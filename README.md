# NNx

Lightweight PyTorch training / eval / visualization toolkit, with first-class support for graph neural networks (GCN / GraphSAGE / GAT). Originally extracted from `thekaveh/ml` to underpin training loops, checkpointing, and result visualization across notebook-based experiments; now standalone.

## What's inside

- **`nnx.NNModel`** — orchestrator. Builds the network from params, runs train / eval / predict, manages checkpoints, supports callbacks (incl. early stopping), warm-resume training, mixed precision, gradient clipping, gradient accumulation, and seeded reproducibility.
- **Networks** — `FeedFwdNN` (vision / tabular), and `GraphConvNN` / `GraphSageNN` / `GraphAttNN` (all built on the shared `GraphNNBase` so they differ only in their PyG layer constructor).
- **Datasets** — `NNDataset` (torchvision `VisionDataset` wrapper), `NNGraphDataset` (PyG single-graph wrapper using `NeighborLoader`), and `NNTabularDataset` (pandas DataFrame → train/val/test loaders).
- **Params** — frozen dataclasses for every config knob: `NNParams` (architecture), `NNModelParams` (orchestration), `NNTrainParams`, `NNOptimParams`, `NNSchedulerParams`. Every params object round-trips through `.state() / .from_state()` for persistence.
- **Enums-as-factories** — `Nets`, `Losses`, `Optims`, `Schedulers`, `Activations`, `Devices`, `Checkpoints`. Each enum value's `__call__` constructs the underlying object, so adding a new option is a single-place change.
- **Callbacks** — `Callback` base class with `on_{train,epoch}_{begin,end}` hooks. Stock: `EarlyStopping`, `LRMonitor`, `ModelCheckpoint`, `TensorBoardCallback` (opt-in via `nnx[tensorboard]`), `WandbCallback` (opt-in via `nnx[wandb]`). Legacy `Callable[[List[IDP]], None]` is still accepted.
- **Persistence** — `NNRun.save / load` writes `run.yaml` + `idps.csv` + `metadata.yaml` (env snapshot) per run under `runs/<md5(state)>/`; saved incrementally after every epoch so interrupted training stays loadable. `NNCheckpoint` saves at FIRST / Q1 / Q2 / Q3 / LAST / BEST tags with optimizer-state `.opt.pt` sidecars for warm-resume.
- **Visualization** — `VisUtils` (and module-level aliases) returns Plotly `Figure` objects: `confusion_matrix`, `classification_report` (returns a DataFrame), `multi_line_plot`, `scatter_plot`, `two_dim_tsne_checkpoint_logits`.
- **Reproducibility** — `nnx.set_seed(seed, strict=False)` pins every RNG + cuDNN; `nnx.dataloader_worker_init_fn` for per-worker seeds; `NNTrainParams.seed` runs `set_seed` at `train()` entry.
- **ONNX export** — `NNModel.to_onnx(path, example_input)` exports the network via legacy `torch.onnx.export` (no `onnxscript` dep needed).
- **Custom training step** — `NNModel.train(..., train_step_fn=...)` swaps out the supervised forward/backward/step for any user-supplied function. Unblocks autoencoder / VAE / link-prediction / recommendation / diffusion paradigms without modifying NNx core. See `docs/concepts.md` and `examples/05_custom_train_step_autoencoder.py`.

## Install

```bash
pip install -e .                       # runtime
pip install -e ".[dev]"                # adds pytest, ruff, coverage, tensorboard, onnx
pip install -e ".[tensorboard]"        # TensorBoardCallback
pip install -e ".[wandb]"              # WandbCallback
pip install -e ".[onnx]"               # NNModel.to_onnx validation tooling
pip install -e ".[docs]"               # mkdocs build (mkdocs-material + mkdocstrings)
```

Python 3.10+. Tested on 3.10 / 3.11 / 3.12. Examples in [examples/](examples/) are runnable on CPU.

## Quickstart

End-to-end CPU example — a tiny random-tensor classification run:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    NNModel, NNParams, NNModelParams, NNTrainParams,
    NNOptimParams, NNSchedulerParams,
    Activations, Devices, Losses, Nets, Optims,
    EarlyStopping,
)

# 1. Data
X_train, y_train = torch.randn(256, 8), torch.randint(0, 3, (256,))
X_val,   y_val   = torch.randn(64, 8),  torch.randint(0, 3, (64,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=32)

# 2. Model
net_params   = NNParams(input_dim=8, output_dim=3, hidden_dims=[32, 16],
                        dropout_prob=0.1, activation=Activations.RELU)
model_params = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU,
                             loss=Losses.CROSS_ENTROPY)
model = NNModel(net_params=net_params, params=model_params)

# 3. Train
train_params = NNTrainParams(
    n_epochs=10,
    train_loader=train_loader,
    val_loader=val_loader,
    optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2,
                        momentum=(0.9, 0.999), weight_decay=5e-5),
    scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5,
                                patience=3, cooldown=1, threshold=1e-3),
)
run = model.train(params=train_params, callbacks=[EarlyStopping(patience=5)])

# 4. Use it
print(f"trained {len(run.idps)} iterations; saved under runs/{run.id}/")
y_log, y_hat = model.predict(X=X_val.numpy())
```

### Other models

Switch architectures by changing the `Nets` enum value passed to `NNModelParams`. NNModel constructs the underlying network for you:

```python
NNModelParams(net=Nets.GRAPH_CONV,  device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
NNModelParams(net=Nets.GRAPH_SAGE,  device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
NNModelParams(net=Nets.GRAPH_ATT,   device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
# Then pass NNParams(..., n_heads=4) for GRAPH_ATT.
# Use NNGraphDataset (PyG NeighborLoader-backed) to feed batches.
```

See [examples/](examples/) for end-to-end scripts.

### Reproducibility

```python
from nnx import set_seed, dataloader_worker_init_fn
set_seed(42)                                        # pins torch / numpy / python / cudnn
DataLoader(..., worker_init_fn=dataloader_worker_init_fn)
NNTrainParams(seed=42, ...)                         # pins again at train() entry
```

### Warm-resume training

```python
run = model.train(params=NNTrainParams(n_epochs=10, ...))

# Build a fresh NNModel and continue from run's LAST checkpoint
# (optimizer state preserved via .opt.pt sidecar):
NNModel(net_params=..., params=...).train(params=NNTrainParams(
    n_epochs=10,
    resume_from_run_id=run.id,
    resume_from_checkpoint="last",   # or "best"
    ...
))
```

### Custom metrics

```python
NNTrainParams(
    ...,
    extra_metrics={
        "my_metric": lambda y, y_hat: float((y == y_hat).mean()),
    },
)
# Available on idp.train_edp.extra / idp.val_edp.extra and survives NNRun.load.
```

### Visualization

```python
from nnx import VisUtils
fig = VisUtils.confusion_matrix(y_true, y_pred, class_names=["a","b","c"])
fig.show()
df = VisUtils.classification_report(y_true, y_pred)  # DataFrame
```

### Auto device detection

```python
from nnx import Devices
NNModelParams(net=Nets.FEED_FWD, device=Devices.get(), loss=Losses.CROSS_ENTROPY)
# Devices.get() picks MPS (Apple) > CUDA > CPU.
```

### Mixed precision (CUDA)

```python
NNModelParams(..., mixed_precision=True)   # silently no-op on CPU/MPS
```

### Scheduler choices

By default the scheduler is `ReduceLROnPlateau` driven by the params dataclass. Pass `kind=` to switch:

```python
from nnx import Schedulers
NNSchedulerParams(..., kind=Schedulers.COSINE_ANNEALING, T_max=100)
# Or: STEP, ONE_CYCLE, LINEAR_WARMUP_DECAY
```

### Loading a run

```python
from nnx import NNRun, NNCheckpoint, Checkpoints
run  = NNRun.load(id="<md5>")                              # rehydrate idps + params
ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
model = NNModel.from_checkpoint(checkpoint=ckpt)
```

## Status

Alpha. API is stable for the existing `thekaveh/ml` notebook consumer; pre-1.0 means we'll fix obvious bugs (see [CHANGELOG](CHANGELOG.md)) without renaming public APIs unless they're broken in ways notebooks can't work around.

## Contributing

Bug reports and PRs welcome via GitHub issues. Running locally:

```bash
pytest                      # full suite (~3s)
pytest tests/test_callbacks.py::test_lr_monitor_records_history  # one test
ruff check src/ tests/      # lint (gates CI)
```

## License

MIT. See [LICENSE](LICENSE).
