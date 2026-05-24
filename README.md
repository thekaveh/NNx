# NNx

Lightweight PyTorch training / eval / visualization toolkit, with first-class support for graph neural networks (GCN / GraphSAGE / GAT). Originally extracted from `thekaveh/ml` to underpin training loops, checkpointing, and result visualization across notebook-based experiments; now standalone.

## What's inside

- **`nnx.NNModel`** — orchestrator. Builds the network from params, runs train / eval / predict, manages checkpoints, supports callbacks (incl. early stopping) and optional CUDA mixed precision.
- **Networks** — `FeedFwdNN` (vision / tabular), and `GraphConvNN` / `GraphSageNN` / `GraphAttNN` (all built on the shared `GraphNNBase` so they differ only in their PyG layer constructor).
- **Datasets** — `NNDataset` (torchvision `VisionDataset` wrapper with train/val/test split — val is carved from train, test stays held-out) and `NNGraphDataset` (PyG single-graph wrapper using `NeighborLoader`).
- **Params** — frozen dataclasses for every config knob: `NNParams` (architecture), `NNModelParams` (orchestration), `NNTrainParams`, `NNOptimParams`, `NNSchedulerParams`. Every params object round-trips through `.state() / .from_state()` for persistence.
- **Enums-as-factories** — `Nets`, `Losses`, `Optims`, `Schedulers`, `Activations`, `Devices`, `Checkpoints`. Each enum value's `__call__` constructs the underlying object, so adding a new option is a single-place change.
- **Callbacks** — `Callback` base class with `on_{train,epoch}_{begin,end}` hooks. Stock: `EarlyStopping`, `LRMonitor`, `ModelCheckpoint`. Legacy `Callable[[List[IDP]], None]` is still accepted.
- **Persistence** — `NNRun.save / load` writes `run.yaml` + `idps.csv` per run under `runs/<md5(state)>/`; `NNCheckpoint` saves at FIRST / Q1 / Q2 / Q3 / LAST / BEST tags. A `runs/best` symlink points at the lowest-error run.
- **Visualization** — `VisUtils` returns Plotly `Figure` objects: `confusion_matrix`, `classification_report` (returns a DataFrame), `multi_line_plot`, `scatter_plot`, `two_dim_tsne_checkpoint_logits`.

## Install

```bash
pip install -e .                # runtime
pip install -e ".[dev]"         # adds pytest, ruff
```

Python 3.10+. Tested on 3.10 / 3.11 / 3.12.

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

```python
from nnx import GraphConvNN, GraphSageNN, GraphAttNN  # used via Nets enum
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
