# Concepts

## Architecture in one diagram

```
         NNParams (architecture)        NNModelParams (orchestration)
                  │                              │
                  └──────────┬───────────────────┘
                             ▼
                          NNModel
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
     train(params)      evaluate(loader)     predict(X)
        │
        ├──> NNRun (md5 of state) ──> runs/<id>/run.yaml + idps.csv
        ├──> NNCheckpoint × 6 tags ─> runs/<id>/checkpoints/*.pt
        │                              + *.opt.pt (optimizer sidecars)
        └──> Callbacks: EarlyStopping, LRMonitor, TensorBoard, Wandb, ...
```

## Params as the source of truth

Every config in nnx is a frozen dataclass with a `state()` / `from_state()` pair:

```python
NNParams         # network shape: dims, dropout, activation, n_heads
NNModelParams    # device + loss + net kind + mixed precision
NNOptimParams    # SGD / Adam + LR / momentum / grad clipping / accumulation
NNSchedulerParams # ReduceLROnPlateau / Step / Cosine / OneCycle / LinearWarmup
NNTrainParams    # epochs + loaders + optim + scheduler + seed + ...
```

The round-trip is the persistence contract: `obj == NNParams.from_state(obj.state())`. New fields default to omitting themselves from `.state()` so existing runs keep hashing to the same `run.id`.

## Enums-as-factories

Every enum's `__call__` builds the underlying object:

```python
Optims.ADAM(net=net, lr_start=1e-3, ...)   # → torch.optim.Adam
Losses.CROSS_ENTROPY()                     # → nn.CrossEntropyLoss
Nets.GRAPH_CONV(params=NNParams(...))      # → GraphConvNN
Schedulers.COSINE_ANNEALING(opt, params, n_epochs)  # → CosineAnnealingLR
```

Adding a new option is a one-place change: extend the enum + the `match` block. No parallel dispatch elsewhere to update.

## What lands on disk

Every `model.train(params)` creates a run directory under `./runs/<md5_of_state>/`:

```
runs/<id>/
├── run.yaml          # NNRun.state() — config-only, hashes to <id>
├── metadata.yaml     # env snapshot (nnx/torch/python/git) — NOT in hash
├── idps.csv          # per-iteration metrics, flushed every epoch
├── checkpoints/
│   ├── first.pt      # NNCheckpoint at epoch 0
│   ├── q1.pt q2.pt q3.pt   # at 1/4, 2/4, 3/4 of n_epochs
│   ├── last.pt       # most recent epoch
│   ├── last.pt.opt.pt    # optimizer state sidecar (warm-resume)
│   ├── best.pt       # lowest-error checkpoint so far
│   └── best.pt.opt.pt
```

The `runs/best` symlink points at the lowest-error run across all runs in the directory (on Windows without developer mode, it's a `POINTER.txt` file instead).

Every write inside `runs/<id>/` (run.yaml, metadata.yaml, idps.csv, every `*.pt`) goes through a tmp-then-rename atomic helper. A `KeyboardInterrupt` during a save leaves either the previous file or the new file at the destination — never a half-written file. Combined with the per-epoch save cadence, this means an interrupted run remains loadable: the last completed epoch's state is intact.

`run.yaml` is the configuration; `metadata.yaml` is the environment. Two runs with identical config but different env both write to the same directory — by design, since they're the same experiment. To distinguish them, use different seeds or different data; both flow into `run.yaml` and so into the id.

## Callbacks

`Callback` has four hooks (`on_train_begin / on_epoch_begin / on_epoch_end / on_train_end`) each receiving a `_CallbackContext`:

```python
ctx.model         # NNModel
ctx.run           # NNRun (in-progress)
ctx.optimizer     # torch.optim.Optimizer
ctx.epoch         # int
ctx.idp           # current NNIterationDataPoint
ctx.idps          # running list of all idps so far
ctx.should_stop   # writable — set True to break out of training
```

Built-in callbacks: `EarlyStopping`, `LRMonitor`, `ModelCheckpoint`, `TensorBoardCallback`, `WandbCallback`. Custom callbacks subclass `Callback` and override whichever hooks they need.

## Custom training paradigms

`NNModel.train()` runs a supervised loop by default — for every batch, it does `loss_fn(net(X), Y)` → backward → step. If your task doesn't fit that shape (autoencoders, VAEs, link prediction with negative sampling, recommendation pairwise losses, diffusion noise prediction), pass a `train_step_fn`:

```python
from nnx import TrainStepContext, NNEvaluationDataPoint

def my_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
    X, _ = ctx.model.net.unpack_batch(ctx.batch)
    X = tuple(x.to(ctx.model.device) for x in X)
    recon, mu, logvar = ctx.model.net(*X)
    recon_loss = F.mse_loss(recon, X[0])
    kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    loss = recon_loss + kl
    loss.backward()
    ctx.optimizer.step()
    return NNEvaluationDataPoint(
        f1=0, recall=0, accuracy=0, precision=0,
        loss=float(loss.detach()),
        extra={"recon": float(recon_loss), "kl": float(kl)},
    )

model.train(params=train_params, train_step_fn=my_step)
```

The hook is one optional kwarg on `train()`. The rest of the loop (scheduler, callbacks, checkpoint cadence, val loop, incremental save) stays exactly the same. Your function is responsible for `zero_grad` / forward / loss / backward / `optimizer.step` / NaN guard / gradient accumulation / AMP — `ctx` carries the relevant knobs (`grad_clip_norm`, `accumulate_grad_batches`, `scaler`); honoring them is on you. To layer logging on top of the standard supervised step instead of replacing it, call `default_train_step(ctx)` from inside your hook.

See [`examples/05_custom_train_step_autoencoder.py`](https://github.com/thekaveh/NNx/blob/main/examples/05_custom_train_step_autoencoder.py) for an end-to-end autoencoder example.

`evaluate()` and `predict()` still assume supervised classification; they'll grow `eval_step_fn` / `predict_fn` equivalents when the first task that needs them lands.

## Fine-tuning (transfer learning)

The standard transfer-learning recipe — "load pretrained weights, freeze most of the model, train only the head" — has three moving parts in nnx, all under `nnx.finetune`:

```python
from nnx import NNModel, load_pretrained, NNParamGroupSpec, NNOptimParams, Optims

model = NNModel(net_params=..., params=...)

# 1. Load weights from an external state-dict / .pt file / other nn.Module.
#    Pass key_map= to rewrite foreign naming (e.g., {"backbone.": "net."}).
result = load_pretrained(model.net, "resnet18.pt", strict=False)
print(f"loaded {len(result.loaded_keys)}, missing {len(result.missing_keys)}")

# 2. Freeze whatever shouldn't train. Glob patterns match the dotted
#    parameter name. `model.freeze` is a shortcut for nnx.finetune.freeze.
model.freeze("layers.0.*", "layers.1.*")          # freeze the backbone
# `model.unfreeze("*")` would reverse it; `frozen(model.net)` lists what's frozen.

# 3. (Optional) Run the unfrozen part with a smaller LR than a fresh head.
#    NNOptimParams.param_groups takes a list of NNParamGroupSpec; each
#    matches parameters by glob and overrides lr / lr_multiplier / weight_decay.
optim = NNOptimParams(
    name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=5e-4,
    param_groups=[
        NNParamGroupSpec(name_pattern="layers.0.*", lr_multiplier=0.01),
        NNParamGroupSpec(name_pattern="*.bias",     weight_decay=0.0),
    ],
)
```

`NNModel.export_state_dict(path)` saves the inverse — `self.net.state_dict()` only, no NNCheckpoint wrapper — for users who want to share weights with non-nnx consumers.

Strict back-compat: `NNOptimParams` with `param_groups=None` (the default) produces exactly the same `state()` dict as before this field existed. Existing `run.id` hashes are unchanged.

See [`examples/06_finetune_with_layer_freezing.py`](https://github.com/thekaveh/NNx/blob/main/examples/06_finetune_with_layer_freezing.py) for the end-to-end recipe.

## Reproducibility

```python
from nnx import set_seed, dataloader_worker_init_fn

set_seed(42, strict=True)                    # pins torch / numpy / python / cudnn
loader = DataLoader(..., worker_init_fn=dataloader_worker_init_fn)
NNTrainParams(seed=42, ...)                  # pins again inside train()
```

`strict=True` opts into `torch.use_deterministic_algorithms(True)` — slower and may raise on ops without a deterministic CUDA kernel, but produces bit-for-bit identical training across runs on the same hardware.

## Resuming training

```python
# First run
run = model.train(params=NNTrainParams(n_epochs=10, ...))

# Continue from LAST (preserves Adam momentum / SGD velocity via .opt.pt sidecar)
NNModel(net_params=..., params=...).train(params=NNTrainParams(
    n_epochs=10,
    resume_from_run_id=run.id,
    resume_from_checkpoint="last",
    ...
))
```

Checkpoints written before resume support (i.e., from runs that predate this feature) don't carry an `.opt.pt` sidecar — weights still load, but the optimizer starts fresh.
