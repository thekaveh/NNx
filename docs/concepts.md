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
