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

## Multi-optimizer training (GANs, actor-critic)

When the per-batch update isn't "one forward → one loss → one optimizer step" but a coordinated dance between multiple optimizers — GAN G/D alternation, actor-critic policy + value, energy-based models — use `nnx.trainer.Trainer` in place of `NNModel.train()`. It accepts one `NNModel` and a name-keyed dict of `NNOptimParams`, builds one `torch.optim.Optimizer` per entry, and hands the dict to a user-supplied `trainer_step_fn`:

```python
from nnx import Trainer, NNTrainerParams, TrainerStepContext, NNEvaluationDataPoint
from nnx import NNOptimParams, NNParamGroupSpec, Optims

trainer = Trainer(model=model)   # `model.net` wraps both G and D as sub-modules

def gan_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
    opt_G, opt_D = ctx.optimizers["G"], ctx.optimizers["D"]
    # ... D step: opt_D.zero_grad() → d_loss.backward() → opt_D.step()
    # ... G step: opt_G.zero_grad() → g_loss.backward() → opt_G.step()
    return NNEvaluationDataPoint(loss=..., error=..., ...)

run = trainer.train(
    params=NNTrainerParams(
        n_epochs=10,
        train_loader=loader,
        optims={
            "G": NNOptimParams(
                name=Optims.ADAM, max_lr=2e-4, momentum=(0.5, 0.999), weight_decay=0.0,
                param_groups=[NNParamGroupSpec(name_pattern="G.*", lr=2e-4)],
            ),
            "D": NNOptimParams(
                name=Optims.ADAM, max_lr=2e-4, momentum=(0.5, 0.999), weight_decay=0.0,
                param_groups=[NNParamGroupSpec(name_pattern="D.*", lr=2e-4)],
            ),
        },
    ),
    trainer_step_fn=gan_step,
)
```

The Trainer enforces **strict** `param_groups` semantics — each optimizer owns ONLY parameters its specs explicitly match. Without that, `opt_G` would also pick up D's parameters in a default bucket and the two optimizers would silently update the same weights. The contract is enforced via `build_param_groups(..., strict=True)`; the same fine-tuning specs from Track A apply, just with unmatched params dropped instead of bucketed.

The Trainer writes the same `NNRun` + `NNCheckpoint` artifacts `NNModel.train()` does, with one extra `trainer` block in `run.yaml` capturing the multi-optim config so `NNRun.load(id)` round-trips. There is **no** `default_trainer_step` — multi-optim updates are inherently scenario-specific, and silently running the wrong update is worse than requiring an explicit fn.

Callbacks work in trainer mode too: `ctx.optimizer` is the **primary** (sorted-first) optimizer for back-compat with `LRMonitor` / `TensorBoardCallback`, and `ctx.optimizers` / `ctx.trainer` are added attributes for trainer-aware callbacks.

Strict back-compat: `NNRun` built without a trainer (the standard `NNModel.train()` path) emits exactly the same `state()` as before — existing `run.id` hashes are unchanged.

See [`examples/09_gan_with_trainer.py`](https://github.com/thekaveh/NNx/blob/main/examples/09_gan_with_trainer.py) for a tiny GAN on a 1D mixture-of-Gaussians distribution. Warm-resume from trainer-mode checkpoints (multi-optim sidecars) is a planned follow-up.

## Diffusion (DDPM)

Diffusion lives entirely on top of the [`train_step_fn` hook](#custom-training-paradigms) — no Trainer, no NNModel changes, no new params dataclass. The pieces are under `nnx.diffusion`:

```python
from nnx import (
    DiffusionMLP, NoiseSchedulers,
    diffusion_train_step_factory, sample,
    NNModel, NNModelParams, NNParams, NNTrainParams, NNOptimParams,
    Activations, Devices, Losses, Nets, Optims,
)

model = NNModel(net_params=..., params=NNModelParams(net=Nets.FEED_FWD, ...))
# The Nets.FEED_FWD factory builds a classifier — wrong shape for diffusion
# (`forward(x_t, t) → ε`). Swap it for the DiffusionMLP. NNModel.train()
# reads model.net.parameters() + model.net_params, both of which survive
# this substitution.
model.net = DiffusionMLP(input_dim=2, hidden_dims=[64, 64], time_embed_dim=16)

schedule = NoiseSchedulers.LINEAR(T=200)         # or NoiseSchedulers.COSINE
step_fn  = diffusion_train_step_factory(schedule)
model.train(params=NNTrainParams(..., train_loader=loader), train_step_fn=step_fn)

# Sample by running the reverse-diffusion loop.
samples = sample(model, schedule, shape=(256, 2))
```

`NoiseSchedulers` is an enum-as-factory matching the rest of the library (`Optims`, `Schedulers`, `Nets`): `NoiseSchedulers.LINEAR(T, beta_min, beta_max)` and `NoiseSchedulers.COSINE(T, s)` each return a `NoiseSchedule` — a frozen dataclass of precomputed tensors (`betas`, `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`, `posterior_variance`). The factory builds tensors on CPU; the train step and sampler migrate them to `model.device` on first use. There's no `state()` / `from_state()` round-trip — the tensors are recoverable from `(kind, T, beta_min, beta_max | s)`, so on-disk runs reconstruct the schedule from the call arguments rather than serializing the tensors.

`DiffusionMLP` is a small conditional MLP: sinusoidal time embed → MLP projection → concat with flattened `x` → MLP → noise prediction. It's *not* a U-Net — for image-space diffusion the same schedule / train step / sampler work against any user-supplied `nn.Module` with the `forward(x, t)` signature.

`diffusion_train_step_factory(schedule)` returns a [`TrainStepFn`](#custom-training-paradigms) that for each batch samples `t ~ Uniform[0, T)`, samples `ε ~ N(0, I)`, computes `x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε`, predicts noise via `model.net(x_t, t)`, and backprops the MSE between predicted and true noise. The returned EDP sets both `loss` and `error` to the noise-prediction MSE so BEST checkpoint tracking and ReduceLROnPlateau have a metric.

`sample(model, schedule, shape, device=, generator=)` runs `T` reverse steps under `torch.no_grad()` and returns generated samples. The optional `generator=` argument enables reproducible sampling — useful for notebook visualization.

See [`examples/08_diffusion_2d_mixture.py`](https://github.com/thekaveh/NNx/blob/main/examples/08_diffusion_2d_mixture.py) for an end-to-end run on a 2D mixture of four Gaussians. After training, samples cluster around all four modes (~25% each).

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
