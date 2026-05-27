# Concepts

This document explains the design decisions behind NNx: the architecture, the foundational patterns every other feature builds on, and the five specialization subpackages.

Sections are ordered from most fundamental to most specialized. Read top-to-bottom on a first pass; jump by anchor for reference.

## 1. Architecture

NNx's public surface, internal orchestration, callback bus, and on-disk persistence are summarized in the architecture diagram. A standalone interactive copy lives at [`architecture.html`](architecture.html); the same SVG appears in the [main README](https://github.com/thekaveh/NNx/blob/main/README.md#11-architecture).

The diagram has eight layers, top-to-bottom:

1. **User code + PyTorch** (slate) — the consumer surface.
2. **`NNModel` / `Trainer`** (cyan) — the two public entry classes.
3. **`train_step_fn` / `trainer_step_fn`** (orange bus) — the optional hook every specialization plugs into.
4. **Specialization subpackages** (amber) — `nnx.finetune`, `nnx.peft`, `nnx.diffusion`, `nnx.paradigms`, `nnx.trainer`, plus the shared `nnx._step_helpers`.
5. **Training-loop internals** (emerald) — the epoch × batch dispatch, `finalize_step` (NaN guard + grad-clip), `_step_scheduler` (Schedulers enum dispatch), `_save_checkpoints` (FIRST/Q1/Q2/Q3/LAST/BEST cadence).
6. **Callback bus** (orange) — `on_train_begin / on_epoch_begin / on_epoch_end / on_train_end`.
7. **Callback listeners** (orange) — `EarlyStopping`, `LRMonitor`, `ModelCheckpoint`, `TensorBoardCallback`, `WandbCallback`.
8. **Persistence** (violet) — `NNRun` writes `run.yaml + idps.csv + metadata.yaml` and `NNCheckpoint` writes `*.pt + *.opt.pt` into `runs/<id>/`.

## 2. Params as the source of truth

Every config in NNx is a frozen, kw-only, slotted dataclass with a `state()` / `from_state()` pair:

```python
NNParams         # network shape: dims, dropout, activation, n_heads
NNModelParams    # device + loss + net kind + mixed precision
NNOptimParams    # SGD / Adam + LR / momentum / grad clipping / accumulation / param_groups
NNSchedulerParams # ReduceLROnPlateau / Step / Cosine / OneCycle / LinearWarmup
NNTrainParams    # epochs + loaders + optim + scheduler + seed + ...
NNTrainerParams  # multi-optim version: dict of optims + dict of schedulers
```

### 2.1. The round-trip contract

`obj == ParamsClass.from_state(obj.state())` for every params class. This is the persistence contract: anything that survives `NNRun.save → run.yaml → NNRun.load` flows through it.

### 2.2. The omit-when-default invariant

New fields added to an existing params class **must omit themselves from `state()` when at their default**. Otherwise every existing `run.id` (md5 of `state()`) shifts and on-disk runs become unfindable. Every params class follows the pattern; regression tests in `tests/test_params_round_trip.py` pin it for `param_groups`, `mixed_precision`, `kind`, `seed`, `save_phase_checkpoints`, `trainer`.

## 3. Enums-as-factories

Every enum's `__call__` constructs the underlying object:

```python
Optims.ADAM(net=net, lr_start=1e-3, ...)              # → torch.optim.Adam
Losses.CROSS_ENTROPY()                                 # → nn.CrossEntropyLoss
Nets.GRAPH_CONV(params=NNParams(...))                  # → GraphConvNN
Schedulers.COSINE_ANNEALING(opt, params, n_epochs)     # → CosineAnnealingLR
NoiseSchedulers.LINEAR(T=1000, beta_min=1e-4, beta_max=2e-2)  # → NoiseSchedule
```

Adding a new option is a one-place change: extend the enum + the `match` block. No parallel dispatch elsewhere to update.

## 4. What lands on disk

Every `model.train(params)` creates a run directory under `runs/<id>/` (where `id` is the md5 of the run's `state()` dict):

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

### 4.1. The `runs/best` pointer

The `runs/best` symlink points at the lowest-error run across all runs in the directory (on Windows without developer mode, it's a `POINTER.txt` file instead).

### 4.2. Atomicity + incremental writes

Every write inside `runs/<id>/` (`run.yaml`, `metadata.yaml`, `idps.csv`, every `*.pt`) goes through a tmp-then-rename atomic helper. A `KeyboardInterrupt` during a save leaves either the previous file or the new file at the destination — never a half-written file. Combined with the per-epoch save cadence, this means an interrupted run remains loadable: the last completed epoch's state is intact.

### 4.3. Config vs environment

`run.yaml` is the configuration; `metadata.yaml` is the environment. Two runs with identical config but different env both write to the same directory — by design, since they're the same experiment. To distinguish them, use different seeds or different data; both flow into `run.yaml` and so into the id.

## 5. Callbacks

`Callback` has four hooks (`on_train_begin / on_epoch_begin / on_epoch_end / on_train_end`) each receiving a `_CallbackContext`:

```python
ctx.model         # NNModel
ctx.run           # NNRun (in-progress)
ctx.optimizer     # torch.optim.Optimizer (primary, sorted-first in Trainer mode)
ctx.optimizers    # dict[str, Optimizer]  (Trainer mode only)
ctx.trainer       # Trainer  (Trainer mode only)
ctx.epoch         # int
ctx.idp           # current NNIterationDataPoint
ctx.idps          # running list of all idps so far
ctx.should_stop   # writable — set True to break out of training
```

Built-in callbacks: `EarlyStopping`, `LRMonitor`, `ModelCheckpoint`, `TensorBoardCallback`, `WandbCallback`. Custom callbacks subclass `Callback` and override whichever hooks they need.

## 6. Custom training paradigms

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

### 6.1. Hook contract

The hook is one optional kwarg on `train()`. The rest of the loop (scheduler, callbacks, checkpoint cadence, val loop, incremental save) stays exactly the same. Your function is responsible for `zero_grad` / forward / loss / backward / `optimizer.step` / NaN guard / gradient accumulation / AMP — `ctx` carries the relevant knobs (`grad_clip_norm`, `accumulate_grad_batches`, `scaler`); honoring them is on you. To layer logging on top of the standard supervised step instead of replacing it, call `default_train_step(ctx)` from inside your hook.

The four paradigm factories in `nnx.paradigms` and `nnx.diffusion.diffusion_train_step_factory` all share an internal helper, `nnx._step_helpers.finalize_step`, that runs the NaN guard before backward and honors `ctx.grad_clip_norm`. AMP and gradient accumulation are not yet handled inside paradigm steps — `finalize_step` raises a clear `ValueError` if either is requested (rather than silently dropping them).

See [`examples/05_custom_train_step_autoencoder.py`](https://github.com/thekaveh/NNx/blob/main/examples/05_custom_train_step_autoencoder.py) for an end-to-end autoencoder example.

### 6.2. What about `evaluate()` and `predict()`?

They still assume supervised classification; they'll grow `eval_step_fn` / `predict_fn` equivalents when the first task that needs them lands.

## 7. Fine-tuning (transfer learning)

The standard transfer-learning recipe — "load pretrained weights, freeze most of the model, train only the head" — has three moving parts in NNx, all under `nnx.finetune`:

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

`NNModel.export_state_dict(path)` saves the inverse — `self.net.state_dict()` only, no `NNCheckpoint` wrapper — for users who want to share weights with non-nnx consumers.

Strict back-compat: `NNOptimParams` with `param_groups=None` (the default) produces exactly the same `state()` dict as before this field existed. Existing `run.id` hashes are unchanged.

See [`examples/06_finetune_with_layer_freezing.py`](https://github.com/thekaveh/NNx/blob/main/examples/06_finetune_with_layer_freezing.py) for the end-to-end recipe.

## 8. Multi-optimizer training (GANs, actor-critic)

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

### 8.1. Strict param-groups semantics

The Trainer enforces **strict** `param_groups` semantics — each optimizer owns ONLY parameters its specs explicitly match. Without that, `opt_G` would also pick up D's parameters in a default bucket and the two optimizers would silently update the same weights. The contract is enforced via `build_param_groups(..., strict=True)`; the same fine-tuning specs from `nnx.finetune` apply, just with unmatched params dropped instead of bucketed.

### 8.2. No default step

There is **no** `default_trainer_step` — multi-optim updates are inherently scenario-specific, and silently running the wrong update is worse than requiring an explicit fn.

### 8.3. NNRun integration

The Trainer writes the same `NNRun` + `NNCheckpoint` artifacts `NNModel.train()` does, with one extra `trainer` block in `run.yaml` capturing the multi-optim config so `NNRun.load(id)` round-trips. Callbacks work in trainer mode too: `ctx.optimizer` is the **primary** (sorted-first) optimizer for back-compat with `LRMonitor` / `TensorBoardCallback`, and `ctx.optimizers` / `ctx.trainer` are added attributes for trainer-aware callbacks.

Strict back-compat: `NNRun` built without a trainer (the standard `NNModel.train()` path) emits exactly the same `state()` as before — existing `run.id` hashes are unchanged.

See [`examples/09_gan_with_trainer.py`](https://github.com/thekaveh/NNx/blob/main/examples/09_gan_with_trainer.py) for a tiny GAN on a 1D mixture-of-Gaussians distribution. Warm-resume from trainer-mode checkpoints (multi-optim sidecars) is a planned follow-up.

## 9. Diffusion (DDPM)

Diffusion lives entirely on top of the `train_step_fn` hook (§6) — no Trainer, no NNModel changes, no new params dataclass. The pieces are under `nnx.diffusion`:

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

### 9.1. Noise schedules

`NoiseSchedulers` is an enum-as-factory matching the rest of the library (`Optims`, `Schedulers`, `Nets`): `NoiseSchedulers.LINEAR(T, beta_min, beta_max)` and `NoiseSchedulers.COSINE(T, s)` each return a `NoiseSchedule` — a frozen dataclass of precomputed tensors (`betas`, `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`, `posterior_variance`). The factory builds tensors on CPU; the train step and sampler migrate them to `model.device` on first use. There's no `state()` / `from_state()` round-trip — the tensors are recoverable from `(kind, T, beta_min, beta_max | s)`, so on-disk runs reconstruct the schedule from the call arguments rather than serializing the tensors.

### 9.2. Denoising network

`DiffusionMLP` is a small conditional MLP: sinusoidal time embed → MLP projection → concat with flattened `x` → MLP → noise prediction. It's *not* a U-Net — for image-space diffusion the same schedule / train step / sampler work against any user-supplied `nn.Module` with the `forward(x, t)` signature.

### 9.3. Training step

`diffusion_train_step_factory(schedule)` returns a `TrainStepFn` (§6) that for each batch samples `t ~ Uniform[0, T)`, samples `ε ~ N(0, I)`, computes `x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε`, predicts noise via `model.net(x_t, t)`, and backprops the MSE between predicted and true noise. The returned EDP sets both `loss` and `error` to the noise-prediction MSE so BEST checkpoint tracking and ReduceLROnPlateau have a metric.

### 9.4. Sampling

`sample(model, schedule, shape, device=, generator=)` runs `T` reverse steps under `torch.no_grad()` and returns generated samples. The optional `generator=` argument enables reproducible sampling — useful for notebook visualization.

See [`examples/08_diffusion_2d_mixture.py`](https://github.com/thekaveh/NNx/blob/main/examples/08_diffusion_2d_mixture.py) for an end-to-end run on a 2D mixture of four Gaussians. After training, samples cluster around all four modes (~25% each).

## 10. Training paradigms (KD, SimCLR, Mixup, CutMix)

All four live in `nnx.paradigms` as `TrainStepFn` factories. Each plugs into `NNModel.train(train_step_fn=...)`:

```python
from nnx import (
    kd_train_step_factory,           # Hinton-style knowledge distillation
    simclr_train_step_factory,       # SimCLR contrastive learning
    mixup_train_step_factory,        # Mixup batch augmentation
    cutmix_train_step_factory,       # CutMix batch augmentation (4D images)
    nt_xent_loss,                    # SimCLR loss exposed for ad-hoc use
)
```

### 10.1. Knowledge distillation

```python
teacher = NNModel.from_checkpoint(...)            # pretrained, larger net
student = NNModel(net_params=..., params=...)     # smaller net, same output_dim
step_fn = kd_train_step_factory(teacher, alpha=0.5, temperature=4.0)
student.train(params=train_params, train_step_fn=step_fn)
```

The factory **freezes the teacher's parameters and sets its net to eval mode** on call — teacher weights are guaranteed not to drift during student training. The loss is `α · KL(softmax(t/T) || softmax(s/T)) · T² + (1-α) · L_hard` — the standard Hinton direction (teacher first), implemented via `F.kl_div(log_softmax(student/T), softmax(teacher/T))`. The hard-label term uses the student's `loss_fn` so KD works for any classification loss (CE, NLL, ...). EDP reports the combined loss and student top-1 error.

### 10.2. SimCLR contrastive

The training dataloader must yield `(view1, view2)` pairs — two augmented views of each source sample. `model.net` forwards each view separately so BatchNorm sees one view at a time:

```python
step_fn = simclr_train_step_factory(temperature=0.5)
model.train(params=..., train_step_fn=step_fn)
```

`nt_xent_loss(z1, z2, temperature)` is exposed as a standalone for users wanting to compose the loss into other pipelines. The augmentation that produces `(view1, view2)` is the caller's responsibility — a paired-view `Dataset` is the common pattern.

### 10.3. Mixup / CutMix

Both interpolate within the batch and re-weight the loss. They're train_step factories (not `collate_fn`s) because mixing labels with arbitrary loss functions needs label-aware computation that doesn't fit the standard `(X, Y)` batch contract:

```python
mixup = mixup_train_step_factory(alpha=0.4)              # any input shape
cutmix = cutmix_train_step_factory(alpha=1.0)            # 4D (B, C, H, W) only
model.train(params=..., train_step_fn=mixup)
```

`λ ~ Beta(α, α)` is the mixing coefficient. The loss is `λ · L(f(x'), y_a) + (1-λ) · L(f(x'), y_b)`. The reported `accuracy` is the λ-weighted correctness (so `accuracy + error == 1`), useful as a signal for `EarlyStopping` / `ReduceLROnPlateau`. CutMix raises on lower-rank input — its spatial cut isn't well-defined without H and W.

See [`examples/10_knowledge_distillation.py`](https://github.com/thekaveh/NNx/blob/main/examples/10_knowledge_distillation.py) for a teacher→student distillation flow on a tabular toy task; the same factory-plus-train_step pattern applies to the other three.

## 11. Parameter-efficient fine-tuning (LoRA, adapters)

When a pretrained model is too large to fine-tune in full, PEFT keeps the original weights frozen and trains a small set of new parameters instead. Two patterns ship in `nnx.peft`:

### 11.1. LoRA — low-rank adaptation

`LoRALinear` wraps an `nn.Linear`, freezes the original weight, and adds two trainable matrices `A` (r × in) and `B` (out × r) whose product is added as a residual:

```
y = W·x  →  y = W·x + (α/r) · B(A(x))
```

`A` is Kaiming-uniform initialized; `B` is **zero-initialized**, so the layer's output at step 0 is exactly `W·x` — fine-tuning starts from the pretrained behavior and diverges only as `B` picks up gradient. The same initialization story as adapters' zero-init `up` projection.

`apply_lora_to(module, *patterns, r, alpha, dropout)` walks a module and replaces every `nn.Linear` whose dotted name matches any glob with a `LoRALinear` wrapper. Returns the count. The match patterns are the same fnmatch globs as `freeze` from §7.

```python
from nnx import NNModel, apply_lora_to, save_lora_weights

model = NNModel(net_params=..., params=...)
model.train(params=pretrain_params)        # full pretraining first

n_wrapped = apply_lora_to(model.net, "layers.*", r=4, alpha=8.0)
# Now every wrapped layer's base weight is frozen (requires_grad=False);
# only the lora_A / lora_B matrices train. Optimizer sees all params, but
# the frozen ones don't move.
model.train(params=finetune_params)        # fine-tune — only LoRA params update

save_lora_weights(model.net, "lora.pt")    # tiny checkpoint, lora_A/B only
```

The `apply_lora_to` mutation is **idempotent**: a second call against patterns that already match LoRA-wrapped layers is a no-op (the inner `.base` is excluded from the walk).

`save_lora_weights(module, path)` writes only the LoRA parameters — typically a small percentage of a full `state_dict` for the same model (single-digit % for production-scale nets with `r=4-8`; closer to ~40% on the tiny demo net where `r/dim` is large). `load_lora_weights(module, source)` loads them back into an already-wrapped module via `load_state_dict(strict=False)`, so the frozen base's missing-from-the-checkpoint keys don't raise. Source can be a path or a state-dict dict.

After wrapping, parameter names gain a `.base.` segment: `layers.0.weight` becomes `layers.0.base.weight`. Code that did `model.net.layers[0].weight` should switch to `model.net.layers[0].base.weight` or use `model.net.layers[0].base` for the wrapped Linear.

### 11.2. Adapter layers

`AdapterLayer(dim, bottleneck, activation=nn.GELU)` is a bottleneck residual: `y = x + up(act(down(x)))` with `up` zero-initialized so the layer starts as the identity. Unlike LoRA, adapters are full modules the caller composes into the forward pass — there's no `apply_adapters_to(module)` helper because "where to insert" depends on the architecture (after each Linear vs after each block vs only at certain depths).

```python
from nnx import AdapterLayer

class AdaptedNet(nn.Module):
    def __init__(self, pretrained_layers):
        super().__init__()
        self.layers = pretrained_layers
        self.adapters = nn.ModuleList([
            AdapterLayer(dim=64, bottleneck=8) for _ in pretrained_layers
        ])

    def forward(self, x):
        for layer, adapter in zip(self.layers, self.adapters):
            x = adapter(layer(x))
        return x
```

See [`examples/07_lora_finetuning.py`](https://github.com/thekaveh/NNx/blob/main/examples/07_lora_finetuning.py) for an end-to-end LoRA flow that explicitly verifies every base parameter is bit-exactly unchanged across the fine-tuning run.

## 12. Reproducibility

```python
from nnx import set_seed, dataloader_worker_init_fn

set_seed(42, strict=True)                    # pins torch / numpy / python / cudnn
loader = DataLoader(..., worker_init_fn=dataloader_worker_init_fn)
NNTrainParams(seed=42, ...)                  # pins again inside train()
```

`strict=True` opts into `torch.use_deterministic_algorithms(True)` — slower and may raise on ops without a deterministic CUDA kernel, but produces bit-for-bit identical training across runs on the same hardware.

## 13. Resuming training

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
