# Concepts

This document explains the design decisions behind NNx: the architecture, the foundational patterns every other feature builds on, and the twelve specialization subpackages — Tier-1 (`finetune`, `peft`, `diffusion`, `paradigms`, `trainer`) and Tier-2 (`quantize`, `prune`, `surgery`, `embeddings`, `interop`, `viz`, `generation`) — plus the decoder-only LM path on top.

Sections are ordered from most fundamental to most specialized. Read top-to-bottom on a first pass; jump by anchor for reference.

## 1. Architecture

NNx's public surface, internal orchestration, callback bus, and on-disk persistence are summarized in the architecture diagram. A standalone interactive copy lives at [`architecture.html`](architecture.html); the same SVG appears in the [main README](https://github.com/thekaveh/NNx/blob/main/README.md#11-architecture).

The diagram has eight layers, top-to-bottom:

1. **User code + PyTorch** (slate) — the consumer surface.
2. **`NNModel` / `Trainer`** (cyan) — the two public entry classes.
3. **`train_step_fn` / `trainer_step_fn`** (orange bus) — the optional hook every specialization plugs into.
4. **Specialization subpackages** (amber) — model-side: `nnx.finetune`, `nnx.peft` (LoRA / DoRA / IA3 / Prefix / Prompt / Adapters), `nnx.prune`, `nnx.surgery`, `nnx.quantize`; data + paradigm side: `nnx.diffusion`, `nnx.paradigms` (KD / feature-KD / SimCLR / Mixup / CutMix / MoE / I-JEPA / DPO / Born-Again), `nnx.trainer`; interop + downstream: `nnx.embeddings` (contrastive + FAISS), `nnx.interop` (GGUF + Ollama + safetensors), `nnx.generation` (LogitsProcessor chain), `nnx.viz` (model-internals viz). All plug into the orange `train_step_fn` / `trainer_step_fn` hook from Layer 3 — plus the shared `nnx._step_helpers`.
5. **Training-loop internals** (emerald) — the epoch × batch dispatch, the inline NaN guard + grad-clip in `default_train_step`, `_step_scheduler` (Schedulers enum dispatch), `_save_checkpoints` (FIRST/Q1/Q2/Q3/LAST/BEST cadence). Note: the shared `finalize_step` helper lives under the **Specialization subpackages** layer (Layer 4, in `nnx._step_helpers`) and is invoked only from paradigm / diffusion step-fn factories — not from the supervised loop, which has its own inline NaN+clip path.
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

New fields added to an existing params class **must omit themselves from `state()` when at their default**. Otherwise every existing `run.id` (md5 of `state()`) shifts and on-disk runs become unfindable. Every params class follows the pattern; regression tests pin it for `mixed_precision` / `kind` / `trainer` in `tests/test_params_round_trip.py`, for `param_groups` in `tests/test_finetune_param_groups.py` and `tests/test_pass2_ou_series.py`, for `seed` / `save_phase_checkpoints` in `tests/test_trainer_params.py` and `tests/test_pass2_ou_series.py`, and for `schedulers` in `tests/test_trainer_params.py`.

### 2.3. Variant-gated construction via `.builder()`

Params dataclasses with **tagged-union shape** (a `kind` field whose value gates which other fields are meaningful) expose a `.builder()` classmethod as an alternative to the direct kwarg constructor. The Builder methods are named after the variants; each writes exactly the fields its variant uses, so the user can't construct an invalid combination by accident.

`NNSchedulerParams` is the first such Builder. The five variant methods —
`reduce_on_plateau`, `step`, `cosine_annealing`, `one_cycle`,
`linear_warmup_decay` — set `kind` plus the right variant-specific
knobs and leave the rest at the dataclass defaults. State() and
from_state() round-trip unchanged; the existing direct-kwarg ctor is
untouched.

```python
from nnx import NNSchedulerParams

scheduler = NNSchedulerParams.builder().one_cycle(
    max_lr=1e-3, total_steps=10_000,
    min_lr=1e-7, factor=0.5, patience=10, cooldown=2, threshold=1e-3,
).build()
```

`NNOptimParams.builder()` extends the pattern with **four optimizer-variant methods** —
`adam`, `adam_amsgrad`, `sgd`, `sgd_nesterov` — plus three optional chained
modifiers — `grad_clip(norm)`, `accumulate_grad(batches)`, `param_groups(specs)`.
The Adam variants take the PyTorch-native `betas: tuple[float, float]` kwarg,
which the Builder maps onto the underlying `NNOptimParams.momentum` field
(the field name stays `momentum` for on-disk back-compat). SGD variants keep
the float `momentum=` kwarg.

```python
from nnx import NNOptimParams

# Adam with PyTorch-native spelling
opt = NNOptimParams.builder().adam(max_lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0).build()

# SGD with grad-clip and gradient accumulation
opt = (
    NNOptimParams.builder()
    .sgd(max_lr=1e-2, momentum=0.9, weight_decay=5e-5)
    .grad_clip(1.0)
    .accumulate_grad(4)
    .build()
)
```

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

The four paradigm factories in `nnx.paradigms` and `nnx.diffusion.diffusion_train_step_factory` all share an internal helper, `nnx._step_helpers.finalize_step`, that runs the NaN guard before backward and honors `ctx.grad_clip_norm`. AMP and gradient accumulation are not yet handled inside paradigm steps — `finalize_step` raises a clear `ValueError` if either is requested (rather than silently dropping them). The AMP rejection only fires when `ctx.scaler` is non-None, which on CPU it never is (the supervised path silently bypasses AMP on CPU/MPS regardless of `NNModelParams.mixed_precision`); the explicit error is the user-facing safety net for the CUDA path, where silent drop would actually matter.

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

**Builder shape (composes Plans 1 + 2):**

The same GAN recipe via the composite Builder. The `.scheduler("d", ...)`
without a prior `.optimizer("d", ...)` would fail at `.build()` time
with an actionable error naming the typo:

```python
from nnx import NNTrainerParams, NNOptimParams, NNSchedulerParams

g_optim = NNOptimParams.builder().adam(max_lr=2e-4, betas=(0.5, 0.999), weight_decay=0.0).build()
d_optim = NNOptimParams.builder().adam(max_lr=2e-4, betas=(0.5, 0.999), weight_decay=0.0).build()
plateau = NNSchedulerParams.builder().reduce_on_plateau(
    min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3
).build()

trainer_params = (
    NNTrainerParams.builder()
    .n_epochs(50)
    .optimizer("g", g_optim)
    .optimizer("d", d_optim)
    .scheduler("g", plateau)
    .scheduler("d", plateau)
    .build()
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

## 10. Training paradigms

All paradigms live in `nnx.paradigms` as `TrainStepFn` factories — Hinton-style knowledge distillation, FitNets-style feature distillation, SimCLR contrastive, Mixup and CutMix batch augmentation, sparse top-k Mixture-of-Experts, I-JEPA self-supervised pretraining, DPO preference fine-tuning, and Born-Again iterated self-distillation. Each plugs into `NNModel.train(train_step_fn=...)`:

```python
from nnx import (
    kd_train_step_factory,           # Hinton-style knowledge distillation
    feature_kd_train_step_factory,   # FitNets-style intermediate-feature KD
    simclr_train_step_factory,       # SimCLR contrastive learning
    mixup_train_step_factory,        # Mixup batch augmentation
    cutmix_train_step_factory,       # CutMix batch augmentation (4D images)
    moe_train_step_factory,          # MoE supervised step + Switch aux loss
    jepa_train_step_factory,         # I-JEPA self-supervised pretraining
    dpo_train_step_factory,          # DPO preference fine-tuning (LM)
    born_again_train,                # Iterated self-distillation wrapper
    nt_xent_loss,                    # SimCLR loss exposed for ad-hoc use
)
```

§§10.1–10.3 below cover the four foundational paradigms (KD, SimCLR, Mixup, CutMix). The newer additions are documented in dedicated pages or under §15 (DPO ties into the LM path):

- **Feature-KD** — extends `kd_train_step_factory` with an MSE term between named teacher/student intermediate activations; full signature in [API §10](api.md).
- **MoE** — `MoELinear` drop-in for `nn.Linear` + `moe_train_step_factory` (sums per-layer `last_aux_loss` into the main loss as a Switch-style load-balancing penalty); demo in `examples/14_moe_classifier.py`.
- **I-JEPA** — masked-patch → latent-prediction against an EMA target encoder; full walkthrough in [`docs/jepa.md`](jepa.md).
- **DPO** — preference-pair fine-tuning against a frozen reference policy; see §15 and [`docs/dpo.md`](dpo.md).
- **Born-Again** — `born_again_train(...)` iterates self-distillation across G generations; see [API §10](api.md).

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

## 11. Parameter-efficient fine-tuning (LoRA, DoRA, IA3, Prefix, Prompt, Adapters)

When a pretrained model is too large to fine-tune in full, PEFT keeps the original weights frozen and trains a small set of new parameters instead. `nnx.peft` ships six adapters covering the full spectrum from rank-decomposed residuals (LoRA) down to a single per-output scaling vector (IA3). §§11.1–11.2 walk through LoRA and `AdapterLayer` in detail; the others share the same wrap-and-freeze idiom — see [API §9](api.md) for full signatures.

- **DoRA** (`DoRALinear` / `apply_dora_to`) — subclass of `LoRALinear` adding a trainable per-output-row `magnitude` vector and recomposing the layer's weight as `W = magnitude · V / ||V||_c`. Often outperforms LoRA at the same rank with only `out_features` extra params. Save/load shares `save_lora_weights` for the `lora_A`/`lora_B` matrices; the `magnitude` parameter rides along via the full `state_dict()` round-trip.
- **IA3** (`IA3Linear` / `apply_ia3_to`) — the smallest adapter in the family: a single learned per-output-dim scaling vector applied to a frozen `nn.Linear`'s output. Dedicated `save_ia3_weights` / `load_ia3_weights` persist only the scaling tensor.
- **PrefixTuner** (`PrefixTuner` / `save_prefix_weights` / `load_prefix_weights`) — prepends a learned key/value prefix to every attention layer of a frozen `TransformerNN`. The model itself is unchanged; the tuner stores the prefix tensors and routes them through the attention forward via a hook.
- **PromptTuner** (`PromptTuner` / `save_prompt_weights` / `load_prompt_weights`) — prepends learned soft-prompt embeddings ahead of the input tokens of a frozen `TransformerNN`. Cheapest of the LM-targeted PEFT methods; useful when even the rank-decomposed LoRA budget is too large.

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

## 12. Model-internals visualization

`nnx.vis_utils` covers **run-output** viz (training curves, confusion matrices, t-SNE of checkpoint logits). The companion `nnx.viz` subpackage covers **model-internals** viz — the model itself, not what the run produced. Six primitives ship today: `summary`, `weight_histogram`, `activation_map`, `attribute`, `gradient_flow`, and `netron_export`.

`nnx.viz.summary(model, input_size=...)` returns a `torchinfo.ModelStatistics` — print it for the Keras-style parameter table; access `.total_params` / `.trainable_params` / `.total_mult_adds` for programmatic regression assertions. Accepts an `NNModel` (unwrapped to `.net`) or any `torch.nn.Module`. Requires the optional `viz` extra (`pip install nnx[viz]` pulls in `torchinfo`).

`nnx.viz.weight_histogram(model)` walks `model.named_parameters()` and emits one Plotly `Histogram` trace per tensor in a grid subplot. Useful for spotting dead layers, NaN / Inf weights, or saturation patterns at a glance.

`nnx.viz.activation_map(model, x, layer_name)` registers a forward hook on the named submodule, runs `model(x)` under `torch.no_grad()`, and returns a Plotly heatmap: a grid of per-channel heatmaps for 4D conv activations `(N, C, H, W)`, or a single `(N, F)` heatmap for 2D dense activations. Pass a dotted name from `model.named_modules()` (`"layers.0"`, `"conv1"`, etc.); a typo raises `ValueError` and lists the first available names so you can fix it.

`nnx.viz.attribute(model, x, *, method, target, **method_kwargs)` is a Captum-backed input-attribution wrapper with single string-keyed dispatch over six methods: `integrated_gradients`, `gradient_shap`, `deep_lift`, `saliency`, `input_x_gradient`, `occlusion`. Returns `(attribution_tensor, plotly.Figure)` — the figure renders the attribution as a Plotly heatmap (3-/4-D image-shaped inputs are mean-pooled over channels first). Captum is lazy-imported at the call site, so the rest of `nnx.viz` keeps working without it; the missing-dep path raises a clear `ImportError`. Sensible per-method defaults (`baselines=zeros` for GradientShap, `sliding_window_shapes` for Occlusion) preserve the one-call ergonomics. Requires the `viz` extra (`pip install nnx[viz]` pulls in `captum` alongside `torchinfo`).

`nnx.viz.gradient_flow(model)` is the diagnostic for vanishing / exploding gradients during training. Call after `loss.backward()` and before `optimizer.zero_grad()`; returns a Plotly bar chart of per-trainable-parameter L2 gradient norms. Frozen parameters (`requires_grad=False`) and parameters that weren't reached by the forward pass (gradient is `None`) are skipped. Raises `ValueError` with a helpful message if no parameter has a gradient — the usual cause is forgetting `loss.backward()`.

`nnx.viz.netron_export(model, "model.onnx", example_input)` exports the underlying network via `torch.onnx.export` so the artifact can be opened in [Netron](https://netron.app/). Passing `launch=True` additionally calls `netron.start(path)` to open the browser viewer; that path requires the `viz-interactive` extra (`pip install nnx[viz-interactive]`).

```python
import torch
from nnx import NNModel, NNParams, NNModelParams, Activations, Devices, Losses, Nets
from nnx.viz import activation_map, netron_export, summary, weight_histogram

model = NNModel(
    net_params=NNParams(input_dim=8, output_dim=3, hidden_dims=[32, 16],
                        dropout_prob=0.1, activation=Activations.RELU),
    params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU,
                         loss=Losses.CROSS_ENTROPY),
)
print(summary(model, input_size=(1, 8)))   # Keras-style parameter table
weight_histogram(model).show()              # Plotly grid of per-tensor weight distributions
activation_map(model, torch.randn(4, 8), "layers.0").show()  # batch x features heatmap
netron_export(model, "model.onnx", torch.randn(1, 8))         # write graph for Netron
```

## 13. Reproducibility

```python
from nnx import set_seed, dataloader_worker_init_fn

set_seed(42, strict=True)                    # pins torch / numpy / python / cudnn
loader = DataLoader(..., worker_init_fn=dataloader_worker_init_fn)
NNTrainParams(seed=42, ...)                  # pins again inside train()
```

`strict=True` opts into `torch.use_deterministic_algorithms(True)` — slower and may raise on ops without a deterministic CUDA kernel, but produces bit-for-bit identical training across runs on the same hardware.

### 13.1. LR finder

Before a long training run, run `nnx.lr_finder` to pick a defensible `max_lr` for a one-cycle scheduler. The sweep is non-destructive — model weights are snapshotted and restored on exit — so you can call it as a pre-flight check inside the same script that trains for real.

```python
from nnx import lr_finder
import torch.nn.functional as F

result = lr_finder(
    model.net, train_loader,
    loss_fn=F.cross_entropy,
    start_lr=1e-7, end_lr=10.0, num_iter=100,
)
print(f"Suggested max_lr: {result.suggested_lr:.2e}")
result.figure.show()
```

`suggested_lr` is the LR at the steepest descent point of the EMA-smoothed loss curve — the Smith (2017) heuristic. Plug it into `NNOptimParams.max_lr` for the real training run.

### 13.2. Non-destructive contract for inference and inspection helpers

`lr_finder` isn't the only helper that snapshots and restores caller state. Nine NNx call sites share the same non-destructive contract — they put the underlying `nn.Module` into `eval()` mode (needed for correct BatchNorm / Dropout semantics) for the duration of the call, then restore `model.training` to whatever it was on entry. The restore runs inside a `try/finally`, so the contract holds even when the body raises mid-call:

- `nnx.lr_finder`
- `NNModel.predict`, `NNModel.evaluate`
- `GenerativeNNModel.generate`
- `nnx.diffusion.sample`
- `nnx.embeddings.embed_texts`
- `nnx.viz.activation_map`, `nnx.viz.netron_export`, `nnx.viz.attribute`

This means a common train → evaluate → train-more (or train → predict → train-more) loop no longer strands the model in `.eval()` mode after the helper returns — Dropout and BatchNorm pick up exactly where they left off on the next training step. Before the post-PR-#40 maintenance pass, `predict` / `evaluate` / `generate` / `sample` / `embed_texts` leaked `.eval()` state, silently disabling Dropout masking and BatchNorm running-stats updates on the next training step unless the caller remembered to call `model.net.train()` themselves.

## 14. Resuming training

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

## 15. Generative language modeling (`TransformerNN` + `GenerativeNNModel`)

The decoder-only LM path is the largest architectural addition since the
foundational training loop. It introduces three new abstractions that sit
alongside `NNModel` rather than replacing it:

- **`TransformerNN`** — a small decoder-only transformer (`RMSNorm` pre-norm,
  rotary positional embeddings, SwiGLU FFN, tied input/output embeddings,
  fused QKV projection). Sits at the same level as `FeedFwdNN` / `GraphConvNN`
  / `GraphSageNN` / `GraphAttNN` and is selected via `Nets.TRANSFORMER`.
- **`NNTransformerParams`** + **`NNTokenizerParams`** — params subclasses that
  add the LM-specific shape knobs (`vocab_size`, `n_layers`, `d_model`,
  `max_seq_len`, `ffn_mult`, `rope_base`, `tie_embeddings`, `attn_dropout`,
  `resid_dropout`). Every optional field omits itself from `state()` at
  default — the same broken-three-times invariant covered in §2.2.
- **`GenerativeNNModel`** — a thin subclass of `NNModel` adding
  `generate(prompt, *, max_new_tokens, logits_processors=...)`. The
  generation loop runs the prompt through `TransformerNN.forward_with_kv`
  for a single prefill step, then incrementally decodes token-by-token using
  the returned KV-cache (measured ≈1.9× speedup at 128 tokens on CPU; the
  gap widens on longer contexts and GPU). Sampling defaults to greedy; pass a
  list of `nnx.generation.LogitsProcessor` (temperature, top-k, top-p,
  repetition-penalty) to switch.

The LM path stays optional behind the `lm` extra (`pip install "nnx[lm]"` —
pulls `tokenizers` + `datasets`); the rest of NNx works without it. See
[`docs/lm.md`](lm.md) for the end-to-end walkthrough and
[`examples/11_tinystories_lm.py`](https://github.com/thekaveh/NNx/blob/main/examples/11_tinystories_lm.py)
for a CPU-friendly TinyStories training run.

Downstream of the LM path, four follow-ons compose on top of it:

- **PEFT for transformers** — `PrefixTuner` / `PromptTuner` (see §11)
  attach learned key/value or input-embedding prefixes to a frozen
  `TransformerNN` for parameter-efficient adaptation.
- **DPO** — `dpo_train_step_factory` (see [`docs/dpo.md`](dpo.md))
  fine-tunes a `TransformerNN` against `(prompt, chosen, rejected)`
  preference triples via the Rafailov et al. 2023 chosen-vs-rejected
  log-ratio objective, no reward modeling or RL.
- **GGUF + Ollama export** — `nnx.interop.write_gguf` writes a
  llama.cpp-compatible `.gguf` (fused-QKV split, SwiGLU `w1`/`w3`/`w2` →
  `ffn_gate`/`ffn_up`/`ffn_down`, RoPE/RMSNorm metadata). See
  [`docs/gguf.md`](gguf.md).
- **HuggingFace Hub publish** — via the `PyTorchModelHubMixin` integration
  on `NNModel` itself (any subclass inherits it). See
  [`docs/hub.md`](hub.md).

## 16. Tier-2 subpackage deep-dives

Four Tier-2 subpackages are large enough to warrant a dedicated section but small enough that the canonical write-up lives elsewhere. This catalog is the pointer index — open the linked page for the full walkthrough.

- **`nnx.quantize`** — PTQ INT8 weight-only (`quantize_int8(model)`) and QAT 8da4w (`qat_train_step_factory` + `QATLifecycleCallback`), both built on `torchao`. The PTQ path is one call, no calibration data, no retraining; the QAT path is a paradigm-style `TrainStepFn` factory that fake-quants during training and converts on commit. Opt-in via `pip install nnx[quantize]`. See [API §11](api.md) for the full surface; `examples/12_quantize_int8.py` and `examples/15_qat_classifier.py` for end-to-end runs.
- **`nnx.prune`** — `magnitude_prune` (mask-based unstructured, checkpoint-safe) and `semi_structured_24` (2:4 semi-structured via `torchao` for Ampere+ inference). The `bake=True` default keeps `state_dict` keys identical to the un-pruned net so pruned checkpoints load into stock code under `strict=True`. See [API §12](api.md); `examples/` does not yet ship a pruning demo (see deferred items in [CHANGELOG](https://github.com/thekaveh/NNx/blob/main/CHANGELOG.md)).
- **`nnx.surgery`** — `widen` / `deepen` (function-preserving Net2Net edits — Chen/Goodfellow/Shlens, ICLR 2016), `drop_layer`, `low_rank_factorize` (SVD truncation, exact at max rank), and `expand_embedding`. Every primitive returns a fresh `nn.Module` and composes with `NNModel.train()` for the "load checkpoint → surgery → refine" loop. Full walkthrough with before/after parameter-count tables in [`docs/surgery.md`](surgery.md).
- **`nnx.embeddings`** — the one RAG-adjacent surface NNx ships. `train_contrastive` reuses the existing NT-Xent machinery for domain-specific text embedders; `export_to_faiss` writes the trained model's outputs to a FAISS index (Flat / HNSW) that any retrieval framework (LangChain / LlamaIndex / Haystack / raw FAISS) can consume. The chunker, reranker, and vector-DB client are deliberately out of scope. See [`docs/embeddings.md`](embeddings.md) for the full when-to-use guide; `examples/13_train_domain_embedder.py` is the runnable demo.

`nnx.generation` (LogitsProcessor chain) is documented inline in §15 since its raison d'être is `GenerativeNNModel.generate(...)`. `nnx.interop` (safetensors + GGUF + Ollama) is documented under §15 and on [`docs/gguf.md`](gguf.md) / [`docs/hub.md`](hub.md). `nnx.viz` is in §12 above.
