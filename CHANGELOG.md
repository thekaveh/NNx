# Changelog

All notable changes to NNx are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is roughly [SemVer](https://semver.org/) — pre-1.0, we allow behavior changes (typically bug fixes) without renaming public APIs.

## [Unreleased]

### Fixed — post-tracks audit

- **`NNModelParams.state()` omits `mixed_precision` when False.** The field was added in pass-2 but always emitted into `state()`, breaking the omit-when-default back-compat invariant. Every default-AMP run had a shifted `run.id` versus pre-pass-2 runs with otherwise identical config. **One-time hash shift:** any existing default-AMP `runs/<id>/` directory will recompute to a different id after this fix — load by the on-disk directory name still works; recomputed ids will land in a fresh directory.
- **`NNSchedulerParams.state()` omits `kind` and the variant-specific knobs** (`step_size` / `T_max` / `max_lr` / `total_steps` / `warmup_steps`) when None. Same omit-when-default invariant: a plain ReduceLROnPlateau `NNSchedulerParams` now hashes to the same `run.id` as it did before the `Schedulers` enum was added. Existing on-disk runs with explicit-None entries still load (the legacy form is tolerated in `from_state`).
- **In-memory `best_checkpoint` tracking aligned with on-disk BEST.** `NNModel.train()`'s `best_checkpoint` reassignment used a different comparison than the BEST write inside `_save_checkpoints`. When `val_loader=None` (so every `val_edp` is None), the in-memory tracker effectively held LAST while the on-disk BEST tracked training error. Both now go through the same `_best_err` helper.
- **`_best_err` deduplicated.** Was triplicated — a local closure in `NNModel._save_checkpoints`, a module-level helper in `nn_run.py`, and another module-level helper in `trainer/trainer.py`. Kept the `nn_run.py` version as canonical; the other two now import it.
- **Paradigm step factories honor `grad_clip_norm` and guard against non-finite loss.** The four paradigm `train_step_fn` factories (diffusion / SimCLR / Mixup / CutMix) plus KD now route through a shared `nnx._step_helpers.finalize_step` helper. Previously they silently dropped `NNOptimParams.grad_clip_norm`, and diffusion / SimCLR / Mixup / CutMix had no NaN/Inf guard — only KD checked. **New explicit rejection:** the helper raises `ValueError` if `NNModelParams.mixed_precision=True` (paradigm steps don't handle the scaler) or if `accumulate_grad_batches != 1` (no cycle-aware accumulation). Previously these were silently ignored; users with those knobs set now see a clear error.
- **`ModelCheckpoint` callback actually saves now.** The body was `if ctx.epoch in self.epochs: pass` — a no-op stub. Now writes `runs/<run.id>/checkpoints/<tag>_e<epoch>.pt` via the atomic-write path on matched epochs.
- **`FeedFwdNN.from_file` uses `torch.load(weights_only=True)`** for consistency with `NNCheckpoint.load_optimizer_state` and `load_pretrained`. State-dicts are tensor-only; the strict loader works AND removes the arbitrary-code-execution risk on user-supplied paths.
- Documentation and comment cleanups: `docs/index.md` listed only pass-2 features (added the five new tracks); `docs/concepts.md` architecture diagram missed the five new subpackages (extended with a Specializations branch); `examples/06`'s `_make_loaders` docstring claimed class-conditional Gaussians that the code didn't implement (rewrote); `freezing.py` docstring incorrectly claimed `fnmatch *` matches segment-boundaries (it matches across dots); `loading.py` `key_map` docstring said "substring replacement" but the code does prefix replacement; `paradigms/distillation.py` comment said "KL(student || teacher)" but the implementation is the standard `KL(teacher || student)`; `peft/adapters.py` activation docstring said `nn.GELU()` (instance) but the default is `nn.GELU` (class factory); README's enums-as-factories bullet was missing `NoiseSchedulers`. Internal phase labels (Track A / Track B / Track C / pass-2 R2 / R3 / R4) that had leaked into published code/docs/tests have been replaced with descriptions of WHAT the referenced thing is.

### Added — PEFT (Track B)

- **`nnx.peft` package** — two complementary patterns for parameter-efficient fine-tuning of pretrained networks.
  - **`LoRALinear(base, *, r, alpha, dropout)`** — wraps an `nn.Linear`, freezes the base's parameters (`requires_grad=False`) on construction, and adds two trainable matrices `lora_A` (r × in, Kaiming-uniform init) and `lora_B` (out × r, **zero-initialized**) whose product is added as a residual scaled by `α/r`. The zero-init on B means output at step 0 equals `base(x)` exactly — fine-tuning starts from the pretrained behavior and diverges only as B picks up gradient. Validates `r > 0`, `alpha > 0`, `0 ≤ dropout < 1` at construction.
  - **`apply_lora_to(module, *patterns, r, alpha, dropout)`** — walks `module.named_modules()` and replaces every `nn.Linear` whose dotted name matches any fnmatch glob with a `LoRALinear` wrapper, in place. Returns the count wrapped. **Idempotent**: re-applying against patterns that already match LoRA-wrapped layers is a no-op (the inner `.base` is excluded from the walk). Same glob conventions as `nnx.finetune.freeze` from Track A.
  - **`save_lora_weights(module, path)`** — writes ONLY the `lora_A` / `lora_B` parameters via `torch.save` of a filtered state-dict subset. Typically 1-5% of the size of a full `state_dict` snapshot for the same model.
  - **`load_lora_weights(module, source)`** — loads LoRA params from a path (`weights_only=True` for safety) or directly from a dict, via `load_state_dict(strict=False)` so the frozen base's missing keys don't raise. Returns the number of tensors loaded.
  - **`AdapterLayer(dim, bottleneck, activation=nn.GELU)`** — bottleneck residual block `y = x + up(act(down(x)))`. `up.weight` and `up.bias` are zero-initialized so the layer is the residual identity at step 0. Composed by the user into a custom `nn.Module` — NNX doesn't ship a "wrap every block" helper because adapter insertion points are architecture-specific.
- Runnable LoRA demo: `examples/07_lora_finetuning.py` — pretrains a small classifier, wraps every Linear with LoRA, fine-tunes on a different distribution, **explicitly verifies every base parameter is bit-exactly unchanged** across the fine-tuning run, and compares the LoRA-only checkpoint size against a full `state_dict` snapshot.
- 23 new tests across `tests/test_peft_{lora,adapters}.py`: LoRALinear validation + base-freezing + zero-init invariant (output == base at step 0) + only-LoRA-trainable invariant + in/out features pass-through; `apply_lora_to` empty-pattern rejection + selective wrap + wildcard wrap + idempotency on re-application + forward-preserves-at-init; save/load round-trip + base-keys-excluded-from-checkpoint + dict-source loading + bad-source-type rejection; end-to-end PEFT contract (every base param bit-exactly unchanged + every lora_B param has moved); AdapterLayer shape + identity-at-init + parameter-count scaling + gradient-flow + dim validation + custom activation.

### Added — training paradigms (Track D)

- **`nnx.paradigms` package** — four `TrainStepFn` factories for non-vanilla supervised paradigms, all consumed via the existing `NNModel.train(train_step_fn=...)` hook. No new params dataclass, no NNModel changes; each is a self-contained closure.
  - **`kd_train_step_factory(teacher, *, alpha, temperature)`** — Hinton-style knowledge distillation. Mixes a temperature-softened KL divergence against the teacher's logits (`α · KL · T²`) with the standard hard-label loss (`(1-α) · L_hard`). The factory **freezes the teacher's parameters and sets its net to eval mode on call**, so the teacher provably cannot drift across the student's training. The hard term goes through the student's `loss_fn` so KD works with any classification loss.
  - **`simclr_train_step_factory(*, temperature)`** — SimCLR contrastive training. The training loader must yield `(view1, view2)` paired-view tensors per source sample. `model.net` is forwarded once per view (BatchNorm sees one view at a time). Reports the NT-Xent loss in both `.loss` and `.error`.
  - **`nt_xent_loss(z1, z2, *, temperature)`** — the SimCLR loss exposed as a standalone for users wanting to compose it into custom training loops.
  - **`mixup_train_step_factory(*, alpha)`** — Mixup batch augmentation: `x' = λx_a + (1-λ)x_b` with `λ ~ Beta(α, α)`. Works for any input rank (tabular, sequence, image). Reports λ-weighted accuracy as the `accuracy` field; `accuracy + error == 1`.
  - **`cutmix_train_step_factory(*, alpha)`** — CutMix batch augmentation for 4D `(B, C, H, W)` image batches. Copies a random rectangle from `x_b` into `x_a`, then re-weights the loss by the actual cut area (which can be smaller than the nominal Beta draw when the box clips at an edge). Raises a clear `ValueError` on lower-rank input — CutMix's spatial cut isn't well-defined without H and W.
- Runnable distillation demo: `examples/10_knowledge_distillation.py` — pretrains a wider teacher then distills into a 1/16-size student. The example explicitly verifies teacher weights are unchanged across the student's training run, demonstrating the factory's freeze guarantee. Honest about scope: doesn't claim to beat a non-distilled baseline on toy tabular data.
- 19 new tests across `tests/test_paradigms_{distillation,contrastive,augmentation}.py`: factory validation (alpha / temperature ranges), teacher freezing guarantee + teacher-eval-mode assertion, end-to-end loss-decreases (KD α=0.5) + α-boundary cases (α=0.0 collapse to supervised, α=1.0 pure distillation), NT-Xent properties (shape mismatch, finite + scalar output, loss smaller for aligned pairs than random), SimCLR step bad-batch-shape error, Mixup self-consistency (accuracy + error == 1), CutMix non-image input rejection + 4D end-to-end.

### Added — diffusion (Track C)

- **`nnx.diffusion` package** — DDPM-style diffusion training and sampling, layered entirely on top of the existing `train_step_fn` hook on `NNModel.train()` (no Trainer, no NNModel internals touched).
  - **`NoiseSchedulers`** — enum-as-factory with two variants: `LINEAR(T, beta_min, beta_max)` (original DDPM linear betas) and `COSINE(T, s)` (Improved-DDPM cosine schedule). Each enum value's `__call__` returns a precomputed `NoiseSchedule`.
  - **`NoiseSchedule`** — frozen dataclass holding the derived tensors (`betas`, `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`, `posterior_variance`). All 1D of length T. `.to(device)` returns a copy with every tensor migrated. Not `state()`-serialized — recoverable from `(kind, T, kind-specific knobs)`.
  - **`DiffusionMLP(input_dim, hidden_dims, time_embed_dim)`** — small conditional MLP: sinusoidal time embed → projection → concat with flat x → MLP → noise prediction. `forward(x, t) → ε_pred`. Handles arbitrary-rank inputs by flattening + un-flattening. Intentionally minimal; image-space diffusion calls for a U-Net the user supplies, with the same schedule / step / sampler machinery.
  - **`diffusion_train_step_factory(schedule) -> TrainStepFn`** — closes over the schedule and returns a `TrainStepFn` suitable for `NNModel.train(train_step_fn=...)`. Per batch: samples `t ~ Uniform[0, T)`, samples `ε ~ N(0, I)`, computes `x_t`, predicts noise, backprops MSE. Reports loss as both `.loss` and `.error` on the EDP so BEST tracking + ReduceLROnPlateau work.
  - **`sample(model, schedule, shape, device=, generator=)`** — reverse-diffusion sampler. Runs T backward steps under `torch.no_grad()` and `model.net.eval()`. The optional `generator=` enables reproducible sampling for notebooks.
  - **`sinusoidal_time_embed(t, dim)`** — standalone helper for the standard sinusoidal positional embedding, exposed for users building their own t-conditioned nets.
- **`NNModel.train()` net-params fallback** — the run-construction line now reads `self.net_params` (always set in `__init__`) instead of `self.net.params` (FeedFwdNN-specific attribute). Back-compat-safe: the values are identical for the existing supervised path. Lets callers swap `model.net` for a custom `nn.Module` post-construction (the same idiom Track G's GAN demo uses) without breaking `NNModel.train()`.
- Runnable diffusion demo: `examples/08_diffusion_2d_mixture.py` — DDPM on a 2D mixture of 4 Gaussians at (±2, ±2). Verified end-to-end (loss 1.0078 → 0.6048; samples land in all four modes at roughly equal counts).
- 27 new tests across `tests/test_diffusion_{schedules,nets,training,sampling}.py` covering schedule shape/monotonicity/clamping, net forward shape, full training + loss-decreases, sampling shape / finiteness / reproducibility / mode coverage.

### Added — multi-optimizer Trainer (Track G)

- **`nnx.trainer` package** — `Trainer` class that parallels `NNModel.train()` for scenarios where the per-batch update isn't a single supervised forward/backward/step. Built around the GAN G/D pattern, but applicable to actor-critic, EBM, contrastive multi-head, or any other multi-optimizer paradigm.
  - **`Trainer(model: NNModel).train(params, trainer_step_fn, callbacks=)`** — builds one `torch.optim.Optimizer` per entry in `NNTrainerParams.optims`, dispatches to a user-supplied `trainer_step_fn(ctx) -> NNEvaluationDataPoint` per batch, writes the same `NNRun` + per-tag `NNCheckpoint` artifacts as `NNModel.train()`. No `default_trainer_step` — multi-optim updates are scenario-specific and silently running the wrong update is worse than requiring an explicit fn.
  - **`NNTrainerParams`** — frozen dataclass with `optims: Mapping[str, NNOptimParams]` (name-keyed multi-optim config), `schedulers: Mapping[str, NNSchedulerParams]` (one per optim, defaults to ReduceLROnPlateau when missing), plus the standard `n_epochs` / `train_loader` / `val_loader` / `seed` / `save_phase_checkpoints` / `extra_metrics`. Validates non-empty `optims` and that every scheduler key matches an optim key. `state()` keys sorted for deterministic `run.id`.
  - **`TrainerStepContext`** — frozen bundle passed into a `trainer_step_fn`: `model`, `batch`, `optimizers` (dict), `schedulers` (dict), `extra_metrics`, `batch_idx`, `epoch_idx`. The companion `TrainerStepFn` type alias is exported.
- **Strict `param_groups` semantics** for multi-optim — `build_param_groups(..., strict=True)` (new keyword) drops parameters that match no spec instead of bucketing them into a default group. Threaded through `Optims.__call__(..., strict_param_groups=True)`. The Trainer passes True so disjoint optimizers don't co-own parameters via implicit default buckets. Default `strict=False` preserves Track A fine-tuning semantics exactly.
- **`NNRun.trainer: Optional[NNTrainerParams]`** — populated by the Trainer; None for `NNModel.train()` runs. **Strict back-compat:** OMITTED from `state()` when None so existing `NNModel` run.id hashes are unchanged. `NNRun.load(id)` round-trips trainer-mode runs by lazy-importing `NNTrainerParams.from_state` when the YAML carries a `trainer` block.
- Runnable GAN demo: `examples/09_gan_with_trainer.py` — generator + discriminator packed into one nn.Module, two disjoint optimizers scoped via `NNParamGroupSpec(name_pattern="G.*" | "D.*")`, alternating updates on a 1D mixture-of-Gaussians. Verified end-to-end on CPU.

**Deferred from this PR:** trainer-mode warm-resume. The Trainer writes only the model net's `state_dict` to its `NNCheckpoint`s — there is no per-optimizer `.opt.<name>.pt` sidecar yet. `NNTrainerParams` does not carry `resume_from_run_id` / `resume_from_checkpoint`. Resuming a GAN's Adam state for both G and D will land as its own follow-up PR once the use case is exercised.

### Added — fine-tuning infrastructure (Track A)

- **`nnx.finetune` package** with three submodules:
  - **`freezing`** — `freeze(module, *patterns)` / `unfreeze(module, *patterns)` / `frozen(module)`. Glob-pattern (`fnmatch`) toggling of `requires_grad` on submodule parameters; the standard transfer-learning idiom. `NNModel.freeze` / `NNModel.unfreeze` are convenience methods delegating to the free functions.
  - **`loading`** — `load_pretrained(module, source, *, key_map, strict, prefix)` returns a `LoadPretrainedResult` with `loaded_keys` / `missing_keys` / `unexpected_keys`. Sources: file paths (loaded with `weights_only=True` for safety), state-dicts, or other `nn.Module`s. Key remapping handles foreign naming conventions (torchvision / HuggingFace / etc.).
  - **`param_groups`** — `NNParamGroupSpec` (frozen, kw_only, slots dataclass) for declarative per-layer LR / weight_decay overrides. The fine-tuning idiom of "small LR on the backbone, large LR on the head" expressed as a list of specs on `NNOptimParams.param_groups`. `build_param_groups(module, specs, default_lr, default_weight_decay)` is the helper the `Optims` enum factory dispatches through.
- **`NNOptimParams.param_groups: Optional[list[NNParamGroupSpec]]`** field. When set, the optimizer factory builds per-group dicts with the spec's lr / lr_multiplier / weight_decay overrides; frozen parameters are dropped. **Strict back-compat:** `param_groups=None` (default) is OMITTED from `state()`, so existing `run.id` hashes are unchanged.
- **`NNModel.export_state_dict(path)`** — saves `self.net.state_dict()` to disk as a plain torch file (no NNCheckpoint wrapper). Companion to `load_pretrained` for the round-trip.

### Added

- **`train_step_fn` hook on `NNModel.train()`.** One optional kwarg that swaps out the supervised forward/backward/step for any user-supplied function. Unblocks non-supervised training paradigms (autoencoder, VAE, link prediction, recommendation, diffusion) without modifying NNx core. Default-None path is byte-identical to the prior loop. New public surface: `TrainStepContext` (frozen dataclass carrying model/batch/optimizer/scaler/grad_clip_norm/extra_metrics/accumulate_grad_batches/batch_idx/epoch_idx), `default_train_step(ctx)` (the standard supervised step, exported for users who want to layer behavior on top), `TrainStepFn` (type alias). Seven tests in `tests/test_train_step_hook.py`; runnable autoencoder example at `examples/05_custom_train_step_autoencoder.py`.
- Public alias for `nnx.PredictResult` (was reachable only via `nnx.nn.nn_model`).

### Changed — internal

- `NNModel.__fwd_pass` → `NNModel._fwd_pass`. Required so the free `default_train_step` can reach it without Python name-mangling. Single underscore is still "weak private"; no external consumer touched the mangled `_NNModel__fwd_pass` name.
- `NNModel._train_step` becomes a one-line wrapper around `default_train_step` for back-compat with any hypothetical subclass that overrode it. The `train()` loop itself no longer dispatches through `_train_step`.

### Fixed

- `_save_checkpoints` / `_step_scheduler` / `_update_tqdm_postfix` now tolerate an `NNEvaluationDataPoint` with `error=None`. Custom `train_step_fn` hooks for non-supervised paradigms (VAE/autoencoder/diffusion) don't always have a classification error to report; the loop falls back through `val_edp.error → val_edp.loss → train_edp.error → train_edp.loss` and skips the scheduler step entirely if nothing is set. Previously these three sites crashed with `TypeError` on `None < float` / `float(None)` / `f"{None:.4f}"`.

### Deferred

- `eval_step_fn` / `predict_fn` — same pattern, but `evaluate()` and `predict()` still assume supervised classification. First ml-lab task that needs custom eval (autoencoder, VAE, DDPM) will drive that.
- Network registry (`Nets.register(...)`) — each new architecture lands a `Nets` enum variant via its task's PR.
- Loss registry — custom losses live inside `train_step_fn` today (the user computes the loss tensor manually). Lift to a registry when multiple tasks duplicate the same custom loss.

## [Pass-2 unreleased] — comprehensive improvements pass 2

Second improvement pass on branch `chore/comprehensive-improvements-pass-2`, building on pass-1. Strict back-compat preserved throughout — every new field on a params dataclass defaults to its old value and omits itself from `state()` when the default holds, so existing `run.id` hashes are unchanged.

### Added — features (F-series)

- **Warm-resume training.** `NNTrainParams.resume_from_run_id` and `resume_from_checkpoint` load weights AND optimizer state from a prior run's checkpoint at the start of `train()`. Optimizer state is written as a `.opt.pt` sidecar so the existing pickled `NNCheckpoint` format is untouched.
- **Gradient accumulation** via `NNOptimParams.accumulate_grad_batches` (default 1). Loss is scaled by 1/N; `zero_grad`/`optimizer.step` fire on cycle boundaries; AMP unscale + grad-clip both honor the cycle.
- **TensorBoardCallback** and **WandbCallback** — stream per-epoch train/val metrics + LR. Lazy import so users not on the path don't pay the dep cost.
- **`NNModel.to_onnx(path, example_input)`** — export the network via the legacy `torch.onnx.export` tracing path (no `onnxscript` needed). Marks dim-0 dynamic by default.
- **`NNTabularDataset`** — wraps a pandas DataFrame into train/val/test loaders matching the `NNDatasetBase` contract.
- **Custom metrics** via `NNTrainParams.extra_metrics={name: fn}`. Each `fn(Y, Y_hat) -> float` populates the new `NNEvaluationDataPoint.extra` dict; survives the `NNRun.save`/`NNRun.load` round-trip via `extra.<name>` CSV columns.

### Added — reproducibility (V-series)

- `nnx.set_seed(seed, strict=False)` pins Python `random`, NumPy, torch CPU+CUDA, and cuDNN. `strict=True` also calls `torch.use_deterministic_algorithms(True)`.
- `nnx.dataloader_worker_init_fn` — pass to `DataLoader(worker_init_fn=...)` for per-worker deterministic seeds.
- `NNTrainParams.seed` runs `set_seed` at `train()` entry; included in `state()` only when set.
- `nnx.env_snapshot()` captures library / torch / numpy / python / platform / CUDA / git-commit info. Written by `NNRun.save()` to `runs/<id>/metadata.yaml` — separate from `run.yaml` so it does NOT contribute to `run.id`.

### Added — API ergonomics (O+U-series)

- `NNModel.predict(X)` accepts `numpy.ndarray`, `torch.Tensor`, tuples thereof, or a `DataLoader` (labels in batches are discarded). Returns a `PredictResult` NamedTuple that unpacks positionally as `(logits, classes)` for back-compat.
- `NNTrainParams.save_phase_checkpoints: bool = True`. Set False to skip the FIRST + Q1/Q2/Q3 cycle (LAST + BEST still always saved) — useful for tiny experiments or huge models.
- `Devices.torch_device()` / `Devices.get_torch_device()` return `torch.device` directly without the `.()` dance.
- `Utils.print_tree` / `print_table` accept `file=` for output redirection.
- `nnx.__version__` resolves from `importlib.metadata`; falls back to `"0.1.0+local"` when editable-installed.
- `pyproject` keywords expanded (training, checkpointing, callbacks, experiments, reproducibility, neural-networks, research).

### Added — reliability (R-series)

- **NaN/Inf guard** in `NNModel._train_step` — raises `FloatingPointError` rather than letting divergence corrupt checkpoints silently.
- **Gradient clipping** via `NNOptimParams.grad_clip_norm: Optional[float]`. AMP-aware (unscales before clipping).
- **Incremental persistence** — `NNRun.save()` runs after every epoch, not just at the end. `KeyboardInterrupt` / OOM mid-training now leaves a loadable partial run.
- **SECURITY note** on `NNCheckpoint.from_file` calling out the arbitrary-code-execution risk of `weights_only=False` on untrusted files.
- Re-pin `loss_fn` to `self.device` on every `evaluate()` call (guards against late device reassignment).

### Fixed — correctness (N-series)

- `NNOptimParams.is_valid()` now returns `False` (not implicit `None`) for unknown enum variants — invalid configs no longer slip past the `not params.optim.is_valid()` pre-flight check.
- `NNModel.train()` tolerates `DataLoader`s without `__len__` (`IterableDataset`-backed). Falls back to a tqdm bar with no total.
- `NNRun.save()` falls back to writing `best/POINTER.txt` when `os.symlink` raises (Windows without developer mode).
- `NNModel.evaluate()` aggregates Y / Y_hat across batches and computes metrics once on the aggregate, fixing unequal-final-batch weighting. Raises `ValueError` on an empty loader instead of returning NaN.
- `NNIterationDataPoint` gets a docstring spelling out that `val_edp` is populated only on the LAST idp of each epoch — readers shouldn't expect it on every row.

### Changed — tooling (S+T-series)

- CI runs pytest under coverage (`pytest-cov`), uploads `coverage.xml` artifact on Python 3.11.
- CI runs pyright in basic mode (`continue-on-error: true` today; will tighten to `--strict` over time).
- `NNX_TQDM_DISABLE=1` silences the training progress bar — autouse'd in `tests/conftest.py` so pytest output stays clean.
- `tests/conftest.py` exposes shared fixtures (`tiny_model`, `tiny_classification_loaders`, `tmp_runs_root`, ...).
- `mkdocs.yml` + `docs/` skeleton (index, quickstart, concepts, api). New `.github/workflows/docs.yml` builds with `--strict` on every push and deploys via `mkdocs gh-deploy` on `main`. New `nnx[docs]` optional extra.
- `.pre-commit-config.yaml` with ruff + standard pre-commit-hooks.
- `CONTRIBUTING.md` covering setup, workflow, back-compat invariants, testing.
- `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.md` + `pull_request_template.md`.
- `.github/workflows/release.yml` — tag-triggered build + PyPI publish via OIDC trusted publishing.

### Added — docs (U3)

- `examples/` folder with four runnable scripts: `01_synthetic_classification.py`, `02_resume_training.py`, `03_custom_metrics.py`, `04_onnx_export.py`. All verified end-to-end on CPU.

### Internal (D6)

- `Utils.print_tree` / `print_table` / `flatten_dict` are now module-level functions in `nnx.utils`. The `Utils` class is a thin shim binding the same functions as staticmethods, so existing `Utils.method(...)` callers continue to work with no semantic change.
- `VisUtils` plotting helpers get module-level aliases (`from nnx.vis_utils import confusion_matrix` works).

### Additional fixes (post-initial-pass)

- `runs/best` POINTER.txt fallback wasn't read during BEST comparison; env_snapshot subprocessed git on every save; `.gitignore` missed `runs/`, `tb_logs/`, `*.onnx`, `coverage.xml`, `site/`.
- **Critical:** `NNOptimParams.state()` unconditionally emitted `grad_clip_norm=None`, changing every existing `run.id` hash. Plus `callbacks.py` top-level IPython import (pulling IPython into every `import nnx`); `NNRun.all()` crashed on missing `runs/` and tried to load stray files.
- `mkdocs build --strict` had 4 warnings (specs in docs but not nav; griffe couldn't parse a docstring; missing type annotation on `to_onnx.example_input`).
- **Critical:** `NNEvaluationDataPoint.extra` didn't actually round-trip through `idps.csv`. json_normalize flattened the dict on save but `NNIterationDataPoint.from_state` never reassembled the `train_edp.extra.*` columns. The pass-2 claim that "extra survives idps.csv" was false until this fix.
- `pytest-cov` listed in dev extras but not installed locally. CI handles via `pip install -e ".[dev]"`; surfaced via cov-run on a fresh venv.
- `NNEvaluationDataPoint.mean_of` silently dropped the `extra` dict from inputs. `NNCheckpoint.load_optimizer_state` now uses `weights_only=True` (the state dict is structured tensors + dicts; the strict loader works AND removes the ACE risk).
- Six conftest fixtures (`tiny_model`, `tiny_classification_loaders`, etc.) defined but unused — premature abstractions deleted; CONTRIBUTING.md updated to match.
- `NNTabularDataset` now validates `feature_cols` / `target_col` against `df.columns` up-front with a clear KeyError; new test for `env_snapshot` cache (introduced in R1 but never explicitly tested).
- stray leading blank line in `nn_graph_dataset.py`.
- **Real recovery gap:** `NNRun.save()`'s three writes (run.yaml, metadata.yaml, idps.csv) were non-atomic. A Ctrl-C mid-write left half-written files. New `_atomic_write_text` helper does tmp + fsync + os.replace.
- `NNCheckpoint.to_file` had the same non-atomic gap (torch.save direct to destination). New `_atomic_torch_save` helper applies the same tmp + rename pattern to both the main checkpoint and the `.opt.pt` sidecar.
- Atomicity also applied to the Windows POINTER.txt fallback; helper reordered (defined before its caller); pyproject `filterwarnings` for the upstream `torch_geometric.distributed` / `torch.jit.script` DeprecationWarnings; fix the scheduler test's optimizer-before-scheduler step order so the runtime UserWarning doesn't fire.
- README "Other models" was a non-functional snippet (imported classes without showing how to wire them through `NNModel`). Replaced with concrete `NNModelParams(net=Nets.GRAPH_*)` examples + a pointer at the `examples/` folder. Added README subsections for Reproducibility, Warm-resume, and Custom metrics so the pass-2 features are visible from the top-level doc.
- `test_imports.py` was missing smoke imports for `nnx.seeding`, `nnx.nn.callbacks`, `nnx.nn.net.graph_nn_base`, `nnx.nn.dataset.nn_tabular_dataset`, and `nnx.nn.enum.schedulers`. The test predated pass-1 and never grew with the codebase. Closed the gap so the cheapest-possible refactor signal is exhaustive again.
- `release.yml` skipped `twine check` between `python -m build` and the PyPI upload step. A malformed README or invalid classifier would only surface when PyPI rejected the upload — by then the tag is burned. Added a `twine check dist/*` verification step; also added `cache: pip` to the setup-python step for parity with the other workflows.
- **R18-R19** — Final sweeps: ran the literal README quickstart end-to-end, manually exercised the four `predict()` input forms (ndarray, tensor, tuple-of-each), verified all internal markdown links resolve, and confirmed `mkdocs build --strict` is silent. No additional actionable findings.

### Deferred (with rationale)

- **D3** (split `NNModel.train()` into a `TrainingLoop` runner): the existing helpers (`_train_step`, `_save_checkpoints`, `_step_scheduler`, `_build_scheduler`, ...) already break the loop body into testable units. A full extraction would be churn without proportional value.
- **D7** (versioned state-dict checkpoint format with a versioned reader): too risky for this back-compat pass. The pickled `NNCheckpoint` continues to work; the `weights_only=False` security note in the docstring guards against the supply-chain risk.
- **D8** (Storage protocol for cloud backends): broad I/O abstraction touching every save/load site. Better as its own focused PR.
- **N5** (md5 of `str(state)` → `json.dumps(sort_keys=True)`): would change every existing `run.id`. Can't ship under strict back-compat.
- **O5 / O6 / O7** (NNTrainParams config-vs-runtime split, callbacks-as-params, NNModel `__init__` param rename): API breaks. Deferred.
- **O4** (frozen `_CallbackContext` view): would change the surface callbacks can mutate — defer to a callback API revision.
- **P1 / P2** (per-batch device sync, loss.item() sync): would sacrifice per-batch metric granularity (idp.train_edp). Deferred.
- **E7** (move ipython/kaleido to optional extras): would break `pip install nnx` for users relying on the default extras. Deferred.

## [Pass-1 unreleased] — comprehensive improvements pass 1

The pass-1 series landed on branch `chore/comprehensive-improvements-pass-1`. Strict back-compat preserved: no public API renames, no on-disk format breaks, deep imports still resolve.

### Fixed — correctness

- `NNDataset` now carves the validation slice out of the source `train=True` split, keeping the source `train=False` split intact for final evaluation. Previously val was a slice of test, leaking the test pool. Reported val metrics will differ between pre/post versions.
- `NNDataset` `random_split` sizes are computed as `(total - val, val)` instead of two truncated halves. Fixes the crash on odd-length source train sets.
- `NNRun.save` no longer crashes when comparing against a prior BEST run that has no `val_edp` (e.g., a no-validation experiment). A new `_best_err` helper falls back to train error, then `+inf`.
- `NNEvaluationDataPoint.of` now defaults `average="macro"` for `f1` / `recall` / `precision`. The prior `"micro"` hardcoding made all three numerically identical to accuracy for single-label multi-class tasks. Pass `average="micro"` to opt back in.
- `VisUtils.multi_line_plot`: removed dead `cs = px.colors.qualitative.Plotly[...]` assignment that was immediately overwritten; replaced the `ls[:len(ys)]` legend loop (which depended on a leaked inner-loop variable) with `n_lines_per_series = len(yss[0])`; raises `ValueError` on empty `yss`.
- `Activations.SOFTMAX` returns a closure that supplies `dim=-1`, avoiding the implicit-dim warning and ambiguity from `F.softmax`.
- `NNModel._train_step`: detach `train_loss` before `float()` to avoid `UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior`.

### Fixed — deprecations / future breakage

- Migrated `torch.cuda.amp.{autocast,GradScaler}` to `torch.amp.*` with explicit `device_type="cuda"`. The `torch.cuda.amp` module has been deprecated since torch 2.4.
- `NNCheckpoint.from_file` calls `torch.load(weights_only=False)`. Without it, torch ≥ 2.6 (where `weights_only` defaults to `True`) raises `UnpicklingError` on any saved `NNCheckpoint` (checkpoints pickle the full Python object, not a bare state dict).
- `NNGraphDataset` reads the underlying `Data` via `dataset[0]` instead of `dataset._data`. The private accessor was renamed/removed across PyG versions.
- Removed the top-level `from IPython.display import clear_output` import in `nn_model.py`. The actual use is in `callbacks._LegacyCallback`; leaving the top-level import made every consumer of `nnx.nn.nn_model` pull in IPython.

### Added

- `nnx/__init__.py` re-exports the curated public surface (`NNModel`, params, callbacks, enums, nets, datasets, utils) with an explicit `__all__`. Deep imports (`from nnx.nn.net.feed_fwd_nn import FeedFwdNN`) still work for existing code.
- `NNRun.save / load / all / checkpoints` and `NNCheckpoint.save / load` accept an optional `root: Optional[str] = None` kwarg. Default is unchanged (cwd-relative); callers wanting to redirect persistence can now pass one.
- `NNEvaluationDataPoint.of` accepts `average: str = "macro"`.
- `VisUtils.{multi_line_plot, scatter_plot, two_dim_tsne_checkpoint_logits, confusion_matrix}` now return the `plotly.graph_objects.Figure` they build. The `.show()` call is gated on a non-None renderer so headless test envs no longer crash.
- `tests/test_params_round_trip.py` — contract test asserting `obj == from_state(state())` for every params dataclass. Fails loudly when fields drift.
- `tests/test_train_integration.py` — end-to-end `NNModel.train()` coverage on a tiny in-memory `TensorDataset`, plus `NNRun.load` round-trip and `NNModel.from_checkpoint` reconstruction.
- `NNOptimParams.momentum` docstring explaining the SGD-vs-Adam dual meaning.
- `NNDataset` docstring documenting that val is carved from train.
- This `CHANGELOG.md`.

### Changed — tooling

- Ruff lint now selects `E`, `F`, `W`, `B` (bugbear), `I` (isort), `UP` (pyupgrade). Style-preserving ignores: `E701` (case style), `B024` (structural base class), `UP007` / `UP045` (keep `Optional` over `X | None`). 213 auto-fixes applied (mostly import ordering).
- CI matrix adds Python 3.12.
- CI ruff step no longer has `continue-on-error: true` — lint gates merges.

### Internal

- `nn_dataset_base.py`: trimmed 9 unused imports.
- `nn_model.py`: removed empty `class NNModel():` parens.
- `nn_dataset.py`: switched to a local `resolved_batch_sizes` so downstream loaders don't read `self.batch_sizes` while it still holds the default tuple.

## [0.1.0] — 2026-05-18

Initial extraction from `thekaveh/ml`.
