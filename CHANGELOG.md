# Changelog

All notable changes to NNx are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is roughly [SemVer](https://semver.org/) — pre-1.0, we allow behavior changes (typically bug fixes) without renaming public APIs.

## [Unreleased] — comprehensive improvements pass 2

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
