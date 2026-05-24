# Changelog

All notable changes to NNx are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is roughly [SemVer](https://semver.org/) — pre-1.0, we allow behavior changes (typically bug fixes) without renaming public APIs.

## [Unreleased] — comprehensive improvements pass

A correctness + deprecations + tests + docs sweep landing on a feature branch (`chore/comprehensive-improvements-pass-1`). Strict back-compat preserved: no public API renames, no on-disk format breaks, deep imports still resolve.

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
