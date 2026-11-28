---
id: FEAT-103
title: "[FEAT-103] Remove small duplicates in nn_model and nn_run"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-22", "SMELL-24", "SMELL-28"]
depends_on: []
related_issues: []
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-103: Five small copies: `nn_model._component_type` re-implements `models._qualified`; `nn_run` keeps its own copies of provenance's two file-name constants; `nn_run._point_best` re-imports `shutil`, `_epoch_series` imports `defaultdict` locally, and `_epoch_series` spells out the series names that `history._SERIES` already defines. This ticket replaces each copy with the original. Behaviour does not change.

## Context

**What happens today.**
1. `nn_model.py` L203–L205 and `models.py` L379–L380:

   ```python
   def _component_type(value: Any) -> str:
       cls = type(value)
       return f"{cls.__module__}.{cls.__qualname__}"
   ```

   ```python
   def _qualified(kind: type) -> str:
       return f"{kind.__module__}.{kind.__qualname__}"
   ```

2. `nn_run.py` L51–L52, used only at L341 (`_release_empty_reservation`):

   ```python
   _PROVENANCE_MANIFEST_FILE = "provenance.json"  # nnx.provenance.MANIFEST_FILE
   _PROVENANCE_ATTEMPT_FILE = "attempt.json"  # nnx.provenance.ATTEMPT_FILE
   ```

   while `provenance.py` L83–L84 defines `MANIFEST_FILE = "provenance.json"` and `ATTEMPT_FILE = "attempt.json"`, and `nn_run` already imports `provenance` at the top (L30).
3. `_point_best` imports `shutil` inside a branch (L529) although `nn_run` imports it at the top (L10). `NNRun._epoch_series` imports `defaultdict` locally (L1002) and builds its series from a literal (L1007–L1009):

   ```python
           series: dict[str, list[Any]] = {
               name: [] for name in ("epochs", "train_loss", "train_err", "val_loss", "val_err", "monitor", "improved")
           }
   ```

   `history.py` defines `_SERIES = ("train_loss", "train_err", "val_loss", "val_err", "monitor")` (L101) and builds the journal's series as `("epochs", *_SERIES, "improved")` (L755).

**Why it matters.** A copied constant or name list drifts silently: a new series in `history._SERIES` would appear in a journal run's chart but not in a CSV run's.

**History.** No board issue covers these.

**Background for the worker.**
- `_component_type(value)`: the qualified class name a training-state sidecar records for an optimizer or scheduler (`"torch.optim.sgd.SGD"`), compared on resume.
- `_release_empty_reservation`: removes a run directory that holds only bookkeeping files, among them `provenance.json` and `attempt.json`.
- `_epoch_series`: the per-epoch chart data of a CSV run (`NNRun._repr_html_`); a journal run gets the same keys from `history._journal_epoch_series`.

**Proposal.**
1. In `src/nnx/nn/nn_model.py`, add `_qualified` to the top-level `from ..models import (...)` (L58) and replace the body of `_component_type` (L204–L205) with `return _qualified(type(value))`.
2. In `src/nnx/nn/params/nn_run.py`:
   - change L30 to `from ...provenance import ATTEMPT_FILE, MANIFEST_FILE, ProvenanceRecord, load_provenance`;
   - delete L51–L52 and, in `_release_empty_reservation` (L341), use `MANIFEST_FILE` and `ATTEMPT_FILE`: `bookkeeping = (_HISTORY_PROTOCOL_FILE, MANIFEST_FILE, ATTEMPT_FILE, ".run.lock")`. No test reads the deleted names (`grep -rn "_PROVENANCE_MANIFEST_FILE\|_PROVENANCE_ATTEMPT_FILE" tests` returns nothing);
   - delete the local `import shutil` in `_point_best` (L529) and the blank line after it;
   - move `from collections import defaultdict` from `_epoch_series` (L1002, with the blank line after it) to the top-level imports (`uv run ruff check --fix` places it above `from collections.abc import ...`);
   - add `_SERIES` to the top-level `from ...history import ...` (L28), and replace the series expression (L1008) with `name: [] for name in ("epochs", *_SERIES, "improved")`.
3. No behaviour or message changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `_component_type` delegates. Check: `grep -n "return _qualified(type(value))" src/nnx/nn/nn_model.py` returns 1 line
- [ ] The provenance file names are not copied. Check: `grep -rn '"provenance.json"\|"attempt.json"' src/nnx/nn` returns 0 lines
- [ ] No local `shutil` or `defaultdict` import is left in `nn_run.py`. Check: `grep -nE "^\s+(import shutil|from collections import defaultdict)" src/nnx/nn/params/nn_run.py` returns 0 lines
- [ ] The CSV chart series come from `history._SERIES`. Check: `grep -n '"epochs", \*_SERIES, "improved"' src/nnx/nn/params/nn_run.py` returns 1 line
- [ ] The suites that cover these paths pass unchanged. Check: `uv run pytest tests/test_component_resume.py tests/test_attempt_provenance.py tests/test_nn_run_repr_html.py tests/test_best_pointer_readers.py tests/test_exact_resume.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L203-L205](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L203-L205) | `_component_type` |
| [models.py L379-L380](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L379-L380) | `_qualified` |
| [nn_run.py L28-L30](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L28-L30) | the top-level `history` and `provenance` imports |
| [nn_run.py L51-L52](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L51-L52), [L341](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L341) | the copied provenance file names and their use |
| [provenance.py L83-L84](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/provenance.py#L83-L84) | `MANIFEST_FILE`, `ATTEMPT_FILE` |
| [nn_run.py L529](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L529), [L10](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L10) | the second `import shutil` |
| [nn_run.py L1002-L1009](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L1002-L1009) | `_epoch_series`: local `defaultdict`, the literal series names |
| [history.py L101](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L101), [L755](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L755) | `_SERIES` and the journal's series |

## Scope

**In:** the five replacements above.

**Out:** the raw `warnings.warn` in `_epoch_series` (FEAT-084). The run-directory file names `run.yaml`, `metadata.yaml`, `idps.csv` (FEAT-098). Other local imports in `nn_model.py` (FEAT-095).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-074
**Related:** FEAT-084, FEAT-098
