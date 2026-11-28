---
id: FEAT-084
title: "[FEAT-084] Replace two raw warnings.warn calls with _warn_at_user_frame"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-27"]
depends_on: ["FEAT-082", "FIX-107"]
related_issues: [522]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-084: two warnings use raw `warnings.warn` with a hand-picked `stacklevel`, while every other NNx warning goes through `_warn_at_user_frame`. The run chart's "history journal is unreadable" warning (`stacklevel=3`) points at NNx's own `NNRun._repr_html_`, not at the user's line, and both warnings show only once per location under Python's default filter. Switch both to `_warn_at_user_frame`, keeping their messages and categories.

## Context

**What happens today.** `NNRun._epoch_series` (`nn/params/nn_run.py` L996):

```python
            except (HistoryCorruptionError, OSError) as exc:
                warnings.warn(
                    f"run {self.id}'s history journal is unreadable ({exc}); the chart shows only the "
                    f"{len(self.idps or [])} records in memory, and their oldest epoch may be partial",
                    RuntimeWarning,
                    stacklevel=3,
                )
```

It is reached through `_repr_html_` (L938) → `_render_metric_chart_html` (L1041) → `_epoch_series`, so `stacklevel=3` lands on `_repr_html_` inside `nn_run.py`. `NNModel.to_onnx` (`nn/nn_model.py` L4398) warns about dropped tensors the same way, with `UserWarning, stacklevel=2`.

**Why it matters.** A warning attributed to NNx's source tells the user nothing about which of their calls caused it, and Python's default filter then shows it once per process for that NNx line, hiding later occurrences.

**History.** #522 set the precedent: NNx's run-reading warnings go through `_warn_at_user_frame` so they point at the caller and stay correct under `-W error`. FIX-107 rewrites the `to_onnx` dropped-state check and moves its warning after the file is written; it lands first, and this ticket converts the warning call as FIX-107 left it. The Hub "no manifest" warning is FIX-113's.

**Background for the worker.**
- *`_warn_at_user_frame(message, category=RuntimeWarning)`*: in `src/nnx/_warnings.py` (FEAT-082); warns at the first stack frame outside `nnx`, with a fresh warning registry per call, so the default filter shows it on every call and user filters (`ignore`, `error`, `once`) still decide.
- *`stacklevel`*: how many frames up `warnings.warn` attributes the warning; it breaks whenever the call path changes.

**Proposal.**

1. `src/nnx/nn/params/nn_run.py`, `NNRun._epoch_series`: replace the `warnings.warn(...)` call (L996–L1001) with
   ```python
                   _warn_at_user_frame(
                       f"run {self.id}'s history journal is unreadable ({exc}); the chart shows only the "
                       f"{len(self.idps or [])} records in memory, and their oldest epoch may be partial",
                   )
   ```
   (`RuntimeWarning` is the default category). `nn_run.py` imports `_warn_at_user_frame` at module top since FEAT-082. Delete `import warnings` only if ruff reports it unused.
2. `src/nnx/nn/nn_model.py`, `NNModel.to_onnx`: replace the dropped-tensor `warnings.warn(<message>, UserWarning, stacklevel=2)` call (L4398–L4403 at `9ad461e`; after FIX-107 find it by its "to_onnx():" message) with `_warn_at_user_frame(<same message>, UserWarning)`. Leave `nn_model.py`'s other `warnings.warn` (L1428, the Hub manifest warning) to FIX-113.
3. Add to `tests/test_history_journal.py`, below `test_the_chart_falls_back_to_the_window_when_the_journal_is_gone`:
   ```python
   def test_the_unreadable_journal_warning_points_at_the_callers_line():
       """FEAT-084: attributed to the user's call, not to NNRun._repr_html_."""
       import shutil

       run = _fit("model", HistoryJournal(retention=7, chunk_size=2))
       shutil.rmtree(run.history)
       with pytest.warns(RuntimeWarning, match="journal is unreadable") as caught:
           run._repr_html_()
       assert [w.filename for w in caught if "journal is unreadable" in str(w.message)] == [__file__]
   ```
4. Add to `tests/test_to_onnx_dropped_state.py`:
   ```python
   def test_the_dropped_tensor_warning_is_given_on_every_export_at_the_callers_line(tmp_path):
       """FEAT-084: one warning per export under the default filter, each at this file's line."""
       model = _feed_forward()
       model.net.register_buffer("class_prior", torch.tensor([0.25, 0.75]))
       with warnings.catch_warnings(record=True) as caught:
           warnings.simplefilter("default")
           for _ in range(2):
               model.to_onnx(str(tmp_path / "model.onnx"), torch.randn(2, 4), dynamic_batch=False)
       dropped = [w for w in caught if "class_prior" in str(w.message)]
       assert len(dropped) == 2 and all(w.filename == __file__ for w in dropped)
   ```
5. Add this bullet at the end of `### Fixed` under `## [Unreleased]` in `CHANGELOG.md`:
   ```markdown
   - **Two more warnings point at your code.** `to_onnx`'s dropped-tensor warning and the run chart's "history journal is unreadable" warning are attributed to the first caller outside NNx, like NNx's other warnings, and are shown on every call instead of once per NNx source line.
   ```

**Worker setup.** Branch from `develop` after FEAT-082 and FIX-107 have merged. `uv sync --frozen --all-extras` installs torch, `onnx` and every extra the tests need.

## Acceptance criteria

- [ ] Both new tests fail on `develop` (the journal warning names `nn_run.py`; the second export is not reported) and pass after the change. Check: `uv run pytest tests/test_history_journal.py::test_the_unreadable_journal_warning_points_at_the_callers_line tests/test_to_onnx_dropped_state.py::test_the_dropped_tensor_warning_is_given_on_every_export_at_the_callers_line`
- [ ] No `stacklevel=` is left in either function. Check: `grep -n "stacklevel" src/nnx/nn/params/nn_run.py` prints nothing, and `grep -n "stacklevel" src/nnx/nn/nn_model.py` prints no line inside `to_onnx`
- [ ] The existing message tests pass unchanged. Check: `uv run pytest tests -k "to_onnx_dropped_state or nn_run_repr_html or history_journal"`
- [ ] The CHANGELOG bullet from step 5 is under `## [Unreleased]` → `### Fixed`. Check: `grep -c "Two more warnings point at your code" CHANGELOG.md` prints `1`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_run.py L982-L1001](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L982-L1001) | `_epoch_series` and its `stacklevel=3` warning |
| [nn_run.py L920-L938](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L920-L938), [L1030-L1041](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L1030-L1041) | The call path `_repr_html_` → `_render_metric_chart_html` → `_epoch_series` |
| [nn_model.py L4394-L4403](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4394-L4403) | `to_onnx`'s `stacklevel=2` warning |
| [test_history_journal.py L1129-L1137](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_history_journal.py#L1129-L1137) | The existing journal-fallback test |
| [test_to_onnx_dropped_state.py L26-L37](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_to_onnx_dropped_state.py#L26-L37) | The existing dropped-tensor tests |
| #522 | The precedent: run-reading warnings through `_warn_at_user_frame` |

## Scope

**In:** the two warning calls, two tests, one CHANGELOG bullet.

**Out:** the Hub "no manifest" warning (FIX-113); which tensors `to_onnx` reports (FIX-107); other warnings.

## Dependencies

**Blocked by:** FEAT-082, FIX-107
**Blocks:** none
**Related:** #522
