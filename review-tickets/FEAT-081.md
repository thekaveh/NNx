---
id: FEAT-081
title: "[FEAT-081] Share the remaining epoch-end blocks between NNModel and Trainer"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-06"]
depends_on: []
related_issues: [536, 541]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-081: `NNModel._train_impl` and `Trainer._train_impl` still repeat three blocks word for word: the progress-bar construction, the empty-epoch guard, and the epoch close (attach the validation record, the epoch summary and the monitored value). Add three module-level helpers in `nn/nn_model.py`, `_training_bar`, `_require_epoch_batches` and `_close_epoch_records`, and call them from both loops. `records.settle_lrs()` stays in each loop, between the guard and validation. No behaviour changes.

## Context

**What happens today.** Both loops build the same bar (`nn_model.py` L5594 and L5598, `trainer.py` L753 and L756):

```python
        tqdm_disabled = os.environ.get("NNX_TQDM_DISABLE", "").lower() in {"1", "true", "yes"}
        with (
            torch.set_grad_enabled(True),
            tqdm(colour="blue", total=n_iter, desc="Training", disable=tqdm_disabled) as tqdm_bar,
```

raise the same error on an empty epoch, then settle the learning rates (`trainer.py` L866–L876; `nn_model.py` L5740–L5751):

```python
                if records.epoch_is_empty():
                    # Same guard as NNModel.train: zero batches would
                    # crash on records.last (first epoch) or corrupt the
                    # previous epoch's logged metrics (later epochs).
                    raise ValueError(
                        f"train_loader yielded no batches in epoch {idx_epoch} — check batch_size vs "
                        "dataset size with drop_last=True, or whether the loader is a one-shot iterable."
                    )
                # #541: as NNModel.train, a tensor learning rate's copies are
                # read back as floats once, before the callbacks and the commit.
                records.settle_lrs()
```

and, after their (different) validation, close the epoch identically (`trainer.py` L883–L892; `nn_model.py` L5786–L5794):

```python
                records.replace_last(records.last.with_val_edp(val_edp))
                record: Optional[MonitorRecord] = None
                if epoch_summary is not None:
                    train_summary = epoch_summary.result()
                    if tracker is not None:
                        assert monitor is not None
                        value = monitor.value(train=train_summary or train_edp, val=val_edp)
                        record = tracker.observe(value, epoch=idx_epoch)
                    records.replace_last(records.last.with_epoch_summary(train_summary, record))
```

**Why it matters.** A fix to one copy can miss the other; the trainer's own comment ("Same guard as NNModel.train") marks the duplicate.

**History.** #536 (FEAT-049) already shares `_resume_fit` and the `NNModel` methods `_commit_epoch`, `_save_final_last` and `_update_tqdm_postfix` between the loops; this ticket finishes that work and keeps the two loops separate by design (#536 did not merge them). #541 (FIX-073) added `records.settle_lrs()`, which must run once per epoch, before validation, the callbacks and the commit; it stays where it is in both loops.

**Background for the worker.**
- *`records`*: each loop's history object for the fit (`_training_history(...)`); `append`, `epoch_is_empty()`, `settle_lrs()`, `last` and `replace_last(...)` are its methods.
- *Epoch summary*: an optional whole-epoch training summary (`_TrainEpochSummary`, from `nnx.monitors`); `epoch_summary.result()` gives an `NNEvaluationDataPoint` or `None`.
- *Monitor / tracker*: the declared `MonitorSpec` and its `MonitorTracker`, which turns the epoch's monitored value into a `MonitorRecord` (best so far, improved or not).
- *`NNX_TQDM_DISABLE`*: an environment variable; `1`, `true` or `yes` in any letter case hides the progress bar (tests, CI, non-TTY runs).

**Proposal.**

1. In `src/nnx/nn/nn_model.py`, add these three functions directly after `_observe_epoch_summary` (ends L2735, before `_batch_sample_count`). The module already imports `os`, `tqdm`, `MonitorRecord`, `MonitorSpec`, `MonitorTracker` and `_TrainEpochSummary`.
   ```python
   def _training_bar(n_iter: int) -> tqdm:
       """The progress bar both training loops show: ``n_iter`` steps, blue,
       titled "Training", hidden when ``NNX_TQDM_DISABLE`` is ``1``, ``true`` or
       ``yes`` (any case), as in tests, CI and non-TTY runs. Use it as a context
       manager."""
       disabled = os.environ.get("NNX_TQDM_DISABLE", "").lower() in {"1", "true", "yes"}
       return tqdm(colour="blue", total=n_iter, desc="Training", disable=disabled)


   def _require_epoch_batches(records: Any, idx_epoch: int) -> None:
       """Refuse an epoch whose train loader yielded no batch (both loops): the
       first epoch would crash on ``records.last``, a later one would attach its
       validation record to the previous epoch's last record. The caller then
       settles the learning rates (#541) before validating."""
       if records.epoch_is_empty():
           raise ValueError(
               f"train_loader yielded no batches in epoch {idx_epoch} — check batch_size vs "
               "dataset size with drop_last=True, or whether the loader is a one-shot iterable."
           )


   def _close_epoch_records(
       records: Any,
       val_edp: Optional[NNEvaluationDataPoint],
       train_edp: Any,
       epoch_summary: Optional[_TrainEpochSummary],
       tracker: Optional[MonitorTracker],
       monitor: Optional[MonitorSpec],
       idx_epoch: int,
   ) -> Optional[MonitorRecord]:
       """Close an epoch's records after validation (both loops): attach
       ``val_edp`` to the epoch's last iteration record and, when the loop keeps
       an epoch summary, that summary and the monitored value ``tracker``
       observed (from the summary, else ``train_edp`` — the last batch's record —
       and ``val_edp``). Returns the monitor record, or ``None`` when nothing was
       monitored."""
       records.replace_last(records.last.with_val_edp(val_edp))
       record: Optional[MonitorRecord] = None
       if epoch_summary is not None:
           train_summary = epoch_summary.result()
           if tracker is not None:
               assert monitor is not None
               value = monitor.value(train=train_summary or train_edp, val=val_edp)
               record = tracker.observe(value, epoch=idx_epoch)
           records.replace_last(records.last.with_epoch_summary(train_summary, record))
       return record
   ```
   The error message is word for word today's, so every test matching `yielded no batches` keeps passing.
2. In `NNModel._train_impl`:
   - delete the four lines L5591–L5594 (the three comment lines and `tqdm_disabled = ...`) and change L5598 to `            _training_bar(n_iter) as tqdm_bar,`;
   - replace L5740–L5748 (the `if records.epoch_is_empty():` block with its comment) with `                _require_epoch_batches(records, idx_epoch)`; keep the `#541` comment and `records.settle_lrs()` (L5749–L5751) right after it;
   - replace L5786–L5794 (from `records.replace_last(records.last.with_val_edp(val_edp))` through the inner `records.replace_last(...with_epoch_summary...)`) with `                record = _close_epoch_records(records, val_edp, train_edp, epoch_summary, tracker, monitor, idx_epoch)`.
3. In `src/nnx/trainer/trainer.py`:
   - add `_close_epoch_records`, `_require_epoch_batches` and `_training_bar` to the `from ..nn.nn_model import (...)` list (L61–L86, sorted);
   - delete L753 (`tqdm_disabled = ...`) and change L756 to `            _training_bar(n_iter) as tqdm_bar,`;
   - replace L866–L873 (the guard and its "Same guard as NNModel.train" comment) with `                _require_epoch_batches(records, idx_epoch)`; keep the `#541` comment and `records.settle_lrs()` (L874–L876);
   - replace L883–L892 with `                record = _close_epoch_records(records, val_edp, train_edp, epoch_summary, tracker, monitor, idx_epoch)`;
   - delete the `tqdm` and `os` imports only if ruff reports them unused.
4. The per-batch `NNIterationDataPoint` append (`nn_model.py` L5703–L5712, `trainer.py` L851–L862) stays inline: it runs on every batch and the two copies differ in which optimizer's rate they record.
5. Add `tests/test_epoch_end_helpers.py`:
   ```python
   """FEAT-081: the epoch-end helpers NNModel.train and Trainer.train share."""

   from types import SimpleNamespace

   import pytest

   from nnx.nn.nn_model import _close_epoch_records, _require_epoch_batches, _training_bar


   class _Records:
       def __init__(self, empty=False):
           self.empty, self.last = empty, SimpleNamespace(val=None, summary=None)

       def epoch_is_empty(self):
           return self.empty

       def replace_last(self, value):
           self.last = value


   def _record(**fields):
       record = SimpleNamespace(**fields)
       record.with_val_edp = lambda val: _record(**{**fields, "val": val})
       record.with_epoch_summary = lambda summary, monitored: _record(**{**fields, "summary": (summary, monitored)})
       return record


   @pytest.mark.parametrize(("value", "disabled"), [("1", True), ("TRUE", True), ("yes", True), ("0", False), ("", False)])
   def test_training_bar_reads_nnx_tqdm_disable(monkeypatch, value, disabled):
       monkeypatch.setenv("NNX_TQDM_DISABLE", value)
       with _training_bar(3) as bar:
           assert bar.total == 3 and bar.disable is disabled


   def test_require_epoch_batches_names_the_epoch():
       _require_epoch_batches(_Records(empty=False), 0)
       with pytest.raises(ValueError, match=r"train_loader yielded no batches in epoch 4 — check batch_size"):
           _require_epoch_batches(_Records(empty=True), 4)


   def test_close_epoch_records_attaches_validation_summary_and_monitor_record():
       records = _Records()
       records.last = _record(val=None, summary=None)
       assert _close_epoch_records(records, "val", "train", None, None, None, 2) is None
       assert records.last.val == "val" and records.last.summary is None
       summary = SimpleNamespace(result=lambda: "summary")
       monitor = SimpleNamespace(value=lambda train, val: (train, val))
       tracker = SimpleNamespace(observe=lambda value, epoch: ("observed", value, epoch))
       records.last = _record(val=None, summary=None)
       record = _close_epoch_records(records, "val", "train", summary, tracker, monitor, 2)
       assert record == ("observed", ("summary", "val"), 2)
       assert records.last.val == "val" and records.last.summary == ("summary", record)
   ```

No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new tests in `tests/test_epoch_end_helpers.py` pass (they fail on `develop`, where the helpers do not exist). Check: `uv run pytest tests/test_epoch_end_helpers.py`
- [ ] The duplicated blocks are gone from both loops. Check: `grep -c "NNX_TQDM_DISABLE\|yielded no batches\|with_epoch_summary" src/nnx/trainer/trainer.py` prints `0`, and the same command on `src/nnx/nn/nn_model.py` prints `4`, all inside the three helpers
- [ ] `records.settle_lrs()` still runs once per epoch in each loop, after the guard and before validation. Check: `grep -n "_require_epoch_batches(records\|records.settle_lrs()" src/nnx/nn/nn_model.py src/nnx/trainer/trainer.py` shows each `settle_lrs` line right after a `_require_epoch_batches` line
- [ ] Both loops behave as before. Check: `uv run pytest tests -k "trainer or train_integration or named_monitors or history_journal or scheduler_clocks or causal_lm_task or ranking_task"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L5591-L5598](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5591-L5598), [trainer.py L753-L756](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/trainer.py#L753-L756) | The same bar, built twice |
| [nn_model.py L5740-L5751](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5740-L5751), [trainer.py L866-L876](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/trainer.py#L866-L876) | The empty-epoch guard and `settle_lrs` |
| [nn_model.py L5786-L5794](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5786-L5794), [trainer.py L883-L892](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/trainer.py#L883-L892) | The epoch close |
| [nn_model.py L5703-L5712](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5703-L5712), [trainer.py L851-L862](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/trainer.py#L851-L862) | The per-batch append, which stays |
| [nn_model.py L2705-L2737](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2705-L2737) | `_enumerate_with_last` and `_observe_epoch_summary`, the shared loop helpers the new ones join |
| [trainer.py L61-L86](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/trainer.py#L61-L86) | `Trainer`'s private imports from `nn_model` |
| #536, #541 | The shared resume/commit path; `settle_lrs` |

## Scope

**In:** the three helpers, their use in both loops, the new test file.

**Out:**
- Merging the two loops (#536 keeps them separate).
- The per-batch append (hot path; the copies differ).
- Extracting `_train_impl`'s other blocks: FEAT-079 (lands after this ticket).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-079, FEAT-074
**Related:** #536, #541
