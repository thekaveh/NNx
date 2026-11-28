---
id: FEAT-079
title: "[FEAT-079] Extract four helpers from NNModel._train_impl"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-03"]
depends_on: ["FEAT-081"]
related_issues: [536, 576, 765, 540, 768]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-079: `NNModel._train_impl` is 645 lines (cyclomatic complexity 102) and defines a nested function and a nested class on every call. Extract four helpers by pure extract-method: the update announcer and reporter become module-level classes, and the validation dispatch, the per-batch stop agreement and the epoch commit become private helpers. No behaviour, message or ordering changes; the two training loops stay separate.

## Context

**What happens today.** On every call, `_train_impl` builds a closure and a class that share mutable locals (`nn_model.py` L5523–L5565):

```python
            listener_errors: list[BaseException] = []
            ddp_listening = False  # agreed at each epoch's start, below (#768)

            def _announce_update() -> None:
                # FEAT-033: count the committed update and tell the listeners.
                ctx.committed_updates += 1
```

```python
            class _ReportUpdate:
                """The step's ``report_update``: the clock's (refusing an
                optimizer name, as ever), then the update listeners. ``listening``
                says whether anything needs a judged step (``listens``), and
                ``skip_window()`` counts a built-in step's window with nothing to
                learn from, which reports no update (#513)."""
```

`ddp_listening` is rebound at each epoch's start (L5659) and read by the closure and by the per-batch stop agreement (L5717–L5735). The validation dispatch (L5753–L5785) and the epoch commit (L5835–L5885) are long inline blocks.

**Why it matters.** The function is the hardest code in the package to review; per-call class definitions also cost time on every `train()`.

**History.** #536 (FEAT-049) shares the resume and commit path with `Trainer`; the loops stay separate by design. #576 (FEAT-054) and #765 (DEC-011) shaped the per-step update path, which this ticket does not touch (no closure is added to the per-batch hot path). #540 and #768 defined the DDP listener-error hold and the epoch-start agreement of `ddp_listening`, which the extracted announcer keeps exactly. FEAT-081 extracts the empty-epoch guard, the epoch close and the progress bar first; this ticket builds on it.

**Narrowed from the finding.** The finding also proposed extracting the whole resume-window and reporter setup (L5468–L5594) and the whole batch-loop body (L5675–L5735) into methods. Those blocks set or rebind about fifteen locals each; as methods they would return large tuples, and the batch body runs on every batch. This ticket extracts the self-contained parts of them instead (helpers 1 and 3 below).

**Background for the worker.**
- *Update listeners*: callables on `ctx.update_listeners` called after each committed optimizer update (FEAT-033, e.g. a search's pruner); `ctx.committed_updates` counts the updates.
- *`ddp_listening`*: under DDP, whether any rank listens for updates; agreed at each epoch's start (#768). While it is set, a listener's error is held and agreed at the batch's stop gather (#540) instead of raising on one rank only.
- *Report-update object*: the `report_update` callable a train step receives; it refuses an optimizer name, closes the resume window (#507), reports to the update clock, then announces the update. Steps also read its `listening` property and call `skip_window()` (#513).
- *Extract-method*: moving a block into a function unchanged, with the block's inputs as parameters and its outputs as return values; no statement is reordered.

**Proposal.**

1. **Helper 1: the announcer and the reporter, at module level.** In `src/nnx/nn/nn_model.py`, add after `class _ObjectiveStep` (ends L3139, before `def _objective_engine`):
   ```python
   class _UpdateAnnouncer:
       """Counts each committed update on ``ctx`` and calls its update listeners
       (FEAT-033). Under DDP, ``ddp_listening`` is agreed at each epoch's start
       (#768); while it is set, a listener's error is held in ``errors`` until
       the batch's stop gather, which every rank reaches (#540)."""

       def __init__(self, ctx: Any) -> None:
           self.ctx = ctx
           self.ddp_listening = False
           self.errors: list[BaseException] = []

       def __call__(self) -> None:
           ctx = self.ctx
           ctx.committed_updates += 1
           for listener in list(ctx.update_listeners):
               if not self.ddp_listening:
                   listener(ctx)
                   continue
               try:
                   listener(ctx)
               except Exception as error:
                   self.errors.append(error)
                   break


   class _ReportUpdate:
       """The step's ``report_update``: the clock's (refusing an optimizer name,
       as ever), then the update listeners. ``listening`` says whether anything
       needs a judged step (``listens``), and ``skip_window()`` counts a built-in
       step's window with nothing to learn from, which reports no update (#513)."""

       def __init__(
           self, clock_report: Callable[..., None], resume_window: Any, ctx: Any, announce: Callable[[], None]
       ) -> None:
           self.clock_report = clock_report
           self.resume_window = resume_window
           self.ctx = ctx
           self.announce = announce
           self.skipped_windows = 0

       def skip_window(self) -> None:
           self.skipped_windows += 1

       def __call__(self, *names: Any) -> None:
           # A refused report is no update: refuse before the window
           # closes, as Trainer checks its name first (#507).
           refuse_optimizer_name(names)
           self.resume_window.close()  # an update NNx sees: a resume's rollback window ends (#507)
           self.clock_report(*names)
           self.announce()

       @property
       def listening(self) -> bool:
           return listens(self.clock_report) or bool(self.ctx.update_listeners)
   ```
   In `_train_impl`:
   - replace L5523–L5537 (`listener_errors: ...`, `ddp_listening = False ...` and the whole `def _announce_update`) with `            announcer = _UpdateAnnouncer(ctx)`; keep the `#540` comment above it;
   - replace L5539–L5564 (the nested `class _ReportUpdate` and `update_reporter = _ReportUpdate()`) with `            update_reporter = _ReportUpdate(clock_report, resume_window, ctx, announcer)`; L5565 (`report_update: Callable[..., None] = update_reporter`) stays;
   - L5571: `engine.listeners.append(lambda event: _announce_update())` → `engine.listeners.append(lambda event: announcer())`;
   - L5659: `ddp_listening = any(` → `announcer.ddp_listening = any(`.
2. **Helper 2: the validation dispatch.** Add an `NNModel` method directly before `_commit_epoch` (L6562):
   ```python
       def _validate_epoch(
           self, params: NNTrainParams, *, validate: bool, ddp: Any, eval_step_fn: Any,
           compile_session: Optional[_CompileSession], idx_epoch: int,
       ) -> Optional[NNEvaluationDataPoint]:
           """The epoch's validation record, or ``None`` when the fit does not
           validate: each rank's shard gathered under DDP (FEAT-030), the caller's
           ``eval_step_fn`` in eval mode (#86), or the default validation."""
   ```
   whose body is L5753–L5785 unchanged, followed by `        return val_edp`. In `_train_impl`, replace L5753–L5785 with:
   ```python
                   val_edp = self._validate_epoch(
                       params, validate=validate, ddp=ddp, eval_step_fn=eval_step_fn,
                       compile_session=compile_session, idx_epoch=idx_epoch,
                   )
   ```
3. **Helper 3: the per-batch stop agreement.** Add, after `_ReportUpdate`:
   ```python
   def _stop_at_update(ctx: Any, ddp: Any, announcer: _UpdateAnnouncer, *, idx_epoch: int, is_last_batch: bool) -> bool:
       """After each batch: agree a stop requested at an update on every rank
       (gathered only when some rank listens; a held listener error is agreed
       too, #540). A stop on the epoch's last batch sets ``ctx.should_stop`` (the
       epoch is committed, then the fit stops); a stop before it returns
       ``True``: the caller discards the epoch in progress and stops (FEAT-033).
       Under DDP with no rank listening, the epoch-end gather decides instead."""
       if ddp is not None and announcer.ddp_listening:
           held = announcer.errors.pop() if announcer.errors else None
           ctx.stop_at_update = any(
               ddp.gather_agreed(bool(ctx.stop_at_update), held, f"epoch {idx_epoch}'s update listeners")
           )
       if ctx.stop_at_update and (ddp is None or announcer.ddp_listening):
           if is_last_batch:
               ctx.should_stop = True  # the epoch is complete: commit it, then stop
           else:
               return True
       return False
   ```
   In `_train_impl`, replace L5717–L5735 with:
   ```python
                       if _stop_at_update(ctx, ddp, announcer, idx_epoch=idx_epoch, is_last_batch=is_last_batch):
                           # FEAT-033: stop at this update boundary; the epoch in
                           # progress is discarded, never committed.
                           stopped_mid_epoch = True
                           records.discard_epoch()
                           break
   ```
4. **Helper 4: the epoch commit on every rank.** Add an `NNModel` method directly after `_validate_epoch`:
   ```python
       def _commit_epoch_everywhere(
           self, *, run, ctx, records, idx_epoch, local_epoch, params, counters, best_checkpoint, optimization,
           scaler, precision, compile_session, train_loader, registry, record, writes: bool, ddp: Any,
       ) -> tuple[NNRun, Optional[NNCheckpoint]]:
           """Commit the epoch: stamp the run with the compile record as it stands
           (FEAT-029), gather every rank's RNG streams under DDP (FEAT-030), let
           the writer commit (history first, then the checkpoint, the commit
           marker; #536), and agree the writer's outcome on every rank. Returns
           the stamped run and the best checkpoint (unchanged on a replica)."""
   ```
   Annotate each parameter as `_commit_epoch` annotates its same-named parameter (L6562 onward). Its body is L5835–L5885 unchanged, except that the `self._commit_epoch(...)` call passes `local_epoch=local_epoch` and `counters=counters` (the two expressions move to the call site below), followed by `        return run, best_checkpoint`. In `_train_impl`, keep the comment lines L5829–L5834 and replace L5835–L5885 with:
   ```python
                   run, best_checkpoint = self._commit_epoch_everywhere(
                       run=run,
                       ctx=ctx,
                       records=records,
                       idx_epoch=idx_epoch,
                       # Phase tags follow the plan's logical epochs on a
                       # planned resume (#394), the session's otherwise.
                       local_epoch=idx_epoch if params.resume_epochs == "planned" else local_epoch,
                       params=params,
                       counters={
                           "global_step": idx_iter,
                           "committed_updates": ctx.committed_updates,
                           "planned_n_epochs": start_epoch + session_epochs,
                       },
                       best_checkpoint=best_checkpoint,
                       optimization=optimization,
                       scaler=scaler,
                       precision=precision,
                       compile_session=compile_session,
                       train_loader=train_loader,
                       registry=registry,
                       record=record,
                       writes=writes,
                       ddp=ddp,
                   )
   ```
   Both expressions are pure, so computing them on a DDP replica too changes nothing.
5. Line numbers are at `9ad461e`; FEAT-081 shifts them. Find each block by the quoted code.
6. No test patch needs a new target: the helpers stay in `nn_model.py`, so patches of `nn_model` globals (`_scale_gradients`, `_check_finite_gradients`, `resolve_precision`, `_resume_fit`, `_best_err`, `_training_history`, `warn_unowned_trainables`, `_capture_training_modes`) and of `NNModel` methods (`_update_tqdm_postfix` in `tests/test_history_journal.py` L325) see the same calls.
7. Add `tests/test_train_impl_helpers.py`:
   ```python
   """FEAT-079: the helpers extracted from NNModel._train_impl."""

   from types import SimpleNamespace

   import pytest

   from nnx.nn.nn_model import NNModel, _ReportUpdate, _stop_at_update, _UpdateAnnouncer


   def test_the_announcer_counts_and_holds_a_listener_error_only_under_ddp_listening():
       calls = []

       def failing(ctx):
           calls.append("failing")
           raise RuntimeError("listener")

       ctx = SimpleNamespace(committed_updates=0, update_listeners=[failing, lambda ctx: calls.append("next")])
       announcer = _UpdateAnnouncer(ctx)
       with pytest.raises(RuntimeError, match="listener"):
           announcer()
       announcer.ddp_listening = True
       announcer()
       assert ctx.committed_updates == 2 and [str(e) for e in announcer.errors] == ["listener"]
       assert calls == ["failing", "failing"]  # the held error stops the round


   def test_the_reporter_closes_the_window_reports_and_announces():
       events = []
       window = SimpleNamespace(close=lambda: events.append("close"))
       ctx = SimpleNamespace(update_listeners=[])
       reporter = _ReportUpdate(lambda *names: events.append("clock"), window, ctx, lambda: events.append("announce"))
       reporter()
       reporter.skip_window()
       assert events == ["close", "clock", "announce"] and reporter.skipped_windows == 1


   @pytest.mark.parametrize(("is_last_batch", "discard", "should_stop"), [(True, False, True), (False, True, False)])
   def test_a_stop_at_an_update_commits_on_the_last_batch_and_discards_before_it(is_last_batch, discard, should_stop):
       ctx = SimpleNamespace(stop_at_update=True, should_stop=False)
       announcer = _UpdateAnnouncer(ctx)
       assert _stop_at_update(ctx, None, announcer, idx_epoch=0, is_last_batch=is_last_batch) is discard
       assert ctx.should_stop is should_stop


   def test_train_impl_defines_no_class_or_announcer_per_call():
       import ast
       import inspect
       import textwrap

       tree = ast.parse(textwrap.dedent(inspect.getsource(NNModel._train_impl)))
       nested = [node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef))][1:]
       assert nested == ["_plan_clock"]
   ```

No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FEAT-081 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need. Run the listed suites before and after the change; both runs must pass.

## Acceptance criteria

- [ ] The new tests in `tests/test_train_impl_helpers.py` pass (they fail on `develop`, where the helpers do not exist). Check: `uv run pytest tests/test_train_impl_helpers.py`
- [ ] `_train_impl` keeps only `_plan_clock` as a nested definition and no longer references the old locals. Check: `grep -n "ddp_listening\b\|listener_errors\|_announce_update" src/nnx/nn/nn_model.py` lists only lines inside `_UpdateAnnouncer`, `_stop_at_update` and the `announcer.ddp_listening` uses
- [ ] Training, resume, DDP, search and comparison behaviour is unchanged. Check: `uv run pytest tests -k "train_integration or trainer or exact_resume or resume_parity or ddp or run_comparison or named_monitors or search or history_journal or eval_step_fn"`
- [ ] The full suite passes. Check: `uv run pytest -n logical --dist loadgroup`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L5318-L5962](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5318-L5962) | `NNModel._train_impl` |
| [nn_model.py L5520-L5571](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5520-L5571) | The per-call `_announce_update` closure and `_ReportUpdate` class |
| [nn_model.py L5652-L5661](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5652-L5661) | `ddp_listening` agreed at each epoch's start (#768) |
| [nn_model.py L5717-L5738](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5717-L5738) | The per-batch stop agreement |
| [nn_model.py L5753-L5785](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5753-L5785) | The validation dispatch |
| [nn_model.py L5829-L5885](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5829-L5885) | The epoch commit and its DDP agreement |
| [nn_model.py L3117-L3141](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L3117-L3141) | `_ObjectiveStep`, after which the module-level classes go |
| #536, #576, #765, #540, #768 | Shared commit path; per-step update path; DDP listener rules |

## Scope

**In:** the four helpers and their use in `NNModel._train_impl`; one test file.

**Out:**
- `Trainer._train_impl`, `NNModel.train`, `default_train_step`, `_resume_fit` and the other complexity hotspots the review listed (not ticketed).
- Extracting the whole resume-window setup or the whole batch body (see "Narrowed").
- The epoch-end blocks shared with `Trainer`: FEAT-081. Moving the resume code: FEAT-074.

## Dependencies

**Blocked by:** FEAT-081
**Blocks:** FEAT-074
**Related:** #536, #576, #765, #540, #768
