# 15. Architecture

## 1. Package and lifecycle overview

NNx is organized around two public entry points (`NNModel` / `Trainer`), a
training-hook family (`train_step_fn`, `eval_step_fn`, and
`trainer_step_fn`), and content-addressed persistence under `runs/<id>/`.
Hook-producing modules inject behavior into the orchestrators; model transforms,
exporters, inference helpers, and diagnostics compose around them. The
inference branch, `nnx.decisions`, sits beside the training path: typed
questions go to a `DecisionProvider` (for a trained classifier,
`FixedHeadProvider` over `NNModel.predict_proba`) and come back as validated
results ([Typed decisions](decisions.md)). The training
loop owns callback dispatch, scheduler updates (once per epoch by default; once
per committed optimizer update for a `clock="optimizer_update"` scheduler),
phase checkpoint cadence, and incremental `NNRun` persistence.

See [Concepts §1](concepts.md#1-architecture) for the full written breakdown.

![NNx architecture](assets/architecture.png)

## 2. Lifecycle order

For each successfully started training run, NNx calls `on_train_begin`, then
dispatches epoch and batch work. A completed epoch aggregates validation through
the built-in path or `eval_step_fn`, updates each epoch-clock scheduler once,
and dispatches `on_epoch_end`. An `optimizer_update`-clock scheduler is stepped
instead right after each committed update of its optimizer — reported by the
update engine, by `default_train_step`, by `finalize_step` (the built-in
paradigm steps), or by a step function calling
`ctx.report_update(...)` — never per microbatch, masked window or skipped step;
an objective's `on_optimizer_update` callbacks see every optimizer's event of a
commit before any clock steps. A run that declares metrics or a named monitor also records the
whole-epoch training summary and the monitor's decision before the scheduler
update; that one decision drives the plateau scheduler, BEST and any
`EarlyStopping` given the same monitor
([Concepts §6.4](concepts.md#64-named-metrics-and-monitors)). In an objective
run the shared update engine accumulates each microbatch's loss terms and, at
the end of every update window, runs unscale → clip → step and dispatches
`on_optimizer_update` once per committed update — before the epoch's
validation and the commit order above
([Concepts §6.5](concepts.md#65-objectives-and-the-shared-update-engine)). Durable state then commits in order: run history, LAST,
phase/BEST, and deferred callback checkpoints. Finalization calls `on_train_end`
in reverse callback order. Both `NNModel` and `Trainer` refresh LAST after
finalization so callback mutations and topology-transform metadata are present
in the persisted checkpoint.

On failure, callbacks whose begin hook completed are still finalized. Every
cleanup hook is attempted; cleanup errors do not mask an exception already
raised by training. A failed LAST commit rolls history back; failures after LAST
retain the durable history/checkpoint pair. On load, history newer than LAST is
truncated, while an empty or corrupt LAST is rejected rather than treated as a
request to erase history. Each text file is replaced on its own through an owned
temporary file: a failed write leaves that destination's previous bytes and
removes its temporary, but the history files are not one multi-file
transaction, and a hard kill can still leave a stale temporary behind.

With an opt-in history journal (`history=HistoryJournal(...)`,
[Concepts §4.5](concepts.md#45-bounded-history-the-history-journal)) the
loop keeps a bounded window of records in memory, and the "run history" step
appends the epoch's records as immutable chunks, indexes them and publishes
the journal manifest — still before LAST, which stays the commit marker.
Readers show records up to LAST's epoch only, so a crash between the
manifest and LAST leaves an uncommitted tail that is ignored; a failed LAST
republishes the previous manifest. Callbacks see the window as `ctx.idps`
unless they declare `history_access = "full"`, and a continuation writes its
own run and chunk files, never its source's.

![NNx training lifecycle](assets/training-lifecycle.png)
