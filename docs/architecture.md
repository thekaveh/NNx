# Architecture

## 1. Package and lifecycle overview

NNx is organized around two public entry points (`NNModel` / `Trainer`), a
training-hook family (`train_step_fn`, `eval_step_fn`, and
`trainer_step_fn`), and content-addressed persistence under `runs/<id>/`.
Hook-producing modules inject behavior into the orchestrators; model transforms,
exporters, inference helpers, and diagnostics compose around them. The training
loop owns callback dispatch, once-per-epoch scheduler updates, phase checkpoint
cadence, and incremental `NNRun` persistence.

See [Concepts §1](concepts.md#1-architecture) for the full written breakdown.

![NNx architecture](assets/architecture.svg)

!!! tip "Standalone version"
    Open the [standalone diagram page](architecture.html){target="_blank" rel="noopener"}
    in a new tab — it renders the same flow with a per-layer explanation card grid.

## 2. Lifecycle order

For each successfully started training run, NNx calls `on_train_begin`, then
dispatches epoch and batch work. A completed epoch aggregates validation through
the built-in path or `eval_step_fn`, updates the scheduler once, saves configured
phase checkpoints and run history, and calls `on_epoch_end`. Finalization calls
`on_train_end` in reverse callback order. Both `NNModel` and `Trainer`
refresh LAST after finalization so callback mutations and topology-transform
metadata are present in the persisted checkpoint.

On failure, callbacks whose begin hook completed are still finalized. Every
cleanup hook is attempted; cleanup errors do not mask an exception already
raised by training.
