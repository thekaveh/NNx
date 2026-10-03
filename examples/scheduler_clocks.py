"""Scheduler clocks: per-epoch vs per-optimizer-update scheduling (FEAT-014).

By default every scheduler steps once per completed epoch, so its horizons
(``step_size``, ``T_max``, ``total_steps``, ``warmup_steps``) count epochs.
``NNSchedulerParams(clock="optimizer_update")`` instead steps it once per
*committed* update of its optimizer — never per microbatch, an all-masked
window or a skipped step — so those horizons count optimizer updates, the
unit warm-up and one-cycle schedules are usually written in.

This script trains a tiny classifier offline and checks three things:

1. **Accumulation.** Two epochs of five microbatches with
   ``accumulate_grad_batches=2`` commit 2 + 2 + 1 updates per epoch: the
   update-clock schedule steps 6 times, the epoch-clock one twice, and
   ``LRMonitor.history`` still holds one LR per epoch (per-update LRs go to
   ``LRMonitor.update_history``).
2. **A skipped update.** An objective with ``nonfinite="skip"`` drops one
   window with a non-finite loss: no update, so no scheduler step (5).
3. **A continuation.** A one-cycle schedule over 12 updates, split 2 + 2
   epochs across a resume, follows exactly the LR sequence of the
   uninterrupted run: the clock's count, owner and budget are checkpointed.

Run:
    python examples/scheduler_clocks.py
"""

from __future__ import annotations

import os
import tempfile

import torch

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    Schedulers,
)
from nnx.nn.callbacks import LRMonitor
from nnx.objectives import LossTerm, Objective, ObjectiveResult, supervised_objective

_PLATEAU_FIELDS = {"min_lr": 0.0, "factor": 0.5, "patience": 0, "cooldown": 0, "threshold": 0.0}


def _model(seed: int = 0) -> NNModel:
    torch.manual_seed(seed)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _batches(n: int = 5) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1)
    return [(torch.randn(4, 4, generator=generator), torch.randint(0, 2, (4,), generator=generator)) for _ in range(n)]


def _train(scheduler: NNSchedulerParams, *, epochs: int = 2, objective=None, data_id: str, **train):
    monitor = LRMonitor()
    run = _model(train.pop("seed", 0)).train(
        NNTrainParams(
            n_epochs=epochs,
            train_loader=_batches(),
            optim=NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0, accumulate_grad_batches=2),
            scheduler=scheduler,
            save_phase_checkpoints=False,
            data_id=data_id,
            **train,
        ),
        callbacks=[monitor],
        objective=objective,
    )
    return monitor, run


class _SkipOneWindow(Objective):
    """The supervised objective with a non-finite loss in epoch 1, batch 2."""

    def __init__(self) -> None:
        super().__init__(nonfinite="skip")
        self.inner = supervised_objective()

    def __call__(self, ctx):
        result = self.inner(ctx)
        if (ctx.epoch_idx, ctx.batch_idx) != (1, 2):
            return result
        (term,) = result.terms
        poisoned = LossTerm(term.name, term.numerator * float("nan"), term.denominator, term.reduction)
        return ObjectiveResult([poisoned], result.record)


def scheduler_clocks_workflow() -> dict:
    """Bounded, self-checking demonstration of both clocks."""
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as workdir:
        os.chdir(workdir)
        try:
            step = {"kind": Schedulers.STEP, "step_size": 1, **_PLATEAU_FIELDS}
            per_update, _ = _train(NNSchedulerParams(clock="optimizer_update", **step), data_id="updates")
            per_epoch, _ = _train(NNSchedulerParams(**step), data_id="epochs")
            assert [k for k, _ in per_update.update_history] == [1, 2, 3, 4, 5, 6]
            assert len(per_update.history) == len(per_epoch.history) == 2  # one LR per epoch either way
            assert per_epoch.update_history == []
            assert per_epoch.history[-1] == 0.1 * 0.5**2 and per_update.history[-1] == 0.1 * 0.5**6

            skipped, _ = _train(
                NNSchedulerParams(clock="optimizer_update", **step), objective=_SkipOneWindow(), data_id="skip"
            )
            assert len(skipped.update_history) == 5  # the non-finite window took no update

            one_cycle = NNSchedulerParams(
                kind=Schedulers.ONE_CYCLE, max_lr=0.5, total_steps=12, clock="optimizer_update", **_PLATEAU_FIELDS
            )
            whole, _ = _train(one_cycle, epochs=4, data_id="whole")
            first, parent = _train(one_cycle, epochs=2, data_id="split")
            second, _ = _train(one_cycle, epochs=2, data_id="split", seed=7, resume_from_run_id=parent.id)
            resumed = first.update_history + second.update_history
            assert [k for k, _ in resumed] == list(range(1, 13))
            assert all(abs(a[1] - b[1]) < 1e-12 for a, b in zip(resumed, whole.update_history, strict=True))
        finally:
            os.chdir(previous)
    summary = {
        "updates_per_run": len(per_update.update_history),
        "epoch_steps_per_run": len(per_epoch.history),
        "updates_with_a_skip": len(skipped.update_history),
        "continued_updates": len(resumed),
    }
    print(f"scheduler clocks: {summary}")
    return summary


def main() -> None:
    scheduler_clocks_workflow()


if __name__ == "__main__":
    main()
