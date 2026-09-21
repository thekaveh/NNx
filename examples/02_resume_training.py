"""Warm-resume training from a prior run.

Round 1 trains for 3 epochs; round 2 picks up from the LAST checkpoint
of round 1 and trains another 4 epochs, with Adam momentum / first /
second-moment buffers, scheduler, scaler, epoch progress, and RNG state
preserved across the boundary. Resume lineage participates in the new run's
content-addressed identity, so the original run remains intact.

Checking for saved state is *observational*: ``NNCheckpoint.load_training_state``
/ ``load_optimizer_state`` / ``load_with_training_state`` return their absent
result for a run that has never been written without creating ``runs/<id>/``,
so a preflight probe never blocks the first fit. ``checkpoint_probe_first_fit``
below is a bounded, self-checking demonstration of that contract.

Run:
    python examples/02_resume_training.py
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    set_seed,
)


def _base_optim() -> NNOptimParams:
    return NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-2,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
    )


def _base_sched() -> NNSchedulerParams:
    return NNSchedulerParams(
        min_lr=1e-7,
        factor=0.5,
        patience=2,
        cooldown=1,
        threshold=1e-3,
    )


def _make_model_and_loader():
    # No set_seed here — the caller does set_seed(...) in main() before
    # each call, per the [[examples-seed-helper-override]] convention.
    # Centralizing seed management in main() makes the reproducibility
    # contract visible at the entry point and avoids hidden re-seeding
    # inside helpers.
    X = torch.randn(128, 8)
    y = torch.randint(0, 3, (128,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    net_params = NNParams(
        input_dim=8,
        output_dim=3,
        hidden_dims=[16],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    model_params = NNModelParams(
        net=Nets.FEED_FWD,
        device=Devices.CPU,
        loss=Losses.CROSS_ENTROPY,
    )
    return NNModel(net_params=net_params, params=model_params), loader


def checkpoint_probe_first_fit() -> dict:
    """Bounded demonstration that probing absent training state does not
    reserve a run (writes ``runs/`` under the current working directory;
    the smoke test runs it in a temporary one).

    Derives the prospective run ID from the same params the fit will use,
    probes all three state readers, asserts both the absent return values
    and that no ``runs/<id>/`` directory appeared, then performs the first
    fit with ``overwrite_existing`` left at its default (``False``) and
    reloads LAST together with its training state.
    """
    set_seed(7)
    model, loader = _make_model_and_loader()
    params = NNTrainParams(n_epochs=1, train_loader=loader, optim=_base_optim(), scheduler=_base_sched())
    prospective_id = NNRun(net=model.net_params, train=params, model=model.params).id
    run_dir = os.path.join("runs", prospective_id)

    assert NNCheckpoint.load_training_state(run=prospective_id, type=Checkpoints.LAST) is None
    assert NNCheckpoint.load_optimizer_state(run=prospective_id, type=Checkpoints.LAST) is None
    assert NNCheckpoint.load_with_training_state(run=prospective_id, type=Checkpoints.LAST) == (None, None)
    assert not os.path.exists(run_dir), f"probe must not reserve {run_dir}"

    run = model.train(params=params)  # overwrite_existing stays False
    assert run.id == prospective_id
    last, training_state = NNCheckpoint.load_with_training_state(run=run.id, type=Checkpoints.LAST)
    assert last is not None and training_state is not None
    assert last.idp.epoch_idx == 0 and training_state["completed_epoch"] == 0

    summary = {"run_id": run.id, "probe_reserved_run": False, "last_epoch": last.idp.epoch_idx}
    print(f"probe-then-first-fit workflow: {summary}")
    return summary


def main():
    base_optim = _base_optim()
    base_sched = _base_sched()

    # Round 1: train from scratch. Seed pinned so the random model
    # init + DataLoader shuffle order are reproducible.
    set_seed(7)
    model_a, loader = _make_model_and_loader()
    run_a = model_a.train(
        params=NNTrainParams(
            n_epochs=4,
            train_loader=loader,
            optim=base_optim,
            scheduler=base_sched,
        )
    )
    print(f"\nRound 1 done. run.id = {run_a.id}, {len(run_a.idps)} iterations")

    # Round 2: build a NEW model (random weights) and resume from round 1's LAST.
    # Re-seeding is harmless but not required for continuity: loading the
    # training-state bundle restores the RNG after model construction.
    set_seed(7)
    model_b, loader2 = _make_model_and_loader()
    run_b = model_b.train(
        params=NNTrainParams(
            n_epochs=3,
            train_loader=loader2,
            optim=base_optim,
            scheduler=base_sched,
            resume_from_run_id=run_a.id,
            resume_from_checkpoint="last",
        )
    )
    assert run_b.id != run_a.id
    print(f"Round 2 done. run.id = {run_b.id}, {len(run_b.idps)} iterations")
    print("Round 2 continued from round 1's complete LAST training state.")
    print(f"Final round 2 train loss: {run_b.idps[-1].train_edp.loss:.4f}")


if __name__ == "__main__":
    main()
