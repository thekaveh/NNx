"""Warm-resume training from a prior run.

Round 1 trains for 3 epochs; round 2 picks up from the LAST checkpoint
of round 1 and trains another 3 epochs, with Adam momentum / first /
second-moment buffers preserved across the boundary.

Run:
    python examples/02_resume_training.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

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
    set_seed,
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


def main():
    base_optim = NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-2,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
    )
    base_sched = NNSchedulerParams(
        min_lr=1e-7,
        factor=0.5,
        patience=2,
        cooldown=1,
        threshold=1e-3,
    )

    # Round 1: train from scratch. Seed pinned so the random model
    # init + DataLoader shuffle order are reproducible.
    set_seed(7)
    model_a, loader = _make_model_and_loader()
    run_a = model_a.train(
        params=NNTrainParams(
            n_epochs=3,
            train_loader=loader,
            optim=base_optim,
            scheduler=base_sched,
        )
    )
    print(f"\nRound 1 done. run.id = {run_a.id}, {len(run_a.idps)} iterations")

    # Round 2: build a NEW model (random weights) and resume from round 1's LAST.
    # Re-seed so the model has the same initial weights as Round 1 (which
    # get overwritten by load_state_dict on resume anyway); the same seed
    # also pins the DataLoader shuffle order for an apples-to-apples
    # continuation of the training trajectory.
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
    print(f"Round 2 done. run.id = {run_b.id}, {len(run_b.idps)} iterations")
    print("Round 2 started from round 1's LAST weights + optimizer state.")
    print(f"Final round 2 train loss: {run_b.idps[-1].train_edp.loss:.4f}")


if __name__ == "__main__":
    main()
