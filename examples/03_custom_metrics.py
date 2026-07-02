"""Plug a custom metric callable into NNTrainParams.extra_metrics.

Demonstrates how to record any metric beyond the four hard-coded ones
(f1/recall/precision/accuracy). Custom metrics show up in
``idp.train_edp.extra`` and ``idp.val_edp.extra`` and survive the
NNRun.save → NNRun.load round-trip.

Run:
    python examples/03_custom_metrics.py
"""

from __future__ import annotations

import numpy as np
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


def main():
    set_seed(1)
    X = torch.randn(128, 8)
    y = torch.randint(0, 3, (128,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    model = NNModel(
        net_params=NNParams(
            input_dim=8,
            output_dim=3,
            hidden_dims=[16],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )

    # Two custom metrics: 0-1 error (mirrors `error` but computed differently)
    # and the predicted-class entropy as a confidence proxy.
    def hamming_error(Y, Y_hat):
        return float((Y != Y_hat).mean())

    def predicted_class_entropy(_Y, Y_hat):
        # Distribution of predicted classes, then Shannon entropy in nats.
        _, counts = np.unique(Y_hat, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p + 1e-12)).sum())

    train_params = NNTrainParams(
        n_epochs=3,
        train_loader=loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=1e-2,
            momentum=(0.9, 0.999),
            weight_decay=0.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-7,
            factor=0.5,
            patience=1,
            cooldown=1,
            threshold=1e-3,
        ),
        extra_metrics={
            "hamming_error": hamming_error,
            "predicted_class_entropy": predicted_class_entropy,
        },
    )

    run = model.train(params=train_params)
    last = run.idps[-1]
    print("\nCustom metrics on the final batch:")
    for name, value in last.train_edp.extra.items():
        print(f"  {name:30s} = {value:.4f}")

    print("\nThese values also survive NNRun.load() — they're in idps.csv as extra.<name> columns.")


if __name__ == "__main__":
    main()
