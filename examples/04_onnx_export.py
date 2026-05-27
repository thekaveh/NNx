"""Export a trained NNModel to ONNX.

Requires ``onnx`` to validate the result:
    pip install nnx[onnx]

Run:
    python examples/04_onnx_export.py
"""

from __future__ import annotations

import os
import tempfile

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
    set_seed(0)
    X = torch.randn(64, 8)
    y = torch.randint(0, 3, (64,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    model = NNModel(
        net_params=NNParams(
            input_dim=8,
            output_dim=3,
            hidden_dims=[32],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )

    # Quick fit so the exported model has non-random weights.
    model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        )
    )

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "model.onnx")
        # An example tensor matching the model's input shape. ONNX records
        # the dtype + shape from this; we mark batch dim dynamic so the
        # exported graph accepts any batch size at inference.
        example = torch.randn(2, 8)
        model.to_onnx(onnx_path, example_input=example)

        print(f"\nExported ONNX model: {onnx_path}")
        print(f"  size on disk: {os.path.getsize(onnx_path):,} bytes")

        # Validate via the `onnx` library.
        try:
            import onnx

            onnx.checker.check_model(onnx_path)
            print("  onnx.checker: model is well-formed.")
        except ImportError:
            print("  (install `onnx` to run onnx.checker.check_model)")


if __name__ == "__main__":
    main()
