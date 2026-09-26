"""Export a trained NNModel to ONNX.

Requires ``onnx`` to validate the result:
    pip install thekaveh-nnx[onnx]

Run:
    python examples/04_onnx_export.py

``registered_module_variant()`` (also run by ``main``) exports a model built
from a registered factory (FEAT-006, ``nnx.models``) — any tensor-output
``nn.Module`` exports the same way — and, when ``onnxruntime`` is
installed, checks the graph's outputs against the CPU model. It is executed
by ``tests/test_examples_smoke.py`` as a bounded helper.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    ModelSpec,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    register_model_factory,
    set_seed,
    unregister_model_factory,
)


class Scorer(torch.nn.Module):
    """A caller-defined, tensor-output module (FEAT-006)."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(8, width), torch.nn.GELU(), torch.nn.Linear(width, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def registered_module_variant() -> None:
    """Export a registered-factory model and, with ``onnxruntime``, compare
    its outputs to the CPU model (rtol=1e-4, atol=1e-5). Raises
    ``ImportError`` without ``onnx``."""
    import onnx

    register_model_factory("examples.scorer", 1, lambda config: Scorer(**config))
    try:
        model = NNModel(params=NNModelParams(net=ModelSpec("examples.scorer", 1, {"width": 16}), device=Devices.CPU))
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = os.path.join(tmp, "scorer.onnx")
            model.to_onnx(onnx_path, example_input=torch.randn(2, 8))
            onnx.checker.check_model(onnx_path)
            print(f"\nExported registered module examples.scorer@v1: {onnx_path}")
            try:
                import onnxruntime
            except ImportError:
                print("  (install `onnxruntime` to compare the graph with the CPU model)")
                return
            X = torch.randn(5, 8, generator=torch.Generator().manual_seed(0))
            session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            (outputs,) = session.run(None, {session.get_inputs()[0].name: X.numpy()})
            np.testing.assert_allclose(outputs, model.predict(X).logits, rtol=1e-4, atol=1e-5)
            print("  onnxruntime outputs match the CPU model.")
    finally:
        unregister_model_factory("examples.scorer", 1)


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

    try:
        registered_module_variant()
    except ImportError:
        print("  (install `onnx` to export the registered-module variant)")


if __name__ == "__main__":
    main()
