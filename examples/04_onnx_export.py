"""Export a trained NNModel to ONNX, and say exactly what was shown.

Requires ``onnx`` to validate the result; ``onnxruntime`` additionally
executes it:
    pip install thekaveh-nnx[onnx]            # checker-only
    pip install thekaveh-nnx[onnx-runtime]    # executed on CPU

Run:
    python examples/04_onnx_export.py

``conformance(model, path, X)`` labels every result (FEAT-038):
``checker-only`` when only ``onnx.checker`` ran — the file is well formed,
and **nothing is claimed about what a runtime computes** — or ``executed``
when ONNX Runtime ran the graph on CPU and its outputs were compared with
the model's at ``rtol=1e-4, atol=1e-5``. Parity is never claimed without
the runtime. The tested, recorded profiles (source revision, artifact
hashes, versions, input cases) live in ``nnx.export_conformance`` and
``scripts/check_export_conformance.py``.

``registered_module_variant()`` (also run by ``main``) exports a model built
from a registered factory (FEAT-006, ``nnx.models``) — any tensor-output
``nn.Module`` exports the same way. It and ``conformance`` are executed by
``tests/test_examples_smoke.py`` as bounded helpers.
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


def conformance(model: NNModel, onnx_path: str, X: torch.Tensor, *, rtol: float = 1e-4, atol: float = 1e-5) -> dict:
    """What an exported file was shown to do — never more.

    Always runs ``onnx.checker`` (raises ``ImportError`` without ``onnx``).
    Without ``onnxruntime`` the result is ``{"level": "checker-only",
    "parity": None}``: structure only, no parity claim. With it, ONNX
    Runtime executes the graph on CPU and ``{"level": "executed", "parity":
    True / False, "max_abs_error": ...}`` compares every raw output with
    the model's at ``rtol`` / ``atol``."""
    import onnx

    onnx.checker.check_model(onnx_path)
    result = {
        "level": "checker-only",
        "parity": None,
        "max_abs_error": None,
        "provider": None,
        "rtol": rtol,
        "atol": atol,
    }
    try:
        import onnxruntime
    except ImportError:
        return result
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    produced = session.run(None, {session.get_inputs()[0].name: X.numpy()})
    reference = np.asarray(model.predict(X).logits)
    result.update(level="executed", provider=session.get_providers()[0], parity=False)
    outputs = np.asarray(produced[0]) if len(produced) == 1 else None  # one output, or a failed comparison
    if outputs is not None and outputs.shape == reference.shape:  # a shape mismatch is a failed comparison, not a crash
        result.update(
            parity=bool(np.allclose(outputs, reference, rtol=rtol, atol=atol)),
            max_abs_error=float(np.abs(outputs - reference).max()),
        )
    return result


def describe(result: dict) -> str:
    """One line that claims exactly the result's level."""
    if result["level"] == "checker-only":
        return (
            "  checker-only: onnx.checker says the file is well formed; no runtime executed it, "
            "so no parity is claimed (install thekaveh-nnx[onnx-runtime] to execute it)"
        )
    verdict = "match" if result["parity"] else "DO NOT match"
    error = "shapes differ" if result["max_abs_error"] is None else f"max abs error {result['max_abs_error']:.2e}"
    return (
        f"  executed: ONNX Runtime ({result['provider']}) outputs {verdict} the model "
        f"({error}, rtol={result['rtol']:g}, atol={result['atol']:g})"
    )


def registered_module_variant() -> dict:
    """Export a registered-factory model and label the result with
    :func:`conformance`. Raises ``ImportError`` without ``onnx``."""
    import onnx  # noqa: F401  (the checker is required; the runtime is optional)

    register_model_factory("examples.scorer", 1, lambda config: Scorer(**config))
    try:
        model = NNModel(params=NNModelParams(net=ModelSpec("examples.scorer", 1, {"width": 16}), device=Devices.CPU))
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = os.path.join(tmp, "scorer.onnx")
            model.to_onnx(onnx_path, example_input=torch.randn(2, 8))
            print(f"\nExported registered module examples.scorer@v1: {onnx_path}")
            X = torch.randn(5, 8, generator=torch.Generator().manual_seed(0))
            result = conformance(model, onnx_path, X)
            print(describe(result))
            if result["level"] == "executed" and not result["parity"]:
                raise AssertionError("the exported graph does not match the model")
            return result
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

        # Label what was shown: checker-only, or executed in ONNX Runtime.
        try:
            print(describe(conformance(model, onnx_path, torch.randn(7, 8))))  # batch 7: the batch dim is dynamic
        except ImportError:
            print("  (install `onnx` to run onnx.checker.check_model; nothing was validated)")

    try:
        registered_module_variant()
    except ImportError:
        print("  (install `onnx` to export the registered-module variant)")


if __name__ == "__main__":
    main()
