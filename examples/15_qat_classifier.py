"""Quantization-aware training — train a classifier with fake-quant ops,
convert to int4 weights / int8 dynamic activations at the end.

QAT trades one training run for accuracy recovery on aggressive low-bit
schemes. The flow:

  1. Build a small classifier (FeedFwdNN with three Linear layers).
  2. Wire up ``QATLifecycleCallback`` + ``qat_train_step_factory`` —
     ``on_train_begin`` swaps every Linear for a fake-quantized variant,
     so the network learns under int4/int8 rounding noise.
  3. Train for several epochs. The standard supervised step + AMP /
     accumulation / clipping all flow through unchanged — the fake-quant
     ops sit inside the module graph.
  4. ``on_train_end`` converts the fake-quant linears to real
     int4/int8 modules (``Int8DynActInt4WeightLinear``). The resulting
     model is ready for inference / dynamo-based ONNX export.

Compared to PTQ (``examples/12_quantize_int8.py``):

  - PTQ is one call, no retraining, and runs on the pre-trained FP32
    model — fast but the accuracy floor is whatever the weight-only
    int8 quantization can preserve.
  - QAT requires a full training run with fake-quant inserted, but
    typically recovers most of the accuracy lost to int4 weights +
    int8 activations.

Run:
    pip install nnx[quantize,onnx-dynamo]
    python examples/15_qat_classifier.py
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
    QATLifecycleCallback,
    qat_train_step_factory,
    set_seed,
)


def _make_classifier() -> NNModel:
    """Three hidden layers of width 128. Each width is a multiple of the
    int4 weight quantizer's default groupsize (32) so the 8da4w recipe
    applies cleanly without padding."""
    return NNModel(
        net_params=NNParams(
            input_dim=32,
            output_dim=3,
            hidden_dims=[128, 128, 64],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def _loaders(seed: int = 0) -> tuple[DataLoader, DataLoader]:
    """Separable 3-class Gaussian-mixture toy task — same shape as the
    PTQ example so the two examples make a clean A/B comparison."""
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(3, 32, generator=g) * 2.0

    def _make(n: int):
        cls = torch.randint(0, 3, (n,), generator=g)
        X = means[cls] + 0.5 * torch.randn(n, 32, generator=g)
        return X, cls

    X_train, y_train = _make(512)
    X_val, y_val = _make(256)
    train = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    return train, val


def _val_accuracy(model: NNModel, val_loader: DataLoader) -> float:
    """Mean argmax accuracy over the val loader. Routes through model.net
    directly so the (possibly-converted) quantized weights are exercised."""
    model.net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, Y in val_loader:
            logits = model.net(X)
            pred = logits.argmax(dim=1)
            correct += int((pred == Y).sum().item())
            total += int(Y.size(0))
    return correct / total


def main():
    set_seed(0)
    train_loader, val_loader = _loaders(seed=0)

    # ---- Build the model + QAT pieces.
    print("=" * 60)
    print("Phase 1: building model + QAT callback")
    print("=" * 60)
    model = _make_classifier()
    fp_total = sum(p.numel() for p in model.net.parameters())
    print(f"net: {fp_total} parameters (FP32)\n")

    callback = QATLifecycleCallback(qat_config="8da4w")
    step_fn = qat_train_step_factory(qat_config="8da4w")
    print("QATLifecycleCallback initialized; quantizer = Int8DynActInt4WeightQATQuantizer")
    print("qat_train_step_factory returned the default supervised step")
    print("(QAT integration happens via the callback's on_train_begin/end hooks).\n")

    # ---- Train. The callback inserts fake-quant on the first epoch
    # and converts on the last.
    print("=" * 60)
    print("Phase 2: training with QAT")
    print("=" * 60)
    model.train(
        params=NNTrainParams(
            n_epochs=6,
            train_loader=train_loader,
            val_loader=val_loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=2,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        callbacks=[callback],
        train_step_fn=step_fn,
    )
    print(f"\ncallback.is_prepared = {callback.is_prepared}")
    print(f"callback.is_converted = {callback.is_converted}")

    # ---- Inspect the converted module hierarchy.
    print("\n" + "=" * 60)
    print("Phase 3: post-convert module inspection")
    print("=" * 60)
    types_present = sorted({type(m).__name__ for m in model.net.modules()})
    print("module types in converted net:")
    for t in types_present:
        print(f"  - {t}")

    # ---- Measure accuracy of the converted model.
    print("\n" + "=" * 60)
    print("Phase 4: converted model accuracy")
    print("=" * 60)
    acc = _val_accuracy(model, val_loader)
    print(f"int4/int8 (8da4w) val accuracy: {acc * 100:.2f}%")

    # ---- Confirm ONNX export still works on the converted model.
    # The legacy TorchScript exporter trips on torchao's quantized
    # matmul; use the dynamo path (requires onnxscript).
    print("\n" + "=" * 60)
    print("Phase 5: ONNX export (dynamo)")
    print("=" * 60)
    try:
        import onnxscript  # noqa: F401
    except ImportError:
        print("onnxscript not installed — skipping ONNX export.")
        print("Install with `pip install nnx[onnx-dynamo]` to enable.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "qat_converted.onnx")
        model.to_onnx(onnx_path, example_input=torch.randn(1, 32), dynamo=True)
        size = os.path.getsize(onnx_path)
        print(f"wrote {onnx_path} ({size} bytes) — converted model exports cleanly")


if __name__ == "__main__":
    main()
