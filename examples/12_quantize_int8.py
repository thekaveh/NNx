"""PTQ INT8 weight-only quantization — train a classifier, quantize, compare.

Flow:

  1. Build a small classifier (FeedFwdNN with three Linear layers).
  2. Train on a 3-class Gaussian-mixture toy task.
  3. Snapshot FP32 validation accuracy.
  4. Quantize the trained model via ``nnx.quantize_int8`` — one call,
     no calibration data, no retraining.
  5. Compare:

     - Size of the pickled state-dict before vs. after.
     - Validation accuracy before vs. after.

The accuracy delta is typically a fraction of a percentage point for
networks this small; at production scale (transformer / ResNet) the
size win is closer to 4x and accuracy loss stays sub-percent for most
classification tasks. The example demonstrates the *mechanism*; the
real-world tradeoff is task-dependent.

Run:
    pip install nnx-pytorch[quantize]   # or: pip install 'torchao>=0.17'
    python examples/12_quantize_int8.py
"""

from __future__ import annotations

import os
import pickle
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
    quantize_int8,
    set_seed,
)


def _make_classifier() -> NNModel:
    """Three hidden layers of width 128. Wide enough that the per-channel
    int8 layout's metadata is dominated by the weight bytes, so the
    quantized state-dict is meaningfully smaller than the FP32 one."""
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
    """A separable 3-class toy task — class means well-separated so the
    network reaches a meaningful accuracy and the quantization delta is
    measurable."""
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
    """Run the model over val_loader and return mean argmax accuracy.
    Routes through model.net directly so the quantized weights are
    exercised (model.predict accepts a DataLoader and goes through the
    same forward, but we do it manually here for transparency)."""
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

    # ---- Phase 1: train.
    print("=" * 60)
    print("Phase 1: training FP32 classifier")
    print("=" * 60)
    model = _make_classifier()
    fp_total = sum(p.numel() for p in model.net.parameters())
    print(f"net: {fp_total} parameters (FP32)\n")
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
        )
    )

    # ---- Phase 2: snapshot FP32 accuracy + state-dict size.
    print("\n" + "=" * 60)
    print("Phase 2: measuring FP32 baseline")
    print("=" * 60)
    fp_acc = _val_accuracy(model, val_loader)
    fp_bytes = len(pickle.dumps(model.net.state_dict()))
    print(f"FP32 val accuracy:        {fp_acc * 100:.2f}%")
    print(f"FP32 state-dict (pickle): {fp_bytes:>9} bytes")

    # ---- Phase 3: quantize. One call. Source NNModel is left untouched.
    print("\n" + "=" * 60)
    print("Phase 3: PTQ INT8 weight-only quantization")
    print("=" * 60)
    model_q = quantize_int8(model)
    print("quantize_int8 returned a new NNModel; source is unchanged.")
    # Show that the original's first Linear weight is still a plain Parameter.
    src_w_type = type(model.net.layers[0].weight).__name__
    q_w_type = type(model_q.net.layers[0].weight).__name__
    print(f"original layers[0].weight: {src_w_type}")
    print(f"quantized layers[0].weight: {q_w_type}")

    # ---- Phase 4: measure deltas.
    print("\n" + "=" * 60)
    print("Phase 4: comparing FP32 vs INT8")
    print("=" * 60)
    q_acc = _val_accuracy(model_q, val_loader)
    q_bytes = len(pickle.dumps(model_q.net.state_dict()))

    print(f"INT8 val accuracy:        {q_acc * 100:.2f}%")
    print(f"INT8 state-dict (pickle): {q_bytes:>9} bytes")
    print()
    print(f"accuracy delta:           {(q_acc - fp_acc) * 100:+.2f} pp")
    print(f"size ratio (INT8 / FP32): {q_bytes / fp_bytes * 100:.1f}%")
    print(f"size reduction:           {(1 - q_bytes / fp_bytes) * 100:.1f}%")

    # ---- Phase 5: confirm ONNX export still works on the quantized model.
    print("\n" + "=" * 60)
    print("Phase 5: ONNX export sanity check")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "quantized.onnx")
        model_q.to_onnx(onnx_path, example_input=torch.randn(1, 32))
        size = os.path.getsize(onnx_path)
        print(f"wrote {onnx_path} ({size} bytes) — the quantized model exports cleanly")


if __name__ == "__main__":
    main()
