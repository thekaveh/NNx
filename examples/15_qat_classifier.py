"""Quantization-aware training — train a classifier with fake-quant ops,
convert to int4 weights / int8 dynamic activations at the end.

QAT trades one training run for accuracy recovery on aggressive low-bit
schemes. The flow:

  1. Build a small classifier (FeedFwdNN with four Linear layers, hidden 128, 128, 64).
  2. Wire up ``QATLifecycleCallback`` + ``qat_train_step_factory`` —
     ``on_train_begin`` swaps every Linear for a fake-quantized variant,
     so the network learns under int4/int8 rounding noise.
  3. Train for several epochs. The standard supervised step + AMP /
     accumulation / clipping all flow through unchanged — the fake-quant
     ops sit inside the module graph.
  4. ``on_train_end`` converts the fake-quant linears to real
     int4/int8 modules (``Int8DynActInt4WeightLinear``). The resulting
     model is ready for inference / dynamo-based ONNX export.
  5. The saved LAST checkpoint holds the CONVERTED (quantized) weights
     — ``NNModel.train`` re-saves it after ``on_train_end`` fires, so
     the on-disk state carries the int4 scales/zeros, not the pre-convert
     FP32 tensors. ``NNModel.from_checkpoint`` reloads it: the checkpoint
     records the ``torchao_qat`` transform, and the loader replays the
     same prepare→convert before it loads the quantized state dict,
     and ``check_reload`` asserts that the reloaded model gives the
     converted model's validation logits. Accuracy alone could not show
     a broken reload: a corrupted checkpoint can keep every prediction.

Compared to PTQ (``examples/12_quantize_int8.py``):

  - PTQ is one call, no retraining, and runs on the pre-trained FP32
    model — fast but the accuracy floor is whatever the weight-only
    int8 quantization can preserve.
  - QAT requires a full training run with fake-quant inserted, but
    typically recovers most of the accuracy lost to int4 weights +
    int8 activations.

Requires the ``quantize`` extra (for ``torchao``'s QAT primitives). The
``onnx-dynamo`` extra enables the optional Phase-6 export of the converted
int4/int8 model through the dynamo-based ONNX path (torchao's quantized
tensor subclasses don't round-trip through the legacy TorchScript
exporter); without it, Phase 6 is skipped:

    pip install 'thekaveh-nnx[quantize,onnx-dynamo]'

Run:
    python examples/15_qat_classifier.py

The bounded ``qat_reload_workflow()`` helper (one QAT epoch, then the
Phase-5 reload check) is executed by ``tests/test_examples_smoke.py``.
"""

from __future__ import annotations

import os
import tempfile

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
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    QATLifecycleCallback,
    qat_train_step_factory,
    set_seed,
)


def _make_classifier() -> NNModel:
    """Three hidden layers (128, 128, 64). Each width is a multiple of the
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
    """A 3-class Gaussian mixture whose classes overlap: the class means lie
    about 3.3 noise standard deviations apart, so even the best rule (the
    nearest class mean) misclassifies about one validation row in ten, and a
    row near a class boundary can change its prediction when the weights are
    quantized. Examples 12 (PTQ) and 15 (QAT) share this function verbatim,
    so the two make a clean A/B comparison."""
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(3, 32, generator=g) * 0.4

    def _make(n: int):
        cls = torch.randint(0, 3, (n,), generator=g)
        X = means[cls] + torch.randn(n, 32, generator=g)
        return X, cls

    X_train, y_train = _make(512)
    X_val, y_val = _make(256)
    train = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    return train, val


def _train_params(n_epochs: int, train_loader: DataLoader, val_loader: DataLoader) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=n_epochs,
        # The QAT callback is not part of the run id: without its own
        # data_id this run would share 12_quantize_int8.py's id.
        data_id="qat-8da4w",
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


def check_reload(model: NNModel, checkpoint: NNCheckpoint, inputs: torch.Tensor) -> float:
    """Phase 5's round-trip check: the model rebuilt from ``checkpoint``
    must give the converted ``model``'s logits on ``inputs``. Logits, not
    accuracy: a corrupted reload can keep every argmax. Returns the largest
    absolute logit difference; raises ``AssertionError`` above 1e-5."""
    reloaded = NNModel.from_checkpoint(checkpoint)
    model.net.eval()
    reloaded.net.eval()
    with torch.no_grad():
        expected, got = model.net(inputs), reloaded.net(inputs)
    largest = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, rtol=0.0, atol=1e-5), (
        f"reloaded quantized model differs from the converted one: max |logit diff| {largest:.4g}"
    )
    return largest


def qat_reload_workflow(n_epochs: int = 1) -> dict:
    """Bounded Phases 2 and 5: QAT-train for ``n_epochs``, load the LAST
    checkpoint and run :func:`check_reload` on the validation inputs.
    Returns the converted model, the checkpoint and the inputs so a caller
    can repeat the check. Without ``torchao`` the QAT callback raises an
    ``ImportError`` naming ``thekaveh-nnx[quantize]``."""
    set_seed(0)
    train_loader, val_loader = _loaders(seed=0)
    model = _make_classifier()
    run = model.train(
        params=_train_params(n_epochs, train_loader, val_loader),
        callbacks=[QATLifecycleCallback(qat_config="8da4w")],
        train_step_fn=qat_train_step_factory(qat_config="8da4w"),
    )
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    inputs = torch.cat([X for X, _ in val_loader])
    check_reload(model, checkpoint, inputs)
    return {"model": model, "checkpoint": checkpoint, "inputs": inputs}


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
    run = model.train(
        params=_train_params(6, train_loader, val_loader),
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

    # ---- Checkpoint round-trip. The LAST checkpoint is re-saved AFTER
    # ``on_train_end`` runs, so it holds the CONVERTED weights — the int4
    # scales/zeros are on disk, not the pre-convert FP32 tensors. A fresh
    # net must go through the same prepare→convert (with the same
    # groupsize — the scales/zeros shapes depend on it) before it can
    # absorb the quantized state dict.
    print("\n" + "=" * 60)
    print("Phase 5: checkpoint round-trip (quantized state persists)")
    print("=" * 60)
    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert ckpt is not None
    quant_keys = sorted(k for k in ckpt.net_state if k.endswith(("scales", "zeros")))
    print(f"quantized keys in the saved LAST checkpoint: {quant_keys}")
    assert quant_keys, "LAST must hold the converted int4 scales/zeros"

    largest = check_reload(model, ckpt, torch.cat([X for X, _ in val_loader]))
    print(f"reloaded val logits == converted model's (max |diff| {largest:.2g})")

    # ---- Confirm ONNX export still works on the converted model.
    # The legacy TorchScript exporter trips on torchao's quantized
    # matmul; use the dynamo path (requires onnxscript).
    print("\n" + "=" * 60)
    print("Phase 6: ONNX export (dynamo)")
    print("=" * 60)
    try:
        import onnxscript  # noqa: F401
    except ImportError:
        print("onnxscript not installed — skipping ONNX export.")
        print("Install with `pip install thekaveh-nnx[onnx-dynamo]` to enable.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "qat_converted.onnx")
        model.to_onnx(onnx_path, example_input=torch.randn(1, 32), dynamo=True)
        size = os.path.getsize(onnx_path)
        print(f"wrote {onnx_path} ({size} bytes) — converted model exports cleanly")


if __name__ == "__main__":
    main()
