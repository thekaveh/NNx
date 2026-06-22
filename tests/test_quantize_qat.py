"""Tests for ``nnx.quantize.qat`` — torchao QAT 8da4w.

torchao is an opt-in extra (``pip install thekaveh-nnx[quantize]``); the entire
module is skipped when it isn't importable so contributors who don't
install the extra still get a clean ``pytest -q``.
"""

from __future__ import annotations

import pytest
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
    NNTrainParams,
    Optims,
    QATLifecycleCallback,
    default_train_step,
    qat_train_step_factory,
)

# Skip the whole module when torchao is unavailable. The factory + callback
# raise clear ImportErrors on construction; running the rest of the suite
# without torchao still has value (just not these tests).
pytest.importorskip("torchao")


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _make_model(hidden_dims: list[int] | None = None) -> NNModel:
    """Small classifier whose hidden widths divide the default int4
    groupsize (32). Width 64 = 32 * 2, so the 8da4w quantizer applies
    cleanly without requiring ``padding_allowed=True``."""
    return NNModel(
        net_params=NNParams(
            input_dim=32,
            output_dim=3,
            hidden_dims=hidden_dims or [64, 64],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def _make_loader(n: int = 64, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(3, 32, generator=g) * 2.0
    cls = torch.randint(0, 3, (n,), generator=g)
    X = means[cls] + 0.5 * torch.randn(n, 32, generator=g)
    return DataLoader(TensorDataset(X, cls), batch_size=16, shuffle=True)


@pytest.fixture
def tiny_model() -> NNModel:
    torch.manual_seed(0)
    return _make_model()


# -------------------------------------------------------------------------
# Factory validation
# -------------------------------------------------------------------------


def test_qat_train_step_factory_rejects_unknown_config():
    """Bad ``qat_config`` must raise eagerly at factory construction —
    before ``model.train(...)`` spins up — so typos surface fast."""
    with pytest.raises(ValueError, match="unknown qat_config"):
        qat_train_step_factory(qat_config="bogus-recipe")


@pytest.mark.parametrize("bad_groupsize", [0, -32])
def test_qat_lifecycle_callback_rejects_non_positive_groupsize(bad_groupsize):
    """``groupsize`` is the int4 weight grouping width. A negative value
    silently builds a mis-quantized model and 0 surfaces only as a cryptic
    ZeroDivisionError deep inside prepare(); both fail fast at construction
    via the shared ``_build_quantizer`` chokepoint — the
    [[params-boundary-validation]] class."""
    with pytest.raises(ValueError, match="groupsize must be a positive int"):
        QATLifecycleCallback(qat_config="8da4w", groupsize=bad_groupsize)


def test_qat_train_step_factory_returns_base_step_unchanged():
    """The factory is a thin wrapper: no base => default_train_step;
    custom base => same callable returned. The fake-quant insertion
    happens in the callback, not in the per-batch step."""
    fn_default = qat_train_step_factory()
    assert fn_default is default_train_step

    def my_step(ctx):  # type: ignore[no-untyped-def] — type signature carried by TrainStepFn
        return None

    fn_custom = qat_train_step_factory(base_step=my_step)
    assert fn_custom is my_step


# -------------------------------------------------------------------------
# Callback lifecycle: prepares + converts at the right times
# -------------------------------------------------------------------------


def _module_class_names(net: torch.nn.Module) -> set[str]:
    """All concrete class names in ``net``'s module tree.

    Used instead of an ``isinstance(m, nn.Linear)`` filter because
    torchao's *converted* linear (``Int8DynActInt4WeightLinear``) does
    NOT subclass ``nn.Linear``: it's a standalone ``nn.Module`` that
    holds packed int4 weights. Filtering by ``nn.Linear`` would hide
    exactly the modules we want to assert on post-convert."""
    return {type(m).__name__ for m in net.modules()}


def test_qat_lifecycle_callback_prepares_and_converts(tiny_model):
    """The callback's on_train_begin must swap nn.Linear for the
    fake-quantized counterpart; on_train_end must convert to the
    truly-quantized variant. Driven manually (bypassing model.train)
    so we exercise the callback contract in isolation."""
    cb = QATLifecycleCallback(qat_config="8da4w")
    assert not cb.is_prepared and not cb.is_converted

    # Snapshot the layer types pre-prepare. Plain nn.Linear from FeedFwdNN.
    pre = _module_class_names(tiny_model.net)
    assert "Linear" in pre, pre

    # Build a minimal stand-in for the train()'s _CallbackContext that
    # exposes just .model — that's all our callback consumes.
    class _Ctx:
        def __init__(self, model):
            self.model = model

    ctx = _Ctx(tiny_model)

    cb.on_train_begin(ctx)
    assert cb.is_prepared and not cb.is_converted
    # Post-prepare: torchao swaps in Int8DynActInt4WeightQATLinear (a
    # subclass of nn.Linear) — identify by class name to stay decoupled
    # from torchao's internal symbol path.
    prepared_types = _module_class_names(tiny_model.net)
    assert any("QAT" in t and "Linear" in t for t in prepared_types), f"no QAT-wrapped Linear found: {prepared_types}"

    # Forward pass still works on the prepared model — fake-quant is
    # transparent to the standard forward.
    out = tiny_model.net(torch.randn(2, 32))
    assert out.shape == (2, 3)

    cb.on_train_end(ctx)
    assert cb.is_prepared and cb.is_converted
    converted_types = _module_class_names(tiny_model.net)
    # After convert: Int8DynActInt4WeightLinear (no "QAT" in the name —
    # this is the inference-time module, not the training-time one).
    # Note: the converted linear is NOT an nn.Linear subclass.
    assert any("Int8" in t and "Int4" in t and "QAT" not in t for t in converted_types), (
        f"no truly-quantized Linear found post-convert: {converted_types}"
    )


def test_qat_lifecycle_callback_idempotent(tiny_model):
    """Calling on_train_begin / on_train_end twice must not double-wrap
    or re-convert. Guards against a regression if a caller registers
    the same callback in two back-to-back train() invocations."""
    cb = QATLifecycleCallback(qat_config="8da4w")

    class _Ctx:
        def __init__(self, model):
            self.model = model

    ctx = _Ctx(tiny_model)
    cb.on_train_begin(ctx)
    types_after_first = _module_class_names(tiny_model.net)
    # Second call must be a no-op (early return on is_prepared).
    cb.on_train_begin(ctx)
    types_after_second = _module_class_names(tiny_model.net)
    assert types_after_first == types_after_second

    cb.on_train_end(ctx)
    converted = _module_class_names(tiny_model.net)
    cb.on_train_end(ctx)
    still_converted = _module_class_names(tiny_model.net)
    assert converted == still_converted


def test_qat_lifecycle_callback_convert_without_prepare_noop(tiny_model):
    """Calling on_train_end without ever calling on_train_begin must
    NOT mutate the model (would otherwise yield a partially-quantized
    state that fails inference). It's a silent no-op since
    ``is_prepared`` is False."""
    cb = QATLifecycleCallback(qat_config="8da4w")

    class _Ctx:
        def __init__(self, model):
            self.model = model

    pre = _module_class_names(tiny_model.net)
    cb.on_train_end(_Ctx(tiny_model))
    post = _module_class_names(tiny_model.net)
    assert pre == post
    assert not cb.is_converted


# -------------------------------------------------------------------------
# End-to-end: QAT integrates with model.train()
# -------------------------------------------------------------------------


def test_qat_end_to_end_training(monkeypatch):
    """Full integration: build a tiny classifier, attach the
    QATLifecycleCallback + factory step, run a couple of epochs via
    NNModel.train(), confirm the model emerges converted and still
    produces sane outputs. This is the contract the user actually
    cares about."""
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    model = _make_model(hidden_dims=[64, 64])
    train_loader = _make_loader(n=64, seed=0)

    cb = QATLifecycleCallback(qat_config="8da4w")
    step_fn = qat_train_step_factory(qat_config="8da4w")

    model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=train_loader,
            val_loader=None,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
        ),
        callbacks=[cb],
        train_step_fn=step_fn,
    )

    # After train(): callback advanced through the full lifecycle.
    assert cb.is_prepared and cb.is_converted

    # The converted model's linears are the truly-quantized variant —
    # not the QAT (fake-quant) variant, which would mean on_train_end
    # silently skipped. Converted module isn't an nn.Linear subclass,
    # so scan all modules by class name.
    classes = _module_class_names(model.net)
    assert any("Int8" in t and "Int4" in t and "QAT" not in t for t in classes), (
        f"converted model still has QAT linears: {classes}"
    )

    # Forward through the converted model returns logits with the right shape.
    with torch.no_grad():
        out = model.net(torch.randn(4, 32))
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all(), "converted model produced non-finite outputs"


# -------------------------------------------------------------------------
# Post-convert ONNX export
# -------------------------------------------------------------------------


def test_qat_converted_model_onnx_exports(tmp_path, monkeypatch, skip_on_dynamo_dispatch_error):
    """The converted model must round-trip through ``NNModel.to_onnx`` —
    same contract as PTQ. torchao's converted int4-weight linear is
    not a TorchScript-traceable module (legacy ``torch.onnx.export``
    path raises an internal assertion), so we use the ``dynamo=True``
    path, which goes through ``torch.export`` and handles the packed
    quantized weights correctly.

    Requires ``onnxscript`` — install via ``thekaveh-nnx[onnx-dynamo]``. Skipped
    when the installed torch / onnxscript combo can't dispatch one of the
    ops the quantized model emits (see ``_skip_if_dynamo_dispatch_error``
    in ``conftest.py``)."""
    pytest.importorskip("onnxscript")

    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    model = _make_model(hidden_dims=[64, 64])

    cb = QATLifecycleCallback(qat_config="8da4w")
    step_fn = qat_train_step_factory(qat_config="8da4w")
    model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=_make_loader(n=32, seed=0),
            val_loader=None,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        ),
        callbacks=[cb],
        train_step_fn=step_fn,
    )
    assert cb.is_converted

    onnx_path = tmp_path / "qat.onnx"
    try:
        model.to_onnx(str(onnx_path), example_input=torch.randn(1, 32), dynamo=True)
    except Exception as e:
        skip_on_dynamo_dispatch_error(e)
        raise  # unreachable — helper either skips or re-raises
    assert onnx_path.exists() and onnx_path.stat().st_size > 0
