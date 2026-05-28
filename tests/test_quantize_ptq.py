"""Tests for ``nnx.quantize.quantize_int8`` — torchao PTQ INT8 weight-only.

torchao is an opt-in extra (``pip install nnx[quantize]``); the entire
module is skipped when it isn't importable so contributors who don't
install the extra still get a clean ``pytest -q``.
"""

from __future__ import annotations

import pickle

import pytest
import torch

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNIterationDataPoint,
    NNModel,
    NNModelParams,
    NNParams,
    quantize_int8,
)

# Skip the whole module when torchao is unavailable. quantize_int8 itself
# raises a clear ImportError; running the rest of the suite without
# torchao still has value (just not these tests).
pytest.importorskip("torchao")


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _make_model(hidden_dims: list[int] | None = None) -> NNModel:
    """Build a small classifier. Hidden dims default to [64, 64] so the
    per-channel int8 layout actually shrinks the state-dict — at
    very small dims the per-channel scales dominate and ``pickle.dumps``
    of the quantized state-dict can be larger than the FP32 one."""
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


@pytest.fixture
def tiny_model() -> NNModel:
    torch.manual_seed(0)
    return _make_model()


# -------------------------------------------------------------------------
# Core behavior
# -------------------------------------------------------------------------


def test_quantize_int8_returns_new_nnmodel(tiny_model):
    """The function returns a fresh NNModel — not the same instance and
    not just the underlying net."""
    m_q = quantize_int8(tiny_model)
    assert isinstance(m_q, NNModel)
    assert m_q is not tiny_model
    assert m_q.net is not tiny_model.net


def test_quantize_int8_preserves_output_shape(tiny_model):
    """Forward shape unchanged — quantized weights are a drop-in replacement."""
    x = torch.randn(4, 32)
    m_q = quantize_int8(tiny_model)
    out_orig = tiny_model.net(x)
    out_quant = m_q.net(x)
    assert out_orig.shape == out_quant.shape == (4, 3)


def test_quantize_int8_does_not_mutate_original(tiny_model):
    """The original NNModel's net.state_dict() must contain plain
    ``torch.Tensor`` weights after quantization — torchao operates on a
    deep-copy under the hood."""
    pre_types = {k: type(v) for k, v in tiny_model.net.state_dict().items()}
    quantize_int8(tiny_model)  # discard the result
    post_types = {k: type(v) for k, v in tiny_model.net.state_dict().items()}
    assert pre_types == post_types, "quantize_int8 mutated the source model's state-dict"
    # Plain forward on the original still works (weights still FP32).
    out = tiny_model.net(torch.randn(2, 32))
    assert out.shape == (2, 3)


def test_quantize_int8_replaces_linear_weights_with_quantized_tensor(tiny_model):
    """Every Linear weight in the quantized copy must be a torchao-quantized
    tensor (not a plain ``torch.nn.Parameter``). torchao ships several
    concrete subclasses across config versions (``AffineQuantizedTensor``
    in v1, ``Int8Tensor`` in v2); both live under the ``torchao``
    package, so we identify quantized weights structurally by their
    module path rather than by importing a specific class."""
    m_q = quantize_int8(tiny_model)
    linear_weights = [(n, m.weight) for n, m in m_q.net.named_modules() if isinstance(m, torch.nn.Linear)]
    assert len(linear_weights) > 0, "test model has no Linear layers"
    for n, w in linear_weights:
        cls = type(w)
        assert cls.__module__.startswith("torchao"), (
            f"Linear weight at {n!r} is {cls.__module__}.{cls.__name__}, expected a torchao-quantized tensor subclass"
        )


def test_quantize_int8_preserves_attached_attrs(tiny_model):
    """Non-net attributes (params, net_params, device, loss_fn) carry over."""
    m_q = quantize_int8(tiny_model)
    assert m_q.params is tiny_model.params
    assert m_q.net_params is tiny_model.net_params
    assert m_q.device == tiny_model.device
    # loss_fn is reused (same instance) — quantize_int8 doesn't rebuild it.
    assert m_q.loss_fn is tiny_model.loss_fn


# -------------------------------------------------------------------------
# Accuracy: output stays close to FP32 (small bounded delta)
# -------------------------------------------------------------------------


def test_quantize_int8_output_close_to_fp32(tiny_model):
    """Int8 weight-only PTQ should keep outputs close to FP32 — large
    drift would mean the quantization is broken. The tolerance is
    deliberately loose: per-channel int8 has a real rounding error,
    but it should still be a small fraction of the activation magnitude
    on a randomly-initialized net."""
    torch.manual_seed(0)
    x = torch.randn(16, 32)
    m_q = quantize_int8(tiny_model)
    with torch.no_grad():
        out_fp = tiny_model.net(x)
        out_q = m_q.net(x)
    # Allow up to ~5% relative L2 deviation.
    rel = (out_fp - out_q).norm() / (out_fp.norm() + 1e-9)
    assert rel < 0.05, f"quantized output drifted too far from FP32: rel={rel:.4f}"


# -------------------------------------------------------------------------
# Size reduction: quantized state-dict pickles smaller than FP32 state-dict
# -------------------------------------------------------------------------


def test_quantize_int8_reduces_pickle_size(tiny_model):
    """At dim >= 64, the per-channel int8 layout is meaningfully smaller
    than the FP32 state-dict — confirms the storage benefit."""
    m_q = quantize_int8(tiny_model)
    fp_bytes = len(pickle.dumps(tiny_model.net.state_dict()))
    q_bytes = len(pickle.dumps(m_q.net.state_dict()))
    assert q_bytes < fp_bytes, f"quantized state-dict ({q_bytes} B) not smaller than FP32 ({fp_bytes} B)"


def test_quantize_int8_nncheckpoint_round_trip_smaller(tiny_model, tmp_path):
    """NNCheckpoint.to_file (pickle path) writes the quantized state-dict
    smaller than the FP32 equivalent. Mirrors the on-disk size benefit
    callers will see in practice."""
    m_q = quantize_int8(tiny_model)

    # Build a minimal idp + the same model/net params for both checkpoints.
    # NNEvaluationDataPoint requires the four sklearn-computed core metrics;
    # use NNEvaluationDataPoint.of with a single-element prediction so the
    # idp is structurally valid without us hand-rolling the field set.
    import numpy as np

    from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

    edp = NNEvaluationDataPoint.of(Y=np.array([0]), Y_hat=np.array([0]))
    idp = NNIterationDataPoint(
        iter_idx=0,
        epoch_idx=0,
        batch_idx=0,
        train_edp=edp,
        lr=1e-3,
    )

    orig_path = tmp_path / "orig.pt"
    quant_path = tmp_path / "quant.pt"
    NNCheckpoint(
        idp=idp,
        model_params=tiny_model.params,
        net_params=tiny_model.net_params,
        net_state=tiny_model.net.state_dict(),
    ).to_file(str(orig_path))
    NNCheckpoint(
        idp=idp,
        model_params=m_q.params,
        net_params=m_q.net_params,
        net_state=m_q.net.state_dict(),
    ).to_file(str(quant_path))

    assert orig_path.exists() and quant_path.exists()
    assert quant_path.stat().st_size < orig_path.stat().st_size, (
        f"quantized checkpoint ({quant_path.stat().st_size} B) not smaller than FP32 ({orig_path.stat().st_size} B)"
    )


# -------------------------------------------------------------------------
# ONNX export sanity — the quantized model still exports cleanly.
# -------------------------------------------------------------------------


def test_quantized_model_onnx_exports(tiny_model, tmp_path):
    """The quantized NNModel must round-trip through NNModel.to_onnx
    cleanly — torchao's AffineQuantizedTensor falls back to dequantized
    matmul during the trace, so the exported ONNX is FP32 with the
    quantized weights baked in. The contract for callers is: 'quantize,
    then export' works."""
    m_q = quantize_int8(tiny_model)
    onnx_path = tmp_path / "quantized.onnx"
    m_q.to_onnx(str(onnx_path), example_input=torch.randn(1, 32))
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0


# -------------------------------------------------------------------------
# Deep-copy isolation — mutating the quantized model doesn't affect original.
# -------------------------------------------------------------------------


def test_quantize_int8_quantized_net_isolated_from_original(tiny_model):
    """Modifying the quantized model's net after quantization must not
    bleed back into the original — confirms the deep-copy boundary."""
    m_q = quantize_int8(tiny_model)
    # Snapshot the original's biases (those stay as plain Parameters
    # post-quantize since Int8WeightOnlyConfig only rewrites .weight).
    orig_bias = tiny_model.net.layers[0].bias.detach().clone()
    with torch.no_grad():
        m_q.net.layers[0].bias.fill_(99.0)
    assert torch.equal(tiny_model.net.layers[0].bias.detach(), orig_bias), (
        "mutating quantized model's bias leaked back into the original"
    )


# -------------------------------------------------------------------------
# Larger smoke: a slightly bigger model still works end-to-end.
# -------------------------------------------------------------------------


def test_quantize_int8_larger_model_predict_round_trip():
    """End-to-end: build a deeper classifier, quantize, run predict()
    via the public API, confirm it returns the expected PredictResult shape."""
    torch.manual_seed(0)
    model = _make_model(hidden_dims=[128, 128, 64])
    m_q = quantize_int8(model)
    import numpy as np

    X = np.random.RandomState(0).randn(8, 32).astype(np.float32)
    result = m_q.predict(X)
    assert result.logits.shape == (8, 3)
    assert result.classes.shape == (8,)


# -------------------------------------------------------------------------
# Error path — opt-in extra missing.
# -------------------------------------------------------------------------


def test_quantize_int8_clear_error_when_torchao_missing(monkeypatch, tiny_model):
    """Simulate torchao being uninstalled and confirm the error message
    points the user at the right pip extra. Mocks the import inside the
    function rather than uninstalling the package."""
    import builtins

    real_import = builtins.__import__

    def _no_torchao(name, *a, **kw):
        if name.startswith("torchao"):
            raise ImportError("simulated: torchao not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_torchao)

    with pytest.raises(ImportError, match=r"nnx\[quantize\]"):
        quantize_int8(tiny_model)


# -------------------------------------------------------------------------
# State-dict structural sanity — keys unchanged, only weight value differs.
# -------------------------------------------------------------------------


def test_quantize_int8_keeps_state_dict_keys(tiny_model):
    """The quantized state-dict must have the same keys as the original
    — Int8WeightOnlyConfig rewrites the .weight tensor in place; it
    does NOT add scale/zero-point top-level keys (those live inside the
    AffineQuantizedTensor's own __tensor_flatten__)."""
    m_q = quantize_int8(tiny_model)
    assert set(m_q.net.state_dict().keys()) == set(tiny_model.net.state_dict().keys())


def test_quantize_int8_idempotent_via_deepcopy(tiny_model):
    """Calling quantize_int8 twice on the same FP32 source must each
    return a valid quantized model — the first call must not have
    mutated the source. Guards against a regression where the deep-copy
    is dropped 'for efficiency'."""
    m_q1 = quantize_int8(tiny_model)
    m_q2 = quantize_int8(tiny_model)
    x = torch.randn(2, 32)
    # Both quantized models produce the same shape; weights came from the
    # same FP32 source so outputs are bit-identical.
    with torch.no_grad():
        o1 = m_q1.net(x)
        o2 = m_q2.net(x)
    assert torch.equal(o1, o2)


def test_quantize_int8_preserves_eval_mode_train_mode_toggle(tiny_model):
    """Standard nn.Module .train() / .eval() must still toggle on the
    quantized net (the AffineQuantizedTensor doesn't override .training)."""
    m_q = quantize_int8(tiny_model)
    m_q.net.eval()
    assert all(not m.training for m in m_q.net.modules())
    m_q.net.train()
    assert m_q.net.training
