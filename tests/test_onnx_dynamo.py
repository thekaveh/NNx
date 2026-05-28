"""Tests for the `dynamo` opt-in on `NNModel.to_onnx`.

PyTorch 2.5+ ships a `torch.export`-based ONNX exporter (`dynamo=True`)
which became the default in 2.9. NNx defaults to the legacy TorchScript
exporter (`dynamo=False`) so that plain `pip install onnx` is sufficient;
callers who want the new path opt in explicitly with `dynamo=True` and
install the `onnx-dynamo` extra (which pulls `onnxscript`).

What's covered here:

- The default (`dynamo=False`) still produces a checker-clean ONNX file —
  the kwarg addition must not break existing callers.
- `dynamo=True` produces a non-empty file via the new exporter (gated by
  an `onnxscript` importorskip — the dep is opt-in).
- When `onnxscript` isn't importable, `dynamo=True` raises a clear
  `ImportError` pointing at the `nnx[onnx-dynamo]` extra rather than
  whatever cryptic error torch would otherwise surface.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from nnx.nn.enum.activations import Activations
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.nn_model import NNModel
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_params import NNParams


def _model() -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=4,
            output_dim=2,
            hidden_dims=[8],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def test_to_onnx_dynamo_true_writes_file(tmp_path, skip_on_dynamo_dispatch_error):
    """`dynamo=True` runs the torch.export-based exporter and writes a
    non-empty ONNX file. Skipped when `onnxscript` isn't installed (the
    dep is intentionally opt-in via `nnx[onnx-dynamo]`), or when the
    installed torch / onnxscript combo can't dispatch one of the prims/aten
    ops this model emits — see ``_skip_if_dynamo_dispatch_error`` in
    ``conftest.py`` for the upstream-version-skew rationale."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    model = _model()
    onnx_path = tmp_path / "model_dynamo.onnx"
    example = torch.randn(2, 4)

    try:
        out = model.to_onnx(str(onnx_path), example_input=example, dynamo=True)
    except Exception as e:
        skip_on_dynamo_dispatch_error(e)
        raise  # unreachable — helper either skips or re-raises

    assert Path(out).exists()
    assert os.path.getsize(out) > 0


def test_to_onnx_default_is_legacy_path(tmp_path):
    """Back-compat: omitting `dynamo` keeps the legacy TorchScript path
    (no `onnxscript` required) and still produces a checker-clean file.
    Guards against the kwarg-addition silently flipping the default."""
    pytest.importorskip("onnx")

    model = _model()
    onnx_path = tmp_path / "model_default.onnx"
    example = torch.randn(2, 4)

    out = model.to_onnx(str(onnx_path), example_input=example)

    assert Path(out).exists()
    assert os.path.getsize(out) > 0

    import onnx

    onnx.checker.check_model(str(onnx_path))


def test_to_onnx_dynamo_true_without_onnxscript_raises_clear_error(tmp_path, monkeypatch):
    """When `onnxscript` is missing, `dynamo=True` must raise an
    `ImportError` that names the `nnx[onnx-dynamo]` extra. Without this
    lazy guard, torch would surface a less actionable error (or even
    appear to succeed in some versions) — both bad UX."""
    # Simulate `onnxscript` being uninstalled even if it happens to be
    # available in the test environment.
    monkeypatch.setitem(sys.modules, "onnxscript", None)

    model = _model()
    onnx_path = tmp_path / "model_no_onnxscript.onnx"
    example = torch.randn(2, 4)

    with pytest.raises(ImportError, match=r"onnx-dynamo"):
        model.to_onnx(str(onnx_path), example_input=example, dynamo=True)
