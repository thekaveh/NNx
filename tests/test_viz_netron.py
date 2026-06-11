"""Tests for nnx.viz.netron_export — ONNX-via-torch.onnx export.

Contracts:
- Produces a non-empty .onnx file at the given path.
- File parses as valid ONNX (round-trip through `onnx.load`).
- Accepts NNModel (unwraps to .net) and raw nn.Module.
- `launch=False` (default) does NOT spawn the Netron viewer — we never
  import / call `netron.start` in CI.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

# Same optional-extra convention every other gated test file follows:
# skip gracefully when the required extras aren't installed (the
# shipped sdist's suite must not hard-fail without them).
# onnx is needed by torch.onnx.export's proto save in EVERY test here,
# not just the explicit round-trip import below.
pytest.importorskip("onnx")

from nnx import (  # noqa: E402
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
)
from nnx.viz import netron_export


@pytest.fixture
def tiny_feedfwd() -> NNModel:
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


def test_netron_export_produces_valid_onnx_file(tmp_path, tiny_feedfwd):
    out = tmp_path / "model.onnx"
    returned = netron_export(tiny_feedfwd, str(out), torch.randn(1, 4))
    assert returned == str(out)
    assert out.exists()
    assert os.path.getsize(out) > 0
    # Round-trip via the onnx package (gated module-level above) to
    # confirm the file is structurally valid, not just non-empty.
    import onnx

    model_proto = onnx.load(str(out))
    onnx.checker.check_model(model_proto)


def test_netron_export_accepts_raw_nn_module(tmp_path):
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    out = tmp_path / "raw.onnx"
    netron_export(net, str(out), torch.randn(1, 4))
    assert out.exists() and os.path.getsize(out) > 0


def test_netron_export_does_not_launch_by_default(tmp_path, tiny_feedfwd, monkeypatch):
    # Sentinel: if netron.start gets called under launch=False the test
    # fails loudly. Guards against future regressions where someone
    # flips the default or forgets the `if launch:` gate.
    called = {"v": False}

    def _fail_if_called(_path: str) -> None:
        called["v"] = True
        raise AssertionError("netron.start must not run under launch=False")

    # Stub the import target so the test doesn't require `netron` to be
    # installed in CI's minimal env. (It *is* installed in this worktree
    # but we want the test resilient to the dep being optional.)
    import sys
    import types

    fake_netron = types.ModuleType("netron")
    fake_netron.start = _fail_if_called  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "netron", fake_netron)

    netron_export(tiny_feedfwd, str(tmp_path / "no_launch.onnx"), torch.randn(1, 4))
    assert called["v"] is False


def test_netron_export_launch_true_invokes_netron_start(tmp_path, tiny_feedfwd, monkeypatch):
    # When the caller explicitly opts in to `launch=True` we should
    # forward to `netron.start(path)`. Stub it out so the test doesn't
    # actually spawn a viewer process.
    seen: dict[str, str] = {}

    def _capture(path: str) -> None:
        seen["path"] = path

    import sys
    import types

    fake_netron = types.ModuleType("netron")
    fake_netron.start = _capture  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "netron", fake_netron)

    out = tmp_path / "launched.onnx"
    netron_export(tiny_feedfwd, str(out), torch.randn(1, 4), launch=True)
    assert seen.get("path") == str(out)


def test_netron_export_launch_true_without_netron_raises(tmp_path, tiny_feedfwd, monkeypatch):
    # Simulate `netron` not being installed so we can verify the
    # ImportError message names the correct extra. The export itself
    # still has to succeed before the import is attempted, so the file
    # should exist on disk even though the launch step failed.
    import sys

    monkeypatch.setitem(sys.modules, "netron", None)
    out = tmp_path / "missing_netron.onnx"
    with pytest.raises(ImportError, match=r"viz-interactive"):
        netron_export(tiny_feedfwd, str(out), torch.randn(1, 4), launch=True)
    assert out.exists()
