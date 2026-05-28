"""Tests for nnx.viz.summary — Keras-style parameter table.

Thin wrapper around torchinfo.summary. Two contracts:
- Accepts an NNModel and unwraps to .net.
- Accepts an nn.Module directly (no NNModel detour required).
"""

from __future__ import annotations

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
)
from nnx.viz import summary


def _tiny() -> NNModel:
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


def test_summary_returns_torchinfo_modelstatistics():
    from torchinfo.model_statistics import ModelStatistics

    m = _tiny()
    s = summary(m, input_size=(1, 4))
    assert isinstance(s, ModelStatistics)
    assert s.total_params > 0


def test_summary_accepts_nn_module_directly():
    m = _tiny()
    s = summary(m.net, input_size=(1, 4))
    assert s.total_params > 0
