"""Tests for nnx.viz.activation_map — forward-hook activation capture.

Contracts:
- 4D conv activation -> Plotly Figure with one Heatmap per channel (capped
  at max_channels).
- 2D dense activation -> single-Heatmap Figure of shape (N, F).
- Unknown layer name -> ValueError naming the candidates.
- Accepts NNModel (unwraps to .net) and raw nn.Module.
- Eval mode is restored to whatever it was on entry (training-state
  invariant — a debug helper shouldn't silently flip the model into
  eval-mode-forever).
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest
import torch
from torch import nn

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
)
from nnx.viz import activation_map


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


class _TinyConv(nn.Module):
    """Hand-built conv net for the 4D activation path.

    NNx ships a feed-forward / GNN / transformer factory but no
    Conv2d-flavored Net enum yet; this fixture stands in until that
    exists.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_activation_map_4d_conv_returns_per_channel_heatmaps():
    model = _TinyConv()
    x = torch.randn(2, 3, 8, 8)
    fig = activation_map(model, x, "conv1")
    assert isinstance(fig, go.Figure)
    # conv1 has 6 output channels; we should get exactly 6 heatmap traces
    # (under the default max_channels=16).
    assert len(fig.data) == 6
    for trace in fig.data:
        assert isinstance(trace, go.Heatmap)
        # First sample, single channel -> (H, W) = (8, 8).
        assert trace.z.shape == (8, 8)


def test_activation_map_4d_respects_max_channels():
    model = _TinyConv()
    x = torch.randn(1, 3, 8, 8)
    fig = activation_map(model, x, "conv1", max_channels=3)
    assert len(fig.data) == 3


def test_activation_map_2d_dense_returns_single_heatmap(tiny_feedfwd):
    # `layers.0` is the first Linear in the FeedFwd Sequential -> 2D output.
    x = torch.randn(5, 4)
    fig = activation_map(tiny_feedfwd, x, "layers.0")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)
    # Shape is (batch, hidden_dim) = (5, 8).
    assert fig.data[0].z.shape == (5, 8)


def test_activation_map_unknown_layer_raises_valueerror(tiny_feedfwd):
    x = torch.randn(1, 4)
    with pytest.raises(ValueError, match="not found in model.named_modules"):
        activation_map(tiny_feedfwd, x, "nope.does.not.exist")


def test_activation_map_accepts_raw_nn_module(tiny_feedfwd):
    fig = activation_map(tiny_feedfwd.net, torch.randn(3, 4), "layers.0")
    assert isinstance(fig, go.Figure)


def test_activation_map_restores_training_state():
    model = _TinyConv()
    model.train()
    activation_map(model, torch.randn(1, 3, 4, 4), "conv1")
    assert model.training is True


def test_activation_map_leaves_eval_mode_alone():
    model = _TinyConv()
    model.eval()
    activation_map(model, torch.randn(1, 3, 4, 4), "conv1")
    assert model.training is False
