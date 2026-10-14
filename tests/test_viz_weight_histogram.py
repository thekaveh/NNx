"""Tests for nnx.viz.weight_histogram — per-parameter Plotly histogram grid.

Three contracts:
- Accepts an NNModel (unwraps to .net) and returns a Plotly Figure.
- Accepts a raw nn.Module directly.
- Emits exactly one trace per named parameter tensor (so visual debugging
  of "where did my dead layer go" is one-to-one with named_parameters()).
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest
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
from nnx.viz import weight_histogram


@pytest.fixture
def tiny_model() -> NNModel:
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


def test_weight_histogram_returns_plotly_figure(tiny_model):
    fig = weight_histogram(tiny_model)
    assert isinstance(fig, go.Figure)
    # Each trainable parameter tensor gets exactly one trace.
    expected_n = sum(1 for _ in tiny_model.net.named_parameters())
    assert len(fig.data) == expected_n


def test_weight_histogram_accepts_nn_module(tiny_model):
    fig = weight_histogram(tiny_model.net)
    assert isinstance(fig, go.Figure)


def test_weight_histogram_raises_on_paramless_module():
    # ReLU has no parameters — there is nothing to plot. Raising beats
    # silently returning an empty figure (which would mislead callers
    # debugging a "where did my model go" issue).
    with pytest.raises(ValueError, match="no named parameters"):
        weight_histogram(nn.ReLU())


def test_weight_histogram_accepts_a_bfloat16_module():
    import torch

    fig = weight_histogram(nn.Linear(4, 2).to(torch.bfloat16))
    assert isinstance(fig, go.Figure) and len(fig.data) == 2  # weight and bias


def test_weight_histogram_renders_a_model_with_many_tensors():
    """Plotly caps subplot spacing at 1 / (rows - 1): 50 tensors (17 rows)
    used to raise ValueError."""
    model = nn.Sequential(*[nn.Linear(2, 2) for _ in range(25)])
    fig = weight_histogram(model)
    assert len(fig.data) == 50


@pytest.mark.parametrize("bad", [{"cols": 0}, {"bins": 0}, {"cols": True}])
def test_weight_histogram_validates_its_grid_arguments(bad):
    with pytest.raises(ValueError, match="positive integer"):
        weight_histogram(nn.Linear(2, 2), **bad)


def test_weight_histogram_counts_non_finite_values_in_the_title():
    import torch

    layer = nn.Linear(2, 2)
    with torch.no_grad():
        layer.weight[0, 0] = float("nan")
        layer.weight[1, 1] = float("inf")
    titles = [annotation.text for annotation in weight_histogram(layer).layout.annotations]
    assert "weight (2 non-finite)" in titles and "bias" in titles
