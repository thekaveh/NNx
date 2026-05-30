"""Tests for ``nnx.viz.gradient_flow`` — per-layer gradient-norm bar chart."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest
import torch
from torch import nn

from nnx.viz import gradient_flow


def _tiny_model() -> nn.Module:
    """Two-Linear FFN with one weight + one bias param per layer."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )


def test_gradient_flow_returns_plotly_figure_after_backward():
    """After loss.backward(), gradient_flow returns a Plotly Figure."""
    model = _tiny_model()
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()

    fig = gradient_flow(model)
    assert isinstance(fig, go.Figure)


def test_gradient_flow_one_bar_per_trainable_param():
    """The bar trace's x-axis has one entry per nn.Parameter with a gradient."""
    model = _tiny_model()
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    nn.functional.cross_entropy(model(x), y).backward()

    fig = gradient_flow(model)
    bar = fig.data[0]
    # Two Linear layers × (weight + bias) = 4 entries.
    assert len(bar.x) == 4


def test_gradient_flow_bar_labels_include_param_names():
    """X-axis labels reproduce model.named_parameters() names."""
    model = _tiny_model()
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    nn.functional.cross_entropy(model(x), y).backward()

    fig = gradient_flow(model)
    labels = list(fig.data[0].x)
    expected = [n for n, p in model.named_parameters() if p.requires_grad]
    assert labels == expected


def test_gradient_flow_norms_are_nonnegative():
    """Each bar height is the L2 norm of the gradient — must be >= 0."""
    model = _tiny_model()
    x = torch.randn(4, 4)
    y = torch.tensor([0, 1, 2, 0])
    nn.functional.cross_entropy(model(x), y).backward()

    fig = gradient_flow(model)
    norms = list(fig.data[0].y)
    assert all(n >= 0 for n in norms)


def test_gradient_flow_raises_when_no_gradients():
    """Calling gradient_flow BEFORE backward should raise ValueError with a
    helpful message."""
    model = _tiny_model()
    with pytest.raises(ValueError, match="loss.backward"):
        gradient_flow(model)


def test_gradient_flow_skips_frozen_parameters():
    """Parameters with requires_grad=False are excluded."""
    model = _tiny_model()
    # Freeze the first Linear's weights.
    for p in model[0].parameters():
        p.requires_grad = False

    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    nn.functional.cross_entropy(model(x), y).backward()

    fig = gradient_flow(model)
    labels = list(fig.data[0].x)
    # Only the second Linear's weight + bias remain.
    assert labels == ["2.weight", "2.bias"]


def test_gradient_flow_skips_unreached_parameters():
    """Parameters whose `.grad` is None (never reached by the forward
    pass) are silently skipped. Documents the second guard in
    `gradient_flow` (alongside the `requires_grad=False` skip).
    """
    # Build a model with a deliberately-unreached `unused` parameter.
    # Putting the parameter in a separate ParameterDict means the
    # forward pass doesn't touch it; `param.grad` stays None after
    # `loss.backward()`.
    model = nn.Module()
    model.linear = nn.Linear(4, 3)
    model.unused = nn.Parameter(torch.randn(5))

    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    # Forward only goes through `linear`, not `unused`.
    nn.functional.cross_entropy(model.linear(x), y).backward()

    fig = gradient_flow(model)
    labels = list(fig.data[0].x)
    # `unused` has requires_grad=True but no .grad — must be skipped.
    assert "unused" not in labels
    # `linear.weight` + `linear.bias` should be present.
    assert "linear.weight" in labels
    assert "linear.bias" in labels
