"""Tests for ``nnx.surgery.deepen`` — Net2DeeperNet identity-init."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import deepen
from nnx.nn.enum.activations import Activations
from nnx.nn.net.feed_fwd_nn import FeedFwdNN
from nnx.nn.params.nn_params import NNParams

# ---------- Function-preservation: the contract ------------------------


def test_deepen_after_relu_in_sequential_is_function_preserving():
    """Insert [Linear(I), ReLU] after a ReLU — output unchanged."""
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    x = torch.randn(3, 4)
    orig_out = net(x)
    deeper = deepen(net, after_layer_name="1")  # the ReLU
    new_out = deeper(x)

    # The Sequential is now [Linear(4,8), ReLU, Linear(I,8,8), ReLU, Linear(8,2)]
    assert len(deeper) == 5
    assert isinstance(deeper[2], nn.Linear)
    assert isinstance(deeper[3], nn.ReLU)
    assert deeper[2].in_features == 8
    assert deeper[2].out_features == 8

    assert torch.allclose(orig_out, new_out, atol=1e-5), (
        f"deepen broke function-preservation: max diff {(orig_out - new_out).abs().max().item():.2e}"
    )


def test_deepen_inserted_linear_is_identity_init():
    """The newly inserted Linear must have weight = I and bias = 0 so
    the forward equals identity on the ReLU's non-negative output."""
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    deeper = deepen(net, after_layer_name="1")
    inserted = deeper[2]
    assert isinstance(inserted, nn.Linear)
    assert torch.equal(inserted.weight.data, torch.eye(8))
    assert torch.equal(inserted.bias.data, torch.zeros(8))


def test_deepen_preserves_function_on_feed_fwd_nn():
    """ModuleList mode: insertion via FeedFwdNN's `layers` list. The
    parent's forward applies ReLU on either side, preserving output."""
    torch.manual_seed(1)
    params = NNParams(
        input_dim=6,
        output_dim=3,
        hidden_dims=[8, 10],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    net = FeedFwdNN(params)
    net.eval()
    x = torch.randn(4, 6)
    orig_out = net(x)

    # Insert after layers.0 (first hidden Linear, out_features=8).
    deeper = deepen(net, after_layer_name="layers.0")
    deeper.eval()
    new_out = deeper(x)

    assert len(deeper.layers) == 4  # was 3, now 3+1
    assert deeper.layers[1].in_features == 8
    assert deeper.layers[1].out_features == 8
    assert torch.allclose(orig_out, new_out, atol=1e-5), (
        f"deepen on FeedFwdNN broke function-preservation: max diff {(orig_out - new_out).abs().max().item():.2e}"
    )


def test_deepen_returns_fresh_module():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    deeper = deepen(net, after_layer_name="1")
    assert deeper is not net
    assert len(net) == 3  # original untouched
    assert len(deeper) == 5


# ---------- Error handling --------------------------------------------


def test_deepen_rejects_nonexistent_layer():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with pytest.raises(KeyError, match="no module named"):
        deepen(net, after_layer_name="nonexistent")


def test_deepen_rejects_sigmoid_activation_in_sequential():
    """Identity-init only function-preserves through ReLU. Insertion
    after a Sigmoid is structurally similar but breaks the math, so the
    primitive refuses it via the 'unsupported insertion site' branch
    (the Sigmoid is neither nn.ReLU nor nn.Linear-in-ModuleList)."""
    net = nn.Sequential(nn.Linear(4, 8), nn.Sigmoid(), nn.Linear(8, 2))
    with pytest.raises(TypeError, match="cannot insert after"):
        deepen(net, after_layer_name="1")


def test_deepen_rejects_non_relu_feed_fwd_nn():
    """ModuleList mode: refuse if the parent's activation isn't ReLU."""
    params = NNParams(
        input_dim=6,
        output_dim=3,
        hidden_dims=[8, 10],
        dropout_prob=0.0,
        activation=Activations.SIGMOID,
    )
    net = FeedFwdNN(params)
    with pytest.raises(ValueError, match="ReLU"):
        deepen(net, after_layer_name="layers.0")


def test_deepen_rejects_last_layer_of_module_list():
    """Inserting after the output head of a FeedFwdNN-like module would
    bypass the surrounding activation, so it's refused."""
    params = NNParams(
        input_dim=6,
        output_dim=3,
        hidden_dims=[8, 10],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    net = FeedFwdNN(params)
    # layers has 3 entries (in→8, 8→10, 10→3); inserting after the last
    # is the disallowed case.
    with pytest.raises(ValueError, match="cannot insert after the last"):
        deepen(net, after_layer_name="layers.2")


def test_deepen_rejects_relu_with_no_upstream_linear():
    """ReLU as the first element of a Sequential has no upstream Linear
    to source the hidden dim from."""
    net = nn.Sequential(nn.ReLU(), nn.Linear(4, 2))
    with pytest.raises(ValueError, match="upstream nn.Linear"):
        deepen(net, after_layer_name="0")
