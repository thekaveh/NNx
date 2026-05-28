"""Tests for ``nnx.surgery.drop_layer``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import drop_layer
from nnx.nn.enum.activations import Activations
from nnx.nn.net.feed_fwd_nn import FeedFwdNN
from nnx.nn.params.nn_params import NNParams

# ---------- Chain-preservation (the local contract) -------------------


def test_drop_layer_replaces_named_layer_with_identity():
    """Dropping a shape-preserving layer (activation) keeps the chain
    intact and forward shapes unchanged."""
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(3, 4)
    # Pre-condition: chain works.
    assert net(x).shape == (3, 2)

    dropped = drop_layer(net, layer_name="1")
    assert isinstance(dropped[1], nn.Identity)
    # Chain still works.
    assert dropped(x).shape == (3, 2)


def test_drop_layer_function_preserved_when_layer_was_already_identity_like():
    """If the dropped module was already a no-op on its input range
    (e.g. ReLU on a strictly positive input), the surged model's
    forward matches the original. This is a sanity check, not a
    general invariant — drop_layer is NOT a function-preserving op."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    # Force the linear's pre-activation positive so ReLU is identity.
    with torch.no_grad():
        net[0].weight.data.abs_()
        net[0].bias.data.fill_(1.0)
    x = torch.abs(torch.randn(3, 4)) + 0.5  # strictly positive input

    orig_out = net(x)
    dropped = drop_layer(net, layer_name="1")
    new_out = dropped(x)
    assert torch.allclose(orig_out, new_out, atol=1e-5)


def test_drop_layer_returns_fresh_module():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    dropped = drop_layer(net, layer_name="1")
    assert dropped is not net
    assert isinstance(net[1], nn.ReLU)  # original untouched


def test_drop_layer_on_feed_fwd_nn():
    """Dotted-name resolution must work on FeedFwdNN's nn.ModuleList."""
    params = NNParams(
        input_dim=4,
        output_dim=2,
        hidden_dims=[8, 8],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    net = FeedFwdNN(params)
    net.eval()
    # Drop the middle Linear (8 → 8, shape-preserving).
    dropped = drop_layer(net, layer_name="layers.1")
    assert isinstance(dropped.layers[1], nn.Identity)
    # The forward still has a valid shape contract since 8 → 8 was
    # square.
    x = torch.randn(2, 4)
    assert dropped(x).shape == (2, 2)


# ---------- Importance-driven selection ------------------------------


def test_drop_layer_picks_minimum_importance_candidate():
    """With a list and an importance fn, the lowest-scoring layer is
    dropped."""
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    # Score Linear layers by their weight L2 norm; force layer "2" to
    # be the lowest-norm candidate.
    with torch.no_grad():
        net[2].weight.data.mul_(1e-4)

    def l2(m):
        if hasattr(m, "weight"):
            return float(m.weight.data.norm().item())
        return float("inf")

    dropped = drop_layer(net, layer_name=["0", "2", "4"], importance=l2)
    assert isinstance(dropped[2], nn.Identity)
    # The other two Linears stayed put.
    assert isinstance(dropped[0], nn.Linear)
    assert isinstance(dropped[4], nn.Linear)


def test_drop_layer_importance_ignored_for_single_string():
    """Passing importance with a single layer_name is silently ignored."""
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    sentinel = {"called": False}

    def score(_):
        sentinel["called"] = True
        return 0.0

    dropped = drop_layer(net, layer_name="1", importance=score)
    assert isinstance(dropped[1], nn.Identity)
    assert sentinel["called"] is False


# ---------- Error handling --------------------------------------------


def test_drop_layer_rejects_nonexistent_layer():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with pytest.raises(KeyError, match="no module named"):
        drop_layer(net, layer_name="bogus")


def test_drop_layer_rejects_empty_candidate_list():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with pytest.raises(ValueError, match="empty candidate list"):
        drop_layer(net, layer_name=[], importance=lambda m: 0.0)


def test_drop_layer_rejects_list_without_importance():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with pytest.raises(ValueError, match="importance="):
        drop_layer(net, layer_name=["0", "2"])


def test_drop_layer_propagates_keyerror_inside_candidate_list():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with pytest.raises(KeyError, match="no module named 'bogus'"):
        drop_layer(net, layer_name=["0", "bogus"], importance=lambda m: 0.0)
