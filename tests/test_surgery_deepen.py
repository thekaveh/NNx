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


def test_deepen_dtype_follows_dim_source_linear_through_dropout():
    """The identity layer's dtype must come from the SAME upstream
    Linear that sourced the hidden dim. Pre-fix, the dtype probe peeked
    at parent[idx-1] — with a Dropout between the Linear and the ReLU
    that's not a Linear, so the probe fell back to float32 and the
    spliced layer crashed a float64 forward with a dtype mismatch."""
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.Dropout(p=0.0),
        nn.ReLU(),
        nn.Linear(8, 2),
    ).double()
    x = torch.randn(3, 4, dtype=torch.float64)
    orig_out = net(x)

    deeper = deepen(net, after_layer_name="2")  # the ReLU
    assert deeper[3].weight.dtype == torch.float64
    new_out = deeper(x)
    assert torch.allclose(orig_out, new_out, atol=1e-10)


def test_deepen_does_not_advance_global_rng():
    """deepen() must be a no-op on the global torch RNG stream — the
    identity Linear is built uninitialized (skip_init) since both its
    params are overwritten. Pre-fix, the fresh layer's kaiming init
    drew from the default generator, silently diverging any seeded
    caller pipeline."""
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    torch.manual_seed(123)
    state = torch.get_rng_state()
    deepen(net, after_layer_name="1")
    assert torch.equal(torch.get_rng_state(), state)


# ---------------- FIX-006: deepen keeps FeedFwdNN per-layer config aligned ----------------


def _ffn(**overrides) -> FeedFwdNN:
    defaults = dict(input_dim=4, output_dim=2, hidden_dims=[4, 4], dropout_prob=0.0, activation=Activations.RELU)
    defaults.update(overrides)
    return FeedFwdNN(NNParams(**defaults))


def _schema(net: FeedFwdNN) -> list[tuple[int, int]]:
    return [(layer.in_features, layer.out_features) for layer in net.layers]


@pytest.mark.parametrize("site", [0, 1])
def test_deepen_aligns_activation_and_dropout_overrides(site):
    """Deepening a FeedFwdNN must update its immutable params together
    with the ModuleList: one more hidden dim at the insertion site, a
    ReLU inserted into an explicit activations list, the new site's
    dropout set to zero with every old override kept in place, and a
    state()/from_state() round trip that rebuilds the same layer schema.
    The source params and network are untouched."""
    torch.manual_seed(0)
    net = _ffn(
        activations=[Activations.RELU, Activations.RELU],
        dropout_probs=[0.2, 0.4],
    ).eval()
    source_state = net.params.state()
    x = torch.randn(3, 4)

    deeper = deepen(net, after_layer_name=f"layers.{site}").eval()

    torch.testing.assert_close(deeper(x), net(x), atol=1e-5, rtol=1e-5)  # the old IndexError path
    assert list(deeper.params.hidden_dims) == [4, 4, 4]
    assert list(deeper.params.activations) == [Activations.RELU] * 3
    expected_dropout = [0.2, 0.4]
    expected_dropout.insert(site + 1, 0.0)
    assert list(deeper.params.dropout_probs) == expected_dropout
    assert [deeper.params.dropout_for(i) for i in range(3)] == expected_dropout
    assert len(deeper.layers) == 4 and _schema(deeper) == [(4, 4), (4, 4), (4, 4), (4, 2)]

    assert list(net.params.hidden_dims) == [4, 4] and len(net.layers) == 3
    assert net.params.state() == source_state

    restored = NNParams.from_state(deeper.params.state())
    assert restored.dims == deeper.params.dims
    assert [restored.activation_for(i) for i in range(3)] == [deeper.params.activation_for(i) for i in range(3)]
    assert [restored.dropout_for(i) for i in range(3)] == [deeper.params.dropout_for(i) for i in range(3)]
    rebuilt = FeedFwdNN(restored)
    assert _schema(rebuilt) == _schema(deeper)
    rebuilt.load_state_dict(deeper.state_dict())
    rebuilt.eval()
    torch.testing.assert_close(rebuilt(x), net(x), atol=1e-5, rtol=1e-5)


def test_deepen_uses_effective_site_activation():
    """The validator must look at the activation actually applied at the
    insertion site — `params.activation_for(idx)` — not the scalar
    default: a non-ReLU override at the site is rejected even when the
    scalar is ReLU, and a ReLU override is accepted when the scalar is
    not (a later non-ReLU site still blocks insertion *there*)."""
    torch.manual_seed(0)
    x = torch.randn(3, 4)

    blocked = _ffn(activation=Activations.RELU, activations=[Activations.TANH, Activations.RELU])
    with pytest.raises(ValueError, match="tanh"):
        deepen(blocked, after_layer_name="layers.0")
    assert len(blocked.layers) == 3 and list(blocked.params.hidden_dims) == [4, 4]

    allowed = _ffn(activation=Activations.TANH, activations=[Activations.RELU, Activations.RELU]).eval()
    deeper = deepen(allowed, after_layer_name="layers.0").eval()
    torch.testing.assert_close(deeper(x), allowed(x), atol=1e-5, rtol=1e-5)
    assert list(deeper.params.activations) == [Activations.RELU] * 3
    assert deeper.params.activation is Activations.TANH  # scalar untouched; overrides carry the truth

    mixed = _ffn(activation=Activations.SIGMOID, activations=[Activations.RELU, Activations.SIGMOID]).eval()
    deeper = deepen(mixed, after_layer_name="layers.0").eval()
    torch.testing.assert_close(deeper(x), mixed(x), atol=1e-5, rtol=1e-5)
    assert list(deeper.params.activations) == [Activations.RELU, Activations.RELU, Activations.SIGMOID]
    with pytest.raises(ValueError, match="sigmoid"):
        deepen(mixed, after_layer_name="layers.1")


def test_deepen_inserted_dropout_is_zero():
    """With a nonzero scalar dropout and no override list, deepening must
    materialize an override list that keeps every old site at its
    probability and gives the new identity site zero — and the new site
    must not consume an extra dropout draw, so a seeded training forward
    matches the original's exactly."""
    torch.manual_seed(0)
    net = _ffn(dropout_prob=0.5)
    deeper = deepen(net, after_layer_name="layers.0")
    assert list(deeper.params.dropout_probs) == [0.5, 0.0, 0.5]
    assert deeper.params.dropout_prob == 0.5  # scalar untouched for the old sites
    assert net.params.dropout_probs is None  # source untouched

    x = torch.randn(5, 4)
    net.train()
    deeper.train()
    torch.manual_seed(11)
    out_net = net(x)
    rng_net = torch.get_rng_state()
    torch.manual_seed(11)
    out_deeper = deeper(x)
    rng_deeper = torch.get_rng_state()
    assert torch.equal(rng_net, rng_deeper), "the inserted zero-dropout site must not draw from the RNG"
    torch.testing.assert_close(out_deeper, out_net, atol=1e-5, rtol=1e-5)
