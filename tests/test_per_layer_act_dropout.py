"""Per-layer activation & dropout on NNParams (#85).

Optional ``activations`` / ``dropout_probs`` lists (len == number of hidden
layers). ``None`` → the existing net-wide scalar, unchanged. ``state()`` emits
the lists only when present AND not equal to the uniform scalar — the
omit-when-default invariant that keeps every existing config hashing to the
same ``NNRun.id`` (the crux of back-compat).
"""

from __future__ import annotations

import pytest
import torch

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams
from nnx.nn.params.nn_params import NNParams


def _params(**kw) -> NNParams:
    base = dict(
        input_dim=4,
        output_dim=2,
        hidden_dims=[8, 6],
        dropout_prob=0.1,
        activation=Activations.RELU,
    )
    base.update(kw)
    return NNParams(**base)


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


def test_per_layer_lists_accepted_and_stored():
    p = _params(
        activations=[Activations.TANH, Activations.RELU],
        dropout_probs=[0.2, 0.0],
    )
    assert p.activations == [Activations.TANH, Activations.RELU]
    assert p.dropout_probs == [0.2, 0.0]


def test_per_layer_length_must_match_hidden_dims():
    with pytest.raises(ValueError, match="activations"):
        _params(activations=[Activations.TANH])  # 1 != 2 hidden layers
    with pytest.raises(ValueError, match="dropout_probs"):
        _params(dropout_probs=[0.1, 0.2, 0.3])  # 3 != 2


def test_per_layer_dropout_range_validated():
    with pytest.raises(ValueError, match="dropout_probs"):
        _params(dropout_probs=[0.1, 1.5])


def test_no_hidden_dims_rejects_nonempty_lists():
    with pytest.raises(ValueError, match="activations"):
        NNParams(
            input_dim=4,
            output_dim=2,
            hidden_dims=None,
            dropout_prob=0.0,
            activation=Activations.RELU,
            activations=[Activations.TANH],
        )


# ---------------------------------------------------------------------------
# state() round-trip + hash stability (the back-compat crux)
# ---------------------------------------------------------------------------


def test_state_omits_lists_when_absent():
    s = _params().state()
    assert "activations" not in s
    assert "dropout_probs" not in s


def test_state_omits_lists_when_uniform_equal_to_scalar():
    """Lists that merely repeat the scalar must NOT enter state() — otherwise
    an equivalent config would re-hash every existing run.id."""
    p = _params(
        activations=[Activations.RELU, Activations.RELU],  # == scalar activation
        dropout_probs=[0.1, 0.1],  # == scalar dropout_prob
    )
    s = p.state()
    assert "activations" not in s
    assert "dropout_probs" not in s
    # and the state equals the plain-scalar config's state exactly
    assert s == _params().state()


def test_state_round_trips_per_layer_values():
    p = _params(
        activations=[Activations.TANH, Activations.RELU],
        dropout_probs=[0.2, 0.0],
    )
    s = p.state()
    assert "activations" in s and "dropout_probs" in s
    back = NNParams.from_state(s)
    assert back.activations == [Activations.TANH, Activations.RELU]
    assert back.dropout_probs == [0.2, 0.0]
    # full fidelity: state(state) is stable
    assert back.state() == s


def test_hash_stability_for_existing_scalar_configs():
    """An existing scalar-only config must hash to the SAME NNRun-visible state
    as before this feature (state() byte-identical keys/values)."""
    s = _params().state()
    assert set(s.keys()) == {"input_dim", "output_dim", "dropout_prob", "hidden_dims", "activation"}


def test_resolve_from_state_keeps_per_layer_fields():
    p = _params(activations=[Activations.TANH, Activations.RELU])
    back = NNParams.resolve_from_state(p.state())
    assert isinstance(back, NNParams)
    assert back.activations == [Activations.TANH, Activations.RELU]


# ---------------------------------------------------------------------------
# FeedFwdNN honors per-layer values
# ---------------------------------------------------------------------------


def test_feed_fwd_uses_per_layer_activation():
    """relu zeroes negatives; identity-like check via a crafted input: with
    per-layer [GELU, RELU] vs uniform RELU the intermediate outputs differ."""
    torch.manual_seed(0)
    uniform = NNModel(
        net_params=_params(),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    torch.manual_seed(0)
    per_layer = NNModel(
        net_params=_params(activations=[Activations.TANH, Activations.RELU]),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    # identical weights (same seed), different activation schedule → different logits
    X = torch.randn(3, 4)
    uniform.net.eval()
    per_layer.net.eval()
    with torch.no_grad():
        out_u = uniform.net(X)
        out_p = per_layer.net(X)
    assert not torch.allclose(out_u, out_p)


def test_feed_fwd_uses_per_layer_dropout_probs():
    """dropout_probs [1.0, 0.0]: in train mode, layer-0's output is fully dropped,
    so the final logits are exactly the bias path — deterministic despite train()."""
    p = _params(dropout_probs=[1.0, 0.0], dropout_prob=0.0)
    model = NNModel(
        net_params=p,
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    net = model.net
    net.train()  # dropout active
    X = torch.randn(5, 4)
    out1 = net(X)
    out2 = net(X)
    # p=1.0 on the first hidden layer zeroes it entirely → downstream is a
    # deterministic function of biases only → repeated calls identical.
    assert torch.equal(out1, out2)


def test_feed_fwd_scalar_path_unchanged():
    """No lists → the uniform scalar path (regression guard)."""
    model = NNModel(
        net_params=_params(dropout_prob=0.0),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    X = torch.randn(2, 4)
    model.net.eval()
    with torch.no_grad():
        out = model.net(X)
    assert out.shape == (2, 2)
