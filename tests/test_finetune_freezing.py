"""Tests for nnx.finetune.freezing — requires_grad management via patterns."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams
from nnx.finetune import freeze, frozen, unfreeze


def _two_layer_net() -> nn.Sequential:
    """Module with named children so dotted-name matching has structure
    to chew on. Returns a Linear→ReLU→Linear stack."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def test_freeze_matches_by_dotted_name():
    """freeze('0.*') matches the first Linear's weight + bias only."""
    net = _two_layer_net()
    n = freeze(net, "0.*")
    assert n == 2  # weight + bias of the first Linear
    assert frozen(net) == ["0.bias", "0.weight"]
    # The second Linear is still trainable.
    assert net[2].weight.requires_grad


def test_freeze_multiple_patterns_unions():
    net = _two_layer_net()
    n = freeze(net, "0.weight", "2.bias")
    assert n == 2
    assert frozen(net) == ["0.weight", "2.bias"]


def test_freeze_idempotent_doesnt_double_count():
    """Freezing an already-frozen parameter doesn't count it again — the
    return value is 'newly frozen', not 'total frozen'."""
    net = _two_layer_net()
    first = freeze(net, "0.weight")
    again = freeze(net, "0.weight")
    assert first == 1
    assert again == 0
    assert "0.weight" in frozen(net)


def test_unfreeze_inverts_freeze():
    net = _two_layer_net()
    freeze(net, "*")
    assert len(frozen(net)) == 4  # all 4 params
    n = unfreeze(net, "0.*")
    assert n == 2
    assert frozen(net) == ["2.bias", "2.weight"]


def test_freeze_requires_at_least_one_pattern():
    """No-arg freeze() is too dangerous to be the default — could silently
    freeze the whole model if a user forgot the pattern arg."""
    net = _two_layer_net()
    with pytest.raises(ValueError, match="at least one pattern"):
        freeze(net)
    with pytest.raises(ValueError, match="at least one pattern"):
        unfreeze(net)


def test_freeze_wildcard_freezes_everything():
    """`freeze('*')` is the explicit way to freeze everything."""
    net = _two_layer_net()
    n = freeze(net, "*")
    assert n == 4  # 2 layers × (weight + bias)
    assert len(frozen(net)) == 4


def test_freeze_pattern_matching_nothing_returns_zero():
    """A pattern that matches no parameter must NOT raise — the
    fnmatch-based API is fundamentally "filter by name", and a no-match
    result is a legitimate filter output (e.g., a configuration loop
    iterates known prefixes and some don't exist on the current net)."""
    net = _two_layer_net()
    n = freeze(net, "nonexistent.*")
    assert n == 0
    assert frozen(net) == []
    # All parameters still trainable.
    assert all(p.requires_grad for p in net.parameters())


def test_unfreeze_pattern_matching_nothing_returns_zero():
    """Same no-match contract for the inverse operation."""
    net = _two_layer_net()
    freeze(net, "*")
    n = unfreeze(net, "absolutely.not.here.*")
    assert n == 0
    # Everything still frozen.
    assert not any(p.requires_grad for p in net.parameters())


def test_frozen_returns_sorted_for_stable_assertions():
    net = _two_layer_net()
    freeze(net, "2.weight", "0.bias")
    # Sorted alphabetically regardless of insertion order.
    assert frozen(net) == ["0.bias", "2.weight"]


def test_nnmodel_freeze_delegates_to_self_net():
    """NNModel.freeze / .unfreeze should affect only self.net, matching
    the documented contract that this is a convenience for the common case."""
    model = NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    # FeedFwdNN's parameters are layers.0.* and layers.1.*
    n = model.freeze("layers.0.*")
    assert n == 2
    assert all(name.startswith("layers.0.") for name in frozen(model.net))

    m = model.unfreeze("layers.0.weight")
    assert m == 1
    assert frozen(model.net) == ["layers.0.bias"]


def test_frozen_params_dont_get_gradients(monkeypatch):
    """End-to-end: freeze a layer, run forward+backward, confirm the
    frozen layer's .grad stays None while the trainable layer's doesn't."""
    torch.manual_seed(0)
    net = _two_layer_net()
    freeze(net, "0.*")  # freeze first layer

    X = torch.randn(4, 4)
    y = torch.randint(0, 2, (4,))
    out = net(X)
    loss = nn.functional.cross_entropy(out, y)
    loss.backward()

    # Frozen layer accumulates no gradient.
    assert net[0].weight.grad is None
    assert net[0].bias.grad is None
    # Trainable layer does.
    assert net[2].weight.grad is not None
    assert net[2].bias.grad is not None
