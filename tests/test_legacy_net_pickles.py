"""Built-in nets pickled whole by NNx 0.3.x and 0.4.x restore whole.

A whole-module pickle (``torch.save(net)``, or a net inside a pickled
``NNModel``) stores each module's ``__dict__``. 0.3.x stored ``params`` as a
plain attribute on every net and cached nothing the forward reads. 0.4.x kept
the plain attribute on ``ConvNN`` and ``FeedFwdMoENN`` (only
``FeedFwdNN.params`` was a property) and cached the forward fields of
``FeedFwdNN``, ``ConvNN`` and ``TransformerNN``. ``params`` of the three
feed-forward nets is now a property over ``_params`` whose setter fills the
cached fields, so such a pickle loaded but failed on first use: ``.params``
and ``str()`` raised ``AttributeError`` for the ``params`` the property hid,
and the forward for the cached fields the pickle lacked.

The legacy layouts are rebuilt from a current net's state, which is what
``torch.load(..., weights_only=False)`` hands ``__setstate__``.
"""

from __future__ import annotations

import io
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn

from nnx import (
    Activations,
    ConvNN,
    Devices,
    FeedFwdMoENN,
    FeedFwdNN,
    Losses,
    Nets,
    NNConvParams,
    NNModel,
    NNModelParams,
    NNMoEParams,
    NNParams,
    NNTransformerParams,
    NNViTParams,
)

_PARAMS = {
    Nets.FEED_FWD: NNParams(
        input_dim=8, hidden_dims=[16, 8], output_dim=3, dropout_prob=0.25, activation=Activations.TANH
    ),
    Nets.CONV: NNConvParams(
        input_dim=64,
        hidden_dims=[8],
        output_dim=3,
        dropout_prob=0.25,
        activation=Activations.TANH,
        conv_channels=[2],
        kernel_size=3,
    ),
    Nets.FEED_FWD_MOE: NNMoEParams(
        input_dim=8,
        hidden_dims=[16],
        output_dim=3,
        dropout_prob=0.25,
        activation=Activations.TANH,
        num_experts=2,
        top_k=1,
    ),
    Nets.TRANSFORMER: NNTransformerParams(
        input_dim=100,
        output_dim=100,
        dropout_prob=0.0,
        activation=Activations.RELU,
        n_heads=4,
        vocab_size=100,
        n_layers=1,
        d_model=32,
        max_seq_len=16,
    ),
    Nets.VIT: NNViTParams(
        input_dim=48, output_dim=8, dropout_prob=0.0, n_heads=2, image_size=4, patch_size=2, d_model=8, n_layers=1
    ),
    Nets.GRAPH_CONV: NNParams(input_dim=8, hidden_dims=[16], output_dim=3, dropout_prob=0.0),
    Nets.GRAPH_SAGE: NNParams(input_dim=8, hidden_dims=[16], output_dim=3, dropout_prob=0.0),
    Nets.GRAPH_ATT: NNParams(input_dim=8, hidden_dims=[16], output_dim=3, dropout_prob=0.0, n_heads=2),
}
_KINDS = {FeedFwdNN: Nets.FEED_FWD, ConvNN: Nets.CONV, FeedFwdMoENN: Nets.FEED_FWD_MOE}

# The fields each feed-forward net's params setter fills for the forward. The
# rest of their pickled attributes, their submodules' included, are the same
# in 0.3.0 and 0.4.1 pickles.
_CACHED = {
    FeedFwdNN: ("_hidden",),
    ConvNN: ("_side", "_in_channels", "_pool_size", "_conv_activation", "_hidden"),
    FeedFwdMoENN: ("_hidden",),
}
# The ones 0.4.1 had cached already, with params still a plain attribute.
_CACHED_0_4_1 = {ConvNN: _CACHED[ConvNN], FeedFwdMoENN: ()}


def _net(kind: Nets) -> nn.Module:
    if kind.name.startswith("GRAPH_"):
        pytest.importorskip("torch_geometric")
    torch.manual_seed(0)
    return kind(_PARAMS[kind])


def _inputs(kind: Nets) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    if kind is Nets.TRANSFORMER:
        return (torch.randint(0, 100, (2, 6)),)
    if kind is Nets.VIT:
        return (torch.randn(2, 3, 4, 4),)
    if kind.name.startswith("GRAPH_"):
        return torch.randn(5, 8), torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    return (torch.randn(4, _PARAMS[kind].input_dim),)


def _legacy_state(net: nn.Module, kept: tuple[str, ...] = ()) -> dict:
    """`net`'s pickled state as an earlier release wrote it: ``params`` a
    plain attribute, and of the cached forward fields only `kept`."""
    state = dict(vars(net))
    state["params"] = state.pop("_params")
    for name in _CACHED[type(net)]:
        if name not in kept:
            del state[name]
    return state


def _restore(net: nn.Module, state: dict) -> nn.Module:
    restored = type(net).__new__(type(net))
    restored.__setstate__(state)  # what torch.load(..., weights_only=False) does
    return restored


@pytest.mark.parametrize(
    ("cls", "kept"),
    [
        pytest.param(FeedFwdMoENN, _CACHED_0_4_1[FeedFwdMoENN], id="FeedFwdMoENN-0.4.1"),
        pytest.param(ConvNN, _CACHED_0_4_1[ConvNN], id="ConvNN-0.4.1"),
        pytest.param(FeedFwdNN, (), id="FeedFwdNN-0.3.x"),
        pytest.param(ConvNN, (), id="ConvNN-0.3.x"),
        pytest.param(FeedFwdMoENN, (), id="FeedFwdMoENN-0.3.x"),
    ],
)
def test_a_legacy_layout_restores_its_params_str_and_logits(cls, kept):
    net = _net(_KINDS[cls]).eval()
    (x,) = _inputs(_KINDS[cls])
    with torch.no_grad():
        expected = net(x)
    state = _legacy_state(net, kept)
    params = state["params"]

    restored = _restore(net, state)

    assert restored.params is params
    assert str(restored) == str(net) == f"{cls.__name__}={params}"
    with torch.no_grad():
        assert torch.equal(restored(x), expected)
    # The cached dropout probabilities are refilled too: a training forward
    # under one seed matches.
    net.train()
    restored.train()
    torch.manual_seed(2)
    trained = net(x)
    torch.manual_seed(2)
    assert torch.equal(restored(x), trained)
    # No stale plain attribute: a re-save writes the current layout.
    assert sorted(vars(restored)) == sorted(vars(net))


def test_a_0_3_x_transformer_forwards_and_still_refuses_an_overlong_sequence():
    net = _net(Nets.TRANSFORMER).eval()
    (tokens,) = _inputs(Nets.TRANSFORMER)
    with torch.no_grad():
        expected = net(tokens)
    state = dict(vars(net))
    del state["_max_seq_len"]  # 0.3.x read params.max_seq_len in the forward

    restored = _restore(net, state)

    assert restored.params is net.params
    assert str(restored) == str(net)
    with torch.no_grad():
        assert torch.equal(restored(tokens), expected)
        with pytest.raises(ValueError, match=r"input sequence length \(17\) exceeds max_seq_len=16"):
            restored(torch.zeros(1, 17, dtype=torch.long))
    assert sorted(vars(restored)) == sorted(vars(net))


@pytest.mark.parametrize(
    ("cls", "kept"),
    [
        pytest.param(FeedFwdMoENN, _CACHED_0_4_1[FeedFwdMoENN], id="FeedFwdMoENN-0.4.1"),
        pytest.param(ConvNN, _CACHED_0_4_1[ConvNN], id="ConvNN-0.4.1"),
        pytest.param(FeedFwdNN, (), id="FeedFwdNN-0.3.x"),
    ],
)
def test_a_pickled_model_wrapping_a_legacy_net_predicts(cls, kept):
    kind = _KINDS[cls]
    torch.manual_seed(0)
    model = NNModel(
        net_params=_PARAMS[kind], params=NNModelParams(net=kind, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    )
    (x,) = _inputs(kind)
    expected = model.predict(x)
    blob = io.BytesIO()
    with mock.patch.object(cls, "__getstate__", lambda self: _legacy_state(self, kept), create=True):
        torch.save(model, blob)  # the whole model, its net in the earlier release's layout
    blob.seek(0)

    restored = torch.load(blob, weights_only=False)

    predicted = restored.predict(x)
    assert np.array_equal(predicted.logits, expected.logits)
    assert np.array_equal(predicted.classes, expected.classes)
    assert restored.net.params == model.net.params
    assert str(restored.net) == str(model.net)


# Each built-in net's own pickled attributes (beyond nn.Module's) and its
# state_dict keys, as NNx writes them at e69fd74: restoring older pickles must
# not change what the current release writes.
_LAYOUT = {
    Nets.FEED_FWD: (
        {"_params", "_hidden"},
        ["layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias", "layers.2.weight", "layers.2.bias"],
    ),
    Nets.CONV: (
        {"_params", "_side", "_in_channels", "_pool_size", "_conv_activation", "_hidden"},
        ["convs.0.weight", "convs.0.bias", "fcs.0.weight", "fcs.0.bias", "fcs.1.weight", "fcs.1.bias"],
    ),
    Nets.FEED_FWD_MOE: (
        {"_params", "_hidden"},
        [
            "layers.0.router.weight",
            "layers.0.experts.0.weight",
            "layers.0.experts.0.bias",
            "layers.0.experts.1.weight",
            "layers.0.experts.1.bias",
            "layers.1.weight",
            "layers.1.bias",
        ],
    ),
    Nets.TRANSFORMER: (
        {"params", "_max_seq_len"},
        [
            "tok_embed.weight",
            "blocks.0.norm1.weight",
            "blocks.0.attn.w_qkv.weight",
            "blocks.0.attn.w_o.weight",
            "blocks.0.norm2.weight",
            "blocks.0.ffn.w1.weight",
            "blocks.0.ffn.w2.weight",
            "blocks.0.ffn.w3.weight",
            "norm_out.weight",
            "lm_head.weight",
        ],
    ),
    Nets.VIT: (
        {"image_size", "patch_size", "in_channels", "d_model", "n_patches"},
        [
            "cls_token",
            "pos_embed",
            "patch_embed.weight",
            "patch_embed.bias",
            "blocks.0.norm1.weight",
            "blocks.0.attn.w_qkv.weight",
            "blocks.0.attn.w_o.weight",
            "blocks.0.norm2.weight",
            "blocks.0.ffn.w1.weight",
            "blocks.0.ffn.w2.weight",
            "blocks.0.ffn.w3.weight",
            "norm_out.weight",
        ],
    ),
    # PyG names its layers' tensors, and has renamed them across releases: the
    # graph nets pin only the layers NNx builds.
    Nets.GRAPH_CONV: ({"params"}, ["layers.0", "layers.1"]),
    Nets.GRAPH_SAGE: ({"params"}, ["layers.0", "layers.1"]),
    Nets.GRAPH_ATT: ({"params"}, ["layers.0", "layers.1"]),
}


def _keys(kind: Nets, net: nn.Module) -> list[str]:
    keys = list(net.state_dict())
    if kind.name.startswith("GRAPH_"):
        return sorted({".".join(key.split(".")[:2]) for key in keys})
    return keys


@pytest.mark.parametrize("kind", list(Nets), ids=lambda kind: kind.name)
def test_a_current_pickle_keeps_its_layout_and_round_trips(kind):
    assert set(_LAYOUT) == set(Nets)
    net = _net(kind).eval()
    inputs = _inputs(kind)
    with torch.no_grad():
        expected = net(*inputs)
    own, keys = _LAYOUT[kind]
    assert set(vars(net)) - set(vars(nn.Module())) == own
    assert _keys(kind, net) == keys

    blob = io.BytesIO()
    torch.save(net, blob)
    blob.seek(0)
    restored = torch.load(blob, weights_only=False)

    assert type(restored) is type(net)
    assert sorted(vars(restored)) == sorted(vars(net))
    assert str(restored) == str(net)
    if hasattr(net, "params"):
        assert restored.params == net.params
    assert list(restored.state_dict()) == list(net.state_dict())
    assert all(torch.equal(restored.state_dict()[key], value) for key, value in net.state_dict().items())
    with torch.no_grad():
        assert torch.equal(restored(*inputs), expected)
