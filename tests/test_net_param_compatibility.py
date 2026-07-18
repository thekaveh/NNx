from __future__ import annotations

import pytest

from nnx import Activations, Nets, NNConvParams, NNMoEParams, NNParams, NNTransformerParams


def _plain() -> NNParams:
    return NNParams(
        input_dim=16,
        output_dim=4,
        hidden_dims=[8],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )


def _conv() -> NNConvParams:
    return NNConvParams(
        input_dim=784,
        output_dim=10,
        hidden_dims=[16],
        dropout_prob=0.0,
        activation=Activations.RELU,
        conv_channels=[6, 16],
    )


def _moe() -> NNMoEParams:
    return NNMoEParams(
        input_dim=16,
        output_dim=4,
        hidden_dims=[8],
        dropout_prob=0.0,
        activation=Activations.RELU,
        num_experts=4,
    )


def _transformer() -> NNTransformerParams:
    return NNTransformerParams(
        input_dim=32,
        output_dim=32,
        hidden_dims=[],
        dropout_prob=0.0,
        activation=Activations.RELU,
        vocab_size=32,
        n_layers=1,
        d_model=8,
        max_seq_len=16,
        n_heads=2,
    )


@pytest.mark.parametrize(
    ("net", "params", "expected"),
    [
        (Nets.CONV, _plain(), "NNConvParams"),
        (Nets.FEED_FWD_MOE, _plain(), "NNMoEParams"),
        (Nets.TRANSFORMER, _plain(), "NNTransformerParams"),
        (Nets.FEED_FWD, _conv(), "CONV"),
        (Nets.GRAPH_CONV, _moe(), "FEED_FWD_MOE"),
        (Nets.GRAPH_SAGE, _transformer(), "TRANSFORMER"),
    ],
)
def test_nets_reject_incompatible_param_types(net, params, expected):
    with pytest.raises(ValueError, match=expected):
        net(params=params)
