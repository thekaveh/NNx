from __future__ import annotations

from enum import Enum
from typing import cast

from torch import nn

from ..net.conv_nn import ConvNN
from ..net.feed_fwd_moe_nn import FeedFwdMoENN
from ..net.feed_fwd_nn import FeedFwdNN
from ..net.graph_att_nn import GraphAttNN
from ..net.graph_conv_nn import GraphConvNN
from ..net.graph_sage_nn import GraphSageNN
from ..net.transformer_nn import TransformerNN
from ..params.nn_params import NNParams


class Nets(Enum):
    # LeNet-style conv classifier (#89); consumed by NNConvParams.
    # Back-compat-safe addition (see the TRANSFORMER comment).
    CONV = "conv"
    FEED_FWD = "feed_fwd"
    # MoE feed-forward (#88): hidden layers are MoELinear; consumed by
    # NNMoEParams. Back-compat-safe addition (see the TRANSFORMER comment).
    FEED_FWD_MOE = "feed_fwd_moe"
    GRAPH_ATT = "graph_att"
    GRAPH_CONV = "graph_conv"
    GRAPH_SAGE = "graph_sage"
    # Decoder-only Transformer; consumed by NNTransformerParams. Adding
    # this enum variant is back-compat-safe: existing run.yaml files
    # that don't reference it deserialize unchanged through Nets(<str>).
    TRANSFORMER = "transformer"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    def __call__(self, params: NNParams) -> nn.Module:
        if params is None:
            raise ValueError("params must not be None")

        from ..params.nn_conv_params import NNConvParams
        from ..params.nn_moe_params import NNMoEParams
        from ..params.nn_transformer_params import NNTransformerParams

        specialized = {
            Nets.CONV: NNConvParams,
            Nets.FEED_FWD_MOE: NNMoEParams,
            Nets.TRANSFORMER: NNTransformerParams,
        }
        expected_type = specialized.get(self)
        if expected_type is not None and not isinstance(params, expected_type):
            raise ValueError(f"Nets.{self.name} requires {expected_type.__name__}; got {type(params).__name__}")
        for matching_net, param_type in specialized.items():
            if isinstance(params, param_type) and self is not matching_net:
                raise ValueError(
                    f"{param_type.__name__} is only compatible with Nets.{matching_net.name}; got Nets.{self.name}"
                )

        match self:
            case Nets.CONV:
                return ConvNN(params=cast(NNConvParams, params))
            case Nets.FEED_FWD:
                return FeedFwdNN(params=params)
            case Nets.FEED_FWD_MOE:
                return FeedFwdMoENN(params=cast(NNMoEParams, params))
            case Nets.GRAPH_ATT:
                return GraphAttNN(params=params)
            case Nets.GRAPH_CONV:
                return GraphConvNN(params=params)
            case Nets.GRAPH_SAGE:
                return GraphSageNN(params=params)
            case Nets.TRANSFORMER:
                return TransformerNN(params=cast(NNTransformerParams, params))
