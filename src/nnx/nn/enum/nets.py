from __future__ import annotations

from enum import Enum
from torch import nn, optim
from typing import Tuple, Union

from ..params.nn_params import NNParams

from ..net.feed_fwd_nn import FeedFwdNN
from ..net.graph_att_nn import GraphAttNN
from ..net.graph_conv_nn import GraphConvNN
from ..net.graph_sage_nn import GraphSageNN

class Nets(Enum):
    FEED_FWD    = "feed_fwd"
    GRAPH_ATT   = "graph_att"
    GRAPH_CONV  = "graph_conv"
    GRAPH_SAGE  = "graph_sage"
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)
    
    def __call__(self, params: NNParams):
        assert params is not None
        
        match self:
            case Nets.FEED_FWD:
                return FeedFwdNN(params=params)
            case Nets.GRAPH_ATT:
                return GraphAttNN(params=params)
            case Nets.GRAPH_CONV:
                return GraphConvNN(params=params)
            case Nets.GRAPH_SAGE:
                return GraphSageNN(params=params)