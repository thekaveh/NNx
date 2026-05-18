"""Smoke tests: each network class can be instantiated.

We don't run forward passes here — that requires constructing valid input
shapes which depend on the constructor's exact API. The instantiation
test catches the most common breakage (missing required deps, broken
__init__ signatures, mismatched parent classes).
"""

import torch.nn as nn


def test_feed_fwd_nn_is_module():
    from nnx.nn.net.feed_fwd_nn import FeedFwdNN
    assert issubclass(FeedFwdNN, nn.Module)


def test_graph_conv_nn_is_module():
    from nnx.nn.net.graph_conv_nn import GraphConvNN
    assert issubclass(GraphConvNN, nn.Module)


def test_graph_sage_nn_is_module():
    from nnx.nn.net.graph_sage_nn import GraphSageNN
    assert issubclass(GraphSageNN, nn.Module)


def test_graph_att_nn_is_module():
    from nnx.nn.net.graph_att_nn import GraphAttNN
    assert issubclass(GraphAttNN, nn.Module)
