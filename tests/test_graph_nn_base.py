"""Tests that the graph-NN refactor preserves the public interface."""

from __future__ import annotations

import torch

from nnx.nn.enum.activations import Activations
from nnx.nn.net.graph_att_nn import GraphAttNN
from nnx.nn.net.graph_conv_nn import GraphConvNN
from nnx.nn.net.graph_nn_base import GraphNNBase
from nnx.nn.net.graph_sage_nn import GraphSageNN
from nnx.nn.params.nn_params import NNParams


def _params(n_heads=None):
    return NNParams(
        dropout_prob=0.1,
        activation=Activations.RELU,
        input_dim=4,
        output_dim=2,
        hidden_dims=[8],
        n_heads=n_heads,
    )


def test_graph_nets_inherit_from_base():
    assert issubclass(GraphConvNN, GraphNNBase)
    assert issubclass(GraphSageNN, GraphNNBase)
    assert issubclass(GraphAttNN, GraphNNBase)


def test_graph_conv_builds_and_forwards():
    net = GraphConvNN(params=_params())
    n_nodes = 5
    X = torch.randn(n_nodes, 4)
    E = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 0, 1, 2, 3, 4]], dtype=torch.long)
    out = net(X, E)
    assert out.shape == (n_nodes, 2)


def test_graph_sage_builds_and_forwards():
    net = GraphSageNN(params=_params())
    n_nodes = 5
    X = torch.randn(n_nodes, 4)
    E = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 0, 1, 2, 3, 4]], dtype=torch.long)
    out = net(X, E)
    assert out.shape == (n_nodes, 2)


def test_graph_att_requires_n_heads():
    import pytest

    with pytest.raises(ValueError, match="n_heads"):
        GraphAttNN(params=_params(n_heads=None))


def test_graph_att_builds_with_n_heads():
    net = GraphAttNN(params=_params(n_heads=2))
    n_nodes = 5
    X = torch.randn(n_nodes, 4)
    E = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 0, 1, 2, 3, 4]], dtype=torch.long)
    out = net(X, E)
    assert out.shape == (n_nodes, 2)


def test_unpack_batch_returns_tuple_tuple_tensor():
    """All graph nets expose the same (X, E), Y unpacking shape."""
    from types import SimpleNamespace

    batch = SimpleNamespace(
        x=torch.zeros(3, 4),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        y=torch.tensor([0, 1, 0], dtype=torch.long),
    )
    for cls in (GraphConvNN, GraphSageNN):
        net = cls(params=_params())
        (X, E), Y = net.unpack_batch(batch)
        assert X.shape == (3, 4)
        assert E.shape == (2, 2)
        assert Y.shape == (3,)


def test_base_class_requires_build_layers():
    import pytest

    with pytest.raises(NotImplementedError):
        GraphNNBase(params=_params())


def test_fwd_pass_scores_seed_nodes_only():
    """NeighborLoader batches put the batch_size seed nodes first and
    append sampled neighbors, which can belong to other splits.
    _fwd_pass must slice logits/labels to the seeds — pre-fix it scored
    every subgraph node, leaking val/test labels into the training loss
    and train labels into val metrics."""
    from types import SimpleNamespace

    from nnx import Devices, Losses, Nets, NNModel, NNModelParams

    model = NNModel(
        net_params=_params(),
        params=NNModelParams(net=Nets.GRAPH_CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    n_nodes, n_seed = 6, 2
    batch = SimpleNamespace(
        x=torch.randn(n_nodes, 4),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long),
        y=torch.tensor([0, 1, 1, 0, 1, 0]),
        batch_size=n_seed,
    )
    _, Y, logits, Y_hat = model._fwd_pass(batch)
    assert Y.shape == (n_seed,)
    assert logits.shape[0] == n_seed
    assert Y_hat.shape == (n_seed,)

    # Plain full-graph Data (no batch_size attribute) stays unsliced.
    full = SimpleNamespace(x=batch.x, edge_index=batch.edge_index, y=batch.y)
    _, Y_full, logits_full, _ = model._fwd_pass(full)
    assert Y_full.shape == (n_nodes,)
    assert logits_full.shape[0] == n_nodes
