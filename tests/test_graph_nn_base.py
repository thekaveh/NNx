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
        # input_id marks a genuine NeighborLoader subgraph (the seed
        # indices) — it gates the slice; see the multi-graph test below.
        input_id=torch.tensor([0, 1]),
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


def test_fwd_pass_does_not_slice_multi_graph_batches():
    """A PyG Batch.from_data_list collation carries batch_size
    (= num_graphs) but NO input_id — node-classification over a
    multi-graph DataLoader must stay unsliced. An over-broad
    discriminator here would silently truncate node-level logits to
    the graph count."""
    from types import SimpleNamespace

    from nnx import Devices, Losses, Nets, NNModel, NNModelParams

    model = NNModel(
        net_params=_params(),
        params=NNModelParams(net=Nets.GRAPH_CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    n_nodes = 5
    collated = SimpleNamespace(
        x=torch.randn(n_nodes, 4),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 2, 4]], dtype=torch.long),
        y=torch.tensor([0, 1, 0, 1, 0]),
        batch_size=2,  # num_graphs — NOT a seed count
    )
    _, Y, logits, _ = model._fwd_pass(collated)
    assert Y.shape == (n_nodes,)
    assert logits.shape[0] == n_nodes


def test_predict_loader_slices_to_seed_nodes():
    """predict() over a NeighborLoader-style loader must return one row
    per SEED node, not per subgraph node — neighbor rows would both
    pollute the output and break row-count alignment with the loader's
    node set."""
    from types import SimpleNamespace

    from nnx import Devices, Losses, Nets, NNModel, NNModelParams

    model = NNModel(
        net_params=_params(),
        params=NNModelParams(net=Nets.GRAPH_CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )

    def _subgraph(n_nodes: int, n_seed: int) -> SimpleNamespace:
        return SimpleNamespace(
            x=torch.randn(n_nodes, 4),
            edge_index=torch.tensor([[i for i in range(n_nodes)], [(i + 1) % n_nodes for i in range(n_nodes)]]),
            y=torch.zeros(n_nodes, dtype=torch.long),
            batch_size=n_seed,
            input_id=torch.arange(n_seed),
        )

    class _Loader:
        def __iter__(self):
            return iter([_subgraph(6, 2), _subgraph(5, 3)])

    # predict() type-checks for DataLoader via isinstance — wrap minimally.
    result = model.predict(X=_FakeDataLoader(_Loader()))
    assert result.logits.shape[0] == 5  # 2 + 3 seeds, not 6 + 5 nodes
    assert result.classes.shape[0] == 5


class _FakeDataLoader(torch.utils.data.DataLoader):
    """A DataLoader subclass whose iteration is fully overridden — lets
    predict()'s isinstance(X, DataLoader) branch run on synthetic
    NeighborLoader-shaped batches without torch_geometric machinery."""

    def __init__(self, inner):
        self._inner = inner
        # Minimal viable parent init: a one-item dataset, never used.
        super().__init__(dataset=[0])

    def __iter__(self):
        return iter(self._inner)
