"""Tests for nnx.vis_utils.

Cover the module import, the public callable surface, AND a real
invocation of `multi_line_plot` so a regression that turns the
function into a no-op or returns `None` is caught.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch


def test_vis_utils_importable():
    import nnx.vis_utils as vis_utils

    assert vis_utils is not None


def test_vis_utils_module_has_callables():
    """vis_utils should expose at least one callable for figure generation."""
    import inspect

    import nnx.vis_utils as vis_utils

    callables = [name for name, obj in inspect.getmembers(vis_utils) if callable(obj) and not name.startswith("_")]
    assert len(callables) > 0, "vis_utils exposes no public callables"


def test_multi_line_plot_returns_figure_with_traces():
    """A real invocation of `multi_line_plot` — verifies the function
    actually builds a Plotly Figure and adds the expected number of
    traces. Catches regressions that would silently return None or skip
    trace creation."""
    import plotly.graph_objects as go

    from nnx.vis_utils import VisUtils

    x = list(range(10))
    yss = [
        [[1.0 * i for i in range(10)], [2.0 * i for i in range(10)]],  # group 1: 2 lines
        [[0.5 * i for i in range(10)]],  # group 2: 1 line
    ]
    fig = VisUtils.multi_line_plot(
        x=x,
        yss=yss,
        title="test plot",
        yss_legend=(["group A", "group B"], ["line 1", "line 2"]),
        x_axis_label="x",
        y_axis_label="y",
        renderer=None,  # headless
    )
    assert isinstance(fig, go.Figure)
    # Expect at least one trace per actual data line — the function may
    # also add no-trace markers for the legend rows; the floor is the
    # data-line count.
    actual_lines = sum(len(group) for group in yss)
    assert len(fig.data) >= actual_lines, f"expected ≥ {actual_lines} traces for the data lines, got {len(fig.data)}"


def test_multi_line_plot_raises_on_empty_yss():
    """The function should raise loudly rather than silently producing an empty plot."""
    from nnx.vis_utils import VisUtils

    with pytest.raises(ValueError, match="at least one series"):
        VisUtils.multi_line_plot(
            x=[],
            yss=[],
            title="t",
            yss_legend=([], []),
            x_axis_label="x",
            y_axis_label="y",
            renderer=None,
        )


def test_generate_colors_are_pairwise_distinct():
    """Hue is circular, so the old closed linspace gave the first and
    last class identical colors in every scatter/t-SNE plot."""
    from nnx.vis_utils import VisUtils

    for n in (2, 3, 5, 8):
        colors = VisUtils.generate_colors(n)
        assert len(set(colors)) == n, f"duplicate colors at n={n}: {colors}"


def test_two_dim_tsne_checkpoint_logits_collects_across_batches(monkeypatch):
    from nnx import vis_utils
    from nnx.vis_utils import VisUtils

    captured = {}

    class _FakeTSNE:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def fit_transform(self, X):
            captured["rows"] = len(X)
            return np.zeros((len(X), 2))

    class _FakeNet:
        @staticmethod
        def unpack_batch(batch):
            return batch

    class _FakeModel:
        net = _FakeNet()

        @staticmethod
        def predict(X):
            n = len(X)
            return types.SimpleNamespace(logits=np.arange(n * 2, dtype=float).reshape(n, 2))

    class _FakeNNModel:
        @staticmethod
        def from_checkpoint(checkpoint):
            return _FakeModel()

    ds = types.SimpleNamespace(
        output_dim=2,
        test_loader=[
            (torch.randn(3, 4), torch.tensor([0, 1, 0])),
            (torch.randn(4, 4), torch.tensor([1, 0, 1, 0])),
        ],
    )
    checkpoint = types.SimpleNamespace(idp=types.SimpleNamespace(epoch_idx=3))

    monkeypatch.setattr(vis_utils, "NNModel", _FakeNNModel)
    monkeypatch.setattr(vis_utils, "TSNE", _FakeTSNE)

    VisUtils.two_dim_tsne_checkpoint_logits(checkpoint=checkpoint, ds=ds, n_samples=5, renderer=None)

    assert captured["rows"] == 5
    assert captured["kwargs"]["perplexity"] == 4
    assert captured["kwargs"]["random_state"] == 0


def test_two_dim_tsne_refuses_multi_output_targets_before_the_projection(monkeypatch):
    from nnx import vis_utils
    from nnx.vis_utils import VisUtils

    class _NeverTSNE:
        def __init__(self, **kwargs):
            raise AssertionError("t-SNE ran before the refusal")

    class _FakeModel:
        net = types.SimpleNamespace(unpack_batch=staticmethod(lambda batch: batch))

        @staticmethod
        def predict(X):
            return types.SimpleNamespace(logits=np.zeros((len(X), 3)))

    monkeypatch.setattr(vis_utils, "NNModel", types.SimpleNamespace(from_checkpoint=lambda checkpoint: _FakeModel()))
    monkeypatch.setattr(vis_utils, "TSNE", _NeverTSNE)
    ds = types.SimpleNamespace(output_dim=3, test_loader=[(torch.randn(4, 4), torch.ones(4, 3))])
    with pytest.raises(ValueError, match="one class index per row"):
        VisUtils.two_dim_tsne_checkpoint_logits(checkpoint=object(), ds=ds, n_samples=4, renderer=None)


def test_multi_line_plot_legends_every_dash_style_and_checks_label_counts():
    from nnx.vis_utils import VisUtils

    x = [0, 1, 2]
    common = dict(title="t", x_axis_label="x", y_axis_label="y", renderer=None)
    fig = VisUtils.multi_line_plot(x=x, yss=[[x], [x, x]], yss_legend=(["a", "b"], ["x", "y"]), **common)
    legend = [trace.name for trace in fig.data if trace.showlegend is not False]
    assert "x" in legend and "y" in legend
    with pytest.raises(ValueError, match="label per group"):
        VisUtils.multi_line_plot(x=x, yss=[[x], [x, x]], yss_legend=(["a"], ["x", "y"]), **common)


# --- #529: a PyG node batch is plotted by its seed rows only ----------------------------------------


def _row_tsne(monkeypatch):
    """Swap t-SNE for a projection that puts row ``i`` at ``x = i``: the
    projected input is captured, and each class trace's ``x`` values name
    the rows drawn in that class's colour."""
    from nnx import vis_utils

    captured = {}

    class _RowTSNE:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, X):
            captured["X"] = np.asarray(X)
            return np.column_stack([np.arange(len(X), dtype=float), np.zeros(len(X))])

    monkeypatch.setattr(vis_utils, "TSNE", _RowTSNE)
    return captured


def _rows_by_class(fig) -> list[list[int]]:
    return [sorted(int(x) for x in trace.x) for trace in fig.data]


class _TinyGraph:  # one 20-node graph: 10 train, 4 val, 6 test nodes (the ticket's example)
    num_features, num_classes = 3, 2

    def __init__(self, root=None, transform=None):
        from torch_geometric.data import Data

        n = torch.arange(20)
        self._data = Data(
            x=torch.randn(20, 3, generator=torch.Generator().manual_seed(7)),
            edge_index=torch.stack([n, (n + 1) % 20]),
            y=n % 2,
            train_mask=n < 10,
            val_mask=(n >= 10) & (n < 14),
            test_mask=n >= 14,
        )

    def __getitem__(self, idx):
        return self._data


def _node_batch(seeds: int, neighbors: int, offset: int):
    """A batch in the NeighborLoader layout: ``seeds`` class-0 seed rows
    (``input_id`` / ``batch_size``), then ``neighbors`` class-1 rows that
    only supply message-passing context."""
    from torch_geometric.data import Data

    n = torch.arange(seeds + neighbors)
    batch = Data(
        x=torch.randn(len(n), 3, generator=torch.Generator().manual_seed(offset)),
        edge_index=torch.stack([n, (n + 1) % len(n)]),
        y=(n >= seeds).long(),
    )
    batch.input_id = torch.arange(offset, offset + seeds)
    batch.batch_size = seeds
    return batch


def _trained_checkpoint(net, train_loader, val_loader, input_dim, tmp_path, monkeypatch, **net_kwargs):
    """(checkpoint, reloaded model) of a one-epoch run of the built-in ``net``."""
    from nnx import (
        Activations,
        Checkpoints,
        Devices,
        Losses,
        NNCheckpoint,
        NNModel,
        NNModelParams,
        NNParams,
        NNTrainParams,
        set_seed,
    )

    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = NNModel(
        net_params=NNParams(
            input_dim=input_dim,
            output_dim=2,
            hidden_dims=[4],
            dropout_prob=0.0,
            activation=Activations.RELU,
            **net_kwargs,
        ),
        params=NNModelParams(net=net, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    run = model.train(params=NNTrainParams(n_epochs=1, train_loader=train_loader, val_loader=val_loader))
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    return checkpoint, NNModel.from_checkpoint(checkpoint=checkpoint)


def _graph_checkpoint(net, tmp_path, monkeypatch, **net_kwargs):
    from nnx import NNGraphDataset

    ds = NNGraphDataset(ds_class=_TinyGraph, sampler="full")
    checkpoint, reloaded = _trained_checkpoint(
        net, ds.train_loader, ds.val_loader, 3, tmp_path, monkeypatch, **net_kwargs
    )
    return ds, checkpoint, reloaded


@pytest.mark.parametrize(
    ("net_name", "net_kwargs"),
    [("GRAPH_CONV", {}), ("GRAPH_SAGE", {}), ("GRAPH_ATT", {"n_heads": 2}), ("FEED_FWD", {})],
)
def test_two_dim_tsne_plots_only_the_test_seed_rows_of_a_full_graph_batch(net_name, net_kwargs, tmp_path, monkeypatch):
    """The full-graph test batch holds all 20 nodes, test seeds first: only
    the 6 seeds are plotted, as ``predict(ds.test_loader)`` returns them —
    not the train and validation nodes behind them."""
    pytest.importorskip("torch_geometric")
    from nnx import Nets
    from nnx.vis_utils import VisUtils

    ds, checkpoint, reloaded = _graph_checkpoint(getattr(Nets, net_name), tmp_path, monkeypatch, **net_kwargs)
    expected = reloaded.predict(ds.test_loader).logits
    (batch,) = ds.test_loader
    labels = batch.y[: batch.batch_size].numpy()
    assert len(expected) == len(labels) == 6 and batch.num_nodes == 20

    captured = _row_tsne(monkeypatch)
    fig = VisUtils.two_dim_tsne_checkpoint_logits(checkpoint, ds, n_samples=20, renderer=None)
    np.testing.assert_array_equal(captured["X"], expected)  # the t-SNE input, row for row
    assert sum(len(trace.x) for trace in fig.data) == 6
    assert _rows_by_class(fig) == [np.flatnonzero(labels == c).tolist() for c in range(2)]  # the test labels


def test_two_dim_tsne_plots_the_seed_rows_of_every_node_batch_in_order(tmp_path, monkeypatch):
    """Two NeighborLoader-shaped batches of 3 class-0 seeds then 5 class-1
    neighbors: the plot takes both batches' seeds in loader order, up to
    ``n_samples``, and never a neighbor row."""
    pytest.importorskip("torch_geometric")
    from nnx import Nets
    from nnx.vis_utils import VisUtils

    _, checkpoint, reloaded = _graph_checkpoint(Nets.GRAPH_CONV, tmp_path, monkeypatch)
    loader = [_node_batch(3, 5, offset=0), _node_batch(3, 5, offset=100)]
    expected = reloaded.predict(loader).logits
    assert len(expected) == 6
    ds = types.SimpleNamespace(output_dim=2, test_loader=loader)

    captured = _row_tsne(monkeypatch)
    for n_samples, rows in ((4, 4), (6, 6), (20, 6)):
        fig = VisUtils.two_dim_tsne_checkpoint_logits(checkpoint, ds, n_samples=n_samples, renderer=None)
        np.testing.assert_array_equal(captured["X"], expected[:rows])
        assert _rows_by_class(fig) == [list(range(rows)), []]  # every plotted row is a class-0 seed


def test_two_dim_tsne_refuses_node_batches_with_fewer_than_two_seed_rows(tmp_path, monkeypatch):
    """One seed row and five neighbors is one sample, not six."""
    pytest.importorskip("torch_geometric")
    from nnx import Nets
    from nnx.vis_utils import VisUtils

    _, checkpoint, _ = _graph_checkpoint(Nets.GRAPH_CONV, tmp_path, monkeypatch)
    ds = types.SimpleNamespace(output_dim=2, test_loader=[_node_batch(1, 5, offset=0)])
    _row_tsne(monkeypatch)
    with pytest.raises(ValueError, match="at least 2 samples"):
        VisUtils.two_dim_tsne_checkpoint_logits(checkpoint, ds, n_samples=6, renderer=None)


@pytest.mark.parametrize(
    ("target", "shape"), [(torch.ones(6, 2), r"\(6, 2\)"), (torch.tensor(0), r"\(\)")], ids=["2-D", "0-d"]
)
def test_two_dim_tsne_refuses_a_node_batch_target_before_cutting_its_seed_rows(target, shape, monkeypatch):
    """The class-index refusal runs before the seed rows are cut: it names
    the batch's own target shape, and a 0-d target is refused, not sliced."""
    from nnx import vis_utils
    from nnx.vis_utils import VisUtils

    class _NeverTSNE:
        def __init__(self, **kwargs):
            raise AssertionError("t-SNE ran before the refusal")

    class _FakeModel:  # a built-in net whose batches lead with 3 seed rows
        net = types.SimpleNamespace(unpack_batch=lambda batch: batch, seed_count=lambda batch: 3)

        @staticmethod
        def predict(X):
            return types.SimpleNamespace(logits=np.zeros((len(X), 2)))

    monkeypatch.setattr(vis_utils, "NNModel", types.SimpleNamespace(from_checkpoint=lambda checkpoint: _FakeModel()))
    monkeypatch.setattr(vis_utils, "TSNE", _NeverTSNE)
    ds = types.SimpleNamespace(output_dim=2, test_loader=[(torch.randn(6, 4), target)])
    with pytest.raises(ValueError, match=rf"the targets are {shape}, not one class index per row"):
        VisUtils.two_dim_tsne_checkpoint_logits(checkpoint=object(), ds=ds, n_samples=6, renderer=None)


def test_two_dim_tsne_plots_every_row_of_a_non_graph_built_in_net(tmp_path, monkeypatch):
    """Tuple batches have no seed rows: a built-in net's rows are collected
    across batches up to ``n_samples``, as before."""
    from torch.utils.data import DataLoader, TensorDataset

    from nnx import Nets
    from nnx.vis_utils import VisUtils

    X = torch.randn(7, 4, generator=torch.Generator().manual_seed(3))
    Y = torch.tensor([0, 1, 0, 1, 1, 0, 1])
    loader = DataLoader(TensorDataset(X, Y), batch_size=3)
    checkpoint, reloaded = _trained_checkpoint(Nets.FEED_FWD, loader, None, 4, tmp_path, monkeypatch)
    expected = reloaded.predict(loader).logits
    ds = types.SimpleNamespace(output_dim=2, test_loader=loader)

    captured = _row_tsne(monkeypatch)
    fig = VisUtils.two_dim_tsne_checkpoint_logits(checkpoint, ds, n_samples=5, renderer=None)
    np.testing.assert_array_equal(captured["X"], expected[:5])
    assert _rows_by_class(fig) == [np.flatnonzero(Y[:5].numpy() == c).tolist() for c in range(2)]
