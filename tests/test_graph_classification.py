"""Graph-level classification with explicit pooling and graph identity (FEAT-026).

A ``GraphCollection`` holds whole graphs with stable ids and graph-level
targets; a ``GraphClassifier`` (encoder → mean / sum pool → head) gives one
row per graph through NNx's ordinary train / evaluate / predict / reload
paths. Loss and metrics are averaged over labeled graphs, prediction rows
keep their graph ids, and node-level paths refuse collection batches.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNTrainParams,
    TaskSpec,
)
from nnx.graph_tasks import (
    IGNORE,
    GraphClassifier,
    GraphCollection,
    GraphPool,
    GraphTaskError,
    graph_classifier_spec,
)
from nnx.nn.enum.checkpoints import Checkpoints


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")


class Identity(nn.Module):
    """An encoder that returns the node features unchanged."""

    def forward(self, x, edge_index):
        return x


def _chain(n: int) -> torch.Tensor:
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    src = torch.arange(n - 1)
    return torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])])


def _graph(n: int, value: float, width: int = 1) -> Data:
    return Data(x=torch.full((n, width), float(value)), edge_index=_chain(n))


def _identity_model(pool: str = "mean", classes: int = 2) -> NNModel:
    return NNModel(
        module=GraphClassifier(Identity(), pool),
        params=NNModelParams(loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(classes, ignore_index=IGNORE)),
    )


# --- AC1 / AC2: one row per graph, graph-local pooling -------------------------------------------------


def test_a_batch_of_three_graphs_gives_three_rows_and_targets_in_id_order():
    graphs = [_graph(2, 1, width=2), _graph(5, 2, width=2), _graph(9, 3, width=2)]
    collection = GraphCollection(graphs, [7, 3, 11], targets=[1, 0, 1])
    (batch,) = list(collection.loader(batch_size=3))
    assert batch.batch_size == 3 and getattr(batch, "input_id", None) is None
    model = _identity_model()
    (x, edge_index, vector, ptr), y = model.net.unpack_batch(batch)
    assert y.tolist() == [1, 0, 1] and batch.graph_id.tolist() == [7, 3, 11]
    result = model.predict_proba(collection.loader(batch_size=3))
    assert result.probabilities.shape == (3, 2) and result.sample_ids.tolist() == [7, 3, 11]
    assert model.predict(collection.loader(batch_size=3)).logits.shape == (3, 2)


def test_mean_and_sum_pooling_are_graph_local_and_node_order_invariant():
    collection = GraphCollection([_graph(2, 1), _graph(5, 2), _graph(9, 3)], [0, 1, 2], targets=[0, 0, 0])
    (batch,) = list(collection.loader(batch_size=3))
    for mode, expected in (("mean", [1.0, 2.0, 3.0]), ("sum", [2.0, 10.0, 27.0])):
        pooled = GraphClassifier(Identity(), mode)(batch.x, batch.edge_index, batch.batch, batch.ptr)
        assert pooled.reshape(-1).tolist() == expected
    # Permuting the nodes inside each graph (edges remapped) changes nothing.
    torch.manual_seed(0)
    graphs = []
    for n in (2, 5, 9):
        x = torch.randn(n, 3)
        perm = torch.randperm(n)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(n)
        graphs.append((Data(x=x, edge_index=_chain(n)), Data(x=x[perm], edge_index=inverse[_chain(n)])))
    for mode in ("mean", "sum"):
        pool = GraphPool(mode)
        outs = []
        for side in (0, 1):
            b = Batch.from_data_list([pair[side] for pair in graphs])
            outs.append(pool(b.x, b.batch, b.num_graphs))
        assert torch.allclose(outs[0], outs[1], atol=1e-6)
    with pytest.raises(GraphTaskError, match="pool must be one of"):
        GraphPool("max")


# --- AC3: malformed collections and batches ------------------------------------------------------------


def test_duplicate_graph_ids_are_rejected():
    with pytest.raises(GraphTaskError, match="duplicate graph ids \\[4\\]"):
        GraphCollection([_graph(2, 1), _graph(3, 1)], [4, 4], targets=[0, 1])
    collection = GraphCollection([_graph(2, 1), _graph(3, 1)], [4, 5], targets=[0, 1])
    (batch,) = list(collection.loader(2))
    batch.graph_id = torch.tensor([4, 4])
    with pytest.raises(GraphTaskError, match="duplicate graph ids in one batch"):
        _identity_model().net.unpack_batch(batch)


def test_zero_node_graphs_are_rejected():
    with pytest.raises(GraphTaskError, match="graph 9 has no nodes"):
        GraphCollection([_graph(2, 1), Data(x=torch.empty((0, 1)), edge_index=_chain(0))], [8, 9], targets=[0, 1])
    collection = GraphCollection([_graph(2, 1), _graph(3, 1)], [4, 5], targets=[0, 1])
    (batch,) = list(collection.loader(2))
    batch.ptr = torch.tensor([0, 0, 5])  # a graph without nodes
    batch.graph_id = torch.tensor([4, 5])
    with pytest.raises(GraphTaskError, match="no nodes"):
        _identity_model().net.unpack_batch(batch)


def test_cross_graph_edges_are_rejected():
    outside = Data(x=torch.ones(2, 1), edge_index=torch.tensor([[0], [2]]))
    with pytest.raises(GraphTaskError, match="cross-graph edge"):
        GraphCollection([outside], [1], targets=[0])
    collection = GraphCollection([_graph(2, 1), _graph(3, 1)], [4, 5], targets=[0, 1])
    (batch,) = list(collection.loader(2))
    batch.edge_index = torch.cat([batch.edge_index, torch.tensor([[0], [4]])], dim=1)  # graph 4 -> graph 5
    with pytest.raises(GraphTaskError, match="joins two different graphs"):
        _identity_model().net.unpack_batch(batch)


def test_an_inconsistent_ptr_is_rejected():
    collection = GraphCollection([_graph(2, 1), _graph(3, 1)], [4, 5], targets=[0, 1])
    for ptr in (torch.tensor([0, 3, 5]), torch.tensor([0, 2, 4]), torch.tensor([1, 2, 5])):
        (batch,) = list(collection.loader(2))
        batch.ptr = ptr
        with pytest.raises(GraphTaskError, match="ptr"):
            _identity_model().net.unpack_batch(batch)


def test_an_unflagged_missing_target_is_rejected():
    with pytest.raises(GraphTaskError, match="graph 2 has no target; give it one or flag it"):
        GraphCollection([_graph(2, 1), _graph(3, 1)], [1, 2], targets=[0, None])
    flagged = GraphCollection([_graph(2, 1), _graph(3, 1)], [1, 2], targets=[0, None], unlabeled=[2])
    assert flagged.labels == (0, IGNORE) and flagged.labeled == 1
    (batch,) = list(flagged.loader(2))
    del batch.y
    batch.y = torch.tensor([0])  # one target for two graphs
    with pytest.raises(GraphTaskError, match="one target per graph"):
        _identity_model().net.unpack_batch(batch)


# --- AC4: per-graph normalisation ------------------------------------------------------------------------


def _nll_graphs(nlls, *, unlabeled=()):
    """Graphs whose mean-pooled logits ``[a, 0]`` give NLL ``k`` for class 0."""
    values = [-math.log(math.exp(k) - 1.0) for k in nlls]
    graphs = [
        Data(x=torch.tensor([[v, 0.0]] * n), edge_index=_chain(n)) for v, n in zip(values, (2, 5, 9), strict=False)
    ]
    ids = list(range(len(graphs)))
    targets = [None if i in unlabeled else 0 for i in ids]
    return GraphCollection(graphs, ids, targets=targets, unlabeled=list(unlabeled))


def test_the_loss_denominator_is_the_labeled_graph_count():
    model = _identity_model()
    record = model.evaluate(_nll_graphs([1.0, 2.0, 3.0]).loader(batch_size=2))  # batches of 2 then 1
    assert record.loss == pytest.approx(2.0) and record.count == 3  # not (1.5 + 3) / 2 = 2.25
    masked = model.evaluate(_nll_graphs([1.0, 2.0, 3.0], unlabeled=[1]).loader(batch_size=2))
    assert masked.loss == pytest.approx(4.0 / 2.0) and masked.count == 2


def test_accuracy_and_confusion_match_a_per_graph_reference():
    rng = np.random.default_rng(0)
    graphs, labels = [], []
    for _ in range(12):
        n = int(rng.integers(2, 10))
        logits = rng.normal(size=3)
        graphs.append(Data(x=torch.tensor(np.tile(logits, (n, 1)), dtype=torch.float32), edge_index=_chain(n)))
        labels.append(int(rng.integers(0, 3)))
    collection = GraphCollection(graphs, list(range(100, 112)), targets=labels)
    model = _identity_model(classes=3)
    record = model.evaluate(collection.loader(batch_size=5))
    predicted = model.predict(collection.loader(batch_size=5)).classes
    reference = [int(np.argmax(g.x[0].numpy())) for g in graphs]
    assert predicted.tolist() == reference
    assert record.accuracy == pytest.approx(np.mean(np.array(reference) == np.array(labels)))
    confusion = np.zeros((3, 3), dtype=int)
    for truth, guess in zip(labels, predicted.tolist(), strict=True):
        confusion[truth, guess] += 1
    expected = np.zeros((3, 3), dtype=int)
    for truth, guess in zip(labels, reference, strict=True):
        expected[truth, guess] += 1
    assert (confusion == expected).all() and confusion.sum() == 12


# --- AC5 / AC8: train, select, save, reload --------------------------------------------------------------


def _synthetic(n_graphs: int = 18, seed: int = 0) -> GraphCollection:
    """Graphs whose class is the dominant feature column, plus one unlabeled."""
    rng = np.random.default_rng(seed)
    graphs, labels = [], []
    for i in range(n_graphs):
        label = i % 3
        n = int(rng.integers(2, 10))
        x = rng.normal(scale=0.3, size=(n, 3))
        x[:, label] += 2.0
        graphs.append(Data(x=torch.tensor(x, dtype=torch.float32), edge_index=_chain(n)))
        labels.append(None if i == n_graphs - 1 else label)
    return GraphCollection(graphs, [1000 + i for i in range(n_graphs)], targets=labels, unlabeled=[1000 + n_graphs - 1])


def _recipe_model(seed: int = 0) -> NNModel:
    spec = graph_classifier_spec(input_dim=3, num_classes=3, hidden_dims=[16], pool="mean", seed=seed)
    return NNModel(
        params=NNModelParams(net=spec, loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(3, ignore_index=IGNORE))
    )


def _train(model: NNModel, collection: GraphCollection, epochs: int = 2):
    return model.train(
        params=NNTrainParams(
            n_epochs=epochs,
            train_loader=collection.loader(batch_size=4, shuffle=True, seed=0),
            val_loader=collection.loader(batch_size=5),
            optim=NNOptimParams.builder().adam(max_lr=1e-2).build(),
            seed=0,
        )
    )


def test_a_synthetic_collection_trains_saves_and_reloads_through_the_registered_recipe():
    collection = _synthetic()
    model = _recipe_model()
    run = _train(model, collection)
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.model_params.net.id == "nnx.graph_classifier"
    reloaded = NNModel.from_checkpoint(checkpoint)
    assert isinstance(reloaded.net, GraphClassifier) and reloaded.net.pool.mode == "mean"
    assert [type(m).__name__ for m in reloaded.net.encoder.layers] == ["GCNConv"]
    assert reloaded.net.head.out_features == 3  # encoder, pool and head rebuilt as declared
    loader = collection.subset([1000, 1001, 1002]).loader(batch_size=3)
    before, after = model.predict_proba(loader), reloaded.predict_proba(loader)
    assert after.sample_ids.tolist() == [1000, 1001, 1002]
    assert np.allclose(before.probabilities, after.probabilities, atol=1e-6)
    assert before.decoded.tolist() == after.decoded.tolist()
    # Records count labeled graphs, and so do BEST selection and the rendered series.
    reloaded_run = NNRun.load(run.id)
    val = [idp.val_edp for idp in reloaded_run.idps if idp.val_edp is not None]
    assert len(val) == 2 and all(edp.count == collection.labeled == 17 for edp in val)
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.val_edp.count == 17
    assert best.idp.val_edp.error == pytest.approx(min(edp.error for edp in val))


def test_rendering_weights_batch_records_by_their_labeled_graph_count():
    from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
    from nnx.nn.params.nn_run import _mean_field

    records = [
        NNEvaluationDataPoint(loss=1.5, kind="categorical", count=2, status="ok"),
        NNEvaluationDataPoint(loss=3.0, kind="categorical", count=1, status="ok"),
    ]
    assert _mean_field(records, "loss") == pytest.approx(2.0)  # (1 + 2 + 3) / 3, not 2.25
    assert _mean_field([NNEvaluationDataPoint(loss=1.5), NNEvaluationDataPoint(loss=3.0)], "loss") == 2.25
    assert math.isnan(_mean_field([NNEvaluationDataPoint()], "error"))


def test_unsupported_exports_fail_before_writing(tmp_path):
    from nnx.viz.netron import netron_export

    collection = _synthetic(6)
    model = _recipe_model()
    (batch,) = list(collection.loader(batch_size=6))
    example = (batch.x, batch.edge_index, batch.batch, batch.ptr)
    for export in (
        lambda path: model.to_onnx(str(path), example),
        lambda path: netron_export(model, str(path), example),
    ):
        path = tmp_path / "graph.onnx"
        with pytest.raises(NotImplementedError, match="export_state_dict"):
            export(path)
        assert not path.exists()
    weights = model.export_state_dict(str(tmp_path / "graph.pt"))
    assert os.path.exists(weights)


# --- AC6 / AC7: identity through every path --------------------------------------------------------------


def test_graph_ids_survive_device_moves_shuffling_and_concatenation():
    collection = _synthetic(9)
    model = _recipe_model()
    ordered = model.predict_proba(collection.loader(batch_size=4))
    assert ordered.sample_ids.tolist() == list(collection.ids)  # three batches, concatenated
    shuffled = model.predict_proba(collection.loader(batch_size=4, shuffle=True, seed=3))
    assert sorted(shuffled.sample_ids.tolist()) == list(collection.ids)
    assert shuffled.sample_ids.tolist() != list(collection.ids)
    by_id = dict(zip(ordered.sample_ids.tolist(), ordered.probabilities, strict=True))
    assert all(
        np.allclose(by_id[i], p, atol=1e-6)
        for i, p in zip(shuffled.sample_ids.tolist(), shuffled.probabilities, strict=True)
    )
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        (batch,) = list(collection.loader(batch_size=9))
        moved = batch.to(device)
        assert moved.graph_id.tolist() == list(collection.ids) and moved.y.shape == (9,)
    record = model.evaluate(collection.loader(batch_size=4, shuffle=True, seed=1))
    assert record.count == collection.labeled  # one row per labeled graph, whatever the order


def test_training_records_one_row_per_labeled_graph_and_never_slices_seeds():
    from nnx.objectives import ObjectiveContext, supervised_objective

    collection = _synthetic(7)
    model = _recipe_model()
    (batch,) = list(collection.loader(batch_size=7))
    assert batch.batch_size == 7  # a PyG Batch carries num_graphs here, yet nothing is sliced
    assert getattr(model.net, "seed_count", None) is None
    result = supervised_objective()(ObjectiveContext(model=model, batch=batch, epoch_idx=0, batch_idx=0))
    assert result.record.count == collection.labeled == 6


# --- AC9: node-level paths refuse collections -----------------------------------------------------------------


def test_node_level_paths_refuse_collection_inputs():
    from nnx import NNGraphDataset
    from nnx.vis_utils import VisUtils

    collection = _synthetic(6)
    node_model = NNModel(
        net_params=NNParams(input_dim=3, output_dim=3, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.GRAPH_CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    with pytest.raises(ValueError, match="graph-collection batch"):
        node_model.evaluate(collection.loader(batch_size=3))
    with pytest.raises(ValueError, match="graph-collection batch"):
        node_model.predict(collection.loader(batch_size=3))

    class Many:
        def __init__(self, root, transform=None):
            self.graphs = [collection[0], collection[1]]

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, index):
            return self.graphs[index]

    with pytest.raises(ValueError, match="classifies the nodes of ONE graph"):
        NNGraphDataset(ds_class=Many, sampler="full")
    run = _train(_recipe_model(), collection, epochs=1)
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)

    class Collected:
        output_dim = 3
        test_loader = collection.loader(batch_size=3)

    with pytest.raises(ValueError, match="does not project graph-collection batches"):
        VisUtils.two_dim_tsne_checkpoint_logits(checkpoint, Collected(), n_samples=4, renderer=None)
    assert not any(hasattr(collection, name) for name in ("train_loader", "val_loader", "test_loader"))


def test_recipe_settings_are_validated():
    with pytest.raises(GraphTaskError, match="encoder must be one of"):
        graph_classifier_spec(input_dim=3, num_classes=2, encoder="gin")
    with pytest.raises(GraphTaskError, match="num_classes"):
        graph_classifier_spec(input_dim=3, num_classes=1)
    with pytest.raises(GraphTaskError, match="hidden_dims"):
        graph_classifier_spec(input_dim=3, num_classes=2, hidden_dims=[])
    with pytest.raises(GraphTaskError, match="features; the collection has"):
        GraphCollection([_graph(2, 1, width=2), _graph(2, 1, width=3)], [1, 2], targets=[0, 1])
    with pytest.raises(GraphTaskError, match="not below num_classes"):
        GraphCollection([_graph(2, 1)], [1], targets=[3], num_classes=3)
    for encoder in ("graph_sage", "graph_att"):
        spec = graph_classifier_spec(input_dim=3, num_classes=3, hidden_dims=[8, 8], encoder=encoder, pool="sum")
        model = NNModel(params=NNModelParams(net=spec, loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(3)))
        assert model.predict_proba(_synthetic(4).subset([1000, 1001, 1002]).loader(3)).probabilities.shape == (3, 3)


# --- review round 1 -------------------------------------------------------------------------------------


def test_round_one_edges(tmp_path):
    from nnx.models import build_module
    from nnx.nn.nn_model import _batch_sample_count
    from nnx.viz.netron import netron_export

    collection = _synthetic(6)
    feed_fwd = NNModel(
        net_params=NNParams(input_dim=3, output_dim=3, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    with pytest.raises(ValueError, match="graph-collection batch"):
        feed_fwd.predict(collection.loader(batch_size=3))
    module = build_module(graph_classifier_spec(input_dim=3, num_classes=3, pool="sum"))
    (batch,) = list(collection.loader(batch_size=6))
    path = tmp_path / "bare.onnx"
    with pytest.raises(NotImplementedError, match="export_state_dict"):
        netron_export(module, str(path), (batch.x, batch.edge_index, batch.batch, batch.ptr))
    assert not path.exists()
    # float64 features (NumPy's default) run through a float32 model.
    doubles = GraphCollection(
        [Data(x=g.x.double(), edge_index=g.edge_index) for g in (collection[0], collection[1])], [1, 2], targets=[0, 1]
    )
    assert _recipe_model().predict_proba(doubles.loader(2)).probabilities.shape == (2, 3)
    # The recipe takes an Activations member and checks the feature width.
    spec = graph_classifier_spec(input_dim=3, num_classes=3, activation=Activations.RELU)
    assert spec.config["activation"] == "relu"
    wide = GraphCollection([_graph(3, 1.0, width=5)], [1], targets=[0])
    with pytest.raises(GraphTaskError, match="5 node features; the classifier reads 3"):
        _recipe_model().predict(wide.loader(1))
    # Custom training steps weigh a graph batch by its graphs, not as one sample.
    model = _recipe_model()
    assert _batch_sample_count(model.net, batch) == 6

    class Tagged:
        def __init__(self, root, transform=None):
            self.graph = collection[0]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return self.graph

    from nnx import NNGraphDataset

    with pytest.raises(ValueError, match="graph-collection item"):
        NNGraphDataset(ds_class=Tagged, sampler="full")


# --- review round 2 ------------------------------------------------------------------------------


def test_round_two_unlabeled_y_subset_iterators_and_unshuffled_seeds():
    labelled = _graph(2, 1)
    labelled.y = torch.tensor([1])
    with pytest.raises(GraphTaskError, match="flagged unlabeled but has a target"):
        GraphCollection([labelled, _graph(3, 1)], [1, 2], targets=None, unlabeled=[1])
    collection = GraphCollection([_graph(2, 1), _graph(3, 1)], [1, 2], targets=[0, None], unlabeled=[2])
    again = GraphCollection(list(collection), [1, 2], unlabeled=[2])  # its own items carry IGNORE as y
    assert again.labels == collection.labels
    assert collection.subset(iter([2, 1])).ids == (2, 1)
    with pytest.raises(GraphTaskError, match="shuffled loader"):
        collection.loader(batch_size=1, seed=3)
    assert len(list(collection.loader(batch_size=1, shuffle=True, seed=3))) == 2
