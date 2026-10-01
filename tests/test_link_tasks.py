"""Leakage-aware link and edge prediction (FEAT-027).

Edges are split, not nodes: a versioned manifest keeps canonical edges per
split, fixed held-out negatives and the topology policy; messages pass over
training edges only; every batch is checked against the split before any
forward pass; evaluation materialises every candidate for exact AUROC / AP.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nnx import Losses, MonitorSpec, NNCheckpoint, NNModel, NNModelParams, NNOptimParams, NNTrainParams
from nnx.components import ComponentRestoreError
from nnx.link_tasks import (
    LinkPredictor,
    LinkSplit,
    LinkTask,
    LinkTaskError,
    link_metrics,
    link_predictor_spec,
    split_links,
)
from nnx.nn.enum.checkpoints import Checkpoints


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")


PATH = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 0-1-2-3: complement {(0,2), (0,3), (1,3)}


def _sbm(n: int = 30, seed: int = 0):
    """A two-community graph: dense inside, sparse across."""
    g = torch.Generator().manual_seed(seed)
    community = torch.arange(n) % 2
    edges = [
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if torch.rand(1, generator=g).item() < (0.4 if community[u] == community[v] else 0.03)
    ]
    x = torch.nn.functional.one_hot(community, 2).float() + 0.1 * torch.randn(n, 2, generator=g)
    return torch.tensor(edges).t().contiguous(), x


def _model(seed: int = 0, **settings) -> NNModel:
    spec = link_predictor_spec(input_dim=2, hidden_dims=[16], seed=seed, **settings)
    return NNModel(params=NNModelParams(net=spec, loss=Losses.MEAN_SQUARED_ERROR))


def _train(task: LinkTask, x: torch.Tensor, epochs: int = 2, **kwargs):
    model = kwargs.pop("model", None) or _model()
    run = model.train(
        params=NNTrainParams(
            n_epochs=epochs,
            train_loader=task.loader("train", x, batch_size=32, seed=0),
            val_loader=task.loader("val", x, batch_size=16),
            optim=NNOptimParams.builder().adam(max_lr=1e-2).build(),
            metrics=task.metric_specs(),
            monitor=MonitorSpec("auroc"),
            **kwargs,
        ),
        objective=task.objective(),
        eval_step_fn=task.eval_step(),
    )
    return model, run


# --- AC1: the manifest ------------------------------------------------------------------------------


def test_manifest_replay():
    edge_index, _ = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=7)
    assert split.replay(edge_index) == split
    assert (
        LinkSplit.from_state(split.state()) == split and LinkSplit.from_state(split.state()).digest() == split.digest()
    )
    state = split.state()
    assert state["topology"] == "train_edges" and state["seed"] == 7 and state["directed"] is False
    assert split_links(edge_index, 30, val=0.1, test=0.1, seed=8).digest() != split.digest()
    with pytest.raises(LinkTaskError, match="does not reproduce"):
        split.replay(edge_index[:, 1:])
    # Membership is independent of the input order, duplicates and reverses.
    shuffled = torch.cat([edge_index.flip(0), edge_index], dim=1)[:, torch.randperm(2 * edge_index.shape[1])]
    assert split_links(shuffled, 30, val=0.1, test=0.1, seed=7) == split


def test_reverse_duplicate_rejected():
    with pytest.raises(LinkTaskError, match="not canonical.*\\(0, 1\\)"):
        LinkSplit(num_nodes=4, train=((0, 1),), test=((1, 0),))
    with pytest.raises(LinkTaskError, match="already in 'train'"):
        LinkSplit(num_nodes=4, train=((0, 1),), test=((0, 1),))
    with pytest.raises(LinkTaskError, match="already in 'val'"):
        LinkSplit(num_nodes=4, train=((0, 1),), val=((1, 2),), val_negatives=((1, 2),))
    with pytest.raises(LinkTaskError, match="self-loop"):
        LinkSplit(num_nodes=4, train=((2, 2),))
    directed = LinkSplit(num_nodes=4, train=((0, 1),), test=((1, 0),), directed=True)  # distinct when directed
    assert directed.test == ((1, 0),)
    both = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])  # every edge with its reverse
    split = split_links(both, 4, val=1, test=1, seed=0, negatives=0)
    assert len(split.train) + len(split.val) + len(split.test) == 3  # one canonical edge each, one split each


# --- AC2: message topology vs supervision ------------------------------------------------------------------


def test_eval_topology_is_train_only():
    split = LinkSplit(num_nodes=4, train=((0, 1),), val=((1, 2),), test=((2, 3),))
    assert split.message_edge_index().tolist() == [[0, 1], [1, 0]]
    task = LinkTask(split, train_negatives=0)
    x = torch.randn(4, 2)
    for name in ("train", "val", "test"):
        for batch in task.loader(name, x, batch_size=4):
            assert batch.edge_index.tolist() == [[0, 1], [1, 0]]
            assert {tuple(e) for e in batch.edge_label_index.t().tolist()} == {split.positives(name)[0]}


def test_edge_label_mode():
    edges = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]])
    split = split_links(edges, 4, val=1, test=1, seed=1, edge_labels=[1, 0, 1, 0], categories=("follows", "blocks"))
    assert split.categories == ("follows", "blocks") and split.mode == "edge_label"
    assert split.val_negatives == () and split.test_negatives == ()  # a non-edge is never a category
    with pytest.raises(LinkTaskError, match="never a category"):
        LinkTask(split, mode="edge_label", train_negatives=1)
    with pytest.raises(LinkTaskError, match="cannot serve"):
        LinkTask(split, mode="binary")
    task = LinkTask(split, mode="edge_label", train_negatives=0)
    labels = split.labels()
    x = torch.randn(4, 2)
    (batch,) = list(task.loader("val", x, batch_size=8))
    pairs = [tuple(e) for e in batch.edge_label_index.t().tolist()]
    assert batch.edge_label.tolist() == [labels[p] for p in pairs] and batch.edge_label.dtype == torch.long
    assert {tuple(e) for e in batch.edge_index.t().tolist()} == {*split.train, *[(v, u) for u, v in split.train]}
    visible = split_links(
        edges,
        4,
        val=1,
        test=1,
        seed=1,
        edge_labels=[1, 0, 1, 0],
        categories=("follows", "blocks"),
        label_existence="visible",
    )
    assert visible.state()["topology"] == "train_edges+labelled_existence"
    assert visible.message_edge_index().shape[1] == 8  # every labelled edge's existence, both directions
    with pytest.raises(LinkTaskError, match="same|different categories"):
        split_links(torch.tensor([[0, 1], [1, 0]]), 4, val=0, test=0, seed=0, edge_labels=[0, 1], categories=("a", "b"))


# --- AC3: negatives -----------------------------------------------------------------------------------------


def test_negative_capacity(monkeypatch):
    import nnx.link_tasks as module

    sampled = []
    original = module._complement_sample
    monkeypatch.setattr(module, "_complement_sample", lambda *a, **k: sampled.append(1) or original(*a, **k))
    with pytest.raises(LinkTaskError, match="4 negatives requested but the complement holds only 3"):
        split_links(PATH, 4, val=1, test=1, seed=0, negatives=2)
    assert sampled == []  # refused before sampling
    split = split_links(PATH, 4, val=1, test=1, seed=0, negatives=1)
    complement = {(0, 2), (0, 3), (1, 3)}
    negatives = set(split.val_negatives) | set(split.test_negatives)
    assert negatives <= complement and len(negatives) == 2
    assert split.replay(PATH).val_negatives == split.val_negatives  # fixed: replay unchanged
    task = LinkTask(split, train_negatives=1)
    x = torch.randn(4, 2)
    for _ in range(3):  # training negatives are re-drawn per pass, always from the complement
        (batch,) = list(task.loader("train", x, batch_size=8))
        drawn = {
            tuple(e)
            for e, t in zip(batch.edge_label_index.t().tolist(), batch.edge_label.tolist(), strict=True)
            if t == 0
        }
        assert drawn <= complement - negatives
    loops = split_links(PATH, 4, val=1, test=1, seed=0, negatives=1, self_loops="allow")
    assert loops.self_loops == "allow"


# --- AC4: shapes and metrics --------------------------------------------------------------------------------


def test_metric_oracle():
    metrics = link_metrics([0.8, 0.3, 0.6, 0.1], [1, 0, 1, 0])
    assert metrics["bce"].value == pytest.approx(-(math.log(0.8) + math.log(0.7) + math.log(0.6) + math.log(0.9)) / 4)
    assert metrics["auroc"].value == 1.0 and metrics["ap"].value == 1.0
    categorical = link_metrics([[0.7, 0.3], [0.2, 0.8]], [0, 1])
    assert categorical["nll"].value == pytest.approx(-(math.log(0.7) + math.log(0.8)) / 2)
    assert categorical["accuracy"].value == 1.0


def test_one_class_unavailable():
    metrics = link_metrics([0.8, 0.6], [1, 1])
    assert metrics["auroc"].value is None and "only class [1]" in metrics["auroc"].reason
    assert metrics["ap"].value is None and metrics["bce"].value is not None
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.15, test=0.1, seed=3, negatives=0)  # validation: positives only
    task = LinkTask(split)
    _, run = _train(task, x, epochs=2)
    records = [idp for idp in run.idps if idp.val_edp is not None]
    assert all("auroc" not in idp.val_edp.metrics and idp.val_edp.metrics["negatives"] == 0 for idp in records)
    assert all(not idp.selection.improved for idp in records)  # an unavailable AUROC never wins BEST


def test_binary_and_categorical_candidates_keep_their_shapes():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=2)
    binary = LinkTask(split).predict(_model(), LinkTask(split).loader("val", x, batch_size=5))
    k = len(split.val) + len(split.val_negatives)
    assert binary.logits.shape == binary.probabilities.shape == binary.targets.shape == (k,)
    assert binary.ids.shape == (k,) and binary.pairs.shape == (k, 2)
    categories = [i % 3 for i in range(edge_index.shape[1])]
    labelled = split_links(
        edge_index, 30, val=0.1, test=0.1, seed=2, edge_labels=categories, categories=("a", "b", "c")
    )
    task = LinkTask(labelled, mode="edge_label", train_negatives=0)
    model = _model(decoder="mlp", num_categories=3)
    out = task.predict(model, task.loader("val", x, batch_size=5))
    assert out.logits.shape == out.probabilities.shape == (len(labelled.val), 3) and out.targets.shape == (
        len(labelled.val),
    )
    assert np.allclose(out.probabilities.sum(axis=1), 1.0)
    with pytest.raises(LinkTaskError, match="needs \\(.*3\\)"):
        task.predict(_model(), task.loader("val", x, batch_size=5))  # a binary head on a categorical task


# --- AC5: leakage, training, reload ----------------------------------------------------------------------------


def test_leakage_detected_before_training():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=4)
    task = LinkTask(split)
    (batch,) = list(task.loader("train", x, batch_size=10_000))
    leaked = split.test[0]
    batch.edge_index = torch.cat(
        [batch.edge_index, torch.tensor([[leaked[0], leaked[1]], [leaked[1], leaked[0]]])], dim=1
    )
    model = _model()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    with pytest.raises(LinkTaskError, match="leak: held-out positive"):
        model.train(
            params=NNTrainParams(
                n_epochs=1, train_loader=[batch], optim=NNOptimParams.builder().sgd(max_lr=0.1).build()
            ),
            objective=task.objective(),
        )
    assert all(torch.equal(before[k], v) for k, v in model.net.state_dict().items())
    (val_batch,) = list(task.loader("val", x, batch_size=10_000))
    val_batch.edge_label = 1 - val_batch.edge_label  # a held-out negative claimed as positive
    with pytest.raises(LinkTaskError, match="positive but is not one|labelled negative"):
        task.check_batch(val_batch)


def test_save_reload_preserves_order():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=5)
    task = LinkTask(split)
    model, run = _train(task, x, epochs=3)
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.model_params.net.id == "nnx.link_predictor"
    reloaded = NNModel.from_checkpoint(checkpoint)
    assert isinstance(reloaded.net, LinkPredictor)
    before = task.predict(model, task.loader("test", x, batch_size=4))
    after = task.predict(reloaded, task.loader("test", x, batch_size=4))
    assert before.ids.tolist() == after.ids.tolist() and np.allclose(
        before.probabilities, after.probabilities, atol=1e-6
    )
    ids = split.candidate_ids("test")
    assert before.ids.tolist() == [ids[tuple(p)] for p in before.pairs.tolist()]
    # Reordering the candidates moves ids and scores together.
    (batch,) = list(task.loader("test", x, batch_size=10_000))
    perm = torch.randperm(batch.edge_label_index.shape[1], generator=torch.Generator().manual_seed(0))
    batch.edge_label_index, batch.edge_label, batch.candidate_id = (
        batch.edge_label_index[:, perm],
        batch.edge_label[perm],
        batch.candidate_id[perm],
    )
    permuted = task.predict(reloaded, [batch])
    by_id = dict(zip(after.ids.tolist(), after.probabilities.tolist(), strict=True))
    assert all(
        math.isclose(by_id[i], p, abs_tol=1e-6)
        for i, p in zip(permuted.ids.tolist(), permuted.probabilities.tolist(), strict=True)
    )
    state = NNCheckpoint.load_training_state(run=run.id, type=Checkpoints.LAST)["components"]["link.task"]["state"]
    assert state["manifest"] == split.state() and state["manifest"]["val_negatives"] == [
        list(e) for e in split.val_negatives
    ]


# --- AC6: checkpointed manifest, adapters, streaming -------------------------------------------------------------


def test_mutated_manifest_rejected():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=6)
    _, run = _train(LinkTask(split), x, epochs=1)
    mutated = split_links(edge_index, 30, val=0.1, test=0.1, seed=9)
    model = _model()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    with pytest.raises(ComponentRestoreError, match="link split"):
        _train(LinkTask(mutated), x, epochs=2, model=model, resume_from_run_id=run.id)
    assert all(torch.equal(before[k], v) for k, v in model.net.state_dict().items())


def test_pool_rejects_edge_batches():
    from nnx import Activations, Devices, Nets, NNParams
    from nnx.graph_tasks import GraphClassifier, GraphTaskError

    edge_index, x = _sbm()
    task = LinkTask(split_links(edge_index, 30, val=0.1, test=0.1, seed=1))
    (batch,) = list(task.loader("val", x, batch_size=100))
    with pytest.raises(GraphTaskError, match="edge-label \\(link\\) batch"):
        GraphClassifier(torch.nn.Identity()).unpack_batch(batch)
    node_model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.GRAPH_CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    with pytest.raises(ValueError, match="edge-label \\(link\\) batch"):
        node_model.net.unpack_batch(batch)
    with pytest.raises(ValueError, match="edge-label \\(link\\) batch"):
        node_model.predict(torch.utils.data.DataLoader([batch], batch_size=None))


def test_streaming_materialises():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.2, test=0.1, seed=3)
    task = LinkTask(split)
    model = _model()

    class Ctx:
        def __init__(self, loader):
            self.model, self.val_loader, self.extra_metrics = model, loader, None

    whole = task.eval_step()(Ctx(task.loader("val", x, batch_size=10_000)))
    for size in (1, 3, 7):  # per-batch AUROC would differ; the materialised one cannot
        assert task.eval_step()(Ctx(task.loader("val", x, batch_size=size))).metrics == whole.metrics
    assert whole.count == len(split.val) + len(split.val_negatives) and "auroc" in whole.metrics
    small = LinkTask(split, max_candidates=5)
    with pytest.raises(LinkTaskError, match="max_candidates=5"):
        small.eval_step()(Ctx(small.loader("val", x, batch_size=4)))
