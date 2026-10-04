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
    seed = kwargs.pop("negatives_seed", 0)
    run = model.train(
        params=NNTrainParams(
            n_epochs=epochs,
            train_loader=task.loader("train", x, batch_size=32, seed=seed),
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


# --- review round 1 ------------------------------------------------------------------------------------------


def test_the_default_recipe_can_predict_no_link():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=5)
    task = LinkTask(split)
    model, _ = _train(task, x, epochs=30, seed=0)
    assert model.net.encoder.last_activation is False  # unconstrained embeddings: dot logits can be negative
    prediction = task.predict(model, task.loader("test", x, batch_size=64))
    assert prediction.logits.min() < 0
    assert link_metrics(prediction.probabilities, prediction.targets)["accuracy"].value > 0.6


def test_replay_without_validation_edges():
    edge_index, _ = _sbm()
    for negatives in (0, 3):
        split = split_links(edge_index, 30, val=0, test=0.2, seed=0, negatives=negatives)
        assert len(split.test_negatives) == negatives * len(split.test)
        assert split.replay(edge_index) == split
    assert split_links(edge_index, 30, val=0, test=0, seed=0, negatives=2).replay(edge_index).test == ()


def test_batches_are_bound_to_their_role():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=4)
    task = LinkTask(split)
    model = _model()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    with pytest.raises(LinkTaskError, match="'train' batches only, got a 'test' batch"):
        model.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=task.loader("test", x, batch_size=8),
                optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            ),
            objective=task.objective(),
        )
    assert all(torch.equal(before[k], v) for k, v in model.net.state_dict().items())

    def negative_batch(edge):
        (batch,) = list(task.loader("train", x, batch_size=10_000))
        batch.edge_label_index = torch.cat([batch.edge_label_index, torch.tensor([[edge[0]], [edge[1]]])], dim=1)
        batch.edge_label = torch.cat([batch.edge_label, torch.zeros(1)])
        batch.candidate_id = torch.cat([batch.candidate_id, torch.tensor([-1])])
        return batch

    with pytest.raises(LinkTaskError, match="held-out \\(val / test\\) negative"):
        task.check_batch(negative_batch(split.test_negatives[0]))
    with pytest.raises(LinkTaskError, match="self-loop"):
        task.check_batch(negative_batch((5, 5)))
    with pytest.raises(LinkTaskError, match="outside the graph"):
        task.check_batch(negative_batch((5, 30)))

    class Ctx:
        def __init__(self, loader):
            self.model, self.val_loader, self.extra_metrics = model, loader, None

    evaluate = task.eval_step()
    assert evaluate(Ctx(task.loader("val", x, batch_size=8))).count == len(split.val) + len(split.val_negatives)
    for name in ("train", "test"):  # selecting BEST on train or test would leak
        with pytest.raises(LinkTaskError, match="reads the 'val' split"):
            evaluate(Ctx(task.loader(name, x, batch_size=8)))
    with pytest.raises(LinkTaskError, match="one split"):
        evaluate(Ctx([*task.loader("val", x, batch_size=8), *task.loader("test", x, batch_size=8)]))
    with pytest.raises(LinkTaskError, match="exactly once"):
        evaluate(Ctx([*task.loader("val", x, batch_size=8), *task.loader("val", x, batch_size=8)]))
    with pytest.raises(LinkTaskError, match="exactly once"):
        evaluate(Ctx(list(task.loader("val", x, batch_size=4))[:1]))


def test_candidate_ids_must_match_their_pairs():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=4)
    task = LinkTask(split)
    (batch,) = list(task.loader("test", x, batch_size=10_000))
    batch.candidate_id = batch.candidate_id.flip(0)
    with pytest.raises(LinkTaskError, match="carries id"):
        task.predict(_model(), [batch])


def test_a_resumed_run_continues_the_training_negatives():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=4)
    whole, _ = _train(LinkTask(split), x, epochs=4, seed=0)
    first, run = _train(LinkTask(split), x, epochs=2, seed=0)
    resumed, _ = _train(LinkTask(split), x, epochs=2, seed=0, model=first, resume_from_run_id=run.id)
    assert all(
        torch.equal(a, b)
        for a, b in zip(whole.net.state_dict().values(), resumed.net.state_dict().values(), strict=True)
    )

    # A resumed loader drawing negatives from another seed is refused before any update.
    model = _model()
    with pytest.raises(LinkTaskError, match="from seed 0, the resumed loader from seed 4"):
        _train(LinkTask(split), x, epochs=1, seed=0, model=model, resume_from_run_id=run.id, negatives_seed=None)


def test_training_negatives_follow_the_epoch_not_the_iteration_count():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=4)
    task = LinkTask(split)
    loader = task.loader("train", x, batch_size=10_000, seed=0)

    def drawn():
        (batch,) = list(loader)
        return batch.link_pass, sorted(map(tuple, batch.edge_label_index.t().tolist()))

    first, second = drawn(), drawn()  # outside training: one pass per iteration
    assert first[0] == 0 and second[0] == 1 and first[1] != second[1]
    loader.set_epoch(5)
    assert drawn() == drawn() and drawn()[0] == 5  # within an epoch, extra iterations draw the same pass

    # The training loop announces each epoch, for NNModel.train and Trainer.train alike.
    seen = []

    class Recording(list):
        def set_epoch(self, epoch):
            seen.append(epoch)

    (batch,) = list(task.loader("train", x, batch_size=10_000, seed=0))
    objective = task.objective()
    model = _model()
    for _ in range(2):  # one borrowed objective, two fresh fits: both start at epoch 0
        model.train(
            params=NNTrainParams(
                n_epochs=2, train_loader=Recording([batch]), optim=NNOptimParams.builder().sgd(max_lr=0.1).build()
            ),
            objective=objective,
            salt=str(len(seen)),
        )
    assert seen == [0, 1, 0, 1]
    from nnx.trainer import NNTrainerParams, Trainer

    seen.clear()
    params = (
        NNTrainerParams.builder()
        .n_epochs(2)
        .train_loader(Recording([batch]))
        .optimizer("default", NNOptimParams.builder().sgd(max_lr=0.1).build())
        .save_phase_checkpoints(False)
        .build()
    )
    Trainer(_model()).train(params=params, objective=objective)
    assert seen == [0, 1]


def test_the_default_paths_refuse_link_batches():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=4)
    task = LinkTask(split)
    for settings in ({}, {"decoder": "mlp"}):
        model = _model(**settings)
        before = {k: v.clone() for k, v in model.net.state_dict().items()}
        with pytest.raises(LinkTaskError, match="through its LinkTask"):
            model.train(
                params=NNTrainParams(
                    n_epochs=1,
                    train_loader=task.loader("train", x, batch_size=8),
                    optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
                )
            )
        assert all(torch.equal(before[k], v) for k, v in model.net.state_dict().items())
        with pytest.raises(LinkTaskError, match="through its LinkTask"):
            model.predict(torch.utils.data.DataLoader(list(task.loader("test", x, batch_size=8)), batch_size=None))


def test_edge_label_defaults_shuffled_training_batches_and_conflicting_labels():
    edges = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]])
    labelled = split_links(edges, 4, val=1, test=1, seed=1, edge_labels=[1, 0, 1, 0], categories=("a", "b"))
    task = LinkTask(labelled)
    assert task.mode == "edge_label" and task.train_negatives == 0
    edge_index, x = _sbm()
    binary = LinkTask(split_links(edge_index, 30, val=0.1, test=0.1, seed=4))
    first = next(iter(binary.loader("train", x, batch_size=16, seed=0)))
    assert set(first.edge_label.tolist()) == {0.0, 1.0}  # not every positive first
    with pytest.raises(LinkTaskError, match="more than one category"):
        LinkSplit(num_nodes=4, train=((0, 1),), categories=("a", "b"), edge_labels=(((0, 1), 0), ((0, 1), 1)))


def test_round_two_batch_and_setting_contracts():
    import numpy as np_

    from nnx import Activations

    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=4)
    task = LinkTask(split)
    (batch,) = list(task.loader("val", x, batch_size=10_000))
    assert split._is_message_graph(batch.edge_index)
    assert split._is_message_graph(batch.edge_index[:, torch.randperm(batch.edge_index.shape[1])])  # any order
    task.check_batch(batch)
    del batch.edge_label
    with pytest.raises(LinkTaskError, match="one edge_label"):
        task.check_batch(batch)
    labelled = LinkSplit(
        num_nodes=4,
        train=((0, 1),),
        categories=("a", "b"),
        edge_labels=(((np_.int64(0), torch.tensor(1)), np_.int64(1)),),
    )
    assert labelled.labels() == {(0, 1): 1} and labelled.digest().startswith("sha256:")
    assert link_predictor_spec(input_dim=2, activation=Activations.TANH).config["activation"] == "tanh"
    # Edge-label training rows are shuffled per pass too.
    edges = torch.tensor([[i for i in range(12)], [i + 1 for i in range(12)]])
    categories = split_links(
        edges, 13, val=1, test=1, seed=0, edge_labels=[i % 2 for i in range(12)], categories=("a", "b")
    )
    loader = LinkTask(categories).loader("train", torch.randn(13, 2), batch_size=100, seed=0)
    orders = [next(iter(loader)).edge_label_index.t().tolist() for _ in range(2)]
    assert orders[0] != orders[1] and sorted(orders[0]) == sorted(orders[1])


# --- review round 3 ------------------------------------------------------------------------------------------


def test_out_of_graph_message_ids_cannot_alias_a_training_edge():
    edge_index, x = _sbm()
    split = split_links(edge_index, 30, val=0.1, test=0.1, seed=4)
    task = LinkTask(split)
    (batch,) = list(task.loader("val", x, batch_size=10_000))
    u, v = (int(i) for i in batch.edge_index[:, 0])
    for alias in ((u + 1, v - 30), (u - 1, v + 30)):  # u*30+v keys that collide with a message edge
        forged = batch.edge_index.clone()
        forged[:, 0] = torch.tensor(alias)
        assert not split._is_message_graph(forged)
        bad = batch.clone()
        bad.edge_index = forged
        with pytest.raises(LinkTaskError, match="training topology"):
            task.check_batch(bad)


def test_the_loader_seed_is_validated():
    edge_index, x = _sbm()
    task = LinkTask(split_links(edge_index, 30, val=0.1, test=0.1, seed=4))
    for seed in (-1, 0.9, True, "7"):
        with pytest.raises(LinkTaskError, match="seed"):
            task.loader("train", x, batch_size=8, seed=seed)
    with pytest.raises(LinkTaskError, match="candidates are fixed"):
        task.loader("val", x, batch_size=8, seed=0)


# --- review round 4 ------------------------------------------------------------------------------------------


class _Wrapper:  # a prefetch-style wrapper
    def __init__(self, inner, forward=True):
        self.inner, self.forward = inner, forward

    def __iter__(self):
        return iter(self.inner)

    def __len__(self):
        return len(self.inner)

    def set_epoch(self, epoch):
        if self.forward:
            self.inner.set_epoch(epoch)


def _fit(loader, objective, epochs, salt=None, callbacks=(), **resume):
    model = _model()
    run = model.train(
        params=NNTrainParams(
            n_epochs=epochs,
            train_loader=loader,
            optim=NNOptimParams.builder().adam(max_lr=1e-2).build(),
            seed=0,
            **resume,
        ),
        objective=objective,
        salt=salt,
        callbacks=list(callbacks),
    )
    return model, run


def _same_weights(a, b):
    return all(torch.equal(p, q) for p, q in zip(a.net.state_dict().values(), b.net.state_dict().values(), strict=True))


def test_a_wrapper_that_forwards_set_epoch_and_materialised_batches_resume_at_parity():
    edge_index, x = _sbm()
    task = LinkTask(split_links(edge_index, 30, val=0.1, test=0.1, seed=4))

    def wrapped():
        return _Wrapper(task.loader("train", x, batch_size=32, seed=0))

    whole, _ = _fit(wrapped(), task.objective(), 4, salt="whole")
    _, half = _fit(wrapped(), task.objective(), 2, salt="half")
    resumed, _ = _fit(wrapped(), task.objective(), 2, resume_from_run_id=half.id)
    assert _same_weights(whole, resumed)

    loader = task.loader("train", x, batch_size=32, seed=0)
    twice = [*loader, *loader]  # one loader iterated twice: passes 0 and 1
    assert {batch.link_pass for batch in twice} == {0, 1}
    whole, _ = _fit(twice, task.objective(), 3, salt="listed-whole")
    _, half = _fit(twice, task.objective(), 1, salt="listed-half")
    resumed, _ = _fit(twice, task.objective(), 2, resume_from_run_id=half.id)
    assert _same_weights(whole, resumed)

    # A source trained through a wrapper that swallows set_epoch (its fresh pass count equals the epoch)
    # resumes with the task's own loader.
    def swallowing():
        return _Wrapper(task.loader("train", x, batch_size=32, seed=0), forward=False)

    whole, _ = _fit(swallowing(), task.objective(), 4, salt="nofwd-whole")
    _, half = _fit(swallowing(), task.objective(), 2, salt="nofwd-half")
    resumed, _ = _fit(task.loader("train", x, batch_size=32, seed=0), task.objective(), 2, resume_from_run_id=half.id)
    assert _same_weights(whole, resumed)


def test_a_failed_resume_leaves_nothing_armed_and_malformed_state_is_refused():
    from nnx.nn.callbacks import Callback

    edge_index, x = _sbm()
    task = LinkTask(split_links(edge_index, 30, val=0.1, test=0.1, seed=4))
    _, source = _fit(task.loader("train", x, batch_size=32, seed=0), task.objective(), 2, salt="source")

    class Boom(Callback):
        def on_epoch_begin(self, ctx):
            raise RuntimeError("boom")

    objective = task.objective()
    with pytest.raises(RuntimeError, match="boom"):
        _fit(
            task.loader("train", x, batch_size=32, seed=0),
            objective,
            1,
            salt="boom",
            callbacks=[Boom()],
            resume_from_run_id=source.id,
        )
    _fit(task.loader("train", x, batch_size=32, seed=7), objective, 1, salt="fresh")  # another seed: a new fit
    with pytest.raises(LinkTaskError, match="from seed 0, the resumed loader from seed 7"):
        _fit(task.loader("train", x, batch_size=32, seed=7), task.objective(), 1, resume_from_run_id=source.id)

    state = NNCheckpoint.load_training_state(run=source.id, type=Checkpoints.LAST)["components"]["link.task"]["state"]
    assert state["negatives_seed"] == 0 and state["epoch"] == 1
    assert task.objective().check_component_state(state, version=1) == []
    for bad in ({"negatives_seed": "0"}, {"epoch": -1}, {"epoch": True}):
        problems = task.objective().check_component_state({**state, **bad}, version=1)
        assert problems and "malformed" in problems[0]


# --- release review: confident logits ------------------------------------------------------------------------


class _Fixed(torch.nn.Module):
    """A link model whose float32 logits are given per candidate pair."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, x, edge_index, edge_label_index):
        return torch.tensor([self.logits[(u, v)] for u, v in edge_label_index.t().tolist()], dtype=torch.float32)


def _evaluate(task, logits, x, batch_size):
    from types import SimpleNamespace

    model = SimpleNamespace(net=_Fixed(logits), device=torch.device("cpu"))
    ctx = SimpleNamespace(model=model, val_loader=task.loader("val", x, batch_size=batch_size), extra_metrics=None)
    return task.eval_step()(ctx), task.predict(model, task.loader("val", x, batch_size=batch_size))


def test_saturated_logits_keep_exact_binary_metrics():
    from sklearn.metrics import average_precision_score, roc_auc_score

    positives = {(1, 2): 40.0, (2, 3): 20.0, (3, 4): -120.0}
    negatives = {(4, 5): 30.0, (5, 6): -20.0, (6, 7): 17.0, (0, 7): -40.0}
    split = LinkSplit(num_nodes=8, train=((0, 1),), val=tuple(positives), val_negatives=tuple(negatives))
    task, x = LinkTask(split), torch.zeros(8, 2)
    z = np.array([*positives.values(), *negatives.values()])
    y = np.array([1.0] * len(positives) + [0.0] * len(negatives))
    # Exact: 40 and 20 outrank four and three negatives (7 of 12 pairs); the positives rank 1st, 3rd and 7th.
    exact = {
        "bce": torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(z), torch.tensor(y)).item(),
        "auroc": 7 / 12,
        "ap": (1 + 2 / 3 + 3 / 7) / 3,
    }
    assert roc_auc_score(y, z) == pytest.approx(exact["auroc"], abs=1e-12)
    assert average_precision_score(y, z) == pytest.approx(exact["ap"], abs=1e-12)
    # float32 probabilities tie 40, 30, 20 and 17 at 1.0 and round -120 to 0: every value moves.
    saturated = link_metrics(torch.sigmoid(torch.tensor(z, dtype=torch.float32)).numpy(), y)
    assert all(saturated[name].value != pytest.approx(value, rel=1e-3) for name, value in exact.items())

    record, prediction = _evaluate(task, {**positives, **negatives}, x, batch_size=10_000)
    assert prediction.logits.tolist() == z.tolist()
    for name, value in exact.items():
        assert record.metrics[name] == pytest.approx(value, rel=1e-12, abs=1e-12)
    assert record.loss == record.metrics["bce"] and record.metrics["accuracy"] == pytest.approx(4 / 7)
    scored = link_metrics(prediction.logits, prediction.targets, from_logits=True)
    assert {name: m.value for name, m in scored.items()} == {
        k: v for k, v in record.metrics.items() if k not in ("positives", "negatives")
    }
    for size in (1, 3):  # the materialised arithmetic does not depend on the batches
        assert _evaluate(task, {**positives, **negatives}, x, batch_size=size)[0].metrics == record.metrics
    # Infinite logits rank first or last and cost what they should: 0 when right, inf when wrong.
    infinite = link_metrics([math.inf, 1.0, -math.inf, -1.0], [1, 1, 0, 0], from_logits=True)
    assert infinite["auroc"].value == 1.0 and infinite["ap"].value == 1.0
    assert infinite["bce"].value == pytest.approx(math.log1p(math.exp(-1.0)) / 2, rel=1e-12)
    assert link_metrics([-math.inf, 1.0], [1, 0], from_logits=True)["bce"].value == math.inf


def test_moderate_logits_score_as_their_probabilities():
    rng = np.random.default_rng(0)
    z = rng.normal(scale=3.0, size=200)
    y = (rng.random(200) < 0.4).astype(np.float64)
    from_logits, from_probabilities = link_metrics(z, y, from_logits=True), link_metrics(1 / (1 + np.exp(-z)), y)
    assert from_logits.keys() == from_probabilities.keys()
    for name, metric in from_logits.items():
        assert metric.value == pytest.approx(from_probabilities[name].value, rel=1e-12, abs=1e-12)
    rows, labels = rng.normal(scale=3.0, size=(200, 4)), rng.integers(0, 4, size=200)
    softmax = np.exp(rows - rows.max(axis=1, keepdims=True))
    softmax /= softmax.sum(axis=1, keepdims=True)
    categorical, reference = link_metrics(rows, labels, from_logits=True), link_metrics(softmax, labels)
    assert categorical["nll"].value == pytest.approx(reference["nll"].value, rel=1e-12)
    assert categorical["accuracy"].value == reference["accuracy"].value
    one_class = link_metrics([30.0, 40.0], [1, 1], from_logits=True)["auroc"]
    assert one_class.value is None and "only class [1]" in str(one_class.reason)
    with pytest.raises(LinkTaskError, match="from_logits must be a bool"):
        link_metrics(z, y, from_logits="False")  # type: ignore[arg-type]


def test_saturated_logits_keep_an_exact_categorical_nll():
    rows = {(1, 2): [0.0, 120.0, 0.0], (2, 3): [0.0, -120.0, 0.0], (3, 4): [50.0, 0.0, -50.0]}
    labels = {(0, 1): 0, (1, 2): 1, (2, 3): 1, (3, 4): 2}
    split = LinkSplit(
        num_nodes=6, train=((0, 1),), val=tuple(rows), categories=("a", "b", "c"), edge_labels=tuple(labels.items())
    )
    task, x = LinkTask(split), torch.zeros(6, 2)
    z, y = torch.tensor(list(rows.values()), dtype=torch.float64), torch.tensor([labels[e] for e in rows])
    exact = torch.nn.functional.cross_entropy(z, y).item()
    assert exact == pytest.approx((0 + (120 + math.log(2)) + 100) / 3, rel=1e-12)
    saturated = link_metrics(torch.softmax(z.float(), dim=-1).numpy(), y.numpy())["nll"].value  # exp(-120) underflows
    assert saturated is not None and saturated > 2 * exact
    for size in (1, 2, 10_000):
        record, prediction = _evaluate(task, rows, x, batch_size=size)
        assert record.loss == record.metrics["nll"] == pytest.approx(exact, rel=1e-12)
        assert record.metrics["accuracy"] == pytest.approx(1 / 3)
    assert link_metrics(prediction.logits, prediction.targets, from_logits=True)["nll"].value == record.loss
