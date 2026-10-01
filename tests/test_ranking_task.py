"""Query-grouped ranking and retrieval evaluation (FEAT-035).

:class:`nnx.ranking.RankingTask` checks query / candidate ids, relevance
grades and masks before any update, trains a scorer with a pairwise
logistic objective over unequal-relevance pairs within one query, and
evaluates MRR@k, Recall@k and NDCG@k per complete query — buffered across
batches under a declared limit, ties broken by candidate id, queries with
no relevant candidate excluded and counted. The metrics drive a named
``MonitorSpec`` (BEST), persist in the run history and reach the loggers
with no classification placeholder.
"""

from __future__ import annotations

import ast
import itertools
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    MetricSpec,
    MonitorSpec,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNTrainParams,
)
from nnx.components import ComponentRestoreError
from nnx.nn.callbacks import WandbCallback
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.objectives import ObjectiveContext
from nnx.ranking import (
    RankingError,
    RankingTask,
    mrr_at_k,
    ndcg_at_k,
    pairwise_logistic_loss,
    rank,
    recall_at_k,
)

D = 4
GROUP = 4
CLASSIFICATION_FIELDS = ("error", "accuracy", "f1", "precision", "recall")


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")


def _scorer(seed: int = 0) -> NNModel:
    torch.manual_seed(seed)
    return NNModel(
        net_params=NNParams(input_dim=D, output_dim=1, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR),
    )


def _data(n_queries: int = 6, seed: int = 0, zero_query: bool = True):
    """``n_queries`` queries of ``GROUP`` candidates graded by a hidden
    linear utility (best 2, next 1, rest 0); the last query has no relevant
    candidate when ``zero_query``."""
    g = torch.Generator().manual_seed(seed)
    w = torch.tensor([1.0, -0.5, 0.25, 0.0])
    feats, queries, candidates, grades = [], [], [], []
    for q in range(n_queries):
        x = torch.randn(GROUP, D, generator=g)
        order = (x @ w).argsort(descending=True)
        rel = torch.zeros(GROUP, dtype=torch.long)
        if not (zero_query and q == n_queries - 1):
            rel[order[0]], rel[order[1]] = 2, 1
        feats.append(x)
        grades.append(rel)
        queries += [100 + q] * GROUP
        candidates += [1000 * q + c for c in range(GROUP)]
    return torch.cat(feats), torch.tensor(queries), torch.tensor(candidates), torch.cat(grades)


def _loader(data, batch_size: int) -> DataLoader:
    return DataLoader(TensorDataset(*data), batch_size=batch_size, shuffle=False)


class _Ctx:
    def __init__(self, model, loader):
        self.model, self.val_loader = model, loader


def _objective_step(task, model, batch):
    return task.objective()(ObjectiveContext(model=model, batch=batch, epoch_idx=0, batch_idx=0))


# --- AC1: validation and candidate identity ---------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"k": ()}, "at least one cutoff"),
        ({"k": (0,)}, "k must be an integer >= 1"),
        ({"k": (-1, 2)}, "k must be an integer >= 1"),
        ({"max_relevance": 0}, "max_relevance"),
        ({"relevance_threshold": 3, "max_relevance": 2}, "above max_relevance"),
        ({"weighting": "list"}, "weighting"),
        ({"candidate_sets": "full"}, "candidate_sets"),
        ({"max_buffered": 0}, "max_buffered"),
        ({"k": (5,), "group_size": 4}, "smaller than k"),
        ({"version": 2}, "version"),
    ],
)
def test_the_task_declares_and_validates_its_configuration(fields, match):
    with pytest.raises(RankingError, match=match):
        RankingTask(**fields)
    task = RankingTask(k=(10, 1, 10), max_relevance=3, relevance_threshold=2, weighting="pair", group_size=12)
    assert task.k == (1, 10)  # sorted, de-duplicated
    assert RankingTask.from_state(task.state()) == task
    with pytest.raises(RankingError, match="unknown keys"):
        RankingTask.from_state({**task.state(), "listwise": True})


def _bad_batches():
    x, q, c, r = _data(2)
    yield "length", (x, q, c[:-1], r), "query ids but"
    yield "duplicate candidate", (x, q, torch.cat([c[:1], c[:-1]]), r), "appears twice for query"
    yield "grade above max", (x, q, c, torch.where(r == 2, 3, r)), "grades must be in 0..2"
    yield "negative grade", (x, q, c, r - 1), "grades must be in 0..2"
    yield "fractional grade", (x, q, c, r.float() + 0.5), "finite integers"
    yield "boolean grade", (x, q, c, r > 0), "not booleans"
    yield "grade shape", (x, q, c, r[:-1]), "relevance must be a"
    yield "mask shape", (x, q, c, r, torch.ones(len(q) - 1, dtype=torch.bool)), "the mask must be a"
    yield "mask values", (x, q, c, r, torch.full((len(q),), 2)), "boolean or 0/1"
    yield "float ids", (x, q.float(), c, r), "query_ids must be"
    yield "mixed ids", (x, q, [*map(str, c[:-1].tolist()), 7], r), "candidate_ids must be"
    yield "empty string id", (x, q, ["a", *map(str, c[1:].tolist())][:-1] + [""], r), "candidate_ids must be"
    yield "tuple arity", (x, q, c), "a ranking batch is"
    yield "mapping keys", {"features": x, "query_ids": q, "relevance": r}, "needs features"
    yield "scorer rows", (x[:-1], q, c, r), "one score per row"


@pytest.mark.parametrize(("case", "batch", "match"), list(_bad_batches()), ids=[c for c, _, _ in _bad_batches()])
def test_malformed_batches_fail_before_any_update(case, batch, match):
    task = RankingTask(k=(2,), max_relevance=2)
    model = _scorer()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    with pytest.raises(RankingError, match=match):
        _objective_step(task, model, batch)
    with pytest.raises(RankingError, match=match):
        model.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=[batch],
                optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            ),
            objective=task.objective(),
        )
    assert all(torch.equal(before[k], v) for k, v in model.net.state_dict().items())  # no update happened


def test_a_masked_duplicate_is_ignored_and_a_masked_row_never_trains():
    task = RankingTask(k=(1,))
    model = _scorer()
    x = torch.randn(3, D)
    q, c, r = torch.tensor([1, 1, 1]), torch.tensor([7, 8, 7]), torch.tensor([1, 0, 0])
    result = _objective_step(task, model, (x, q, c, r, torch.tensor([True, True, False])))
    s = model.net(x)[:, 0].detach()
    assert result.terms[0].value == pytest.approx(math.log1p(math.exp(-(s[0] - s[1]).item())), rel=1e-5)


def test_candidate_identity_survives_sorting_device_moves_and_reload(tmp_path):
    # Sorting: positions come back in ranked order, ties broken by id.
    ids = ["doc-b", "doc-a", "doc-c", "doc-d"]
    assert [ids[i] for i in rank([0.5, 0.5, 0.9, 0.1], ids)] == ["doc-c", "doc-a", "doc-b", "doc-d"]
    with pytest.raises(RankingError, match="integers or all strings"):
        rank([1.0, 2.0], [1, "a"])
    # Device moves: ids are host values read before the features move, so a
    # batch held as int32 / int64 / non-contiguous tensors (or moved to any
    # available device) yields the same ids.
    task = RankingTask(k=(2,), max_relevance=2)
    x, q, c, r = _data(2)
    base = task.split((x, q, c, r))
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device, dtype in itertools.product(devices, (torch.int32, torch.int64)):
        moved = (x.to(device), q.to(device, dtype), c.flip(0).to(device, dtype).flip(0), r.to(device))
        _, qs, cs, rs, _ = task.split(moved)
        assert (qs, cs) == (base[1], base[2]) and torch.equal(rs, base[3]) and rs.device.type == "cpu"
    # Reload: a saved scorer reloaded from its checkpoint ranks the same ids.
    model = _scorer()
    loader = _loader(_data(3, seed=1), batch_size=5)
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=_loader(_data(3), batch_size=GROUP),
            val_loader=loader,
            optim=NNOptimParams.builder().sgd(max_lr=0.05).build(),
        ),
        objective=task.objective(),
        eval_step_fn=task.eval_step(),
    )
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    reloaded = NNModel.from_checkpoint(checkpoint)
    first = task.eval_step()(_Ctx(model, loader))
    second = task.eval_step()(_Ctx(reloaded, loader))
    assert first.metrics == second.metrics
    xs, qs, cs, _ = _data(3, seed=1)
    for query in sorted(set(qs.tolist())):
        rows = [i for i, v in enumerate(qs.tolist()) if v == query]
        ranked = [
            [cs[rows[i]].item() for i in rank(m.net(xs[rows]).detach()[:, 0].tolist(), cs[rows].tolist())]
            for m in (model, reloaded)
        ]
        assert ranked[0] == ranked[1]


# --- AC2: pairwise logistic objective -----------------------------------------------------------


def test_pairwise_logistic_loss_anchor_ties_and_no_pair_queries():
    numerator, denominator = pairwise_logistic_loss(torch.tensor([2.0, 0.0]), torch.tensor([1, 0]), [1, 1])
    assert denominator == 1 and float(numerator) == pytest.approx(0.126928, abs=1e-6)
    # Ties contribute nothing; a query whose grades are all equal is skipped.
    numerator, denominator = pairwise_logistic_loss(
        torch.tensor([2.0, 0.0, 5.0, 1.0, 3.0]), torch.tensor([1, 0, 0, 1, 1]), ["a", "a", "a", "b", "b"]
    )
    a = [math.log1p(math.exp(-(2.0 - 0.0))), math.log1p(math.exp(-(2.0 - 5.0)))]
    assert denominator == 1 and float(numerator) == pytest.approx(sum(a) / 2)
    # Query vs pair weighting: one query with one pair, one with three.
    scores = torch.tensor([1.0, 0.0, 0.3, 0.2, 0.1, 0.4])
    grades = torch.tensor([1, 0, 2, 1, 0, 0])
    queries = [1, 1, 2, 2, 2, 2]
    sp = lambda d: math.log1p(math.exp(-d))  # noqa: E731
    q1 = [sp(1.0)]
    q2 = [sp(0.3 - 0.2), sp(0.3 - 0.1), sp(0.3 - 0.4), sp(0.2 - 0.1), sp(0.2 - 0.4)]
    n, d = pairwise_logistic_loss(scores, grades, queries, weighting="query")
    assert d == 2 and float(n) / d == pytest.approx((q1[0] + sum(q2) / len(q2)) / 2)
    n, d = pairwise_logistic_loss(scores, grades, queries, weighting="pair")
    assert d == 6 and float(n) / d == pytest.approx((q1[0] + sum(q2)) / 6)
    # A microbatch with no unequal pair is an empty term (no update, no record value).
    task = RankingTask(k=(1,))
    result = _objective_step(task, _scorer(), (torch.randn(2, D), [1, 1], [1, 2], torch.tensor([0, 0])))
    assert result.terms[0].denominator == 0 and result.terms[0].value is None and result.record.status == "empty"
    # Gradients flow to the scorer.
    model = _scorer()
    result = _objective_step(task, model, (torch.randn(2, D), [1, 1], [1, 2], torch.tensor([1, 0])))
    result.terms[0].numerator.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.net.parameters())


# --- AC3: MRR@k, Recall@k, NDCG@k ------------------------------------------------------------------


def _reference_ndcg(grades, scores, ids, k):
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], ids[i]))[:k]
    dcg = sum((2 ** grades[i] - 1) / math.log2(r + 2) for r, i in enumerate(order))
    ideal = sorted(grades, reverse=True)[:k]
    return dcg / sum((2**g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))


def test_metrics_match_references():
    grades, scores, ids = [2, 0, 1], [3.0, 2.0, 1.0], [1, 2, 3]
    assert mrr_at_k(grades, scores, ids, 2) == 1.0
    assert recall_at_k(grades, scores, ids, 2) == 0.5
    assert ndcg_at_k(grades, scores, ids, 2) == pytest.approx(3 / (3 + 1 / math.log2(3)))
    # Multiple positives, graded relevance, a stricter threshold.
    grades, scores, ids = [0, 3, 1, 2, 0], [0.9, 0.1, 0.8, 0.5, 0.7], [10, 11, 12, 13, 14]
    assert mrr_at_k(grades, scores, ids, 3) == pytest.approx(1 / 2)  # 12 is second
    assert mrr_at_k(grades, scores, ids, 3, threshold=2) == 0.0  # missing positive in the top 3
    assert recall_at_k(grades, scores, ids, 4, threshold=2) == pytest.approx(1 / 2)  # 13 of {11, 13}
    for k in range(1, 6):
        assert ndcg_at_k(grades, scores, ids, k) == pytest.approx(_reference_ndcg(grades, scores, ids, k))
    # Ties break by candidate id (ascending), whatever the input order.
    assert mrr_at_k([0, 1], [0.5, 0.5], [3, 5], 1) == 0.0  # 3 ranks first
    assert mrr_at_k([1, 0], [0.5, 0.5], [5, 3], 1) == 0.0
    assert mrr_at_k([0, 1], [0.5, 0.5], ["b", "a"], 1) == 1.0
    # k over the group size, and k <= 0, are rejected; so are malformed queries.
    for bad in (0, -1, 4):
        with pytest.raises(RankingError, match="k"):
            ndcg_at_k([1, 0, 0], [1.0, 0.0, 0.5], [1, 2, 3], bad)
    with pytest.raises(RankingError, match="same candidate id twice"):
        mrr_at_k([1, 0], [1.0, 0.0], [1, 1], 1)
    with pytest.raises(RankingError, match="non-finite"):
        mrr_at_k([1, 0], [math.nan, 0.0], [1, 2], 1)
    with pytest.raises(RankingError, match="one grade, score and candidate id"):
        recall_at_k([1, 0, 0], [1.0, 0.0], [1, 2], 1)
    with pytest.raises(RankingError, match="no relevant"):
        recall_at_k([0, 0], [1.0, 0.0], [1, 2], 1)
    with pytest.raises(RankingError, match="every grade is 0"):
        ndcg_at_k([0, 0], [1.0, 0.0], [1, 2], 1)


# --- AC4: per complete query -------------------------------------------------------------------


def test_metrics_aggregate_per_complete_query_and_reordering_changes_nothing():
    task = RankingTask(k=(1, 2), max_relevance=2)
    model = _scorer()
    data = _data(5, seed=3)
    whole = task.eval_step()(_Ctx(model, _loader(data, batch_size=len(data[0]))))
    # Queries split across batches are joined: any batching gives the same values.
    for batch_size in (1, 3, 5, 7):
        assert task.eval_step()(_Ctx(model, _loader(data, batch_size))).metrics == whole.metrics
    # Reordering rows (queries interleaved) gives identical values.
    perm = torch.randperm(len(data[0]), generator=torch.Generator().manual_seed(0))
    shuffled = tuple(t[perm] for t in data)
    assert task.eval_step()(_Ctx(model, _loader(shuffled, 6))).metrics == whole.metrics
    # The mean is over queries: one value per complete query.
    x, q, c, r = data
    per_query = []
    for query in sorted(set(q.tolist())):
        rows = [i for i, v in enumerate(q.tolist()) if v == query]
        if int(r[rows].max()) == 0:
            continue
        s = model.net(x[rows]).detach()[:, 0].double().tolist()
        per_query.append(ndcg_at_k(r[rows].tolist(), s, c[rows].tolist(), 2))
    assert whole.metrics["ndcg_at_2"] == pytest.approx(sum(per_query) / len(per_query))
    # The all-zero-relevance query is excluded and counted, never scored 0.
    assert whole.count == 4 and whole.metrics["queries"] == 4.0 and whole.metrics["excluded_queries"] == 1.0
    assert whole.metrics["candidates"] == 4.0 * GROUP and whole.metrics["candidates_per_query"] == GROUP
    assert whole.metrics["exhaustive_candidates"] == 0.0


def test_duplicate_incomplete_and_oversized_streams_are_rejected():
    model = _scorer()
    x, q, c, r = _data(3)
    task = RankingTask(k=(2,), max_relevance=2)
    # The same candidate of a query arriving in two batches.
    stream = [(x[:4], q[:4], c[:4], r[:4]), (x[3:4], q[3:4], c[3:4], r[3:4])]
    with pytest.raises(RankingError, match="appears twice for query 100 in the evaluation stream"):
        task.eval_step()(_Ctx(model, stream))
    # A declared group size rejects an incomplete query.
    sized = RankingTask(k=(2,), max_relevance=2, group_size=GROUP)
    assert sized.eval_step()(_Ctx(model, _loader((x, q, c, r), 5))).count == 2
    with pytest.raises(RankingError, match="has 3 candidates; the task declares group_size=4"):
        sized.eval_step()(_Ctx(model, _loader((x[1:], q[1:], c[1:], r[1:]), 5)))
    # Without one, a query smaller than k is still rejected.
    with pytest.raises(RankingError, match="k=2 is larger than query 102's 1 candidates"):
        task.eval_step()(_Ctx(model, _loader((x[:9], q[:9], c[:9], r[:9]), 5)))
    # Buffering is bounded.
    with pytest.raises(RankingError, match="exceeds max_buffered=8"):
        RankingTask(k=(2,), max_relevance=2, max_buffered=8).eval_step()(_Ctx(model, _loader((x, q, c, r), 5)))
    # A scorer producing a non-finite score cannot be ranked.
    with torch.no_grad():
        model.net.state_dict()[next(iter(model.net.state_dict()))].fill_(math.nan)
    with pytest.raises(RankingError, match="non-finite score"):
        task.eval_step()(_Ctx(model, _loader((x, q, c, r), 5)))


def test_an_all_excluded_stream_is_unavailable_and_the_mode_is_restored():
    task = RankingTask(k=(1,))
    model = _scorer()
    model.net.train()
    x, q, c, r = _data(2)
    record = task.eval_step()(_Ctx(model, _loader((x, q, c, torch.zeros_like(r)), 3)))
    assert record.status == "empty" and record.count == 0 and record.loss is None
    assert dict(record.metrics) == {"queries": 0.0, "excluded_queries": 2.0, "exhaustive_candidates": 0.0}
    assert model.net.training


# --- AC5 / AC6: train, select, checkpoint, history, loggers, rendering --------------------------------


def _params(task: RankingTask, epochs: int = 3, **kwargs) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=epochs,
        train_loader=_loader(_data(6), batch_size=8),
        val_loader=_loader(_data(4, seed=1), batch_size=5),
        optim=NNOptimParams.builder().sgd(max_lr=0.2).build(),
        metrics=task.metric_specs(),
        monitor=MonitorSpec("ndcg_at_2"),
        **kwargs,
    )


class _FakeWandb:
    def __init__(self):
        self.logs = []

    def log(self, values, step):
        self.logs.append(dict(values))


def test_a_tiny_scorer_trains_selects_by_named_ndcg_and_checkpoints_the_task():
    task = RankingTask(k=(1, 2), max_relevance=2, candidate_sets="exhaustive", group_size=GROUP)
    wandb = _FakeWandb()
    run = _scorer().train(
        params=_params(task),
        objective=task.objective(),
        eval_step_fn=task.eval_step(),
        callbacks=[WandbCallback(wandb_run=wandb)],
    )
    reloaded = NNRun.load(run.id)  # the run history, as persisted
    val = [idp.val_edp for idp in reloaded.idps if idp.val_edp is not None]
    assert len(val) == 3
    for edp in val:
        assert edp.kind == "ranking" and edp.status == "ok" and edp.count == 3  # 3 scored queries
        assert set(edp.metrics) == {
            *task.metric_names(),
            "queries",
            "excluded_queries",
            "candidates",
            "candidates_per_query",
            "exhaustive_candidates",
        }
        assert edp.metrics["excluded_queries"] == 1.0 and edp.metrics["exhaustive_candidates"] == 1.0
        assert edp.metrics["candidates_per_query"] == GROUP
        assert all(getattr(edp, name) is None for name in CLASSIFICATION_FIELDS)  # no placeholder
    # Named BEST: the monitor selects on validation NDCG@2.
    values = [edp.metrics["ndcg_at_2"] for edp in val]
    selections = [idp.selection for idp in reloaded.idps if idp.selection is not None]
    assert selections[0].monitor.key == "val.ndcg_at_2" and selections[0].monitor.mode == "max"
    best_so_far, expected = -math.inf, []
    for value in values:
        expected.append(value > best_so_far)
        best_so_far = max(best_so_far, value)
    assert [s.improved for s in selections] == expected
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.val_edp.metrics["ndcg_at_2"] == max(values)
    # Loggers and rendering carry the ranking metrics, never a classification field.
    logged = [entry for entry in wandb.logs if "val/ndcg_at_2" in entry]
    assert len(logged) == 3 and all("val/excluded_queries" in e and "val/queries" in e for e in logged)
    assert not any(f"val/{name}" in entry for entry in wandb.logs for name in CLASSIFICATION_FIELDS)
    assert [entry["monitor/val.ndcg_at_2"] for entry in logged] == pytest.approx(values)
    html = reloaded._repr_html_()
    assert "ndcg_at_2" in html and "monitor: val.ndcg_at_2" in html
    assert "ndcg_at_2" in str(reloaded)
    assert reloaded._epoch_series()["val_err"] == pytest.approx([math.nan] * 3, nan_ok=True)
    # The task configuration is checkpointed component state.
    state = NNCheckpoint.load_training_state(run=run.id, type=Checkpoints.LAST)["components"]["ranking.task"]["state"]
    assert state["task"] == task.state() and RankingTask.from_state(state["task"]) == task
    pairs_per_epoch = 5  # 5 graded training queries per epoch, query weighting
    assert state["units"] == 3 * pairs_per_epoch

    resumed = task.objective()
    child = _scorer().train(
        params=_params(task, 1, resume_from_run_id=run.id), objective=resumed, eval_step_fn=task.eval_step()
    )
    assert resumed.units == 4 * pairs_per_epoch and child.resume_status.mode == "stateful"
    changed = RankingTask(k=(1, 2), max_relevance=2, candidate_sets="sampled", group_size=GROUP)
    model = _scorer()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    with pytest.raises(ComponentRestoreError, match="ranking task changed"):
        model.train(
            params=_params(changed, 1, resume_from_run_id=run.id, overwrite_existing=True),
            objective=changed.objective(),
            eval_step_fn=changed.eval_step(),
            salt="changed",
        )
    assert all(torch.equal(before[k], v) for k, v in model.net.state_dict().items())  # refused before any update


def test_ranking_metrics_need_the_ranking_eval_step():
    task = RankingTask(k=(2,), max_relevance=2)
    model = _scorer()
    loader = [(torch.randn(4, D), torch.randn(4, 1))]
    with pytest.raises(RankingError, match="query-grouped metrics"):
        model.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=loader,
                val_loader=loader,
                optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
                metrics=task.metric_specs(),
                monitor=MonitorSpec("ndcg_at_2"),
            )
        )
    with pytest.raises(ValueError, match="exactly one option, k"):
        MetricSpec("ranking.ndcg", config={"k": 2, "gain": "linear"}).check()


def test_the_module_needs_no_faiss_and_the_embeddings_scope_is_unchanged():
    tree = ast.parse(Path(__import__("nnx.ranking").ranking.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
    assert "faiss" not in imported and "sklearn" not in imported
    from nnx.embeddings import faiss_export

    assert "NNx's job ends here" in (faiss_export.__doc__ or "")
    assert faiss_export.export_to_faiss.__annotations__.get("return") in (str, "str")


# --- hardening --------------------------------------------------------------------------------------------


def test_numpy_settings_give_a_weights_only_state_and_units_restart_per_fresh_run():
    import io

    import numpy as np

    task = RankingTask(
        k=(np.int64(2),), max_relevance=np.int64(2), relevance_threshold=np.int32(1), max_buffered=np.int64(64)
    )
    state = task.state()
    assert all(type(state[key]) is int for key in ("max_relevance", "relevance_threshold", "max_buffered", "version"))
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    assert RankingTask.from_state(torch.load(buffer, weights_only=True)) == task
    objective = task.objective()
    for index in range(2):
        run = _scorer().train(
            params=_params(task, 1), objective=objective, eval_step_fn=task.eval_step(), salt=f"fit-{index}"
        )
        saved = NNCheckpoint.load_training_state(run=run.id, type=Checkpoints.LAST)["components"]["ranking.task"]
        assert saved["state"]["units"] == 5  # one epoch's graded queries, not the sum over runs


def test_a_scorer_output_goes_through_its_batch_adapter():
    from torch import nn

    from nnx.models import BatchAdapter

    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(D, 1)

        def forward(self, x):
            return self.linear(x), {"aux": None}

    class First(BatchAdapter):
        def split(self, batch):
            return (batch,), {}, None

        def output(self, raw):
            return raw[0]

    task = RankingTask(k=(1,))
    x, q, c, r = _data(2)
    model = NNModel(module=Pair(), params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR), batch_adapter=First())
    assert task.scores(model, x, len(q)).shape == (len(q),)
    plain = NNModel(module=Pair(), params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR))
    with pytest.raises(RankingError, match="floating scores"):
        task.scores(plain, x, len(q))


def test_a_pairless_microbatch_is_finite_even_with_infinite_scores():
    scores = torch.full((3,), math.inf, requires_grad=True)
    numerator, denominator = pairwise_logistic_loss(scores, torch.tensor([1, 1, 1]), ["q", "q", "q"])
    assert denominator == 0 and float(numerator.detach()) == 0.0
    numerator.backward()
    assert scores.grad is not None and float(scores.grad.abs().sum()) == 0.0


def test_an_empty_validation_loader_and_extra_metrics_are_refused():
    task = RankingTask(k=(1,))
    with pytest.raises(RankingError, match="yielded no batches"):
        task.eval_step()(_Ctx(_scorer(), []))
    params = NNTrainParams(
        n_epochs=1,
        train_loader=_loader(_data(2), batch_size=GROUP),
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        extra_metrics={"acc": lambda y, p: 0.0},
    )
    with pytest.raises(RankingError, match="extra_metrics"):
        _scorer().train(params=params, objective=task.objective())


def test_without_a_monitor_an_all_excluded_epoch_never_becomes_best():
    task = RankingTask(k=(1,), max_relevance=2)

    class ExcludedSecondTime:
        def __init__(self, data):
            self.data, self.passes = data, 0

        def __iter__(self):
            self.passes += 1
            x, q, c, r = self.data
            grades = r if self.passes == 1 else torch.zeros_like(r)
            return iter([(x, q, c, grades)])

    params = NNTrainParams(
        n_epochs=2,
        train_loader=_loader(_data(6), batch_size=8),
        val_loader=ExcludedSecondTime(_data(3, seed=1, zero_query=False)),
        optim=NNOptimParams.builder().sgd(max_lr=0.2).build(),
    )
    with pytest.warns(RuntimeWarning, match="validation record is unavailable"):  # the plateau step is skipped
        run = _scorer().train(params=params, objective=task.objective(), eval_step_fn=task.eval_step())
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.epoch_idx == 0 and best.idp.val_edp.status == "ok"


# --- review round 1 -----------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_objective_trains_on_a_cuda_scorer():  # pragma: no cover - GPU only
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=D, output_dim=1, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CUDA, loss=Losses.MEAN_SQUARED_ERROR),
    )
    task = RankingTask(k=(1,), max_relevance=2)
    result = _objective_step(task, model, next(iter(_loader(_data(2), batch_size=2 * GROUP))))
    assert result.terms[0].numerator.device.type == "cuda" and result.terms[0].denominator == 2


def test_relevance_on_another_device_is_moved_to_the_scores_before_any_indexing():
    """CPU stand-in for the GPU case: the grades are moved (whole) to the
    scores' device before a device index ever touches them."""
    moved = []

    class Grades(torch.Tensor):
        def to(self, *args, **kwargs):
            moved.append(tuple(self.shape))
            return super().to(*args, **kwargs)

    grades = torch.tensor([1, 0, 2, 0, 1, 1]).as_subclass(Grades)
    numerator, denominator = pairwise_logistic_loss(torch.rand(6), grades, ["a"] * 3 + ["b"] * 3)
    assert moved[0] == (6,) and denominator == 2


def test_pairs_are_scored_in_bounded_blocks(monkeypatch):
    import nnx.ranking as ranking

    seen = []
    original = torch.nn.functional.softplus

    def spy(x, *args, **kwargs):
        seen.append(x.numel())
        return original(x, *args, **kwargs)

    monkeypatch.setattr(ranking, "PAIR_BLOCK", 64)
    monkeypatch.setattr(torch.nn.functional, "softplus", spy)
    n = 300
    scores = torch.linspace(0, 1, n, dtype=torch.float64)
    grades = torch.tensor([2 if i < 40 else 1 if i < 100 else 0 for i in range(n)])
    numerator, denominator = pairwise_logistic_loss(scores, grades, ["q"] * n, weighting="pair")
    assert max(seen) <= max(64, n)  # never the n x n matrix
    s, r = scores.tolist(), grades.tolist()
    expected = [math.log1p(math.exp(-(s[i] - s[j]))) for i in range(n) for j in range(n) if r[i] > r[j]]
    assert denominator == len(expected) and float(numerator) == pytest.approx(math.fsum(expected), rel=1e-12)


def test_the_validation_loss_covers_only_scored_queries():
    task = RankingTask(k=(1,), max_relevance=2, relevance_threshold=2)
    model = _scorer()
    x = torch.randn(6, D, generator=torch.Generator().manual_seed(5))
    q, c = torch.tensor([1, 1, 1, 2, 2, 2]), torch.tensor([1, 2, 3, 1, 2, 3])
    r = torch.tensor([2, 0, 0, 1, 0, 0])  # query 2 has no candidate at the threshold: excluded
    record = task.eval_step()(_Ctx(model, [(x, q, c, r)]))
    alone = task.eval_step()(_Ctx(model, [(x[:3], q[:3], c[:3], r[:3])]))
    assert record.metrics["excluded_queries"] == 1.0 and record.loss == pytest.approx(alone.loss)


def test_masked_rows_may_carry_any_grade_and_features_with_to_are_moved():
    task = RankingTask(k=(1,))
    x = torch.randn(3, D)
    q, c = torch.tensor([1, 1, 1]), torch.tensor([1, 2, 3])
    _, _, _, grades, _ = task.split((x, q, c, torch.tensor([1, 0, -1]), torch.tensor([True, True, False])))
    assert grades.tolist() == [1, 0, -1]
    with pytest.raises(RankingError, match="grades must be in 0..1"):
        task.split((x, q, c, torch.tensor([1, 0, -1])))

    class Features:
        def __init__(self, x):
            self.x, self.devices = x, []

        def to(self, device):
            self.devices.append(torch.device(device))
            return self.x

    features = Features(x)
    assert task.scores(_scorer(), features, 3).shape == (3,) and features.devices == [torch.device("cpu")]


def test_a_validation_task_record_without_a_value_never_falls_back_to_training():
    from nnx._metrics import _resolve_metric
    from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

    no_loss = NNEvaluationDataPoint(kind="ranking", count=2, status="ok", metrics={"ndcg_at_1": 1.0})
    assert _resolve_metric(no_loss, NNEvaluationDataPoint(loss=0.5)) is None
    legacy = NNEvaluationDataPoint()  # a record without a task kind keeps the fallback
    assert _resolve_metric(legacy, NNEvaluationDataPoint(loss=0.5)) == 0.5


def test_numpy_strings_become_builtin_strings():
    import numpy as np

    task = RankingTask(k=(1,), weighting=np.str_("pair"), candidate_sets=np.str_("exhaustive"))
    assert type(task.state()["weighting"]) is str and type(task.state()["candidate_sets"]) is str


# --- review round 2 ------------------------------------------------------------------------------


def test_training_keeps_block_inputs_not_the_pair_matrix():
    n = 2000
    scores = torch.randn(n, requires_grad=True)
    grades = (torch.arange(n) % 2).long()  # 1000 x 1000 unequal pairs
    saved = []

    def pack(tensor):
        saved.append(tensor.numel() * tensor.element_size())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        numerator, denominator = pairwise_logistic_loss(scores, grades, ["q"] * n, weighting="pair")
    numerator.backward()
    pair_matrix = 1000 * 1000 * 4
    assert denominator == 1000 * 1000 and sum(saved) < pair_matrix / 10
    reference = torch.nn.functional.softplus(-(scores[grades == 1][:, None] - scores[grades == 0][None, :])).sum()
    assert float(numerator.detach()) == pytest.approx(float(reference.detach()), rel=1e-5)
    assert scores.grad is not None and float(scores.grad.abs().sum()) > 0


def test_round_two_settings_and_streams():
    from enum import Enum

    class Weighting(str, Enum):
        PAIR = "pair"

    class Sets(str, Enum):
        EXHAUSTIVE = "exhaustive"

    task = RankingTask(k=(1,), weighting=Weighting.PAIR, candidate_sets=Sets.EXHAUSTIVE)
    assert task.state()["weighting"] == "pair" and type(task.state()["candidate_sets"]) is str
    assert RankingTask.from_state(task.state()) == task
    x = torch.randn(3, D)
    q, c = torch.tensor([1, 1, 1]), torch.tensor([1, 2, 3])
    padded = torch.tensor([1.0, 0.0, math.nan])
    _, _, _, grades, _ = task.split((x, q, c, padded, torch.tensor([True, True, False])))
    assert grades[:2].tolist() == [1, 0]
    with pytest.raises(RankingError, match="finite integers"):
        task.split((x, q, c, padded))
    for bad in (-3, 7, 1.9, True):
        with pytest.raises(RankingError, match="grade"):
            task.evaluate_rows([("q", 1, 0.5, bad), ("q", 2, 0.1, 0)])
    small = RankingTask(k=(1,), max_buffered=4)
    seen = []

    class Spy:
        def __iter__(self):
            seen.append("batch")
            yield (torch.randn(5, D), torch.tensor([1] * 5), torch.arange(5), torch.tensor([1, 0, 0, 0, 0]))

    model = _scorer()
    calls = []
    original = model.net.forward
    model.net.forward = lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs)  # type: ignore[method-assign]
    with pytest.raises(RankingError, match="max_buffered=4"):
        small.eval_step()(_Ctx(model, Spy()))
    assert calls == []  # refused before the batch was scored or buffered


# --- review round 3 ------------------------------------------------------------------------------


def test_rank_refuses_a_non_finite_score():
    with pytest.raises(RankingError, match="non-finite score"):
        rank([math.nan, 0.5, 0.9], [1, 2, 3])
    with pytest.raises(RankingError, match="non-finite score"):
        rank([0.5, math.inf, 0.9], ["a", "b", "c"])


def test_only_blocks_worth_it_are_recomputed_in_backward(monkeypatch):
    import nnx.ranking as ranking

    calls = []
    original = ranking.checkpoint

    def spy(*args, **kwargs):
        calls.append(args[1].shape[0] * args[2].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(ranking, "checkpoint", spy)
    scores = torch.randn(40, requires_grad=True)
    grades = (torch.arange(40) % 3).long()
    small, _ = pairwise_logistic_loss(scores, grades, ["q"] * 40, weighting="pair")
    (small_grad,) = torch.autograd.grad(small, scores)
    assert calls == []  # 20 x 13 pairs per query: cheaper to keep than to recompute

    monkeypatch.setattr(ranking, "CHECKPOINT_PAIRS", 1)
    every, _ = pairwise_logistic_loss(scores, grades, ["q"] * 40, weighting="pair")
    (every_grad,) = torch.autograd.grad(every, scores)
    assert calls and min(calls) >= 1
    assert float(every.detach()) == pytest.approx(float(small.detach()), rel=1e-12)
    assert torch.allclose(every_grad, small_grad, rtol=1e-6, atol=1e-7)
