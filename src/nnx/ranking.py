"""Query-grouped ranking and retrieval evaluation (FEAT-035).

A scorer ranks the **candidates of each query**; it is judged per query,
never per batch. :class:`RankingTask` declares the relevance grades, the
cutoffs ``k``, the relevance threshold, how the pairwise objective weights
queries and pairs, the candidate-set kind and the buffer limit, and checks
every batch — query ids, candidate ids, relevance grades and an optional
mask — before any update.

A batch is ``(features, query_ids, candidate_ids, relevance)`` or
``(..., mask)`` (or a mapping with those keys): the scorer maps
``features`` to one score per candidate row; ids are integer tensors or
lists of strings; relevance grades are integers in ``0..max_relevance``.

- :meth:`RankingTask.objective` — the **pairwise logistic** objective
  (FEAT-004): for each query, every pair of its candidates with unequal
  relevance contributes ``log(1 + exp(-(s_hi - s_lo)))``; equal-relevance
  pairs (ties) contribute nothing and a query without such a pair is
  skipped. ``weighting="query"`` averages each query's pairs and then the
  queries (a query weighs the same however many pairs it has);
  ``weighting="pair"`` weighs every pair equally. Pairs are formed within a
  microbatch. Checkpointed as component ``"ranking.task"``.
- :meth:`RankingTask.eval_step` — an ``eval_step_fn``: it buffers the whole
  validation stream (at most ``max_buffered`` candidate rows), groups it by
  query — so a query split across batches is joined, in any order — and
  rejects a candidate seen twice for one query. Each query's candidates are
  ranked by score, **ties broken by candidate id** (ascending); then
  ``MRR@k`` (the reciprocal rank of the first relevant candidate in the top
  ``k``, else 0), ``Recall@k`` (relevant candidates in the top ``k`` over
  all relevant) and ``NDCG@k`` (gain ``2^rel - 1``, discount
  ``log2(rank + 1)``) are averaged over queries. A candidate is relevant
  when its grade is at least ``relevance_threshold``; a query with no
  relevant candidate is **excluded** (and counted), never scored 0. ``k``
  above a query's candidate count is rejected.

The record (``kind="ranking"``) carries ``mrr_at_<k>``, ``recall_at_<k>``,
``ndcg_at_<k>``, the counts of scored and excluded queries and candidates,
and ``exhaustive_candidates`` (``1.0`` when every query's candidate set is
its full corpus, ``0.0`` when it is sampled — then the metrics measure the
candidate set, not full-corpus retrieval). :meth:`RankingTask.metric_specs`
declares the metrics so a :class:`~nnx.MonitorSpec` can name them (for
example ``MonitorSpec("ndcg_at_10")``) for BEST, early stopping and plateau
scheduling.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch.utils.checkpoint import checkpoint

from .components import ComponentSpec
from .objectives import LossTerm, Objective, ObjectiveContext, ObjectiveResult

__all__ = [
    "KIND",
    "RankingError",
    "RankingEval",
    "RankingObjective",
    "RankingTask",
    "ndcg_at_k",
    "pairwise_logistic_loss",
    "rank",
    "mrr_at_k",
    "recall_at_k",
]

KIND = "ranking"
TASK_VERSION = 1
WEIGHTINGS = ("query", "pair")
CANDIDATE_SETS = ("sampled", "exhaustive")
Id = Union[int, str]


class RankingError(ValueError):
    """A configuration, batch or query the ranking task rejects."""


def _count(value: Any, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise RankingError(f"{what} must be an integer >= {minimum}, got {value!r}")
    return int(value)


# --- per-query metrics ----------------------------------------------------------------------------


def rank(scores: Sequence[float], candidate_ids: Sequence[Id]) -> list[int]:
    """Positions of the candidates in ranked order: by score, highest first,
    ties broken by candidate id (ascending) — a stable, documented order."""
    if len(scores) != len(candidate_ids):
        raise RankingError(f"{len(scores)} scores for {len(candidate_ids)} candidate ids")
    kinds = {type(c) for c in candidate_ids}
    if len(kinds) > 1:
        raise RankingError("candidate ids of one query must all be integers or all strings")
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), candidate_ids[i]))


def _check_query(relevance: Sequence[int], scores: Sequence[float], candidate_ids: Sequence[Id]) -> None:
    if not (len(relevance) == len(scores) == len(candidate_ids)):
        raise RankingError(
            f"a query needs one grade, score and candidate id per candidate; got {len(relevance)} grades, "
            f"{len(scores)} scores and {len(candidate_ids)} ids"
        )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RankingError("a query lists the same candidate id twice")
    if not all(math.isfinite(float(score)) for score in scores):
        raise RankingError("a query has a non-finite score; it cannot be ranked")


def _check_k(k: int, n: int) -> int:
    k = _count(k, "k", minimum=1)
    if k > n:
        raise RankingError(f"k={k} is larger than the query's {n} candidates")
    return k


def mrr_at_k(
    relevance: Sequence[int], scores: Sequence[float], candidate_ids: Sequence[Id], k: int, *, threshold: int = 1
) -> float:
    """Reciprocal rank of the first relevant candidate within the top ``k``
    (0 when none is)."""
    _check_query(relevance, scores, candidate_ids)
    order = rank(scores, candidate_ids)[: _check_k(k, len(scores))]
    for position, i in enumerate(order, start=1):
        if relevance[i] >= threshold:
            return 1.0 / position
    return 0.0


def recall_at_k(
    relevance: Sequence[int], scores: Sequence[float], candidate_ids: Sequence[Id], k: int, *, threshold: int = 1
) -> float:
    """Relevant candidates in the top ``k`` over all relevant candidates
    (the query must have at least one)."""
    _check_query(relevance, scores, candidate_ids)
    total = sum(r >= threshold for r in relevance)
    if total == 0:
        raise RankingError("recall is undefined for a query with no relevant candidate")
    order = rank(scores, candidate_ids)[: _check_k(k, len(scores))]
    return sum(relevance[i] >= threshold for i in order) / total


def ndcg_at_k(relevance: Sequence[int], scores: Sequence[float], candidate_ids: Sequence[Id], k: int) -> float:
    """``DCG@k / IDCG@k`` with gain ``2^rel - 1`` and discount
    ``log2(rank + 1)`` (the query must have a nonzero grade)."""
    _check_query(relevance, scores, candidate_ids)
    k = _check_k(k, len(scores))
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum((2.0**r - 1.0) / math.log2(position + 2) for position, r in enumerate(ideal))
    if idcg == 0.0:
        raise RankingError("NDCG is undefined for a query whose every grade is 0")
    order = rank(scores, candidate_ids)[:k]
    dcg = sum((2.0 ** relevance[i] - 1.0) / math.log2(position + 2) for position, i in enumerate(order))
    return dcg / idcg


PAIR_BLOCK = 1 << 20  # pairs scored at once: memory stays linear in a query's size


def _block_loss(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(-(high.unsqueeze(1) - low.unsqueeze(0))).sum()


def pairwise_logistic_loss(
    scores: torch.Tensor, relevance: torch.Tensor, query_ids: Sequence[Id], *, weighting: str = "query"
) -> tuple[torch.Tensor, int]:
    """``(numerator, denominator)`` of the pairwise logistic loss over the
    unequal-relevance pairs within each query (see the module docstring).

    Pairs are formed grade by grade (each grade's candidates against every
    lower-graded one) in blocks of at most :data:`PAIR_BLOCK` pairs, so no
    query's full pair matrix is ever held — in training too: each block is
    recomputed in backward (``torch.utils.checkpoint``) rather than kept. ``relevance`` may live on any
    device; it is moved to the scores'."""
    if weighting not in WEIGHTINGS:
        raise RankingError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
    if len(query_ids) != scores.shape[0] or relevance.shape[0] != scores.shape[0]:
        raise RankingError(f"{scores.shape[0]} scores for {relevance.shape[0]} grades and {len(query_ids)} query ids")
    relevance = relevance.to(scores.device)
    groups: dict[Id, list[int]] = {}
    for row, query in enumerate(query_ids):
        groups.setdefault(query, []).append(row)
    numerator = scores.reshape(-1)[:0].sum()  # an empty sum: on the graph, finite whatever the scores hold
    denominator = 0
    for rows in groups.values():
        index = torch.tensor(rows, device=scores.device)
        s, r = scores[index], relevance[index]
        total = s[:0].sum()
        pairs = 0
        for grade in torch.unique(r).tolist()[1:]:  # each grade above the lowest, against every lower one
            high, low = s[r == grade], s[r < grade]
            block = max(1, PAIR_BLOCK // max(1, low.shape[0]))
            for start in range(0, high.shape[0], block):
                chunk = high[start : start + block]
                if torch.is_grad_enabled() and (chunk.requires_grad or low.requires_grad):
                    # Recomputed in backward: autograd keeps the block's inputs, never its pair matrix.
                    part = checkpoint(_block_loss, chunk, low, use_reentrant=False)
                else:
                    part = _block_loss(chunk, low)
                total = total + part
                pairs += chunk.shape[0] * low.shape[0]
        if pairs == 0:
            continue  # no unequal-relevance pair: the query teaches nothing
        if weighting == "pair":
            numerator = numerator + total
            denominator += pairs
        else:
            numerator = numerator + total / pairs
            denominator += 1
    return numerator, denominator


# --- the task -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingTask:
    """A query-grouped ranking task.

    Args:
        k: the cutoffs the metrics are reported at (each ``>= 1``).
        max_relevance: grades are integers in ``0..max_relevance``
            (``1``: binary relevance).
        relevance_threshold: a candidate is relevant (for MRR and Recall,
            and for excluding queries) when its grade is at least this.
        weighting: the objective's ``"query"`` or ``"pair"`` weighting.
        candidate_sets: ``"sampled"`` (each query sees a sample of its
            corpus) or ``"exhaustive"`` (its whole corpus); recorded with
            the metrics.
        max_buffered: the most candidate rows the evaluation buffers to
            join each query's candidates.
        group_size: when set, every evaluated query must have exactly this
            many candidates — an incomplete (or over-full) query is
            rejected rather than scored.
        version: the task schema version.
    """

    k: tuple[int, ...] = (10,)
    max_relevance: int = 1
    relevance_threshold: int = 1
    weighting: str = "query"
    candidate_sets: str = "sampled"
    max_buffered: int = 1_000_000
    group_size: Optional[int] = None
    version: int = TASK_VERSION

    def __post_init__(self) -> None:
        ks = tuple(self.k) if isinstance(self.k, Iterable) else (self.k,)
        if not ks:
            raise RankingError("k needs at least one cutoff")
        ks = tuple(sorted({_count(k, "k", minimum=1) for k in ks}))
        object.__setattr__(self, "k", ks)
        top = _count(self.max_relevance, "max_relevance", minimum=1)
        threshold = _count(self.relevance_threshold, "relevance_threshold", minimum=1)
        if threshold > top:
            raise RankingError(f"relevance_threshold {threshold} is above max_relevance {top}")
        # Builtin ints: the checkpointed state stays weights_only-loadable.
        object.__setattr__(self, "max_relevance", top)
        object.__setattr__(self, "relevance_threshold", threshold)
        if self.weighting not in WEIGHTINGS:
            raise RankingError(f"weighting must be one of {WEIGHTINGS}, got {self.weighting!r}")
        if self.candidate_sets not in CANDIDATE_SETS:
            raise RankingError(f"candidate_sets must be one of {CANDIDATE_SETS}, got {self.candidate_sets!r}")
        # The canonical builtin str (a numpy string or a str Enum compares equal but pickles unsafely).
        object.__setattr__(self, "weighting", WEIGHTINGS[WEIGHTINGS.index(self.weighting)])
        object.__setattr__(self, "candidate_sets", CANDIDATE_SETS[CANDIDATE_SETS.index(self.candidate_sets)])
        object.__setattr__(self, "max_buffered", _count(self.max_buffered, "max_buffered", minimum=1))
        if self.group_size is not None:
            size = _count(self.group_size, "group_size", minimum=1)
            if size < self.k[-1]:
                raise RankingError(f"group_size {size} is smaller than k={self.k[-1]}")
            object.__setattr__(self, "group_size", size)
        if isinstance(self.version, bool) or self.version != TASK_VERSION:
            raise RankingError(f"this NNx supports ranking task version {TASK_VERSION}, got {self.version!r}")
        object.__setattr__(self, "version", TASK_VERSION)

    def state(self) -> dict[str, Any]:
        return {
            "kind": KIND,
            "version": self.version,
            "k": list(self.k),
            "max_relevance": self.max_relevance,
            "relevance_threshold": self.relevance_threshold,
            "weighting": self.weighting,
            "candidate_sets": self.candidate_sets,
            "max_buffered": self.max_buffered,
            "group_size": self.group_size,
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> RankingTask:
        if not isinstance(state, Mapping) or state.get("kind") != KIND:
            raise RankingError(f"not a ranking task state: {state!r}")
        fields = (
            "version",
            "k",
            "max_relevance",
            "relevance_threshold",
            "weighting",
            "candidate_sets",
            "max_buffered",
            "group_size",
        )
        unknown = sorted(set(state) - {"kind", *fields})
        if unknown:
            raise RankingError(f"a ranking task state has unknown keys {unknown}")
        values: dict[str, Any] = {name: state[name] for name in fields if name in state}
        if isinstance(values.get("k"), list):
            values["k"] = tuple(values["k"])
        return RankingTask(**values)

    def metric_names(self) -> list[str]:
        return [f"{metric}_at_{k}" for k in self.k for metric in ("mrr", "recall", "ndcg")]

    def metric_specs(self) -> list[Any]:
        """``MetricSpec``\\ s declaring this task's metrics — registered ids
        ``ranking.mrr`` / ``ranking.recall`` / ``ranking.ndcg`` with
        ``config={"k": k}``, reported as ``<metric>_at_<k>`` — so a
        ``MonitorSpec`` can select on them. :meth:`eval_step` computes them;
        a declaration under another name finds no value."""
        from .monitors import MetricSpec

        _register_metrics()
        return [
            MetricSpec(f"ranking.{metric}", config={"k": k}, name=f"{metric}_at_{k}")
            for k in self.k
            for metric in ("mrr", "recall", "ndcg")
        ]

    # ---------- batches ----------

    def split(self, batch: Any) -> tuple[Any, list[Id], list[Id], torch.Tensor, Optional[torch.Tensor]]:
        """``(features, query_ids, candidate_ids, relevance, mask)``, checked."""
        if isinstance(batch, Mapping):
            parts = [batch.get(name) for name in ("features", "query_ids", "candidate_ids", "relevance")]
            if any(part is None for part in parts):
                raise RankingError("a ranking batch mapping needs features, query_ids, candidate_ids and relevance")
            parts.append(batch.get("mask"))
        elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)) and len(batch) in (4, 5):
            parts = [*batch, None] if len(batch) == 4 else list(batch)
        else:
            raise RankingError(
                "a ranking batch is (features, query_ids, candidate_ids, relevance[, mask]) or a mapping with those keys"
            )
        features, queries, candidates, relevance, mask = parts
        query_ids = self._ids(queries, "query_ids")
        candidate_ids = self._ids(candidates, "candidate_ids")
        n = len(query_ids)
        if len(candidate_ids) != n:
            raise RankingError(f"{n} query ids but {len(candidate_ids)} candidate ids")
        if not isinstance(relevance, torch.Tensor) or relevance.ndim != 1 or relevance.shape[0] != n:
            got = tuple(relevance.shape) if isinstance(relevance, torch.Tensor) else type(relevance).__name__
            raise RankingError(f"relevance must be a ({n},) tensor of integer grades, got {got}")
        if relevance.dtype == torch.bool:
            raise RankingError("relevance grades must be integers, not booleans")
        raw = relevance.detach().cpu()
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (n,):
                got = tuple(mask.shape) if isinstance(mask, torch.Tensor) else type(mask).__name__
                raise RankingError(f"the mask must be a ({n},) tensor, got {got}")
            if mask.dtype != torch.bool:
                if mask.is_floating_point() or not bool(((mask == 0) | (mask == 1)).all()):
                    raise RankingError("the mask must be boolean or 0/1 integers")
                mask = mask.bool()
            mask = mask.detach().cpu()
        read = raw if mask is None else raw[mask]  # a masked row's grade is never read
        if raw.is_floating_point() and (not bool(torch.isfinite(read).all()) or not bool((read == read.round()).all())):
            raise RankingError("relevance grades must be finite integers")
        relevance = (
            torch.where(torch.isfinite(raw), raw, torch.zeros_like(raw)).long()
            if raw.is_floating_point()
            else raw.long()
        )
        bad = (relevance < 0) | (relevance > self.max_relevance)
        if mask is not None:
            bad &= mask  # a masked row's grade is never read
        if bool(bad.any()):
            raise RankingError(
                f"relevance grades must be in 0..{self.max_relevance}; found {relevance[bad][:5].tolist()}"
            )
        seen: set[tuple[Id, Id]] = set()
        for row, pair in enumerate(zip(query_ids, candidate_ids, strict=True)):
            if mask is not None and not bool(mask[row]):
                continue
            if pair in seen:
                raise RankingError(f"candidate {pair[1]!r} appears twice for query {pair[0]!r} in one batch")
            seen.add(pair)
        return features, query_ids, candidate_ids, relevance, mask

    @staticmethod
    def _ids(value: Any, what: str) -> list[Id]:
        if isinstance(value, torch.Tensor):
            if value.ndim != 1 or value.is_floating_point() or value.dtype == torch.bool:
                raise RankingError(
                    f"{what} must be a 1-D integer tensor or a list of strings, got {value.dtype} {tuple(value.shape)}"
                )
            return [int(v) for v in value.tolist()]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            ids = list(value)
            if all(isinstance(v, str) and v for v in ids) or all(
                isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in ids
            ):
                return [v if isinstance(v, str) else int(v) for v in ids]
        raise RankingError(f"{what} must be a 1-D integer tensor or a list of non-empty strings (one id per row)")

    def scores(self, model: Any, features: Any, n: int) -> torch.Tensor:
        """The scorer's one score per row, checked."""
        from .nn.nn_model import _to_device

        raw = model.net(_to_device(features, model.device))  # a tensor, or anything with .to (a graph batch)
        scores = getattr(raw, "logits", None)
        if scores is None:
            adapter = getattr(model, "_batch_adapter", None)  # a registered module's own output rule (FEAT-006)
            try:
                scores = raw if adapter is None else adapter.output(raw)
            except TypeError as error:
                raise RankingError(f"the scorer must return floating scores: {error}") from error
        if not isinstance(scores, torch.Tensor) or not scores.is_floating_point():
            raise RankingError(f"the scorer must return floating scores, got {type(scores).__name__}")
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        if scores.ndim != 1 or scores.shape[0] != n:
            raise RankingError(f"the scorer returned shape {tuple(scores.shape)}; expected one score per row ({n},)")
        return scores

    # ---------- training and evaluation ----------

    def objective(self, *, nonfinite: str = "fail") -> RankingObjective:
        return RankingObjective(self, nonfinite=nonfinite)

    def eval_step(self) -> RankingEval:
        return RankingEval(self)

    def evaluate_rows(self, rows: Iterable[tuple[Id, Id, float, int]]) -> tuple[dict[str, float], int, int, int]:
        """Metrics over complete queries from ``(query, candidate, score,
        relevance)`` rows in any order: ``(means, scored queries, excluded
        queries, candidates of the scored queries)``. Duplicate candidates of a query raise."""
        queries: dict[Id, dict[Id, tuple[float, int]]] = {}
        candidates = 0
        for query, candidate, score, grade in rows:
            group = queries.setdefault(query, {})
            if candidate in group:
                raise RankingError(
                    f"candidate {candidate!r} appears twice for query {query!r} in the evaluation stream"
                )
            if (
                isinstance(grade, bool)
                or not isinstance(grade, numbers.Integral)
                or not 0 <= grade <= self.max_relevance
            ):
                raise RankingError(f"query {query!r}: grade {grade!r} is not an integer in 0..{self.max_relevance}")
            group[candidate] = (float(score), int(grade))
        sums = dict.fromkeys(self.metric_names(), 0.0)
        scored = excluded = 0
        for query in sorted(queries, key=lambda q: (type(q).__name__, q)):  # order-independent
            ids = list(queries[query])
            scores = [queries[query][c][0] for c in ids]
            grades = [queries[query][c][1] for c in ids]
            if self.group_size is not None and len(ids) != self.group_size:
                raise RankingError(
                    f"query {query!r} has {len(ids)} candidates; the task declares group_size={self.group_size}"
                )
            if not all(math.isfinite(score) for score in scores):
                raise RankingError(f"query {query!r} has a non-finite score; it cannot be ranked")
            if self.k[-1] > len(ids):  # checked for every query, excluded or not
                raise RankingError(f"k={self.k[-1]} is larger than query {query!r}'s {len(ids)} candidates")
            if not any(g >= self.relevance_threshold for g in grades):
                excluded += 1  # no relevant candidate: excluded, never scored 0
                continue
            for k in self.k:
                sums[f"mrr_at_{k}"] += mrr_at_k(grades, scores, ids, k, threshold=self.relevance_threshold)
                sums[f"recall_at_{k}"] += recall_at_k(grades, scores, ids, k, threshold=self.relevance_threshold)
                sums[f"ndcg_at_{k}"] += ndcg_at_k(grades, scores, ids, k)
            scored += 1
            candidates += len(ids)
        means = {name: total / scored for name, total in sums.items()} if scored else {}
        return means, scored, excluded, candidates


def _record(
    task: RankingTask, *, loss: Optional[float], means: Mapping[str, float], scored: int, excluded: int, candidates: int
) -> Any:
    from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

    exhaustive = 1.0 if task.candidate_sets == "exhaustive" else 0.0
    if scored == 0:  # no query could be scored: unavailable, with what was excluded
        counts = {"queries": 0.0, "excluded_queries": float(excluded), "exhaustive_candidates": exhaustive}
        return NNEvaluationDataPoint(kind=KIND, count=0, status="empty", metrics=counts if excluded else {})
    metrics = {
        **means,
        "queries": float(scored),
        "excluded_queries": float(excluded),
        "candidates": float(candidates),
        "candidates_per_query": candidates / scored,
        "exhaustive_candidates": exhaustive,
    }
    return NNEvaluationDataPoint(loss=loss, kind=KIND, count=scored, status="ok", metrics=metrics)


def _no_extra_metrics(ctx: Any) -> None:
    if getattr(ctx, "extra_metrics", None):
        raise RankingError(
            "extra_metrics (y_true, y_pred callables over rows) do not apply to query-grouped ranking; its "
            "records carry MRR / Recall / NDCG per query — compute anything else in your own eval_step_fn"
        )


class RankingObjective(Objective):
    """The pairwise logistic objective of a :class:`RankingTask` (see the
    module docstring); checkpointed as component ``"ranking.task"`` with the
    task configuration and the run's training units — pairs
    (``weighting="pair"``) or queries (``"query"``) scored, from 0 for each
    fresh run and restored on resume (a window the engine then skips is
    still counted)."""

    def __init__(self, task: RankingTask, *, nonfinite: str = "fail") -> None:
        super().__init__(nonfinite=nonfinite)
        if not isinstance(task, RankingTask):
            raise RankingError(f"a RankingTask is needed, got {type(task).__name__}")
        self.task = task
        self.units = 0  # pairs (weighting="pair") or queries ("query") scored in this run

    def __call__(self, ctx: ObjectiveContext) -> ObjectiveResult:
        _no_extra_metrics(ctx)
        if ctx.epoch_idx == 0 and ctx.batch_idx == 0:
            self.units = 0  # a fresh run (a resume starts at a later epoch, its count restored)
        model = ctx.model
        model.net.train()
        features, query_ids, candidate_ids, relevance, mask = self.task.split(ctx.batch)
        scores = self.task.scores(model, features, len(query_ids))
        rows = list(range(len(query_ids))) if mask is None else [i for i in range(len(query_ids)) if bool(mask[i])]
        index = torch.tensor(rows, dtype=torch.long)
        numerator, denominator = pairwise_logistic_loss(
            scores[index.to(scores.device)],
            relevance[index],
            [query_ids[i] for i in rows],
            weighting=self.task.weighting,
        )
        term = LossTerm("pairwise_logistic", numerator, denominator)
        self.units += denominator
        from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

        record = (
            NNEvaluationDataPoint(loss=term.value, kind=KIND, count=denominator, status="ok")
            if denominator
            else NNEvaluationDataPoint(kind=KIND, count=0, status="empty")
        )
        return ObjectiveResult((term,), record)

    def component_spec(self) -> ComponentSpec:
        return ComponentSpec("ranking.task", version=1)

    def component_state(self) -> dict[str, Any]:
        return {"task": self.task.state(), "units": self.units}

    def check_component_state(self, state: Mapping[str, Any], *, version: int) -> list[str]:
        saved = state.get("task") if isinstance(state, Mapping) else None
        if saved != self.task.state():
            return [f"the ranking task changed since the checkpoint: {saved} -> {self.task.state()}"]
        units = state.get("units")
        if isinstance(units, bool) or not isinstance(units, int) or units < 0:
            return [f"the checkpoint's ranking unit count is not a non-negative integer: {units!r}"]
        return []

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None:
        self.units = int(state["units"])


class RankingEval:
    """The :class:`RankingTask`'s ``eval_step_fn``: buffers the validation
    stream (at most ``max_buffered`` rows), groups it by query and reports
    the per-query metrics (see the module docstring), with the pairwise
    loss over the same queries as the record's ``loss``."""

    def __init__(self, task: RankingTask) -> None:
        if not isinstance(task, RankingTask):
            raise RankingError(f"a RankingTask is needed, got {type(task).__name__}")
        self.task = task

    def __call__(self, ctx: Any) -> Any:
        from .utils import _capture_training_modes, _restore_training_modes

        _no_extra_metrics(ctx)
        model = ctx.model
        modes = _capture_training_modes(model.net)
        rows: list[tuple[Id, Id, float, int]] = []
        batches = 0
        try:
            model.net.eval()
            with torch.no_grad():
                for batch in ctx.val_loader:
                    batches += 1
                    features, query_ids, candidate_ids, relevance, mask = self.task.split(batch)
                    kept = len(query_ids) if mask is None else int(mask.sum())
                    if len(rows) + kept > self.task.max_buffered:  # refused before the batch is buffered
                        raise RankingError(
                            f"the evaluation stream exceeds max_buffered={self.task.max_buffered} candidate rows"
                        )
                    scores = self.task.scores(model, features, len(query_ids)).detach().double().cpu().tolist()
                    for i, (query, candidate) in enumerate(zip(query_ids, candidate_ids, strict=True)):
                        if mask is not None and not bool(mask[i]):
                            continue
                        rows.append((query, candidate, scores[i], int(relevance[i])))
        finally:
            _restore_training_modes(modes)
        if batches == 0:
            raise RankingError("the validation loader yielded no batches")
        means, scored, excluded, candidates = self.task.evaluate_rows(rows)
        loss = None
        if scored:  # the pairwise loss over the scored queries only (never an excluded one)
            relevant = {row[0] for row in rows if row[3] >= self.task.relevance_threshold}
            kept = [row for row in rows if row[0] in relevant]
            scores = torch.tensor([row[2] for row in kept], dtype=torch.float64)
            grades = torch.tensor([row[3] for row in kept])
            numerator, denominator = pairwise_logistic_loss(
                scores, grades, [row[0] for row in kept], weighting=self.task.weighting
            )
            loss = float(numerator) / denominator if denominator else None
        return _record(self.task, loss=loss, means=means, scored=scored, excluded=excluded, candidates=candidates)


class _Unavailable:
    """The default evaluation path sees rows, not queries: a declared
    ranking metric is computed by :class:`RankingEval` only, and refuses
    anywhere else rather than reporting a per-batch value."""

    def update(self, target: Any, prediction: Any) -> None:
        raise RankingError(
            "ranking.mrr / ranking.recall / ranking.ndcg are query-grouped metrics: pass eval_step_fn=RankingTask.eval_step() "
            "(and objective=RankingTask.objective()) so they are computed per complete query"
        )

    def result(self) -> Optional[float]:
        return None


def _check_k_config(config: Mapping[str, Any]) -> None:
    if set(config) != {"k"}:
        raise ValueError(f"a ranking metric takes exactly one option, k; got {sorted(config)}")
    _count(config["k"], "k", minimum=1)


def _register_metrics() -> None:
    from .monitors import register_metric, registered_metrics

    known = set(registered_metrics())
    for metric in ("mrr", "recall", "ndcg"):
        if (f"ranking.{metric}", 1) not in known:
            register_metric(
                f"ranking.{metric}",
                1,
                lambda config: _Unavailable(),
                input="continuous",
                mode="max",
                check_config=_check_k_config,
            )


_register_metrics()
