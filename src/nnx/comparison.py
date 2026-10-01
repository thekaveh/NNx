"""Multi-seed experiment summaries and paired comparisons (FEAT-032).

Repeated runs of one configuration (seeds, replicates) and two
configurations compared over the same replicates — reported **read-only**:
nothing here builds or loads a model, reads a checkpoint, elects a
``runs/best`` pointer or writes into a run.

- :class:`Observation` — one attempt's committed evaluation of one metric,
  declaring everything a comparison depends on: the :class:`Metric` (name,
  direction, unit), the evaluation split, the checkpoint-selection rule,
  the configuration, data and split identities, the replicate key, the
  completion status and the evaluation behind the value. A fact that is not
  known is ``None`` and stays unknown — it is never guessed.
- :func:`summarize` — groups observations that may be pooled (same metric,
  split, selection rule, configuration, data and split identity) and
  reports each group's count, mean and **sample** standard deviation
  (denominator ``n - 1``; unavailable for ``n = 1``, and both for
  ``n = 0``). Failed and non-finite attempts stay listed and counted, never
  dropped. Observations that differ in any of those fields are never
  pooled: the summary is stratified, one group each, naming the fields that
  differ (:meth:`Summary.pooled` refuses).
- :func:`compare` — a **paired** comparison of configuration B against A
  over replicate keys whose meaning you declare (``pairing=``; equal seeds
  alone do not prove two runs are paired). Each matched pair gives the
  signed delta ``B - A`` — never flipped for a metric to minimize — and
  unmatched or incomplete replicates are listed. An optional
  :class:`Bootstrap` records its unit, seed and method, and its interval is
  labelled seed variability.
- :class:`ComparisonReport` — a machine-readable :meth:`~ComparisonReport.table`
  and a concise :meth:`~ComparisonReport.text` view carrying every run and
  attempt id, status, selection rule, unmatched pair and signed delta;
  saved as strict JSON (``nnx.comparison/1``) and reloaded with its results
  re-derived and checked. Input order never changes a report.
- :func:`observations_from_runs` — reads observations from saved runs
  (``run.yaml``, ``idps.csv`` and the FEAT-019 provenance files, each read
  once) without loading a model or a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from ._artifacts import atomic_write, check_keys, parse_json, read_text

__all__ = [
    "FORMAT",
    "Bootstrap",
    "ComparisonError",
    "ComparisonReport",
    "GroupSummary",
    "IncompatibleObservations",
    "Metric",
    "Observation",
    "Pair",
    "PairedComparison",
    "Summary",
    "compare",
    "observations_from_runs",
    "summarize",
]

FORMAT = "nnx.comparison/1"
DIRECTIONS = ("maximize", "minimize")
STATUSES = ("completed", "failed", "cancelled", "running", "unknown")
# The fields observations must share to be pooled into one group, in report order.
POOL_FIELDS = ("metric", "split", "selection", "config", "data", "split_id")
# Identity facts that may be unknown (None); reported per group when they are.
_OPTIONAL_IDENTITIES = ("config", "data", "split_id")


class ComparisonError(ValueError):
    """A malformed observation, a duplicate attempt, an invalid pairing or a
    report file that does not hold what it claims."""


class IncompatibleObservations(ComparisonError):
    """Observations that may not be pooled or paired: ``fields`` names what
    differs."""

    def __init__(self, message: str, *, fields: Sequence[str]) -> None:
        super().__init__(message)
        self.fields = tuple(fields)


def _text(value: Any, what: str, *, optional: bool = False) -> Optional[str]:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ComparisonError(f"{what} must be a non-empty string{' or None' if optional else ''}, got {value!r}")
    return value


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"an observation's value must be a number or None, got {value!r}")
    return float(value)


def _encode(value: Optional[float]) -> Any:
    """Strict JSON for a value: non-finite numbers become strings."""
    if value is None or math.isfinite(value):
        return value
    return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")


def _decode(value: Any) -> Optional[float]:
    if isinstance(value, str):
        decoded = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}.get(value)
        if decoded is None:
            raise ComparisonError(f"a stored value is a number, null, 'NaN', 'Infinity' or '-Infinity', got {value!r}")
        return decoded
    return _number(value)


# --- what is compared ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """What a value measures: its ``name``, whether larger or smaller is
    better (``direction``: ``"maximize"`` / ``"minimize"``) and its ``unit``
    (``None``: unitless or not declared)."""

    name: str
    direction: str
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.name, "Metric.name")
        if self.direction not in DIRECTIONS:
            raise ComparisonError(f"Metric.direction must be one of {DIRECTIONS}, got {self.direction!r}")
        _text(self.unit, "Metric.unit", optional=True)

    def state(self) -> dict[str, Any]:
        return {"name": self.name, "direction": self.direction, "unit": self.unit}

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> Metric:
        check_keys(state, required=("name", "direction", "unit"), what="metric", error=ComparisonError)
        return Metric(state["name"], state["direction"], state["unit"])

    def __str__(self) -> str:
        return f"{self.name} ({self.direction}{f', {self.unit}' if self.unit else ''})"


@dataclass(frozen=True)
class Observation:
    """One attempt's committed evaluation of one metric.

    Attributes:
        run_id: the run that produced it.
        attempt_id: the attempt within that run (FEAT-019), when known.
        metric: what ``value`` measures.
        value: the value (``None`` when the attempt has none; NaN and
            infinities are kept and counted as non-finite).
        status: the attempt's completion status: ``"completed"``,
            ``"failed"``, ``"cancelled"``, ``"running"`` or ``"unknown"``.
        split: the evaluation split the value was measured on (for example
            ``"validation"``).
        selection: the checkpoint-selection rule behind the value (for
            example ``"last"``, or ``"best:loss"`` for a monitor's election).
        config: the configuration identity.
        data: the data identity.
        split_id: the split identity (a split manifest digest, for example).
        replicate: the replicate key (for example ``"seed=3"``) — what
            :func:`compare` pairs on.
        evaluation: the committed evaluation behind the value (for example
            the epoch record and the checkpoint it was committed with).

    ``config``, ``data``, ``split_id``, ``replicate``, ``attempt_id`` and
    ``evaluation`` are ``None`` when unknown.
    """

    run_id: str
    metric: Metric
    value: Optional[float]
    split: str
    selection: str
    status: str = "completed"
    attempt_id: Optional[str] = None
    config: Optional[str] = None
    data: Optional[str] = None
    split_id: Optional[str] = None
    replicate: Optional[str] = None
    evaluation: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.run_id, "Observation.run_id")
        if not isinstance(self.metric, Metric):
            raise ComparisonError(f"Observation.metric must be a Metric, got {type(self.metric).__name__}")
        object.__setattr__(self, "value", _number(self.value))
        _text(self.split, "Observation.split")
        _text(self.selection, "Observation.selection")
        if self.status not in STATUSES:
            raise ComparisonError(f"Observation.status must be one of {STATUSES}, got {self.status!r}")
        for name in ("attempt_id", "config", "data", "split_id", "replicate", "evaluation"):
            _text(getattr(self, name), f"Observation.{name}", optional=True)

    @property
    def id(self) -> str:
        """The attempt id, or ``run:<run id>`` when the attempt is unknown."""
        return self.attempt_id if self.attempt_id is not None else f"run:{self.run_id}"

    @property
    def finite(self) -> bool:
        """Whether this is a completed attempt with a finite value."""
        return self.status == "completed" and self.value is not None and math.isfinite(self.value)

    def _pool_key(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in POOL_FIELDS)

    def _sort_key(self) -> tuple[str, ...]:
        return (self.replicate or "", self.run_id, self.attempt_id or "")

    def state(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "metric": self.metric.state(),
            "value": _encode(self.value),
            "status": self.status,
            "split": self.split,
            "selection": self.selection,
            "config": self.config,
            "data": self.data,
            "split_id": self.split_id,
            "replicate": self.replicate,
            "evaluation": self.evaluation,
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> Observation:
        fields = ("run_id", "attempt_id", "metric", "value", "status", "split", "selection")
        fields += ("config", "data", "split_id", "replicate", "evaluation")
        check_keys(state, required=fields, what="observation", error=ComparisonError)
        if not isinstance(state["metric"], Mapping):
            raise ComparisonError(f"an observation's metric is a mapping, got {state['metric']!r}")
        return Observation(
            run_id=state["run_id"],
            attempt_id=state["attempt_id"],
            metric=Metric.from_state(state["metric"]),
            value=_decode(state["value"]),
            status=state["status"],
            split=state["split"],
            selection=state["selection"],
            config=state["config"],
            data=state["data"],
            split_id=state["split_id"],
            replicate=state["replicate"],
            evaluation=state["evaluation"],
        )


def _checked(observations: Iterable[Observation], what: str = "observations") -> tuple[Observation, ...]:
    """The observations in a canonical order (input order never matters);
    a duplicate attempt (per metric) is refused."""
    items = tuple(observations)
    bad = [type(item).__name__ for item in items if not isinstance(item, Observation)]
    if bad:
        raise ComparisonError(f"{what} must be Observations, got {bad}")
    seen: set[tuple[str, str]] = set()
    duplicates = []
    for item in items:
        key = (item.metric.name, item.id)
        if key in seen:
            duplicates.append(item.id)
        seen.add(key)
    if duplicates:
        raise ComparisonError(f"duplicate attempt ids in {what}: {sorted(set(duplicates))}")
    return tuple(sorted(items, key=lambda item: (repr(item._pool_key()), item._sort_key())))


# --- statistics ------------------------------------------------------------------------------------


def _mean_std(values: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
    """Mean and sample standard deviation (denominator ``n - 1``)."""
    n = len(values)
    if n == 0:
        return None, None
    mean = math.fsum(values) / n
    if n == 1:
        return mean, None
    return mean, math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))


@dataclass(frozen=True)
class GroupSummary:
    """One poolable group: the shared fields (``metric``, ``split``,
    ``selection``, ``config``, ``data``, ``split_id``), every observation
    (sorted), and the statistics over its completed, finite values.

    ``n`` counts those values; ``n_attempts`` every observation,
    ``n_failed`` those not completed (failed, cancelled, running or of
    unknown status) and ``n_nonfinite`` completed ones without a finite
    value. ``std`` is the sample standard deviation (``n - 1``): ``None``
    when ``n < 2``; ``mean`` is ``None`` when ``n == 0``. ``unknown`` names
    the identity fields that are not known."""

    metric: Metric
    split: str
    selection: str
    config: Optional[str]
    data: Optional[str]
    split_id: Optional[str]
    observations: tuple[Observation, ...]
    n: int = field(init=False)
    n_attempts: int = field(init=False)
    n_failed: int = field(init=False)
    n_nonfinite: int = field(init=False)
    mean: Optional[float] = field(init=False)
    std: Optional[float] = field(init=False)

    def __post_init__(self) -> None:
        values = [_value_of(item.value) for item in self.observations if item.finite]
        mean, std = _mean_std(values)
        object.__setattr__(self, "n", len(values))
        object.__setattr__(self, "n_attempts", len(self.observations))
        object.__setattr__(self, "n_failed", sum(item.status != "completed" for item in self.observations))
        object.__setattr__(
            self, "n_nonfinite", sum(item.status == "completed" and not item.finite for item in self.observations)
        )
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @property
    def unknown(self) -> tuple[str, ...]:
        return tuple(name for name in _OPTIONAL_IDENTITIES if getattr(self, name) is None)

    def label(self) -> str:
        return f"{self.config or '<unknown config>'} | {self.metric} on {self.split} @ {self.selection}"

    def state(self) -> dict[str, Any]:
        return {
            "metric": self.metric.state(),
            "split": self.split,
            "selection": self.selection,
            "config": self.config,
            "data": self.data,
            "split_id": self.split_id,
            "unknown": list(self.unknown),
            "attempts": [item.id for item in self.observations],
            "n": self.n,
            "n_attempts": self.n_attempts,
            "n_failed": self.n_failed,
            "n_nonfinite": self.n_nonfinite,
            "mean": self.mean,
            "std": self.std,
        }


def _value_of(value: Optional[float]) -> float:
    if value is None:  # callers filter on Observation.finite
        raise ComparisonError("an observation without a value has no finite value")
    return value


def _group(observations: tuple[Observation, ...]) -> GroupSummary:
    first = observations[0]
    return GroupSummary(
        metric=first.metric,
        split=first.split,
        selection=first.selection,
        config=first.config,
        data=first.data,
        split_id=first.split_id,
        observations=tuple(sorted(observations, key=Observation._sort_key)),
    )


def _differing(groups: Sequence[Any]) -> tuple[str, ...]:
    return tuple(name for name in POOL_FIELDS if len({repr(getattr(group, name)) for group in groups}) > 1)


@dataclass(frozen=True)
class Summary:
    """A stratified summary: one :class:`GroupSummary` per poolable group,
    and the fields that differ between groups (``differing``)."""

    groups: tuple[GroupSummary, ...]

    @property
    def differing(self) -> tuple[str, ...]:
        return _differing(self.groups)

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(item for group in self.groups for item in group.observations)

    def pooled(self) -> GroupSummary:
        """The single group, when every observation may be pooled; otherwise
        :class:`IncompatibleObservations` naming the differing fields."""
        if len(self.groups) != 1:
            raise IncompatibleObservations(
                f"these observations form {len(self.groups)} groups and are not pooled: they differ in "
                f"{list(self.differing)}",
                fields=self.differing,
            )
        return self.groups[0]

    def state(self) -> dict[str, Any]:
        return {"groups": [group.state() for group in self.groups], "differing": list(self.differing)}


def summarize(observations: Iterable[Observation]) -> Summary:
    """Group ``observations`` into poolable groups (same metric, split,
    selection rule, configuration, data and split identity) and summarize
    each; see :class:`GroupSummary`. Duplicate attempts are refused. A
    known identity never pools with an unknown one."""
    items = _checked(observations)
    if not items:
        raise ComparisonError("summarize() needs at least one observation")
    groups: dict[tuple[Any, ...], list[Observation]] = {}
    for item in items:
        groups.setdefault(item._pool_key(), []).append(item)
    return Summary(tuple(_group(tuple(members)) for members in groups.values()))


# --- paired comparisons ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bootstrap:
    """A percentile bootstrap of the mean paired delta: ``resamples``
    resamples of the replicate pairs (the unit), drawn with replacement by
    ``numpy.random.default_rng(seed)``, giving a ``level`` interval. It
    describes **seed variability** over these replicates — not a
    population-level or significance claim."""

    seed: int
    resamples: int = 2000
    level: float = 0.95

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ComparisonError(f"Bootstrap.seed must be a non-negative integer, got {self.seed!r}")
        if isinstance(self.resamples, bool) or not isinstance(self.resamples, int) or self.resamples < 1:
            raise ComparisonError(f"Bootstrap.resamples must be a positive integer, got {self.resamples!r}")
        if isinstance(self.level, bool) or not isinstance(self.level, (int, float)) or not 0 < self.level < 1:
            raise ComparisonError(f"Bootstrap.level must be in (0, 1), got {self.level!r}")

    def state(self) -> dict[str, Any]:
        return {
            "method": "percentile bootstrap of the mean paired delta",
            "unit": "replicate pair",
            "seed": self.seed,
            "resamples": self.resamples,
            "level": float(self.level),
            "label": "seed variability",
        }

    def interval(self, deltas: Sequence[float]) -> Optional[tuple[float, float]]:
        if len(deltas) < 2:
            return None
        import numpy as np

        values = np.asarray(deltas, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        means = values[rng.integers(0, len(values), size=(self.resamples, len(values)))].mean(axis=1)
        tail = (1.0 - float(self.level)) / 2.0
        low, high = np.quantile(means, [tail, 1.0 - tail])
        return float(low), float(high)


@dataclass(frozen=True)
class Pair:
    """One replicate key on both sides: the observations' ids and the signed
    delta ``b - a`` (``None`` unless both are completed and finite)."""

    replicate: str
    a: str
    b: str
    delta: Optional[float]

    def state(self) -> dict[str, Any]:
        return {"replicate": self.replicate, "a": self.a, "b": self.b, "delta": self.delta}


@dataclass(frozen=True)
class PairedComparison:
    """Configuration B against A over shared replicate keys.

    ``pairs`` holds every matched replicate (``delta`` is ``B - A``, never
    flipped: for a metric to minimize, a negative delta means B is lower);
    ``unmatched_a`` / ``unmatched_b`` list the replicate keys found on one
    side only, with their attempt ids. ``n``, ``mean`` and ``std`` (``n - 1``)
    summarize the finite deltas; ``interval`` is the bootstrap's, when one
    was requested and ``n >= 2``."""

    a: GroupSummary
    b: GroupSummary
    pairing: str
    bootstrap: Optional[Bootstrap] = None
    pairs: tuple[Pair, ...] = field(init=False)
    unmatched_a: tuple[tuple[str, str], ...] = field(init=False)
    unmatched_b: tuple[tuple[str, str], ...] = field(init=False)
    n: int = field(init=False)
    mean: Optional[float] = field(init=False)
    std: Optional[float] = field(init=False)
    interval: Optional[tuple[float, float]] = field(init=False)

    def __post_init__(self) -> None:
        by_a = {item.replicate: item for item in self.a.observations}
        by_b = {item.replicate: item for item in self.b.observations}
        pairs = []
        for key in sorted(set(by_a) & set(by_b), key=str):
            left, right = by_a[key], by_b[key]
            delta = _value_of(right.value) - _value_of(left.value) if left.finite and right.finite else None
            pairs.append(Pair(str(key), left.id, right.id, delta))
        deltas = [pair.delta for pair in pairs if pair.delta is not None]
        mean, std = _mean_std(deltas)
        object.__setattr__(self, "pairs", tuple(pairs))
        object.__setattr__(
            self, "unmatched_a", tuple((str(k), by_a[k].id) for k in sorted(set(by_a) - set(by_b), key=str))
        )
        object.__setattr__(
            self, "unmatched_b", tuple((str(k), by_b[k].id) for k in sorted(set(by_b) - set(by_a), key=str))
        )
        object.__setattr__(self, "n", len(deltas))
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "interval", None if self.bootstrap is None else self.bootstrap.interval(deltas))

    @property
    def metric(self) -> Metric:
        return self.a.metric

    def state(self) -> dict[str, Any]:
        return {
            "a": [item.id for item in self.a.observations],
            "b": [item.id for item in self.b.observations],
            "a_config": self.a.config,
            "b_config": self.b.config,
            "metric": self.metric.state(),
            "delta": "b - a",
            "pairing": self.pairing,
            "pairs": [pair.state() for pair in self.pairs],
            "unmatched_a": [list(item) for item in self.unmatched_a],
            "unmatched_b": [list(item) for item in self.unmatched_b],
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "bootstrap": None if self.bootstrap is None else self.bootstrap.state(),
            "interval": None if self.interval is None else list(self.interval),
        }


def _side(observations: Iterable[Observation], what: str) -> GroupSummary:
    group = summarize(_checked(observations, what)).groups
    if len(group) != 1:
        raise IncompatibleObservations(
            f"side {what} must be one poolable group; it differs in {list(_differing(group))}",
            fields=_differing(group),
        )
    side = group[0]
    missing = [item.id for item in side.observations if item.replicate is None]
    if missing:
        raise ComparisonError(f"pairing needs a replicate key on every observation; side {what} lacks one on {missing}")
    keys = [item.replicate for item in side.observations]
    repeated = sorted({key for key in keys if keys.count(key) > 1}, key=str)
    if repeated:
        raise ComparisonError(f"side {what} repeats replicate keys {repeated}: a replicate pairs once")
    return side


def compare(
    a: Iterable[Observation],
    b: Iterable[Observation],
    *,
    pairing: str,
    bootstrap: Optional[Bootstrap] = None,
) -> PairedComparison:
    """Pair configuration ``b`` against ``a`` by replicate key.

    Each side must be one poolable group with a unique replicate key per
    observation, and the sides must agree on everything but the
    configuration (metric, split, selection rule, data and split identity).
    ``pairing`` declares what a shared replicate key means (for example
    ``"same seed for initialisation and data order, same split"``): NNx
    cannot prove it, so it is required and recorded. Deltas are signed
    ``B - A``."""
    _text(pairing, "pairing")
    left, right = _side(a, "a"), _side(b, "b")
    differ = tuple(name for name in _differing((left, right)) if name != "config")
    if differ:
        raise IncompatibleObservations(f"the two sides differ in {list(differ)} and are not paired", fields=differ)
    if bootstrap is not None and not isinstance(bootstrap, Bootstrap):
        raise ComparisonError(f"bootstrap must be a Bootstrap or None, got {type(bootstrap).__name__}")
    ids = {item.id for item in left.observations} & {item.id for item in right.observations}
    if ids:
        raise ComparisonError(f"an attempt cannot be on both sides: {sorted(ids)}")
    return PairedComparison(left, right, pairing, bootstrap)


# --- the report -----------------------------------------------------------------------------------


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.6g}" if value != 0 else "+0"


@dataclass(frozen=True)
class ComparisonReport:
    """A summary of ``observations`` and any paired comparisons, as one
    artifact: :meth:`table` (machine-readable rows), :meth:`text` (a concise
    view), :meth:`state` / :meth:`save` / :meth:`load` (strict JSON,
    ``nnx.comparison/1``)."""

    summary: Summary
    comparisons: tuple[PairedComparison, ...] = ()

    @staticmethod
    def build(
        observations: Iterable[Observation],
        comparisons: Sequence[tuple[Iterable[Observation], Iterable[Observation], str]] = (),
        *,
        bootstrap: Optional[Bootstrap] = None,
    ) -> ComparisonReport:
        """Summarize ``observations`` and run each ``(a, b, pairing)``
        comparison (every compared observation must be among them)."""
        summary = summarize(observations)
        known = {(item.metric.name, item.id) for item in summary.observations}
        results = []
        for a, b, pairing in comparisons:
            a, b = tuple(a), tuple(b)
            stray = sorted(item.id for item in (*a, *b) if (item.metric.name, item.id) not in known)
            if stray:
                raise ComparisonError(f"compared observations must be in the report's observations; not found: {stray}")
            results.append(compare(a, b, pairing=pairing, bootstrap=bootstrap))
        return ComparisonReport(summary, tuple(results))

    # ---------- views ----------

    def table(self) -> list[dict[str, Any]]:
        """One row per observation, group, pair and unmatched replicate,
        each with a ``row`` kind; values are plain JSON types."""
        rows: list[dict[str, Any]] = []
        for index, group in enumerate(self.summary.groups):
            stats = {key: value for key, value in group.state().items() if key != "attempts"}
            rows.append({"row": "group", "group": index, **stats})
            for item in group.observations:
                rows.append({"row": "observation", "group": index, **item.state()})
        for index, result in enumerate(self.comparisons):
            head = {"comparison": index, "a_config": result.a.config, "b_config": result.b.config}
            for pair in result.pairs:
                rows.append({"row": "pair", **head, **pair.state()})
            for side, unmatched in (("a", result.unmatched_a), ("b", result.unmatched_b)):
                for replicate, attempt in unmatched:
                    rows.append({"row": "unmatched", **head, "side": side, "replicate": replicate, "attempt": attempt})
            summary = {k: v for k, v in result.state().items() if k not in ("pairs", "unmatched_a", "unmatched_b")}
            rows.append({"row": "comparison", **head, **{k: v for k, v in summary.items() if k not in head}})
        return rows

    def text(self) -> str:
        """A concise, fixed-order text view."""
        lines = [f"{FORMAT}: {len(self.summary.observations)} observation(s) in {len(self.summary.groups)} group(s)"]
        if len(self.summary.groups) > 1:
            lines.append(f"groups differ in: {', '.join(self.summary.differing)} (stratified, not pooled)")
        for index, group in enumerate(self.summary.groups):
            lines.append(f"[{index}] {group.label()}")
            lines.append(
                f"    data={group.data or 'unknown'} split_id={group.split_id or 'unknown'}"
                f" n={group.n}/{group.n_attempts} failed={group.n_failed} nonfinite={group.n_nonfinite}"
                f" mean={_fmt(group.mean)} sd(n-1)={_fmt(group.std)}"
            )
            for item in group.observations:
                lines.append(
                    f"    - {item.replicate or '<no replicate>'} run={item.run_id} attempt={item.attempt_id or 'unknown'}"
                    f" status={item.status} value={_fmt(item.value)}"
                )
        for index, result in enumerate(self.comparisons):
            lines.append(
                f"compare[{index}] B={result.b.config} vs A={result.a.config}: delta = B - A on {result.metric}"
                f" (pairing: {result.pairing})"
            )
            for pair in result.pairs:
                lines.append(f"    {pair.replicate}: {pair.b} - {pair.a} = {_fmt(pair.delta)}")
            for side, unmatched in (("A", result.unmatched_a), ("B", result.unmatched_b)):
                for replicate, attempt in unmatched:
                    lines.append(f"    unmatched {side}: {replicate} ({attempt})")
            line = f"    n={result.n} mean delta={_fmt(result.mean)} sd(n-1)={_fmt(result.std)}"
            if result.interval is not None and result.bootstrap is not None:
                low, high = result.interval
                line += (
                    f"; {result.bootstrap.level:.0%} seed-variability interval [{_fmt(low)}, {_fmt(high)}]"
                    f" (bootstrap seed {result.bootstrap.seed}, {result.bootstrap.resamples} resamples)"
                )
            lines.append(line)
        return "\n".join(lines) + "\n"

    # ---------- serialization ----------

    def state(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "observations": [item.state() for item in self.summary.observations],
            "summary": self.summary.state(),
            "comparisons": [result.state() for result in self.comparisons],
        }

    def to_json(self) -> str:
        return json.dumps(self.state(), sort_keys=True, indent=2, allow_nan=False) + "\n"

    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        atomic_write(path, self.to_json())

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> ComparisonReport:
        """Rebuild a report from its observations and comparison specs; the
        stored results must equal the re-derived ones."""
        if not isinstance(state, Mapping):
            raise ComparisonError(f"a comparison report is a JSON object, got {type(state).__name__}")
        check_keys(
            state, required=("format", "observations", "summary", "comparisons"), what="report", error=ComparisonError
        )
        if state["format"] != FORMAT:
            raise ComparisonError(f"unsupported comparison report format {state['format']!r} (expected {FORMAT!r})")
        observations = [Observation.from_state(item) for item in state["observations"]]
        by_id = {(item.metric.name, item.id): item for item in observations}
        specs = []
        bootstrap_states = []
        for result in state["comparisons"]:
            metric = result["metric"]["name"]
            try:
                a = [by_id[(metric, attempt)] for attempt in result["a"]]
                b = [by_id[(metric, attempt)] for attempt in result["b"]]
            except KeyError as error:
                raise ComparisonError(f"a comparison names an attempt the report does not hold: {error}") from error
            specs.append((a, b, result["pairing"]))
            bootstrap_states.append(result["bootstrap"])
        rebuilt = ComparisonReport(
            summarize(observations),
            tuple(
                compare(a, b, pairing=pairing, bootstrap=_bootstrap_from_state(boot))
                for (a, b, pairing), boot in zip(specs, bootstrap_states, strict=True)
            ),
        )
        if json.loads(json.dumps(rebuilt.state(), sort_keys=True)) != json.loads(
            json.dumps(dict(state), sort_keys=True)
        ):
            raise ComparisonError("the report's stored results do not match its observations (edited or corrupt)")
        return rebuilt

    @staticmethod
    def load(path: Union[str, os.PathLike[str]]) -> ComparisonReport:
        text = read_text(path, "comparison report", ComparisonError)
        return ComparisonReport.from_state(parse_json(text, "comparison report", ComparisonError))


def _bootstrap_from_state(state: Optional[Mapping[str, Any]]) -> Optional[Bootstrap]:
    if state is None:
        return None
    return Bootstrap(seed=state["seed"], resamples=state["resamples"], level=state["level"])


# --- reading saved runs ---------------------------------------------------------------------------

_SCALARS = ("loss", "error", "accuracy", "f1", "recall", "precision")


def _canonical_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _config_identity(run_state: Mapping[str, Any]) -> str:
    """The run's configuration without what varies per replicate: the run
    id, the seed and the salt."""
    state = {key: value for key, value in run_state.items() if key not in ("id", "salt")}
    train = dict(state.get("train") or {})
    train.pop("seed", None)
    state["train"] = train
    return "sha256:" + hashlib.sha256(_canonical_text(state).encode("utf-8")).hexdigest()[:16]


def _identities(refs: Mapping[str, Any]) -> Optional[str]:
    """``name=kind:value`` per identity (``name=unknown`` when unknown), or
    ``None`` when the manifest declares none."""
    if not refs:
        return None
    return ";".join(
        f"{name}={ref.kind}" + ("" if ref.value is None else f":{ref.value}") for name, ref in sorted(refs.items())
    )


def _metric_value(edp: Any, name: str) -> Optional[float]:
    if edp is None:
        return None
    if name in edp.metrics:
        return float(edp.metrics[name])
    if name in _SCALARS:
        value = getattr(edp, name)
        return None if value is None else float(value)
    if name in edp.extra:
        return float(edp.extra[name])
    return None


def _read_run(run_id: str, root: Optional[str]) -> tuple[Mapping[str, Any], list[Any], Any, bool]:
    import pandas as pd
    import yaml

    from .nn.params.nn_iteration_data_point import NNIterationDataPoint
    from .nn.params.nn_run import _HISTORY_PROTOCOL_FILE, _runs_root, _validate_run_id
    from .provenance import load_provenance

    run_path = os.path.join(_runs_root(root), _validate_run_id(run_id))
    with open(os.path.join(run_path, "run.yaml"), encoding="utf-8") as handle:
        run_state = yaml.safe_load(handle)
    if not isinstance(run_state, Mapping):
        raise ComparisonError(f"malformed run.yaml for run {run_id}")
    csv_path = os.path.join(run_path, "idps.csv")
    try:
        rows = pd.read_csv(csv_path).to_dict(orient="records") if os.path.isfile(csv_path) else []
    except pd.errors.EmptyDataError:
        rows = []
    idps = [NNIterationDataPoint.from_state(row) for row in rows]
    committed_by_last = os.path.isfile(os.path.join(run_path, _HISTORY_PROTOCOL_FILE))
    return run_state, idps, load_provenance(run_id, root), committed_by_last


def observations_from_runs(
    run_ids: Iterable[str],
    *,
    metric: Metric,
    split: str = "validation",
    selection: str = "last",
    root: Optional[str] = None,
    replicate: Optional[str] = "seed",
    config: Optional[Mapping[str, str]] = None,
) -> list[Observation]:
    """Observations of ``metric`` read from saved runs, without loading a
    model or a checkpoint (``run.yaml``, ``idps.csv`` and the provenance
    files are read once each, and nothing is written).

    Args:
        run_ids: the runs (under ``root``'s ``runs/``).
        metric: the metric to read — a named metric (``train.metrics``), a
            built-in field (``loss``, ``error``, ``accuracy``, ``f1``,
            ``recall``, ``precision``) or an ``extra_metrics`` name.
        split: ``"validation"`` (the epoch's validation record) or
            ``"train"`` (its training summary).
        selection: ``"last"`` (the last committed epoch) or ``"best"`` (the
            last committed epoch the run's monitor elected; unknown for a
            run without one).
        replicate: ``"seed"`` keys each observation ``seed=<train.seed>``
            (unknown for an unseeded run); ``None`` leaves it unknown.
        config: run id → declared configuration label; by default, a digest
            of the run's configuration without its seed and salt.

    The status and attempt id come from the run's attempt record
    (FEAT-019; ``"unknown"`` without one), the data and split identities
    from its manifest. When the run's history is committed by its LAST
    checkpoint and no attempt record names that checkpoint's epoch, the
    committed epoch is unknown and so is the value.
    """
    if not isinstance(metric, Metric):
        raise ComparisonError(f"metric must be a Metric, got {type(metric).__name__}")
    if split not in ("validation", "train"):
        raise ComparisonError(f"split must be 'validation' or 'train', got {split!r}")
    if selection not in ("last", "best"):
        raise ComparisonError(f"selection must be 'last' or 'best', got {selection!r}")
    if replicate not in ("seed", None):
        raise ComparisonError(f"replicate must be 'seed' or None, got {replicate!r}")
    observations = []
    for run_id in run_ids:
        run_state, idps, provenance, committed_by_last = _read_run(run_id, root)
        attempt = None if provenance is None else provenance.attempt
        status = "unknown" if attempt is None else attempt.status
        if status not in STATUSES:
            status = "unknown"
        committed: Optional[int] = None
        known = True
        if committed_by_last:
            last = None if attempt is None else attempt.last_committed
            if last is not None and last.get("epoch") is not None:
                committed = int(last["epoch"])
            else:
                known = False
        epochs: dict[int, Any] = {}
        for idp in idps:
            if committed is None or idp.epoch_idx <= committed:
                epochs[idp.epoch_idx] = idp  # the epoch's last record
        chosen = None
        rule = selection
        if known and epochs:
            ordered = [epochs[epoch] for epoch in sorted(epochs)]
            if selection == "last":
                chosen = ordered[-1]
            else:
                elected = [idp for idp in ordered if idp.selection is not None and idp.selection.improved]
                chosen = elected[-1] if elected else None
                if elected:
                    monitor = elected[-1].selection.monitor
                    rule = f"best:{monitor.metric}({monitor.split},{monitor.mode or 'metric default'})"
        value = None
        evaluation = None
        if chosen is not None:
            edp = chosen.val_edp if split == "validation" else chosen.monitored_train_edp()
            value = _metric_value(edp, metric.name)
            evaluation = f"epoch {chosen.epoch_idx} {split} record (idps.csv)"
            last = None if attempt is None else attempt.last_committed
            if last is not None and last.get("checkpoint") is not None:
                evaluation += f"; committed with {last['checkpoint']} generation {last.get('generation')}"
        elif not known:
            evaluation = "unknown: the committed epoch is not recorded"
        train = run_state.get("train") or {}
        seed = train.get("seed")
        manifest = None if provenance is None else provenance.manifest
        data = None if manifest is None else _identities(manifest.data)
        if data is None and train.get("data_id") is not None:
            data = f"data_id:{train['data_id']}"
        observations.append(
            Observation(
                run_id=run_id,
                attempt_id=None if attempt is None else attempt.attempt_id,
                metric=metric,
                value=value,
                status=status,
                split=split,
                selection=rule,
                config=(config or {}).get(run_id) or _config_identity(run_state),
                data=data,
                split_id=None if manifest is None else _identities(manifest.splits),
                replicate=f"seed={seed}" if replicate == "seed" and seed is not None else None,
                evaluation=evaluation,
            )
        )
    return observations
