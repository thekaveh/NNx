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
  (``run.yaml``, ``idps.csv`` — or a FEAT-036 history journal's committed
  records — and the FEAT-019 provenance files, each read once; for a run with a parent, ``metadata.yaml`` and every ancestor's
  ``run.yaml`` and provenance files) without loading a model or a
  checkpoint: the committed epoch is the LAST epoch the attempt record
  names, and a journal is read only up to it.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
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
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ComparisonError(f"an observation's value must be a number or None, got {value!r}")
    try:
        return float(value)
    except OverflowError as error:  # an integer too large for a float
        raise ComparisonError(f"an observation's value does not fit a float: {error}") from error


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
    try:
        total = math.fsum(values)
    except OverflowError:  # fsum raises on an intermediate overflow
        total = math.inf
    mean = total / n if math.isfinite(total) else math.fsum(v / n for v in values)  # no overflow in the sum
    if n == 1:
        return mean, None
    deviations = [v - mean for v in values]
    if not all(math.isfinite(d) for d in deviations):
        deviations = [v / 2 - mean / 2 for v in values]  # halved: the spread itself overflows a float
        scale = 2.0
    else:
        scale = 1.0
    largest = max(abs(d) for d in deviations)
    if largest == 0.0:
        return mean, 0.0
    std = largest * math.sqrt(math.fsum((d / largest) ** 2 for d in deviations) / (n - 1))  # scaled: no overflow
    return mean, std * scale  # a float product saturates to inf, never raises


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
        object.__setattr__(self, "observations", tuple(self.observations))  # an iterator is read once
        if not isinstance(self.metric, Metric):
            raise ComparisonError(f"GroupSummary.metric must be a Metric, got {type(self.metric).__name__}")
        _text(self.split, "GroupSummary.split")
        _text(self.selection, "GroupSummary.selection")
        if not self.observations:
            raise ComparisonError("a GroupSummary needs at least one observation")
        _checked(self.observations, "GroupSummary.observations")
        # Canonical order, whatever order they were given in: a report then
        # saves and reloads identically.
        object.__setattr__(self, "observations", tuple(sorted(self.observations, key=Observation._sort_key)))
        key = tuple(getattr(self, name) for name in POOL_FIELDS)
        strays = sorted(item.id for item in self.observations if item._pool_key() != key)
        if strays:
            raise IncompatibleObservations(
                f"a group's observations share its metric, split, selection, config, data and split_id; "
                f"{strays} do not",
                fields=POOL_FIELDS,
            )
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
            "mean": _encode(self.mean),
            "std": _encode(self.std),
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))  # an iterator is read once
        bad = [type(group).__name__ for group in self.groups if not isinstance(group, GroupSummary)]
        if bad:
            raise ComparisonError(f"Summary.groups must be GroupSummary objects, got {bad}")
        if not self.groups:
            raise ComparisonError("a Summary needs at least one group")
        keys = [tuple(getattr(group, name) for name in POOL_FIELDS) for group in self.groups]
        if len(set(map(repr, keys))) != len(keys):
            raise ComparisonError("a Summary's groups must be distinct poolable groups")
        _checked(self.observations, "Summary observations")
        # The order summarize() gives: canonical, whatever order they were given in.
        object.__setattr__(
            self,
            "groups",
            tuple(sorted(self.groups, key=lambda group: repr(tuple(getattr(group, name) for name in POOL_FIELDS)))),
        )

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


MAX_RESAMPLES = 1_000_000
BOOTSTRAP_BLOCK = 1 << 22  # resampled pairs held at once


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
        if (
            isinstance(self.resamples, bool)
            or not isinstance(self.resamples, int)
            or not 1 <= self.resamples <= MAX_RESAMPLES
        ):
            raise ComparisonError(
                f"Bootstrap.resamples must be an integer in [1, {MAX_RESAMPLES}], got {self.resamples!r}"
            )
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
        n = len(values)
        rng = np.random.default_rng(self.seed)
        means = np.empty(self.resamples, dtype=np.float64)
        chunk = max(1, BOOTSTRAP_BLOCK // n)  # resamples drawn at once: memory bounded whatever the pair count
        for start in range(0, self.resamples, chunk):
            stop = min(start + chunk, self.resamples)
            drawn = rng.integers(0, n, size=(stop - start, n))
            with np.errstate(over="ignore", invalid="ignore"):
                block = values[drawn].mean(axis=1)
            if not np.isfinite(block).all():  # a sum overflowed: divide first (cannot overflow)
                block = (values / n)[drawn].sum(axis=1)
            means[start:stop] = block
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
        return {"replicate": self.replicate, "a": self.a, "b": self.b, "delta": _encode(self.delta)}


@dataclass(frozen=True)
class PairedComparison:
    """Configuration B against A over shared replicate keys.

    ``pairs`` holds every matched replicate (``delta`` is ``B - A``, never
    flipped: for a metric to minimize, a negative delta means B is lower);
    ``unmatched_a`` / ``unmatched_b`` list the replicate keys found on one
    side only, with their attempt ids. ``n``, ``mean`` and ``std`` (``n - 1``)
    summarize the finite deltas; ``n_nonfinite`` counts the pairs of two
    finite values whose delta overflows a float (excluded from ``n`` and
    reported); ``interval`` is the bootstrap's, when one was requested and
    ``n >= 2``."""

    a: GroupSummary
    b: GroupSummary
    pairing: str
    bootstrap: Optional[Bootstrap] = None
    pairs: tuple[Pair, ...] = field(init=False)
    unmatched_a: tuple[tuple[str, str], ...] = field(init=False)
    unmatched_b: tuple[tuple[str, str], ...] = field(init=False)
    n: int = field(init=False)
    n_nonfinite: int = field(init=False)
    mean: Optional[float] = field(init=False)
    std: Optional[float] = field(init=False)
    interval: Optional[tuple[float, float]] = field(init=False)

    def __post_init__(self) -> None:
        for side, name in ((self.a, "a"), (self.b, "b")):
            if not isinstance(side, GroupSummary):
                raise ComparisonError(f"PairedComparison.{name} must be a GroupSummary, got {type(side).__name__}")
        if self.bootstrap is not None and not isinstance(self.bootstrap, Bootstrap):
            raise ComparisonError(f"bootstrap must be a Bootstrap or None, got {type(self.bootstrap).__name__}")
        _check_replicates(self.a, "a")
        _check_replicates(self.b, "b")
        differ = tuple(name for name in _differing((self.a, self.b)) if name != "config")
        if differ:
            raise IncompatibleObservations(f"the two sides differ in {list(differ)} and are not paired", fields=differ)
        _text(self.pairing, "pairing")
        both = {item.id for item in self.a.observations} & {item.id for item in self.b.observations}
        if both:
            raise ComparisonError(f"an attempt cannot be on both sides: {sorted(both)}")
        by_a = {item.replicate: item for item in self.a.observations}
        by_b = {item.replicate: item for item in self.b.observations}
        pairs = []
        for key in sorted(set(by_a) & set(by_b), key=str):
            left, right = by_a[key], by_b[key]
            delta = _value_of(right.value) - _value_of(left.value) if left.finite and right.finite else None
            pairs.append(Pair(str(key), left.id, right.id, delta))
        deltas = [pair.delta for pair in pairs if pair.delta is not None and math.isfinite(pair.delta)]
        mean, std = _mean_std(deltas)
        object.__setattr__(self, "pairs", tuple(pairs))
        object.__setattr__(
            self, "unmatched_a", tuple((str(k), by_a[k].id) for k in sorted(set(by_a) - set(by_b), key=str))
        )
        object.__setattr__(
            self, "unmatched_b", tuple((str(k), by_b[k].id) for k in sorted(set(by_b) - set(by_a), key=str))
        )
        object.__setattr__(self, "n", len(deltas))
        object.__setattr__(
            self, "n_nonfinite", sum(pair.delta is not None and not math.isfinite(pair.delta) for pair in pairs)
        )
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
            "n_nonfinite": self.n_nonfinite,
            "mean": _encode(self.mean),
            "std": _encode(self.std),
            "bootstrap": None if self.bootstrap is None else self.bootstrap.state(),
            "interval": None if self.interval is None else [_encode(v) for v in self.interval],
        }


def _side(observations: Iterable[Observation], what: str) -> GroupSummary:
    group = summarize(_checked(observations, what)).groups
    if len(group) != 1:
        raise IncompatibleObservations(
            f"side {what} must be one poolable group; it differs in {list(_differing(group))}",
            fields=_differing(group),
        )
    side = group[0]
    _check_replicates(side, what)
    return side


def _check_replicates(side: GroupSummary, what: str) -> None:
    missing = [item.id for item in side.observations if item.replicate is None]
    if missing:
        raise ComparisonError(f"pairing needs a replicate key on every observation; side {what} lacks one on {missing}")
    keys = Counter(item.replicate for item in side.observations)
    repeated = sorted((key for key, n in keys.items() if n > 1), key=str)
    if repeated:
        raise ComparisonError(f"side {what} repeats replicate keys {repeated}: a replicate pairs once")


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
    return PairedComparison(left, right, pairing, bootstrap)


# --- the report -----------------------------------------------------------------------------------


def _percent(level: float) -> str:
    """``level`` as an exact percentage (0.95 -> ``95``, 0.9999999 -> ``99.99999``)."""
    text = format(Decimal(repr(float(level))) * 100, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparisons", tuple(self.comparisons))  # an iterator is read once; hashable
        if not isinstance(self.summary, Summary):
            raise ComparisonError(f"ComparisonReport.summary must be a Summary, got {type(self.summary).__name__}")
        bad = [type(result).__name__ for result in self.comparisons if not isinstance(result, PairedComparison)]
        if bad:
            raise ComparisonError(f"ComparisonReport.comparisons must be PairedComparisons, got {bad}")
        known = {(item.metric.name, item.id): item.state() for item in self.summary.observations}
        for result in self.comparisons:
            stray = sorted(
                item.id
                for item in (*result.a.observations, *result.b.observations)
                if known.get((item.metric.name, item.id)) != item.state()
            )
            if stray:
                raise ComparisonError(
                    f"compared observations must be the report's own observations (same id and every field); "
                    f"not found or different: {stray}"
                )

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
        known = {(item.metric.name, item.id): item for item in summary.observations}
        results = []
        for a, b, pairing in comparisons:
            a, b = tuple(a), tuple(b)
            stray = sorted(
                getattr(item, "id", repr(item))
                for item in (*a, *b)
                if not isinstance(item, Observation)
                or (known.get((item.metric.name, item.id)) is None)
                or known[(item.metric.name, item.id)].state() != item.state()
            )
            if stray:
                raise ComparisonError(
                    f"compared observations must be the report's own observations (same id and every field); "
                    f"not found or different: {stray}"
                )
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
                f" n={group.n}/{group.n_attempts} not_completed={group.n_failed} nonfinite={group.n_nonfinite}"
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
            if result.n_nonfinite:
                line += f" ({result.n_nonfinite} matched pair(s) with a delta beyond float range excluded)"
            if result.interval is not None and result.bootstrap is not None:
                low, high = result.interval
                line += (
                    f"; {_percent(result.bootstrap.level)}% seed-variability interval [{_fmt(low)}, {_fmt(high)}]"
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
        try:
            if not isinstance(state["observations"], list) or not isinstance(state["comparisons"], list):
                raise TypeError("observations and comparisons are lists")
            observations = [Observation.from_state(item) for item in state["observations"]]
            by_id = {(item.metric.name, item.id): item for item in observations}
            specs = []
            bootstrap_states = []
            for result in state["comparisons"]:
                metric = result["metric"]["name"]
                sides = (result["a"], result["b"])  # a missing key is a malformed report
                try:
                    a = [by_id[(metric, attempt)] for attempt in sides[0]]
                    b = [by_id[(metric, attempt)] for attempt in sides[1]]
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
        except ComparisonError:
            raise
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ComparisonError(f"malformed comparison report: {type(error).__name__}: {error}") from error
        try:
            stored = json.loads(json.dumps(dict(state), sort_keys=True))
        except (TypeError, ValueError, RecursionError) as error:
            raise ComparisonError(f"malformed comparison report: {type(error).__name__}: {error}") from error
        if json.loads(json.dumps(rebuilt.state(), sort_keys=True)) != stored:
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


_PER_REPLICATE = ("seed", "parent_run_id", "parent_checkpoint")


MAX_LINEAGE = 64  # ancestors a run may have before its lineage is refused (a corrupt history)
ALIASES = ("best",)  # moving pointers a run can resume from: never followed as a fixed parent


def _parent(run_state: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The training (or ``Trainer``) section that names a parent run."""
    for name in ("train", "trainer"):
        section = run_state.get(name)
        if isinstance(section, Mapping) and section.get("parent_run_id") is not None:
            return section
    return None


def _resume_mode(run_path: str, run_id: str) -> Optional[str]:
    """How a run continued its parent — ``stateful``, ``weights_only``, or
    ``fresh`` when it did not resume (a born-again generation, say) — from
    ``metadata.yaml``; ``None`` when the run predates the record. A file that cannot be read, or
    an unknown mode, is refused: the run's procedure would be unknown."""
    import yaml

    path = os.path.join(run_path, "metadata.yaml")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ComparisonError(
            f"run {run_id}: metadata.yaml is unreadable ({type(error).__name__}: {error}); "
            "how it continued its parent is unknown"
        ) from error
    resume = metadata.get("resume") if isinstance(metadata, Mapping) else None
    mode = resume.get("mode") if isinstance(resume, Mapping) else None
    if mode is not None and mode not in ("fresh", "stateful", "weights_only"):
        raise ComparisonError(f"run {run_id}: metadata.yaml records an unknown resume mode {mode!r}")
    return mode


def _resolvable(run_id: str, root: Optional[str]) -> bool:
    """Whether ``run_id`` names a run directory (not an alias, a malformed id
    or a deleted run). A directory whose files cannot be read is resolvable:
    reading it then raises."""
    from .nn.params.nn_run import _runs_root, _validate_run_id

    if run_id in ALIASES:
        return False
    try:
        path = os.path.join(_runs_root(root), _validate_run_id(run_id))
    except (TypeError, ValueError):
        return False
    return os.path.lexists(path)


def _provenance(run_id: str, root: Optional[str]) -> Any:
    """A run's FEAT-019 provenance (``None`` without it); unreadable files
    raise :class:`ComparisonError`."""
    from .provenance import load_provenance

    try:
        return load_provenance(run_id, root)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ComparisonError(
            f"run {run_id}: its provenance files (provenance.json / attempt.json) are unreadable: "
            f"{type(error).__name__}: {error}"
        ) from error


def _attempt_id(provenance: Any) -> Optional[str]:
    attempt = None if provenance is None else provenance.attempt
    return None if attempt is None else attempt.attempt_id


def _recorded_parent(provenance: Any) -> Mapping[str, Any]:
    """The parent a run recorded in its ``attempt.json`` when it started
    (FEAT-019 records it for a resume): the parent's attempt, checkpoint
    tag, epoch and generation. Empty when nothing was recorded."""
    attempt = None if provenance is None else provenance.attempt
    parent = None if attempt is None else attempt.parent
    return parent if isinstance(parent, Mapping) else {}


def _nonempty(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _instant(value: Any) -> Optional[datetime]:
    """A recorded ISO timestamp as an aware instant, or ``None`` when it is
    not one (an unknown time orders nothing)."""
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def _completed_before(own: Any, current: Any) -> bool:
    """Whether the parent's current attempt had completed when the child
    started — recorded facts only, fixed once the parent ends. A parent
    still training, killed or retrained after the child started had not;
    when either time is unknown, it is not known to have."""
    parent = None if current is None else current.attempt
    child = None if own is None else own.attempt
    if parent is None or child is None or parent.status != "completed":
        return False
    finished, started = _instant(parent.finished_at), _instant(child.started_at)
    return finished is not None and started is not None and finished <= started


def _parent_facts(provenance: Any) -> dict[str, Any]:
    """What a parent's provenance declares beyond its ``run.yaml``: its data
    and split identities (not its status, which changes while it trains)."""
    manifest = None if provenance is None else provenance.manifest
    return {
        "data": None if manifest is None else _identities(manifest.data),
        "splits": None if manifest is None else _identities(manifest.splits),
    }


def _run_identity(
    run_id: str,
    root: Optional[str],
    cache: dict[str, tuple[str, int]],
    known: Optional[tuple[Mapping[str, Any], str]] = None,
) -> str:
    """:func:`_config_identity` of a saved run, with the identity of every
    run it descends from: a continuation, a fine-tune or a born-again
    generation is the procedure *parent then child*, so it pools only with
    runs whose parents had the same configuration, data and splits, and
    were continued the same way from the same checkpoint
    tag — and the epoch it started at, as the child recorded it in its
    ``attempt.json``, when that epoch is part of the procedure: a fixed
    checkpoint tag, or a parent that had not completed (still training,
    killed). From a completed parent's ``last`` or ``best`` the epoch is an
    outcome of the parent's training and is left out. A parent is followed
    when its current attempt is the one the child recorded or, with nothing
    recorded, when it had completed before the child started, or has no
    provenance and the child recorded the epoch it started at; a needed
    start epoch that was not recorded follows nothing. Otherwise (the moving
    ``best`` alias, a deleted run, a retrained parent, or a child whose order
    with its parent is unknown) it is named by the recorded parent
    checkpoint generation (else attempt and start epoch), so only siblings
    of that checkpoint pool; without a record, the lineage is unknown and
    names nothing shared — an unknown lineage never pools. Errors in an
    ancestor's files are raised, as for the run's own."""
    return _resolve(run_id, root, cache, (), known)[0]


def _resolve(
    run_id: str,
    root: Optional[str],
    cache: dict[str, tuple[str, int]],
    path: tuple[str, ...],
    known: Optional[tuple[Mapping[str, Any], str]] = None,
) -> tuple[str, int]:
    """``(identity, ancestors)``, cached per call: the same answer whatever
    order runs are listed in."""
    if run_id in cache:
        return cache[run_id]
    if run_id in path:
        raise ComparisonError(f"run {path[0]}: its lineage is a cycle ({' -> '.join((*path, run_id))})")
    if len(path) > MAX_LINEAGE:
        raise ComparisonError(f"run {path[0]}: a lineage of more than {MAX_LINEAGE} ancestors")
    run_state, run_path = known if known is not None else _read_run_state(run_id, root)
    parent = _parent(run_state)
    lineage, ancestors = None, 0
    if parent is not None:
        parent_id = str(parent["parent_run_id"])
        lineage = {"checkpoint": parent.get("parent_checkpoint"), "mode": _resume_mode(run_path, run_id)}
        own = _provenance(run_id, root)
        record = _recorded_parent(own)
        recorded, generation = _nonempty(record.get("attempt_id")), _nonempty(record.get("generation"))
        resolvable = _resolvable(parent_id, root)
        current = _provenance(parent_id, root) if resolvable else None
        done = _completed_before(own, current)
        epoch = record.get("epoch")
        epoch = epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else None
        fixed = record.get("checkpoint", lineage["checkpoint"]) not in ("last", "best")
        # A resumed run (or one whose mode predates the record) started at an epoch
        # of its parent; a born-again generation (``fresh``) at none. That epoch is
        # part of the procedure for a fixed checkpoint tag, or when the parent had
        # not completed (still training, killed); from a completed parent's
        # ``last`` or ``best`` it is an outcome of its training.
        needs_epoch = lineage["mode"] != "fresh" and (fixed or not done)
        unrecorded_parent = current is None or current.attempt is None  # a parent without provenance
        if (
            resolvable
            and (epoch is not None or not needs_epoch)
            and (
                recorded == _attempt_id(current)
                if recorded is not None
                # Nothing recorded about the attempt: a parent known to have completed
                # before the child started, or a provenance-less one the child resumed.
                else (done or (unrecorded_parent and epoch is not None))
            )
        ):
            # The parent attempt the run started from (a parent without
            # provenance has only its run.yaml to declare).
            identity, above = _resolve(parent_id, root, cache, (*path, run_id))
            lineage.update(parent=identity, **_parent_facts(current))
            if needs_epoch:
                lineage["start_epoch"] = epoch
            ancestors = above + 1
        else:
            # An alias (``best`` moves), a deleted parent, one retrained in place
            # since, or one whose order with the child is unknown: the parent
            # checkpoint generation (else attempt, with its start epoch) recorded
            # when the run started names it; with neither, the lineage is
            # unknown and names nothing shared.
            if generation is not None:
                lineage["parent"] = f"parent checkpoint {generation}"
            elif recorded is not None and epoch is not None:
                lineage["parent"] = f"parent attempt {recorded} at epoch {epoch}"
            else:
                lineage["parent"] = f"unknown lineage of run {run_id}"
            ancestors = 1
    if ancestors > MAX_LINEAGE:
        raise ComparisonError(f"run {run_id}: a lineage of more than {MAX_LINEAGE} ancestors")
    cache[run_id] = (_config_identity(run_state, lineage), ancestors)
    return cache[run_id]


def _config_identity(run_state: Mapping[str, Any], lineage: Optional[Mapping[str, Any]] = None) -> str:
    """The run's configuration without what varies per replicate: the run
    id, the salt, the seed of its training (or ``Trainer``) parameters, and
    the id of its parent run — replaced by the parent's identity, how it was
    continued and from which checkpoint tag (``lineage``, see
    :func:`_run_identity`)."""
    state = {key: value for key, value in run_state.items() if key not in ("id", "salt")}
    if lineage is not None:
        state[" lineage"] = dict(lineage)  # the space keeps it apart from any run.yaml key
    for section in ("train", "trainer"):
        if isinstance(state.get(section), Mapping):
            state[section] = {k: v for k, v in state[section].items() if k not in _PER_REPLICATE}
    model = state.get("model")
    if isinstance(model, Mapping):
        model = {k: v for k, v in model.items() if k != "device"}  # where it ran is not what it is
        net = model.get("net")
        if isinstance(net, Mapping) and net.get("kind") == "registered":
            model["net"] = {k: v for k, v in net.items() if k != "seed"}  # a ModelSpec's init seed (FEAT-006)
        state["model"] = model
    return "sha256:" + hashlib.sha256(_canonical_text(state).encode("utf-8")).hexdigest()[:16]


def _replicate_key(run_state: Mapping[str, Any]) -> Optional[str]:
    """``seed=<train or Trainer seed>``, plus ``init_seed=<ModelSpec seed>``
    when a registered model's own initialization seed differs from it;
    ``None`` for a run without a training seed (its data order is not
    reproducible, whatever its initialization seed)."""
    train = run_state.get("train") or {}
    trainer = run_state.get("trainer") or {}
    seed = train.get("seed") if train.get("seed") is not None else trainer.get("seed")
    net = (run_state.get("model") or {}).get("net")
    init = net.get("seed") if isinstance(net, Mapping) and net.get("kind") == "registered" else None
    if seed is None:
        return None
    parts = [f"seed={seed}"]
    if init is not None and init != seed:
        parts.append(f"init_seed={init}")
    return ",".join(parts)


def _declared_monitor(run_state: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    for section in ("train", "trainer"):
        monitor = (run_state.get(section) or {}).get("monitor") if isinstance(run_state.get(section), Mapping) else None
        if isinstance(monitor, Mapping):
            return monitor
    return None


def _identities(refs: Mapping[str, Any]) -> Optional[str]:
    """``name=kind:value`` per identity (``name=unknown`` when unknown), or
    ``None`` when the manifest declares none."""
    if not refs or all(ref.kind == "unknown" for ref in refs.values()):
        return None
    return ";".join(
        f"{name}={ref.kind}" + ("" if ref.value is None else f":{ref.value}") for name, ref in sorted(refs.items())
    )


def _metric_value(edp: Any, name: str) -> Optional[float]:
    try:
        return _raw_metric_value(edp, name)
    except (TypeError, ValueError) as error:
        raise ComparisonError(f"the record's {name!r} is not a number: {error}") from error


def _raw_metric_value(edp: Any, name: str) -> Optional[float]:
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


def _read_run_state(run_id: str, root: Optional[str]) -> tuple[Mapping[str, Any], str]:
    """A run's ``run.yaml`` and its directory."""
    import yaml

    from .nn.params.nn_run import _runs_root, _validate_run_id

    try:
        run_path = os.path.join(_runs_root(root), _validate_run_id(run_id))
    except (TypeError, ValueError) as error:
        raise ComparisonError(f"not a run id: {run_id!r} ({error})") from error
    try:
        with open(os.path.join(run_path, "run.yaml"), encoding="utf-8") as handle:
            run_state = yaml.safe_load(handle)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ComparisonError(f"run {run_id}: run.yaml is missing or unreadable: {error}") from error
    if not isinstance(run_state, Mapping):
        raise ComparisonError(f"malformed run.yaml for run {run_id}")
    for section in ("model", "train", "trainer"):
        if run_state.get(section) is not None and not isinstance(run_state[section], Mapping):
            raise ComparisonError(f"malformed run.yaml for run {run_id}: {section!r} is not a mapping")
    return run_state, run_path


def _read_run(run_id: str, root: Optional[str]) -> tuple[Mapping[str, Any], list[Any], Any, Optional[int], bool, str]:
    """A run's ``run.yaml``, records, provenance, committed epoch (``None``:
    not filtered), whether that epoch is known, and its directory. Under the
    history protocol LAST commits the history, and the attempt record names
    LAST's epoch: no checkpoint is read."""
    import pandas as pd

    from .history import _own_records, has_journal
    from .nn.params.nn_iteration_data_point import NNIterationDataPoint
    from .nn.params.nn_run import _HISTORY_PROTOCOL_FILE

    run_state, run_path = _read_run_state(run_id, root)
    provenance = _provenance(run_id, root)
    attempt = None if provenance is None else provenance.attempt
    last = None if attempt is None else attempt.last_committed
    if last is not None and not isinstance(last, Mapping):
        raise ComparisonError(f"run {run_id}: attempt.json's last_committed is not a mapping: {last!r}")
    committed: Optional[int] = None
    known = True
    if os.path.isfile(os.path.join(run_path, _HISTORY_PROTOCOL_FILE)):
        epoch = None if last is None else last.get("epoch")
        if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0):
            raise ComparisonError(f"run {run_id}: attempt.json's committed epoch is not an epoch: {epoch!r}")
        committed, known = epoch, epoch is not None
    if has_journal(run_path):  # a FEAT-036 history journal writes no idps.csv
        try:
            last_records: dict[Any, Any] = {}
            # Its records up to the committed epoch, checked chunk by chunk —
            # none while that epoch is unknown (the uncommitted tail stays unread).
            for record in _own_records(run_id, root, committed, {}) if known else ():
                last_records[record.epoch_idx] = record
        except Exception as error:
            raise ComparisonError(
                f"run {run_id}: unreadable history journal: {type(error).__name__}: {error}"
            ) from error
        idps = list(last_records.values())
    else:
        csv_path = os.path.join(run_path, "idps.csv")
        try:
            rows = pd.read_csv(csv_path).to_dict(orient="records") if os.path.isfile(csv_path) else []
            last_rows: dict[Any, Any] = {}
            for row in rows:  # only each epoch's last record is ever read
                last_rows[row.get("epoch_idx")] = row
            idps = [NNIterationDataPoint.from_state(row) for row in last_rows.values()]
        except Exception as error:  # an empty or damaged history is not an empty one
            raise ComparisonError(f"run {run_id}: malformed idps.csv: {type(error).__name__}: {error}") from error
    return run_state, idps, provenance, committed, known, run_path


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
    model or a checkpoint (``run.yaml``, ``idps.csv`` — or, for a run
    trained with a ``HistoryJournal``, the journal's committed records,
    each chunk checked — and the provenance files are read once each — plus, for a run with a parent, its
    ``metadata.yaml`` and every ancestor's ``run.yaml`` and provenance
    files, whose data and split identities join the configuration identity
    and whose attempt records decide which parent attempt a child started
    from; an ancestor's unreadable file is refused — and nothing is
    written).

    Args:
        run_ids: the runs (under ``root``'s ``runs/``).
        metric: the metric to read — a named metric (``train.metrics``), a
            built-in field (``loss``, ``error``, ``accuracy``, ``f1``,
            ``recall``, ``precision``) or an ``extra_metrics`` name.
        split: ``"validation"`` (the epoch's validation record) or
            ``"train"`` (its whole-epoch training summary — recorded only by
            runs that declare metrics or a monitor; never a last batch).
        selection: ``"last"`` (the last committed epoch) or ``"best"`` (the
            last committed epoch the run's monitor elected; the rule names
            the declared monitor, and the value is unknown for a run without
            one or whose monitor elected nothing).
        replicate: ``"seed"`` keys each observation ``seed=<seed>`` (the
            training or ``Trainer`` seed), plus ``init_seed=<seed>`` for a
            registered ``ModelSpec`` whose own seed differs; unknown for a
            run without a training seed. ``None`` leaves it unknown.
        config: run id → declared configuration label, for every run; by
            default, a digest of the run's configuration without its salt,
            seeds and device, in which a parent run's id is replaced by the
            parent's own identity, the checkpoint tag, the resume mode and,
            when it is part of the procedure, the epoch the run started at —
            so a continuation, fine-tune or later generation pools only
            with runs descended the same way from the same configuration
            (an unknown lineage pools with nothing).

    Runs should be trained with ``provenance=`` (FEAT-019): the status and
    attempt id come from the run's attempt record (``"unknown"`` without
    one) and the data and split identities from its manifest. A run's
    history is committed by its LAST checkpoint, so without an attempt
    record naming that checkpoint's epoch the committed epoch — and so the
    value — is unknown. The recorded epoch is used as is (LAST itself is
    never opened, so nothing is unpickled), and a history journal is read
    only up to it — not at all while it is unknown, so an uncommitted tail
    is never read. A legacy run (no commit marker) keeps its value,
    but its status is unknown, so it is never counted in a group's ``n``.
    A NaN metric is read back as missing: ``idps.csv`` writes NaN as an
    empty cell.
    """
    if not isinstance(metric, Metric):
        raise ComparisonError(f"metric must be a Metric, got {type(metric).__name__}")
    if split not in ("validation", "train"):
        raise ComparisonError(f"split must be 'validation' or 'train', got {split!r}")
    if selection not in ("last", "best"):
        raise ComparisonError(f"selection must be 'last' or 'best', got {selection!r}")
    if replicate not in ("seed", None):
        raise ComparisonError(f"replicate must be 'seed' or None, got {replicate!r}")
    if isinstance(run_ids, (str, bytes)):
        raise ComparisonError("run_ids is a sequence of run ids, not one string")
    run_ids = list(run_ids)
    if config is not None:
        unlabelled = [run_id for run_id in run_ids if run_id not in config]
        if unlabelled:
            raise ComparisonError(f"config= labels some runs but not {unlabelled}; label every run or none")
    from .history import has_journal

    observations = []
    identities: dict[str, tuple[str, int]] = {}  # run id -> (identity with its parents', ancestor count)
    for run_id in run_ids:
        run_state, idps, provenance, committed, known, run_path = _read_run(run_id, root)
        source = "history journal" if has_journal(run_path) else "idps.csv"
        attempt = None if provenance is None else provenance.attempt
        status = "unknown" if attempt is None else attempt.status
        if status not in STATUSES:
            status = "unknown"
        last = None if attempt is None else attempt.last_committed
        epochs: dict[int, Any] = {}
        for idp in idps:
            if committed is None or idp.epoch_idx <= committed:
                epochs[idp.epoch_idx] = idp  # the epoch's last record
        chosen = None
        rule = selection
        unknown_reason = "unknown: the committed epoch is not recorded" if not known else None
        if known and not epochs:
            unknown_reason = f"unknown: the run records no committed epoch ({source})"
        if selection == "best":
            # The rule is the run's declared monitor, whether or not any epoch was elected.
            monitor = _declared_monitor(run_state)
            if monitor is not None:
                rule = f"best:{monitor.get('metric')}({monitor.get('split')},{monitor.get('mode') or 'metric default'})"
        if known and epochs:
            ordered = [epochs[epoch] for epoch in sorted(epochs)]
            if selection == "last":
                chosen = ordered[-1]
            else:
                elected = [idp for idp in ordered if idp.selection is not None and idp.selection.improved]
                chosen = elected[-1] if elected else None
                if chosen is None:
                    unknown_reason = (
                        "unknown: the run's monitor elected no epoch"
                        if _declared_monitor(run_state) is not None
                        else "unknown: the run declares no monitor"
                    )
        value = None
        evaluation = unknown_reason
        if chosen is not None:
            if split == "validation":
                edp = chosen.val_edp
            else:  # the whole-epoch summary only: a last batch is not an epoch's value
                edp = chosen.train_summary
            value = _metric_value(edp, metric.name)
            evaluation = f"epoch {chosen.epoch_idx} {split} record ({source})"
            if edp is None:
                evaluation = (
                    "unknown: no whole-epoch training summary (declare metrics or a monitor)"
                    if split == "train"
                    else f"unknown: epoch {chosen.epoch_idx} has no validation record"
                )
            elif value is None:
                evaluation = f"unknown: the epoch {chosen.epoch_idx} {split} record has no {metric.name!r} value " + (
                    "(missing, or NaN — idps.csv writes NaN as an empty cell)"
                    if source == "idps.csv"
                    else "(missing, or NaN)"
                )
            if committed is not None and last is not None and last.get("checkpoint") is not None:
                evaluation += f"; committed with {last['checkpoint']} generation {last.get('generation')}"
        train = run_state.get("train") or {}
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
                config=(
                    config[run_id]
                    if config is not None
                    else _run_identity(run_id, root, identities, known=(run_state, run_path))
                ),
                data=data,
                split_id=None if manifest is None else _identities(manifest.splits),
                replicate=_replicate_key(run_state) if replicate == "seed" else None,
                evaluation=evaluation,
            )
        )
    return observations
