"""Reproducible group, time and stratified splits (FEAT-017).

``plan_split`` assigns sample ids to train / validation / test and returns a
:class:`SplitManifest`: the memberships, the strategy, its parameters and
seed, and the identity of the source it was planned on. A manifest is plain
JSON (``to_json`` / ``save`` / ``load``); its ``digest()`` is the split
identity that provenance records (``ExperimentManifest(splits={...})``).

Strategies:

- ``"group"`` — every group (a patient, a user, a document) lands in exactly
  one split. Groups are visited in a seeded order and each goes to the active
  split furthest below its row target; an active split that ends up empty then
  takes the smallest group from the split holding the most groups. Row counts
  therefore approximate ``proportions``.
- ``"chronological"`` — ``cutoffs=(validation_start, test_start)``:
  train is ``t < validation_start``, validation ``validation_start <= t <
  test_start``, test ``t >= test_start``. A timestamp on a cutoff starts the
  later split, so tied rows never separate. ``gap`` excludes the rows within
  ``gap`` before each cutoff (``cutoff - gap <= t < cutoff``). Either cutoff
  may be ``None`` to leave that split empty. Times are numbers, dates, or
  naive or tz-aware datetimes (``numpy.datetime64`` included), one kind per
  plan; a date plan takes a whole-day ``timedelta`` gap.
- ``"stratified"`` — per class, ids are ranked by seed and cut by the largest
  remainder, carrying the rounding across classes so the totals follow
  ``proportions``; every class reaches every active split, and that one-row
  minimum (never carried to other classes) is the only thing that can push a
  small class's rows beyond its share. A class with fewer rows than active
  splits raises (``insufficient="raise"``, the default) or stays in training
  only (``insufficient="train"``).

Plans depend only on the ids and their groups, times or labels: never on row
order, and never on any global RNG (ids are ordered by SHA-256 of the seed
and the id). ``seed=None`` draws a fresh seed from the OS and records it.

:meth:`SplitManifest.resolve` maps a manifest onto the current source's ids.
Duplicate, missing or unexpected ids, a changed source identity or a
positional-only plan (row positions with no source identity) raise
:class:`SplitError` before any loader exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import secrets
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import cached_property
from typing import Any, Optional, Union

import numpy as np

from ._validation import require_count, require_finite_real
from .provenance import IdentityRef, _freeze, _thaw, canonical_bytes, hash_bytes

__all__ = [
    "FORMAT",
    "SPLITS",
    "STRATEGIES",
    "SplitError",
    "SplitIndices",
    "SplitManifest",
    "plan_split",
]

FORMAT = "nnx.split/1"
"""Format version of a serialized manifest (part of its digest)."""

SPLITS = ("train", "validation", "test")
STRATEGIES = ("group", "chronological", "stratified", "explicit")
"""``plan_split`` produces the first three; ``"explicit"`` marks a manifest
built by hand from memberships you already have."""

_PLANNED = STRATEGIES[:3]
_INSUFFICIENT = ("raise", "train")
_MEMBERS = (*SPLITS, "excluded")

SampleId = Union[str, int]


class SplitError(ValueError):
    """A split that cannot be planned or replayed as asked."""


# --- ids --------------------------------------------------------------------------------------


def _key(value: Any, what: str) -> SampleId:
    if isinstance(value, bool):
        raise TypeError(f"{what} must be str or int, not bool ({value!r})")
    if isinstance(value, str):
        return str(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    raise TypeError(f"{what} must be str or int, got {type(value).__name__} {value!r}")


def _keys(values: Iterable[Any], what: str) -> tuple[SampleId, ...]:
    out = tuple(_key(value, what) for value in values)
    if len({type(value) for value in out}) > 1:
        raise TypeError(f"{what} must share one id type (all str or all int), got a mix")
    return out


def _preview(values: Sequence[Any]) -> str:
    shown = list(values[:5])
    return f"{shown}{' …' if len(values) > 5 else ''}"


def _source(value: Union[IdentityRef, str, None]) -> IdentityRef:
    if isinstance(value, IdentityRef):
        return value
    if value is None:
        return IdentityRef.unknown()
    if isinstance(value, str):
        return IdentityRef.declared(value)
    raise TypeError(f"source must be an IdentityRef, a declared id string or None, got {value!r}")


def _ranker(seed: int, tag: str) -> Callable[[SampleId], bytes]:
    """Seeded order without an RNG: SHA-256 of the seed, a tag and the
    type-tagged id, stable across processes and library versions."""
    base = hashlib.sha256(f"nnx.split.rank\0{seed}\0{tag}\0".encode())

    def rank(key: SampleId) -> bytes:
        digest = base.copy()
        digest.update((f"s{key}" if isinstance(key, str) else f"i{key}").encode("utf-8"))
        return digest.digest()

    return rank


def _sequence(value: Any, size: int, what: str) -> tuple[Any, ...]:
    """A fixed-size tuple from an ordered iterable (tuple, list, array,
    Series). Strings, mappings and sets are rejected: a set has no order."""
    if isinstance(value, (str, bytes, Mapping, Set)) or not isinstance(value, Iterable):
        raise ValueError(f"{what}, got {value!r}")
    items = tuple(value)
    if len(items) != size:
        raise ValueError(f"{what}, got {value!r}")
    return items


# --- the manifest -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitIndices:
    """Row positions of each split in the source a manifest was resolved on,
    in the manifest's membership order."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    excluded: tuple[int, ...] = ()


@dataclass(frozen=True, eq=False)
class SplitManifest:
    """Who is in which split, and how that was decided.

    Memberships are sample ids (``ids="sample"``) or row positions
    (``ids="position"``, which can be replayed only against a recorded source
    identity). They are disjoint, stored in sorted order, and ``excluded``
    holds the rows a chronological ``gap`` left out. Equality and hashing
    follow :meth:`canonical_bytes`, computed once (the manifest is frozen).
    """

    strategy: str
    train: tuple[SampleId, ...]
    validation: tuple[SampleId, ...]
    test: tuple[SampleId, ...]
    excluded: tuple[SampleId, ...] = ()
    ids: str = "sample"
    source: IdentityRef = field(default_factory=IdentityRef.unknown)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        if self.ids not in ("sample", "position"):
            raise ValueError(f"ids must be 'sample' or 'position', got {self.ids!r}")
        members = {name: _keys(getattr(self, name), f"{name} ids") for name in _MEMBERS}
        _keys((value for values in members.values() for value in values), "split manifest ids")
        seen: dict[SampleId, str] = {}
        for name, values in members.items():
            if len(set(values)) != len(values):
                raise ValueError(f"{name} lists a sample id more than once")
            for value in values:
                if value in seen:
                    raise ValueError(f"sample id {value!r} is in both {seen[value]} and {name}")
                seen[value] = name
        if self.ids == "position" and set(seen) != set(range(len(seen))):
            raise ValueError("a positional manifest must cover row positions 0..n-1 exactly")
        for name, values in members.items():
            object.__setattr__(self, name, tuple(sorted(values)))
        object.__setattr__(self, "source", _source(self.source))
        if not isinstance(self.parameters, Mapping):
            raise TypeError(f"parameters must be a mapping, got {type(self.parameters).__name__}")
        object.__setattr__(self, "parameters", _freeze(json.loads(canonical_bytes(dict(self.parameters)))))
        if self.seed is not None:
            object.__setattr__(self, "seed", require_count(self.seed, "seed", owner="SplitManifest"))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SplitManifest) and self.canonical_bytes() == other.canonical_bytes()

    def __repr__(self) -> str:  # counts, not every id: a plan can hold millions
        sizes = ", ".join(f"{name}={len(getattr(self, name))}" for name in _MEMBERS)
        return (
            f"SplitManifest(strategy={self.strategy!r}, ids={self.ids!r}, {sizes}, seed={self.seed!r}, "
            f"source={self.source.kind!r}, digest={self.digest()[:19]!r})"
        )

    def __hash__(self) -> int:
        return hash(self.canonical_bytes())

    @property
    def n_rows(self) -> int:
        """Rows the plan covers, excluded ones included."""
        return sum(len(getattr(self, name)) for name in _MEMBERS)

    def state(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "strategy": self.strategy,
            "ids": self.ids,
            "source": self.source.state(),
            "parameters": _thaw(self.parameters),
            "seed": self.seed,
            **{name: list(getattr(self, name)) for name in _MEMBERS},
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> SplitManifest:
        if state.get("format") != FORMAT:
            raise ValueError(f"unsupported split manifest format {state.get('format')!r}; expected {FORMAT!r}")
        return SplitManifest(
            strategy=state["strategy"],
            ids=state.get("ids", "sample"),
            source=IdentityRef.from_state(state["source"]),
            parameters=state.get("parameters") or {},
            seed=state.get("seed"),
            **{name: tuple(state.get(name) or ()) for name in _MEMBERS},
        )

    @cached_property
    def _canonical(self) -> bytes:
        return canonical_bytes(self.state())

    def canonical_bytes(self) -> bytes:
        return self._canonical

    @cached_property
    def _identity(self) -> IdentityRef:
        return hash_bytes(self.canonical_bytes())

    def digest(self) -> str:
        """``sha256:<hex>`` of the canonical manifest: the split identity."""
        return str(self._identity.value)

    def identity(self) -> IdentityRef:
        """The split identity as a verified provenance reference."""
        return self._identity

    def to_json(self) -> str:
        return json.dumps(self.state(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"

    @staticmethod
    def from_json(text: str) -> SplitManifest:
        return SplitManifest.from_state(json.loads(text))

    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        """Write the manifest as JSON, atomically."""
        from .nn.params.nn_run import _atomic_write_text

        _atomic_write_text(os.fspath(path), self.to_json())

    @staticmethod
    def load(path: Union[str, os.PathLike[str]]) -> SplitManifest:
        with open(path, encoding="utf-8") as handle:
            return SplitManifest.from_json(handle.read())

    def resolve(self, ids: Iterable[Any], *, source: Union[IdentityRef, str, None] = None) -> SplitIndices:
        """Map this manifest onto a source whose rows carry ``ids``.

        ``ids`` are the source's sample ids in row order (``range(n)`` for a
        positional manifest). When the manifest records a source identity,
        ``source`` must match it; passing one to a manifest that records none
        raises too, since nothing could be checked. Raises
        :class:`SplitError` on a changed identity, a positional-only
        manifest, ids of another type, and duplicate, missing or unexpected
        ids, and on an empty train split. Excluded sample ids may be absent
        (they are in no split); a positional manifest needs every position,
        since a dropped row would shift the rest.
        """
        given = _source(source)
        if self.source.kind == "unknown" and given.kind != "unknown":
            raise SplitError(
                f"the plan records no source identity, so source identity {given.state()} cannot be checked; "
                "plan with source= (plan_split) to pin one"
            )
        if self.source.kind != "unknown" and given != self.source:
            raise SplitError(
                f"source identity changed: the plan records {self.source.state()}, got {given.state()}; "
                "pass the identity the plan was made with (resolve(source=...) / "
                "NNTabularDataset(source_identity=...))"
            )
        if self.ids == "position" and self.source.kind == "unknown":
            raise SplitError(
                "positional-only manifest: row positions cannot detect a reordered or changed source; plan with "
                "stable sample ids, or record a source identity (source=) and pass the same one when replaying"
            )
        if not self.train:
            raise SplitError("the split manifest has no training rows")
        rows = list(ids)  # a non-iterable is a caller bug: its TypeError propagates
        try:
            keys = _keys(rows, "source ids")
        except TypeError as exc:
            raise SplitError(f"{exc}; convert the id column to str or int (e.g. .astype('int64'))") from exc
        planned = [value for name in _MEMBERS for value in getattr(self, name)]
        if keys and planned and type(keys[0]) is not type(planned[0]):
            raise SplitError(
                f"the plan's sample ids are {type(planned[0]).__name__}, the source's are {type(keys[0]).__name__}"
            )
        position: dict[SampleId, int] = {}
        duplicates = []
        for index, key in enumerate(keys):
            if key in position:
                duplicates.append(key)
            position[key] = index
        if duplicates:
            raise SplitError(f"duplicate sample ids in the source: {_preview(sorted(set(duplicates)))}")
        required = _MEMBERS if self.ids == "position" else SPLITS
        missing = sorted(value for name in required for value in getattr(self, name) if value not in position)
        if missing:
            raise SplitError(f"{len(missing)} planned sample ids are missing from the source: {_preview(missing)}")
        known = set(planned)
        unexpected = [key for key in keys if key not in known]
        if unexpected:
            raise SplitError(f"{len(unexpected)} source sample ids are not in the plan: {_preview(unexpected)}")
        return SplitIndices(
            *(tuple(position[value] for value in getattr(self, name) if value in position) for name in _MEMBERS)
        )


# --- planning ---------------------------------------------------------------------------------


def _proportions(value: Any, strategy: str) -> tuple[float, float, float]:
    if value is None:
        raise TypeError(f"the {strategy} strategy needs proportions=(train, validation, test)")
    items = _sequence(value, 3, "proportions must be a (train, validation, test) triple")
    out = [
        require_finite_real(item, f"proportions[{name}]", owner="plan_split", minimum=0.0)
        for name, item in zip(SPLITS, items, strict=True)
    ]
    if out[0] <= 0:
        raise ValueError("proportions: the train proportion must be > 0")
    if not math.isclose(sum(out), 1.0, abs_tol=1e-9):
        raise ValueError(f"proportions must sum to 1, got {sum(out)!r}")
    return out[0], out[1], out[2]


def _group_members(
    keys: Sequence[SampleId], groups: Sequence[SampleId], proportions: Sequence[float], seed: int
) -> list[list[SampleId]]:
    rows: dict[SampleId, list[SampleId]] = {}
    for key, group in zip(keys, groups, strict=True):
        rows.setdefault(group, []).append(key)
    active = [index for index, share in enumerate(proportions) if share > 0]
    if len(rows) < len(active):
        raise SplitError(
            f"{len(rows)} groups cannot fill {len(active)} active splits; every active split needs a whole group"
        )
    rank = _ranker(seed, "group")
    order = sorted(rows, key=rank)
    target = [share * len(keys) for share in proportions]
    assigned: list[list[SampleId]] = [[], [], []]
    counts = [0, 0, 0]
    for group in order:
        split = max(active, key=lambda index: (target[index] - counts[index], -index))
        assigned[split].append(group)
        counts[split] += len(rows[group])
    for split in active:  # an active split never ends up empty
        if not assigned[split]:
            donor = max(
                (index for index in active if len(assigned[index]) > 1),
                key=lambda index: (len(assigned[index]), -index),
            )
            group = min(assigned[donor], key=lambda group: (len(rows[group]), rank(group)))
            assigned[donor].remove(group)
            assigned[split].append(group)
    return [[key for group in groups_ for key in rows[group]] for groups_ in assigned]


def _allocate(size: int, proportions: Sequence[float], active: Sequence[int], carry: list[float]) -> list[int]:
    """Largest-remainder counts for one class, carrying the rounding residual
    across classes. The one-row minimum per active split is applied after the
    carry is recorded, so it never accumulates into later classes."""
    desired = {index: size * proportions[index] + carry[index] for index in active}
    counts = {index: max(0, math.floor(desired[index])) for index in active}
    while sum(counts.values()) < size:
        counts[max(active, key=lambda index: (desired[index] - counts[index], -index))] += 1
    while sum(counts.values()) > size:
        surplus = min((i for i in active if counts[i] > 0), key=lambda index: (desired[index] - counts[index], index))
        counts[surplus] -= 1
    for index in active:
        carry[index] = min(1.0, max(-1.0, desired[index] - counts[index]))
    for index in active:  # every active split receives the class
        if counts[index] == 0:
            donor = max((i for i in active if counts[i] > 1), key=lambda i: (counts[i] - desired[i], -i))
            counts[donor] -= 1
            counts[index] += 1
    return [counts.get(index, 0) for index in range(3)]


def _stratified_members(
    keys: Sequence[SampleId],
    labels: Sequence[SampleId],
    proportions: Sequence[float],
    seed: int,
    insufficient: str,
) -> list[list[SampleId]]:
    by_class: dict[SampleId, list[SampleId]] = {}
    for key, label in zip(keys, labels, strict=True):
        by_class.setdefault(label, []).append(key)
    active = [index for index, share in enumerate(proportions) if share > 0]
    small = {label for label, rows in by_class.items() if len(rows) < len(active)}
    if small and insufficient == "raise":
        first, *others = sorted(small)
        more = f"; {len(others)} more classes are too small: {_preview(others)}" if others else ""
        raise SplitError(
            f"class {first!r} has {len(by_class[first])} rows but {len(active)} active splits need at least one "
            f"row each{more}. Pass insufficient='train' to keep such classes in training only, or use fewer "
            "active splits"
        )
    members: list[list[SampleId]] = [[], [], []]
    carry = [0.0, 0.0, 0.0]
    rank = _ranker(seed, "sample")
    for label in sorted(by_class):
        ranked = sorted(by_class[label], key=rank)
        if label in small:
            members[0].extend(ranked)
            continue
        counts = _allocate(len(ranked), proportions, active, carry)
        start = 0
        for index, count in enumerate(counts):
            members[index].extend(ranked[start : start + count])
            start += count
    return members


def _missing(value: Any) -> bool:
    """None, NaN, NaT or pandas' NA (whose comparisons cannot be truth-tested)."""
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return False
    try:
        return bool(value != value)
    except TypeError:
        return True


def _time(value: Any, what: str) -> Any:
    if _missing(value):
        raise ValueError(f"{what} is missing ({value!r})")
    if isinstance(value, bool):
        raise TypeError(f"{what} must be a number or a date / datetime, not bool")
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"{what} is not finite ({value!r})")
        return int(value) if isinstance(value, numbers.Integral) else float(value)
    if isinstance(value, np.datetime64):  # e.g. df["day"].to_numpy()
        import pandas as pd

        return pd.Timestamp(value)
    if isinstance(value, date):
        return value
    raise TypeError(f"{what} must be a number or a date / datetime, got {type(value).__name__} {value!r}")


def _time_kind(value: Any) -> str:
    if isinstance(value, datetime):
        return "tz-aware datetime" if value.utcoffset() is not None else "naive datetime"
    return "date" if isinstance(value, date) else "number"


def _time_parameter(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, timedelta):
        return {"seconds": value.total_seconds()}
    return value


def _chronological_members(
    keys: Sequence[SampleId], times: Sequence[Any], cutoffs: Any, gap: Any
) -> tuple[list[list[SampleId]], list[SampleId], dict[str, Any]]:
    values = [_time(value, f"times[{index}]") for index, value in enumerate(times)]
    kinds = sorted({_time_kind(value) for value in values})
    if len(kinds) > 1:
        raise TypeError(f"times must share one kind (numbers, dates, naive or tz-aware datetimes), got {kinds}")
    kind = kinds[0] if kinds else "number"
    is_temporal = kind != "number"
    pair = _sequence(cutoffs, 2, "cutoffs must be a (validation_start, test_start) pair")
    bounds = [None if value is None else _time(value, f"cutoffs[{index}]") for index, value in enumerate(pair)]
    if bounds == [None, None]:
        raise ValueError("cutoffs: at least one of (validation_start, test_start) must be set")
    for index, value in enumerate(bounds):
        if value is not None and _time_kind(value) != kind:
            raise TypeError(f"cutoffs[{index}] is a {_time_kind(value)} but times are {kind} values")
    start_val, start_test = bounds
    if start_val is not None and start_test is not None and start_val > start_test:
        raise ValueError(f"cutoffs must be ordered (validation_start <= test_start), got {cutoffs!r}")
    if gap is None:
        gap = timedelta(0) if is_temporal else 0
    if isinstance(gap, np.timedelta64):
        import pandas as pd

        gap = pd.Timedelta(gap)
    if is_temporal:
        if not isinstance(gap, timedelta):
            raise TypeError(f"gap must be a datetime.timedelta for date / datetime times, got {gap!r}")
        if gap < timedelta(0):
            raise ValueError(f"gap must be >= 0, got {gap!r}")
        if kind == "date" and gap % timedelta(days=1):
            raise ValueError(f"gap must be whole days for date times (date arithmetic drops the rest), got {gap!r}")
    else:
        real = require_finite_real(gap, "gap", owner="plan_split", minimum=0.0)
        gap = int(gap) if isinstance(gap, numbers.Integral) else real
    members: list[list[SampleId]] = [[], [], []]
    excluded: list[SampleId] = []
    for key, value in zip(keys, values, strict=True):
        if start_test is not None and value >= start_test:
            split, boundary = 2, None
        elif start_val is not None and value >= start_val:
            split, boundary = 1, start_test
        else:
            split, boundary = 0, start_val if start_val is not None else start_test
        if boundary is not None and value >= boundary - gap:
            excluded.append(key)
        else:
            members[split].append(key)
    parameters = {"cutoffs": [_time_parameter(value) for value in bounds], "gap": _time_parameter(gap)}
    return members, excluded, parameters


_USES = {
    "group": {"groups", "proportions", "seed"},
    "chronological": {"times", "cutoffs", "gap"},
    "stratified": {"labels", "proportions", "seed", "insufficient"},
}


def plan_split(
    ids: Optional[Iterable[Any]] = None,
    *,
    strategy: str,
    groups: Optional[Iterable[Any]] = None,
    times: Optional[Iterable[Any]] = None,
    labels: Optional[Iterable[Any]] = None,
    proportions: Optional[Iterable[float]] = None,
    cutoffs: Optional[Iterable[Any]] = None,
    gap: Any = None,
    seed: Optional[int] = None,
    insufficient: Optional[str] = None,
    source: Union[IdentityRef, str, None] = None,
) -> SplitManifest:
    """Plan a group, chronological or stratified split.

    Args:
        ids: stable sample ids (str or int), one per row. ``None`` plans on
            row positions, which needs ``source`` (row positions alone cannot
            detect a reordered source) and replays only against it.
        strategy: ``"group"``, ``"chronological"`` or ``"stratified"``.
        groups: per-row group keys (``"group"``).
        times: per-row numbers or dates / datetimes (``"chronological"``).
        labels: per-row class labels (``"stratified"``).
        proportions: ``(train, validation, test)`` shares summing to 1, train
            > 0; a zero share leaves that split empty (``"group"``,
            ``"stratified"``).
        cutoffs: ``(validation_start, test_start)``; either may be ``None``
            (``"chronological"``).
        gap: rows within ``gap`` before a cutoff are excluded; a number, or a
            ``timedelta`` for dates (``"chronological"``, default 0).
        seed: orders groups / ids (``"group"``, ``"stratified"``); ``None``
            draws one from the OS and records it. Chronological plans take
            no seed.
        insufficient: ``"raise"`` (default) or ``"train"`` for a class with
            fewer rows than active splits (``"stratified"``).
        source: the source's identity — an ``IdentityRef`` (e.g.
            ``nnx.provenance.hash_file``), a declared id string, or ``None``.

    Raises:
        SplitError: duplicate ids, too few groups or class rows, or no
            training rows. ``TypeError`` / ``ValueError`` for malformed
            arguments, including ones the strategy does not use.
    """
    if strategy not in _PLANNED:
        raise ValueError(f"strategy must be one of {_PLANNED}, got {strategy!r}")
    given = {
        "groups": groups,
        "times": times,
        "labels": labels,
        "proportions": proportions,
        "cutoffs": cutoffs,
        "gap": gap,
        "seed": seed,
        "insufficient": insufficient,
    }
    if strategy == "chronological" and seed is not None:
        raise ValueError("chronological splits are deterministic and take no seed")
    for name, value in given.items():
        if value is not None and name not in _USES[strategy]:
            raise TypeError(f"{name} is not used by the {strategy} strategy")
    per_row_name = {"group": "groups", "chronological": "times", "stratified": "labels"}[strategy]
    per_row = given[per_row_name]
    if per_row is None:
        raise TypeError(f"the {strategy} strategy needs {per_row_name}=")
    per_row = list(per_row)
    identity = _source(source)
    if ids is None:
        if identity.kind == "unknown":
            raise SplitError(
                "a positional plan (ids=None) needs source=: row positions alone cannot detect a reordered or "
                "changed source; pass stable sample ids, or a source identity such as hash_file(path)"
            )
        keys: tuple[SampleId, ...] = tuple(range(len(per_row)))
        kind = "position"
    else:
        keys = _keys(ids, "ids")
        kind = "sample"
        if len(keys) != len(per_row):
            raise ValueError(f"{per_row_name} has length {len(per_row)} but ids has length {len(keys)}")
    if len(set(keys)) != len(keys):
        seen: set[SampleId] = set()
        duplicates = sorted({key for key in keys if key in seen or seen.add(key)})
        raise SplitError(f"duplicate sample ids: {_preview(duplicates)}")

    excluded: list[SampleId] = []
    if strategy == "chronological":
        members, excluded, parameters = _chronological_members(keys, per_row, cutoffs, gap)
        effective_seed = None
    else:
        shares = _proportions(proportions, strategy)
        effective_seed = require_count(seed, "seed", owner="plan_split") if seed is not None else secrets.randbits(32)
        if strategy == "group":
            members = _group_members(keys, _keys(per_row, "groups"), shares, effective_seed)
            parameters = {"proportions": list(shares)}
        else:
            policy = "raise" if insufficient is None else insufficient
            if policy not in _INSUFFICIENT:
                raise ValueError(f"insufficient must be one of {_INSUFFICIENT}, got {policy!r}")
            members = _stratified_members(keys, _keys(per_row, "labels"), shares, effective_seed, policy)
            parameters = {"proportions": list(shares), "insufficient": policy}
    if not members[0]:
        raise SplitError(f"the {strategy} plan has no training rows")
    return SplitManifest(
        strategy=strategy,
        train=tuple(members[0]),
        validation=tuple(members[1]),
        test=tuple(members[2]),
        excluded=tuple(excluded),
        ids=kind,
        source=identity,
        parameters=parameters,
        seed=effective_seed,
    )
