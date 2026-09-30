"""Abstention policies and risk-coverage evaluation (FEAT-008).

A classifier that must answer every row cannot decline an uncertain one.
An :class:`AbstentionPolicy` accepts a row only when its score reaches a
threshold and **abstains** otherwise:

- ``"max_probability"`` — the score is the top class probability; an
  abstention's reason is ``"below_probability_threshold"``;
- ``"margin"`` — the score is the gap between the top two probabilities (a
  tied top two gives 0); the reason is ``"small_margin"``.

Policies are pure functions over categorical probabilities ``[N, C]``
(``C >= 2``) with stable sample ids and ordered labels. For bare
probabilities the prediction is the first class of maximal probability
(ties break by label order); a ``PredictionResult`` or
``CalibratedPrediction`` keeps its own ``decoded`` class, which must be a
class of maximal probability. The prediction is kept for accepted and
abstained rows alike. A row is accepted when its
score is **at or above** the threshold, so a threshold of 1.0 still accepts
a score of exactly 1. Scores are float64: a margin of ``0.7 - 0.2`` is
``0.49999999999999994`` (the binary values differ by slightly less than
0.5), so a threshold of exactly 0.5 abstains on it; thresholds from
:func:`select_threshold` are actual scores. Malformed probabilities raise
:class:`AbstentionError` — they are never silently abstained on.

Every outcome is one of three kinds:

- **accepted** — the prediction stands;
- **abstained** — the policy declined the row (a success, with its reason);
- **invalid** — the input was malformed or of another schema (an error).

A policy records the probability field it reads (``"probabilities"`` or a
calibrator's ``"calibrated_probabilities"``), the threshold, the label
order, the model id, the calibrator id and the tuning split id, and
serializes as JSON (``nnx.abstention/1``). Applying it to input of another
schema raises :class:`AbstentionSchemaError` before any probability is read.

**Reports** use two denominators: ``coverage = accepted / total`` over every
row, and the selective ``risk = incorrect_accepted / accepted`` over the
accepted rows only. With nothing accepted, risk is **unavailable** (status
``"no_accepted"``), never 0. :func:`select_threshold` tunes a threshold on a
named validation split — never the test split — against a risk ceiling that
is **empirical** (measured on that split, not guaranteed elsewhere). The
risk-coverage curve ends in an **accept-none** endpoint (coverage 0, risk
unavailable) that exists only on the curve: it is not a deployable
threshold, and a policy cannot hold or load it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Optional, cast

import numpy as np

from ._artifacts import JsonArtifact, check_keys, override_record, parse_json, read_text
from ._config import _thaw_config
from ._probability import (
    categorical_probabilities,
    categorical_probabilities_eps,
    class_labels,
    class_targets,
    is_prediction,
    is_tensor,
    prediction_labels,
    read_array,
    rounding_tolerance,
    row_ids,
    same_ids,
)
from ._validation import checked, require_count, require_finite_real, required_id

__all__ = [
    "FORMAT",
    "INPUT_FIELDS",
    "MAX_DEFAULT_CANDIDATES",
    "POLICY_KINDS",
    "REASONS",
    "AbstentionError",
    "AbstentionPolicy",
    "AbstentionResult",
    "AbstentionSchemaError",
    "AcceptedRows",
    "CoverageAccumulator",
    "CoverageReport",
    "CurvePoint",
    "Outcome",
    "SelectiveDecision",
    "ThresholdSelection",
    "decide",
    "risk_coverage_curve",
    "scores",
    "select_threshold",
]

FORMAT = "nnx.abstention/1"
"""Format version of a serialized policy, report or selection."""

POLICY_KINDS = ("max_probability", "margin")
REASONS: Mapping[str, str] = MappingProxyType(
    {"max_probability": "below_probability_threshold", "margin": "small_margin"}
)
"""The reason an abstained row carries, per policy kind (read-only)."""

MAX_DEFAULT_CANDIDATES = 1001
"""The most default candidate thresholds: every distinct score up to this
many, else this many evenly spaced order statistics of the scores (coverage
steps of about 0.1%). Pass ``candidates=`` for any other set."""

INPUT_FIELDS = ("probabilities", "calibrated_probabilities")
"""The probability fields a policy can read: a prediction's raw
probabilities, or a calibrator's calibrated ones."""


class AbstentionError(ValueError):
    """Malformed probabilities, targets, configuration or serialized state."""


class AbstentionSchemaError(AbstentionError):
    """A declared schema — labels, model id, probability field, calibrator
    or sample ids — that contradicts the policy's, the prediction's own, or
    itself (calibrated probabilities without their calibrator, say)."""


# --- validation ----------------------------------------------------------------------------------


def _id(value: Any, what: str) -> str:
    return required_id(value, what, error=AbstentionError)


def _labels(labels: Any) -> tuple[str, ...]:
    return class_labels(labels, error=AbstentionError, minimum=2)


def _kind(kind: Any) -> str:
    if kind not in POLICY_KINDS:
        raise AbstentionError(f"policy kind must be one of {POLICY_KINDS}, got {kind!r}")
    return kind


def _checked(check: Any, value: Any, what: str, **domain: Any) -> Any:
    """A shared ``nnx._validation`` check, raising :class:`AbstentionError`."""
    return checked(check, value, what, owner="nnx.abstention", error=AbstentionError, **domain)


def _unit(value: Any, what: str, **domain: Any) -> float:
    """A finite number in ``[0, 1]``, with -0.0 as 0.0 (one value, one state)."""
    return _checked(require_finite_real, value, what, minimum=0.0, maximum=1.0, **domain) + 0.0


def _threshold(value: Any) -> float:
    if value is None:
        raise AbstentionError(
            "threshold None is the accept-none endpoint of a risk-coverage curve, not a deployable threshold; "
            "a threshold is a finite number in [0, 1] (1.0 still accepts a score of exactly 1)"
        )
    return _unit(
        value,
        "threshold",
        domain_message=f"threshold must lie in [0, 1], got {value!r} (1.0 still accepts a score of exactly 1)",
    )


def _field(value: Any) -> str:
    if value not in INPUT_FIELDS:
        raise AbstentionError(f"input_field must be one of {INPUT_FIELDS}, got {value!r}")
    return value


def _calibrator(
    input_field: str, calibrator_id: Any, *, error: type[AbstentionError] = AbstentionError, hint: str = ""
) -> Optional[str]:
    """The calibrator id a field carries: required for calibrated
    probabilities, absent for raw ones (``error`` otherwise; a malformed id
    is :class:`AbstentionError`)."""
    if input_field == "calibrated_probabilities":
        if calibrator_id is None:
            raise error(
                f"calibrated_probabilities{hint} must declare calibrator_id=... naming their calibrator "
                "(input_field='probabilities' reads raw ones)"
            )
        return _id(calibrator_id, "calibrator_id")
    if calibrator_id is not None:
        raise error(
            f"calibrator_id {calibrator_id!r} given for raw probabilities; raw probabilities have no calibrator"
        )
    return None


def _count(value: Any, what: str) -> int:
    return _checked(require_count, value, what, minimum=0)


def _ids(values: Any) -> np.ndarray:
    """Read-only int64 sample ids."""
    ids = row_ids(values, None, error=AbstentionError)
    ids.setflags(write=False)
    return ids


def _strict_keys(state: Mapping[str, Any], known: set[str], what: str) -> None:
    check_keys(state, required=known, what=what, error=AbstentionError)


def _check_tuning_split(split_id: Any, test_split_id: Any) -> str:
    tuning = _id(split_id, "split_id")
    if test_split_id is not None and _id(test_split_id, "test_split_id") == tuning:
        raise AbstentionError(f"the tuning split {tuning!r} is the test split; tune on a separate validation split")
    return tuning


def _within(part: np.ndarray, whole: np.ndarray) -> bool:
    """Whether ``part`` is a sub-multiset of ``whole`` (ids may repeat)."""
    ids, counts = np.unique(part, return_counts=True)
    pool, available = np.unique(whole, return_counts=True)
    at = np.searchsorted(pool, ids)
    if (at >= pool.size).any():  # an id above every one in the pool
        return False
    return bool((pool[at] == ids).all() and (available[at] >= counts).all())


def _agrees(stored: Sequence[Any], derived: Sequence[Any]) -> bool:
    """Whether stored JSON values equal derived ones: numbers by value (a
    JSON writer may print 1.0 as 1), but a boolean only equals a boolean
    (``True == 1`` in Python)."""
    return all(
        (s is d) if isinstance(s, bool) or isinstance(d, bool) else s == d for s, d in zip(stored, derived, strict=True)
    )


def _artifact(state: Any, artifact: str) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        raise AbstentionError(f"a {artifact} state is a mapping, got {type(state).__name__}")
    if state.get("format") != FORMAT:
        raise AbstentionError(f"unsupported abstention format {state.get('format')!r}; expected {FORMAT!r}")
    if state.get("artifact") != artifact:
        raise AbstentionError(f"not an abstention {artifact}: artifact is {state.get('artifact')!r}")
    return state


# --- scoring -------------------------------------------------------------------------------------


def _scores(p: np.ndarray, kind: str, decoded: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """``(prediction, score)``. ``decoded`` — a prediction's own argmax of its
    raw logits — wins over the probabilities' argmax, which rounding can
    turn into a tie; without it, the first maximum (label order) wins."""
    rows = np.arange(p.shape[0])
    prediction = p.argmax(axis=1).astype(np.int64) if decoded is None else decoded.astype(np.int64, copy=False)
    top = p[rows, prediction]
    if kind == "max_probability":
        return prediction, top  # fancy indexing already made a new array
    second = np.partition(p, -2, axis=1)[:, -2] if p.shape[0] else top
    # A tied top two gives exactly 0; so does a decoded class tied with the
    # maximum only up to rounding (``_read`` bounds that gap), never below 0.
    return prediction, np.maximum(top - second, 0.0)


def scores(probabilities: Any, *, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """``(prediction, score)`` per row of categorical ``(N, C)``
    probabilities: the first class of maximal probability and the policy
    score (the top probability, or the top-two margin). A prediction's own
    ``decoded`` class is kept by :meth:`AbstentionPolicy.apply` and
    :func:`select_threshold` instead."""
    policy_kind = _kind(kind)  # before any probability is read
    return _scores(categorical_probabilities(probabilities, error=AbstentionError, allow_empty=True), policy_kind)


# --- declared inputs -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Declared:
    probabilities: Any
    labels: tuple[str, ...]
    model_id: str
    input_field: str
    calibrator_id: Optional[str]
    sample_ids: Any
    decoded: Optional[np.ndarray] = None
    override: Optional[Mapping[str, Any]] = None


def _declared(
    source: Any,
    *,
    labels: Any,
    model_id: Any,
    sample_ids: Any,
    input_field: Any,
    calibrator_id: Any,
    default_field: Optional[str],
) -> _Declared:
    """What ``source`` declares — without reading its probabilities.

    A ``CalibratedPrediction`` offers both views: ``input_field`` (or
    ``default_field``) picks one, its calibrator id follows, and keywords
    that contradict the prediction are refused. A categorical
    ``PredictionResult`` offers raw probabilities only. Arrays declare
    everything through the keywords."""
    from .calibration import CalibratedPrediction

    if isinstance(source, CalibratedPrediction):
        chosen = input_field if input_field is not None else default_field
        if chosen is None:
            raise AbstentionError(
                "a CalibratedPrediction holds raw and calibrated probabilities; pass input_field="
                "'probabilities' or 'calibrated_probabilities'"
            )
        field_name = _field(chosen)
        calibrated = field_name == "calibrated_probabilities"
        declared = _Declared(
            probabilities=getattr(source, field_name),
            labels=_labels(source.labels),
            model_id=_id(source.model_id, "model_id"),
            input_field=field_name,
            calibrator_id=source.calibrator_id if calibrated else None,
            sample_ids=source.sample_ids,
            decoded=np.asarray(source.decoded),
            override=override_record(source.override, error=AbstentionError, what="a calibration override")
            if calibrated
            else None,
        )
        # Malformed keywords are invalid input, not a contradiction.
        given_model = None if model_id is None else _id(model_id, "model_id")
        given_calibrator = None if calibrator_id is None else _id(calibrator_id, "calibrator_id")
        conflicts: list[str] = [
            name
            for name, given, own in (
                ("labels", None if labels is None else _labels(labels), declared.labels),
                ("model_id", given_model, declared.model_id),
                ("calibrator_id", given_calibrator, declared.calibrator_id),
            )
            if given is not None and given != own
        ]
        if not same_ids(sample_ids, declared.sample_ids, error=AbstentionError):
            conflicts.append("sample_ids")
        if conflicts:
            raise AbstentionSchemaError(f"{conflicts} contradict the calibrated prediction's own values")
        return declared
    if is_prediction(source):
        resolved = prediction_labels(
            source, labels, error=AbstentionError, conflict_error=AbstentionSchemaError, what="abstention", minimum=2
        )
        if resolved is None:
            raise AbstentionError("pass labels=[...] naming the prediction's columns in order")
        _require_prediction_arrays(source)
        if model_id is None:
            raise AbstentionError("pass model_id=... naming the model that made the prediction")
        if input_field is not None:
            _field(input_field)  # a typo is invalid input, not another schema
        if input_field not in (None, "probabilities") or calibrator_id is not None:
            raise AbstentionSchemaError(
                "a PredictionResult holds raw probabilities only: its input_field is 'probabilities' with no "
                "calibrator; calibrate it first (nnx.calibration) for calibrated_probabilities"
            )
        if not same_ids(sample_ids, source.sample_ids, error=AbstentionError):
            raise AbstentionSchemaError("sample_ids= contradicts the prediction's own sample_ids")
        return _Declared(
            probabilities=source.probabilities,
            labels=resolved,
            model_id=_id(model_id, "model_id"),
            input_field="probabilities",
            calibrator_id=None,
            sample_ids=source.sample_ids,
            decoded=np.asarray(source.decoded),
        )
    if labels is None or model_id is None:
        raise AbstentionError("for bare probabilities pass labels=[...] and model_id=... to declare their schema")
    field_name = _field(input_field if input_field is not None else default_field or "probabilities")
    implied = "" if input_field is not None else " (the policy's field, as input_field= was not given)"
    # A field and calibrator that do not belong together are another schema.
    declared_calibrator = _calibrator(field_name, calibrator_id, error=AbstentionSchemaError, hint=implied)
    return _Declared(
        probabilities=source,
        labels=_labels(labels),
        model_id=_id(model_id, "model_id"),
        input_field=field_name,
        calibrator_id=declared_calibrator,
        sample_ids=sample_ids,
    )


def _require_prediction_arrays(source: Any) -> None:
    missing = [name for name in ("probabilities", "decoded") if not hasattr(source, name)]
    if missing:
        raise AbstentionError(f"a prediction needs {' and '.join(missing)}, as a PredictionResult has")


def _check_width(labels: tuple[str, ...], p: np.ndarray) -> None:
    if p.shape[1] != len(labels):
        raise AbstentionError(f"labels must name one label per class: {len(labels)} labels for {p.shape[1]} classes")


def _decoded(decoded: Optional[np.ndarray], p: np.ndarray, tolerance: float) -> Optional[np.ndarray]:
    """A prediction's decoded classes as int64 — integer, in range and a
    class of maximal probability in every row (within ``tolerance``, one
    entry's rounding in the probabilities' own dtype), so the score and
    margin describe the class that is kept."""
    if decoded is None:
        return None
    if decoded.shape != (p.shape[0],) or not np.issubdtype(decoded.dtype, np.integer):
        raise AbstentionError(
            f"the prediction's decoded classes must be {p.shape[0]} integers, got {decoded.dtype} of shape "
            f"{decoded.shape}"
        )
    if decoded.size and (decoded.min() < 0 or decoded.max() >= p.shape[1]):
        raise AbstentionError(f"the prediction's decoded classes must lie in [0, {p.shape[1]})")
    classes = decoded.astype(np.int64)
    if p.shape[0] and bool((p[np.arange(p.shape[0]), classes] < p.max(axis=1) - tolerance).any()):
        raise AbstentionError("the prediction's decoded class is not a class of maximal probability in every row")
    return classes


def _read(declared: _Declared, *, copy: bool) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """``(probabilities, sample_ids, decoded)``, validated. Ids may repeat,
    as a ``PredictionResult``'s may (an oversampled split, say): a policy
    decides each row on its own. Malformed ids fail before the
    probabilities are read; their count is checked after."""
    given = None if declared.sample_ids is None else row_ids(declared.sample_ids, None, error=AbstentionError)
    p, eps = categorical_probabilities_eps(declared.probabilities, error=AbstentionError, allow_empty=True, copy=copy)
    _check_width(declared.labels, p)
    decoded = _decoded(declared.decoded, p, rounding_tolerance(eps, floor=0.0))  # one entry's rounding, its dtype
    ids = row_ids(given, p.shape[0], error=AbstentionError)  # 0..N-1 by default; the count checked here
    return p, ids, decoded


# --- the policy ----------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AbstentionPolicy(JsonArtifact):
    """Accept a row when its score reaches ``threshold``; abstain otherwise.

    Args:
        kind: ``"max_probability"`` or ``"margin"``.
        threshold: a finite number in ``[0, 1]``; a row is accepted when its
            score is at or above it.
        labels: the ordered class names the probabilities' columns carry.
        model_id: the model the probabilities must come from.
        tuning_split_id: the validation split the threshold was chosen on.
        input_field: ``"probabilities"`` (raw) or
            ``"calibrated_probabilities"``.
        calibrator_id: the calibrator of a calibrated field (required
            then, and ``None`` for raw probabilities).

    Equality, hashing and :attr:`id` follow the canonical state.
    """

    kind: str
    threshold: float
    labels: tuple[str, ...]
    model_id: str
    tuning_split_id: str
    input_field: str = "probabilities"
    calibrator_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _kind(self.kind))
        object.__setattr__(self, "threshold", _threshold(self.threshold))
        object.__setattr__(self, "labels", _labels(self.labels))
        object.__setattr__(self, "model_id", _id(self.model_id, "model_id"))
        object.__setattr__(self, "tuning_split_id", _id(self.tuning_split_id, "tuning_split_id"))
        object.__setattr__(self, "input_field", _field(self.input_field))
        object.__setattr__(self, "calibrator_id", _calibrator(self.input_field, self.calibrator_id))

    @property
    def id(self) -> str:
        """``sha256:<hex>`` of the canonical state."""
        return self._digest

    @property
    def reason(self) -> str:
        """The reason an abstained row carries."""
        return REASONS[self.kind]

    def _check(self, labels: tuple[str, ...], model_id: str, input_field: str, calibrator_id: Optional[str]) -> None:
        mismatches = [
            f"{name} {got!r} differs from the policy's {want!r}"
            for name, got, want in (
                ("labels", list(labels), list(self.labels)),
                ("model_id", model_id, self.model_id),
                ("input_field", input_field, self.input_field),
                ("calibrator_id", calibrator_id, self.calibrator_id),
            )
            if got != want
        ]
        if mismatches:
            raise AbstentionSchemaError("incompatible schema: " + "; ".join(mismatches))

    def apply(
        self,
        source: Any,
        *,
        labels: Optional[Sequence[str]] = None,
        model_id: Optional[str] = None,
        sample_ids: Any = None,
        input_field: Optional[str] = None,
        calibrator_id: Optional[str] = None,
    ) -> AbstentionResult:
        """Accept or abstain on every row of ``source``.

        ``source`` is a ``CalibratedPrediction`` (the policy's
        ``input_field`` picks its raw or calibrated view), a categorical
        ``PredictionResult`` (raw; pass ``model_id=``), or ``(N, C)``
        probabilities with ``labels=`` / ``model_id=`` (and
        ``input_field=`` / ``calibrator_id=`` for calibrated ones). The
        declared schema must equal the policy's — otherwise
        :class:`AbstentionSchemaError`, before any probability is read.
        Malformed probabilities raise :class:`AbstentionError`. The input is
        never modified.
        """
        declared = _declared(
            source,
            labels=labels,
            model_id=model_id,
            sample_ids=sample_ids,
            input_field=input_field,
            calibrator_id=calibrator_id,
            default_field=self.input_field,
        )
        self._check(declared.labels, declared.model_id, declared.input_field, declared.calibrator_id)
        p, ids, decoded = _read(declared, copy=True)  # the result owns its probabilities
        prediction, score = _scores(p, self.kind, decoded)
        for array in (p, ids, prediction, score):
            array.setflags(write=False)
        accepted = score >= self.threshold
        accepted.setflags(write=False)
        return AbstentionResult(
            policy_id=self.id,
            kind=self.kind,
            threshold=self.threshold,
            input_field=self.input_field,
            calibrator_id=self.calibrator_id,
            model_id=self.model_id,
            labels=self.labels,
            sample_ids=ids,
            probabilities=p,
            prediction=prediction,
            score=score,
            accepted=accepted,
            calibration_override=declared.override,
        )

    def state(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "artifact": "policy",
            "kind": self.kind,
            "threshold": self.threshold,
            "labels": list(self.labels),
            "model_id": self.model_id,
            "tuning_split_id": self.tuning_split_id,
            "input_field": self.input_field,
            "calibrator_id": self.calibrator_id,
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> AbstentionPolicy:
        """Rebuild from :meth:`state`; anything malformed — including a
        non-deployable threshold such as a curve's accept-none endpoint —
        raises :class:`AbstentionError`."""
        state = _artifact(state, "policy")
        _strict_keys(
            state,
            {"format", "artifact", "kind", "threshold", "labels", "model_id", "tuning_split_id", "input_field"}
            | {"calibrator_id"},
            "policy",
        )
        return AbstentionPolicy(
            kind=state["kind"],
            threshold=state["threshold"],
            labels=state["labels"],
            model_id=state["model_id"],
            tuning_split_id=state["tuning_split_id"],
            input_field=state["input_field"],
            calibrator_id=state["calibrator_id"],
        )

    @staticmethod
    def from_json(text: str) -> AbstentionPolicy:
        return AbstentionPolicy.from_state(parse_json(text, "abstention policy", AbstentionError))

    @staticmethod
    def load(path: Any) -> AbstentionPolicy:
        return AbstentionPolicy.from_json(read_text(path, "abstention policy", AbstentionError))


# --- results -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """One row: ``status`` ``"accepted"`` or ``"abstained"`` (with its
    ``reason``); the ``prediction`` (label) and its index, the ordered
    ``distribution``, the ``score`` and the ``sample_id`` are kept either
    way."""

    sample_id: Optional[int]
    status: str
    prediction: str
    prediction_index: int
    distribution: tuple[tuple[str, float], ...]
    score: float
    reason: Optional[str]

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


@dataclass(frozen=True, eq=False)
class AcceptedRows:
    """The accepted rows only — what a hard-label report (accuracy, a
    confusion matrix, ``VisUtils.classification_report``) may receive. An
    abstention is never turned into a pseudo-class."""

    sample_ids: np.ndarray
    targets: np.ndarray
    predictions: np.ndarray


@dataclass(frozen=True, eq=False)
class AbstentionResult:
    """A policy applied to ``N`` rows.

    ``probabilities`` is a float64 copy of the field the policy read;
    ``prediction`` the first class of maximal probability (a prediction's
    own ``decoded`` class, for a prediction); ``score`` the
    policy score; ``accepted`` whether the row was accepted. The policy's
    identity (``policy_id``, ``kind``, ``threshold``, ``input_field``,
    ``calibrator_id``, ``model_id``, ``labels``) travels with the result, and
    so does ``calibration_override``: the named override under which a
    calibrated prediction was produced, if any (``nnx.calibration``).
    """

    policy_id: str
    kind: str
    threshold: float
    input_field: str
    calibrator_id: Optional[str]
    model_id: str
    labels: tuple[str, ...]
    sample_ids: np.ndarray
    probabilities: np.ndarray
    prediction: np.ndarray
    score: np.ndarray
    accepted: np.ndarray
    calibration_override: Optional[Mapping[str, Any]] = None

    @property
    def abstained(self) -> np.ndarray:
        return ~self.accepted

    @property
    def reasons(self) -> tuple[Optional[str], ...]:
        reason = REASONS[self.kind]
        return tuple(None if accepted else reason for accepted in self.accepted.tolist())

    @property
    def accepted_ids(self) -> np.ndarray:
        return self.sample_ids[self.accepted]

    @property
    def abstained_ids(self) -> np.ndarray:
        return self.sample_ids[~self.accepted]

    def outcomes(self) -> tuple[Outcome, ...]:
        reason = REASONS[self.kind]
        return tuple(
            Outcome(
                sample_id=int(sample_id),
                status="accepted" if accepted else "abstained",
                prediction=self.labels[index],
                prediction_index=int(index),
                distribution=tuple(zip(self.labels, row, strict=True)),
                score=float(score),
                reason=None if accepted else reason,
            )
            for sample_id, accepted, index, row, score in zip(
                self.sample_ids.tolist(),
                self.accepted.tolist(),
                self.prediction.tolist(),
                self.probabilities.tolist(),
                self.score.tolist(),
                strict=True,
            )
        )

    def _targets(self, targets: Any) -> np.ndarray:
        return class_targets(targets, self.prediction.shape[0], len(self.labels), error=AbstentionError)

    def accepted_rows(self, targets: Any) -> AcceptedRows:
        """Sample ids, targets and predictions of the accepted rows only."""
        y = self._targets(targets)
        return AcceptedRows(
            sample_ids=self.sample_ids[self.accepted].copy(),
            targets=y[self.accepted],
            predictions=self.prediction[self.accepted].copy(),
        )

    def report(self, targets: Any) -> CoverageReport:
        """Coverage and selective risk against integer class ``targets``."""
        y = self._targets(targets)
        wrong = self.accepted & (self.prediction != y)
        return CoverageReport(
            policy_id=self.policy_id,
            total=int(self.accepted.shape[0]),
            accepted=int(np.count_nonzero(self.accepted)),
            incorrect=int(np.count_nonzero(wrong)),
            sample_ids=self.sample_ids,
            accepted_ids=self.accepted_ids,
        )


# --- reports -------------------------------------------------------------------------------------


def _coverage(accepted: int, total: int) -> Optional[float]:
    return accepted / total if total else None


def _risk(incorrect: int, accepted: int) -> Optional[float]:
    return incorrect / accepted if accepted else None


def _status(accepted: int, total: int) -> str:
    return "empty" if total == 0 else "no_accepted" if accepted == 0 else "ok"


def _set_counts(counted: Any, what: str) -> None:
    """Validate a frozen dataclass's ``accepted`` / ``incorrect`` / ``total``
    as counts with ``incorrect <= accepted <= total``."""
    for name in ("accepted", "incorrect", "total"):
        object.__setattr__(counted, name, _count(getattr(counted, name), name))
    if not counted.incorrect <= counted.accepted <= counted.total:
        raise AbstentionError(
            f"a {what} needs incorrect <= accepted <= total, got {counted.incorrect} / {counted.accepted} / "
            f"{counted.total}"
        )


class _Counted:
    """Coverage over every row and selective risk over the accepted ones."""

    accepted: int
    incorrect: int
    total: int

    @property
    def coverage(self) -> Optional[float]:
        return _coverage(self.accepted, self.total)

    @property
    def risk(self) -> Optional[float]:
        return _risk(self.incorrect, self.accepted)


@dataclass(frozen=True, eq=False)
class CoverageReport(_Counted, JsonArtifact):
    """Coverage over every row and selective risk over the accepted rows.

    ``coverage = accepted / total`` and ``risk = incorrect / accepted``.
    With nothing accepted, ``risk`` is ``None`` and ``status`` is
    ``"no_accepted"`` — never a risk of 0; with no rows at all both are
    ``None`` and ``status`` is ``"empty"``. ``sample_ids`` keeps every
    original id and ``accepted_ids`` the accepted ones, in row order, as
    read-only int64 arrays.
    """

    policy_id: Optional[str]
    total: int
    accepted: int
    incorrect: int
    sample_ids: np.ndarray
    accepted_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.policy_id is not None:
            _id(self.policy_id, "policy_id")
        _set_counts(self, "report")
        object.__setattr__(self, "sample_ids", _ids(self.sample_ids))
        object.__setattr__(self, "accepted_ids", _ids(self.accepted_ids))
        if len(self.sample_ids) != self.total or len(self.accepted_ids) != self.accepted:
            raise AbstentionError("a report lists one sample id per row and one accepted id per accepted row")
        if not _within(self.accepted_ids, self.sample_ids):
            raise AbstentionError("a report's accepted ids must be among its sample ids, each at most as often")

    @property
    def status(self) -> str:
        return _status(self.accepted, self.total)

    def summary(self) -> str:
        if self.status == "empty":
            return "no rows: coverage and risk unavailable"
        coverage = f"coverage {self.accepted}/{self.total} = {self.coverage:.4g}"
        if self.status == "no_accepted":
            return f"{coverage}; risk unavailable (no accepted rows)"
        return f"{coverage}; selective risk {self.incorrect}/{self.accepted} = {self.risk:.4g}"

    def state(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "artifact": "coverage_report",
            "policy_id": self.policy_id,
            "total": self.total,
            "accepted": self.accepted,
            "incorrect": self.incorrect,
            "coverage": self.coverage,
            "risk": self.risk,
            "status": self.status,
            "sample_ids": self.sample_ids.tolist(),
            "accepted_ids": self.accepted_ids.tolist(),
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> CoverageReport:
        state = _artifact(state, "coverage_report")
        _strict_keys(
            state,
            {"format", "artifact", "policy_id", "total", "accepted", "incorrect", "coverage", "risk", "status"}
            | {"sample_ids", "accepted_ids"},
            "coverage report",
        )
        report = CoverageReport(
            policy_id=state["policy_id"],
            total=state["total"],
            accepted=state["accepted"],
            incorrect=state["incorrect"],
            sample_ids=state["sample_ids"],  # validated as int64 ids by __post_init__
            accepted_ids=state["accepted_ids"],
        )
        derived = (report.coverage, report.risk, report.status)
        if not _agrees([state["coverage"], state["risk"], state["status"]], list(derived)):
            raise AbstentionError(f"the report's coverage / risk / status contradict its counts, which give {derived}")
        return report

    @staticmethod
    def from_json(text: str) -> CoverageReport:
        return CoverageReport.from_state(parse_json(text, "coverage report", AbstentionError))

    @staticmethod
    def load(path: Any) -> CoverageReport:
        return CoverageReport.from_json(read_text(path, "coverage report", AbstentionError))


class _SeenIds:
    """Every id added so far, as sorted int64 runs merged like a binary
    counter: ``O(log n)`` runs, so a membership check of ``k`` ids costs
    ``O(k log² n)`` (a binary search per run)."""

    def __init__(self) -> None:
        self._runs: list[np.ndarray] = []

    def overlaps(self, ids: np.ndarray) -> bool:
        for run in self._runs:
            at = np.minimum(np.searchsorted(run, ids), run.size - 1)
            if bool((run[at] == ids).any()):
                return True
        return False

    def add(self, ids: np.ndarray) -> None:
        if not ids.size:
            return
        run = np.sort(ids)
        while self._runs and self._runs[-1].size <= run.size:
            # A stable (Timsort) sort finds the two sorted runs and merges them in linear time.
            run = np.sort(np.concatenate([self._runs.pop(), run]), kind="stable")
        self._runs.append(run)


class CoverageAccumulator:
    """Coverage and selective risk aggregated over chunks.

    ``update(result, targets)`` adds one chunk; :meth:`report` equals the
    eager ``AbstentionResult.report`` over the concatenated rows — the same
    accepted / incorrect / total counts and the same risk, including when
    nothing is accepted. Every chunk must come from the same policy, and by
    default every id must be new — within its chunk and across chunks —
    since the default ``0..n-1`` repeats from chunk to chunk: pass each
    chunk its own ``sample_ids=``. For a split whose ids legitimately repeat
    (oversampled), pass ``allow_repeated_ids=True``; ids are then not
    checked.
    A ``PredictionResult`` or ``CalibratedPrediction`` chunk carries the ids
    it was made with (``prediction_from_logits(sample_ids=...)``,
    ``TemperatureCalibrator.transform(sample_ids=...)``). Ids are kept as
    int64 arrays — in row order, the accepted ones, and sorted for the repeat
    check, which costs ``O(k log² n)`` for ``k`` new ids among ``n`` so far.
    """

    def __init__(self, *, allow_repeated_ids: bool = False) -> None:
        if not isinstance(allow_repeated_ids, bool):
            raise AbstentionError(f"allow_repeated_ids must be a bool, got {allow_repeated_ids!r}")
        self._check_ids = not allow_repeated_ids
        self._chunks = 0
        self._policy_id: Optional[str] = None
        self._total = self._accepted = self._incorrect = 0
        self._sample_ids: list[np.ndarray] = []
        self._accepted_ids: list[np.ndarray] = []
        self._seen = _SeenIds()

    def update(self, result: AbstentionResult, targets: Any) -> None:
        if not isinstance(result, AbstentionResult):
            raise TypeError(f"update() needs the AbstentionResult of policy.apply(...), got {type(result).__name__}")
        if self._chunks and result.policy_id != self._policy_id:
            raise AbstentionError(f"chunk from policy {result.policy_id!r}; earlier chunks used {self._policy_id!r}")
        chunk = result.report(targets)  # the eager counts, so the two cannot drift
        ids = chunk.sample_ids  # the report's own fresh copy: later edits to the result cannot leak in
        if self._check_ids and np.unique(ids).size != ids.size:
            raise AbstentionError(
                "this chunk repeats a sample id; pass allow_repeated_ids=True for a split whose ids repeat — the "
                "chunk was not added"
            )
        if self._check_ids and self._seen.overlaps(ids):
            raise AbstentionError(
                "this chunk repeats sample ids of an earlier chunk (the default 0..n-1 repeats); pass each chunk "
                "its own sample_ids= (a prediction's when it is made: prediction_from_logits(sample_ids=...), "
                "TemperatureCalibrator.transform(sample_ids=...)), or allow_repeated_ids=True for a split whose "
                "ids repeat — the chunk was not added"
            )
        if self._check_ids:
            self._seen.add(ids)
        self._chunks += 1
        self._policy_id = result.policy_id
        self._total += chunk.total
        self._accepted += chunk.accepted
        self._incorrect += chunk.incorrect
        self._sample_ids.append(ids)
        self._accepted_ids.append(chunk.accepted_ids)

    def report(self) -> CoverageReport:
        """The aggregated report over every chunk added so far."""

        def joined(parts: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

        return CoverageReport(
            policy_id=self._policy_id,
            total=self._total,
            accepted=self._accepted,
            incorrect=self._incorrect,
            sample_ids=joined(self._sample_ids),
            accepted_ids=joined(self._accepted_ids),
        )


# --- risk-coverage curves and threshold selection -------------------------------------------------


@dataclass(frozen=True)
class CurvePoint(_Counted):
    """One point of a risk-coverage curve: rows with a score at or above
    ``threshold`` are accepted. The final ``"accept_none"`` point
    (``threshold=None``, coverage 0, risk unavailable) exists only on the
    curve and is not ``deployable``."""

    threshold: Optional[float]
    accepted: int
    incorrect: int
    total: int

    def __post_init__(self) -> None:
        _set_counts(self, "curve point")
        if self.threshold is not None:
            object.__setattr__(self, "threshold", _threshold(self.threshold))
        if self.threshold is None and self.accepted:
            raise AbstentionError("the accept-none endpoint accepts no rows")

    @property
    def status(self) -> str:
        return "accept_none" if self.threshold is None else _status(self.accepted, self.total)

    @property
    def deployable(self) -> bool:
        return self.threshold is not None

    def state(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "accepted": self.accepted,
            "incorrect": self.incorrect,
            "total": self.total,
            "coverage": self.coverage,
            "risk": self.risk,
            "status": self.status,
        }


def _best(points: Sequence[CurvePoint], ceiling: float) -> Optional[CurvePoint]:
    """The feasible point (deployable, some row accepted, risk within the
    ceiling) with the highest coverage. Equal coverage means the same
    accepted set, hence the same risk; the higher, more conservative
    threshold then wins."""
    feasible = [p for p in points if p.deployable and p.risk is not None and p.risk <= ceiling]
    return max(feasible, key=lambda p: (p.accepted, p.threshold), default=None)


def _distinct(ascending: np.ndarray) -> np.ndarray:
    """The distinct values of a sorted array, without sorting it again."""
    keep = np.ones(ascending.shape[0], dtype=bool)
    np.not_equal(ascending[1:], ascending[:-1], out=keep[1:])
    return ascending[keep]


def _candidates(values: Any, ascending: np.ndarray) -> np.ndarray:
    """Candidate thresholds, ascending: ``values``, or by default drawn from
    the sorted scores ``ascending``."""
    if values is None:
        distinct = _distinct(ascending)  # each distinct score is a distinct accepted set
        if distinct.size <= MAX_DEFAULT_CANDIDATES:
            return distinct
        # Evenly spaced order statistics: actual scores (so every point is an
        # exact accepted set), from accept-all to the top score.
        ranks = np.linspace(0, ascending.size - 1, MAX_DEFAULT_CANDIDATES).round().astype(np.int64)
        return _distinct(ascending[ranks])
    flat = f"candidates must be a flat list of thresholds in [0, 1], got {values!r}"
    if is_tensor(values):
        values = read_array(values, copy=False, error=AbstentionError)  # torch.linspace(...), say
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise AbstentionError(flat)
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise AbstentionError(flat)
    values = list(values)  # generators and sets too
    if any(isinstance(value, (str, bytes)) or np.ndim(cast(Any, value)) for value in values):
        raise AbstentionError(flat)  # nested or ragged lists never reach NumPy
    try:
        checked = [_threshold(value) for value in values]
    except AbstentionError as exc:
        raise AbstentionError(f"candidates must be thresholds in [0, 1]: {exc}") from exc
    if not checked:
        raise AbstentionError("candidates must list at least one threshold")
    return np.unique(np.asarray(checked, dtype=np.float64))


def _curve(score: np.ndarray, wrong: np.ndarray, candidates: Any) -> tuple[CurvePoint, ...]:
    order = np.argsort(score, kind="stable")  # the one sort: candidates and counts both use it
    ascending = score[order]
    thresholds = _candidates(candidates, ascending)
    wrong_from = np.concatenate([np.cumsum(wrong[order][::-1])[::-1], [0]])  # wrong rows at or after index i
    total = int(score.shape[0])
    first = np.searchsorted(ascending, thresholds, side="left")  # each threshold's first accepted row
    accepted, incorrect = (total - first).tolist(), wrong_from[first].tolist()
    points = [CurvePoint(t, a, i, total) for t, a, i in zip(thresholds.tolist(), accepted, incorrect, strict=True)]
    points.append(CurvePoint(None, 0, 0, total))  # the curve-only accept-none endpoint
    return tuple(points)


def _points(
    p: np.ndarray, decoded: Optional[np.ndarray], targets: Any, kind: str, candidates: Any
) -> tuple[CurvePoint, ...]:
    """The curve of validated probabilities against ``targets`` — one
    pipeline for :func:`risk_coverage_curve` and :func:`select_threshold`."""
    y = class_targets(targets, p.shape[0], p.shape[1], error=AbstentionError)
    prediction, score = _scores(p, kind, decoded)
    return _curve(score, prediction != y, candidates)


def _view(source: Any, input_field: Optional[str]) -> tuple[Any, Optional[np.ndarray], Optional[tuple[str, ...]]]:
    """The probabilities a curve reads, a prediction's own decoded classes
    and its labels: a ``CalibratedPrediction``'s ``input_field`` view, a
    categorical ``PredictionResult``'s probabilities, or bare arrays."""
    from .calibration import CalibratedPrediction

    if isinstance(source, CalibratedPrediction):
        if input_field is None:
            raise AbstentionError(
                "a CalibratedPrediction holds raw and calibrated probabilities; pass input_field="
                "'probabilities' or 'calibrated_probabilities'"
            )
        return getattr(source, _field(input_field)), np.asarray(source.decoded), _labels(source.labels)
    if is_prediction(source):
        labels = prediction_labels(
            source, None, error=AbstentionError, conflict_error=AbstentionSchemaError, what="a curve", minimum=2
        )  # a categorical prediction
        if input_field is not None and _field(input_field) != "probabilities":
            raise AbstentionSchemaError("a PredictionResult holds raw probabilities only")
        _require_prediction_arrays(source)
        return source.probabilities, np.asarray(source.decoded), labels
    if input_field is not None and _field(input_field) != "probabilities":
        raise AbstentionError(
            "input_field= picks a CalibratedPrediction's view; bare probabilities are read as given (pass no "
            "input_field, or 'probabilities')"
        )
    return source, None, None


def risk_coverage_curve(
    source: Any,
    targets: Any,
    *,
    kind: str,
    candidates: Optional[Iterable[float]] = None,
    input_field: Optional[str] = None,
) -> tuple[CurvePoint, ...]:
    """Coverage and selective risk at each candidate threshold, ascending.

    ``source`` is ``(N, C)`` probabilities (the prediction is each row's
    first class of maximal probability), a categorical ``PredictionResult``
    or a ``CalibratedPrediction`` with ``input_field=`` (their own
    ``decoded`` classes, as :func:`select_threshold` reads them). By default
    the candidates are the distinct scores (bounded by
    :data:`MAX_DEFAULT_CANDIDATES`). Coverage never rises as the threshold
    rises; risk is measured at each point, never assumed monotone. The last
    point is the accept-none endpoint.
    """
    policy_kind = _kind(kind)  # before any probability is read
    probabilities, decoded, labels = _view(source, input_field)
    p, eps = categorical_probabilities_eps(probabilities, error=AbstentionError, allow_empty=True)
    if labels is not None:
        _check_width(labels, p)
    classes = _decoded(decoded, p, rounding_tolerance(eps, floor=0.0))
    return _points(p, classes, targets, policy_kind, candidates)


@dataclass(frozen=True, eq=False)
class ThresholdSelection(JsonArtifact):
    """The outcome of :func:`select_threshold`.

    ``status`` is ``"ok"`` with the chosen ``policy`` or
    ``"no_feasible_policy"`` with ``policy=None``. ``points`` lists every
    candidate threshold with its counts, coverage and risk, then the
    accept-none endpoint. ``basis`` is ``"empirical"``: the ceiling was met
    on ``split_id``, which does not guarantee it anywhere else.
    ``labels``, ``model_id``, ``input_field`` and ``calibrator_id`` record
    what was tuned on — a chosen policy carries the same — so even a
    ``"no_feasible_policy"`` selection says which model and view failed.
    ``calibration_override`` is the named override (``nnx.calibration``)
    under which the calibrated probabilities tuned on were produced, if any.
    """

    status: str
    policy: Optional[AbstentionPolicy]
    points: tuple[CurvePoint, ...]
    kind: str
    risk_ceiling: float
    split_id: str
    labels: tuple[str, ...]
    model_id: str
    input_field: str = "probabilities"
    calibrator_id: Optional[str] = None
    test_split_id: Optional[str] = None
    basis: str = "empirical"
    calibration_override: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", _labels(self.labels))
        object.__setattr__(self, "model_id", _id(self.model_id, "model_id"))
        object.__setattr__(self, "input_field", _field(self.input_field))
        object.__setattr__(self, "calibrator_id", _calibrator(self.input_field, self.calibrator_id))
        record = override_record(self.calibration_override, error=AbstentionError, what="a calibration_override")
        if record is not None and self.input_field != "calibrated_probabilities":
            raise AbstentionError("a calibration_override belongs to a selection tuned on calibrated_probabilities")
        object.__setattr__(self, "calibration_override", record)
        if self.status not in ("ok", "no_feasible_policy"):
            raise AbstentionError(f"selection status must be 'ok' or 'no_feasible_policy', got {self.status!r}")
        if (self.status == "ok") != (self.policy is not None):
            raise AbstentionError("an 'ok' selection holds a policy and a 'no_feasible_policy' one holds none")
        if self.policy is not None and not isinstance(self.policy, AbstentionPolicy):
            raise AbstentionError(f"a selection's policy must be an AbstentionPolicy, got {self.policy!r}")
        object.__setattr__(self, "kind", _kind(self.kind))
        object.__setattr__(self, "risk_ceiling", _unit(self.risk_ceiling, "risk_ceiling"))
        object.__setattr__(self, "split_id", _check_tuning_split(self.split_id, self.test_split_id))
        if self.basis != "empirical":
            raise AbstentionError(f"a selection's basis is 'empirical', got {self.basis!r}")
        points = tuple(self.points)
        if not points or not all(isinstance(p, CurvePoint) for p in points):
            raise AbstentionError("a selection's points are CurvePoints")
        thresholds = [p.threshold for p in points[:-1]]
        if (
            points[-1].threshold is not None
            or any(t is None for t in thresholds)
            or any(a >= b for a, b in pairwise(thresholds))  # type: ignore[operator]
            or len({p.total for p in points}) != 1
        ):
            raise AbstentionError(
                "a selection's points are strictly ascending thresholds over one total, ending in the single "
                "accept-none endpoint"
            )
        if any(b.accepted > a.accepted or b.incorrect > a.incorrect for a, b in pairwise(points)):
            raise AbstentionError(
                "a higher threshold accepts a subset of the rows: accepted and incorrect counts never rise along "
                "a selection's points"
            )
        object.__setattr__(self, "points", points)
        best = _best(points, self.risk_ceiling)
        if (best is None) != (self.policy is None) or (
            self.policy is not None
            and (
                self.policy.kind != self.kind
                or self.policy.tuning_split_id != self.split_id
                or (self.policy.labels, self.policy.model_id) != (self.labels, self.model_id)
                or (self.policy.input_field, self.policy.calibrator_id) != (self.input_field, self.calibrator_id)
                or best is None
                or self.policy.threshold != best.threshold
            )
        ):
            raise AbstentionError(
                "the selected policy must be the best feasible candidate of the selection's kind, split and schema "
                f"(best: {None if best is None else best.threshold!r})"
            )

    def require(self) -> AbstentionPolicy:
        """The chosen policy, or :class:`AbstentionError`."""
        if self.policy is None:
            raise AbstentionError(
                f"no feasible policy: no {self.kind} threshold reaches risk <= {self.risk_ceiling} on split "
                f"{self.split_id!r} with any accepted row"
            )
        return self.policy

    def state(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "artifact": "threshold_selection",
            "status": self.status,
            "policy": None if self.policy is None else self.policy.state(),
            "points": [point.state() for point in self.points],
            "kind": self.kind,
            "risk_ceiling": self.risk_ceiling,
            "split_id": self.split_id,
            "labels": list(self.labels),
            "model_id": self.model_id,
            "input_field": self.input_field,
            "calibrator_id": self.calibrator_id,
            "test_split_id": self.test_split_id,
            "basis": self.basis,
            "calibration_override": None
            if self.calibration_override is None
            else _thaw_config(self.calibration_override),
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> ThresholdSelection:
        state = _artifact(state, "threshold_selection")
        _strict_keys(
            state,
            {"format", "artifact", "status", "policy", "points", "kind", "risk_ceiling", "split_id", "test_split_id"}
            | {"basis", "calibration_override", "labels", "model_id", "input_field", "calibrator_id"},
            "threshold selection",
        )
        if not isinstance(state["points"], list):
            raise AbstentionError("a selection's points are a list")
        points = []
        for stored in state["points"]:
            if not isinstance(stored, Mapping):
                raise AbstentionError(f"a selection point is a mapping, got {stored!r}")
            _strict_keys(stored, {"threshold", "accepted", "incorrect", "total", "coverage", "risk", "status"}, "point")
            point = CurvePoint(  # validated by __post_init__
                threshold=stored["threshold"],
                accepted=stored["accepted"],
                incorrect=stored["incorrect"],
                total=stored["total"],
            )
            if not _agrees([stored[key] for key in point.state()], list(point.state().values())):
                raise AbstentionError(f"a selection point's coverage / risk / status contradict its counts: {stored!r}")
            points.append(point)
        return ThresholdSelection(
            status=state["status"],
            policy=None if state["policy"] is None else AbstentionPolicy.from_state(state["policy"]),
            points=tuple(points),
            kind=state["kind"],
            risk_ceiling=state["risk_ceiling"],
            split_id=state["split_id"],
            labels=state["labels"],
            model_id=state["model_id"],
            input_field=state["input_field"],
            calibrator_id=state["calibrator_id"],
            test_split_id=state["test_split_id"],
            basis=state["basis"],
            calibration_override=state["calibration_override"],
        )

    @staticmethod
    def from_json(text: str) -> ThresholdSelection:
        return ThresholdSelection.from_state(parse_json(text, "threshold selection", AbstentionError))

    @staticmethod
    def load(path: Any) -> ThresholdSelection:
        return ThresholdSelection.from_json(read_text(path, "threshold selection", AbstentionError))


def select_threshold(
    source: Any,
    targets: Any,
    *,
    kind: str,
    risk_ceiling: float,
    split_id: str,
    test_split_id: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    model_id: Optional[str] = None,
    input_field: Optional[str] = None,
    calibrator_id: Optional[str] = None,
    sample_ids: Any = None,
    candidates: Optional[Iterable[float]] = None,
) -> ThresholdSelection:
    """Choose a ``kind`` threshold on the validation split ``split_id``.

    ``source`` is declared as for :meth:`AbstentionPolicy.apply`; a
    ``CalibratedPrediction`` needs ``input_field=`` to say which of its two
    views to tune on. Every candidate threshold is evaluated: by default
    every distinct score, or, beyond :data:`MAX_DEFAULT_CANDIDATES` of them,
    that many evenly spaced order statistics of the scores (pass
    ``candidates=`` for another set). The feasible ones accept at least one
    row with ``risk <= risk_ceiling``; among them the highest coverage wins
    (equal coverage is the same accepted set, so the same risk), then the
    higher, more conservative threshold.
    Nothing feasible gives ``status="no_feasible_policy"``. A ``split_id``
    equal to ``test_split_id`` is refused: tuning on the test split would
    leak it. The ceiling is empirical — met on this split, not guaranteed.
    """
    tuning = _check_tuning_split(split_id, test_split_id)
    policy_kind = _kind(kind)
    ceiling = _unit(risk_ceiling, "risk_ceiling")
    declared = _declared(
        source,
        labels=labels,
        model_id=model_id,
        sample_ids=sample_ids,
        input_field=input_field,
        calibrator_id=calibrator_id,
        default_field=None,  # a CalibratedPrediction must say which view to tune on
    )
    p, _, decoded = _read(declared, copy=False)  # only read; the ids are checked for form, then unused
    points = _points(p, decoded, targets, policy_kind, candidates)
    best = _best(points, ceiling)
    policy = None
    if best is not None:
        policy = AbstentionPolicy(
            kind=policy_kind,
            threshold=float(best.threshold),  # type: ignore[arg-type]
            labels=declared.labels,
            model_id=declared.model_id,
            tuning_split_id=tuning,
            input_field=declared.input_field,
            calibrator_id=declared.calibrator_id,
        )
    return ThresholdSelection(
        status="ok" if policy is not None else "no_feasible_policy",
        policy=policy,
        points=points,
        kind=policy_kind,
        risk_ceiling=ceiling,
        split_id=tuning,
        labels=declared.labels,
        model_id=declared.model_id,
        input_field=declared.input_field,
        calibrator_id=declared.calibrator_id,
        test_split_id=test_split_id,
        calibration_override=declared.override,
    )


# --- typed decisions -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectiveDecision:
    """A typed decision with a policy's verdict: ``result`` is the
    provider's ``ChoiceResult`` / ``ScoreResult``, unchanged — abstaining
    is a success, not an error — and ``outcome`` says whether it was
    accepted. ``input_field`` / ``calibrator_id`` record which
    probabilities the verdict read (a provider's distribution is raw)."""

    result: Any
    outcome: Outcome
    policy_id: str
    input_field: str
    calibrator_id: Optional[str]

    @property
    def accepted(self) -> bool:
        return self.outcome.accepted

    @property
    def abstained(self) -> bool:
        return not self.outcome.accepted


def decide(result: Any, policy: AbstentionPolicy, *, model_id: str) -> SelectiveDecision:
    """Apply ``policy`` to a typed decision (``nnx.decisions``).

    ``result`` is a ``ChoiceResult`` or ``ScoreResult``; its option ids, in
    order, are the labels. A provider's distribution is raw, so the policy
    must read ``"probabilities"``; labels and ``model_id`` must equal the
    policy's (:class:`AbstentionSchemaError` otherwise). The result is
    returned unchanged inside a :class:`SelectiveDecision`, accepted or not.
    """
    from .decisions.schema import ChoiceResult, ScoreResult

    if not isinstance(result, (ChoiceResult, ScoreResult)):
        raise TypeError(
            f"decide() needs a ChoiceResult or ScoreResult (the answer to a Choice or Score), got {result!r}"
        )
    if not isinstance(policy, AbstentionPolicy):
        raise TypeError(
            f"decide() needs an AbstentionPolicy (a ThresholdSelection's is selection.require()), got {policy!r}"
        )
    distribution = tuple((option_id, float(p)) for option_id, p in result.distribution)
    labels = _labels([option_id for option_id, _ in distribution])
    # A provider's distribution is raw: declared as such, whatever the policy reads.
    policy._check(labels, _id(model_id, "model_id"), "probabilities", None)
    # nnx.decisions already validated the distribution (fsum within its own
    # tolerance); it is scored as is, never re-validated or renormalized.
    prediction, score = _scores(np.array([[p for _, p in distribution]], dtype=np.float64), policy.kind)
    accepted = bool(score[0] >= policy.threshold)
    index = int(prediction[0])
    return SelectiveDecision(
        result=result,
        outcome=Outcome(
            sample_id=None,
            status="accepted" if accepted else "abstained",
            prediction=labels[index],
            prediction_index=index,
            distribution=distribution,
            score=float(score[0]),
            reason=None if accepted else policy.reason,
        ),
        policy_id=policy.id,
        input_field="probabilities",
        calibrator_id=None,
    )
