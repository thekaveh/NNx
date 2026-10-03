"""Fitted classifier calibration (FEAT-007).

A classifier's softmax probabilities are often over- or under-confident.
Temperature scaling fits one scalar ``T > 0`` on a held-out **calibration
split** and serves ``softmax(logits / T)``: the argmax never changes, only
the confidence does. This is a *fitted artifact*, unlike the sampling-time
``nnx.generation.TemperatureScaling`` processor, which rescales language-model
logits by a temperature you choose and fits nothing.

- :func:`fit_temperature` minimizes the calibration split's negative
  log-likelihood in float64 over a copy of the logits (categorical ``(N, C)``,
  ``C >= 2``). It rejects non-finite logits, out-of-range targets and a
  calibration split id equal to the training or test split id; ids that
  differ do not prove the rows are disjoint, and the calibrator records that
  (``disjointness: "unverified"``). A fit with no interior optimum (a
  separable split, constant logits, logits no better than uniform) or that
  does not converge returns a failed :class:`CalibrationFit`, never a NaN
  calibrator.
- :class:`TemperatureCalibrator` stores the temperature with its label
  schema, model id, fit configuration and split ids, serializes as primitive
  JSON (``nnx.calibration/1``) and reloads exactly. ``transform`` refuses
  logits whose labels or model id differ from the fitted ones — before any
  computation — unless a named ``override`` records the mismatch, and returns
  a :class:`CalibratedPrediction` that keeps raw logits, raw probabilities,
  calibrated probabilities and the calibrator id as separate fields.
- :func:`negative_log_likelihood`, :func:`brier_score` and
  :func:`reliability_bins` score categorical probabilities; ``report`` puts
  them side by side before and after calibration on a held-out split and
  says so when calibration made things worse.

The module's own imports are NumPy, the standard library and NumPy-only
internal helpers — none of its consumers (``nnx.prediction``,
``nnx.monitors``, provenance or decisions) — and nothing here modifies a
model, its predictions or its run. (Importing it as ``nnx.calibration``
still runs the package's ``__init__``, which imports torch.) ``save`` reuses
NNx's atomic file writer, as the other JSON artifacts do, and
:func:`model_fingerprint` imports torch when it is called.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union, cast

import numpy as np

from ._artifacts import JsonArtifact, check_keys, frozen_json, override_record, parse_json, read_text
from ._config import _canonical_json, _freeze_config, _FrozenConfig, _thaw_config
from ._probability import (
    NLL_EPSILON,
    brier_terms,
    categorical_probabilities,
    class_labels,
    class_targets,
    is_prediction,
    nll_terms,
    prediction_labels,
    read_array,
    row_ids,
    same_ids,
    softmax_,
)
from ._validation import checked, require_count, require_finite_real, required_id

__all__ = [
    "DEFAULT_EPSILON",
    "FORMAT",
    "CalibratedPrediction",
    "CalibrationError",
    "CalibrationFit",
    "CalibrationFitError",
    "CalibrationMetrics",
    "CalibrationMismatchError",
    "CalibrationReport",
    "ReliabilityBin",
    "TemperatureCalibrator",
    "brier_score",
    "expected_calibration_error",
    "fit_temperature",
    "model_fingerprint",
    "negative_log_likelihood",
    "reliability_bins",
]

FORMAT = "nnx.calibration/1"
"""Format version of a serialized calibrator or report (part of its digest)."""

DEFAULT_EPSILON = NLL_EPSILON
"""Probability floor of :func:`negative_log_likelihood` — the same floor the
named ``nll`` metric (``nnx.monitors``) uses, so the two agree."""

_DISJOINTNESS_NOTE = (
    "the split ids differ, but ids, arrays and loaders cannot prove that the calibration rows are "
    "disjoint from the training and test rows; keep the calibration split separate yourself"
)
_OUTCOMES = ("improved", "worsened", "unchanged", "mixed")


class CalibrationError(ValueError):
    """Calibration inputs, configuration or a serialized state that cannot be
    used as declared."""


class CalibrationFitError(CalibrationError):
    """:meth:`CalibrationFit.require` on a failed fit."""


class CalibrationMismatchError(CalibrationError):
    """Logits whose label schema or model id differ from the calibrator's,
    without a named ``override``."""


# --- validation ----------------------------------------------------------------------------------


def _id(value: Any, what: str) -> str:
    return required_id(value, what, error=CalibrationError)


def _optional_id(value: Any, what: str) -> Optional[str]:
    return None if value is None else _id(value, what)


def _labels(labels: Any) -> tuple[str, ...]:
    # No minimum here: a wrong label count is an identity mismatch, checked
    # against the calibrator's own labels.
    return class_labels(labels, error=CalibrationError, minimum=0)


def _check_held_out(split_id: str, calibration_split_id: str, train_split_id: Optional[str]) -> None:
    if split_id in (calibration_split_id, train_split_id):
        raise CalibrationError(
            f"split_id {split_id!r} is not held out: it is the calibration or training split; "
            "report on a split the calibrator never saw"
        )


def _check_splits(split_id: str, train_split_id: Optional[str], test_split_id: Optional[str]) -> None:
    for name, other in (("train_split_id", train_split_id), ("test_split_id", test_split_id)):
        if other is not None and other == split_id:
            raise CalibrationError(
                f"the calibration split_id {split_id!r} must differ from {name}: fitting on the training or "
                "test split would leak it into the calibrator"
            )
    if train_split_id is not None and train_split_id == test_split_id:
        raise CalibrationError(f"train_split_id and test_split_id must differ, both are {train_split_id!r}")


def _split_record(calibration: str, train: Optional[str], test: Optional[str], *, note: bool = False) -> dict[str, Any]:
    """The split ids with their (always unverified) disjointness; ``note``
    adds the human-readable reason, which is not part of the serialized state."""
    record: dict[str, Any] = {"calibration": calibration, "train": train, "test": test, "disjointness": "unverified"}
    if note:
        record["note"] = _DISJOINTNESS_NOTE
    return record


def _override_record(value: Any) -> Optional[Mapping[str, Any]]:
    """Validate a serialized override record (shared with ``nnx.abstention``)."""
    return override_record(value, error=CalibrationError)


def _checked(check: Any, value: Any, what: str, **domain: Any) -> Any:
    """Run a shared ``nnx._validation`` check, raising :class:`CalibrationError`."""
    return checked(check, value, what, owner="nnx.calibration", error=CalibrationError, **domain)


def _positive_float(value: Any, what: str) -> float:
    message = f"{what} must be a finite positive number, got {value!r}"
    return _checked(require_finite_real, value, what, minimum=0.0, exclusive_min=True, domain_message=message)


def _finite(value: Any, what: str) -> float:
    return _checked(require_finite_real, value, what)


def _count(value: Any, what: str, minimum: int) -> int:
    return _checked(require_count, value, what, minimum=minimum)


class _malformed:
    """Turn a missing or mistyped entry of a serialized state into a
    :class:`CalibrationError` naming ``what``."""

    def __init__(self, what: str) -> None:
        self.what = what

    def __enter__(self) -> None:
        return None

    def __exit__(self, kind: Any, exc: Any, traceback: Any) -> None:  # never suppresses
        if isinstance(exc, (KeyError, TypeError, ValueError, AttributeError)) and not isinstance(exc, CalibrationError):
            raise CalibrationError(f"malformed {self.what}: {exc!r}") from exc


def _epsilon(epsilon: Any) -> Optional[float]:
    if epsilon is None:
        return None
    number = _positive_float(epsilon, "epsilon")
    if number >= 1:
        raise CalibrationError(f"epsilon must be in (0, 1) or None for the exact NLL, got {epsilon!r}")
    return number


def _numpy(value: Any, *, copy: bool = True) -> np.ndarray:
    return read_array(value, copy=copy, error=CalibrationError)


def _logits(value: Any, *, copy: bool = True, allow_empty: bool = False) -> np.ndarray:
    what = "logits"
    array = _numpy(value, copy=copy)
    if array.dtype.kind not in "fiu":
        raise CalibrationError(f"{what} must be a real numeric array, got dtype {array.dtype}")
    if array.ndim != 2:
        raise CalibrationError(f"{what} must be categorical (N, C) with classes on axis 1, got shape {array.shape}")
    if array.shape[1] < 2:
        raise CalibrationError(f"{what} need at least 2 classes, got {array.shape[1]}")
    if array.shape[0] < 1 and not allow_empty:
        raise CalibrationError(f"{what} hold no rows")
    finite = np.isfinite(array)
    if not finite.all():
        raise CalibrationError(
            f"{what} contain {int(finite.size - np.count_nonzero(finite))} non-finite value(s) (NaN or ±inf); "
            "replace masked logits with a large finite negative value"
        )
    return array


def _targets(value: Any, n_rows: int, n_classes: int) -> np.ndarray:
    return class_targets(value, n_rows, n_classes, error=CalibrationError)


def _probabilities(value: Any, *, allow_empty: bool = False) -> np.ndarray:
    return categorical_probabilities(value, error=CalibrationError, allow_empty=allow_empty)


# --- metrics -------------------------------------------------------------------------------------


def negative_log_likelihood(probabilities: Any, targets: Any, *, epsilon: Optional[float] = DEFAULT_EPSILON) -> float:
    """Mean ``-log p[target]`` over the rows of categorical ``(N, C)``
    probabilities.

    ``epsilon`` floors the true-class probability (default
    :data:`DEFAULT_EPSILON`, the named ``nll`` metric's floor); ``epsilon=None``
    is the exact value, which is ``+inf`` when any true-class probability is 0.
    """
    floor = _epsilon(epsilon)
    p = _probabilities(probabilities)
    return _nll(p, _targets(targets, p.shape[0], p.shape[1]), floor)


def _nll(p: np.ndarray, y: np.ndarray, floor: Optional[float]) -> float:
    return float(nll_terms(y, p, floor).mean())  # the named `nll` metric's own terms


def brier_score(probabilities: Any, targets: Any) -> float:
    """Mean over rows of the class-summed squared error ``sum_c (p_c - 1[c == target])^2``
    (range ``[0, 2]``)."""
    p = _probabilities(probabilities)
    return _brier(p, _targets(targets, p.shape[0], p.shape[1]))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(brier_terms(y, p).mean())  # the named `brier` metric's own terms


@dataclass(frozen=True)
class ReliabilityBin:
    """One confidence bin: ``[lower, upper)`` (the last bin is closed at 1),
    the rows whose top-1 confidence falls in it, their mean ``confidence`` and
    top-1 ``accuracy``. An empty bin has ``count == 0`` and ``None`` means."""

    lower: float
    upper: float
    count: int
    confidence: Optional[float]
    accuracy: Optional[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "count", _count(self.count, "reliability bin count", 0))
        for name in ("lower", "upper", "confidence", "accuracy"):  # plain floats: JSON-serializable
            value = getattr(self, name)
            if value is not None or name in ("lower", "upper"):
                object.__setattr__(self, name, _finite(value, f"reliability bin {name}"))
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise CalibrationError(f"a reliability bin needs 0 <= lower < upper <= 1, got [{self.lower}, {self.upper})")
        means = (self.confidence, self.accuracy)
        if self.count == 0 and means != (None, None):
            raise CalibrationError("an empty reliability bin has no confidence or accuracy")
        if self.count > 0 and not all(mean is not None and 0.0 <= mean <= 1.0 for mean in means):
            raise CalibrationError(
                f"a reliability bin with {self.count} rows needs confidence and accuracy in [0, 1], got {means}"
            )

    def state(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "confidence": self.confidence,
            "accuracy": self.accuracy,
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> ReliabilityBin:
        """Rebuild from :meth:`state`; malformed entries raise :class:`CalibrationError`."""
        with _malformed("reliability bin"):
            return ReliabilityBin(  # every field is validated by __post_init__
                lower=state["lower"],
                upper=state["upper"],
                count=state["count"],
                confidence=state.get("confidence"),
                accuracy=state.get("accuracy"),
            )


_MAX_BINS = 10_000


def _n_bins(n_bins: Any) -> int:
    return _checked(require_count, n_bins, "n_bins", minimum=1, maximum=_MAX_BINS)


def reliability_bins(probabilities: Any, targets: Any, *, n_bins: int = 10) -> tuple[ReliabilityBin, ...]:
    """Equal-width top-1 reliability bins over ``[0, 1]``.

    Bin ``i`` covers ``[i / n_bins, (i + 1) / n_bins)``; the last bin is
    closed at 1, and a confidence exactly on an inner edge belongs to the
    upper bin. The prediction is the first class of maximal probability, and
    empty bins are kept with ``count == 0``.
    """
    bins = _n_bins(n_bins)
    p = _probabilities(probabilities, allow_empty=True)
    return _bins(p, _targets(targets, p.shape[0], p.shape[1]), bins)


def _bins(
    p: np.ndarray, y: np.ndarray, bins: int, predicted: Optional[np.ndarray] = None
) -> tuple[ReliabilityBin, ...]:
    """Bins of validated arrays. ``predicted`` fixes each row's top-1 class
    (a report passes the raw-logit argmax for both views, so a float tie
    after calibration cannot move a row to another prediction); by default
    it is the first class of maximal probability."""
    edges = np.arange(bins + 1, dtype=np.float64) / bins
    rows = np.arange(p.shape[0])
    top = p.argmax(axis=1) if predicted is None else predicted
    confidence = p[rows, top]
    correct = (top == y).astype(np.float64)
    index = np.clip(np.searchsorted(edges, confidence, side="right") - 1, 0, bins - 1)
    counts = np.bincount(index, minlength=bins)  # one pass for counts and sums
    confidence_sums = np.bincount(index, weights=confidence, minlength=bins)
    correct_sums = np.bincount(index, weights=correct, minlength=bins)
    out = []
    for i in range(bins):
        count = int(counts[i])
        out.append(
            ReliabilityBin(
                lower=float(edges[i]),
                upper=float(edges[i + 1]),
                count=count,
                # A float mean of values in [0, 1] can round a hair past 1.
                confidence=min(float(confidence_sums[i] / count), 1.0) if count else None,
                accuracy=float(correct_sums[i] / count) if count else None,
            )
        )
    return tuple(out)


def expected_calibration_error(bins: Iterable[ReliabilityBin]) -> Optional[float]:
    """Count-weighted mean ``|accuracy - confidence|`` over the bins; ``None``
    when the bins hold no rows."""
    bins = tuple(bins)  # read twice below: a generator must not run dry
    total = sum(b.count for b in bins)
    if total == 0:
        return None
    return float(
        sum(b.count / total * abs(float(b.accuracy) - float(b.confidence)) for b in bins if b.count)  # type: ignore[arg-type]
    )


# --- float64 softmax -----------------------------------------------------------------------------


def _shifted(logits: np.ndarray) -> np.ndarray:
    """A float64 copy of ``logits`` with each row's maximum moved to 0 —
    softmax- and slope-invariant, and a temperature below 1 can no longer
    overflow it. Rejects rows whose spread exceeds the float64 range."""
    shifted = logits.astype(np.float64, copy=True)
    with np.errstate(over="ignore"):
        shifted -= shifted.max(axis=1, keepdims=True)
    if shifted.size and not np.isfinite(shifted.min(axis=1)).all():  # finite input: only the shift can overflow
        raise CalibrationError(
            "logits span more than the float64 range within a row; use a large finite negative value "
            "(e.g. -1e9) for masked classes"
        )
    return shifted


def _softmax(shifted: np.ndarray, temperature: float) -> np.ndarray:
    """``softmax(logits / temperature)`` over axis 1 from :func:`_shifted`
    logits, through ``nnx.prediction``'s own softmax: at temperature 1 the
    probabilities equal its float64 ones bit for bit (dividing by 1 and
    re-subtracting a zero maximum change nothing)."""
    with np.errstate(over="ignore"):  # a huge masked logit over a small T is -inf: probability 0
        return softmax_(shifted / temperature, 1)


def _mean_nll(shifted: np.ndarray, targets: np.ndarray, beta: float) -> float:
    """Exact mean NLL of ``softmax(beta * logits)`` in the log domain
    (``shifted`` has each row's maximum at 0)."""
    scaled = beta * shifted
    log_norm = np.log(np.exp(scaled).sum(axis=1))
    return float((log_norm - scaled[np.arange(scaled.shape[0]), targets]).mean())


class _Slope:
    """``d NLL / d beta`` for ``beta = 1 / T``: ``mean(E_p[z] - z_target)``,
    non-decreasing in ``beta`` because the NLL is convex in ``beta``. One
    ``(N, C)`` work buffer is reused across the bisection's evaluations."""

    def __init__(self, shifted: np.ndarray, targets: np.ndarray) -> None:
        self.shifted = shifted
        self.true = shifted[np.arange(shifted.shape[0]), targets]
        self.buffer = np.empty_like(shifted)

    def __call__(self, beta: float) -> float:
        with np.errstate(over="ignore"):  # a huge masked logit times beta is -inf: weight 0
            weights = np.multiply(self.shifted, beta, out=self.buffer)
        np.exp(weights, out=weights)
        expected = np.einsum("ij,ij->i", weights, self.shifted) / weights.sum(axis=1)
        return float((expected - self.true).mean())


# --- the calibrator ------------------------------------------------------------------------------


def _frozen(value: Any, what: str) -> _FrozenConfig:
    """An immutable, picklable JSON-like copy of a mapping, so a calibrator's
    state cannot drift from its digest."""
    return frozen_json(value, what, CalibrationError)


@dataclass(frozen=True, eq=False)
class CalibratedPrediction:
    """Raw and calibrated views of one categorical prediction, kept apart.

    Attributes:
        logits: the raw logits as given (a copy; tensors become arrays).
        probabilities: uncalibrated float64 ``softmax(logits)``.
        calibrated_probabilities: float64 ``softmax(logits / temperature)``.
        decoded: argmax class indices of the raw logits — calibration never
            changes the argmax.
        labels: the ordered class names of the columns.
        sample_ids: ``int64[N]`` row identities (a ``PredictionResult``'s
            own ids, else ``transform``'s ``sample_ids=``, else ``0..N-1``).
        calibrator_id: :meth:`TemperatureCalibrator.digest` of the calibrator.
        model_id: the model id the logits were declared to come from.
        temperature: the calibrator's temperature.
        override: ``None``, or the named override and the label / model-id
            mismatches it accepted.
    """

    logits: np.ndarray
    probabilities: np.ndarray
    calibrated_probabilities: np.ndarray
    decoded: np.ndarray
    labels: tuple[str, ...]
    sample_ids: np.ndarray
    calibrator_id: str
    model_id: str
    temperature: float
    override: Optional[Mapping[str, Any]] = None

    def decoded_labels(self) -> np.ndarray:
        """Decoded class names."""
        return np.asarray(self.labels, dtype=object)[self.decoded]


def _source(value: Any, labels: Any) -> tuple[Any, Optional[tuple[str, ...]], Optional[Any]]:
    """Split a ``PredictionResult`` (duck-typed) into logits, labels and
    sample ids without reading the logits; other inputs pass through."""
    if isinstance(value, CalibratedPrediction):
        raise CalibrationError(
            "this prediction is already calibrated; pass its raw .logits (with labels=...) or the original "
            "PredictionResult"
        )
    if is_prediction(value):
        resolved = prediction_labels(
            value,
            labels,
            error=CalibrationError,
            conflict_error=CalibrationError,
            what="temperature scaling",
            minimum=0,
        )
        return value.logits, resolved, value.sample_ids
    return value, None if labels is None else _labels(labels), None


def _labelled(logits: Any, labels: Any) -> tuple[Any, tuple[str, ...], Optional[Any]]:
    """:func:`_source` that insists on labels (explicit or the prediction's)."""
    raw, resolved, sample_ids = _source(logits, labels)
    if resolved is None:
        raise CalibrationError("pass labels=[...] naming the logits' columns in order (or a labelled prediction)")
    return raw, resolved, sample_ids


def _read_logits(raw: Any, labels: tuple[str, ...], *, copy: bool, allow_empty: bool = False) -> np.ndarray:
    array = _logits(raw, copy=copy, allow_empty=allow_empty)
    if array.shape[1] != len(labels):
        raise CalibrationError(
            f"labels must name one label per class: {len(labels)} labels for {array.shape[1]} classes"
        )
    return array


def _fit_config(min_temperature: Any, max_temperature: Any, tolerance: Any, max_iterations: Any) -> dict[str, Any]:
    low = _positive_float(min_temperature, "min_temperature")
    high = _positive_float(max_temperature, "max_temperature")
    if high <= low:
        raise CalibrationError(f"max_temperature ({high!r}) must exceed min_temperature ({low!r})")
    tol = _positive_float(tolerance, "tolerance")
    if not math.isfinite(1.0 / low):
        raise CalibrationError(f"min_temperature ({low!r}) is too small: its reciprocal overflows float64")
    if tol >= math.log(high / low):
        raise CalibrationError(
            f"tolerance ({tol!r}) must be smaller than the log-temperature search range "
            f"log(max_temperature / min_temperature) = {math.log(high / low):.6g}, or nothing would be fitted"
        )
    iterations = _count(max_iterations, "max_iterations", 1)
    return {
        "method": "temperature",
        "objective": "nll",
        "dtype": "float64",
        "min_temperature": low,
        "max_temperature": high,
        "tolerance": tol,
        "max_iterations": iterations,
    }


@dataclass(frozen=True, eq=False)
class TemperatureCalibrator(JsonArtifact):
    """A fitted scalar temperature bound to its label schema and model.

    Build one with :func:`fit_temperature`; reload one with :meth:`load` /
    :meth:`from_state`. ``labels`` are the ordered class names the logits'
    columns must carry, ``model_id`` the model they must come from,
    ``split_id`` the calibration split it was fitted on (and, when declared,
    ``train_split_id`` / ``test_split_id``, which must differ from it).
    ``fit_config`` / ``fit_result`` record how it was fitted (the result's
    ``nll_before`` / ``nll_after`` are the calibration split's exact,
    unfloored NLL, computed in the log domain). Equality,
    hashing and :attr:`id` follow the canonical state.
    """

    temperature: float
    labels: tuple[str, ...]
    model_id: str
    split_id: str
    train_split_id: Optional[str] = None
    test_split_id: Optional[str] = None
    fit_config: Mapping[str, Any] = field(default_factory=dict)
    fit_result: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature", _positive_float(self.temperature, "temperature"))
        labels = _labels(self.labels)
        if len(labels) < 2:
            raise CalibrationError(f"a categorical calibrator needs at least 2 labels, got {list(labels)}")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "model_id", _id(self.model_id, "model_id"))
        object.__setattr__(self, "split_id", _id(self.split_id, "split_id"))
        object.__setattr__(self, "train_split_id", _optional_id(self.train_split_id, "train_split_id"))
        object.__setattr__(self, "test_split_id", _optional_id(self.test_split_id, "test_split_id"))
        _check_splits(self.split_id, self.train_split_id, self.test_split_id)
        for name in ("fit_config", "fit_result"):
            object.__setattr__(self, name, _frozen(getattr(self, name), name))
        n_classes = self.fit_result.get("n_classes")
        if n_classes is not None and n_classes != len(labels):
            raise CalibrationError(f"fit_result n_classes {n_classes!r} contradicts the {len(labels)} labels")
        low, high = self.fit_config.get("min_temperature"), self.fit_config.get("max_temperature")
        if low is not None or high is not None:
            low = _positive_float(low, "fit_config.min_temperature")
            high = _positive_float(high, "fit_config.max_temperature")
            if not low <= self.temperature <= high:
                raise CalibrationError(
                    f"temperature {self.temperature!r} lies outside its own search range [{low!r}, {high!r}]"
                )

    @property
    def split(self) -> dict[str, Any]:
        """The split ids, and that their disjointness is unverified."""
        return _split_record(self.split_id, self.train_split_id, self.test_split_id, note=True)

    @property
    def id(self) -> str:
        """The calibrator id: :meth:`digest`."""
        return self.digest()

    # --- applying ---------------------------------------------------------------------------

    def _identity(self, labels: tuple[str, ...], model_id: Any, override: Any) -> Optional[Mapping[str, Any]]:
        """Compare the declared identity with the fitted one before anything
        is read; return the override record, or raise on a mismatch."""
        model = _id(model_id, "model_id")
        if override is not None:
            _id(override, "override")
        if len(labels) != len(self.labels):
            raise CalibrationMismatchError(
                f"{len(labels)} labels {list(labels)} for a calibrator fitted on {len(self.labels)} classes "
                f"{list(self.labels)}: a temperature fitted for one class count means nothing for another, "
                "and no override can accept it"
            )
        mismatches: dict[str, Any] = {}
        if labels != self.labels:
            mismatches["labels"] = {"expected": list(self.labels), "actual": list(labels)}
        if model != self.model_id:
            mismatches["model_id"] = {"expected": self.model_id, "actual": model}
        if not mismatches:
            return None
        if override is None:
            details = []
            if "labels" in mismatches:
                details.append(f"labels {list(labels)} differ from the fitted label order {list(self.labels)}")
            if "model_id" in mismatches:
                details.append(f"model id {model!r} differs from the fitted model {self.model_id!r}")
            raise CalibrationMismatchError(
                "; ".join(details) + ". Refit on this model's calibration split, or pass a named "
                "override=... to apply it anyway (the mismatch is recorded)"
            )
        return _override_record({"name": override, "mismatches": mismatches})

    def transform(
        self,
        logits: Any,
        *,
        model_id: str,
        labels: Optional[Sequence[str]] = None,
        override: Optional[str] = None,
        sample_ids: Any = None,
    ) -> CalibratedPrediction:
        """Calibrate ``logits`` — an ``(N, C)`` array or tensor, or a
        categorical ``nnx.prediction.PredictionResult``.

        ``sample_ids`` names the rows of an array (``0..N-1`` by default); a
        prediction keeps its own ids, and ``sample_ids=`` must then equal
        them.

        ``model_id`` and the column ``labels`` (taken from the prediction's
        spec when omitted) must equal the fitted ones; otherwise
        :class:`CalibrationMismatchError` is raised before the logits are
        read, unless ``override`` names the exception — then the result
        records the name and each mismatch. An override exists only to accept
        a mismatch: with none, the name is validated and nothing is recorded.
        Non-finite logits or a width that differs from the labels raise
        :class:`CalibrationError`. The input is never modified.
        """
        array, resolved, ids, record = self._checked_logits(
            logits,
            labels,
            model_id,
            override,
            copy=True,
            allow_empty=True,  # the result keeps these logits
            sample_ids=sample_ids,
        )
        return self._calibrate(array, resolved, ids, model_id, record)

    def _checked_logits(
        self,
        logits: Any,
        labels: Any,
        model_id: Any,
        override: Any,
        *,
        copy: bool,
        allow_empty: bool,
        sample_ids: Any = None,
        with_ids: bool = True,
    ) -> tuple[np.ndarray, tuple[str, ...], Optional[np.ndarray], Optional[Mapping[str, Any]]]:
        """Resolve the labels and sample ids and check the declared identity
        **before** the logits are read; then read and validate them. Given
        ``sample_ids`` name an array's rows and must equal a prediction's
        own; ``with_ids=False`` (a report) skips the ids and returns
        ``None`` for them."""
        raw, resolved, own_ids = _labelled(logits, labels)
        record = self._identity(resolved, model_id, override)
        ids = None
        if with_ids:
            if own_ids is not None and not same_ids(sample_ids, own_ids, error=CalibrationError):
                raise CalibrationError("sample_ids= contradicts the prediction's own sample_ids")
            given = own_ids if own_ids is not None else sample_ids
            ids = None if given is None else row_ids(given, None, error=CalibrationError)  # before the logits
        array = _read_logits(raw, resolved, copy=copy, allow_empty=allow_empty)
        if ids is not None and ids.shape[0] != array.shape[0]:
            raise CalibrationError(
                f"sample_ids must be one integer id per row ({array.shape[0]}), got shape {ids.shape} and dtype "
                f"{ids.dtype}"
            )
        return array, resolved, ids, record

    def _calibrate(
        self,
        array: np.ndarray,
        labels: tuple[str, ...],
        sample_ids: Optional[np.ndarray],
        model_id: str,
        record: Optional[Mapping[str, Any]],
    ) -> CalibratedPrediction:
        ids = np.arange(array.shape[0], dtype=np.int64) if sample_ids is None else sample_ids  # validated
        shifted = _shifted(array)
        calibrated = _softmax(shifted, self.temperature)  # a new array, so...
        raw = softmax_(shifted, 1)  # ...the raw softmax can reuse `shifted` in place
        return CalibratedPrediction(
            logits=array,
            probabilities=raw,
            calibrated_probabilities=calibrated,
            decoded=array.argmax(axis=1).astype(np.int64),
            labels=labels,
            sample_ids=ids,
            calibrator_id=self.id,
            model_id=model_id,
            temperature=self.temperature,
            override=record,
        )

    def report(
        self,
        logits: Any,
        targets: Any,
        *,
        model_id: str,
        split_id: str,
        labels: Optional[Sequence[str]] = None,
        override: Optional[str] = None,
        n_bins: int = 10,
        epsilon: Optional[float] = DEFAULT_EPSILON,
    ) -> CalibrationReport:
        """Held-out NLL, Brier, ECE and reliability bins before and after
        calibration.

        ``split_id`` names the held-out split and must differ from the
        calibration and training split ids. Identity checks match
        :meth:`transform`. The report's :attr:`CalibrationReport.outcome` is
        ``"improved"`` only when neither NLL nor Brier got worse and one got
        better; a worse result is reported as ``"worsened"`` / ``"mixed"``.
        """
        held_out = _id(split_id, "split_id")
        _check_held_out(held_out, self.split_id, self.train_split_id)  # before any work
        floor = _epsilon(epsilon)
        bins = _n_bins(n_bins)
        array, resolved, _, record = self._checked_logits(
            logits,
            labels,
            model_id,
            override,
            copy=False,
            allow_empty=False,  # only read
            with_ids=False,  # a report scores rows; their ids play no part
        )
        y = _targets(targets, array.shape[0], array.shape[1])  # before any softmax work
        shifted = _shifted(array)
        decoded = array.argmax(axis=1)
        # This calibrator's own float64 softmax output: no re-validation.
        after = _metrics(_softmax(shifted, self.temperature), y, bins, floor, decoded)
        before = _metrics(softmax_(shifted, 1), y, bins, floor, decoded)
        return CalibrationReport(
            calibrator_id=self.id,
            model_id=model_id,
            labels=resolved,
            temperature=self.temperature,
            split_id=held_out,
            calibration_split_id=self.split_id,
            n_samples=int(y.shape[0]),
            n_bins=bins,
            epsilon=floor,
            before=before,
            after=after,
            outcome=_outcome(before, after),
            override=record,
            train_split_id=self.train_split_id,
            test_split_id=self.test_split_id,
        )

    # --- serialization ----------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "kind": "temperature",
            "temperature": self.temperature,
            "labels": list(self.labels),
            "model_id": self.model_id,
            "split": _split_record(self.split_id, self.train_split_id, self.test_split_id),
            "fit": {"config": _thaw_config(self.fit_config), "result": _thaw_config(self.fit_result)},
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> TemperatureCalibrator:
        """Rebuild from :meth:`state`, rejecting anything malformed with
        :class:`CalibrationError`."""
        if not isinstance(state, Mapping):
            raise CalibrationError(f"a calibrator state is a mapping, got {type(state).__name__}")
        if state.get("format") != FORMAT:
            raise CalibrationError(f"unsupported calibration format {state.get('format')!r}; expected {FORMAT!r}")
        if state.get("kind") != "temperature":
            raise CalibrationError(f"unsupported calibrator kind {state.get('kind')!r}")
        check_keys(
            state,
            required=("format", "kind", "temperature", "labels", "model_id", "split"),
            optional=("fit",),
            what="calibrator state",
            error=CalibrationError,
        )
        split, fit = state["split"], state.get("fit", {})
        if not isinstance(split, Mapping) or not isinstance(fit, Mapping):
            raise CalibrationError("the calibrator state's 'split' and 'fit' must be mappings")
        check_keys(
            split,
            required=("calibration",),
            optional=("train", "test", "disjointness"),
            what="split",
            error=CalibrationError,
        )
        check_keys(fit, required=(), optional=("config", "result"), what="fit", error=CalibrationError)
        if split.get("disjointness", "unverified") != "unverified":
            raise CalibrationError(f"unsupported split disjointness {split.get('disjointness')!r}")
        with _malformed("calibrator state"):
            return TemperatureCalibrator(
                temperature=state["temperature"],
                labels=state["labels"],
                model_id=state["model_id"],
                split_id=split.get("calibration"),  # type: ignore[arg-type]
                train_split_id=split.get("train"),
                test_split_id=split.get("test"),
                fit_config=fit.get("config", {}),
                fit_result=fit.get("result", {}),
            )

    def digest(self) -> str:
        """``sha256:<hex>`` of the canonical state."""
        return self._digest

    @staticmethod
    def from_json(text: str) -> TemperatureCalibrator:
        return TemperatureCalibrator.from_state(parse_json(text, "calibrator", CalibrationError))

    @staticmethod
    def load(path: Union[str, os.PathLike[str]]) -> TemperatureCalibrator:
        return TemperatureCalibrator.from_json(read_text(path, "calibrator", CalibrationError))


# --- fitting -------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CalibrationFit:
    """The outcome of :func:`fit_temperature`: ``status == "ok"`` with a
    ``calibrator``, or ``"failed"`` with ``calibrator=None`` and a ``reason``.
    ``config`` and ``split`` are recorded either way (read-only). Compared
    and hashed by identity; compare the calibrators themselves."""

    status: str
    calibrator: Optional[TemperatureCalibrator]
    reason: Optional[str]
    config: Mapping[str, Any]
    split: Mapping[str, Any]

    def __post_init__(self) -> None:
        ok = self.status == "ok" and isinstance(self.calibrator, TemperatureCalibrator) and self.reason is None
        failed = self.status == "failed" and self.calibrator is None and isinstance(self.reason, str) and self.reason
        if not (ok or failed):
            raise CalibrationError(
                "a CalibrationFit is status='ok' with a calibrator and no reason, or status='failed' with a reason "
                f"and no calibrator; got status={self.status!r}, calibrator={self.calibrator!r}, reason={self.reason!r}"
            )

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def require(self) -> TemperatureCalibrator:
        """The calibrator, or :class:`CalibrationFitError` with the reason."""
        if self.calibrator is None:
            raise CalibrationFitError(f"temperature calibration failed: {self.reason}")
        return self.calibrator


def fit_temperature(
    logits: Any,
    targets: Any,
    *,
    model_id: str,
    split_id: str,
    labels: Optional[Sequence[str]] = None,
    train_split_id: Optional[str] = None,
    test_split_id: Optional[str] = None,
    min_temperature: float = 1e-3,
    max_temperature: float = 1e3,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> CalibrationFit:
    """Fit a scalar temperature on the calibration split ``split_id``.

    ``logits`` are categorical ``(N, C)`` raw scores (array, tensor, or a
    categorical ``nnx.prediction.PredictionResult``, whose spec supplies the
    ``labels``) and ``targets`` their integer class indices. The NLL of
    ``softmax(logits / T)`` is minimized over ``T`` in
    ``[min_temperature, max_temperature]`` in float64 on a copy of the
    logits, by bisection on its (monotone) slope in ``1 / T`` until the
    log-temperature bracket is narrower than ``tolerance``.

    Raises :class:`CalibrationError` for non-finite logits, ``C < 2``,
    out-of-range targets, labels that do not name each class once, and a
    ``split_id`` equal to ``train_split_id`` or ``test_split_id``. Returns a
    failed :class:`CalibrationFit` — never a calibrator with a NaN or
    boundary temperature — when the optimum lies outside the search range (a
    separable split, logits no better than uniform), the NLL does not depend
    on ``T`` (constant logits), or the search does not converge.
    """
    calibration_split = _id(split_id, "split_id")
    train = _optional_id(train_split_id, "train_split_id")
    test = _optional_id(test_split_id, "test_split_id")
    _check_splits(calibration_split, train, test)
    model = _id(model_id, "model_id")
    config = _fit_config(min_temperature, max_temperature, tolerance, max_iterations)
    raw, resolved, _ = _labelled(logits, labels)
    array = _read_logits(raw, resolved, copy=False)  # only read: _shifted() below makes the one float64 copy
    y = _targets(targets, array.shape[0], array.shape[1])
    split = _frozen(_split_record(calibration_split, train, test, note=True), "split")

    def failed(reason: str) -> CalibrationFit:
        return CalibrationFit("failed", None, reason, _frozen(config, "fit config"), split)

    n_samples, n_classes = array.shape
    shifted = _shifted(array)
    del array
    if not shifted.any():  # every row's logits are equal: softmax is uniform at any temperature
        return failed("the calibration NLL does not depend on temperature (every row's logits are constant)")
    slope = _Slope(shifted, y)
    beta_low, beta_high = 1.0 / config["max_temperature"], 1.0 / config["min_temperature"]
    slope_low, slope_high = slope(beta_low), slope(beta_high)
    if not (math.isfinite(slope_low) and math.isfinite(slope_high)):
        return failed("the calibration NLL is not finite over the temperature range")
    if slope_low == 0.0 and slope_high == 0.0:
        return failed(
            "the calibration NLL is flat over the whole temperature range: the logits' scale saturates the "
            "softmax (e.g. classes masked with huge negative values), so no temperature can be fitted"
        )
    if slope_low >= 0.0:
        return failed(
            f"the optimal temperature is at or above max_temperature={config['max_temperature']!r}: the logits "
            "are no better than uniform on the calibration split (check for targets on masked classes), or "
            "their scale needs a larger max_temperature"
        )
    if slope_high <= 0.0:
        return failed(
            f"the optimal temperature is at or below min_temperature={config['min_temperature']!r}: the "
            "calibration split is separable (its NLL keeps falling as the temperature shrinks)"
        )
    low, high = math.log(beta_low), math.log(beta_high)
    iterations = 0
    while high - low > config["tolerance"]:
        middle = 0.5 * (low + high)
        if not low < middle < high:  # the bracket is at float64 resolution: converged
            break
        if iterations == config["max_iterations"]:
            return failed(
                f"the temperature search did not converge within max_iterations={config['max_iterations']} "
                f"(log-temperature bracket {high - low:.3g} > tolerance {config['tolerance']:g})"
            )
        iterations += 1
        if slope(math.exp(middle)) < 0.0:
            low = middle
        else:
            high = middle
    temperature = 1.0 / math.exp(0.5 * (low + high))
    if not config["min_temperature"] < temperature < config["max_temperature"]:
        # Only reachable when the optimum sits within float64 resolution of a bound.
        return failed(
            f"the fitted temperature {temperature!r} reached the search bound "
            f"[{config['min_temperature']!r}, {config['max_temperature']!r}]; widen the range"
        )
    with np.errstate(over="ignore"):
        nll_before, nll_after = _mean_nll(shifted, y, 1.0), _mean_nll(shifted, y, 1.0 / temperature)
    if not (math.isfinite(nll_before) and math.isfinite(nll_after)):
        return failed(
            f"the calibration NLL is not finite at temperature {temperature!r}: some true-class logits lie "
            "beyond the float64 range below their row maximum (extreme masked logits)"
        )
    calibrator = TemperatureCalibrator(
        temperature=temperature,
        labels=resolved,
        model_id=model,
        split_id=calibration_split,
        train_split_id=train,
        test_split_id=test,
        fit_config=config,
        fit_result={
            "n_samples": int(n_samples),
            "n_classes": int(n_classes),
            "iterations": iterations,
            "nll_before": nll_before,
            "nll_after": nll_after,
        },
    )
    return CalibrationFit("ok", calibrator, None, calibrator.fit_config, split)


# --- reports -------------------------------------------------------------------------------------


def _encode(value: float) -> Any:
    """JSON form of an NLL: a finite float as a number, ``+inf`` (the exact
    NLL of a zero true-class probability) as ``"inf"``. Anything else is
    left for ``allow_nan=False`` to refuse."""
    return "inf" if value == math.inf else value


@dataclass(frozen=True)
class CalibrationMetrics:
    """NLL, Brier, expected calibration error and reliability bins of one
    set of probabilities."""

    nll: float
    brier: float
    ece: Optional[float]
    bins: tuple[ReliabilityBin, ...]

    def __post_init__(self) -> None:
        nll = self.nll if self.nll == math.inf else _finite(self.nll, "nll")
        if nll < 0:
            raise CalibrationError(f"nll must be >= 0 or inf, got {self.nll!r}")
        brier = _finite(self.brier, "brier")
        if brier < 0:  # rows may exceed a sum of 1 by the rounding tolerance, so no upper bound of 2
            raise CalibrationError(f"brier must be >= 0, got {self.brier!r}")
        bins = tuple(self.bins)
        if not all(isinstance(b, ReliabilityBin) for b in bins):
            raise CalibrationError("calibration metrics' bins must be ReliabilityBin objects")
        ece = None if self.ece is None else _finite(self.ece, "ece")
        expected = expected_calibration_error(bins)
        if (ece is None) != (expected is None) or (
            ece is not None and expected is not None and not math.isclose(ece, expected, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise CalibrationError(f"ece {self.ece!r} does not match its bins, which give {expected!r}")
        for name, value in (("nll", float(nll)), ("brier", brier), ("ece", ece), ("bins", bins)):
            object.__setattr__(self, name, value)

    @staticmethod
    def of(
        probabilities: Any, targets: Any, *, n_bins: int = 10, epsilon: Optional[float] = DEFAULT_EPSILON
    ) -> CalibrationMetrics:
        """All four metrics of ``probabilities``, validated once."""
        floor, count = _epsilon(epsilon), _n_bins(n_bins)
        p = _probabilities(probabilities)
        return _metrics(p, _targets(targets, p.shape[0], p.shape[1]), count, floor)

    def state(self) -> dict[str, Any]:
        return {
            "nll": _encode(self.nll),
            "brier": self.brier,
            "ece": self.ece,
            "bins": [b.state() for b in self.bins],
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> CalibrationMetrics:
        """Rebuild from :meth:`state`; malformed entries raise :class:`CalibrationError`."""
        with _malformed("calibration metrics"):
            nll = state["nll"]
            return CalibrationMetrics(  # the values are validated by __post_init__
                nll=math.inf if nll == "inf" else nll,
                brier=state["brier"],
                ece=state["ece"],
                bins=tuple(ReliabilityBin.from_state(b) for b in state["bins"]),
            )


def _metrics(
    p: np.ndarray, y: np.ndarray, n_bins: int, floor: Optional[float], predicted: Optional[np.ndarray] = None
) -> CalibrationMetrics:
    """The metrics of already-validated float64 probabilities and targets."""
    bins = _bins(p, y, n_bins, predicted)
    return CalibrationMetrics(
        nll=_nll(p, y, floor), brier=_brier(p, y), ece=expected_calibration_error(bins), bins=bins
    )


_SAME_REL, _SAME_ABS = 1e-9, 1e-12


def _direction(before: float, after: float) -> int:
    """-1 better, +1 worse, 0 the same — within float64 rounding noise
    (relative 1e-9, absolute 1e-12), so a last-bit difference never decides
    the outcome."""
    if before == after or math.isclose(before, after, rel_tol=_SAME_REL, abs_tol=_SAME_ABS):  # inf == inf too
        return 0
    return -1 if after < before else 1


def _outcome(before: CalibrationMetrics, after: CalibrationMetrics) -> str:
    moves = {_direction(before.nll, after.nll), _direction(before.brier, after.brier)}
    if moves == {0}:
        return "unchanged"
    if 1 not in moves:
        return "improved"
    if -1 not in moves:
        return "worsened"
    return "mixed"


_REPORT_KEYS = frozenset(
    {"format", "kind", "calibrator_id", "model_id", "labels", "temperature", "split", "n_samples", "n_bins"}
    | {"epsilon", "before", "after", "outcome", "override"}
)


@dataclass(frozen=True, eq=False)
class CalibrationReport(JsonArtifact):
    """Held-out metrics before and after calibration.

    ``outcome`` is ``"improved"`` (neither NLL nor Brier worse, one better),
    ``"worsened"`` (neither better, one worse), ``"unchanged"`` or
    ``"mixed"``; :attr:`improved` is true only for ``"improved"``.
    ``override`` is the named override that accepted a label / model-id
    mismatch, or ``None``. Equality follows the canonical state, so a
    reloaded calibrator reproduces an equal report.
    """

    calibrator_id: str
    model_id: str
    labels: tuple[str, ...]
    temperature: float
    split_id: str
    calibration_split_id: str
    n_samples: int
    n_bins: int
    epsilon: Optional[float]
    before: CalibrationMetrics
    after: CalibrationMetrics
    outcome: str
    override: Optional[Mapping[str, Any]] = None
    train_split_id: Optional[str] = None
    test_split_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not (isinstance(self.before, CalibrationMetrics) and isinstance(self.after, CalibrationMetrics)):
            raise CalibrationError("a report's before / after must be CalibrationMetrics")
        _id(self.calibrator_id, "calibrator_id")
        _id(self.model_id, "model_id")
        object.__setattr__(self, "labels", _labels(self.labels))
        if len(self.labels) < 2:
            raise CalibrationError(f"a categorical report needs at least 2 labels, got {list(self.labels)}")
        object.__setattr__(self, "override", _override_record(self.override))
        object.__setattr__(self, "temperature", _positive_float(self.temperature, "temperature"))
        _id(self.split_id, "split_id")
        _id(self.calibration_split_id, "calibration_split_id")
        _optional_id(self.train_split_id, "train_split_id")
        _optional_id(self.test_split_id, "test_split_id")
        _check_splits(self.calibration_split_id, self.train_split_id, self.test_split_id)
        object.__setattr__(self, "n_samples", _count(self.n_samples, "n_samples", 1))
        object.__setattr__(self, "n_bins", _n_bins(self.n_bins))
        object.__setattr__(self, "epsilon", _epsilon(self.epsilon))
        if self.outcome not in _OUTCOMES:
            raise CalibrationError(f"report outcome must be one of {_OUTCOMES}, got {self.outcome!r}")
        _check_held_out(self.split_id, self.calibration_split_id, self.train_split_id)
        derived = _outcome(self.before, self.after)
        if self.outcome != derived:
            raise CalibrationError(f"report outcome {self.outcome!r} contradicts its metrics, which give {derived!r}")
        edges = [(i / self.n_bins, (i + 1) / self.n_bins) for i in range(self.n_bins)]
        for name in ("before", "after"):
            metrics = getattr(self, name)
            if [(b.lower, b.upper) for b in metrics.bins] != edges or sum(
                b.count for b in metrics.bins
            ) != self.n_samples:
                raise CalibrationError(
                    f"the report's {name} bins must be the {self.n_bins} equal-width bins [i/{self.n_bins}, "
                    f"(i+1)/{self.n_bins}) holding {self.n_samples} rows"
                )

    @property
    def improved(self) -> bool:
        return self.outcome == "improved"

    def summary(self) -> str:
        """One line that states the outcome — never success when the
        calibrated probabilities are worse."""
        changes = (
            f"NLL {self.before.nll:.6g} -> {self.after.nll:.6g} and Brier {self.before.brier:.6g} -> "
            f"{self.after.brier:.6g} on split {self.split_id!r} ({self.n_samples} samples, temperature "
            f"{self.temperature:.6g})"
        )
        text = {
            "improved": f"calibration improved held-out {changes}",
            "worsened": f"calibration worsened held-out {changes}; keep the uncalibrated probabilities",
            "unchanged": f"calibration left held-out {changes} unchanged",
            "mixed": f"calibration gave mixed held-out results, {changes}; it is not better on both",
        }[self.outcome]
        if self.override is not None:
            fields = ", ".join(sorted(self.override["mismatches"]))
            text += f"; override {self.override['name']!r} accepted a mismatched {fields}"
        return text

    def state(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "kind": "report",
            "calibrator_id": self.calibrator_id,
            "model_id": self.model_id,
            "labels": list(self.labels),
            "temperature": self.temperature,
            "split": {
                "evaluation": self.split_id,
                "calibration": self.calibration_split_id,
                "train": self.train_split_id,
                "test": self.test_split_id,
            },
            "n_samples": self.n_samples,
            "n_bins": self.n_bins,
            "epsilon": self.epsilon,
            "before": self.before.state(),
            "after": self.after.state(),
            "outcome": self.outcome,
            "override": None if self.override is None else _thaw_config(self.override),
        }

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> CalibrationReport:
        if not isinstance(state, Mapping) or state.get("format") != FORMAT or state.get("kind") != "report":
            raise CalibrationError(f"not a {FORMAT} calibration report")
        check_keys(  # epsilon: null is exact, absent is malformed
            state,
            required=_REPORT_KEYS - {"override"},
            optional=("override",),
            what="calibration report",
            error=CalibrationError,
        )
        split = state["split"]
        if not isinstance(split, Mapping) or set(split) != {"evaluation", "calibration", "train", "test"}:
            raise CalibrationError(
                f"a report's split holds exactly 'evaluation', 'calibration', 'train' and 'test', got {split!r}"
            )
        with _malformed("calibration report"):
            return CalibrationReport(  # scalar fields are validated by __post_init__
                calibrator_id=state["calibrator_id"],
                model_id=state["model_id"],
                labels=state["labels"],
                temperature=state["temperature"],
                split_id=split["evaluation"],
                calibration_split_id=split["calibration"],
                n_samples=state["n_samples"],
                n_bins=state["n_bins"],
                epsilon=state["epsilon"],
                before=CalibrationMetrics.from_state(state["before"]),
                after=CalibrationMetrics.from_state(state["after"]),
                outcome=state["outcome"],
                override=state.get("override"),
                train_split_id=split["train"],
                test_split_id=split["test"],
            )

    @staticmethod
    def from_json(text: str) -> CalibrationReport:
        return CalibrationReport.from_state(parse_json(text, "calibration report", CalibrationError))

    @staticmethod
    def load(path: Union[str, os.PathLike[str]]) -> CalibrationReport:
        return CalibrationReport.from_json(read_text(path, "calibration report", CalibrationError))


# --- model identity ------------------------------------------------------------------------------


def model_fingerprint(model: Any) -> str:
    """``sha256:<hex>`` of a model's weights: every ``state_dict()`` entry's
    name, dtype, shape and little-endian bytes, in name order (the same on
    every host).

    ``model`` is an ``nn.Module`` or anything with a ``.net`` module (an
    ``NNModel``). Use it as a ``model_id`` so a calibrator refuses logits
    from a model whose weights changed; a declared id such as
    ``f"{run.id}:BEST"`` works too, but is only as reliable as its naming.
    """
    import torch

    # A module is fingerprinted whole — never through a submodule that
    # happens to be called ``net``; only a non-module wrapper (an NNModel)
    # is unwrapped to its ``.net``.
    module = model if isinstance(model, torch.nn.Module) else getattr(model, "net", None)
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"model_fingerprint needs a torch module (or a .net) with state_dict(), got {model!r}")
    # Wrappers that only prefix the state_dict keys — torch.compile's
    # `_orig_mod.`, DataParallel / DistributedDataParallel's `module.` — are
    # unwrapped, so the same weights keep the same fingerprint.
    parallel = (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    while True:
        inner = module.module if isinstance(module, parallel) else getattr(module, "_orig_mod", None)
        if not isinstance(inner, torch.nn.Module):
            break
        module = inner
    return _state_fingerprint(cast(Mapping[str, Any], module.state_dict()))


_FINGERPRINT_SEED = b"nnx.calibration.model\0"


def _fingerprint_header(name: str, kind: str, dtype: Any, shape: Sequence[int], numel: int) -> bytes:
    """The bytes :func:`model_fingerprint` hashes before one tensor's data."""
    return f"{name}\0{kind}\0{dtype}\0{tuple(shape)}\0{numel}\0".encode()


def _state_fingerprint(entries: Mapping[str, Any]) -> str:
    """:func:`model_fingerprint` of a ``state_dict()`` itself — the same
    digest, for weights held without their module (``nnx.bundles``)."""
    import torch

    digest = hashlib.sha256(_FINGERPRINT_SEED)
    for name, value in sorted(entries.items()):
        if isinstance(value, torch.Tensor):
            kind = "tensor"
            if value.is_quantized:  # the integers plus every quantization parameter
                scheme = value.qscheme()
                if scheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
                    params = f"{value.q_scale()!r},{value.q_zero_point()}"
                else:
                    scales = value.q_per_channel_scales().to(torch.float64).tolist()
                    params = f"{value.q_per_channel_axis()},{scales!r},{value.q_per_channel_zero_points().tolist()}"
                kind = f"quantized\0{value.dtype}\0{scheme}\0{params}"
                value = value.int_repr()
            try:  # conjugate / negative views are materialized; sparse or meta tensors cannot be read
                flat = value.detach().to("cpu").resolve_conj().resolve_neg().contiguous().reshape(-1)
                raw = flat.view(torch.uint8).numpy().data  # the buffer itself: no extra copy
            except (RuntimeError, NotImplementedError, ValueError) as exc:  # ValueError: lazy parameters
                raise TypeError(
                    f"model_fingerprint cannot hash the {value.layout} / {value.device} tensor {name!r}; "
                    "pass a declared model_id instead"
                ) from exc
            if sys.byteorder == "big" and flat.element_size() > 1:  # hash little-endian bytes on every host
                unit = flat.element_size() // (2 if flat.is_complex() else 1)
                raw = flat.view(torch.uint8).reshape(-1, unit).flip(-1).contiguous().numpy().data
            digest.update(_fingerprint_header(name, kind, flat.dtype, tuple(value.shape), flat.numel()))
            digest.update(raw)
            continue
        try:  # get_extra_state() entries: JSON-like values (str keys, finite numbers) hash canonically
            payload = _canonical_json(_thaw_config(_freeze_config(value, "", owner="extra state")))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"model_fingerprint cannot hash the non-tensor state entry {name!r} ({type(value).__name__}); "
                "pass a declared model_id instead"
            ) from exc
        digest.update(f"{name}\0extra\0".encode() + payload + b"\0")
    return f"sha256:{digest.hexdigest()}"
