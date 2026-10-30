"""Probability-aware prediction results.

``NNModel.predict()`` returns raw logits and decoded classes. Callers who
need probabilities — for NLL, Brier score, calibration or a decision
policy — can instead declare the task explicitly with a
:class:`ProbabilitySpec` and get a :class:`PredictionResult` carrying the
raw logits, the probabilities, the decoded values and each row's sample
identity::

    spec = ProbabilitySpec(kind="categorical", class_axis=1, labels=("cat", "dog", "fox"))
    result = model.predict_proba(X, spec)
    result.probabilities.sum(axis=1)       # 1.0 per row
    result.decoded_labels()                # array(['dog', 'cat', ...])

Two task kinds are supported, and the kind is always declared — never
inferred from tensor shape or loss:

* ``"categorical"`` — mutually exclusive classes: softmax over
  ``class_axis`` (each row sums to 1) and argmax decoding.
* ``"bernoulli"`` — independent binary outputs (multi-label, or a single
  binary logit): element-wise sigmoid (rows are *not* normalized) and
  ``logit >= 0`` decoding into 0/1 indicators. These are indicators, not
  class indices: :attr:`PredictionResult.class_indices` refuses them.

The legacy ``PredictResult`` and ``predict()`` are unchanged.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ._probability import class_labels, read_array, row_ids, softmax_

__all__ = [
    "PredictionResult",
    "PredictionValidationError",
    "NonFiniteLogitsError",
    "ProbabilitySpec",
    "prediction_from_logits",
]

_KINDS = ("categorical", "bernoulli")


class PredictionValidationError(ValueError):
    """An invalid :class:`ProbabilitySpec`, or logits / sample ids that do
    not satisfy one (wrong label count, invalid class axis, non-finite
    logits, misaligned sample ids)."""


class NonFiniteLogitsError(PredictionValidationError):
    """The logits hold NaN or +-inf: the model's output, not the request,
    is at fault (a decision provider reports it as a provider failure)."""


@dataclass(frozen=True, slots=True)
class ProbabilitySpec:
    """Explicit declaration of how raw logits become probabilities.

    Args:
        kind: ``"categorical"`` (softmax over ``class_axis``, argmax
            decoding) or ``"bernoulli"`` (independent sigmoid per output,
            ``logit >= 0`` decoding).
        class_axis: axis of the logits that indexes classes / outputs.
            Negative values count from the end; ``-1`` (default) fits
            ``(N, C)`` classifier logits and class-last ``(B, T, V)``
            language-model logits, while class-first outputs such as
            ``(N, C, H, W)`` need ``class_axis=1``. Axis 0 is always the
            sample axis and cannot be the class axis.
        labels: optional ordered class / output names, one per entry
            along ``class_axis``; unique, non-empty strings, in a sequence,
            NumPy array or pandas ``Index`` (stored as a tuple of ``str``).
            A string, mapping or set is refused.

    Serializes with :meth:`state` / :meth:`from_state` (plain YAML-safe
    types, label order preserved).
    """

    kind: str
    class_axis: int = -1
    labels: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise PredictionValidationError(
                f"ProbabilitySpec kind must be one of {', '.join(repr(k) for k in _KINDS)}, got {self.kind!r}"
            )
        axis = self.class_axis
        if isinstance(axis, bool) or not isinstance(axis, numbers.Integral):
            raise PredictionValidationError(
                f"ProbabilitySpec class_axis must be an integer axis, got {axis!r} of type {type(axis).__name__}"
            )
        object.__setattr__(self, "class_axis", int(axis))
        if self.labels is None:
            return
        # The shared rule (NumPy arrays and pandas indexes accepted; a string,
        # mapping or set refused), stored as plain str: numpy.str_ is not YAML-safe.
        labels = class_labels(self.labels, error=PredictionValidationError, minimum=0, what="ProbabilitySpec labels")
        minimum = 2 if self.kind == "categorical" else 1
        if len(labels) < minimum:
            raise PredictionValidationError(
                f"a {self.kind} ProbabilitySpec needs at least {minimum} label(s), got {len(labels)}"
            )
        object.__setattr__(self, "labels", labels)

    def resolve_class_axis(self, ndim: int) -> int:
        """Return ``class_axis`` as a non-negative axis of an ``ndim``-D
        logits array, or raise if it is out of range or the sample axis."""
        if ndim < 2:
            raise PredictionValidationError(f"logits need a sample axis and a class axis (at least 2-D), got {ndim}-D")
        axis = self.class_axis + ndim if self.class_axis < 0 else self.class_axis
        if not 0 <= axis < ndim:
            raise PredictionValidationError(
                f"class_axis {self.class_axis} is out of range for {ndim}-D logits (valid: {-ndim}..{ndim - 1})"
            )
        if axis == 0:
            raise PredictionValidationError(
                "class_axis resolves to axis 0, the sample axis; declare the class axis explicitly"
            )
        return axis

    def state(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "class_axis": self.class_axis}
        if self.labels is not None:
            d["labels"] = list(self.labels)
        return d

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> ProbabilitySpec:
        # Labels go through the constructor's validation unchanged, so a
        # hand-edited `labels: abc` is rejected rather than split into letters.
        return ProbabilitySpec(kind=state["kind"], class_axis=state.get("class_axis", -1), labels=state.get("labels"))


@dataclass(frozen=True, eq=False)
class PredictionResult:
    """Probability-aware prediction for ``N`` samples.

    Attributes:
        logits: raw network output, exactly what ``predict().logits``
            returns for the same input.
        probabilities: same shape as ``logits``; softmax over the class
            axis (categorical, rows sum to 1) or element-wise sigmoid
            (bernoulli, not normalized). ``None`` for a continuous
            (regression-task) result, which has no probabilities.
        decoded: categorical — argmax class indices (``logits`` without
            the class axis); bernoulli — 0/1 indicators shaped like
            ``logits`` (``logit >= 0``, or the task's threshold);
            continuous — the predicted values themselves.
        sample_ids: ``int64[N]`` identity of each row: the input row index
            for arrays, tensors and ordinary loaders, and the global node
            index (``input_id``) for seed rows of a graph loader.
        spec: the :class:`ProbabilitySpec` that produced the result, or
            ``None`` for a continuous result (``NNModel.predict_proba``
            on a regression task).
    """

    logits: np.ndarray
    probabilities: Optional[np.ndarray]
    decoded: np.ndarray
    sample_ids: np.ndarray
    spec: Optional[ProbabilitySpec]

    @property
    def kind(self) -> str:
        """``"categorical"``, ``"bernoulli"`` or ``"continuous"`` (no spec)."""
        return self.spec.kind if self.spec is not None else "continuous"

    @property
    def class_axis(self) -> int:
        """The class axis of :attr:`logits` / :attr:`probabilities`, non-negative."""
        if self.spec is None:
            raise TypeError("class_axis is only defined for probabilistic results; a continuous result has no classes")
        return self.spec.resolve_class_axis(self.logits.ndim)

    @property
    def labels(self) -> Optional[tuple[str, ...]]:
        return self.spec.labels if self.spec is not None else None

    @property
    def class_indices(self) -> np.ndarray:
        """Decoded class indices — categorical results only.

        Bernoulli outputs are independent per-output indicators, not
        mutually exclusive classes, so they are refused here (and must
        never be fed to class-index consumers such as
        ``VisUtils.confusion_matrix``)."""
        if self.spec is None or self.spec.kind != "categorical":
            raise TypeError(
                "class_indices is only defined for categorical predictions; bernoulli `decoded` holds "
                "independent 0/1 indicators per output, not class indices"
            )
        return self.decoded

    def decoded_labels(self) -> np.ndarray:
        """Decoded class names (categorical results with ``spec.labels``)."""
        labels = self.labels
        if labels is None:
            raise PredictionValidationError("decoded_labels() needs ProbabilitySpec.labels")
        return np.asarray(labels, dtype=object)[self.class_indices]


def _as_array(logits: Any) -> np.ndarray:
    array = read_array(logits, copy=False, error=PredictionValidationError)  # shared with nnx.calibration
    if array.dtype.kind not in "fiu":
        raise PredictionValidationError(f"logits must be a real numeric array, got dtype {array.dtype}")
    return array


def _check_spec_fits(logits: Any, spec: ProbabilitySpec) -> int:
    """Check ``spec`` against the logits' shape (class axis in range and
    not the sample axis, label count, categorical class count) and return
    the non-negative class axis. Shape-only: cheap enough to run on a
    loader's first batch before the rest of inference."""
    shape = np.shape(logits)
    axis = spec.resolve_class_axis(len(shape))
    n_classes = shape[axis]
    if spec.labels is not None and len(spec.labels) != n_classes:
        raise PredictionValidationError(
            f"ProbabilitySpec has {len(spec.labels)} label(s) but the logits have {n_classes} "
            f"entries along class_axis {spec.class_axis}"
        )
    if spec.kind == "categorical" and n_classes < 2:
        raise PredictionValidationError(
            f"categorical predictions need at least 2 classes along class_axis {spec.class_axis}, got {n_classes}"
        )
    return axis


def prediction_from_logits(
    logits: Any,
    spec: ProbabilitySpec,
    *,
    sample_ids: Any = None,
) -> PredictionResult:
    """Build a :class:`PredictionResult` from raw logits.

    ``logits`` is an array or tensor whose axis 0 indexes samples and whose
    ``spec.class_axis`` indexes classes / outputs. Raises
    :class:`PredictionValidationError` when the logits are no numeric array
    (a ragged list, say), the class axis is invalid, the label count
    differs from the class-axis size, any logit is NaN or ±inf (masked
    logits must use a large finite negative value), or ``sample_ids`` is
    not one integer id per sample (ragged ids included). Probabilities are
    computed stably (max-shifted softmax, ``exp(-|x|)`` sigmoid) in the
    logits' float precision and returned in their dtype — float16 is
    computed in float32 and returned as float16, bfloat16 tensors are
    upcast to float32, integer logits give float64.
    """
    if not isinstance(spec, ProbabilitySpec):
        raise TypeError(f"spec must be a ProbabilitySpec, got {type(spec).__name__}")
    array = _as_array(logits)
    axis = _check_spec_fits(array, spec)
    finite = np.isfinite(array)
    if not finite.all():
        raise NonFiniteLogitsError(
            f"logits contain {int(finite.size - np.count_nonzero(finite))} non-finite value(s) (NaN or ±inf); "
            "probabilities would be undefined — replace masked logits with a large finite negative value"
        )
    del finite

    # One working copy updated in place, in the logits' own float precision
    # (float16 widened to float32, integers to float64), so large class-last
    # LM logits are not multiplied in memory.
    compute = np.dtype(np.float32) if array.dtype == np.float16 else array.dtype
    if compute.kind != "f":
        compute = np.dtype(np.float64)
    work = array.astype(compute, copy=True)
    if spec.kind == "categorical":
        softmax_(work, axis)  # shared with nnx.calibration: one max-shifted softmax
        decoded = array.argmax(axis=axis).astype(np.int64)
    else:
        positive = work >= 0
        np.abs(work, out=work)
        np.negative(work, out=work)
        np.exp(work, out=work)  # e = exp(-|x|), in (0, 1]
        # sigmoid(x) = 1 / (1 + e) for x >= 0 and e / (1 + e) otherwise.
        np.divide(np.where(positive, 1.0, work), 1.0 + work, out=work)
        decoded = positive.astype(np.int64)
    out_dtype = array.dtype if array.dtype.kind == "f" else np.dtype(np.float64)
    probabilities = work if work.dtype == out_dtype else work.astype(out_dtype)

    # 0..N-1 by default; ragged, misaligned, non-integer and over-int64 ids are refused, never wrapped.
    ids = row_ids(sample_ids, array.shape[0], error=PredictionValidationError)
    return PredictionResult(logits=array, probabilities=probabilities, decoded=decoded, sample_ids=ids, spec=spec)
