"""Probability arithmetic shared by metrics, predictions and calibration.
Internal; NumPy only.

One implementation of each piece, so the modules that use it cannot drift
apart:

- ``softmax_`` — the max-shifted softmax of ``nnx.prediction``; a
  temperature-1 calibrator (``nnx.calibration``) repeats it bit for bit.
- ``nll_terms`` / ``brier_terms`` — the per-row (categorical) or
  per-output (Bernoulli) terms of the named ``nll`` / ``brier`` metrics
  (``nnx.monitors``) and of ``nnx.calibration``'s
  ``negative_log_likelihood`` / ``brier_score``.
- ``NLL_EPSILON`` — the probability floor of both NLLs.
- ``to_numpy`` — the tensor-to-NumPy conversion of predictions and
  calibration (duck-typed: torch is never imported here).
- ``categorical_probabilities`` / ``class_targets`` / ``row_ids`` — the
  validation of categorical ``(N, C)`` probabilities, integer class targets
  and per-row sample ids shared by calibration and abstention, raising the
  caller's own error type.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

NLL_EPSILON = 1e-12
SUM_TOLERANCE = 1e-6
"""Smallest row-sum tolerance of :func:`categorical_probabilities`."""


def is_tensor(value: Any) -> bool:
    """Whether ``value`` is a ``torch.Tensor`` (checked by type, never by
    duck typing, and without importing torch)."""
    return any(cls.__name__ == "Tensor" and cls.__module__ == "torch" for cls in type(value).__mro__)


def to_numpy(value: Any, *, copy: bool) -> np.ndarray:
    """A NumPy array of an array-like or a torch tensor (detached, moved to
    the CPU, bfloat16 upcast losslessly to float32). ``copy=True`` always
    returns memory the caller owns; ``copy=False`` may share it with a CPU
    tensor or the input array."""
    if is_tensor(value):
        tensor, fresh = value.detach(), False
        if str(tensor.dtype) == "torch.bfloat16":  # no NumPy equivalent
            tensor, fresh = tensor.float(), True
        array = tensor.cpu().numpy()
        # A CPU tensor shares memory with its array; an upcast or device tensor is already a copy.
        return array.copy() if copy and tensor.device.type == "cpu" and not fresh else array
    return np.array(value, copy=True) if copy else np.asarray(value)


def softmax_(work: np.ndarray, axis: int) -> np.ndarray:
    """Softmax over ``axis``, in place on a floating array the caller owns:
    subtract the maximum, exponentiate, normalize. Returns ``work``."""
    work -= work.max(axis=axis, keepdims=True)
    np.exp(work, out=work)
    work /= work.sum(axis=axis, keepdims=True)
    return work


def is_categorical(target: np.ndarray, probabilities: np.ndarray) -> bool:
    """Categorical probabilities carry one more (class) axis than targets."""
    return probabilities.ndim == target.ndim + 1


def nll_terms(target: np.ndarray, probabilities: np.ndarray, epsilon: Optional[float] = NLL_EPSILON) -> np.ndarray:
    """``-log p(target)`` per row (categorical: class axis last) or per
    output (Bernoulli), in float64. Probabilities are clipped to
    ``[epsilon, 1]`` first; ``epsilon=None`` is exact (``+inf`` at 0)."""
    low = 0.0 if epsilon is None else epsilon
    with np.errstate(divide="ignore", invalid="ignore"):
        if is_categorical(target, probabilities):  # read one entry per row, then widen
            true = np.take_along_axis(probabilities, target.astype(np.int64)[..., None], axis=-1)[..., 0]
            return -np.log(np.clip(true.astype(np.float64), low, 1.0))
        p = probabilities.astype(np.float64)
        t = target.astype(np.float64)
        positive = t * np.log(np.clip(p, low, 1.0))
        negative = (1.0 - t) * np.log(np.clip(1.0 - p, low, 1.0))
        if epsilon is None:  # exact: a zero-weight term is 0, never 0 * log 0 = NaN
            positive = np.where(t == 0, 0.0, positive)
            negative = np.where(t == 1, 0.0, negative)
        return -(positive + negative)


def brier_terms(target: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Squared error per row, summed over classes (categorical), or per
    output (Bernoulli), in float64."""
    error = probabilities.astype(np.float64)  # one float64 working copy, no one-hot array
    if is_categorical(target, probabilities):
        index = target.astype(np.int64)[..., None]
        np.put_along_axis(error, index, np.take_along_axis(error, index, axis=-1) - 1.0, axis=-1)
        return (error**2).sum(axis=-1)
    return (error - target.astype(np.float64)) ** 2


# --- shared input validation ------------------------------------------------------------------


def read_array(value: Any, *, copy: bool, error: type[Exception]) -> np.ndarray:
    """:func:`to_numpy`, with conversion failures (ragged lists, tensors of
    an unconvertible dtype) raised as ``error``."""
    try:
        return to_numpy(value, copy=copy)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise error(f"cannot read {type(value).__name__} as a numeric array: {exc}") from exc


def categorical_probabilities(
    value: Any, *, error: type[Exception], allow_empty: bool = False, what: str = "probabilities", copy: bool = False
) -> np.ndarray:
    """Validated float64 ``(N, C)`` categorical probabilities (``C >= 2``):
    finite, in ``[0, 1]``, each row summing to 1 within the input's own
    rounding. The input is never modified; float64 input is returned as is
    unless ``copy=True`` (other dtypes are always a fresh float64 array).
    Raises ``error`` otherwise."""
    return categorical_probabilities_eps(value, error=error, allow_empty=allow_empty, what=what, copy=copy)[0]


def rounding_tolerance(eps: float, terms: int = 1, *, floor: float = SUM_TOLERANCE) -> float:
    """How far rounding at machine epsilon ``eps`` can move a sum of
    ``terms`` entries (a single entry by default): each entry's own rounding
    plus a pairwise sum's, which grows like ``eps * log2(terms)``. ``4 eps (1
    + log2 terms)`` allows for both, floored at ``floor``."""
    return max(floor, 4 * eps * (1 + math.log2(max(terms, 1))))


def categorical_probabilities_eps(
    value: Any, *, error: type[Exception], allow_empty: bool = False, what: str = "probabilities", copy: bool = False
) -> tuple[np.ndarray, float]:
    """:func:`categorical_probabilities` and the machine epsilon of the
    input's own dtype (``0.0`` for integers), for :func:`rounding_tolerance`
    of any other comparison its rounding can blur."""
    # bfloat16 tensors arrive upcast to float32; their rounding is bfloat16's.
    source_eps = 2.0**-7 if is_tensor(value) and str(value.dtype) == "torch.bfloat16" else None
    array = read_array(value, copy=False, error=error)
    if array.dtype.kind not in "fiu":
        raise error(f"{what} must be a real numeric array, got dtype {array.dtype}")
    if array.ndim != 2:
        raise error(f"{what} must be 2-D (N, C), got shape {array.shape}")
    if array.shape[1] < 2:
        raise error(f"categorical {what} need at least 2 classes, got {array.shape[1]}")
    if array.shape[0] == 0 and not allow_empty:
        raise error(f"{what} hold no rows")
    # Row sums carry the input's own rounding, which stays far below 1 (float16
    # at C = 1e9 still rejects rows off by 0.12).
    eps = source_eps if source_eps is not None else float(np.finfo(array.dtype).eps) if array.dtype.kind == "f" else 0.0
    tolerance = rounding_tolerance(eps, array.shape[1])
    # The cast below, or a list read into a new array, is this call's own
    # memory; anything else (an ndarray, a tensor, a pandas frame) may be shared.
    fresh = array.dtype != np.float64 or isinstance(value, (list, tuple))
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise error(f"{what} contain non-finite values")
    if ((array < 0) | (array > 1)).any():
        raise error(f"{what} must lie in [0, 1]")
    sums = array.sum(axis=1)
    if array.shape[0] and np.abs(sums - 1.0).max() > tolerance:
        raise error(
            f"rows of categorical {what} must sum to 1 (within {tolerance:g}), got sums up to "
            f"{float(sums[np.abs(sums - 1.0).argmax()])!r}"
        )
    return (array.copy() if copy and not fresh else array), eps


def class_targets(value: Any, n_rows: int, n_classes: int, *, error: type[Exception]) -> np.ndarray:
    """One int64 class index in ``[0, n_classes)`` per row, or ``error``."""
    array = read_array(value, copy=False, error=error)  # only read; the int64 cast below is a fresh array
    if array.size == 0 and array.ndim == 1:  # an empty list is float64 in NumPy
        array = array.astype(np.int64)
    if array.dtype.kind not in "iu":
        raise error(f"targets must be integer class indices, got dtype {array.dtype}")
    if array.ndim != 1 or array.shape[0] != n_rows:
        raise error(f"targets need one target per row ({n_rows}), got shape {array.shape}")
    if array.dtype.kind == "u" and array.size and int(array.max()) >= n_classes:  # before an int64 cast can wrap
        raise error(
            f"targets {sorted(set(array[array >= n_classes][:5].tolist()))} are out of range for {n_classes} "
            f"classes (0..{n_classes - 1})"
        )
    array = array.astype(np.int64)
    outside = array[(array < 0) | (array >= n_classes)]
    if outside.size:
        raise error(
            f"targets {sorted(set(outside[:5].tolist()))} are out of range for {n_classes} classes (0..{n_classes - 1})"
        )
    return array


def row_ids(value: Any, n_rows: Optional[int], *, error: type[Exception]) -> np.ndarray:
    """int64 sample ids: one per row (``0..N-1`` when ``value`` is None;
    any count when ``n_rows`` is None); ids may repeat. Always
    a fresh array."""
    if value is None:
        if n_rows is None:
            raise error("sample_ids are required")
        return np.arange(n_rows, dtype=np.int64)
    ids = read_array(value, copy=False, error=error)
    if ids.size == 0 and ids.ndim == 1:  # an empty list is float64 in NumPy
        ids = ids.astype(np.int64)
    if ids.ndim != 1 or (n_rows is not None and ids.shape[0] != n_rows) or ids.dtype.kind not in "iu":
        rows = "one integer id per row" + ("" if n_rows is None else f" ({n_rows})")
        raise error(f"sample_ids must be {rows}, got shape {ids.shape} and dtype {ids.dtype}")
    if ids.dtype.kind == "u" and ids.size and int(ids.max()) > np.iinfo(np.int64).max:
        raise error("sample_ids above 2**63 - 1 do not fit int64 ids")
    ids = ids.astype(np.int64)  # always a fresh array
    return ids


def same_ids(given: Any, own: Any, *, error: type[Exception]) -> bool:
    """Whether caller-given sample ids equal a prediction's own (``None``
    agrees; another count disagrees); malformed ids — either side's — raise
    ``error``."""
    return given is None or np.array_equal(row_ids(given, None, error=error), row_ids(own, None, error=error))


def class_labels(value: Any, *, error: type[Exception], minimum: int = 1) -> tuple[str, ...]:
    """Ordered, unique, non-empty class names (NumPy arrays and pandas
    indexes accepted) — at least ``minimum`` of them — or ``error``."""
    if not isinstance(value, (str, bytes, Mapping)) and hasattr(value, "tolist"):
        value = value.tolist()  # NumPy arrays, pandas Index / Series
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise error(f"labels must be an ordered sequence of class names, got {value!r}")
    bad = [label for label in value if not isinstance(label, str) or not label]
    if bad:
        raise error(f"labels must be non-empty strings, got {bad!r}")
    out = tuple(str(label) for label in value)  # NumPy str_ -> str
    if len(set(out)) != len(out):
        raise error(f"labels must be unique, got {list(out)}")
    if len(out) < minimum:
        raise error(
            f"a categorical schema needs at least {minimum} label{'' if minimum == 1 else 's'}, got {list(out)}"
        )
    return out


def is_prediction(value: Any) -> bool:
    """Whether ``value`` is an ``nnx.prediction.PredictionResult`` (duck-typed
    on ``logits``, ``spec`` and ``sample_ids``: ``nnx.prediction`` imports
    torch, which callers here must not)."""
    return all(hasattr(value, name) for name in ("logits", "sample_ids", "spec"))


def prediction_labels(
    value: Any,
    labels: Any,
    *,
    error: type[Exception],
    conflict_error: type[Exception],
    what: str,
    minimum: int = 1,
) -> Optional[tuple[str, ...]]:
    """The column labels of a categorical prediction: ``labels`` when given
    (they must match the spec's own, else ``conflict_error``), else the
    spec's, else ``None``. A non-categorical prediction raises ``error``."""
    spec = value.spec
    if spec is None or getattr(spec, "kind", None) != "categorical":
        kind = "continuous" if spec is None else getattr(spec, "kind", None)
        raise error(f"{what} needs a categorical prediction, got a {kind} one")
    declared = getattr(spec, "labels", None)
    if labels is None:
        return None if declared is None else class_labels(declared, error=error, minimum=minimum)
    resolved = class_labels(labels, error=error, minimum=minimum)
    if declared is not None and tuple(declared) != resolved:
        raise conflict_error(f"labels {list(resolved)} disagree with the prediction's own labels {list(declared)}")
    return resolved
