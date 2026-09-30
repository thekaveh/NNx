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
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

NLL_EPSILON = 1e-12


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
