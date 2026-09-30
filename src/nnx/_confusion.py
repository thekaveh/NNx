"""Mergeable confusion counts for label metrics. Internal; NumPy only.

Accuracy, precision, recall and F1 need no stored predictions: counts of the
``(target, prediction)`` pairs seen are enough, and two counts over disjoint
samples add. Memory grows with the number of distinct classes (or labels),
never with the number of samples. The values reproduce scikit-learn's
``accuracy_score`` and ``precision_score`` / ``recall_score`` / ``f1_score``
with ``zero_division=0`` over the same samples:

- :class:`ConfusionCounts` — class labels (multiclass or binary): the label
  set is every class seen in the targets or the predictions, and
  ``"binary"`` scores class 1 of a two-class set;
- :class:`MultilabelCounts` — 0/1 indicator rows ``(N, K)``: accuracy is the
  exact-row (subset) accuracy and the averages run over the ``K`` labels.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

AVERAGES = ("macro", "micro", "weighted", "binary")


def integers(values: np.ndarray, what: str) -> np.ndarray:
    """``values`` as ``int64``; non-integral labels are rejected (scikit-learn
    refuses continuous labels too)."""
    values = np.asarray(values)
    if values.dtype.kind == "u" and values.size and int(values.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{what} above 2**63 - 1 do not fit int64 class labels")  # never wrapped negative
    if values.dtype.kind in "iub":
        return values.astype(np.int64, copy=False)
    with np.errstate(invalid="ignore"):
        cast = values.astype(np.int64)
    if not np.array_equal(cast, values):  # NaN, ±inf and fractions never equal their cast
        raise ValueError(f"{what} must be integer class labels, got non-integral values")
    return cast


_DENSE_LIMIT = 1 << 18  # class labels in [0, 262144) — up to a very large vocabulary — are counted densely


class ConfusionCounts:
    """Per-class counts of class labels: true positives, true (support) and
    predicted totals. Labels in ``[0, 262144)`` — class indices, up to a
    very large vocabulary — are counted in dense arrays grown on demand
    (``np.bincount``); any other label in sorted sparse arrays. Both are
    vectorized. Memory grows with the classes seen, never
    with the samples."""

    __slots__ = ("_dense", "_keys", "_sparse", "count", "correct")

    def __init__(self) -> None:
        self._dense = np.zeros((3, 0), dtype=np.int64)  # rows: true, predicted, true positives
        self._keys = np.zeros(0, dtype=np.int64)  # the other labels, sorted
        self._sparse = np.zeros((3, 0), dtype=np.int64)  # their counts, same rows
        self.count = 0
        self.correct = 0

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        """Add one batch of class labels (any shape; flattened)."""
        target = integers(np.asarray(target).reshape(-1), "targets")
        prediction = integers(np.asarray(prediction).reshape(-1), "predictions")
        if target.shape != prediction.shape:
            raise ValueError(f"targets and predictions differ in size: {target.size} vs {prediction.size}")
        if target.size == 0:
            return
        hits = target == prediction
        for row, labels in enumerate((target, prediction, target[hits])):  # true, predicted, true positives
            self._add(row, labels)
        self.count += int(target.size)
        self.correct += int(np.count_nonzero(hits))

    def _add(self, row: int, labels: np.ndarray) -> None:
        dense = (labels >= 0) & (labels < _DENSE_LIMIT)
        inside = labels[dense]
        if inside.size:
            counts = np.bincount(inside)
            self._grow(counts.size)
            self._dense[row, : counts.size] += counts
        outside = labels[~dense]
        if outside.size:  # negative or very large labels
            keys, numbers = np.unique(outside, return_counts=True)
            table = np.zeros((3, keys.size), dtype=np.int64)
            table[row] = numbers
            self._add_sparse(keys, table)

    def _add_sparse(self, keys: np.ndarray, table: np.ndarray) -> None:
        merged = np.union1d(self._keys, keys)
        counts = np.zeros((3, merged.size), dtype=np.int64)
        counts[:, np.searchsorted(merged, self._keys)] += self._sparse
        counts[:, np.searchsorted(merged, keys)] += table
        self._keys, self._sparse = merged, counts

    def _grow(self, size: int) -> None:
        if size > self._dense.shape[1]:
            self._dense = np.pad(self._dense, ((0, 0), (0, size - self._dense.shape[1])))

    def merge(self, other: ConfusionCounts) -> None:
        """Add ``other``'s counts in place."""
        self._grow(other._dense.shape[1])
        self._dense[:, : other._dense.shape[1]] += other._dense
        if other._keys.size:
            self._add_sparse(other._keys, other._sparse)
        self.count += other.count
        self.correct += other.correct

    def accuracy(self) -> Optional[float]:
        return self.correct / self.count if self.count else None

    def _table(self) -> tuple[list[int], np.ndarray]:
        """The sorted label set (every class seen as a target or a
        prediction) and its ``(3, labels)`` counts."""
        seen = np.flatnonzero(self._dense[0] + self._dense[1])
        if not self._keys.size:
            return seen.tolist(), self._dense[:, seen]
        labels = np.concatenate([seen, self._keys])
        table = np.concatenate([self._dense[:, seen], self._sparse], axis=1)
        order = np.argsort(labels, kind="stable")  # dense and sparse labels never overlap
        return labels[order].tolist(), table[:, order]

    def scores(self, average: str = "macro") -> Optional[tuple[float, float, float]]:
        """``(precision, recall, f1)`` under ``average``, or ``None`` before
        any sample."""
        _check_average(average)
        if not self.count:
            return None
        labels, table = self._table()
        if average == "binary":
            # scikit-learn: a two-class label set scored for pos_label=1.
            if len(labels) > 2:
                raise ValueError(
                    f"average='binary' needs at most two classes, got {labels}; use 'macro', 'micro' or 'weighted'"
                )
            if 1 not in labels:
                if len(labels) == 2:
                    raise ValueError(f"average='binary' scores class 1, which is not among the classes {labels}")
                return 0.0, 0.0, 0.0  # class 1 never appears: nothing to score
            table = table[:, [labels.index(1)]]
        true, predicted, tp = table
        return _average(tp, predicted, true, average)


class MultilabelCounts:
    """Per-label counts of 0/1 indicator rows ``(N, K)``."""

    __slots__ = ("tp", "predicted", "true", "count", "correct")

    def __init__(self) -> None:
        self.tp: Optional[np.ndarray] = None
        self.predicted: Optional[np.ndarray] = None
        self.true: Optional[np.ndarray] = None
        self.count = 0  # rows
        self.correct = 0  # rows predicted exactly

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        target = integers(target, "targets")
        prediction = integers(prediction, "predictions")
        if target.ndim != 2 or target.shape != prediction.shape:
            raise ValueError(f"multilabel indicators must be (N, K) alike, got {target.shape} and {prediction.shape}")
        if not (np.isin(target, (0, 1)).all() and np.isin(prediction, (0, 1)).all()):
            raise ValueError("multilabel indicators must be 0/1")
        if self.tp is None:
            width = target.shape[1]
            self.tp = np.zeros(width, dtype=np.int64)
            self.predicted = np.zeros(width, dtype=np.int64)
            self.true = np.zeros(width, dtype=np.int64)
        elif target.shape[1] != self.tp.shape[0]:
            raise ValueError(f"multilabel width changed from {self.tp.shape[0]} to {target.shape[1]}")
        assert self.predicted is not None and self.true is not None
        self.tp += (target & prediction).sum(axis=0)
        self.predicted += prediction.sum(axis=0)
        self.true += target.sum(axis=0)
        self.count += int(target.shape[0])
        self.correct += int((target == prediction).all(axis=1).sum())

    def merge(self, other: MultilabelCounts) -> None:
        if other.tp is None:
            return
        if self.tp is None:
            self.tp, self.predicted, self.true = other.tp.copy(), other.predicted.copy(), other.true.copy()  # type: ignore[union-attr]
        else:
            if other.tp.shape != self.tp.shape:
                raise ValueError(f"multilabel widths differ: {self.tp.shape[0]} vs {other.tp.shape[0]}")
            self.tp = self.tp + other.tp
            self.predicted = self.predicted + other.predicted  # type: ignore[operator]
            self.true = self.true + other.true  # type: ignore[operator]
        self.count += other.count
        self.correct += other.correct

    def accuracy(self) -> Optional[float]:
        return self.correct / self.count if self.count else None

    def scores(self, average: str = "macro") -> Optional[tuple[float, float, float]]:
        _check_average(average)
        if not self.count:
            return None
        if average == "binary":
            raise ValueError(
                "average='binary' does not apply to multilabel indicators; use 'macro', 'micro' or 'weighted'"
            )
        assert self.tp is not None and self.predicted is not None and self.true is not None
        return _average(self.tp, self.predicted, self.true, average)


LabelCounts = Union[ConfusionCounts, MultilabelCounts]


def label_counts(target: np.ndarray) -> LabelCounts:
    """The counts for a label array shaped like ``target``: multilabel
    indicators for ``(N, K)`` with ``K > 1``, class labels otherwise
    (``(N,)`` and column vectors ``(N, 1)``). Other shapes are refused, as
    scikit-learn refuses them."""
    shape = np.shape(target)
    if len(shape) <= 1 or (len(shape) == 2 and shape[1] == 1):
        return ConfusionCounts()
    if len(shape) == 2:
        return MultilabelCounts()
    raise ValueError(f"classification labels must be (N,) classes or (N, K) indicators, got shape {shape}")


def record_scores(counts: LabelCounts) -> tuple[float, float, float, float]:
    """``(accuracy, precision, recall, f1)`` with macro averaging — the
    classification fields of an evaluation record."""
    accuracy = counts.accuracy()
    scores = counts.scores("macro")
    if accuracy is None or scores is None:
        raise ValueError("no samples were counted")
    return (float(accuracy), *scores)


def _check_average(average: str) -> None:
    if average not in AVERAGES:
        raise ValueError(f"average must be one of {', '.join(repr(a) for a in AVERAGES)}, got {average!r}")


def average_scores(tp: np.ndarray, predicted: np.ndarray, true: np.ndarray, average: str) -> tuple[float, float, float]:
    """``(precision, recall, f1)`` from per-label true positives, predicted
    and true totals, averaged as scikit-learn does with ``zero_division=0``."""
    return _average(tp, predicted, true, average)


def _average(tp: np.ndarray, predicted: np.ndarray, true: np.ndarray, average: str) -> tuple[float, float, float]:
    if average == "micro":
        tp, predicted, true = tp.sum(keepdims=True), predicted.sum(keepdims=True), true.sum(keepdims=True)
    precision = _divide(tp, predicted)
    recall = _divide(tp, true)
    f1 = _divide(2 * tp, true + predicted)
    if average == "weighted":
        weights = true.astype(np.float64)
        if not weights.sum():
            return 0.0, 0.0, 0.0
        return (
            float(np.average(precision, weights=weights)),
            float(np.average(recall, weights=weights)),
            float(np.average(f1, weights=weights)),
        )
    return float(precision.mean()), float(recall.mean()), float(f1.mean())


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``numerator / denominator`` with 0 where the denominator is 0
    (``zero_division=0``)."""
    out = np.zeros(numerator.shape, dtype=np.float64)
    return np.divide(numerator, denominator, out=out, where=denominator > 0)
