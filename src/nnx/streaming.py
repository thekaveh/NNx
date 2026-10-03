"""Streaming prediction and mergeable metrics (FEAT-020).

``NNModel.predict()`` and ``NNModel.evaluate()`` are eager: they return only
after the whole loader has run, holding every batch's outputs until then.
This module adds the bounded counterparts.

**A prediction stream.** ``NNModel.iter_predict(loader)`` returns a
:class:`PredictionStream` that yields one result per loader batch, in loader
order, and holds nothing once a batch has been handed over::

    with model.iter_predict(loader) as stream:
        for batch in stream:              # PredictionBatch(logits, classes, sample_ids)
            write(batch.sample_ids, batch.classes)

Concatenating the batches gives exactly what the eager call returns:
``predict(loader)``'s logits and classes, and ``predict_proba(loader)``'s
sample ids — the same seed-row slicing for graph loaders and the same
categorical / multilabel / continuous decoding. With a
:class:`~nnx.prediction.ProbabilitySpec`, or ``rich=True`` for a model with a
task, every batch is a :class:`~nnx.prediction.PredictionResult` instead.

- Each batch runs in eval mode under ``no_grad``, and every submodule's
  training mode is restored before the batch is yielded, so the network is in
  its own mode between batches — also when the forward pass raises.
- Closing the stream (leaving the ``with`` block, ``close()``, an exception)
  drops its references to the loader's iterator and to the model. A closed or
  consumed stream refuses to be iterated again. The loader stays the caller's:
  it is never closed and can be iterated again.
- The stream holds at most the batch in flight. What the consumer keeps is
  the consumer's memory: store only what you need (for example the sample
  ids and classes), not the batches.

**Mergeable metrics.** :class:`StreamingMetrics` accumulates declared
:class:`~nnx.MetricSpec` metrics over any number of batches with ``update``,
combines two accumulations over disjoint samples with ``merge`` and returns a
read-only :class:`MetricSnapshot` from ``finalize``. Built-in metrics keep
sums and counts (``accuracy``, ``nll``, ``brier``, ``mae``, ``mse``) or
confusion counts (``f1``), so memory never grows with the number of samples,
and a merge adds them — the result does not depend on the order. A metric that
needs every stored score (a rank metric such as AUROC is never additive) is
rejected unless ``materialize=True`` asks for O(N) storage.

**Bounded validation.** :func:`streaming_eval_step` is a ready-made
``eval_step_fn`` for ``NNModel.train``: the default validation record from
counts and sums instead of stored predictions. ``evaluate()`` itself is
unchanged and eager.
"""

from __future__ import annotations

import copy
import math
import numbers
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch

from ._confusion import integers
from ._probability import to_numpy
from .monitors import (
    MetricAccumulator,
    MetricSpec,
    _batch_inputs,
    _check_metric_inputs,
    _unique_metrics,
)
from .prediction import PredictionResult, ProbabilitySpec

if TYPE_CHECKING:
    from .nn.nn_model import EvalStepContext
    from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
    from .nn.params.nn_train_params import NNTrainParams
    from .tasks import TaskSpec

__all__ = [
    "MetricMergeError",
    "MetricSnapshot",
    "PredictionBatch",
    "PredictionStream",
    "StreamClosedError",
    "StreamingMetrics",
    "concatenate_predictions",
    "streaming_eval_step",
]

SEMANTICS = ("categorical", "bernoulli", "continuous")


class StreamClosedError(RuntimeError):
    """A closed or already consumed :class:`PredictionStream` was used again."""


class MetricMergeError(ValueError):
    """Two :class:`StreamingMetrics` with different declarations were merged:
    other metrics (id, version or config), other probability semantics, other
    task labels, another decision threshold or another ignore index."""


# --- prediction streams ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PredictionBatch:
    """One loader batch of :meth:`NNModel.iter_predict <nnx.NNModel.iter_predict>`.

    Attributes:
        logits: the raw network output for the batch's rows (graph loaders:
            seed rows only), as ``predict()`` returns them.
        classes: the decoded predictions, as ``predict().classes``: argmax
            classes, 0/1 indicators for a multilabel or ``BCEWithLogitsLoss``
            model, the values themselves for a regression task.
        sample_ids: ``int64`` identity of each row, as ``predict_proba()``
            reports it: the position in iteration order, or the global node
            index for graph seed rows.
    """

    logits: np.ndarray
    classes: np.ndarray
    sample_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.logits.shape[0])


class PredictionStream:
    """A context-managed iterator of prediction batches; see the module
    docstring. Obtain one with ``NNModel.iter_predict``."""

    def __init__(self, batches: Iterator[Any]) -> None:
        self._batches: Optional[Iterator[Any]] = batches
        self._state = "open"  # "open" | "consumed" | "closed"
        self._started = False  # a batch was handed out: a second pass would silently skip it

    @property
    def closed(self) -> bool:
        """Whether the stream is closed or consumed (it cannot be iterated again)."""
        return self._state != "open"

    def __enter__(self) -> PredictionStream:
        self._require_open()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __iter__(self) -> PredictionStream:
        self._require_open()
        if self._started:
            raise StreamClosedError(
                "this prediction stream is partly consumed; iterate a stream once from its first batch, or call "
                "iter_predict() again"
            )
        return self

    def __next__(self) -> Union[PredictionBatch, PredictionResult]:
        if self._state != "open":
            raise StopIteration  # like a closed generator: close() inside a for loop ends it
        self._started = True
        assert self._batches is not None
        try:
            return next(self._batches)
        except StopIteration:
            self._release()
            self._state = "consumed"
            raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Stop the stream and drop its references to the loader's iterator
        and the model. Idempotent; the loader itself is left to the caller."""
        if self._state == "open":
            self._state = "closed"  # first, so a failing cleanup never leaves it half open
        self._release()

    def _release(self) -> None:
        batches, self._batches = self._batches, None
        close = getattr(batches, "close", None)
        if callable(close):
            close()  # a generator: finalizes it, releasing the loader iterator

    def _require_open(self) -> None:
        if self._state != "open":
            what = "consumed" if self._state == "consumed" else "closed"
            raise StreamClosedError(
                f"this prediction stream is {what}; call iter_predict() again for a new pass over the loader"
            )


def concatenate_predictions(
    batches: Iterable[Union[PredictionBatch, PredictionResult]],
) -> Union[PredictionBatch, PredictionResult]:
    """Concatenate streamed batches into one result, in order — what the
    eager ``predict()`` / ``predict_proba()`` returns for the same loader.
    Materializes every batch (O(N) memory)."""
    items = list(batches)
    if not items:
        raise ValueError("concatenate_predictions() needs at least one batch")
    if all(isinstance(item, PredictionBatch) for item in items):
        return PredictionBatch(
            logits=np.concatenate([b.logits for b in items]),  # type: ignore[union-attr]
            classes=np.concatenate([b.classes for b in items]),  # type: ignore[union-attr]
            sample_ids=np.concatenate([b.sample_ids for b in items]),
        )
    if all(isinstance(item, PredictionResult) for item in items):
        results = [item for item in items if isinstance(item, PredictionResult)]
        spec = results[0].spec
        if any(result.spec != spec for result in results):
            raise ValueError("concatenate_predictions() got results with different ProbabilitySpecs")
        with_probabilities = [r.probabilities is not None for r in results]
        if any(with_probabilities) and not all(with_probabilities):
            raise ValueError("concatenate_predictions() got results with and without probabilities")
        probabilities = [r.probabilities for r in results] if all(with_probabilities) else None
        return PredictionResult(
            logits=np.concatenate([r.logits for r in results]),
            probabilities=None if probabilities is None else np.concatenate(probabilities),  # type: ignore[arg-type]
            decoded=np.concatenate([r.decoded for r in results]),
            sample_ids=np.concatenate([r.sample_ids for r in results]),
            spec=spec,
        )
    raise TypeError("concatenate_predictions() needs all PredictionBatch or all PredictionResult items")


def _check_stream_source(X: Any) -> None:
    """``iter_predict`` streams a loader of batches, never one in-memory input
    (an array, a tensor, a tuple or mapping of them, or one graph)."""
    one_graph = type(X).__module__.startswith("torch_geometric.data")  # Data / HeteroData / Batch iterate fields
    if (
        one_graph
        or isinstance(X, (np.ndarray, torch.Tensor, tuple, Mapping, str, bytes))
        or not isinstance(X, Iterable)
    ):
        raise TypeError(
            f"iter_predict() streams a DataLoader or another iterable of batches, got {type(X).__name__}; "
            "for one input (arrays, tensors, tuples of them, one graph) call predict() / predict_proba(), "
            "or wrap it in a DataLoader"
        )


# --- mergeable metrics ----------------------------------------------------------------------------


class _NeedsStoredScores(ValueError):
    pass


_REMEDY_STREAMING = (
    "pass StreamingMetrics(..., materialize=True) to store every score (O(N) memory), or compute it with the "
    "default evaluate()"
)
_REMEDY_VALIDATION = "use the default validation step (drop eval_step_fn=streaming_eval_step) for it"


def _bounded_accumulator(spec: MetricSpec, *, remedy: str = _REMEDY_VALIDATION) -> MetricAccumulator:
    """``spec``'s accumulator in a mergeable form whose memory does not grow
    with the samples: the built-ins, or a registered accumulator that
    implements ``merge`` and does not declare ``stores_scores = True``.
    ``remedy`` ends the refusal with what the caller can do instead."""
    accumulator = spec.accumulator()
    if getattr(accumulator, "stores_scores", False):
        why = "its accumulator declares stores_scores=True: it keeps every score (a rank metric such as AUROC is never additive)"
    elif not callable(getattr(accumulator, "merge", None)):
        why = "its accumulator has no merge(), so NNx cannot combine it or tell that it keeps only running sums"
    else:
        return accumulator
    raise _NeedsStoredScores(
        f"metric {spec.label!r} ({spec.id}@v{spec.version}) has no bounded, mergeable form: {why}. Bounded "
        f"streaming rejects it; implement merge() for a metric that keeps only sums or counts, or {remedy}"
    )


class _Stored:
    """A metric that needs every score, kept as stored arrays and replayed
    into a fresh accumulator at finalize (``materialize=True``, O(N))."""

    stores_scores = True

    def __init__(self, spec: MetricSpec) -> None:
        self._spec = spec
        self._targets: list[np.ndarray] = []
        self._predictions: list[np.ndarray] = []

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        self._spec.accumulator().update(target, prediction)  # malformed inputs fail on their own batch
        self._targets.append(np.array(target, copy=True))
        self._predictions.append(np.array(prediction, copy=True))

    def merge(self, other: _Stored) -> None:
        # The stored arrays are private copies that are never written again, so
        # sharing them is safe: a later update only appends to one list.
        self._targets.extend(other._targets)
        self._predictions.extend(other._predictions)

    def __deepcopy__(self, memo: dict) -> _Stored:
        clone = _Stored(self._spec)
        clone._targets, clone._predictions = list(self._targets), list(self._predictions)
        return clone

    def result(self) -> Optional[float]:
        if not self._targets:
            return None
        accumulator = self._spec.accumulator()
        accumulator.update(np.concatenate(self._targets), np.concatenate(self._predictions))
        return accumulator.result()


@dataclass(frozen=True)
class MetricSnapshot:
    """A finalized :class:`StreamingMetrics`: read-only values that later
    updates never change.

    Attributes:
        count: the valid samples scored (categorical rows; multilabel and
            continuous entries). ``0`` for an empty or fully masked stream.
        values: each available metric's value, by name.
        unavailable: the declared metrics with no value (every one when
            ``count`` is 0).
    """

    count: int
    values: Mapping[str, float] = field(default_factory=dict)
    unavailable: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.count > 0


class StreamingMetrics:
    """Mergeable accumulators for declared metrics (see the module docstring).

    Args:
        metrics: the :class:`~nnx.MetricSpec` s to accumulate (unique names).
        semantics: how predictions are read — ``"categorical"`` (class
            probabilities over the last axis of ``probabilities``, argmax
            labels), ``"bernoulli"`` (independent per-output probabilities,
            labels at ``threshold``) or ``"continuous"`` (values). Every
            metric's input must be one these semantics provide.
        labels: optional ordered class / output names; a categorical
            ``probabilities`` width must match them. Part of the merge schema.
        threshold: the bernoulli decision threshold for labels derived from
            probabilities (default 0.5).
        materialize: store every score of a metric that has no bounded form
            (O(N) memory) instead of rejecting it.
        ignore_index: categorical only — the target value that is never
            scored (e.g. ``-100`` for padding).
        num_outputs: the class count (categorical) or the output width along
            axis 1 (bernoulli, continuous), when no ``labels`` name them;
            batches of another width are refused.

    Two accumulators merge only when their metrics (id, version, config and
    name), semantics, labels, output count, threshold and ignore index are
    equal. If a
    metric's update raises part-way through a batch, the accumulation no
    longer describes one set of samples and refuses further use.
    """

    def __init__(
        self,
        metrics: Sequence[MetricSpec],
        semantics: str,
        *,
        labels: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
        materialize: bool = False,
        ignore_index: Optional[int] = None,
        num_outputs: Optional[int] = None,
    ) -> None:
        specs = _unique_metrics(metrics, "StreamingMetrics")
        if semantics not in SEMANTICS:
            raise ValueError(f"semantics must be one of {', '.join(repr(s) for s in SEMANTICS)}, got {semantics!r}")
        if labels is not None:
            labels = tuple(labels)
            if not labels or any(not isinstance(label, str) or not label for label in labels):
                raise ValueError(f"labels must be non-empty strings, got {labels!r}")
            if len(set(labels)) != len(labels):
                raise ValueError(f"labels must be unique, got {labels!r}")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, numbers.Real)
            or not math.isfinite(threshold)
            or not 0.0 < threshold < 1.0
        ):
            raise ValueError(f"threshold must be a probability in (0, 1), got {threshold!r}")
        if ignore_index is not None and (
            semantics != "categorical"
            or isinstance(ignore_index, bool)
            or not isinstance(ignore_index, numbers.Integral)
        ):
            raise ValueError(f"ignore_index must be an integer for categorical semantics, got {ignore_index!r}")
        if num_outputs is not None:
            least = 2 if semantics == "categorical" else 1
            if isinstance(num_outputs, bool) or not isinstance(num_outputs, numbers.Integral) or num_outputs < least:
                raise ValueError(f"num_outputs must be an integer of at least {least}, got {num_outputs!r}")
            if labels is not None and len(labels) != num_outputs:
                raise ValueError(f"num_outputs={num_outputs} does not match the {len(labels)} labels")
        width = int(num_outputs) if num_outputs is not None else (len(labels) if labels is not None else None)
        n_classes = width if semantics == "categorical" else None
        _check_metric_inputs(specs, semantics, where="StreamingMetrics", n_classes=n_classes)
        self._specs = specs
        self._semantics = semantics
        self._labels: Optional[tuple[str, ...]] = labels
        self._threshold = float(threshold)
        self._materialize = bool(materialize)
        self._ignore_index = None if ignore_index is None else int(ignore_index)
        self._num_outputs: Optional[int] = width
        self._logit_threshold = math.log(self._threshold / (1.0 - self._threshold))  # the task's decision rule
        self._accumulators: dict[str, Any] = {}
        for spec in specs:
            try:
                self._accumulators[spec.label] = _bounded_accumulator(spec, remedy=_REMEDY_STREAMING)
            except _NeedsStoredScores:
                if not materialize:
                    raise
                self._accumulators[spec.label] = _Stored(spec)
        self._count = 0
        self._broken = False
        self._seen_width: Optional[int] = None
        self._top_class = -1  # the largest categorical class index scored (-1: none yet)

    @classmethod
    def for_task(cls, metrics: Sequence[MetricSpec], task: TaskSpec, *, materialize: bool = False) -> StreamingMetrics:
        """Accumulators for a model's :class:`~nnx.TaskSpec`: categorical
        (with the task's ``ignore_index``), multilabel (bernoulli at the
        task's threshold) or regression (continuous), with the task's labels."""
        semantics = {"categorical": "categorical", "multilabel": "bernoulli", "regression": "continuous"}[task.kind]
        return cls(
            metrics,
            semantics,
            labels=task.labels,
            threshold=task.threshold,
            materialize=materialize,
            ignore_index=task.ignore_index if semantics == "categorical" else None,
            num_outputs=task.num_outputs,
        )

    @property
    def metrics(self) -> tuple[MetricSpec, ...]:
        return self._specs

    @property
    def semantics(self) -> str:
        return self._semantics

    @property
    def labels(self) -> Optional[tuple[str, ...]]:
        return self._labels

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def ignore_index(self) -> Optional[int]:
        return self._ignore_index

    @property
    def num_outputs(self) -> Optional[int]:
        """The class count (categorical) or output width, if declared
        (``num_outputs``, else the labels')."""
        return self._num_outputs

    @property
    def count(self) -> int:
        """The valid samples scored so far."""
        return self._count

    @property
    def bounded(self) -> bool:
        """Whether no metric stores its scores (memory independent of N)."""
        return not any(isinstance(accumulator, _Stored) for accumulator in self._accumulators.values())

    # --- updates ------------------------------------------------------------------------------

    def update(
        self,
        target: Any,
        *,
        probabilities: Any = None,
        labels: Any = None,
        values: Any = None,
        valid: Any = None,
    ) -> int:
        """Add one batch and return the number of valid samples scored.

        ``target`` holds class indices ``(N,)`` (categorical), 0/1 outcomes
        (bernoulli; soft targets count as 1 from 0.5) or values (continuous).
        Give each input a declared metric reads: ``probabilities`` —
        ``(N, C)`` class probabilities (categorical) or per-output
        probabilities shaped like ``target`` (bernoulli); ``labels`` — the
        decoded predictions (derived from ``probabilities`` when omitted);
        ``values`` — continuous predictions shaped like ``target``.
        ``valid`` (boolean, shaped like ``target``) excludes entries; a NaN
        target, or a categorical target equal to ``ignore_index``, is
        excluded too. Arrays and tensors are accepted."""
        self._require_intact()  # before any check of the batch
        target_np = to_numpy(target, copy=False)
        if valid is None:
            mask = np.ones(target_np.shape, dtype=bool)
        else:
            mask = _same_rows(to_numpy(valid, copy=False), target_np, "valid").astype(bool)
        mask = _target_rule(target_np, mask, self._ignore_index)
        inputs: dict[str, np.ndarray] = {}
        given = {
            "probabilities": None if probabilities is None else to_numpy(probabilities, copy=False),
            "labels": None if labels is None else to_numpy(labels, copy=False),
            "continuous": None if values is None else to_numpy(values, copy=False),
        }
        needed = {spec.input for spec in self._specs}
        width: Optional[int] = None  # the batch's classes / outputs, when it shows them
        if self._semantics == "categorical":
            if target_np.ndim != 1:
                raise ValueError(f"categorical targets must be class indices shaped (N,), got {target_np.shape}")
            truth = integers(target_np[mask], "categorical targets")
            probs = given["probabilities"]
            if probs is not None:
                if probs.ndim != 2 or probs.shape[0] != target_np.shape[0]:
                    raise ValueError(
                        f"categorical probabilities must be (N, C) with N={target_np.shape[0]}, got {probs.shape}"
                    )
                width = self._check_width(probs.shape[1], "probabilities")
                inputs["probabilities"] = probs[mask]
            if given["labels"] is not None:
                inputs["labels"] = integers(_same_rows(given["labels"], target_np, "labels")[mask], "labels")
            elif probs is not None and "labels" in needed:
                inputs["labels"] = inputs["probabilities"].argmax(axis=1)  # the rows already selected
            # Class indices address the declared classes, else the probabilities'
            # columns; with neither known they are at least never negative.
            n_classes = self._num_outputs
            if n_classes is None:  # this batch's probabilities, else the width earlier batches fixed
                n_classes = probs.shape[1] if probs is not None else self._seen_width
            _check_class_range(truth, n_classes)
            if "labels" in inputs:
                _check_class_range(inputs["labels"], n_classes, "decoded labels")
        else:
            width = self._check_width(target_np.shape[1] if target_np.ndim >= 2 else 1, "targets")
            for name in ("probabilities", "labels", "continuous"):
                array = given[name]
                if array is not None:
                    inputs[name] = _same_rows(array, target_np, name)[mask].reshape(-1)
            if "labels" in inputs and self._semantics == "bernoulli" and not np.isin(inputs["labels"], (0, 1)).all():
                raise ValueError("bernoulli labels must be 0/1 decisions (pass probabilities= for probabilities)")
            if "labels" in needed and "labels" not in inputs and "probabilities" in inputs:
                inputs["labels"] = (inputs["probabilities"] >= self._threshold).astype(np.int64)
            truth = target_np[mask].reshape(-1)
            if self._semantics == "bernoulli":
                truth = (truth >= 0.5).astype(np.int64)
        for spec in self._specs:
            if spec.input not in inputs:
                keyword = "values" if spec.input == "continuous" else spec.input
                raise ValueError(f"metric {spec.label!r} needs {spec.input} inputs: pass update(..., {keyword}=...)")
        return self._feed(truth, inputs, width)

    def update_logits(self, target: Any, logits: Any, *, valid: Any = None) -> int:
        """Add one batch of raw model outputs (class axis 1): softmax
        (categorical), sigmoid decided at ``threshold`` (bernoulli) or the
        values (continuous) — the inputs ``evaluate()`` derives. Returns the
        number of valid samples scored."""
        self._require_intact()  # before any check of the batch
        target_t = _tensor(target)
        logits_t = _tensor(logits)
        if self._semantics == "categorical" and logits_t.ndim < 2:
            raise ValueError(f"categorical logits need a class axis 1, (N, C, ...), got {tuple(logits_t.shape)}")
        width = self._check_width(int(logits_t.shape[1]) if logits_t.ndim >= 2 else 1, "logits")
        categorical = self._semantics == "categorical"
        soft = categorical and target_t.is_floating_point() and target_t.shape == logits_t.shape
        rows = (logits_t.shape[0], *logits_t.shape[2:]) if logits_t.ndim >= 2 else None
        if categorical and not soft and tuple(target_t.shape) != rows:
            raise ValueError(
                f"categorical targets must be class indices shaped like the logits without their class axis "
                f"{rows}, or one-hot / soft rows shaped like the logits, got {tuple(target_t.shape)}"
            )
        if not categorical and target_t.shape != logits_t.shape:
            raise ValueError(
                f"logits must be shaped like the targets {tuple(target_t.shape)}, got {tuple(logits_t.shape)}"
            )
        if soft:
            # One-hot / soft class targets: whole rows decided by their argmax, masked by
            # valid (shaped like the rows) and — as every NaN target here — a row holding a
            # NaN; never by ignore_index, a class index rule.
            rows = (target_t.shape[0], *target_t.shape[2:])
            valid_t = torch.ones(rows, dtype=torch.bool, device=target_t.device) if valid is None else _tensor(valid)
            valid_t = valid_t.to(dtype=torch.bool, device=target_t.device)
            if tuple(valid_t.shape) != rows:
                raise ValueError(f"valid must be shaped like the target rows {rows}, got {tuple(valid_t.shape)}")
            valid_t = valid_t & ~torch.isnan(target_t).any(dim=1)
        else:
            valid_t = self._target_mask(target_t, valid)
            if categorical and target_t.is_floating_point():
                integers(target_t[valid_t].cpu().numpy(), "categorical targets")  # float class indices
        truth, inputs = _batch_inputs(
            self._semantics,
            target_t,
            logits_t,
            valid_t,
            None,  # the ignore index is part of valid_t
            frozenset(spec.input for spec in self._specs),
            self._logit_threshold,
        )
        if categorical:
            _check_class_range(truth, int(logits_t.shape[1]))  # on the host copy _batch_inputs made
        return self._feed(truth, inputs, width)

    def _target_mask(self, target: torch.Tensor, valid: Any) -> torch.Tensor:
        """:meth:`update_logits`'s mask: ``valid`` (every entry by default),
        then :func:`_target_rule`."""
        if valid is None:
            mask = torch.ones(target.shape, dtype=torch.bool, device=target.device)
        else:
            mask = _tensor(valid).to(dtype=torch.bool, device=target.device)
            if tuple(mask.shape) != tuple(target.shape):
                raise ValueError(
                    f"valid must be shaped like the targets {tuple(target.shape)}, got {tuple(mask.shape)}"
                )
        return _target_rule(target, mask, self._ignore_index)

    def _check_width(self, width: int, what: str) -> int:
        """The batch's classes / outputs along axis 1 must be the declared
        ``num_outputs`` — or, undeclared, the width of every earlier batch.
        The first width of an undeclared categorical accumulation must hold
        every class index earlier (labels-only) batches brought."""
        expected = self._num_outputs if self._num_outputs is not None else self._seen_width
        if expected is not None and width != expected:
            source = "are declared" if self._num_outputs is not None else "came in earlier batches"
            raise ValueError(f"{what} have {width} outputs along axis 1 but {expected} {source}")
        if expected is None:
            _check_top_class(self._top_class, width, f"these {what}")  # -1 unless categorical
        return width

    def _feed(self, target: np.ndarray, inputs: Mapping[str, np.ndarray], width: Optional[int] = None) -> int:
        self._require_intact()
        if width is not None and self._seen_width is None:
            self._seen_width = width  # an accepted batch fixes the width of an undeclared accumulation
        n = int(target.shape[0])
        if not n:
            return 0
        # Categorical classes range-checked against no width yet: kept for the
        # width a later batch or a merge fixes, once the batch is scored.
        unchecked = self._semantics == "categorical" and self._num_outputs is None and self._seen_width is None
        try:
            for spec in self._specs:
                self._accumulators[spec.label].update(target, inputs[spec.input])
        except BaseException:
            # Metrics before the failing one took the batch and the rest did not:
            # the accumulation no longer describes one set of samples.
            self._broken = True
            raise
        if unchecked:
            labels = inputs.get("labels")
            top = int(target.max()) if labels is None else max(int(target.max()), int(labels.max()))
            self._top_class = max(self._top_class, top)
        self._count += n
        return n

    def _require_intact(self) -> None:
        if self._broken:
            raise RuntimeError(
                "a metric failed part-way through an update, so this accumulation is inconsistent; start a new "
                "StreamingMetrics"
            )

    # --- merge / finalize -----------------------------------------------------------------------

    def merge(self, other: StreamingMetrics) -> StreamingMetrics:
        """A new accumulation covering both inputs' samples; neither input
        changes, and later updates to either never reach the result.
        Raises :class:`MetricMergeError` for different declarations."""
        if not isinstance(other, StreamingMetrics):
            raise TypeError(f"merge() needs StreamingMetrics, got {type(other).__name__}")
        self._require_intact()
        other._require_intact()
        self._check_schema(other)
        merged = copy.copy(self)
        # Copy only this side: an accumulator's merge() reads the other side and
        # keeps nothing of it that a later update could change.
        merged._accumulators = {label: copy.deepcopy(acc) for label, acc in self._accumulators.items()}
        for label, accumulator in merged._accumulators.items():
            if accumulator.merge(other._accumulators[label]) is not None:
                raise TypeError(
                    f"the accumulator of metric {label!r} returned a value from merge(); merge(other) must add "
                    "other's state to the accumulator in place and return None"
                )
        merged._count += other._count
        merged._seen_width = self._seen_width if self._seen_width is not None else other._seen_width
        merged._top_class = max(self._top_class, other._top_class)
        return merged

    def _check_schema(self, other: StreamingMetrics) -> None:
        """Declarations first, then what the batches fixed (widths, class
        indices), so a merge names the real mismatch."""
        mine = [(spec.label, spec.id, spec.version, dict(spec.state())) for spec in self._specs]
        theirs = [(spec.label, spec.id, spec.version, dict(spec.state())) for spec in other._specs]
        if mine != theirs:
            raise MetricMergeError(
                f"cannot merge different metric declarations: {list(self._specs)} vs {list(other._specs)} "
                "(names, ids, versions and configs must match)"
            )
        if self._semantics != other._semantics:
            raise MetricMergeError(
                f"cannot merge different probability semantics: {self._semantics!r} vs {other._semantics!r}"
            )
        if self._labels != other._labels:
            raise MetricMergeError(f"cannot merge different task labels: {self._labels!r} vs {other._labels!r}")
        if self._semantics == "bernoulli" and self._threshold != other._threshold:  # the only semantics it decides
            raise MetricMergeError(
                f"cannot merge different decision thresholds: {self._threshold!r} vs {other._threshold!r}"
            )
        if self._num_outputs != other._num_outputs:
            raise MetricMergeError(
                f"cannot merge different output counts: {self._num_outputs!r} vs {other._num_outputs!r}"
            )
        if self._ignore_index != other._ignore_index:
            raise MetricMergeError(
                f"cannot merge different ignore indices: {self._ignore_index!r} vs {other._ignore_index!r}"
            )
        if self._seen_width is not None and other._seen_width is not None and self._seen_width != other._seen_width:
            raise MetricMergeError(
                f"cannot merge accumulations of different widths: {self._seen_width} vs {other._seen_width} outputs"
            )
        width = self._seen_width if self._seen_width is not None else other._seen_width
        top = max(self._top_class, other._top_class)  # one side's classes, the other side's width
        if self._semantics == "categorical" and width is not None and top >= width:
            raise MetricMergeError(f"cannot merge: {_top_class_problem(top, width, 'the merged accumulation')}")

    def finalize(self) -> MetricSnapshot:
        """The metrics over every sample seen, as a read-only snapshot.
        Repeatable, and never changes the accumulation."""
        from .trainer.params import _FrozenMapping

        self._require_intact()
        values: dict[str, float] = {}
        unavailable: list[str] = []
        for spec in self._specs:
            value = self._accumulators[spec.label].result() if self._count else None
            if value is None:
                unavailable.append(spec.label)
            else:
                values[spec.label] = float(value)
        return MetricSnapshot(count=self._count, values=_FrozenMapping(values), unavailable=tuple(unavailable))

    def __repr__(self) -> str:
        return (
            f"StreamingMetrics({[spec.label for spec in self._specs]}, {self._semantics!r}, count={self._count}"
            + (", materialize=True" if self._materialize else "")
            + ")"
        )


def _target_rule(target: Any, mask: Any, ignore_index: Optional[int]) -> Any:
    """The one masking rule of ``update`` and ``update_logits``, for NumPy
    arrays and tensors alike: a NaN target, or a target equal to
    ``ignore_index``, is never scored."""
    if isinstance(target, torch.Tensor):
        if target.is_floating_point():
            mask = mask & ~torch.isnan(target)
    elif target.dtype.kind == "f":
        mask = mask & ~np.isnan(target)
    if ignore_index is not None:
        mask = mask & (target != ignore_index)
    return mask


def _tensor(value: Any) -> torch.Tensor:
    """A tensor view of an array. A read-only array (torch never shares memory
    it could write) or one with negative strides (which torch cannot view) is
    copied first."""
    if isinstance(value, np.ndarray) and (not value.flags.writeable or any(stride < 0 for stride in value.strides)):
        value = value.copy()
    return torch.as_tensor(value)


def _top_class_problem(top: int, width: int, where: str) -> str:
    return f"class index {top} was scored before any width was known, but {where}: {width} classes"


def _check_top_class(top: int, width: int, where: str) -> None:
    """Class indices scored before any width was known must fit the width
    that ``where`` now fixes."""
    if top >= width:
        raise ValueError(_top_class_problem(top, width, where))


def _check_class_range(values: np.ndarray, n_classes: Optional[int], what: str = "categorical targets") -> None:
    """Categorical class indices must address one of the ``n_classes``
    classes — or, with ``n_classes`` unknown, be non-negative (a masked or
    ignored target is removed before)."""
    if not values.size:
        return
    low, high = int(values.min()), int(values.max())
    if low < 0 or (n_classes is not None and high >= n_classes):
        bound = f"[0, {n_classes})" if n_classes is not None else "[0, C)"
        raise ValueError(
            f"{what} must be class indices in {bound}, got values in [{low}, {high}]; "
            "mask the others with valid= or ignore_index"
        )


def _same_rows(array: np.ndarray, target: np.ndarray, name: str) -> np.ndarray:
    if array.shape != target.shape:
        raise ValueError(f"{name} must be shaped like the targets {target.shape}, got {array.shape}")
    return array


# --- bounded validation ---------------------------------------------------------------------------


def streaming_eval_step(ctx: EvalStepContext) -> NNEvaluationDataPoint:
    """An ``eval_step_fn`` for ``NNModel.train`` that builds the default
    validation record from counts and sums (FEAT-020).

    The record is the one NNx's own ``evaluate()`` returns — the same task
    counts and status, loss (each batch's numerator over the summed loss
    denominators), classification or task metrics and declared metrics —
    without storing a target or prediction per sample, so memory does not
    grow with the validation set. A model subclass that overrides
    ``evaluate()`` keeps its override only on the default validation step. It cannot compute ``extra_metrics`` (callables on the
    full arrays) or a declared metric that needs every stored score; both are
    rejected when training starts. ``evaluate()`` and the default validation
    step are unchanged."""
    from .nn.nn_model import _evaluate

    return _evaluate(
        ctx.model,
        ctx.val_loader,
        ctx.extra_metrics,
        tuple(ctx.metrics),
        bounded=True,
        who="streaming_eval_step()",
    )


def _streaming_problems(params: NNTrainParams) -> list[tuple[str, str]]:
    """What ``streaming_eval_step`` cannot compute for a run with these
    training parameters, as ``(field path, message)`` — pure apart from
    building each declared metric's accumulator."""
    from .tasks import _bounded_extra_metrics_error

    found: list[tuple[str, str]] = []
    if params.extra_metrics:
        found.append(("train.extra_metrics", str(_bounded_extra_metrics_error("streaming_eval_step()"))))
    for index, spec in enumerate(params.metrics):
        try:
            spec.check()
        except (KeyError, TypeError, ValueError):
            continue  # the metric's own check reports it
        try:
            _bounded_accumulator(spec)
        except _NeedsStoredScores as exc:
            found.append((f"train.metrics[{index}]", str(exc)))
        except Exception as exc:  # the registered factory runs its own code
            found.append((f"train.metrics[{index}]", f"building metric {spec.label!r}'s accumulator failed: {exc!r}"))
    return found


def _streaming_preflight(model: Any, params: NNTrainParams) -> None:
    """Fail before the run is reserved when the streaming step cannot
    produce the validation record the run declares."""
    from .nn.nn_model import _metric_context

    # One check shared with ExperimentPlan.validate(): the first problem is the error.
    problems = _streaming_problems(params)
    if problems:
        raise ValueError(problems[0][1])
    domain, _, _, n_classes = _metric_context(model)  # inputs checked, no accumulator built again
    _check_metric_inputs(tuple(params.metrics), domain, where="streaming_eval_step()", n_classes=n_classes)
    adapter = getattr(model, "task_adapter", None)
    if adapter is not None:
        from .nn.nn_model import _bounded_task_accumulator

        _bounded_task_accumulator(adapter, "streaming_eval_step()")  # the check the step repeats per epoch


def _as_probability_spec(spec: Any) -> Optional[ProbabilitySpec]:
    if spec is not None and not isinstance(spec, ProbabilitySpec):
        raise TypeError(f"spec must be a ProbabilitySpec, got {type(spec).__name__}")
    return spec
