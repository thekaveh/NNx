from __future__ import annotations

import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Optional, cast

import numpy as np
from sklearn import metrics


class _FrozenMetrics(Mapping[str, float]):
    __slots__ = ("_items", "_values")

    def __init__(self, values: Mapping[str, float] | None = None) -> None:
        self._items = tuple(sorted((values or {}).items()))
        self._values = dict(self._items)

    def __getitem__(self, key: str) -> float:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __hash__(self) -> int:
        return hash(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())


_RECORD_STATUSES = ("ok", "empty")


@dataclass(frozen=True, kw_only=True, slots=True)
class NNEvaluationDataPoint:
    """Per-batch / per-epoch evaluation metrics.

    The four classification fields (f1, recall, accuracy, precision) are
    computed by `of()` via sklearn. `loss` and `error` are typically
    attached after the fact by NNModel during training / evaluation.

    `extra` is a free-form dict of user-supplied custom metric names to
    floats. Populated when NNTrainParams.extra_metrics or evaluate(extra_metrics=)
    is set; empty by default (and omitted from state() when empty so that
    pre-extra runs hash to the same run.id and pre-extra YAML loads cleanly).

    **Task records (FEAT-002).** A model with a ``TaskSpec`` writes records
    that also carry the task `kind` (``"categorical"`` / ``"multilabel"`` /
    ``"regression"``), the `count` of valid (unmasked) targets they
    summarize, a `status` (``"ok"``, or ``"empty"`` when every target was
    masked) and the task's own `metrics` (``mse`` / ``mae`` for regression,
    ``subset_accuracy`` / ``element_accuracy`` for multilabel). A
    regression record leaves the classification fields and `error` as
    ``None`` rather than fabricating them; an empty record has no loss or
    metrics at all. All four task fields are omitted from `state()` on
    legacy records, whose serialization is unchanged.
    """

    f1: Optional[float] = None
    recall: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    loss: Optional[float] = None
    error: Optional[float] = None

    # Custom metrics injected by the caller. Keys are metric names; values
    # are floats. Default factory keeps the dataclass hashable-by-value via
    # the dict default.
    extra: Mapping[str, float] = field(default_factory=_FrozenMetrics)

    # Task-record fields (FEAT-002) — None / empty on legacy records.
    kind: Optional[str] = None
    count: Optional[int] = None
    status: Optional[str] = None
    metrics: Mapping[str, float] = field(default_factory=_FrozenMetrics)

    def __post_init__(self) -> None:
        if not isinstance(self.extra, _FrozenMetrics):
            object.__setattr__(self, "extra", _FrozenMetrics(self.extra))
        if not isinstance(self.metrics, _FrozenMetrics):
            object.__setattr__(self, "metrics", _FrozenMetrics(self.metrics))
        if self.kind is None:
            if self.count is not None or self.status is not None:
                raise ValueError("NNEvaluationDataPoint count / status describe a task record and need a kind")
            return
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError(f"NNEvaluationDataPoint kind must be a non-empty string, got {self.kind!r}")
        count = self.count
        # CSV reload yields floats for integer columns: accept integral values.
        if isinstance(count, bool) or not isinstance(count, numbers.Real) or count != int(count) or count < 0:
            raise ValueError(f"NNEvaluationDataPoint count must be a non-negative integer, got {count!r}")
        object.__setattr__(self, "count", int(count))
        expected = "ok" if self.count else "empty"
        if self.status != expected:
            raise ValueError(
                f"NNEvaluationDataPoint status must be {expected!r} for count={self.count}, got {self.status!r}"
            )

    def with_loss(self, value: float):
        return replace(self, loss=value)

    def with_error(self, value: float):
        return replace(self, error=value)

    def with_extra(self, name: str, value: float) -> NNEvaluationDataPoint:
        merged = {**self.extra, name: float(value)}
        return replace(self, extra=merged)

    @staticmethod
    def of(
        Y: np.ndarray,
        Y_hat: np.ndarray,
        average: str = "macro",
        extra_metrics: Optional[Mapping[str, Callable]] = None,
    ):
        """Compute per-batch evaluation metrics.

        `average` controls how f1/precision/recall reduce across classes.
        Default "macro" treats all classes equally — the right choice for
        multi-class classification and the only one that makes f1/precision/
        recall mathematically distinct from accuracy. Pass "micro" to
        recover the legacy behavior (numerically identical to accuracy for
        single-label multi-class). Accuracy itself is not affected.

        `extra_metrics` is a {name -> callable(y_true, y_pred) -> float} map
        of user-supplied custom metrics, called as ``fn(Y, Y_hat)`` — truth
        first, decoded predictions second — once on the arrays passed here
        (one batch in the default training step, the aggregate in
        `NNModel.evaluate`) and stored in the returned object's `extra` dict.
        """
        extra: dict[str, float] = {}
        if extra_metrics:
            for name, fn in extra_metrics.items():
                extra[name] = float(fn(Y, Y_hat))

        return NNEvaluationDataPoint(
            accuracy=float(metrics.accuracy_score(y_true=Y, y_pred=Y_hat)),
            f1=float(metrics.f1_score(y_true=Y, y_pred=Y_hat, average=average, zero_division=cast(Any, 0))),
            recall=float(metrics.recall_score(y_true=Y, y_pred=Y_hat, average=average, zero_division=cast(Any, 0))),
            precision=float(
                metrics.precision_score(y_true=Y, y_pred=Y_hat, average=average, zero_division=cast(Any, 0))
            ),
            extra=extra,
        )

    @staticmethod
    def mean_of(edps: list[NNEvaluationDataPoint]) -> NNEvaluationDataPoint:
        """Unweighted-mean reduce a list of EDPs across every metric.

        .. warning::

            This is a **simple mean across edps**, NOT a sample-weighted
            mean. With unequal batch sizes (the common case), the result
            is statistically incorrect — a 1024-sample batch counts the
            same as an 8-sample tail batch. For correct sample-weighted
            metrics across batches, use :meth:`NNModel.evaluate`, which
            concatenates predictions across the loader and computes once
            on the full sample.

            ``mean_of`` is kept for back-compat with callers that already
            depend on the unweighted-mean semantics; new code should
            prefer :meth:`NNModel.evaluate` unless the unweighted form is
            specifically what's wanted (e.g., averaging across runs, not
            across batches within a run).

        An ``extra`` key present on some but not all edps is averaged over
        the edps where it IS present (skipped on the rest).
        """
        if not edps:
            raise ValueError("mean_of() requires at least one evaluation data point")

        # Aggregate the classification fields over the edps that carry them
        # (task records such as regression leave them None).
        def _mean(name: str) -> Optional[float]:
            values = [getattr(edp, name) for edp in edps if getattr(edp, name) is not None]
            return float(np.mean(values)) if values else None

        ret = NNEvaluationDataPoint(
            f1=_mean("f1"),
            recall=_mean("recall"),
            accuracy=_mean("accuracy"),
            precision=_mean("precision"),
        )

        if len([edp.loss for edp in edps if edp.loss is not None]) > 0:
            ret = ret.with_loss(float(np.mean([edp.loss for edp in edps if edp.loss is not None])))

        if len([edp.error for edp in edps if edp.error is not None]) > 0:
            ret = ret.with_error(float(np.mean([edp.error for edp in edps if edp.error is not None])))

        # Propagate extras: union the key set across edps, mean per key
        # over the edps that have it. Keys missing from some edps are
        # skipped on those, not zero-filled.
        all_extra_keys: set[str] = set()
        for edp in edps:
            all_extra_keys.update(edp.extra.keys())
        if all_extra_keys:
            extra_mean: dict[str, float] = {}
            for k in all_extra_keys:
                values = [edp.extra[k] for edp in edps if k in edp.extra]
                if values:
                    extra_mean[k] = float(np.mean(values))
            ret = replace(ret, extra=extra_mean)

        # Task metrics (FEAT-002) follow the same per-key rule. The result is
        # a plain mean, not a merged task record, so kind / count / status
        # are not carried (use NNModel.evaluate for whole-dataset records).
        metric_keys = sorted({k for edp in edps for k in edp.metrics})
        if metric_keys:
            ret = replace(
                ret,
                metrics={k: float(np.mean([edp.metrics[k] for edp in edps if k in edp.metrics])) for k in metric_keys},
            )

        return ret

    def state(self) -> dict:
        d: dict[str, object] = dict(
            f1=self.f1,
            recall=self.recall,
            accuracy=self.accuracy,
            precision=self.precision,
            loss=self.loss,
            error=self.error,
        )
        # Omit `extra` when empty so EDPs from before this field existed
        # remain bit-for-bit identical in state() form (preserves run.id
        # back-compat).
        if self.extra:
            d["extra"] = dict(self.extra)
        # Task-record fields (FEAT-002): omitted on legacy records, so their
        # serialized form (idps.csv columns, checkpoint metadata) is unchanged.
        if self.kind is not None:
            d["kind"] = self.kind
            d["count"] = self.count
            d["status"] = self.status
        if self.metrics:
            d["metrics"] = dict(self.metrics)
        return d

    @staticmethod
    def from_state(state: dict) -> NNEvaluationDataPoint:
        return NNEvaluationDataPoint(
            f1=state.get("f1"),
            recall=state.get("recall"),
            accuracy=state.get("accuracy"),
            precision=state.get("precision"),
            loss=state.get("loss"),
            error=state.get("error"),
            extra=dict(state.get("extra") or {}),
            kind=state.get("kind"),
            count=state.get("count"),
            status=state.get("status"),
            metrics=dict(state.get("metrics") or {}),
        )
