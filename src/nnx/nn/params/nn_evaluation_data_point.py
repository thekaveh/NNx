from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from sklearn import metrics


@dataclass(frozen=True, kw_only=True, slots=True)
class NNEvaluationDataPoint:
    """Per-batch / per-epoch evaluation metrics.

    The four core fields (f1, recall, accuracy, precision) are computed by
    `of()` via sklearn. `loss` and `error` are typically attached after the
    fact by NNModel during training / evaluation.

    `extra` is a free-form dict of user-supplied custom metric names to
    floats. Populated when NNTrainParams.extra_metrics or evaluate(metrics=)
    is set; empty by default (and omitted from state() when empty so that
    pre-extra runs hash to the same run.id and pre-extra YAML loads cleanly).
    """

    f1          : float
    recall      : float
    accuracy    : float
    precision   : float
    loss        : Optional[float]   = None
    error       : Optional[float]   = None

    # Custom metrics injected by the caller. Keys are metric names; values
    # are floats. Default factory keeps the dataclass hashable-by-value via
    # the dict default.
    extra       : dict              = field(default_factory=dict)

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

        `extra_metrics` is a {name -> callable(Y, Y_hat) -> float} map of
        user-supplied custom metrics. Each is invoked once on the aggregate
        predictions and stored in the returned object's `extra` dict.
        """
        extra: dict[str, float] = {}
        if extra_metrics:
            for name, fn in extra_metrics.items():
                extra[name] = float(fn(Y, Y_hat))

        return NNEvaluationDataPoint(
            accuracy=metrics.accuracy_score(y_true=Y, y_pred=Y_hat)
            , f1=metrics.f1_score(y_true=Y, y_pred=Y_hat, average=average, zero_division=0)
            , recall=metrics.recall_score(y_true=Y, y_pred=Y_hat, average=average, zero_division=0)
            , precision=metrics.precision_score(y_true=Y, y_pred=Y_hat, average=average, zero_division=0)
            , extra=extra
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
        # Aggregate the standard fields with the existing logic.
        ret = NNEvaluationDataPoint(
            f1=np.mean([edp.f1 for edp in edps])
            , recall=np.mean([edp.recall for edp in edps])
            , accuracy=np.mean([edp.accuracy for edp in edps])
            , precision=np.mean([edp.precision for edp in edps])
        )

        if len([edp.loss for edp in edps if edp.loss is not None]) > 0:
            ret = ret.with_loss(np.mean([edp.loss for edp in edps if edp.loss is not None]))

        if len([edp.error for edp in edps if edp.error is not None]) > 0:
            ret = ret.with_error(np.mean([edp.error for edp in edps if edp.error is not None]))

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

        return ret

    def state(self) -> dict:
        d = dict(
            f1          = self.f1
            , recall    = self.recall
            , accuracy  = self.accuracy
            , precision = self.precision
            , loss      = self.loss
            , error     = self.error
        )
        # Omit `extra` when empty so EDPs from before this field existed
        # remain bit-for-bit identical in state() form (preserves run.id
        # back-compat).
        if self.extra:
            d['extra'] = dict(self.extra)
        return d

    @staticmethod
    def from_state(state: dict) -> NNEvaluationDataPoint:
        return NNEvaluationDataPoint(
            f1          = state['f1']
            , recall    = state['recall']
            , accuracy  = state['accuracy']
            , precision = state['precision']
            , loss      = state['loss']
            , error     = state['error']
            , extra     = dict(state.get('extra') or {})
        )
