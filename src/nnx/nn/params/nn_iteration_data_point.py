from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .nn_evaluation_data_point import NNEvaluationDataPoint


@dataclass(frozen=True, kw_only=True, slots=True)
class NNIterationDataPoint:
    """One row in the per-iteration training log.

    `train_edp` is computed from the current batch only. `val_edp` is the
    per-epoch validation evaluation — populated **only on the last idp of
    each epoch** (the idp at which the validation loop ran). Other idps in
    the same epoch have `val_edp=None`. When reading idps.csv, group by
    epoch_idx and take the row with val_edp set for per-epoch validation
    metrics.
    """

    lr          : float
    iter_idx    : int
    epoch_idx   : int
    batch_idx   : int
    train_edp   : NNEvaluationDataPoint
    val_edp     : Optional[NNEvaluationDataPoint]   = None

    def with_val_edp(self, value: NNEvaluationDataPoint):
        return replace(self, val_edp=value)

    def state(self) -> dict:
        return dict(
            lr          = self.lr
            , iter_idx  = self.iter_idx
            , epoch_idx = self.epoch_idx
            , batch_idx = self.batch_idx
            , train_edp = self.train_edp.state()
            , val_edp   = self.val_edp.state() if self.val_edp is not None else None
        )

    @staticmethod
    def from_state(state: dict) -> NNIterationDataPoint:
        val_edp = None
        if state.get('val_edp.loss') is not None or any(
            state.get(f'val_edp.{k}') is not None
            for k in ('error', 'accuracy', 'f1', 'recall', 'precision')
        ):
            val_edp = NNEvaluationDataPoint.from_state(
                dict(
                    loss=state.get('val_edp.loss')
                    , error=state.get('val_edp.error')
                    , accuracy=state.get('val_edp.accuracy')
                    , f1=state.get('val_edp.f1')
                    , recall=state.get('val_edp.recall')
                    , precision=state.get('val_edp.precision')
                )
            )
        return NNIterationDataPoint(
            lr          = state['lr']
            , iter_idx  = state['iter_idx']
            , epoch_idx = state['epoch_idx']
            , batch_idx = state['batch_idx']
            , train_edp = NNEvaluationDataPoint.from_state(
                dict(
                    loss=state['train_edp.loss']
                    , error=state['train_edp.error']
                    , accuracy=state['train_edp.accuracy']
                    , f1=state['train_edp.f1']
                    , recall=state['train_edp.recall']
                    , precision=state['train_edp.precision']
                )
            )
            , val_edp = val_edp
        )
