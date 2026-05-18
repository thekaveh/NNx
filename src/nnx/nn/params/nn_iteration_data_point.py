from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, replace

from .nn_evaluation_data_point import NNEvaluationDataPoint

@dataclass(frozen=True, kw_only=True, slots=True)
class NNIterationDataPoint:
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
            , val_edp = NNEvaluationDataPoint.from_state(
                dict(
                    loss=state['val_edp.loss']
                    , error=state['val_edp.error']
                    , accuracy=state['val_edp.accuracy']
                    , f1=state['val_edp.f1']
                    , recall=state['val_edp.recall']
                    , precision=state['val_edp.precision']
                )
            )
        )