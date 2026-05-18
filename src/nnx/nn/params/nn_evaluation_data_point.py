from __future__ import annotations

import numpy as np

from sklearn import metrics

from typing import List, Optional
from dataclasses import dataclass, replace
    
@dataclass(frozen=True, kw_only=True, slots=True)
class NNEvaluationDataPoint:
    f1          : float
    recall      : float
    accuracy    : float
    precision   : float
    loss        : Optional[float]   = None
    error       : Optional[float]   = None
    
    def with_loss(self, value: float):
        return replace(self, loss=value)
    
    def with_error(self, value: float):
        return replace(self, error=value)
    
    @staticmethod
    def of(Y: np.ndarray, Y_hat: np.ndarray):
        return NNEvaluationDataPoint(
            accuracy=metrics.accuracy_score(y_true=Y, y_pred=Y_hat)
            , f1=metrics.f1_score(y_true=Y, y_pred=Y_hat, average="micro", zero_division=0)
            , recall=metrics.recall_score(y_true=Y, y_pred=Y_hat, average="micro", zero_division=0)
            , precision=metrics.precision_score(y_true=Y, y_pred=Y_hat, average="micro", zero_division=0)
        )
    
    @staticmethod
    def mean_of(edps: List[NNEvaluationDataPoint]):
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

        return ret
    
    def state(self) -> dict:
        return dict(
            f1          = self.f1
            , recall    = self.recall
            , accuracy  = self.accuracy
            , precision = self.precision
            , loss      = self.loss
            , error     = self.error
        )
    
    @staticmethod
    def from_state(state: dict) -> NNEvaluationDataPoint:
        return NNEvaluationDataPoint(
            f1          = state['f1']
            , recall    = state['recall']
            , accuracy  = state['accuracy']
            , precision = state['precision']
            , loss      = state['loss'] if state['loss'] is not None else None
            , error     = state['error'] if state['error'] is not None else None
        )