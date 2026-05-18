from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True, slots=True)
class NNSchedulerParams:
    min_lr      : float
    factor      : float
    patience    : int
    cooldown    : int
    threshold   : float
    
    def __str__(self) -> str:
        return f"[patience={self.patience}, cooldown={self.cooldown}, factor={self.factor:1.0e}, threshold={self.threshold:1.0e}, min_lr={self.min_lr:1.0e}]"
    
    def state(self) -> dict:
        return dict(
            min_lr      = self.min_lr
            , factor    = self.factor
            , cooldown  = self.cooldown
            , patience  = self.patience
            , threshold = self.threshold
        )
    
    @staticmethod
    def from_state(state: dict) -> NNSchedulerParams:
        return NNSchedulerParams(
            min_lr      = state['min_lr']
            , factor    = state['factor']
            , patience  = state['patience']
            , cooldown  = state['cooldown']
            , threshold = state['threshold']
        )