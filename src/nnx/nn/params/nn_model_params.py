from __future__ import annotations

from dataclasses import dataclass

from ..enum.nets import Nets
from ..enum.losses import Losses
from ..enum.devices import Devices

@dataclass(frozen=True, kw_only=True, slots=True)
class NNModelParams:
    net   : Nets
    device: Devices = Devices.CPU
    loss  : Losses  = Losses.CROSS_ENTROPY
    
    def __str__(self):
        return f"[net={self.net}, device={self.device}, loss={self.loss}]"
    
    def is_valid(self):
        return (
            self.net is not None
            and self.device is not None
            and self.loss is not None
        )
        
    def state(self) -> dict:
        return dict(
            net         = str(self.net)
            , loss      = str(self.loss)
            , device    = str(self.device)
        )
        
    @staticmethod
    def from_state(state: dict) -> NNModelParams:
        return NNModelParams(
            net         = Nets(state['net'])
            , loss      = Losses(state['loss'])
            , device    = Devices(state['device'])
        )