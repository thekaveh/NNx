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

    # Opt-in fp16 autocast + GradScaler in train(). Only effective on CUDA;
    # silently bypassed on CPU/MPS where torch.cuda.amp is a no-op or unavailable.
    mixed_precision: bool = False

    def __str__(self) -> str:
        return f"[net={self.net}, device={self.device}, loss={self.loss}, mixed_precision={self.mixed_precision}]"

    def is_valid(self) -> bool:
        return (
            self.net is not None
            and self.device is not None
            and self.loss is not None
        )

    def state(self) -> dict:
        return dict(
            net               = str(self.net),
            loss              = str(self.loss),
            device            = str(self.device),
            mixed_precision   = self.mixed_precision,
        )

    @staticmethod
    def from_state(state: dict) -> NNModelParams:
        return NNModelParams(
            net               = Nets(state['net']),
            loss              = Losses(state['loss']),
            device            = Devices(state['device']),
            mixed_precision   = state.get('mixed_precision', False),
        )