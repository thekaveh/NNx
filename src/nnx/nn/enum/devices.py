from __future__ import annotations

import torch

from enum import Enum

class Devices(Enum):
    CPU     = "cpu"
    MPS     = "mps"
    CUDA    = "cuda"
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return str(self)

    def __call__(self):
        return torch.device(self.value)

    @staticmethod
    def get():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return Devices.MPS
        elif torch.cuda.is_available():
            return Devices.CUDA
        else:
            return Devices.CPU