from __future__ import annotations

from enum import Enum

import torch


class Devices(Enum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    def __call__(self) -> torch.device:
        return torch.device(self.value)

    def torch_device(self) -> torch.device:
        """Explicit alias for ``self()`` — more readable in code that mixes
        the enum and torch.device usage."""
        return torch.device(self.value)

    @staticmethod
    def get() -> Devices:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return Devices.MPS
        elif torch.cuda.is_available():
            return Devices.CUDA
        else:
            return Devices.CPU

    @staticmethod
    def get_torch_device() -> torch.device:
        """Convenience: auto-detect and return the corresponding torch.device
        directly. Equivalent to ``Devices.get().torch_device()``."""
        return Devices.get().torch_device()
