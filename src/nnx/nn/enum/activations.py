from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from functools import partial

import torch.nn.functional as F


class Activations(Enum):
    ELU = "elu"
    SELU = "selu"
    TANH = "tanh"
    RELU = "relu"
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    LEAKY_RELU = "leaky_relu"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    def __call__(self) -> Callable:
        match self:
            case Activations.ELU:
                return F.elu
            case Activations.SELU:
                return F.selu
            case Activations.TANH:
                return F.tanh
            case Activations.RELU:
                return F.relu
            # F.softmax needs an explicit dim; -1 matches the last-axis
            # convention used everywhere in nnx (batch-major). functools.partial
            # (rather than a lambda) is picklable, which matters under DDP /
            # multiprocessing dataloader workers.
            case Activations.SOFTMAX:
                return partial(F.softmax, dim=-1)
            case Activations.SIGMOID:
                return F.sigmoid
            case Activations.SOFTPLUS:
                return F.softplus
            case Activations.LEAKY_RELU:
                return F.leaky_relu
