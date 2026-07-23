from __future__ import annotations

from enum import Enum

from torch import nn


class Losses(Enum):
    CROSS_ENTROPY = "cross_entropy"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    NEGATIVE_LOG_LIKELIHOOD = "negative_log_likelihood"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    def __call__(self) -> nn.Module:
        match self:
            case Losses.CROSS_ENTROPY:
                return nn.CrossEntropyLoss()
            case Losses.MEAN_SQUARED_ERROR:
                return nn.MSELoss()
            case Losses.BINARY_CROSS_ENTROPY:
                return nn.BCEWithLogitsLoss()
            case Losses.NEGATIVE_LOG_LIKELIHOOD:
                return nn.NLLLoss()
