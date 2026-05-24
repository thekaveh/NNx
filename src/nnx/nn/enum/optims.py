from __future__ import annotations

from enum import Enum
from typing import Union

from torch import nn, optim


class Optims(Enum):
    SGD             = "sgd"
    ADAM            = "adam"
    ADAM_AMSGRAD    = "adam_amsgrad"
    SGD_NESTEROV    = "sgd_nesterov"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    def __call__(
        self
        , net           : nn.Module
        , lr_start      : float
        , weight_decay  : float
        , momentum      : Union[float, tuple[float, float]]
    ) -> optim.Optimizer:
        if net is None:
            raise ValueError("net must not be None")

        match self:
            case Optims.SGD:
                return optim.SGD(
                    lr=lr_start
                    , momentum=momentum
                    , params=net.parameters()
                    , weight_decay=weight_decay
                )
            case Optims.ADAM:
                return optim.Adam(
                    lr=lr_start
                    , betas=momentum
                    , params=net.parameters()
                    , weight_decay=weight_decay
                )
            case Optims.ADAM_AMSGRAD:
                return optim.Adam(
                    amsgrad=True
                    , lr=lr_start
                    , betas=momentum
                    , params=net.parameters()
                    , weight_decay=weight_decay
                )
            case Optims.SGD_NESTEROV:
                return optim.SGD(
                    nesterov=True
                    , lr=lr_start
                    , momentum=momentum
                    , params=net.parameters()
                    , weight_decay=weight_decay
                )
