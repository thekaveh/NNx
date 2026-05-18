from __future__ import annotations

from enum import Enum
from torch import nn, optim
from typing import Tuple, Union

class Optims(Enum):
    SGD             = "sgd"
    ADAM            = "adam"
    ADAM_AMSGRAD    = "adam_amsgrad"
    SGD_NESTEROV    = "sgd_nesterov"
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)
    
    def __call__(
        self
        , net           : nn.Module
        , lr_start      : float
        , weight_decay  : float
        , momentum      : Union[float, Tuple[float, float]]
    ):
        assert net is not None
        
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