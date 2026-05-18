from __future__ import annotations

import torch.nn.functional as F

from enum import Enum

class Activations(Enum):
    ELU         = 'elu'
    SELU        = 'selu'
    TANH        = 'tanh'
    RELU        = 'relu'
    SOFTMAX     = 'softmax'
    SIGMOID     = 'sigmoid'
    SOFTPLUS    = 'softplus'
    LEAKY_RELU  = 'leaky_relu'

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return str(self)
        
    def __call__(self):
        match self:
            case Activations.ELU           : return F.elu
            case Activations.SELU          : return F.selu
            case Activations.TANH          : return F.tanh
            case Activations.RELU          : return F.relu
            case Activations.SOFTMAX       : return F.softmax
            case Activations.SIGMOID       : return F.sigmoid
            case Activations.SOFTPLUS      : return F.softplus
            case Activations.LEAKY_RELU    : return F.leaky_relu