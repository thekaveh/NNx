from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Union

from ..enum.optims import Optims


@dataclass(frozen=True, kw_only=True, slots=True)
class NNOptimParams:
    """Optimizer config.

    `momentum` is overloaded by optimizer kind:
      - For SGD / SGD_NESTEROV: a single float, the SGD momentum coefficient.
      - For ADAM / ADAM_AMSGRAD: a (beta1, beta2) tuple, passed as the
        Adam `betas=` argument. The name is retained for backwards
        compatibility — `is_valid()` enforces the per-optim shape.
    """

    name            : Optims
    max_lr          : float
    weight_decay    : float
    momentum        : Union[float, tuple[float, float]]

    def __str__(self):
        return f"[name={self.name}, max_lr={self.max_lr:1.0e}, weight_decay={self.weight_decay:1.0e}, momentum={self.momentum}]"

    def state(self):
        return dict(
            max_lr          = self.max_lr
            , momentum      = str(self.momentum)
            , name          = str(self.name)
            , weight_decay  = self.weight_decay
        )

    @staticmethod
    def from_state(rep: dict) -> NNOptimParams:
        return NNOptimParams(
            max_lr          = rep['max_lr']
            , name          = Optims(rep['name'])
            , weight_decay  = rep['weight_decay']
            , momentum      = ast.literal_eval(rep['momentum'])
        )

    def is_valid(self) -> bool:
        if self.name == Optims.SGD or self.name == Optims.SGD_NESTEROV:
            return isinstance(self.momentum, float)
        if self.name == Optims.ADAM or self.name == Optims.ADAM_AMSGRAD:
            return (
                isinstance(self.momentum, tuple)
                and len(self.momentum) == 2
                and all(isinstance(x, float) for x in self.momentum)
            )
        # Unknown enum variant — refuse rather than implicitly returning None
        # (which would short-circuit `not params.optim.is_valid()` in train()).
        return False
