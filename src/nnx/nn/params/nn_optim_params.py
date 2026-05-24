from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional, Union

from ..enum.optims import Optims


@dataclass(frozen=True, kw_only=True, slots=True)
class NNOptimParams:
    """Optimizer config.

    `momentum` is overloaded by optimizer kind:
      - For SGD / SGD_NESTEROV: a single float, the SGD momentum coefficient.
      - For ADAM / ADAM_AMSGRAD: a (beta1, beta2) tuple, passed as the
        Adam `betas=` argument. The name is retained for backwards
        compatibility — `is_valid()` enforces the per-optim shape.

    `grad_clip_norm` clips gradients by global L2 norm before optimizer.step().
    None = no clipping (back-compat default). Typical values: 1.0 for
    transformers, 5.0 for RNNs.

    `accumulate_grad_batches` enables gradient accumulation — the effective
    batch size becomes batch_size * accumulate_grad_batches. The loss is
    scaled by 1/N so the accumulated gradient is the mean across N batches.
    Default 1 (back-compat: step every batch).
    """

    name            : Optims
    max_lr          : float
    weight_decay    : float
    momentum        : Union[float, tuple[float, float]]

    grad_clip_norm  : Optional[float] = None
    accumulate_grad_batches: int      = 1

    def __str__(self):
        return f"[name={self.name}, max_lr={self.max_lr:1.0e}, weight_decay={self.weight_decay:1.0e}, momentum={self.momentum}, grad_clip={self.grad_clip_norm}, accum={self.accumulate_grad_batches}]"

    def state(self):
        d = dict(
            max_lr          = self.max_lr
            , momentum      = str(self.momentum)
            , name          = str(self.name)
            , weight_decay  = self.weight_decay
        )
        # grad_clip_norm / accumulate_grad_batches: only emit when set to a
        # non-default value, so a NNOptimParams with neither set hashes to
        # the same run.id as before these fields existed. Existing on-disk
        # YAML without these keys is loadable via the .get() defaults below.
        if self.grad_clip_norm is not None:
            d['grad_clip_norm'] = self.grad_clip_norm
        if self.accumulate_grad_batches != 1:
            d['accumulate_grad_batches'] = self.accumulate_grad_batches
        return d

    @staticmethod
    def from_state(rep: dict) -> NNOptimParams:
        return NNOptimParams(
            max_lr          = rep['max_lr']
            , name          = Optims(rep['name'])
            , weight_decay  = rep['weight_decay']
            , momentum      = ast.literal_eval(rep['momentum'])
            # .get() preserves back-compat with older YAML that predates
            # grad_clip_norm / accumulate_grad_batches.
            , grad_clip_norm= rep.get('grad_clip_norm')
            , accumulate_grad_batches = rep.get('accumulate_grad_batches', 1)
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
