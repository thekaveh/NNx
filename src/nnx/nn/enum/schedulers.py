"""Schedulers enum — wraps common torch.optim.lr_scheduler classes.

The enum's __call__ is invoked from NNModel.train() via NNSchedulerParams.kind.
Each variant takes the optimizer plus the scheduler params dataclass (which
carries variant-specific config like T_max, step_size, max_lr).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from torch.optim import Optimizer, lr_scheduler

if TYPE_CHECKING:
    from ..params.nn_scheduler_params import NNSchedulerParams


class Schedulers(Enum):
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"
    STEP = "step"
    COSINE_ANNEALING = "cosine_annealing"
    ONE_CYCLE = "one_cycle"
    LINEAR_WARMUP_DECAY = "linear_warmup_decay"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    def __call__(
        self,
        optimizer: Optimizer,
        params: NNSchedulerParams,
        n_epochs: int,
    ):
        match self:
            case Schedulers.REDUCE_LR_ON_PLATEAU:
                return lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    min_lr=params.min_lr,
                    factor=params.factor,
                    cooldown=params.cooldown,
                    patience=params.patience,
                    threshold=params.threshold,
                )
            case Schedulers.STEP:
                step_size = params.step_size if params.step_size is not None else max(1, n_epochs // 3)
                gamma = params.factor
                return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            case Schedulers.COSINE_ANNEALING:
                T_max = params.T_max if params.T_max is not None else n_epochs
                return lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=T_max,
                    eta_min=params.min_lr,
                )
            case Schedulers.ONE_CYCLE:
                max_lr = params.max_lr if params.max_lr is not None else optimizer.param_groups[0]["lr"]
                total_steps = params.total_steps if params.total_steps is not None else n_epochs
                _reject_short_total_steps(total_steps, n_epochs)
                return lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=total_steps,
                )
            case Schedulers.LINEAR_WARMUP_DECAY:
                warmup_steps = params.warmup_steps if params.warmup_steps is not None else max(1, n_epochs // 10)
                total_steps = params.total_steps if params.total_steps is not None else n_epochs
                _reject_short_total_steps(total_steps, n_epochs)

                def _lr_lambda(step: int) -> float:
                    if step < warmup_steps:
                        # step+1: the scheduler steps once per EPOCH, so
                        # a 0.0 factor at step 0 would train the entire
                        # first epoch at LR=0 (HF's per-batch stepping
                        # only wastes one batch with the 0-based form).
                        return float(step + 1) / float(max(1, warmup_steps))
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    return max(0.0, 1.0 - progress)

                return lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
def _reject_short_total_steps(total_steps: int, n_epochs: int) -> None:
    """NNx steps schedulers once per EPOCH (not per batch, the HF habit),
    so an explicit total_steps < n_epochs is always a config error:
    OneCycleLR raises mid-train at epoch total_steps+1 (losing that
    epoch's idps), and LINEAR_WARMUP_DECAY's decay clamp silently trains
    every remaining epoch at LR=0."""
    if total_steps < n_epochs:
        raise ValueError(
            f"total_steps ({total_steps}) must be >= n_epochs ({n_epochs}) — "
            "NNx schedulers step once per epoch, not per batch."
        )
