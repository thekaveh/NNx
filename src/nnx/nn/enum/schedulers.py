"""Schedulers enum — wraps common torch.optim.lr_scheduler classes.

The enum's __call__ is invoked from NNModel.train() via NNSchedulerParams.kind.
Each variant takes the optimizer plus the scheduler params dataclass (which
carries variant-specific config like T_max, step_size, max_lr).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

from torch.optim import Optimizer, lr_scheduler

from ..._scheduler_clock import uses_update_clock

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
        *,
        n_updates: Optional[int] = None,
    ):
        # FEAT-014: an "optimizer_update" clock counts committed updates, so
        # its default horizon is the run's planned updates (``n_updates``,
        # None when the loader has no length — then an explicit budget is
        # required); the "epoch" clock keeps counting epochs.
        updates = uses_update_clock(params)
        horizon = n_updates if updates else n_epochs

        def budget(value: Optional[int], field: str) -> int:
            if value is not None:
                return value
            if horizon is None:
                raise ValueError(
                    f"an optimizer_update-clock {self.value} scheduler needs an explicit {field} (its budget in "
                    "optimizer updates): NNx plans the updates only when it owns the update windows (the default "
                    "step or an objective) over a loader with a length"
                )
            return horizon

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
                step_size = params.step_size if params.step_size is not None else max(1, budget(None, "step_size") // 3)
                gamma = params.factor
                return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            case Schedulers.COSINE_ANNEALING:
                T_max = budget(params.T_max, "T_max")
                return lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=T_max,
                    eta_min=params.min_lr,
                )
            case Schedulers.ONE_CYCLE:
                max_lr = params.max_lr if params.max_lr is not None else optimizer.param_groups[0]["lr"]
                total_steps = budget(params.total_steps, "total_steps")
                _reject_short_total_steps(total_steps, horizon, updates=updates)
                return lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=total_steps,
                )
            case Schedulers.LINEAR_WARMUP_DECAY:
                total_steps = budget(params.total_steps, "total_steps")
                # The default warm-up is a tenth of the schedule: of the
                # run's epochs on the epoch clock (unchanged), and of its
                # total_steps updates on the update clock, so a split
                # update-clock run warms up exactly as the whole one.
                default_warmup = max(1, (total_steps if updates else n_epochs) // 10)
                warmup_steps = params.warmup_steps if params.warmup_steps is not None else default_warmup
                _reject_short_total_steps(total_steps, horizon, updates=updates)

                def _lr_lambda(step: int) -> float:
                    if step < warmup_steps:
                        # step+1: the first step already trains (on the
                        # epoch clock, a 0.0 factor at step 0 would train
                        # the entire first epoch at LR=0).
                        return float(step + 1) / float(max(1, warmup_steps))
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    return max(0.0, 1.0 - progress)

                return lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _reject_short_total_steps(total_steps: int, horizon: Optional[int], *, updates: bool = False) -> None:
    """An explicit total_steps shorter than the run is a config error:
    OneCycleLR raises mid-train once it is exhausted (losing that epoch's
    idps), and LINEAR_WARMUP_DECAY's decay clamp silently trains the rest at
    LR=0. On the epoch clock NNx steps once per EPOCH (not per batch, the HF
    habit); on the optimizer_update clock once per committed update."""
    if horizon is None:
        return  # unknown length: the scheduler clock refuses an overrun when it happens
    if total_steps < horizon:
        if updates:
            raise ValueError(
                f"total_steps ({total_steps}) must be >= the run's planned optimizer updates ({horizon}) — an "
                "optimizer_update-clock scheduler steps once per committed update."
            )
        raise ValueError(
            f"total_steps ({total_steps}) must be >= n_epochs ({horizon}) — "
            "NNx schedulers step once per epoch, not per batch."
        )
