"""Schedulers enum — wraps common torch.optim.lr_scheduler classes.

The enum's __call__ is invoked from NNModel.train() via NNSchedulerParams.kind.
Each variant takes the optimizer plus the scheduler params dataclass (which
carries variant-specific config like T_max, step_size, max_lr).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

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
                # One peak per param group: each group's own LR, or an
                # explicit max_lr scaled by the group's LR relative to the
                # optimizer's base LR (the default group, which NNParamGroupSpec
                # multipliers are relative to), so the default group peaks at
                # exactly max_lr and spec order never changes a peak.
                group_lrs = [float(group["lr"]) for group in optimizer.param_groups]
                # build_optimizer records the config's max_lr; a hand-built
                # optimizer falls back to its own default LR.
                base_lr = float(getattr(optimizer, "_nnx_base_lr", optimizer.defaults.get("lr", group_lrs[0])))
                if params.max_lr is None:
                    max_lr: Any = group_lrs if len(group_lrs) > 1 else group_lrs[0]
                elif base_lr > 0:
                    # Also for one group: a Trainer's strict G.* x0.1 group
                    # peaks at 0.1 * max_lr whether or not a sibling exists.
                    scaled = [params.max_lr * lr / base_lr for lr in group_lrs]
                    max_lr = scaled if len(scaled) > 1 else scaled[0]
                else:
                    max_lr = params.max_lr
                total_steps = budget(params.total_steps, "total_steps")
                _refuse(_horizon_problem(self, params, n_epochs=n_epochs, n_updates=n_updates))
                return lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=total_steps,
                )
            case Schedulers.LINEAR_WARMUP_DECAY:
                total_steps = budget(params.total_steps, "total_steps")
                _refuse(_horizon_problem(self, params, n_epochs=n_epochs, n_updates=n_updates))
                warmup_steps = _warmup_steps(params, total_steps, n_epochs=n_epochs)

                def _lr_lambda(step: int) -> float:
                    if step < warmup_steps:
                        # step+1: the first step already trains (on the
                        # epoch clock, a 0.0 factor at step 0 would train
                        # the entire first epoch at LR=0).
                        return float(step + 1) / float(max(1, warmup_steps))
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    return max(0.0, 1.0 - progress)

                return lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _horizon_problem(
    kind: Any, params: NNSchedulerParams, *, n_epochs: int, n_updates: Optional[int] = None
) -> Optional[str]:
    """Why building a ``kind`` schedule from ``params`` refuses its horizon
    or warm-up, or ``None``. Pure: no optimizer or scheduler is built. One
    implementation for :meth:`Schedulers.__call__` and the configuration
    checks of ``ExperimentPlan.validate()`` / ``fit()`` (#506).

    ``n_updates`` is the run's planned optimizer updates (``None`` when they
    are unknown). A budget the run cannot resolve (an optimizer_update clock
    with no planned updates and no explicit budget) is not this check's: the
    build refuses it on its own.

    - An explicit ``total_steps`` shorter than the run: ``OneCycleLR`` raises
      mid-train once it is exhausted (losing that epoch's records), and
      LINEAR_WARMUP_DECAY's decay clamp silently trains the rest at LR=0. On
      the epoch clock NNx steps once per EPOCH (not per batch, the HF habit);
      on the optimizer_update clock once per committed update.
    - A warm-up longer than the resolved ``total_steps`` (a default one
      too): the schedule would never reach the optimizer's LR, nor decay.
    """
    if kind not in (Schedulers.ONE_CYCLE, Schedulers.LINEAR_WARMUP_DECAY):
        return None
    updates = uses_update_clock(params)
    horizon = n_updates if updates else n_epochs
    total_steps = params.total_steps if params.total_steps is not None else horizon
    if total_steps is None:
        return None  # no budget to check: the build refuses the missing one
    if horizon is not None and total_steps < horizon:
        # An unknown length (horizon None) is not checked here: the
        # scheduler clock refuses an overrun when it happens.
        if updates:
            return (
                f"total_steps ({total_steps}) must be >= the run's planned optimizer updates ({horizon}) — an "
                "optimizer_update-clock scheduler steps once per committed update."
            )
        return (
            f"total_steps ({total_steps}) must be >= n_epochs ({horizon}) — "
            "NNx schedulers step once per epoch, not per batch."
        )
    if kind is Schedulers.LINEAR_WARMUP_DECAY:
        warmup_steps = _warmup_steps(params, total_steps, n_epochs=n_epochs)
        if warmup_steps > total_steps:
            return f"warmup_steps ({warmup_steps}) must be <= the schedule's total_steps ({total_steps})"
    return None


def _warmup_steps(params: NNSchedulerParams, total_steps: int, *, n_epochs: int) -> int:
    """A warmup-decay schedule's warm-up: ``warmup_steps``, else a tenth of
    the schedule — of the run's epochs on the epoch clock (unchanged), and
    of its ``total_steps`` updates on the update clock, so a split
    update-clock run warms up exactly as the whole one."""
    if params.warmup_steps is not None:
        return params.warmup_steps
    return max(1, (total_steps if uses_update_clock(params) else n_epochs) // 10)


def _refuse(problem: Optional[str]) -> None:
    if problem is not None:
        raise ValueError(problem)
