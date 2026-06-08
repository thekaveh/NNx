"""Builder for NNSchedulerParams — variant-gated construction.

The classic-GoF Builder pattern applied to a tagged-union dataclass.
Each `<variant>(...)` method sets the variant's `kind` plus the
fields the variant uses; the user can't construct an invalid
combination (e.g., `kind=STEP` without `step_size`). The Builder is
purely additive — `NNSchedulerParams(**kwargs)` still works for callers
who prefer the direct-kwarg form.

The Builder holds `self._fields: dict[str, Any]` and forwards only the
user-touched fields to the dataclass at `.build()` time. This is the
mechanism that preserves the omit-when-default state() invariant: any
field the user doesn't touch stays at the dataclass default and is
absent from `self._fields`, so `state()` continues to omit it.

See `docs/superpowers/specs/2026-05-31-builder-pattern-investigation.md`
§3.1 for the rubric scoring and design rationale.
"""

from __future__ import annotations

from typing import Any

from ..enum.schedulers import Schedulers
from .nn_scheduler_params import NNSchedulerParams


class NNSchedulerParamsBuilder:
    """Variant-aware builder for `NNSchedulerParams`.

    Reach this via `NNSchedulerParams.builder()`. Each variant method
    is self-contained — the user calls exactly one of them per builder
    instance. Calling a second variant overwrites the first (last
    write wins); `.build()` produces the dataclass.
    """

    def __init__(self) -> None:
        self._fields: dict[str, Any] = {}

    def reduce_on_plateau(
        self,
        *,
        min_lr: float,
        factor: float,
        patience: int,
        cooldown: int,
        threshold: float,
    ) -> NNSchedulerParamsBuilder:
        """ReduceLROnPlateau — the default scheduler.

        Sets the five plateau fields. `kind` is left at None (the
        dataclass default), which preserves the omit-when-default
        state() invariant for callers who used the original pre-enum
        config.
        """
        self._fields = {
            "min_lr": min_lr,
            "factor": factor,
            "patience": patience,
            "cooldown": cooldown,
            "threshold": threshold,
        }
        return self

    def step(
        self,
        *,
        step_size: int,
        min_lr: float,
        factor: float,
        patience: int,
        cooldown: int,
        threshold: float,
    ) -> NNSchedulerParamsBuilder:
        """torch.optim.lr_scheduler.StepLR — decay LR by `factor` every
        `step_size` epochs. The plateau-shape fields (`min_lr`,
        `patience`, `cooldown`, `threshold`) are not consumed by
        StepLR but are required by the underlying NNSchedulerParams
        dataclass and serialised for back-compat.
        """
        self._fields = {
            "kind": Schedulers.STEP,
            "step_size": step_size,
            "min_lr": min_lr,
            "factor": factor,
            "patience": patience,
            "cooldown": cooldown,
            "threshold": threshold,
        }
        return self

    def cosine_annealing(
        self,
        *,
        T_max: int,
        min_lr: float,
        factor: float,
        patience: int,
        cooldown: int,
        threshold: float,
    ) -> NNSchedulerParamsBuilder:
        """torch.optim.lr_scheduler.CosineAnnealingLR — anneal LR over
        `T_max` steps.
        """
        self._fields = {
            "kind": Schedulers.COSINE_ANNEALING,
            "T_max": T_max,
            "min_lr": min_lr,
            "factor": factor,
            "patience": patience,
            "cooldown": cooldown,
            "threshold": threshold,
        }
        return self

    def one_cycle(
        self,
        *,
        max_lr: float,
        total_steps: int,
        min_lr: float,
        factor: float,
        patience: int,
        cooldown: int,
        threshold: float,
    ) -> NNSchedulerParamsBuilder:
        """torch.optim.lr_scheduler.OneCycleLR — Smith one-cycle schedule
        with peak LR `max_lr` over `total_steps` steps.
        """
        self._fields = {
            "kind": Schedulers.ONE_CYCLE,
            "max_lr": max_lr,
            "total_steps": total_steps,
            "min_lr": min_lr,
            "factor": factor,
            "patience": patience,
            "cooldown": cooldown,
            "threshold": threshold,
        }
        return self

    def linear_warmup_decay(
        self,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr: float,
        factor: float,
        patience: int,
        cooldown: int,
        threshold: float,
    ) -> NNSchedulerParamsBuilder:
        """Linear warm-up to `max_lr` over `warmup_steps`, linear decay
        to 0 over the remaining `total_steps - warmup_steps`. Used by
        most transformer training recipes.
        """
        self._fields = {
            "kind": Schedulers.LINEAR_WARMUP_DECAY,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "min_lr": min_lr,
            "factor": factor,
            "patience": patience,
            "cooldown": cooldown,
            "threshold": threshold,
        }
        return self

    def build(self) -> NNSchedulerParams:
        """Construct the dataclass from the fields the user touched.

        Pre-empts the dataclass's missing-required-argument TypeError
        with an actionable Builder-level ValueError naming the variant
        methods — matches the [[builder-pattern-shape]] §11b convention
        that PR #52 established on NNTrainerParamsBuilder.

        Forwards only the keys present in `self._fields` so the
        dataclass defaults govern every untouched field — that's what
        preserves the omit-when-default state() invariant.

        Raises:
            ValueError: if no variant method (`.reduce_on_plateau`,
                `.step`, `.cosine_annealing`, `.one_cycle`,
                `.linear_warmup_decay`) was called before `.build()`.
                The message names the five methods so the user can
                fix the chain without consulting the dataclass schema.
        """
        # Every variant fills the 5 plateau-shape fields. An empty
        # _fields means no variant was called.
        if "min_lr" not in self._fields:
            raise ValueError(
                "NNSchedulerParamsBuilder: call one of .reduce_on_plateau(...), "
                ".step(...), .cosine_annealing(...), .one_cycle(...), or "
                ".linear_warmup_decay(...) before .build() — a variant selects "
                "the scheduler kind and sets the required plateau-shape fields."
            )
        return NNSchedulerParams(**self._fields)
