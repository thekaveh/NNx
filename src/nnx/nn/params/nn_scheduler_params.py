from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..enum.schedulers import Schedulers

if TYPE_CHECKING:
    from .nn_scheduler_params_builder import NNSchedulerParamsBuilder


@dataclass(frozen=True, kw_only=True, slots=True)
class NNSchedulerParams:
    min_lr: float
    factor: float
    patience: int
    cooldown: int
    threshold: float

    # Optional. Default None preserves ReduceLROnPlateau behavior (the only
    # scheduler before the Schedulers enum was added).
    kind: Optional[Schedulers] = None

    # Kind-specific config. Each variant uses a subset; unused fields stay None.
    step_size: Optional[int] = None  # STEP
    T_max: Optional[int] = None  # COSINE_ANNEALING
    max_lr: Optional[float] = None  # ONE_CYCLE
    total_steps: Optional[int] = None  # ONE_CYCLE, LINEAR_WARMUP_DECAY
    warmup_steps: Optional[int] = None  # LINEAR_WARMUP_DECAY

    def __str__(self) -> str:
        if self.kind is None:
            return (
                f"[plateau, patience={self.patience}, cooldown={self.cooldown}, "
                f"factor={self.factor:1.0e}, threshold={self.threshold:1.0e}, "
                f"min_lr={self.min_lr:1.0e}]"
            )
        return f"[{self.kind}, factor={self.factor:1.0e}, min_lr={self.min_lr:1.0e}]"

    def state(self) -> dict:
        d = dict(
            min_lr=self.min_lr,
            factor=self.factor,
            cooldown=self.cooldown,
            patience=self.patience,
            threshold=self.threshold,
        )
        # `kind` and its variant-specific knobs (step_size, T_max, max_lr,
        # total_steps, warmup_steps) are omitted from state() when at
        # their defaults so a plain ReduceLROnPlateau NNSchedulerParams
        # hashes to the same run.id as before the Schedulers enum
        # existed. Same omit-when-default invariant as NNTrainParams.seed
        # / NNModelParams.mixed_precision / NNOptimParams.param_groups.
        if self.kind is not None:
            d["kind"] = self.kind.value
        if self.step_size is not None:
            d["step_size"] = self.step_size
        if self.T_max is not None:
            d["T_max"] = self.T_max
        if self.max_lr is not None:
            d["max_lr"] = self.max_lr
        if self.total_steps is not None:
            d["total_steps"] = self.total_steps
        if self.warmup_steps is not None:
            d["warmup_steps"] = self.warmup_steps
        return d

    @staticmethod
    def from_state(state: dict) -> NNSchedulerParams:
        kind_str = state.get("kind")
        return NNSchedulerParams(
            min_lr=state["min_lr"],
            factor=state["factor"],
            patience=state["patience"],
            cooldown=state["cooldown"],
            threshold=state["threshold"],
            kind=Schedulers(kind_str) if kind_str else None,
            step_size=state.get("step_size"),
            T_max=state.get("T_max"),
            max_lr=state.get("max_lr"),
            total_steps=state.get("total_steps"),
            warmup_steps=state.get("warmup_steps"),
        )

    @classmethod
    def builder(cls) -> NNSchedulerParamsBuilder:
        """Return a variant-aware builder. See `NNSchedulerParamsBuilder`."""
        from .nn_scheduler_params_builder import NNSchedulerParamsBuilder

        return NNSchedulerParamsBuilder()
