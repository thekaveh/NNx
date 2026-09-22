from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..._validation import require_count, require_finite_real
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

    def __post_init__(self):
        # Fail-fast on out-of-range numeric fields. None of these are emitted
        # into state() when at their defaults, so validation never shifts a
        # run.id — same [[params-boundary-validation]] contract as the other
        # params dataclasses. `factor` is the LR multiplier (the torch
        # ReduceLROnPlateau ctor additionally requires factor < 1, which it
        # enforces itself for that variant); `patience`/`cooldown` are epoch
        # counts; `min_lr`/`threshold` are non-negative bounds. Present
        # (non-None) variant knobs are positive step/length counts.
        # Real-valued fields: finite first, then the domain (FIX-020).
        require_finite_real(self.factor, "factor", owner="NNSchedulerParams", minimum=0.0, exclusive_min=True)
        require_finite_real(self.min_lr, "min_lr", owner="NNSchedulerParams", minimum=0.0)
        require_finite_real(self.threshold, "threshold", owner="NNSchedulerParams", minimum=0.0)
        if self.max_lr is not None:
            require_finite_real(
                self.max_lr,
                "max_lr",
                owner="NNSchedulerParams",
                minimum=0.0,
                exclusive_min=True,
                domain_message=f"NNSchedulerParams requires max_lr > 0 when set, got {self.max_lr}",
            )
        # Epoch counts: nonnegative integers, normalized to plain `int`
        # (FIX-021); zero is a valid "react immediately" setting for both.
        for name in ("patience", "cooldown"):
            object.__setattr__(
                self, name, require_count(getattr(self, name), name, owner="NNSchedulerParams", minimum=0)
            )
        # Positive step/length counts when supplied; `None` stays the
        # "not this variant" sentinel and is never coerced (FIX-021).
        for name in ("step_size", "T_max", "total_steps", "warmup_steps"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    require_count(
                        value,
                        name,
                        owner="NNSchedulerParams",
                        minimum=0,
                        exclusive_min=True,
                        domain_message=f"NNSchedulerParams requires {name} > 0 when set, got {value}",
                    ),
                )

    def __str__(self) -> str:
        if self.kind is None:
            return (
                f"[plateau, patience={self.patience}, cooldown={self.cooldown}, "
                f"factor={self.factor:1.0e}, threshold={self.threshold:1.0e}, "
                f"min_lr={self.min_lr:1.0e}]"
            )
        return f"[{self.kind}, factor={self.factor:1.0e}, min_lr={self.min_lr:1.0e}]"

    def state(self) -> dict:
        d: dict[str, object] = dict(
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
