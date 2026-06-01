"""Builder for NNSchedulerParams — variant-gated construction.

The classic-GoF Builder pattern applied to a tagged-union dataclass.
Each `with_<variant>(...)` method sets the variant's `kind` plus the
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

    def build(self) -> NNSchedulerParams:
        """Construct the dataclass from the fields the user touched.

        Forwards only the keys present in `self._fields` so the
        dataclass defaults govern every untouched field — that's what
        preserves the omit-when-default state() invariant.
        """
        return NNSchedulerParams(**self._fields)
