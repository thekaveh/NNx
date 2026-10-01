"""Optimizer-update scheduler clocks (FEAT-014). Internal.

A scheduler configured with ``clock="optimizer_update"`` steps once per
*committed* update of the optimizer it belongs to — never per microbatch, an
all-masked window or a skipped AMP step. The update source is explicit:

- the shared update engine (FEAT-004) for an objective run;
- ``default_train_step``, after each optimizer step it actually takes;
- a custom step function that calls ``ctx.report_update()`` (``NNModel``)
  or ``ctx.report_update(name)`` (``Trainer``) after each update — NNx
  never infers updates around an opaque step.

A :class:`SchedulerClock` counts its optimizer's committed updates and, when
attached (``Trainer``'s ``auto_step_schedulers=False`` detaches it), steps
the scheduler after each. Its owner, clock, count and horizon are
checkpointed component state, so a stateful resume continues the count and
refuses a mismatched configuration before anything steps.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Optional

CLOCK = "optimizer_update"
_HORIZON_KINDS = frozenset({"one_cycle", "linear_warmup_decay"})


def uses_update_clock(scheduler_params: Any) -> bool:
    return getattr(scheduler_params, "clock", "epoch") == CLOCK


def planned_updates(loader: Any, window: int, n_epochs: int) -> Optional[int]:
    """The run's planned committed updates when NNx owns the windows:
    ``ceil(len(loader) / window)`` per epoch (a short final window still
    commits); ``None`` for a loader without a length."""
    try:
        batches = len(loader)
    except TypeError:
        return None
    return n_epochs * math.ceil(batches / max(1, window))


def update_horizon(scheduler_params: Any) -> Optional[int]:
    """The hard budget an update-clock scheduler may not step past: a
    one-cycle / warmup-decay schedule's ``total_steps`` (``None`` for the
    open-ended kinds)."""
    kind = getattr(scheduler_params, "kind", None)
    if kind is None or str(kind) not in _HORIZON_KINDS:
        return None
    return scheduler_params.total_steps


class SchedulerClock:
    """Steps one optimizer's scheduler once per committed update of that
    optimizer, and persists the count (component state)."""

    def __init__(
        self,
        owner: str,
        scheduler: Any,
        *,
        horizon: Optional[int],
        planned: Optional[int] = None,
        attached: bool = True,
        component_name: str = "nnx.scheduler_clock",
    ) -> None:
        self.owner = owner
        self.scheduler = scheduler
        self.horizon = horizon
        self.planned = planned
        self.attached = attached
        self.component_name = component_name
        self.count = 0
        # (update count, learning rate after the step) since the epoch began.
        self.trace: list[tuple[int, float]] = []

    def committed(self, optimizer: Optional[str] = None) -> None:
        """One committed update of ``optimizer`` (this clock's owner when
        ``None``): count it and, when attached, step the scheduler."""
        if optimizer is not None and optimizer != self.owner:
            return
        if self.attached and self.horizon is not None and self.count >= self.horizon:
            raise ValueError(
                f"optimizer {self.owner!r} committed update {self.count + 1}, beyond its scheduler's budget of "
                f"{self.horizon} optimizer updates (total_steps); set total_steps to cover every update of the run"
            )
        self.count += 1
        if self.attached:
            self.scheduler.step()
            self.trace.append((self.count, float(self.scheduler.optimizer.param_groups[0]["lr"])))

    # ---------- checkpointable component (FEAT-005) ----------

    def component_spec(self) -> Any:
        from .components import ComponentSpec

        return ComponentSpec(self.component_name, version=1, required=True)

    def component_state(self) -> dict[str, Any]:
        return {"clock": CLOCK, "owner": self.owner, "count": self.count, "horizon": self.horizon}

    def check_component_state(self, state: Mapping[str, Any], *, version: int) -> list[str]:
        problems: list[str] = []
        if state.get("clock") != CLOCK or state.get("owner") != self.owner:
            problems.append(
                f"the checkpoint's scheduler clock is {state.get('clock')!r} owned by {state.get('owner')!r}, this "
                f"run's is {CLOCK!r} owned by {self.owner!r}"
            )
        if state.get("horizon") != self.horizon:
            problems.append(
                f"the checkpoint's scheduler budget is {state.get('horizon')} optimizer updates, this run's is "
                f"{self.horizon}; configure the same total_steps"
            )
        count = state.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            problems.append(f"malformed update count {count!r}")
        elif self.horizon is not None and self.planned is not None and count + self.planned > self.horizon:
            problems.append(
                f"resuming at update {count} with {self.planned} planned updates would pass the scheduler's budget "
                f"of {self.horizon} (total_steps); configure one horizon covering the original and resumed updates"
            )
        return problems

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None:
        self.count = int(state["count"])
