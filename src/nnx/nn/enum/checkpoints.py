from __future__ import annotations

from enum import Enum
from typing import Optional


class Checkpoints(Enum):
    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    BEST = "best"
    LAST = "last"
    FIRST = "first"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)


def phase_tag(idx_epoch: int, n_epochs: int) -> Optional[Checkpoints]:
    """Return the phase-checkpoint tag for `idx_epoch` of `n_epochs`, or None.

    The tag schedule is FIRST at epoch 0, then Q1 / Q2 / Q3 at approximately
    the 1/4, 2/4, 3/4 boundaries (floor-arithmetic, off-by-one allowed when
    `n_epochs` isn't divisible by 4). LAST is always written by the caller
    after the loop body and is intentionally NOT returned here.

    Single source of truth used by both `NNModel.train` and
    `nnx.trainer.Trainer` so the phase-boundary semantics stay in lock-step.

    **Small-`n_epochs` caveat (silent, by design):** with `n_epochs <= 7`,
    the Q1 epoch index is 0 (or negative) and collides with FIRST, so Q1 is
    never written — FIRST always takes precedence. Q2 is also missing for
    `n_epochs <= 3`, and Q3 for `n_epochs <= 2` (their indices are negative
    or zero). Callers who need every quartile should train for at least
    `n_epochs >= 8`; below that, expect partial coverage. The caveat is intentional rather than buggy because changing
    the index math would shift run.id-relevant trajectories for anyone
    already relying on the current schedule.
    """
    if idx_epoch == 0:
        return Checkpoints.FIRST
    if idx_epoch == int(n_epochs * 1 / 4) - 1:
        return Checkpoints.Q1
    if idx_epoch == int(n_epochs * 2 / 4) - 1:
        return Checkpoints.Q2
    if idx_epoch == int(n_epochs * 3 / 4) - 1:
        return Checkpoints.Q3
    return None
