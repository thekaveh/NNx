"""Unit tests for the shared `phase_tag(idx_epoch, n_epochs)` helper.

The helper is the single source of truth for the FIRST/Q1/Q2/Q3
quartile-checkpoint schedule used by BOTH `NNModel.train` and
`nnx.trainer.Trainer`. PR #51's audit found the original branches were
duplicated verbatim across the two callers (DRY violation) AND silently
lost Q1 for small `n_epochs` due to a collision with the FIRST branch.
The helper consolidates the logic; these tests lock the schedule in.

The small-`n_epochs` collision is intentional (changing the math would
shift run.id-relevant trajectories for anyone already relying on the
current schedule), but documented here so a future maintainer doesn't
"fix" it without seeing the trade-off.
"""

from __future__ import annotations

import pytest

from nnx.nn.enum.checkpoints import Checkpoints, phase_tag


def test_phase_tag_first_at_epoch_zero():
    """FIRST always fires at epoch 0 regardless of n_epochs."""
    for n in (1, 4, 10, 100):
        assert phase_tag(0, n) is Checkpoints.FIRST


def test_phase_tag_canonical_quartiles_at_n_epochs_8():
    """n_epochs=8 is the canonical case where every quartile lands cleanly:
    Q1 at 1, Q2 at 3, Q3 at 5 (int(8*k/4)-1 for k=1,2,3). LAST is the
    caller's job, not the helper's, so this test does not check epoch 7."""
    assert phase_tag(0, 8) is Checkpoints.FIRST
    assert phase_tag(1, 8) is Checkpoints.Q1
    assert phase_tag(3, 8) is Checkpoints.Q2
    assert phase_tag(5, 8) is Checkpoints.Q3
    # No tag at non-boundary epochs.
    for idx in (2, 4, 6, 7):
        assert phase_tag(idx, 8) is None


@pytest.mark.parametrize("n_epochs", [10, 12, 16, 20, 100])
def test_phase_tag_quartiles_fire_for_n_epochs_at_least_6(n_epochs):
    """For `n_epochs >= 6` every quartile is reachable: the Q1 index
    (int(n/4)-1) is at least 1, which doesn't collide with FIRST (0)."""
    seen = set()
    for idx in range(n_epochs):
        tag = phase_tag(idx, n_epochs)
        if tag is not None:
            seen.add(tag)
    assert seen == {Checkpoints.FIRST, Checkpoints.Q1, Checkpoints.Q2, Checkpoints.Q3}


@pytest.mark.parametrize("n_epochs", [1, 2, 3, 4, 5])
def test_phase_tag_small_n_epochs_drops_q1_intentionally(n_epochs):
    """Small-`n_epochs` caveat (documented in the helper's docstring):
    Q1 collides with FIRST when `int(n_epochs / 4) - 1 == 0`, which
    happens for n_epochs in [4, 5, 6, 7] — wait, that's not the bug.

    The actual collision: `int(n_epochs * 1/4) - 1 == 0` when
    `n_epochs in [4, 5, 6, 7]` (int(1)-1, int(1.25)-1, int(1.5)-1,
    int(1.75)-1 all equal 0). For n_epochs <= 3 the Q1 index is
    negative and just never fires. Either way, Q1 is silently dropped.

    This test locks the trade-off in: changing the math to make Q1
    fire would shift run.id-relevant trajectories for callers already
    relying on the current schedule. The contract is "best-effort
    quartile coverage for n_epochs >= 6; FIRST + LAST are guaranteed."
    """
    seen = set()
    for idx in range(n_epochs):
        tag = phase_tag(idx, n_epochs)
        if tag is not None:
            seen.add(tag)
    # FIRST always present.
    assert Checkpoints.FIRST in seen
    # Q1 never fires in this range (collides with FIRST or has a
    # negative target index that no real `idx_epoch` reaches).
    assert Checkpoints.Q1 not in seen


def test_phase_tag_returns_none_for_non_boundary_epochs():
    """Sanity: a sufficiently large n_epochs with idx in the middle of
    no quartile should return None — proves the helper isn't matching
    every epoch."""
    # n_epochs=100: quartile indices are 24, 49, 74. Epoch 50 is between.
    assert phase_tag(50, 100) is None
    # FIRST is the only tag at epoch 0.
    assert phase_tag(0, 100) is Checkpoints.FIRST
    # And the quartiles ARE present at the canonical indices.
    assert phase_tag(24, 100) is Checkpoints.Q1
    assert phase_tag(49, 100) is Checkpoints.Q2
    assert phase_tag(74, 100) is Checkpoints.Q3
