"""Tests for NNSchedulerParams.builder() — variant-gated construction.

Asserts the Builder produces dataclasses that round-trip through
state()/from_state() AND preserve the omit-when-default invariant
(no `kind` / no variant-specific key in state() when the user used
the plateau path).
"""

from __future__ import annotations

from nnx.nn.params.nn_scheduler_params import NNSchedulerParams


def test_builder_reduce_on_plateau_produces_plateau_dataclass():
    """The reduce_on_plateau variant sets the 5 plateau fields and
    leaves kind=None + every variant-specific knob None."""
    sp = (
        NNSchedulerParams.builder()
        .reduce_on_plateau(min_lr=1e-7, factor=0.5, patience=3, cooldown=1, threshold=1e-3)
        .build()
    )
    assert sp.min_lr == 1e-7
    assert sp.factor == 0.5
    assert sp.patience == 3
    assert sp.cooldown == 1
    assert sp.threshold == 1e-3
    assert sp.kind is None
    assert sp.step_size is None
    assert sp.T_max is None
    assert sp.max_lr is None
    assert sp.total_steps is None
    assert sp.warmup_steps is None


def test_builder_reduce_on_plateau_preserves_omit_when_default_invariant():
    """CRITICAL: a Builder-produced plateau NNSchedulerParams must emit
    the same state() as a direct-ctor plateau NNSchedulerParams — no
    `kind` / `step_size` / `T_max` / etc. keys. This is the
    omit-when-default invariant that PR #10 broke three times; the
    Builder rollout must preserve it.
    """
    built = (
        NNSchedulerParams.builder()
        .reduce_on_plateau(min_lr=1e-7, factor=0.5, patience=5, cooldown=2, threshold=1e-3)
        .build()
    )
    direct = NNSchedulerParams(
        min_lr=1e-7,
        factor=0.5,
        patience=5,
        cooldown=2,
        threshold=1e-3,
    )
    assert built.state() == direct.state()
    assert "kind" not in built.state()
    assert "step_size" not in built.state()
    assert "T_max" not in built.state()
    assert "max_lr" not in built.state()
    assert "total_steps" not in built.state()
    assert "warmup_steps" not in built.state()
