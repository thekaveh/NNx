"""Tests for NNSchedulerParams.builder() — variant-gated construction.

Asserts the Builder produces dataclasses that round-trip through
state()/from_state() AND preserve the omit-when-default invariant
(no `kind` / no variant-specific key in state() when the user used
the plateau path).
"""

from __future__ import annotations

import pytest

from nnx.nn.enum.schedulers import Schedulers
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


def test_builder_step_variant():
    """STEP variant: kind=STEP, step_size set, T_max / max_lr /
    total_steps / warmup_steps remain None. Plateau defaults stay
    too — they're harmless defaults for ReduceLROnPlateau-equivalent
    fields the StepLR ctor ignores."""
    sp = (
        NNSchedulerParams.builder()
        .step(step_size=30, min_lr=1e-7, factor=0.5, patience=10, cooldown=2, threshold=1e-3)
        .build()
    )
    assert sp.kind == Schedulers.STEP
    assert sp.step_size == 30
    assert sp.T_max is None
    assert sp.max_lr is None
    assert sp.total_steps is None
    assert sp.warmup_steps is None
    # Round-trips through state() / from_state().
    assert NNSchedulerParams.from_state(sp.state()) == sp


def test_builder_cosine_annealing_variant():
    sp = (
        NNSchedulerParams.builder()
        .cosine_annealing(T_max=100, min_lr=1e-7, factor=0.5, patience=10, cooldown=2, threshold=1e-3)
        .build()
    )
    assert sp.kind == Schedulers.COSINE_ANNEALING
    assert sp.T_max == 100
    assert sp.step_size is None
    assert NNSchedulerParams.from_state(sp.state()) == sp


def test_builder_one_cycle_variant():
    sp = (
        NNSchedulerParams.builder()
        .one_cycle(max_lr=1e-3, total_steps=10_000, min_lr=1e-7, factor=0.5, patience=10, cooldown=2, threshold=1e-3)
        .build()
    )
    assert sp.kind == Schedulers.ONE_CYCLE
    assert sp.max_lr == 1e-3
    assert sp.total_steps == 10_000
    assert sp.step_size is None
    assert sp.T_max is None
    assert NNSchedulerParams.from_state(sp.state()) == sp


def test_builder_linear_warmup_decay_variant():
    sp = (
        NNSchedulerParams.builder()
        .linear_warmup_decay(
            warmup_steps=500,
            total_steps=10_000,
            min_lr=1e-7,
            factor=0.5,
            patience=10,
            cooldown=2,
            threshold=1e-3,
        )
        .build()
    )
    assert sp.kind == Schedulers.LINEAR_WARMUP_DECAY
    assert sp.warmup_steps == 500
    assert sp.total_steps == 10_000
    assert NNSchedulerParams.from_state(sp.state()) == sp


def test_builder_last_variant_wins_when_called_twice():
    """The Builder is forgiving: calling two variant methods on the
    same instance overwrites the first. Documents the contract."""
    sp = (
        NNSchedulerParams.builder()
        .step(step_size=30, min_lr=1e-7, factor=0.5, patience=10, cooldown=2, threshold=1e-3)
        .one_cycle(max_lr=1e-3, total_steps=10_000, min_lr=1e-7, factor=0.5, patience=10, cooldown=2, threshold=1e-3)
        .build()
    )
    assert sp.kind == Schedulers.ONE_CYCLE
    assert sp.step_size is None  # overwritten
    assert sp.max_lr == 1e-3


def test_builder_build_without_variant_raises():
    """Calling .build() before selecting a variant raises an actionable
    Builder-level ValueError naming the five variant methods — matches
    the [[builder-pattern-shape]] §11b convention PR #52 established on
    NNTrainerParamsBuilder. The error message must reference the
    Builder methods (.reduce_on_plateau(), .step(), ...), not the
    dataclass fields, so the user knows what to call next.
    """
    with pytest.raises(ValueError, match=r"NNSchedulerParamsBuilder.*\.reduce_on_plateau.*\.step"):
        NNSchedulerParams.builder().build()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"factor": 0.0}, "factor > 0"),
        ({"factor": -0.5}, "factor > 0"),
        ({"min_lr": -1e-7}, "min_lr >= 0"),
        ({"threshold": -1e-3}, "threshold >= 0"),
        ({"patience": -1}, "patience >= 0"),
        ({"cooldown": -1}, "cooldown >= 0"),
        ({"step_size": 0}, "step_size > 0 when set"),
        ({"T_max": -10}, "T_max > 0 when set"),
        ({"total_steps": 0}, "total_steps > 0 when set"),
        ({"warmup_steps": -5}, "warmup_steps > 0 when set"),
    ],
)
def test_direct_ctor_rejects_out_of_range_fields(overrides, match):
    """`NNSchedulerParams.__post_init__` fails fast on out-of-range numeric
    fields — the [[params-boundary-validation]] class. Required fields have
    non-negative/positive bounds; present (non-None) variant knobs must be
    positive step/length counts. None of these are emitted into state() at
    their defaults, so validation never shifts a run.id."""
    kwargs = dict(min_lr=1e-7, factor=0.5, patience=8, cooldown=2, threshold=1e-3)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        NNSchedulerParams(**kwargs)


def test_builder_rejects_invalid_one_cycle_max_lr():
    """Negative max_lr is nonsensical. `NNSchedulerParams.__post_init__`
    now fails-fast at dataclass-construction time (via the Builder's
    `.build()`), per the [[params-boundary-validation]] contract — the
    error surfaces at config construction rather than deep inside the
    OneCycleLR constructor at scheduler-build time."""
    with pytest.raises(ValueError, match="max_lr > 0 when set"):
        (
            NNSchedulerParams.builder()
            .one_cycle(
                max_lr=-1.0,
                total_steps=10_000,
                min_lr=1e-7,
                factor=0.5,
                patience=10,
                cooldown=2,
                threshold=1e-3,
            )
            .build()
        )
