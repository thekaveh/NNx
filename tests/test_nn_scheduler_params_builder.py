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


def test_one_cycle_keeps_each_param_groups_own_peak():
    """OneCycle with several param groups peaks each group at its own LR
    (or the explicit max_lr scaled by the group's ratio to the base LR),
    never one shared value taken from the first group."""
    import torch

    from nnx import NNSchedulerParams, Schedulers

    def groups():
        # The specific group first and the default (base-LR) group last, as
        # build_param_groups orders them.
        backbone, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        return torch.optim.SGD(
            [{"params": backbone.parameters(), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-3}], lr=1e-3
        )

    base = dict(min_lr=0.0, factor=0.5, patience=0, cooldown=0, threshold=0.0, total_steps=10)
    own = groups()
    Schedulers.ONE_CYCLE(own, NNSchedulerParams(kind=Schedulers.ONE_CYCLE, **base), n_epochs=10)
    assert [group["max_lr"] for group in own.param_groups] == [1e-5, 1e-3]
    scaled = groups()
    Schedulers.ONE_CYCLE(scaled, NNSchedulerParams(kind=Schedulers.ONE_CYCLE, max_lr=1e-2, **base), n_epochs=10)
    # The default group peaks at exactly max_lr; the backbone keeps its 1/100.
    assert scaled.param_groups[1]["max_lr"] == 1e-2
    assert scaled.param_groups[0]["max_lr"] == pytest.approx(1e-4)


def test_one_cycle_default_group_peaks_at_exactly_max_lr_beside_a_multiplier_group():
    """The ratio to the base LR is taken before scaling max_lr, so the
    default group (ratio exactly 1.0) peaks at exactly max_lr even when
    max_lr * lr / base_lr would round 1 ulp off it (#547)."""
    import torch

    from nnx import NNOptimParams, NNParamGroupSpec, NNSchedulerParams, Optims, Schedulers
    from nnx.optimizers import build_optimizer

    net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    optimizer = build_optimizer(
        net,
        NNOptimParams(
            name=Optims.SGD,
            max_lr=0.1,
            momentum=0.9,
            weight_decay=0.0,
            param_groups=[NNParamGroupSpec(name_pattern="0.*", lr_multiplier=0.01)],
        ),
    )
    params = NNSchedulerParams(
        kind=Schedulers.ONE_CYCLE,
        max_lr=0.1,
        min_lr=0.0,
        factor=0.5,
        patience=0,
        cooldown=0,
        threshold=0.0,
        total_steps=10,
    )
    Schedulers.ONE_CYCLE(optimizer, params, n_epochs=10)
    scoped = {id(p) for p in net[0].parameters()}
    (multiplier_group,) = [g for g in optimizer.param_groups if {id(p) for p in g["params"]} == scoped]
    (default_group,) = [g for g in optimizer.param_groups if g is not multiplier_group]
    assert default_group["max_lr"] == 0.1
    assert multiplier_group["max_lr"] == pytest.approx(0.001)


def test_one_cycle_peaks_against_a_factory_optimizers_configured_base_lr():
    """A registered factory builds its optimizer without an lr, so its
    defaults["lr"] is the class default: the peaks follow the config's
    max_lr (the default group's rate), as for a built-in optimizer."""
    import torch

    from nnx import NNParamGroupSpec, NNSchedulerParams, Schedulers
    from nnx.optimizers import (
        NNOptimFactoryParams,
        OptimizerFactorySpec,
        build_optimizer,
        register_optimizer_factory,
        unregister_optimizer_factory,
    )

    register_optimizer_factory("test.adam_default_lr", 1, lambda groups, config: torch.optim.Adam(groups))
    try:
        net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        optimizer = build_optimizer(
            net,
            NNOptimFactoryParams(
                factory=OptimizerFactorySpec(id="test.adam_default_lr", version=1),
                max_lr=0.1,
                param_groups=[NNParamGroupSpec(name_pattern="0.*", lr_multiplier=0.5)],
            ),
        )
        params = NNSchedulerParams(
            kind=Schedulers.ONE_CYCLE,
            max_lr=1.0,
            min_lr=0.0,
            factor=0.5,
            patience=0,
            cooldown=0,
            threshold=0.0,
            total_steps=10,
        )
        Schedulers.ONE_CYCLE(optimizer, params, n_epochs=10)
        assert sorted(group["max_lr"] for group in optimizer.param_groups) == pytest.approx([0.5, 1.0])
    finally:
        unregister_optimizer_factory("test.adam_default_lr", 1)


def test_one_cycle_keeps_a_single_scoped_groups_multiplier():
    """A Trainer's strict optimizer may hold one group at lr_multiplier=0.1:
    its peak is 0.1 * max_lr, as it is beside a sibling group."""
    import torch

    from nnx import NNOptimParams, NNParamGroupSpec, NNSchedulerParams, Optims, Schedulers
    from nnx.optimizers import build_optimizer

    net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    optimizer = build_optimizer(
        net,
        NNOptimParams(
            name=Optims.SGD,
            max_lr=1e-2,
            momentum=0.9,
            weight_decay=0.0,
            param_groups=[NNParamGroupSpec(name_pattern="0.*", lr_multiplier=0.1)],
        ),
        strict_param_groups=True,
    )
    params = NNSchedulerParams(
        kind=Schedulers.ONE_CYCLE,
        max_lr=1.0,
        min_lr=0.0,
        factor=0.5,
        patience=0,
        cooldown=0,
        threshold=0.0,
        total_steps=10,
    )
    Schedulers.ONE_CYCLE(optimizer, params, n_epochs=10)
    assert [group["max_lr"] for group in optimizer.param_groups] == pytest.approx([0.1])
