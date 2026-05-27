"""Tests for NNTrainerParams — validation + state()/from_state() round-trip."""
from __future__ import annotations

import pytest

from nnx import (
    NNOptimParams,
    NNParamGroupSpec,
    NNSchedulerParams,
    NNTrainerParams,
    Optims,
    Schedulers,
)


def _g_optim() -> NNOptimParams:
    return NNOptimParams(
        name=Optims.ADAM, max_lr=2e-4, momentum=(0.5, 0.999), weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="G.*", lr=2e-4)],
    )


def _d_optim() -> NNOptimParams:
    return NNOptimParams(
        name=Optims.ADAM, max_lr=2e-4, momentum=(0.5, 0.999), weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="D.*", lr=2e-4)],
    )


def _sched() -> NNSchedulerParams:
    return NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3)


def test_trainer_params_round_trip_minimal():
    p = NNTrainerParams(
        n_epochs=5,
        optims={"G": _g_optim(), "D": _d_optim()},
    )
    p2 = NNTrainerParams.from_state(p.state())
    assert p2.n_epochs == 5
    assert set(p2.optims.keys()) == {"G", "D"}
    assert p2.optims["G"].max_lr == 2e-4
    assert p2.optims["D"].param_groups[0].name_pattern == "D.*"
    assert p2.schedulers == {}
    assert p2.seed is None
    assert p2.save_phase_checkpoints is True


def test_trainer_params_round_trip_with_schedulers_and_seed():
    p = NNTrainerParams(
        n_epochs=3,
        optims={"G": _g_optim(), "D": _d_optim()},
        schedulers={"G": _sched(), "D": _sched()},
        seed=42,
        save_phase_checkpoints=False,
    )
    p2 = NNTrainerParams.from_state(p.state())
    assert p2.seed == 42
    assert p2.save_phase_checkpoints is False
    assert set(p2.schedulers.keys()) == {"G", "D"}
    assert p2.schedulers["G"].patience == 2


def test_trainer_params_empty_optims_raises():
    with pytest.raises(ValueError, match="at least one entry"):
        NNTrainerParams(n_epochs=1, optims={})


def test_trainer_params_orphan_scheduler_raises():
    with pytest.raises(ValueError, match="not present in optims"):
        NNTrainerParams(
            n_epochs=1,
            optims={"G": _g_optim()},
            schedulers={"D": _sched()},
        )


def test_trainer_params_state_keys_sorted_deterministic():
    """Two NNTrainerParams that differ only by dict insertion order
    must produce the same state() so run.id is deterministic."""
    p_ab = NNTrainerParams(n_epochs=1, optims={"A": _g_optim(), "B": _d_optim()})
    p_ba = NNTrainerParams(n_epochs=1, optims={"B": _d_optim(), "A": _g_optim()})
    assert p_ab.state() == p_ba.state()


def test_trainer_params_seed_omitted_from_state_when_none():
    """Back-compat preservation: state() must NOT include `seed` when it
    is at its default None — same omit-when-default pattern as
    NNTrainParams. Future fields should follow the same convention."""
    p = NNTrainerParams(n_epochs=1, optims={"G": _g_optim()})
    assert "seed" not in p.state()
    assert "save_phase_checkpoints" not in p.state()


def test_trainer_params_schedulers_omitted_from_state_when_empty():
    """Same omit-when-default invariant for the `schedulers` field. Default
    is an empty mapping; emitting `schedulers: {}` would diverge from the
    project-wide canonical pattern (every other params dataclass omits
    optional fields at their default). The omit-when-default contract is
    what keeps run.id hashes stable when new fields are introduced —
    broken three times before; pinned with regression tests now."""
    p = NNTrainerParams(n_epochs=1, optims={"G": _g_optim()})
    assert "schedulers" not in p.state()


def test_trainer_params_state_picks_up_non_plateau_scheduler():
    p = NNTrainerParams(
        n_epochs=2,
        optims={"G": _g_optim()},
        schedulers={"G": NNSchedulerParams(
            min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3,
            kind=Schedulers.COSINE_ANNEALING, T_max=10,
        )},
    )
    p2 = NNTrainerParams.from_state(p.state())
    assert p2.schedulers["G"].kind == Schedulers.COSINE_ANNEALING
    assert p2.schedulers["G"].T_max == 10
