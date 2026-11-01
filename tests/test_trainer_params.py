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
        name=Optims.ADAM,
        max_lr=2e-4,
        momentum=(0.5, 0.999),
        weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="G.*", lr=2e-4)],
    )


def _d_optim() -> NNOptimParams:
    return NNOptimParams(
        name=Optims.ADAM,
        max_lr=2e-4,
        momentum=(0.5, 0.999),
        weight_decay=0.0,
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
        auto_step_schedulers=False,
    )
    p2 = NNTrainerParams.from_state(p.state())
    assert p2.seed == 42
    assert p2.save_phase_checkpoints is False
    assert p2.auto_step_schedulers is False
    assert set(p2.schedulers.keys()) == {"G", "D"}
    assert p2.schedulers["G"].patience == 2


def test_trainer_params_empty_optims_raises():
    with pytest.raises(ValueError, match="at least one entry"):
        NNTrainerParams(n_epochs=1, optims={})


@pytest.mark.parametrize("bad_n_epochs", [0, -1])
def test_trainer_params_rejects_non_positive_n_epochs(bad_n_epochs):
    """`NNTrainerParams.__post_init__` fails fast on `n_epochs < 1` — the
    [[params-boundary-validation]] class. `n_epochs` drives
    `range(params.n_epochs)` in Trainer.train, so a 0/negative value would
    otherwise make training a silent no-op (empty idps, no BEST checkpoint)
    rather than raising. Symmetric with NNTrainParams."""
    with pytest.raises(ValueError, match="n_epochs >= 1"):
        NNTrainerParams(n_epochs=bad_n_epochs, optims={"G": _g_optim()})


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
    assert "auto_step_schedulers" not in p.state()


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
        schedulers={
            "G": NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=2,
                cooldown=1,
                threshold=1e-3,
                kind=Schedulers.COSINE_ANNEALING,
                T_max=10,
            )
        },
    )
    p2 = NNTrainerParams.from_state(p.state())
    assert p2.schedulers["G"].kind == Schedulers.COSINE_ANNEALING
    assert p2.schedulers["G"].T_max == 10


def test_trainer_params_round_trip_a_one_cycle_scheduler_with_an_unused_warmup():
    """#546: NNRun.load reads a Trainer run's schedulers{} through
    NNTrainerParams.from_state; a ONE_CYCLE entry keeps an unused
    warmup_steps > total_steps (valid on 0.4.1) and its state()."""
    one_cycle = NNSchedulerParams(
        min_lr=0.0,
        factor=0.5,
        patience=0,
        cooldown=0,
        threshold=0.0,
        kind=Schedulers.ONE_CYCLE,
        max_lr=0.1,
        total_steps=10,
        warmup_steps=20,
    )
    p = NNTrainerParams(n_epochs=2, optims={"G": _g_optim()}, schedulers={"G": one_cycle})
    p2 = NNTrainerParams.from_state(p.state())
    assert p2 == p
    assert p2.state() == p.state()
    assert p2.schedulers["G"] == one_cycle


def _metric_identity(y, y_hat):
    """Module-level so the pickle round-trip below can serialize it."""
    return 0.5


def test_direct_trainer_params_reject_mapping_mutation():
    """FIX-010: direct construction snapshots caller-owned dicts, and the
    exposed mappings reject item assignment/deletion. `extra_metrics=None`
    stays None; an explicit empty mapping stays an (empty) mapping."""
    optims = {"g": _g_optim()}
    schedulers = {"g": _sched()}
    metrics = {"m": _metric_identity}
    params = NNTrainerParams(n_epochs=1, optims=optims, schedulers=schedulers, extra_metrics=metrics)

    optims["d"] = _d_optim()
    schedulers.clear()
    metrics.clear()
    assert set(params.optims) == {"g"}
    assert set(params.schedulers) == {"g"}
    assert params.extra_metrics is not None and set(params.extra_metrics) == {"m"}
    assert params.extra_metrics["m"] is _metric_identity

    for mapping in (params.optims, params.schedulers, params.extra_metrics):
        with pytest.raises(TypeError):
            mapping["x"] = _g_optim()  # type: ignore[index]
        with pytest.raises(TypeError):
            del mapping["g" if mapping is not params.extra_metrics else "m"]  # type: ignore[attr-defined]

    assert NNTrainerParams(n_epochs=1, optims={"g": _g_optim()}).extra_metrics is None
    explicit_empty = NNTrainerParams(n_epochs=1, optims={"g": _g_optim()}, extra_metrics={})
    assert explicit_empty.extra_metrics is not None and len(explicit_empty.extra_metrics) == 0
    assert len(NNTrainerParams(n_epochs=1, optims={"g": _g_optim()}).schedulers) == 0


def test_trainer_params_key_validation_uses_snapshot():
    """Orphan-scheduler validation must see the captured mapping — it
    cannot be satisfied by an optim the caller adds after construction."""
    optims = {"g": _g_optim()}
    with pytest.raises(ValueError, match="not present in optims"):
        NNTrainerParams(n_epochs=1, optims=optims, schedulers={"d": _sched()})


def test_trainer_mapping_copy_and_roundtrip():
    """FIX-010: the read-only mappings survive shallow/deep copy, pickle,
    dataclasses.replace and state()/from_state() — insertion order is
    kept at runtime while state() stays sorted / default-omitted."""
    import copy
    import pickle
    from dataclasses import replace

    params = NNTrainerParams(
        n_epochs=2,
        optims={"z": _g_optim(), "a": _d_optim()},
        schedulers={"z": _sched()},
        extra_metrics={"m": _metric_identity},
    )
    assert list(params.optims) == ["z", "a"]
    assert list(params.state()["optims"]) == ["a", "z"]
    assert params.optims == {"z": _g_optim(), "a": _d_optim()}

    for clone in (copy.copy(params), copy.deepcopy(params), pickle.loads(pickle.dumps(params))):
        assert list(clone.optims) == ["z", "a"]
        assert clone.optims == params.optims
        assert clone.schedulers == params.schedulers
        assert clone.extra_metrics is not None and clone.extra_metrics["m"](None, None) == 0.5
        assert clone.state() == params.state()
        with pytest.raises(TypeError):
            clone.optims["q"] = _g_optim()  # type: ignore[index]

    replaced = replace(params, n_epochs=5)
    assert replaced.optims == params.optims and list(replaced.optims) == ["z", "a"]
    assert replaced.extra_metrics is params.extra_metrics

    restored = NNTrainerParams.from_state(params.state())
    assert restored.optims == params.optims
    assert restored.schedulers == params.schedulers
    assert "extra_metrics" not in params.state()
    with pytest.raises(TypeError):
        restored.optims["q"] = _g_optim()  # type: ignore[index]


@pytest.mark.parametrize("names", [{1: None, "a": None}, {"": None}])
def test_optimizer_and_scheduler_names_must_be_non_empty_strings(names):
    """A non-string name passed construction and broke state() / str()
    (both sort the names) far from its origin."""
    optims = {key: _g_optim() for key in names}
    with pytest.raises(ValueError, match="names must be non-empty strings"):
        NNTrainerParams(n_epochs=1, optims=optims)


def test_trainer_params_store_numpy_strings_as_plain_values():
    """#515 twins of NNTrainParams: identity, lineage, optimizer and
    scheduler names, metrics and monitor reach state() as plain str."""
    import numpy as np
    import yaml

    from nnx import MetricSpec, MonitorSpec

    s = np.str_
    run_id = "4c968b3c916f88b0f36bdfd2fc855938"
    common = dict(
        n_epochs=1,
        optims={s("G"): _g_optim(), "D": _d_optim()},
        schedulers={s("G"): _sched()},
        data_id=s("ds-1"),
        metrics=(MetricSpec(s("accuracy"), name=s("acc")),),
        monitor=MonitorSpec(
            metric=s("acc"), split=s("val"), mode=s("max"), on_missing=s("error"), on_nonfinite=s("skip")
        ),
    )
    for lineage in (
        dict(parent_run_id=s(run_id), resume_from_checkpoint=s("best")),
        dict(resume_from_run_id=s(run_id), resume_from_checkpoint=s("best")),
    ):
        p = NNTrainerParams(**common, **lineage)
        state = p.state()
        yaml.safe_dump(state)  # raised RepresenterError on the first numpy.str_
        assert all(type(name) is str for name in (*p.optims, *p.schedulers, *state["optims"], *state["schedulers"]))
        assert all(type(state[key]) is str for key in ("data_id", "parent_run_id", "parent_checkpoint"))
        assert type(p.resume_from_run_id or p.parent_run_id) is str
        assert (type(state["metrics"][0]["id"]), type(state["metrics"][0]["name"])) == (str, str)
        assert all(
            type(state["monitor"][key]) is str for key in ("metric", "split", "mode", "on_missing", "on_nonfinite")
        )
        plain = {key: str(value) for key, value in lineage.items()}
        assert (
            state
            == NNTrainerParams(
                n_epochs=1,
                optims={"G": _g_optim(), "D": _d_optim()},
                schedulers={"G": _sched()},
                data_id="ds-1",
                metrics=(MetricSpec("accuracy", name="acc"),),
                monitor=MonitorSpec(metric="acc", split="val", mode="max", on_missing="error"),
                **plain,
            ).state()
        )


@pytest.mark.parametrize("lineage", [12345, "", "  ", "run"])
def test_trainer_params_refuse_a_parent_run_id_that_is_not_a_run_id_string(lineage):
    """#515 Example 3 twin: NNTrainerParams accepted an int (recorded as the
    lineage) and an NNRun (failed the first run.yaml write)."""
    from nnx import Activations, Devices, Losses, Nets, NNModelParams, NNParams, NNTrainParams
    from nnx.nn.params.nn_run import NNRun

    if lineage == "run":
        lineage = NNRun(
            net=NNParams(input_dim=4, output_dim=2, hidden_dims=[3], dropout_prob=0.0, activation=Activations.RELU),
            model=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
            train=NNTrainParams(n_epochs=1),
        )
        match = r"NNTrainerParams\.parent_run_id .*got an NNRun; pass its \.id"
    else:
        match = "NNTrainerParams.parent_run_id must be a non-empty run id string"
    with pytest.raises(ValueError, match=match) as refused:
        NNTrainerParams(n_epochs=1, optims={"G": _g_optim()}, parent_run_id=lineage)
    assert ("a copy of them must set parent_run_id" in str(refused.value)) is (lineage in (12345, "  "))


def test_trainer_params_from_state_keeps_a_saved_non_string_lineage():
    """run.yaml written before #515 may record `parent_run_id: 12345`; the
    run must reload with that lineage, so its id does not change."""
    from dataclasses import replace

    from nnx import NNTrainerParamsBuilder

    saved = {**NNTrainerParams(n_epochs=1, optims={"G": _g_optim()}).state()}
    saved.update(parent_run_id=12345, parent_checkpoint="last")
    p = NNTrainerParams.from_state(saved)
    assert p.parent_run_id == 12345 and p.state() == saved
    # A copy is refused until the lineage is a run id string or None.
    with pytest.raises(ValueError, match="a copy of them must set parent_run_id"):
        p.with_val_loader([])
    with pytest.raises(ValueError, match="NNTrainerParams.parent_run_id"):
        NNTrainerParamsBuilder.from_params(p).build()
    assert NNTrainerParamsBuilder.from_params(replace(p, parent_run_id=None)).build().parent_run_id is None


# --- FIX-095 (#563): equal configurations hash equal whatever their mapping order ------------------


def _hashable_optim(max_lr: float) -> NNOptimParams:
    """No `param_groups`: unlike `_g_optim()`, this one is hashable."""
    return NNOptimParams(name=Optims.ADAM, max_lr=max_lr, momentum=(0.9, 0.999), weight_decay=0.0)


def _metric_other(y, y_hat):
    return 0.25


@pytest.mark.parametrize("mapping", ["optims", "schedulers", "extra_metrics"])
def test_trainer_params_hash_ignores_mapping_insertion_order(mapping):
    """Equality ignores the insertion order of `optims`, `schedulers` and
    `extra_metrics`, so the hash must too: two configurations differing only
    in that order are one set member and find each other as dict keys, and
    each keeps its own iteration order."""
    entries = {
        "optims": {"G": _hashable_optim(1e-3), "D": _hashable_optim(1e-2)},
        "schedulers": {
            "G": _sched(),
            "D": NNSchedulerParams(min_lr=1e-6, factor=0.1, patience=5, cooldown=0, threshold=1e-4),
        },
        "extra_metrics": {"acc": _metric_identity, "f1": _metric_other},
    }[mapping]
    common = {"n_epochs": 1, "optims": {"G": _hashable_optim(1e-3), "D": _hashable_optim(1e-2)}}
    p1 = NNTrainerParams(**{**common, mapping: entries})
    p2 = NNTrainerParams(**{**common, mapping: dict(reversed(list(entries.items())))})

    assert p1 == p2
    assert hash(p1) == hash(p2)
    assert len({p1, p2}) == 1
    assert {p1: "run"}[p2] == "run"
    assert list(getattr(p1, mapping)) == list(entries)
    assert list(getattr(p2, mapping)) == list(reversed(list(entries)))


def test_trainer_params_from_state_round_trip_hashes_equal():
    """state() sorts the names, so `from_state` (and `NNRun.load`) rebuilds
    the mappings in sorted order; the reloaded configuration still hashes
    equal to one built with unsorted names."""
    p = (
        NNTrainerParams.builder()
        .n_epochs(1)
        .optimizer("g", _hashable_optim(1e-3))
        .optimizer("d", _hashable_optim(1e-2))
        .scheduler("g", _sched())
        .scheduler("d", _sched())
        .build()
    )
    loaded = NNTrainerParams.from_state(p.state())
    assert list(p.optims) == ["g", "d"] and list(loaded.optims) == ["d", "g"]
    assert loaded == p
    assert hash(loaded) == hash(p)
    assert len({loaded, p}) == 1


def test_frozen_mapping_hash_needs_no_orderable_keys():
    """`plans._frozen_mapping` wraps any mapping, so the order-free hash
    must not sort keys of mixed types."""
    from nnx.trainer.params import _FrozenMapping

    first, second = _FrozenMapping({1: "x", "a": "y"}), _FrozenMapping({"a": "y", 1: "x"})
    assert first == second
    assert hash(first) == hash(second)
    assert list(first) == [1, "a"] and list(second) == ["a", 1]
