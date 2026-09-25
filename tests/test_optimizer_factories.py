"""FEAT-013: registered optimizer factories and the shared build hook.

Covers the `OptimizerFactorySpec(id, version, config)` contract, the
process-local registry, `build_optimizer` (one call per optimizer with the
resolved groups, exact parameter ownership), failure before any run
directory exists, resume validation of the factory identity and topology,
offline metadata reload (NNModel and Trainer runs), and the legacy run ids
of built-in configurations.
"""

from __future__ import annotations

import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNEvaluationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimFactoryParams,
    NNOptimParams,
    NNParamGroupSpec,
    NNParams,
    NNRun,
    NNTrainerParams,
    NNTrainParams,
    OptimizerFactorySpec,
    Optims,
    Trainer,
    TrainerStepContext,
    build_optimizer,
    register_optimizer_factory,
    registered_optimizer_factories,
    unregister_optimizer_factory,
)
from nnx.optimizers import optim_params_from_state
from nnx.trainer.trainer import _representative_train_params

FACTORY_ID = "tests.counting_sgd"


class _CountingSGD:
    """Local SGD factory that records every call it receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    def __call__(self, param_groups, config):
        self.calls.append(([dict(g, params=list(g["params"])) for g in param_groups], dict(config)))
        return torch.optim.SGD(param_groups, momentum=config.get("momentum", 0.0))


@pytest.fixture
def counting_sgd():
    factory = _CountingSGD()
    register_optimizer_factory(FACTORY_ID, 1, factory)
    try:
        yield factory
    finally:
        unregister_optimizer_factory(FACTORY_ID, 1)


def _factory_params(**overrides) -> NNOptimFactoryParams:
    fields = dict(
        factory=OptimizerFactorySpec(id=FACTORY_ID, version=1, config={"momentum": 0.9}),
        max_lr=0.05,
        weight_decay=1e-4,
    )
    fields.update(overrides)
    return NNOptimFactoryParams(**fields)


def _net_params() -> NNParams:
    return NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU)


def _model_params() -> NNModelParams:
    return NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)


def _model() -> NNModel:
    return NNModel(net_params=_net_params(), params=_model_params())


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(16, 4, generator=generator)
    y = torch.randint(0, 2, (16,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def _trainer_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
    model = ctx.model
    for optimizer in ctx.optimizers.values():
        optimizer.zero_grad()
    (x,), y = model.net.unpack_batch(ctx.batch)
    loss = model.loss_fn(model.net(x), y)
    loss.backward()
    for optimizer in ctx.optimizers.values():
        optimizer.step()
    return NNEvaluationDataPoint(f1=0.0, recall=0.0, accuracy=0.0, precision=0.0, loss=float(loss.detach()), error=0.0)


# ---------------------------------------------------------------- spec


def test_spec_is_stable_json_like_data():
    spec = OptimizerFactorySpec(
        id="acme.lion-v2_x", version=3, config={"betas": [0.9, 0.99], "opts": {"b": 1, "a": None}}
    )
    assert spec.state() == {
        "id": "acme.lion-v2_x",
        "version": 3,
        "config": {"betas": [0.9, 0.99], "opts": {"a": None, "b": 1}},
    }
    assert OptimizerFactorySpec.from_state(spec.state()) == spec
    assert hash(OptimizerFactorySpec.from_state(spec.state())) == hash(spec)
    assert str(spec) == "acme.lion-v2_x@v3"
    assert spec.config["betas"] == (0.9, 0.99)  # lists are frozen to tuples
    with pytest.raises(TypeError):
        spec.config["betas"] = (0.5, 0.5)  # type: ignore[index]
    with pytest.raises(AttributeError):
        spec.version = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"id": "", "version": 1}, ValueError),
        ({"id": "has space", "version": 1}, ValueError),
        ({"id": ".hidden", "version": 1}, ValueError),
        ({"id": "ok", "version": 0}, ValueError),
        ({"id": "ok", "version": 1.0}, ValueError),
        ({"id": "ok", "version": True}, ValueError),
        ({"id": "ok", "version": 1, "config": {"lr": float("nan")}}, ValueError),
        ({"id": "ok", "version": 1, "config": {1: "int key"}}, TypeError),
        ({"id": "ok", "version": 1, "config": {"t": torch.zeros(1)}}, TypeError),
        ({"id": "ok", "version": 1, "config": [("not", "a mapping")]}, TypeError),
    ],
)
def test_spec_rejects_invalid_identity_and_config(kwargs, error):
    with pytest.raises(error):
        OptimizerFactorySpec(**kwargs)


def test_spec_equality_is_type_strict_and_hash_consistent():
    """1, 1.0 and True serialize differently (and give different run ids),
    so they are different specs; equal specs hash equally."""
    import numpy as np

    as_int = OptimizerFactorySpec(id="ok", version=1, config={"k": 1})
    as_float = OptimizerFactorySpec(id="ok", version=1, config={"k": 1.0})
    as_bool = OptimizerFactorySpec(id="ok", version=1, config={"k": True})
    assert len({as_int, as_float, as_bool}) == 3 and as_int != as_float != as_bool
    # NumPy scalars are normalized to plain int / float (YAML-portable).
    from_numpy = OptimizerFactorySpec(id="ok", version=1, config={"k": np.int64(1), "beta": np.float32(0.5)})
    assert from_numpy.state()["config"] == {"beta": 0.5, "k": 1}
    assert type(from_numpy.config["k"]) is int and type(from_numpy.config["beta"]) is float
    assert from_numpy == OptimizerFactorySpec(id="ok", version=1, config={"k": 1, "beta": 0.5})
    assert hash(from_numpy) == hash(OptimizerFactorySpec(id="ok", version=1, config={"beta": 0.5, "k": 1}))


def test_anonymous_callable_fails_serialization():
    """A factory is referenced by id/version, never stored: a lambda as the
    factory field or inside the config is rejected before it could reach
    run.yaml."""
    with pytest.raises(TypeError, match="register_optimizer_factory"):
        NNOptimFactoryParams(factory=lambda groups, config: torch.optim.SGD(groups), max_lr=0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        OptimizerFactorySpec(id="ok", version=1, config={"build": lambda groups: None})


def test_factory_params_state_round_trip_and_discriminator():
    params = _factory_params(grad_clip_norm=1.0, param_groups=[NNParamGroupSpec(name_pattern="layers.0.*", lr=0.1)])
    state = params.state()
    assert "factory" in state and "name" not in state and "momentum" not in state
    assert optim_params_from_state(state) == params
    builtin = NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.9, weight_decay=0.0)
    assert optim_params_from_state(builtin.state()) == builtin
    with pytest.raises(ValueError, match="max_lr"):
        _factory_params(max_lr=-1.0)
    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        _factory_params(accumulate_grad_batches=0)


# ------------------------------------------------------------ registry


def test_registry_rejects_duplicates_and_non_callables(counting_sgd):
    assert (FACTORY_ID, 1) in registered_optimizer_factories()
    with pytest.raises(ValueError, match="already registered"):
        register_optimizer_factory(FACTORY_ID, 1, counting_sgd)
    register_optimizer_factory(FACTORY_ID, 1, counting_sgd, replace=True)
    with pytest.raises(TypeError, match="callable"):
        register_optimizer_factory("tests.not_callable", 1, "torch.optim.SGD")  # type: ignore[arg-type]
    assert unregister_optimizer_factory("tests.never_registered", 1) is False


# --------------------------------------------------------------- build


class _GroupsAsGiven(torch.optim.Optimizer):
    """Optimizer that skips torch's own duplicate check (``param_groups``
    set verbatim), so the factory result check itself is exercised."""

    def __init__(self, groups):  # noqa: D107 — deliberately bypasses Optimizer.__init__
        self.param_groups = groups
        self.state = {}
        self.defaults = {}


def test_factory_is_called_once_with_resolved_groups(counting_sgd):
    net = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    net[0].bias.requires_grad_(False)
    params = _factory_params(param_groups=[NNParamGroupSpec(name_pattern="1.*", lr=0.2)])

    optimizer = build_optimizer(net, params)
    assert len(counting_sgd.calls) == 1
    groups, config = counting_sgd.calls[0]
    assert config == {"momentum": 0.9}
    # Every group is resolved: explicit lr / weight_decay, frozen params absent.
    assert [(g["lr"], g["weight_decay"], len(g["params"])) for g in groups] == [(0.2, 1e-4, 2), (0.05, 1e-4, 1)]
    held = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert id(net[0].bias) not in held and held == {id(net[0].weight), id(net[1].weight), id(net[1].bias)}

    # Strict ownership (the Trainer contract): unmatched parameters are not owned.
    build_optimizer(net, params, strict_param_groups=True)
    strict_groups, _ = counting_sgd.calls[1]
    assert [len(g["params"]) for g in strict_groups] == [2]


def test_factory_receives_what_a_builtin_would_without_param_groups(counting_sgd):
    """One ownership rule: with param_groups=None the factory gets every
    parameter in one group — frozen ones included, exactly like a built-in
    optimizer — so both record the same topology."""
    from nnx.nn.nn_model import _optimizer_topology

    net = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    net[0].weight.requires_grad_(False)
    factory_opt = build_optimizer(net, _factory_params())
    builtin_opt = build_optimizer(net, NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.9, weight_decay=1e-4))
    groups, _ = counting_sgd.calls[0]
    assert [(g["lr"], g["weight_decay"], len(g["params"])) for g in groups] == [(0.05, 1e-4, 4)]
    assert _optimizer_topology(factory_opt, net) == _optimizer_topology(builtin_opt, net)


def test_unmatched_parameters_still_fail_under_strict_ownership(counting_sgd):
    """Strict ownership never silently falls back to "all parameters": a
    spec that matches nothing fails for built-ins and factories alike, and
    the factory is never called."""
    net = nn.Linear(3, 2)
    scoped = [NNParamGroupSpec(name_pattern="nothing.*")]
    builtin = NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0, param_groups=scoped)
    for params in (_factory_params(param_groups=scoped), builtin):
        with pytest.raises(ValueError, match="no parameter groups"):
            build_optimizer(net, params, strict_param_groups=True)
    assert counting_sgd.calls == []


@pytest.mark.parametrize(
    ("result", "error", "match"),
    [
        (lambda groups, net: "not an optimizer", TypeError, "expected a torch.optim.Optimizer"),
        (lambda groups, net: torch.optim.SGD([net.weight], lr=0.1), ValueError, r"missing \['bias'\]"),
        (
            lambda groups, net: _GroupsAsGiven([{"params": [net.weight, net.bias]}, {"params": [net.bias]}]),
            ValueError,
            r"duplicated \['bias'\]",
        ),
        (
            lambda groups, net: torch.optim.SGD([net.weight, net.bias, nn.Parameter(torch.zeros(2))], lr=0.1),
            ValueError,
            "foreign",
        ),
    ],
    ids=["wrong-type", "missing", "duplicate", "foreign"],
)
def test_factory_result_must_own_exactly_the_resolved_parameters(result, error, match):
    net = nn.Linear(3, 2)
    register_optimizer_factory("tests.bad_factory", 1, lambda groups, config: result(groups, net))
    try:
        with pytest.raises(error, match=match):
            build_optimizer(net, _factory_params(factory=OptimizerFactorySpec(id="tests.bad_factory", version=1)))
    finally:
        unregister_optimizer_factory("tests.bad_factory", 1)


def test_unknown_id_or_version_fails_before_the_run_importing_nothing(tmp_path, monkeypatch, counting_sgd):
    monkeypatch.chdir(tmp_path)
    unknown_id = _factory_params(factory=OptimizerFactorySpec(id="tests.nowhere.to_be_found", version=1))
    unknown_version = _factory_params(factory=OptimizerFactorySpec(id=FACTORY_ID, version=7))
    for params, match in (
        (unknown_id, "is not registered"),
        (unknown_version, r"no registered version 7 \(registered: v1\)"),
    ):
        before = set(sys.modules)
        with pytest.raises(ValueError, match=match):
            _model().train(NNTrainParams(n_epochs=1, train_loader=_loader(), optim=params))
        trainer_params = NNTrainerParams(n_epochs=1, train_loader=_loader(), optims={"main": params})
        with pytest.raises(ValueError, match=match):
            Trainer(_model()).train(params=trainer_params, trainer_step_fn=_trainer_step)
        assert set(sys.modules) == before, "resolving a factory must not import anything"
    assert not (tmp_path / "runs").exists()
    assert counting_sgd.calls == []


def test_wrong_return_type_fails_before_the_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    register_optimizer_factory("tests.returns_none", 1, lambda groups, config: None)
    try:
        params = _factory_params(factory=OptimizerFactorySpec(id="tests.returns_none", version=1))
        with pytest.raises(TypeError, match="returned NoneType"):
            _model().train(NNTrainParams(n_epochs=1, train_loader=_loader(), optim=params))
        with pytest.raises(TypeError, match="returned NoneType"):
            Trainer(_model()).train(
                params=NNTrainerParams(n_epochs=1, train_loader=_loader(), optims={"main": params}),
                trainer_step_fn=_trainer_step,
            )
    finally:
        unregister_optimizer_factory("tests.returns_none", 1)
    assert not (tmp_path / "runs").exists()


def test_nn_model_and_trainer_share_one_build_hook(tmp_path, monkeypatch, counting_sgd):
    import nnx.optimizers

    monkeypatch.chdir(tmp_path)
    seen: list[tuple[str, bool]] = []
    original = nnx.optimizers.build_optimizer

    def spy(net, params, *, strict_param_groups=False):
        seen.append((type(params).__name__, strict_param_groups))
        return original(net, params, strict_param_groups=strict_param_groups)

    monkeypatch.setattr(nnx.optimizers, "build_optimizer", spy)
    _model().train(NNTrainParams(n_epochs=1, train_loader=_loader(), optim=_factory_params()))
    builtin = NNOptimParams(name=Optims.ADAMW, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.01)
    Trainer(_model()).train(
        params=NNTrainerParams(n_epochs=1, train_loader=_loader(), optims={"main": builtin}),
        trainer_step_fn=_trainer_step,
    )
    assert seen == [("NNOptimFactoryParams", False), ("NNOptimParams", True)]


# -------------------------------------------------------------- resume


def _train_source(tmp_path, monkeypatch, optim) -> NNRun:
    monkeypatch.chdir(tmp_path)
    return _model().train(NNTrainParams(n_epochs=1, train_loader=_loader(), optim=optim))


def _resume(first: NNRun, optim):
    return _model().train(NNTrainParams(n_epochs=1, train_loader=_loader(), resume_from_run_id=first.id, optim=optim))


def test_factory_identity_is_saved_and_resume_accepts_the_same_factory(tmp_path, monkeypatch, counting_sgd):
    from nnx.nn.enum.checkpoints import Checkpoints
    from nnx.nn.params.nn_checkpoint import NNCheckpoint

    params = _factory_params()
    first = _train_source(tmp_path, monkeypatch, params)
    _, state = NNCheckpoint.load_with_training_state(run=first.id, type=Checkpoints.LAST)
    assert state is not None
    assert state["optimizer_factory"] == params.factory.state()
    assert [[entry["name"] for entry in group] for group in state["optimizer_topology"]] == [
        ["layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias"]
    ]
    resumed = _resume(first, params)
    assert resumed.train.optim == params
    assert len(counting_sgd.calls) == 2  # one construction per train() call


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            {"factory": OptimizerFactorySpec(id=FACTORY_ID, version=1, config={"momentum": 0.5})},
            "optimizer factory mismatch",
        ),
        (
            {"factory": OptimizerFactorySpec(id=FACTORY_ID, version=2, config={"momentum": 0.9})},
            "optimizer factory mismatch",
        ),
        (
            {"factory": OptimizerFactorySpec(id=FACTORY_ID, version=1, config={"momentum": 0.9, "x": 1})},
            "optimizer factory mismatch",
        ),
        ({"param_groups": [NNParamGroupSpec(name_pattern="*.bias", weight_decay=0.0)]}, "optimizer parameter topology"),
    ],
    ids=["config", "version", "extra-config-key", "topology"],
)
def test_resume_rejects_a_changed_factory_identity_or_topology(tmp_path, monkeypatch, counting_sgd, change, match):
    register_optimizer_factory(FACTORY_ID, 2, counting_sgd)
    try:
        first = _train_source(tmp_path, monkeypatch, _factory_params())
        with pytest.raises(ValueError, match=match):
            _resume(first, _factory_params(**change))
    finally:
        unregister_optimizer_factory(FACTORY_ID, 2)


def test_resume_identity_is_type_strict(tmp_path, monkeypatch, counting_sgd):
    """A config written as 1 does not resume as 1.0: they are different
    specs in run.yaml, so the checkpoint belongs to a different optimizer."""
    as_int = _factory_params(factory=OptimizerFactorySpec(id=FACTORY_ID, version=1, config={"momentum": 1}))
    first = _train_source(tmp_path, monkeypatch, as_int)
    as_float = _factory_params(factory=OptimizerFactorySpec(id=FACTORY_ID, version=1, config={"momentum": 1.0}))
    with pytest.raises(ValueError, match="optimizer factory mismatch"):
        _resume(first, as_float)


def test_resume_rejects_switching_between_builtin_and_factory(tmp_path, monkeypatch, counting_sgd):
    builtin = NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.9, weight_decay=1e-4)
    first = _train_source(tmp_path, monkeypatch, builtin)
    with pytest.raises(ValueError, match="optimizer factory mismatch"):
        _resume(first, _factory_params())
    second = _model().train(NNTrainParams(n_epochs=1, train_loader=_loader(), optim=_factory_params(), data_id="b"))
    with pytest.raises(ValueError, match="optimizer factory mismatch"):
        _resume(second, builtin)


# ------------------------------------------------------ metadata reload


def test_trainer_run_with_factory_primary_reloads_offline(tmp_path, monkeypatch, counting_sgd):
    """The sorted-first optimizer ("a_body") is the registered variant: the
    representative `train` block and the full `trainer` block both reload
    with the factory unregistered, discriminator intact, without calling
    the factory, and display it by id@version."""
    monkeypatch.chdir(tmp_path)
    optims = {
        "a_body": _factory_params(param_groups=[NNParamGroupSpec(name_pattern="layers.0.*")]),
        "b_head": NNOptimParams(
            name=Optims.ADAMW,
            max_lr=1e-3,
            momentum=(0.9, 0.999),
            weight_decay=0.01,
            param_groups=[NNParamGroupSpec(name_pattern="layers.1.*")],
        ),
    }
    run = Trainer(_model()).train(
        params=NNTrainerParams(n_epochs=1, train_loader=_loader(), optims=optims), trainer_step_fn=_trainer_step
    )
    assert len(counting_sgd.calls) == 1

    unregister_optimizer_factory(FACTORY_ID, 1)
    try:
        loaded = NNRun.load(run.id)
    finally:
        register_optimizer_factory(FACTORY_ID, 1, counting_sgd)
    assert loaded.id == run.id and loaded.state() == run.state()
    assert isinstance(loaded.train.optim, NNOptimFactoryParams) and loaded.train.optim == optims["a_body"]
    assert loaded.trainer is not None and dict(loaded.trainer.optims) == optims
    assert type(loaded.trainer.optims["b_head"]) is NNOptimParams
    text, html = str(loaded), loaded._repr_html_()
    # The factory's own config is shown; no fabricated `momentum=` / `name=` field.
    assert f"optimizer={FACTORY_ID}@v1 {{'momentum': 0.9}}" in text and ", momentum=" not in text
    assert f"registered factory {FACTORY_ID}@v1" in html
    assert len(counting_sgd.calls) == 1  # reload and display ran no factory code


def test_builtin_run_ids_are_unchanged():
    """Legacy ids pinned before FEAT-013: adding ADAMW, `eps` and factory
    decoding must not shift any built-in configuration's run id."""
    adam = NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0)
    run = NNRun(train=NNTrainParams(n_epochs=1, optim=adam), model=_model_params(), net=_net_params())
    assert run.id == "5c741de80116c9858a7e2a56256cb917"
    trainer_params = NNTrainerParams(
        n_epochs=1,
        optims={
            "a": NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
                param_groups=[NNParamGroupSpec(name_pattern="layers.0.*")],
            ),
            "b": NNOptimParams(
                name=Optims.SGD,
                max_lr=1e-2,
                momentum=0.9,
                weight_decay=1e-4,
                param_groups=[NNParamGroupSpec(name_pattern="layers.1.*")],
            ),
        },
    )
    trainer_run = NNRun(
        train=_representative_train_params(trainer_params),
        trainer=trainer_params,
        model=_model_params(),
        net=_net_params(),
    )
    assert trainer_run.id == "25fa561cf7227484a896c5bf32ef5eb2"
