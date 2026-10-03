"""FEAT-012: immutable experiment plans — branching and the pure
``validate()`` boundary (execution is in ``test_plan_execution.py``)."""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import nnx
from nnx import plans
from nnx.monitors import MetricSpec, MonitorSpec
from nnx.nn.callbacks import Callback, EarlyStopping
from nnx.nn.enum.activations import Activations
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_train_params import NNTrainParams
from nnx.plans import Diagnostic, ExperimentPlan, PlanError, PlanValidation

NET = NNParams(input_dim=4, output_dim=3, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU)
MODEL = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
TRAIN = NNTrainParams(n_epochs=2)


def _loader(n: int = 16) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    return DataLoader(TensorDataset(torch.randn(n, 4, generator=generator), torch.arange(n) % 3), batch_size=8)


def _plan(**data: Any) -> ExperimentPlan:
    return ExperimentPlan().with_net(NET).with_model(MODEL).with_train(TRAIN).with_data(_loader(), **data)


@contextlib.contextmanager
def _registered(factory_id: str):
    from nnx.models import register_model_factory, unregister_model_factory

    register_model_factory(factory_id, 1, lambda config: torch.nn.Linear(4, 3))
    try:
        yield
    finally:
        unregister_model_factory(factory_id, 1)


class CountingLoader:
    """A re-iterable of batches that counts every pass over it."""

    def __init__(self) -> None:
        self.passes = 0
        self.batches = [(torch.randn(8, 4), torch.arange(8) % 3)]

    def __iter__(self):
        self.passes += 1
        return iter(self.batches)


# --- branching (AC1) ---------------------------------------------------------------------------


def test_branching_leaves_the_source_and_siblings_unchanged():
    base = _plan()
    before = [getattr(base, f.name) for f in dataclasses.fields(base)]
    short, long = base.with_epochs(1), base.with_epochs(5)
    seeded = base.with_seed(3)
    assert (base.train.n_epochs, short.train.n_epochs, long.train.n_epochs) == (2, 1, 5)
    assert base.seed is None and seeded.seed == 3 and short.seed is None
    assert all(getattr(base, f.name) is value for f, value in zip(dataclasses.fields(base), before, strict=True))
    assert short.net is base.net and short.model is base.model  # immutable parameters are shared, not copied


def test_nested_collections_do_not_leak_between_branches():
    first, second = Callback(), Callback()
    metrics = {"acc": lambda y, p: 1.0}
    base = _plan().with_callbacks(first).with_extra_metrics(metrics)
    sibling = base.with_callbacks(first, second).with_extra_metrics({**metrics, "other": lambda y, p: 0.0})
    metrics["late"] = lambda y, p: 2.0  # the caller edits their dict afterwards
    assert base.callbacks == (first,) and sibling.callbacks == (first, second)
    assert set(base.train.extra_metrics) == {"acc"} and set(sibling.train.extra_metrics) == {"acc", "other"}
    with pytest.raises(TypeError):
        base.train.extra_metrics["x"] = lambda y, p: 0.0  # read-only
    with pytest.raises(dataclasses.FrozenInstanceError):
        base.seed = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        base.net.hidden_dims.append(4)  # type: ignore[union-attr]
    params = NNTrainParams(n_epochs=1, extra_metrics=metrics)
    frozen = ExperimentPlan().with_train(params)
    metrics["later"] = lambda y, p: 3.0
    assert "later" not in frozen.train.extra_metrics  # with_train copies the borrowed mapping too


def test_edits_to_training_parameters_need_a_training_configuration():
    with pytest.raises(PlanError, match="with_train"):
        ExperimentPlan().with_epochs(3)


def test_docstrings_name_the_borrowed_runtime_objects():
    text = " ".join(filter(None, (plans.__doc__, ExperimentPlan.__doc__, ExperimentPlan.with_data.__doc__)))
    for borrowed in ("loader", "callback", "extra_metrics", "step functions", "objective", "components", "borrowed"):
        assert borrowed in text
    assert "factory" in (ExperimentPlan.with_callback_factories.__doc__ or "") + text


# --- validate() (AC2) --------------------------------------------------------------------------


def test_validate_aggregates_field_path_diagnostics():
    bad = (
        ExperimentPlan()
        .with_seed(-1)
        .with_callbacks("not a callback")
        .with_callback_factories(7)
        .with_data(iter([(torch.zeros(1, 4), torch.zeros(1))]), identity=" ")
        .resuming("", checkpoint="nope", mode="sometimes")
    )
    report = bad.validate()
    assert not report.ok
    assert set(report.paths) >= {
        "model",
        "train",
        "data.train",
        "data.identity",
        "seed",
        "callbacks[0]",
        "callback_factories[0]",
        "resume.run_id",
        "resume.checkpoint",
        "resume.mode",
    }
    with pytest.raises(PlanError) as caught:
        report.raise_for_errors()
    assert caught.value.diagnostics == report.diagnostics
    assert all(path in str(caught.value) for path in report.paths)
    assert _plan().validate() == PlanValidation(())


def test_validate_reports_conflicts_statically():
    lineage = _plan().with_train(NNTrainParams(n_epochs=2, parent_run_id="abc")).resuming("def")
    assert "resume" in lineage.validate().paths  # resume and parent lineage cannot both be set
    seeded = _plan().with_train(NNTrainParams(n_epochs=2, seed=1))
    assert seeded.with_seed(2).validate().ok and seeded.with_seed(2)._compile_train(None, None).seed == 2
    assert seeded.with_seed(None)._compile_train(None, None).seed == 1  # the plan's seed overrides; None inherits
    overwrite = _plan().with_train(NNTrainParams(n_epochs=2, overwrite_existing=True))
    assert overwrite.validate().paths == ("train.overwrite_existing",)
    no_val = _plan().with_metrics(monitor=MonitorSpec(metric="loss", split="val"))
    assert no_val.validate().paths == ("train.monitor",)  # a val monitor with no val data
    assert _plan(val=_loader()).with_metrics(monitor=MonitorSpec(metric="loss")).validate().ok
    unknown = _plan().with_metrics([MetricSpec("no-such-metric")])
    assert unknown.validate().paths == ("train.metrics[0]",)
    both = _plan().with_objective(lambda batch: None).with_step_fns(train_step_fn=lambda ctx: None)
    assert both.validate().paths == ("objective",)
    extra = _plan().with_extra_metrics({"bad": 3})
    assert extra.validate().paths == ("train.extra_metrics['bad']",)


def test_validate_checks_the_network_descriptor():
    from nnx.models import ModelSpec

    assert ExperimentPlan().with_model(MODEL).with_train(TRAIN).with_data(_loader()).validate().paths == ("net",)
    spec = NNModelParams(net=ModelSpec("some-factory"), device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    assert set(_plan().with_model(spec).validate().paths) == {"net", "model.net"}  # takes no NNParams; unregistered
    with _registered("some-factory"):
        assert _plan().with_model(spec).validate().paths == ("net",)
        assert _plan().with_model(spec).with_net(None).validate().ok
    runtime = nnx.nn.nn_model.NNModel(params=NNModelParams(loss=Losses.CROSS_ENTROPY), module=torch.nn.Linear(4, 3))
    assert {"model.net"} <= set(_plan().with_model(runtime.params).validate().paths)  # never a wrapped module


def test_validate_consumes_no_loader_calls_no_factory_builds_no_model_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    counts = {"factory": 0, "callback_factory": 0, "models": 0, "loads": 0}
    loader = CountingLoader()

    def factory():
        counts["factory"] += 1
        return CountingLoader()

    def callback_factory():
        counts["callback_factory"] += 1
        return Callback()

    real_init = nnx.nn.nn_model.NNModel.__init__

    def counting_init(self, *args, **kwargs):
        counts["models"] += 1
        real_init(self, *args, **kwargs)

    def no_load(*args, **kwargs):
        counts["loads"] += 1
        raise AssertionError("validate() must not read weights")

    monkeypatch.setattr(nnx.nn.nn_model.NNModel, "__init__", counting_init)
    monkeypatch.setattr(torch, "load", no_load)
    plan = (
        _plan()
        .with_data(loader, val=factory, identity="toy")
        .with_callback_factories(callback_factory)
        .resuming("0123456789abcdef")
        .with_metrics(monitor=MonitorSpec(metric="loss"))
    )
    early = EarlyStopping(patience=2)
    before = dict(vars(early))
    assert plan.with_callbacks(early).validate().ok
    assert counts == {"factory": 0, "callback_factory": 0, "models": 0, "loads": 0}
    assert loader.passes == 0 and dict(vars(early)) == before  # a borrowed callback is not even bound
    assert list(tmp_path.iterdir()) == []


def test_a_diagnostic_is_a_path_and_a_message():
    diagnostic = Diagnostic("data.train", "required")
    assert (diagnostic.path, diagnostic.message) == ("data.train", "required")
    assert nnx.ExperimentPlan is ExperimentPlan and nnx.plans is plans


# --- review regressions -------------------------------------------------------------------------


def test_round_one_malformed_extra_metrics_and_training_edits_are_plan_errors():
    assert _plan().with_extra_metrics([("acc", lambda y, p: 1.0)]).validate().paths == ("train.extra_metrics",)
    with pytest.raises(PlanError, match="train.n_epochs"):
        _plan().with_epochs(0)
    with pytest.raises(PlanError, match="train.monitor"):
        _plan().with_metrics(monitor="val.loss")


def test_round_one_with_metrics_and_step_fns_keep_the_field_not_given():
    spec, monitor = MetricSpec("accuracy", name="acc"), MonitorSpec(metric="acc")
    both = _plan(val=_loader()).with_metrics([spec]).with_metrics(monitor=monitor)
    assert both.train.metrics == (spec,) and both.train.monitor == monitor
    assert both.with_metrics([spec]).train.monitor == monitor
    assert both.with_metrics(monitor=None).train.monitor is None
    step, evaluate = (lambda ctx: None), (lambda ctx: None)
    stepped = _plan().with_step_fns(train_step_fn=step).with_step_fns(eval_step_fn=evaluate)
    assert (stepped.train_step_fn, stepped.eval_step_fn) == (step, evaluate)
    assert stepped.with_step_fns(train_step_fn=None).eval_step_fn is evaluate


def test_round_one_resume_accepts_the_loops_own_checkpoints():
    from nnx.nn.enum.checkpoints import Checkpoints

    assert _plan().resuming("abc", checkpoint="custom_e3").validate().ok  # a ModelCheckpoint stem
    member = _plan().resuming("abc", checkpoint=Checkpoints.BEST)
    assert member.validate().ok and member.resume.checkpoint == "best"
    assert _plan().resuming("abc", checkpoint="LAST").validate().paths == ("resume.checkpoint",)


def test_round_one_callback_monitors_are_checked_without_binding():
    early = EarlyStopping(monitor=MonitorSpec(metric="loss", split="val"), patience=2)
    before = dict(vars(early))
    assert _plan().with_callbacks(early).validate().paths == ("callbacks[0].monitor",)  # no val data
    assert dict(vars(early)) == before
    assert _plan(val=_loader()).with_callbacks(early).validate().ok
    stepped = _plan().with_step_fns(train_step_fn=lambda ctx: None)
    spec = MetricSpec("accuracy", name="acc")
    train_monitor = stepped.with_metrics([spec], monitor=MonitorSpec(metric="acc", split="train"))
    assert train_monitor.validate().paths == ("train.monitor",)  # only the default step records it


def test_round_two_data_edits_and_paths():
    with pytest.raises(PlanError) as caught:
        _plan().with_metrics(None)  # type: ignore[arg-type]
    assert caught.value.diagnostics[0].path == "train.metrics"
    with pytest.raises(PlanError) as caught:
        _plan().with_metrics([MetricSpec("accuracy", name="acc")], monitor="val.acc")
    assert caught.value.diagnostics[0].path == "train.monitor"  # the changed field the error names
    with_val_only = ExperimentPlan().with_net(NET).with_model(MODEL).with_train(TRAIN).with_data(None, val=_loader())
    assert {"data.train"} <= set(with_val_only.validate().paths)


def test_round_two_read_only_mappings_copy_and_pickle():
    import copy
    import pickle

    plan = _plan().with_extra_metrics({"acc": len})
    assert copy.deepcopy(plan.train).extra_metrics == plan.train.extra_metrics
    assert pickle.loads(pickle.dumps(plan.train.extra_metrics)) == plan.train.extra_metrics
    with pytest.raises(TypeError):
        plan.train.extra_metrics["x"] = len  # type: ignore[index]


def test_round_three_the_effective_seed_and_orphan_identity_are_checked():
    inherited = _plan().with_train(NNTrainParams(n_epochs=1, seed=-5))
    assert inherited.validate().paths == ("train.seed",)  # the seed set_seed would receive
    identity_only = ExperimentPlan().with_net(NET).with_model(MODEL).with_train(TRAIN).with_data(None, identity="x")
    assert {"data.train"} <= set(identity_only.validate().paths)


def test_round_three_resume_mode_defaults_to_the_training_parameters():
    stateful = _plan().with_train(NNTrainParams(n_epochs=2, resume_mode="stateful"))
    assert stateful.resuming("abc")._compile_train(None, None).resume_mode == "stateful"
    assert stateful.resuming("abc", mode="weights_only")._compile_train(None, None).resume_mode == "weights_only"


def test_round_four_optimizer_net_and_data_diagnostics():
    from nnx.models import ModelSpec
    from nnx.nn.enum.optims import Optims
    from nnx.nn.params.nn_optim_params import NNOptimParams

    sgd = NNOptimParams(name=Optims.SGD, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0)
    assert not sgd.is_valid()
    assert _plan().with_optim(sgd).validate().paths == ("train.optim",)
    spec = NNModelParams(net=ModelSpec("some-factory"), device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    typed = _plan().with_model(spec).with_net({"input_dim": 4})  # type: ignore[arg-type]
    with _registered("some-factory"):
        assert typed.validate().paths == ("net",)  # one diagnostic, not two contradicting ones
    orphan = ExperimentPlan().with_net(NET).with_model(MODEL).with_train(TRAIN).with_data(None, val=_loader())
    assert orphan.validate().paths.count("data.train") == 1


def test_round_four_callback_monitors_use_the_bind_protocol_without_binding():
    class Bound(Callback):
        def __init__(self) -> None:
            self.bound = 0

        def _bind_metrics(self, metrics):
            self.bound += 1
            return MonitorSpec(metric="loss", split="val").resolve(metrics)

    callback = Bound()
    assert _plan().with_callbacks(callback).validate().paths == ("callbacks[0].monitor",)
    assert callback.bound == 0  # checked on a copy


def test_round_four_resume_mode_uses_the_training_rule():
    report = _plan().resuming("abc", mode="sometimes").validate()
    assert report.paths == ("resume.mode",) and "resume_mode must be one of" in report.diagnostics[0].message


def test_round_five_errors_pickle_and_name_the_right_field():
    import pickle

    error = PlanError((Diagnostic("seed", "bad"),))
    restored = pickle.loads(pickle.dumps(error))
    assert restored.diagnostics == error.diagnostics and str(restored) == str(error)
    assert isinstance(error, ValueError) and isinstance(error, TypeError)
    spec = MetricSpec("accuracy", name="acc")
    monitored = _plan(val=_loader()).with_metrics([spec], monitor=MonitorSpec(metric="acc"))
    with pytest.raises(PlanError) as caught:
        monitored.with_metrics([])  # the kept monitor now names an undeclared metric
    assert caught.value.diagnostics[0].path == "train"
    with pytest.raises(TypeError):
        _plan().with_epochs("3")  # still catchable as the TypeError NNTrainParams raises


def test_round_five_a_callback_class_is_not_a_callback():
    report = _plan().with_callbacks(EarlyStopping).validate()
    assert report.paths == ("callbacks[0]",) and "with_callback_factories" in report.diagnostics[0].message


def test_round_six_metric_inputs_and_provenance_are_checked_without_a_model(monkeypatch):
    import nnx.nn.nn_model as nn_model

    built = []
    monkeypatch.setattr(nn_model.NNModel, "__init__", lambda self, *a, **k: built.append(1))
    binary_f1 = MetricSpec("f1", config={"average": "binary"})
    report = _plan().with_metrics([binary_f1]).validate()  # a 3-class model
    assert report.paths == ("train.metrics",) and "exactly 2 classes" in report.diagnostics[0].message
    assert built == []
    assert _plan().with_provenance({"not": "a manifest"}).validate().paths == ("provenance",)


def test_round_seven_task_factory_and_callback_checks_are_pure():
    from nnx.models import KeywordInputs, ModelSpec
    from nnx.tasks import TaskSpec

    tasked = NNModelParams(
        net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(5)
    )
    assert _plan().with_model(tasked).validate().paths == ("model.task",)  # 5 classes on a 3-output net
    missing = NNModelParams(net=ModelSpec("nope.unknown"), device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    assert _plan().with_net(None).with_model(missing).validate().paths == ("model.net",)
    assert _plan().with_callbacks(lambda: EarlyStopping()).validate().paths == ("callbacks[0]",)  # a factory
    assert _plan().with_callbacks(lambda idps: None).validate().ok  # a legacy fn(idps) callback
    assert _plan().with_batch_adapter("keywords").validate().paths == ("batch_adapter",)
    adapter = KeywordInputs(["x"])
    assert (
        _plan().with_batch_adapter(adapter).batch_adapter is adapter
        and _plan().with_batch_adapter(adapter).validate().ok
    )


def test_round_eight_every_model_problem_is_reported_at_once():
    lossless = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=None)
    assert "model" in _plan().with_model(lossless).validate().paths
    report = ExperimentPlan().with_batch_adapter("bogus").validate()
    assert {"model", "batch_adapter"} <= set(report.paths)  # no early return hides the adapter


def test_round_eight_an_uncopyable_callback_is_left_to_train():
    class Handle(Callback):
        def __copy__(self):
            raise RuntimeError("holds a C handle")

        def _bind_metrics(self, metrics):
            raise AssertionError("never bound by validate()")

    assert _plan().with_callbacks(Handle()).validate().ok


def test_round_nine_a_lossless_model_is_one_diagnostic_not_a_crash():
    import functools

    from nnx.tasks import TaskSpec

    lossless = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=None, task=TaskSpec.categorical(3))
    report = _plan().with_model(lossless).with_metrics([MetricSpec("accuracy")]).validate()
    assert report.paths == ("model",)  # no task or metric diagnostic piled on, no AttributeError
    with pytest.raises(PlanError, match="needs a device and a loss"):
        _plan().with_model(lossless).probe((torch.zeros(1, 4), torch.zeros(1, dtype=torch.long)))
    factory = functools.partial(EarlyStopping, patience=3)
    assert _plan().with_callbacks(factory).validate().paths == ("callbacks[0]",)


def test_round_ten_factories_components_and_the_constructor():
    needs_size = _plan().with_data(lambda batch_size: _loader())
    assert needs_size.validate().paths == ("data.train",)
    assert _plan().with_callback_factories(lambda idps: None).validate().paths == ("callback_factories[0]",)
    assert _plan().with_components(object()).validate().paths == ("components[0]",)
    callbacks, metrics = [Callback()], {"acc": lambda y, p: 1.0}
    direct = ExperimentPlan(
        net=NET,
        model=MODEL,
        train=NNTrainParams(n_epochs=1, extra_metrics=metrics),
        train_data=_loader(),
        callbacks=callbacks,
    )
    callbacks.append(Callback())
    metrics["late"] = lambda y, p: 0.0
    assert len(direct.callbacks) == 1 and isinstance(direct.callbacks, tuple)
    assert set(direct.train.extra_metrics) == {"acc"}  # the constructor freezes too


def test_round_eleven_the_constructor_freezes_any_iterable():
    callbacks = [Callback(), Callback()]
    plan = ExperimentPlan(net=NET, model=MODEL, train=TRAIN, train_data=_loader(), callbacks=(c for c in callbacks))
    assert plan.callbacks == tuple(callbacks) and plan.validate().ok and plan.validate().ok  # read once, kept
    single = ExperimentPlan(net=NET, model=MODEL, train=TRAIN, train_data=_loader(), callbacks=Callback())
    assert single.validate().paths == ("callbacks",)


def test_round_twelve_every_collection_problem_is_reported():
    report = ExperimentPlan(
        net=NET, model=MODEL, train=TRAIN, train_data=_loader(), callbacks=(1,), components=5
    ).validate()
    assert set(report.paths) == {"components", "callbacks[0]"}


def test_round_twelve_a_failing_loss_constructor_is_a_diagnostic(monkeypatch):
    from nnx.nn.enum.losses import Losses as LossEnum

    def explode(self, *args, **kwargs):
        raise RuntimeError("cannot build the loss")

    monkeypatch.setattr(LossEnum, "__call__", explode)
    report = _plan().with_metrics([MetricSpec("accuracy")]).validate()
    assert report.paths == ("model.loss",) and "cannot build the loss" in report.diagnostics[0].message
    # NNModel builds the loss whatever the plan declares, so validate() reports it without metrics too
    assert _plan().with_metrics([]).validate().paths == ("model.loss",)


def test_round_sixteen_a_sequence_protocol_source_is_accepted():
    class Batches:  # iterable through __getitem__ alone, as the training loop accepts
        def __init__(self, batches):
            self._batches = batches

        def __len__(self):
            return len(self._batches)

        def __getitem__(self, index):
            return self._batches[index]

    source = Batches([(torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))])
    assert _plan().with_data(source, val=None).validate().ok


def test_round_thirteen_copied_and_pickled_plans_keep_their_markers():
    import copy
    import pickle

    base = (
        ExperimentPlan()
        .with_net(NET)
        .with_model(MODEL)
        .with_train(TRAIN)
        .with_data([(torch.zeros(2, 4), torch.zeros(2))])
    )
    for clone in (copy.deepcopy(base), copy.copy(base)):
        assert clone._sources(clone.train)[1] is None and clone._identity(clone.train) is None
        assert clone.validate().ok
    restored = pickle.loads(pickle.dumps(ExperimentPlan()))
    assert restored.val_data is plans._INHERIT and restored.data_identity is plans._INHERIT


def test_round_thirteen_with_data_changes_only_what_it_is_given():
    loader, val = _loader(), _loader(8)
    plan = _plan().with_data(loader).with_data(val=val)
    assert plan.train_data is loader and plan._sources(plan.train)[1] is val
    assert plan.with_data(identity="x").train_data is loader


def test_round_thirteen_a_broken_metric_is_reported_once():
    spec = MetricSpec("no-such-metric", name="bad")
    plan = _plan(val=_loader()).with_train(
        NNTrainParams(n_epochs=1, metrics=(spec,), monitor=MonitorSpec(metric="bad"))
    )
    assert plan.with_callbacks(EarlyStopping(monitor=MonitorSpec(metric="bad"))).validate().paths == (
        "train.metrics[0]",
    )


def test_round_seventeen_samples_and_mappings_are_not_batches():
    samples = TensorDataset(torch.zeros(4, 4), torch.zeros(4, dtype=torch.long))
    report = _plan().with_data(samples, val={"x": torch.zeros(2, 4)}).validate()
    messages = dict((d.path, d.message) for d in report.diagnostics)
    assert "wrap it in a DataLoader" in messages["data.train"] and "iterates its keys" in messages["data.val"]
    assert _plan().with_data(DataLoader(samples, batch_size=2), val=None).validate().ok


def test_round_eighteen_a_callable_with_optional_parameters_is_a_factory():
    def make_early_stopping(patience=3):
        return EarlyStopping(patience=patience)

    for factory in (make_early_stopping, lambda *args: EarlyStopping()):
        report = _plan().with_callbacks(factory).validate()
        assert report.paths == ("callbacks[0]",) and "with_callback_factories" in report.diagnostics[0].message
    assert _plan().with_callback_factories(make_early_stopping).validate().ok
    assert _plan().with_callbacks(lambda idps, *extra: None).validate().ok  # the history is required


def test_round_nineteen_wrong_optimizer_scheduler_and_row_sources_are_diagnostics():
    report = _plan().with_optim("adam").with_scheduler("plateau").validate()
    assert set(report.paths) == {"train.optim", "train.scheduler"}  # reported, not an AttributeError
    for rows in (torch.zeros(4, 4), np.zeros((4, 4))):
        report = _plan().with_data(rows, val=None).validate()
        assert report.paths == ("data.train",) and "single rows" in report.diagnostics[0].message


def test_round_twenty_optimizer_factories_and_monitors_beside_a_broken_metric():
    from nnx.optimizers import (
        NNOptimFactoryParams,
        OptimizerFactorySpec,
        register_optimizer_factory,
        unregister_optimizer_factory,
    )

    register_optimizer_factory("plan-sgd", 1, lambda params, config: torch.optim.SGD(params, lr=0.05))
    try:
        optim = NNOptimFactoryParams(factory=OptimizerFactorySpec("plan-sgd", 1), max_lr=0.05)
        assert _plan().with_optim(optim).validate().ok  # a registered optimizer factory, as train() accepts
    finally:
        unregister_optimizer_factory("plan-sgd", 1)
    broken = MetricSpec("no-such-metric")
    plan = _plan().with_metrics([broken], monitor=MonitorSpec(metric="loss", split="val"))  # no val data
    assert plan.validate().paths == ("train.metrics[0]", "train.monitor")  # every problem, at once

    class Fragile(EarlyStopping):
        def _bind_metrics(self, metrics):
            raise RuntimeError("a copy cannot bind")

    assert _plan().with_callbacks(Fragile()).validate().ok  # train() binds the callback itself


def test_round_twenty_one_a_device_or_loss_of_the_wrong_type_is_a_diagnostic():
    from nnx.tasks import TaskSpec

    for model in (
        dataclasses.replace(MODEL, loss=torch.nn.MSELoss, task=TaskSpec.regression()),
        dataclasses.replace(MODEL, device="cpu"),
    ):
        plan = _plan().with_model(model)
        assert plan.validate().paths == ("model",)  # reported, not an AttributeError / TypeError
        with pytest.raises(PlanError, match="a Devices and a Losses member"):
            plan.probe((torch.zeros(2, 4), torch.zeros(2, dtype=torch.long)))


def test_round_twenty_two_split_metrics_are_hashable_values():
    from nnx.plans import SplitMetrics

    first = SplitMetrics("val", available=True, epoch=1, source="epoch", values=plans._frozen_mapping({"loss": 0.5}))
    second = SplitMetrics("val", available=True, epoch=1, source="epoch", values=plans._frozen_mapping({"loss": 0.5}))
    assert first == second and hash(first) == hash(second) and len({first, second}) == 1


def test_round_twenty_four_a_list_of_callbacks_is_told_to_spread():
    report = _plan().with_callbacks([EarlyStopping(), EarlyStopping()]).validate()
    assert report.paths == ("callbacks[0]",) and "separate arguments" in report.diagnostics[0].message
    assert "with_callback_factories" not in report.diagnostics[0].message


def test_feat020_a_streaming_eval_step_reports_what_it_cannot_bound():
    from nnx.streaming import streaming_eval_step

    class _Scores:  # no merge(): not provably bounded
        def update(self, target, prediction):
            pass

        def result(self):
            return 0.5

    plans_metrics = (MetricSpec("accuracy"), MetricSpec("tests.plan-rank"))
    nnx.register_metric("tests.plan-rank", 1, lambda config: _Scores(), input="probabilities", mode="max")
    try:
        plan = (
            _plan(val=_loader())
            .with_step_fns(eval_step_fn=streaming_eval_step)
            .with_metrics(plans_metrics)
            .with_extra_metrics({"n": lambda y, y_hat: 0.0})
        )
        assert set(plan.validate().paths) == {"train.extra_metrics", "train.metrics[1]"}
        assert plan.with_step_fns(eval_step_fn=None).validate().ok  # the default step computes both
    finally:
        nnx.unregister_metric("tests.plan-rank", 1)


def test_feat020_review_validate_checks_metric_inputs_for_the_streaming_step():
    from nnx.streaming import streaming_eval_step

    regression = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR)
    plan = (
        _plan(val=_loader())
        .with_model(regression)
        .with_step_fns(train_step_fn=lambda ctx: None, eval_step_fn=streaming_eval_step)
        .with_metrics([MetricSpec("nll")])  # probabilities a continuous model cannot provide
    )
    report = plan.validate()
    assert "train.metrics" in report.paths  # as train()'s preflight would refuse it
    assert any("streaming_eval_step()" in d.message for d in report.diagnostics)  # named as train() names it


def test_feat020_review_round_four_plan_checks_follow_train():
    from nnx.streaming import streaming_eval_step

    regression = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR)
    no_val = (
        _plan()
        .with_model(regression)
        .with_step_fns(train_step_fn=lambda ctx: None, eval_step_fn=streaming_eval_step)
        .with_metrics([MetricSpec("nll")])
    )
    assert "train.metrics" not in no_val.validate().paths  # no validation data: train() never checks it either

    class _Broken(Exception):
        pass

    def factory(config):
        raise _Broken("optional dependency missing")

    nnx.register_metric("tests.broken-factory", 1, factory, input="labels", mode="max")
    try:
        plan = _plan(val=_loader()).with_step_fns(eval_step_fn=streaming_eval_step)
        report = plan.with_metrics([MetricSpec("tests.broken-factory")]).validate()  # a diagnostic, not a crash
        assert report.paths == ("train.metrics[0]",) and "optional dependency missing" in report.diagnostics[0].message
    finally:
        nnx.unregister_metric("tests.broken-factory", 1)


def test_feat020_review_round_eight_only_the_streaming_step_itself_is_checked():
    from unittest import mock

    step = mock.Mock()  # exposes every attribute, _problems included
    plan = _plan(val=_loader()).with_step_fns(eval_step_fn=step).with_extra_metrics({"n": lambda y, y_hat: 0.0})
    assert plan.validate().ok  # a custom step computes what it declares; nothing is sniffed from it


def test_precision_round_five_plans_check_the_policy_before_anything_runs():
    from nnx.paradigms.augmentation import mixup_train_step_factory
    from nnx.precision import PrecisionPolicy

    fp16_on_cpu = _plan().with_model(dataclasses.replace(MODEL, precision=PrecisionPolicy("fp16")))
    report = fp16_on_cpu.validate()
    assert report.paths == ("model.precision",) and "fp16" in report.diagnostics[0].message
    bf16 = _plan().with_model(dataclasses.replace(MODEL, precision=PrecisionPolicy("bf16")))
    assert bf16.validate().ok
    mixup = bf16.with_step_fns(train_step_fn=mixup_train_step_factory())
    report = mixup.validate()
    assert report.paths == ("train_step_fn",) and "full precision only" in report.diagnostics[0].message
    assert _plan().with_step_fns(train_step_fn=mixup_train_step_factory()).validate().ok  # fp32 takes it
