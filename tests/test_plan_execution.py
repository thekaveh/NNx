"""FEAT-012: experiment plan execution — ``probe()`` and ``fit()`` against
the imperative ``NNModel`` loop they compile to."""

from __future__ import annotations

import hashlib
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx.monitors import MonitorSpec
from nnx.nn.callbacks import Callback, EarlyStopping
from nnx.nn.enum.activations import Activations
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.enum.optims import Optims
from nnx.nn.nn_model import NNModel
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_run import NNRun
from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
from nnx.nn.params.nn_train_params import NNTrainParams
from nnx.plans import ATTEMPT_SALT_PREFIX, ExperimentPlan, FitResult, PlanError
from nnx.seeding import set_seed

NET = NNParams(input_dim=4, output_dim=3, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU)
MODEL = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
TRAIN = NNTrainParams(
    n_epochs=2,
    optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
    scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=0, threshold=1e-3),
)
DATA = torch.Generator().manual_seed(11)
X_TRAIN, Y_TRAIN = torch.randn(24, 4, generator=DATA), torch.arange(24) % 3
X_VAL, Y_VAL = torch.randn(12, 4, generator=DATA), torch.arange(12) % 3


def _train_loader() -> DataLoader:
    return DataLoader(TensorDataset(X_TRAIN, Y_TRAIN), batch_size=8, shuffle=True)


def _val_loader() -> DataLoader:
    return DataLoader(TensorDataset(X_VAL, Y_VAL), batch_size=6)


def _accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _plan(*, val: bool = True) -> ExperimentPlan:
    return (
        ExperimentPlan()
        .with_net(NET)
        .with_model(MODEL)
        .with_train(TRAIN)
        .with_data(_train_loader, val=_val_loader if val else None, identity="toy")
        .with_extra_metrics({"acc": _accuracy})
        .with_seed(7)
    )


class Recorder(Callback):
    """Records every hook with its epoch, in order."""

    def __init__(self, log: list[tuple[str, Any]], tag: str) -> None:
        self.log, self.tag = log, tag

    def on_train_begin(self, ctx) -> None:
        self.log.append((self.tag, "train_begin", None))

    def on_epoch_begin(self, ctx) -> None:
        self.log.append((self.tag, "epoch_begin", ctx.epoch))

    def on_epoch_end(self, ctx) -> None:
        self.log.append((self.tag, "epoch_end", ctx.epoch))

    def on_train_end(self, ctx) -> None:
        self.log.append((self.tag, "train_end", None))


def _imperative(log: list, *, val: bool = True) -> tuple[NNModel, NNRun]:
    """The plan's imperative twin, step by step."""
    set_seed(7)
    train_loader, val_loader = _train_loader(), (_val_loader() if val else None)
    params = NNTrainParams(
        n_epochs=TRAIN.n_epochs,
        optim=TRAIN.optim,
        scheduler=TRAIN.scheduler,
        seed=7,
        data_id="toy",
        extra_metrics={"acc": _accuracy},
        train_loader=train_loader,
        val_loader=val_loader,
    )
    model = NNModel(net_params=NET, params=MODEL)
    run = model.train(params, callbacks=[Recorder(log, "a"), Recorder(log, "b")])
    return model, run


def _comparable(run: NNRun) -> dict:
    state = run.state()
    state.pop("id")
    state.pop("salt", None)
    return state


def _digest(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


# --- fit() (AC4 / AC6 / AC10) -------------------------------------------------------------------


def test_fit_matches_an_equally_seeded_imperative_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan_log: list = []
    result = _plan().with_callbacks(Recorder(plan_log, "a"), Recorder(plan_log, "b")).fit()
    twin_log: list = []
    model, run = _imperative(twin_log)
    assert isinstance(result, FitResult) and isinstance(result.run, NNRun)  # train() still returns an NNRun
    for (name, weight), (twin_name, twin_weight) in zip(
        result.model.net.state_dict().items(), model.net.state_dict().items(), strict=True
    ):
        assert name == twin_name and torch.equal(weight, twin_weight)
    assert [idp.state() for idp in result.run.idps] == [idp.state() for idp in run.idps]
    assert _comparable(result.run) == _comparable(run)  # the same NNRun snapshot, bar the attempt salt
    assert plan_log == twin_log  # the same loop and callback order (on_train_end in reverse)
    assert plan_log[-2:] == [("b", "train_end", None), ("a", "train_end", None)]
    assert result.run.id != run.id and result.run.salt == ATTEMPT_SALT_PREFIX + result.attempt_id


def test_a_train_only_plan_invents_no_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _plan(val=False).fit()
    _, run = _imperative([], val=False)
    assert all(idp.val_edp is None for idp in result.run.idps)
    assert [idp.state() for idp in result.run.idps] == [idp.state() for idp in run.idps]
    val = result.metrics["val"]
    assert not val.available and val.reason == "the plan has no validation data" and not val.values


def test_repeated_fits_are_distinct_attempts_and_never_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = _plan()
    first, second = plan.fit(), plan.fit()
    assert first.attempt_id != second.attempt_id and first.run.id != second.run.id
    assert (tmp_path / "runs" / first.run.id).is_dir() and (tmp_path / "runs" / second.run.id).is_dir()
    assert first.run.train.overwrite_existing is False and second.run.train.overwrite_existing is False
    named = plan.fit(attempt="nightly-1")
    assert named.attempt_id == "nightly-1" and named.run.salt == "plan-attempt:nightly-1"
    with pytest.raises(FileExistsError):
        plan.fit(attempt="nightly-1")  # the same attempt of the same configuration is never overwritten
    with pytest.raises(PlanError, match="attempt"):
        plan.fit(attempt=" ")


def test_an_explicit_resume_keeps_parent_lineage_and_leaves_the_parent_intact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parent = _plan().fit()
    parent_dir = tmp_path / "runs" / parent.run.id
    before = _digest(parent_dir)
    child = _plan().with_epochs(4).resuming(parent.run.id).fit()
    assert child.run.id != parent.run.id
    assert child.run.state()["train"]["parent_run_id"] == parent.run.id
    assert child.run.state()["train"]["parent_checkpoint"] == "last"
    assert child.run.train.overwrite_existing is False
    assert child.run.resume_status is not None and child.run.idps[0].epoch_idx == 2  # warm-restarted
    assert _digest(parent_dir) == before  # the parent's artifacts are untouched


def test_metrics_name_their_split_and_availability_without_a_second_loader_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    passes = {"train": 0, "val": 0}

    class Counted:
        def __init__(self, split: str, loader: DataLoader) -> None:
            self.split, self.loader = split, loader

        def __iter__(self):
            passes[self.split] += 1
            return iter(self.loader)

        def __len__(self) -> int:
            return len(self.loader)

    plan = _plan().with_data(Counted("train", _train_loader()), val=Counted("val", _val_loader()))
    result = plan.fit()
    fitted = dict(passes)
    last = result.run.idps[-1]
    train, val = result.metrics["train"], result.metrics["val"]
    assert (train.split, train.available, train.epoch, train.source) == ("train", True, 1, "last_batch")
    assert (val.split, val.available, val.epoch, val.source) == ("val", True, 1, "epoch")
    assert val.values["loss"] == last.val_edp.loss and val.values["extra/acc"] == last.val_edp.extra["acc"]
    assert train.values["loss"] == last.train_edp.loss
    assert passes == fitted  # reading the metrics iterated no loader
    assert passes["val"] == TRAIN.n_epochs  # one validation pass per epoch, as the loop does


def test_factories_run_once_per_fit_and_never_during_validate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = {"train": 0, "val": 0, "callback": 0}

    def train_factory():
        calls["train"] += 1
        return _train_loader()

    def val_factory():
        calls["val"] += 1
        return _val_loader()

    made: list[EarlyStopping] = []

    def callback_factory():
        calls["callback"] += 1
        made.append(EarlyStopping(patience=5))
        return made[-1]

    plan = _plan().with_data(train_factory, val=val_factory).with_callback_factories(callback_factory)
    assert plan.validate().ok and calls == {"train": 0, "val": 0, "callback": 0}
    plan.fit()
    assert calls == {"train": 1, "val": 1, "callback": 1}
    plan.fit()
    assert calls == {"train": 2, "val": 2, "callback": 2} and made[0] is not made[1]  # fresh per fit


def test_an_invalid_plan_fails_before_anything_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    plan = _plan().with_data(lambda: calls.append("factory")).with_seed(-5)
    with pytest.raises(PlanError, match="seed"):
        plan.fit()
    assert calls == [] and not (tmp_path / "runs").exists()


# --- probe() (AC3 / AC8) ------------------------------------------------------------------------


def _ambient() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state()[1].tolist(),
        "torch": torch.get_rng_state().tolist(),
        "cudnn": (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "env": (os.environ.get("PYTHONHASHSEED"), os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
    }


def test_probe_seeds_builds_a_temporary_model_and_restores_the_ambient_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONHASHSEED", "ambient")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    random.seed(123)
    before = _ambient()
    grad_enabled = torch.is_grad_enabled()
    batch = (X_TRAIN[:5], Y_TRAIN[:5])
    probe = _plan().probe(batch)
    assert _ambient() == before and torch.is_grad_enabled() == grad_enabled
    assert probe.output_shape == (5, 3) and probe.output_dtype == "float32" and probe.target_shape == (5,)
    assert probe.loss is not None and probe.loss > 0
    assert probe.n_parameters == probe.n_trainable == 4 * 8 + 8 + 8 * 3 + 3
    assert _plan().probe(batch) == probe  # seeded: the same temporary model every time
    inputs_only = _plan().probe((X_TRAIN[:2],))
    assert inputs_only.output_shape == (2, 3) and inputs_only.loss is None and inputs_only.target_shape is None
    assert list(tmp_path.iterdir()) == []  # no run directory


def test_probe_runs_under_no_grad():
    seen: list[bool] = []

    class Spy(torch.nn.Module):
        def forward(self, module, inputs, output):  # a forward hook
            seen.append(torch.is_grad_enabled())

    real_init = NNModel.__init__

    def hooked(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self.net.register_forward_hook(Spy())

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(NNModel, "__init__", hooked)
        _plan().probe((X_TRAIN[:3], Y_TRAIN[:3]))
    assert seen and not any(seen)  # every forward pass (the raw output and the loss) ran without grad


def test_a_failed_probe_changes_no_callback_rng_backend_or_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONHASHSEED", "ambient")
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    early = EarlyStopping(patience=3)
    early._wait = 2  # borrowed state from an earlier fit
    log: list = []
    plan = _plan().with_callbacks(early, Recorder(log, "a"))
    wrong = (torch.randn(2, 7), torch.zeros(2, dtype=torch.long))  # 7 features for a 4-input net
    before, callback_state = _ambient(), dict(vars(early))
    with pytest.raises(RuntimeError):
        plan.probe(wrong)
    assert _ambient() == before
    assert dict(vars(early)) == callback_state and log == []
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(PlanError, match="net"):
        ExperimentPlan().with_model(MODEL).probe((X_TRAIN[:1], Y_TRAIN[:1]))  # nothing to build
    assert _ambient() == before


# --- review regressions -------------------------------------------------------------------------


def test_round_one_factories_inside_training_params_run_like_plan_factories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def make():
        calls.append("train")
        return _train_loader()

    params = NNTrainParams(n_epochs=1, optim=TRAIN.optim, train_loader=make)
    result = ExperimentPlan().with_net(NET).with_model(MODEL).with_train(params).fit()
    assert calls == ["train"] and len(result.run.idps) == 3


def test_round_one_a_factory_returning_a_one_shot_iterator_fails_before_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def batches():
        yield from _train_loader()

    with pytest.raises(PlanError, match="data.train"):
        _plan().with_data(batches).fit()  # a generator function: one pass only
    with pytest.raises(PlanError, match="callback_factories"):
        _plan().with_callback_factories(lambda: 3).fit()
    assert not (tmp_path / "runs").exists()


def test_round_one_a_diverged_loss_is_reported_not_hidden():
    from types import SimpleNamespace

    from nnx.plans import _values

    record = SimpleNamespace(loss=float("nan"), error=0.5, accuracy=None, metrics={}, extra={"acc": float("inf")})
    values = _values(record)
    assert np.isnan(values["loss"]) and values["error"] == 0.5 and values["extra/acc"] == float("inf")
    assert "accuracy" not in values  # not recorded


def test_round_two_with_data_carries_its_own_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = NNTrainParams(n_epochs=1, optim=TRAIN.optim, data_id="mnist", train_loader=_train_loader())
    base = ExperimentPlan().with_net(NET).with_model(MODEL).with_train(params)
    assert base._compile_train(None, None).data_id == "mnist"  # the training parameters' own loaders
    assert base.with_data(_train_loader)._compile_train(None, None).data_id == "mnist"  # kept unless given
    assert base.with_data(_train_loader, identity="cifar")._compile_train(None, None).data_id == "cifar"
    assert base.with_data(_train_loader, identity=None)._compile_train(None, None).data_id is None
    with_val = base.with_train(replace(params, val_loader=_val_loader()))
    kept = with_val.with_data(_train_loader)
    assert kept._sources(kept.train)[1] is with_val.train.val_loader  # the val loader is kept
    dropped = with_val.with_data(_train_loader, val=None)
    assert dropped._sources(dropped.train)[1] is None  # train-only, explicitly
    # The order of the calls does not matter: the training parameters' loader and id are resolved late.
    late = ExperimentPlan().with_net(NET).with_model(MODEL).with_data(_train_loader).with_train(with_val.train)
    assert late._sources(late.train)[1] is with_val.train.val_loader
    assert late._compile_train(None, None).data_id == "mnist"


def test_round_two_probe_validates_the_seed_first():
    before = torch.get_rng_state()
    for seed in (1.9, True, -1, 2**40, "x"):
        with pytest.raises(PlanError, match="seed"):
            _plan().with_seed(seed).probe((X_TRAIN[:2], Y_TRAIN[:2]))
    assert torch.equal(torch.get_rng_state(), before)


def test_round_two_an_extra_metric_never_shadows_a_recorded_field(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _plan().with_extra_metrics({"loss": lambda y, p: 123.0}).fit()
    values = result.metrics["val"].values
    assert values["extra/loss"] == 123.0 and values["loss"] == result.run.idps[-1].val_edp.loss != 123.0


def test_round_three_a_val_factory_must_return_a_loader(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PlanError, match="data.val"):
        _plan().with_data(_train_loader, val=lambda: None).fit()
    assert not (tmp_path / "runs").exists()


def test_round_three_probe_runs_the_task_preflight(monkeypatch):
    calls = []
    real = NNModel._check_task_preflight

    def spy(self):
        calls.append(True)
        return real(self)

    monkeypatch.setattr(NNModel, "_check_task_preflight", spy)
    _plan().probe((X_TRAIN[:2], Y_TRAIN[:2]))
    assert calls == [True]


def test_round_four_a_bad_attempt_is_listed_with_the_other_problems(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PlanError) as caught:
        _plan().with_seed(-1).fit(attempt="")
    assert {d.path for d in caught.value.diagnostics} == {"seed", "attempt"}


def test_round_six_probe_reports_no_loss_for_a_custom_step():
    batch = (X_TRAIN[:3], Y_TRAIN[:3])
    assert _plan().probe(batch).loss is not None
    custom = _plan().with_step_fns(train_step_fn=lambda ctx: None).probe(batch)
    assert custom.loss is None and custom.output_shape == (3, 3) and custom.target_shape == (3,)
    assert _plan().with_objective(lambda ctx: None).probe(batch).loss is None


def test_round_six_zero_dimensional_tensor_metrics_are_kept():
    from types import SimpleNamespace

    from nnx.plans import _values

    record = SimpleNamespace(loss=torch.tensor(0.25), error=np.float64(0.5), metrics={"m": np.array(1.5)}, extra={})
    assert dict(_values(record)) == {"loss": 0.25, "error": 0.5, "m": 1.5}


def test_round_seven_a_seed_scope_restores_settings_even_if_the_rng_restore_fails(monkeypatch):
    import nnx.seeding as seeding
    from nnx.seeding import _seed_scope

    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setenv("PYTHONHASHSEED", "ambient")

    def broken(state, loader=None):
        raise RuntimeError("cannot restore")

    monkeypatch.setattr(seeding, "_restore_rng_state", broken)
    with pytest.raises(RuntimeError, match="cannot restore"), _seed_scope():
        set_seed(3)
    assert torch.backends.cudnn.benchmark is True and os.environ["PYTHONHASHSEED"] == "ambient"


def test_round_seven_a_non_array_target_has_no_shape():
    from nnx.models import PositionalInputs

    class Labels(PositionalInputs):
        def split(self, batch):
            args, kwargs, target = super().split(batch)
            return args, kwargs, [int(y) for y in target]  # a list of labels

    result = (
        _plan()
        .with_step_fns(train_step_fn=lambda ctx: None)
        .with_batch_adapter(Labels())
        .probe((X_TRAIN[:2], Y_TRAIN[:2]))
    )
    assert result.target_shape is None and result.output_shape == (2, 3)


def test_round_eight_the_probe_loss_is_the_first_training_steps_loss():
    from nnx.nn.nn_model import _step_loss_terms

    dropout = NNParams(input_dim=4, output_dim=3, hidden_dims=[8], dropout_prob=0.5, activation=Activations.RELU)
    batch = (X_TRAIN[:8], Y_TRAIN[:8])
    probe = _plan().with_net(dropout).probe(batch)
    set_seed(7)
    model = NNModel(net_params=dropout, params=MODEL)
    model.net.train()  # dropout active, as in the default training step
    with torch.no_grad():
        expected = float(_step_loss_terms(model, batch, None, 1).train_loss)
    assert probe.loss == pytest.approx(expected)


def test_round_ten_an_empty_final_record_is_not_available():
    from types import SimpleNamespace

    from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
    from nnx.plans import _fit_metrics

    empty = NNEvaluationDataPoint()
    idp = SimpleNamespace(
        epoch_idx=0, train_summary=None, train_edp=empty, val_edp=empty, monitored_train_edp=lambda: empty
    )
    metrics = _fit_metrics(SimpleNamespace(idps=[idp]), has_val=True)
    assert not metrics["train"].available and "no values" in metrics["train"].reason
    assert not metrics["val"].available and "no values" in metrics["val"].reason


def test_round_eleven_a_failing_restore_never_hides_the_blocks_error(monkeypatch):
    import nnx.seeding as seeding
    from nnx.seeding import _seed_scope

    def broken(state, loader=None):
        raise RuntimeError("cannot restore")

    monkeypatch.setattr(seeding, "_restore_rng_state", broken)
    with pytest.raises(ValueError, match="the probe's own error") as caught, _seed_scope():
        raise ValueError("the probe's own error")
    if sys.version_info >= (3, 11):  # the failed restore is noted on the block's error
        assert any("cannot restore" in note for note in getattr(caught.value, "__notes__", ()))


def test_round_thirteen_a_cpu_probe_on_a_gpu_host_queues_no_cuda_seed(monkeypatch):
    # CPU simulation of a GPU host whose CUDA context is not initialized yet.
    seeded: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: seeded.append(seed))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("an idle CUDA context is never read"))
    _plan().probe((X_TRAIN[:2], Y_TRAIN[:2]))
    assert seeded == []  # nothing queued for the first CUDA use after the probe
    set_seed(1)
    assert seeded and set(seeded) == {1}  # set_seed itself still seeds CUDA by default


def test_round_sixteen_cuda_started_by_the_build_is_restored(monkeypatch):
    # CPU simulation of a GPU host whose idle CUDA context the model build
    # starts (as nnx.models.build_module does when it reads the streams).
    started = {"cuda": False}
    restored: list[Any] = []
    build = ExperimentPlan._build_model

    def build_and_start(plan):
        started["cuda"] = True
        return build(plan)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: started["cuda"])
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: ["streams after the build"])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)
    monkeypatch.setattr(ExperimentPlan, "_build_model", build_and_start)
    _plan().probe((X_TRAIN[:2], Y_TRAIN[:2]))
    assert restored == [["streams after the build"]]

    started["cuda"] = False
    restored.clear()
    with pytest.raises(RuntimeError, match="shape|size"):
        _plan().probe((torch.zeros(2, 99), Y_TRAIN[:2]))  # a failing forward pass restores them too
    assert restored == [["streams after the build"]]


def test_round_seventeen_a_cuda_probe_starts_cuda_first_and_restores_it(monkeypatch):
    # CPU simulation of a GPU host with an idle CUDA context: a CUDA model's
    # probe starts CUDA before capturing, so even a failed build restores it.
    started = {"cuda": False}
    restored: list[Any] = []

    def failing_build(plan):
        raise RuntimeError("the build failed")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: started["cuda"])
    monkeypatch.setattr(torch.cuda, "init", lambda: started.update(cuda=True))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: ["streams before the probe"])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)
    monkeypatch.setattr(ExperimentPlan, "_build_model", failing_build)
    plan = _plan().with_model(replace(MODEL, device=Devices.CUDA))
    with pytest.raises(RuntimeError, match="the build failed"):
        plan.probe((X_TRAIN[:2], Y_TRAIN[:2]))
    assert started["cuda"] and restored == [["streams before the probe"]]


def test_round_seventeen_a_train_only_run_without_epochs_names_the_missing_val_data():
    from types import SimpleNamespace

    from nnx.plans import _fit_metrics

    run: Any = SimpleNamespace(idps=[])  # a run stopped before its first epoch was recorded
    metrics = _fit_metrics(run, has_val=False)
    assert metrics["val"].reason == "the plan has no validation data"
    assert metrics["train"].reason == "the run recorded no epochs"
    assert _fit_metrics(run, has_val=True)["val"].reason == "the run recorded no epochs"


def test_round_twenty_three_a_factory_made_callbacks_monitor_is_a_plan_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = _plan(val=False).with_callback_factories(
        lambda: EarlyStopping(monitor=MonitorSpec(metric="loss", split="val"), patience=1)
    )
    assert plan.validate().ok  # a factory is never called by validate()
    with pytest.raises(PlanError) as caught:
        plan.fit()
    assert caught.value.diagnostics[0].path == "callback_factories[0].monitor"
    assert not (tmp_path / "runs").exists()  # raised before any run was reserved


def test_round_twenty_three_an_all_masked_batch_has_no_probe_loss():
    from nnx.tasks import TaskSpec

    net = NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU)
    model = NNModelParams(
        net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(2)
    )
    plan = _plan().with_net(net).with_model(model)
    assert plan.probe((X_TRAIN[:4], torch.full((4, 2), float("nan")))).loss is None  # as the default step records it
    assert plan.probe((X_TRAIN[:4], torch.zeros(4, 2))).loss is not None


def test_round_twenty_three_a_probe_seeds_only_the_streams_it_restores(monkeypatch):
    seeded: list[int] = []
    monkeypatch.setattr(torch, "manual_seed", lambda seed: seeded.append(seed))  # every device, XPU included
    _plan().probe((X_TRAIN[:2], Y_TRAIN[:2]))
    assert seeded == []


def test_round_twenty_four_every_setting_is_restored_and_a_failure_is_never_silent(monkeypatch):
    import nnx.seeding as seeding
    from nnx.seeding import _seed_scope

    before = os.environ.get("PYTHONHASHSEED")
    real = torch.use_deterministic_algorithms
    with pytest.raises(RuntimeError, match="cannot restore"):
        with _seed_scope():
            os.environ["PYTHONHASHSEED"] = "999"

            def broken(*args, **kwargs):
                raise RuntimeError("cannot restore")

            monkeypatch.setattr(torch, "use_deterministic_algorithms", broken)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", real)
    assert os.environ.get("PYTHONHASHSEED") == before  # restored although the torch setting failed

    class NoNotes(Exception):  # as on Python 3.10, where exceptions have no add_note
        add_note = None

    def failing(state, loader=None):
        raise RuntimeError("cannot restore")

    monkeypatch.setattr(seeding, "_restore_rng_state", failing)
    with pytest.warns(RuntimeWarning, match="cannot restore"), pytest.raises(NoNotes):
        with _seed_scope():
            raise NoNotes("the probe's own error")


def test_round_twenty_five_each_setting_is_restored_on_its_own():
    from nnx.seeding import _capture_seed_settings, _restore_seed_settings

    state = _capture_seed_settings()
    broken = {**state, "cudnn": None}  # the cuDNN flags cannot be restored
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(not state["deterministic"][0])
    try:
        with pytest.raises(TypeError):
            _restore_seed_settings(broken)
        assert torch.are_deterministic_algorithms_enabled() == state["deterministic"][0]  # restored regardless
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == state["env"]["CUBLAS_WORKSPACE_CONFIG"]
    finally:
        _restore_seed_settings(state)
