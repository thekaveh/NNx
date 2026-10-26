"""One warm resume, one rollback guard and one best-epoch decision for
``NNModel.train`` and ``Trainer.train`` (#536).

The two loops share the resume (validation, restore, ``ResumeStatus``),
its rollback, the epoch commit, the final LAST re-save and the
reconstructible-topology preflight, so a resume failure ends the same way
in both. The documented differences stay: a ``Trainer`` always resumes
``"additional"`` epochs and refuses a multi-rank process group.
"""

from __future__ import annotations

import dataclasses
import inspect
import random
import sys
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

import nnx.nn.nn_model as nn_model
import nnx.trainer.trainer as trainer_module
from nnx import (
    Callback,
    ComponentSpec,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNEvaluationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainerParams,
    NNTrainParams,
    Trainer,
)
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import ResumePointError
from nnx.objectives import supervised_objective

LOOPS = ("NNModel", "Trainer")
SGD = NNOptimParams.builder().sgd(max_lr=0.05).build()
_g = torch.Generator().manual_seed(0)
DATA = TensorDataset(torch.randn(16, 4, generator=_g), torch.randint(0, 2, (16,), generator=_g))


def _model(seed: int = 0) -> NNModel:
    torch.manual_seed(seed)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _plain(data: Any = DATA, **options: Any) -> DataLoader:
    return DataLoader(data, batch_size=8, **options)


def _shuffled() -> DataLoader:  # one torch.Generator, which a resume restores
    return DataLoader(DATA, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(0))


def _step(ctx: Any) -> NNEvaluationDataPoint:
    x, y = ctx.batch
    optimizer = ctx.optimizers["main"]
    optimizer.zero_grad()
    loss = ctx.model.loss_fn(ctx.model.net(x), y)
    loss.backward()
    optimizer.step()
    ctx.report_update("main")
    return NNEvaluationDataPoint(loss=loss.item(), error=loss.item())


def _fit(
    loop: str,
    model: NNModel,
    loader: Any,
    *,
    n_epochs: int = 1,
    resume: str | None = None,
    data_id: str = "parity",
    **kwargs: Any,
) -> Any:
    """``loop``'s fit of ``model``: ``NNModel.train`` with the default step,
    or ``Trainer.train`` with one optimizer ``"main"`` and a supervised step."""
    if loop == "NNModel":
        params = NNTrainParams(
            n_epochs=n_epochs, train_loader=loader, optim=SGD, data_id=data_id, resume_from_run_id=resume
        )
        return model.train(params=params, **kwargs)
    builder = NNTrainerParams.builder().n_epochs(n_epochs).train_loader(loader).optimizer("main", SGD).data_id(data_id)
    if resume is not None:
        builder = builder.resume_from(resume)
    return Trainer(model).train(params=builder.build(), trainer_step_fn=_step, **kwargs)


def _state(model: NNModel) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.net.state_dict().items()}


def _rng() -> tuple[Any, ...]:
    return random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone()


def _same_rng(before: tuple[Any, ...], after: tuple[Any, ...]) -> bool:
    return before[0] == after[0] and np.array_equal(before[1], after[1]) and torch.equal(before[2], after[2])


# --- the parity matrix ---------------------------------------------------------------------------------------


class _Refuse(Callback):
    def __init__(self, hook: str) -> None:
        self.hook = hook

    def on_train_begin(self, ctx: Any) -> None:
        if self.hook == "on_train_begin":
            raise RuntimeError("on_train_begin refused")

    def on_epoch_begin(self, ctx: Any) -> None:
        if self.hook == "on_epoch_begin":
            raise RuntimeError("on_epoch_begin refused")


class _Flaky:
    """A checkpointable component whose restore fails when asked to."""

    def __init__(self, fail: bool) -> None:
        self.spec, self.value, self.fail = ComponentSpec("flaky"), 0, fail

    def component_spec(self) -> ComponentSpec:
        return self.spec

    def component_state(self) -> dict[str, Any]:
        return {"value": self.value}

    def load_component_state(self, state: Any, *, version: int) -> None:
        self.value = state["value"]
        if self.fail:  # once: the component restore's own rollback reloads it
            self.fail = False
            raise RuntimeError("component restore failed")


class _Unreadable(Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> Any:
        raise OSError(f"cannot read sample {index}")


def _republish_between_verify_and_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """A concurrent writer republishes the resume point between its
    ``verify()`` and its load (Part 1 (a) of #536)."""
    real_verify = NNCheckpoint.verify
    accepted = inspect.signature(NNCheckpoint.save).parameters
    renamed = {
        "optimizer": "optimizer_state",
        "scheduler": "scheduler_state",
        "scaler": "scaler_state",
        "rng": "rng_state",
        "optimizers": "optimizers_state",
        "schedulers": "schedulers_state",
        "model": "resume_net_state",
    }

    def verify_then_republish(run: str, type: Any, root: Any = None) -> Any:
        manifest = real_verify(run=run, type=type, root=root)
        checkpoint, state = NNCheckpoint.load_with_training_state(run=run, type=type, root=root)
        assert checkpoint is not None and state is not None
        keywords = {renamed.get(key, key): value for key, value in state.items()}
        checkpoint.save(run=run, type=type, root=root, **{k: v for k, v in keywords.items() if k in accepted})
        return manifest

    monkeypatch.setattr(NNCheckpoint, "verify", staticmethod(verify_then_republish))


# case -> (source loader, resume loader, resume kwargs, setup)
CASES = {
    "failing_restore": lambda: (_shuffled, _plain, {}, None),
    "refused_on_train_begin": lambda: (_plain, _plain, {"callbacks": [_Refuse("on_train_begin")]}, None),
    "refused_on_epoch_begin": lambda: (_plain, _plain, {"callbacks": [_Refuse("on_epoch_begin")]}, None),
    "failing_component_restore": lambda: (_plain, _plain, {"components": [_Flaky(fail=True)]}, None),
    "warning_raised_as_error": lambda: (_plain, lambda: _plain(num_workers=2), {}, "warnings"),
    "failing_first_fetch": lambda: (_plain, lambda: _plain(_Unreadable()), {}, None),
    "republished_resume_point": lambda: (_plain, _plain, {}, "republish"),
}


def _resume_outcome(loop: str, case: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run ``case``'s failing resume through ``loop``: what it raised, whether
    the model and the RNG streams came back, and the ``ResumeStatus`` the
    shared resume produced (``None`` when it failed before producing one)."""
    source_loader, resume_loader, kwargs, setup = CASES[case]()
    source_components = [_Flaky(fail=False)] if "components" in kwargs else None
    source = _fit(loop, _model(0), source_loader(), data_id=f"source-{case}", components=source_components)

    statuses: list[Any] = []
    real_resume = nn_model._resume_fit

    def recording_resume(*args: Any, **keywords: Any) -> Any:
        statuses.append(None)
        resumed = real_resume(*args, **keywords)
        statuses[-1] = resumed.status
        return resumed

    with monkeypatch.context() as patch:
        patch.setattr(nn_model, "_resume_fit", recording_resume)
        patch.setattr(trainer_module, "_resume_fit", recording_resume)
        if setup == "republish":
            _republish_between_verify_and_load(patch)
        model = _model(1)
        before_model, before_rng = _state(model), _rng()
        with warnings.catch_warnings():
            if setup == "warnings":
                warnings.filterwarnings("error", message="exact warm-resume continuity")
            with pytest.raises(Exception) as raised:
                _fit(loop, model, resume_loader(), resume=source.id, data_id=f"resume-{case}", **kwargs)
        after_model, after_rng = _state(model), _rng()
    assert len(statuses) == 1, f"{loop} did not resume through the shared implementation"
    status = statuses[0]
    return {
        "error": raised.type,
        "model": after_model,
        "model_unchanged": all(torch.equal(after_model[key], value) for key, value in before_model.items()),
        "rng_unchanged": _same_rng(before_rng, after_rng),
        # The two loops resume different source runs (their checkpoints only resume
        # through the loop that wrote them): compare everything but the source's identity.
        "status": None if status is None else replace(status, source_run_id=None, source_checkpoint_id=None),
    }


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_failing_resume_ends_the_same_way_in_both_loops(case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outcomes = {loop: _resume_outcome(loop, case, monkeypatch) for loop in LOOPS}
    model_fit, trainer_fit = outcomes["NNModel"], outcomes["Trainer"]
    expected = {
        "failing_restore": ValueError,
        "refused_on_train_begin": RuntimeError,
        "refused_on_epoch_begin": RuntimeError,
        "failing_component_restore": RuntimeError,
        "warning_raised_as_error": RuntimeWarning,
        "failing_first_fetch": OSError,
        "republished_resume_point": ResumePointError,
    }[case]
    assert model_fit["error"] is trainer_fit["error"]
    assert issubclass(model_fit["error"], expected)
    # Nothing trained from the loaded state: both hand the caller's model and RNG back.
    assert model_fit["model_unchanged"] and trainer_fit["model_unchanged"]
    assert model_fit["rng_unchanged"] and trainer_fit["rng_unchanged"]
    assert all(torch.equal(value, trainer_fit["model"][key]) for key, value in model_fit["model"].items())
    assert model_fit["status"] == trainer_fit["status"]
    if case in {"failing_restore", "warning_raised_as_error", "republished_resume_point"}:
        assert model_fit["status"] is None  # the shared resume itself failed
    else:
        assert model_fit["status"] is not None and model_fit["status"].mode == "stateful"


# --- one rollback per failure --------------------------------------------------------------------------------


@pytest.mark.parametrize("loop", LOOPS)
def test_a_failing_stateful_restore_rolls_back_once(loop, tmp_path, monkeypatch):
    """Part 3 of #536: the resuming loader has no torch.Generator, so the RNG
    restore raises after the weights and global RNG were loaded. One restore
    load and one RNG restore, then one rollback load and one RNG rollback."""
    monkeypatch.chdir(tmp_path)
    source = _fit(loop, _model(0), _shuffled(), data_id="source")
    model = _model(1)
    before_model, before_rng = _state(model), torch.get_rng_state().clone()
    calls: list[str] = []
    real_load, real_set_rng = model.net.load_state_dict, torch.set_rng_state

    def load(state: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append("load")
        return real_load(state, *args, **kwargs)

    def set_rng(state: torch.Tensor) -> None:
        calls.append("rng")
        real_set_rng(state)

    with monkeypatch.context() as patch:
        patch.setattr(model.net, "load_state_dict", load)
        patch.setattr(torch, "set_rng_state", set_rng)
        with pytest.raises(ValueError, match="different number of torch.Generator"):
            _fit(loop, model, _plain(), resume=source.id, data_id="resumed")

    assert calls == ["load", "rng", "load", "rng"]  # restore, then exactly one rollback
    assert all(torch.equal(value, before_model[key]) for key, value in _state(model).items())
    assert torch.equal(torch.get_rng_state(), before_rng)


# --- one best-epoch decision ---------------------------------------------------------------------------------

ERRORS = [0.5, 0.7, 0.3, 0.3, 0.2]  # scripted per-epoch error; no val_loader, so BEST ranks the train record


@pytest.mark.parametrize("loop", LOOPS)
def test_the_save_reports_best_and_neither_loop_reevaluates_it(loop, tmp_path, monkeypatch):
    """Part 2 of #536: ``_save_checkpoints`` alone evaluates the BEST rule and
    reports what it published; each loop's in-memory best is that report."""
    monkeypatch.chdir(tmp_path)
    loader = DataLoader(DATA, batch_size=16)
    callers: list[str] = []
    real_best_err, real_save = nn_model._best_err, NNModel._save_checkpoints
    log: list[tuple[int, Any, int]] = []

    def counting_best_err(checkpoint: Any) -> float:
        callers.append(sys._getframe(1).f_code.co_name)
        return real_best_err(checkpoint)

    def logging_save(self: NNModel, *args: Any, **kwargs: Any) -> Any:
        saved = real_save(self, *args, **kwargs)
        handed_in, on_disk = kwargs["best_checkpoint"], NNCheckpoint.load(run=kwargs["run_id"], type=Checkpoints.BEST)
        assert on_disk is not None
        epoch = saved.checkpoint.idp.epoch_idx
        assert saved.best == (on_disk.idp.epoch_idx == epoch)  # the report is what reached disk
        log.append((epoch, None if handed_in is None else handed_in.idp.epoch_idx, on_disk.idp.epoch_idx))
        return saved

    def scripted(ctx: Any) -> NNEvaluationDataPoint:
        x, y = ctx.batch
        optimizer = ctx.optimizer if loop == "NNModel" else ctx.optimizers["main"]
        optimizer.zero_grad()
        loss = ctx.model.loss_fn(ctx.model.net(x), y)
        loss.backward()
        optimizer.step()
        return NNEvaluationDataPoint(loss=loss.item(), error=ERRORS[ctx.epoch_idx])

    monkeypatch.setattr(nn_model, "_best_err", counting_best_err)
    monkeypatch.setattr(trainer_module, "_best_err", counting_best_err, raising=False)
    monkeypatch.setattr(NNModel, "_save_checkpoints", logging_save)
    if loop == "NNModel":
        _model().train(
            params=NNTrainParams(n_epochs=len(ERRORS), train_loader=loader, optim=SGD), train_step_fn=scripted
        )
    else:
        params = NNTrainerParams.builder().n_epochs(len(ERRORS)).train_loader(loader).optimizer("main", SGD).build()
        Trainer(_model()).train(params=params, trainer_step_fn=scripted)

    assert set(callers) == {"_save_checkpoints"}  # neither loop's _train_impl
    # The epoch log of e69fd74: the in-memory best each save is handed is the previous on-disk BEST.
    assert log == [(0, None, 0), (1, 0, 0), (2, 0, 2), (3, 2, 2), (4, 2, 4)]


# --- the shared epoch commit and final LAST ------------------------------------------------------------------


def test_the_trainer_commits_epochs_and_the_final_last_through_the_shared_helpers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    real_commit, real_final = NNModel._commit_epoch, NNModel._save_final_last

    def commit(self: NNModel, **kwargs: Any) -> Any:
        calls.append(f"commit {kwargs['idx_epoch']}")
        return real_commit(self, **kwargs)

    def final(self: NNModel, **kwargs: Any) -> None:
        calls.append("final LAST")
        real_final(self, **kwargs)

    monkeypatch.setattr(NNModel, "_commit_epoch", commit)
    monkeypatch.setattr(NNModel, "_save_final_last", final)
    run = _fit("Trainer", _model(), _plain(), n_epochs=2)

    assert calls == ["commit 0", "commit 1", "final LAST"]
    checkpoint, state = NNCheckpoint.load_with_training_state(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None and state is not None
    assert state["completed_epoch"] == 1 and set(state["optimizers"]) == {"main"}
    assert set(state["optimizer_topologies"]) == {"main"} and state["optimizer"] is None


# --- one topology preflight ----------------------------------------------------------------------------------


@pytest.mark.parametrize("objective", [True, False], ids=["objective", "step_function"])
def test_both_train_methods_run_one_topology_preflight(objective, tmp_path, monkeypatch):
    """Part 1 (b) of #536: the same preflight, with the same verdict on who
    owns the update, before either loop reserves a run."""
    monkeypatch.chdir(tmp_path)
    preflights: dict[str, list[bool]] = {loop: [] for loop in LOOPS}
    asserted: dict[str, int] = dict.fromkeys(LOOPS, 0)
    real_preflight, real_assert = nn_model._topology_preflight, NNModel._assert_reconstructible_topology
    loop = "NNModel"

    def preflight(model: Any, *, nnx_updates: bool) -> None:
        preflights[loop].append(nnx_updates)
        real_preflight(model, nnx_updates=nnx_updates)

    def counting_assert(self: NNModel) -> None:
        asserted[loop] += 1
        real_assert(self)

    monkeypatch.setattr(nn_model, "_topology_preflight", preflight)
    monkeypatch.setattr(trainer_module, "_topology_preflight", preflight)
    monkeypatch.setattr(NNModel, "_assert_reconstructible_topology", counting_assert)
    params = NNTrainParams(n_epochs=1, train_loader=_plain(), optim=SGD, data_id="preflight")
    if objective:
        _model().train(params=params, objective=supervised_objective())
    else:
        _model().train(params=params, train_step_fn=lambda ctx: NNEvaluationDataPoint(loss=0.0, error=0.0))
    loop = "Trainer"
    trainer_params = NNTrainerParams.builder().n_epochs(1).train_loader(_plain()).optimizer("main", SGD).build()
    if objective:
        Trainer(_model()).train(params=trainer_params, objective=supervised_objective())
    else:
        Trainer(_model()).train(params=trainer_params, trainer_step_fn=_step)

    assert preflights == {"NNModel": [objective], "Trainer": [objective]}
    assert asserted == {"NNModel": int(objective), "Trainer": int(objective)}


# --- the documented differences ------------------------------------------------------------------------------


def test_a_trainer_always_resumes_additional_epochs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = _fit("Trainer", _model(), _plain(), n_epochs=2, data_id="source")
    assert "resume_epochs" not in {field.name for field in dataclasses.fields(NNTrainerParams)}
    resumed = _fit("Trainer", _model(), _plain(), n_epochs=2, resume=source.id, data_id="resumed")

    assert sorted({idp.epoch_idx for idp in resumed.idps}) == [2, 3]  # n_epochs more, not the plan's rest
    assert resumed.resume_status.mode == "stateful" and resumed.resume_status.source_epoch == 1


def test_trainer_train_still_refuses_a_multi_rank_process_group(tmp_path, monkeypatch):
    import torch.distributed as dist

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *args, **kwargs: 2)
    with pytest.raises(RuntimeError, match=r"Trainer\.train\(\) trains in a single process"):
        _fit("Trainer", _model(), _plain())


# --- the messages the shared resume gives both loops ---------------------------------------------------------


def test_a_source_without_training_state_warns_alike_in_both_loops(tmp_path, monkeypatch):
    """A weights-only resume point (``resume_mode="auto"``) restores the
    weights and warns, in the same words, that everything else, the scaler
    included, starts fresh."""
    monkeypatch.chdir(tmp_path)
    messages = {}
    for loop in LOOPS:
        source = _fit(loop, _model(0), _plain(), data_id=f"source-{loop}")
        last = NNCheckpoint.load(run=source.id, type=Checkpoints.LAST)
        assert last is not None
        last.save(run=source.id, type=Checkpoints.LAST)  # republished without training state
        with pytest.warns(RuntimeWarning, match="checkpoint has no training-state sidecar") as caught:
            run = _fit(loop, _model(1), _plain(), resume=source.id, data_id=f"resume-{loop}")
        assert run.resume_status.mode == "weights_only" and run.resume_status.source_epoch == 0
        messages[loop] = [str(w.message) for w in caught if "training-state sidecar" in str(w.message)]
    assert messages["NNModel"] == messages["Trainer"]
    assert "optimizer, scheduler, scaler, RNG and component state restart" in messages["Trainer"][0]


@pytest.mark.parametrize("loop", LOOPS)
def test_a_stateful_resume_without_a_horizon_is_refused_before_the_schedule_is_built(loop, tmp_path, monkeypatch):
    """The resume's horizon refusal comes before a schedule's own
    construction refusal (an optimizer_update-clock one-cycle schedule with
    no total_steps and a custom step, which plans no updates)."""
    from nnx import NNSchedulerParams, Schedulers

    monkeypatch.chdir(tmp_path)
    one_cycle = NNSchedulerParams(
        kind=Schedulers.ONE_CYCLE,
        clock="optimizer_update",
        min_lr=0.0,
        factor=0.5,
        patience=1,
        cooldown=0,
        threshold=0.0,
    )
    resume = "a" * 32  # never read: the refusal comes first
    with pytest.raises(ValueError, match=r"resuming one_cycle( for 'main')? requires scheduler.total_steps"):
        if loop == "NNModel":
            params = NNTrainParams(
                n_epochs=1, train_loader=_plain(), optim=SGD, scheduler=one_cycle, resume_from_run_id=resume
            )
            _model().train(params=params, train_step_fn=lambda ctx: NNEvaluationDataPoint(loss=0.0, error=0.0))
        else:
            params = (
                NNTrainerParams.builder()
                .n_epochs(1)
                .train_loader(_plain())
                .optimizer("main", SGD)
                .scheduler("main", one_cycle)
                .resume_from(resume)
                .build()
            )
            Trainer(_model()).train(params=params, trainer_step_fn=_step)
