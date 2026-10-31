"""FEAT-005: component state survives a warm resume split at an epoch boundary.

A deterministic CPU JEPA run for 4 epochs and the same run split 2+2
(the second half in a fresh model, target encoder and callbacks, as a new
process would build them) end with identical model, optimizer, scheduler,
EarlyStopping and EMA target-encoder state. EarlyStopping's restored
patience stops a resumed run at the epoch an uninterrupted one stops.
The resumed step must use the checkpoint's EMA momentum, as the JEPA
objective must (#560).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    ComponentRestoreError,
    Devices,
    EarlyStopping,
    EvalStepContext,
    JEPAPredictor,
    Losses,
    Nets,
    NNCheckpoint,
    NNEvaluationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNSchedulerParams,
    NNTrainParams,
    ViTNN,
    build_target_encoder,
    jepa_objective,
    jepa_train_step_factory,
    random_block_mask,
    set_seed,
)
from nnx.paradigms.jepa import JEPATrainStep
from tests._checkpoint_files import restamp

# ---------------------------------------------------------------- JEPA fixture


def _jepa_mask_fn(n_patches, device):
    # Global-RNG sampling: a warm resume must restore the RNG for the split
    # run to draw the same masks as the continuous one.
    return random_block_mask(n_patches=n_patches, grid_size=4, device=device)


def _jepa_parts(seed: int = 0, ema_momentum: float = 0.9):
    set_seed(seed)
    model = NNModel(
        net_params=NNParams(input_dim=12, output_dim=4, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    model.net = ViTNN(image_size=16, patch_size=4, in_channels=3, d_model=16, n_layers=1, n_heads=2)
    target = build_target_encoder(model.net)
    predictor = JEPAPredictor(embed_dim=16, n_patches=model.net.n_patches, predictor_dim=8, n_layers=1, n_heads=2)
    model.net.add_module("_jepa_predictor", predictor)
    step = jepa_train_step_factory(
        target_encoder=target, predictor=predictor, mask_fn=_jepa_mask_fn, ema_momentum=ema_momentum
    )
    return model, target, step


def _jepa_loader() -> DataLoader:
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(8, 3, 16, 16, generator=generator)
    return DataLoader(TensorDataset(x, torch.zeros(8, dtype=torch.long)), batch_size=4, shuffle=False)


def _scripted_eval(values: list[float]):
    def eval_step(ctx: EvalStepContext) -> NNEvaluationDataPoint:
        return NNEvaluationDataPoint(loss=values[ctx.epoch_idx], error=values[ctx.epoch_idx])

    return eval_step


def _jepa_params(n_epochs: int, **resume) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=n_epochs,
        optim=NNOptimParams.builder().adam(max_lr=1e-3).build(),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=0, cooldown=0, threshold=1e-3),
        train_loader=_jepa_loader(),
        val_loader=_jepa_loader(),
        **resume,
    )


VAL = [0.9, 0.8, 0.85, 0.7]


def test_jepa_continuous_and_split_runs_end_in_identical_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    model_a, target_a, step_a = _jepa_parts()
    stopper_a = EarlyStopping(monitor="val_edp.loss", patience=5)
    run_a = model_a.train(
        params=_jepa_params(4), train_step_fn=step_a, eval_step_fn=_scripted_eval(VAL), callbacks=[stopper_a]
    )

    model_b, target_b, step_b = _jepa_parts()
    first = model_b.train(
        params=_jepa_params(2),
        train_step_fn=step_b,
        eval_step_fn=_scripted_eval(VAL),
        callbacks=[EarlyStopping(monitor="val_edp.loss", patience=5)],
    )
    model_c, target_c, step_c = _jepa_parts(seed=123)  # a fresh "process": different init
    stopper_c = EarlyStopping(monitor="val_edp.loss", patience=5)
    second = model_c.train(
        params=_jepa_params(2, resume_from_run_id=first.id),
        train_step_fn=step_c,
        eval_step_fn=_scripted_eval(VAL),
        callbacks=[stopper_c],
    )
    assert second.resume_status is not None and second.resume_status.mode == "stateful"
    assert set(second.resume_status.restored_components) == {"jepa.target_encoder", "early_stopping"}

    for key, value in model_a.net.state_dict().items():
        torch.testing.assert_close(model_c.net.state_dict()[key], value, msg=key)
    for key, value in target_a.state_dict().items():
        torch.testing.assert_close(target_c.state_dict()[key], value, msg=f"target {key}")
    assert stopper_c.component_state() == stopper_a.component_state()

    state_a = NNCheckpoint.load_training_state(run=run_a.id, type=Checkpoints.LAST)
    state_c = NNCheckpoint.load_training_state(run=second.id, type=Checkpoints.LAST)
    assert state_a is not None and state_c is not None
    assert state_a["completed_epoch"] == state_c["completed_epoch"] == 3
    assert state_c["scheduler"] == state_a["scheduler"]
    for group_a, group_c in zip(
        state_a["optimizer"]["state"].values(), state_c["optimizer"]["state"].values(), strict=True
    ):
        torch.testing.assert_close(group_c["exp_avg"], group_a["exp_avg"])
        torch.testing.assert_close(group_c["exp_avg_sq"], group_a["exp_avg_sq"])
    assert set(state_c["components"]) == {"jepa.target_encoder", "early_stopping"}
    assert NNRun.load(second.id).resume_status == second.resume_status


# ------------------------------------------------ JEPA EMA momentum on resume


def _jepa_fit(mode: str, seed: int = 0, ema_momentum: float = 0.9) -> tuple[NNModel, torch.nn.Module, dict[str, Any]]:
    """A fresh model, its EMA target and the ``train()`` keyword that trains
    it with the imperative step or with its objective twin."""
    model, target, step = _jepa_parts(seed, ema_momentum)
    if mode == "step":
        return model, target, {"train_step_fn": step}
    predictor = model.net.get_submodule("_jepa_predictor")
    return model, target, {"objective": jepa_objective(target, predictor, _jepa_mask_fn, ema_momentum=ema_momentum)}


def _rewrite_last_training_state(tmp_path, run_id: str, edit) -> Any:
    """Apply ``edit`` to LAST's training state on disk; returns the sidecar."""
    checkpoint = NNCheckpoint.load(run_id, Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.training_state_id is not None
    sidecar = tmp_path / "runs" / run_id / "checkpoints" / f"last.pt.opt.{checkpoint.training_state_id}.pt"
    state = torch.load(sidecar, weights_only=True)
    edit(state)
    torch.save(state, sidecar)
    restamp(sidecar)
    return sidecar


_MOMENTUM_REFUSAL = {
    "step": "incompatible component 'jepa.target_encoder': the checkpoint's JEPA EMA momentum 0.9 does not match "
    "this step's 0.5: resume with ema_momentum=0.9",
    "objective": "incompatible component 'jepa.objective': the checkpoint's JEPA objective {'objective': 'jepa', "
    "'loss': 'latent_mse', 'ema_momentum': 0.9} does not match this one {'objective': 'jepa', 'loss': 'latent_mse', "
    "'ema_momentum': 0.5}",
}


@pytest.mark.parametrize("mode", sorted(_MOMENTUM_REFUSAL))
def test_a_changed_ema_momentum_is_refused_before_anything_is_restored(tmp_path, monkeypatch, mode):
    """#560: the imperative step's resume restored a 0.9 target and went on
    with 0.5, while the objective refused the change. Both now raise one
    ComponentRestoreError naming the component and both momenta, and leave
    the resumed net and target as they were built."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model, _, fit = _jepa_fit(mode, ema_momentum=0.9)
    first = model.train(params=_jepa_params(1), eval_step_fn=_scripted_eval(VAL), **fit)

    model, target, fit = _jepa_fit(mode, seed=123, ema_momentum=0.5)
    net_before, target_before = copy.deepcopy(model.net.state_dict()), copy.deepcopy(target.state_dict())
    with pytest.raises(ComponentRestoreError) as caught:
        model.train(params=_jepa_params(1, resume_from_run_id=first.id), eval_step_fn=_scripted_eval(VAL), **fit)
    assert caught.value.problems == (_MOMENTUM_REFUSAL[mode],)
    assert all(torch.equal(model.net.state_dict()[key], value) for key, value in net_before.items())
    assert all(torch.equal(target.state_dict()[key], value) for key, value in target_before.items())


@pytest.mark.parametrize(
    ("saved", "problem"),
    [
        (None, "the checkpoint's JEPA target-encoder state records no ema_momentum"),
        ("0.9", "malformed JEPA EMA momentum '0.9'"),
        (1.5, "malformed JEPA EMA momentum 1.5"),
    ],
    ids=["missing", "not_a_number", "out_of_range"],
)
def test_a_target_state_without_its_ema_momentum_joins_the_pre_restore_report(tmp_path, monkeypatch, saved, problem):
    """#560: nothing read the saved momentum back. A target state without one
    (or with a malformed one: a non-number or a value outside [0, 1)) is now
    a problem in the one ComponentRestoreError that lists the other
    components' problems too, before anything is restored."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model, _, step = _jepa_parts()
    first = model.train(
        params=_jepa_params(1),
        train_step_fn=step,
        eval_step_fn=_scripted_eval(VAL),
        callbacks=[EarlyStopping(monitor="val_edp.loss", patience=5)],
    )

    def edit(state: dict[str, Any]) -> None:
        target_state = state["components"]["jepa.target_encoder"]["state"]
        if saved is None:
            del target_state["ema_momentum"]
        else:
            target_state["ema_momentum"] = saved

    _rewrite_last_training_state(tmp_path, first.id, edit)
    model, target, step = _jepa_parts(seed=123)  # the momentum the run was trained with
    target_before = copy.deepcopy(target.state_dict())
    with pytest.raises(ComponentRestoreError) as caught:
        model.train(
            params=_jepa_params(1, resume_from_run_id=first.id),
            train_step_fn=step,
            eval_step_fn=_scripted_eval(VAL),
            callbacks=[EarlyStopping(monitor="train_edp.loss", patience=5)],  # a second problem
        )
    problems = caught.value.problems
    assert len(problems) == 2 and f"incompatible component 'jepa.target_encoder': {problem}" in problems
    assert any(found.startswith("incompatible component 'early_stopping': ") for found in problems)
    assert all(torch.equal(target.state_dict()[key], value) for key, value in target_before.items())


def test_a_jepa_checkpoint_without_component_state_resumes_with_a_fresh_target(tmp_path, monkeypatch):
    """A sidecar written before component state existed records no momentum:
    the step still resumes with its fresh target and a warning (jepa.md
    §4.5), whatever momentum it is built with."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model, _, step = _jepa_parts()
    first = model.train(params=_jepa_params(1), train_step_fn=step, eval_step_fn=_scripted_eval(VAL))

    def predate_component_state(state: dict[str, Any]) -> None:
        for key in ("components", "optimizers", "optimizer_types", "schedulers", "scheduler_types"):
            state.pop(key)
        state["nnx_training_state_version"] = 3  # as written before FEAT-005

    sidecar = _rewrite_last_training_state(tmp_path, first.id, predate_component_state)
    (sidecar.parent / "last.pt.manifest.json").unlink()  # legacy checkpoints carry no manifest (#394)

    loaded: list[int] = []
    load = JEPATrainStep.load_component_state

    def spy(self: JEPATrainStep, state: Any, *, version: int) -> None:
        loaded.append(version)
        load(self, state, version=version)

    monkeypatch.setattr(JEPATrainStep, "load_component_state", spy)
    model, _, step = _jepa_parts(seed=123, ema_momentum=0.5)
    with pytest.warns(RuntimeWarning, match="predates component state; jepa.target_encoder start fresh"):
        second = model.train(
            params=_jepa_params(1, resume_from_run_id=first.id), train_step_fn=step, eval_step_fn=_scripted_eval(VAL)
        )
    assert second.resume_status is not None and second.resume_status.mode == "stateful"
    assert second.resume_status.restored_components == ()
    assert second.resume_status.fresh_components == ("jepa.target_encoder",)
    assert loaded == []


def test_a_numpy_ema_momentum_is_saved_as_a_float_and_resumes(tmp_path, monkeypatch):
    """The step saved its momentum as given, so a NumPy scalar (a sweep-grid
    point) wrote a training state torch.load(weights_only=True) refuses. It is
    saved as a float, as jepa_objective saves it, and a resume with the same
    value restores the target."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model, _, step = _jepa_parts(ema_momentum=np.float64(0.9))
    first = model.train(params=_jepa_params(1), train_step_fn=step, eval_step_fn=_scripted_eval(VAL))
    state = NNCheckpoint.load_training_state(run=first.id, type=Checkpoints.LAST)
    assert state is not None
    assert type(state["components"]["jepa.target_encoder"]["state"]["ema_momentum"]) is float

    model, _, step = _jepa_parts(seed=123, ema_momentum=0.9)
    second = model.train(
        params=_jepa_params(1, resume_from_run_id=first.id), train_step_fn=step, eval_step_fn=_scripted_eval(VAL)
    )
    assert second.resume_status is not None and second.resume_status.restored_components == ("jepa.target_encoder",)


# ------------------------------------------------ EarlyStopping continuity


def _tiny_supervised():
    set_seed(0)
    return NNModel(
        net_params=NNParams(input_dim=3, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(2)
    return DataLoader(
        TensorDataset(torch.randn(6, 3, generator=generator), torch.randint(0, 2, (6,), generator=generator)),
        batch_size=3,
    )


def _params(n_epochs: int, **resume) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=n_epochs,
        optim=NNOptimParams.builder().sgd(max_lr=0.01).build(),
        train_loader=_loader(),
        val_loader=_loader(),
        **resume,
    )


class _ResetCounter(EarlyStopping):
    """EarlyStopping that records the order of reset and restore."""

    def __init__(self, events):
        super().__init__(monitor="val_edp.error", patience=2)
        self.events = events

    def on_train_begin(self, ctx):
        self.events.append(("reset", self._wait))
        super().on_train_begin(ctx)

    def load_component_state(self, state, *, version):
        super().load_component_state(state, version=version)
        self.events.append(("restore", self._wait))

    def on_epoch_end(self, ctx):
        self.events.append(("epoch", ctx.epoch))
        super().on_epoch_end(ctx)


def test_restored_patience_stops_at_the_uninterrupted_epoch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    values = [0.6, 0.4, 0.41, 0.42]

    continuous = _tiny_supervised().train(
        params=_params(10),
        eval_step_fn=_scripted_eval(values + [0.43] * 6),
        callbacks=[EarlyStopping(monitor="val_edp.error", patience=2)],
    )
    assert continuous.idps[-1].epoch_idx == 3  # best at epoch 1; patience 2 runs out at epoch 3

    first = _tiny_supervised().train(
        params=_params(3),
        eval_step_fn=_scripted_eval(values),
        callbacks=[EarlyStopping(monitor="val_edp.error", patience=2)],
    )
    events: list[tuple[str, int]] = []
    resumed = _tiny_supervised().train(
        params=_params(7, resume_from_run_id=first.id),
        eval_step_fn=_scripted_eval(values + [0.43] * 6),
        callbacks=[_ResetCounter(events)],
    )
    assert resumed.idps[-1].epoch_idx == 3  # one resumed epoch, then the restored patience stops it
    # Reset once, then the restored state (wait=1), then the first resumed epoch.
    assert events[:3] == [("reset", 0), ("restore", 1), ("epoch", 3)]
    assert [name for name, _ in events].count("reset") == 1


def test_a_fresh_early_stopping_restarts_patience_without_resume(tmp_path, monkeypatch):
    """Contrast: without restored state the same split would run on."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    values = [0.6, 0.4, 0.41, 0.42]
    first = _tiny_supervised().train(
        params=_params(3),
        eval_step_fn=_scripted_eval(values),
        callbacks=[EarlyStopping(monitor="val_edp.error", patience=2)],
    )
    resumed = _tiny_supervised().train(
        params=_params(4, resume_from_run_id=first.id, resume_mode="weights_only"),
        eval_step_fn=_scripted_eval(values + [0.43] * 6),
        callbacks=[EarlyStopping(monitor="val_edp.error", patience=2)],
    )
    assert resumed.resume_status is not None and resumed.resume_status.mode == "weights_only"
    assert resumed.resume_status.fresh_components == ("early_stopping",)
    assert resumed.idps[-1].epoch_idx > 3


def test_weights_only_and_stateful_modes_are_reported_and_enforced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    first = _tiny_supervised().train(params=_params(1))
    fresh = _tiny_supervised().train(params=_params(1, data_id="other"))
    assert fresh.resume_status is not None and fresh.resume_status.mode == "fresh"
    stateful = _tiny_supervised().train(params=_params(1, resume_from_run_id=first.id))
    assert stateful.resume_status is not None and stateful.resume_status.mode == "stateful"
    assert NNRun.load(stateful.id).resume_status == stateful.resume_status
    with pytest.raises(ValueError, match="resume_mode"):
        NNTrainParams(n_epochs=1, resume_mode="sometimes")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["auto", "weights_only"])
def test_a_failure_right_after_the_resume_restore_hands_the_model_back(tmp_path, monkeypatch, mode):
    import nnx.nn.nn_model as nn_model_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    first = _tiny_supervised().train(params=_params(1))
    model = _tiny_supervised()
    with torch.no_grad():
        for parameter in model.net.parameters():
            parameter.add_(1.0)  # differs from the checkpoint
    before = {name: value.clone() for name, value in model.net.state_dict().items()}
    torch.manual_seed(5)
    rng_before = torch.get_rng_state()

    real = nn_model_module.ResumeStatus

    def refuse(**fields):  # stands in for a warnings-as-errors filter after the restore
        if fields.get("mode") in ("stateful", "weights_only"):
            raise RuntimeError("after the restore")
        return real(**fields)

    monkeypatch.setattr(nn_model_module, "ResumeStatus", refuse)
    with pytest.raises(RuntimeError, match="after the restore"):
        model.train(params=_params(2, resume_from_run_id=first.id, resume_mode=mode))
    assert all(torch.equal(before[name], value) for name, value in model.net.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_a_sidecar_warning_raised_as_an_error_hands_the_model_back(tmp_path, monkeypatch):
    import warnings

    from nnx import Checkpoints, NNCheckpoint

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    first = _tiny_supervised().train(params=_params(1))
    last = NNCheckpoint.load(run=first.id, type=Checkpoints.LAST)
    assert last is not None
    last.save(run=first.id, type=Checkpoints.LAST)  # no optimizer_state: weights only
    model = _tiny_supervised()
    with torch.no_grad():
        for parameter in model.net.parameters():
            parameter.add_(1.0)
    before = {name: value.clone() for name, value in model.net.state_dict().items()}
    torch.manual_seed(5)
    rng_before = torch.get_rng_state()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="checkpoint has no training-state sidecar")
        with pytest.raises(Warning, match="training-state sidecar"):
            model.train(params=_params(2, resume_from_run_id=first.id, resume_from_checkpoint="last"))
    assert all(torch.equal(before[name], value) for name, value in model.net.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_an_unowned_warning_raised_as_an_error_on_the_first_resumed_epoch_hands_the_model_back(tmp_path, monkeypatch):
    import warnings

    import nnx.nn.nn_model as nn_model_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    first = _tiny_supervised().train(params=_params(1))

    def warn(*_args, **_kwargs):
        warnings.warn("parameters ['x'] are trainable but no optimizer owns them", RuntimeWarning, stacklevel=2)

    monkeypatch.setattr(nn_model_module, "warn_unowned_trainables", warn)
    model = _tiny_supervised()
    with torch.no_grad():
        for parameter in model.net.parameters():
            parameter.add_(1.0)
    before = {name: value.clone() for name, value in model.net.state_dict().items()}
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*no optimizer owns")
        with pytest.raises(RuntimeWarning, match="no optimizer owns"):
            model.train(params=_params(2, resume_from_run_id=first.id))
    assert all(torch.equal(before[name], value) for name, value in model.net.state_dict().items())


def test_the_worker_rng_warning_raised_as_an_error_hands_the_model_back(tmp_path, monkeypatch):
    import warnings

    import nnx.nn.nn_model as nn_model_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    first = _tiny_supervised().train(params=_params(1))
    monkeypatch.setattr(nn_model_module, "_loader_num_workers", lambda _loader: 1)
    model = _tiny_supervised()
    with torch.no_grad():
        for parameter in model.net.parameters():
            parameter.add_(1.0)
    before = {name: value.clone() for name, value in model.net.state_dict().items()}
    torch.manual_seed(5)
    rng_before = torch.get_rng_state()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="exact warm-resume continuity")
        with pytest.raises(Warning, match="exact warm-resume continuity"):
            model.train(params=_params(2, resume_from_run_id=first.id))
    assert all(torch.equal(before[name], value) for name, value in model.net.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng_before)
