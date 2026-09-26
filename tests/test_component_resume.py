"""FEAT-005: component state survives a warm resume split at an epoch boundary.

A deterministic CPU JEPA run for 4 epochs and the same run split 2+2
(the second half in a fresh model, target encoder and callbacks, as a new
process would build them) end with identical model, optimizer, scheduler,
EarlyStopping and EMA target-encoder state. EarlyStopping's restored
patience stops a resumed run at the epoch an uninterrupted one stops.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
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
    jepa_train_step_factory,
    random_block_mask,
    set_seed,
)

# ---------------------------------------------------------------- JEPA fixture


def _jepa_parts(seed: int = 0):
    set_seed(seed)
    model = NNModel(
        net_params=NNParams(input_dim=12, output_dim=4, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    model.net = ViTNN(image_size=16, patch_size=4, in_channels=3, d_model=16, n_layers=1, n_heads=2)
    target = build_target_encoder(model.net)
    predictor = JEPAPredictor(embed_dim=16, n_patches=model.net.n_patches, predictor_dim=8, n_layers=1, n_heads=2)
    model.net.add_module("_jepa_predictor", predictor)

    def mask_fn(n_patches, device):
        # Global-RNG sampling: a warm resume must restore the RNG for the split
        # run to draw the same masks as the continuous one.
        return random_block_mask(n_patches=n_patches, grid_size=4, device=device)

    step = jepa_train_step_factory(target_encoder=target, predictor=predictor, mask_fn=mask_fn, ema_momentum=0.9)
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
