"""FEAT-040: objective adapters at the FEAT-005 registry boundary.

Two epochs equal one epoch plus a stateful resume (into a freshly built
model, target and objective, as a new process would build them): the JEPA
target encoder, predictor reference, objective spec and EMA counter, and
the diffusion objective's generator, all continue. A saved state with a
second predictor payload, another predictor reference or another spec is
rejected before anything is restored; LAST and custom snapshots keep the
predictor's weights once, with the net.
"""

from __future__ import annotations

import copy

import pytest
import torch

from nnx import (
    Activations,
    Checkpoints,
    ComponentRestoreError,
    Devices,
    DiffusionMLP,
    JEPAObjective,
    JEPAPredictor,
    Losses,
    ModelCheckpoint,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    NoiseSchedulers,
    ViTNN,
    build_target_encoder,
    diffusion_objective,
    random_block_mask,
    set_seed,
)

_PLATEAU = NNSchedulerParams(min_lr=0.0, factor=0.5, patience=10, cooldown=0, threshold=0.0)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)


def _params(loader, n_epochs: int, **fields) -> NNTrainParams:
    fields.setdefault("overwrite_existing", True)
    return NNTrainParams(
        n_epochs=n_epochs,
        train_loader=loader,
        optim=NNOptimParams.builder().adam(max_lr=1e-3).build(),
        scheduler=_PLATEAU,
        save_phase_checkpoints=False,
        **fields,
    )


def _shell(seed: int) -> NNModel:
    set_seed(seed)
    return NNModel(
        net_params=NNParams(input_dim=12, output_dim=4, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


# ---------------------------------------------------------------- JEPA


def _jepa(seed: int = 0, *, use: str = "_jepa_predictor", momentum: float = 0.9, cls=JEPAObjective, spare=False):
    model = _shell(seed)
    model.net = ViTNN(image_size=16, patch_size=4, in_channels=3, d_model=16, n_layers=1, n_heads=2)
    target = build_target_encoder(model.net)

    def predictor_module():
        return JEPAPredictor(embed_dim=16, n_patches=model.net.n_patches, predictor_dim=8, n_layers=1, n_heads=2)

    names = ("_jepa_predictor", "_spare_predictor") if spare else ("_jepa_predictor",)
    for name in names:
        model.net.add_module(name, predictor_module())
    predictor = getattr(model.net, use)

    def mask_fn(n_patches, device):
        # Global-RNG masks: the resume restores that RNG, so the split run draws the same masks.
        return random_block_mask(n_patches=n_patches, grid_size=4, device=device)

    return model, target, cls(target, predictor, mask_fn, ema_momentum=momentum)


def _images():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(6, 3, 16, 16, generator=generator)
    return [(x[:4], torch.zeros(4, dtype=torch.long)), (x[4:], torch.zeros(2, dtype=torch.long))]


def test_jepa_two_epochs_equal_one_plus_a_stateful_resume():
    model_a, target_a, objective_a = _jepa()
    model_a.train(_params(_images(), 2), objective=objective_a)

    model_b, _, objective_b = _jepa()
    first = model_b.train(_params(_images(), 1), objective=objective_b)
    model_c, target_c, objective_c = _jepa(seed=123)  # a fresh "process": different init
    second = model_c.train(_params(_images(), 1, resume_from_run_id=first.id), objective=objective_c)

    assert second.resume_status is not None and second.resume_status.mode == "stateful"
    assert "jepa.objective" in second.resume_status.restored_components
    for key, value in model_a.net.state_dict().items():
        torch.testing.assert_close(model_c.net.state_dict()[key], value, msg=key)
    for key, value in target_a.state_dict().items():
        torch.testing.assert_close(target_c.state_dict()[key], value, msg=f"target {key}")
    assert objective_c.ema_updates == objective_a.ema_updates == 4  # two committed updates per epoch
    assert not target_c.training and all(not p.requires_grad for p in target_c.parameters())


def test_last_and_custom_snapshots_keep_the_predictor_once_with_the_net():
    model, _, objective = _jepa()
    run = model.train(_params(_images(), 1), objective=objective, callbacks=[ModelCheckpoint(epochs=[0], tag="snap")])
    state = NNCheckpoint.load_training_state(run=run.id, type=Checkpoints.LAST)
    assert state is not None
    component = state["components"]["jepa.objective"]["state"]
    assert set(component) == {"spec", "predictor", "target_encoder", "ema_updates"}
    assert component["predictor"] == "_jepa_predictor" and component["ema_updates"] == 2
    assert not any(key.startswith("_jepa_predictor") for key in component["target_encoder"])
    last = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    snapshot = NNCheckpoint.from_file(f"runs/{run.id}/checkpoints/snap_e0.pt")
    for checkpoint in (last, snapshot):
        assert checkpoint is not None
        predictor_keys = [key for key in checkpoint.net_state if key.startswith("_jepa_predictor.")]
        assert predictor_keys and len(predictor_keys) == len(set(predictor_keys))


def _resume_with(writer=JEPAObjective, *, seed: int = 5, **parts):
    model, _, objective = _jepa(cls=writer, spare=parts.get("spare", False))
    first = model.train(_params(_images(), 1), objective=objective)
    fresh, target, fresh_objective = _jepa(seed=seed, **parts)
    before = copy.deepcopy(fresh.net.state_dict()), copy.deepcopy(target.state_dict())
    with pytest.raises(ComponentRestoreError) as caught:
        fresh.train(_params(_images(), 1, resume_from_run_id=first.id), objective=fresh_objective)
    assert all(torch.equal(fresh.net.state_dict()[k], v) for k, v in before[0].items())
    assert all(torch.equal(target.state_dict()[k], v) for k, v in before[1].items())
    return str(caught.value)


class _DuplicatingObjective(JEPAObjective):
    """A writer that also saved the predictor's weights in its own state."""

    def component_state(self):
        return {**super().component_state(), "predictor_state": dict(self.predictor.state_dict())}


def test_a_duplicate_predictor_payload_is_rejected_before_anything_is_restored():
    assert "second predictor (or other) payload" in _resume_with(_DuplicatingObjective)


def test_another_predictor_reference_or_spec_is_rejected():
    # The same net (and optimizer topology) with two predictor-shaped modules: the
    # resumed objective points at the other one.
    assert "this run's is model.net._spare_predictor" in _resume_with(spare=True, use="_spare_predictor")
    assert "does not match this one" in _resume_with(momentum=0.5)


# ---------------------------------------------------------------- diffusion


def _diffusion(seed: int = 0):
    model = _shell(seed)
    model.net = DiffusionMLP(input_dim=2, hidden_dims=[16], time_embed_dim=8)
    return model


def _points():
    x = torch.randn(10, 2, generator=torch.Generator().manual_seed(2))
    return [(x[:4], torch.zeros(4)), (x[4:7], torch.zeros(3)), (x[7:], torch.zeros(3))]


def test_diffusion_two_epochs_equal_one_plus_a_stateful_resume():
    schedule = NoiseSchedulers.COSINE(T=40)
    model_a = _diffusion()
    objective_a = diffusion_objective(schedule)  # seeded from the global RNG at first use
    model_a.train(_params(_points(), 2, seed=4), objective=objective_a)

    model_b = _diffusion()
    first = model_b.train(_params(_points(), 1, seed=4), objective=diffusion_objective(schedule))
    model_c = _diffusion(seed=77)
    objective_c = diffusion_objective(schedule)
    second = model_c.train(_params(_points(), 1, seed=4, resume_from_run_id=first.id), objective=objective_c)

    assert second.resume_status is not None and "diffusion.objective" in second.resume_status.restored_components
    for key, value in model_a.net.state_dict().items():
        torch.testing.assert_close(model_c.net.state_dict()[key], value, msg=key)
    assert torch.equal(objective_c.generator.get_state(), objective_a.generator.get_state())


def test_a_diffusion_resume_with_another_schedule_is_rejected():
    model = _diffusion()
    first = model.train(_params(_points(), 1), objective=diffusion_objective(NoiseSchedulers.LINEAR(T=40), seed=0))
    fresh = _diffusion(seed=3)
    before = copy.deepcopy(fresh.net.state_dict())
    with pytest.raises(ComponentRestoreError, match="resume with the same schedule"):
        fresh.train(
            _params(_points(), 1, resume_from_run_id=first.id),
            objective=diffusion_objective(NoiseSchedulers.COSINE(T=40), seed=0),
        )
    assert all(torch.equal(fresh.net.state_dict()[k], v) for k, v in before.items())
