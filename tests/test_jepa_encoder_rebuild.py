"""FEAT-047: a JEPA-trained encoder rebuilds without its predictor through
every rebuild path — a run bundle (``reconstruct_bundle``) and a Hugging Face
Hub save (``from_pretrained``) as well as ``from_checkpoint`` (#395) — and
each path's error names its own option."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Devices,
    JEPAPredictor,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNTrainParams,
    NNViTParams,
    ViTNN,
    build_target_encoder,
    jepa_train_step_factory,
    random_block_mask,
)
from nnx.bundles import BundleCapabilityError, export_bundle, reconstruct_bundle

PREDICTOR = "_jepa_predictor"
PROBE = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(5))


@pytest.fixture
def jepa(tmp_path, monkeypatch):
    """A one-epoch JEPA run whose predictor is registered under model.net, as
    the documented recipe does."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    params = NNViTParams(
        input_dim=3 * 8 * 8, output_dim=16, dropout_prob=0.0, image_size=8, patch_size=4, d_model=16, n_layers=2,
        n_heads=2,
    )  # fmt: skip
    model = NNModel(
        net_params=params, params=NNModelParams(net=Nets.VIT, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR)
    )
    target = build_target_encoder(model.net)
    predictor = JEPAPredictor(embed_dim=16, n_patches=model.net.n_patches, predictor_dim=8, n_layers=1, n_heads=2)
    model.net.add_module(PREDICTOR, predictor)
    images = torch.randn(8, 3, 8, 8)
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=DataLoader(TensorDataset(images, torch.zeros(8, dtype=torch.long)), batch_size=4),
            optim=NNOptimParams.builder().adam(max_lr=1e-3).build(),
        ),
        train_step_fn=jepa_train_step_factory(
            target_encoder=target,
            predictor=predictor,
            mask_fn=lambda n, device: random_block_mask(n_patches=n, grid_size=2, device=device),
            ema_momentum=0.99,
        ),
    )
    return model, run


def _assert_same_encoder(rebuilt: NNModel, trained: NNModel) -> None:
    assert isinstance(rebuilt.net, ViTNN) and not hasattr(rebuilt.net, PREDICTOR)
    trained_state = {k: v for k, v in trained.net.state_dict().items() if not k.startswith(f"{PREDICTOR}.")}
    rebuilt_state = rebuilt.net.state_dict()
    assert rebuilt_state.keys() == trained_state.keys()
    for name, weight in trained_state.items():
        assert torch.equal(rebuilt_state[name], weight), name
    trained.net.eval()
    rebuilt.net.eval()
    with torch.no_grad():
        assert torch.equal(rebuilt.net(PROBE), trained.net(PROBE))


# --- run bundles -------------------------------------------------------------------------------


def test_a_jepa_run_bundle_rebuilds_its_encoder_bit_exact(jepa, tmp_path):
    model, run = jepa
    bundle = tmp_path / "bundle"
    export_bundle(run.id, bundle)
    with pytest.raises(RuntimeError) as caught:
        reconstruct_bundle(bundle)
    message = str(caught.value)
    assert "reconstruct_bundle(path, exclude_submodules=('_jepa_predictor',))" in message
    assert "from_checkpoint" not in message  # the option of the entry point the caller used
    with pytest.raises(ValueError, match="do not hold"):
        reconstruct_bundle(bundle, exclude_submodules=("_jepa_predicter",))
    rebuilt = reconstruct_bundle(bundle, exclude_submodules=(PREDICTOR,))
    _assert_same_encoder(rebuilt.model, model)


def test_an_encoder_rebuilt_without_its_predictor_does_not_resume(jepa, tmp_path):
    _, run = jepa
    bundle = tmp_path / "bundle"
    export_bundle(run.id, bundle)
    rebuilt = reconstruct_bundle(bundle, exclude_submodules=(PREDICTOR,))
    loader = DataLoader(TensorDataset(torch.randn(4, 3, 8, 8), torch.zeros(4, dtype=torch.long)), batch_size=4)
    with pytest.raises(BundleCapabilityError, match="_jepa_predictor.*inference only"):
        rebuilt.resume(NNTrainParams(n_epochs=1, train_loader=loader))


# --- Hugging Face Hub --------------------------------------------------------------------------


def test_a_hub_save_with_the_predictor_attached_rebuilds_its_encoder(jepa, tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("safetensors")
    model, _ = jepa
    saved = tmp_path / "hub"
    model.save_pretrained(str(saved))
    with pytest.raises(RuntimeError) as caught:
        NNModel.from_pretrained(str(saved))
    message = str(caught.value)
    assert "from_pretrained(model_id, exclude_submodules=('_jepa_predictor',))" in message
    assert "from_checkpoint" not in message
    with pytest.raises(ValueError, match="do not hold"):
        NNModel.from_pretrained(str(saved), exclude_submodules=("_jepa_predicter",))
    _assert_same_encoder(NNModel.from_pretrained(str(saved), exclude_submodules=(PREDICTOR,)), model)


def test_hub_exclusion_stays_strict_for_every_other_key(jepa, tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file, save_file

    model, _ = jepa
    saved = tmp_path / "hub"
    model.save_pretrained(str(saved))
    weights = os.path.join(saved, "model.safetensors")
    state = load_file(weights)
    dropped = next(key for key in state if not key.startswith(f"{PREDICTOR}."))
    del state[dropped]
    save_file(state, weights)
    with pytest.raises(RuntimeError, match="Missing key"):
        NNModel.from_pretrained(str(saved), exclude_submodules=(PREDICTOR,))


# --- a registered module (its state schema is checked before the load) ---------------------------


class _Encoder(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = torch.nn.Linear(4, width)

    def forward(self, x):
        return self.body(x)


def _encoder_factory(config):
    return _Encoder(**config)


@pytest.fixture
def registered(tmp_path, monkeypatch):
    """A registered module trained with two submodules attached under
    ``model.net`` that its factory does not build."""
    from nnx.models import ModelSpec, register_model_factory, unregister_model_factory

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    register_model_factory("tests.jepa_encoder", 1, _encoder_factory)
    try:
        torch.manual_seed(0)
        spec = ModelSpec("tests.jepa_encoder", 1, {"width": 3}, seed=5)
        model = NNModel(params=NNModelParams(net=spec, device=Devices.CPU, loss=Losses.CROSS_ENTROPY))
        model.net.add_module(PREDICTOR, torch.nn.Linear(3, 3))
        model.net.add_module("_extra", torch.nn.Linear(3, 3))
        run = model.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=DataLoader(TensorDataset(torch.randn(8, 4), torch.arange(8) % 3), batch_size=4),
                optim=NNOptimParams.builder().adam(max_lr=1e-3).build(),
            )
        )
        yield model, run
    finally:
        unregister_model_factory("tests.jepa_encoder", 1)


def _same_body(rebuilt: NNModel, trained: NNModel) -> None:
    assert set(rebuilt.net.state_dict()) == {"body.weight", "body.bias"}
    for name, weight in rebuilt.net.state_dict().items():
        assert torch.equal(weight, trained.net.state_dict()[name]), name


def test_a_registered_module_bundle_names_the_option_and_merges_names_already_passed(registered, tmp_path):
    model, run = registered
    export_bundle(run.id, tmp_path / "bundle")
    with pytest.raises(ValueError) as caught:  # the schema check, before any weight is loaded
        reconstruct_bundle(tmp_path / "bundle")
    assert "reconstruct_bundle(path, exclude_submodules=('_extra', '_jepa_predictor'))" in str(caught.value)
    with pytest.raises(ValueError) as caught:
        reconstruct_bundle(tmp_path / "bundle", exclude_submodules=[PREDICTOR])
    assert "exclude_submodules=('_jepa_predictor', '_extra')" in str(caught.value)  # pasteable as is
    _same_body(reconstruct_bundle(tmp_path / "bundle", exclude_submodules=iter((PREDICTOR, "_extra"))).model, model)
    with pytest.raises(TypeError, match="not a string"):
        reconstruct_bundle(tmp_path / "bundle", exclude_submodules=PREDICTOR)  # type: ignore[arg-type]


def test_a_registered_module_hub_save_names_the_option(registered, tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("safetensors")
    model, _ = registered
    model.save_pretrained(str(tmp_path / "hub"))
    with pytest.raises(ValueError) as caught:
        NNModel.from_pretrained(str(tmp_path / "hub"))
    assert "from_pretrained(model_id, exclude_submodules=('_extra', '_jepa_predictor'))" in str(caught.value)
    _same_body(NNModel.from_pretrained(str(tmp_path / "hub"), exclude_submodules=(PREDICTOR, "_extra")), model)


def test_a_factory_loading_a_checkpoint_of_its_own_gets_the_from_checkpoint_hint(jepa, tmp_path):
    """The bundle's hint names reconstruct_bundle only for the bundle's own load."""
    from nnx.models import ModelSpec, register_model_factory, unregister_model_factory
    from nnx.nn.enum.checkpoints import Checkpoints
    from nnx.nn.params.nn_checkpoint import NNCheckpoint

    _, run = jepa
    inner = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    seen: list[str] = []

    def factory(config):
        try:
            NNModel.from_checkpoint(inner)
        except RuntimeError as error:
            seen.append(str(error))
        return _Encoder(**config)

    register_model_factory("tests.nested", 1, factory)
    try:
        spec = ModelSpec("tests.nested", 1, {"width": 3}, seed=5)
        outer = NNModel(params=NNModelParams(net=spec, device=Devices.CPU, loss=Losses.CROSS_ENTROPY))
        outer_run = outer.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=DataLoader(TensorDataset(torch.randn(4, 4), torch.arange(4) % 3), batch_size=4),
            )
        )
        export_bundle(outer_run.id, tmp_path / "nested")
        seen.clear()
        reconstruct_bundle(tmp_path / "nested")
    finally:
        unregister_model_factory("tests.nested", 1)
    assert seen and all("NNModel.from_checkpoint(checkpoint, exclude_submodules=" in message for message in seen)
