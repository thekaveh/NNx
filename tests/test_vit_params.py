"""#395: the public ViT network type — ``Nets.VIT``, ``NNViTParams`` and
checkpoint reconstruction, so a stored ViT run rebuilds through public APIs."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
    NNViTParams,
    ViTNN,
)
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_run import NNRun


def _vit(**overrides) -> NNViTParams:
    fields = dict(
        input_dim=3 * 8 * 8,
        output_dim=16,
        dropout_prob=0.0,
        image_size=8,
        patch_size=4,
        d_model=16,
        n_layers=2,
        n_heads=2,
    )
    fields.update(overrides)
    return NNViTParams(**fields)


def _model(params: NNViTParams) -> NNModel:
    return NNModel(
        net_params=params, params=NNModelParams(net=Nets.VIT, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR)
    )


# --- parameters -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"image_size": 0}, "image_size"),
        ({"patch_size": -4}, "patch_size"),
        ({"d_model": 0, "output_dim": 1}, "d_model"),
        ({"n_layers": 0}, "n_layers"),
        ({"in_channels": 0}, "in_channels"),
        ({"ffn_mult": 0}, "ffn_mult"),
        ({"n_heads": None}, "n_heads"),
        ({"image_size": 10, "input_dim": 300}, "divisible by patch_size"),
        ({"d_model": 18, "output_dim": 18, "n_heads": 4}, "divisible by n_heads"),
        ({"attn_dropout": 1.5}, "attn_dropout"),
        ({"resid_dropout": -0.1}, "resid_dropout"),
        ({"image_size": True}, "image_size"),
        ({"n_layers": 2.5}, "n_layers"),
        ({"input_dim": 100}, "input_dim"),
        ({"output_dim": 10}, "output_dim"),
        ({"dropout_prob": 0.1}, "attn_dropout"),
        ({"hidden_dims": [8]}, "hidden_dims"),
    ],
)
def test_invalid_settings_fail_at_construction(overrides, match):
    with pytest.raises(ValueError, match=match):
        _vit(**overrides)


def test_state_always_carries_the_architecture_and_omits_defaulted_knobs():
    state = _vit().state()
    for key in ("image_size", "patch_size", "d_model", "n_layers", "n_heads"):
        assert key in state
    # Omit-when-default: a vanilla config keeps a stable run id as knobs accrue.
    for key in ("in_channels", "ffn_mult", "attn_dropout", "resid_dropout"):
        assert key not in state
    tuned = _vit(input_dim=1 * 8 * 8, in_channels=1, ffn_mult=2, attn_dropout=0.1, resid_dropout=0.2).state()
    assert (tuned["in_channels"], tuned["ffn_mult"], tuned["attn_dropout"], tuned["resid_dropout"]) == (
        1,
        2,
        0.1,
        0.2,
    )


def test_every_setting_takes_part_in_the_run_identity():
    base = _vit().state()
    for changed in (
        _vit(image_size=12, input_dim=3 * 12 * 12),
        _vit(patch_size=2),
        _vit(d_model=32, output_dim=32),
        _vit(n_layers=3),
        _vit(n_heads=4),
        _vit(input_dim=8 * 8, in_channels=1),
        _vit(ffn_mult=2),
        _vit(attn_dropout=0.1),
        _vit(resid_dropout=0.1),
    ):
        assert changed.state() != base


def test_state_round_trips_and_resolves_to_the_vit_params():
    params = _vit(input_dim=1 * 8 * 8, in_channels=1, ffn_mult=2, attn_dropout=0.1)
    restored = NNParams.resolve_from_state(params.state())
    assert isinstance(restored, NNViTParams) and restored == params
    assert restored.state() == params.state()


def test_existing_states_and_net_values_are_unchanged():
    """Back compatibility: a stored feed-forward config still resolves to the
    base params, and every existing ``Nets`` value keeps its spelling."""
    plain = NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0)
    assert type(NNParams.resolve_from_state(plain.state())) is NNParams
    assert [net.value for net in Nets if net is not Nets.VIT] == [
        "conv",
        "feed_fwd",
        "feed_fwd_moe",
        "graph_att",
        "graph_conv",
        "graph_sage",
        "transformer",
    ]
    assert Nets("vit") is Nets.VIT


# --- construction -----------------------------------------------------------------------------------


def test_nets_vit_builds_the_configured_encoder():
    net = Nets.VIT(_vit(ffn_mult=2))
    assert isinstance(net, ViTNN)
    assert (net.image_size, net.patch_size, net.in_channels, net.d_model, net.n_patches) == (8, 4, 3, 16, 4)
    assert len(net.blocks) == 2
    out = net(torch.randn(2, 3, 8, 8))
    assert out.shape == (2, 5, 16)  # CLS + 4 patches


def test_nets_and_params_must_match():
    with pytest.raises(ValueError, match="NNViTParams"):
        Nets.VIT(NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0))
    with pytest.raises(ValueError, match="Nets.VIT"):
        Nets.FEED_FWD(_vit())


# --- reconstruction ---------------------------------------------------------------------------------


def test_a_trained_vit_rebuilds_from_its_checkpoint_and_run(tmp_path, monkeypatch):
    """Build, train one step, save, rebuild: the same forward on a fixed input,
    through NNModel.from_checkpoint and NNRun.load, with no caller-side swap."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    params = _vit()
    images = torch.randn(4, 3, 8, 8)
    targets = torch.zeros(4, 5, 16)  # regress every token toward zero: one real update
    model = _model(params)
    before = {k: v.clone() for k, v in model.net.state_dict().items()}

    def token_step(ctx):  # an encoder has no labels to score: regress its tokens (as a JEPA step would)
        from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

        net = ctx.model.net
        net.train()
        net.zero_grad()
        x, y = ctx.batch
        loss = torch.nn.functional.mse_loss(net(x), y)
        loss.backward()
        ctx.optimizer.step()
        value = float(loss.detach())
        return NNEvaluationDataPoint(f1=0.0, recall=0.0, accuracy=0.0, precision=0.0, loss=value, error=value)

    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=DataLoader(TensorDataset(images, targets), batch_size=4),
            optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        ),
        train_step_fn=token_step,
    )
    assert any(not torch.equal(before[k], v) for k, v in model.net.state_dict().items())  # it trained

    probe = torch.randn(2, 3, 8, 8)
    model.net.eval()
    with torch.no_grad():
        expected = model.net(probe)

    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert isinstance(checkpoint.net_params, NNViTParams) and checkpoint.net_params == params
    rebuilt = NNModel.from_checkpoint(checkpoint)
    assert isinstance(rebuilt.net, ViTNN)
    rebuilt.net.eval()
    with torch.no_grad():
        assert torch.equal(rebuilt.net(probe), expected)

    stored = NNRun.load(run.id)
    assert isinstance(stored.net, NNViTParams) and stored.net == params
    assert stored.model.net is Nets.VIT


def test_a_jepa_trained_encoder_rebuilds_without_its_predictor(tmp_path, monkeypatch):
    """The documented JEPA recipe registers the predictor under model.net, so
    its checkpoint holds predictor weights the encoder architecture lacks:
    from_checkpoint names them, and exclude_submodules rebuilds the encoder
    alone with its exact trained weights."""
    from nnx import JEPAPredictor, build_target_encoder, jepa_train_step_factory, random_block_mask

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    model = _model(_vit())
    target = build_target_encoder(model.net)
    predictor = JEPAPredictor(embed_dim=16, n_patches=model.net.n_patches, predictor_dim=8, n_layers=1, n_heads=2)
    model.net.add_module("_jepa_predictor", predictor)
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
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert any(key.startswith("_jepa_predictor.") for key in checkpoint.net_state)
    with pytest.raises(RuntimeError, match=r"exclude_submodules=\('_jepa_predictor',\)"):
        NNModel.from_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="do not hold"):
        NNModel.from_checkpoint(checkpoint, exclude_submodules=("_jepa_predicter",))
    encoder = NNModel.from_checkpoint(checkpoint, exclude_submodules=("_jepa_predictor",))
    assert isinstance(encoder.net, ViTNN) and not hasattr(encoder.net, "_jepa_predictor")
    probe = torch.randn(2, 3, 8, 8)
    model.net.eval()
    encoder.net.eval()
    with torch.no_grad():
        assert torch.equal(encoder.net(probe), model.net(probe))


def test_dropout_probabilities_are_normalized_floats():
    import numpy as np
    import yaml

    params = _vit(attn_dropout=np.float32(0.25), resid_dropout=1)
    assert type(params.attn_dropout) is float and type(params.resid_dropout) is float
    assert params.state() == _vit(attn_dropout=0.25, resid_dropout=1.0).state()
    yaml.safe_dump(params.state())
    for bad in (float("nan"), True, "0.1"):
        with pytest.raises(ValueError, match="attn_dropout"):
            _vit(attn_dropout=bad)


def test_a_run_stored_before_nets_vit_still_loads_with_its_id(tmp_path, monkeypatch):
    """A feed-forward run written by NNx before #395 (486ad73) loads unchanged:
    its net params resolve to the base class and re-hash to the stored id."""
    import shutil
    from pathlib import Path

    stored = Path(__file__).parent / "fixtures" / "run_before_vit"
    shutil.copytree(stored, tmp_path / "runs")
    monkeypatch.chdir(tmp_path)
    run = NNRun.load("caccfb2e2e30a6de9234b7cfea8fb070")
    assert type(run.net) is NNParams and run.model.net is Nets.FEED_FWD
    assert run.id == "caccfb2e2e30a6de9234b7cfea8fb070" and len(run.idps) == 1
