"""Paradigm settings refuse NaN and infinity at the boundary, before any
training: a NaN clip norm or an infinite temperature would otherwise train
silently wrong (all-NaN weights, a zero-gradient loss)."""

from __future__ import annotations

import pytest
import torch

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
    cutmix_train_step_factory,
    feature_kd_train_step_factory,
    kd_train_step_factory,
    mixup_train_step_factory,
    moe_train_step_factory,
    nt_xent_loss,
    simclr_train_step_factory,
)

BAD = [float("nan"), float("inf"), float("-inf")]


def _teacher() -> NNModel:
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize(
    "build",
    [
        lambda v: nt_xent_loss(torch.randn(2, 3), torch.randn(2, 3), temperature=v),
        lambda v: simclr_train_step_factory(temperature=v),
        lambda v: kd_train_step_factory(_teacher(), temperature=v),
        lambda v: feature_kd_train_step_factory(_teacher(), auxiliary_layers={"layers.0": "layers.0"}, temperature=v),
        lambda v: moe_train_step_factory(aux_loss_weight=v),
        lambda v: mixup_train_step_factory(alpha=v),
        lambda v: cutmix_train_step_factory(alpha=v),
    ],
    ids=["nt_xent", "simclr", "kd", "feature_kd", "moe", "mixup", "cutmix"],
)
def test_a_non_finite_setting_is_refused(build, bad):
    with pytest.raises(ValueError, match="finite"):
        build(bad)


@pytest.mark.parametrize("bad", BAD)
def test_text_contrastive_settings_refuse_non_finite_values(bad):
    pytest.importorskip("sentence_transformers")
    from nnx.embeddings import text_contrastive_train_step_factory, train_contrastive

    with pytest.raises(ValueError, match="finite"):
        text_contrastive_train_step_factory(temperature=bad)
    pairs = [("a", "b"), ("c", "d")]
    with pytest.raises(ValueError, match="finite"):
        train_contrastive(object(), pairs, batch_size=2, temperature=bad)
    with pytest.raises(ValueError, match="finite"):
        train_contrastive(object(), pairs, batch_size=2, grad_clip_norm=bad)


def _train_params():
    from nnx import NNOptimParams, NNTrainParams

    return NNTrainParams(
        n_epochs=1,
        train_loader=[(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))],
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
    )


def _born_again(**kwargs):
    from nnx import born_again_train

    return born_again_train(_teacher(), train_params=_train_params(), **kwargs)


def _jepa(v):
    from nnx.paradigms import jepa_train_step_factory

    return jepa_train_step_factory(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), lambda n, d: None, ema_momentum=v)


def _update_ema(v):
    from nnx.paradigms import update_ema

    update_ema(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), momentum=v)


def _block_mask(*aspect):
    from nnx.paradigms import random_block_mask

    return random_block_mask(n_patches=16, grid_size=4, block_aspect=aspect)


def _kd_objective(**kwargs):
    from nnx.objectives import kd_objective

    return kd_objective(_teacher(), **kwargs)


# A bool is an int (True == 1), so it passed every bare range comparison and
# trained as 1.0 / 0.0: kd_objective(alpha=True) ran pure distillation.
BOOL_SETTINGS = {
    "kd-alpha": (lambda v: kd_train_step_factory(_teacher(), alpha=v), "alpha"),
    "kd-temperature": (lambda v: kd_train_step_factory(_teacher(), temperature=v), "temperature"),
    "kd_objective-alpha": (lambda v: _kd_objective(alpha=v), "alpha"),
    "kd_objective-temperature": (lambda v: _kd_objective(temperature=v), "temperature"),
    "born_again-alpha": (lambda v: _born_again(generations=2, alpha=v), "alpha"),
    "born_again-generations": (lambda v: _born_again(generations=v), "generations"),
    "feature_kd-alpha": (
        lambda v: feature_kd_train_step_factory(_teacher(), auxiliary_layers={"layers.0": "layers.0"}, alpha=v),
        "alpha",
    ),
    "feature_kd-beta": (
        lambda v: feature_kd_train_step_factory(_teacher(), auxiliary_layers={"layers.0": "layers.0"}, beta=v),
        "beta",
    ),
    "feature_kd-temperature": (
        lambda v: feature_kd_train_step_factory(_teacher(), auxiliary_layers={"layers.0": "layers.0"}, temperature=v),
        "temperature",
    ),
    "mixup-alpha": (lambda v: mixup_train_step_factory(alpha=v), "alpha"),
    "cutmix-alpha": (lambda v: cutmix_train_step_factory(alpha=v), "alpha"),
    "moe-aux_loss_weight": (lambda v: moe_train_step_factory(aux_loss_weight=v), "aux_loss_weight"),
    "simclr-temperature": (lambda v: simclr_train_step_factory(temperature=v), "temperature"),
    "nt_xent-temperature": (
        lambda v: nt_xent_loss(torch.randn(2, 3), torch.randn(2, 3), temperature=v),
        "temperature",
    ),
    "jepa-ema_momentum": (_jepa, "ema_momentum"),
    "update_ema-momentum": (_update_ema, "momentum"),
    "random_block_mask-block_aspect": (lambda v: _block_mask(v, v), "block_aspect"),
    "random_block_mask-block_aspect_max": (lambda v: _block_mask(1.0, v), "block_aspect"),
}


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("name", list(BOOL_SETTINGS))
def test_a_bool_setting_is_refused_naming_the_field(name, flag, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def no_training(*args, **kwargs):
        raise AssertionError("trained before the refusal")

    monkeypatch.setattr(NNModel, "train", no_training)
    build, field = BOOL_SETTINGS[name]
    with pytest.raises(ValueError, match=field) as raised:
        build(flag)
    assert "bool" in str(raised.value)


@pytest.mark.parametrize("flag", [True, False])
def test_text_contrastive_settings_refuse_bool_values(flag):
    pytest.importorskip("sentence_transformers")
    from nnx.embeddings import text_contrastive_train_step_factory, train_contrastive

    with pytest.raises(ValueError, match="temperature.*bool"):
        text_contrastive_train_step_factory(temperature=flag)
    pairs = [("a", "b"), ("c", "d")]
    with pytest.raises(ValueError, match="temperature.*bool"):
        train_contrastive(object(), pairs, batch_size=2, temperature=flag)
    with pytest.raises(ValueError, match="grad_clip_norm.*bool"):
        train_contrastive(object(), pairs, batch_size=2, grad_clip_norm=flag)


def test_integer_settings_are_still_accepted(tmp_path, monkeypatch):
    from nnx.paradigms import update_ema

    monkeypatch.chdir(tmp_path)
    kd_train_step_factory(_teacher(), alpha=1, temperature=4)
    kd_train_step_factory(_teacher(), alpha=0, temperature=4)
    assert _kd_objective(alpha=1, temperature=4).temperature == 4.0
    feature_kd_train_step_factory(_teacher(), auxiliary_layers={"layers.0": "layers.0"}, alpha=1, beta=0, temperature=4)
    mixup_train_step_factory(alpha=1)
    cutmix_train_step_factory(alpha=2)
    moe_train_step_factory(aux_loss_weight=0)
    simclr_train_step_factory(temperature=1)
    assert torch.isfinite(nt_xent_loss(torch.randn(2, 3), torch.randn(2, 3), temperature=1))
    _jepa(0)
    source, target = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    update_ema(source, target, momentum=0)
    assert torch.equal(target.weight, source.weight)
    context, block = _block_mask(1, 2)
    assert block.any() and context.any()
    runs = _born_again(generations=2, alpha=1, temperature=4)
    assert len(runs) == 2


def test_text_contrastive_integer_settings_and_disabled_clipping_are_accepted():
    pytest.importorskip("sentence_transformers")
    from nnx.embeddings import text_contrastive_train_step_factory, train_contrastive

    text_contrastive_train_step_factory(temperature=1)

    class Embedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Embedding(8, 4)

        def forward(self, texts):
            return self.table(torch.tensor([ord(t[0]) % 8 for t in texts]))

    pairs = [("a", "b"), ("c", "d")]
    backbone = Embedder()
    assert train_contrastive(backbone, pairs, n_epochs=1, batch_size=2, temperature=1, grad_clip_norm=None) is backbone
    assert train_contrastive(backbone, pairs, n_epochs=1, batch_size=2, grad_clip_norm=1) is backbone
