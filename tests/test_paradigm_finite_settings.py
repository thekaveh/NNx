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
