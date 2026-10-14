"""Every imperative paradigm step (it steps the optimizer itself) is
refused as ``objective=`` before any run is reserved, naming its
paradigm — not after a run directory exists, on its first batch."""

from __future__ import annotations

import os

import pytest
import torch

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNOptimParams, NNParams, NNTrainParams
from nnx.embeddings import text_contrastive_train_step_factory
from nnx.paradigms import (
    cutmix_train_step_factory,
    dpo_train_step_factory,
    feature_kd_train_step_factory,
    kd_train_step_factory,
    mixup_train_step_factory,
    moe_train_step_factory,
    simclr_train_step_factory,
)


def _model() -> NNModel:
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


FACTORIES = {
    "mixup": lambda: mixup_train_step_factory(),
    "cutmix": lambda: cutmix_train_step_factory(),
    "moe": lambda: moe_train_step_factory(),
    "kd": lambda: kd_train_step_factory(_model()),
    "feature_kd": lambda: feature_kd_train_step_factory(_model(), auxiliary_layers={"layers.0": "layers.0"}),
    "dpo": lambda: dpo_train_step_factory(_model()),
    "contrastive": lambda: simclr_train_step_factory(),
    "text_contrastive": lambda: text_contrastive_train_step_factory(),
}


@pytest.mark.parametrize("paradigm", sorted(FACTORIES))
def test_an_imperative_paradigm_step_is_refused_as_an_objective_before_any_run(paradigm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = NNTrainParams(
        n_epochs=1,
        train_loader=[(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))],
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
    )
    with pytest.raises(ValueError, match=f"imperative {paradigm} step"):
        _model().train(params, objective=FACTORIES[paradigm]())
    assert not os.path.exists("runs") or not [n for n in os.listdir("runs") if not n.startswith(".")]
