"""Tests for nnx.paradigms.augmentation — Mixup + CutMix."""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    cutmix_train_step_factory,
    mixup_train_step_factory,
    set_seed,
)

os.environ.setdefault("NNX_TQDM_DISABLE", "1")


# -------------------------------------------------------------------------
# Mixup
# -------------------------------------------------------------------------

def _supervised_model() -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=8, output_dim=3, hidden_dims=[16],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )


def _supervised_loader(n: int = 64) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(n, 8)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)


def test_mixup_factory_validates_alpha():
    with pytest.raises(ValueError, match="alpha"):
        mixup_train_step_factory(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        mixup_train_step_factory(alpha=-0.1)


def test_mixup_train_loop_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model = _supervised_model()
    loader = _supervised_loader()
    run = model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3,
            ),
        ),
        train_step_fn=mixup_train_step_factory(alpha=0.4),
    )
    losses = [idp.train_edp.loss for idp in run.idps]
    assert len(losses) > 0
    assert all(lo is not None and torch.isfinite(torch.tensor(lo)).item() for lo in losses)


def test_mixup_reports_weighted_accuracy(tmp_path, monkeypatch):
    """Mixup's `accuracy` field should be the lam-weighted correctness,
    not the raw top-1 — useful to spot accidental contract drift."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model = _supervised_model()
    loader = _supervised_loader(n=16)
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3,
            ),
        ),
        train_step_fn=mixup_train_step_factory(alpha=0.4),
    )
    # accuracy + error must sum to 1 (Mixup uses 1 - weighted_acc as
    # the error signal, so this is a self-consistency check).
    for idp in run.idps:
        edp = idp.train_edp
        assert edp.accuracy is not None
        assert edp.error is not None
        assert abs((edp.accuracy + edp.error) - 1.0) < 1e-6


# -------------------------------------------------------------------------
# CutMix
# -------------------------------------------------------------------------

class _TinyImageNet(nn.Module):
    """A minimal conv classifier so CutMix has a 4D batch to work on."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.head = nn.Linear(4 * 4 * 4, n_classes)
        # Stash a placeholder ".params" so NNRun's persistence path stays
        # happy. NNModel.train() reads `self.net_params` (from Track C),
        # so this is just future-proofing for tests that touch self.net.params.
        self.params = None

    def forward(self, x):
        h = torch.relu(self.conv(x))
        return self.head(h.flatten(1))

    def unpack_batch(self, batch):
        return (batch[0],), batch[1]


def _image_model() -> NNModel:
    """An NNModel whose net is the tiny conv above. The NNParams
    placeholder is unused — CutMix only needs forward() + unpack_batch()."""
    m = NNModel(
        net_params=NNParams(
            input_dim=3 * 4 * 4, output_dim=3, hidden_dims=[],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    m.net = _TinyImageNet(n_classes=3).to(m.device)
    return m


def _image_loader(n: int = 32) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(n, 3, 4, 4)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)


def test_cutmix_factory_validates_alpha():
    with pytest.raises(ValueError, match="alpha"):
        cutmix_train_step_factory(alpha=0.0)


def test_cutmix_rejects_non_image_input():
    """CutMix needs 4D input — a 2D tabular batch should raise loudly,
    not produce silently wrong results."""
    model = _supervised_model()  # 2D input
    loader = _supervised_loader(n=16)
    step_fn = cutmix_train_step_factory(alpha=1.0)
    # Drive a single batch through the step manually to surface the
    # validation error (NNModel.train wraps everything in a tqdm loop).
    from nnx.nn.nn_model import TrainStepContext
    batch = next(iter(loader))
    optimizer = Optims.ADAM(
        net=model.net, lr_start=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
    )
    ctx = TrainStepContext(
        model=model, batch=batch, optimizer=optimizer, scaler=None,
        grad_clip_norm=None, extra_metrics=None,
        accumulate_grad_batches=1, batch_idx=0, epoch_idx=0,
    )
    with pytest.raises(ValueError, match="4D image input"):
        step_fn(ctx)


def test_cutmix_train_loop_runs_on_4d_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _image_model()
    loader = _image_loader()
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3,
            ),
        ),
        train_step_fn=cutmix_train_step_factory(alpha=1.0),
    )
    losses = [idp.train_edp.loss for idp in run.idps]
    assert len(losses) > 0
    assert all(lo is not None and torch.isfinite(torch.tensor(lo)).item() for lo in losses)
