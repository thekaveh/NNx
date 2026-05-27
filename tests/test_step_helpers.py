"""Tests for the shared finalize_step helper used by paradigm factories.

Verifies the contract: NaN-guard, grad-clip honoring, explicit rejection
of unsupported NNOptimParams knobs (AMP, gradient accumulation).
"""
from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    DiffusionMLP,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    NoiseSchedulers,
    Optims,
    diffusion_train_step_factory,
    mixup_train_step_factory,
    set_seed,
)
from nnx._step_helpers import finalize_step
from nnx.nn.nn_model import TrainStepContext

os.environ.setdefault("NNX_TQDM_DISABLE", "1")


def _model_and_optim():
    m = NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    opt = Optims.ADAM(
        net=m.net, lr_start=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
    )
    return m, opt


def _ctx(model, optim, *, scaler=None, grad_clip=None, accumulate=1):
    return TrainStepContext(
        model=model, batch=None, optimizer=optim, scaler=scaler,
        grad_clip_norm=grad_clip, extra_metrics=None,
        accumulate_grad_batches=accumulate, batch_idx=0, epoch_idx=0,
    )


def test_finalize_step_raises_on_amp_scaler():
    """ctx.scaler != None → ValueError. Paradigm factories don't handle AMP;
    silently dropping the user's mixed_precision=True setting would be worse."""
    m, opt = _model_and_optim()
    # Build a dummy loss that depends on m.net so backward() has gradients.
    X = torch.randn(2, 4)
    loss = F.cross_entropy(m.net(X), torch.zeros(2, dtype=torch.long))
    fake_scaler = object()  # any non-None sentinel triggers the check
    with pytest.raises(ValueError, match="mixed precision"):
        finalize_step(loss, _ctx(m, opt, scaler=fake_scaler), paradigm="testparadigm")


def test_finalize_step_raises_on_gradient_accumulation():
    """ctx.accumulate_grad_batches != 1 → ValueError."""
    m, opt = _model_and_optim()
    X = torch.randn(2, 4)
    loss = F.cross_entropy(m.net(X), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="gradient accumulation"):
        finalize_step(loss, _ctx(m, opt, accumulate=4), paradigm="testparadigm")


def test_finalize_step_raises_on_non_finite_loss():
    """Non-finite loss must raise loudly — silent divergence corrupts checkpoints."""
    m, opt = _model_and_optim()
    # NaN loss tensor (still requires_grad so the backward() call before
    # the check completes without error).
    X = torch.randn(2, 4, requires_grad=True)
    loss = (m.net(X).sum() * float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        finalize_step(loss, _ctx(m, opt), paradigm="testparadigm")


def test_finalize_step_honors_grad_clip_norm():
    """grad_clip_norm in the ctx must actually clip gradients."""
    m, opt = _model_and_optim()
    X = torch.randn(2, 4)
    loss = F.cross_entropy(m.net(X) * 1e6, torch.zeros(2, dtype=torch.long))  # huge gradient
    finalize_step(loss, _ctx(m, opt, grad_clip=1.0), paradigm="testparadigm")
    # After clipping, every parameter's grad norm should be ≤ 1.0 globally.
    total_norm = math.sqrt(sum(
        float(p.grad.detach().norm(2)) ** 2 for p in m.net.parameters() if p.grad is not None
    ))
    assert total_norm <= 1.0 + 1e-4, f"grad_clip_norm=1.0 not honored; got {total_norm}"


def test_finalize_step_returns_loss_float():
    """Return value should be the detached float for use in EDP."""
    m, opt = _model_and_optim()
    X = torch.randn(2, 4)
    loss = F.cross_entropy(m.net(X), torch.zeros(2, dtype=torch.long))
    val = finalize_step(loss, _ctx(m, opt), paradigm="testparadigm")
    assert isinstance(val, float)
    assert math.isfinite(val)


# -------------------------------------------------------------------------
# End-to-end: paradigm factories surface the helper's error correctly
# -------------------------------------------------------------------------

def test_diffusion_factory_rejects_mixed_precision(tmp_path, monkeypatch):
    """The diffusion step inherits the finalize_step contract — if AMP
    is on, the train() call should fail with the helper's clear error
    rather than silently running un-scaled."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    m = NNModel(
        net_params=NNParams(
            input_dim=2, output_dim=2, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        # mixed_precision=True forces ctx.scaler to be non-None on CUDA.
        # On CPU, the scaler is None — so we simulate by patching the
        # _build_grad_scaler method to return a sentinel.
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    # Patch _build_grad_scaler to force a non-None scaler so the AMP path
    # fires on CPU (no actual scaling happens; we just want the rejection).
    m._build_grad_scaler = lambda: object()

    m.net = DiffusionMLP(input_dim=2, hidden_dims=[8], time_embed_dim=8).to(m.device)

    loader = DataLoader(
        TensorDataset(torch.randn(8, 2), torch.zeros(8, dtype=torch.long)),
        batch_size=4, shuffle=False,
    )
    schedule = NoiseSchedulers.LINEAR(T=10)
    with pytest.raises(ValueError, match="mixed precision"):
        m.train(
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
            train_step_fn=diffusion_train_step_factory(schedule),
        )


def test_mixup_factory_rejects_gradient_accumulation(tmp_path, monkeypatch):
    """Setting NNOptimParams.accumulate_grad_batches > 1 with a Mixup
    step must raise rather than silently dropping the knob."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    m = NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    loader = DataLoader(
        TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,))),
        batch_size=8, shuffle=False,
    )
    with pytest.raises(ValueError, match="gradient accumulation"):
        m.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=loader,
                optim=NNOptimParams(
                    name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
                    accumulate_grad_batches=4,
                ),
                scheduler=NNSchedulerParams(
                    min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3,
                ),
            ),
            train_step_fn=mixup_train_step_factory(alpha=0.4),
        )
