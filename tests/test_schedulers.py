"""Tests for the Schedulers enum factory."""
from __future__ import annotations

import torch
from torch.optim import SGD, lr_scheduler

from nnx.nn.enum.schedulers import Schedulers
from nnx.nn.params.nn_scheduler_params import NNSchedulerParams


def _make_optimizer():
    params = [torch.nn.Parameter(torch.randn(2, 2))]
    return SGD(params, lr=1e-2)


def _base_params(**overrides):
    return NNSchedulerParams(
        min_lr=1e-7, factor=0.5, patience=5, cooldown=2, threshold=1e-3,
        **overrides,
    )


def test_reduce_lr_on_plateau():
    opt = _make_optimizer()
    sched = Schedulers.REDUCE_LR_ON_PLATEAU(opt, _base_params(), n_epochs=10)
    assert isinstance(sched, lr_scheduler.ReduceLROnPlateau)


def test_step_scheduler():
    opt = _make_optimizer()
    sched = Schedulers.STEP(opt, _base_params(step_size=3), n_epochs=10)
    assert isinstance(sched, lr_scheduler.StepLR)
    assert sched.step_size == 3


def test_cosine_annealing():
    opt = _make_optimizer()
    sched = Schedulers.COSINE_ANNEALING(opt, _base_params(T_max=10), n_epochs=10)
    assert isinstance(sched, lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 10


def test_one_cycle():
    opt = _make_optimizer()
    sched = Schedulers.ONE_CYCLE(opt, _base_params(max_lr=1e-1, total_steps=50), n_epochs=50)
    assert isinstance(sched, lr_scheduler.OneCycleLR)


def test_linear_warmup_decay():
    opt = _make_optimizer()
    sched = Schedulers.LINEAR_WARMUP_DECAY(
        opt, _base_params(warmup_steps=5, total_steps=20), n_epochs=20,
    )
    assert isinstance(sched, lr_scheduler.LambdaLR)
    # PyTorch (since 1.1) wants optimizer.step before lr_scheduler.step.
    # Step optimizer once with a fake loss so the warning doesn't fire.
    opt.zero_grad()
    for p in opt.param_groups[0]["params"]:
        p.grad = torch.zeros_like(p)
    opt.step()

    sched.step()
    lr_after_step1 = opt.param_groups[0]["lr"]
    # After warmup, LR should be at or near base_lr
    for _ in range(5):
        sched.step()
    lr_after_warmup = opt.param_groups[0]["lr"]
    assert lr_after_warmup > lr_after_step1


def test_scheduler_params_state_round_trip():
    """state() → from_state() preserves kind and kind-specific fields."""
    original = NNSchedulerParams(
        min_lr=1e-7, factor=0.5, patience=5, cooldown=2, threshold=1e-3,
        kind=Schedulers.COSINE_ANNEALING, T_max=100,
    )
    reconstructed = NNSchedulerParams.from_state(original.state())
    assert reconstructed.kind == Schedulers.COSINE_ANNEALING
    assert reconstructed.T_max == 100


def test_scheduler_params_backwards_compat_no_kind():
    """A params object without `kind` set deserializes as None — preserves
    pre-Schedulers behavior (ReduceLROnPlateau in train())."""
    p = NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=5, cooldown=2, threshold=1e-3)
    state = p.state()
    assert state["kind"] is None
    reconstructed = NNSchedulerParams.from_state(state)
    assert reconstructed.kind is None
