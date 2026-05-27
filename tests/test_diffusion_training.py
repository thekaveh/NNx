"""End-to-end test for the diffusion training step factory.

Exercises the full path: build a schedule, build the network, build the
step fn, drive it through NNModel.train(), and verify the loss is finite
and decreases as training proceeds. Saves into a tmp runs dir so the
test doesn't pollute the repo.
"""
from __future__ import annotations

import os

import torch
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
    set_seed,
)

os.environ.setdefault("NNX_TQDM_DISABLE", "1")


def _make_model() -> NNModel:
    """An NNModel with placeholder NNParams, whose .net is swapped for
    a DiffusionMLP. The placeholder mirrors DiffusionMLP's input dim so
    the run.yaml stays interpretable."""
    m = NNModel(
        net_params=NNParams(
            input_dim=2, output_dim=2, hidden_dims=[16],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    m.net = DiffusionMLP(input_dim=2, hidden_dims=[32, 32], time_embed_dim=16).to(m.device)
    return m


def _gaussian_mixture_loader(n: int = 256, batch_size: int = 64) -> DataLoader:
    """2D mixture of 4 isotropic Gaussians at the corners of a square.
    Small + multimodal — diffusion has to learn the structure, not just
    fit a single mean."""
    torch.manual_seed(0)
    centers = torch.tensor([[-2, -2], [-2, 2], [2, -2], [2, 2]], dtype=torch.float32)
    idx = torch.randint(0, 4, (n,))
    means = centers[idx]
    X = means + 0.3 * torch.randn(n, 2)
    y_dummy = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(X, y_dummy), batch_size=batch_size, shuffle=True)


def test_diffusion_train_step_runs_and_loss_decreases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model = _make_model()
    loader = _gaussian_mixture_loader()
    schedule = NoiseSchedulers.LINEAR(T=100)

    step_fn = diffusion_train_step_factory(schedule)
    run = model.train(
        params=NNTrainParams(
            n_epochs=4,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3,
            ),
        ),
        train_step_fn=step_fn,
    )

    assert run.idps is not None and len(run.idps) > 0
    losses = [idp.train_edp.loss for idp in run.idps]
    # All losses finite — no NaN/Inf from a misbroadcast schedule.
    for lo in losses:
        assert lo is not None
        assert lo == lo  # NaN-safe finiteness check
        assert abs(lo) < 1e9
    # Loss should decrease in expectation; we average the first vs last
    # quarter of iterations because per-step loss is noisy (each batch
    # samples a fresh random timestep).
    n = len(losses)
    early = sum(losses[: n // 4]) / max(1, n // 4)
    late = sum(losses[3 * n // 4:]) / max(1, n - 3 * n // 4)
    assert late < early, (
        f"diffusion loss did not decrease across training: "
        f"early-quarter mean {early:.4f} vs late-quarter mean {late:.4f}"
    )


def test_diffusion_train_step_cosine_schedule(tmp_path, monkeypatch):
    """COSINE schedule should produce a working training step just like
    LINEAR — the factory is schedule-agnostic. Mirrors LINEAR's
    loss-decreases check rather than just asserting "ran without error"."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model = _make_model()
    loader = _gaussian_mixture_loader()
    schedule = NoiseSchedulers.COSINE(T=100)

    run = model.train(
        params=NNTrainParams(
            n_epochs=4,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3,
            ),
        ),
        train_step_fn=diffusion_train_step_factory(schedule),
    )
    losses = [idp.train_edp.loss for idp in run.idps]
    assert len(losses) > 0
    for lo in losses:
        assert lo is not None and torch.isfinite(torch.tensor(lo)).item()
    n = len(losses)
    early = sum(losses[: n // 4]) / max(1, n // 4)
    late = sum(losses[3 * n // 4:]) / max(1, n - 3 * n // 4)
    assert late < early, (
        f"diffusion COSINE loss did not decrease: early {early:.4f} vs late {late:.4f}"
    )


def test_diffusion_step_reports_loss_and_error_equal(tmp_path, monkeypatch):
    """Documented invariant in diffusion/training.py: train_edp.loss and
    train_edp.error must be set to the same value so BEST tracking and
    ReduceLROnPlateau have a metric. Without this, the scheduler falls
    back through edp.error → edp.loss and works anyway, but the contract
    deserves a regression test — a future refactor could break it silently."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _make_model()
    schedule = NoiseSchedulers.LINEAR(T=50)
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=_gaussian_mixture_loader(n=32, batch_size=8),
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3,
            ),
        ),
        train_step_fn=diffusion_train_step_factory(schedule),
    )
    for idp in run.idps:
        assert idp.train_edp.loss == idp.train_edp.error, (
            f"diffusion step contract violated: loss={idp.train_edp.loss} "
            f"!= error={idp.train_edp.error}"
        )
