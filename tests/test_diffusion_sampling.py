"""Tests for the reverse-diffusion sampler.

Verifies the sampler runs, returns the requested shape, and produces
finite samples that — after enough training — are closer to the source
distribution than random noise.
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
    sample,
    set_seed,
)

os.environ.setdefault("NNX_TQDM_DISABLE", "1")


def _trained_diffusion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    centers = torch.tensor([[-2, 2], [2, -2]], dtype=torch.float32)
    idx = torch.randint(0, 2, (256,))
    X = centers[idx] + 0.2 * torch.randn(256, 2)
    loader = DataLoader(
        TensorDataset(X, torch.zeros(256, dtype=torch.long)),
        batch_size=64, shuffle=True,
    )

    m = NNModel(
        net_params=NNParams(
            input_dim=2, output_dim=2, hidden_dims=[16],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    m.net = DiffusionMLP(input_dim=2, hidden_dims=[64, 64], time_embed_dim=16).to(m.device)

    schedule = NoiseSchedulers.LINEAR(T=100)
    m.train(
        params=NNTrainParams(
            n_epochs=6,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=2e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3,
            ),
        ),
        train_step_fn=diffusion_train_step_factory(schedule),
    )
    return m, schedule


def test_sample_returns_requested_shape(tmp_path, monkeypatch):
    m, schedule = _trained_diffusion(tmp_path, monkeypatch)
    out = sample(m, schedule, shape=(16, 2))
    assert out.shape == (16, 2)


def test_sample_produces_finite_values(tmp_path, monkeypatch):
    m, schedule = _trained_diffusion(tmp_path, monkeypatch)
    out = sample(m, schedule, shape=(32, 2))
    assert torch.isfinite(out).all()


def test_sample_is_reproducible_with_generator(tmp_path, monkeypatch):
    """Passing the same Generator should produce the same samples — required
    for reproducible eval / visualization in notebooks."""
    m, schedule = _trained_diffusion(tmp_path, monkeypatch)

    g1 = torch.Generator(device="cpu").manual_seed(42)
    g2 = torch.Generator(device="cpu").manual_seed(42)
    out1 = sample(m, schedule, shape=(8, 2), generator=g1)
    out2 = sample(m, schedule, shape=(8, 2), generator=g2)
    assert torch.equal(out1, out2)


def test_sample_closer_to_data_than_pure_noise(tmp_path, monkeypatch):
    """A trained model's samples should land closer to the source
    distribution (two clusters at (-2, 2) and (2, -2)) than pure
    Gaussian noise. We measure 'closer' by the minimum Euclidean
    distance from each sample to either cluster center."""
    m, schedule = _trained_diffusion(tmp_path, monkeypatch)

    out = sample(m, schedule, shape=(64, 2))
    centers = torch.tensor([[-2.0, 2.0], [2.0, -2.0]])
    # Distances to nearest center for both the model's samples and
    # i.i.d. Gaussian noise.
    def min_dist(pts):
        d = torch.cdist(pts, centers).min(dim=1).values
        return float(d.mean())

    trained_dist = min_dist(out)
    noise_dist = min_dist(torch.randn(64, 2))

    assert trained_dist < noise_dist, (
        f"trained samples are no closer to the modes than pure noise: "
        f"trained {trained_dist:.3f} vs noise {noise_dist:.3f}"
    )
