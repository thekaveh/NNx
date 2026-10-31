"""Tests for the reverse-diffusion sampler.

Verifies the sampler runs, returns the requested shape, and produces
finite samples that — after enough training — are closer to the source
distribution than random noise.
"""

from __future__ import annotations

import copy

import pytest
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
from tests._adapter_outputs import REMEDY, DictDenoiser, Extract


def _trained_diffusion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    centers = torch.tensor([[-2, 2], [2, -2]], dtype=torch.float32)
    idx = torch.randint(0, 2, (256,))
    X = centers[idx] + 0.2 * torch.randn(256, 2)
    loader = DataLoader(
        TensorDataset(X, torch.zeros(256, dtype=torch.long)),
        batch_size=64,
        shuffle=True,
    )

    m = NNModel(
        net_params=NNParams(
            input_dim=2,
            output_dim=2,
            hidden_dims=[16],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )
    m.net = DiffusionMLP(input_dim=2, hidden_dims=[64, 64], time_embed_dim=16).to(m.device)

    schedule = NoiseSchedulers.LINEAR(T=100)
    m.train(
        params=NNTrainParams(
            n_epochs=6,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=2e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=2,
                cooldown=1,
                threshold=1e-3,
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


# --- #561: the denoiser's return value through the model's batch adapter ---------------------------------------


def _three_denoisers():
    """One DiffusionMLP's weights three ways: swapped onto a built-in model
    (no batch adapter), wrapped with its default adapter, and inside a
    mapping-returning module read through ``Extract("sample")``."""
    set_seed(0)
    mlp = DiffusionMLP(input_dim=2, hidden_dims=[16], time_embed_dim=8)
    swapped = NNModel(
        net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    swapped.net = copy.deepcopy(mlp)
    wrapped = NNModel(module=copy.deepcopy(mlp), params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR))
    extract = Extract("sample")
    mapped = NNModel(
        module=DictDenoiser(copy.deepcopy(mlp)),
        params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR),
        batch_adapter=extract,
    )
    return (swapped, wrapped, mapped), extract


def test_a_dict_returning_denoiser_samples_through_the_adapters_output():
    """The sampler fed the raw model.net(x, t) into its posterior mean, so a
    denoiser returning a mapping (diffusers' UNet2DOutput) failed there
    (#561). Read through Extract("sample"), it draws exactly the samples the
    same weights returning a tensor draw, with the default adapter or none."""
    schedule = NoiseSchedulers.LINEAR(T=20)
    models, extract = _three_denoisers()
    seeded = [sample(m, schedule, shape=(5, 2), generator=torch.Generator().manual_seed(9)) for m in models]
    drawn = []
    for model in models:  # the global RNG when no generator is given
        set_seed(2)
        drawn.append(sample(model, schedule, shape=(5, 2)))

    assert seeded[2].shape == (5, 2) and torch.isfinite(seeded[2]).all()
    assert torch.equal(seeded[0], seeded[1]) and torch.equal(seeded[0], seeded[2])
    assert torch.equal(drawn[0], drawn[1]) and torch.equal(drawn[0], drawn[2])
    assert extract.grad_modes == [False] * (2 * schedule.T)  # one no-grad forward per reverse step


def test_a_non_tensor_denoiser_output_names_the_adapter_remedy():
    """With the default adapter, a mapping return raises BatchAdapter.output()'s
    TypeError, naming the remedy, not a TypeError from the posterior-mean
    arithmetic; the sampler still restores the net's mode."""
    set_seed(0)
    model = NNModel(
        module=DictDenoiser(DiffusionMLP(input_dim=2, hidden_dims=[16], time_embed_dim=8)),
        params=NNModelParams(loss=Losses.MEAN_SQUARED_ERROR),
    )
    model.net.train()
    with pytest.raises(TypeError, match=REMEDY):
        sample(model, NoiseSchedulers.LINEAR(T=20), shape=(5, 2))
    assert all(module.training for module in model.net.modules())
