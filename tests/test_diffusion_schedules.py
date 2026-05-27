"""Tests for nnx.diffusion.schedules — linear/cosine noise schedules."""
from __future__ import annotations

import math

import pytest
import torch

from nnx import NoiseSchedule, NoiseSchedulers


def test_linear_schedule_shape_and_endpoints():
    s = NoiseSchedulers.LINEAR(T=100, beta_min=1e-4, beta_max=2e-2)
    assert isinstance(s, NoiseSchedule)
    assert s.T == 100
    assert s.kind is NoiseSchedulers.LINEAR
    assert s.betas.shape == (100,)
    assert math.isclose(s.betas[0].item(), 1e-4, rel_tol=1e-5)
    assert math.isclose(s.betas[-1].item(), 2e-2, rel_tol=1e-5)


def test_linear_schedule_monotone_betas():
    """The DDPM linear schedule has strictly increasing β_t."""
    s = NoiseSchedulers.LINEAR(T=200)
    diffs = s.betas[1:] - s.betas[:-1]
    assert (diffs > 0).all()


def test_alphas_cumprod_strictly_decreasing():
    """ᾱ_t is the cumulative product of (1 - β_t), so it strictly
    decreases as t grows — a precondition for the forward process making
    x_t progressively noisier."""
    s = NoiseSchedulers.LINEAR(T=100)
    diffs = s.alphas_cumprod[1:] - s.alphas_cumprod[:-1]
    assert (diffs < 0).all()


def test_alphas_cumprod_at_zero_is_alpha_zero():
    s = NoiseSchedulers.LINEAR(T=50)
    # ᾱ_0 = α_0 = 1 - β_0
    assert torch.isclose(s.alphas_cumprod[0], 1.0 - s.betas[0])


def test_sqrt_consistency():
    """Derived sqrt tensors must equal the analytic sqrt of the source —
    if they drift, the diffusion train step uses inconsistent coefficients."""
    s = NoiseSchedulers.LINEAR(T=100)
    assert torch.allclose(s.sqrt_alphas_cumprod, s.alphas_cumprod.sqrt())
    assert torch.allclose(
        s.sqrt_one_minus_alphas_cumprod, (1.0 - s.alphas_cumprod).sqrt(),
    )


def test_posterior_variance_non_negative():
    s = NoiseSchedulers.LINEAR(T=100)
    assert (s.posterior_variance >= 0).all()


def test_cosine_schedule_shape():
    s = NoiseSchedulers.COSINE(T=100)
    assert s.kind is NoiseSchedulers.COSINE
    assert s.betas.shape == (100,)
    # COSINE α_bar must still be decreasing.
    diffs = s.alphas_cumprod[1:] - s.alphas_cumprod[:-1]
    assert (diffs < 0).all()


def test_cosine_betas_are_clamped():
    """Cosine schedule clamps β to ≤ 0.999 — without this, late-step
    betas blow up and the posterior variance becomes ill-conditioned."""
    s = NoiseSchedulers.COSINE(T=1000)
    assert s.betas.max().item() <= 0.999 + 1e-6
    assert s.betas.min().item() >= 1e-8 - 1e-12


def test_to_migrates_all_tensors():
    """Schedule.to(device) must move every tensor field. Using CPU
    here since CUDA may be unavailable on test machines; the test
    asserts the no-op identity (a deepcopy, but on the same device)
    rather than a device transfer."""
    s = NoiseSchedulers.LINEAR(T=10)
    s2 = s.to("cpu")
    assert s2.betas.device == torch.device("cpu")
    assert torch.equal(s.betas, s2.betas)
    assert s2.T == s.T
    assert s2.kind is s.kind


def test_invalid_T_raises():
    with pytest.raises(ValueError, match="T must be positive"):
        NoiseSchedulers.LINEAR(T=0)
    with pytest.raises(ValueError, match="T must be positive"):
        NoiseSchedulers.COSINE(T=-5)


def test_linear_invalid_beta_bounds_raises():
    with pytest.raises(ValueError, match="beta_min"):
        NoiseSchedulers.LINEAR(T=10, beta_min=0.5, beta_max=0.1)
    with pytest.raises(ValueError, match="beta_min"):
        NoiseSchedulers.LINEAR(T=10, beta_min=0.0, beta_max=0.5)


def test_cosine_invalid_s_raises():
    with pytest.raises(ValueError, match="s > 0"):
        NoiseSchedulers.COSINE(T=10, s=-0.1)


def test_linear_schedule_T_one_is_valid():
    """The smallest valid schedule (T=1) must build without raising; many
    array-indexing bugs (e.g., `betas[1:] - betas[:-1]` on a length-1
    tensor) surface only at the boundary."""
    s = NoiseSchedulers.LINEAR(T=1)
    assert s.betas.shape == (1,)
    assert torch.isfinite(s.betas).all()
    assert torch.isfinite(s.alphas_cumprod).all()
    assert torch.isfinite(s.sqrt_alphas_cumprod).all()
    assert torch.isfinite(s.sqrt_one_minus_alphas_cumprod).all()


def test_cosine_schedule_T_one_is_valid():
    """Same boundary test for the cosine schedule."""
    s = NoiseSchedulers.COSINE(T=1)
    assert s.betas.shape == (1,)
    assert torch.isfinite(s.betas).all()
    assert torch.isfinite(s.alphas_cumprod).all()
