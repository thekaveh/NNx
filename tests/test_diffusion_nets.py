"""Tests for nnx.diffusion.nets — DiffusionMLP + sinusoidal_time_embed."""

from __future__ import annotations

import pytest
import torch

from nnx import DiffusionMLP
from nnx.diffusion import sinusoidal_time_embed


def test_sinusoidal_time_embed_shape():
    t = torch.arange(8)
    out = sinusoidal_time_embed(t, dim=16)
    assert out.shape == (8, 16)


def test_sinusoidal_time_embed_odd_dim_raises():
    with pytest.raises(ValueError, match="even"):
        sinusoidal_time_embed(torch.arange(4), dim=15)


def test_sinusoidal_time_embed_distinct_per_t():
    """Different timesteps must produce different embeddings — required
    for the conditioning to carry information."""
    out = sinusoidal_time_embed(torch.arange(10), dim=16)
    # Pairwise distinct rows.
    for i in range(out.shape[0]):
        for j in range(i + 1, out.shape[0]):
            assert not torch.allclose(out[i], out[j])


def test_diffusion_mlp_forward_shape_2d():
    net = DiffusionMLP(input_dim=2, hidden_dims=[32, 32], time_embed_dim=16)
    x = torch.randn(8, 2)
    t = torch.randint(0, 100, (8,))
    out = net(x, t)
    assert out.shape == x.shape


def test_diffusion_mlp_forward_preserves_image_shape():
    """For higher-rank inputs (e.g., (B, C, H, W)), the network flattens
    internally but must reshape the output back."""
    net = DiffusionMLP(input_dim=3 * 4 * 4, hidden_dims=[16], time_embed_dim=8)
    x = torch.randn(2, 3, 4, 4)
    t = torch.randint(0, 100, (2,))
    out = net(x, t)
    assert out.shape == x.shape


def test_diffusion_mlp_input_dim_mismatch_raises():
    net = DiffusionMLP(input_dim=4, time_embed_dim=8)
    x = torch.randn(2, 7)
    t = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="flattened input dim"):
        net(x, t)


def test_diffusion_mlp_invalid_input_dim_raises():
    with pytest.raises(ValueError, match="input_dim must be positive"):
        DiffusionMLP(input_dim=0)


def test_diffusion_mlp_unpack_batch_handles_tuple_and_tensor():
    net = DiffusionMLP(input_dim=2, time_embed_dim=8)
    x = torch.randn(4, 2)
    # Tuple form (X, Y) — the conventional dataloader contract.
    (X_t,), Y_t = net.unpack_batch((x, torch.zeros(4)))
    assert torch.equal(X_t, x)
    assert Y_t is not None
    # Bare tensor — diffusion datasets that yield only x_0.
    (X_t2,), Y_t2 = net.unpack_batch(x)
    assert torch.equal(X_t2, x)
    assert Y_t2 is None
