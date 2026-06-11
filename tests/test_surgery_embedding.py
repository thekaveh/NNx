"""Tests for ``nnx.surgery.expand_embedding``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import expand_embedding

# ---------- Function-preservation: the contract ------------------------


def test_expand_embedding_preserves_original_rows_exactly():
    """Looking up any original index in the surged Embedding returns
    the same vector as the original Embedding."""
    torch.manual_seed(0)
    old = nn.Embedding(num_embeddings=10, embedding_dim=4)
    new_emb, frozen_mask = expand_embedding(old, new_num_embeddings=15)

    assert new_emb.num_embeddings == 15
    assert new_emb.embedding_dim == 4

    # Compare lookup outputs at every original index.
    idx = torch.arange(10)
    assert torch.equal(new_emb(idx), old(idx))


def test_expand_embedding_zeros_init():
    """The default 'zeros' init must zero-fill the new rows."""
    torch.manual_seed(1)
    old = nn.Embedding(num_embeddings=5, embedding_dim=3)
    new_emb, _ = expand_embedding(old, new_num_embeddings=8, init="zeros")

    new_rows = new_emb.weight.data[5:]
    assert torch.equal(new_rows, torch.zeros(3, 3))


def test_expand_embedding_copy_mean_init():
    """The 'copy_mean' init replicates the per-column mean of the
    original rows into every new row."""
    torch.manual_seed(2)
    old = nn.Embedding(num_embeddings=6, embedding_dim=4)
    new_emb, _ = expand_embedding(old, new_num_embeddings=10, init="copy_mean")

    expected_row = old.weight.data.mean(dim=0)
    for i in range(6, 10):
        assert torch.allclose(new_emb.weight.data[i], expected_row, atol=1e-6)


def test_expand_embedding_frozen_mask_shape_and_contents():
    """Mask shape == (new_num,); True for original rows, False for new."""
    old = nn.Embedding(num_embeddings=4, embedding_dim=2)
    _, mask = expand_embedding(old, new_num_embeddings=7)
    assert mask.dtype == torch.bool
    assert mask.shape == (7,)
    assert mask[:4].all().item() is True
    assert (~mask[4:]).all().item() is True


def test_expand_embedding_returns_fresh_module():
    old = nn.Embedding(num_embeddings=4, embedding_dim=2)
    new_emb, _ = expand_embedding(old, new_num_embeddings=8)
    assert new_emb is not old
    assert old.num_embeddings == 4  # untouched


def test_expand_embedding_preserves_padding_idx_and_other_kwargs():
    """nn.Embedding has several auxiliary kwargs (padding_idx, max_norm,
    scale_grad_by_freq, sparse). The surged copy must preserve them."""
    old = nn.Embedding(num_embeddings=6, embedding_dim=3, padding_idx=0, max_norm=1.0, scale_grad_by_freq=True)
    new_emb, _ = expand_embedding(old, new_num_embeddings=10)
    assert new_emb.padding_idx == 0
    assert new_emb.max_norm == 1.0
    assert new_emb.scale_grad_by_freq is True


def test_expand_embedding_frozen_mask_supports_row_level_freeze():
    """The frozen_mask must be usable to zero-out gradient updates on
    original rows after a training step — that's its purpose."""
    torch.manual_seed(3)
    old = nn.Embedding(num_embeddings=5, embedding_dim=3)
    new_emb, mask = expand_embedding(old, new_num_embeddings=8, init="zeros")

    # Simulate one optimizer step: every row gets a gradient of 1.0,
    # then we mask out the frozen rows so they don't update.
    new_emb.weight.requires_grad_(True)
    fake_loss = new_emb.weight.sum()
    fake_loss.backward()
    grad_before = new_emb.weight.grad.clone()
    # Zero the gradient on frozen rows.
    new_emb.weight.grad[mask] = 0.0
    # Original rows now have zero gradient; new rows still have 1.0.
    assert torch.equal(new_emb.weight.grad[:5], torch.zeros(5, 3))
    assert torch.equal(new_emb.weight.grad[5:], torch.ones(3, 3))
    # Sanity: before masking, all grads were 1.
    assert torch.equal(grad_before, torch.ones(8, 3))


# ---------- Error handling --------------------------------------------


def test_expand_embedding_rejects_non_embedding():
    with pytest.raises(TypeError, match="requires an nn.Embedding"):
        expand_embedding(nn.Linear(3, 3), new_num_embeddings=10)


def test_expand_embedding_rejects_shrinking():
    old = nn.Embedding(num_embeddings=10, embedding_dim=4)
    with pytest.raises(ValueError, match="must be > current"):
        expand_embedding(old, new_num_embeddings=5)


def test_expand_embedding_rejects_equal_count():
    old = nn.Embedding(num_embeddings=10, embedding_dim=4)
    with pytest.raises(ValueError, match="must be > current"):
        expand_embedding(old, new_num_embeddings=10)


def test_expand_embedding_rejects_unknown_init():
    old = nn.Embedding(num_embeddings=10, embedding_dim=4)
    with pytest.raises(ValueError, match="unknown init strategy"):
        expand_embedding(old, new_num_embeddings=12, init="random")  # type: ignore[arg-type]


def test_expand_embedding_does_not_advance_global_rng():
    """expand_embedding() must be a no-op on the global torch RNG
    stream — every row of the fresh Embedding is overwritten, so it is
    built uninitialized (skip_init). Pre-fix, the fresh layer's normal_
    init drew from the default generator, silently diverging any
    seeded caller pipeline. Covers both init modes."""
    old = nn.Embedding(num_embeddings=10, embedding_dim=4)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    expand_embedding(old, new_num_embeddings=15, init="zeros")
    expand_embedding(old, new_num_embeddings=15, init="copy_mean")
    assert torch.equal(torch.get_rng_state(), state)
