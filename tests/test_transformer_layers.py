"""Tests for the decoder-only Transformer building blocks.

Covers each layer individually so a regression in one (RMSNorm, RoPE,
SwiGLU, multi-head causal attention, TransformerBlock) is bisectable
without having to debug the full TransformerNN at once.
"""

from __future__ import annotations

import math

import pytest
import torch

from nnx.nn.net.transformer_layers import (
    RMSNorm,
    RoPE,
    SwiGLU,
    TransformerBlock,
    apply_rotary,
    build_causal_mask,
    multi_head_causal_attention,
)

# ---------------- RMSNorm ----------------


def test_rmsnorm_preserves_shape():
    norm = RMSNorm(dim=16)
    x = torch.randn(2, 7, 16)
    y = norm(x)
    assert y.shape == x.shape


def test_rmsnorm_normalizes_to_unit_rms_with_weight_one():
    # By default weight=ones, so RMSNorm scales the last-dim vector to have
    # RMS ~ 1 (within eps). Pick a non-unit-RMS input and verify.
    norm = RMSNorm(dim=8, eps=1e-12)
    x = torch.randn(3, 5, 8) * 4.0  # large scale
    y = norm(x)
    rms = y.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), rms


def test_rmsnorm_weight_is_learnable_param():
    norm = RMSNorm(dim=4)
    assert norm.weight.requires_grad
    assert norm.weight.shape == (4,)


# ---------------- RoPE ----------------


def test_rope_returns_cos_sin_of_expected_shape():
    rope = RoPE(dim=8, max_seq_len=32, base=10000.0)
    cos, sin = rope.forward(seq_len=5)
    # cos/sin are (seq_len, dim/2) — applied per pair of dimensions.
    assert cos.shape == (5, 4)
    assert sin.shape == (5, 4)


def test_apply_rotary_is_norm_preserving():
    # Rotation matrices preserve L2 norm; pass random Q through rotary and
    # check norms are within float tolerance of original.
    torch.manual_seed(0)
    rope = RoPE(dim=16, max_seq_len=64)
    cos, sin = rope.forward(seq_len=10)
    q = torch.randn(2, 3, 10, 16)  # (batch, n_heads, seq, head_dim)
    q_rot = apply_rotary(q, cos, sin)
    pre = q.norm(dim=-1)
    post = q_rot.norm(dim=-1)
    assert torch.allclose(pre, post, atol=1e-5), (pre, post)


def test_apply_rotary_at_position_zero_is_identity():
    # At position 0, cos=1 and sin=0, so the rotation is identity.
    rope = RoPE(dim=8, max_seq_len=4)
    cos, sin = rope.forward(seq_len=1)
    q = torch.randn(1, 2, 1, 8)
    q_rot = apply_rotary(q, cos, sin)
    assert torch.allclose(q, q_rot, atol=1e-6)


def test_rope_rotates_correctly_for_known_inputs():
    """Hand-computed: d=4, base=10000, position=1.

    theta_0 = 10000^(0/4) = 1, freq_0 = 1/1 = 1 → angle = 1.0 rad at t=1
    theta_1 = 10000^(2/4) = 100, freq_1 = 1/100 → angle = 0.01 rad at t=1

    For input x = [1, 0, 1, 0], paired as [(1,0), (1,0)]:
      pair0 rotated by 1.0 rad: (cos(1), sin(1)) = (0.5403, 0.8415)
      pair1 rotated by 0.01 rad: (cos(0.01), sin(0.01)) ≈ (0.99995, 0.01)
    """
    rope = RoPE(dim=4, max_seq_len=4, base=10000.0)
    cos, sin = rope.forward(seq_len=2)
    # We only check t=1 (the second row).
    q = torch.tensor([[[[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]]])
    # q shape: (batch=1, n_heads=1, seq=2, head_dim=4)
    q_rot = apply_rotary(q, cos, sin)
    out_t1 = q_rot[0, 0, 1].tolist()  # row for t=1
    expected = [math.cos(1.0), math.sin(1.0), math.cos(0.01), math.sin(0.01)]
    assert all(abs(a - b) < 1e-5 for a, b in zip(out_t1, expected, strict=False)), (
        out_t1,
        expected,
    )


# ---------------- SwiGLU ----------------


def test_swiglu_preserves_shape():
    ff = SwiGLU(d_model=16, ffn_mult=4)
    x = torch.randn(2, 7, 16)
    y = ff(x)
    assert y.shape == x.shape


def test_swiglu_hidden_dim_follows_2_3_convention():
    # SwiGLU convention: hidden = int(2 * ffn_mult * d / 3)
    ff = SwiGLU(d_model=12, ffn_mult=4)
    expected_hidden = int(2 * 4 * 12 / 3)
    assert ff.w1.out_features == expected_hidden
    assert ff.w3.out_features == expected_hidden
    assert ff.w2.in_features == expected_hidden
    assert ff.w2.out_features == 12


def test_swiglu_has_no_bias():
    # LLaMA/SwiGLU conventionally have bias=False everywhere.
    ff = SwiGLU(d_model=8, ffn_mult=2)
    assert ff.w1.bias is None
    assert ff.w2.bias is None
    assert ff.w3.bias is None


# ---------------- Causal mask ----------------


def test_build_causal_mask_blocks_future_positions():
    mask = build_causal_mask(seq_len=4)
    # Upper triangle (j > i) should be -inf; lower-and-diag should be 0.
    assert mask.shape == (4, 4)
    assert mask[0, 0].item() == 0.0
    assert mask[3, 3].item() == 0.0
    assert mask[0, 1].item() == float("-inf")
    assert mask[1, 0].item() == 0.0
    assert mask[2, 3].item() == float("-inf")


# ---------------- multi_head_causal_attention ----------------


def test_attention_output_shape():
    torch.manual_seed(0)
    batch, seq, n_heads, head_dim = 2, 5, 4, 8
    q = torch.randn(batch, n_heads, seq, head_dim)
    k = torch.randn(batch, n_heads, seq, head_dim)
    v = torch.randn(batch, n_heads, seq, head_dim)
    mask = build_causal_mask(seq_len=seq)
    out = multi_head_causal_attention(q, k, v, mask)
    assert out.shape == (batch, n_heads, seq, head_dim)


def test_attention_is_causal_first_token_only_sees_itself():
    """When seq is masked causally, the output for position 0 only
    depends on input at position 0 — perturbing later positions of V
    must NOT change the output at position 0."""
    torch.manual_seed(0)
    batch, seq, n_heads, head_dim = 1, 4, 2, 4
    q = torch.randn(batch, n_heads, seq, head_dim)
    k = torch.randn(batch, n_heads, seq, head_dim)
    v = torch.randn(batch, n_heads, seq, head_dim)
    mask = build_causal_mask(seq_len=seq)

    out1 = multi_head_causal_attention(q, k, v, mask)
    v_perturbed = v.clone()
    v_perturbed[:, :, 1:, :] += 100.0  # perturb only positions >= 1
    out2 = multi_head_causal_attention(q, k, v_perturbed, mask)

    # Position 0 outputs must be identical.
    assert torch.allclose(out1[:, :, 0, :], out2[:, :, 0, :], atol=1e-5)
    # And later positions should change (sanity).
    assert not torch.allclose(out1[:, :, -1, :], out2[:, :, -1, :], atol=1e-3)


# ---------------- TransformerBlock ----------------


def test_transformer_block_preserves_shape():
    block = TransformerBlock(d_model=16, n_heads=4, ffn_mult=4, max_seq_len=32)
    x = torch.randn(2, 8, 16)
    y, _ = block(x)
    assert y.shape == x.shape


def test_transformer_block_residual_when_zero_init():
    """If we zero-init the output projections of attention and FFN, the
    block reduces to identity (the residual passes through). This is a
    nice integration check that the residual connections actually wire
    through both sub-layers."""
    block = TransformerBlock(d_model=8, n_heads=2, ffn_mult=2, max_seq_len=16)
    # Zero the output projection of attention and the down-projection of FFN.
    torch.nn.init.zeros_(block.attn.w_o.weight)
    torch.nn.init.zeros_(block.ffn.w2.weight)
    x = torch.randn(1, 5, 8)
    y, _ = block(x)
    assert torch.allclose(y, x, atol=1e-5)


def test_transformer_block_kv_cache_seam_returns_none_when_disabled():
    """The low-level default is `use_cache=False`; the returned kv must
    be None so callers aren't tempted to use the cache prematurely.
    `GenerativeNNModel.generate` flips it on via `forward_with_cache`
    for the high-level decode path."""
    block = TransformerBlock(d_model=8, n_heads=2, ffn_mult=2, max_seq_len=16)
    x = torch.randn(1, 4, 8)
    y, kv = block(x, use_cache=False)
    assert kv is None
    assert y.shape == x.shape


def test_transformer_block_requires_d_model_divisible_by_n_heads():
    with pytest.raises(ValueError, match="divisible"):
        TransformerBlock(d_model=15, n_heads=4, ffn_mult=4, max_seq_len=32)
