"""Decoder-only Transformer building blocks (RMSNorm, RoPE, SwiGLU,
multi-head causal attention, TransformerBlock).

These match the LLaMA/Mistral family of architectural choices:
  * Pre-norm with RMSNorm (no LayerNorm bias term).
  * Rotary positional embedding (RoPE) applied to Q and K only.
  * SwiGLU feed-forward block (gated SiLU; bias=False everywhere).
  * Multi-head causal attention with a KV-cache seam — the low-level
    attention call defaults to ``use_cache=False`` (used by the
    training-forward path); ``GenerativeNNModel.generate`` flips it on
    via ``TransformerNN.forward_with_cache`` for incremental decoding.

Scope explicit: TinyStories-class single-GPU LM, not a production
inference path. FlashAttention v3 / tensor parallelism / multi-node
training are out of scope.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019).

    Cheaper than LayerNorm — no mean subtraction, no learnable bias.
    Used by LLaMA / Mistral / most modern decoder-only LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for the norm computation so half-precision
        # training doesn't underflow on small RMS values; cast back at
        # the end so the rest of the network keeps its dtype.
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RoPE(nn.Module):
    """Rotary positional embedding (Su et al., 2021).

    Precomputes cos/sin tables once at construction; ``forward(seq_len)``
    slices them so a 4-token decoding step doesn't pay the full
    `max_seq_len` precompute on every call.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        # inv_freq[k] = 1 / (base ** (2k/dim))  for k in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) slices of shape (seq_len, dim/2).

        ``offset`` lets the KV-cache path request positions
        [offset, offset + seq_len) without rebuilding the table.
        """
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ValueError(f"RoPE requested positions up to {end} but max_seq_len={self.max_seq_len}")
        return self.cos_cached[offset:end], self.sin_cached[offset:end]


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to a (batch, n_heads, seq, head_dim) tensor.

    Layout convention: pairs ``(x[..., 2k], x[..., 2k+1])`` are rotated
    by angle ``theta_{t,k}`` for token at position ``t``. This matches
    the original RoPE paper (and is the layout the hand-computed
    `test_rope_rotates_correctly_for_known_inputs` test expects).
    """
    # x: (B, H, T, D); cos/sin: (T, D/2)
    # Split into even/odd indexed halves.
    x_even = x[..., 0::2]  # (B, H, T, D/2)
    x_odd = x[..., 1::2]  # (B, H, T, D/2)
    # Broadcast cos/sin over batch/head dims.
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    # Re-interleave.
    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


class SwiGLU(nn.Module):
    """Gated feed-forward block: ``w2(silu(w1(x)) * w3(x))``.

    The hidden dimension follows the LLaMA convention
    ``hidden = int(2 * ffn_mult * d_model / 3)`` so that the parameter
    count of the gated variant matches a plain ``4*d_model`` MLP.
    """

    def __init__(self, d_model: int, ffn_mult: int = 4):
        super().__init__()
        hidden = int(2 * ffn_mult * d_model / 3)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def build_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Additive attention mask: 0 where allowed, -inf where future.

    Shape (seq_len, seq_len); broadcast across batch + head dims by the
    caller. We use additive masking (added to the pre-softmax logits)
    so the same path works under fp16 (where multiplicative masks can
    underflow).
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask = mask.masked_fill(torch.triu(torch.ones_like(mask), diagonal=1).bool(), float("-inf"))
    return mask


def multi_head_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Vanilla scaled-dot-product attention with an additive causal mask.

    Shapes:
      q, k, v: (B, H, T, D_head)
      mask:    (T, T) — broadcast over batch + head.
    Returns: (B, H, T, D_head).

    Implementation note: we deliberately use ``F.scaled_dot_product_attention``
    when no dropout is configured (PyTorch picks the fastest available
    kernel — Flash, mem-efficient, or math), falling back to the explicit
    math path when dropout is requested so training-mode dropout is
    deterministic with the rest of the run's seed.
    """
    if dropout_p == 0.0:
        # PyTorch's SDPA accepts an additive (float) mask.
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

    head_dim = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5)
    scores = scores + mask  # additive causal mask
    attn = F.softmax(scores, dim=-1)
    attn = F.dropout(attn, p=dropout_p, training=True)
    return torch.matmul(attn, v)


class MultiHeadCausalAttention(nn.Module):
    """Multi-head causal self-attention with a KV-cache seam.

    The low-level default is ``use_cache=False`` (training-forward
    path) — the seam is exercised by ``TransformerNN.forward_with_cache``
    which threads cached ``(k, v)`` tuples through every block on
    behalf of ``GenerativeNNModel.generate``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        rope_base: float = 10000.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = attn_dropout

        # Fused QKV projection: 3x faster on small models than three separate
        # Linear layers. bias=False matches LLaMA conventions.
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(dim=self.head_dim, max_seq_len=max_seq_len, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Run attention.

        Args:
            x: (B, T, d_model) input.
            past_kv: ignored when ``use_cache=False`` (training path).
            use_cache: when True, returns the updated (k, v) tuple for
                the caller to thread into the next call —
                ``GenerativeNNModel.generate`` does this via
                ``TransformerNN.forward_with_cache``.
        Returns:
            (output, new_kv) where ``new_kv`` is None when
            ``use_cache=False``.
        """
        b, t, _ = x.shape
        qkv = self.w_qkv(x)  # (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape to (B, H, T, D_head).
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K. The position offset is 0 in the
        # train/full-forward path; the KV-cache path passes a non-zero
        # offset equal to the cached prefix length.
        offset = 0
        if use_cache and past_kv is not None:
            offset = past_kv[0].size(-2)
        cos, sin = self.rope(seq_len=t, offset=offset)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if use_cache and past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # Build the additive causal mask. Under the cache path (T_q < T_kv),
        # only the trailing T_q rows of the full T_kv x T_kv mask apply.
        kv_len = k.size(-2)
        mask = build_causal_mask(seq_len=kv_len, device=x.device)
        if t < kv_len:
            mask = mask[-t:, :]

        attn_out = multi_head_causal_attention(q, k, v, mask, dropout_p=self.attn_dropout if self.training else 0.0)
        # (B, H, T, D_head) -> (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.w_o(attn_out)

        new_kv = (k, v) if use_cache else None
        return out, new_kv


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: x = x + attn(RMSNorm(x)); x = x + ffn(RMSNorm(x)).

    Pre-norm (norm before each sub-layer) is the modern default — it
    stabilizes deep stacks without warmup tricks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int,
        max_seq_len: int,
        rope_base: float = 10000.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadCausalAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            attn_dropout=attn_dropout,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, ffn_mult=ffn_mult)
        self.resid_dropout = resid_dropout

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + F.dropout(attn_out, p=self.resid_dropout, training=self.training)
        x = x + F.dropout(self.ffn(self.norm2(x)), p=self.resid_dropout, training=self.training)
        return x, new_kv
