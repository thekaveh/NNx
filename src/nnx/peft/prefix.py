"""Prefix tuning — learnable K/V prefixes per transformer layer.

Prefix tuning (`li:prefix-tuning`) freezes the entire pretrained model
and learns a small set of per-layer "virtual" key/value vectors that are
prepended to each attention layer's K and V tensors. Queries (real
tokens) attend back to the prefix slots, but the prefix slots are not
themselves part of the input sequence — they only exist on the K/V side.

The effect: gradient flows only through the prefix tensors during
fine-tuning. The pretrained transformer is fully preserved, and the
adapter is tiny (``n_layers * n_prefix * n_heads * head_dim * 2``
parameters total — typically <1% of the base model).

Mechanism in this file:

  - :class:`PrefixTuner` wraps a :class:`TransformerNN`, freezes every
    base parameter, and allocates one ``(n_prefix, n_heads, head_dim)``
    K-tensor and one V-tensor per :class:`TransformerBlock` it
    targets (all blocks by default, or a prefix of them via
    ``n_layers``).
  - At construction time we MONKEY-PATCH each target block's
    ``MultiHeadCausalAttention.forward`` with a closure that runs the
    original attention but injects the learned K/V prefix after RoPE
    is applied to the real K. The prefix tensors are RoPE-free (they're
    learned content, not positional offsets — same convention as the
    original prefix-tuning paper).
  - The attention mask is extended so the ``n_prefix`` prefix columns
    are unmasked for every query row — queries always see the full
    prefix; the causal mask only applies between real tokens.

We monkey-patch (rather than registering a forward hook) because a
hook fires on outputs, not on intermediate K/V — and the K/V live
strictly inside the MHA forward. Subclassing :class:`MultiHeadCausalAttention`
would require swapping module instances inside the block, which is more
intrusive than per-instance forward replacement.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from ..nn.net.transformer_layers import (
    MultiHeadCausalAttention,
    apply_rotary,
    build_causal_mask,
    multi_head_causal_attention,
)
from ..nn.net.transformer_nn import TransformerNN
from ._source import _resolve_source_to_state_dict


class PrefixTuner(nn.Module):
    """Wrap a :class:`TransformerNN` with learnable per-layer K/V prefixes.

    Freezes every parameter of the wrapped model on construction and
    registers ``n_layers`` pairs of ``(n_prefix, n_heads, head_dim)``
    K / V tensors as the only trainable parameters.

    Args:
        model: a :class:`TransformerNN` instance. Its parameters are
            mutated in place (set to ``requires_grad=False``); the
            attention forward of each targeted block is monkey-patched.
        n_prefix: number of virtual prefix tokens per layer. Must be > 0.
        n_layers: number of leading transformer blocks to attach a prefix
            to. ``None`` (default) targets every block in
            ``model.blocks``. When set, the first ``n_layers`` blocks
            are targeted; later blocks run un-prefixed.

    Note on shape: the prefix uses ``n_heads`` and ``head_dim`` taken
    from the model's ``params`` — there's no per-block override, since
    every block in a TransformerNN shares the same attention shape.
    """

    def __init__(
        self,
        model: TransformerNN,
        *,
        n_prefix: int = 10,
        n_layers: Optional[int] = None,
    ):
        super().__init__()
        if not isinstance(model, TransformerNN):
            raise TypeError(f"PrefixTuner requires a TransformerNN, got {type(model).__name__}")
        if n_prefix <= 0:
            raise ValueError(f"n_prefix must be positive, got {n_prefix}")

        total_blocks = len(model.blocks)
        if n_layers is None:
            n_layers = total_blocks
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if n_layers > total_blocks:
            raise ValueError(f"n_layers={n_layers} exceeds model depth ({total_blocks} blocks)")

        self.model = model
        self.n_prefix = n_prefix
        self.n_target_layers = n_layers

        # Freeze every parameter of the wrapped TransformerNN — the
        # prefix tensors are the only trainable bit going forward.
        for p in self.model.parameters():
            p.requires_grad = False

        # Allocate (n_prefix, n_heads, head_dim) K and V per targeted
        # block. We store them as a ParameterList so they live in
        # state_dict and are visible to `self.parameters()`.
        n_heads = model.params.n_heads
        head_dim = model.params.d_model // n_heads

        self.prefix_keys = nn.ParameterList(
            [nn.Parameter(torch.empty(n_prefix, n_heads, head_dim)) for _ in range(n_layers)]
        )
        self.prefix_values = nn.ParameterList(
            [nn.Parameter(torch.empty(n_prefix, n_heads, head_dim)) for _ in range(n_layers)]
        )
        # Init with small Gaussian noise — matches the original
        # prefix-tuning paper's "random init" baseline.
        for p in self.prefix_keys:
            nn.init.normal_(p, std=0.02)
        for p in self.prefix_values:
            nn.init.normal_(p, std=0.02)

        # Monkey-patch each targeted block's MHA forward to inject the
        # learned prefix. We keep a reference to each MHA's original
        # forward in a sidecar list so the patch can be undone in
        # principle (not exposed as public API, but useful for clean
        # teardown in tests).
        self._original_attn_forwards: list = []
        for i in range(n_layers):
            block = model.blocks[i]
            mha = block.attn
            self._original_attn_forwards.append(mha.forward)
            mha.forward = self._make_patched_forward(mha, layer_idx=i)

    def _make_patched_forward(self, mha: MultiHeadCausalAttention, layer_idx: int):
        """Build a closure that replaces ``mha.forward`` with the
        prefix-injecting variant for layer ``layer_idx``.

        The body mirrors :meth:`MultiHeadCausalAttention.forward`
        exactly — same QKV projection, same RoPE application — except
        after RoPE on K it prepends the layer's learned prefix K/V
        slots and extends the causal mask so queries can attend back
        to the prefix unmasked.
        """
        tuner = self

        def patched_forward(
            x: torch.Tensor,
            past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
        ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
            b, t, _ = x.shape
            qkv = mha.w_qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(b, t, mha.n_heads, mha.head_dim).transpose(1, 2)
            k = k.view(b, t, mha.n_heads, mha.head_dim).transpose(1, 2)
            v = v.view(b, t, mha.n_heads, mha.head_dim).transpose(1, 2)

            # RoPE on q / k. Prefix-tuning does NOT apply RoPE to the
            # prefix slots — they're learned content, not positional.
            offset = 0
            if use_cache and past_kv is not None:
                offset = past_kv[0].size(-2)
            cos, sin = mha.rope(seq_len=t, offset=offset)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

            if use_cache and past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)

            # Snapshot the cache BEFORE prefix injection: the cache must
            # hold real-token K/V only. Caching the prefix-injected
            # tensors would re-prepend the prefix on top of the cached
            # copy every decode step (n_prefix duplicate slots per step)
            # and inflate the RoPE offset above by n_prefix per step —
            # cached logits drifted ~2.0 from the full forward.
            new_kv = (k, v) if use_cache else None

            # Inject the learned K/V prefix for this layer. Shape:
            # (n_prefix, n_heads, head_dim) -> (1, n_heads, n_prefix, head_dim)
            # broadcast over the batch dim by .expand.
            pk = tuner.prefix_keys[layer_idx].permute(1, 0, 2).unsqueeze(0)
            pv = tuner.prefix_values[layer_idx].permute(1, 0, 2).unsqueeze(0)
            pk = pk.expand(b, -1, -1, -1)
            pv = pv.expand(b, -1, -1, -1)
            k = torch.cat([pk, k], dim=-2)  # (B, H, n_prefix + T_kv, D_head)
            v = torch.cat([pv, v], dim=-2)

            # Build the attention mask. The query dim is t (the real
            # tokens). The key dim is n_prefix + kv_len. The prefix
            # columns are always unmasked; the real-token columns get a
            # standard causal mask.
            kv_len = k.size(-2) - tuner.n_prefix  # length of real K/V
            real_mask = build_causal_mask(seq_len=max(kv_len, t), device=x.device)
            if t < kv_len:
                # Cache path: only the trailing t rows of the full mask
                # apply, like the unpatched MHA forward does.
                real_mask = real_mask[-t:, :kv_len]
            else:
                # Train / no-cache path: kv_len == t.
                real_mask = real_mask[:t, :kv_len]
            prefix_mask = torch.zeros(t, tuner.n_prefix, device=x.device)
            mask = torch.cat([prefix_mask, real_mask], dim=-1)  # (t, n_prefix + kv_len)

            attn_out = multi_head_causal_attention(q, k, v, mask, dropout_p=mha.attn_dropout if mha.training else 0.0)
            attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, mha.d_model)
            out = mha.w_o(attn_out)

            return out, new_kv

        return patched_forward

    # ------------------------------------------------------------------
    # Forward + trainable params
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        """Delegate to the wrapped model. The prefix injection happens
        inside each block's monkey-patched MHA forward."""
        return self.model(*args, **kwargs)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the learned prefix tensors.

        The wrapped model's parameters are frozen on construction; this
        is the iterator you hand to an optimizer.
        """
        yield from self.prefix_keys
        yield from self.prefix_values

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def prefix_state_dict(self) -> dict:
        """Return a state-dict containing only the prefix tensors,
        keyed for round-trip via :meth:`load_prefix_weights`.

        The keys are the same as ``self.state_dict()`` filtered to the
        prefix entries — i.e., ``prefix_keys.0``, ``prefix_values.0``,
        ``prefix_keys.1``, …
        """
        return {k: v for k, v in self.state_dict().items() if "prefix_" in k}


def save_prefix_weights(tuner: PrefixTuner, path: Union[str, Path]) -> str:
    """Save ONLY the prefix tensors of ``tuner`` to ``path``.

    Args:
        tuner: a :class:`PrefixTuner` instance.
        path: destination file path.

    Returns:
        The path written, so calls can be chained.
    """
    sd = tuner.prefix_state_dict()
    torch.save(sd, str(path))
    return str(path)


def load_prefix_weights(tuner: PrefixTuner, source: Union[str, Path, dict]) -> int:
    """Load prefix tensors into ``tuner`` from ``source``.

    Args:
        tuner: must already have the same prefix shape as the source
            (same n_prefix, n_heads, head_dim, n_layers). Otherwise
            ``load_state_dict`` will surface the mismatch.
        source: a path to a file produced by :func:`save_prefix_weights`,
            or a state-dict dict directly.

    Returns:
        The number of parameter tensors loaded.
    """
    sd = _resolve_source_to_state_dict(source, "load_prefix_weights")
    # Filter to prefix-only keys defensively so a full-model state-dict
    # accidentally passed in doesn't blow up the loader.
    sd = {k: v for k, v in sd.items() if "prefix_" in k}
    result = tuner.load_state_dict(sd, strict=False)
    # strict=False silently drops keys that don't exist on the tuner —
    # subtract them so the return value counts tensors that landed.
    return len(sd) - len(result.unexpected_keys)
