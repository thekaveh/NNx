"""ViT-S — a tiny Vision Transformer encoder built on the LM path's
``transformer_layers`` primitives (``RMSNorm``, ``SwiGLU``).

Scope: small enough to demonstrate I-JEPA on 32x32 images on CPU, not
to chase SOTA on ImageNet. Three architectural choices worth flagging:

  * **Patch embedding via Conv2d** with kernel/stride = ``patch_size``
    — the standard ViT trick that yields one token per non-overlapping
    patch with one matrix multiply.
  * **Learned positional embeddings** (additive, one per patch + CLS).
    The decoder TransformerNN ships RoPE on Q/K because LMs need to
    sample beyond the training context length; vision tokens don't —
    the patch grid is fixed at construction, so learned absolute
    positions are simpler and cheaper.
  * **Non-causal (bidirectional) self-attention**. The
    ``MultiHeadCausalAttention`` in ``transformer_layers`` hard-codes
    a causal mask, so we implement a small ``MultiHeadSelfAttention``
    here. The SwiGLU MLP, RMSNorm, and pre-norm block shape are reused
    unchanged.

The encoder accepts an optional ``mask: BoolTensor[B, n_patches]`` so
the I-JEPA "context encoder" can drop the masked patches before
attention runs — JEPA's whole point is "don't even look at the
prediction targets". Mask = ``True`` means **keep**; the tokens
attached to the CLS are concatenated unconditionally.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .transformer_layers import RMSNorm, SwiGLU


class _MultiHeadSelfAttention(nn.Module):
    """Bidirectional multi-head self-attention (no causal mask, no RoPE).

    Lives in the ViT file rather than ``transformer_layers`` because
    that file's ``MultiHeadCausalAttention`` couples RoPE and the
    causal mask into a single unit — pulling them apart for vision
    tokens would invasively refactor a path that the LM tests pin.
    Keeping the two attention variants separate is simpler and matches
    what every published ViT does.
    """

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = attn_dropout

        # Fused QKV projection to match the LM path's convention. bias=False
        # is the ViT default (Dosovitskiy et al. use bias on QKV, but every
        # modern reimplementation drops it — the LayerNorm/RMSNorm before
        # makes the bias redundant).
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.w_o(attn_out)


class ViTBlock(nn.Module):
    """Pre-norm ViT block: ``x = x + attn(RMSNorm(x)); x = x + ffn(RMSNorm(x))``.

    Same shape as :class:`nnx.nn.net.transformer_layers.TransformerBlock`
    but with bidirectional attention instead of causal. SwiGLU is
    reused unchanged.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int = 4,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = _MultiHeadSelfAttention(d_model, n_heads, attn_dropout=attn_dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, ffn_mult=ffn_mult)
        self.resid_dropout = resid_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + F.dropout(self.attn(self.norm1(x)), p=self.resid_dropout, training=self.training)
        x = x + F.dropout(self.ffn(self.norm2(x)), p=self.resid_dropout, training=self.training)
        return x


class ViTNN(nn.Module):
    """Small Vision Transformer encoder.

    Forward contract:

      ``forward(x: (B, C, H, W), mask: Optional[BoolTensor[B, n_patches]]=None)``
      → ``(B, T_kept + 1, d_model)`` if ``mask`` provided (T_kept = mask.sum())
      → ``(B, n_patches + 1, d_model)`` otherwise.

    The leading token is the learned CLS. Patches are flattened in
    raster order (row-major over the patch grid). The optional ``mask``
    is the I-JEPA "context" mask: True positions are kept, False ones
    are dropped before any attention runs, so gradients do not flow
    through masked patches.

    ``__init__`` requires ``image_size``, ``patch_size``, and
    ``in_channels`` for the patch-embedding convolution. ``image_size``
    must be divisible by ``patch_size`` — validated at construction.
    """

    def __init__(
        self,
        *,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_mult: int = 4,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_patches = (image_size // patch_size) ** 2

        # Patch embedding: Conv2d with kernel=stride=patch_size produces
        # one (d_model)-vector per non-overlapping patch. (B, C, H, W) ->
        # (B, d_model, H/p, W/p) -> flatten -> (B, n_patches, d_model).
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # Learned CLS + position embeddings. One extra position for the
        # CLS token (prepended at the front of the sequence).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        # Init pos_embed / cls_token with small noise so the symmetry
        # at step 0 doesn't lead to degenerate attention patterns.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_mult=ffn_mult,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_out = RMSNorm(d_model)

    def patch_positions(self) -> torch.Tensor:
        """Return ``LongTensor[n_patches]`` of patch-token positions in
        the full sequence (i.e., ``arange(1, n_patches + 1)`` — CLS is
        position 0).

        Exposed so the I-JEPA step factory can derive its context /
        target position indices by boolean-masking this tensor instead
        of rebuilding the arange (see ``jepa_train_step_factory``).
        """
        return torch.arange(1, self.n_patches + 1, device=self.pos_embed.device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the encoder.

        Args:
            x: (B, C, H, W) input image tensor.
            mask: optional BoolTensor of shape ``(B, n_patches)`` —
                True positions are kept, False ones are dropped *before*
                attention. Per-sample masks may have different
                ``True``-counts, but the resulting batch must have the
                same kept-count per row (asserted). I-JEPA's typical
                context mask is uniform across the batch (same set of
                patches kept on every sample in a step).

        Returns:
            ``(B, T_kept + 1, d_model)`` where T_kept is the number of
            kept patches (or ``n_patches`` when mask is None). The +1
            is the CLS token at position 0.
        """
        if x.dim() != 4:
            raise ValueError(f"ViTNN expects (B, C, H, W) input, got shape {tuple(x.shape)}")
        B = x.shape[0]
        # Patch embed -> (B, n_patches, d_model). Add positional embeds
        # to the patch tokens (pos 1..n_patches; index 0 is CLS-only).
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, n_patches, d_model)
        patches = patches + self.pos_embed[:, 1 : self.n_patches + 1, :]

        if mask is not None:
            if mask.shape != (B, self.n_patches):
                raise ValueError(f"mask shape {tuple(mask.shape)} does not match expected ({B}, {self.n_patches})")
            # Per-row kept-count must agree so we can stack into a
            # rectangular tensor. JEPA's typical use is a uniform
            # mask anyway, but we validate to fail loudly when callers
            # construct ragged masks.
            kept_counts = mask.sum(dim=1)
            if kept_counts.unique().numel() != 1:
                raise ValueError(
                    f"mask has ragged kept-counts per row {kept_counts.tolist()}; "
                    "ViTNN requires a uniform kept-count across the batch."
                )
            T_kept = int(kept_counts[0].item())
            # Gather kept tokens. ``mask`` is (B, n_patches) bool; result
            # reshapes to (B, T_kept, d_model).
            patches = patches[mask].view(B, T_kept, self.d_model)

        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, patches], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)

    def unpack_batch(self, batch):
        """Standard ``(X-tuple, Y)`` adapter. JEPA doesn't use Y but
        the supervised linear-probe path on top of a frozen ViTNN does.
        """
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            y = batch[1] if len(batch) > 1 else None
            return (x,), y
        return (batch,), None

    def __str__(self) -> str:
        return (
            f"ViTNN[image={self.image_size}, patch={self.patch_size}, "
            f"d_model={self.d_model}, n_patches={self.n_patches}]"
        )
