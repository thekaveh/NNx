"""Expand an :class:`nn.Embedding`'s row count.

Use cases: extending the vocabulary of a pretrained embedding for a
downstream task, adding extra special tokens, or appending an
"out-of-vocab" / "unknown" bin. The contract:

  - The original ``num_embeddings`` rows are copied verbatim into the
    new layer — pretrained behavior on existing token IDs is unchanged.
  - The new rows (indices ``[old_num, new_num)``) are initialized by a
    declared strategy: ``"zeros"`` (deterministic, safe default) or
    ``"copy_mean"`` (each new row equals the mean of the original rows
    — a reasonable warm-start for embedding fine-tuning).
  - A boolean *frozen_mask* of shape ``(new_num,)`` is returned
    alongside the new layer: ``True`` for the original rows (caller
    should keep them frozen), ``False`` for the new rows (trainable).
    The caller can plug the mask into a custom optimizer / hook to
    enforce row-level freezing — :class:`nn.Embedding` itself doesn't
    expose per-row gradients.

The function returns a tuple ``(new_emb, frozen_mask)``. The original
``emb`` is untouched.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import nn
from torch.nn.utils import skip_init

InitStrategy = Literal["zeros", "copy_mean"]


def expand_embedding(
    emb: nn.Embedding,
    *,
    new_num_embeddings: int,
    init: InitStrategy = "zeros",
) -> tuple[nn.Embedding, torch.Tensor]:
    """Return a larger Embedding whose first rows match ``emb`` exactly.

    Args:
        emb: the source embedding. Its weights are read but not
            mutated.
        new_num_embeddings: the desired ``num_embeddings`` for the
            returned layer. Must be strictly greater than the current.
        init: how to initialize the new rows. ``"zeros"`` — fill with
            zeros (default; deterministic, safe). ``"copy_mean"`` —
            fill each new row with the per-column mean of the original
            rows.

    Returns:
        ``(new_emb, frozen_mask)`` where ``new_emb`` is a fresh
        :class:`nn.Embedding` with the original rows preserved, and
        ``frozen_mask`` is a bool tensor of shape
        ``(new_num_embeddings,)`` marking the original rows (``True``)
        as candidates for freezing during refinement.

    Raises:
        TypeError: if ``emb`` is not :class:`nn.Embedding`.
        ValueError: if ``new_num_embeddings`` is not strictly greater,
            or if ``init`` is unknown.
    """
    if not isinstance(emb, nn.Embedding):
        raise TypeError(f"expand_embedding requires an nn.Embedding, got {type(emb).__name__}")
    old_num = emb.num_embeddings
    if new_num_embeddings <= old_num:
        raise ValueError(f"new_num_embeddings must be > current ({old_num}); got {new_num_embeddings}")
    if init not in ("zeros", "copy_mean"):
        raise ValueError(f"unknown init strategy {init!r}; expected 'zeros' or 'copy_mean'")

    dim = emb.embedding_dim
    # skip_init: every row is overwritten below (originals copied, new
    # rows zeroed / mean-filled), so meta-device construction avoids
    # consuming global RNG on a discarded normal_ init.
    new_emb = cast(
        nn.Embedding,
        skip_init(
            nn.Embedding,
            num_embeddings=new_num_embeddings,
            embedding_dim=dim,
            padding_idx=emb.padding_idx,
            max_norm=emb.max_norm,
            norm_type=emb.norm_type,
            scale_grad_by_freq=emb.scale_grad_by_freq,
            sparse=emb.sparse,
            dtype=emb.weight.dtype,
            device=emb.weight.device,
        ),
    )
    with torch.no_grad():
        # Preserve the original rows exactly.
        new_emb.weight[:old_num].copy_(emb.weight.data)

        # Initialize the new rows by the chosen strategy.
        if init == "zeros":
            new_emb.weight[old_num:].zero_()
        elif init == "copy_mean":
            # Mean across rows of the original embedding — a per-column
            # vector — broadcast into every new row.
            row_mean = emb.weight.data.mean(dim=0, keepdim=True)  # (1, dim)
            new_emb.weight[old_num:].copy_(row_mean.expand(new_num_embeddings - old_num, dim))

    # Frozen mask: True for rows that came from the original embedding.
    frozen_mask = torch.zeros(new_num_embeddings, dtype=torch.bool, device=emb.weight.device)
    frozen_mask[:old_num] = True
    return new_emb, frozen_mask
