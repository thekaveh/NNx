"""Embedding row expansion. Placeholder filled in by PR 5 of SP-6."""

from __future__ import annotations

from torch import nn


def expand_embedding(
    emb: nn.Embedding,
    *,
    new_num_embeddings: int,
    init: str = "zeros",
) -> tuple:  # pragma: no cover
    raise NotImplementedError("expand_embedding is implemented in a subsequent PR")
