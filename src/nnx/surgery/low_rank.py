"""Low-rank SVD factorization. Placeholder filled in by PR 4 of SP-6."""

from __future__ import annotations

from torch import nn


def low_rank_factorize(linear: nn.Linear, *, rank: int, method: str = "svd") -> nn.Module:  # pragma: no cover
    raise NotImplementedError("low_rank_factorize is implemented in a subsequent PR")
