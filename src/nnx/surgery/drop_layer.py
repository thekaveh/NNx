"""Layer removal. Placeholder filled in by PR 3 of SP-6."""

from __future__ import annotations

from torch import nn


def drop_layer(model: nn.Module, *, layer_name: str, importance=None) -> nn.Module:  # pragma: no cover
    raise NotImplementedError("drop_layer is implemented in a subsequent PR")
