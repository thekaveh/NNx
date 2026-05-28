"""Net2DeeperNet — function-preserving depth expansion. Placeholder
that will be filled in by PR 2 of the SP-6 surgery series."""

from __future__ import annotations

from torch import nn


def deepen(model: nn.Module, *, after_layer_name: str) -> nn.Module:  # pragma: no cover
    raise NotImplementedError("deepen is implemented in a subsequent PR")
