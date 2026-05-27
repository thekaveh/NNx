"""Adapter layers — bottleneck residual blocks for parameter-efficient
fine-tuning.

The classic Houlsby/Pfeiffer adapter: insert a tiny down-project →
nonlinearity → up-project block after each pretrained layer, with a
residual connection so the initial output equals the input. The
adapter's up-projection is zero-initialized for the same reason
:class:`LoRALinear`'s ``B`` matrix is — fine-tuning starts from the
pretrained behavior and diverges only as the adapter picks up gradient.

Unlike LoRA (which transparently wraps an existing linear), adapters
are full modules that the caller inserts into the forward pass
themselves. NNX doesn't ship a "wrap every block" helper because
"after each pretrained layer" depends on the architecture; adapters
in `FeedFwdNN` go in different places than adapters in a transformer.
Compose :class:`AdapterLayer` into a custom :class:`nn.Module` as
needed.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class AdapterLayer(nn.Module):
    """Bottleneck residual block: ``y = x + up(act(down(x)))``.

    ``up.weight`` and ``up.bias`` are zero-initialized so at step 0
    the layer's output equals its input exactly. Gradient flow
    through ``up`` and ``down`` is unblocked from the first step;
    only the magnitude of the residual starts at zero.

    Args:
        dim: input and output feature dimension. The adapter is
            shape-preserving.
        bottleneck: hidden dimension. Typically much smaller than
            ``dim`` (e.g., dim=768 → bottleneck=64 in the original
            Houlsby setup). Lower bottleneck = fewer params,
            potentially less expressive.
        activation: callable applied between down- and up-projection.
            Defaults to ``torch.nn.GELU()`` — the modern adapter
            choice; ReLU works too.
    """

    def __init__(
        self,
        dim: int,
        bottleneck: int,
        activation: Callable[[], nn.Module] = nn.GELU,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if bottleneck <= 0:
            raise ValueError(f"bottleneck must be positive, got {bottleneck}")

        self.down = nn.Linear(dim, bottleneck)
        self.act = activation()
        self.up = nn.Linear(bottleneck, dim)

        # Zero-init the up-projection so the adapter is the identity
        # residual at step 0. Gradients still flow through `up` from
        # the first backward pass; only the OUTPUT magnitude starts
        # at zero.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))

    def extra_repr(self) -> str:
        return f"dim={self.down.in_features}, bottleneck={self.down.out_features}"
