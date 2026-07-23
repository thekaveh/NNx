"""Convolutional (LeNet-style) params (#89).

``NNConvParams`` subclasses ``NNParams`` — the same lift-via-subclassing
pattern ``NNTransformerParams``/``NNMoEParams`` use. It adds the conv-stack
knobs consumed by :class:`~nnx.nn.net.conv_nn.ConvNN`:

- ``conv_channels`` — out-channels per Conv→Pool block (required; no
  meaningful default).
- ``in_channels`` / ``kernel_size`` / ``stride`` / ``padding`` /
  ``pool_size`` — LeNet-5 defaults (1 / 5 / 1 / 0 / 2). Omitted from
  ``state()`` at their defaults — the omit-when-default invariant that keeps
  a "vanilla" conv config hashing to a stable run.id as knobs accrue.

``conv_channels`` is ALWAYS emitted by ``state()``: it is the discriminator
``NNParams.resolve_from_state`` dispatches on (mirroring ``vocab_size`` →
transformer, ``num_experts`` → MoE), and hashing it is exactly what keeps a
conv run's id distinct from its plain-FeedFwd twin.

Unlike the base feed-forward params, convolutional params require a non-null
scalar ``activation`` because every Conv→Pool block uses it. Optional
``activations`` remain overrides for the fully connected hidden layers only.

v1 targets small SQUARE images: the spatial side is derived as
``sqrt(input_dim / in_channels)`` (MNIST: 784/1 → 28), so the base
``input_dim`` keeps its meaning and no extra height/width fields are needed.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

from .nn_params import NNParams, _ImmutableList

_LENET_DEFAULTS = {"in_channels": 1, "kernel_size": 5, "stride": 1, "padding": 0, "pool_size": 2}


@dataclass(frozen=True, kw_only=True, slots=True)
class NNConvParams(NNParams):
    """Parameters for a ConvNN with a required conv-block activation."""

    # Required: out-channels per Conv→Pool block.
    conv_channels: list[int]
    # LeNet-5 defaults.
    in_channels: int = 1
    kernel_size: int = 5
    stride: int = 1
    padding: int = 0
    pool_size: int = 2

    def __post_init__(self):
        # Explicit unbound call — same slotted-dataclass reasoning as
        # NNTransformerParams.__post_init__.
        NNParams.__post_init__(self)
        object.__setattr__(self, "conv_channels", _ImmutableList(self.conv_channels))
        if self.activation is None:
            raise ValueError("NNConvParams requires a scalar activation for convolution blocks")
        if not self.conv_channels or not all(c > 0 for c in self.conv_channels):
            raise ValueError(f"NNConvParams requires non-empty conv_channels with all > 0, got {self.conv_channels}")
        if self.in_channels <= 0:
            raise ValueError(f"NNConvParams requires in_channels > 0, got {self.in_channels}")
        if self.kernel_size <= 0:
            raise ValueError(f"NNConvParams requires kernel_size > 0, got {self.kernel_size}")
        if self.stride <= 0:
            raise ValueError(f"NNConvParams requires stride > 0, got {self.stride}")
        if self.padding < 0:
            raise ValueError(f"NNConvParams requires padding >= 0, got {self.padding}")
        if self.pool_size <= 0:
            raise ValueError(f"NNConvParams requires pool_size > 0, got {self.pool_size}")

        # v1 square-image contract: input_dim = in_channels * side².
        if self.input_dim % self.in_channels != 0:
            raise ValueError(
                f"NNConvParams requires input_dim divisible by in_channels, "
                f"got input_dim={self.input_dim}, in_channels={self.in_channels}"
            )
        pixels = self.input_dim // self.in_channels
        side = math.isqrt(pixels)
        if side * side != pixels:
            raise ValueError(
                f"NNConvParams v1 requires a square image: input_dim/in_channels must be "
                f"a perfect square, got {pixels} (input_dim={self.input_dim}, in_channels={self.in_channels})"
            )

        # Fail-fast on spatial collapse: every Conv→Pool block must leave a
        # feature map of at least 1×1 (validated here, far from the first
        # forward — the [[params-boundary-validation]] contract).
        for i, s in enumerate(self.spatial_sizes()):
            if s < 1:
                raise ValueError(
                    f"NNConvParams conv/pool stack collapses the spatial size below 1×1 "
                    f"at block {i} (sizes: {self.spatial_sizes()}, image side {self.image_side()})"
                )

    def image_side(self) -> int:
        """Spatial side of the (square) input image."""
        return math.isqrt(self.input_dim // self.in_channels)

    def spatial_sizes(self) -> list[int]:
        """Feature-map side after each Conv→Pool block (floor arithmetic,
        matching Conv2d/MaxPool2d)."""
        sizes = []
        side = self.image_side()
        for _ in self.conv_channels:
            side = (side + 2 * self.padding - self.kernel_size) // self.stride + 1
            side = side // self.pool_size
            sizes.append(side)
        return sizes

    def flatten_dim(self) -> int:
        """Input width of the first FC layer: last block's channels × side²."""
        side = self.spatial_sizes()[-1]
        return self.conv_channels[-1] * side * side

    def state(self) -> dict:
        d = NNParams.state(self)
        # Always emitted: the resolve_from_state discriminator AND the hash
        # distinctness guard vs a plain-FeedFwd config.
        d["conv_channels"] = str(self.conv_channels)
        # Omit-when-default (module docstring).
        for key, default in _LENET_DEFAULTS.items():
            value = getattr(self, key)
            if value != default:
                d[key] = value
        return d

    @staticmethod
    def from_state(state: dict) -> NNConvParams:
        base = NNParams.from_state(state)
        return NNConvParams(
            input_dim=base.input_dim,
            output_dim=base.output_dim,
            hidden_dims=base.hidden_dims,
            dropout_prob=base.dropout_prob,
            activation=base.activation,
            activations=base.activations,
            dropout_probs=base.dropout_probs,
            n_heads=base.n_heads,
            conv_channels=ast.literal_eval(state["conv_channels"]),
            **{key: state.get(key, default) for key, default in _LENET_DEFAULTS.items()},
        )
