"""LeNet-style convolutional classifier.

Conv→Pool→…→Flatten→FC for small square images (MNIST/FashionMNIST scale).
Same interface contract as :class:`FeedFwdNN` (``forward(x)→logits``,
``unpack_batch``), accepting either ``(B, C, H, W)`` images or the flattened
``(B, input_dim)`` rows vision loaders may hand a feed-forward net — the
forward reshapes to the square image derived from the params.

Conv blocks use the net-wide scalar ``activation``; the per-layer
``activations``/``dropout_probs`` lists (#85) stay FC-only — their length
contract is ``len == len(hidden_dims)``, which counts FC hidden layers.

v1 scope: LeNet-style only; no ResNet/skip connections.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..params.nn_conv_params import NNConvParams
from ._pickling import restore_params_property


class ConvNN(nn.Module):
    """LeNet-style convolutional classifier built from :class:`NNConvParams`:
    conv and pool blocks, then fully connected layers. ``forward(x)`` takes
    ``(B, C, H, W)`` images or flattened rows and returns logits."""

    _params: NNConvParams
    _side: int
    _in_channels: int
    _pool_size: int
    _conv_activation: Optional[Callable[[torch.Tensor], torch.Tensor]]
    _hidden: tuple[tuple[Callable[[torch.Tensor], torch.Tensor], float], ...]

    def __init__(self, params: NNConvParams):
        super().__init__()

        self.params = params

        channels = [params.in_channels, *params.conv_channels]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=params.kernel_size,
                    stride=params.stride,
                    padding=params.padding,
                )
                for in_c, out_c in zip(channels, channels[1:], strict=False)
            ]
        )

        fc_dims = [params.flatten_dim()]
        fc_dims += params.hidden_dims if params.hidden_dims is not None else []
        fc_dims += [params.output_dim]
        self.fcs = nn.ModuleList(
            [
                nn.Linear(in_features=in_dim, out_features=out_dim)
                for in_dim, out_dim in zip(fc_dims, fc_dims[1:], strict=False)
            ]
        )

    @property
    def params(self) -> NNConvParams:
        """The net's configuration; setting it (as model surgery does)
        re-resolves the per-layer state the forward reads."""
        return self._params

    @params.setter
    def params(self, params: NNConvParams) -> None:
        # Everything the forward reads from the params, resolved whenever
        # they are set (model surgery replaces them, as for FeedFwdNN): the
        # forward touches neither the params object nor the Activations enum,
        # which torch.compile cannot trace on older torch. Plain attributes:
        # nothing is registered as a module or saved.
        self.__dict__["_params"] = params
        self.__dict__["_side"] = params.image_side()
        self.__dict__["_in_channels"] = params.in_channels
        self.__dict__["_pool_size"] = params.pool_size
        self.__dict__["_conv_activation"] = params.activation() if params.activation is not None else None
        n_hidden = len(params.hidden_dims) if params.hidden_dims is not None else 0
        self.__dict__["_hidden"] = tuple((params.activation_for(i)(), params.dropout_for(i)) for i in range(n_hidden))

    def __setstate__(self, state: dict[str, Any]) -> None:
        # A whole-module pickle from 0.3.x or 0.4.x stored params as a plain
        # attribute, and 0.3.x no cached fields: set it through the property
        # (FIX-065).
        restore_params_property(self, state)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape(X.size(0), self._in_channels, self._side, self._side)  # channels_last too

        activation = self._conv_activation
        if activation is None:
            raise ValueError("ConvNN requires a scalar activation for convolution blocks")
        for conv in self.convs:
            X = conv(X)
            X = activation(X)
            X = F.max_pool2d(X, kernel_size=self._pool_size)

        X = X.reshape(X.size(0), -1)

        for fc, (hidden_activation, dropout) in zip(self.fcs[:-1], self._hidden, strict=True):
            X = fc(X)
            X = hidden_activation(X)
            X = F.dropout(X, p=dropout, training=self.training)

        return self.fcs[-1](X)

    def unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            X, Y = batch
            return (X,), Y
        raise TypeError(f"ConvNN.unpack_batch expects a (list/tuple) batch; got {type(batch).__name__}.")

    def __str__(self):
        return f"ConvNN={self.params}"
