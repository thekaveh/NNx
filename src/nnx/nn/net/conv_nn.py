"""LeNet-style convolutional classifier (#89).

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

import torch
import torch.nn.functional as F
from torch import nn

from ..params.nn_conv_params import NNConvParams


class ConvNN(nn.Module):
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        side = self.params.image_side()
        X = X.view(X.size(0), self.params.in_channels, side, side)

        for conv in self.convs:
            X = conv(X)
            X = self.params.activation()(X)
            X = F.max_pool2d(X, kernel_size=self.params.pool_size)

        X = X.view(X.size(0), -1)

        for i, fc in enumerate(self.fcs[:-1]):
            X = fc(X)
            X = self.params.activation_for(i)()(X)
            X = F.dropout(X, p=self.params.dropout_for(i), training=self.training)

        return self.fcs[-1](X)

    def unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            X, Y = batch
            return (X,), Y
        raise TypeError(f"ConvNN.unpack_batch expects a (list/tuple) batch; got {type(batch).__name__}.")

    def __str__(self):
        return f"ConvNN={self.params}"
