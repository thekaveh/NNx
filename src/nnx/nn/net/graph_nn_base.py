"""Shared base for the GNN architectures.

GraphConvNN, GraphSageNN, GraphAttNN previously had ~95% duplicate code.
The common forward loop + unpack_batch live here; subclasses override
_build_layers() to provide the PyG conv constructor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..params.nn_params import NNParams


class GraphNNBase(nn.Module):
    """Abstract base for GNN architectures.

    Subclasses must implement `_build_layers()` returning an `nn.ModuleList`
    of PyG message-passing layers. The forward loop applies all-but-last
    layers with the configured activation + dropout, then a bare final layer.
    """

    def __init__(self, params: NNParams):
        super().__init__()
        self.params = params
        self.layers = self._build_layers()

    def _build_layers(self) -> nn.ModuleList:
        raise NotImplementedError("subclass must implement _build_layers()")

    def forward(self, X: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            X = layer(X, E)
            X = self.params.activation()(X)
            X = F.dropout(X, p=self.params.dropout_prob, training=self.training)
        return self.layers[-1](X, E)

    def unpack_batch(self, batch) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return (batch.x, batch.edge_index), batch.y
