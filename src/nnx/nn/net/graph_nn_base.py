"""Shared base for the GNN architectures.

GraphConvNN, GraphSageNN, GraphAttNN previously had ~95% duplicate code.
The common forward loop + unpack_batch live here; subclasses override
_build_layers() to provide the PyG conv constructor.
"""

from __future__ import annotations

from typing import Optional

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
        for i, layer in enumerate(self.layers[:-1]):
            X = layer(X, E)
            X = self.params.activation_for(i)()(X)
            X = F.dropout(X, p=self.params.dropout_for(i), training=self.training)
        return self.layers[-1](X, E)

    def unpack_batch(self, batch) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return (batch.x, batch.edge_index), batch.y

    def seed_count(self, batch) -> Optional[int]:
        """Number of seed rows at the head of a NeighborLoader subgraph.

        NeighborLoader puts the ``batch_size`` seed nodes first and
        appends their sampled neighbors — which can belong to *other*
        splits. Loss and metrics must be computed on the seed rows only;
        scoring neighbor rows leaks val/test labels into the training
        loss and train labels into val metrics.

        Returns None (no slicing) for anything that isn't a
        NeighborLoader subgraph: plain full-graph ``Data`` has no
        ``batch_size``, and a multi-graph ``Batch.from_data_list``
        collation DOES carry ``batch_size`` (= ``num_graphs``) but no
        ``input_id`` — slicing there would truncate node-level output
        to the graph count. ``input_id`` is the NeighborLoader-specific
        marker (the seed indices), so it gates the slice.
        """
        if getattr(batch, "input_id", None) is None:
            return None
        n = getattr(batch, "batch_size", None)
        return int(n) if n is not None else None
