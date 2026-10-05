from __future__ import annotations

from torch import nn

from ..._optional import require
from .graph_nn_base import GraphNNBase

# The graph extra (FEAT-031): importing this module needs torch_geometric.
GCNConv = require("torch_geometric.nn", "GraphConvNN").GCNConv


class GraphConvNN(GraphNNBase):
    """Node-level graph net of PyG ``GCNConv`` layers (the ``graph`` extra:
    ``pip install "thekaveh-nnx[graph]"``)."""

    def _build_layers(self) -> nn.ModuleList:
        return nn.ModuleList(
            [
                GCNConv(in_channels=in_dim, out_channels=out_dim)
                for in_dim, out_dim in zip(self.params.dims, self.params.dims[1:], strict=False)
            ]
        )

    def __str__(self) -> str:
        return f"GraphConvNN={self.params}"
