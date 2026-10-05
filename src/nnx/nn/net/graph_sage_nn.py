from __future__ import annotations

from torch import nn

from ..._optional import require
from .graph_nn_base import GraphNNBase

# The graph extra (FEAT-031): importing this module needs torch_geometric.
SAGEConv = require("torch_geometric.nn", "GraphSageNN").SAGEConv


class GraphSageNN(GraphNNBase):
    """Node-level graph net of PyG ``SAGEConv`` layers (the ``graph`` extra:
    ``pip install "thekaveh-nnx[graph]"``)."""

    def _build_layers(self) -> nn.ModuleList:
        return nn.ModuleList(
            [
                SAGEConv(in_channels=in_dim, out_channels=out_dim)
                for in_dim, out_dim in zip(self.params.dims, self.params.dims[1:], strict=False)
            ]
        )

    def __str__(self) -> str:
        return f"GraphSageNN={self.params}"
