from __future__ import annotations

from torch import nn

from ..._optional import require
from .graph_nn_base import GraphNNBase

# The graph extra (FEAT-031): importing this module needs torch_geometric.
GATConv = require("torch_geometric.nn", "GraphAttNN").GATConv


class GraphAttNN(GraphNNBase):
    """Node-level graph net of PyG ``GATConv`` layers (the ``graph`` extra:
    ``pip install "thekaveh-nnx[graph]"``)."""

    def _build_layers(self) -> nn.ModuleList:
        if self.params.n_heads is None or self.params.n_heads <= 0:
            raise ValueError(f"GraphAttNN requires NNParams.n_heads > 0, got {self.params.n_heads!r}")
        n_heads = self.params.n_heads
        dim_pairs = list(zip(self.params.dims, self.params.dims[1:], strict=False))
        return nn.ModuleList(
            [
                GATConv(
                    out_channels=out_dim,
                    heads=n_heads,
                    concat=(idx_dim != len(dim_pairs) - 1),
                    in_channels=in_dim if idx_dim == 0 else in_dim * n_heads,
                )
                for idx_dim, (in_dim, out_dim) in enumerate(dim_pairs)
            ]
        )

    def __str__(self) -> str:
        return f"GraphAttNN={self.params}"
