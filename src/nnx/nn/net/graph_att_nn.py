from __future__ import annotations

import torch_geometric as pyg
from torch import nn

from .graph_nn_base import GraphNNBase


class GraphAttNN(GraphNNBase):
    def _build_layers(self) -> nn.ModuleList:
        if self.params.n_heads is None or self.params.n_heads <= 0:
            raise ValueError(f"GraphAttNN requires NNParams.n_heads > 0, got {self.params.n_heads!r}")
        n_heads = self.params.n_heads
        dim_pairs = list(zip(self.params.dims, self.params.dims[1:], strict=False))
        return nn.ModuleList(
            [
                pyg.nn.GATConv(
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
