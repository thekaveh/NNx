from __future__ import annotations

from torch import nn
from torch_geometric.nn import SAGEConv

from .graph_nn_base import GraphNNBase


class GraphSageNN(GraphNNBase):
    def _build_layers(self) -> nn.ModuleList:
        return nn.ModuleList(
            [
                SAGEConv(in_channels=in_dim, out_channels=out_dim)
                for in_dim, out_dim in zip(self.params.dims, self.params.dims[1:], strict=False)
            ]
        )

    def __str__(self) -> str:
        return f"GraphSageNN={self.params}"
