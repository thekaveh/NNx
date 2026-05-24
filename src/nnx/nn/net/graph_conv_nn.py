import torch_geometric as pyg
from torch import nn

from .graph_nn_base import GraphNNBase


class GraphConvNN(GraphNNBase):
    def _build_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            pyg.nn.GCNConv(in_channels=in_dim, out_channels=out_dim)
            for in_dim, out_dim in zip(self.params.dims, self.params.dims[1:], strict=False)
        ])

    def __str__(self) -> str:
        return f"GraphConvNN={self.params}"
