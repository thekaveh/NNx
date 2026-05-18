import torch
import torch_geometric as pyg
import torch.nn.functional as F

from torch import nn
from ..params.nn_params import NNParams

class GraphAttNN(nn.Module):
    def __init__(self, params: NNParams):
        super(GraphAttNN, self).__init__()
        
        assert params.n_heads is not None and params.n_heads > 0
        
        self.params = params
        
        dim_pairs = list(zip(self.params.dims, self.params.dims[1:]))
        
        self.layers = nn.ModuleList(
            [
                pyg.nn.GATConv(
                    out_channels=out_dim
                    , heads=self.params.n_heads
                    , concat=True if (idx_dim != len(dim_pairs) - 1) else False
                    , in_channels=in_dim if idx_dim == 0 else in_dim * self.params.n_heads
                )   for idx_dim, (in_dim, out_dim) in enumerate(dim_pairs)
            ]
        )

    def forward(self, X: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            X = layer(X, E)
            X = self.params.activation()(X)
            X = F.dropout(X, p=self.params.dropout_prob, training=self.training)
                
        X = self.layers[-1](X, E)
        
        return X
    
    def unpack_batch(self, batch):
        X, E, Y = batch.x, batch.edge_index, batch.y
        
        return (X, E), Y
    
    def __str__(self):
        return f"GraphAttNN={self.params}"