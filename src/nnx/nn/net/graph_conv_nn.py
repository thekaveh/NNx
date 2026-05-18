import torch
import torch_geometric as pyg
import torch.nn.functional as F

from torch import nn
from ..params.nn_params import NNParams
from ..enum.activations import Activations

class GraphConvNN(nn.Module):
    def __init__(self, params: NNParams):
        super(GraphConvNN, self).__init__()
        
        self.params = params
        
        self.layers = nn.ModuleList(
            [
                pyg.nn.GCNConv(
                    in_channels=in_dim
                    , out_channels=out_dim
                )   for in_dim, out_dim in zip(self.params.dims, self.params.dims[1:])
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
        return f"GraphConvNN={self.params}"