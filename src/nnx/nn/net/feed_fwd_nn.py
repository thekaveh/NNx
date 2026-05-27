import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import nn

from ..params.nn_params import NNParams


class FeedFwdNN(nn.Module):
    def __init__(self, params: NNParams):
        super().__init__()

        self.params = params

        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=in_dim
                    , out_features=out_dim
                )   for in_dim, out_dim in zip(self.params.dims, self.params.dims[1:], strict=False)
            ]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.view(X.size(0), -1)

        for layer in self.layers[:-1]:
            X = layer(X)
            X = self.params.activation()(X)
            X = F.dropout(X, p=self.params.dropout_prob, training=self.training)

        X = self.layers[-1](X)

        return X

    def unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            X, Y = batch
        elif isinstance(batch, pyg.data.data.Data):
            X, Y = batch.x, batch.y
        else:
            raise TypeError("The input 'batch' must be either a tuple or an instance of torch_geometric.data.data.Data.")

        return (X,), Y

    def __str__(self):
        return f"FeedFwdNN={self.params}"

    def to_file(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @staticmethod
    def from_file(path: str, params: NNParams) -> 'FeedFwdNN':
        # weights_only=True: a state-dict is plain tensors + standard
        # scalar/dict types — the strict loader works and removes the
        # arbitrary-code-execution risk on user-supplied paths. Matches
        # NNCheckpoint.load_optimizer_state and load_pretrained.
        net = FeedFwdNN(params)
        net.load_state_dict(torch.load(path, weights_only=True))

        return net

    @staticmethod
    def from_state(state_dict: dict, params: NNParams) -> 'FeedFwdNN':
        net = FeedFwdNN(params)
        net.load_state_dict(state_dict)

        return net
