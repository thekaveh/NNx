from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..params.nn_params import NNParams
from ._pickling import restore_params_property


class FeedFwdNN(nn.Module):
    """Feed-forward classifier built from :class:`NNParams`: flatten, a stack
    of hidden ``nn.Linear`` layers (per-layer activation and dropout), then a
    linear head. ``forward(x)`` returns logits."""

    _params: NNParams
    _hidden: tuple[tuple[Callable[[torch.Tensor], torch.Tensor], float], ...]

    def __init__(self, params: NNParams):
        super().__init__()

        self.params = params

        self.layers = nn.ModuleList(
            [
                nn.Linear(in_features=in_dim, out_features=out_dim)
                for in_dim, out_dim in zip(self.params.dims, self.params.dims[1:], strict=False)
            ]
        )

    @property
    def params(self) -> NNParams:
        return self._params

    @params.setter
    def params(self, params: NNParams) -> None:
        # Each hidden layer's activation function (stateless functionals) and
        # dropout probability, resolved whenever the params are set (model
        # surgery replaces them): the forward touches neither the params
        # object nor the Activations enum, which torch.compile cannot trace
        # on older torch (FEAT-029). Plain attributes: nothing is registered
        # as a module or saved.
        self.__dict__["_params"] = params
        self.__dict__["_hidden"] = tuple(
            (params.activation_for(i)(), params.dropout_for(i)) for i in range(len(params.dims) - 2)
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        # A whole-module pickle from 0.3.x stored params as a plain attribute
        # and no cached fields: set it through the property (FIX-065).
        restore_params_property(self, state)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape(X.size(0), -1)  # a non-contiguous input (a transposed array) too

        for layer, (activation, dropout) in zip(self.layers[:-1], self._hidden, strict=True):
            X = layer(X)
            X = activation(X)
            X = F.dropout(X, p=dropout, training=self.training)

        X = self.layers[-1](X)

        return X

    def seed_count(self, batch) -> Optional[int]:
        """The seed rows of a PyG NeighborLoader / full-graph split batch,
        as ``GraphNNBase.seed_count``: this net also reads PyG node batches,
        whose other rows belong to other splits and must not be scored."""
        if isinstance(batch, (list, tuple)) or getattr(batch, "input_id", None) is None:
            return None
        n = getattr(batch, "batch_size", None)
        return int(n) if n is not None else None

    def unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            X, Y = batch
            return (X,), Y
        # PyG batches go through the GNN subclasses (GraphConvNN /
        # GraphSageNN / GraphAttNN), but we keep the branch here for
        # back-compat with the original contract. A PyG batch exists only
        # once torch_geometric is imported, so look it up rather than
        # import it: the graph stack is an extra (FEAT-031).
        pyg_data = sys.modules.get("torch_geometric.data.data")
        if pyg_data is not None and isinstance(batch, pyg_data.Data):
            if getattr(batch, "edge_label_index", None) is not None:
                raise ValueError(
                    "an edge-label (link) batch asks about candidate edges; a node-level FeedFwdNN cannot score it "
                    "— use nnx.link_tasks.LinkTask with a LinkPredictor"
                )
            if getattr(batch, "graph_id", None) is not None:
                # A batch of whole graphs (FEAT-026): one target per graph, never per node.
                raise ValueError(
                    "a graph-collection batch (graph_id) holds whole graphs, one target per graph; a node-level "
                    "FeedFwdNN cannot score it — classify graphs with nnx.graph_tasks.GraphClassifier"
                )
            X, Y = batch.x, batch.y
            return (X,), Y
        raise TypeError(
            "FeedFwdNN.unpack_batch expects a (list/tuple) batch or a "
            "torch_geometric.data.Data instance; got "
            f"{type(batch).__name__}."
        )

    def __str__(self):
        return f"FeedFwdNN={self.params}"

    def to_file(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @staticmethod
    def from_file(path: str, params: NNParams, map_location="cpu") -> FeedFwdNN:
        # weights_only=True: a state-dict is plain tensors + standard
        # scalar/dict types — the strict loader works and removes the
        # arbitrary-code-execution risk on user-supplied paths. Matches
        # NNCheckpoint.load_optimizer_state and load_pretrained.
        net = FeedFwdNN(params)
        net.load_state_dict(torch.load(path, weights_only=True, map_location=map_location))

        return net

    @staticmethod
    def from_state(state_dict: dict, params: NNParams) -> FeedFwdNN:
        net = FeedFwdNN(params)
        net.load_state_dict(state_dict)

        return net
