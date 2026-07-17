"""Mixture-of-Experts feed-forward classifier (#88).

Same shape contract as :class:`FeedFwdNN` (flatten → hidden stack → linear
head, ``unpack_batch``), but every HIDDEN layer is a
:class:`~nnx.nn.moe.MoELinear` (top-k routed experts). The classifier head
stays a plain ``nn.Linear`` — routing the head is atypical and would
complicate the aux-loss story.

Pairs with :func:`nnx.moe_train_step_factory`, which sums
``layer.last_aux_loss`` over every ``MoELinear`` in the net (load-balancing
term). Without this net there was no public-params path that produced
``MoELinear`` layers, so the factory silently degenerated to plain
supervised training.

Tutorial-scale perf note: ``MoELinear.forward`` dispatches experts in
O(top_k · num_experts) Python loops — fine at MNIST/iris scale, a footgun
for large expert counts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..moe import MoELinear
from ..params.nn_moe_params import NNMoEParams


class FeedFwdMoENN(nn.Module):
    def __init__(self, params: NNMoEParams):
        super().__init__()

        self.params = params

        dims = params.dims
        hidden_pairs = list(zip(dims, dims[1:], strict=False))[:-1]
        head_in, head_out = dims[-2], dims[-1]

        self.layers = nn.ModuleList(
            [
                MoELinear(
                    in_features=in_dim,
                    out_features=out_dim,
                    num_experts=params.num_experts,
                    top_k=params.top_k,
                )
                for in_dim, out_dim in hidden_pairs
            ]
            + [nn.Linear(in_features=head_in, out_features=head_out)]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.view(X.size(0), -1)

        for i, layer in enumerate(self.layers[:-1]):
            X = layer(X)
            X = self.params.activation_for(i)()(X)
            X = F.dropout(X, p=self.params.dropout_for(i), training=self.training)

        return self.layers[-1](X)

    def unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            X, Y = batch
            return (X,), Y
        raise TypeError(f"FeedFwdMoENN.unpack_batch expects a (list/tuple) batch; got {type(batch).__name__}.")

    def __str__(self):
        return f"FeedFwdMoENN={self.params}"
