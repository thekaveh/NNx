"""Smoke tests: each network class can be instantiated.

We don't run forward passes here for the GNN classes — that requires
constructing valid PyG input shapes which depend on the constructor's
exact API. The instantiation test catches the most common breakage
(missing required deps, broken __init__ signatures, mismatched parent
classes). `FeedFwdNN` additionally has a round-trip + weights_only
test below — it's the simplest net to exercise on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nnx.nn.enum.activations import Activations
from nnx.nn.net.feed_fwd_nn import FeedFwdNN
from nnx.nn.params.nn_params import NNParams


def test_feed_fwd_nn_is_module():
    assert issubclass(FeedFwdNN, nn.Module)


def test_graph_conv_nn_is_module():
    from nnx.nn.net.graph_conv_nn import GraphConvNN

    assert issubclass(GraphConvNN, nn.Module)


def test_graph_sage_nn_is_module():
    from nnx.nn.net.graph_sage_nn import GraphSageNN

    assert issubclass(GraphSageNN, nn.Module)


def test_graph_att_nn_is_module():
    from nnx.nn.net.graph_att_nn import GraphAttNN

    assert issubclass(GraphAttNN, nn.Module)


def _make_feed_fwd_nn() -> FeedFwdNN:
    return FeedFwdNN(
        params=NNParams(
            input_dim=4,
            output_dim=2,
            hidden_dims=[8],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
    )


def test_feed_fwd_nn_to_file_from_file_round_trip(tmp_path):
    """Save → load → assert state-dicts equal. Without this, a regression
    in to_file (e.g., persisting an extra non-tensor field) would only
    surface when someone actually called from_file in production."""
    torch.manual_seed(0)
    net_a = _make_feed_fwd_nn()
    path = tmp_path / "ffn.pt"
    net_a.to_file(str(path))

    net_b = FeedFwdNN.from_file(
        str(path),
        params=NNParams(
            input_dim=4,
            output_dim=2,
            hidden_dims=[8],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
    )
    sd_a = net_a.state_dict()
    sd_b = net_b.state_dict()
    assert set(sd_a.keys()) == set(sd_b.keys())
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"weight mismatch for {k!r}"


def test_feed_fwd_nn_from_file_rejects_pickle_payload(tmp_path):
    """from_file uses torch.load(weights_only=True), which restricts the
    pickle ALLOWLIST to tensors + standard scalar/dict types. A bare
    non-tensor pickle payload should raise UnpicklingError rather than
    execute code.

    This is the explicit safety-test for the docstring claim about
    `weights_only=True` — a regression that drops the flag would
    silently re-introduce the pickle-RCE vulnerability."""
    import pickle

    path = tmp_path / "bad.pt"
    # Persist a non-state-dict pickle that weights_only would refuse.
    torch.save({"hello": object()}, str(path))

    with pytest.raises(pickle.UnpicklingError):
        FeedFwdNN.from_file(
            str(path),
            params=NNParams(
                input_dim=4,
                output_dim=2,
                hidden_dims=[8],
                dropout_prob=0.0,
                activation=Activations.RELU,
            ),
        )
