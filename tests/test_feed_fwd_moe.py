"""FeedFwdMoENN + Nets.FEED_FWD_MOE + NNMoEParams (#88).

Composes the two pre-existing MoE primitives (``MoELinear``,
``moe_train_step_factory``) into a buildable model: an ``NNParams`` subclass
carrying ``num_experts``/``top_k`` (hashed via ``state()``, round-tripped via
``resolve_from_state``) and a FeedFwd-shaped net whose hidden layers are
``MoELinear`` (plain ``nn.Linear`` head).
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNMoEParams,
    NNOptimParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    moe_train_step_factory,
)
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.moe import MoELinear
from nnx.nn.net.feed_fwd_moe_nn import FeedFwdMoENN
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_params import NNParams


def _moe_params(**kw) -> NNMoEParams:
    base = dict(
        input_dim=8,
        output_dim=3,
        hidden_dims=[16, 12],
        dropout_prob=0.0,
        activation=Activations.RELU,
        num_experts=4,
        top_k=2,
    )
    base.update(kw)
    return NNMoEParams(**base)


def _model(params: NNMoEParams) -> NNModel:
    return NNModel(
        net_params=params,
        params=NNModelParams(net=Nets.FEED_FWD_MOE, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


# ---------------------------------------------------------------------------
# Params: validation + state round-trip + hash distinctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_experts", [0, 1])
def test_moe_params_rejects_fewer_than_two_experts(num_experts):
    with pytest.raises(ValueError, match="num_experts >= 2"):
        _moe_params(num_experts=num_experts, top_k=1)


def test_moe_params_accepts_two_experts():
    params = _moe_params(num_experts=2, top_k=1)
    assert params.num_experts == 2
    assert params.top_k == 1


@pytest.mark.parametrize("hidden_dims", [None, []])
def test_moe_params_requires_at_least_one_hidden_layer(hidden_dims):
    with pytest.raises(ValueError, match="at least one hidden layer"):
        _moe_params(hidden_dims=hidden_dims)


def test_moe_params_from_state_rejects_empty_hidden_layers():
    state = _moe_params(hidden_dims=[8]).state()
    state["hidden_dims"] = "[]"
    with pytest.raises(ValueError, match="at least one hidden layer"):
        NNParams.resolve_from_state(state)


def test_plain_nn_params_still_accepts_no_hidden_layers():
    assert NNParams(input_dim=8, output_dim=3, hidden_dims=[], dropout_prob=0.0).hidden_dims == []
    assert NNParams(input_dim=8, output_dim=3, hidden_dims=None, dropout_prob=0.0).hidden_dims is None


def test_moe_params_from_state_rejects_single_expert():
    state = _moe_params(num_experts=2, top_k=1).state()
    state["num_experts"] = 1
    with pytest.raises(ValueError, match="num_experts >= 2"):
        NNParams.resolve_from_state(state)


def test_moe_params_top_k_validation():
    with pytest.raises(ValueError, match="top_k"):
        _moe_params(top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        _moe_params(num_experts=2, top_k=3)  # top_k > num_experts


def test_moe_params_state_round_trip():
    p = _moe_params(num_experts=6, top_k=3)
    s = p.state()
    assert s["num_experts"] == 6
    assert s["top_k"] == 3
    back = NNParams.resolve_from_state(s)
    assert isinstance(back, NNMoEParams)
    assert back.num_experts == 6
    assert back.top_k == 3
    assert back.state() == s


def test_moe_state_differs_from_plain_config():
    """NNRun.id hashes net.state() — the MoE fields must make it distinct from
    the equivalent plain-NNParams config (the silent-collision guard)."""
    plain = NNParams(input_dim=8, output_dim=3, hidden_dims=[16, 12], dropout_prob=0.0, activation=Activations.RELU)
    assert _moe_params().state() != plain.state()


def test_moe_top_k_default_omitted_from_state():
    """top_k=2 is the default → omit-when-default keeps future configs stable."""
    s = _moe_params(top_k=2).state()
    assert "top_k" not in s
    assert NNParams.resolve_from_state(s).top_k == 2  # default restored


# ---------------------------------------------------------------------------
# Net: MoELinear hidden layers + plain Linear head
# ---------------------------------------------------------------------------


def test_net_contains_moe_linear_hidden_layers():
    model = _model(_moe_params())
    assert isinstance(model.net, FeedFwdMoENN)
    hidden = list(model.net.layers[:-1])
    assert len(hidden) == 2
    assert all(isinstance(layer, MoELinear) for layer in hidden)
    # classifier head stays a plain Linear (routing the head is atypical)
    assert isinstance(model.net.layers[-1], torch.nn.Linear)


def test_minimal_valid_moe_contains_an_expert_layer():
    model = _model(_moe_params(hidden_dims=[4]))
    assert any(isinstance(layer, MoELinear) for layer in model.net.modules())


def test_forward_shape_and_aux_loss():
    model = _model(_moe_params())
    X = torch.randn(5, 8)
    out = model.net(X)
    assert out.shape == (5, 3)
    # every MoELinear recorded a finite aux loss on forward
    for layer in model.net.layers[:-1]:
        assert layer.last_aux_loss is not None
        assert torch.isfinite(layer.last_aux_loss)


# ---------------------------------------------------------------------------
# Training: moe_train_step_factory over the built net
# ---------------------------------------------------------------------------


def test_moe_train_step_produces_finite_loss_with_nonzero_aux(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    X = torch.randn(32, 8)
    y = torch.randint(0, 3, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)
    model = _model(_moe_params())

    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        ),
        train_step_fn=moe_train_step_factory(aux_loss_weight=0.01),
    )
    assert run.idps
    last = run.idps[-1].train_edp
    assert last.loss is not None and torch.isfinite(torch.tensor(last.loss))
    # aux term contributed: at least one layer's last_aux_loss is non-zero
    assert any(float(layer.last_aux_loss.detach()) != 0.0 for layer in model.net.layers[:-1])


def test_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    X = torch.randn(16, 8)
    y = torch.randint(0, 3, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)
    model = _model(_moe_params())
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        ),
        train_step_fn=moe_train_step_factory(),
    )
    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert ckpt is not None
    assert isinstance(ckpt.net_params, NNMoEParams)  # resolve_from_state dispatch
    reloaded = NNModel.from_checkpoint(ckpt)
    assert isinstance(reloaded.net, FeedFwdMoENN)
    # keys match exactly
    assert set(reloaded.net.state_dict().keys()) == set(ckpt.net_state.keys())
