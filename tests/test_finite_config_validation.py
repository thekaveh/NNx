"""FIX-020: public real-valued hyperparameters reject NaN and ±infinity at
their configuration boundary — before an optimizer, a run directory, or a
RoPE table is created — while meaningful zeros and ``None`` sentinels stay
valid and serialized state is unchanged.

The inventory below is the contract: every (constructor, field, domain)
row is parametrized over NaN, +inf and -inf, asserts the error names the
field, and pairs with valid near-boundary values.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from nnx import (
    Activations,
    Devices,
    DoRALinear,
    EarlyStopping,
    LoRALinear,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParamGroupSpec,
    NNParams,
    NNSchedulerParams,
    NNTrainerParams,
    NNTrainParams,
    NNTransformerParams,
    Optims,
)
from nnx.nn.net.transformer_layers import RMSNorm, RoPE

NONFINITE = [float("nan"), float("inf"), float("-inf")]
_ADAM = dict(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0)
_SGD = dict(name=Optims.SGD, max_lr=1e-3, momentum=0.9, weight_decay=0.0)
_SCHED = dict(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3)
_TP = dict(
    input_dim=16,
    output_dim=16,
    dropout_prob=0.0,
    vocab_size=16,
    n_layers=1,
    n_heads=2,
    d_model=16,
    ffn_mult=2,
    max_seq_len=8,
)


def _optim(field: str, value):
    return NNOptimParams(**{**_ADAM, field: value})


def _sgd_momentum(value):
    return NNOptimParams(**{**_SGD, "momentum": value})


def _adam_beta(index: int, value):
    betas = [0.9, 0.999]
    betas[index] = value
    return NNOptimParams(**{**_ADAM, "momentum": tuple(betas)})


def _sched(field: str, value):
    return NNSchedulerParams(**{**_SCHED, field: value})


def _group(field: str, value):
    return NNParamGroupSpec(name_pattern="x.*", **{field: value})


INVENTORY = [
    pytest.param(lambda v: _optim("max_lr", v), "max_lr", id="optim.max_lr"),
    pytest.param(lambda v: _optim("weight_decay", v), "weight_decay", id="optim.weight_decay"),
    pytest.param(lambda v: _optim("grad_clip_norm", v), "grad_clip_norm", id="optim.grad_clip_norm"),
    pytest.param(_sgd_momentum, "momentum", id="optim.sgd_momentum"),
    pytest.param(lambda v: _adam_beta(0, v), "momentum", id="optim.adam_beta1"),
    pytest.param(lambda v: _adam_beta(1, v), "momentum", id="optim.adam_beta2"),
    pytest.param(lambda v: _sched("factor", v), "factor", id="sched.factor"),
    pytest.param(lambda v: _sched("min_lr", v), "min_lr", id="sched.min_lr"),
    pytest.param(lambda v: _sched("threshold", v), "threshold", id="sched.threshold"),
    pytest.param(lambda v: _sched("max_lr", v), "max_lr", id="sched.max_lr"),
    pytest.param(lambda v: _group("lr", v), "lr", id="group.lr"),
    pytest.param(lambda v: _group("lr_multiplier", v), "lr_multiplier", id="group.lr_multiplier"),
    pytest.param(lambda v: _group("weight_decay", v), "weight_decay", id="group.weight_decay"),
    pytest.param(lambda v: NNTransformerParams(**_TP, rope_base=v), "rope_base", id="transformer.rope_base"),
    pytest.param(lambda v: RoPE(dim=4, max_seq_len=4, base=v), "base", id="RoPE.base"),
    pytest.param(lambda v: RMSNorm(4, eps=v), "eps", id="RMSNorm.eps"),
    pytest.param(lambda v: EarlyStopping(min_delta=v), "min_delta", id="EarlyStopping.min_delta"),
    pytest.param(lambda v: LoRALinear(nn.Linear(4, 4), r=2, alpha=v), "alpha", id="LoRA.alpha"),
    pytest.param(lambda v: DoRALinear(nn.Linear(4, 4), r=2, alpha=v), "alpha", id="DoRA.alpha"),
]


@pytest.mark.parametrize(("build", "field"), INVENTORY)
@pytest.mark.parametrize("value", NONFINITE, ids=["nan", "+inf", "-inf"])
def test_nonfinite_values_are_rejected_with_field_name(build, field, value):
    with pytest.raises(ValueError, match=field):
        build(value)


@pytest.mark.parametrize(("build", "field"), INVENTORY)
def test_non_real_values_are_rejected(build, field):
    """Booleans and strings are not real numbers: neither is coerced."""
    with pytest.raises((ValueError, TypeError), match=field):
        build(True)
    with pytest.raises((ValueError, TypeError), match=field):
        build("0.5")


def test_meaningful_zeros_and_none_sentinels_stay_valid():
    """Existing zero / None semantics are unchanged and hash identically."""
    zero_lr = NNOptimParams(**{**_ADAM, "max_lr": 0.0, "weight_decay": 0.0})
    assert zero_lr.max_lr == 0.0 and zero_lr.grad_clip_norm is None
    assert zero_lr.state() == {"max_lr": 0.0, "momentum": "(0.9, 0.999)", "name": "adam", "weight_decay": 0.0}
    assert NNSchedulerParams(**{**_SCHED, "threshold": 0.0, "min_lr": 0.0}).threshold == 0.0
    assert NNSchedulerParams(**_SCHED).max_lr is None
    assert NNParamGroupSpec(name_pattern="x.*", weight_decay=0.0).weight_decay == 0.0
    assert NNParamGroupSpec(name_pattern="x.*").lr is None
    assert NNOptimParams(**{**_SGD, "momentum": 0.0}).momentum == 0.0
    assert EarlyStopping(min_delta=0.0).min_delta == 0.0
    assert RMSNorm(4, eps=0.0).eps == 0.0
    assert NNTransformerParams(**_TP, rope_base=1.0).rope_base == 1.0
    assert "rope_base" not in NNTransformerParams(**_TP).state()  # default still omitted


@pytest.mark.parametrize("base", [1.0, 10.0, 10000.0, 1e6])
def test_valid_rope_bases_produce_finite_tables(base):
    rope = RoPE(dim=8, max_seq_len=16, base=base)
    assert torch.isfinite(rope.cos_cached).all() and torch.isfinite(rope.sin_cached).all()
    params = NNTransformerParams(**_TP, rope_base=base)
    assert params.rope_base == base


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_rope_base_domain_is_positive(value):
    with pytest.raises(ValueError, match="rope_base"):
        NNTransformerParams(**_TP, rope_base=value)
    with pytest.raises(ValueError, match="base"):
        RoPE(dim=4, max_seq_len=4, base=value)


def test_negative_rmsnorm_eps_is_rejected():
    with pytest.raises(ValueError, match="eps"):
        RMSNorm(4, eps=-1e-6)


def test_replace_and_from_state_routes_validate_too():
    """`dataclasses.replace`, `from_state` and the trainer params route all
    pass through `__post_init__`, so an invalid value embedded in run
    state fails with its field identity before any optimizer exists."""
    valid = NNOptimParams.builder().adam(max_lr=0.0, weight_decay=0.0).build()
    with pytest.raises(ValueError, match="grad_clip_norm"):
        replace(valid, grad_clip_norm=float("nan"))
    bad_state = {**valid.state(), "max_lr": float("nan")}
    with pytest.raises(ValueError, match="max_lr"):
        NNOptimParams.from_state(bad_state)
    train = NNTrainParams(n_epochs=1, optim=valid, scheduler=NNSchedulerParams(**_SCHED))
    bad_train_state = train.state()
    bad_train_state["scheduler"] = {**bad_train_state["scheduler"], "factor": float("inf")}
    with pytest.raises(ValueError, match="factor"):
        NNTrainParams.from_state(bad_train_state)
    with pytest.raises(ValueError, match="max_lr"):
        NNTrainerParams(n_epochs=1, optims={"main": replace(valid, max_lr=float("inf"))})


def test_invalid_config_fails_before_run_directory_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    with pytest.raises(ValueError, match="max_lr"):
        NNTrainParams(n_epochs=1, optim=NNOptimParams(**{**_ADAM, "max_lr": float("nan")}))
    assert not (tmp_path / "runs").exists()


def test_group_lr_product_must_stay_finite():
    """A finite multiplier can still overflow the resolved group LR; the
    resolved value is validated before the optimizer is built."""
    from nnx.finetune.param_groups import build_param_groups

    net = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    huge = NNParamGroupSpec(name_pattern="0.*", lr_multiplier=1e308)
    with pytest.raises(ValueError, match="lr"):
        build_param_groups(net, [huge], default_lr=1e10, default_weight_decay=0.0)
    ok = build_param_groups(
        net, [NNParamGroupSpec(name_pattern="0.*", lr_multiplier=2.0)], default_lr=1e-3, default_weight_decay=0.0
    )
    assert all(math.isfinite(g["lr"]) for g in ok)


def test_tiny_valid_optimization_and_transformer_forward_stay_finite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    X, y = torch.randn(16, 4), torch.randint(0, 2, (16,))
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=DataLoader(TensorDataset(X, y), batch_size=8),
            optim=NNOptimParams(**{**_ADAM, "max_lr": 1e-3, "grad_clip_norm": 1.0}),
            scheduler=NNSchedulerParams(**{**_SCHED, "threshold": 0.0}),
        )
    )
    assert all(idp.train_edp.loss is not None and math.isfinite(idp.train_edp.loss) for idp in run.idps)
    tp = NNTransformerParams(**_TP, rope_base=10000.0)
    net = Nets.TRANSFORMER(params=tp)
    out = net(torch.randint(0, 16, (2, 4)))
    assert torch.isfinite(out).all()
