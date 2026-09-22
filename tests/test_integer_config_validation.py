"""FIX-021: public configuration *counts* are validated as integers at
construction — ``numbers.Integral`` (NumPy integers included) normalized
to plain ``int``; booleans, every float (``2.0`` included), strings and
non-finite values rejected with a field-named ``ValueError`` — before any
``range``, modulo, ``isqrt`` or layer allocation. Real-valued knobs keep
their FIX-020 domains; dataset batch-size tuples belong to FIX-022.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    EarlyStopping,
    Losses,
    Nets,
    NNConvParams,
    NNModel,
    NNModelParams,
    NNMoEParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNSchedulerParams,
    NNTrainerParams,
    NNTrainParams,
    NNTransformerParams,
    Optims,
)
from nnx.nn.net.transformer_layers import RoPE

BAD_COUNTS = [
    pytest.param(2.5, id="fraction"),
    pytest.param(2.0, id="integral-float"),
    pytest.param(True, id="bool"),
    pytest.param("2", id="string"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
]
_ADAM = dict(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0)
_SCHED = dict(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3)
_BASE = dict(input_dim=4, output_dim=2, dropout_prob=0.0)
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
_CONV = dict(
    input_dim=64, output_dim=2, dropout_prob=0.0, hidden_dims=[8], conv_channels=[4], activation=Activations.RELU
)
_MOE = dict(input_dim=4, output_dim=2, dropout_prob=0.0, hidden_dims=[8], num_experts=4)


def _p(**kw):
    return NNParams(**{**_BASE, **kw})


def _tp(**kw):
    return NNTransformerParams(**{**_TP, **kw})


def _conv(**kw):
    return NNConvParams(**{**_CONV, **kw})


def _moe(**kw):
    return NNMoEParams(**{**_MOE, **kw})


def _sched(**kw):
    return NNSchedulerParams(**{**_SCHED, **kw})


INVENTORY = [
    pytest.param(lambda v: _p(input_dim=v), "input_dim", id="NNParams.input_dim"),
    pytest.param(lambda v: _p(output_dim=v), "output_dim", id="NNParams.output_dim"),
    pytest.param(lambda v: _p(hidden_dims=[8, v]), "hidden_dims", id="NNParams.hidden_dims"),
    pytest.param(lambda v: _p(n_heads=v), "n_heads", id="NNParams.n_heads"),
    pytest.param(lambda v: _tp(vocab_size=v), "vocab_size", id="Transformer.vocab_size"),
    pytest.param(lambda v: _tp(n_layers=v), "n_layers", id="Transformer.n_layers"),
    pytest.param(lambda v: _tp(d_model=v), "d_model", id="Transformer.d_model"),
    pytest.param(lambda v: _tp(n_heads=v), "n_heads", id="Transformer.n_heads"),
    pytest.param(lambda v: _tp(max_seq_len=v), "max_seq_len", id="Transformer.max_seq_len"),
    pytest.param(lambda v: _tp(ffn_mult=v), "ffn_mult", id="Transformer.ffn_mult"),
    pytest.param(lambda v: _conv(conv_channels=[v]), "conv_channels", id="Conv.conv_channels"),
    pytest.param(lambda v: _conv(in_channels=v), "in_channels", id="Conv.in_channels"),
    pytest.param(lambda v: _conv(kernel_size=v), "kernel_size", id="Conv.kernel_size"),
    pytest.param(lambda v: _conv(stride=v), "stride", id="Conv.stride"),
    pytest.param(lambda v: _conv(padding=v), "padding", id="Conv.padding"),
    pytest.param(lambda v: _conv(pool_size=v), "pool_size", id="Conv.pool_size"),
    pytest.param(lambda v: _moe(num_experts=v), "num_experts", id="MoE.num_experts"),
    pytest.param(lambda v: _moe(top_k=v), "top_k", id="MoE.top_k"),
    pytest.param(
        lambda v: NNOptimParams(**_ADAM, accumulate_grad_batches=v), "accumulate_grad_batches", id="Optim.accumulate"
    ),
    pytest.param(lambda v: _sched(patience=v), "patience", id="Sched.patience"),
    pytest.param(lambda v: _sched(cooldown=v), "cooldown", id="Sched.cooldown"),
    pytest.param(lambda v: _sched(step_size=v), "step_size", id="Sched.step_size"),
    pytest.param(lambda v: _sched(T_max=v), "T_max", id="Sched.T_max"),
    pytest.param(lambda v: _sched(total_steps=v), "total_steps", id="Sched.total_steps"),
    pytest.param(lambda v: _sched(warmup_steps=v), "warmup_steps", id="Sched.warmup_steps"),
    pytest.param(lambda v: NNTrainParams(n_epochs=v), "n_epochs", id="TrainParams.n_epochs"),
    pytest.param(
        lambda v: NNTrainerParams(n_epochs=v, optims={"m": NNOptimParams(**_ADAM)}),
        "n_epochs",
        id="TrainerParams.n_epochs",
    ),
    pytest.param(lambda v: EarlyStopping(patience=v), "patience", id="EarlyStopping.patience"),
    pytest.param(lambda v: RoPE(dim=v, max_seq_len=4), "dim", id="RoPE.dim"),
    pytest.param(lambda v: RoPE(dim=4, max_seq_len=v), "max_seq_len", id="RoPE.max_seq_len"),
]


@pytest.mark.parametrize(("build", "field"), INVENTORY)
@pytest.mark.parametrize("value", BAD_COUNTS)
def test_non_integral_counts_are_rejected_with_field_name(build, field, value):
    with pytest.raises((ValueError, TypeError), match=field):
        build(value)


def test_numpy_integers_normalize_to_plain_int_and_serialize():
    """NumPy integers are accepted, normalized to ``int`` before immutable
    lists and ``state()`` (a raw ``np.int64`` used to make ``yaml.safe_dump``
    raise), and hash identically to the plain-int config."""
    base = NNOptimParams.builder().adam(max_lr=1e-3).build()
    good = replace(base, accumulate_grad_batches=np.int64(2))
    assert type(good.accumulate_grad_batches) is int
    assert good.state() == replace(base, accumulate_grad_batches=2).state()
    yaml.safe_dump(good.state())

    params = _p(
        input_dim=np.int32(4), output_dim=np.int64(2), hidden_dims=[np.int16(8), np.int64(6)], n_heads=np.int8(2)
    )
    assert type(params.input_dim) is int and type(params.output_dim) is int and type(params.n_heads) is int
    assert all(type(d) is int for d in params.hidden_dims) and params.dims == [4, 8, 6, 2]
    assert params.state() == _p(input_dim=4, output_dim=2, hidden_dims=[8, 6], n_heads=2).state()
    yaml.safe_dump(params.state())

    tp = _tp(d_model=np.int64(16), n_layers=np.int64(1), n_heads=np.int32(2))
    assert type(tp.d_model) is int and tp.state() == _tp().state()
    conv = _conv(kernel_size=np.int64(3), padding=np.int64(0), conv_channels=[np.int64(4)])
    assert type(conv.kernel_size) is int and all(type(c) is int for c in conv.conv_channels)
    assert conv.state() == _conv(kernel_size=3).state()
    train = NNTrainParams(
        n_epochs=np.int64(3), optim=good, scheduler=_sched(patience=np.int64(2), step_size=np.int64(1))
    )
    assert (
        type(train.n_epochs) is int and type(train.scheduler.patience) is int and type(train.scheduler.step_size) is int
    )
    yaml.safe_dump(train.state())


def test_zero_and_positive_boundaries_keep_their_distinct_policies():
    """Zero is valid for padding / patience / cooldown and invalid for
    epochs / accumulation / dimensions; None sentinels stay None."""
    assert _conv(padding=0).padding == 0
    assert _sched(patience=0, cooldown=0).patience == 0
    assert EarlyStopping(patience=0).patience == 0
    assert _sched().step_size is None and _sched().T_max is None
    for build, field in [
        (lambda: NNTrainParams(n_epochs=0), "n_epochs"),
        (lambda: NNTrainerParams(n_epochs=0, optims={"m": NNOptimParams(**_ADAM)}), "n_epochs"),
        (lambda: NNOptimParams(**_ADAM, accumulate_grad_batches=0), "accumulate_grad_batches"),
        (lambda: _p(input_dim=0), "input_dim"),
        (lambda: _sched(step_size=0), "step_size"),
        (lambda: _sched(warmup_steps=0), "warmup_steps"),
        (lambda: _moe(num_experts=1), "num_experts"),
        (lambda: _moe(top_k=5), "top_k"),
    ]:
        with pytest.raises(ValueError, match=field):
            build()


def test_transformer_head_dim_must_be_even_and_integral():
    """`d_model % n_heads == 0` is not enough: RoPE needs an even head
    width, and the divisibility check must run on validated integers."""
    with pytest.raises(ValueError, match="head"):
        _tp(d_model=6, n_heads=2)  # head_dim 3 — RoPE would reject it later
    with pytest.raises(ValueError, match="divisible"):
        _tp(d_model=18, n_heads=4)
    assert _tp(d_model=16, n_heads=4).d_model == 16


def test_builder_and_from_state_routes_reject_the_same_cases():
    with pytest.raises((ValueError, TypeError), match="d_model"):
        NNTransformerParams.builder().vocab(16).layers(n=1, heads=2, d_model=16.0)
    bad = _tp().state()
    bad["n_layers"] = 1.5
    with pytest.raises(ValueError, match="n_layers"):
        NNParams.resolve_from_state(bad)
    bad_train = NNTrainParams(n_epochs=1).state()
    bad_train["n_epochs"] = 2.0
    with pytest.raises(ValueError, match="n_epochs"):
        NNTrainParams.from_state(bad_train)


def test_malformed_count_in_run_yaml_fails_at_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    X, y = torch.randn(8, 4), torch.randint(0, 2, (8,))
    model = NNModel(
        net_params=_p(hidden_dims=[8], activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=DataLoader(TensorDataset(X, y), batch_size=4),
            optim=NNOptimParams(**_ADAM),
            scheduler=_sched(),
        )
    )
    yaml_path = tmp_path / "runs" / run.id / "run.yaml"
    state = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    state["train"]["n_epochs"] = 1.5
    yaml_path.write_text(yaml.safe_dump(state, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="n_epochs"):
        NNRun.load(run.id)


def test_valid_accumulation_windows_end_at_documented_indices(tmp_path, monkeypatch):
    """accumulate_grad_batches=2 over five microbatches steps after batches
    1, 3 and 4 (the final short window) — validation must not change the
    cadence. Parameter updates are compared against a reference that
    applies the same three normalized windows explicitly."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    X, y = torch.randn(5, 4), torch.tensor([0, 1, 0, 1, 1])
    loader = DataLoader(TensorDataset(X, y), batch_size=1, shuffle=False)

    def _model():
        return NNModel(
            net_params=_p(hidden_dims=[8], activation=Activations.RELU),
            params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        )

    torch.manual_seed(1)
    trained = _model()
    stepped: list[int] = []
    original_step = torch.optim.SGD.step

    def spy(self, *a, **k):
        stepped.append(len(stepped))
        return original_step(self, *a, **k)

    monkeypatch.setattr(torch.optim.SGD, "step", spy)
    trained.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0, accumulate_grad_batches=np.int64(2)
            ),
            scheduler=_sched(),
        )
    )
    monkeypatch.setattr(torch.optim.SGD, "step", original_step)
    assert len(stepped) == 3  # windows [0,1], [2,3], [4]

    torch.manual_seed(1)
    reference = _model()
    opt = torch.optim.SGD([p for p in reference.net.parameters()], lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    reference.net.train()
    for window in ([0, 1], [2, 3], [4]):
        opt.zero_grad()
        loss = sum(loss_fn(reference.net(X[i : i + 1]), y[i : i + 1]) for i in window) / len(window)
        loss.backward()
        opt.step()
    for (name, a), (_, b) in zip(trained.net.named_parameters(), reference.net.named_parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6, msg=name)
    assert all(
        math.isfinite(idp.train_edp.loss)
        for idp in trained.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=loader,
                data_id="again",
                optim=NNOptimParams(name=Optims.SGD, max_lr=0.0, momentum=0.0, weight_decay=0.0),
                scheduler=_sched(),
            )
        ).idps
        if idp.train_edp.loss is not None
    )
