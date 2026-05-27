"""Tests for nnx.finetune.param_groups + the NNOptimParams integration."""
from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParamGroupSpec,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
)
from nnx.finetune.param_groups import build_param_groups


def _net() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def test_param_group_spec_round_trip():
    """state() / from_state() must reconstruct the spec identically."""
    spec = NNParamGroupSpec(
        name_pattern="encoder.*", lr=1e-5, weight_decay=0.0,
    )
    rt = NNParamGroupSpec.from_state(spec.state())
    assert rt == spec


def test_param_group_spec_lr_xor_multiplier():
    """Specifying both lr AND lr_multiplier is ambiguous; raise early."""
    with pytest.raises(ValueError, match="at most one"):
        NNParamGroupSpec(name_pattern="*", lr=1e-3, lr_multiplier=0.1)


def test_param_group_spec_state_omits_unset_fields():
    """A spec with only name_pattern set should produce a 1-key state()."""
    spec = NNParamGroupSpec(name_pattern="head.*")
    assert spec.state() == {"name_pattern": "head.*"}


def test_nn_optim_params_state_omits_param_groups_when_none():
    """CRITICAL back-compat invariant: NNOptimParams with param_groups=None
    must emit the same state() it did before this field existed —
    otherwise every existing run.id shifts. The same invariant is now
    enforced on every params dataclass; see the matching regression
    tests in test_params_round_trip.py for mixed_precision and
    NNSchedulerParams.kind."""
    p = NNOptimParams(
        name=Optims.ADAM, max_lr=1e-3,
        momentum=(0.9, 0.999), weight_decay=0.0,
    )
    state = p.state()
    assert "param_groups" not in state, (
        "param_groups=None must be omitted from state() to preserve run.id back-compat; "
        f"got {state!r}"
    )
    assert set(state.keys()) == {"max_lr", "momentum", "name", "weight_decay"}


def test_nn_optim_params_state_emits_param_groups_when_set():
    p = NNOptimParams(
        name=Optims.ADAM, max_lr=1e-3,
        momentum=(0.9, 0.999), weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="*", lr_multiplier=0.1)],
    )
    state = p.state()
    assert "param_groups" in state
    assert state["param_groups"][0]["name_pattern"] == "*"

    rt = NNOptimParams.from_state(state)
    assert rt.param_groups is not None
    assert len(rt.param_groups) == 1
    assert rt.param_groups[0] == p.param_groups[0]


def test_build_param_groups_drops_frozen_params():
    """Frozen params should be excluded — they don't need optimizer state."""
    net = _net()
    net[0].weight.requires_grad = False  # freeze first layer's weight
    groups = build_param_groups(
        net,
        [NNParamGroupSpec(name_pattern="*")],
        default_lr=1e-3, default_weight_decay=0.0,
    )
    # Should have 3 trainable params (0.bias, 2.weight, 2.bias), not 4.
    n_params = sum(len(g["params"]) for g in groups)
    assert n_params == 3


def test_build_param_groups_first_matching_spec_wins():
    """Spec priority: earlier specs in the list claim parameters first."""
    net = _net()
    groups = build_param_groups(
        net,
        [
            NNParamGroupSpec(name_pattern="0.weight", lr=1e-5),
            NNParamGroupSpec(name_pattern="*", lr=1e-3),
        ],
        default_lr=1e-2, default_weight_decay=0.0,
    )
    # First group should contain exactly one param (0.weight @ lr=1e-5).
    # Second group should contain everything else (3 params @ lr=1e-3).
    # Unmatched bucket should be empty (no default group).
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-5 and len(groups[0]["params"]) == 1
    assert groups[1]["lr"] == 1e-3 and len(groups[1]["params"]) == 3


def test_build_param_groups_unmatched_fall_into_default_group():
    """Params not claimed by any spec get the default LR/WD."""
    net = _net()
    groups = build_param_groups(
        net,
        [NNParamGroupSpec(name_pattern="0.*", lr=1e-5)],   # matches 0.weight + 0.bias
        default_lr=1e-2, default_weight_decay=5e-4,
    )
    # Spec group: 2 params at lr=1e-5; default group: 2 params at lr=1e-2.
    assert len(groups) == 2
    matched = next(g for g in groups if g["lr"] == 1e-5)
    default = next(g for g in groups if g["lr"] == 1e-2)
    assert len(matched["params"]) == 2
    assert len(default["params"]) == 2
    assert default["weight_decay"] == 5e-4


def test_build_param_groups_strict_drops_unmatched():
    """Strict mode: unmatched params are dropped from the optimizer entirely
    instead of going into a default bucket. This is the Trainer's contract
    for disjoint multi-optimizer setups."""
    net = _net()
    groups = build_param_groups(
        net,
        [NNParamGroupSpec(name_pattern="0.*", lr=1e-5)],   # matches 0.weight + 0.bias
        default_lr=1e-2, default_weight_decay=5e-4,
        strict=True,
    )
    # Only the matched spec group exists — no default bucket.
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-5
    assert len(groups[0]["params"]) == 2


def test_build_param_groups_strict_raises_when_nothing_matches():
    """Strict mode + no specs matching ANY param should raise so the
    misconfiguration is caught at construction, not silently during
    the first .step()."""
    net = _net()
    with pytest.raises(ValueError, match="strict mode"):
        build_param_groups(
            net,
            [NNParamGroupSpec(name_pattern="nonexistent.*", lr=1e-5)],
            default_lr=1e-2, default_weight_decay=0.0,
            strict=True,
        )


def test_optims_strict_param_groups_passes_through():
    """The Optims.__call__ wrapper should thread strict_param_groups
    through to build_param_groups."""
    net = _net()
    optimizer = Optims.ADAM(
        net=net, lr_start=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="0.*", lr=1e-5)],
        strict_param_groups=True,
    )
    # Only the matched group, no default.
    assert len(optimizer.param_groups) == 1
    assert len(optimizer.param_groups[0]["params"]) == 2


def test_build_param_groups_lr_multiplier_scales_default():
    net = _net()
    groups = build_param_groups(
        net,
        [NNParamGroupSpec(name_pattern="0.*", lr_multiplier=0.01)],
        default_lr=1e-3, default_weight_decay=0.0,
    )
    matched = next(g for g in groups if g["lr"] == 1e-3 * 0.01)
    assert len(matched["params"]) == 2


def test_optims_adam_param_groups_per_group_lr_in_state():
    """Optims.ADAM(net=..., param_groups=...) should produce a torch optimizer
    whose param_groups[i]['lr'] matches what the spec asked for."""
    net = _net()
    optimizer = Optims.ADAM(
        net=net, lr_start=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
        param_groups=[
            NNParamGroupSpec(name_pattern="0.*", lr_multiplier=0.01),
            NNParamGroupSpec(name_pattern="2.*", lr=5e-4),
        ],
    )
    lrs = sorted(g["lr"] for g in optimizer.param_groups)
    assert lrs == [1e-5, 5e-4]  # 0.* @ 0.01 * 1e-3 = 1e-5; 2.* @ 5e-4


def test_optims_sgd_param_groups():
    """Same path through SGD's branch of the Optims.__call__ match."""
    net = _net()
    optimizer = Optims.SGD(
        net=net, lr_start=1e-2, momentum=0.9, weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="*", lr=5e-3)],
    )
    assert all(g["lr"] == 5e-3 for g in optimizer.param_groups)


def test_train_end_to_end_with_param_groups(tmp_path, monkeypatch):
    """An end-to-end train() call with param_groups set must run cleanly
    and persist param_groups into NNRun.state() / run.yaml."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)

    model = NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    run = model.train(params=NNTrainParams(
        n_epochs=1,
        train_loader=loader,
        optim=NNOptimParams(
            name=Optims.ADAM, max_lr=1e-2,
            momentum=(0.9, 0.999), weight_decay=0.0,
            param_groups=[
                NNParamGroupSpec(name_pattern="layers.0.*", lr=1e-4),
            ],
        ),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
    ))
    # Reload the run; param_groups should round-trip.
    from nnx import NNRun
    reloaded = NNRun.load(id=run.id)
    assert reloaded.train.optim.param_groups is not None
    assert reloaded.train.optim.param_groups[0].name_pattern == "layers.0.*"
    assert reloaded.train.optim.param_groups[0].lr == 1e-4
