"""FEAT-013: `Optims.ADAMW` (decoupled weight decay) and the Adam-family `eps`.

`Optims.ADAMW` must build exactly `torch.optim.AdamW` — including through
parameter-group overrides — while ADAM keeps its coupled L2 decay. `eps`
is validated, forwarded to the Adam family only, and omitted from
`state()` at torch's default so existing run ids are unchanged.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import NNOptimParams, NNParamGroupSpec, Optims, build_optimizer
from nnx.finetune.param_groups import build_param_groups


def _scalar_module(value: float) -> nn.Module:
    module = nn.Module()
    module.p = nn.Parameter(torch.tensor([value]))
    return module


@pytest.mark.parametrize(("name", "expected"), [(Optims.ADAMW, 1.96), (Optims.ADAM, 1.9)])
def test_adamw_decays_decoupled_where_adam_couples(name, expected):
    """lr=0.1, weight_decay=0.2, betas=(0, 0), zero gradient: AdamW scales
    the weight by 1 - lr*wd (2.0 -> 1.96) and the Adam step is zero; Adam
    turns the decay into a gradient of 0.4 whose normalized step is lr
    (2.0 -> 1.9)."""
    module = _scalar_module(2.0)
    optimizer = build_optimizer(module, NNOptimParams(name=name, max_lr=0.1, weight_decay=0.2, momentum=(0.0, 0.0)))
    module.p.grad = torch.zeros_like(module.p)
    optimizer.step()
    assert module.p.item() == pytest.approx(expected, abs=1e-6)


def test_adamw_matches_native_adamw_with_param_group_overrides():
    """Three fixed gradients through Optims.ADAMW and through a native
    torch.optim.AdamW over the same resolved groups leave identical
    weights, exp_avg and exp_avg_sq — group lr / weight_decay overrides
    included."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    reference = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    reference.load_state_dict(net.state_dict())
    groups = [
        NNParamGroupSpec(name_pattern="0.*", lr=0.05, weight_decay=0.3),
        NNParamGroupSpec(name_pattern="1.bias", lr_multiplier=0.5),
    ]
    params = NNOptimParams(
        name=Optims.ADAMW, max_lr=0.02, momentum=(0.8, 0.95), weight_decay=0.1, eps=1e-6, param_groups=groups
    )
    optimizer = build_optimizer(net, params)
    native = torch.optim.AdamW(
        build_param_groups(reference, groups, default_lr=0.02, default_weight_decay=0.1),
        lr=0.02,
        betas=(0.8, 0.95),
        weight_decay=0.1,
        eps=1e-6,
    )
    assert type(optimizer) is torch.optim.AdamW
    assert [(g["lr"], g["weight_decay"]) for g in optimizer.param_groups] == [
        (g["lr"], g["weight_decay"]) for g in native.param_groups
    ]
    assert {(g["lr"], g["weight_decay"]) for g in optimizer.param_groups} >= {(0.05, 0.3), (0.01, 0.1)}

    generator = torch.Generator().manual_seed(1)
    for _ in range(3):
        grads = [torch.randn(p.shape, generator=generator) for p in net.parameters()]
        for model, opt in ((net, optimizer), (reference, native)):
            for param, grad in zip(model.parameters(), grads, strict=True):
                param.grad = grad.clone()
            opt.step()
    for param, ref in zip(net.parameters(), reference.parameters(), strict=True):
        assert torch.equal(param, ref)
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(optimizer.state[param][key], native.state[ref][key])


def test_eps_is_forwarded_to_the_adam_family_and_omitted_at_default():
    net = nn.Linear(2, 1)
    for name in (Optims.ADAM, Optims.ADAM_AMSGRAD, Optims.ADAMW):
        default = NNOptimParams(name=name, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0)
        assert "eps" not in default.state()
        assert build_optimizer(net, default).defaults["eps"] == 1e-8
        custom = NNOptimParams(name=name, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0, eps=1e-6)
        assert custom.state()["eps"] == 1e-6
        assert NNOptimParams.from_state(custom.state()) == custom
        assert build_optimizer(net, custom).defaults["eps"] == 1e-6
    adamw = NNOptimParams(name=Optims.ADAMW, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.01)
    assert adamw.state()["name"] == "adamw" and NNOptimParams.from_state(adamw.state()) == adamw


@pytest.mark.parametrize("eps", [0.0, -1e-8, float("nan"), float("inf"), True, "1e-8"])
def test_eps_must_be_finite_and_positive(eps):
    with pytest.raises(ValueError, match="eps"):
        NNOptimParams(name=Optims.ADAMW, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0, eps=eps)


@pytest.mark.parametrize("name", [Optims.SGD, Optims.SGD_NESTEROV])
def test_eps_is_rejected_on_sgd_variants(name):
    with pytest.raises(ValueError, match="Adam-family"):
        NNOptimParams(name=name, max_lr=1e-3, momentum=0.9, weight_decay=0.0, eps=1e-6)
    assert "eps" not in NNOptimParams(name=name, max_lr=1e-3, momentum=0.9, weight_decay=0.0).state()


def test_adamw_is_valid_only_with_betas():
    assert NNOptimParams(name=Optims.ADAMW, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0).is_valid()
    assert not NNOptimParams(name=Optims.ADAMW, max_lr=1e-3, momentum=0.9, weight_decay=0.0).is_valid()


def test_builder_adamw_and_eps():
    built = NNOptimParams.builder().adamw(max_lr=3e-4).build()
    assert built == NNOptimParams(name=Optims.ADAMW, max_lr=3e-4, momentum=(0.9, 0.999), weight_decay=1e-2)
    assert "eps" not in built.state()
    tuned = NNOptimParams.builder().adam(max_lr=1e-3, eps=1e-6).grad_clip(1.0).build()
    assert tuned.eps == 1e-6 and tuned.grad_clip_norm == 1.0
    # A later variant replaces the earlier variant's eps (last variant wins).
    switched = NNOptimParams.builder().adam(max_lr=1e-3, eps=1e-6).sgd(max_lr=1e-2).build()
    assert switched.name == Optims.SGD and switched.eps == 1e-8
    with pytest.raises(ValueError, match=r"\.adamw\(\.\.\.\)"):
        NNOptimParams.builder().build()
