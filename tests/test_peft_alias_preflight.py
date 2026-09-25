"""FIX-014: PEFT injection preflight for shared-module aliases.

``named_modules()`` deduplicates repeated module objects, so a ``Linear``
registered under two names used to be wrapped at one path only (the
aliases stopped being one layer), and selecting the non-first alias was a
silent no-op. The ``apply_*_to`` helpers now see every registration and
reject a selected ``Linear`` registered in more than one slot *before*
building any wrapper (construction freezes the base). A ``Linear`` inside
a shared *container* has a single slot, so it is wrapped once and every
path observes the same wrapper.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

import nnx
import nnx.peft
from nnx import LoRALinear, NNParamGroupSpec
from nnx.finetune import build_param_groups

_HELPERS = {
    "lora": lambda m, *p: nnx.peft.apply_lora_to(m, *p, r=2, alpha=4.0),
    "dora": lambda m, *p: nnx.peft.apply_dora_to(m, *p, r=2, alpha=4.0),
    "ia3": lambda m, *p: nnx.peft.apply_ia3_to(m, *p),
}


def _aliased_model() -> nn.Module:
    """``first`` (independent) precedes ``a`` / ``b`` (one shared Linear);
    ``c`` is another independent Linear."""
    torch.manual_seed(0)
    shared = nn.Linear(4, 4)
    return nn.ModuleDict(
        OrderedDict([("first", nn.Linear(4, 4)), ("a", shared), ("b", shared), ("c", nn.Linear(4, 4))])
    )


def _snapshot(model: nn.Module):
    return (
        {name: id(m) for name, m in model.named_modules(remove_duplicate=False)},
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        {name: p.requires_grad for name, p in model.named_parameters(remove_duplicate=False)},
    )


def _assert_unchanged(model: nn.Module, before) -> None:
    ids, state, flags = before
    assert {name: id(m) for name, m in model.named_modules(remove_duplicate=False)} == ids
    after = model.state_dict()
    assert after.keys() == state.keys()
    assert all(torch.equal(after[k], state[k]) for k in state)
    assert {name: p.requires_grad for name, p in model.named_parameters(remove_duplicate=False)} == flags


@pytest.mark.parametrize("kind", sorted(_HELPERS))
@pytest.mark.parametrize("pattern", ["*", "b", "a"])
def test_peft_rejects_selected_shared_alias_before_mutation(kind, pattern):
    model = _aliased_model()
    before = _snapshot(model)
    with pytest.raises(ValueError, match=r"registered under more than one name \(a, b\)"):
        _HELPERS[kind](model, pattern)
    _assert_unchanged(model, before)
    assert model["a"] is model["b"] and isinstance(model["a"], nn.Linear)


@pytest.mark.parametrize("kind", sorted(_HELPERS))
def test_peft_alias_preflight_is_transactional(kind):
    """The whole matched set is validated before the first wrapper is built:
    a valid target ordered *before* the aliased one stays untouched."""
    model = _aliased_model()
    before = _snapshot(model)
    with pytest.raises(ValueError, match="Nothing was modified"):
        _HELPERS[kind](model, "first", "b")
    _assert_unchanged(model, before)
    assert type(model["first"]) is nn.Linear
    assert all(p.requires_grad for p in model["first"].parameters())


@pytest.mark.parametrize("kind", sorted(_HELPERS))
def test_unselected_alias_does_not_block_unaliased_target(kind):
    model = _aliased_model()
    assert _HELPERS[kind](model, "c", "first") == 2
    assert type(model["c"]) is not nn.Linear and type(model["first"]) is not nn.Linear
    assert model["a"] is model["b"] and type(model["a"]) is nn.Linear
    assert all(p.requires_grad for p in model["a"].parameters())
    assert _HELPERS[kind](model, "c", "first") == 0  # idempotent: no double wrap of .base


@pytest.mark.parametrize("kind", sorted(_HELPERS))
@pytest.mark.parametrize("pattern", ["*", "enc.*", "dec.*"])
def test_shared_container_is_wrapped_once_for_every_path(kind, pattern):
    """A Linear inside a *shared container* has one registration slot: it is
    wrapped once (count 1), selecting it through the non-first path works,
    and both paths observe the same wrapper."""
    block = nn.Sequential(nn.Linear(4, 4))
    model = nn.ModuleDict({"enc": block, "dec": block})
    assert _HELPERS[kind](model, pattern) == 1
    assert model["enc"][0] is model["dec"][0]
    assert type(model["enc"][0]) is not nn.Linear


def test_alias_of_an_existing_wrapper_base_is_rejected():
    """A Linear already wrapped in one slot and still registered raw in
    another would get a second, independent adapter."""
    shared = nn.Linear(4, 4)
    model = nn.ModuleDict({"a": LoRALinear(shared, r=2), "b": shared})
    with pytest.raises(ValueError, match=r"\(a\.base, b\)"):
        nnx.peft.apply_lora_to(model, "b", r=2)
    assert model["b"] is shared


def test_cyclic_registration_does_not_recurse_forever():
    root = nn.Module()
    root.child = nn.Module()
    root.child.back = root  # cycle
    root.lin = nn.Linear(4, 4)
    assert nnx.peft.apply_lora_to(root, "*", r=2) == 1
    assert isinstance(root.lin, LoRALinear)


def test_alias_rejection_keeps_param_groups_and_allows_retry():
    model = _aliased_model()
    specs = [NNParamGroupSpec(name_pattern="c.*", lr_multiplier=0.1)]

    def membership():
        groups = build_param_groups(model, specs, default_lr=1e-3, default_weight_decay=0.0)
        return [sorted(id(p) for p in g["params"]) for g in groups]

    before = membership()
    with pytest.raises(ValueError):
        nnx.peft.apply_lora_to(model, "*", r=2)
    assert membership() == before
    assert nnx.peft.apply_lora_to(model, "c", r=2) == 1
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert {id(model["c"].lora_A), id(model["c"].lora_B)} <= trainable


@pytest.mark.parametrize("name", ["apply_lora_to", "apply_dora_to", "apply_ia3_to"])
def test_facades_raise_the_same_alias_error(name):
    assert getattr(nnx, name) is getattr(nnx.peft, name)
    messages = []
    for facade in (nnx, nnx.peft):
        model = _aliased_model()
        with pytest.raises(ValueError) as info:
            getattr(facade, name)(model, "*")
        messages.append(str(info.value))
    assert messages[0] == messages[1]
    assert name in messages[0] and "Tied" in messages[0]


def test_alias_error_lists_a_bounded_number_of_paths():
    """A heavily shared alias is named without an unbounded message."""
    shared = nn.Linear(4, 4)
    blocks = nn.ModuleList([nn.ModuleDict({"x": shared, "y": shared}) for _ in range(12)])
    with pytest.raises(ValueError, match=r"… and \d+ more\)") as info:
        nnx.peft.apply_lora_to(nn.ModuleDict({"blocks": blocks}), "*", r=2)
    assert len(str(info.value)) < 1000


def test_tied_output_head_is_not_an_alias_but_shares_its_tensor():
    """Documented boundary: the tied embedding / output head are distinct
    modules (no alias error), but wrapping the head freezes the tensor
    they share."""
    from nnx import NNTransformerParams, TransformerNN

    net = TransformerNN(
        NNTransformerParams(
            input_dim=32,
            output_dim=32,
            dropout_prob=0.0,
            vocab_size=32,
            n_layers=1,
            n_heads=2,
            d_model=16,
            ffn_mult=2,
            max_seq_len=16,
            tie_embeddings=True,
        )
    )
    assert net.lm_head.weight is net.tok_embed.weight
    assert nnx.peft.apply_lora_to(net, "*", r=2) > 0
    assert isinstance(net.lm_head, LoRALinear)
    assert net.tok_embed.weight.requires_grad is False
