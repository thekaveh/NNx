"""Tests for NNOptimParams.builder() — variant-gated optimizer config.

Asserts the Builder produces dataclasses that:
  * Use the PyTorch-native `betas=` kwarg on the Adam variants (the
    Builder maps it onto NNOptimParams.momentum, which is the
    backwards-compatible underlying field).
  * Use the float `momentum=` kwarg on the SGD variants.
  * Preserve the omit-when-default state() invariant — Builders that
    don't touch optional fields (`grad_clip_norm`,
    `accumulate_grad_batches`, `param_groups`) must leave them at the
    dataclass defaults so state() omits them.
  * Pass NNOptimParams.is_valid() — the contract that momentum's
    shape matches the optimizer kind.
"""

from __future__ import annotations

from nnx.nn.enum.optims import Optims
from nnx.nn.params.nn_optim_params import NNOptimParams


def test_builder_adam_uses_betas_kwarg_and_maps_to_momentum_field():
    """The Adam variant takes `betas` (PyTorch convention) and stores
    it on the dataclass's `momentum` field. The rename is
    Builder-side only — `from_state` / direct-kwarg ctor still take
    `momentum`."""
    op = NNOptimParams.builder().adam(max_lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-5).build()
    assert op.name == Optims.ADAM
    assert op.max_lr == 1e-3
    assert op.weight_decay == 5e-5
    # The Builder's `betas` kwarg is stored on the dataclass `momentum`
    # field — that's the back-compat shape.
    assert op.momentum == (0.9, 0.999)
    assert op.is_valid()


def test_builder_adam_preserves_omit_when_default_invariant():
    """CRITICAL: an Adam config built via `.builder().adam(...).build()`
    must produce the same state() as a direct-ctor Adam config. The
    optional fields (`grad_clip_norm`, `accumulate_grad_batches`,
    `param_groups`) must stay at their defaults and NOT appear in state().

    PR #10 + earlier broke this three times for related fields; new
    Builders preserving the invariant is non-negotiable.
    """
    built = NNOptimParams.builder().adam(max_lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0).build()
    direct = NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-3,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
    )
    assert built.state() == direct.state()
    assert "grad_clip_norm" not in built.state()
    assert "accumulate_grad_batches" not in built.state()
    assert "param_groups" not in built.state()


def test_builder_sgd_uses_float_momentum():
    """SGD keeps the float `momentum` kwarg — `betas` is an Adam term.
    The dataclass `momentum` field stores the float directly."""
    op = NNOptimParams.builder().sgd(max_lr=1e-2, momentum=0.9, weight_decay=5e-5).build()
    assert op.name == Optims.SGD
    assert op.max_lr == 1e-2
    assert op.momentum == 0.9
    assert op.is_valid()


def test_builder_sgd_nesterov():
    op = NNOptimParams.builder().sgd_nesterov(max_lr=1e-2, momentum=0.9, weight_decay=0.0).build()
    assert op.name == Optims.SGD_NESTEROV
    assert op.momentum == 0.9
    assert op.is_valid()


def test_builder_adam_amsgrad():
    op = NNOptimParams.builder().adam_amsgrad(max_lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0).build()
    assert op.name == Optims.ADAM_AMSGRAD
    assert op.momentum == (0.9, 0.999)
    assert op.is_valid()


def test_builder_chains_grad_clip_after_variant():
    """The optional modifier methods chain after the variant method.
    `grad_clip` writes the `grad_clip_norm` field; without it, state()
    omits the key (omit-when-default invariant)."""
    op = NNOptimParams.builder().adam(max_lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0).grad_clip(1.0).build()
    assert op.grad_clip_norm == 1.0
    # Round-trips through state() / from_state().
    assert NNOptimParams.from_state(op.state()) == op
    # And the user-set value DOES appear in state() (it's no longer at default).
    assert op.state().get("grad_clip_norm") == 1.0


def test_builder_chains_accumulate_grad_after_variant():
    op = NNOptimParams.builder().adam(max_lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0).accumulate_grad(4).build()
    assert op.accumulate_grad_batches == 4
    assert NNOptimParams.from_state(op.state()) == op
    assert op.state().get("accumulate_grad_batches") == 4


def test_builder_chains_param_groups_after_variant():
    """param_groups composes with the GAN-recipe pattern of two
    optims with different sub-net scopes (the §3.4 plan's main
    user). Each NNOptimParams takes a list of NNParamGroupSpec; the
    Builder just stores the list, no transformation.

    `NNParamGroupSpec` takes `name_pattern` (single fnmatch glob) +
    one of `lr` / `lr_multiplier` + optional `weight_decay`. See
    `src/nnx/finetune/param_groups.py`.
    """
    from nnx import NNParamGroupSpec

    g = NNParamGroupSpec(name_pattern="encoder.*", lr_multiplier=0.01, weight_decay=0.0)
    op = NNOptimParams.builder().adam(max_lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0).param_groups([g]).build()
    assert op.param_groups == [g]
    assert NNOptimParams.from_state(op.state()) == op
