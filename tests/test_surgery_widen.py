"""Tests for ``nnx.surgery.widen`` — Net2WiderNet.

Every primitive in :mod:`nnx.surgery` has the same correctness contract:
the surged module's forward output equals the original's *before any
training step*. That's the whole point of Net2Net — you can resume
training immediately without an accuracy cliff. If function-preservation
fails, the surgery is broken; **do not relax the tolerance**.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import widen
from nnx.nn.enum.activations import Activations
from nnx.nn.net.feed_fwd_nn import FeedFwdNN
from nnx.nn.params.nn_params import NNParams

# ---------- Function-preservation: the contract ------------------------


def test_widen_linear_is_function_preserving():
    """The core Net2WiderNet invariant: forward output unchanged."""
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    x = torch.randn(3, 4)
    orig_out = net(x)

    widened = widen(net, layer_name="0", new_width=16)

    assert isinstance(widened[0], nn.Linear)
    assert widened[0].out_features == 16
    assert widened[2].in_features == 16

    new_out = widened(x)
    assert torch.allclose(orig_out, new_out, atol=1e-5), (
        f"widen() broke function-preservation: max diff {(orig_out - new_out).abs().max().item():.2e}"
    )


def test_widen_returns_fresh_module():
    """The deep-copy contract: the original is untouched."""
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    widened = widen(net, layer_name="0", new_width=16)
    # Different identity, different shapes, but original is intact.
    assert widened is not net
    assert widened[0] is not net[0]
    assert net[0].out_features == 8
    assert net[2].in_features == 8


def test_widen_preserves_function_with_bias_false():
    """Bias-less Linears must also preserve the forward."""
    torch.manual_seed(1)
    net = nn.Sequential(
        nn.Linear(3, 5, bias=False),
        nn.ReLU(),
        nn.Linear(5, 2, bias=False),
    )
    x = torch.randn(4, 3)
    orig_out = net(x)
    widened = widen(net, layer_name="0", new_width=9)
    new_out = widened(x)
    assert torch.allclose(orig_out, new_out, atol=1e-5)
    assert widened[0].bias is None
    assert widened[2].bias is None


def test_widen_preserves_function_on_feed_fwd_nn():
    """Function-preservation must hold across NNx's own FeedFwdNN
    (an ``nn.ModuleList``-backed net, dotted names are ``layers.<i>``)."""
    torch.manual_seed(2)
    params = NNParams(
        input_dim=6,
        output_dim=3,
        hidden_dims=[10, 8],
        dropout_prob=0.0,  # dropout would break identity comparison
        activation=Activations.RELU,
    )
    net = FeedFwdNN(params)
    net.eval()  # disable any train-mode noise
    x = torch.randn(5, 6)
    orig_out = net(x)

    widened = widen(net, layer_name="layers.0", new_width=20)
    widened.eval()
    new_out = widened(x)

    assert isinstance(widened, FeedFwdNN)
    assert widened.layers[0].out_features == 20
    assert widened.layers[1].in_features == 20
    assert torch.allclose(orig_out, new_out, atol=1e-5), (
        f"widen on FeedFwdNN broke function-preservation: max diff {(orig_out - new_out).abs().max().item():.2e}"
    )


def test_widen_is_deterministic_with_seed():
    """Same rng_seed → identical surgery output across calls."""
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    a = widen(net, layer_name="0", new_width=16, rng_seed=42)
    b = widen(net, layer_name="0", new_width=16, rng_seed=42)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


# ---------- Error handling --------------------------------------------


def test_widen_rejects_nonexistent_layer():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with pytest.raises(KeyError, match="no module named"):
        widen(net, layer_name="nonexistent", new_width=16)


def test_widen_rejects_shrinking():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with pytest.raises(ValueError, match="new_width must be > current"):
        widen(net, layer_name="0", new_width=4)


def test_widen_rejects_equal_width():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with pytest.raises(ValueError, match="new_width must be > current"):
        widen(net, layer_name="0", new_width=8)


def test_widen_rejects_non_linear_target():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with pytest.raises(TypeError, match="expected nn.Linear"):
        widen(net, layer_name="1", new_width=16)  # the ReLU


def test_widen_rejects_when_no_downstream_linear():
    """If the named layer is the *last* Linear, there's nothing to
    rescale and function-preservation can't be enforced."""
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    with pytest.raises(ValueError, match="no downstream nn.Linear"):
        widen(net, layer_name="0", new_width=16)


# ---------- Parameter count ------------------------------------------


def test_widen_parameter_count_grows():
    """Widening a layer from k to k+q should grow its params and the
    downstream layer's input fan-in, leaving other layers alone."""
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    orig_params = sum(p.numel() for p in net.parameters())
    widened = widen(net, layer_name="0", new_width=16)
    new_params = sum(p.numel() for p in widened.parameters())
    # +8 new units → +8*4 weight + +8 bias on layer 0,
    # and +8 new in-features on layer 2 → +8*2 weight (no extra bias).
    expected = orig_params + (8 * 4 + 8) + (8 * 2)
    assert new_params == expected


def test_widen_does_not_advance_global_rng():
    """widen() must be a no-op on the global torch RNG stream — the
    fresh layers are built uninitialized (skip_init) because every
    param is fully overwritten. Pre-fix, their kaiming inits drew from
    the default generator, so any seeded caller pipeline silently
    diverged after a surgery call. Covers both the seeded and the
    rng_seed=None (local-entropy) paths."""
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    torch.manual_seed(123)
    state = torch.get_rng_state()
    widen(net, layer_name="0", new_width=16)
    widen(net, layer_name="0", new_width=16, rng_seed=None)
    assert torch.equal(torch.get_rng_state(), state)


# ---------------- FIX-016: replacements preserve trainability and modes ----------------

_ROLE_CASES = [
    pytest.param(True, True, True, id="w_train-b_train"),
    pytest.param(True, False, True, id="w_train-b_frozen"),
    pytest.param(False, True, True, id="w_frozen-b_train"),
    pytest.param(False, False, True, id="w_frozen-b_frozen"),
    pytest.param(False, None, False, id="w_frozen-no_bias"),
    pytest.param(True, None, False, id="w_train-no_bias"),
]


@pytest.mark.parametrize(("w_flag", "b_flag", "has_bias"), _ROLE_CASES)
def test_surgery_preserves_weight_bias_roles(w_flag, b_flag, has_bias):
    """widen must give the target replacement the target's own weight/bias
    `requires_grad` flags and the downstream replacement the downstream's
    own (opposing) flags — never cross-propagate — while the source model
    keeps its flags and the new tensors are independent storage."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 4, bias=has_bias), nn.ReLU(), nn.Linear(4, 2, bias=has_bias))
    net[0].weight.requires_grad_(w_flag)
    net[2].weight.requires_grad_(not w_flag)
    if has_bias:
        net[0].bias.requires_grad_(b_flag)
        net[2].bias.requires_grad_(not b_flag)
    source_flags = {n: p.requires_grad for n, p in net.named_parameters()}

    wider = widen(net, layer_name="0", new_width=7)

    assert wider[0].weight.requires_grad is w_flag
    assert wider[2].weight.requires_grad is (not w_flag)
    if has_bias:
        assert wider[0].bias.requires_grad is b_flag
        assert wider[2].bias.requires_grad is (not b_flag)
    else:
        assert wider[0].bias is None and wider[2].bias is None
    assert {n: p.requires_grad for n, p in net.named_parameters()} == source_flags
    assert wider[0].weight is not net[0].weight and wider[2].weight is not net[2].weight

    x = torch.randn(3, 4)
    torch.testing.assert_close(wider(x), net(x), atol=1e-5, rtol=1e-5)
    trainable = [p for p in wider.parameters() if p.requires_grad]
    if trainable:
        wider(x).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        assert all(p.grad is None for p in wider.parameters() if not p.requires_grad)


def test_surgery_preserves_mixed_module_modes():
    """Replaced children keep the mode of the module they replace; the
    copied root keeps its own mode; a mixed root/child configuration
    survives widening unchanged."""
    net = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    net.eval()
    net[0].train()  # mixed: root eval, target train, downstream eval
    wider = widen(net, layer_name="0", new_width=6)
    assert wider.training is False
    assert wider[0].training is True
    assert wider[2].training is False

    net.train()
    net[2].eval()
    wider = widen(net, layer_name="0", new_width=6)
    assert wider.training is True and wider[0].training is True and wider[2].training is False


def test_widen_frozen_network_cannot_enter_explicit_param_groups():
    """The practical consequence: an optimizer built AFTER surgery on a
    fully frozen network must not receive any parameter — replacement
    constructors used to default every new tensor back to trainable."""
    from nnx import NNParamGroupSpec
    from nnx.finetune.param_groups import build_param_groups

    net = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    net.requires_grad_(False)
    wider = widen(net, layer_name="0", new_width=7)
    assert not any(p.requires_grad for p in wider.parameters())
    assert not wider[0].training and not wider[2].training
    with pytest.raises(ValueError, match="no parameter groups"):
        build_param_groups(
            wider, [NNParamGroupSpec(name_pattern="*", lr=1e-2)], default_lr=1e-2, default_weight_decay=0.0, strict=True
        )
