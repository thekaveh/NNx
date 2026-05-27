"""Tests for nnx.peft.adapters — AdapterLayer."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import AdapterLayer


def test_adapter_layer_forward_shape():
    a = AdapterLayer(dim=16, bottleneck=4)
    x = torch.randn(3, 16)
    out = a(x)
    assert out.shape == x.shape


def test_adapter_layer_initial_output_is_identity():
    """up.weight and up.bias are zero-initialized → output == input
    at step 0. This is the residual-identity invariant; like LoRA's
    B=0 init, it lets fine-tuning start from the unchanged behavior."""
    torch.manual_seed(0)
    a = AdapterLayer(dim=16, bottleneck=4)
    x = torch.randn(3, 16)
    assert torch.allclose(a(x), x, atol=1e-6)


def test_adapter_layer_param_count_is_bottleneck_scaled():
    """Param count should be O(dim * bottleneck), much less than a
    full dim*dim Linear. Verifies the "parameter-efficient" claim."""
    a = AdapterLayer(dim=64, bottleneck=4)
    n_params = sum(p.numel() for p in a.parameters())
    # down: 64*4 + 4 = 260; up: 4*64 + 64 = 320; total 580.
    assert n_params == 64 * 4 + 4 + 4 * 64 + 64
    # And much less than a full Linear(64, 64) which would be 4160.
    full_linear_params = 64 * 64 + 64
    assert n_params < full_linear_params / 4


def test_adapter_layer_gradients_flow_through_up_immediately():
    """At step 0 the output equals the input (`up.weight` is zero-init),
    so `up.weight.grad` picks up a non-zero signal on the first
    backward. `down.weight.grad` is structurally zero at step 0 because
    its gradient chain passes through `up.weight = 0` — it becomes
    non-zero only after the first optimizer step moves `up.weight`
    off zero. This test pins both invariants explicitly so a
    regression that decouples gradient flow from forward dependency
    is caught."""
    torch.manual_seed(0)
    a = AdapterLayer(dim=8, bottleneck=2)
    x = torch.randn(4, 8)
    loss = a(x).pow(2).sum()
    loss.backward()

    # up.weight / up.bias get non-zero gradients on the first backward —
    # the loss depends on them through the residual term directly.
    assert a.up.weight.grad is not None
    assert (a.up.weight.grad != 0).any(), "up.weight.grad must be non-zero at step 0"
    assert a.up.bias.grad is not None
    assert (a.up.bias.grad != 0).any(), "up.bias.grad must be non-zero at step 0"

    # down.weight gradient is structurally zero at step 0 because the
    # chain rule routes its gradient through `up.weight`, which is
    # zero. After one optimizer step (which moves up.weight off zero),
    # the next backward populates down.weight.grad with non-zeros.
    assert a.down.weight.grad is not None
    assert (a.down.weight.grad == 0).all(), (
        "down.weight.grad must be all-zero at step 0 (gradient chain "
        "passes through up.weight=0)"
    )

    # Take one SGD-like step on up.weight to break the zero, then verify
    # the next backward DOES populate down.weight.grad non-trivially.
    import torch as _torch
    with _torch.no_grad():
        a.up.weight += 0.01
    a.down.weight.grad = None
    a.up.weight.grad = None
    loss2 = a(x).pow(2).sum()
    loss2.backward()
    assert (a.down.weight.grad != 0).any(), (
        "after up.weight is non-zero, down.weight.grad must pick up signal"
    )


def test_adapter_layer_validates_dims():
    with pytest.raises(ValueError, match="dim"):
        AdapterLayer(dim=0, bottleneck=4)
    with pytest.raises(ValueError, match="bottleneck"):
        AdapterLayer(dim=8, bottleneck=0)


def test_adapter_layer_custom_activation():
    """Caller can pass any nn.Module factory for the activation."""
    a = AdapterLayer(dim=8, bottleneck=2, activation=nn.ReLU)
    assert isinstance(a.act, nn.ReLU)
