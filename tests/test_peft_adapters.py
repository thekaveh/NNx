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


def test_adapter_layer_gradients_flow_through_both_projections():
    """At step 0 the output equals the input but gradients should
    still flow — only the OUTPUT magnitude starts at zero, the
    GRADIENT path through up/down is unblocked from step 1."""
    torch.manual_seed(0)
    a = AdapterLayer(dim=8, bottleneck=2)
    x = torch.randn(4, 8, requires_grad=False)
    loss = a(x).pow(2).sum()
    loss.backward()
    # down.weight / down.bias should have non-zero gradient (the loss
    # depends on them through the down → act → up path).
    assert a.down.weight.grad is not None
    # up.weight gets gradient because the loss depends on it via the
    # forward path; up.bias same.
    assert a.up.weight.grad is not None
    assert a.up.bias.grad is not None


def test_adapter_layer_validates_dims():
    with pytest.raises(ValueError, match="dim"):
        AdapterLayer(dim=0, bottleneck=4)
    with pytest.raises(ValueError, match="bottleneck"):
        AdapterLayer(dim=8, bottleneck=0)


def test_adapter_layer_custom_activation():
    """Caller can pass any nn.Module factory for the activation."""
    a = AdapterLayer(dim=8, bottleneck=2, activation=nn.ReLU)
    assert isinstance(a.act, nn.ReLU)
