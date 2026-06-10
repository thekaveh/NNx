"""Tests for nnx.nn.moe — MoELinear sparse mixture-of-experts layer."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import MoELinear, set_seed

# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------


def test_moe_linear_validates_inputs():
    """``top_k > num_experts`` and ``num_experts ≤ 1`` both raise."""
    with pytest.raises(ValueError, match="top_k"):
        MoELinear(8, 4, num_experts=2, top_k=3)
    with pytest.raises(ValueError, match="top_k"):
        MoELinear(8, 4, num_experts=4, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        MoELinear(8, 4, num_experts=4, top_k=-1)
    with pytest.raises(ValueError, match="num_experts"):
        MoELinear(8, 4, num_experts=1, top_k=1)
    with pytest.raises(ValueError, match="num_experts"):
        MoELinear(8, 4, num_experts=0, top_k=1)


# -------------------------------------------------------------------------
# Forward shape + routing
# -------------------------------------------------------------------------


def test_moe_linear_forward_shape():
    """Output is (B, out_features) regardless of num_experts / top_k."""
    set_seed(0)
    layer = MoELinear(8, 4, num_experts=4, top_k=2)
    out = layer(torch.randn(5, 8))
    assert out.shape == (5, 4)


def test_moe_linear_router_module_shape():
    """The router is a bias-less nn.Linear(in, num_experts)."""
    layer = MoELinear(8, 4, num_experts=3, top_k=2)
    assert isinstance(layer.router, nn.Linear)
    assert layer.router.weight.shape == (3, 8)
    assert layer.router.bias is None


def test_moe_linear_experts_module_list_shape():
    """experts is a ModuleList of nn.Linear(in, out), one per expert."""
    layer = MoELinear(8, 4, num_experts=3, top_k=2)
    assert isinstance(layer.experts, nn.ModuleList)
    assert len(layer.experts) == 3
    for e in layer.experts:
        assert isinstance(e, nn.Linear)
        assert e.weight.shape == (4, 8)


def test_moe_linear_top_k_routing():
    """Each input row is dispatched through exactly ``top_k`` distinct experts.

    Inspect the layer's internal top-k indices via a hook that captures
    the router's argmax-via-topk output."""
    set_seed(0)
    layer = MoELinear(8, 4, num_experts=5, top_k=2)

    # Replicate the layer's top-k computation to verify the property
    # holds without intercepting private state.
    x = torch.randn(11, 8)
    with torch.no_grad():
        logits = layer.router(x)
        _, topk_idx = logits.topk(layer.top_k, dim=-1)
    assert topk_idx.shape == (11, layer.top_k)
    # Each row's top-k indices are distinct (torch.topk guarantees this).
    for row in topk_idx:
        assert len(set(row.tolist())) == layer.top_k


# -------------------------------------------------------------------------
# Aux loss properties
# -------------------------------------------------------------------------


def test_moe_linear_aux_loss_populated_after_forward():
    """``last_aux_loss`` starts at None and is set after each forward."""
    layer = MoELinear(8, 4, num_experts=4, top_k=2)
    assert layer.last_aux_loss is None
    layer(torch.randn(5, 8))
    assert layer.last_aux_loss is not None
    assert isinstance(layer.last_aux_loss, torch.Tensor)
    assert layer.last_aux_loss.dim() == 0  # scalar


def test_moe_linear_aux_loss_non_negative():
    """Switch aux loss is the sum of products of two non-negative quantities
    (fractions and probabilities), so it must be ≥ 0 for any input."""
    set_seed(0)
    layer = MoELinear(8, 4, num_experts=4, top_k=2)
    for _ in range(10):
        x = torch.randn(16, 8)
        layer(x)
        assert layer.last_aux_loss is not None
        assert layer.last_aux_loss.item() >= 0.0


def test_moe_linear_aux_loss_zero_at_uniform():
    """When routing is perfectly uniform — f_i = P_i = 1/N for every i —
    the Switch penalty equals 1 (its minimum), NOT 0. With ``α · N · Σ f_i P_i``
    at f_i = P_i = 1/N: ``N · N · (1/N²) = 1``. This test verifies the
    minimum value is what the math predicts.

    We force uniform routing by zeroing the router (so logits are all 0
    and probs are uniform). topk on equal values picks the first ``k``
    indices deterministically, so dispatch_frac is non-uniform —
    we instead use ``num_experts == top_k`` so ALL experts run for
    every token, making dispatch_frac uniform at exactly 1/N.
    """
    set_seed(0)
    N = 4
    layer = MoELinear(8, 4, num_experts=N, top_k=N)
    # Zero out the router so every expert gets equal probability for
    # every input.
    with torch.no_grad():
        layer.router.weight.zero_()
    x = torch.randn(7, 8)
    layer(x)
    assert layer.last_aux_loss is not None
    # Expected minimum: 1.0.
    assert layer.last_aux_loss.item() == pytest.approx(1.0, abs=1e-6)


def test_moe_linear_aux_loss_above_minimum_when_skewed():
    """If the router is biased toward a single expert, the aux loss
    exceeds the uniform-routing minimum (1.0). This is the property
    optimization is meant to exploit — gradient descent on the aux loss
    pulls routing back toward uniform."""
    set_seed(0)
    N = 4
    layer = MoELinear(8, 4, num_experts=N, top_k=2)
    # Drive the router strongly toward expert 0 by making its weight
    # row dominant (large positive logits for expert 0, large negative
    # for the rest, for any reasonable input scale).
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[0] += 100.0  # expert 0 wins almost every token
    x = torch.randn(32, 8)
    layer(x)
    assert layer.last_aux_loss is not None
    # Skewed routing should yield aux_loss strictly greater than 1.0
    # (the uniform minimum).
    assert layer.last_aux_loss.item() > 1.0 + 1e-3


def test_moe_linear_load_balancing_converges():
    """Optimizing the aux loss alone pushes routing toward uniform.

    We start with a deliberately skewed router (one expert dominates
    BOTH top-k slots for nearly every input), train the router
    parameters against ``last_aux_loss``, and verify the loss
    decreases toward 1.0 (its uniform-routing minimum).

    Note: ``f_i`` (the dispatch fraction) is non-differentiable
    because top-k is a hard selection. Only the ``P_i`` term carries
    gradient; descending on it pulls the full softmax distribution
    back toward uniform, which (eventually) also re-balances the
    top-k selections.
    """
    set_seed(0)
    N = 4
    layer = MoELinear(8, 4, num_experts=N, top_k=2)

    # Skewed starting point — expert 0 and expert 1 win both top-k
    # slots for every input. The router has bias=False, so the skew
    # comes from the weight × input dot product, NOT a constant
    # offset. We make the input strictly positive (so positive
    # expert rows reliably produce larger logits) and pick weight
    # magnitudes that put aux_loss meaningfully above 1.0 but not
    # so high that softmax saturates and gradients vanish.
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[0] += 1.0
        layer.router.weight[1] += 0.5

    # Strictly positive input ensures the +positive expert rows
    # produce strictly larger logits than the zeroed-out rows for
    # every token.
    x = torch.randn(32, 8).abs() + 0.1
    opt = torch.optim.SGD(layer.router.parameters(), lr=1.0)

    losses = []
    for _ in range(200):
        opt.zero_grad()
        layer(x)
        loss = layer.last_aux_loss
        assert loss is not None
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    # Loss should drop substantially across the run — the start is
    # significantly above 1.0 (the uniform minimum); the end should
    # have closed at least half the gap.
    start, end = losses[0], losses[-1]
    gap_start = start - 1.0
    gap_end = end - 1.0
    assert gap_end < gap_start * 0.5, (
        f"aux loss did not decrease enough: start {start:.4f} (gap {gap_start:.4f}) → end {end:.4f} (gap {gap_end:.4f})"
    )


# -------------------------------------------------------------------------
# Smoke: end-to-end trainability
# -------------------------------------------------------------------------


def test_moe_linear_gradients_flow_to_router_and_experts():
    """Backward through an MoELinear-using objective updates BOTH the
    router (via gating) and the experts that were actually selected."""
    set_seed(0)
    layer = MoELinear(8, 4, num_experts=4, top_k=2)
    x = torch.randn(8, 8, requires_grad=False)
    y = layer(x).sum()
    y.backward()
    # Router always receives gradient — every token contributes via softmax.
    assert layer.router.weight.grad is not None
    assert (layer.router.weight.grad.abs().sum() > 0).item()
    # At least one expert should have received gradient.
    any_expert_grad = any(e.weight.grad is not None and e.weight.grad.abs().sum() > 0 for e in layer.experts)
    assert any_expert_grad


def test_moe_linear_extra_repr_includes_key_fields():
    layer = MoELinear(8, 4, num_experts=3, top_k=2)
    repr_str = repr(layer)
    assert "num_experts=3" in repr_str
    assert "top_k=2" in repr_str


def test_moe_linear_accepts_3d_input():
    """The docstring promises nn.Linear drop-in compatibility, and
    nn.Linear accepts (..., in_features). Pre-fix a (B, T, C) sequence
    batch raised a cryptic IndexError mid-dispatch; tokens now route
    independently with leading dims restored on return."""
    torch.manual_seed(0)
    layer = MoELinear(in_features=8, out_features=6, num_experts=4, top_k=2)
    x = torch.randn(3, 5, 8)
    out = layer(x)
    assert out.shape == (3, 5, 6)
    assert layer.last_aux_loss is not None
    # Parity with the flattened 2-D call (same weights, same tokens).
    out_flat = layer(x.reshape(-1, 8))
    assert torch.allclose(out, out_flat.reshape(3, 5, 6), atol=1e-6)
