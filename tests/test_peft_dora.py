"""Tests for nnx.peft.dora — DoRALinear + apply_dora_to."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import (
    DoRALinear,
    LoRALinear,
    apply_dora_to,
    load_lora_weights,
    save_lora_weights,
)

# -------------------------------------------------------------------------
# DoRALinear basics
# -------------------------------------------------------------------------


def test_dora_linear_rejects_non_linear_base():
    with pytest.raises(TypeError, match="nn.Linear"):
        DoRALinear(nn.Conv2d(3, 4, 3), r=2)


def test_dora_linear_validates_r_alpha_dropout():
    base = nn.Linear(8, 4)
    with pytest.raises(ValueError, match="rank r"):
        DoRALinear(base, r=0)
    with pytest.raises(ValueError, match="alpha"):
        DoRALinear(base, r=2, alpha=0.0)
    with pytest.raises(ValueError, match="dropout"):
        DoRALinear(base, r=2, dropout=1.0)
    with pytest.raises(ValueError, match="dropout"):
        DoRALinear(base, r=2, dropout=-0.1)


def test_dora_freezes_base():
    base = nn.Linear(8, 4)
    # Base starts trainable.
    assert all(p.requires_grad for p in base.parameters())
    DoRALinear(base, r=2)
    # After wrap: every base param frozen.
    assert all(not p.requires_grad for p in base.parameters())


def test_dora_initial_output_equals_base():
    """B is zero-init AND magnitude is initialized from ||W_0||_c, so
    V = W_0 + BA = W_0 → m · V/||V||_c == W_0 at step 0. The layer's
    output equals the base layer's output exactly — fine-tuning starts
    from the pretrained behavior."""
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    dora = DoRALinear(base, r=2, alpha=4.0)
    x = torch.randn(3, 8)
    assert torch.allclose(dora(x), base(x), atol=1e-6)


def test_dora_initial_output_equals_base_no_bias():
    """Same invariant with bias=False — the magnitude/normalize math
    must work regardless of bias term."""
    torch.manual_seed(1)
    base = nn.Linear(8, 4, bias=False)
    dora = DoRALinear(base, r=2, alpha=4.0)
    x = torch.randn(3, 8)
    assert torch.allclose(dora(x), base(x), atol=1e-6)


def test_dora_forward_shape():
    base = nn.Linear(8, 4)
    dora = DoRALinear(base, r=2, alpha=4.0)
    out = dora(torch.randn(3, 8))
    assert out.shape == (3, 4)


def test_dora_trainable_set():
    """DoRA's trainable parameters are exactly {lora_A, lora_B, magnitude}
    — the base is frozen, and `magnitude` is the only addition over
    LoRA's trainable set. Get this wrong and either the frozen base
    leaks gradient or the magnitude vector doesn't actually update."""
    base = nn.Linear(8, 4)
    dora = DoRALinear(base, r=2)
    trainable = {n for n, p in dora.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B", "magnitude"}


def test_dora_magnitude_init_matches_column_norm():
    """magnitude is initialized from the per-output-row L2 norm of the
    base weight so V/||V||_c · m == W_0 exactly at step 0."""
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    dora = DoRALinear(base, r=2)
    expected = base.weight.norm(p=2, dim=1)
    assert torch.allclose(dora.magnitude, expected, atol=1e-6)


def test_dora_in_out_features_passthrough():
    base = nn.Linear(8, 4)
    dora = DoRALinear(base, r=2)
    assert dora.in_features == 8
    assert dora.out_features == 4


def test_dora_inherits_loralinear():
    """DoRA is a LoRA refinement, not a parallel rewrite. The subclass
    relationship lets `save_lora_weights` / `load_lora_weights` capture
    the lora_A/B matrices unchanged; only the new `magnitude` parameter
    is DoRA-specific."""
    base = nn.Linear(8, 4)
    dora = DoRALinear(base, r=2)
    assert isinstance(dora, LoRALinear)


# -------------------------------------------------------------------------
# apply_dora_to
# -------------------------------------------------------------------------


class _TinyNet(nn.Module):
    """3-layer MLP — the canonical apply_dora_to target."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(8, 16),
                nn.Linear(16, 8),
                nn.Linear(8, 3),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_apply_dora_to_requires_pattern():
    with pytest.raises(ValueError, match="at least one"):
        apply_dora_to(_TinyNet())


def test_apply_dora_to_wraps_matched_only():
    net = _TinyNet()
    n = apply_dora_to(net, "layers.0", r=2, alpha=4.0)
    assert n == 1
    assert isinstance(net.layers[0], DoRALinear)
    # Unmatched layers untouched.
    assert isinstance(net.layers[1], nn.Linear) and not isinstance(net.layers[1], LoRALinear)
    assert isinstance(net.layers[2], nn.Linear) and not isinstance(net.layers[2], LoRALinear)


def test_apply_dora_to_wildcard_wraps_all_linears():
    net = _TinyNet()
    n = apply_dora_to(net, "layers.*", r=2)
    assert n == 3
    assert all(isinstance(net.layers[i], DoRALinear) for i in range(3))


def test_apply_dora_to_is_idempotent_for_already_wrapped():
    """Second call against the same patterns must NOT re-wrap the inner
    .base of an existing DoRALinear (the parent-is-DoRALinear check
    inherits LoRALinear's skip behavior)."""
    net = _TinyNet()
    n_first = apply_dora_to(net, "layers.*", r=2)
    assert n_first == 3
    n_second = apply_dora_to(net, "layers.*", r=2)
    assert n_second == 0
    assert all(isinstance(net.layers[i], DoRALinear) for i in range(3))
    for i in range(3):
        assert isinstance(net.layers[i].base, nn.Linear)
        assert not isinstance(net.layers[i].base, LoRALinear)


def test_apply_dora_to_preserves_forward_at_init():
    """Post-wrap forward at step 0 == pre-wrap forward, by the same
    base-equals-init invariant tested on DoRALinear alone."""
    torch.manual_seed(0)
    net = _TinyNet()
    x = torch.randn(2, 8)
    pre = net(x)
    apply_dora_to(net, "layers.*", r=2, alpha=4.0)
    post = net(x)
    assert torch.allclose(pre, post, atol=1e-6)


# -------------------------------------------------------------------------
# Interop with save_lora_weights / load_lora_weights
# -------------------------------------------------------------------------


def test_save_lora_weights_captures_dora_lora_matrices(tmp_path):
    """DoRA inherits LoRALinear's lora_A / lora_B attributes; the existing
    save_lora_weights filter still picks them up. (The magnitude vector
    is captured via state_dict normally; users wanting just the LoRA
    half can re-use save_lora_weights as-is.)"""
    torch.manual_seed(0)
    net = _TinyNet()
    apply_dora_to(net, "layers.*", r=2, alpha=4.0)
    with torch.no_grad():
        for n, p in net.named_parameters():
            if "lora_" in n:
                p.fill_(0.42)

    path = save_lora_weights(net, tmp_path / "dora_lora.pt")

    sd = torch.load(path, weights_only=True)
    assert len(sd) > 0
    for k in sd:
        assert "lora_A" in k or "lora_B" in k, f"unexpected non-LoRA key: {k!r}"

    net_b = _TinyNet()
    apply_dora_to(net_b, "layers.*", r=2, alpha=4.0)
    n_loaded = load_lora_weights(net_b, path)
    assert n_loaded > 0
    for n, p in net_b.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            assert torch.all(p == 0.42)
