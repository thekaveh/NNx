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


# -------------------------------------------------------------------------
# FIX-015: row normalization must be finite in FP16 and differentiable
# -------------------------------------------------------------------------


def _reference_dora_forward(dora: DoRALinear, x: torch.Tensor) -> torch.Tensor:
    """Promoted-precision oracle: normalize V in float64 with the same
    epsilon, rescale by magnitude, cast the effective weight back to the
    layer dtype and apply F.linear."""
    lora_update = (dora.lora_B.double() @ dora.lora_A.double()) * dora.scaling
    V = dora.base.weight.double() + lora_update
    norm = V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
    W = (dora.magnitude.double().unsqueeze(1) * V / norm).to(x.dtype)
    return torch.nn.functional.linear(x, W, dora.base.bias)


@pytest.mark.parametrize("bias", [False, True])
def test_dora_half_zero_rows_are_finite(bias):
    """FP16 rounds the 1e-8 guard to zero, so a zero row of V used to be
    0/0 = NaN. The whole wrapper is converted after wrapping, so this
    isolates normalization from constructor placement (FIX-003)."""
    base = nn.Linear(4, 4, bias=bias)
    with torch.no_grad():
        base.weight.zero_()
        if bias:
            base.bias.fill_(0.5)
    adapted = DoRALinear(base, r=2).half()
    x = torch.ones(2, 4, dtype=torch.float16, requires_grad=True)
    out = adapted(x)
    assert out.dtype == torch.float16 and torch.isfinite(out).all()
    torch.testing.assert_close(out, base(x), atol=0, rtol=0)
    out.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapted.parameters())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dora_small_rows_match_promoted_reference(dtype):
    """Mixed zero / tiny-but-representable / ordinary rows: every dtype
    stays finite in forward and backward and matches the promoted
    reference at its own tolerance; FP64 is not truncated to FP32."""
    torch.manual_seed(0)
    base = nn.Linear(6, 4, bias=True)
    with torch.no_grad():
        base.weight[0].zero_()  # all-zero row
        base.weight[1].mul_(0).add_(torch.tensor([6e-5, -6e-5, 6e-5, 0.0, 0.0, 0.0]))  # tiny representable row
    adapted = DoRALinear(base, r=2, alpha=4.0).to(dtype)
    x = (torch.randn(3, 6) * 0.5).to(dtype).requires_grad_(True)

    out = adapted(x)
    assert out.dtype == dtype and torch.isfinite(out).all()
    tol = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1e-6, torch.float64: 1e-12}[dtype]
    torch.testing.assert_close(out, _reference_dora_forward(adapted, x.detach()), atol=tol, rtol=tol)
    # At init the zero row stays zero — the guard never invents a direction.
    assert torch.equal(out[:, 0], base(x.detach())[:, 0])

    out.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapted.parameters())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dora_nonzero_learned_update_gradients(dtype):
    """A learned nonzero adapter/magnitude state must stay differentiable:
    output and the input/A/B/magnitude gradients match the promoted
    reference (via torch.autograd on the oracle) at well-conditioned
    points, and nothing is detached."""
    torch.manual_seed(1)
    base = nn.Linear(6, 4, bias=True)
    adapted = DoRALinear(base, r=2, alpha=4.0)
    with torch.no_grad():
        adapted.lora_A.normal_(std=0.3)
        adapted.lora_B.normal_(std=0.3)
        adapted.magnitude.mul_(1.5)
    adapted = adapted.to(dtype)
    x = torch.randn(3, 6).to(dtype).requires_grad_(True)

    out = adapted(x)
    out.float().sum().backward()
    grads = {"x": x.grad, "A": adapted.lora_A.grad, "B": adapted.lora_B.grad, "m": adapted.magnitude.grad}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    assert all(bool((g != 0).any()) for g in grads.values())

    # Oracle gradients in float64 from the same parameters.
    xr = x.detach().double().requires_grad_(True)
    A = adapted.lora_A.detach().double().requires_grad_(True)
    B = adapted.lora_B.detach().double().requires_grad_(True)
    m = adapted.magnitude.detach().double().requires_grad_(True)
    V = base.weight.detach().double() + (B @ A) * adapted.scaling
    W = m.unsqueeze(1) * V / V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
    ref = torch.nn.functional.linear(xr, W, base.bias.detach().double())
    ref.sum().backward()
    tol = {torch.float16: 5e-2, torch.bfloat16: 1e-1, torch.float32: 1e-4, torch.float64: 1e-9}[dtype]
    torch.testing.assert_close(out.double(), ref.detach(), atol=tol, rtol=tol)
    for name, got, want in (
        ("x", grads["x"], xr.grad),
        ("A", grads["A"], A.grad),
        ("B", grads["B"], B.grad),
        ("m", grads["m"], m.grad),
    ):
        assert got is not None and want is not None
        torch.testing.assert_close(got.double(), want, atol=tol, rtol=tol, msg=name)


def test_dora_half_classifier_step_and_full_state_round_trip():
    """apply_dora_to on a tiny classifier with a zero row in a wrapped
    layer, whole net converted to half: forward through the following
    projection is finite, an optimizer update on a float32 loss
    reduction moves only the adapter/magnitude, the frozen base is
    unchanged, and the complete state_dict (magnitude included)
    reloads to identical finite outputs."""
    torch.manual_seed(0)
    net = _TinyNet()
    with torch.no_grad():
        net.layers[0].weight[0].zero_()
    apply_dora_to(net, "layers.*", r=2, alpha=4.0)
    net = net.half()
    base_snapshot = {n: p.detach().clone() for n, p in net.named_parameters() if not p.requires_grad}
    x = torch.randn(5, 8).half()

    out = net(x)
    assert out.dtype == torch.float16 and torch.isfinite(out).all()

    optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1)
    loss = (net(x).float() ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    optimizer.step()
    assert all(torch.equal(p.detach(), base_snapshot[n]) for n, p in net.named_parameters() if n in base_snapshot)

    reloaded = _TinyNet()
    apply_dora_to(reloaded, "layers.*", r=2, alpha=4.0)
    reloaded = reloaded.half()
    reloaded.load_state_dict(net.state_dict())
    assert "layers.0.magnitude" in net.state_dict()
    torch.testing.assert_close(reloaded(x), net(x), atol=0, rtol=0)
    assert torch.isfinite(reloaded(x)).all()
