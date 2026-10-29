"""Tests for nnx.peft.dora — DoRALinear + apply_dora_to."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import (
    DoRALinear,
    LoRALinear,
    apply_dora_to,
    apply_lora_to,
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


def test_dora_refuses_a_base_with_its_own_forward_and_leaves_the_model_unchanged():
    import pytest
    from torch import nn

    from nnx.peft import apply_dora_to

    class Scaled(nn.Linear):
        def forward(self, x):
            return super().forward(x) * 2

    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), Scaled(4, 2))
    with pytest.raises(TypeError, match="Scaled"):
        apply_dora_to(model, "*")
    assert type(model[0]) is nn.Linear and type(model[2]) is Scaled  # nothing replaced
    assert all(p.requires_grad for p in model.parameters())  # nothing frozen


def test_a_dora_base_replaced_later_by_a_custom_forward_is_refused_at_forward():
    import pytest
    import torch
    from torch import nn

    from nnx.peft import DoRALinear

    class Scaled(nn.Linear):
        def forward(self, x):
            return super().forward(x) * 2

    layer = DoRALinear(nn.Linear(3, 2), r=2)
    swapped = Scaled(3, 2)
    swapped.load_state_dict(layer.base.state_dict())
    layer.base = swapped  # what a later QAT prepare does to the slot
    with pytest.raises(TypeError, match="QAT prepare"):
        layer(torch.randn(1, 3))


def test_a_wrapper_that_fails_to_build_leaves_every_slot_and_flag_unchanged():
    import pytest
    from torch import nn

    from nnx.peft import LoRALinear
    from nnx.peft._targets import wrap_selected_linears

    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    built = []

    def wrap(base):
        if built:
            raise RuntimeError("cannot build the second wrapper")
        built.append(LoRALinear(base, r=2))  # freezes the first base
        return built[-1]

    with pytest.raises(RuntimeError, match="second wrapper"):
        wrap_selected_linears(model, ("*",), skip_inside=LoRALinear, helper="test", wrap=wrap)
    assert type(model[0]) is nn.Linear and type(model[2]) is nn.Linear
    assert all(p.requires_grad for p in model.parameters())


def test_dora_wraps_a_subclass_without_its_own_forward_and_keeps_frozen_flags_on_refusal():
    import pytest
    import torch
    from torch import nn

    from nnx.peft import apply_dora_to

    class Tagged(nn.Linear):  # no forward of its own
        pass

    model = nn.Sequential(Tagged(3, 4), nn.ReLU(), nn.MultiheadAttention(4, 1).out_proj)
    x = torch.randn(2, 3)
    expected = model[0](x)
    assert apply_dora_to(model, "*") == 2  # NonDynamicallyQuantizableLinear passes too
    assert torch.allclose(model[0](x), expected, atol=1e-6)

    class Scaled(nn.Linear):
        def forward(self, x):
            return super().forward(x) * 2

    refused = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), Scaled(4, 2))
    refused[0].bias.requires_grad_(False)  # frozen by the caller before the call
    flags = [p.requires_grad for p in refused.parameters()]
    with pytest.raises(TypeError):
        apply_dora_to(refused, "*")
    assert [p.requires_grad for p in refused.parameters()] == flags


# -------------------------------------------------------------------------
# FIX-056 (#523): bases whose forward path DoRA's recomposed matmul skips
# -------------------------------------------------------------------------


_X = torch.randn(5, 6, generator=torch.Generator().manual_seed(3))


def _mlp(seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 2))


def _instance_forward(layer: nn.Linear) -> None:
    stock = layer.forward
    layer.forward = lambda inp: stock(inp) + 1.0  # type: ignore[method-assign]


def _legacy(norm: str):
    import warnings

    from torch.nn.utils import spectral_norm, weight_norm

    def apply(layer: nn.Linear) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # legacy weight_norm's deprecation notice
            (weight_norm if norm == "weight_norm" else spectral_norm)(layer)

    return apply


def _prune_mask(layer: nn.Linear) -> None:
    from nnx.prune import magnitude_prune

    magnitude_prune(nn.Sequential(layer), 0.5, bake=False)


# Each feature with the remedy its refusal names. Context of #523 measured the
# wrapper's silent step-0 drift: forward hook 0.3573, pre-hook 0.1937, instance
# forward 0.7405, legacy spectral_norm 3.5342; masks and legacy weight_norm
# crashed the second training step.
_BYPASSED = {
    "forward_hook": (lambda layer: layer.register_forward_hook(lambda m, i, o: o * 2), "forward hook.*LoRA"),
    "forward_pre_hook": (
        lambda layer: layer.register_forward_pre_hook(lambda m, args: (args[0] * 0.5,)),
        "forward pre-hook.*LoRA",
    ),
    "backward_hook": (lambda layer: layer.register_full_backward_hook(lambda m, gi, go: None), "backward hook.*LoRA"),
    "backward_pre_hook": (
        lambda layer: layer.register_full_backward_pre_hook(lambda m, go: None),
        "backward hook.*LoRA",
    ),
    "instance_forward": (_instance_forward, "instance forward override.*LoRA"),
    "prune_mask": (_prune_mask, "unbaked prune mask.*Bake the mask"),
    "legacy_weight_norm": (_legacy("weight_norm"), "legacy weight_norm.*Remove the norm"),
    "legacy_spectral_norm": (_legacy("spectral_norm"), "legacy weight_norm / spectral_norm.*Remove the norm"),
}


def _state(net: nn.Module):
    return (
        [(name, type(m)) for name, m in net.named_modules()],
        [(name, p.requires_grad) for name, p in net.named_parameters()],
        [(name, m.training) for name, m in net.named_modules()],
    )


@pytest.mark.parametrize("kind", list(_BYPASSED))
@pytest.mark.parametrize("how", ["apply_dora_to", "DoRALinear"])
def test_dora_refuses_a_base_whose_forward_path_it_would_skip_and_changes_nothing(kind, how):
    add, match = _BYPASSED[kind]
    net = _mlp()
    net[2].eval()  # a mixed-mode net: every flag must survive the refusal
    add(net[0])
    before = _state(net)
    with pytest.raises(TypeError, match=match):
        if how == "apply_dora_to":
            apply_dora_to(net, "*", r=2)
        else:
            DoRALinear(net[0], r=2)
    assert _state(net) == before


def test_lora_still_wraps_every_base_dora_refuses():
    for kind, (add, _) in _BYPASSED.items():
        net = _mlp().eval()  # spectral_norm's power iteration runs in train mode only
        add(net[0])
        with torch.no_grad():
            before = net(_X)
            apply_lora_to(net, "0", r=2)
            assert torch.equal(net(_X), before), kind


def _l1_prune_base(wrapper: DoRALinear) -> None:
    torch.nn.utils.prune.l1_unstructured(wrapper.base, "weight", 0.5)


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda w: w.base.register_forward_hook(lambda m, i, o: None), "forward hook"),
        (lambda w: w.base.register_forward_pre_hook(lambda m, args: None), "forward pre-hook"),
        (lambda w: w.base.register_full_backward_hook(lambda m, gi, go: None), "backward hook"),
        (lambda w: w.base.register_full_backward_pre_hook(lambda m, go: None), "backward hook"),
        (_l1_prune_base, "unbaked prune mask"),
    ],
    ids=["forward_hook", "forward_pre_hook", "backward_hook", "backward_pre_hook", "l1_unstructured"],
)
def test_a_base_changed_after_wrapping_is_refused_at_forward(change, match):
    """#523 Context: hooks on ``.base`` fired 0 times under DoRA (LoRA: 1),
    and a mask on ``.base`` reloaded 0.6289 away from the saved model."""
    net = _mlp()
    apply_dora_to(net, "0", r=2)
    change(net[0])
    with pytest.raises(TypeError, match=f"DoRA's base now has an? {match}.*, added after wrapping"):
        net(_X)


def _parametrized(kind: str):
    from torch.nn.utils import parametrizations

    def apply(layer: nn.Linear) -> None:
        (parametrizations.weight_norm if kind == "weight_norm" else parametrizations.spectral_norm)(layer)

    return apply


class _Tagged(nn.Linear):  # no forward of its own
    pass


def _tagged(net: nn.Sequential) -> None:
    tagged = _Tagged(6, 6)
    tagged.load_state_dict(net[0].state_dict())
    net[0] = tagged


@pytest.mark.parametrize(
    "prepare",
    [
        lambda net: _parametrized("weight_norm")(net[0]),
        lambda net: _parametrized("spectral_norm")(net[0]),
        _tagged,
    ],
    ids=["parametrizations.weight_norm", "parametrizations.spectral_norm", "subclass_without_forward"],
)
def test_dora_still_wraps_parametrized_and_subclassed_bases_with_parity_and_trains(prepare):
    net = _mlp()
    prepare(net)
    net.eval()  # spectral_norm's power iteration runs in train mode only
    with torch.no_grad():
        before = net(_X)
        assert apply_dora_to(net, "0", r=2) == 1
        after = net(_X)
    assert (after - before).abs().max() <= 1e-6
    net.train()
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1)
    for _ in range(3):  # the refused legacy forms crashed the second step
        loss = net(_X).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(net(_X)).all()


@pytest.mark.parametrize("strict", [False, True], ids=["non_strict", "strict"])
def test_a_plain_dora_net_still_exports(strict):
    """#523 review: non-strict ``torch.export`` traces with FakeTensor weights,
    which the quantized-weight check must not take for a quantized one."""
    net = _mlp()
    apply_dora_to(net, "0", r=2)
    net.eval()
    exported = torch.export.export(net, (_X,), strict=strict)
    with torch.no_grad():
        assert torch.allclose(exported.module()(_X), net(_X))


def test_a_plain_dora_net_runs_under_fake_tensor_mode():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.func import functional_call

    net = _mlp()
    apply_dora_to(net, "0", r=2)
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake = {k: mode.from_tensor(v) for k, v in net.state_dict().items()}
        out = functional_call(net, fake, (mode.from_tensor(_X),))
    assert out.shape == (5, 2)


def test_a_training_forward_runs_a_dora_base_parametrization_once():
    """#523 review: inspecting the weight must not read ``base.weight``, which
    reruns the parametrization (a second spectral_norm power-iteration step)."""
    from torch.nn.utils import parametrizations

    net = _mlp()
    parametrizations.spectral_norm(net[0])
    norm = net[0].parametrizations.weight[0]
    apply_dora_to(net, "0", r=2)
    calls = []
    stock = type(norm).forward
    norm.forward = lambda w: (calls.append(1), stock(norm, w))[1]  # type: ignore[method-assign]
    for _ in range(3):
        net(_X).sum().backward()
    assert len(calls) == 3


# -------------------------------------------------------------------------
# #523 review: NNx's own hook-based tools (nnx.viz.summary) stay exempt
# -------------------------------------------------------------------------


def test_only_hooks_an_nnx_tool_registers_are_exempt_and_never_after_it():
    from nnx.peft.dora import _tool_hooks

    net = _mlp()
    apply_dora_to(net, "0", r=2)
    before = net[0].base.register_forward_hook(lambda m, i, o: None)
    with _tool_hooks():
        with pytest.raises(TypeError, match="forward hook"):
            net(_X)  # a user's hook from before the tool stays refused
        before.remove()
        tool = net[0].base.register_forward_pre_hook(lambda m, args: None)
        with _tool_hooks():  # nested: the outer tool's hooks stay exempt inside
            net(_X)
        net(_X)  # the inner block's exit restores the outer one
    with pytest.raises(TypeError, match="forward pre-hook"):
        net(_X)  # nothing outlives the block
    tool.remove()
    net(_X)


@pytest.mark.parametrize("kind", ["instance_forward", "prune_mask", "legacy_weight_norm", "legacy_spectral_norm"])
def test_a_tool_block_relaxes_only_the_hook_check(kind):
    from nnx.peft.dora import _tool_hooks

    add, match = _BYPASSED[kind]
    net = _mlp()
    apply_dora_to(net, "0", r=2)
    with _tool_hooks():
        add(net[0].base)  # applied after wrapping, inside the block
        with pytest.raises(TypeError, match=match.split(".*")[0]):
            net(_X)


def test_a_tool_block_does_not_reach_another_thread():
    import threading

    from nnx.peft.dora import _tool_hooks

    net = _mlp()
    apply_dora_to(net, "0", r=2)
    errors: list[BaseException] = []
    entered, hooked = threading.Event(), threading.Event()

    def other() -> None:
        entered.wait(5)
        net[0].base.register_forward_hook(lambda m, i, o: None)
        hooked.set()
        try:
            net(_X)
        except TypeError as e:
            errors.append(e)

    worker = threading.Thread(target=other)
    worker.start()
    with _tool_hooks():
        entered.set()
        hooked.wait(5)
        worker.join(5)
    assert len(errors) == 1 and "forward hook" in str(errors[0])


# -------------------------------------------------------------------------
# DoRA + QAT (FIX-059, #526)
# -------------------------------------------------------------------------


class _CountingLoader(list):
    """A list loader that counts the epochs that started reading it."""

    reads = 0

    def __iter__(self):
        self.reads += 1
        return super().__iter__()


def _qat_fit(model, loop: str, callback, loader) -> None:
    """One QAT epoch through ``NNModel.train`` or ``Trainer.train``."""
    from nnx import NNEvaluationDataPoint, NNOptimParams, NNTrainerParams, NNTrainParams, Trainer
    from nnx.quantize import qat_train_step_factory

    optim = NNOptimParams.builder().sgd(max_lr=0.1).build()
    if loop == "nnmodel":
        params = NNTrainParams(n_epochs=1, train_loader=loader, optim=optim)
        model.train(params=params, callbacks=[callback], train_step_fn=qat_train_step_factory())
        return

    def trainer_step(ctx):
        ctx.model.net.train()
        optimizer = ctx.optimizers["main"]
        optimizer.zero_grad()
        x, target = ctx.batch
        loss = ctx.model.loss_fn(ctx.model.net(x), target)
        loss.backward()
        optimizer.step()
        return NNEvaluationDataPoint(loss=float(loss.detach()), error=float(loss.detach()))

    params = NNTrainerParams(n_epochs=1, train_loader=loader, optims={"main": optim})
    Trainer(model).train(params=params, trainer_step_fn=trainer_step, callbacks=[callback])


def _qat_dora_model(input_dim: int = 64):
    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams

    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(
            input_dim=input_dim, output_dim=2, hidden_dims=[64], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    apply_dora_to(model.net, "layers.0")
    return model


def _types(net: nn.Module) -> dict[str, str]:
    return {name: type(module).__name__ for name, module in net.named_modules() if name}


@pytest.mark.parametrize("loop", ["nnmodel", "trainer"])
def test_qat_refuses_a_dora_base_before_prepare_and_changes_nothing(loop, tmp_path, monkeypatch):
    """#526: QAT used to prepare (and, as the DoRA forward's refusal
    propagated, convert) the whole net before DoRA refused at the first
    batch, leaving an int8 net no forward could run."""
    pytest.importorskip("torchao")
    from nnx.quantize import QATLifecycleCallback

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model = _qat_dora_model()
    x, y = torch.randn(4, 64), torch.tensor([0, 1, 0, 1])
    before, weight = _types(model.net), model.net.layers[1].weight
    original = weight.detach().clone()
    expected = model.predict(x).logits
    loader = _CountingLoader([(x, y)])
    callback = QATLifecycleCallback(groupsize=32)

    with pytest.raises(TypeError, match=r"base of the DoRA layer layers\.0: .*Use LoRA") as refusal:
        _qat_fit(model, loop, callback, loader)

    assert "Nothing was modified" in str(refusal.value)
    assert loader.reads == 0  # refused before the first batch
    assert _types(model.net) == before
    assert model.net.layers[1].weight is weight and weight.dtype == torch.float32
    assert torch.equal(weight, original)
    assert not callback.is_prepared and not callback.is_converted
    assert callback.checkpoint_transforms() == ()
    assert (model.predict(x).logits == expected).all()


@pytest.mark.parametrize("loop", ["nnmodel", "trainer"])
def test_qat_trains_a_net_whose_dora_base_it_does_not_swap(loop, tmp_path, monkeypatch):
    """#526: only a DoRA base QAT would swap is refused; one whose
    in_features the groupsize does not divide stays a plain Linear."""
    pytest.importorskip("torchao")
    from nnx.quantize import QATLifecycleCallback

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model = _qat_dora_model(input_dim=48)  # 32 does not divide 48
    x, y = torch.randn(4, 48), torch.tensor([0, 1, 0, 1])
    callback = QATLifecycleCallback(groupsize=32)

    _qat_fit(model, loop, callback, [(x, y)])

    assert callback.is_prepared and callback.is_converted
    types = _types(model.net)
    assert types["layers.0"] == "DoRALinear" and types["layers.0.base"] == "Linear"
    assert types["layers.1"] == "Int8DynActInt4WeightLinear"
    with torch.no_grad():
        assert torch.isfinite(model.net(x)).all()
