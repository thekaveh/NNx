"""Tests for ``nnx.surgery.low_rank_factorize``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams, low_rank_factorize

# ---------- Function-preservation: exact at max rank ------------------


def test_low_rank_factorize_is_function_preserving_at_max_rank():
    """At rank == min(out, in), SVD reconstructs the original Linear
    within FP rounding."""
    torch.manual_seed(0)
    linear = nn.Linear(16, 8)
    x = torch.randn(5, 16)
    orig_out = linear(x)

    factored = low_rank_factorize(linear, rank=8)
    new_out = factored(x)

    assert isinstance(factored, nn.Sequential)
    assert torch.allclose(orig_out, new_out, atol=1e-5), (
        f"low_rank_factorize at full rank broke function-preservation: "
        f"max diff {(orig_out - new_out).abs().max().item():.2e}"
    )


def test_low_rank_factorize_preserves_bias():
    """The output Linear must carry the original bias verbatim."""
    linear = nn.Linear(10, 6)
    factored = low_rank_factorize(linear, rank=6)
    assert factored[0].bias is None  # down: bias=False
    assert factored[1].bias is not None  # up: bias preserved
    assert torch.equal(factored[1].bias.data, linear.bias.data)


def test_low_rank_factorize_handles_bias_false():
    """Bias-less Linears must factorize without complaint and still
    preserve function at full rank."""
    torch.manual_seed(1)
    linear = nn.Linear(12, 8, bias=False)
    x = torch.randn(3, 12)
    orig_out = linear(x)
    factored = low_rank_factorize(linear, rank=8)
    new_out = factored(x)
    assert factored[1].bias is None
    assert torch.allclose(orig_out, new_out, atol=1e-5)


# ---------- Approximation behaviour at lower ranks --------------------


def test_low_rank_factorize_reduces_parameter_count():
    """Param count drops from out*in + out_bias to k*(in+out) + out_bias."""
    linear = nn.Linear(64, 32)
    orig_params = sum(p.numel() for p in linear.parameters())
    factored = low_rank_factorize(linear, rank=8)
    new_params = sum(p.numel() for p in factored.parameters())
    expected = 8 * (64 + 32) + 32  # k*(in+out) + bias
    assert new_params == expected
    assert new_params < orig_params


def test_surgery_without_recipe_is_rejected_before_persistent_training():
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    model.net.layers[1] = low_rank_factorize(model.net.layers[1], rank=2)
    with pytest.raises(ValueError, match="no reconstruction recipe"):
        model._assert_reconstructible_topology()


def test_low_rank_factorize_approx_error_bounded_by_dropped_singular_values():
    """The Frobenius error of the rank-k reconstruction equals the
    L2 norm of the dropped singular values (Eckart-Young)."""
    torch.manual_seed(2)
    linear = nn.Linear(16, 8, bias=False)
    W = linear.weight.data
    S = torch.linalg.svdvals(W)

    rank = 3
    factored = low_rank_factorize(linear, rank=rank)
    W_approx = factored[1].weight.data @ factored[0].weight.data  # (out, k) @ (k, in)
    error = (W - W_approx).norm().item()
    expected = S[rank:].norm().item()
    # FP slack on the SVD path.
    assert error == pytest.approx(expected, rel=1e-4, abs=1e-5)


# ---------- Error handling --------------------------------------------


def test_low_rank_factorize_rejects_non_linear():
    with pytest.raises(TypeError, match="requires an nn.Linear"):
        low_rank_factorize(nn.ReLU(), rank=2)


def test_low_rank_factorize_rejects_rank_zero():
    linear = nn.Linear(8, 4)
    with pytest.raises(ValueError, match="rank must be in"):
        low_rank_factorize(linear, rank=0)


def test_low_rank_factorize_rejects_rank_above_max():
    linear = nn.Linear(8, 4)  # max rank = 4
    with pytest.raises(ValueError, match="rank must be in"):
        low_rank_factorize(linear, rank=5)


def test_low_rank_factorize_rejects_unknown_method():
    linear = nn.Linear(8, 4)
    with pytest.raises(ValueError, match="unsupported method"):
        low_rank_factorize(linear, rank=2, method="bogus")


def test_low_rank_factorize_method_kwarg_accepts_svd():
    """The default and the explicit 'svd' must behave identically."""
    torch.manual_seed(3)
    linear = nn.Linear(10, 6)
    a = low_rank_factorize(linear, rank=4)
    b = low_rank_factorize(linear, rank=4, method="svd")
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_low_rank_factorize_does_not_advance_global_rng():
    """low_rank_factorize() must be a no-op on the global torch RNG
    stream — both factored Linears are fully overwritten, so they are
    built uninitialized (skip_init). Pre-fix, their kaiming inits drew
    from the default generator, silently diverging any seeded caller
    pipeline."""
    linear = nn.Linear(16, 8)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    low_rank_factorize(linear, rank=4)
    assert torch.equal(torch.get_rng_state(), state)


# ---------------- FIX-016: factors preserve trainability and modes ----------------


@pytest.mark.parametrize(
    ("w_flag", "b_flag", "has_bias"),
    [
        pytest.param(True, True, True, id="w_train-b_train"),
        pytest.param(True, False, True, id="w_train-b_frozen"),
        pytest.param(False, True, True, id="w_frozen-b_train"),
        pytest.param(False, False, True, id="w_frozen-b_frozen"),
        pytest.param(False, None, False, id="w_frozen-no_bias"),
        pytest.param(True, None, False, id="w_train-no_bias"),
    ],
)
def test_low_rank_preserves_weight_bias_roles(w_flag, b_flag, has_bias):
    """Both SVD factors represent the one source weight and inherit its
    flag; only `up.bias` carries the source bias and inherits the bias
    flag; `down` has no bias. The source Linear is untouched."""
    torch.manual_seed(0)
    base = nn.Linear(6, 4, bias=has_bias)
    base.weight.requires_grad_(w_flag)
    if has_bias:
        base.bias.requires_grad_(b_flag)
    source_flags = {n: p.requires_grad for n, p in base.named_parameters()}

    factors = low_rank_factorize(base, rank=4)
    down, up = factors[0], factors[1]
    assert down.weight.requires_grad is w_flag and up.weight.requires_grad is w_flag
    assert down.bias is None
    if has_bias:
        assert up.bias.requires_grad is b_flag and up.bias is not base.bias
    else:
        assert up.bias is None
    assert {n: p.requires_grad for n, p in base.named_parameters()} == source_flags

    x = torch.randn(3, 6)
    torch.testing.assert_close(factors(x), base(x), atol=1e-5, rtol=1e-5)
    trainable = [p for p in factors.parameters() if p.requires_grad]
    if trainable:
        factors(x).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        assert all(p.grad is None for p in factors.parameters() if not p.requires_grad)


@pytest.mark.parametrize("mode", [True, False], ids=["train", "eval"])
def test_low_rank_preserves_module_mode(mode):
    base = nn.Linear(6, 4).train(mode)
    factors = low_rank_factorize(base, rank=4)
    assert factors.training is mode
    assert all(m.training is mode for m in factors)
