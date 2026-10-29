"""Tests for nnx.prune.magnitude — magnitude_prune."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import prune as nnx_prune


class _TinyNet(nn.Module):
    """3-layer MLP — the canonical pruning target."""

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


# -------------------------------------------------------------------------
# magnitude_prune core behavior
# -------------------------------------------------------------------------


def test_magnitude_prune_zeros_correct_fraction():
    """50% sparsity should set ~half of each layer's weights to exactly 0.

    L1-unstructured at sparsity=s zeroes the s-fraction smallest-magnitude
    weights per layer. For a Linear(in, out) we expect ⌊s · in · out⌋
    zeros (PyTorch rounds the count, not the fraction)."""
    torch.manual_seed(0)
    net = _TinyNet()
    nnx_prune.magnitude_prune(net, sparsity=0.5)

    for i, layer in enumerate(net.layers):
        total = layer.weight.numel()
        zeros = (layer.weight == 0).sum().item()
        # PyTorch's prune.l1_unstructured zeros round(s * numel) entries.
        expected = round(0.5 * total)
        assert zeros == expected, f"layer {i}: expected {expected} zeros at 50% sparsity, got {zeros} / {total}"


def test_magnitude_prune_preserves_state_dict_keys_when_bake_true():
    """bake=True (default) MUST keep state_dict keys shape-compatible with
    the unpruned network — this is THE checkpoint-compat invariant. After
    pruning, the state_dict should still carry plain `layers.N.weight` /
    `layers.N.bias`, NOT `weight_orig` + `weight_mask`."""
    net = _TinyNet()
    pre_keys = set(net.state_dict().keys())
    nnx_prune.magnitude_prune(net, sparsity=0.3)
    post_keys = set(net.state_dict().keys())
    assert pre_keys == post_keys, (
        f"state_dict keys diverged under bake=True: added={post_keys - pre_keys}, removed={pre_keys - post_keys}"
    )


def test_magnitude_prune_returns_count():
    """The function must report how many layers it actually pruned, so
    callers can verify their pattern matched the intended set."""
    net = _TinyNet()
    n = nnx_prune.magnitude_prune(net, sparsity=0.2)
    assert n == 3  # three nn.Linear in _TinyNet


def test_magnitude_prune_pattern_filters_correctly():
    """layer_pattern is an fnmatch glob against dotted module name —
    only matched layers should pick up zeros."""
    torch.manual_seed(0)
    net = _TinyNet()
    # Snapshot pre-prune so we can assert non-matched layers are untouched.
    pre = {n: p.clone() for n, p in net.named_parameters()}

    n_pruned = nnx_prune.magnitude_prune(net, sparsity=0.5, layer_pattern="layers.0")
    assert n_pruned == 1

    # layers.0: zeros present.
    assert (net.layers[0].weight == 0).sum().item() > 0
    # layers.1 and layers.2: bit-exactly unchanged.
    assert torch.equal(net.layers[1].weight, pre["layers.1.weight"])
    assert torch.equal(net.layers[2].weight, pre["layers.2.weight"])


def test_magnitude_prune_idempotent_on_already_zero():
    """Calling magnitude_prune twice at the same sparsity should NOT
    double-prune — l1_unstructured picks the smallest-magnitude entries,
    which after the first prune are the same already-zeroed positions.
    Total zeros must equal the post-first-prune count, not 2x."""
    torch.manual_seed(0)
    net = _TinyNet()
    nnx_prune.magnitude_prune(net, sparsity=0.5)

    zeros_after_first = {i: (l.weight == 0).sum().item() for i, l in enumerate(net.layers)}

    nnx_prune.magnitude_prune(net, sparsity=0.5)
    zeros_after_second = {i: (l.weight == 0).sum().item() for i, l in enumerate(net.layers)}

    for i in zeros_after_first:
        assert zeros_after_first[i] == zeros_after_second[i], (
            f"layer {i}: second prune at same sparsity changed zero count "
            f"({zeros_after_first[i]} → {zeros_after_second[i]}) — not idempotent"
        )


def test_iterative_pruning_via_bake_false():
    """bake=False keeps the prune reparameterization in place — so the
    state_dict carries `weight_orig` + `weight_mask` instead of plain
    `weight`. This is the path users on iterative-pruning schedules
    (10% per epoch for N epochs, etc.) need so torch.nn.utils.prune
    can continue to compose with the existing mask."""
    net = _TinyNet()
    nnx_prune.magnitude_prune(net, sparsity=0.3, bake=False)
    sd_keys = set(net.state_dict().keys())
    # Every Linear's `weight` should have been replaced by `weight_orig`
    # + `weight_mask` in the state_dict.
    for i in range(3):
        assert f"layers.{i}.weight_orig" in sd_keys
        assert f"layers.{i}.weight_mask" in sd_keys
        assert f"layers.{i}.weight" not in sd_keys
        # Bias is untouched.
        assert f"layers.{i}.bias" in sd_keys


# -------------------------------------------------------------------------
# Edge cases — sparsity bounds, empty matches
# -------------------------------------------------------------------------


def test_magnitude_prune_rejects_invalid_sparsity():
    """Sparsity must be in [0, 1). A value outside the range can't
    correspond to any l1_unstructured amount, so we surface a clear
    error before delegating to torch."""
    net = _TinyNet()
    with pytest.raises(ValueError, match="sparsity"):
        nnx_prune.magnitude_prune(net, sparsity=-0.1)
    with pytest.raises(ValueError, match="sparsity"):
        nnx_prune.magnitude_prune(net, sparsity=1.0)
    with pytest.raises(ValueError, match="sparsity"):
        nnx_prune.magnitude_prune(net, sparsity=1.5)


def test_magnitude_prune_sparsity_zero_is_noop():
    """sparsity=0 is a valid call — zero zeros added — and must leave
    every weight bit-exactly unchanged."""
    torch.manual_seed(0)
    net = _TinyNet()
    pre = {n: p.clone() for n, p in net.named_parameters()}
    n = nnx_prune.magnitude_prune(net, sparsity=0.0)
    assert n == 3
    for n, post in net.named_parameters():
        assert torch.equal(post.detach(), pre[n]), f"sparsity=0 mutated {n!r}"


def test_magnitude_prune_no_match_returns_zero():
    """A pattern that matches no Linear submodule must return 0 and
    leave the net untouched."""
    torch.manual_seed(0)
    net = _TinyNet()
    pre = {n: p.clone() for n, p in net.named_parameters()}
    n = nnx_prune.magnitude_prune(net, sparsity=0.5, layer_pattern="nonexistent.*")
    assert n == 0
    for name, post in net.named_parameters():
        assert torch.equal(post.detach(), pre[name]), f"no-match prune mutated {name!r}"


def test_magnitude_prune_smallest_magnitudes_go_to_zero():
    """l1_unstructured zeros the smallest-magnitude entries. Verify the
    correctness property directly: after pruning at s, every surviving
    (non-zero) weight has |w| >= max |w| among the zeroed entries."""
    torch.manual_seed(0)
    net = _TinyNet()
    pre_weights = [layer.weight.detach().clone() for layer in net.layers]
    nnx_prune.magnitude_prune(net, sparsity=0.5)

    for layer, pre_w in zip(net.layers, pre_weights, strict=True):
        mask = layer.weight != 0
        if mask.any() and (~mask).any():
            kept_abs = pre_w[mask].abs()
            zeroed_abs = pre_w[~mask].abs()
            assert kept_abs.min() >= zeroed_abs.max(), "magnitude prune kept a smaller-magnitude weight than it zeroed"


def test_magnitude_prune_state_dict_load_round_trips_under_bake():
    """state_dict from a baked-pruned net should load CLEANLY into a
    fresh unpruned net with strict=True — proving the bake=True path
    is checkpoint-shape-compatible with the original layer layout.

    Note: PyTorch's ``prune.remove`` re-registers the baked ``weight``
    parameter at the END of the layer's parameter list (it was first
    de-registered as a Parameter when l1_unstructured turned it into
    a tensor, then re-added as a Parameter by remove). The state_dict
    CONTENTS still match key-for-key with the original network — that's
    the load_state_dict invariant — but the iteration order of
    ``named_parameters()`` shifts. Verify via the dict view instead."""
    torch.manual_seed(0)
    net_a = _TinyNet()
    nnx_prune.magnitude_prune(net_a, sparsity=0.5)
    sd = net_a.state_dict()

    net_b = _TinyNet()
    missing, unexpected = net_b.load_state_dict(sd, strict=True)
    # strict=True returns (missing_keys=[], unexpected_keys=[]) on success.
    assert missing == [] and unexpected == []

    # And every parameter on net_b matches the source by name.
    src = dict(net_a.named_parameters())
    dst = dict(net_b.named_parameters())
    assert set(src.keys()) == set(dst.keys())
    for name in src:
        assert torch.equal(src[name].detach(), dst[name].detach()), f"loaded {name!r} mismatch"


# -------------------------------------------------------------------------
# FIX-056 (#523): DoRALinear reads its base's weight directly
# -------------------------------------------------------------------------


def _dora_net() -> nn.Sequential:
    from nnx.peft import apply_dora_to

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 2))
    apply_dora_to(net, "0", r=2)
    return net


@pytest.mark.parametrize("pattern", ["0.base", "*"])
def test_unbaked_pruning_refuses_a_dora_base_and_modifies_nothing(pattern):
    """#523 Context: the accepted mask never applied, and a strict reload of a
    pruned DoRA net came back 0.6289 away from the saved model."""
    net = _dora_net()
    state = {k: v.clone() for k, v in net.state_dict().items()}
    hooks = [len(m._forward_pre_hooks) for m in net.modules()]
    with pytest.raises(ValueError, match=r"0\.base are the base of a DoRALinear.*Nothing was modified"):
        nnx_prune.magnitude_prune(net, 0.5, layer_pattern=pattern, bake=False)
    assert net.state_dict().keys() == state.keys()
    assert all(torch.equal(net.state_dict()[k], v) for k, v in state.items())
    assert [len(m._forward_pre_hooks) for m in net.modules()] == hooks


def test_baked_pruning_of_a_dora_base_and_unbaked_pruning_of_lora_and_ia3_bases_still_work():
    from nnx.peft import apply_ia3_to, apply_lora_to

    net = _dora_net()
    assert nnx_prune.magnitude_prune(net, 0.5, layer_pattern="0.base") == 1  # baked: a plain weight again
    assert (net[0].base.weight == 0).sum() == 18
    net(torch.randn(2, 6))
    for apply in (apply_lora_to, apply_ia3_to):
        other = nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 2))
        apply(other, "0")
        assert nnx_prune.magnitude_prune(other, 0.5, layer_pattern="0.base", bake=False) == 1
        assert hasattr(other[0].base, "weight_mask")


# -------------------------------------------------------------------------
# FIX-086 (#554): parametrized weights are refused before any layer is pruned
# -------------------------------------------------------------------------


def _legacy(norm):
    def apply(lin):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the legacy hooks' deprecation notice
            norm(lin)

    return apply


class _Scale(nn.Module):
    def forward(self, w):
        return 2 * w


def _parametrize_weight(lin):
    from torch.nn.utils import parametrize

    parametrize.register_parametrization(lin, "weight", _Scale())


_WEIGHT_REPARAMETERIZATIONS = {
    "weight_norm": nn.utils.parametrizations.weight_norm,
    "spectral_norm": nn.utils.parametrizations.spectral_norm,
    "orthogonal": nn.utils.parametrizations.orthogonal,
    "custom": _parametrize_weight,
    "legacy_weight_norm": _legacy(nn.utils.weight_norm),
    "legacy_spectral_norm": _legacy(nn.utils.spectral_norm),
}


def _zero_fraction(lin) -> float:
    return float((lin.weight == 0).float().mean())


def _snapshot(net):
    return {k: v.clone() for k, v in net.state_dict().items()}


def _assert_unchanged(net, state):
    assert sorted(net.state_dict()) == sorted(state)
    assert all(torch.equal(net.state_dict()[k], v) for k, v in state.items())


def _context_net():
    """The #554 Context net: only the SECOND Linear is weight-normed, built
    through ``NNModel(module=...)`` as NNx advises for such a net."""
    from nnx import Losses, NNModel, NNModelParams

    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    nn.utils.parametrizations.weight_norm(module[2])
    return NNModel(module=module, params=NNModelParams(loss=Losses.CROSS_ENTROPY)).net


@pytest.mark.parametrize("bake", [True, False])
def test_context_refuses_a_later_parametrized_linear_and_modifies_nothing(bake):
    """#554 Context: a raw TypeError after layer 0 was already pruned to 0.50."""
    net = _context_net()
    state = _snapshot(net)
    hooks = [len(m._forward_pre_hooks) for m in net.modules()]
    with pytest.raises(ValueError) as info:
        nnx_prune.magnitude_prune(net, 0.5, bake=bake)
    message = str(info.value)
    assert "Linear 2 (parametrized by _WeightNorm) has a parametrized weight" in message
    assert "remove_parametrizations" in message and "exclude it with layer_pattern" in message
    assert message.endswith("Nothing was modified.")
    _assert_unchanged(net, state)
    assert [len(m._forward_pre_hooks) for m in net.modules()] == hooks
    assert _zero_fraction(net[0]) == 0.0


def test_retry_after_removing_the_parametrization_prunes_each_layer_once():
    """#554 Context: the bake=False retry left layer 0 at 0.75 (pruned twice)."""
    from torch.nn.utils import parametrize

    net = _context_net()
    with pytest.raises(ValueError):
        nnx_prune.magnitude_prune(net, 0.5, bake=False)
    parametrize.remove_parametrizations(net[2], "weight")
    assert nnx_prune.magnitude_prune(net, 0.5, bake=False) == 2
    assert _zero_fraction(net[0]) == 0.5
    assert _zero_fraction(net[2]) == 0.5


@pytest.mark.parametrize("bake", [True, False])
@pytest.mark.parametrize("form", list(_WEIGHT_REPARAMETERIZATIONS))
def test_every_weight_reparameterization_after_a_plain_linear_is_refused(form, bake):
    torch.manual_seed(0)
    net = _TinyNet()
    _WEIGHT_REPARAMETERIZATIONS[form](net.layers[1])
    state = _snapshot(net)
    with pytest.raises(ValueError, match=r"Linear layers\.1 \(.+\) has a parametrized weight.*Nothing was modified"):
        nnx_prune.magnitude_prune(net, 0.5, bake=bake)
    _assert_unchanged(net, state)
    assert _zero_fraction(net.layers[0]) == 0.0


def test_one_error_lists_every_parametrized_linear():
    torch.manual_seed(0)
    net = _TinyNet()
    nn.utils.parametrizations.spectral_norm(net.layers[1])
    _legacy(nn.utils.weight_norm)(net.layers[2])
    state = _snapshot(net)
    with pytest.raises(
        ValueError,
        match=r"Linears layers\.1 \(parametrized by _SpectralNorm\), layers\.2 \(legacy weight_norm\) have a "
        r"parametrized weight.*exclude them with layer_pattern",
    ):
        nnx_prune.magnitude_prune(net, 0.5)
    _assert_unchanged(net, state)


@pytest.mark.parametrize("bake", [True, False])
def test_a_parametrized_root_linear_is_refused_without_the_layer_pattern_remedy(bake):
    """Excluding the root Linear with layer_pattern would leave nothing to prune."""
    torch.manual_seed(0)
    net = nn.Linear(16, 4)
    nn.utils.parametrizations.weight_norm(net)
    state = _snapshot(net)
    with pytest.raises(ValueError) as info:
        nnx_prune.magnitude_prune(net, 0.5, bake=bake)
    message = str(info.value)
    assert message.startswith(
        "magnitude_prune: the root Linear (parametrized by _WeightNorm) has a parametrized weight"
    )
    assert "remove_parametrizations" in message and "layer_pattern" not in message
    assert message.endswith("Nothing was modified.")
    _assert_unchanged(net, state)


def test_excluding_the_parametrized_linear_with_layer_pattern_prunes_the_rest():
    torch.manual_seed(0)
    net = _TinyNet()
    nn.utils.parametrizations.weight_norm(net.layers[1])
    assert nnx_prune.magnitude_prune(net, 0.5, layer_pattern="layers.[02]") == 2
    assert _zero_fraction(net.layers[0]) == 0.5


def test_a_bias_only_parametrization_still_prunes_every_target():
    from torch.nn.utils import parametrize

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    parametrize.register_parametrization(net[2], "bias", _Scale())
    assert nnx_prune.magnitude_prune(net, 0.5) == 2
    assert _zero_fraction(net[0]) == 0.5
    assert _zero_fraction(net[2]) == 0.5


def test_two_unbaked_calls_compose():
    """An unbaked prune mask is a reparameterization too, but not one the
    refusal counts: successive bake=False calls keep composing."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    assert nnx_prune.magnitude_prune(net, 0.5, bake=False) == 2
    assert nnx_prune.magnitude_prune(net, 0.5, bake=False) == 2
    assert _zero_fraction(net[0]) == 0.75
    assert _zero_fraction(net[2]) == 0.75
