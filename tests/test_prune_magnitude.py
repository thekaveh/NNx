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
