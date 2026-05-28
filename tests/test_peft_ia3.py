"""Tests for nnx.peft.ia3 — IA3Linear + apply_ia3_to + save/load."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import (
    IA3Linear,
    apply_ia3_to,
    load_ia3_weights,
    save_ia3_weights,
)

# -------------------------------------------------------------------------
# IA3Linear basics
# -------------------------------------------------------------------------


def test_ia3_linear_rejects_non_linear_base():
    with pytest.raises(TypeError, match="nn.Linear"):
        IA3Linear(nn.Conv2d(3, 4, 3))


def test_ia3_freezes_base():
    base = nn.Linear(8, 4)
    assert all(p.requires_grad for p in base.parameters())
    IA3Linear(base)
    assert all(not p.requires_grad for p in base.parameters())


def test_ia3_initial_output_equals_base():
    """IA3's scaling vector is initialized to all-ones, so the layer
    output at step 0 is base(x) * 1.0 == base(x). Fine-tuning starts
    from the pretrained behavior exactly."""
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    ia3 = IA3Linear(base)
    x = torch.randn(3, 8)
    assert torch.allclose(ia3(x), base(x), atol=1e-6)


def test_ia3_initial_output_equals_base_no_bias():
    torch.manual_seed(1)
    base = nn.Linear(8, 4, bias=False)
    ia3 = IA3Linear(base)
    x = torch.randn(3, 8)
    assert torch.allclose(ia3(x), base(x), atol=1e-6)


def test_ia3_forward_shape():
    base = nn.Linear(8, 4)
    ia3 = IA3Linear(base)
    out = ia3(torch.randn(3, 8))
    assert out.shape == (3, 4)


def test_ia3_only_scaling_trainable():
    """The smallest adapter on the market — the ONLY trainable parameter
    on the wrapper is the per-dim scaling vector. The frozen base is
    contract-critical: if the base's weight/bias accidentally trains,
    IA3 collapses to a full fine-tune."""
    base = nn.Linear(8, 4)
    ia3 = IA3Linear(base)
    trainable = {n for n, p in ia3.named_parameters() if p.requires_grad}
    assert trainable == {"scaling"}


def test_ia3_scaling_init_is_ones():
    base = nn.Linear(8, 4)
    ia3 = IA3Linear(base)
    assert ia3.scaling.shape == (4,)
    assert torch.all(ia3.scaling == 1.0)


def test_ia3_in_out_features_passthrough():
    base = nn.Linear(8, 4)
    ia3 = IA3Linear(base)
    assert ia3.in_features == 8
    assert ia3.out_features == 4


def test_ia3_scaling_actually_scales_output():
    """Setting the scaling vector to a known non-unit value must produce
    base(x) * scaling exactly — this is the IA3 forward contract."""
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    ia3 = IA3Linear(base)
    with torch.no_grad():
        ia3.scaling.copy_(torch.tensor([2.0, 0.5, -1.0, 3.0]))
    x = torch.randn(3, 8)
    expected = base(x) * torch.tensor([2.0, 0.5, -1.0, 3.0])
    assert torch.allclose(ia3(x), expected, atol=1e-6)


# -------------------------------------------------------------------------
# apply_ia3_to
# -------------------------------------------------------------------------


class _TinyNet(nn.Module):
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


def test_apply_ia3_to_requires_pattern():
    with pytest.raises(ValueError, match="at least one"):
        apply_ia3_to(_TinyNet())


def test_apply_ia3_to_wraps_matched_only():
    net = _TinyNet()
    n = apply_ia3_to(net, "layers.0")
    assert n == 1
    assert isinstance(net.layers[0], IA3Linear)
    assert not isinstance(net.layers[1], IA3Linear)
    assert not isinstance(net.layers[2], IA3Linear)


def test_apply_ia3_to_wildcard_wraps_all_linears():
    net = _TinyNet()
    n = apply_ia3_to(net, "layers.*")
    assert n == 3
    assert all(isinstance(net.layers[i], IA3Linear) for i in range(3))


def test_apply_ia3_to_is_idempotent_for_already_wrapped():
    net = _TinyNet()
    n_first = apply_ia3_to(net, "layers.*")
    assert n_first == 3
    n_second = apply_ia3_to(net, "layers.*")
    assert n_second == 0
    for i in range(3):
        assert isinstance(net.layers[i].base, nn.Linear)
        assert not isinstance(net.layers[i].base, IA3Linear)


def test_apply_ia3_to_preserves_forward_at_init():
    torch.manual_seed(0)
    net = _TinyNet()
    x = torch.randn(2, 8)
    pre = net(x)
    apply_ia3_to(net, "layers.*")
    post = net(x)
    assert torch.allclose(pre, post, atol=1e-6)


# -------------------------------------------------------------------------
# save / load ia3 weights
# -------------------------------------------------------------------------


def test_save_load_ia3_weights_round_trip(tmp_path):
    """Apply IA3, mutate the scaling vectors, save, load into a fresh
    wrapped net — the scaling vectors must come back identical."""
    torch.manual_seed(0)
    net_a = _TinyNet()
    apply_ia3_to(net_a, "layers.*")
    with torch.no_grad():
        for n, p in net_a.named_parameters():
            if "scaling" in n:
                p.fill_(0.42)

    path = save_ia3_weights(net_a, tmp_path / "ia3.pt")
    assert path.endswith("ia3.pt")

    net_b = _TinyNet()
    apply_ia3_to(net_b, "layers.*")
    # Pre-load: scalings are still 1.0 on net_b.
    for n, p in net_b.named_parameters():
        if "scaling" in n:
            assert torch.all(p == 1.0)

    n_loaded = load_ia3_weights(net_b, path)
    assert n_loaded > 0
    sa = dict(net_a.named_parameters())
    sb = dict(net_b.named_parameters())
    for n in sa:
        if "scaling" in n:
            assert torch.equal(sa[n].detach(), sb[n].detach())


def test_save_ia3_weights_excludes_base_params(tmp_path):
    """Saved checkpoint must contain ONLY scaling keys — that's the
    point of IA3's storage efficiency (smallest adapter on the market)."""
    net = _TinyNet()
    apply_ia3_to(net, "layers.*")
    path = save_ia3_weights(net, tmp_path / "ia3.pt")

    sd = torch.load(path, weights_only=True)
    assert len(sd) > 0
    for k in sd:
        assert "scaling" in k, f"unexpected non-IA3 key in saved checkpoint: {k!r}"


def test_load_ia3_weights_from_dict():
    torch.manual_seed(0)
    net_a = _TinyNet()
    apply_ia3_to(net_a, "layers.*")
    with torch.no_grad():
        for n, p in net_a.named_parameters():
            if "scaling" in n:
                p.fill_(0.7)

    sd = {k: v for k, v in net_a.state_dict().items() if "scaling" in k}

    net_b = _TinyNet()
    apply_ia3_to(net_b, "layers.*")
    load_ia3_weights(net_b, sd)
    for n, p in net_b.named_parameters():
        if "scaling" in n:
            assert torch.all(p == 0.7)


def test_load_ia3_weights_rejects_bad_source_type():
    net = _TinyNet()
    apply_ia3_to(net, "layers.*")
    with pytest.raises(TypeError, match="path or dict"):
        load_ia3_weights(net, 12345)


def test_load_ia3_weights_with_empty_dict_is_zero_op():
    """`load_ia3_weights(net, {})` is a no-op that returns 0 — it must
    NOT overwrite or wipe the existing scaling vectors."""
    torch.manual_seed(0)
    net = _TinyNet()
    apply_ia3_to(net, "layers.*")
    with torch.no_grad():
        for n, p in net.named_parameters():
            if "scaling" in n:
                p.fill_(0.33)
    pre = {n: p.clone() for n, p in net.named_parameters() if "scaling" in n}

    n_loaded = load_ia3_weights(net, {})
    assert n_loaded == 0

    post = {n: p.clone() for n, p in net.named_parameters() if "scaling" in n}
    for k in pre:
        assert torch.equal(pre[k], post[k]), f"empty-dict load_ia3_weights mutated {k!r}"
