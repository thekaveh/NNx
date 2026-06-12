"""Tests for nnx.peft.lora — LoRALinear + apply_lora_to + save/load."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nnx import (
    Activations,
    Devices,
    LoRALinear,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    apply_lora_to,
    load_lora_weights,
    save_lora_weights,
    set_seed,
)

# -------------------------------------------------------------------------
# LoRALinear basics
# -------------------------------------------------------------------------


def test_lora_linear_rejects_non_linear_base():
    with pytest.raises(TypeError, match="nn.Linear"):
        LoRALinear(nn.Conv2d(3, 4, 3), r=2)


def test_lora_linear_validates_r_alpha_dropout():
    base = nn.Linear(8, 4)
    with pytest.raises(ValueError, match="rank r"):
        LoRALinear(base, r=0)
    with pytest.raises(ValueError, match="alpha"):
        LoRALinear(base, r=2, alpha=0.0)
    with pytest.raises(ValueError, match="dropout"):
        LoRALinear(base, r=2, dropout=1.0)
    with pytest.raises(ValueError, match="dropout"):
        LoRALinear(base, r=2, dropout=-0.1)


def test_lora_linear_freezes_base_on_construction():
    base = nn.Linear(8, 4)
    # Base starts trainable.
    assert all(p.requires_grad for p in base.parameters())
    LoRALinear(base, r=2)
    # After wrap: every base param frozen.
    assert all(not p.requires_grad for p in base.parameters())


def test_lora_linear_initial_output_equals_base():
    """B is zero-initialized so y == base(x) at step 0. This is the
    invariant that lets LoRA fine-tuning start from the pretrained
    behavior exactly."""
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2, alpha=4.0)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_linear_forward_shape():
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2, alpha=4.0)
    out = lora(torch.randn(3, 8))
    assert out.shape == (3, 4)


def test_lora_linear_only_lora_params_trainable():
    """The frozen base means only lora_A and lora_B should appear in
    a list of trainable parameters. This is THE LoRA contract — get
    this wrong and full-rank weights also update during fine-tuning."""
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2)
    trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
    assert set(trainable) == {"lora_A", "lora_B"}


def test_lora_linear_in_out_features_passthrough():
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2)
    assert lora.in_features == 8
    assert lora.out_features == 4


# -------------------------------------------------------------------------
# apply_lora_to
# -------------------------------------------------------------------------


class _TinyNet(nn.Module):
    """3-layer MLP — the canonical apply_lora_to target."""

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


def test_apply_lora_to_requires_pattern():
    with pytest.raises(ValueError, match="at least one"):
        apply_lora_to(_TinyNet())


def test_apply_lora_to_wraps_matched_only():
    net = _TinyNet()
    n = apply_lora_to(net, "layers.0", r=2, alpha=4.0)
    assert n == 1
    assert isinstance(net.layers[0], LoRALinear)
    # Unmatched layers untouched.
    assert isinstance(net.layers[1], nn.Linear) and not isinstance(net.layers[1], LoRALinear)
    assert isinstance(net.layers[2], nn.Linear) and not isinstance(net.layers[2], LoRALinear)


def test_apply_lora_to_wildcard_wraps_all_linears():
    net = _TinyNet()
    n = apply_lora_to(net, "layers.*", r=2)
    assert n == 3
    assert all(isinstance(net.layers[i], LoRALinear) for i in range(3))


def test_apply_lora_to_is_idempotent_for_already_wrapped():
    """A second apply_lora_to call against patterns that already match
    LoRA-wrapped layers should be a no-op: the inner .base of an
    existing LoRALinear must not be re-wrapped."""
    net = _TinyNet()
    n_first = apply_lora_to(net, "layers.*", r=2)
    assert n_first == 3
    # All three layers are now LoRALinear. Re-applying with the same
    # pattern should NOT double-wrap.
    n_second = apply_lora_to(net, "layers.*", r=2)
    assert n_second == 0
    assert all(isinstance(net.layers[i], LoRALinear) for i in range(3))
    # And no .base.base double-wrapping.
    for i in range(3):
        assert isinstance(net.layers[i].base, nn.Linear)
        assert not isinstance(net.layers[i].base, LoRALinear)


def test_apply_lora_to_preserves_forward_at_init():
    """After wrapping, the forward pass at step 0 should equal the
    pre-wrap forward exactly — base weights unchanged + LoRA B=0."""
    torch.manual_seed(0)
    net = _TinyNet()
    x = torch.randn(2, 8)
    pre = net(x)
    apply_lora_to(net, "layers.*", r=2, alpha=4.0)
    post = net(x)
    assert torch.allclose(pre, post, atol=1e-6)


# -------------------------------------------------------------------------
# save / load lora weights
# -------------------------------------------------------------------------


def test_save_load_lora_weights_round_trip(tmp_path):
    """Apply LoRA, mutate the A/B matrices, save, load into a fresh
    wrapped net — the LoRA matrices must come back identical."""
    torch.manual_seed(0)
    net_a = _TinyNet()
    apply_lora_to(net_a, "layers.*", r=2, alpha=4.0)
    # Mutate the LoRA params away from their init so the round-trip is
    # detectable (zero-init B would otherwise match trivially).
    with torch.no_grad():
        for n, p in net_a.named_parameters():
            if "lora_" in n:
                p.fill_(0.42)

    path = save_lora_weights(net_a, tmp_path / "lora.pt")
    assert path.endswith("lora.pt")

    net_b = _TinyNet()
    apply_lora_to(net_b, "layers.*", r=2, alpha=4.0)
    # Pre-load: B's are still zero on net_b.
    for n, p in net_b.named_parameters():
        if "lora_B" in n:
            assert torch.all(p == 0)

    n_loaded = load_lora_weights(net_b, path)
    assert n_loaded > 0
    # Post-load: every LoRA param on net_b matches net_a's.
    sa = dict(net_a.named_parameters())
    sb = dict(net_b.named_parameters())
    for n in sa:
        if "lora_" in n:
            assert torch.equal(sa[n].detach(), sb[n].detach())


def test_save_lora_weights_excludes_base_params(tmp_path):
    """The saved checkpoint must contain ONLY lora_A / lora_B keys,
    never base.weight / base.bias — that's the point of LoRA's
    storage efficiency."""
    net = _TinyNet()
    apply_lora_to(net, "layers.*", r=2)
    path = save_lora_weights(net, tmp_path / "lora.pt")

    sd = torch.load(path, weights_only=True)
    assert len(sd) > 0
    for k in sd:
        assert "lora_A" in k or "lora_B" in k, f"unexpected non-LoRA key in saved checkpoint: {k!r}"


def test_load_lora_weights_from_dict():
    """Passing a dict directly works the same as a file path."""
    torch.manual_seed(0)
    net_a = _TinyNet()
    apply_lora_to(net_a, "layers.*", r=2)
    with torch.no_grad():
        for n, p in net_a.named_parameters():
            if "lora_A" in n:
                p.fill_(0.7)

    sd = {k: v for k, v in net_a.state_dict().items() if "lora_" in k}

    net_b = _TinyNet()
    apply_lora_to(net_b, "layers.*", r=2)
    load_lora_weights(net_b, sd)
    for n, p in net_b.named_parameters():
        if "lora_A" in n:
            assert torch.all(p == 0.7)


def test_load_lora_weights_rejects_bad_source_type():
    net = _TinyNet()
    apply_lora_to(net, "layers.*", r=2)
    with pytest.raises(TypeError, match="path or dict"):
        load_lora_weights(net, 12345)


def test_load_lora_weights_with_empty_dict_is_zero_op():
    """A partial / empty LoRA state-dict must not silently corrupt the
    target net. Document the contract: `load_lora_weights(net, {})` is a
    no-op that returns 0 (nothing loaded) rather than wiping out the
    existing matrices or raising."""
    torch.manual_seed(0)
    net = _TinyNet()
    apply_lora_to(net, "layers.*", r=2)
    # Mutate the LoRA matrices so we can verify the empty-dict load
    # does NOT overwrite them.
    with torch.no_grad():
        for n, p in net.named_parameters():
            if "lora_" in n:
                p.fill_(0.33)
    pre = {n: p.clone() for n, p in net.named_parameters() if "lora_" in n}

    n_loaded = load_lora_weights(net, {})
    assert n_loaded == 0

    post = {n: p.clone() for n, p in net.named_parameters() if "lora_" in n}
    for k in pre:
        assert torch.equal(pre[k], post[k]), f"empty-dict load_lora_weights mutated {k!r}"


def test_load_lora_weights_into_unadapted_model_returns_zero():
    """Loading a LoRA checkpoint into a module that was never LoRA-fied
    must report 0 tensors loaded — pre-fix it returned len(source dict)
    because load_state_dict(strict=False) silently drops keys the
    module doesn't have, masking exactly the misuse the docstring warns
    about."""
    torch.manual_seed(0)
    net = _TinyNet()
    apply_lora_to(net, "layers.*", r=2)
    sd = {n: p.clone() for n, p in net.named_parameters() if "lora_" in n}
    assert len(sd) > 0

    plain = _TinyNet()  # never adapted — no lora_* keys exist
    n_loaded = load_lora_weights(plain, sd)
    assert n_loaded == 0, f"reported {n_loaded} loaded into an un-adapted model"


# -------------------------------------------------------------------------
# End-to-end: PEFT fine-tuning preserves base weights
# -------------------------------------------------------------------------


def _classification_loaders(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(3, 8, generator=g)
    cls = torch.randint(0, 3, (256,), generator=g)
    X = means[cls] + 0.5 * torch.randn(256, 8, generator=g)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, cls),
        batch_size=32,
        shuffle=True,
    )


def test_lora_finetune_leaves_base_weights_frozen(tmp_path, monkeypatch):
    """End-to-end: pretrain → snapshot → apply_lora_to → fine-tune →
    verify every base.weight / base.bias is BIT-EXACTLY unchanged but
    every lora_A / lora_B HAS moved. This is THE PEFT contract."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    model = NNModel(
        net_params=NNParams(
            input_dim=8,
            output_dim=3,
            hidden_dims=[16, 16],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )
    # Phase 1: pretrain
    model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=_classification_loaders(seed=0),
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=1,
                cooldown=1,
                threshold=1e-3,
            ),
        )
    )

    # Snapshot every parameter pre-LoRA so we can compare term-by-term
    # after fine-tuning.
    pre_snapshot = {n: p.clone() for n, p in model.net.named_parameters()}

    # Phase 2: wrap with LoRA
    n_wrapped = apply_lora_to(model.net, "layers.*", r=2, alpha=4.0)
    assert n_wrapped == 3  # three Linear layers in the FeedFwdNN

    # Snapshot the LoRA params at init so we can verify they MOVED.
    lora_init = {n: p.clone() for n, p in model.net.named_parameters() if "lora_" in n}

    # Phase 3: fine-tune on a DIFFERENT distribution (different seed)
    model.train(
        params=NNTrainParams(
            n_epochs=3,
            train_loader=_classification_loaders(seed=42),
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=1,
                cooldown=1,
                threshold=1e-3,
            ),
        )
    )

    # Invariant 1: every base parameter is bit-exactly unchanged.
    for n, post in model.net.named_parameters():
        if "lora_" in n:
            continue
        # After apply_lora_to, the parameter names change: what was
        # `layers.0.weight` is now `layers.0.base.weight`. Strip the
        # `.base` to get the pre-LoRA key.
        pre_key = n.replace(".base.", ".")
        assert pre_key in pre_snapshot, f"no pre-snapshot entry for {n!r} (pre {pre_key!r})"
        assert torch.equal(post.detach(), pre_snapshot[pre_key]), (
            f"base parameter {n!r} drifted during LoRA fine-tuning"
        )

    # Invariant 2: every LoRA parameter has moved at least once
    # (lora_A could in principle stay close to init for a few steps;
    # lora_B is zero-init so any gradient at all moves it).
    for n, post in model.net.named_parameters():
        if "lora_B" in n:
            assert not torch.equal(post.detach(), lora_init[n]), (
                f"LoRA parameter {n!r} did not change during fine-tuning"
            )
