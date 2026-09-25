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


# -------------------------------------------------------------------------
# FIX-003: adapters inherit the wrapped tensor's dtype / device
# -------------------------------------------------------------------------

_PLACEMENT_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]


def _wrap_lora(base):
    return LoRALinear(base, r=2, alpha=4.0)


def _wrap_dora(base):
    from nnx import DoRALinear

    return DoRALinear(base, r=2, alpha=4.0)


def _wrap_ia3(base):
    from nnx import IA3Linear

    return IA3Linear(base)


@pytest.mark.parametrize("wrap", [_wrap_lora, _wrap_dora, _wrap_ia3], ids=["lora", "dora", "ia3"])
@pytest.mark.parametrize("dtype", _PLACEMENT_DTYPES, ids=str)
def test_adapter_inherits_base_dtype_device(wrap, dtype):
    """Wrapping an already-converted base must work immediately: adapter
    parameters take the base weight's dtype/device, the forward runs
    without a corrective `.to()`, the output keeps the base dtype, and
    the base tensor is neither replaced nor recast nor trained."""
    torch.manual_seed(0)
    base = nn.Linear(4, 4).to(dtype)
    original_weight = base.weight
    adapted = wrap(base)
    assert base.weight is original_weight and base.weight.dtype == dtype
    for name, p in adapted.named_parameters():
        assert p.dtype == dtype, name
        assert p.device == base.weight.device, name

    x = torch.ones(2, 4, dtype=dtype, requires_grad=True)
    out = adapted(x)
    assert out.dtype == dtype and torch.isfinite(out).all()
    out.float().sum().backward()
    assert base.weight.grad is None and (base.bias is None or base.bias.grad is None)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapted.parameters())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_ia3_preserves_half_output_and_composes(dtype):
    """IA3 used to allocate a float32 scaling vector, silently promoting
    a half pipeline to float32 so the next half Linear crashed."""
    from nnx import IA3Linear

    torch.manual_seed(0)
    first = IA3Linear(nn.Linear(4, 4).to(dtype))
    nxt = nn.Linear(4, 2).to(dtype)
    x = torch.ones(2, 4, dtype=dtype)
    hidden = first(x)
    assert hidden.dtype == dtype
    assert nxt(hidden).dtype == dtype


@pytest.mark.parametrize("wrap", [_wrap_lora, _wrap_dora, _wrap_ia3], ids=["lora", "dora", "ia3"])
def test_meta_base_allocates_meta_adapters(wrap):
    """A meta-device base must get meta adapters (no materialization and
    no forward is attempted — meta construction is not a forward pass)."""
    base = nn.Linear(4, 4, device="meta")
    adapted = wrap(base)
    for name, p in adapted.named_parameters():
        assert p.device.type == "meta", name
    assert base.weight.device.type == "meta"


def test_apply_lora_to_preconverted_net_trains_new_params():
    """apply_lora_to on a net converted BEFORE wrapping: the traversal
    inherits placement per layer, the forward composes with the next
    layer immediately, and an optimizer built afterwards owns exactly
    the newly registered adapter Parameters."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 3)).double()
    n = apply_lora_to(net, "*", r=2, alpha=4.0)
    assert n == 2
    assert all(p.dtype == torch.float64 for p in net.parameters())
    trainable = [p for p in net.parameters() if p.requires_grad]
    assert {id(p) for p in trainable} == {id(net[0].lora_A), id(net[0].lora_B), id(net[2].lora_A), id(net[2].lora_B)}
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    x = torch.randn(5, 8, dtype=torch.float64)
    before = [p.detach().clone() for p in trainable]
    frozen = {name: p.detach().clone() for name, p in net.named_parameters() if not p.requires_grad}
    loss = (net(x) ** 2).mean()
    loss.backward()
    optimizer.step()
    assert net(x).dtype == torch.float64
    assert any(not torch.equal(a, b.detach()) for a, b in zip(before, trainable, strict=True))
    assert all(torch.equal(p.detach(), frozen[name]) for name, p in net.named_parameters() if name in frozen)


# -------------------------------------------------------------------------
# FIX-002: adapter artifacts are selected by ownership, not name substrings
# -------------------------------------------------------------------------


def test_lora_collision_name_cannot_overwrite_base(tmp_path):
    """A submodule whose *name* contains ``lora_A`` must not leak its frozen
    base tensors into an adapter-only file, and a source dict carrying
    such colliding base keys must not modify any non-adapter tensor."""
    torch.manual_seed(0)
    net = nn.ModuleDict({"lora_A_projection": nn.Linear(4, 4), "lora_B_head": nn.Linear(4, 2)})
    assert apply_lora_to(net, "*", r=2) == 2
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}

    path = save_lora_weights(net, tmp_path / "lora.pt")
    saved = torch.load(path, weights_only=True)
    assert set(saved) == {
        "lora_A_projection.lora_A",
        "lora_A_projection.lora_B",
        "lora_B_head.lora_A",
        "lora_B_head.lora_B",
    }

    # Full state dict with every non-adapter tensor deliberately altered.
    source = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for k, v in source.items():
        if not (k.endswith(".lora_A") or k.endswith(".lora_B")):
            v.zero_()
    assert load_lora_weights(net, source) == 4
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_load_lora_weights_counts_only_accepted_adapter_tensors():
    """Unrelated keys are ignored (counted as not loaded); a wrong-shaped
    but genuinely owned key still raises the native load error; a root
    wrapper has unprefixed keys."""
    torch.manual_seed(0)
    net = _TinyNet()
    apply_lora_to(net, "layers.*", r=2)
    valid = net.state_dict()["layers.0.lora_A"].clone().fill_(0.5)
    n = load_lora_weights(
        net, {"layers.0.lora_A": valid, "layers.0.base.weight": torch.zeros(16, 8), "nonsense": torch.ones(1)}
    )
    assert n == 1
    assert torch.all(net.layers[0].lora_A == 0.5)
    assert not torch.all(net.layers[0].base.weight == 0)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_lora_weights(net, {"layers.0.lora_A": torch.zeros(3, 8)})

    root = LoRALinear(nn.Linear(4, 4), r=2)
    sd = {k: v for k, v in root.state_dict().items()}
    assert set(sd) == {"base.weight", "base.bias", "lora_A", "lora_B"}
    assert (
        load_lora_weights(
            root, {"lora_A": torch.zeros(2, 4), "lora_B": torch.zeros(4, 2), "base.weight": torch.zeros(4, 4)}
        )
        == 2
    )
    assert not torch.all(root.base.weight == 0)


# -------------------------------------------------------------------------
# FIX-013: wrappers inherit the wrapped layer's train/eval mode
# -------------------------------------------------------------------------


def _dropout_wrappers():
    from nnx import DoRALinear, apply_dora_to

    return {"lora": (LoRALinear, apply_lora_to), "dora": (DoRALinear, apply_dora_to)}


def _set_nonzero_adapter(wrapper) -> None:
    with torch.no_grad():
        nn.init.normal_(wrapper.lora_B, std=0.5)


def _reference_forward(wrapper, x, *, dropout_active: bool):
    """Recompute a LoRA / DoRA forward with an explicit dropout switch, so
    a seeded wrapper call can be compared with a seeded reference instead
    of two random masks."""
    import torch.nn.functional as F

    from nnx import DoRALinear

    p = wrapper.lora_dropout.p if isinstance(wrapper.lora_dropout, nn.Dropout) else 0.0
    if isinstance(wrapper, DoRALinear):
        update = F.dropout((wrapper.lora_B @ wrapper.lora_A) * wrapper.scaling, p, dropout_active)
        v = wrapper.base.weight + update
        w = wrapper.magnitude.unsqueeze(1) * (v / v.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8))
        return F.linear(x, w, wrapper.base.bias)
    dropped = F.dropout(x, p, dropout_active)
    return wrapper.base(x) + (dropped @ wrapper.lora_A.t() @ wrapper.lora_B.t()) * wrapper.scaling


def _mode_map(module: nn.Module) -> dict[str, bool]:
    return {name: m.training for name, m in module.named_modules()}


@pytest.mark.parametrize("kind", ["lora", "dora"])
def test_eval_adapter_injection_preserves_mode(kind):
    """Injecting a nonzero-dropout adapter into an eval model keeps every
    wrapper and dropout child in eval, so inference with nonzero adapter
    weights is deterministic."""
    _, apply = _dropout_wrappers()[kind]
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    assert apply(model, "*", r=2, alpha=4.0, dropout=0.5) == 2
    assert not any(m.training for m in model.modules())
    for idx in (0, 2):
        _set_nonzero_adapter(model[idx])
    x = torch.randn(8, 4)
    with torch.no_grad():
        first, second = model(x), model(x)
    assert torch.equal(first, second)
    for idx in (0, 2):
        assert torch.allclose(model[idx](x), _reference_forward(model[idx], x, dropout_active=False))


@pytest.mark.parametrize("kind", ["lora", "dora"])
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
def test_direct_wrapper_inherits_base_mode(kind, training):
    """Direct construction, not only the apply helpers, inherits the base
    mode; a train-mode base keeps dropout active (checked against a seeded
    dropout reference)."""
    cls, _ = _dropout_wrappers()[kind]
    torch.manual_seed(0)
    base = nn.Linear(4, 3)
    base.train(training)
    wrapper = cls(base, r=2, alpha=4.0, dropout=0.5)
    assert wrapper.training is training
    assert wrapper.lora_dropout.training is training
    assert wrapper.base.training is training
    _set_nonzero_adapter(wrapper)
    x = torch.randn(6, 4)
    with torch.no_grad():
        torch.manual_seed(123)
        out = wrapper(x)
        torch.manual_seed(123)
        expected = _reference_forward(wrapper, x, dropout_active=training)
        torch.manual_seed(123)
        no_dropout = _reference_forward(wrapper, x, dropout_active=False)
    assert torch.allclose(out, expected)
    assert torch.allclose(out, no_dropout) is (not training)


@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
def test_ia3_wrapper_inherits_base_mode(training):
    """IA3 has no dropout, but its wrapper still reports the base mode so
    per-module mode maps stay faithful after injection."""
    from nnx import IA3Linear

    base = nn.Linear(4, 3)
    base.train(training)
    assert IA3Linear(base).training is training


def test_apply_ia3_to_preserves_mixed_child_modes():
    from nnx import apply_ia3_to

    parent = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)).eval()
    parent[1].train()
    assert apply_ia3_to(parent, "*") == 2
    assert _mode_map(parent) == {"": False, "0": False, "0.base": False, "1": True, "1.base": True}


@pytest.mark.parametrize("tuner_name", ["PromptTuner", "PrefixTuner"])
def test_transformer_tuners_inherit_model_mode(tuner_name):
    """Prompt / prefix tuners report the wrapped model's mode and leave the
    model's own (mixed) submodule modes untouched."""
    import nnx
    from nnx import NNTransformerParams, TransformerNN

    model = TransformerNN(
        NNTransformerParams(
            input_dim=32,
            output_dim=32,
            dropout_prob=0.0,
            vocab_size=32,
            n_layers=2,
            n_heads=2,
            d_model=16,
            ffn_mult=2,
            max_seq_len=16,
        )
    ).eval()
    model.blocks[1].train()
    model_modes = _mode_map(model)
    tuner = getattr(nnx, tuner_name)(model)
    assert tuner.training is False
    assert all(not m.training for name, m in tuner.named_modules() if not name.startswith("model"))
    assert _mode_map(model) == model_modes


@pytest.mark.parametrize("kind", ["lora", "dora"])
def test_adapter_injection_preserves_mixed_child_modes(kind):
    """A parent with mixed child modes keeps each replaced child's own
    mode; nothing is flattened to the parent's (or a fresh module's) flag."""
    _, apply = _dropout_wrappers()[kind]
    parent = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)).eval()
    parent[1].train()
    apply(parent, "*", r=2, alpha=4.0, dropout=0.5)
    assert parent.training is False
    assert parent[0].training is False
    assert parent[0].base.training is False
    assert parent[0].lora_dropout.training is False
    assert parent[1].training is True
    assert parent[1].base.training is True
    assert parent[1].lora_dropout.training is True


@pytest.mark.parametrize("kind", ["lora", "dora"])
def test_later_train_call_reactivates_adapter_dropout(kind):
    """Preservation does not pin the wrapper in eval: a later
    ``model.train()`` activates LoRA input dropout / DoRA matrix dropout."""
    _, apply = _dropout_wrappers()[kind]
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4)).eval()
    apply(model, "*", r=2, alpha=4.0, dropout=0.5)
    wrapper = model[0]
    _set_nonzero_adapter(wrapper)
    model.train()
    assert wrapper.training and wrapper.lora_dropout.training
    x = torch.randn(6, 4)
    with torch.no_grad():
        torch.manual_seed(7)
        out = model(x)
        torch.manual_seed(7)
        expected = _reference_forward(wrapper, x, dropout_active=True)
    assert torch.allclose(out, expected)
    assert not torch.allclose(out, _reference_forward(wrapper, x, dropout_active=False))


@pytest.mark.parametrize("kind", ["lora", "dora"])
def test_loading_adapter_state_keeps_modes_and_adds_no_mode_key(kind, tmp_path):
    """Modes are runtime state: loading full or adapter-only state leaves
    the caller's per-module modes alone and serializes no mode field."""
    _, apply = _dropout_wrappers()[kind]
    torch.manual_seed(0)
    target = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2)).eval()
    target[1].train()
    apply(target, "*", r=2, alpha=4.0, dropout=0.5)
    source = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2)).train()
    apply(source, "*", r=2, alpha=4.0, dropout=0.5)
    _set_nonzero_adapter(source[0])
    before = _mode_map(target)

    state = source.state_dict()
    registered = {n for n, _ in source.named_parameters()} | {n for n, _ in source.named_buffers()}
    assert set(state) == registered  # no extra (mode) entry is serialized
    target.load_state_dict(state)
    assert _mode_map(target) == before

    path = tmp_path / "adapter.pt"
    save_lora_weights(source, path)
    saved = torch.load(path, weights_only=True)
    assert set(saved) == {"0.lora_A", "0.lora_B", "1.lora_A", "1.lora_B"}
    assert load_lora_weights(target, path) == len(saved)
    assert _mode_map(target) == before


def test_inference_helpers_restore_wrapped_mixed_modes():
    """``NNModel.predict`` / ``evaluate`` on a LoRA-wrapped net with mixed
    modes restore every wrapper, base and dropout flag, not just the root."""
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=3, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    model.net.eval()
    assert apply_lora_to(model.net, "*", r=2, alpha=4.0, dropout=0.5) >= 2
    wrappers = [m for m in model.net.modules() if isinstance(m, LoRALinear)]
    wrappers[0].train()  # deliberate mixed mode under an eval root
    before = _mode_map(model.net)
    assert any(before.values()) and not all(before.values())

    x = torch.randn(5, 4)
    model.predict(x)
    assert _mode_map(model.net) == before
    loader = DataLoader(TensorDataset(x, torch.tensor([0, 1, 2, 0, 1])), batch_size=5)
    model.evaluate(loader=loader)
    assert _mode_map(model.net) == before
