"""Tests for nnx.peft.prefix — PrefixTuner + save/load."""

from __future__ import annotations

import pytest
import torch

from nnx import (
    Activations,
    NNTransformerParams,
    PrefixTuner,
    TransformerNN,
    load_prefix_weights,
    save_prefix_weights,
    set_seed,
)


def _tiny_transformer() -> TransformerNN:
    """Small TransformerNN fixture — kept tiny for test speed."""
    params = NNTransformerParams(
        # Base NNParams fields (input_dim=output_dim=vocab_size is the LM convention).
        input_dim=100,
        output_dim=100,
        dropout_prob=0.0,
        activation=Activations.RELU,
        n_heads=4,
        # Transformer-specific.
        vocab_size=100,
        n_layers=2,
        d_model=32,
        max_seq_len=64,
    )
    return TransformerNN(params)


def test_prefix_tuner_freezes_base():
    """Every parameter of the wrapped TransformerNN must be frozen
    on construction — the PEFT contract."""
    set_seed(0)
    model = _tiny_transformer()
    tuner = PrefixTuner(model, n_prefix=4)
    # Every base param frozen.
    for name, p in tuner.model.named_parameters():
        assert not p.requires_grad, f"base parameter {name!r} not frozen"
    # The prefix tensors themselves ARE trainable.
    for p in tuner.prefix_keys:
        assert p.requires_grad
    for p in tuner.prefix_values:
        assert p.requires_grad


def test_prefix_tuner_only_prefix_trainable():
    """`trainable_parameters()` must yield exactly the prefix tensors —
    nothing from the wrapped model."""
    set_seed(0)
    model = _tiny_transformer()
    tuner = PrefixTuner(model, n_prefix=4)
    trainable = list(tuner.trainable_parameters())
    # 2 layers × {K, V} = 4 tensors.
    assert len(trainable) == 2 * model.params.n_layers
    # All trainable items are the exact tensor objects from
    # prefix_keys / prefix_values (identity, not just value-equal).
    expected_ids = {id(p) for p in tuner.prefix_keys} | {id(p) for p in tuner.prefix_values}
    assert {id(p) for p in trainable} == expected_ids


def test_prefix_tuner_forward_shape_unchanged():
    """Output shape from the wrapper must match the base model's
    output shape exactly — the prefix lives on the K/V side, not the
    query side, so the sequence dimension is preserved."""
    set_seed(0)
    model = _tiny_transformer()
    tuner = PrefixTuner(model, n_prefix=5)
    tokens = torch.randint(0, model.params.vocab_size, (2, 16))
    out = tuner(tokens)
    assert out.shape == (2, 16, model.params.vocab_size)


def test_prefix_tuner_save_load_round_trip(tmp_path):
    """Save the prefix tensors, load into a fresh tuner, verify the
    loaded values match bit-exactly."""
    set_seed(0)
    model_a = _tiny_transformer()
    tuner_a = PrefixTuner(model_a, n_prefix=4)
    # Mutate the prefix tensors away from init so the round-trip is
    # detectable.
    with torch.no_grad():
        for p in tuner_a.prefix_keys:
            p.fill_(0.42)
        for p in tuner_a.prefix_values:
            p.fill_(-0.17)

    path = save_prefix_weights(tuner_a, tmp_path / "prefix.pt")
    assert path.endswith("prefix.pt")

    # Fresh model + fresh tuner (different init, but same shape).
    set_seed(1)
    model_b = _tiny_transformer()
    tuner_b = PrefixTuner(model_b, n_prefix=4)

    n_loaded = load_prefix_weights(tuner_b, path)
    assert n_loaded > 0

    for pa, pb in zip(tuner_a.prefix_keys, tuner_b.prefix_keys, strict=False):
        assert torch.equal(pa.detach(), pb.detach())
    for pa, pb in zip(tuner_a.prefix_values, tuner_b.prefix_values, strict=False):
        assert torch.equal(pa.detach(), pb.detach())


def test_prefix_tuner_param_count_under_1pct():
    """Prefix params must be <1% of base params on a representative
    config — the central efficiency claim of prefix tuning.

    Uses a slightly larger TransformerNN than the rest of this file's
    tests: the absolute prefix budget grows linearly in
    n_layers * n_heads * head_dim, but the base parameter count grows
    quadratically in d_model (the SwiGLU FFN and QKV projections both
    dominate). On a TINY model the <1% claim doesn't hold — bumping
    d_model to 64 and n_layers to 4 puts us well inside the regime
    where prefix tuning is actually efficient (the central PEFT claim).
    """
    set_seed(0)
    params = NNTransformerParams(
        input_dim=100,
        output_dim=100,
        dropout_prob=0.0,
        activation=Activations.RELU,
        n_heads=4,
        vocab_size=100,
        n_layers=4,
        d_model=64,
        max_seq_len=64,
    )
    model = TransformerNN(params)
    tuner = PrefixTuner(model, n_prefix=2)
    base_params = sum(p.numel() for p in tuner.model.parameters())
    prefix_params = sum(p.numel() for p in tuner.trainable_parameters())
    ratio = prefix_params / base_params
    assert ratio < 0.01, (
        f"prefix params {prefix_params} are {ratio:.2%} of base params {base_params}; "
        f"prefix tuning's efficiency claim wants <1%"
    )


def test_prefix_tuner_validates_n_prefix():
    """n_prefix <= 0 must raise — a zero-prefix tuner is a no-op."""
    model = _tiny_transformer()
    with pytest.raises(ValueError, match="n_prefix"):
        PrefixTuner(model, n_prefix=0)
    with pytest.raises(ValueError, match="n_prefix"):
        PrefixTuner(model, n_prefix=-1)


def test_prefix_tuner_rejects_already_tuned_model():
    """A second PrefixTuner on the same net silently hijacked the first:
    the patched forwards read mha._nnx_prefix_tuner, which the second
    tuner overwrote — so the first tuner's parameters stopped receiving
    gradients while its forward injected the SECOND tuner's prefixes
    (training an optimizer over tuner-1's params became a silent no-op).
    Loud rejection instead; deepcopy remains the supported way to fork
    a tuned net."""
    set_seed(0)
    model = _tiny_transformer()
    PrefixTuner(model, n_prefix=2)
    with pytest.raises(ValueError, match="already prefix-tuned"):
        PrefixTuner(model, n_prefix=2)


def test_prefix_tuner_kv_cache_matches_full_forward():
    """Cache-path parity: incremental forward_with_cache on a
    prefix-tuned TransformerNN must match the full forward's last-token
    logits at every decode step, and the cache must hold real-token K/V
    only. Pre-fix, the patched MHA cached the prefix-INJECTED K/V, so
    every decode step re-prepended the prefix on top of the cached copy
    (n_prefix duplicate slots per step) and inflated the RoPE offset by
    n_prefix — cached logits drifted ~2.0 from the full forward."""
    set_seed(0)
    net = _tiny_transformer()
    tuner = PrefixTuner(net, n_prefix=3)
    assert tuner is not None  # patching happens in __init__
    net.eval()

    seq = torch.randint(0, 100, (1, 12))
    with torch.no_grad():
        _, past = net.forward_with_cache(seq[:, :4], past_kvs=None)
        for i in range(4, 12):
            inc, past = net.forward_with_cache(seq[:, i : i + 1], past_kvs=past)
            full = net(seq[:, : i + 1])
            assert torch.allclose(inc[:, -1, :], full[:, -1, :], atol=1e-4), (
                f"cached logits diverged at position {i}: "
                f"max diff {(inc[:, -1, :] - full[:, -1, :]).abs().max().item():.4f}"
            )
            # No prefix slots may leak into the cache.
            assert past[0][0].size(-2) == i + 1, (
                f"cache holds {past[0][0].size(-2)} slots at position {i}; expected {i + 1} real tokens"
            )


def test_prefix_tuner_deepcopy_is_independent():
    """copy.deepcopy of a prefix-tuned net must be fully decoupled.
    Pre-fix, the patched MHA forwards were instance closures, which
    deepcopy treats as atomic — the copy's attention silently kept
    reading the ORIGINAL weights and prefix params (corrupting
    quantize-on-copy, born-again frozen teachers, and surgery copies).
    The patch is now a MethodType-bound module-level function with its
    refs stored on the MHA, so deepcopy rebinds and re-references
    through the memo."""
    import copy

    set_seed(0)
    net = _tiny_transformer()
    tuner = PrefixTuner(net, n_prefix=3)
    net.eval()
    ids = torch.randint(0, 100, (1, 6))

    with torch.no_grad():
        baseline = net(ids).clone()
        clone = copy.deepcopy(net)
        clone.eval()
        assert torch.equal(clone(ids), baseline)

        # Mutating the ORIGINAL (weights + prefix) must not move the clone.
        net.blocks[0].attn.w_qkv.weight.add_(1.0)
        tuner.prefix_keys[0].add_(5.0)
        assert torch.equal(clone(ids), baseline), "deepcopy aliases the original"

        # The clone's own prefix params are live (its attention reads them).
        clone.blocks[0].attn._nnx_prefix_tuner.prefix_keys[0].add_(5.0)
        assert not torch.equal(clone(ids), baseline)


def test_prefix_tuned_net_torch_save_round_trips(tmp_path):
    """torch.save/load of a WHOLE prefix-tuned net must round-trip:
    MethodType pickles by name-lookup on the instance, which the
    class-level _prefix_patched_forward alias resolves. Pre-fix the
    save succeeded but the load died with AttributeError — a silently
    unloadable artifact."""
    set_seed(0)
    net = _tiny_transformer()
    PrefixTuner(net, n_prefix=3)
    net.eval()
    ids = torch.randint(0, 100, (1, 6))
    with torch.no_grad():
        baseline = net(ids).clone()

    path = tmp_path / "prefix_net.pt"
    torch.save(net, path)
    loaded = torch.load(path, weights_only=False)
    loaded.eval()
    with torch.no_grad():
        assert torch.equal(loaded(ids), baseline)
        # The loaded net's prefix refs are live and self-consistent.
        loaded.blocks[0].attn._nnx_prefix_tuner.prefix_keys[0].add_(5.0)
        assert not torch.equal(loaded(ids), baseline)
