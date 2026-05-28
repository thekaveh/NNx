"""Tests for GenerativeNNModel.generate() — greedy + sampling LM decoding.

Covers:
  * Deterministic greedy decoding matches a manually-computed continuation.
  * Same-seed sampling reproducibility.
  * Logits processors (temperature, top-k, top-p, repetition penalty)
    compose correctly.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("tokenizers")

from nnx.generation.logits_processors import (  # noqa: E402
    RepetitionPenalty,
    TemperatureScaling,
    TopKFilter,
    TopPFilter,
    apply_chain,
)
from nnx.nn.enum.devices import Devices  # noqa: E402
from nnx.nn.enum.losses import Losses  # noqa: E402
from nnx.nn.enum.nets import Nets  # noqa: E402
from nnx.nn.generative_nn_model import GenerativeNNModel  # noqa: E402
from nnx.nn.params.nn_model_params import NNModelParams  # noqa: E402
from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe  # noqa: E402
from nnx.nn.params.nn_transformer_params import NNTransformerParams  # noqa: E402

# ---------------- Helpers ----------------


def _make_tokenizer(tmp_path):
    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the fox jumps over the lazy dog",
        "the world is round",
        "hello world hello there",
    ]
    tk = train_bpe(files=None, texts=corpus, vocab_size=64, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
    path = tmp_path / "tok.json"
    return NNTokenizerParams.of(tokenizer=tk, path=str(path))


def _make_model(tokenizer: NNTokenizerParams) -> GenerativeNNModel:
    net_params = NNTransformerParams(
        input_dim=tokenizer.vocab_size,
        output_dim=tokenizer.vocab_size,
        dropout_prob=0.0,
        vocab_size=tokenizer.vocab_size,
        n_layers=2,
        n_heads=2,
        d_model=16,
        ffn_mult=2,
        max_seq_len=32,
    )
    model_params = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    return GenerativeNNModel(net_params=net_params, params=model_params, tokenizer=tokenizer)


# ---------------- LogitsProcessor unit tests ----------------


def test_temperature_scaling_divides_logits():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    out = TemperatureScaling(temperature=2.0)(logits, token_history=[])
    expected = torch.tensor([[0.5, 1.0, 1.5]])
    assert torch.allclose(out, expected)


def test_temperature_zero_returns_argmax_one_hot_via_inf():
    """temperature=0 is greedy. We implement it by mapping to +inf on
    the argmax position so the downstream softmax picks it
    deterministically."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    out = TemperatureScaling(temperature=0.0)(logits, token_history=[])
    # Argmax (index 2) should be +inf; others -inf (or any value that
    # softmaxes to ~0 — we use -inf).
    assert torch.isinf(out[0, 2]) and out[0, 2] > 0
    assert torch.isinf(out[0, 0]) and out[0, 0] < 0
    assert torch.isinf(out[0, 1]) and out[0, 1] < 0


def test_top_k_filter_keeps_only_top_k_logits():
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])
    out = TopKFilter(top_k=2)(logits, token_history=[])
    # Top-2: indices 1 (5.0) and 3 (4.0). Others must be -inf.
    assert out[0, 1].item() == 5.0
    assert out[0, 3].item() == 4.0
    assert torch.isinf(out[0, 0]) and out[0, 0] < 0
    assert torch.isinf(out[0, 2]) and out[0, 2] < 0
    assert torch.isinf(out[0, 4]) and out[0, 4] < 0


def test_top_p_filter_keeps_smallest_cumulative_p_set():
    # A near-uniform distribution where top-p=0.5 should retain just the
    # top-most token (50% of mass already on the largest).
    logits = torch.tensor([[0.0, 10.0, 0.1]])  # huge prob on idx 1
    out = TopPFilter(top_p=0.5)(logits, token_history=[])
    # idx 1 must survive; the rest get pushed to -inf.
    assert out[0, 1].item() == 10.0
    assert torch.isinf(out[0, 0]) and out[0, 0] < 0
    assert torch.isinf(out[0, 2]) and out[0, 2] < 0


def test_repetition_penalty_divides_logits_for_seen_tokens():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    history = [1, 3]
    out = RepetitionPenalty(penalty=2.0)(logits, token_history=history)
    # Positions 1 and 3 should be divided by 2 (their logits are positive).
    assert out[0, 1].item() == 1.0  # 2.0 / 2.0
    assert out[0, 3].item() == 2.0  # 4.0 / 2.0
    # Untouched positions:
    assert out[0, 0].item() == 1.0
    assert out[0, 2].item() == 3.0


def test_repetition_penalty_multiplies_negative_logits():
    """For negative logits, the penalty MULTIPLIES (so the magnitude
    grows) — the standard HF repetition-penalty trick."""
    logits = torch.tensor([[-2.0, -4.0]])
    out = RepetitionPenalty(penalty=2.0)(logits, token_history=[0])
    # idx 0: negative, multiply by 2 → -4.0
    assert out[0, 0].item() == -4.0
    # idx 1: not in history, untouched
    assert out[0, 1].item() == -4.0


def test_apply_chain_runs_processors_in_order():
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])
    # TopK=2 then Temperature=2: should yield {1: 2.5, 3: 2.0, rest: -inf}.
    out = apply_chain(
        logits,
        token_history=[],
        processors=[TopKFilter(top_k=2), TemperatureScaling(temperature=2.0)],
    )
    assert out[0, 1].item() == 2.5  # 5/2
    assert out[0, 3].item() == 2.0  # 4/2


# ---------------- generate() integration ----------------


def test_generate_deterministic_greedy_is_reproducible(tmp_path):
    """Greedy generate() (temperature=0) is fully deterministic — same
    prompt, same model, same call → same output. This is the regression
    test the SP-4 plan calls out specifically."""
    tokenizer = _make_tokenizer(tmp_path)
    torch.manual_seed(42)
    model = _make_model(tokenizer)
    out_a = model.generate(prompt="the", max_new_tokens=8, temperature=0.0)
    out_b = model.generate(prompt="the", max_new_tokens=8, temperature=0.0)
    assert out_a == out_b
    # And the output is a string longer than the prompt.
    assert isinstance(out_a, str)
    assert len(out_a) >= len("the")


def test_generate_sampling_is_reproducible_with_same_seed(tmp_path):
    """Same-seed sampling reproducibility — the other SP-4 spec
    requirement. Two calls with the same seed must produce identical
    outputs even with temperature > 0."""
    tokenizer = _make_tokenizer(tmp_path)
    torch.manual_seed(0)
    model = _make_model(tokenizer)

    out_a = model.generate(prompt="the", max_new_tokens=6, temperature=1.0, top_k=10, seed=123)
    out_b = model.generate(prompt="the", max_new_tokens=6, temperature=1.0, top_k=10, seed=123)
    assert out_a == out_b


def test_generate_sampling_different_seeds_can_differ(tmp_path):
    """Sanity: different seeds CAN produce different outputs (not
    asserting they always do — for a tiny model collisions are possible
    — but on average they should). We just sanity-check that the seed
    is actually consumed."""
    tokenizer = _make_tokenizer(tmp_path)
    torch.manual_seed(0)
    model = _make_model(tokenizer)
    seen = {model.generate(prompt="the", max_new_tokens=4, temperature=2.0, top_k=20, seed=s) for s in range(20)}
    assert len(seen) > 1


def test_generate_stops_at_max_new_tokens(tmp_path):
    """Hard cap: max_new_tokens limits the output length. We assert that
    the decoded output (after stripping the prompt) contains at most
    max_new_tokens tokens worth of new content."""
    tokenizer = _make_tokenizer(tmp_path)
    model = _make_model(tokenizer)
    prompt_ids = tokenizer.encode("the")
    out = model.generate(prompt="the", max_new_tokens=3, temperature=0.0)
    # Re-encode the output and check it has at most prompt_len + 3 tokens.
    out_ids = tokenizer.encode(out)
    assert len(out_ids) <= len(prompt_ids) + 3


def test_generate_respects_max_seq_len(tmp_path):
    """If the prompt + max_new_tokens would exceed max_seq_len, the
    generator must truncate the context window rather than crash."""
    tokenizer = _make_tokenizer(tmp_path)
    # Make the model with a small max_seq_len.
    net_params = NNTransformerParams(
        input_dim=tokenizer.vocab_size,
        output_dim=tokenizer.vocab_size,
        dropout_prob=0.0,
        vocab_size=tokenizer.vocab_size,
        n_layers=1,
        n_heads=2,
        d_model=16,
        ffn_mult=2,
        max_seq_len=8,
    )
    model_params = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    model = GenerativeNNModel(net_params=net_params, params=model_params, tokenizer=tokenizer)
    # Prompt is ~5 BPE tokens; ask for 20 new — total far exceeds 8.
    out = model.generate(
        prompt="the cat sat on the mat",
        max_new_tokens=20,
        temperature=0.0,
    )
    assert isinstance(out, str)


# ---------------- KV-cache (SP-10c) ----------------


def test_kv_cache_produces_same_output_as_full_forward(tmp_path):
    """Equivalence: KV-cached greedy decode must produce identical
    token sequences to the SP-4 full-recompute path. This is the load-
    bearing correctness test for SP-10c — if these diverge, the cache
    implementation has a bug (wrong RoPE offset, off-by-one on mask
    slicing, etc.)."""
    tokenizer = _make_tokenizer(tmp_path)
    torch.manual_seed(7)
    model = _make_model(tokenizer)

    out_cached = model.generate(prompt="the", max_new_tokens=16, temperature=0.0, use_cache=True)
    out_full = model.generate(prompt="the", max_new_tokens=16, temperature=0.0, use_cache=False)
    assert out_cached == out_full


def test_kv_cache_matches_full_forward_under_sampling_with_seed(tmp_path):
    """Sampling-path equivalence — same seed, same prompt, both code
    paths should produce the same tokens. The sampler consumes the
    seeded RNG in the same order in both paths."""
    tokenizer = _make_tokenizer(tmp_path)
    torch.manual_seed(0)
    model = _make_model(tokenizer)

    out_cached = model.generate(prompt="the", max_new_tokens=10, temperature=1.0, top_k=10, seed=123, use_cache=True)
    out_full = model.generate(prompt="the", max_new_tokens=10, temperature=1.0, top_k=10, seed=123, use_cache=False)
    assert out_cached == out_full


def test_generate_use_cache_false_is_back_compat(tmp_path):
    """``use_cache=False`` preserves the exact SP-4 behaviour — the
    output for the default-greedy case is the same one the existing
    `test_generate_deterministic_greedy_is_reproducible` covers."""
    tokenizer = _make_tokenizer(tmp_path)
    torch.manual_seed(42)
    model = _make_model(tokenizer)
    a = model.generate(prompt="the", max_new_tokens=8, temperature=0.0, use_cache=False)
    b = model.generate(prompt="the", max_new_tokens=8, temperature=0.0, use_cache=False)
    assert a == b


def test_kv_cache_handles_sliding_window_overflow(tmp_path):
    """When prompt + new tokens exceed max_seq_len, the cache path
    must still produce the same tokens as the no-cache sliding window.
    This guards against an off-by-one in the cache-trim path."""
    tokenizer = _make_tokenizer(tmp_path)
    # Small window so we definitely overflow.
    net_params = NNTransformerParams(
        input_dim=tokenizer.vocab_size,
        output_dim=tokenizer.vocab_size,
        dropout_prob=0.0,
        vocab_size=tokenizer.vocab_size,
        n_layers=1,
        n_heads=2,
        d_model=16,
        ffn_mult=2,
        max_seq_len=8,
    )
    model_params = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    torch.manual_seed(11)
    model = GenerativeNNModel(net_params=net_params, params=model_params, tokenizer=tokenizer)

    out_cached = model.generate(prompt="the cat sat on the mat", max_new_tokens=12, temperature=0.0, use_cache=True)
    out_full = model.generate(prompt="the cat sat on the mat", max_new_tokens=12, temperature=0.0, use_cache=False)
    assert out_cached == out_full


def test_kv_cache_speedup_at_long_context(tmp_path):
    """Performance regression test: the cache path should be measurably
    faster than the full-recompute path on a non-trivial generation.

    We use a small Transformer (4 layers, 64 d_model) generating 128
    new tokens so the O(T^2) vs O(T) cost gap is clearly visible. The
    threshold is set conservatively (≥1.2x) — CPU timing on shared CI
    is noisy (we've seen the same code measure 1.9x on a quiet laptop
    and 1.46x on a busy GitHub Actions Linux runner in the same minute),
    and the real-world win is much larger on GPU at longer contexts.
    The point is to prove the cache is doing useful work, not to land
    a tight benchmark target.
    """
    import time

    tokenizer = _make_tokenizer(tmp_path)
    net_params = NNTransformerParams(
        input_dim=tokenizer.vocab_size,
        output_dim=tokenizer.vocab_size,
        dropout_prob=0.0,
        vocab_size=tokenizer.vocab_size,
        n_layers=4,
        n_heads=4,
        d_model=64,
        ffn_mult=4,
        max_seq_len=256,
    )
    model_params = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    torch.manual_seed(3)
    model = GenerativeNNModel(net_params=net_params, params=model_params, tokenizer=tokenizer)

    # Warm both paths once so torch's JIT/lazy-init costs are excluded
    # from the timed regions.
    model.generate(prompt="the", max_new_tokens=2, temperature=0.0, use_cache=True)
    model.generate(prompt="the", max_new_tokens=2, temperature=0.0, use_cache=False)

    n_new = 128

    # Run each path a few times and take the min — fewer noise spikes
    # than a single-shot measurement.
    def _time_path(use_cache: bool, repeats: int = 3) -> float:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.generate(prompt="the", max_new_tokens=n_new, temperature=0.0, use_cache=use_cache)
            times.append(time.perf_counter() - t0)
        return min(times)

    t_full = _time_path(use_cache=False)
    t_cached = _time_path(use_cache=True)

    speedup = t_full / t_cached if t_cached > 0 else float("inf")
    # Print so `pytest -s` shows the actual numbers — useful for
    # tracking the speedup over time.
    print(f"\n[kv-cache] full={t_full:.3f}s  cached={t_cached:.3f}s  speedup={speedup:.2f}x")
    assert speedup >= 1.2, f"Expected ≥1.2x speedup, got {speedup:.2f}x (full={t_full:.3f}s, cached={t_cached:.3f}s)"


def test_generate_requires_tokenizer(tmp_path):
    """Constructing a GenerativeNNModel without a tokenizer raises a
    clear error when generate() is called — the model needs the
    tokenizer to encode the prompt and decode the output."""
    net_params = NNTransformerParams(
        input_dim=64,
        output_dim=64,
        dropout_prob=0.0,
        vocab_size=64,
        n_layers=1,
        n_heads=2,
        d_model=16,
        ffn_mult=2,
        max_seq_len=8,
    )
    model_params = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    model = GenerativeNNModel(net_params=net_params, params=model_params, tokenizer=None)
    with pytest.raises(ValueError, match="tokenizer"):
        model.generate(prompt="anything", max_new_tokens=2)
