"""Tests for LogitsChain + LogitsChainBuilder.

Pure-torch tests — no tokenizers dep. The integration test for
GenerativeNNModel.generate(logits_chain=...) lives in
test_generative_nn_model.py (which gates on the `lm` extra).
"""

from __future__ import annotations

import pytest
import torch

from nnx.generation.logits_chain import LogitsChain
from nnx.generation.logits_processors import (
    RepetitionPenalty,
    TemperatureScaling,
    TopKFilter,
    TopPFilter,
)


def test_builder_empty_chain_is_no_op():
    """Calling .build() with no modifiers returns a LogitsChain whose
    .apply() is a pass-through. Establishes the "empty Builder is
    safe" contract."""
    chain = LogitsChain.builder().build()
    assert chain.processors == []
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    out = chain.apply(logits, token_history=[])
    assert torch.equal(out, logits)


def test_builder_temperature_only():
    """Single temperature processor; verify the chain produces the
    same effect as TemperatureScaling directly."""
    chain = LogitsChain.builder().temperature(0.5).build()
    assert len(chain.processors) == 1
    assert isinstance(chain.processors[0], TemperatureScaling)
    assert chain.processors[0].temperature == 0.5


def test_builder_enforces_canonical_order():
    """The canonical HF order is `RepetitionPenalty → TopKFilter →
    TopPFilter → TemperatureScaling`. The Builder enforces this
    regardless of the call order — the user can chain the methods in
    any sequence; .build() sorts the processors before returning."""
    chain = (
        LogitsChain.builder()
        .temperature(0.7)  # called first
        .repetition_penalty(1.1)  # then this
        .top_p(0.9)
        .top_k(50)
        .build()
    )
    types = [type(p) for p in chain.processors]
    assert types == [RepetitionPenalty, TopKFilter, TopPFilter, TemperatureScaling]


def test_builder_custom_processor_appended_after_canonical():
    """A custom processor goes AFTER the canonical group, in the
    order added. Two custom processors append in chain order."""

    class LogitBias:
        """Toy custom processor — adds a constant to all logits."""

        def __init__(self, bias: float):
            self.bias = bias

        def __call__(self, logits: torch.Tensor, token_history: list[int]) -> torch.Tensor:
            return logits + self.bias

    bias1 = LogitBias(0.5)
    bias2 = LogitBias(-0.1)

    chain = LogitsChain.builder().temperature(0.8).custom(bias1).top_k(50).custom(bias2).build()
    # Canonical processors first (top-k, then temperature), then customs in chain order.
    assert chain.processors[0].__class__.__name__ == "TopKFilter"
    assert chain.processors[1].__class__.__name__ == "TemperatureScaling"
    assert chain.processors[2] is bias1
    assert chain.processors[3] is bias2


def test_builder_multiple_calls_to_same_method_overwrite():
    """Calling `.temperature(0.5).temperature(0.8)` keeps only the
    last value — the standard processors are stored by class in a
    dict, so the last write wins. Documents the contract."""
    chain = LogitsChain.builder().temperature(0.5).temperature(0.8).build()
    assert len(chain.processors) == 1
    assert chain.processors[0].temperature == 0.8


def test_chain_apply_matches_direct_apply_chain_call():
    """LogitsChain.apply() is semantically equivalent to calling
    apply_chain(processors=chain.processors) directly."""
    from nnx.generation.logits_processors import apply_chain

    chain = LogitsChain.builder().repetition_penalty(1.1).top_k(2).temperature(0.8).build()
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    history = [0, 2]
    via_chain = chain.apply(logits, token_history=history)
    via_direct = apply_chain(logits, token_history=history, processors=chain.processors)
    assert torch.equal(via_chain, via_direct)


def test_chain_apply_passes_token_history_to_custom_processor():
    """The apply path must forward `token_history` to every processor
    in the chain — including custom ones. Pre-existing tests only
    checked storage order, not call-time argument propagation, so a
    regression that swallowed history or replaced it with [] would
    have passed all the existing tests."""

    class HistoryRecorder:
        """Custom processor that records every (history) it sees."""

        def __init__(self):
            self.seen_histories: list[list[int]] = []

        def __call__(self, logits: torch.Tensor, token_history: list[int]) -> torch.Tensor:
            self.seen_histories.append(list(token_history))
            return logits

    rec = HistoryRecorder()
    chain = LogitsChain.builder().custom(rec).build()
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    history = [0, 1, 2, 5]
    chain.apply(logits, token_history=history)
    assert rec.seen_histories == [history], (
        f"Expected custom processor to receive token_history={history!r}, got {rec.seen_histories!r}"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_decoding_settings_are_refused(value):
    """NaN compares False to every bound: without an explicit check a NaN
    penalty made the most-penalized token win every draw."""
    from nnx.generation.logits_processors import RepetitionPenalty, TemperatureScaling

    with pytest.raises(ValueError, match="finite"):
        TemperatureScaling(value)
    with pytest.raises(ValueError, match="finite"):
        RepetitionPenalty(value)


def test_greedy_refuses_custom_processors_it_would_corrupt():
    """temperature(0.0) turns logits into ±inf argmax markers: a later
    token ban would leave all -inf (an arbitrary token) and a bias would be
    ignored, so build() refuses the pair."""
    from nnx.generation.logits_chain import LogitsChainBuilder

    def ban(logits, history):
        out = logits.clone()
        out[..., 2] = -float("inf")
        return out

    with pytest.raises(ValueError, match="OrderedLogitsPipeline"):
        LogitsChainBuilder().temperature(0.0).custom(ban).build()
    LogitsChainBuilder().temperature(0.7).custom(ban).build()  # sampling keeps custom stages


def _count_tensor_bools(monkeypatch) -> list[int]:
    """Count tensor-to-bool conversions (each one a device-to-host sync on CUDA)."""
    count = [0]
    original = torch.Tensor.__bool__

    def counting(self):
        count[0] += 1
        return original(self)

    monkeypatch.setattr(torch.Tensor, "__bool__", counting)
    return count


def test_temperature_one_returns_the_logits_without_a_sync(monkeypatch):
    """Default generate() runs TemperatureScaling(1.0): it is the identity,
    with no upcast and no tensor-to-bool conversion (#578), through all
    three construction routes."""
    from nnx.generation.pipeline import LogitsStage, OrderedLogitsPipeline

    logits = torch.randn(1, 64)
    count = _count_tensor_bools(monkeypatch)
    assert TemperatureScaling(1.0)(logits, []) is logits
    assert TemperatureScaling(1)(logits, []) is logits
    assert LogitsChain.builder().temperature(1.0).build().apply(logits, token_history=[]) is logits
    assert OrderedLogitsPipeline.of(LogitsStage.temperature(1.0))(logits, []) is logits
    assert count[0] == 0


@pytest.mark.parametrize("temperature", [1.0 + 1e-7, 2.0, 1e30])
def test_temperature_above_one_only_divides(monkeypatch, temperature):
    """T > 1 cannot overflow, so the result is exactly logits / T, with no
    tensor-to-bool conversion, extreme inputs included (#578)."""
    from nnx.generation.pipeline import LogitsStage, OrderedLogitsPipeline

    big = torch.finfo(torch.float32).max
    tiny = torch.finfo(torch.float32).smallest_normal / 4  # a denormal
    logits = torch.tensor([[big, -big, tiny, -tiny, float("inf"), float("-inf"), float("nan"), 0.0, 3.5]])
    expected = logits / temperature
    count = _count_tensor_bools(monkeypatch)
    outs = (
        TemperatureScaling(temperature)(logits, []),
        LogitsChain.builder().temperature(temperature).build().apply(logits, token_history=[]),
        OrderedLogitsPipeline.of(LogitsStage.temperature(temperature))(logits, []),
    )
    assert count[0] == 0
    for out in outs:
        torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_temperature_at_or_above_one_keeps_half_precision(dtype, temperature):
    """At T >= 1 half-precision logits keep their dtype, as before the
    small-temperature guard (#578); only 0 < T < 1 upcasts to float32."""
    logits = torch.randn(1, 16).to(dtype)
    out = TemperatureScaling(temperature)(logits, [])
    assert out.dtype == dtype
    torch.testing.assert_close(out, logits / temperature, rtol=0, atol=0)
    assert TemperatureScaling(0.5)(logits, []).dtype == torch.float32


def test_temperature_below_one_keeps_the_overflow_guard(monkeypatch):
    """0 < T < 1 still runs the guard: an overflow marks the argmax."""
    logits = torch.tensor([[1.0e30, 3.0e38, -1.0]])
    count = _count_tensor_bools(monkeypatch)
    out = TemperatureScaling(1e-30)(logits, [])
    assert count[0] > 0
    assert out.argmax().item() == 1
    assert out.tolist() == [[float("-inf"), float("inf"), float("-inf")]]
