"""Tests for LogitsChain + LogitsChainBuilder.

Pure-torch tests — no tokenizers dep. The integration test for
GenerativeNNModel.generate(logits_chain=...) lives in
test_generative_nn_model.py (which gates on the `lm` extra).
"""

from __future__ import annotations

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
