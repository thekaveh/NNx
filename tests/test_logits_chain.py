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
