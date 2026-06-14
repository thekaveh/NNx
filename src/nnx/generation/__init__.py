"""Autoregressive generation utilities for NNx language models.

Public surface:
  * :class:`LogitsProcessor` — protocol for the chain-of-transformations
    that adjust raw logits before sampling.
  * :class:`TemperatureScaling` — divides logits by ``T``; placed last in
    the canonical chain order so the ``T=0`` greedy marker survives.
  * :class:`TopKFilter` — keep the top-``k`` logits, mask the rest.
  * :class:`TopPFilter` — keep the smallest set of logits whose softmax
    mass reaches ``p`` (nucleus sampling).
  * :class:`RepetitionPenalty` — divide / multiply logits for previously
    sampled tokens; sign-aware so it works for both positive and
    negative scores.
  * :func:`apply_chain` — run a list of LogitsProcessors over a logits
    tensor in order.
  * :class:`LogitsChain` — frozen container for an ordered list of
    processors with a single :meth:`__call__` entry point.
  * :class:`LogitsChainBuilder` — fluent builder for ``LogitsChain``;
    matches the :class:`nnx.NNTransformerParamsBuilder` convention.
  * :func:`sample_next_token` — sample one next token given prepared
    logits + an optional torch.Generator for seeded sampling.

The chain design mirrors HF transformers' ``LogitsProcessorList`` — the
caller composes the processors in the order they want them applied, and
the model only needs to know how to plug the chain in. This keeps
:class:`GenerativeNNModel.generate` readable: the loop is "forward, run
chain, sample, append, repeat."
"""

from __future__ import annotations

from .logits_chain import LogitsChain, LogitsChainBuilder
from .logits_processors import (
    LogitsProcessor,
    RepetitionPenalty,
    TemperatureScaling,
    TopKFilter,
    TopPFilter,
    apply_chain,
)
from .sampling import sample_next_token

__all__ = [
    "LogitsChain",
    "LogitsChainBuilder",
    "LogitsProcessor",
    "TemperatureScaling",
    "TopKFilter",
    "TopPFilter",
    "RepetitionPenalty",
    "apply_chain",
    "sample_next_token",
]
