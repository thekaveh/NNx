"""Autoregressive generation utilities for NNx language models.

Public surface:
  * :class:`LogitsProcessor` — protocol for the chain-of-transformations
    that adjust raw logits before sampling (temperature, top-k, top-p,
    repetition penalty).
  * :func:`apply_chain` — run a list of LogitsProcessors over a logits
    tensor in order.
  * :func:`sample_next_token` — sample one next token given prepared
    logits + an optional torch.Generator for seeded sampling.

The chain design mirrors HF transformers' ``LogitsProcessorList`` — the
caller composes the processors in the order they want them applied, and
the model only needs to know how to plug the chain in. This keeps
:class:`GenerativeNNModel.generate` readable: the loop is "forward, run
chain, sample, append, repeat."
"""

from __future__ import annotations

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
    "LogitsProcessor",
    "TemperatureScaling",
    "TopKFilter",
    "TopPFilter",
    "RepetitionPenalty",
    "apply_chain",
    "sample_next_token",
]
