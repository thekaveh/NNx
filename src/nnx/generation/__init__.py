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
  * :class:`OrderedLogitsPipeline` (FEAT-037) — immutable stages run in
    exactly their declared order (``append`` / ``prepend`` return new
    pipelines); pass via ``generate(logits_pipeline=...)``.
  * :class:`LogitsStage` — a validated built-in stage spec
    (temperature / top-k / top-p / repetition penalty); a zero-temperature
    stage is terminal.
  * :class:`CustomStage` — a caller-owned callable at a declared position.
  * :class:`LogitsStageCodec` — serializes custom stages of one tag.
  * :data:`PIPELINE_VERSION` — the pipeline state version (1).
  * :func:`register_logits_stage_codec` /
    :func:`unregister_logits_stage_codec` /
    :func:`registered_logits_stage_codecs` — the in-process codec registry
    (data never names code to import).

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
from .pipeline import (
    PIPELINE_VERSION,
    CustomStage,
    LogitsStage,
    LogitsStageCodec,
    OrderedLogitsPipeline,
    register_logits_stage_codec,
    registered_logits_stage_codecs,
    unregister_logits_stage_codec,
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
    "OrderedLogitsPipeline",
    "LogitsStage",
    "CustomStage",
    "LogitsStageCodec",
    "PIPELINE_VERSION",
    "register_logits_stage_codec",
    "unregister_logits_stage_codec",
    "registered_logits_stage_codecs",
]
