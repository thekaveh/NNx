"""Ordered logits pipelines: decoding stages run exactly as declared (FEAT-037).

``nnx.generation.OrderedLogitsPipeline`` is the order-preserving companion to
``LogitsChain`` (whose builder sorts processors into NNx's canonical order):

  1. **The order oracle.** On logits ``[2, 1, 0]``, temperature 0.5 then
     top-p 0.8 keeps one finite logit; top-p 0.8 then temperature 0.5 keeps
     two. Each pipeline matches ``apply_chain`` over the same explicit list.
  2. **Immutable data.** ``append`` / ``prepend`` return new pipelines; two
     temperature-0.5 stages equal one 0.25; a zero-temperature (greedy)
     stage is terminal, so a stage after it is refused.
  3. **Versioned round trip.** ``state()`` writes ``{"version": 1,
     "stages": [...]}``; a custom stage serializes only through a codec
     registered for its tag.
  4. **Generation.** A tiny transformer with a locally trained BPE tokenizer
     generates with ``generate(logits_pipeline=...)``, seeded, on the cached
     and uncached paths — identical to the same explicit processor list.

Requires the ``lm`` extra (``tokenizers``) for step 4:
``pip install "thekaveh-nnx[lm]"``. Fully offline, CPU, deterministic.

Run:
    python examples/ordered_logits_pipeline.py

The bounded ``ordered_logits_pipeline_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from nnx.generation import (
    CustomStage,
    LogitsChain,
    LogitsStage,
    LogitsStageCodec,
    OrderedLogitsPipeline,
    TemperatureScaling,
    TopPFilter,
    apply_chain,
    register_logits_stage_codec,
    unregister_logits_stage_codec,
)

LOGITS = torch.tensor([[2.0, 1.0, 0.0]])


def _finite(logits: torch.Tensor) -> int:
    return int(torch.isfinite(logits).sum())


def _bias(amount: float):
    def stage(logits, token_history):
        return logits + amount

    stage.amount = amount  # type: ignore[attr-defined]
    return stage


def ordered_logits_pipeline_workflow() -> dict:
    """Run every step offline and return what it measured."""
    # 1. Order changes the result.
    cool_first = OrderedLogitsPipeline.of(LogitsStage.temperature(0.5), LogitsStage.top_p(0.8))
    nucleus_first = OrderedLogitsPipeline.of(LogitsStage.top_p(0.8), LogitsStage.temperature(0.5))
    kept = {
        "temperature_then_top_p": _finite(cool_first(LOGITS, [])),
        "top_p_then_temperature": _finite(nucleus_first(LOGITS, [])),
    }
    assert kept == {"temperature_then_top_p": 1, "top_p_then_temperature": 2}
    explicit = apply_chain(LOGITS, token_history=[], processors=[TemperatureScaling(0.5), TopPFilter(0.8)])
    assert torch.equal(cool_first(LOGITS, []), explicit)
    reversed_explicit = apply_chain(LOGITS, token_history=[], processors=[TopPFilter(0.8), TemperatureScaling(0.5)])
    assert torch.equal(nucleus_first(LOGITS, []), reversed_explicit)

    # 2. Immutable, duplicates kept, greedy is terminal.
    halved_twice = OrderedLogitsPipeline.of(LogitsStage.temperature(0.5)).append(LogitsStage.temperature(0.5))
    assert torch.allclose(halved_twice(LOGITS, []), OrderedLogitsPipeline.of(LogitsStage.temperature(0.25))(LOGITS, []))
    assert len(cool_first.prepend(LogitsStage.repetition_penalty(1.2))) == 3 and len(cool_first) == 2
    greedy = OrderedLogitsPipeline.of(LogitsStage.top_k(2), LogitsStage.temperature(0))
    refusals = 0
    for attempt in (
        lambda: greedy.append(LogitsStage.top_p(0.9)),
        lambda: cool_first.prepend(LogitsStage.temperature(0)),
    ):
        try:
            attempt()
        except ValueError:
            refusals += 1
    terminal_refused = refusals == 2
    assert terminal_refused and len(greedy) == 2

    # 3. Versioned round trip, a custom stage through its codec.
    biased = cool_first.append(CustomStage(_bias(1.0), tag="bias"))
    codec = LogitsStageCodec(tag="bias", encode=lambda fn: {"amount": fn.amount}, decode=lambda c: _bias(c["amount"]))
    register_logits_stage_codec(codec)
    try:
        state = json.loads(json.dumps(biased.state()))
        restored = OrderedLogitsPipeline.from_state(state)
    finally:
        unregister_logits_stage_codec("bias")
    assert torch.equal(restored(LOGITS, []), biased(LOGITS, []))

    # 4. Generation with a pipeline equals the explicit processor list.
    generated = _generate()
    return {"kept_finite": kept, "terminal_refused": terminal_refused, "state": state, "generated": generated}


def _generate() -> dict:
    from nnx import Devices, GenerativeNNModel, Losses, Nets, NNModelParams, NNTransformerParams
    from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe

    corpus = ["the cat sat on the mat", "the dog ran in the park", "hello world hello there"]
    tokenizer_model = train_bpe(files=None, texts=corpus, vocab_size=64, special_tokens=["<unk>", "<pad>"])
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer = NNTokenizerParams.of(tokenizer=tokenizer_model, path=str(Path(tmp) / "tok.json"))
        torch.manual_seed(0)
        model = GenerativeNNModel(
            net_params=NNTransformerParams(
                input_dim=tokenizer.vocab_size,
                output_dim=tokenizer.vocab_size,
                dropout_prob=0.0,
                vocab_size=tokenizer.vocab_size,
                n_layers=2,
                n_heads=2,
                d_model=16,
                ffn_mult=2,
                max_seq_len=32,
            ),
            params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
            tokenizer=tokenizer,
        )
        pipeline = OrderedLogitsPipeline.of(LogitsStage.temperature(0.7), LogitsStage.top_p(0.9))
        chain = LogitsChain(processors=[TemperatureScaling(0.7), TopPFilter(0.9)])
        out = {}
        for use_cache in (True, False):
            text = model.generate("the cat", max_new_tokens=6, seed=1, use_cache=use_cache, logits_pipeline=pipeline)
            same = model.generate("the cat", max_new_tokens=6, seed=1, use_cache=use_cache, logits_chain=chain)
            assert text == same
            out["cached" if use_cache else "uncached"] = text
    return out


def main() -> None:
    print(json.dumps(ordered_logits_pipeline_workflow(), indent=2))


if __name__ == "__main__":
    main()
