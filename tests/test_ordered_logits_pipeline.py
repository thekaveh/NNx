"""FEAT-037: immutable, ordered logits pipelines beside the canonical LogitsChain."""

from __future__ import annotations

import json
import math

import pytest
import torch

from nnx.generation import (
    CustomStage,
    LogitsChain,
    LogitsStage,
    LogitsStageCodec,
    OrderedLogitsPipeline,
    RepetitionPenalty,
    TemperatureScaling,
    TopKFilter,
    TopPFilter,
    apply_chain,
    register_logits_stage_codec,
    registered_logits_stage_codecs,
    unregister_logits_stage_codec,
)

LOGITS = torch.tensor([[2.0, 1.0, 0.0]])


def _finite(t: torch.Tensor) -> int:
    return int(torch.isfinite(t).sum())


# ---------------- isolation ----------------


def test_append_and_prepend_return_new_pipelines_with_immutable_stages():
    base = OrderedLogitsPipeline.of(LogitsStage.top_p(0.8))
    longer = base.append(LogitsStage.temperature(0.5))
    front = base.prepend(LogitsStage.repetition_penalty(1.2))
    assert base.stages == (LogitsStage.top_p(0.8),)
    assert [s.kind for s in longer.stages] == ["top_p", "temperature"]
    assert [s.kind for s in front.stages] == ["repetition_penalty", "top_p"]
    assert isinstance(longer.stages, tuple)
    with pytest.raises(AttributeError):
        longer.stages[1].value = 3.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        longer.stages = ()  # type: ignore[misc]


def test_the_callers_list_and_processors_never_leak_in():
    stages = [LogitsStage.top_k(2)]
    processor = TemperatureScaling(temperature=0.5)
    pipeline = OrderedLogitsPipeline(stages).append(processor)
    stages.append(LogitsStage.top_k(1))  # mutating the caller's list
    processor.temperature = 9.0  # mutating the processor passed in
    assert pipeline.stages == (LogitsStage.top_k(2), LogitsStage.temperature(0.5))
    compiled = pipeline.processors()
    compiled[1].temperature = 7.0  # mutating a compiled processor
    assert pipeline.processors()[1].temperature == 0.5  # each run compiles fresh
    sibling = pipeline.append(LogitsStage.top_p(0.9))
    assert len(pipeline) == 2 and len(sibling) == 3


# ---------------- execution order ----------------


def test_declared_order_is_kept_and_reversal_differs():
    """The oracle: temperature-then-nucleus keeps 1 finite logit; reversed keeps 2."""
    cool_first = OrderedLogitsPipeline.of(LogitsStage.temperature(0.5), LogitsStage.top_p(0.8))
    nucleus_first = OrderedLogitsPipeline.of(LogitsStage.top_p(0.8), LogitsStage.temperature(0.5))
    assert _finite(cool_first(LOGITS, [])) == 1
    assert _finite(nucleus_first(LOGITS, [])) == 2
    for pipeline, processors in (
        (cool_first, [TemperatureScaling(0.5), TopPFilter(0.8)]),
        (nucleus_first, [TopPFilter(0.8), TemperatureScaling(0.5)]),
    ):
        assert torch.equal(pipeline(LOGITS, []), apply_chain(LOGITS, token_history=[], processors=processors))


def test_duplicates_and_custom_positions_run_where_declared():
    twice = OrderedLogitsPipeline.of(LogitsStage.temperature(0.5), LogitsStage.temperature(0.5))
    once = OrderedLogitsPipeline.of(LogitsStage.temperature(0.25))
    assert torch.allclose(twice(LOGITS, []), once(LOGITS, []))
    seen = []

    def spy(name):
        def stage(logits, history):
            seen.append(name)
            return logits + 1.0

        return stage

    pipeline = OrderedLogitsPipeline.of(
        spy("a"), LogitsStage.top_k(2), CustomStage(spy("b")), LogitsStage.temperature(1.0)
    )
    out = pipeline(LOGITS, [0])
    assert seen == ["a", "b"]
    expected = apply_chain(LOGITS, token_history=[0], processors=pipeline.processors())
    assert torch.equal(out, expected)


# ---------------- validation and placement ----------------


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("temperature", -0.1),
        ("temperature", math.inf),
        ("temperature", math.nan),
        ("top_p", 0.0),
        ("top_p", 1.5),
        ("top_k", 0),
        ("top_k", 2.5),
        ("top_k", True),
        ("repetition_penalty", 0.9),
        ("repetition_penalty", "1.2"),
        ("unknown", 1.0),
    ],
)
def test_values_are_validated_when_the_stage_is_made(kind, value):
    with pytest.raises(ValueError):
        LogitsStage(kind, value)


def test_a_zero_temperature_stage_is_terminal_and_nothing_is_reordered():
    greedy = OrderedLogitsPipeline.of(LogitsStage.top_k(2), LogitsStage.temperature(0))
    out = greedy(LOGITS, [])
    assert out[0, 0] == math.inf and _finite(out) == 0
    with pytest.raises(ValueError, match="must be last"):
        greedy.append(LogitsStage.top_p(0.9))
    with pytest.raises(ValueError, match="must be last"):
        OrderedLogitsPipeline.of(LogitsStage.top_p(0.9)).prepend(LogitsStage.temperature(0))
    with pytest.raises(ValueError, match="must be last"):
        OrderedLogitsPipeline.of(LogitsStage.temperature(0), CustomStage(lambda logits, history: logits))
    assert [s.kind for s in greedy.stages] == ["top_k", "temperature"]  # unchanged
    with pytest.raises(TypeError, match="pipeline stage"):
        OrderedLogitsPipeline.of(3)


# ---------------- generation ----------------


@pytest.fixture
def lm(tmp_path):
    pytest.importorskip("tokenizers")
    from nnx.nn.enum.devices import Devices
    from nnx.nn.enum.losses import Losses
    from nnx.nn.enum.nets import Nets
    from nnx.nn.generative_nn_model import GenerativeNNModel
    from nnx.nn.params.nn_model_params import NNModelParams
    from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe
    from nnx.nn.params.nn_transformer_params import NNTransformerParams

    corpus = ["the cat sat on the mat", "the dog ran in the park", "hello world hello there"]
    tk = train_bpe(files=None, texts=corpus, vocab_size=64, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
    tokenizer = NNTokenizerParams.of(tokenizer=tk, path=str(tmp_path / "tok.json"))
    torch.manual_seed(0)
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
    params = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    return GenerativeNNModel(net_params=net_params, params=params, tokenizer=tokenizer)


@pytest.mark.parametrize("use_cache", [True, False])
def test_generate_with_a_pipeline_equals_the_explicit_processor_list(lm, use_cache):
    pipeline = OrderedLogitsPipeline.of(
        LogitsStage.repetition_penalty(1.3), LogitsStage.temperature(0.7), LogitsStage.top_p(0.9)
    )
    explicit = LogitsChain(processors=[RepetitionPenalty(1.3), TemperatureScaling(0.7), TopPFilter(0.9)])
    streamed: list[int] = []
    a = lm.generate(
        "the cat", max_new_tokens=8, seed=3, use_cache=use_cache, logits_pipeline=pipeline, on_token=streamed.append
    )
    explicit_stream: list[int] = []
    b = lm.generate(
        "the cat", max_new_tokens=8, seed=3, use_cache=use_cache, logits_chain=explicit, on_token=explicit_stream.append
    )
    assert a == b and streamed == explicit_stream and len(streamed) == 8
    stopped = lm.generate(
        "the cat", max_new_tokens=8, seed=3, use_cache=use_cache, logits_pipeline=pipeline, stop=["the"]
    )
    stopped_explicit = lm.generate(
        "the cat", max_new_tokens=8, seed=3, use_cache=use_cache, logits_chain=explicit, stop=["the"]
    )
    assert stopped == stopped_explicit


def test_generate_compiles_the_pipeline_once_per_call(lm, monkeypatch):
    pipeline = OrderedLogitsPipeline.of(LogitsStage.temperature(0))
    compiled = []
    original = OrderedLogitsPipeline.processors

    def counting(self):
        compiled.append(1)
        return original(self)

    monkeypatch.setattr(OrderedLogitsPipeline, "processors", counting)
    lm.generate("the cat", max_new_tokens=5, logits_pipeline=pipeline)
    assert compiled == [1]


def test_a_chain_and_a_pipeline_together_raise_before_touching_modes(lm, monkeypatch):
    import nnx.nn.generative_nn_model as module

    monkeypatch.setattr(module, "_capture_training_modes", lambda net: pytest.fail("modes captured"))
    with pytest.raises(ValueError, match="not both"):
        lm.generate(
            "the cat",
            logits_chain=LogitsChain(processors=[]),
            logits_pipeline=OrderedLogitsPipeline.of(LogitsStage.temperature(1.0)),
        )


def test_a_raising_custom_stage_restores_every_submodule_mode(lm):
    lm.net.train()
    lm.net.blocks[0].eval()  # a mixed-mode net
    modes = {name: m.training for name, m in lm.net.named_modules()}

    def broken(logits, history):
        raise RuntimeError("stage failed")

    with pytest.raises(RuntimeError, match="stage failed"):
        lm.generate("the cat", max_new_tokens=3, logits_pipeline=OrderedLogitsPipeline.of(broken))
    assert {name: m.training for name, m in lm.net.named_modules()} == modes


def test_the_legacy_paths_are_unchanged(lm):
    chain = LogitsChain.builder().temperature(0.5).top_k(5).build()
    assert [type(p) for p in chain.processors] == [TopKFilter, TemperatureScaling]  # canonical sort kept
    assert lm.generate("the cat", max_new_tokens=4, temperature=0) == lm.generate(
        "the cat", max_new_tokens=4, logits_pipeline=OrderedLogitsPipeline.of(LogitsStage.temperature(0))
    )


# ---------------- serialization ----------------


def test_built_in_pipelines_round_trip_through_versioned_ordered_state():
    pipeline = OrderedLogitsPipeline.of(
        LogitsStage.top_p(0.8), LogitsStage.temperature(0.5), LogitsStage.top_p(0.8), LogitsStage.top_k(3)
    )
    state = pipeline.state()
    assert state == {
        "version": 1,
        "stages": [
            {"kind": "top_p", "value": 0.8},
            {"kind": "temperature", "value": 0.5},
            {"kind": "top_p", "value": 0.8},
            {"kind": "top_k", "value": 3},
        ],
    }
    restored = OrderedLogitsPipeline.from_state(json.loads(json.dumps(state)))
    assert restored == pipeline and torch.equal(restored(LOGITS, []), pipeline(LOGITS, []))
    with pytest.raises(ValueError, match="version"):
        OrderedLogitsPipeline.from_state({"version": 2, "stages": []})
    with pytest.raises(ValueError, match="unknown keys"):
        OrderedLogitsPipeline.from_state({"version": 1, "stages": [{"kind": "top_k", "value": 3, "module": "os"}]})


def test_custom_stages_are_runtime_only_unless_a_codec_is_registered():
    def bias(logits, history):
        return logits + 1.0

    pipeline = OrderedLogitsPipeline.of(LogitsStage.top_k(2), CustomStage(bias, tag="bias"))
    with pytest.raises(ValueError, match="runtime-only"):
        pipeline.state()
    with pytest.raises(ValueError, match="runtime-only"):
        OrderedLogitsPipeline.of(bias).state()  # an untagged callable

    def make_bias(config):
        amount = float(config["amount"])

        def stage(logits, history):
            return logits + amount

        return stage

    codec = LogitsStageCodec(tag="bias", encode=lambda fn: {"amount": 1.0}, decode=make_bias)
    register_logits_stage_codec(codec)
    try:
        state = pipeline.state()
        assert state["stages"][1] == {"kind": "custom", "tag": "bias", "config": {"amount": 1.0}}
        restored = OrderedLogitsPipeline.from_state(state)
        assert torch.equal(restored(LOGITS, []), pipeline(LOGITS, []))
        assert "bias" in registered_logits_stage_codecs()
    finally:
        unregister_logits_stage_codec("bias")
    # Data never names code to import: an unregistered tag cannot be rebuilt.
    with pytest.raises(ValueError, match="no LogitsStageCodec"):
        OrderedLogitsPipeline.from_state(
            {"version": 1, "stages": [{"kind": "custom", "tag": "os.system", "config": {"cmd": "echo"}}]}
        )


def test_new_types_export_from_generation_and_nnx():
    import nnx
    import nnx.generation as generation

    for name in ("OrderedLogitsPipeline", "LogitsStage", "CustomStage", "LogitsStageCodec"):
        assert getattr(nnx, name) is getattr(generation, name)
        assert name in nnx.__all__ and name in generation.__all__


# ---------------- review regressions ----------------


def test_a_subclass_of_a_built_in_processor_keeps_its_own_behaviour():
    class Shifted(TemperatureScaling):
        def __call__(self, logits, token_history):
            return logits + 100.0

    (stage,) = OrderedLogitsPipeline.of(Shifted(1.0)).stages
    assert isinstance(stage, CustomStage)
    assert torch.equal(OrderedLogitsPipeline.of(Shifted(1.0))(LOGITS, []), LOGITS + 100.0)


def test_custom_stages_compare_by_the_callable_they_hold():
    def keep(logits, history):
        return logits

    def zero(logits, history):
        return logits * 0

    assert CustomStage(keep) == CustomStage(keep) and CustomStage(keep) != CustomStage(zero)
    assert OrderedLogitsPipeline.of(keep) != OrderedLogitsPipeline.of(zero)
    assert len({OrderedLogitsPipeline.of(keep), OrderedLogitsPipeline.of(zero), OrderedLogitsPipeline.of(keep)}) == 2


def test_a_nested_pipeline_is_flattened_so_the_terminal_rule_sees_it():
    inner = OrderedLogitsPipeline.of(LogitsStage.top_k(2), LogitsStage.temperature(0))
    with pytest.raises(ValueError, match="must be last"):
        OrderedLogitsPipeline.of(inner, LogitsStage.top_p(0.5))
    assert OrderedLogitsPipeline.of(LogitsStage.top_p(0.5), inner).stages[1:] == inner.stages


@pytest.mark.parametrize(
    "state",
    [
        {"version": True, "stages": []},
        {"version": 1.0, "stages": []},
        {"version": 1, "stages": [{"kind": "custom", "tag": "bias", "config": {}, "evil": "x"}]},
        {"version": 1, "stages": [{"kind": "custom", "tag": "bias", "config": "rm -rf"}]},
    ],
)
def test_from_state_refuses_loose_or_odd_data(state):
    register_logits_stage_codec(LogitsStageCodec(tag="bias", encode=lambda fn: {}, decode=lambda c: lambda l, h: l))
    try:
        with pytest.raises(ValueError):
            OrderedLogitsPipeline.from_state(state)
    finally:
        unregister_logits_stage_codec("bias")


def test_a_huge_value_is_a_value_error():
    with pytest.raises(ValueError, match="too large"):
        LogitsStage.temperature(10**400)


def test_wrong_decoding_arguments_raise_type_errors_before_the_model_is_used(lm):
    with pytest.raises(TypeError, match="OrderedLogitsPipeline"):
        lm.generate("the cat", logits_pipeline=[TemperatureScaling(0.5)])
    with pytest.raises(TypeError, match="pass it as logits_pipeline="):
        lm.generate("the cat", logits_chain=OrderedLogitsPipeline.of(LogitsStage.temperature(0.5)))


def test_a_pipeline_never_enters_the_run_identity_or_checkpoint(lm):
    before = lm.params.state(), lm.net_params.state()
    lm.generate("the cat", max_new_tokens=2, logits_pipeline=OrderedLogitsPipeline.of(LogitsStage.temperature(0)))
    assert (lm.params.state(), lm.net_params.state()) == before
