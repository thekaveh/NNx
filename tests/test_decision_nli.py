"""FEAT-011: the local label-conditioned NLI baseline provider.

Everything runs offline against a tiny stub: a word-level HuggingFace-style
tokenizer and an NLI "model" whose entailment logit is ``log(1 + overlap)``
— the number of distinct hypothesis words found in the premise — with
contradiction and neutral at ``0``.
"""

from __future__ import annotations

import inspect
import math
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nnx.decisions import (
    Boolean,
    BooleanResult,
    Choice,
    ChoiceResult,
    InvalidDecisionRequest,
    NLIProvider,
    ProviderFailure,
    Score,
    UnsupportedCapability,
)

CLS, SEP, PAD = 1, 2, 0


class WordTokenizer:
    """``tokenizer(premises, hypotheses, truncation=..., max_length=...,
    padding=..., return_tensors=...)`` → ``[CLS] premise [SEP] hypothesis
    [SEP]`` with token types 0 / 1; ``"only_first"`` cuts the premise."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.calls: list[dict] = []

    def ids(self, text: str) -> list[int]:
        words = [word.strip(".,?!").lower() for word in text.split()]
        return [self.vocab.setdefault(word, len(self.vocab) + 3) for word in words if word]

    def __call__(self, premises, hypotheses, *, truncation, max_length=None, padding=False, return_tensors=None):
        self.calls.append(
            {
                "premises": list(premises),
                "hypotheses": list(hypotheses),
                "truncation": truncation,
                "tensors": return_tensors,
            }
        )
        rows, types = [], []
        for premise, hypothesis in zip(premises, hypotheses, strict=True):
            p, h = self.ids(premise), self.ids(hypothesis)
            if truncation == "only_first" and max_length is not None:
                p = p[: max(max_length - len(h) - 3, 0)]
            rows.append([CLS, *p, SEP, *h, SEP])
            types.append([0] * (len(p) + 2) + [1] * (len(h) + 1))
        if return_tensors != "pt":
            return {"input_ids": rows, "token_type_ids": types}
        width = max(len(row) for row in rows)
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        pad = [row + [PAD] * (width - len(row)) for row in rows]
        types = [row + [0] * (width - len(row)) for row in types]
        return {
            "input_ids": torch.tensor(pad),
            "token_type_ids": torch.tensor(types),
            "attention_mask": torch.tensor(mask),
        }


class OverlapNLI(nn.Module):
    """Entailment logit ``log(1 + overlap)``; contradiction and neutral 0."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_labels=3, label2id={"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2})
        self.scale = nn.Parameter(torch.ones(()))
        self.head = nn.Linear(1, 1)  # a submodule, for mixed training modes
        self.seen: list[tuple[str, bool, int]] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("the provider must never call from_pretrained")

    def forward(self, input_ids, token_type_ids, attention_mask):
        self.seen.append((input_ids.device.type, self.training, int(input_ids.shape[0])))
        rows = []
        for ids, kinds, mask in zip(input_ids, token_type_ids, attention_mask, strict=True):
            live = mask.bool()
            premise = set(ids[live & (kinds == 0)].tolist()) - {CLS, SEP}
            hypothesis = set(ids[live & (kinds == 1)].tolist()) - {SEP}
            rows.append([0.0, 0.0, math.log1p(len(premise & hypothesis))])
        return SimpleNamespace(logits=torch.tensor(rows) * self.scale)


def _ids_and_p(result) -> tuple[tuple[str, ...], list[float]]:
    return tuple(i for i, _ in result.distribution), [p for _, p in result.distribution]


def _provider(**overrides) -> NLIProvider:
    fields = {"entailment_id": "entailment", "contradiction_id": "contradiction", "hypothesis_template": "{}"}
    fields.update(overrides)
    model = fields.pop("model", OverlapNLI())
    tokenizer = fields.pop("tokenizer", WordTokenizer())
    return NLIProvider(model, tokenizer, **fields)


SPORT = ("t-sport", "late goal")
ECON = ("t-econ", "interest rates")
TEXT = "the match ended with a late goal"


# --- AC1: descriptions arrive at inference -----------------------------------------------------------------


def test_candidates_supplied_at_inference_move_with_their_ids():
    provider = _provider()  # built with no candidates at all
    (forward,) = provider.decide(Choice("Topic?", (SPORT, ECON)), [TEXT])
    (backward,) = provider.decide(Choice("Topic?", (ECON, SPORT)), [TEXT])
    assert isinstance(forward, ChoiceResult)
    ids, p = _ids_and_p(forward)
    assert ids == ("t-sport", "t-econ") and p == pytest.approx([0.75, 0.25])
    ids, p = _ids_and_p(backward)
    assert ids == ("t-econ", "t-sport") and p == pytest.approx([0.25, 0.75])  # the ids move with their p
    assert forward.probabilities() == pytest.approx(backward.probabilities())


def test_a_paraphrased_description_is_rescored_under_the_same_id():
    provider = _provider()
    tokenizer = provider.tokenizer
    original = Choice("Topic?", (SPORT, ECON))
    paraphrase = Choice("Topic?", (("t-sport", "goal scored late"), ECON))
    (a,) = provider.decide(original, [TEXT])
    (b,) = provider.decide(paraphrase, [TEXT])
    assert original.digest() != paraphrase.digest() and a.question_digest != b.question_digest
    assert tokenizer.calls[-1]["hypotheses"] == ["goal scored late", "interest rates"]
    assert b.top == "t-sport" and b.probabilities()["t-sport"] == pytest.approx(0.75)


# --- AC2: explicit, validated settings -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"entailment_id": 2, "contradiction_id": 2}, "must differ"),
        ({"entailment_id": 3}, "out of range for a model with 3 classes"),
        ({"contradiction_id": -1}, "out of range"),
        ({"entailment_id": "entails"}, "unknown entailment_id label 'entails'"),
        ({"entailment_id": 2.0}, "class id"),
        ({"hypothesis_template": "no placeholder"}, "exactly one"),
        ({"hypothesis_template": "{} and {}"}, "exactly one"),
        ({"boolean_template": "{prompt}"}, "exactly one"),
        ({"hypothesis_template": "About {!r}."}, "exactly one bare"),
        ({"hypothesis_template": "About {:.3}."}, "exactly one bare"),
        ({"hypothesis_template": "About {:>40}."}, "exactly one bare"),
        ({"hypothesis_template": "About {:{}}."}, "exactly one bare"),
        ({"boolean_template": "{:d}"}, "exactly one bare"),
        ({"boolean_template": "{!x}"}, "exactly one bare"),
        ({"hypothesis_template": "About {."}, "not a valid format string"),
        ({"choice_scoring": "renormalized_probability"}, "choice_scoring"),
        ({"boolean_scoring": "joint"}, "boolean_scoring"),
        ({"truncation": "longest_first"}, "truncation"),
        ({"max_length": 0}, "max_length"),
        ({"pair_batch_size": 0}, "pair_batch_size"),
    ],
)
def test_settings_are_explicit_and_validated(overrides, match):
    with pytest.raises(InvalidDecisionRequest, match=match):
        _provider(**overrides)


def test_label_names_resolve_through_the_models_config():
    provider = _provider(entailment_id="Entailment", contradiction_id="CONTRADICTION")
    assert (provider.entailment_id, provider.contradiction_id) == (2, 0)
    unnamed = OverlapNLI()
    del unnamed.config
    with pytest.raises(InvalidDecisionRequest, match="unknown entailment_id label"):
        _provider(model=unnamed)
    assert _provider(model=unnamed, entailment_id=2, contradiction_id=0).entailment_id == 2


def test_boolean_questions_are_never_normalised_together():
    provider = _provider(boolean_template="{}")
    question = Boolean("late goal match")
    texts = ["the match ended with a late goal", "rates rose", "a goal"]
    batch = provider.decide(question, texts)
    singles = [provider.decide(question, [text])[0] for text in texts]
    assert all(isinstance(result, BooleanResult) for result in batch)
    assert [r.p_true for r in batch] == pytest.approx([r.p_true for r in singles])
    assert [r.p_true for r in batch] == pytest.approx([0.8, 0.5, 2 / 3])  # each its own pair: 4/5, 1/2, 2/3
    assert sum(r.p_true for r in batch) != pytest.approx(1.0)


# --- AC3: nothing is downloaded --------------------------------------------------------------------------


def test_importing_and_constructing_download_nothing(tmp_path):
    # Network and every from_pretrained entry point are refused BEFORE nnx is
    # imported, so importing, constructing and deciding are all covered.
    script = textwrap.dedent(
        f"""
        import socket, sys
        calls = []
        def refuse(*args, **kwargs):
            calls.append(args)
            raise AssertionError("network access or from_pretrained")
        socket.socket.connect = refuse
        socket.create_connection = refuse
        sys.path.insert(0, {str(tmp_path)!r})
        try:
            import transformers
            for name in ("PreTrainedModel", "PreTrainedTokenizerBase", "AutoModel", "AutoTokenizer",
                         "AutoModelForSequenceClassification", "AutoConfig"):
                if hasattr(transformers, name):
                    setattr(getattr(transformers, name), "from_pretrained", classmethod(refuse))
        except ImportError:
            pass
        try:
            import huggingface_hub
            huggingface_hub.hf_hub_download = refuse
            huggingface_hub.snapshot_download = refuse
        except ImportError:
            pass
        import nnx
        from nnx.decisions import Boolean, NLIProvider
        from stub import OverlapNLI, WordTokenizer
        provider = NLIProvider(OverlapNLI(), WordTokenizer(), entailment_id=2, contradiction_id=0)
        result = provider.decide(Boolean("late goal"), ["a late goal"])
        assert calls == [], calls
        print(result[0].p_true)
        """
    )
    stub = inspect.getsource(sys.modules[__name__])
    (tmp_path / "stub.py").write_text(stub)
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False)
    assert run.returncode == 0, run.stderr
    assert float(run.stdout.strip().splitlines()[-1]) == pytest.approx(0.75)


def test_the_provider_module_imports_no_nli_library():
    import ast

    import nnx.decisions.nli as module

    tree = ast.parse(inspect.getsource(module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__",
        "collections",
        "dataclasses",
        "math",
        "numbers",
        "string",
        "typing",
        "numpy",
        "torch",
    }


def test_embed_texts_and_the_faiss_export_keep_their_signatures():
    from nnx.embeddings import embed_texts, export_to_faiss

    # Pinned as they were before FEAT-011: the NLI provider adds nothing to either.
    assert str(inspect.signature(embed_texts)) == (
        "(backbone: 'Any', texts: 'list[str]', *, batch_size: 'int' = 64, "
        "device: 'Optional[Union[str, torch.device]]' = None, normalize: 'bool' = True) -> 'torch.Tensor'"
    )
    assert list(inspect.signature(export_to_faiss).parameters) == [
        "backbone",
        "corpus",
        "out_path",
        "batch_size",
        "index_type",
        "normalize",
        "device",
    ]


# --- AC4: fixtures -------------------------------------------------------------------------------------


def test_batched_requests_equal_one_input_at_a_time_and_any_pair_batch_size():
    question = Choice("Topic?", (SPORT, ECON, ("t-weather", "heavy rain")))
    texts = ["late goal in the rain", "interest rates rose", "heavy rain and a goal"]
    reference = [_provider().decide(question, [text])[0] for text in texts]
    for pair_batch_size in (1, 2, 4, 7, 64):
        batch = _provider(pair_batch_size=pair_batch_size).decide(question, texts)
        for got, want in zip(batch, reference, strict=True):
            assert _ids_and_p(got)[0] == _ids_and_p(want)[0]
            assert _ids_and_p(got)[1] == pytest.approx(_ids_and_p(want)[1])


def test_pair_batch_tails_are_scored():
    provider = _provider(pair_batch_size=4)
    question = Choice("Topic?", (SPORT, ECON, ("t-weather", "heavy rain")))
    results = provider.decide(question, ["late goal", "rates", "rain"])  # 9 pairs: chunks 4, 4, 1
    assert [rows for _, _, rows in provider.model.seen] == [4, 4, 1]
    assert len(results) == 3 and provider.model_calls == 1
    pairs = [call for call in provider.tokenizer.calls if call["tensors"] == "pt"]
    assert [len(call["premises"]) for call in pairs] == [4, 4, 1]


def test_long_inputs_are_reported_or_refused():
    question = Choice("Topic?", (SPORT, ECON))
    long_text = TEXT + " " + " ".join(f"word{i}" for i in range(40))
    reporting = _provider(max_length=16)
    short, long = reporting.decide(question, [TEXT, long_text])
    assert short.raw["truncated"] == [False, False] and long.raw["truncated"] == [True, True]
    refusing = _provider(max_length=16, truncation="error")
    with pytest.raises(InvalidDecisionRequest, match="exceed max_length=16"):
        refusing.decide(question, [TEXT, long_text])
    assert refusing.model_calls == 0 and refusing.model.seen == []


def test_a_hypothesis_that_cannot_fit_is_refused_before_the_model():
    long_description = ("t-long", "one two three four five six")
    question = Choice("Topic?", (SPORT, long_description))
    quiet = _provider(max_length=6)  # this stub cuts the premise to nothing and still overflows
    with pytest.raises(InvalidDecisionRequest, match="does not fit max_length=6"):
        quiet.decide(question, [TEXT])
    assert quiet.model_calls == 0 and quiet.model.seen == []

    class Strict(WordTokenizer):  # like a HuggingFace fast tokenizer: refuses instead
        def __call__(self, premises, hypotheses, *, truncation, max_length=None, **kwargs):
            if truncation == "only_first" and any(len(self.ids(h)) + 3 > max_length for h in hypotheses):
                raise ValueError("Truncation error: Sequence to truncate too short to respect the provided max_length")
            return super().__call__(premises, hypotheses, truncation=truncation, max_length=max_length, **kwargs)

    strict = _provider(max_length=6, tokenizer=Strict())
    with pytest.raises(InvalidDecisionRequest, match="too short") as caught:
        strict.decide(question, [TEXT])
    assert isinstance(caught.value.__cause__, ValueError) and strict.model_calls == 0


def test_malformed_model_outputs_are_provider_failures():
    class Scalar(OverlapNLI):
        def forward(self, input_ids, token_type_ids, attention_mask):
            return torch.tensor(1.0)

    class Ragged(OverlapNLI):
        def forward(self, input_ids, token_type_ids, attention_mask):
            return torch.zeros(int(input_ids.shape[0]), 3 + int(input_ids.shape[0]))  # width varies per chunk

    with pytest.raises(ProviderFailure, match="inconsistent shapes"):
        _provider(model=Scalar()).decide(Boolean("late goal"), [TEXT])
    with pytest.raises(ProviderFailure, match="inconsistent shapes"):
        _provider(model=Ragged(), pair_batch_size=2).decide(Choice("Topic?", (SPORT, ECON, ("t-x", "x"))), [TEXT])


def test_unknown_entailment_labels_never_reach_the_model():
    model = OverlapNLI()
    model.config.label2id = {"yes": 0, "no": 1}
    with pytest.raises(InvalidDecisionRequest, match="unknown entailment_id label 'entailment'"):
        _provider(model=model)
    assert model.seen == []


# --- AC6: the provider contract ----------------------------------------------------------------------------


def _mixed_modes(model: nn.Module) -> list[bool]:
    model.train()
    model.head.eval()
    return [part.training for part in model.modules()]


def test_a_batch_reaches_the_tokenizer_and_model_in_candidate_order_and_modes_survive():
    provider = _provider(hypothesis_template="This text is about {}.")
    before = _mixed_modes(provider.model)
    question = Choice("Topic?", (ECON, SPORT))
    provider.decide(question, ["a late goal", "rates"])
    (call,) = [c for c in provider.tokenizer.calls if c["tensors"] == "pt"]
    assert call["premises"] == ["a late goal", "a late goal", "rates", "rates"]
    assert call["hypotheses"] == ["This text is about interest rates.", "This text is about late goal."] * 2
    assert call["truncation"] == "only_first"
    assert provider.model.seen == [("cpu", False, 4)]  # eval mode, on the model's device
    assert [part.training for part in provider.model.modules()] == before
    assert next(provider.model.parameters()).device.type == "cpu"


def test_modes_survive_a_truncation_refusal_and_a_model_failure():
    refusing = _provider(max_length=8, truncation="error")
    before = _mixed_modes(refusing.model)
    with pytest.raises(InvalidDecisionRequest):
        refusing.decide(Choice("Topic?", (SPORT, ECON)), [TEXT])
    assert [part.training for part in refusing.model.modules()] == before

    class Broken(OverlapNLI):
        def forward(self, input_ids, token_type_ids, attention_mask):
            raise RuntimeError("kernel failure")

    failing = _provider(model=Broken())
    before = _mixed_modes(failing.model)
    with pytest.raises(ProviderFailure, match="kernel failure") as caught:
        failing.decide(Boolean("late goal"), [TEXT])
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert [part.training for part in failing.model.modules()] == before

    class FailingTensors(WordTokenizer):  # fails while building the truncated model inputs
        def __call__(self, premises, hypotheses, *, return_tensors=None, **kwargs):
            if return_tensors == "pt":
                raise RuntimeError("tokenizer failure while truncating")
            return super().__call__(premises, hypotheses, return_tensors=return_tensors, **kwargs)

    truncating = _provider(max_length=8, tokenizer=FailingTensors())
    before = _mixed_modes(truncating.model)
    with pytest.raises(ProviderFailure, match="while truncating"):
        truncating.decide(Choice("Topic?", (SPORT, ECON)), [TEXT])  # an over-long pair, cut under only_first
    assert [part.training for part in truncating.model.modules()] == before
    assert next(truncating.model.parameters()).device.type == "cpu" and truncating.model.seen == []


# --- AC7: records and unsupported requests ----------------------------------------------------------------


def test_results_keep_the_template_revision_and_scoring_and_are_never_calibrated():
    provider = _provider(hypothesis_template="It is about {}.", revision="stub-rev-1")
    (choice,) = provider.decide(Choice("Topic?", (SPORT, ECON)), [TEXT])
    (boolean,) = provider.decide(Boolean("late goal"), [TEXT])
    for result in (choice, boolean):
        raw = result.raw
        assert raw["method"] == "nli" and raw["revision"] == "stub-rev-1" and raw["calibrated"] is False
        assert raw["hypothesis_template"] == "It is about {}." and len(raw["logits"]) == len(raw["truncated"])
    assert choice.raw["choice_scoring"] == "entailment_softmax"
    assert boolean.raw["boolean_scoring"] == "entailment_vs_contradiction"
    assert provider.record() == {key: choice.raw[key] for key in provider.record()}
    caps = provider.capabilities()
    assert caps.primitives == frozenset({"choice", "boolean"}) and caps.dynamic_labels and not caps.training


def test_unsupported_or_invalid_requests_never_call_the_model():
    provider = _provider()
    with pytest.raises(UnsupportedCapability, match="score"):
        provider.decide(Score("How severe?", (("low", "minor"), ("high", "major"))), [TEXT])
    with pytest.raises(UnsupportedCapability, match="sequence of texts"):
        provider.decide(Boolean("late goal"), TEXT)  # a bare string, not a batch
    with pytest.raises(InvalidDecisionRequest, match="not non-empty strings"):
        provider.decide(Boolean("late goal"), [TEXT, "  "])
    with pytest.raises(UnsupportedCapability, match="max_batch"):
        _provider(max_batch=1).decide(Boolean("late goal"), [TEXT, TEXT])
    assert provider.model_calls == 0 and provider.model.seen == [] and provider.tokenizer.calls == []
    assert provider.decide(Boolean("late goal"), []) == [] and provider.model_calls == 0
