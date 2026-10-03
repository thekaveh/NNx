"""A local label-conditioned baseline: decisions from an NLI model (FEAT-011).

``nnx.decisions.NLIProvider`` scores a text (the premise) against candidate
descriptions supplied at inference time (each rendered into a hypothesis),
with a caller-supplied natural-language-inference model and tokenizer:

  1. **Choice.** Every (text, candidate) pair is scored; the entailment
     logits are normalized across the candidates with a softmax — logits
     ``log(3)`` and ``0`` give ``(0.75, 0.25)``. No candidate existed when
     the provider was built.
  2. **Boolean.** One pair per text; the softmax over that pair's
     contradiction and entailment logits — contradiction ``0``, entailment
     ``log(4)`` give ``p_true = 0.8``. Texts are never normalized together.
  3. **Evaluation on supplied records.** Accuracy, macro-F1 and the
     categorical NLL / Brier score of the Choice distributions on a small
     labelled split, recorded with the split, the model revision and the
     provider's settings. The scores are zero-shot, **not calibrated**:
     how well an NLI model transfers to a decision task is measured here,
     not assumed.
  4. **Refusals before any model call.** A ``Score`` (unsupported) and an
     invalid request leave the call counter at zero.

NNx downloads nothing: in practice the caller loads a model and tokenizer
(pinning a revision) and passes them in. This example runs fully offline
with a tiny deterministic stand-in — a word-overlap "NLI model" whose
entailment logit is ``log(1 + overlap)`` — so every number is reproducible.

Run:
    python examples/decision_nli.py

The bounded ``decision_nli_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from nnx.calibration import brier_score, negative_log_likelihood
from nnx.decisions import (
    Boolean,
    Choice,
    InvalidDecisionRequest,
    NLIProvider,
    Score,
    UnsupportedCapability,
)

CLS, SEP, PAD = 1, 2, 0


class WordTokenizer:
    """A HuggingFace-style pair tokenizer over whole words (stand-in)."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def ids(self, text: str) -> list[int]:
        words = [word.strip(".,?!").lower() for word in text.split()]
        return [self.vocab.setdefault(word, len(self.vocab) + 3) for word in words if word]

    def __call__(self, premises, hypotheses, *, truncation, max_length=None, padding=False, return_tensors=None):
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
        return {
            "input_ids": torch.tensor([row + [PAD] * (width - len(row)) for row in rows]),
            "token_type_ids": torch.tensor([row + [0] * (width - len(row)) for row in types]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
        }


class OverlapNLI(nn.Module):
    """Entailment logit ``log(1 + overlap)`` (distinct hypothesis words in
    the premise); contradiction and neutral ``0``. A stand-in for a real
    NLI cross-encoder, with the same calling convention."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_labels=3, label2id={"contradiction": 0, "neutral": 1, "entailment": 2})
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, input_ids, token_type_ids, attention_mask):
        rows = []
        for ids, kinds, mask in zip(input_ids, token_type_ids, attention_mask, strict=True):
            live = mask.bool()
            premise = set(ids[live & (kinds == 0)].tolist()) - {CLS, SEP}
            hypothesis = set(ids[live & (kinds == 1)].tolist()) - {SEP}
            rows.append([0.0, 0.0, math.log1p(len(premise & hypothesis))])
        return SimpleNamespace(logits=torch.tensor(rows) * self.scale)


TOPICS = Choice(
    "What is the text about?",
    (("t-sport", "late goal match"), ("t-econ", "interest rates market"), ("t-weather", "heavy rain storm")),
)

# A small labelled split (supplied records): text and the expected topic id.
RECORDS = [
    ("the match ended with a late goal", "t-sport"),
    ("a goal in the last minute won the match", "t-sport"),
    ("the market fell as interest rates rose", "t-econ"),
    ("rates are expected to climb", "t-econ"),
    ("a heavy storm brought rain all night", "t-weather"),
    ("the rain stopped the match", "t-weather"),  # a hard case: shares a word with sport
]


def decision_nli_workflow() -> dict:
    provider = NLIProvider(
        OverlapNLI(),
        WordTokenizer(),
        entailment_id="entailment",
        contradiction_id="contradiction",
        hypothesis_template="{}",
        boolean_template="{}",
        revision="stub-overlap-1",
    )

    # 1. Choice: candidates supplied now, scored by entailment softmax.
    (result,) = provider.decide(
        Choice("Topic?", (("t-sport", "late goal"), ("t-econ", "interest rates"))), [RECORDS[0][0]]
    )
    print("choice:", dict(result.distribution))
    assert np.allclose([p for _, p in result.distribution], [0.75, 0.25])  # logits log(3), 0

    # 2. Boolean: one pair per text, contradiction vs entailment.
    answers = provider.decide(Boolean("late goal match"), [RECORDS[0][0], "rates rose"])
    print("boolean p_true:", [round(a.p_true, 4) for a in answers])
    assert np.isclose(answers[0].p_true, 0.8) and np.isclose(answers[1].p_true, 0.5)  # never normalized together

    # 3. Evaluation on the supplied records.
    texts = [text for text, _ in RECORDS]
    ids = list(TOPICS.option_ids)
    targets = np.array([ids.index(label) for _, label in RECORDS])
    results = provider.decide(TOPICS, texts)
    probabilities = np.array([[p for _, p in r.distribution] for r in results])
    predicted = probabilities.argmax(axis=1)
    report = {
        "split": "demo-labelled-v1",
        "n_records": len(RECORDS),
        "revision": provider.revision,
        "settings": provider.record(),
        "accuracy": float((predicted == targets).mean()),
        "macro_f1": float(f1_score(targets, predicted, average="macro")),
        "nll": negative_log_likelihood(probabilities, targets),
        "brier": brier_score(probabilities, targets),
    }
    print(json.dumps(report, indent=2))
    assert report["settings"]["calibrated"] is False and 0.0 <= report["accuracy"] <= 1.0

    # 4. Refused before any model call: a fresh provider's call counter stays zero.
    fresh = NLIProvider(OverlapNLI(), WordTokenizer(), entailment_id=2, contradiction_id=0)
    for bad in (
        lambda: fresh.decide(Score("Severity?", (("low", "minor"), ("high", "major"))), texts),
        lambda: fresh.decide(TOPICS, ["valid text", ""]),
    ):
        try:
            bad()
            raise AssertionError("expected a refusal")
        except (UnsupportedCapability, InvalidDecisionRequest) as error:
            print("refused:", type(error).__name__, "-", error)
    assert fresh.model_calls == 0
    return report


def main() -> None:
    decision_nli_workflow()


if __name__ == "__main__":
    main()
