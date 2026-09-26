"""Typed decisions from a local NNx model (FEAT-009).

``nnx.decisions`` asks provider-neutral questions — ``Choice``, ``Boolean``
and ``Score`` — and returns validated, labelled probabilities. This example
runs the whole path locally, with no hosted provider:

  1. **A deterministic local model.** A two-feature, three-class
     ``NNModel`` whose weights are set explicitly, so every number below is
     reproducible.
  2. **A typed question.** A ``Choice`` whose options carry bookkeeping ids
     (``"sp-cat"`` …) separate from their model-facing descriptions, in an
     order different from the head's columns. ``model_view()`` shows what a
     text model would see — never the ids — and ``digest()`` pins the exact
     question.
  3. **The fixed-head provider.** ``FixedHeadProvider`` declares what the
     head justifies (``capabilities()``), maps the option ids onto the
     head's labels through a bijection, calls ``predict_proba`` once and
     returns ``ChoiceResult``\\ s in the question's option order, each
     validated (sums to 1, no renormalizing). An unseen label is rejected
     before the model is called.
  4. **Other primitives.** An ordinal ``Score`` from a head whose class
     order is the level order (``ordinal=True``; ``expected_index`` is a
     position between levels, not an interval score) and a ``Boolean`` from
     a one-logit head.

Fully offline, CPU only.

Run:
    python examples/decision_fixed_head.py

The bounded ``decision_fixed_head_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory.
"""

from __future__ import annotations

import numpy as np
import torch

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams, TaskSpec
from nnx.decisions import (
    Boolean,
    Choice,
    FixedHeadProvider,
    Option,
    Score,
    UnsupportedCapability,
    validate_response,
)

SPECIES = ("cat", "dog", "fox")


def _head(weight: list[list[float]], loss: Losses, task: TaskSpec) -> NNModel:
    model = NNModel(
        net_params=NNParams(
            input_dim=2, output_dim=len(weight), hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=loss, task=task),
    )
    with torch.no_grad():
        model.net.layers[0].weight.copy_(torch.tensor(weight))
        model.net.layers[0].bias.zero_()
    return model


def decision_fixed_head_workflow() -> None:
    # 1. A deterministic three-class head whose task names its classes.
    species = _head(
        [[2.0, 0.0], [0.0, 1.0], [-1.0, 1.0]],
        Losses.CROSS_ENTROPY,
        TaskSpec.categorical(3, labels=SPECIES),
    )
    photos = torch.tensor([[1.5, 0.0], [0.5, 2.0], [-1.0, 1.0]])

    # 2. A typed question: bookkeeping ids, model-facing descriptions, own order.
    question = Choice(
        "Which animal is in the photo?",
        (Option("sp-fox", "A fox"), Option("sp-cat", "A cat"), Option("sp-dog", "A dog")),
    )
    print("model sees:", question.model_view())
    assert "sp-fox" not in repr(question.model_view())
    print("digest:", question.digest()[:16], "…")

    # 3. The fixed-head provider over a bijection from option ids to labels.
    provider = FixedHeadProvider(species, option_map={"sp-cat": "cat", "sp-dog": "dog", "sp-fox": "fox"})
    caps = provider.capabilities()
    print("capabilities:", sorted(caps.primitives), sorted(caps.modalities), "labels", caps.labels)
    results = provider.decide(question, photos)
    for photo, result in zip(photos.tolist(), results, strict=True):
        shown = ", ".join(f"{option_id}={p:.3f}" for option_id, p in result.distribution)
        print(f"photo {photo}: {shown} -> {result.top}")
    assert [r.top for r in results] == ["sp-cat", "sp-dog", "sp-fox"]
    assert all(abs(sum(p for _, p in r.distribution) - 1.0) <= 1e-6 for r in results)
    assert all(r.question_digest == question.digest() for r in results) and provider.model_calls == 1

    # An unseen label never reaches the model.
    try:
        provider.decide(
            Choice("Which animal?", (("sp-cat", "A cat"), ("sp-dog", "A dog"), ("sp-cow", "A cow"))), photos
        )
        raise AssertionError("an unseen label must be rejected")
    except UnsupportedCapability as error:
        print("rejected before any model call:", error)
    assert provider.model_calls == 1

    # A provider's keyed output is validated the same way, whatever its order.
    keyed = validate_response(question, {"sp-dog": 0.1, "sp-fox": 0.6, "sp-cat": 0.3}, provider="elsewhere")
    assert [option_id for option_id, _ in keyed.distribution] == ["sp-fox", "sp-cat", "sp-dog"]

    # 4a. An ordinal Score from a head whose class order is the level order.
    severity = _head(
        [[-2.0, 0.0], [0.0, 0.5], [2.0, 0.0]],
        Losses.CROSS_ENTROPY,
        TaskSpec.categorical(3, labels=("low", "mid", "high")),
    )
    levels = Score("How severe is the damage?", (("low", "Cosmetic"), ("mid", "Degraded"), ("high", "Outage")))
    (score,) = FixedHeadProvider(severity, ordinal=True).decide(levels, photos[:1])
    print("severity:", dict(score.distribution), f"expected level index {score.expected_index:.3f} (ordinal)")
    assert score.expected_index > 1.5  # photo 0 leans to "high"

    # 4b. A Boolean from a one-logit head.
    outdoors = _head([[0.0, 3.0]], Losses.BINARY_CROSS_ENTROPY, TaskSpec.multilabel(1))
    answers = FixedHeadProvider(outdoors).decide(Boolean("Was the photo taken outdoors?"), photos)
    expected = 1.0 / (1.0 + np.exp(-3.0 * photos[:, 1].numpy()))
    np.testing.assert_allclose([a.p_true for a in answers], expected, rtol=1e-6)
    print("outdoors:", [round(a.p_true, 3) for a in answers])


def main() -> None:
    decision_fixed_head_workflow()


if __name__ == "__main__":
    main()
