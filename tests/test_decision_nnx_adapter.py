"""FEAT-009: the fixed-head NNx decision adapter.

A trained classifier answers only the primitives its head justifies, over
its exact label space (or a bijection onto it). Columns come back in the
request's option order; unseen or missing labels, unsupported primitives,
modalities and batch sizes are rejected before the model-call counter
advances; every submodule's training mode is restored.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
    TaskSpec,
)
from nnx.decisions import (
    Boolean,
    BooleanResult,
    Choice,
    ChoiceResult,
    DecisionProvider,
    FixedHeadProvider,
    InvalidDecisionRequest,
    Option,
    ProviderFailure,
    Score,
    ScoreResult,
    UnsupportedCapability,
)

LABELS = ("cat", "dog", "fox")
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])


def _classifier(output_dim: int = 3, loss: Losses = Losses.CROSS_ENTROPY, task=None) -> NNModel:
    """A deterministic two-feature head: logits = X @ W.T (no hidden layer)."""
    model = NNModel(
        net_params=NNParams(
            input_dim=2, output_dim=output_dim, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=loss, task=task),
    )
    weight = torch.tensor([[2.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])[:output_dim]
    with torch.no_grad():
        model.net.layers[0].weight.copy_(weight)
        model.net.layers[0].bias.zero_()
    return model


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def _animals(*order: str) -> Choice:
    names = {"cat": "A cat", "dog": "A dog", "fox": "A fox", "cow": "A cow"}
    return Choice("Which animal is it?", tuple(Option(name, names[name]) for name in order))


def test_the_adapter_reorders_columns_into_the_request_order():
    model = _classifier()
    provider = FixedHeadProvider(model, labels=LABELS)
    assert isinstance(provider, DecisionProvider)
    question = _animals("fox", "cat", "dog")
    results = provider.decide(question, X)
    expected = _softmax(X.numpy() @ np.array([[2.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]).T)
    assert len(results) == 3 and all(isinstance(result, ChoiceResult) for result in results)
    for result, row in zip(results, expected, strict=True):
        assert [option_id for option_id, _ in result.distribution] == ["fox", "cat", "dog"]
        np.testing.assert_allclose([p for _, p in result.distribution], row[[2, 0, 1]], rtol=1e-6)
        assert result.question_digest == question.digest() and result.provider == "nnx.fixed_head"
        assert result.raw.shape == (3,)  # the row's logits, kept apart from the distribution
    assert provider.model_calls == 1


def test_the_label_space_must_match_exactly_or_through_a_bijection():
    model = _classifier()
    provider = FixedHeadProvider(model, labels=LABELS)
    with pytest.raises(UnsupportedCapability, match="unseen labels \\['cow'\\]"):
        provider.decide(_animals("cat", "dog", "cow"), X)
    with pytest.raises(UnsupportedCapability, match="missing \\['fox'\\]"):
        provider.decide(_animals("cat", "dog"), X)
    assert provider.model_calls == 0  # rejected before any model call

    mapped = FixedHeadProvider(model, labels=LABELS, option_map={"f": "fox", "c": "cat", "d": "dog"})
    question = Choice("Which?", (Option("c", "Cat"), Option("d", "Dog"), Option("f", "Fox")))
    direct = provider.decide(_animals("cat", "dog", "fox"), X)
    via_map = mapped.decide(question, X)
    for a, b in zip(direct, via_map, strict=True):
        assert [p for _, p in a.distribution] == [p for _, p in b.distribution]
    with pytest.raises(UnsupportedCapability, match="unseen option ids \\['cat', 'dog', 'fox'\\]"):
        mapped.decide(_animals("cat", "dog", "fox"), X)
    with pytest.raises(InvalidDecisionRequest, match="bijection"):
        FixedHeadProvider(model, labels=LABELS, option_map={"a": "cat", "b": "cat", "c": "dog"})
    with pytest.raises(InvalidDecisionRequest, match="2 labels for a head with 3 output classes"):
        FixedHeadProvider(model, labels=("cat", "dog"))
    with pytest.raises(InvalidDecisionRequest, match="needs its label space"):
        FixedHeadProvider(model)


def test_task_labels_supply_the_label_space():
    model = _classifier(task=TaskSpec.categorical(3, labels=LABELS))
    provider = FixedHeadProvider(model)
    assert provider.capabilities().labels == LABELS
    assert provider.decide(_animals("dog", "fox", "cat"), X)[0].top in LABELS


def test_heads_serve_only_the_primitives_they_justify():
    categorical = FixedHeadProvider(_classifier(), labels=LABELS)
    caps = categorical.capabilities()
    assert caps.primitives == {"choice"} and caps.modalities == {"tensor"} and not caps.dynamic_labels
    assert (caps.inference, caps.training, caps.export) == (True, True, True)
    with pytest.raises(UnsupportedCapability, match="not 'boolean'"):
        categorical.decide(Boolean("Is it a cat?"), X)
    levels = Score("How big is it?", (("cat", "Small"), ("fox", "Medium"), ("dog", "Large")))
    with pytest.raises(UnsupportedCapability, match="not 'score'"):
        categorical.decide(levels, X)

    ordinal = FixedHeadProvider(_classifier(), labels=LABELS, ordinal=True)
    with pytest.raises(UnsupportedCapability, match="follow the head's class order"):
        ordinal.decide(levels, X)  # levels shuffled against the head's class order
    in_order = Score("How big is it?", (("cat", "Small"), ("dog", "Medium"), ("fox", "Large")))
    (score, *_) = ordinal.decide(in_order, X)
    assert isinstance(score, ScoreResult) and [level for level, _ in score.distribution] == ["cat", "dog", "fox"]
    assert score.expected_index == pytest.approx(sum(i * p for i, (_, p) in enumerate(score.distribution)))
    assert ordinal.model_calls == 1

    binary = FixedHeadProvider(_classifier(1, Losses.BINARY_CROSS_ENTROPY))
    assert binary.capabilities().primitives == {"boolean"}
    results = binary.decide(Boolean("Is it a cat?"), X)
    assert all(isinstance(result, BooleanResult) for result in results)
    np.testing.assert_allclose([r.p_true for r in results], 1 / (1 + np.exp(-2.0 * X[:, 0].numpy())), rtol=1e-6)
    with pytest.raises(UnsupportedCapability, match="not 'choice'"):
        binary.decide(_animals("cat", "dog", "fox"), X)
    with pytest.raises(UnsupportedCapability, match="regression head"):
        FixedHeadProvider(_classifier(1, Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(1)))
    with pytest.raises(UnsupportedCapability, match="multi-output Bernoulli"):
        FixedHeadProvider(_classifier(3, Losses.BINARY_CROSS_ENTROPY, task=TaskSpec.multilabel(3)))


def test_unsupported_inputs_and_batches_fail_before_the_model_is_called():
    provider = FixedHeadProvider(_classifier(), labels=LABELS, max_batch=2)
    with pytest.raises(UnsupportedCapability, match="max_batch=2"):
        provider.decide(_animals("cat", "dog", "fox"), X)
    with pytest.raises(UnsupportedCapability, match="tensor inputs"):
        provider.decide(_animals("cat", "dog", "fox"), ["a photo of a fox"])
    assert provider.model_calls == 0


def test_model_modes_are_restored_and_backend_failures_are_typed():
    model = _classifier()
    model.net.train()
    model.net.layers[0].eval()  # mixed modes must come back exactly
    provider = FixedHeadProvider(model, labels=LABELS)
    provider.decide(_animals("cat", "dog", "fox"), X)
    assert model.net.training and not model.net.layers[0].training

    def broken(*args, **kwargs):
        raise RuntimeError("device lost")

    model.net.forward = broken  # type: ignore[method-assign]
    with pytest.raises(ProviderFailure, match="device lost") as caught:
        provider.decide(_animals("cat", "dog", "fox"), X)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert model.net.training and not model.net.layers[0].training
    assert provider.model_calls == 2


# --- review hardening ------------------------------------------------------------------------


def test_multi_output_bernoulli_heads_are_rejected_with_or_without_a_task():
    from torch import nn

    with pytest.raises(UnsupportedCapability, match="multi-output Bernoulli"):
        FixedHeadProvider(_classifier(3, Losses.BINARY_CROSS_ENTROPY))  # no task: the net's width decides
    wide = NNModel(module=nn.Linear(2, 3), params=NNModelParams(loss=Losses.BINARY_CROSS_ENTROPY))
    provider = FixedHeadProvider(wide)  # width unknown until the head runs
    with pytest.raises(UnsupportedCapability, match="one-output head"):
        provider.decide(Boolean("Is it a cat?"), X)


def test_explicit_labels_must_agree_with_the_task_and_fit_the_head():
    from torch import nn

    task_model = _classifier(task=TaskSpec.categorical(3, labels=LABELS))
    with pytest.raises(InvalidDecisionRequest, match="disagree with the model's task labels"):
        FixedHeadProvider(task_model, labels=("fox", "dog", "cat"))
    assert FixedHeadProvider(task_model, labels=LABELS).labels == LABELS
    unknown_width = NNModel(module=nn.Linear(2, 4), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    provider = FixedHeadProvider(unknown_width, labels=LABELS)  # 3 labels for a 4-way head, found on the call
    with pytest.raises(InvalidDecisionRequest, match="does not fit its labels"):
        provider.decide(_animals("cat", "dog", "fox"), X)


def test_half_precision_logits_still_validate(monkeypatch):
    from nnx.prediction import prediction_from_logits

    model = _classifier()
    provider = FixedHeadProvider(model, labels=LABELS)
    logits16 = torch.randn(64, 3, generator=torch.Generator().manual_seed(0)).numpy().astype(np.float16) * 4

    def half_predict_proba(inputs, spec):
        return prediction_from_logits(logits16, spec)  # a float16 head's output

    monkeypatch.setattr(model, "predict_proba", half_predict_proba)
    results = provider.decide(_animals("cat", "dog", "fox"), torch.zeros(64, 2))
    assert len(results) == 64 and all(abs(sum(p for _, p in r.distribution) - 1) <= 1e-6 for r in results)


def test_empty_batches_answer_nothing_without_calling_the_model_and_raw_is_owned():
    provider = FixedHeadProvider(_classifier(), labels=LABELS)
    assert provider.decide(_animals("cat", "dog", "fox"), torch.zeros(0, 2)) == []
    assert provider.model_calls == 0
    question = _animals("cat", "dog", "fox")
    first, *_ = provider.decide(question, X)
    assert first.raw.base is None  # a copy, not a view keeping the whole batch alive
    assert question.digest() is question.digest()  # computed once per question
