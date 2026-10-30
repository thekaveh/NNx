"""FEAT-009: the fixed-head NNx decision adapter.

A trained classifier answers only the primitives its head justifies, over
its exact label space (or a bijection onto it). Columns come back in the
request's option order; unseen or missing labels, unsupported primitives,
modalities and batch sizes are rejected before the model-call counter
advances; every submodule's training mode is restored.
"""

from __future__ import annotations

import re

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
    order = ("dog", "fox", "cat")  # a permutation of the task's label order
    with torch.no_grad():
        probs = torch.softmax(model.net(X), dim=1)[:, [LABELS.index(name) for name in order]]
    expected = [order[i] for i in probs.argmax(dim=1).tolist()]  # first maximum on ties, as .top
    assert [result.top for result in provider.decide(_animals(*order), X)] == expected == ["cat", "dog", "cat"]


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

    binary = FixedHeadProvider(_classifier(1, Losses.BINARY_CROSS_ENTROPY), question=Boolean("Is it a cat?"))
    assert binary.capabilities().primitives == {"boolean"}
    results = binary.decide(Boolean("Is it a cat?"), X)
    assert all(isinstance(result, BooleanResult) for result in results)
    np.testing.assert_allclose([r.p_true for r in results], 1 / (1 + np.exp(-2.0 * X[:, 0].numpy())), rtol=1e-6)
    with pytest.raises(UnsupportedCapability, match="not 'choice'"):
        binary.decide(_animals("cat", "dog", "fox"), X)
    with pytest.raises(UnsupportedCapability, match="regression head"):
        FixedHeadProvider(_classifier(1, Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(1)))
    with pytest.raises(UnsupportedCapability, match="regression head"):  # a regression loss without a task
        FixedHeadProvider(_classifier(2, Losses.MEAN_SQUARED_ERROR), labels=("lo", "hi"))
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
    provider = FixedHeadProvider(wide, question=Boolean("Is it a cat?"))  # width unknown until the head runs
    with pytest.raises(InvalidDecisionRequest, match="one-output head"):
        provider.decide(Boolean("Is it a cat?"), X)


def test_a_bernoulli_head_answers_only_the_boolean_it_was_trained_for():
    cat = Boolean("Is it a cat?")
    binary = FixedHeadProvider(_classifier(1, Losses.BINARY_CROSS_ENTROPY), question=cat)
    assert len(binary.decide(Boolean("Is it a cat?"), X)) == 3  # the same question: the same digest
    for other in (Boolean("Is it a dog?"), Boolean("Is it a fox?")):
        with pytest.raises(UnsupportedCapability, match="only the Boolean it was trained for, 'Is it a cat\\?'"):
            binary.check(other, X)
        with pytest.raises(UnsupportedCapability, match=re.escape(f"not {other.prompt!r}")):
            binary.decide(other, X)
    assert binary.model_calls == 1  # another Boolean never reaches the model
    with pytest.raises(InvalidDecisionRequest, match="needs the Boolean it answers"):
        FixedHeadProvider(_classifier(1, Losses.BINARY_CROSS_ENTROPY))
    with pytest.raises(InvalidDecisionRequest, match="must be a Boolean"):
        FixedHeadProvider(_classifier(1, Losses.BINARY_CROSS_ENTROPY), question=_animals("cat", "dog", "fox"))  # type: ignore[arg-type]
    with pytest.raises(InvalidDecisionRequest, match="a categorical head answers by its labels"):
        FixedHeadProvider(_classifier(), labels=LABELS, question=cat)


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


@pytest.mark.parametrize("bad", [{"max_batch": 0}, {"max_batch": True}, {"ordinal": "no"}, {"name": ""}])
def test_a_fixed_head_provider_is_configured_at_construction(bad):
    """A bad setting fails at construction as a decision error, not at the
    first call (where jobs reported it as a provider failure)."""
    with pytest.raises(InvalidDecisionRequest):
        FixedHeadProvider(_classifier(3), labels=("cat", "dog", "fox"), **bad)


@pytest.mark.parametrize(
    "bad",
    [{"primitives": "choice"}, {"primitives": {"chioce"}}, {"labels": "ab"}, {"inference": "yes"}, {"max_batch": 0}],
)
def test_capabilities_refuse_malformed_declarations(bad):
    from nnx.decisions import Capabilities

    fields = {"primitives": {"choice"}, "modalities": {"text"}, **bad}
    with pytest.raises(InvalidDecisionRequest):
        Capabilities(**fields)


def test_non_finite_logits_are_a_provider_failure():
    """NaN logits are the backend failing on a valid request, not a
    malformed request (decide_result returns Err("provider_failure"))."""
    from torch import nn

    from nnx.decisions import ProviderFailure

    model = NNModel(module=nn.Linear(2, 3), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    with torch.no_grad():
        model.net.weight.fill_(float("nan"))
    provider = FixedHeadProvider(model, labels=("cat", "dog", "fox"))
    with pytest.raises(ProviderFailure, match="non-finite"):
        provider.decide(_animals("cat", "dog", "fox"), X)


def test_fixed_head_labels_and_output_shapes_are_decision_errors():
    from torch import nn

    with pytest.raises(InvalidDecisionRequest, match="not the string"):
        FixedHeadProvider(NNModel(module=nn.Linear(2, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY)), labels="ab")
    bce = NNModel(module=nn.Linear(2, 1), params=NNModelParams(loss=Losses.BINARY_CROSS_ENTROPY))
    with pytest.raises(InvalidDecisionRequest, match="non-empty strings"):
        FixedHeadProvider(bce, labels=("",), question=Boolean("Is it a cat?"))

    class Seg(nn.Module):
        def forward(self, x):
            return x.new_zeros(x.shape[0], 3, 4)  # (N, C, L): class-first but not (rows, classes)

    seg = FixedHeadProvider(
        NNModel(module=Seg(), params=NNModelParams(loss=Losses.CROSS_ENTROPY)), labels=("cat", "dog", "fox")
    )
    with pytest.raises(InvalidDecisionRequest, match="rows, classes"):
        seg.decide(_animals("cat", "dog", "fox"), X)


def test_a_bernoulli_head_refuses_categorical_settings_and_labels_must_be_iterable():
    from nnx.decisions import InvalidDecisionRequest

    bce = _classifier(1, Losses.BINARY_CROSS_ENTROPY)
    for extra in ({"ordinal": True}, {"option_map": {"x": "y"}}):
        with pytest.raises(InvalidDecisionRequest, match="categorical head, not a Bernoulli"):
            FixedHeadProvider(bce, question=Boolean("Is it a cat?"), **extra)
    with pytest.raises(InvalidDecisionRequest, match="sequence of strings, got int"):
        FixedHeadProvider(_classifier(3, Losses.CROSS_ENTROPY), labels=3)


# ------------------------------------------- FIX-047 (#514): one precision on every producer

TWO = ("a", "b")
AB = Choice("Which class?", (("a", "class a"), ("b", "class b")))


def _gap_head(dtype: torch.dtype) -> NNModel:
    """A categorical task head whose logits are ``[0, x]``: the input is the
    logit gap, in ``dtype`` (float16 runs on the CPU too)."""
    model = NNModel(
        net_params=NNParams(input_dim=1, output_dim=2, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(2, labels=TWO)
        ),
    )
    with torch.no_grad():
        model.net.layers[0].weight.copy_(torch.tensor([[0.0], [1.0]]))
        model.net.layers[0].bias.zero_()
    model.net.to(dtype)
    return model


def _streamed(model: NNModel, X: torch.Tensor, spec, batch_size: int) -> list:
    from torch.utils.data import DataLoader, TensorDataset

    with model.iter_predict(DataLoader(TensorDataset(X), batch_size=batch_size), spec) as stream:
        return list(stream)


@pytest.mark.parametrize("kind", ["max_probability", "margin"])
@pytest.mark.parametrize(
    ("dtype", "low", "high", "wrong_high"),
    [
        pytest.param(torch.float32, 17.0, 30.0, 20.0, id="float32-gaps-17-30"),
        pytest.param(torch.float16, 8.0, 12.0, 9.0, id="float16-gaps-8-12"),
    ],
)
def test_a_policy_tuned_on_any_producer_decides_alike_on_every_path(kind, dtype, low, high, wrong_high):
    """A policy tuned on any producer of the head's ``"probabilities"`` accepts
    the same rows through every path that scores them: all compute in float64
    from the same logits, so the saturation band of the head's own precision
    (float32 rounds to 1 from a logit gap of 16.6, float16 from 8.3) moves no
    decision."""
    from nnx.abstention import select_threshold
    from nnx.calibration import TemperatureCalibrator
    from nnx.decisions import DecisionJob
    from nnx.decisions.benchmark import Budget, Sample, collect, evaluate
    from nnx.prediction import ProbabilitySpec
    from nnx.result import decide_result

    model = _gap_head(dtype)
    provider = FixedHeadProvider(model)
    spec = ProbabilitySpec("categorical", labels=TWO)
    calibrator = TemperatureCalibrator(temperature=1.0, labels=TWO, model_id="m", split_id="cal")
    rng = np.random.default_rng(0)
    column = lambda gaps: torch.tensor(gaps, dtype=dtype).unsqueeze(1)  # noqa: E731
    x_val = column(np.concatenate([rng.uniform(low, high, 40), rng.uniform(low, wrong_high, 10)]))
    y_val = np.array([1] * 40 + [0] * 10)  # 40 right, 10 wrong
    raw = model.predict_proba(x_val, spec)
    assert raw.logits.dtype == np.dtype(str(dtype).removeprefix("torch."))
    head_rows = np.array([[p for _, p in result.distribution] for result in provider.decide(AB, x_val)])
    tuning = {
        "predict_proba": (raw, {"model_id": "m"}),
        "task predict_proba": (model.predict_proba(x_val), {"model_id": "m"}),
        "iter_predict": (_streamed(model, x_val, spec, batch_size=50)[0], {"model_id": "m"}),
        "raw view": (calibrator.transform(raw.logits, labels=TWO, model_id="m"), {"input_field": "probabilities"}),
        "FixedHeadProvider": (head_rows, {"labels": TWO, "model_id": "m"}),
    }
    selections = {
        name: select_threshold(source, y_val, kind=kind, risk_ceiling=0.0, split_id="val", **declared)
        for name, (source, declared) in tuning.items()
    }
    policy = selections["predict_proba"].require()
    for name, selection in selections.items():
        assert selection == selections["predict_proba"], name

    x_new = column(rng.uniform(low, high, 20))
    new = model.predict_proba(x_new, spec)
    accepted = {
        "predict_proba": policy.apply(new, model_id="m").accepted,
        "task predict_proba": policy.apply(model.predict_proba(x_new), model_id="m").accepted,
        "iter_predict": np.concatenate(
            [policy.apply(chunk, model_id="m").accepted for chunk in _streamed(model, x_new, spec, batch_size=7)]
        ),
        "raw view": policy.apply(calibrator.transform(new.logits, labels=TWO, model_id="m")).accepted,
        "decide_result": np.array(
            [d.accepted for d in decide_result(provider, AB, x_new, policy=policy, model_id="m").unwrap()]
        ),
    }
    rows = DecisionJob.ask(AB, id="q", policy=policy, model_id="m").run(provider, state=x_new).outcomes["q"].rows
    accepted["DecisionJob"] = np.array([row.kind == "answered" for row in rows])
    reference = accepted["predict_proba"]
    assert 0 < reference.sum() < 20  # the threshold lies inside the head's saturation band
    for name, mask in accepted.items():
        np.testing.assert_array_equal(mask, reference, err_msg=name)
    samples = [Sample(id=f"s{i}", question=AB, input=x_new[i], label="b") for i in range(20)]
    records = collect(provider, samples, provider_id="head", budget=Budget(max_calls=20)).records
    report = evaluate(samples, records, split="test", policy=policy, model_id="m")
    assert report.slices["in_family"].metrics["selective_coverage"].value == reference.mean()


# ------------------------------------------- FIX-063 (#530): a task-less head's kind from its loss


def _taskless(loss_fn: torch.nn.Module, output_dim: int = 3) -> tuple[NNModel, list[int]]:
    """A no-task built-in head (its width is known) with a runtime ``loss_fn``,
    and the list its forward pre-hook appends to: empty while the model never ran."""
    model = _classifier(output_dim)
    model.loss_fn = loss_fn
    calls: list[int] = []
    model.net.register_forward_pre_hook(lambda module, args: calls.append(1))
    return model, calls


class _FocalLoss(torch.nn.Module):
    """A custom categorical loss NNx cannot recognize."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        return (-((1 - p) ** 2) * torch.log(p)).mean()


def _name(loss_fn: torch.nn.Module) -> str:
    return type(loss_fn).__name__


@pytest.mark.parametrize("output_dim", [1, 2])
@pytest.mark.parametrize(
    "loss_fn",
    [
        torch.nn.MSELoss(),
        torch.nn.L1Loss(),
        torch.nn.SmoothL1Loss(),
        torch.nn.HuberLoss(),
        torch.nn.PoissonNLLLoss(),
        torch.nn.GaussianNLLLoss(),
    ],
    ids=_name,
)
def test_a_taskless_regression_loss_is_refused_before_the_model_runs(loss_fn, output_dim):
    """A count (Poisson) or Gaussian likelihood head is a regression head like
    an MSELoss one: a softmax of its outputs is no distribution it was
    trained to give."""
    model, calls = _taskless(loss_fn, output_dim)
    with pytest.raises(UnsupportedCapability, match="a regression head justifies no decision primitive"):
        FixedHeadProvider(model, labels=LABELS[:output_dim])
    assert calls == []  # refused at construction, before any forward pass


@pytest.mark.parametrize(
    ("loss_fn", "output_dim", "match"),
    [
        pytest.param(loss_fn, width, match, id=f"{_name(loss_fn)}-{width}")
        for loss_fn, width, match in [
            (torch.nn.BCEWithLogitsLoss(), 2, "multi-output Bernoulli"),
            (torch.nn.MultiLabelSoftMarginLoss(), 2, "multi-output Bernoulli"),
            (torch.nn.MultiLabelSoftMarginLoss(), 3, "multi-output Bernoulli"),
            (torch.nn.SoftMarginLoss(), 2, "multi-output Bernoulli"),
            (torch.nn.SoftMarginLoss(), 3, "multi-output Bernoulli"),
            (torch.nn.MultiLabelMarginLoss(), 2, "MultiLabelMarginLoss head is a multilabel hinge"),
            (torch.nn.MultiLabelMarginLoss(), 3, "MultiLabelMarginLoss head is a multilabel hinge"),
            (torch.nn.MultiLabelMarginLoss(), 1, "MultiLabelMarginLoss head is a multilabel hinge"),
        ]
    ],
)
def test_a_taskless_multilabel_loss_is_refused_like_a_multi_output_bce_head(loss_fn, output_dim, match):
    """The logistic margin losses train BCEWithLogitsLoss's objective, so a
    multi-output head is refused like a multi-output BCEWithLogitsLoss one;
    a multilabel hinge's margin scores justify no probability at any width."""
    model, calls = _taskless(loss_fn, output_dim)
    for settings in ({"labels": LABELS[:output_dim]}, {"question": Boolean("Is it a cat?")}):
        with pytest.raises(UnsupportedCapability, match=match):
            FixedHeadProvider(model, **settings)
    assert calls == []


@pytest.mark.parametrize("loss_fn", [torch.nn.MultiLabelSoftMarginLoss(), torch.nn.SoftMarginLoss()], ids=_name)
def test_a_one_output_logistic_margin_head_answers_its_boolean_by_sigmoid(loss_fn):
    """Like a one-output BCEWithLogitsLoss head (its objective): the sigmoid
    of its one logit is the probability it was trained to give."""
    model, _ = _taskless(loss_fn, 1)
    binary = FixedHeadProvider(model, question=Boolean("Is it a cat?"))
    assert binary.capabilities().primitives == {"boolean"}
    results = binary.decide(Boolean("Is it a cat?"), X)
    np.testing.assert_allclose([r.p_true for r in results], 1 / (1 + np.exp(-2.0 * X[:, 0].numpy())), rtol=1e-6)


def _sigmoid_head() -> NNModel:
    from torch import nn

    return NNModel(module=nn.Sequential(nn.Linear(2, 2), nn.Sigmoid()), params=NNModelParams())


@pytest.mark.parametrize(
    ("build", "settings"),
    [
        pytest.param(lambda: _classifier(1), {"question": Boolean("Is it a cat?")}, id="one-output"),
        pytest.param(lambda: _classifier(2), {"labels": ("cat", "dog")}, id="two-output"),
        pytest.param(_sigmoid_head, {"labels": ("cat", "dog")}, id="width-unknown"),
        pytest.param(
            lambda: _classifier(1, Losses.BINARY_CROSS_ENTROPY, task=TaskSpec.multilabel(1)),
            {"question": Boolean("Is it a cat?")},
            id="multilabel-task",
        ),
        pytest.param(lambda: _classifier(task=TaskSpec.categorical(3, labels=LABELS)), {}, id="categorical-task"),
    ],
)
def test_a_bce_loss_head_is_never_served_from_its_outputs(build, settings):
    """A BCELoss head outputs probabilities, not logits: a softmax or sigmoid
    of them is no probability the head gives (with or without a task, whose
    own loss check refuses BCELoss)."""
    model = build()
    model.loss_fn = torch.nn.BCELoss()
    calls: list[int] = []
    model.net.register_forward_pre_hook(lambda module, args: calls.append(1))
    with pytest.raises(UnsupportedCapability, match="BCELoss head outputs probabilities.*BCEWithLogitsLoss"):
        FixedHeadProvider(model, **settings)
    assert calls == []


@pytest.mark.parametrize("output_dim", [2, 3])
@pytest.mark.parametrize(
    ("loss_fn", "hinge"),
    [(torch.nn.MultiMarginLoss(), "multi-class"), (torch.nn.MultiLabelMarginLoss(), "multilabel")],
    ids=["MultiMarginLoss", "MultiLabelMarginLoss"],
)
def test_a_taskless_hinge_loss_is_refused_without_a_remedy(loss_fn, hinge, output_dim):
    """A hinge head's outputs are margin scores, not logits: no loss_fn or
    TaskSpec makes a softmax of them a probability, so none is offered."""
    model, calls = _taskless(loss_fn, output_dim)
    with pytest.raises(UnsupportedCapability, match=f"a {_name(loss_fn)} head is a {hinge} hinge") as caught:
        FixedHeadProvider(model, labels=LABELS[:output_dim])
    assert "TaskSpec" not in str(caught.value) and "loss_fn" not in str(caught.value)
    assert calls == []


@pytest.mark.parametrize("loss_fn", [_FocalLoss(), torch.nn.KLDivLoss(reduction="batchmean")], ids=_name)
def test_an_unknown_taskless_loss_is_refused_with_the_fix(loss_fn):
    """Without a task only the losses NNx knows name a head's kind; any other
    is refused, not served as a softmax, and the refusal names both fixes for
    a head of logits (a hinge's margin scores are none: see above)."""
    model, calls = _taskless(loss_fn)
    with pytest.raises(UnsupportedCapability, match=re.escape(_name(loss_fn))) as caught:
        FixedHeadProvider(model, labels=LABELS)
    assert "declare the model's TaskSpec" in str(caught.value)
    assert "set model.loss_fn to the matching CrossEntropyLoss, NLLLoss or BCEWithLogitsLoss" in str(caught.value)
    assert calls == []
    model.loss_fn = torch.nn.CrossEntropyLoss()  # one fix: the loss whose logits the head gives
    assert len(FixedHeadProvider(model, labels=LABELS).decide(_animals("cat", "dog", "fox"), X)) == 3
    declared = _classifier(task=TaskSpec.categorical(3, labels=LABELS))
    declared.loss_fn = loss_fn  # the other: a declared task names the head's kind
    assert len(FixedHeadProvider(declared).decide(_animals("cat", "dog", "fox"), X)) == 3
