"""FEAT-003: typed metric declarations.

Built-in metrics receive the input they declare — decoded labels,
probabilities or continuous outputs — and are computed over the full sample
(never unweighted batch means); custom metrics round-trip by registered
id/config, and an unknown registration is rejected before training without
running any metric code.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from nnx import (
    Activations,
    Devices,
    Losses,
    MetricSpec,
    MonitorSpec,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
    NNTrainParams,
    TaskSpec,
    register_metric,
    registered_metrics,
    unregister_metric,
)

# Evidence fixture: probabilities [[0.8, 0.2], [0.4, 0.6]], targets [0, 1]
# → accuracy 1.0, NLL 0.36698458754, Brier 0.2.
PROBABILITIES = np.array([[0.8, 0.2], [0.4, 0.6]])
TARGETS = np.array([0, 1])


def _accumulate(spec: MetricSpec, batches) -> float:
    accumulator = spec.accumulator()
    for target, prediction in batches:
        accumulator.update(np.asarray(target), np.asarray(prediction))
    value = accumulator.result()
    assert value is not None
    return value


@pytest.mark.parametrize(
    ("metric", "prediction", "expected"),
    [
        ("accuracy", PROBABILITIES.argmax(axis=1), 1.0),
        ("nll", PROBABILITIES, 0.36698458754),
        ("brier", PROBABILITIES, 0.2),
    ],
)
def test_builtin_accumulators_match_the_evidence_fixture_whole_and_one_example_per_batch(metric, prediction, expected):
    spec = MetricSpec(metric)
    whole = _accumulate(spec, [(TARGETS, prediction)])
    per_example = _accumulate(spec, [(TARGETS[i : i + 1], prediction[i : i + 1]) for i in range(2)])
    assert whole == pytest.approx(expected, abs=1e-11)
    assert per_example == pytest.approx(expected, abs=1e-11)


def _fixture_model(**model_kwargs) -> NNModel:
    """A linear classifier whose logits equal its inputs, so feeding
    log-probabilities makes softmax return exactly those probabilities."""
    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, **model_kwargs),
    )
    with torch.no_grad():
        model.net.layers[0].weight.copy_(torch.eye(2))
        model.net.layers[0].bias.zero_()
    return model


@pytest.mark.parametrize("task", [None, TaskSpec.categorical(2)], ids=["legacy", "task"])
def test_evaluate_feeds_labels_and_probabilities_as_declared(task):
    model = _fixture_model(task=task)
    X = torch.log(torch.tensor(PROBABILITIES, dtype=torch.float32))
    Y = torch.tensor(TARGETS)
    metrics = (MetricSpec("accuracy"), MetricSpec("nll"), MetricSpec("brier"), MetricSpec("f1"))
    whole = model.evaluate([(X, Y)], metrics=metrics)
    one_per_batch = model.evaluate([(X[:1], Y[:1]), (X[1:], Y[1:])], metrics=metrics)
    for record in (whole, one_per_batch):
        assert record.metrics["accuracy"] == pytest.approx(1.0)
        assert record.metrics["nll"] == pytest.approx(0.36698458754, abs=1e-6)
        assert record.metrics["brier"] == pytest.approx(0.2, abs=1e-6)
        assert record.metrics["f1"] == pytest.approx(1.0)


def test_evaluate_feeds_continuous_predictions_to_mae_and_mse():
    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(2)
        ),
    )
    with torch.no_grad():
        model.net.layers[0].weight.copy_(torch.eye(2))
        model.net.layers[0].bias.zero_()
    # FEAT-002 regression oracle: [[1, 3], [5, 7]] vs [[0, 1], [4, 5]] → MSE 2.5, MAE 1.5,
    # batches [1, 1]; the masked (NaN) target is excluded like the task's own metrics.
    X = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    Y = torch.tensor([[0.0, 1.0], [4.0, 5.0]])
    record = model.evaluate([(X[:1], Y[:1]), (X[1:], Y[1:])], metrics=(MetricSpec("mae"), MetricSpec("mse")))
    assert record.metrics["mae"] == pytest.approx(1.5) and record.metrics["mse"] == pytest.approx(2.5)
    Y_masked = Y.clone()
    Y_masked[1, 1] = math.nan
    masked = model.evaluate([(X, Y_masked)], metrics=(MetricSpec("mae"),))
    assert masked.metrics["mae"] == pytest.approx((1 + 2 + 1) / 3)


def test_multilabel_probabilities_are_independent_per_output():
    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.BINARY_CROSS_ENTROPY,
            task=TaskSpec.multilabel(2),
        ),
    )
    with torch.no_grad():
        model.net.layers[0].weight.copy_(torch.eye(2))
        model.net.layers[0].bias.zero_()
    p = torch.tensor([[0.9, 0.2]])
    X = torch.log(p / (1 - p))  # logits → sigmoid gives p back
    Y = torch.tensor([[1.0, 0.0]])
    record = model.evaluate([(X, Y)], metrics=(MetricSpec("brier"), MetricSpec("nll")))
    assert record.metrics["brier"] == pytest.approx((0.1**2 + 0.2**2) / 2, abs=1e-6)
    assert record.metrics["nll"] == pytest.approx(-(math.log(0.9) + math.log(0.8)) / 2, abs=1e-6)


def test_declared_inputs_and_directions():
    inputs = {m: MetricSpec(m).input for m in ("accuracy", "f1", "nll", "brier", "mae", "mse")}
    assert inputs == {
        "accuracy": "labels",
        "f1": "labels",
        "nll": "probabilities",
        "brier": "probabilities",
        "mae": "continuous",
        "mse": "continuous",
    }
    assert MetricSpec("nll").mode == "min" and MetricSpec("accuracy").mode == "max"


def test_additive_metrics_use_full_sample_denominators():
    target = np.zeros(3)
    prediction = np.array([1.0, 1.0, 4.0])
    # batches [2, 1]: the mean of batch means would be (1 + 4) / 2 = 2.5
    batches = [(target[:2], prediction[:2]), (target[2:], prediction[2:])]
    assert _accumulate(MetricSpec("mae"), batches) == pytest.approx(2.0)


def test_f1_is_computed_once_over_the_full_sample():
    from sklearn.metrics import f1_score

    target = np.array([0, 1, 1, 0, 1])
    prediction = np.array([0, 1, 0, 0, 1])
    batches = [(target[:2], prediction[:2]), (target[2:4], prediction[2:4]), (target[4:], prediction[4:])]
    assert _accumulate(MetricSpec("f1"), batches) == pytest.approx(f1_score(target, prediction, average="macro"))
    per_batch_mean = np.mean([f1_score(t, p, average="macro", zero_division=0) for t, p in batches])
    assert per_batch_mean != pytest.approx(_accumulate(MetricSpec("f1"), batches))


def test_spec_validation_and_serialization():
    spec = MetricSpec("f1", config={"average": "micro"}, name="f1_micro")
    assert spec.state() == {"id": "f1", "version": 1, "config": {"average": "micro"}, "name": "f1_micro"}
    assert MetricSpec.from_state(spec.state()) == spec and hash(MetricSpec.from_state(spec.state())) == hash(spec)
    assert MetricSpec("nll").state() == {"id": "nll", "version": 1}
    with pytest.raises(ValueError, match="reserved"):
        MetricSpec("nll", name="loss")
    with pytest.raises(ValueError, match="slug"):
        MetricSpec("bad id")
    with pytest.raises(TypeError, match="JSON-like"):
        MetricSpec("f1", config={"average": len})
    with pytest.raises(ValueError, match="average must be one of"):
        MetricSpec("f1", config={"average": "sideways"}).check()
    with pytest.raises(ValueError, match="takes no config"):
        MetricSpec("nll", config={"eps": 1e-9}).check()
    with pytest.raises(ValueError, match="unique"):
        NNTrainParams(n_epochs=1, metrics=[MetricSpec("nll"), MetricSpec("nll")])


class _Top:
    def __init__(self, calls):
        self.values: list[float] = []
        calls.append("accumulator")

    def update(self, target, prediction):
        self.values.extend(np.asarray(prediction).reshape(-1).tolist())

    def result(self):
        return max(self.values) if self.values else None


def test_custom_metrics_round_trip_by_id_and_config_and_unknown_ones_never_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls: list = []
    register_metric("tests.top", 1, lambda config: _Top(calls), input="continuous", mode="max")
    try:
        spec = MetricSpec("tests.top", config={"k": 2}, name="top")
        params = NNTrainParams(n_epochs=1, metrics=[spec], monitor=MonitorSpec(metric="top"))
        reloaded = NNTrainParams.from_state(params.state())
        assert reloaded.metrics == (spec,) and reloaded.state() == params.state()
        assert params.state()["metrics"] == [{"id": "tests.top", "version": 1, "config": {"k": 2}, "name": "top"}]
        assert calls == []  # declaring and serializing never execute metric code
    finally:
        unregister_metric("tests.top", 1)
    assert ("tests.top", 1) not in registered_metrics()

    # A run config naming an unregistered metric still reloads offline …
    offline = NNTrainParams.from_state(params.state())
    assert offline.metrics[0].label == "top"
    # … but training rejects it before any run directory or metric code.
    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=1, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(1)
        ),
    )
    loader = [(torch.randn(3, 2), torch.randn(3, 1))]
    with pytest.raises(ValueError, match="unknown metric 'tests.top'@v1"):
        model.train(params=NNTrainParams(n_epochs=1, metrics=offline.metrics).with_train_loader(loader))
    assert calls == [] and not os.path.exists("runs")


def test_a_metric_input_the_model_cannot_provide_fails_before_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=1, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(1)
        ),
    )
    loader = [(torch.randn(3, 2), torch.randn(3, 1))]
    with pytest.raises(ValueError, match="'nll' needs probabilities inputs"):
        model.train(params=NNTrainParams(n_epochs=1, metrics=[MetricSpec("nll")]).with_train_loader(loader))
    with pytest.raises(ValueError, match="'nll' needs probabilities inputs"):
        model.evaluate(loader, metrics=[MetricSpec("nll")])
    assert not os.path.exists("runs")


def _multilabel_model(threshold: float = 0.5) -> NNModel:
    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.BINARY_CROSS_ENTROPY,
            task=TaskSpec.multilabel(2, threshold=threshold),
        ),
    )
    with torch.no_grad():
        model.net.layers[0].weight.copy_(torch.eye(2))
        model.net.layers[0].bias.zero_()
    return model


def test_multilabel_labels_use_the_task_threshold():
    X = torch.tensor([[0.5, 2.0]])  # sigmoid 0.62 and 0.88
    Y = torch.tensor([[0.0, 1.0]])
    strict = _multilabel_model(threshold=0.8).evaluate([(X, Y)], metrics=(MetricSpec("accuracy"),))
    assert strict.metrics["accuracy"] == pytest.approx(1.0)  # 0.62 < 0.8 decodes to 0, as predict() does
    default = _multilabel_model().evaluate([(X, Y)], metrics=(MetricSpec("accuracy"),))
    assert default.metrics["accuracy"] == pytest.approx(0.5)


def test_f1_averaging_must_fit_the_model_outputs():
    X = torch.tensor([[0.5, 2.0]])
    Y = torch.tensor([[0.0, 1.0]])
    for average in ("macro", "micro"):
        with pytest.raises(ValueError, match="needs config=\\{'average': 'binary'\\}"):
            _multilabel_model().evaluate([(X, Y)], metrics=(MetricSpec("f1", config={"average": average}),))
    pooled = _multilabel_model().evaluate([(X, Y)], metrics=(MetricSpec("f1", config={"average": "binary"}),))
    assert pooled.metrics["f1"] == pytest.approx(2 / 3)  # TP=1, FP=1 (0.62 ≥ 0.5 on a negative), FN=0
    three_classes = NNModel(
        net_params=NNParams(input_dim=2, output_dim=3, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    with pytest.raises(ValueError, match="average='binary' needs exactly 2 classes, this model has 3"):
        three_classes.evaluate(
            [(torch.randn(2, 2), torch.tensor([0, 2]))], metrics=(MetricSpec("f1", config={"average": "binary"}),)
        )
    binary = _fixture_model().evaluate(
        [(torch.log(torch.tensor(PROBABILITIES, dtype=torch.float32)), torch.tensor(TARGETS))],
        metrics=(MetricSpec("f1", config={"average": "binary"}),),
    )
    assert binary.metrics["f1"] == pytest.approx(1.0)


def test_only_the_inputs_declared_metrics_need_are_built():
    from nnx.monitors import _batch_inputs

    logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    target = torch.tensor([0, 1])
    _, labels_only = _batch_inputs("categorical", target, logits, None, -100, frozenset({"labels"}))
    assert set(labels_only) == {"labels"} and labels_only["labels"].tolist() == [0, 1]
    _, both = _batch_inputs("categorical", target, logits, None, -100, frozenset({"labels", "probabilities"}))
    assert set(both) == {"labels", "probabilities"}
    assert both["probabilities"].sum(axis=1) == pytest.approx([1.0, 1.0])


def test_the_declared_metric_mask_compares_ignore_index_on_int64_targets():
    """#511: without a task, a uint8 156 (-100 wrapped) is a real class,
    while an ignore value the dtype holds still masks."""
    from nnx.monitors import _batch_inputs

    logits = torch.zeros(3, 200)
    target = torch.tensor([156, 3, 255], dtype=torch.uint8)
    kept, _ = _batch_inputs("categorical", target, logits, None, -100, frozenset({"labels"}))
    assert kept.tolist() == [156, 3, 255]
    kept, _ = _batch_inputs("categorical", target, logits, None, 255, frozenset({"labels"}))
    assert kept.tolist() == [156, 3]


def test_numpy_string_ids_and_names_are_stored_as_plain_str():
    """`MetricSpec(np.str_("accuracy"), name=np.str_("acc"))` passed the
    slug checks but its state() failed yaml.safe_dump (#515)."""
    import yaml

    spec = MetricSpec(np.str_("accuracy"), version=np.int64(1), name=np.str_("acc"))
    assert (type(spec.id), type(spec.name), type(spec.label)) == (str, str, str)
    assert spec == MetricSpec("accuracy", name="acc") and spec.state() == {
        "id": "accuracy",
        "version": 1,
        "name": "acc",
    }
    assert all(type(spec.state()[key]) is str for key in ("id", "name"))
    yaml.safe_dump(spec.state())
    assert spec.definition.id == "accuracy"  # the registry lookup is unchanged


# --- #518: the Bernoulli nll in float64, never saturated by a float32 sigmoid --------------------


def _identity_bernoulli(width: int, task: bool = True) -> NNModel:
    """A BCE model whose input rows are its logits, with or without a TaskSpec."""
    model = NNModel(
        net_params=NNParams(
            input_dim=width, output_dim=width, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.BINARY_CROSS_ENTROPY,
            task=TaskSpec.multilabel(width) if task else None,
        ),
    )
    with torch.no_grad():
        model.net.layers[0].weight.copy_(torch.eye(width))
        model.net.layers[0].bias.zero_()
    return model


def _bernoulli_nll_terms(logits: torch.Tensor, target: float) -> np.ndarray:
    """The named nll's per-element terms, as _MetricSet feeds them."""
    from nnx._probability import nll_terms
    from nnx.monitors import _batch_inputs

    truth, inputs = _batch_inputs(
        "bernoulli", torch.full_like(logits, target), logits, None, None, frozenset({"probabilities"})
    )
    return nll_terms(truth, inputs["probabilities"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bernoulli_nll_terms_match_bce_with_logits_up_to_the_floor(dtype):
    """A float32 sigmoid is exactly 1 from a logit of 16.6355, so a target-0
    term read the 1e-12 floor (27.631 nats) instead of its logit (#518)."""
    logits = (torch.arange(-27600, 27601, dtype=torch.float64) / 1000).to(dtype)
    for target in (0.0, 1.0):
        exact = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.double(), torch.full_like(logits, target, dtype=torch.float64), reduction="none"
        )
        np.testing.assert_allclose(_bernoulli_nll_terms(logits, target), exact.numpy(), rtol=0, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bernoulli_nll_false_positives_and_negatives_score_alike(dtype):
    from nnx._probability import NLL_EPSILON

    x = (torch.arange(0, 30001, dtype=torch.float64) / 1000).to(dtype)
    positive, negative = _bernoulli_nll_terms(x, 0.0), _bernoulli_nll_terms(-x, 1.0)
    np.testing.assert_allclose(positive, negative, rtol=0, atol=1e-3)
    floor = -math.log(NLL_EPSILON)
    beyond = torch.nn.functional.softplus(x.double()).numpy() > floor
    assert beyond.any() and (positive[beyond] == floor).all() and (negative[beyond] == floor).all()


@pytest.mark.parametrize("task", [True, False], ids=["task", "no-task"])
def test_evaluate_scores_a_confident_false_positive_by_its_logit(task):
    record = _identity_bernoulli(1, task=task).evaluate(
        [(torch.tensor([[17.0]]), torch.tensor([[0.0]]))], metrics=(MetricSpec("nll"),)
    )
    assert record.metrics["nll"] == pytest.approx(17.0, abs=1e-3)
    assert record.metrics["nll"] == pytest.approx(record.loss, abs=1e-3)


def test_a_multilabel_batch_nll_equals_its_bce_with_logits_loss():
    X = torch.tensor([[17.5, 3.0, -2.0, 1.0], [2.0, -1.5, 0.5, 4.0]])
    Y = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    record = _identity_bernoulli(4).evaluate([(X, Y)], metrics=(MetricSpec("nll"), MetricSpec("brier")))
    bce = torch.nn.functional.binary_cross_entropy_with_logits(X.double(), Y.double()).item()
    assert bce == pytest.approx(2.413668, abs=1e-6)
    assert record.metrics["nll"] == pytest.approx(bce, abs=1e-4)
    assert record.metrics["nll"] == pytest.approx(record.loss, abs=1e-4)
    # brier keeps the same input: within 1e-6 of the float32 sigmoid it read before.
    float32_brier = float(np.mean((torch.sigmoid(X).numpy().astype(np.float64) - Y.numpy()) ** 2))
    assert record.metrics["brier"] == pytest.approx(float32_brier, abs=1e-6)
