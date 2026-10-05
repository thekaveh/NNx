"""FEAT-025: the Result boundary wrappers convert exactly what they declare."""

from __future__ import annotations

import os
import pickle

import pytest
import torch

import nnx.bundles as bundles
from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNOptimParams, NNParams, NNTrainParams
from nnx.abstention import AbstentionPolicy
from nnx.bundles import export_bundle
from nnx.decisions import (
    Boolean,
    Choice,
    ChoiceResult,
    InvalidDecisionRequest,
    ProviderFailure,
    UnsupportedCapability,
    validate_response,
)
from nnx.result import (
    BoundaryError,
    Err,
    Ok,
    decide_result,
    inspect_bundle_result,
    validate_decision_request_result,
)
from nnx.tasks import TaskSpec

TOPIC = Choice("Topic?", (("t-sport", "sports"), ("t-econ", "the economy")))


class Spy:
    """A provider recording calls; answers with fixed distributions or raises."""

    def __init__(self, raises=None, distributions=((0.9, 0.1),)):
        self.calls = 0
        self.raises = raises
        self.distributions = distributions

    def decide(self, question, inputs):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return [
            validate_response(question, dict(zip(question.option_ids, row, strict=True)), provider="spy")
            for row in self.distributions
        ]


# ---------------- validate_decision_request_result ----------------


def test_a_valid_request_builds_the_typed_question():
    result = validate_decision_request_result("choice", "Topic?", [("t-sport", "sports"), ("t-econ", "the economy")])
    assert result == Ok(TOPIC)
    assert validate_decision_request_result("boolean", "Spam?") == Ok(Boolean("Spam?"))


@pytest.mark.parametrize(
    ("kind", "prompt", "options"),
    [
        ("choice", "Topic?", [("a", "x"), ("a", "y")]),  # duplicate option ids
        ("choice", "Topic?", [("a", "x")]),  # too few options
        ("boolean", "", None),  # empty prompt
        ("verdict", "Topic?", None),  # unknown kind
        ("choice", "Topic?", 5),  # options that are not a sequence
        ("score", "Level?", 3),
        ("choice", "Topic?", None),
    ],
)
def test_a_malformed_request_is_an_err_and_nothing_reaches_a_provider(kind, prompt, options):
    provider = Spy()
    result = validate_decision_request_result(kind, prompt, options).bind(
        lambda question: decide_result(provider, question, ["text"])
    )
    assert result.is_err and provider.calls == 0
    error = result.error
    assert isinstance(error, BoundaryError) and error.code == "invalid_decision_request"
    assert error.where == "request" and error.context["kind"] == kind
    assert isinstance(error.cause, InvalidDecisionRequest)


def test_only_invalid_decision_request_is_converted(monkeypatch):
    import nnx.decisions.schema as schema

    def broken(state):
        raise KeyError("a bug, not a request error")

    monkeypatch.setattr(schema, "question_from_state", broken)
    with pytest.raises(KeyError):
        validate_decision_request_result("boolean", "Spam?")


# ---------------- decide_result ----------------


def test_a_provider_failure_is_an_err_keeping_its_cause_and_request_id():
    failure = ProviderFailure("backend down")
    failure.request_id = "req-7"  # type: ignore[attr-defined]
    result = decide_result(Spy(raises=failure), TOPIC, ["text"])
    assert result.is_err
    assert (result.error.code, result.error.where, result.error.cause) == (
        "provider_failure",
        "provider.decide",
        failure,
    )
    assert result.error.context["request_id"] == "req-7"
    assert result.error.context["question"] == TOPIC.digest()
    refused = decide_result(Spy(raises=UnsupportedCapability("no text")), TOPIC, ["text"])
    assert refused.error.code == "unsupported"


def test_other_provider_exceptions_propagate():
    with pytest.raises(RuntimeError, match="a bug"):
        decide_result(Spy(raises=RuntimeError("a bug")), TOPIC, ["text"])


def test_abstention_is_a_success_with_probabilities_and_reasons():
    policy = AbstentionPolicy(
        "max_probability",
        0.8,
        labels=("t-sport", "t-econ"),
        model_id="m",
        tuning_split_id="v",
        input_field="probabilities",
    )
    provider = Spy(distributions=((0.9, 0.1), (0.6, 0.4)))
    result = decide_result(provider, TOPIC, ["confident", "unsure"], policy=policy, model_id="m")
    assert result.is_ok
    accepted, abstained = result.unwrap()
    assert accepted.accepted and abstained.abstained
    assert abstained.outcome.reason and abstained.outcome.distribution == (("t-sport", 0.6), ("t-econ", 0.4))
    # composition keeps the decisions intact
    labels = result.map(lambda rows: [row.outcome.status for row in rows]).unwrap()
    assert labels == ["accepted", "abstained"]
    plain = decide_result(Spy(), TOPIC, ["x"])
    assert isinstance(plain.unwrap()[0], ChoiceResult)
    for model_id in (None, 7):
        with pytest.raises(TypeError, match="model_id"):
            decide_result(Spy(), TOPIC, ["x"], policy=policy, model_id=model_id)  # type: ignore[arg-type]


# ---------------- inspect_bundle_result ----------------


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(2)
        ),
    )
    X, y = torch.randn(8, 4), torch.randint(0, 2, (8,))
    run = model.train(
        params=NNTrainParams(n_epochs=1, train_loader=[(X, y)], optim=NNOptimParams.builder().adam(max_lr=0.01).build())
    )
    export_bundle(run.id, "bundle")
    return "bundle"


@pytest.fixture
def no_unpickling(monkeypatch):
    """Fail on any unpickling, tensor read, dynamic import or registry lookup."""
    import importlib

    import nnx.models as models

    def refuse(what):
        def trap(*args, **kwargs):
            raise AssertionError(f"{what} was called")

        return trap

    monkeypatch.setattr(torch, "load", refuse("torch.load"))
    monkeypatch.setattr(pickle, "load", refuse("pickle.load"))
    monkeypatch.setattr(pickle, "loads", refuse("pickle.loads"))
    monkeypatch.setattr(bundles, "_load_tensors", refuse("a tensor read"))
    monkeypatch.setattr(importlib, "import_module", refuse("importlib.import_module"))
    for name in ("build_module", "resolve_model_factory"):
        monkeypatch.setattr(models, name, refuse(f"nnx.models.{name}"))


def test_inspecting_a_bundle_never_reaches_torch_load(bundle, no_unpickling):
    result = inspect_bundle_result(bundle)
    assert result.is_ok and result.unwrap().capability in {"inference", "resume"}


def test_a_missing_bundle_is_an_artifact_error(tmp_path, no_unpickling):
    result = inspect_bundle_result(tmp_path / "nowhere")
    assert result.error.code == "artifact_missing" and result.error.where.endswith("nowhere")
    assert isinstance(result.error.cause, FileNotFoundError)
    (tmp_path / "file").write_text("x", encoding="utf-8")
    through_a_file = inspect_bundle_result(tmp_path / "file" / "sub")
    assert through_a_file.error.code == "artifact_missing"
    assert isinstance(through_a_file.error.cause, NotADirectoryError)


def test_an_unusable_path_is_invalid_not_raised(tmp_path, no_unpickling):
    assert inspect_bundle_result(str(tmp_path / ("x" * 300))).error.code in {"bundle_invalid", "artifact_missing"}
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    assert inspect_bundle_result(loop / "x").error.code in {"bundle_invalid", "artifact_missing"}


def test_a_malformed_bundle_keeps_code_path_and_cause_through_composition(bundle, no_unpickling):
    with open(os.path.join(bundle, "bundle.json"), "w", encoding="utf-8") as handle:
        handle.write("{not json")
    result = inspect_bundle_result(bundle)
    assert result.is_err and result.error.code == "bundle_invalid"
    cause = result.error.cause
    assert isinstance(cause, bundles.BundleError)
    relabelled = result.map_error(lambda error: BoundaryError(error.code, error.where, {"seen": True}, error.cause))
    assert (relabelled.error.code, relabelled.error.where, relabelled.error.cause) == ("bundle_invalid", bundle, cause)
    passed_on = result.recover(lambda error: Err(error) if error.code == "bundle_invalid" else Ok(None))
    assert passed_on.error is result.error


def test_a_path_that_is_not_a_bundle_is_invalid(tmp_path, no_unpickling):
    (tmp_path / "empty").mkdir()
    assert inspect_bundle_result(tmp_path / "empty").error.code == "bundle_invalid"
    (tmp_path / "file").write_text("not a bundle", encoding="utf-8")
    assert inspect_bundle_result(tmp_path / "file").error.code == "bundle_invalid"


@pytest.mark.parametrize(
    "manifest",
    ['{"format": "nnx.bundle/999"}', "[]", '{"files": 3}', "null", ""],
    ids=["future-format", "list", "wrong-types", "null", "empty"],
)
def test_malformed_manifests_never_reach_a_loader(bundle, no_unpickling, manifest):
    with open(os.path.join(bundle, "bundle.json"), "w", encoding="utf-8") as handle:
        handle.write(manifest)
    result = inspect_bundle_result(bundle)
    assert result.is_err and result.error.code == "bundle_invalid"
    assert isinstance(result.error.cause, bundles.BundleError)


def test_boundary_errors_pickle_copy_and_hash():
    import copy

    failure = ProviderFailure("backend down")
    errors = [
        decide_result(Spy(raises=failure), TOPIC, ["text"]),
        validate_decision_request_result("choice", "Topic?", [("a", "x"), ("a", "y")]),
    ]
    for result in errors:
        for clone in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
            assert isinstance(clone, Err) and clone == result
            assert clone.error.context == result.error.context
            assert type(clone.error.cause) is type(result.error.cause)
        assert hash(result.error) == hash(pickle.loads(pickle.dumps(result.error)))
        with pytest.raises(TypeError):
            result.error.context["code"] = "edited"  # type: ignore[index]


def test_the_exception_apis_are_unchanged(tmp_path):
    with pytest.raises(bundles.BundleError):
        bundles.inspect_bundle(tmp_path / "nowhere")
    with pytest.raises(InvalidDecisionRequest):
        Choice("Topic?", (("a", "x"), ("a", "y")))
