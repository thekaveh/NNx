"""An opt-in Result at two fallible boundaries, next to try/except (FEAT-025).

``nnx.result`` adds ``Ok`` / ``Err`` and three boundary wrappers; every
existing exception API is unchanged. This example writes each flow both
ways so the difference in branching is visible:

  1. **A decision request from plain data.** A batch of user-supplied
     requests (one with duplicate option ids, one with an empty prompt) is
     validated and only the valid ones are asked. With exceptions, each
     request needs a ``try`` / ``except InvalidDecisionRequest``; with
     ``validate_decision_request_result(...).bind(decide_result(...))``,
     invalid requests flow through as ``Err`` values with a code and a
     cause — and the provider is never called for them (zero calls).
  2. **A run bundle someone handed over.** A missing directory, a
     malformed one and a valid bundle are inspected. With exceptions the
     caller distinguishes ``FileNotFoundError`` from ``BundleError`` in two
     ``except`` clauses; with ``inspect_bundle_result(...)`` the code
     (``artifact_missing`` / ``bundle_invalid``) is data, and
     ``.map(...)`` / ``.recover(...)`` compose a summary. Inspection reads
     the manifest and JSON records only — it never unpickles a checkpoint.

Offline, CPU, deterministic.

Run:
    python examples/result_boundaries.py

The bounded ``result_boundaries_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory.
"""

from __future__ import annotations

import json
import os
import tempfile

import torch

from nnx.decisions import InvalidDecisionRequest, question_from_state, validate_response
from nnx.result import Err, Ok, decide_result, inspect_bundle_result, validate_decision_request_result

REQUESTS = [
    {"kind": "choice", "prompt": "Topic?", "options": [("t-sport", "sports"), ("t-econ", "the economy")]},
    {"kind": "choice", "prompt": "Topic?", "options": [("dup", "sports"), ("dup", "the economy")]},
    {"kind": "boolean", "prompt": "", "options": None},
]


class CountingProvider:
    """A deterministic local provider that counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, question, inputs):
        self.calls += 1
        n = len(question.option_ids)
        return [validate_response(question, {o: 1.0 / n for o in question.option_ids}) for _ in inputs]


def _as_state(request: dict) -> dict:
    key = "levels" if request["kind"] == "score" else "options"
    state = {"kind": request["kind"], "prompt": request["prompt"]}
    if request["options"] is not None:
        state[key] = request["options"]
    return state


def requests_with_exceptions(provider: CountingProvider) -> list[str]:
    outcomes = []
    for request in REQUESTS:
        try:
            question = question_from_state(_as_state(request))
        except InvalidDecisionRequest as error:
            outcomes.append(f"rejected: {error}")
            continue
        outcomes.append(f"answered {len(provider.decide(question, ['a text']))}")
    return outcomes


def requests_with_results(provider: CountingProvider) -> list[str]:
    outcomes = []
    for request in REQUESTS:
        result = validate_decision_request_result(request["kind"], request["prompt"], request["options"]).bind(
            lambda question: decide_result(provider, question, ["a text"])
        )
        outcomes.append(f"answered {len(result.value)}" if isinstance(result, Ok) else f"rejected: {result.error.code}")
    return outcomes


def _bundle(root: str) -> str:
    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNOptimParams, NNParams, NNTrainParams
    from nnx.bundles import export_bundle
    from nnx.tasks import TaskSpec

    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(2)
        ),
    )
    X, y = torch.randn(8, 4), torch.randint(0, 2, (8,))
    # The run (runs/<id>/) and its bundle live in the temporary directory,
    # not the caller's working directory.
    previous = os.getcwd()
    os.chdir(root)
    try:
        run = model.train(
            params=NNTrainParams(
                n_epochs=1, train_loader=[(X, y)], optim=NNOptimParams.builder().adam(max_lr=0.01).build()
            )
        )
        export_bundle(run.id, "bundle")
    finally:
        os.chdir(previous)
    return os.path.join(root, "bundle")


def bundle_summaries(paths: list[str]) -> tuple[list[str], list[str]]:
    from nnx.bundles import BundleError, inspect_bundle

    with_exceptions = []
    for path in paths:
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with_exceptions.append(f"capability {inspect_bundle(path).capability}")
        except FileNotFoundError:
            with_exceptions.append("unavailable: missing")
        except BundleError:
            with_exceptions.append("unavailable: invalid")

    with_results = [
        inspect_bundle_result(path)
        .map(lambda info: f"capability {info.capability}")
        .recover(
            lambda error: Ok("unavailable: missing" if error.code == "artifact_missing" else "unavailable: invalid")
        )
        .unwrap()
        for path in paths
    ]
    return with_exceptions, with_results


def result_boundaries_workflow() -> dict:
    """Run both flows both ways and check they agree."""
    by_exceptions, by_results = CountingProvider(), CountingProvider()
    exception_outcomes = requests_with_exceptions(by_exceptions)
    result_outcomes = requests_with_results(by_results)
    # Same verdict for every request both ways (the exception style keeps the message, Result the code).
    assert [o.split(":")[0] for o in exception_outcomes] == [o.split(":")[0] for o in result_outcomes]
    assert exception_outcomes[0] == result_outcomes[0] == "answered 1"
    assert by_results.calls == 1  # only the valid request reached the provider
    assert result_outcomes[1:] == ["rejected: invalid_decision_request"] * 2

    with tempfile.TemporaryDirectory() as root:
        good = _bundle(root)
        broken = os.path.join(root, "broken")
        os.makedirs(broken)
        with open(os.path.join(broken, "bundle.json"), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        paths = [good, os.path.join(root, "missing"), broken]
        with_exceptions, with_results = bundle_summaries(paths)
    assert with_exceptions == with_results
    missing = inspect_bundle_result("definitely/not/here")
    assert isinstance(missing, Err) and missing.error.code == "artifact_missing"
    return {
        "requests": result_outcomes,
        "provider_calls": by_results.calls,
        "bundles": with_results,
    }


def main() -> None:
    print(json.dumps(result_boundaries_workflow(), indent=2))


if __name__ == "__main__":
    main()
