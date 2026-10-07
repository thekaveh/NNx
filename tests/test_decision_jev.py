"""FEAT-010: typed decisions on Jev models through the TypeSafe SDK.

A fake ``system_one`` recorder returns real SDK responses parsed from canned
HTTP responses, so no test needs credentials or the network.
"""

from __future__ import annotations

import asyncio
import json
import pickle

import pytest

typesafe_sdk = pytest.importorskip("typesafe_sdk")
httpx2 = pytest.importorskip("httpx2")

from nnx.decisions import (  # noqa: E402
    Boolean,
    BooleanResult,
    Choice,
    ChoiceResult,
    DecisionJob,
    InvalidDecisionRequest,
    InvalidDecisionResponse,
    JevAuthenticationError,
    JevError,
    JevMalformedResponse,
    JevProvider,
    JevRateLimited,
    JevTimeout,
    JobFailed,
    ProviderFailure,
    Score,
    ScoreResult,
    UnsupportedCapability,
)
from nnx.decisions import jev as jev_module  # noqa: E402

TOPIC = Choice("Topic?", (("t-sport", "sports"), ("t-econ", "the economy"), ("t-tech", "technology")))
SPAM = Boolean("Is this spam?")
URGENCY = Score("How urgent?", (("u0", "can wait"), ("u1", "this week"), ("u2", "today")))


def _response(answers, *, model="jev-1.13.0", usage=None, request_id="req-1", status=200, headers=None):
    body = {"model": model, "usage": usage if usage is not None else {"input_tokens": 12, "output_tokens": 3}}
    body["answers"] = answers
    hdrs = {"x-typesafe-request-id": request_id} if request_id else {}
    hdrs.update(headers or {})
    request = httpx2.Request("POST", "https://api.typesafe.ai/v1/systemone")
    response = httpx2.Response(status, json=body, headers=hdrs, request=request)
    return typesafe_sdk.SystemOneResponse.from_http_response(response)


def _api_error(status, *, request_id="req-err", headers=None):
    hdrs = {"x-typesafe-request-id": request_id}
    hdrs.update(headers or {})
    request = httpx2.Request("POST", "https://api.typesafe.ai/v1/systemone")
    response = httpx2.Response(status, json={"error": "nope"}, headers=hdrs, request=request)
    with pytest.raises(typesafe_sdk.TypeSafeError) as caught:
        typesafe_sdk.SystemOneResponse.from_http_response(response)
    return caught.value


ANSWERS = {
    # Labels arrive in another order than the question's options.
    "q0": {
        "type": "choice",
        "choice": "technology",
        "confidence": 0.7,
        "probabilities": {"technology": 0.6, "sports": 0.1, "the economy": 0.3},
    },
    "q1": {"type": "noul", "noul": 0.25},
    "q2": {
        "type": "score",
        "score": 1.3,
        "confidence": 0.55,
        "legend": {"0": "can wait", "1": "this week", "2": "today"},
        "probabilities": {"2": 0.4, "0": 0.1, "1": 0.5},
    },
}


BY_TYPE = {ANSWERS[name]["type"]: ANSWERS[name] for name in ANSWERS}


def _answer_by_type(state, questions, kwargs):
    """Answer every requested question with the canned answer of its type."""
    return _response({name: BY_TYPE[question["type"]] for name, question in questions.items()})


class Recorder:
    """A fake TypeSafe client: records every call and close, returns or
    raises what it is told to."""

    def __init__(self, outcome=None):
        self.calls = []
        self.closed = 0
        self.outcome = outcome if outcome is not None else _answer_by_type

    def system_one(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        result = self.outcome(state, questions, kwargs)
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self):
        self.closed += 1


class AsyncRecorder(Recorder):
    async def system_one(self, state, questions, **kwargs):  # type: ignore[override]
        await asyncio.sleep(0)
        return Recorder.system_one(self, state, questions, **kwargs)

    async def aclose(self):
        self.closed += 1


def test_one_call_returns_aligned_choice_boolean_and_score_results():
    client = Recorder()
    provider = JevProvider(client, model="jev-latest")
    topic, spam, urgency = provider.decide_many([TOPIC, SPAM, URGENCY], ["Rates rose again today."])

    assert len(client.calls) == 1  # every question of one input shares one request
    state, wire, kwargs = client.calls[0]
    assert state == "Rates rose again today."
    assert kwargs == {"model": "jev-latest"}
    assert wire == {
        "q0": {
            "type": "choice",
            "instructions": "Topic?",
            "criteria": {"sports": None, "the economy": None, "technology": None},
        },
        "q1": {"type": "noul", "instructions": "Is this spam?"},
        "q2": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "this week", "today"]},
    }
    assert "t-sport" not in json.dumps(wire)  # ids are bookkeeping, never sent

    (choice,), (boolean,), (score,) = topic, spam, urgency
    assert isinstance(choice, ChoiceResult) and choice.question_digest == TOPIC.digest()
    assert choice.distribution == (("t-sport", 0.1), ("t-econ", 0.3), ("t-tech", 0.6))  # the question's order
    assert isinstance(boolean, BooleanResult) and boolean.p_true == 0.25
    assert isinstance(score, ScoreResult)
    assert score.distribution == (("u0", 0.1), ("u1", 0.5), ("u2", 0.4))  # levels in order
    assert score.vendor_score == 1.3
    for result in (choice, boolean, score):
        assert result.provider == "jev"
        assert result.raw["model"] == "jev-1.13.0"  # the resolved model, not the alias requested
        assert result.raw["requested_model"] == "jev-latest"
        assert result.raw["request_id"] == "req-1"
        assert result.raw["usage"] == {"input_tokens": 12, "output_tokens": 3}


def test_decide_answers_one_question_per_input_in_input_order():
    client = Recorder(lambda state, questions, kwargs: _response({"q0": {"type": "noul", "noul": len(state) / 10}}))
    results = JevProvider(client).decide(SPAM, ["a", "abc"])
    assert [r.p_true for r in results] == [0.1, 0.3]
    assert [call[0] for call in client.calls] == ["a", "abc"]
    assert client.calls[0][2] == {}  # no model: the client's default


def test_confidence_stays_provider_confidence_and_boolean_gets_none():
    topic, spam, urgency = JevProvider(Recorder()).decide_many([TOPIC, SPAM, URGENCY], ["x"])
    assert topic[0].raw["provider_confidence"] == 0.7
    assert urgency[0].raw["provider_confidence"] == 0.55
    assert "provider_confidence" not in spam[0].raw
    # Confidence is metadata, never a probability of the answer being right.
    assert topic[0].probabilities()["t-tech"] == 0.6 and topic[0].raw["provider_confidence"] != 0.6


def test_serialised_metadata_holds_only_plain_reported_fields():
    usage_unreported = Recorder(lambda s, q, k: _response(ANSWERS, usage={"input_tokens": 5}, request_id=None))
    provider = JevProvider(usage_unreported)
    (topic,), (spam,), _ = provider.decide_many([TOPIC, SPAM, URGENCY], ["x"])
    assert topic.raw == {
        "provider": "jev",
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 5},
        "provider_confidence": 0.7,
    }
    assert set(spam.raw) == {"provider", "model", "usage"}  # absent stays absent
    for raw in (topic.raw, spam.raw):
        json.dumps(raw)  # plain JSON
        assert not {"client", "headers", "api_key", "authorization"} & {key.lower() for key in raw}
    assert provider.record() == {"provider": "jev", "backend": "typesafe-sdk", "sdk_range": ">=0.7,<0.8", "model": None}
    assert "secret" not in repr(JevProvider(api_key="secret-key"))


@pytest.mark.parametrize(
    ("make_error", "expected", "request_id"),
    [
        (lambda: typesafe_sdk.TypeSafeAPITimeoutError(5.0), JevTimeout, None),
        (lambda: _api_error(401), JevAuthenticationError, "req-err"),
        (lambda: _api_error(403), JevAuthenticationError, "req-err"),
        (lambda: _api_error(429, headers={"retry-after-ms": "250"}), JevRateLimited, "req-err"),
        (lambda: _api_error(500), JevError, "req-err"),
    ],
    ids=["timeout", "auth-401", "auth-403", "rate-limit", "server"],
)
def test_failures_become_distinct_errors_keeping_cause_and_request_id(make_error, expected, request_id):
    error = make_error()
    client = Recorder(lambda s, q, k: error)
    with pytest.raises(expected) as caught:
        JevProvider(client).decide(SPAM, ["x"])
    assert type(caught.value) is expected
    assert isinstance(caught.value, ProviderFailure)
    assert caught.value.__cause__ is error
    assert caught.value.request_id == request_id
    assert len(client.calls) == 1  # the adapter adds no retry above the SDK's RetryPolicy
    if expected is JevRateLimited:
        assert caught.value.retry_after_ms == 250
    if request_id is not None:
        assert request_id in str(caught.value)


def test_malformed_responses_are_typed_and_keep_the_request_id():
    sdk_invalid = Recorder(lambda s, q, k: _response({"q0": {"type": "noul"}}, request_id="req-bad"))
    with pytest.raises(JevMalformedResponse) as caught:
        JevProvider(sdk_invalid).decide(SPAM, ["x"])
    assert isinstance(caught.value.__cause__, typesafe_sdk.TypeSafeAPIResponseValidationError)
    assert caught.value.request_id == "req-bad"

    def drift(s, q, k):
        bad = {
            "type": "choice",
            "choice": "sports",
            "confidence": 0.9,
            "probabilities": {"sports": 0.9, "the economy": 0.3, "technology": 0.1},
        }
        return _response({"q0": bad}, request_id="req-sum")

    for outcome, reason in (
        (drift, "sum"),
        (lambda s, q, k: _response({}, request_id="req-none"), "no answer"),
        (lambda s, q, k: _response({"q0": {"type": "noul", "noul": 0.5}}, request_id="req-kind"), "expected a choice"),
    ):
        client = Recorder(outcome)
        with pytest.raises(JevMalformedResponse, match=reason) as caught:
            JevProvider(client).decide(TOPIC, ["x"])
        assert isinstance(caught.value, InvalidDecisionResponse)
        assert isinstance(caught.value.__cause__, InvalidDecisionResponse)
        assert caught.value.request_id.startswith("req-")
        assert len(client.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        JevError("APIError: 500", request_id="req-9"),
        JevTimeout("Timeout: slow"),
        JevAuthenticationError("Auth: no", request_id="req-9"),
        JevRateLimited("RateLimit: slow down", request_id="req-9", retry_after_ms=10.0),
        JevMalformedResponse("Invalid: bad", request_id="req-9"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_errors_pickle_and_copy_with_their_request_id(error):
    import copy

    for restored in (pickle.loads(pickle.dumps(error)), copy.copy(error), copy.deepcopy(error)):
        assert type(restored) is type(error)
        assert restored.__dict__ == error.__dict__ and str(restored) == str(error)


def test_a_failed_job_keeps_a_typed_timeout_across_pickling():
    error = typesafe_sdk.TypeSafeAPITimeoutError(5.0)
    job = DecisionJob.ask(SPAM, id="spam")
    with pytest.raises(JobFailed) as caught:
        job.run(JevProvider(Recorder(lambda s, q, k: error)), state=["x"])
    restored = pickle.loads(pickle.dumps(caught.value))
    assert type(restored.outcomes["spam"].error) is JevTimeout


def test_requests_are_checked_before_any_call():
    client = Recorder()
    provider = JevProvider(client, max_batch=1)
    with pytest.raises(UnsupportedCapability, match="texts"):
        provider.decide(SPAM, [1, 2])
    with pytest.raises(UnsupportedCapability, match="max_batch"):
        provider.decide(SPAM, ["a", "b"])
    twin = Choice("Pick", (("a", "same"), ("b", "same")))
    with pytest.raises(UnsupportedCapability, match="repeated"):
        provider.decide(twin, ["a"])
    with pytest.raises(InvalidDecisionRequest, match="injected client"):
        JevProvider(client, api_key="k")
    assert client.calls == []


def test_injected_clients_stay_caller_owned():
    sync_client, async_client = Recorder(), AsyncRecorder()
    with JevProvider(sync_client, async_client=async_client) as provider:
        provider.decide(SPAM, ["x"])

    async def run():
        async with JevProvider(sync_client, async_client=async_client) as provider:
            await provider.adecide(SPAM, ["x"])

    asyncio.run(run())
    assert sync_client.closed == 0 and async_client.closed == 0


def test_a_sync_only_injection_is_sync_only_and_a_job_runs_it_one_call_at_a_time(built):
    import threading
    import time

    from nnx.decisions import Limits

    active, peak, lock = [0], [0], threading.Lock()

    def slow(state, questions, kwargs):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with lock:
            active[0] -= 1
        return _answer_by_type(state, questions, kwargs)

    client = Recorder(slow)
    provider = JevProvider(client)
    assert provider.adecide is None and provider.adecide_many is None  # no async path to race on
    asks = {f"q{i}": DecisionJob.ask(Boolean(f"Question {i}?"), id=f"q{i}") for i in range(4)}
    result = asyncio.run(DecisionJob.collect(asks).arun(provider, state=["y"], limits=Limits(max_concurrency=4)))
    assert {outcome.kind for outcome in result.outcomes.values()} == {"answered"}
    assert peak[0] == 1  # the caller's client is never called from two threads at once
    assert built == [] and client.closed == 0


def test_an_async_only_injection_refuses_sync_calls(built):
    client = AsyncRecorder()
    provider = JevProvider(async_client=client)
    with pytest.raises(InvalidDecisionRequest, match="only an async client"):
        provider.decide(SPAM, ["x"])
    asyncio.run(provider.adecide(SPAM, ["x"]))
    assert built == [] and len(client.calls) == 1


def test_a_client_that_cannot_be_built_fails_typed(monkeypatch):
    def refuse(**settings):
        raise typesafe_sdk.TypeSafeError("No API key was provided.")

    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", refuse)
    with pytest.raises(JevError, match="No API key") as caught:
        JevProvider().decide(SPAM, ["x"])
    assert type(caught.value.__cause__) is typesafe_sdk.TypeSafeError


@pytest.fixture
def built(monkeypatch):
    """Clients the provider builds itself, recorded."""
    made = []

    class FakeSDK:
        TypeSafeError = typesafe_sdk.TypeSafeError

        @staticmethod
        def TypeSafeClient(**settings):
            client = Recorder()
            client.settings = settings
            made.append(client)
            return client

        @staticmethod
        def AsyncTypeSafeClient(**settings):
            client = AsyncRecorder()
            client.settings = settings
            made.append(client)
            return client

    monkeypatch.setattr(jev_module, "_sdk", lambda: FakeSDK)
    return made


def test_built_clients_close_on_exit_and_on_error(built):
    policy = typesafe_sdk.RetryPolicy(max_retries=3)
    with JevProvider(api_key="k", retry=policy) as provider:
        assert built == []  # nothing is built until the first call
        provider.decide(SPAM, ["x"])
    assert built[0].closed == 1
    assert built[0].settings == {"api_key": "k", "retry": policy}  # the SDK owns retries

    with pytest.raises(RuntimeError, match="boom"):
        with JevProvider() as provider:
            provider.decide(SPAM, ["x"])
            raise RuntimeError("boom")
    assert built[1].closed == 1


def test_built_async_clients_close_on_exit_error_and_cancellation(built):
    async def ok():
        async with JevProvider() as provider:
            await provider.adecide(SPAM, ["x"])

    async def error():
        async with JevProvider() as provider:
            await provider.adecide(SPAM, ["x"])
            raise RuntimeError("boom")

    async def cancelled():
        started = asyncio.Event()

        async def work():
            async with JevProvider() as provider:
                await provider.adecide(SPAM, ["x"])
                started.set()
                await asyncio.sleep(3600)

        task = asyncio.ensure_future(work())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(ok())
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(error())
    asyncio.run(cancelled())
    assert [client.closed for client in built] == [1, 1, 1]


def test_a_built_async_client_belongs_to_one_event_loop(built):
    provider = JevProvider()

    async def ask():
        async with provider:
            await provider.adecide(SPAM, ["x"])

    asyncio.run(ask())
    asyncio.run(ask())  # another loop gets its own client
    assert [client.closed for client in built] == [1, 1]

    # Used without 'async with': the next loop cannot await the old client
    # closed, so it is dropped with a warning — never awaited on a dead loop.
    unmanaged = JevProvider()
    asyncio.run(unmanaged.adecide(SPAM, ["x"]))
    with pytest.warns(ResourceWarning, match="another event loop"):
        asyncio.run(unmanaged.adecide(SPAM, ["x"]))
    with pytest.warns(ResourceWarning, match="another event loop"):
        asyncio.run(unmanaged.aclose())


def test_close_warns_when_a_built_async_client_is_still_open(built):
    provider = JevProvider()
    asyncio.run(provider.adecide(SPAM, ["x"]))
    with pytest.warns(ResourceWarning, match="aclose"):
        provider.close()


def test_a_failure_crossing_a_decision_job_keeps_its_reason_and_request_id():
    error = _api_error(429)
    client = Recorder(lambda s, q, k: error)
    job = DecisionJob.collect({"spam": DecisionJob.ask(SPAM, id="spam"), "topic": DecisionJob.ask(TOPIC, id="topic")})
    with pytest.raises(JobFailed) as caught:
        job.run(JevProvider(client), state=["one text"])
    cause = caught.value.__cause__
    assert isinstance(cause, JevRateLimited) and cause.request_id == "req-err"
    assert caught.value.outcomes["spam"].error is cause
    assert len(client.calls) == 1  # both questions shared the one request; nothing was retried


def test_a_failure_crossing_a_benchmark_keeps_its_reason_and_request_id():
    from nnx.decisions.benchmark import Budget, Sample, collect

    client = Recorder(lambda s, q, k: _api_error(429, request_id="req-bench"))
    samples = [Sample(f"s{i}", SPAM, f"text {i}", label=True) for i in range(3)]
    collection = collect(JevProvider(client), samples, provider_id="jev", budget=Budget(max_calls=5), batch_size=3)
    assert len(client.calls) == 1  # the first failure ends the batch; no retry
    assert {record.status for record in collection.records} == {"failed"}
    for record in collection.records:
        assert record.reason.startswith("JevRateLimited:")
        assert "request_id=req-bench" in record.reason


def test_the_live_smoke_is_opt_in_and_pins_a_version(monkeypatch, capsys):
    """CI has no credentials: without a key the live smoke refuses before
    building a client, and it pins a model version, never an alias."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_jev_live.py"
    spec = importlib.util.spec_from_file_location("smoke_jev_live", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", lambda **kwargs: pytest.fail("built a client"))
    assert module.main([]) == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err
    assert module.PINNED_MODEL == "jev-1.13.0"


def test_the_live_smoke_refuses_an_alias(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_jev_live.py"
    spec = importlib.util.spec_from_file_location("smoke_jev_live", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", lambda **kwargs: pytest.fail("built a client"))
    assert module.main(["--model", "jev-latest"]) == 2
    assert "pinned version" in capsys.readouterr().err


# --- FEAT-044: a JevProvider through the Result boundary ---------------------------------------------


@pytest.mark.parametrize(
    ("make_error", "expected", "request_id"),
    [
        (lambda: typesafe_sdk.TypeSafeAPITimeoutError(5.0), JevTimeout, None),
        (lambda: _api_error(401, request_id="req-auth"), JevAuthenticationError, "req-auth"),
        (lambda: _api_error(429, request_id="req-rate"), JevRateLimited, "req-rate"),
        (lambda: _api_error(500, request_id="req-down"), JevError, "req-down"),
    ],
    ids=["timeout", "auth", "rate-limit", "server"],
)
def test_decide_result_turns_a_jev_failure_into_a_provider_failure_err(make_error, expected, request_id):
    from nnx.result import decide_result

    client = Recorder(lambda s, q, k: make_error())
    result = decide_result(JevProvider(client), SPAM, ["x"])
    assert result.is_err
    error = result.error
    assert (error.code, error.where) == ("provider_failure", "provider.decide")
    assert type(error.cause) is expected and error.context["request_id"] == request_id
    assert error.context["provider"] == "JevProvider" and error.context["question"] == SPAM.digest()
    assert len(client.calls) == 1  # no retry above the SDK's own policy


def _drifted(s, q, k):
    bad = {
        "type": "choice",
        "choice": "sports",
        "confidence": 0.9,
        "probabilities": {"sports": 0.9, "the economy": 0.3, "technology": 0.1},
    }
    return _response({"q0": bad}, request_id="req-sum")


@pytest.mark.parametrize(
    ("outcome", "request_id"),
    [
        (lambda s, q, k: _response({"q0": {"type": "noul"}}, request_id="req-bad"), "req-bad"),
        (lambda s, q, k: _response({}, request_id="req-none"), "req-none"),
        (_drifted, "req-sum"),
        (lambda s, q, k: _response({"q0": {"type": "noul", "noul": 0.5}}, request_id="req-kind"), "req-kind"),
    ],
    ids=["sdk-validation", "no-answer", "sum-drift", "wrong-kind"],
)
def test_decide_result_reports_a_malformed_jev_answer_as_an_invalid_response(outcome, request_id):
    """A malformed answer is data the service sent, not an outage: it gets
    its own code, apart from ``provider_failure``."""
    from nnx.result import decide_result

    client = Recorder(outcome)
    result = decide_result(JevProvider(client), TOPIC, ["x"])
    assert result.is_err
    error = result.error
    assert (error.code, error.where) == ("invalid_decision_response", "provider.decide")
    assert isinstance(error.cause, JevMalformedResponse) and error.context["request_id"] == request_id
    assert len(client.calls) == 1


def test_decide_result_returns_aligned_jev_results():
    from nnx.result import decide_result

    result = decide_result(JevProvider(Recorder()), TOPIC, ["one", "two"])
    assert result.is_ok
    rows = result.unwrap()
    assert len(rows) == 2 and all(isinstance(row, ChoiceResult) for row in rows)
    assert all([option for option, _ in row.distribution] == list(TOPIC.option_ids) for row in rows)
