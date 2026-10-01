"""FEAT-024: deferred decision jobs — batching, continuations, fail-fast,
async cancellation and the effect boundary."""

from __future__ import annotations

import asyncio
import pickle
import random
import time

import numpy as np
import pytest
import torch

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams, TaskSpec
from nnx.abstention import AbstentionPolicy
from nnx.decisions import (
    Boolean,
    Capabilities,
    Choice,
    DecisionJob,
    FixedHeadProvider,
    Follow,
    InvalidJob,
    JobFailed,
    JobLimitExceeded,
    JobTimeout,
    Limits,
    Score,
    validate_response,
)

Job = DecisionJob


class TextProvider:
    """A deterministic text provider: Choice → mass on the option whose
    description shares a word with the text; Boolean → 0.9 when the prompt's
    last word is in the text. Batches questions when ``batching``."""

    def __init__(self, *, batching: bool = True, max_questions=None, fail_on=None, delay: float = 0.0):
        self.batching = batching
        self.max_questions = max_questions
        self.fail_on = set(fail_on or ())
        self.delay = delay
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self.closed = False
        if not batching:
            self.decide_many = None  # type: ignore[assignment]

    def capabilities(self) -> Capabilities:
        return Capabilities(
            primitives=frozenset({"choice", "boolean"}), modalities=frozenset({"text"}), dynamic_labels=True
        )

    def close(self) -> None:  # a job must never call this
        self.closed = True

    def _answer(self, question, texts):
        if question.prompt in self.fail_on:
            raise RuntimeError(f"backend failure on {question.prompt!r}")
        results = []
        for text in texts:
            words = set(text.lower().split())
            if isinstance(question, Boolean):
                p = 0.9 if question.prompt.split()[-1].lower() in words else 0.2
                results.append(validate_response(question, p))
                continue
            weights = [1.0 + 4.0 * bool(words & set(o.description.lower().split())) for o in question.options]
            total = sum(weights)
            results.append(
                validate_response(question, {o.id: w / total for o, w in zip(question.options, weights, strict=True)})
            )
        return results

    def decide(self, question, texts):
        self.calls.append(((question.prompt,), tuple(texts)))
        if self.delay:
            time.sleep(self.delay)
        return self._answer(question, texts)

    def decide_many(self, questions, texts):
        self.calls.append((tuple(q.prompt for q in questions), tuple(texts)))
        if self.delay:
            time.sleep(self.delay)
        return [self._answer(q, texts) for q in questions]


TEXTS = ["the striker scored a goal", "rates rose at the bank"]
TOPIC = Choice("Topic?", (("t-sport", "goal match"), ("t-econ", "rates bank")))


def _questions(n: int) -> dict[str, Job]:
    return {f"q{i}": Job.ask(Boolean(f"Mentions word{i}"), id=f"q{i}") for i in range(n)}


# --- AC1: building is pure; run / arun are the effect boundaries ---------------------------------------------


def test_zero_calls_before_run_one_after():
    provider = TextProvider()
    mapped: list[int] = []
    rng_before = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state())
    job = Job.collect({"topic": Job.ask(TOPIC, id="topic"), "goal": Job.ask(Boolean("Mentions goal"), id="goal")})
    job = job.map(lambda answers: mapped.append(1) or answers)
    assert provider.calls == [] and mapped == []
    assert random.getstate() == rng_before[0] and np.array_equal(np.random.get_state()[1], rng_before[1])
    assert torch.equal(torch.get_rng_state(), rng_before[2])
    result = job.run(provider, state=TEXTS)
    assert len(provider.calls) == 1 and result.calls == 1 and mapped == [1]
    assert result.value["topic"][0].top == "t-sport" and result.value["goal"][0].p_true == pytest.approx(0.9)


def test_provider_usable_after_run():
    provider = TextProvider()
    Job.ask(TOPIC, id="topic").run(provider, state=TEXTS)
    with pytest.raises(JobFailed):
        Job.ask(Boolean("boom"), id="boom").run(TextProvider(fail_on={"boom"}), state=TEXTS)
    assert not provider.closed
    assert provider.decide(TOPIC, TEXTS)[1].top == "t-econ"  # still usable directly


# --- AC2: batching ---------------------------------------------------------------------------------------


def test_three_calls_at_cap_two():
    provider = TextProvider()
    result = Job.collect(_questions(5)).run(provider, state=TEXTS, limits=Limits(max_questions=2))
    assert result.calls == 3 == len(provider.calls)  # ceil(5 / 2)
    assert [prompts for prompts, _ in provider.calls] == [
        ("Mentions word0", "Mentions word1"),
        ("Mentions word2", "Mentions word3"),
        ("Mentions word4",),
    ]
    assert list(result.outcomes) == [f"q{i}" for i in range(5)]


def test_five_calls_at_cap_one():
    provider = TextProvider(batching=False)  # answers one question per call
    result = Job.collect(_questions(5)).run(provider, state=TEXTS)
    assert result.calls == 5 and [prompts for prompts, _ in provider.calls] == [
        (f"Mentions word{i}",) for i in range(5)
    ]
    assert Job.collect(_questions(5)).run(TextProvider(max_questions=4), state=TEXTS).calls == 2  # the provider's cap


def test_collect_key_order():
    jobs = {"zeta": Job.ask(TOPIC, id="a"), "alpha": Job.ask(Boolean("Mentions goal"), id="b")}
    result = Job.collect(jobs).run(TextProvider(), state=TEXTS)
    assert list(result.value) == ["zeta", "alpha"] and list(result.outcomes) == ["a", "b"]
    assert Job.collect(jobs).question_ids() == ("a", "b")


def test_duplicate_ids_rejected():
    provider = TextProvider()
    twice = Job.collect({"x": Job.ask(TOPIC, id="same"), "y": Job.ask(Boolean("Mentions goal"), id="same")})
    with pytest.raises(InvalidJob, match="duplicate question id 'same'"):
        twice.run(provider, state=TEXTS)
    with pytest.raises(InvalidJob, match="cannot be served"):  # an unsupported shape (Score)
        Job.ask(Score("Severity?", (("lo", "low"), ("hi", "high"))), id="s").run(provider, state=TEXTS)
    with pytest.raises(InvalidJob, match="cannot be enforced"):  # no count_tokens
        Job.ask(TOPIC, id="t").run(provider, state=TEXTS, limits=Limits(max_tokens=100))
    assert provider.calls == []


def test_a_token_cap_splits_calls_when_the_provider_counts_tokens():
    class Counting(TextProvider):
        def count_tokens(self, questions, texts):
            return 10 * len(questions)

    provider = Counting()
    result = Job.collect(_questions(5)).run(provider, state=TEXTS, limits=Limits(max_tokens=25))
    assert result.calls == 3 and [len(p) for p, _ in provider.calls] == [2, 2, 1]
    with pytest.raises(InvalidJob, match="alone exceeds max_tokens=5"):
        Job.collect(_questions(2)).run(Counting(), state=TEXTS, limits=Limits(max_tokens=5))


# --- AC3: dependent continuations ------------------------------------------------------------------------


def test_continuation_carries_answer_state():
    provider = TextProvider()
    ran: list[str] = []

    def route(results):
        ran.append(results[0].top)
        sport = [text for text, result in zip(TEXTS, results, strict=True) if result.top == "t-sport"]
        return Follow(Job.ask(Boolean("Mentions striker"), id="striker"), state=sport)

    job = Job.collect(
        {"topic": Job.ask(TOPIC, id="topic").then(route), "rates": Job.ask(Boolean("Mentions rates"), id="rates")}
    )
    result = job.run(provider, state=TEXTS)
    assert ran == ["t-sport"]  # once, after its prerequisite succeeded
    assert [texts for _, texts in provider.calls] == [tuple(TEXTS), ("the striker scored a goal",)]
    assert provider.calls[0][0] == ("Topic?", "Mentions rates")  # the sibling batched with the prerequisite
    assert [r.p_true for r in result.value["topic"]] == pytest.approx([0.9])
    assert [r.p_true for r in result.value["rates"]] == pytest.approx([0.2, 0.9])  # untouched by the continuation


def test_depth_limit():
    def deeper(level):
        def step(_results):
            return Follow(
                Job.ask(Boolean(f"Mentions level{level}"), id=f"l{level}").then(deeper(level + 1)), state=TEXTS
            )

        return step

    provider = TextProvider()
    with pytest.raises(JobLimitExceeded, match="max_depth=3") as caught:
        Job.ask(TOPIC, id="root").then(deeper(1)).run(provider, state=TEXTS, limits=Limits(max_depth=3))
    assert set(caught.value.completed) == {"root", "l1", "l2", "l3"} and len(provider.calls) == 4
    with pytest.raises(JobLimitExceeded, match="max_requests=2"):
        Job.ask(TOPIC, id="root").then(deeper(1)).run(TextProvider(), state=TEXTS, limits=Limits(max_requests=2))


# --- AC4: fail-fast ---------------------------------------------------------------------------------------


def test_mid_batch_failure_typed_error():
    provider = TextProvider(fail_on={"Mentions word2"})
    with pytest.raises(JobFailed) as caught:
        Job.collect(_questions(5)).run(provider, state=TEXTS, limits=Limits(max_questions=2))
    error = caught.value
    assert set(error.completed) == {"q0", "q1"} and error.failed == ("q2", "q3") and error.skipped == ("q4",)
    assert isinstance(error.__cause__, RuntimeError) and len(provider.calls) == 2  # nothing scheduled after it
    assert all(outcome.kind == "answered" for outcome in error.completed.values())


def test_continuation_skipped_after_failure():
    continued: list[int] = []
    provider = TextProvider(fail_on={"Mentions boom"})
    job = Job.collect(
        {
            "safe": Job.ask(TOPIC, id="safe").then(
                lambda r: continued.append(1) or Follow(Job.ask(TOPIC, id="next"), TEXTS)
            ),
            "boom": Job.ask(Boolean("Mentions boom"), id="boom"),
        }
    )
    with pytest.raises(JobFailed) as caught:
        job.run(provider, state=TEXTS, limits=Limits(max_questions=1))
    assert continued == [] and caught.value.failed == ("boom",) and "safe" in caught.value.completed
    assert [p for p, _ in provider.calls] == [("Topic?",), ("Mentions boom",)]  # "safe" was not re-run


# --- AC5: async, cancellation, outcome kinds, fixed-head semantics ----------------------------------------


class SlowAsyncProvider(TextProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.started = asyncio.Event()
        self.active = 0
        self.peak = 0

    async def adecide_many(self, questions, texts):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.set()
        try:
            await asyncio.sleep(0.05)
            return self.decide_many(questions, texts)
        finally:
            self.active -= 1


def test_async_runs_match_sync_runs_and_bound_concurrency():
    job = Job.collect(_questions(6))
    sync = job.run(TextProvider(), state=TEXTS, limits=Limits(max_questions=2))
    provider = SlowAsyncProvider()
    result = asyncio.run(job.arun(provider, state=TEXTS, limits=Limits(max_questions=2, max_concurrency=2)))
    assert result.value.keys() == sync.value.keys() and result.calls == 3
    assert [r.p_true for r in result.value["q0"]] == [r.p_true for r in sync.value["q0"]]
    assert provider.peak == 2


def test_async_outcomes_keep_scheduling_order_and_sync_providers_run_one_call_at_a_time():
    import threading

    class Uneven(TextProvider):
        async def adecide_many(self, questions, texts):
            await asyncio.sleep(0.05 if questions[0].prompt == "Mentions word0" else 0.0)  # q0 finishes last
            return self.decide_many(questions, texts)

    result = asyncio.run(
        Job.collect(_questions(3)).arun(Uneven(), state=TEXTS, limits=Limits(max_questions=1, max_concurrency=3))
    )
    assert list(result.outcomes) == ["q0", "q1", "q2"] and list(result.value) == ["q0", "q1", "q2"]

    class Guarded(TextProvider):  # synchronous only, and not thread-safe
        def __init__(self):
            super().__init__(batching=False)
            self.lock = threading.Lock()
            self.overlaps = 0

        def decide(self, question, texts):
            if not self.lock.acquire(blocking=False):
                self.overlaps += 1
                raise AssertionError("two worker threads called the provider at once")
            try:
                time.sleep(0.02)
                return super().decide(question, texts)
            finally:
                self.lock.release()

    guarded = Guarded()
    result = asyncio.run(Job.collect(_questions(4)).arun(guarded, state=TEXTS, limits=Limits(max_concurrency=4)))
    assert result.calls == 4 and guarded.overlaps == 0 and list(result.outcomes) == [f"q{i}" for i in range(4)]


def test_async_cancellation():
    async def scenario():
        provider = SlowAsyncProvider()
        cancel = asyncio.Event()
        run = asyncio.ensure_future(
            Job.collect(_questions(4)).arun(provider, state=TEXTS, limits=Limits(max_questions=1), cancel=cancel)
        )
        await provider.started.wait()
        cancel.set()
        result = await run
        assert result.status == "cancelled" and result.value is None
        kinds = {qid: (o.kind, o.sent) for qid, o in result.outcomes.items()}
        assert kinds["q0"] == ("cancelled", True)  # sent, and never presented as rolled back
        assert all(kinds[q] == ("cancelled", False) for q in ("q1", "q2", "q3"))
        # Task cancellation: job-owned tasks are cleaned up and the error propagates.
        before = len(asyncio.all_tasks())
        task = asyncio.ensure_future(
            Job.collect(_questions(3)).arun(provider, state=TEXTS, limits=Limits(max_questions=1))
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert len(asyncio.all_tasks()) == before
        assert provider.decide(TOPIC, TEXTS)  # the borrowed provider stays usable

    asyncio.run(scenario())


def test_timeouts_are_explicit():
    with pytest.raises(JobTimeout):
        Job.collect(_questions(3)).run(
            TextProvider(batching=False, delay=0.05), state=TEXTS, limits=Limits(timeout=0.06)
        )

    async def slow():
        await Job.collect(_questions(2)).arun(SlowAsyncProvider(), state=TEXTS, limits=Limits(timeout=0.01))

    with pytest.raises(JobTimeout, match="not rolled back"):
        asyncio.run(slow())


def test_outcome_kinds():
    policy = AbstentionPolicy(
        "max_probability", 0.7, labels=("t-sport", "t-econ"), model_id="text-v1", tuning_split_id="v"
    )
    abstaining = Job.ask(TOPIC, id="topic", policy=policy, model_id="text-v1")
    result = abstaining.run(TextProvider(), state=["the striker scored a goal", "nothing relevant here"])
    outcome = result.outcomes["topic"]
    assert outcome.kind == "answered" and [row.kind for row in outcome.rows] == ["answered", "abstained"]
    assert outcome.abstained and result.value[1].outcome.status == "abstained"
    with pytest.raises(JobFailed) as caught:
        Job.ask(Boolean("boom"), id="b").run(TextProvider(fail_on={"boom"}), state=TEXTS)
    assert caught.value.failed == ("b",)
    kinds = {"answered", "failed", "skipped", "cancelled"}
    assert {o.kind for o in result.outcomes.values()} <= kinds


def _species_head() -> NNModel:
    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=3, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
            task=TaskSpec.categorical(3, labels=("cat", "dog", "fox")),
        ),
    )
    with torch.no_grad():
        model.net.layers[0].weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]))
        model.net.layers[0].bias.zero_()
    return model


def test_fixed_head_capability_check():
    provider = FixedHeadProvider(_species_head(), option_map={"sp-cat": "cat", "sp-dog": "dog", "sp-fox": "fox"})
    photos = torch.tensor([[1.5, 0.0], [0.5, 2.0]])
    species = Choice("Which animal?", (("sp-fox", "A fox"), ("sp-cat", "A cat"), ("sp-dog", "A dog")))
    cow = Choice("Which animal?", (("sp-cat", "A cat"), ("sp-dog", "A dog"), ("sp-cow", "A cow")))
    with pytest.raises(InvalidJob, match="cannot be served"):
        Job.collect({"ok": Job.ask(species, id="ok"), "cow": Job.ask(cow, id="cow")}).run(provider, state=photos)
    assert provider.model_calls == 0  # refused before any call
    result = Job.ask(species, id="species").run(provider, state=photos)
    assert [r.top for r in result.value] == ["sp-cat", "sp-dog"] and provider.model_calls == 1  # its label mapping

    def broken(*args, **kwargs):
        raise RuntimeError("head failure")

    provider.model.predict_proba = broken  # type: ignore[method-assign]
    with pytest.raises(JobFailed):
        Job.ask(species, id="again").run(provider, state=photos)
    assert provider.model_calls == 2  # one attempt: the job never retries


# --- AC6: laws and serialisation ----------------------------------------------------------------------------


def test_map_laws():
    job = Job.collect({"topic": Job.ask(TOPIC, id="topic"), "goal": Job.ask(Boolean("Mentions goal"), id="goal")})

    def f(answers):
        return {key: [type(r).__name__ for r in value] for key, value in answers.items()}

    def g(names):
        return sorted(names)

    plain = job.run(TextProvider(), state=TEXTS).value
    assert job.map(lambda x: x).run(TextProvider(), state=TEXTS).value == plain  # identity
    assert (
        job.map(f).map(g).run(TextProvider(), state=TEXTS).value
        == job.map(lambda x: g(f(x))).run(TextProvider(), state=TEXTS).value
    )  # composition


def test_serialisation_rejected():
    pure = Job.collect({"topic": Job.ask(TOPIC, id="topic"), "goal": Job.ask(Boolean("Mentions goal"), id="goal")})
    restored = pickle.loads(pickle.dumps(pure))
    assert restored.state() == pure.state() and restored.question_ids() == ("topic", "goal")
    for runtime in (pure.map(len), pure.then(lambda value: Follow(pure, TEXTS))):
        with pytest.raises(TypeError, match="cannot be serialised"):
            pickle.dumps(runtime)
    with pytest.raises(AttributeError, match="immutable"):
        pure._node = None  # type: ignore[misc]


def test_logits_chain_is_untouched():
    from nnx.generation import LogitsChain

    assert not hasattr(LogitsChain, "ask") and not hasattr(LogitsChain.builder(), "then")


# --- review hardening -------------------------------------------------------------------------------------------


def test_a_synchronous_provider_gets_no_call_after_a_failure_and_unsent_requests_say_so():
    provider = TextProvider(batching=False, delay=0.02, fail_on={"Mentions word0"})
    with pytest.raises(JobFailed) as caught:
        asyncio.run(Job.collect(_questions(4)).arun(provider, state=TEXTS, limits=Limits(max_concurrency=4)))
    assert [prompts for prompts, _ in provider.calls] == [("Mentions word0",)]  # nothing after the failure
    outcomes = caught.value.outcomes
    assert outcomes["q0"].kind == "failed" and outcomes["q0"].sent
    assert all((outcomes[q].kind, outcomes[q].sent) == ("skipped", False) for q in ("q1", "q2", "q3"))

    async def cancelled_mid_call():
        cancel = asyncio.Event()
        slow = TextProvider(batching=False, delay=0.1)
        run = asyncio.ensure_future(
            Job.collect(_questions(3)).arun(slow, state=TEXTS, limits=Limits(max_concurrency=3), cancel=cancel)
        )
        await asyncio.sleep(0.03)
        cancel.set()
        return await run, slow

    result, slow = asyncio.run(cancelled_mid_call())
    assert result.calls == 1 == len(slow.calls)
    assert [(o.kind, o.sent) for o in result.outcomes.values()] == [
        ("cancelled", True),
        ("cancelled", False),
        ("cancelled", False),
    ]


def test_cancelling_the_task_while_a_thread_runs_is_delivered_after_the_thread():
    async def scenario():
        provider = TextProvider(batching=False, delay=0.3)
        provider.finished = 0

        original = provider.decide

        def decide(question, texts):
            try:
                return original(question, texts)
            finally:
                provider.finished += 1

        provider.decide = decide
        cancel = asyncio.Event()
        task = asyncio.ensure_future(Job.ask(TOPIC, id="t").arun(provider, state=TEXTS, cancel=cancel))
        await asyncio.sleep(0.02)
        cancel.set()
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.finished == 1  # the borrowed provider is no longer in use
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(Job.ask(TOPIC, id="t").arun(provider, state=TEXTS), timeout=0.05)
        assert provider.finished == 2

    asyncio.run(scenario())


def test_task_cancellation_cleans_up_the_cancel_event_waiter():
    async def scenario():
        before = len(asyncio.all_tasks())
        task = asyncio.ensure_future(
            Job.collect(_questions(2)).arun(
                SlowAsyncProvider(), state=TEXTS, limits=Limits(max_questions=1), cancel=asyncio.Event()
            )
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert len(asyncio.all_tasks()) == before

    asyncio.run(scenario())


def test_a_success_finishing_with_a_failure_is_kept():
    class Together(TextProvider):
        async def adecide_many(self, questions, texts):
            await asyncio.sleep(0.01)
            return self.decide_many(questions, texts)

    provider = Together(fail_on={"Mentions word0"})
    with pytest.raises(JobFailed) as caught:
        asyncio.run(
            Job.collect(_questions(2)).arun(provider, state=TEXTS, limits=Limits(max_questions=1, max_concurrency=2))
        )
    assert caught.value.failed == ("q0",) and set(caught.value.completed) == {"q1"}


def test_a_limit_lets_calls_in_flight_finish_and_reports_every_outcome():
    provider = SlowAsyncProvider()
    with pytest.raises(JobLimitExceeded, match="max_requests=3") as caught:
        asyncio.run(
            Job.collect(_questions(4)).arun(
                provider, state=TEXTS, limits=Limits(max_questions=1, max_concurrency=2, max_requests=3)
            )
        )
    kinds = {q: (o.kind, o.sent) for q, o in caught.value.outcomes.items()}
    assert kinds == {
        "q0": ("answered", True),
        "q1": ("answered", True),
        "q2": ("answered", True),
        "q3": ("skipped", False),
    }
    assert len(provider.calls) == 3


def test_an_abstention_policy_that_does_not_fit_its_question_is_refused_at_ask():
    policy = AbstentionPolicy(
        "max_probability", 0.7, labels=("t-sport", "t-econ"), model_id="text-v1", tuning_split_id="v"
    )
    with pytest.raises(InvalidJob, match="model_id"):
        Job.ask(TOPIC, id="t", policy=policy, model_id="other-model")
    reordered = Choice("Topic?", (("t-econ", "rates bank"), ("t-sport", "goal match")))
    with pytest.raises(InvalidJob, match="labels"):
        Job.ask(reordered, id="t", policy=policy, model_id="text-v1")


def test_continuations_returning_the_same_state_object_batch_together():
    shared = ["the keeper saved it", "rates and the bank"]
    provider = TextProvider()
    job = Job.collect(
        {
            "a": Job.ask(Boolean("Mentions goal"), id="a").then(
                lambda _: Follow(Job.ask(Boolean("Mentions keeper"), id="a2"), state=shared)
            ),
            "b": Job.ask(Boolean("Mentions bank"), id="b").then(
                lambda _: Follow(Job.ask(Boolean("Mentions rates"), id="b2"), state=shared)
            ),
        }
    )
    result = job.run(provider, state=TEXTS)
    assert result.calls == 2 and provider.calls[1] == (("Mentions keeper", "Mentions rates"), tuple(shared))


def test_an_answer_count_mismatch_is_a_typed_failure_with_skips():
    class Short(TextProvider):
        def decide_many(self, questions, texts):
            return super().decide_many(questions, texts)[:1]

    with pytest.raises(JobFailed) as caught:
        Job.collect(_questions(4)).run(Short(), state=TEXTS, limits=Limits(max_questions=2))
    error = caught.value
    assert error.failed == ("q0", "q1") and error.skipped == ("q2", "q3")
    assert error.outcomes["q0"].kind == "failed" and error.outcomes["q2"].kind == "skipped"
    assert type(error.__cause__).__name__ == "InvalidDecisionResponse"


def test_job_errors_pickle_with_their_outcomes_and_concurrency_is_required():
    with pytest.raises(JobFailed) as caught:
        Job.collect(_questions(3)).run(
            TextProvider(fail_on={"Mentions word1"}), state=TEXTS, limits=Limits(max_questions=1)
        )
    restored = pickle.loads(pickle.dumps(caught.value))
    assert type(restored) is JobFailed and str(restored) == str(caught.value)
    assert restored.failed == ("q1",) and restored.skipped == ("q2",) and set(restored.completed) == {"q0"}
    with pytest.raises(InvalidJob, match="max_concurrency"):
        Limits(max_concurrency=None)  # type: ignore[arg-type]
