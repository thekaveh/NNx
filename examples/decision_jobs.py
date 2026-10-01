"""Deferred decision jobs: batch independent questions, chain dependent
ones (FEAT-024).

A ``DecisionJob`` describes decision work without doing any: ``ask``,
``collect``, ``map`` and ``then`` build an immutable tree, and ``run`` /
``arun`` are the only places a provider is called. This example runs the
whole path locally, with no hosted provider:

  1. **Building is pure.** Five independent questions over the same texts
     are collected and mapped; the provider has been called zero times.
  2. **Fewest calls, stable order.** With a provider that answers up to two
     questions per call, the five questions run in ``ceil(5 / 2) = 3``
     calls, in the order they were collected; the result keeps the
     ``collect`` key order and the question ids.
  3. **A dependent question.** ``then`` maps the topic answers explicitly
     into follow-up state (only the sports texts) and asks one more
     question of those, once, after the topic question succeeded.
  4. **Fail-fast.** When a call fails, nothing more is scheduled:
     ``JobFailed`` carries the completed answers, names the failed and
     skipped questions, and the continuation waiting on the failed answer
     never runs.
  5. **Async.** ``arun`` runs the same job with up to two calls at once and
     returns the same answers; a ``cancel`` event set before the first call
     stops scheduling, and every question is reported cancelled and not
     sent (a request whose call had begun would be reported sent, never
     rolled back).

Fully offline, CPU only.

Run:
    python examples/decision_jobs.py

The bounded ``decision_jobs_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory.
"""

from __future__ import annotations

import asyncio

from nnx.decisions import (
    Boolean,
    Capabilities,
    Choice,
    DecisionJob,
    Follow,
    JobFailed,
    Limits,
    validate_response,
)

Job = DecisionJob
TEXTS = [
    "the striker scored a late goal",
    "the central bank raised rates",
    "the keeper saved a penalty",
]
TOPIC = Choice("What is the text about?", (("sport", "goal striker keeper penalty"), ("economy", "bank rates")))


class KeywordProvider:
    """A deterministic local text provider that answers up to
    ``max_questions`` questions per call (``decide_many``). A ``Boolean``
    "Mentions <word>" is 0.9 when the word is in the text, else 0.2; a
    ``Choice`` puts its mass on the options whose descriptions share a word
    with the text."""

    def __init__(self, *, max_questions: int = 2, fail_on: tuple[str, ...] = ()) -> None:
        self.max_questions = max_questions
        self.fail_on = set(fail_on)
        self.calls: list[tuple[str, ...]] = []

    def capabilities(self) -> Capabilities:
        return Capabilities(
            primitives=frozenset({"choice", "boolean"}), modalities=frozenset({"text"}), dynamic_labels=True
        )

    def _answer(self, question, texts):
        if question.prompt in self.fail_on:
            raise RuntimeError(f"the backend failed on {question.prompt!r}")
        results = []
        for text in texts:
            words = set(text.split())
            if isinstance(question, Boolean):
                results.append(validate_response(question, 0.9 if question.prompt.split()[-1] in words else 0.2))
                continue
            weights = [1.0 + 4.0 * len(words & set(option.description.split())) for option in question.options]
            total = sum(weights)
            results.append(
                validate_response(question, {o.id: w / total for o, w in zip(question.options, weights, strict=True)})
            )
        return results

    def decide(self, question, texts):
        return self.decide_many([question], texts)[0]

    def decide_many(self, questions, texts):
        self.calls.append(tuple(question.prompt for question in questions))
        return [self._answer(question, texts) for question in questions]

    async def adecide_many(self, questions, texts):  # used by arun
        await asyncio.sleep(0)  # a real provider would await its I/O here
        return self.decide_many(questions, texts)


def _mentions(*words: str) -> dict[str, DecisionJob]:
    return {word: Job.ask(Boolean(f"Mentions {word}"), id=word) for word in words}


def decision_jobs_workflow() -> None:
    # 1. Building is pure: no provider call, no callback, no RNG.
    provider = KeywordProvider(max_questions=2)
    flags = Job.collect(_mentions("goal", "bank", "penalty", "rates", "keeper"))
    flagged = flags.map(lambda answers: {word: [round(r.p_true, 1) for r in rows] for word, rows in answers.items()})
    assert provider.calls == []
    print("built:", flagged, "- provider calls so far:", len(provider.calls))

    # 2. Five independent questions at two per call: ceil(5 / 2) = 3 calls, in order.
    result = flagged.run(provider, state=TEXTS)
    print("calls:", provider.calls)
    assert result.calls == 3 == len(provider.calls)
    assert provider.calls == [
        ("Mentions goal", "Mentions bank"),
        ("Mentions penalty", "Mentions rates"),
        ("Mentions keeper",),
    ]
    assert list(result.value) == ["goal", "bank", "penalty", "rates", "keeper"]  # collect key order
    assert list(result.outcomes) == list(flags.question_ids())
    print("flags:", result.value)

    # 3. A dependent question: the topic answers become explicit follow-up state.
    def sports_only(topics):
        sport = [text for text, answer in zip(TEXTS, topics, strict=True) if answer.top == "sport"]
        return Follow(Job.ask(Boolean("Mentions keeper"), id="keeper"), state=sport)

    routed = Job.collect({"topic": Job.ask(TOPIC, id="topic").then(sports_only), **_mentions("rates")})
    provider = KeywordProvider(max_questions=2)
    result = routed.run(provider, state=TEXTS)
    # Round 1 batches the topic with its independent sibling; round 2 asks the follow-up.
    assert provider.calls == [("What is the text about?", "Mentions rates"), ("Mentions keeper",)]
    assert [round(r.p_true, 1) for r in result.value["topic"]] == [0.2, 0.9]  # over the two sports texts
    print("keeper, among sports texts:", [round(r.p_true, 1) for r in result.value["topic"]])

    # 4. Fail-fast: nothing more is scheduled, completed answers are kept, the
    #    continuation waiting on the failed answer never runs.
    continued: list[str] = []

    def follow_up(answers):
        continued.append("ran")
        return Follow(Job.ask(Boolean("Mentions striker"), id="striker"), state=TEXTS)

    failing = Job.collect(
        {
            "goal": Job.ask(Boolean("Mentions goal"), id="goal"),
            "bank": Job.ask(Boolean("Mentions bank"), id="bank").then(follow_up),
            "rates": Job.ask(Boolean("Mentions rates"), id="rates"),
        }
    )
    provider = KeywordProvider(max_questions=1, fail_on=("Mentions bank",))
    try:
        failing.run(provider, state=TEXTS)
        raise AssertionError("the failing call must raise JobFailed")
    except JobFailed as error:
        print(f"failed: {error.failed}, skipped: {error.skipped}, completed: {sorted(error.completed)}")
        assert error.failed == ("bank",) and error.skipped == ("rates",) and sorted(error.completed) == ["goal"]
    assert continued == [] and len(provider.calls) == 2  # the skipped continuation and question never ran

    # 5. Async: the same answers, two calls at once; a cancel event stops scheduling.
    async def run_async():
        answers = await flagged.arun(KeywordProvider(max_questions=1), state=TEXTS, limits=Limits(max_concurrency=2))
        cancel = asyncio.Event()
        cancel.set()  # cancelled before the first call: nothing is sent
        stopped = await flags.arun(KeywordProvider(), state=TEXTS, cancel=cancel)
        return answers, stopped

    answers, stopped = asyncio.run(run_async())
    assert answers.value == flagged.run(KeywordProvider(), state=TEXTS).value and answers.calls == 5
    assert stopped.status == "cancelled" and stopped.calls == 0
    assert all(outcome.kind == "cancelled" and not outcome.sent for outcome in stopped.outcomes.values())
    print("async:", answers.calls, "calls; cancelled run:", stopped.status, "with", stopped.calls, "calls")


def main() -> None:
    decision_jobs_workflow()


if __name__ == "__main__":
    main()
