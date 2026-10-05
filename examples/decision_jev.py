"""Typed decisions on Jev models through the TypeSafe SDK (FEAT-010).

``nnx.decisions.JevProvider`` sends NNx's typed questions to a Jev model with
one ``system_one`` call per text — every question about that text in the same
request — and returns validated results in each question's own order:

  1. **One call, three primitives.** A ``Choice`` (keyed by its options'
     descriptions, never their ids), a ``Boolean`` (a Jev ``noul``) and a
     ``Score`` (its levels' descriptions, lowest first) about one text share
     one request; the answers come back as a ``ChoiceResult``,
     ``BooleanResult`` and ``ScoreResult``.
  2. **Metadata, not correctness.** Each result's ``raw`` records the model
     the service resolved (even when an alias was requested), the request id
     and token usage. A Choice's or Score's ``provider_confidence`` is Jev's
     own certainty — not a probability that the answer is right; a Boolean
     has none.
  3. **Typed failures, one attempt.** A rate-limited request surfaces as
     ``JevRateLimited`` with the service's request id; the SDK's
     ``RetryPolicy`` owns retries (disabled here), so the adapter sends it
     once.
  4. **Async.** The same questions through ``adecide_many`` and an async
     client.

Requires the ``jev`` extra: ``pip install "thekaveh-nnx[jev]"``. In practice
the provider builds its own client from ``TYPESAFE_API_KEY``. This example
runs fully offline: the clients are real SDK clients whose HTTP transport is
an ``httpx2.MockTransport`` answering deterministically, so no request leaves
the machine and every number is reproducible. No client is built at import.

Run:
    python examples/decision_jev.py

The bounded ``decision_jev_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory.
"""

from __future__ import annotations

import asyncio
import json

from nnx.decisions import Boolean, Choice, JevProvider, JevRateLimited, Score

TOPIC = Choice("What is this message about?", (("t-bill", "billing"), ("t-ship", "shipping"), ("t-other", "other")))
URGENT = Boolean("Does the customer need an answer today?")
SEVERITY = Score("How severe is the problem?", (("s0", "minor"), ("s1", "serious"), ("s2", "critical")))
TEXTS = ["I was charged twice for my order.", "Where is my parcel? It is a week late."]
KEYWORDS = {"billing": ("charged", "invoice", "refund"), "shipping": ("parcel", "late", "delivery")}


def _answer(question: dict, text: str) -> dict:
    """A deterministic stand-in for the model: keyword hits."""
    words = text.lower()
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.8 if "late" in words or "twice" in words else 0.2}
    if question["type"] == "choice":
        hits = {label: 1 + sum(word in words for word in KEYWORDS.get(label, ())) for label in question["criteria"]}
        total = sum(hits.values())
        probabilities = {label: hit / total for label, hit in hits.items()}
        best = max(probabilities, key=probabilities.__getitem__)
        return {"type": "choice", "choice": best, "confidence": probabilities[best], "probabilities": probabilities}
    levels = question["criteria"]
    weights = [1.0 + (i == len(levels) // 2) for i in range(len(levels))]  # the middle level is likelier
    probabilities = {str(i): w / sum(weights) for i, w in enumerate(weights)}
    score = sum(i * p for i, p in enumerate(probabilities.values()))
    return {
        "type": "score",
        "score": score,
        "confidence": 0.5,
        "legend": {str(i): level for i, level in enumerate(levels)},
        "probabilities": probabilities,
    }


def _service(*, rate_limited: bool = False):
    """An offline stand-in for the TypeSafe API, with a request counter."""
    import httpx2

    seen = {"requests": 0}

    def handle(request):
        seen["requests"] += 1
        request_id = f"req-{seen['requests']}"
        if rate_limited:
            return httpx2.Response(429, json={"error": "slow down"}, headers={"x-typesafe-request-id": request_id})
        body = json.loads(request.content)
        answers = {name: _answer(question, body["state"]) for name, question in body["questions"].items()}
        payload = {"model": "jev-1.13.0", "usage": {"input_tokens": 40, "output_tokens": 6}, "answers": answers}
        return httpx2.Response(200, json=payload, headers={"x-typesafe-request-id": request_id})

    return httpx2.MockTransport(handle), seen


def decision_jev_workflow() -> dict:
    """Run every step offline and return what it measured."""
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeClient

    no_retries = RetryPolicy(max_retries=0)

    # 1-2. One request per text holds all three questions.
    transport, seen = _service()
    with TypeSafeClient(api_key="offline-key", transport=transport, retry=no_retries) as client:
        provider = JevProvider(client, model="jev-latest")
        topics, urgent, severity = provider.decide_many([TOPIC, URGENT, SEVERITY], TEXTS)
    assert seen["requests"] == len(TEXTS)
    assert [result.top for result in topics] == ["t-bill", "t-ship"]
    assert [round(result.p_true, 2) for result in urgent] == [0.8, 0.8]
    first = topics[0].raw
    assert first["model"] == "jev-1.13.0" and first["requested_model"] == "jev-latest"
    assert "provider_confidence" not in urgent[0].raw
    assert severity[0].vendor_score is not None

    # 3. A rate limit is typed, keeps the request id and is sent once.
    transport, limited = _service(rate_limited=True)
    with TypeSafeClient(api_key="offline-key", transport=transport, retry=no_retries) as client:
        try:
            JevProvider(client).decide(URGENT, TEXTS[:1])
        except JevRateLimited as error:
            rate_limit = {"type": type(error).__name__, "request_id": error.request_id, "requests": limited["requests"]}
        else:  # pragma: no cover - the stand-in always refuses
            raise AssertionError("expected a rate limit")
    assert rate_limit == {"type": "JevRateLimited", "request_id": "req-1", "requests": 1}

    # 4. The same questions through the async client.
    async def run_async():
        transport, _ = _service()
        async with AsyncTypeSafeClient(api_key="offline-key", transport=transport, retry=no_retries) as client:
            provider = JevProvider(async_client=client, model="jev-latest")
            return await provider.adecide_many([TOPIC, URGENT], TEXTS)

    async_topics, _ = asyncio.run(run_async())
    assert [result.top for result in async_topics] == [result.top for result in topics]

    return {
        "topics": [result.top for result in topics],
        "p_urgent": [round(result.p_true, 3) for result in urgent],
        "severity_expected_index": [round(result.expected_index, 3) for result in severity],
        "provider_confidence": round(first["provider_confidence"], 3),
        "resolved_model": first["model"],
        "usage": first["usage"],
        "rate_limit": rate_limit,
    }


def main() -> None:
    print(json.dumps(decision_jev_workflow(), indent=2))


if __name__ == "__main__":
    main()
