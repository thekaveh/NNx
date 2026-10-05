"""Opt-in live smoke of ``nnx.decisions.JevProvider`` against the TypeSafe API (FEAT-010).

Sends one Choice, one Boolean and one Score about one text to a pinned Jev
model — ``jev-1.13.0``, never an alias — and prints, as JSON, the model the
service resolved, the SDK version, the request id, token usage and wall time.

It spends real API quota, so nothing runs it automatically (CI has no
credentials): run it by hand with an API key in the environment::

    TYPESAFE_API_KEY=... python scripts/smoke_jev_live.py [--model jev-1.13.0]

Requires the ``jev`` extra: ``pip install "thekaveh-nnx[jev]"``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

PINNED_MODEL = "jev-1.13.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=PINNED_MODEL, help=f"a pinned model version (default {PINNED_MODEL})")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"jev-\d+\.\d+\.\d+", args.model):
        print(f"--model must be a pinned version such as {PINNED_MODEL}, not {args.model!r}", file=sys.stderr)
        return 2
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        print("TYPESAFE_API_KEY is not set: this opt-in smoke needs real credentials.", file=sys.stderr)
        return 2

    import typesafe_sdk

    from nnx.decisions import Boolean, Choice, JevProvider, Score

    questions = [
        Choice("What is this message about?", (("t-bill", "billing"), ("t-ship", "shipping"), ("t-other", "other"))),
        Boolean("Does the customer need an answer today?"),
        Score("How severe is the problem?", (("s0", "minor"), ("s1", "serious"), ("s2", "critical"))),
    ]
    started = time.perf_counter()
    with JevProvider(model=args.model) as provider:
        (topic,), (urgent,), (severity,) = provider.decide_many(questions, ["I was charged twice for my order."])
    seconds = time.perf_counter() - started
    raw = topic.raw
    report = {
        "requested_model": args.model,
        "resolved_model": raw.get("model"),
        "sdk_version": typesafe_sdk.__version__,
        "request_id": raw.get("request_id"),
        "usage": raw.get("usage"),
        "seconds": round(seconds, 3),
        "answers": {
            "topic": dict(topic.distribution),
            "p_urgent": urgent.p_true,
            "severity": dict(severity.distribution),
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
