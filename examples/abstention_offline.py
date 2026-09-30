"""Abstention policies and risk-coverage evaluation, offline (FEAT-008).

A classifier that must answer every row cannot say "I'm not sure". An
``AbstentionPolicy`` accepts a row only when its top probability (or the
margin between its top two) reaches a threshold, and abstains otherwise:

  1. **Select on validation only.** Candidate thresholds are evaluated on
     the validation split (``split_id="val"``, never the test split) and
     the highest-coverage one whose *empirical* selective risk stays within
     the ceiling is chosen; every candidate's counts are reported, ending in
     the curve-only accept-none endpoint.
  2. **JSON round trip.** The policy records its probability field,
     threshold, label order, model id and tuning split; it is saved to and
     reloaded from a temporary directory unchanged.
  3. **Apply to test.** Accepted sample ids, coverage over every row and
     selective risk over the accepted rows. Abstained rows keep their
     prediction, distribution and reason; a hard-label report sees only the
     accepted rows.
  4. **An all-rejected set** reports risk *unavailable*, never 0.

Fully offline, CPU only: the probabilities are supplied arrays.

Run:
    python examples/abstention_offline.py

The bounded ``abstention_offline_workflow()`` helper is executed by
``tests/test_examples_smoke.py -k abstention_offline`` in a temporary
working directory.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from nnx.abstention import AbstentionPolicy, select_threshold

LABELS = ("cat", "dog", "fox")
MODEL_ID = "pet-classifier:run-3:BEST"

# Validation: (probabilities, target). Confident rows are right, unsure ones often wrong.
VALIDATION = np.array(
    [
        [0.92, 0.05, 0.03],
        [0.10, 0.85, 0.05],
        [0.05, 0.15, 0.80],
        [0.45, 0.40, 0.15],
        [0.38, 0.34, 0.28],
        [0.70, 0.20, 0.10],
        [0.30, 0.35, 0.35],
        [0.20, 0.72, 0.08],
    ]
)
VALIDATION_TARGETS = np.array([0, 1, 2, 1, 2, 0, 1, 1])
TEST = np.array([[0.88, 0.07, 0.05], [0.40, 0.35, 0.25], [0.06, 0.14, 0.80], [0.36, 0.33, 0.31]])
TEST_TARGETS = np.array([0, 2, 2, 0])
TEST_IDS = [1001, 1002, 1003, 1004]


def abstention_offline_workflow() -> dict:
    # 1. Select a threshold on the validation split only.
    selection = select_threshold(
        VALIDATION,
        VALIDATION_TARGETS,
        kind="max_probability",
        risk_ceiling=0.1,
        split_id="val",
        test_split_id="test",
        labels=LABELS,
        model_id=MODEL_ID,
    )
    policy = selection.require()
    assert policy.tuning_split_id == "val" and selection.basis == "empirical"
    assert selection.points[-1].status == "accept_none" and not selection.points[-1].deployable

    # 2. Round-trip the policy through JSON in a temporary directory.
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "policy.json")
        policy.save(path)
        reloaded = AbstentionPolicy.load(path)
    assert reloaded == policy and reloaded.id == policy.id

    # 3. Apply it to the test rows.
    result = reloaded.apply(TEST, labels=LABELS, model_id=MODEL_ID, sample_ids=TEST_IDS)
    report = result.report(TEST_TARGETS)
    rows = result.accepted_rows(TEST_TARGETS)  # what a hard-label report may receive
    assert rows.sample_ids.tolist() == result.accepted_ids.tolist()
    abstained = [outcome for outcome in result.outcomes() if not outcome.accepted]
    assert all(outcome.reason == "below_probability_threshold" for outcome in abstained)

    # 4. A set where every row is rejected: risk is unavailable, never 0.
    unsure = reloaded.apply(TEST[[1, 3]], labels=LABELS, model_id=MODEL_ID, sample_ids=[1002, 1004])
    none = unsure.report(TEST_TARGETS[[1, 3]])
    assert none.status == "no_accepted" and none.risk is None

    summary = {
        "threshold": policy.threshold,
        "candidates": [(point.threshold, point.accepted, point.incorrect) for point in selection.points],
        "accepted_ids": result.accepted_ids.tolist(),
        "abstained_ids": result.abstained_ids.tolist(),
        "coverage": report.coverage,
        "risk": report.risk,
        "all_rejected": {"status": none.status, "risk": none.risk},
    }
    print(f"threshold chosen on validation (risk ceiling 0.1, empirical): {policy.threshold}")
    for point in selection.points:
        shown = "accept-none (curve only)" if point.threshold is None else f"{point.threshold:.2f}"
        print(f"  candidate {shown}: accepted {point.accepted}/{point.total}, incorrect {point.incorrect}")
    print(f"accepted test ids: {summary['accepted_ids']}; abstained: {summary['abstained_ids']}")
    print(report.summary())
    for outcome in abstained:
        print(
            f"  abstained {outcome.sample_id}: predicted {outcome.prediction!r} at {outcome.score:.2f} ({outcome.reason})"
        )
    print(f"all-rejected set: {none.summary()}")
    return summary


def main() -> None:
    abstention_offline_workflow()


if __name__ == "__main__":
    main()
