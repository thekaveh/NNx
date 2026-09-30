"""Fitted temperature calibration from supplied arrays, offline (FEAT-007).

A classifier's softmax confidence is often too high or too low. Temperature
scaling fits one scalar ``T`` on a separate calibration split and serves
``softmax(logits / T)``: the argmax never changes, only the confidence. This
is unrelated to the *generation* temperature of
``nnx.generation.TemperatureScaling``, which is chosen for sampling and never
fitted.

  1. **Fit on the calibration split.** Supplied ``(N, 2)`` logits with 3 of
     every 4 rows correct at a logit margin of 2: the model says 0.88 where
     0.75 is right, so the fit finds ``T = 2 / ln 3 ≈ 1.82``. The calibration
     split id must differ from the training and test ids, and the calibrator
     records that ids cannot prove the rows are disjoint.
  2. **Save and reload.** The calibrator is JSON (label schema, model id, fit
     config, split ids) and reloads with identical probabilities.
  3. **Report on held-out data.** Before/after NLL, Brier and reliability
     bins on the test split, from the *reloaded* calibrator. A second
     held-out set on which the model was already right shows the report
     saying "worsened" instead of claiming success.
  4. **Identity checks.** Logits with reordered labels or from another model
     are refused before any transform, unless a named override records the
     mismatch.

Fully offline, CPU only, no model needed: the logits are supplied arrays.

Run:
    python examples/calibration_offline.py

The bounded ``calibration_offline_workflow()`` helper is executed by
``tests/test_examples_smoke.py -k calibration_offline`` in a temporary
working directory.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

from nnx.calibration import CalibrationMismatchError, TemperatureCalibrator, fit_temperature

LABELS = ("negative", "positive")
MODEL_ID = "sentiment-head:run-7:BEST"


def _split(n_blocks: int, margin: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """``n_blocks`` blocks of 4 rows: 3 correct and 1 wrong at the same margin."""
    logits = np.array([[margin, 0.0], [margin, 0.0], [margin, 0.0], [0.0, margin]] * n_blocks)
    targets = np.array([0, 0, 1, 1] * n_blocks)
    return logits, targets


def calibration_offline_workflow() -> dict:
    # 1. Fit on the calibration split only.
    calib_logits, calib_targets = _split(8)
    fit = fit_temperature(
        calib_logits,
        calib_targets,
        labels=LABELS,
        model_id=MODEL_ID,
        split_id="calibration",
        train_split_id="train",
        test_split_id="test",
    )
    calibrator = fit.require()
    assert math.isclose(calibrator.temperature, 2.0 / math.log(3.0), rel_tol=1e-8), calibrator.temperature
    assert calibrator.split["disjointness"] == "unverified"

    # 2. Save and reload in a temporary directory.
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "calibrator.json")
        calibrator.save(path)
        reloaded = TemperatureCalibrator.load(path)
    assert reloaded == calibrator and reloaded.id == calibrator.id

    # 3. Held-out reports from the reloaded calibrator.
    test_logits, test_targets = _split(4)
    report = reloaded.report(test_logits, test_targets, labels=LABELS, model_id=MODEL_ID, split_id="test", n_bins=5)
    assert report.improved and report.after.nll < report.before.nll
    served = reloaded.transform(test_logits, labels=LABELS, model_id=MODEL_ID)
    original = calibrator.transform(test_logits, labels=LABELS, model_id=MODEL_ID)
    assert np.array_equal(served.calibrated_probabilities, original.calibrated_probabilities)
    assert np.array_equal(served.decoded, test_logits.argmax(axis=1))  # the argmax never changes

    confident = np.array([[2.0, 0.0], [0.0, 2.0]] * 4)  # all correct: softening them hurts
    worse = reloaded.report(
        confident, np.array([0, 1] * 4), labels=LABELS, model_id=MODEL_ID, split_id="shift-check", n_bins=5
    )
    assert worse.outcome == "worsened" and not worse.improved

    # 4. Identity checks before any transform.
    try:
        reloaded.transform(test_logits, labels=LABELS[::-1], model_id=MODEL_ID)
    except CalibrationMismatchError as exc:
        refused = str(exc).split(".")[0]
    else:  # pragma: no cover - the check above always raises
        raise AssertionError("a reordered label list must be refused")
    overridden = reloaded.transform(test_logits, labels=LABELS, model_id="retrained-head", override="audit-2026-09")
    assert overridden.override is not None and "model_id" in overridden.override["mismatches"]

    summary = {
        "temperature": calibrator.temperature,
        "calibrator_id": calibrator.id,
        "before": {"nll": report.before.nll, "brier": report.before.brier, "ece": report.before.ece},
        "after": {"nll": report.after.nll, "brier": report.after.brier, "ece": report.after.ece},
        "outcome": report.outcome,
        "worsened_outcome": worse.outcome,
        "bins": [(b.lower, b.upper, b.count) for b in report.after.bins],
    }
    print(f"fitted temperature on the calibration split: {summary['temperature']:.6f} (2 / ln 3)")
    print(f"calibrator id {summary['calibrator_id'][:19]}... reloaded with identical probabilities")
    print(report.summary())
    print("reliability bins after calibration (lower, upper, count):", summary["bins"])
    print(worse.summary())
    print(f"refused: {refused}")
    print(f"override recorded: {overridden.override}")
    return summary


def main() -> None:
    calibration_offline_workflow()


if __name__ == "__main__":
    main()
