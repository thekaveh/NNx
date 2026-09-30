"""FEAT-007: fitted classifier calibration (``nnx.calibration``).

Scalar temperature scaling fitted on a named calibration split, float64 over
copied logits; NLL, Brier and reliability bins; an identity-bound calibrator
(label schema, model id, fit config, split ids) that serializes, reloads and
reports before/after held-out metrics without claiming success.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nnx import calibration
from nnx.calibration import (
    DEFAULT_EPSILON,
    CalibratedPrediction,
    CalibrationError,
    CalibrationFitError,
    CalibrationMismatchError,
    CalibrationReport,
    ReliabilityBin,
    TemperatureCalibrator,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    model_fingerprint,
    negative_log_likelihood,
    reliability_bins,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("a", "b", "c")
BINARY = ("neg", "pos")

# Fixture P: hand-computed metrics (row 3 gives the true class probability 0).
P = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4], [0.0, 1.0, 0.0]])
Y = np.array([0, 2, 2, 0])

# Fixture K: 2-class calibration split whose NLL optimum is T* = 2 / ln 3
# (3 of 4 rows correct at logit margin 2, so the optimal confidence is 0.75).
K_LOGITS = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
K_TARGETS = np.array([0, 0, 1, 1])
T_STAR = 2.0 / math.log(3.0)


def _fit(logits=K_LOGITS, targets=K_TARGETS, **overrides):
    kwargs = {"labels": BINARY, "model_id": "run-1:BEST", "split_id": "calib", "train_split_id": "train"}
    kwargs.update(overrides)
    return fit_temperature(logits, targets, **kwargs)


def _calibrator(**overrides) -> TemperatureCalibrator:
    return _fit(**overrides).require()


# --------------------------------------------------------------- fitting rejects bad inputs (AC1)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_fit_rejects_non_finite_logits(bad):
    logits = K_LOGITS.copy()
    logits[1, 0] = bad
    with pytest.raises(CalibrationError, match="non-finite"):
        _fit(logits=logits)


def test_fit_rejects_fewer_than_two_classes():
    with pytest.raises(CalibrationError, match="at least 2 classes"):
        _fit(logits=np.zeros((3, 1)), targets=np.zeros(3, dtype=int), labels=("only",))


@pytest.mark.parametrize("targets", [[0, 0, 2, 1], [0, -1, 1, 1]])
def test_fit_rejects_out_of_range_targets(targets):
    with pytest.raises(CalibrationError, match="out of range"):
        _fit(targets=np.array(targets))


def test_fit_rejects_non_integer_and_misaligned_targets():
    with pytest.raises(CalibrationError, match="integer class indices"):
        _fit(targets=np.array([0.0, 0.0, 1.0, 1.0]))
    with pytest.raises(CalibrationError, match="one target per row"):
        _fit(targets=np.array([0, 1]))


@pytest.mark.parametrize("field", ["train_split_id", "test_split_id"])
def test_fit_rejects_a_calibration_split_equal_to_train_or_test(field):
    with pytest.raises(CalibrationError, match="must differ"):
        _fit(split_id="shared", **{field: "shared"})


def test_fit_rejects_missing_ids_and_bad_labels():
    with pytest.raises(CalibrationError, match="split_id"):
        _fit(split_id="")
    with pytest.raises(CalibrationError, match="model_id"):
        _fit(model_id="")
    with pytest.raises(CalibrationError, match="one label per class"):
        _fit(labels=("only",))
    with pytest.raises(CalibrationError, match="unique"):
        _fit(labels=("x", "x"))


def test_fit_records_that_loaders_cannot_prove_disjointness():
    fit = _fit(test_split_id="test")
    split = fit.require().split
    assert split["calibration"] == "calib" and split["train"] == "train" and split["test"] == "test"
    assert split["disjointness"] == "unverified"
    assert "cannot prove" in split["note"] and "loader" in split["note"]
    assert fit.split == split


# ------------------------------------------------------- temperature, failures and argmax (AC2)


def test_fit_recovers_the_closed_form_temperature_and_stays_positive():
    fit = _fit()
    assert fit.ok and fit.status == "ok" and fit.reason is None
    calibrator = fit.require()
    assert calibrator.temperature > 0
    assert calibrator.temperature == pytest.approx(T_STAR, rel=1e-8)
    result = calibrator.fit_result
    assert result["n_samples"] == 4 and result["n_classes"] == 2
    assert result["nll_after"] < result["nll_before"]
    assert result["nll_after"] == pytest.approx(-(3 * math.log(0.75) + math.log(0.25)) / 4, rel=1e-9)


@pytest.mark.parametrize(
    ("logits", "targets", "reason"),
    [
        (np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([0, 1]), "separable"),
        (np.array([[1.0, 1.0], [3.0, 3.0]]), np.array([0, 1]), "does not depend on temperature"),
        (np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([1, 0]), "no better than uniform"),
    ],
)
def test_a_failed_fit_returns_a_failure_not_a_nan_calibrator(logits, targets, reason):
    fit = _fit(logits=logits, targets=targets)
    assert not fit.ok and fit.status == "failed" and fit.calibrator is None
    assert reason in fit.reason
    assert fit.config["method"] == "temperature"
    with pytest.raises(CalibrationFitError, match=reason):
        fit.require()


def test_non_convergence_is_a_failure():
    fit = _fit(max_iterations=2)
    assert not fit.ok and "did not converge" in fit.reason


def test_fit_config_validation():
    with pytest.raises(CalibrationError, match="min_temperature"):
        _fit(min_temperature=0.0)
    with pytest.raises(CalibrationError, match="max_temperature"):
        _fit(min_temperature=2.0, max_temperature=1.0)
    with pytest.raises(CalibrationError, match="max_iterations"):
        _fit(max_iterations=0)
    with pytest.raises(CalibrationError, match="tolerance"):
        _fit(tolerance=0.0)


def test_transform_preserves_argmax_and_the_inputs():
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=3.0, size=(64, 3)).astype(np.float32)
    targets = rng.integers(0, 3, size=64)
    calibrator = fit_temperature(logits, targets, labels=LABELS, model_id="m", split_id="calib").require()
    held_out = rng.normal(scale=3.0, size=(32, 3)).astype(np.float32)
    before = held_out.copy()
    result = calibrator.transform(held_out, labels=LABELS, model_id="m")
    np.testing.assert_array_equal(held_out, before)  # the caller's logits are never modified
    assert np.array_equal(result.calibrated_probabilities.argmax(axis=1), held_out.argmax(axis=1))
    assert np.array_equal(result.decoded, held_out.argmax(axis=1))
    assert result.calibrated_probabilities.dtype == np.float64


def test_fitting_leaves_model_parameters_and_predictions_unchanged(tmp_path, monkeypatch):
    from torch.utils.data import DataLoader, TensorDataset

    from nnx import Activations, Devices, Nets, NNModel, NNModelParams, NNOptimParams, NNParams, NNTrainParams
    from nnx.prediction import ProbabilitySpec

    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=3, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU),
    )
    X = torch.randn(48, 4)
    Yt = torch.randint(0, 3, (48,))
    loader = DataLoader(TensorDataset(X[:16], Yt[:16]), batch_size=8)
    run = model.train(
        params=NNTrainParams(
            n_epochs=1, train_loader=loader, optim=NNOptimParams.builder().sgd(max_lr=0.1).build(), data_id="calib"
        )
    )
    run_id, run_state = run.id, run.state()
    weights = {name: value.clone() for name, value in model.net.state_dict().items()}
    fingerprint = model_fingerprint(model)
    predicted = model.predict(X[16:])

    spec = ProbabilitySpec(kind="categorical", class_axis=1, labels=LABELS)
    calib = model.predict_proba(X[16:32], spec)
    # The model's own argmax with every fourth row flipped: neither separable
    # nor worse than uniform, so the fit has an interior optimum.
    targets = calib.decoded.copy()
    targets[::4] = (targets[::4] + 1) % 3
    fit = fit_temperature(calib, targets, model_id=fingerprint, split_id="calib", train_split_id="train")
    assert fit.ok, fit.reason
    held_out = model.predict_proba(X[32:], spec)
    calibrated = fit.require().transform(held_out, model_id=fingerprint)
    assert calibrated.labels == LABELS
    np.testing.assert_array_equal(calibrated.logits, held_out.logits)
    np.testing.assert_array_equal(calibrated.decoded, held_out.decoded)

    after = model.predict(X[16:])
    np.testing.assert_array_equal(after.logits, predicted.logits)
    np.testing.assert_array_equal(after.classes, predicted.classes)
    assert all(torch.equal(value, weights[name]) for name, value in model.net.state_dict().items())
    assert model_fingerprint(model) == fingerprint
    assert run.id == run_id and run.state() == run_state


def test_model_fingerprint_changes_with_the_weights():
    net = torch.nn.Linear(3, 2)
    first = model_fingerprint(net)
    assert first.startswith("sha256:") and model_fingerprint(net) == first
    with torch.no_grad():
        net.weight[0, 0] += 1.0
    assert model_fingerprint(net) != first
    with pytest.raises(TypeError, match="state_dict"):
        model_fingerprint(object())


# ----------------------------------------------------------------- metrics and bins (AC3)


def test_temperature_one_reproduces_uncalibrated_probabilities():
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    logits = np.log(P[:3])
    identity = TemperatureCalibrator(temperature=1.0, labels=LABELS, model_id="m", split_id="calib")
    result = identity.transform(logits, labels=LABELS, model_id="m")
    np.testing.assert_array_equal(result.calibrated_probabilities, result.probabilities)
    reference = prediction_from_logits(logits, ProbabilitySpec(kind="categorical", labels=LABELS))
    np.testing.assert_array_equal(result.probabilities, reference.probabilities)
    np.testing.assert_allclose(result.probabilities, P[:3], atol=1e-15)


def test_nll_matches_the_hand_computed_fixture():
    exact = negative_log_likelihood(P[:3], Y[:3], epsilon=None)
    assert exact == pytest.approx(-(math.log(0.7) + math.log(0.3) + math.log(0.4)) / 3, rel=1e-15)
    assert negative_log_likelihood(P, Y, epsilon=None) == math.inf  # a zero true-class probability
    floored = -(math.log(0.7) + math.log(0.3) + math.log(0.4) + math.log(DEFAULT_EPSILON)) / 4
    assert negative_log_likelihood(P, Y) == pytest.approx(floored, rel=1e-15)
    assert negative_log_likelihood(P, Y, epsilon=1e-3) == pytest.approx(
        -(math.log(0.7) + math.log(0.3) + math.log(0.4) + math.log(1e-3)) / 4, rel=1e-15
    )
    with pytest.raises(CalibrationError, match="epsilon"):
        negative_log_likelihood(P, Y, epsilon=0.0)


def test_brier_is_the_mean_class_sum_of_squared_error():
    assert brier_score(P, Y) == pytest.approx((0.14 + 0.78 + 0.54 + 2.0) / 4, rel=1e-12)
    assert brier_score(np.eye(3), np.arange(3)) == 0.0


def test_reliability_bins_are_equal_width_half_open_with_the_last_closed():
    bins = reliability_bins(P, Y, n_bins=5)
    assert all(isinstance(b, ReliabilityBin) for b in bins)
    assert [(b.lower, b.upper) for b in bins] == [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    assert [b.count for b in bins] == [0, 0, 2, 1, 1]  # 0.4 opens [0.4, 0.6); 1.0 closes the last bin
    assert bins[0].confidence is None and bins[0].accuracy is None  # empty bins count 0
    assert bins[2].confidence == pytest.approx(0.45) and bins[2].accuracy == 0.5
    assert bins[3].confidence == pytest.approx(0.7) and bins[3].accuracy == 1.0
    assert bins[4].confidence == 1.0 and bins[4].accuracy == 0.0
    assert expected_calibration_error(bins) == pytest.approx(0.5 * 0.05 + 0.25 * 0.3 + 0.25 * 1.0)
    # A confidence exactly on an inner edge opens the upper bin.
    edge = reliability_bins(np.array([[0.6, 0.4], [0.4, 0.6]]), np.array([0, 0]), n_bins=5)
    assert [b.count for b in edge] == [0, 0, 0, 2, 0]
    assert expected_calibration_error(reliability_bins(np.empty((0, 2)), np.empty(0, dtype=int))) is None


@pytest.mark.parametrize(
    ("probabilities", "match"),
    [
        (np.array([[0.5, 0.6]]), "sum to 1"),
        (np.array([[1.5, -0.5]]), r"\[0, 1\]"),
        (np.array([[math.nan, 1.0]]), "non-finite"),
        (np.array([0.5, 0.5]), "2-D"),
    ],
)
def test_metrics_reject_malformed_probabilities(probabilities, match):
    with pytest.raises(CalibrationError, match=match):
        brier_score(probabilities, np.zeros(len(probabilities), dtype=int))
    with pytest.raises(CalibrationError):
        reliability_bins(probabilities, np.zeros(len(probabilities), dtype=int), n_bins=0)


# ---------------------------------------------------------------- identity binding (AC4 / AC7)


def test_calibrator_stores_labels_model_fit_config_and_split():
    calibrator = _calibrator(test_split_id="test")
    assert calibrator.labels == BINARY and calibrator.model_id == "run-1:BEST"
    assert calibrator.split_id == "calib"
    config = calibrator.fit_config
    assert config["method"] == "temperature" and config["objective"] == "nll" and config["dtype"] == "float64"
    assert {"min_temperature", "max_temperature", "tolerance", "max_iterations"} <= set(config)
    assert calibrator.id.startswith("sha256:") and calibrator.id == calibrator.digest()


class _AccessSpy:
    """An array-like that records whether anything read it."""

    def __init__(self, array):
        self.array = array
        self.reads = 0

    def __array__(self, dtype=None, copy=None):
        self.reads += 1
        return np.asarray(self.array, dtype=dtype)

    def __len__(self):
        return len(self.array)


def test_reordered_labels_or_another_model_fail_before_any_transform():
    calibrator = _calibrator()
    spy = _AccessSpy(K_LOGITS)
    with pytest.raises(CalibrationMismatchError, match="label"):
        calibrator.transform(spy, labels=("pos", "neg"), model_id="run-1:BEST")
    with pytest.raises(CalibrationMismatchError, match="model"):
        calibrator.transform(spy, labels=BINARY, model_id="run-2:BEST")
    with pytest.raises(CalibrationMismatchError, match="model"):
        calibrator.report(spy, K_TARGETS, labels=BINARY, model_id="run-2:BEST", split_id="test")
    assert spy.reads == 0
    with pytest.raises(CalibrationError, match="labels"):
        calibrator.transform(K_LOGITS, model_id="run-1:BEST")  # bare logits must name their labels


def test_a_named_override_records_the_mismatch():
    calibrator = _calibrator()
    result = calibrator.transform(K_LOGITS, labels=("pos", "neg"), model_id="run-2", override="relabelled-head")
    assert result.override == {
        "name": "relabelled-head",
        "mismatches": {
            "labels": {"expected": ("neg", "pos"), "actual": ("pos", "neg")},
            "model_id": {"expected": "run-1:BEST", "actual": "run-2"},
        },
    }
    assert result.labels == ("pos", "neg")
    clean = calibrator.transform(K_LOGITS, labels=BINARY, model_id="run-1:BEST", override="unused")
    assert clean.override is None  # nothing to override
    with pytest.raises(CalibrationError, match="override"):
        calibrator.transform(K_LOGITS, labels=("pos", "neg"), model_id="run-1:BEST", override="")


def test_an_override_stays_visible_in_the_report():
    calibrator = _calibrator()
    report = calibrator.report(
        K_LOGITS, K_TARGETS, labels=BINARY, model_id="run-9", split_id="test", override="retrained-head"
    )
    assert report.override["name"] == "retrained-head"
    assert report.override["mismatches"]["model_id"] == {"expected": "run-1:BEST", "actual": "run-9"}
    assert "override 'retrained-head'" in report.summary()
    assert CalibrationReport.from_json(report.to_json()).override == report.override


# -------------------------------------------------------------- save / reload and reports (AC5)


def test_save_and_reload_give_identical_probabilities(tmp_path):
    calibrator = _calibrator(test_split_id="test")
    path = tmp_path / "calibrator.json"
    calibrator.save(path)
    reloaded = TemperatureCalibrator.load(path)
    assert reloaded == calibrator and reloaded.id == calibrator.id
    assert reloaded.temperature == calibrator.temperature  # float64-exact through JSON
    held_out = np.random.default_rng(1).normal(size=(10, 2))
    first = calibrator.transform(held_out, labels=BINARY, model_id="run-1:BEST")
    second = reloaded.transform(held_out, labels=BINARY, model_id="run-1:BEST")
    np.testing.assert_array_equal(first.calibrated_probabilities, second.calibrated_probabilities)
    assert json.loads(path.read_text())["format"] == calibration.FORMAT


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s: s.update(format="nnx.calibration/0"), "format"),
        (lambda s: s.update(temperature=0.0), "positive"),
        (lambda s: s.update(temperature="inf"), "finite"),
        (lambda s: s.update(labels=["neg", "neg"]), "unique"),
        (lambda s: s.update(model_id=""), "model_id"),
        (lambda s: s["split"].update(train="calib"), "must differ"),
        (lambda s: s.pop("labels"), "labels"),
    ],
)
def test_loading_rejects_malformed_state(mutate, match):
    state = _calibrator().state()
    mutate(state)
    with pytest.raises(CalibrationError, match=match):
        TemperatureCalibrator.from_state(state)


def test_report_shows_before_and_after_held_out_metrics_when_calibration_helps():
    calibrator = _calibrator()
    held_out = np.array([[2.0, 0.0]] * 4 + [[0.0, 2.0]] * 4)
    targets = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    report = calibrator.report(held_out, targets, labels=BINARY, model_id="run-1:BEST", split_id="test", n_bins=4)
    assert report.outcome == "improved" and report.improved
    assert report.after.nll < report.before.nll and report.after.brier < report.before.brier
    raw = calibrator.transform(held_out, labels=BINARY, model_id="run-1:BEST")
    assert report.before.nll == negative_log_likelihood(raw.probabilities, targets)
    assert report.after.brier == brier_score(raw.calibrated_probabilities, targets)
    assert len(report.before.bins) == 4 and report.n_samples == 8
    assert report.split_id == "test" and report.calibration_split_id == "calib"
    assert report.calibrator_id == calibrator.id


def test_report_never_claims_success_when_calibration_worsens():
    calibrator = _calibrator()  # T* > 1 softens every prediction
    held_out = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])  # all correct: softening hurts
    report = calibrator.report(held_out, np.array([0, 0, 1, 1]), labels=BINARY, model_id="run-1:BEST", split_id="t")
    assert report.outcome == "worsened" and not report.improved
    assert report.after.nll > report.before.nll and report.after.brier > report.before.brier
    summary = report.summary()
    assert "worsened" in summary and "improved" not in summary
    assert "keep the uncalibrated probabilities" in summary


def test_report_outcome_mixed_and_unchanged():
    identity = TemperatureCalibrator(temperature=1.0, labels=BINARY, model_id="m", split_id="calib")
    unchanged = identity.report(K_LOGITS, K_TARGETS, labels=BINARY, model_id="m", split_id="test")
    assert unchanged.outcome == "unchanged" and not unchanged.improved
    before = unchanged.before
    assert (before.nll, before.brier) == (unchanged.after.nll, unchanged.after.brier)


def test_report_refuses_the_calibration_split_as_held_out_data():
    calibrator = _calibrator()
    with pytest.raises(CalibrationError, match="held out"):
        calibrator.report(K_LOGITS, K_TARGETS, labels=BINARY, model_id="run-1:BEST", split_id="calib")
    with pytest.raises(CalibrationError, match="held out"):
        calibrator.report(K_LOGITS, K_TARGETS, labels=BINARY, model_id="run-1:BEST", split_id="train")


def test_exact_nll_report_serializes_an_infinite_value():
    calibrator = TemperatureCalibrator(temperature=1.0, labels=BINARY, model_id="m", split_id="calib")
    report = calibrator.report(
        np.array([[1e4, 0.0]]), np.array([1]), labels=BINARY, model_id="m", split_id="t", epsilon=None
    )
    assert report.before.nll == math.inf
    reloaded = CalibrationReport.from_json(report.to_json())
    assert reloaded == report and reloaded.before.nll == math.inf


# ----------------------------------------------------- richer result and a reloaded report (AC6 / AC7)


def test_calibrated_prediction_keeps_separate_ordered_label_fields():
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    calibrator = _calibrator()
    prediction = prediction_from_logits(
        K_LOGITS.astype(np.float32), ProbabilitySpec(kind="categorical", labels=BINARY), sample_ids=[7, 8, 9, 10]
    )
    result = calibrator.transform(prediction, model_id="run-1:BEST")
    assert isinstance(result, CalibratedPrediction)
    assert result.labels == BINARY and result.calibrator_id == calibrator.id and result.model_id == "run-1:BEST"
    assert result.temperature == calibrator.temperature
    np.testing.assert_array_equal(result.logits, prediction.logits)
    assert result.logits.dtype == np.float32  # raw logits kept as given
    assert result.sample_ids.tolist() == [7, 8, 9, 10]
    np.testing.assert_allclose(result.probabilities, prediction.probabilities, rtol=1e-6)
    assert not np.shares_memory(result.probabilities, result.calibrated_probabilities)
    assert not np.array_equal(result.probabilities, result.calibrated_probabilities)
    np.testing.assert_allclose(result.calibrated_probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert result.decoded_labels().tolist() == ["neg", "neg", "neg", "pos"]


def test_a_prediction_result_must_be_categorical_two_dimensional():
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    calibrator = _calibrator()
    bernoulli = prediction_from_logits(K_LOGITS, ProbabilitySpec(kind="bernoulli", labels=BINARY))
    with pytest.raises(CalibrationError, match="categorical"):
        calibrator.transform(bernoulli, model_id="run-1:BEST")
    with pytest.raises(CalibrationError, match=r"\(N, C\)"):
        calibrator.transform(np.zeros((2, 2, 2)), labels=BINARY, model_id="run-1:BEST")


def test_a_reloaded_calibrator_reproduces_its_report(tmp_path):
    calibrator = _calibrator()
    held_out = np.random.default_rng(3).normal(scale=2.0, size=(40, 2))
    targets = (held_out[:, 1] > held_out[:, 0]).astype(int)
    targets[:6] = 1 - targets[:6]
    report = calibrator.report(held_out, targets, labels=BINARY, model_id="run-1:BEST", split_id="test")
    calibrator.save(tmp_path / "c.json")
    again = TemperatureCalibrator.load(tmp_path / "c.json").report(
        held_out, targets, labels=BINARY, model_id="run-1:BEST", split_id="test"
    )
    assert again == report and again.to_json() == report.to_json()
    report.save(tmp_path / "report.json")
    assert CalibrationReport.load(tmp_path / "report.json") == report


def test_torch_logits_are_accepted_and_copied():
    calibrator = _calibrator()
    tensor = torch.tensor(K_LOGITS, dtype=torch.bfloat16)
    result = calibrator.transform(tensor, labels=BINARY, model_id="run-1:BEST")
    assert result.logits.dtype == np.float32 and result.calibrated_probabilities.shape == (4, 2)


# ------------------------------------------------ named metrics agree; import is self-contained (AC8)


def test_named_metrics_match_these_functions_on_the_same_fixtures():
    from nnx.monitors import MetricSpec

    for metric, function in (("nll", negative_log_likelihood), ("brier", brier_score)):
        accumulator = MetricSpec(metric).accumulator()
        accumulator.update(Y[:2], P[:2])
        accumulator.update(Y[2:], P[2:])  # batched accumulation over the full sample
        assert accumulator.result() == pytest.approx(function(P, Y), rel=1e-12, abs=0.0)


def test_calibration_imports_no_optional_consumer():
    """The module depends on the standard library, NumPy and NumPy-only
    internal helpers: loaded without the nnx package's own ``__init__``, it
    imports no consumer (prediction, monitors, provenance, decisions) and no
    torch, pandas or scikit-learn."""
    tree = ast.parse((ROOT / "src" / "nnx" / "calibration.py").read_text())
    top_level, internal = set(), set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level:
            internal.add(node.module)
        elif isinstance(node, ast.ImportFrom):
            top_level.add((node.module or "").split(".")[0])
    stdlib = {"__future__", "collections", "dataclasses", "functools", "hashlib", "json", "math", "numbers", "os"} | {
        "sys",
        "typing",
    }
    assert top_level <= stdlib | {"numpy"}, top_level
    assert internal == {"_config", "_probability", "_validation"}, internal
    code = f"""
import sys, types
package = types.ModuleType("nnx")
package.__path__ = [{str(ROOT / "src" / "nnx")!r}]  # a bare package: nnx/__init__.py never runs
sys.modules["nnx"] = package
import numpy as np
from nnx import calibration as module
fit = module.fit_temperature(np.array([[2.0, 0.0]] * 3 + [[0.0, 2.0]]), np.array([0, 0, 1, 1]),
                             labels=("n", "p"), model_id="m", split_id="calib")
report = fit.require().report(np.array([[1.0, 0.0]]), np.array([0]), labels=("n", "p"), model_id="m", split_id="t")
module.TemperatureCalibrator.from_json(fit.require().to_json())
nnx_modules = sorted(name for name in sys.modules if name.split(".")[0] == "nnx")
assert nnx_modules == ["nnx", "nnx._config", "nnx._probability", "nnx._validation", "nnx.calibration"], nnx_modules
heavy = sorted(name for name in sys.modules if name.split(".")[0] in {{"torch", "sklearn", "pandas"}})
assert heavy == [], heavy
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_surface_is_complete():
    import nnx

    assert nnx.calibration is calibration and "calibration" in nnx.__all__
    for name in calibration.__all__:
        assert getattr(calibration, name, None) is not None, name


# ------------------------------------------------------------------------ review regressions


def test_model_fingerprint_hashes_a_whole_module_with_a_net_submodule():
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 2)

    wrapper = Wrapper()
    before = model_fingerprint(wrapper)
    with torch.no_grad():
        wrapper.head.weight.zero_()
    assert model_fingerprint(wrapper) != before  # the head counts, not only `.net`


def test_model_fingerprint_covers_extra_state_and_rejects_opaque_entries():
    class WithExtra(torch.nn.Linear):
        def __init__(self, extra):
            super().__init__(2, 2)
            self.extra = extra

        def get_extra_state(self):
            return self.extra

        def set_extra_state(self, state):
            self.extra = state

    module = WithExtra({"vocab": 3})
    first = model_fingerprint(module)
    module.extra = {"vocab": 4}
    assert model_fingerprint(module) != first
    with pytest.raises(TypeError, match="declared model_id"):
        model_fingerprint(WithExtra(object()))


def test_low_temperature_does_not_overflow_large_finite_logits():
    sharp = TemperatureCalibrator(temperature=0.5, labels=BINARY, model_id="m", split_id="calib")
    result = sharp.transform(np.array([[1e308, 0.9e308]]), labels=BINARY, model_id="m")
    assert np.isfinite(result.calibrated_probabilities).all()
    assert result.calibrated_probabilities.tolist() == [[1.0, 0.0]]


def test_reliability_bins_accept_an_empty_target_list():
    bins = reliability_bins(np.zeros((0, 2)), [], n_bins=3)
    assert [b.count for b in bins] == [0, 0, 0]


def _report_state() -> dict:
    return _calibrator().report(K_LOGITS, K_TARGETS, labels=BINARY, model_id="run-1:BEST", split_id="test").state()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(override={"name": "x"}),
        lambda s: s.update(override={"name": "x", "mismatches": {"weights": {"expected": 1, "actual": 2}}}),
        lambda s: s.update(n_samples="x"),
        lambda s: s.update(n_samples=0),
        lambda s: s["before"].update(brier="abc"),
        lambda s: s["before"].update(nll=None),
        lambda s: s["after"]["bins"][0].update(lower="?"),
        lambda s: s["after"]["bins"][0].update(count=-1),
        lambda s: s.pop("split"),
        lambda s: s.update(outcome="great"),
    ],
)
def test_loading_a_malformed_report_raises_calibration_error(mutate):
    state = _report_state()
    mutate(state)
    with pytest.raises(CalibrationError):
        CalibrationReport.from_state(state)


def test_a_malformed_calibrator_state_raises_calibration_error():
    state = _calibrator().state()
    state["fit"]["config"] = {"bad": math.inf}
    with pytest.raises(CalibrationError, match="JSON-like"):
        TemperatureCalibrator.from_state(state)
    state = _calibrator().state()
    state["split"] = "calib"
    with pytest.raises(CalibrationError, match="mappings"):
        TemperatureCalibrator.from_state(state)


def test_calibrators_and_fits_pickle_and_deepcopy():
    import copy
    import pickle

    fit = _fit(test_split_id="test")
    calibrator = fit.require()
    for clone in (pickle.loads(pickle.dumps(calibrator)), copy.deepcopy(calibrator)):
        assert clone == calibrator and clone.id == calibrator.id
    assert pickle.loads(pickle.dumps(fit)).require() == calibrator
    failed = _fit(logits=np.array([[1.0, 1.0], [2.0, 2.0]]), targets=np.array([0, 1]))
    assert pickle.loads(pickle.dumps(failed)).reason == failed.reason


def test_nested_fit_records_are_immutable_so_the_id_cannot_drift():
    calibrator = TemperatureCalibrator(
        temperature=1.5, labels=BINARY, model_id="m", split_id="calib", fit_result={"x": {"y": [1, 2]}}
    )
    digest = calibrator.digest()
    with pytest.raises(TypeError):
        calibrator.fit_result["x"]["y"] = 2  # type: ignore[index]
    assert calibrator.fit_result["x"]["y"] == (1, 2)
    assert calibrator.state()["fit"]["result"] == {"x": {"y": [1, 2]}}
    assert TemperatureCalibrator.from_json(calibrator.to_json()).digest() == digest


@pytest.mark.parametrize("nll", [math.nan, "-inf", -0.5, "abc"])
def test_a_report_nll_must_be_non_negative_or_inf(nll):
    state = _report_state()
    state["before"]["nll"] = nll
    with pytest.raises(CalibrationError, match="nll"):
        CalibrationReport.from_state(state)


def test_a_loaded_report_must_be_held_out_and_consistent():
    state = _report_state()
    state["split"]["evaluation"] = state["split"]["calibration"]
    with pytest.raises(CalibrationError, match="not held out"):
        CalibrationReport.from_state(state)
    state = _report_state()
    assert state["split"]["train"] == "train"
    state["split"]["evaluation"] = "train"
    with pytest.raises(CalibrationError, match="not held out"):
        CalibrationReport.from_state(state)
    state = _report_state()
    assert state["outcome"] == "improved"
    state["after"], state["before"] = state["before"], state["after"]  # swap: the claim no longer holds
    with pytest.raises(CalibrationError, match="contradicts"):
        CalibrationReport.from_state(state)


def test_logits_spanning_more_than_float64_are_rejected_clearly(recwarn):
    with pytest.raises(CalibrationError, match="float64 range"):
        _fit(logits=np.array([[1e308, -1e308], [0.0, 1.0]]), targets=np.array([0, 1]))
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_a_tolerance_below_float_resolution_still_converges():
    fit = _fit(tolerance=1e-17)
    assert fit.ok and fit.require().temperature == pytest.approx(T_STAR, rel=1e-12)


def test_labels_may_be_numpy_or_pandas_arrays():
    import pandas as pd

    fit = _fit(labels=np.array(["neg", "pos"]))
    assert fit.require().labels == BINARY and all(type(label) is str for label in fit.require().labels)
    assert _fit(labels=pd.Index(["neg", "pos"])).require().labels == BINARY


def test_override_records_are_immutable_and_hash_stably():
    calibrator = _calibrator()
    result = calibrator.transform(K_LOGITS, labels=BINARY, model_id="run-2", override="audit")
    with pytest.raises(TypeError):
        result.override["name"] = "x"  # type: ignore[index]
    report = calibrator.report(K_LOGITS, K_TARGETS, labels=BINARY, model_id="run-2", split_id="t", override="audit")
    before = hash(report)
    with pytest.raises(TypeError):
        report.override["mismatches"]["model_id"] = {}  # type: ignore[index]
    assert hash(report) == before and report in {report}
    assert report.state()["override"]["mismatches"]["model_id"] == {"expected": "run-1:BEST", "actual": "run-2"}


def test_float16_probabilities_from_predict_proba_are_accepted():
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    logits = np.random.default_rng(0).normal(scale=3.0, size=(8, 10)).astype(np.float16)
    result = prediction_from_logits(logits, ProbabilitySpec(kind="categorical"))
    assert result.probabilities.dtype == np.float16
    targets = np.arange(8) % 10
    from nnx.monitors import MetricSpec

    accumulator = MetricSpec("brier").accumulator()
    accumulator.update(targets, result.probabilities)
    assert brier_score(result.probabilities, targets) == pytest.approx(accumulator.result(), rel=1e-12)
    with pytest.raises(CalibrationError, match="sum to 1"):
        brier_score(np.array([[0.5, 0.6]], dtype=np.float16), [0])  # still far off


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["after"]["bins"][0].update(count=3, confidence=None, accuracy=None),
        lambda s: s["after"]["bins"].pop(),
        lambda s: s["after"]["bins"][-1].update(count=s["after"]["bins"][-1]["count"] + 1),
        lambda s: s["after"]["bins"][0].update(lower=0.5),
        lambda s: s["before"]["bins"][0].update(lower=0.9, upper=0.95),
    ],
)
def test_report_bins_must_be_consistent_with_their_counts(mutate):
    state = _report_state()
    mutate(state)
    with pytest.raises(CalibrationError):
        CalibrationReport.from_state(state)


def test_model_fingerprint_sees_through_a_compiled_wrapper():
    class OptimizedModule(torch.nn.Module):  # the shape of torch.compile's wrapper
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

    net = torch.nn.Linear(3, 2)
    assert set(OptimizedModule(net).state_dict()) == {"_orig_mod.weight", "_orig_mod.bias"}
    assert model_fingerprint(OptimizedModule(net)) == model_fingerprint(net)


def test_report_bins_keep_the_raw_argmax_through_a_float_tie():
    calibrator = TemperatureCalibrator(temperature=T_STAR, labels=BINARY, model_id="m", split_id="calib")
    logits = np.array([[-6e-17, 0.0]])  # class 1 wins; after T*, exp() rounds both entries to 1.0
    result = calibrator.transform(logits, labels=BINARY, model_id="m")
    assert result.decoded.tolist() == [1]
    assert result.calibrated_probabilities[0, 0] == result.calibrated_probabilities[0, 1]  # an exact float tie
    report = calibrator.report(logits, np.array([1]), labels=BINARY, model_id="m", split_id="t", n_bins=2)
    for metrics in (report.before, report.after):
        hit = [b for b in metrics.bins if b.count]
        assert len(hit) == 1 and hit[0].accuracy == 1.0  # the same prediction (class 1) before and after


def test_report_outcome_ignores_float_rounding_noise():
    nearly_one = TemperatureCalibrator(temperature=1.0 + 1e-13, labels=BINARY, model_id="m", split_id="calib")
    report = nearly_one.report(K_LOGITS, K_TARGETS, labels=BINARY, model_id="m", split_id="t")
    assert report.before.nll != report.after.nll  # last-bit differences only
    assert report.outcome == "unchanged" and not report.improved


def test_a_fit_is_hashable_and_its_split_read_only():
    fit = _fit()
    assert fit in {fit}
    with pytest.raises(TypeError):
        fit.split["disjointness"] = "verified"  # type: ignore[index]
    assert fit.split["disjointness"] == "unverified"


def test_model_fingerprint_sees_through_data_parallel():
    net = torch.nn.Linear(3, 2)
    wrapped = torch.nn.DataParallel(net)  # CPU: no devices, the wrapper only prefixes keys with "module."
    assert set(wrapped.state_dict()) == {"module.weight", "module.bias"}
    assert model_fingerprint(wrapped) == model_fingerprint(net)


def test_failure_reasons_name_the_real_cause():
    wide = np.array([[1e4, 0.0]] * 9 + [[0.0, 1e4]])  # 90% right, but at a scale beyond max_temperature
    fit = _fit(logits=wide, targets=np.array([0] * 8 + [1, 1]))
    assert not fit.ok and "larger max_temperature" in fit.reason
    assert _fit(logits=wide, targets=np.array([0] * 8 + [1, 1]), max_temperature=1e5).ok
    masked = _fit(logits=np.array([[0.0, -1e300], [-1e300, 0.0]]), targets=np.array([0, 1]))
    assert not masked.ok and "saturates" in masked.reason and "constant" not in masked.reason


def test_shared_exact_bernoulli_nll_terms_are_zero_not_nan(recwarn):
    from nnx._probability import nll_terms

    terms = nll_terms(np.array([0.0, 1.0]), np.array([0.0, 1.0]), None)
    assert terms.tolist() == [0.0, 0.0]
    assert nll_terms(np.array([1.0]), np.array([0.0]), None).tolist() == [math.inf]
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_a_tolerance_wider_than_the_search_range_is_rejected():
    with pytest.raises(CalibrationError, match="search range"):
        _fit(tolerance=100.0)


def test_masked_logits_fit_and_serve_without_overflow_warnings():
    logits = np.array([[2.0, 0.0, -1e306]] * 3 + [[0.0, 2.0, -1e306]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):  # any unguarded overflow would raise
        fit = fit_temperature(logits, K_TARGETS, labels=LABELS, model_id="m", split_id="calib")
        assert fit.ok and fit.require().temperature == pytest.approx(T_STAR, rel=1e-8)
        sharp = TemperatureCalibrator(temperature=0.01, labels=LABELS, model_id="m", split_id="calib")
        served = sharp.transform(logits, labels=LABELS, model_id="m")
    assert (served.calibrated_probabilities[:, 2] == 0.0).all()


def test_a_negative_bin_count_is_rejected():
    with pytest.raises(CalibrationError, match="count"):
        ReliabilityBin(0.0, 0.5, -3, None, None)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(temprature=1.0),
        lambda s: s["split"].update(calibraton="c"),
        lambda s: s["fit"].update(extra={}),
    ],
)
def test_unknown_keys_in_a_saved_calibrator_are_rejected(mutate):
    state = _calibrator().state()
    mutate(state)
    with pytest.raises(CalibrationError, match="unknown keys"):
        TemperatureCalibrator.from_state(state)
    report = _report_state()
    report["extra"] = 1
    with pytest.raises(CalibrationError, match="unknown keys"):
        CalibrationReport.from_state(report)


def test_extra_state_fingerprints_use_the_shared_canonical_encoding():
    class WithExtra(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 2)

        def get_extra_state(self):
            return {"name": "café"}

        def set_extra_state(self, state):
            pass

    assert model_fingerprint(WithExtra()).startswith("sha256:")


def test_model_fingerprint_rejects_sparse_buffers_with_guidance():
    module = torch.nn.Linear(3, 3)
    module.register_buffer("sparse", torch.eye(3).to_sparse())
    with pytest.raises(TypeError, match="declared model_id"):
        model_fingerprint(module)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("epsilon"),
        lambda s: s["split"].update(extra="x"),
        lambda s: s["split"].pop("train"),
    ],
)
def test_a_report_needs_its_epsilon_and_an_exact_split(mutate):
    state = _report_state()
    mutate(state)
    with pytest.raises(CalibrationError):
        CalibrationReport.from_state(state)


def test_the_row_sum_check_stays_meaningful_for_many_float16_classes():
    with pytest.raises(CalibrationError, match="sum to 1"):
        negative_log_likelihood(np.zeros((3, 1100), dtype=np.float16), [0, 1, 2])


def test_model_fingerprint_materializes_conjugate_views():
    module = torch.nn.Module()
    module.register_buffer("z", torch.tensor([1 + 2j]).conj())
    same = torch.nn.Module()
    same.register_buffer("z", torch.tensor([1 - 2j]))
    assert model_fingerprint(module) == model_fingerprint(same)


def test_exact_bernoulli_terms_accept_zero_dimensional_inputs():
    from nnx._probability import nll_terms

    assert float(nll_terms(np.array(0.0), np.array(0.0), None)) == 0.0


def test_round_nine_argument_and_input_checks():
    with pytest.raises(CalibrationError, match="reciprocal overflows"):
        _fit(min_temperature=1e-310)
    with pytest.raises(CalibrationError, match="n_bins"):
        reliability_bins(P, Y, n_bins=10**6)
    with pytest.raises(CalibrationError, match="numeric array"):
        _fit(logits=[[1.0, 2.0], [3.0]])  # ragged
    with pytest.raises(CalibrationError, match="JSON-like") as caught:
        TemperatureCalibrator(temperature=1.0, labels=BINARY, model_id="m", split_id="c", fit_config={"x": object()})
    assert "register_optimizer_factory" not in str(caught.value)


def test_a_target_on_a_masked_class_is_named_in_the_failure():
    logits = np.array([[2.0, 0.0, -1e9]] * 30 + [[0.0, 2.0, -1e9]] * 10)
    targets = np.array([0] * 20 + [1] * 10 + [1] * 10)
    targets[0] = 2  # a target on the masked class
    fit = fit_temperature(logits, targets, labels=LABELS, model_id="m", split_id="calib")
    assert not fit.ok and "masked classes" in fit.reason


def test_round_ten_inputs_files_and_splits(tmp_path):
    probabilities = torch.softmax(torch.randn(16, 7, generator=torch.Generator().manual_seed(0)), 1).bfloat16()
    targets = np.arange(16) % 7
    assert 0.0 <= brier_score(probabilities, targets) <= 2.0  # bfloat16 rounding is tolerated

    class LooksLikeATensor:
        def detach(self):
            return self

        def cpu(self):
            return self

    with pytest.raises(CalibrationError):
        _fit(logits=LooksLikeATensor())
    binary = tmp_path / "model.pt"
    binary.write_bytes(b"\x80\x04\x95\xff\xfe")
    with pytest.raises(CalibrationError, match="not text"):
        TemperatureCalibrator.load(binary)
    with pytest.raises(CalibrationError, match="not text"):
        CalibrationReport.load(binary)
    with pytest.raises(CalibrationError, match="must differ"):
        _fit(train_split_id="data", test_split_id="data")


def test_round_eleven_batches_counts_ids_and_ranges():
    calibrator = _calibrator()
    empty = calibrator.transform(np.zeros((0, 2)), labels=BINARY, model_id="run-1:BEST")
    assert empty.calibrated_probabilities.shape == (0, 2) and empty.decoded.shape == (0,)
    with pytest.raises(CalibrationMismatchError, match="no override"):
        calibrator.transform(np.zeros((4, 5)), labels=list("vwxyz"), model_id="run-1:BEST", override="x")
    from types import SimpleNamespace

    from nnx.prediction import ProbabilitySpec

    duck = SimpleNamespace(
        logits=np.zeros((2, 2)),
        spec=ProbabilitySpec("categorical", labels=BINARY),
        sample_ids=np.array([2**63, 1], dtype=np.uint64),
    )
    with pytest.raises(CalibrationError, match="int64"):
        calibrator.transform(duck, model_id="run-1:BEST")
    state = calibrator.state()
    state["temperature"] = 1e-300
    with pytest.raises(CalibrationError, match="search range"):
        TemperatureCalibrator.from_state(state)


def test_model_fingerprint_rejects_non_string_extra_state_keys():
    class WithExtra(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 2)

        def get_extra_state(self):
            return {1: "a"}

        def set_extra_state(self, state):
            pass

    with pytest.raises(TypeError, match="declared model_id"):
        model_fingerprint(WithExtra())


def test_a_report_built_directly_is_validated_like_a_loaded_one():
    report = CalibrationReport.from_state(_report_state())
    import dataclasses

    for change in ({"n_samples": 0}, {"n_bins": 0}, {"temperature": -1.0}, {"labels": ("a", "a")}):
        with pytest.raises(CalibrationError):
            dataclasses.replace(report, **change)
    assert (report == object()) is False and report != "report"


def test_metrics_and_bins_are_validated_and_plain_floats():
    with pytest.raises(CalibrationError, match="nll"):
        calibration.CalibrationMetrics(nll=math.nan, brier=0.1, ece=None, bins=())
    with pytest.raises(CalibrationError, match="brier"):
        calibration.CalibrationMetrics(nll=0.1, brier=-1.0, ece=None, bins=())
    with pytest.raises(CalibrationError, match="does not match its bins"):
        calibration.CalibrationMetrics(nll=0.1, brier=0.1, ece=0.3, bins=())
    # A row summing to 1 + rounding can score a Brier a hair above 2; that still reports.
    assert calibration.CalibrationMetrics.of(np.array([[1.0, 9e-7, 0.0]]), [2]).brier > 2.0
    assert calibration.CalibrationMetrics(nll=math.inf, brier=0.1, ece=None, bins=()).nll == math.inf
    bin_ = ReliabilityBin(np.float32(0.0), 1.0, 1, np.float32(0.5), 1.0)
    assert type(bin_.lower) is float and type(bin_.confidence) is float
    json.dumps(bin_.state())  # JSON-serializable


def test_reports_record_the_declared_test_split_and_need_two_labels():
    calibrator = _calibrator(test_split_id="test")
    report = calibrator.report(K_LOGITS, K_TARGETS, labels=BINARY, model_id="run-1:BEST", split_id="val")
    assert report.test_split_id == "test" and report.state()["split"]["test"] == "test"
    state = report.state()
    state.pop("ece", None)
    state["before"].pop("ece")
    with pytest.raises(CalibrationError):
        CalibrationReport.from_state(state)
    state = report.state()
    state["labels"] = ["a"]
    with pytest.raises(CalibrationError, match="2 labels"):
        CalibrationReport.from_state(state)


def test_lazy_parameters_get_the_documented_type_error():
    with pytest.raises(TypeError, match="declared model_id"):
        model_fingerprint(torch.nn.LazyLinear(3))


def test_a_report_rejects_non_metric_before_and_after():
    report = CalibrationReport.from_state(_report_state())
    import dataclasses

    with pytest.raises(CalibrationError, match="CalibrationMetrics"):
        dataclasses.replace(report, before={"nll": 0.5})


def test_large_float32_vocabularies_pass_the_row_sum_check():
    probabilities = torch.softmax(torch.randn(4, 152064, generator=torch.Generator().manual_seed(0)) * 2, -1)
    assert negative_log_likelihood(probabilities, [0, 1, 2, 3]) > 0


def test_report_split_ids_must_be_consistent():
    state = _report_state()
    state["split"].update(calibration="test", test="test", evaluation="holdout")
    with pytest.raises(CalibrationError, match="must differ"):
        CalibrationReport.from_state(state)


def test_uint64_targets_report_the_values_given():
    with pytest.raises(CalibrationError, match=str(2**63)):
        brier_score(np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0, 2**63], dtype=np.uint64))


def test_model_fingerprint_is_byte_order_independent(monkeypatch):
    values = torch.tensor([1.5, -2.25, 3.0])
    little = torch.nn.Module()
    little.register_buffer("w", values)
    expected = model_fingerprint(little)
    # Simulate a big-endian host: the same values stored with each element's bytes reversed.
    swapped = values.view(torch.uint8).reshape(-1, 4).flip(-1).contiguous().view(torch.float32).reshape(3)
    big = torch.nn.Module()
    big.register_buffer("w", swapped)
    monkeypatch.setattr(calibration.sys, "byteorder", "big")
    assert model_fingerprint(big) == expected


@pytest.mark.filterwarnings("ignore:torch.quantize_per_tensor:UserWarning")  # torch's own deprecation notice
def test_model_fingerprint_hashes_quantized_values_not_just_integers():
    x = torch.tensor([0.1, 0.2, 0.3])
    first, second = torch.nn.Module(), torch.nn.Module()
    first.register_buffer("q", torch.quantize_per_tensor(x, 0.1, 0, torch.qint8))
    second.register_buffer("q", torch.quantize_per_tensor(2 * x, 0.2, 0, torch.qint8))
    assert torch.equal(first.q.int_repr(), second.q.int_repr())
    assert model_fingerprint(first) != model_fingerprint(second)


def test_ece_and_target_checks_on_iterables_and_shapes():
    bins = reliability_bins(P, Y, n_bins=5)
    assert expected_calibration_error(b for b in bins) == expected_calibration_error(bins)
    with pytest.raises(CalibrationError, match="one target per row"):
        _fit(targets=np.full((4, 1), 5, dtype=np.uint8))


@pytest.mark.filterwarnings("ignore:torch.quantize_per_tensor:UserWarning")
def test_model_fingerprint_separates_quantization_settings_with_equal_values():
    first, second = torch.nn.Module(), torch.nn.Module()
    first.register_buffer("q", torch.quantize_per_tensor(torch.tensor([1.0]), 0.1, 0, torch.qint8))
    second.register_buffer("q", torch.quantize_per_tensor(torch.tensor([1.0]), 0.2, 0, torch.qint8))
    assert torch.equal(first.q.dequantize(), second.q.dequantize())
    assert model_fingerprint(first) != model_fingerprint(second)


def test_final_consistency_checks():
    with pytest.raises(CalibrationError, match="n_classes"):
        TemperatureCalibrator(temperature=1.0, labels=BINARY, model_id="m", split_id="c", fit_result={"n_classes": 5})
    with pytest.raises(CalibrationError, match="status"):
        calibration.CalibrationFit("ok", None, None, {}, {})
    calibrated = _calibrator().transform(K_LOGITS, labels=BINARY, model_id="run-1:BEST")
    with pytest.raises(CalibrationError, match="already calibrated"):
        _calibrator().transform(calibrated, model_id="run-1:BEST")
