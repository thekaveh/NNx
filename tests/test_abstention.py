"""FEAT-008: abstention policies and risk-coverage evaluation (``nnx.abstention``).

Pure policies over categorical probabilities ``[N, C]`` accept a row when its
score (top probability, or top-two margin) reaches a threshold and abstain
otherwise — never on malformed input. Reports use two denominators
(coverage over every row, risk over accepted rows); thresholds are chosen on
a named validation split against an empirical risk ceiling.
"""

from __future__ import annotations

import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from nnx import abstention
from nnx.abstention import (
    AbstentionError,
    AbstentionPolicy,
    AbstentionSchemaError,
    CoverageAccumulator,
    CoverageReport,
    decide,
    risk_coverage_curve,
    select_threshold,
)

LABELS = ("a", "b", "c")
# Fixture A: max probability / top-two margin / prediction (label order breaks ties) / correct?
#   row 0: .7 / .5  / a / yes      row 1: .4 / 0 (tied top two) / a / no (true b)
#   row 2: .6 / .3  / c / yes      row 3: .34 / .01 / a / no (true c)
P = np.array([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [0.1, 0.3, 0.6], [0.34, 0.33, 0.33]])
Y = np.array([0, 1, 2, 2])


def _policy(kind="max_probability", threshold=0.6, **overrides) -> AbstentionPolicy:
    kwargs = {"labels": LABELS, "model_id": "m", "tuning_split_id": "val"}
    kwargs.update(overrides)
    return AbstentionPolicy(kind=kind, threshold=threshold, **kwargs)


def _apply(policy, probabilities=P, **kwargs):
    kwargs.setdefault("labels", LABELS)
    kwargs.setdefault("model_id", "m")
    return policy.apply(probabilities, **kwargs)


# ------------------------------------------------------------------- policies (AC1 / AC2)


def test_max_probability_accepts_at_or_above_the_threshold():
    result = _apply(_policy(threshold=0.6))
    assert result.accepted.tolist() == [True, False, True, False]  # 0.6 >= 0.6 is accepted
    assert result.score.tolist() == [0.7, 0.4, 0.6, 0.34]
    assert _apply(_policy(threshold=0.7)).accepted.tolist() == [True, False, False, False]


def test_margin_uses_the_top_two_gap_and_a_tie_gives_zero():
    result = _apply(_policy("margin", 0.25))
    np.testing.assert_allclose(result.score, [0.5, 0.0, 0.3, 0.01])
    assert result.score[1] == 0.0  # tied top two
    assert result.accepted.tolist() == [True, False, True, False]
    assert _apply(_policy("margin", 0.0)).accepted.all()  # a zero threshold accepts a tie


def test_ties_break_by_label_order():
    result = _apply(_policy())
    assert result.prediction.tolist() == [0, 0, 2, 0]  # row 1 ties a / b: the first label wins
    assert [o.prediction for o in result.outcomes()] == ["a", "a", "c", "a"]


@pytest.mark.parametrize(
    ("probabilities", "match"),
    [
        (np.array([[0.5, 0.6]]), "sum to 1"),
        (np.array([[math.nan, 1.0]]), "non-finite"),
        (np.array([[1.5, -0.5]]), r"\[0, 1\]"),
        (np.array([0.5, 0.5]), "2-D"),
        (np.array([[1.0]]), "at least 2"),
    ],
)
def test_malformed_probabilities_raise_instead_of_abstaining(probabilities, match):
    policy = AbstentionPolicy(
        kind="max_probability", threshold=0.5, labels=("x", "y"), model_id="m", tuning_split_id="v"
    )
    with pytest.raises(AbstentionError, match=match):
        policy.apply(probabilities, labels=("x", "y"), model_id="m")


def test_outcomes_keep_prediction_distribution_score_id_and_reason():
    before = P.copy()
    result = _apply(_policy(), sample_ids=[10, 11, 12, 13])
    np.testing.assert_array_equal(P, before)  # inputs unchanged
    accepted, abstained = result.outcomes()[0], result.outcomes()[1]
    assert (accepted.status, accepted.reason, accepted.sample_id) == ("accepted", None, 10)
    assert accepted.distribution == (("a", 0.7), ("b", 0.2), ("c", 0.1)) and accepted.score == 0.7
    assert (abstained.status, abstained.reason, abstained.sample_id) == ("abstained", "below_probability_threshold", 11)
    assert abstained.prediction == "a" and abstained.distribution == (("a", 0.4), ("b", 0.4), ("c", 0.2))
    assert _apply(_policy("margin", 0.25)).outcomes()[1].reason == "small_margin"
    assert result.reasons == (None, "below_probability_threshold", None, "below_probability_threshold")
    assert result.accepted_ids.tolist() == [10, 12] and result.abstained_ids.tolist() == [11, 13]
    assert not np.shares_memory(result.probabilities, P)


# --------------------------------------------------------- serialization and schema (AC3)


def test_policy_json_round_trip_records_its_declarations(tmp_path):
    policy = _policy(input_field="calibrated_probabilities", calibrator_id="sha256:cal", threshold=0.75)
    state = json.loads(policy.to_json())
    assert state["format"] == abstention.FORMAT
    assert state["input_field"] == "calibrated_probabilities" and state["threshold"] == 0.75
    assert state["model_id"] == "m" and state["calibrator_id"] == "sha256:cal" and state["tuning_split_id"] == "val"
    assert AbstentionPolicy.from_json(policy.to_json()) == policy
    policy.save(tmp_path / "policy.json")
    reloaded = AbstentionPolicy.load(tmp_path / "policy.json")
    assert reloaded == policy and reloaded.id == policy.id


@pytest.mark.parametrize(
    "declared",
    [
        {"labels": ("b", "a", "c")},
        {"model_id": "other"},
        {"input_field": "calibrated_probabilities", "calibrator_id": "sha256:x"},
    ],
)
def test_an_incompatible_schema_raises_before_reading(declared):
    class Spy:
        reads = 0

        def __array__(self, dtype=None, copy=None):
            Spy.reads += 1
            return P

    kwargs = {"labels": LABELS, "model_id": "m", **declared}
    with pytest.raises(AbstentionSchemaError):
        _policy().apply(Spy(), **kwargs)
    assert Spy.reads == 0


def test_a_calibrated_policy_needs_the_same_calibrator():
    policy = _policy(input_field="calibrated_probabilities", calibrator_id="sha256:one")
    with pytest.raises(AbstentionSchemaError, match="calibrator"):
        policy.apply(P, labels=LABELS, model_id="m", input_field="calibrated_probabilities", calibrator_id="sha256:two")
    with pytest.raises(AbstentionError, match="calibrator_id"):
        _policy(input_field="calibrated_probabilities")  # a calibrated field names its calibrator
    with pytest.raises(AbstentionError, match="calibrator_id"):
        _policy(calibrator_id="sha256:one")  # raw probabilities have no calibrator


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s: s.update(format="nnx.abstention/0"), "format"),
        (lambda s: s.update(threshold=None), "threshold"),
        (lambda s: s.update(threshold="inf"), "threshold"),
        (lambda s: s.update(threshold=1.5), "threshold"),
        (lambda s: s.update(kind="entropy"), "kind"),
        (lambda s: s.update(input_field="logits"), "input_field"),
        (lambda s: s.update(extra=1), "unknown"),
        (lambda s: s.update(labels=["a"]), "2 labels"),
    ],
)
def test_loading_rejects_malformed_or_undeployable_policies(mutate, match):
    state = _policy().state()
    mutate(state)
    with pytest.raises(AbstentionError, match=match):
        AbstentionPolicy.from_state(state)


def test_threshold_one_accepts_a_score_of_one():
    policy = _policy(threshold=1.0)
    result = policy.apply(np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]), labels=LABELS, model_id="m")
    assert result.accepted.tolist() == [True, False]
    assert AbstentionPolicy.from_json(policy.to_json()).threshold == 1.0


def test_prediction_and_calibrated_sources_keep_their_identity():
    from nnx.calibration import TemperatureCalibrator
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    logits = np.log(P)
    raw = prediction_from_logits(logits, ProbabilitySpec("categorical", labels=LABELS), sample_ids=[5, 6, 7, 8])
    result = _policy().apply(raw, model_id="m")
    assert result.input_field == "probabilities" and result.calibrator_id is None
    assert result.sample_ids.tolist() == [5, 6, 7, 8]
    calibrator = TemperatureCalibrator(temperature=2.0, labels=LABELS, model_id="m", split_id="calib")
    calibrated = calibrator.transform(raw, model_id="m")
    on_calibrated = _policy(input_field="calibrated_probabilities", calibrator_id=calibrator.id).apply(calibrated)
    np.testing.assert_array_equal(on_calibrated.probabilities, calibrated.calibrated_probabilities)
    assert on_calibrated.calibrator_id == calibrator.id and on_calibrated.input_field == "calibrated_probabilities"
    on_raw = _policy().apply(calibrated)  # a raw-field policy reads the raw view of a calibrated prediction
    np.testing.assert_array_equal(on_raw.probabilities, calibrated.probabilities)
    with pytest.raises(AbstentionSchemaError, match="input_field"):
        _policy(input_field="calibrated_probabilities", calibrator_id=calibrator.id).apply(raw, model_id="m")
    with pytest.raises(AbstentionError, match="categorical"):
        _policy().apply(prediction_from_logits(logits, ProbabilitySpec("bernoulli", labels=LABELS)), model_id="m")


def test_decision_results_keep_abstention_a_success():
    from nnx.decisions import BooleanResult, ChoiceResult

    confident = ChoiceResult(question_digest="q", distribution=(("a", 0.8), ("b", 0.15), ("c", 0.05)))
    unsure = ChoiceResult(question_digest="q", distribution=(("a", 0.4), ("b", 0.35), ("c", 0.25)))
    policy = _policy(threshold=0.6)
    kept = decide(confident, policy, model_id="m")
    held = decide(unsure, policy, model_id="m")
    assert kept.accepted and not kept.abstained and kept.outcome.prediction == "a"
    assert held.abstained and held.outcome.reason == "below_probability_threshold"
    assert held.result is unsure and held.result.distribution == unsure.distribution  # a success, unchanged
    assert held.input_field == "probabilities" and held.calibrator_id is None and held.policy_id == policy.id
    with pytest.raises(AbstentionSchemaError):
        decide(
            ChoiceResult(question_digest="q", distribution=(("b", 0.5), ("a", 0.3), ("c", 0.2))), policy, model_id="m"
        )
    with pytest.raises(AbstentionSchemaError, match="input_field"):
        decide(confident, _policy(input_field="calibrated_probabilities", calibrator_id="sha256:c"), model_id="m")
    with pytest.raises(TypeError, match="Choice"):
        decide(BooleanResult(question_digest="q", p_true=0.9), policy, model_id="m")


# ------------------------------------------------------------------ reports (AC4 / AC7)


def test_report_uses_two_denominators():
    report = _apply(_policy(threshold=0.4)).report(Y)
    assert (report.total, report.accepted, report.incorrect) == (4, 3, 1)
    assert report.coverage == 0.75 and report.risk == pytest.approx(1 / 3) and report.status == "ok"


def test_zero_accepted_reports_risk_unavailable_never_zero():
    report = _apply(_policy(threshold=0.95)).report(Y)
    assert report.accepted == 0 and report.coverage == 0.0
    assert report.risk is None and report.status == "no_accepted"
    assert "risk unavailable" in report.summary()
    assert CoverageReport.from_json(report.to_json()) == report


def test_hard_label_consumers_see_only_accepted_rows():
    from nnx.vis_utils import VisUtils

    result = _apply(_policy(threshold=0.6), sample_ids=[10, 11, 12, 13])
    rows = result.accepted_rows(Y)
    assert rows.sample_ids.tolist() == [10, 12]
    assert rows.targets.tolist() == [0, 2] and rows.predictions.tolist() == [0, 2]
    table = VisUtils.classification_report(rows.targets, rows.predictions)
    classes = {index for index in table.index if index not in ("accuracy", "macro avg", "weighted avg")}
    assert classes <= {"0", "1", "2"}  # never a rejection pseudo-class
    report = result.report(Y)
    assert report.sample_ids.tolist() == [10, 11, 12, 13] and report.total == 4  # every original id is kept
    assert report.accepted_ids.tolist() == [10, 12]


def test_chunked_aggregation_matches_one_eager_report():
    policy = _policy(threshold=0.6)
    eager = _apply(policy, sample_ids=[0, 1, 2, 3]).report(Y)
    accumulator = CoverageAccumulator()
    for rows in ([0], [1, 2], [3]):
        chunk = policy.apply(P[rows], labels=LABELS, model_id="m", sample_ids=rows)
        accumulator.update(chunk, Y[rows])
    assert accumulator.report() == eager
    none = CoverageAccumulator()
    for rows in ([1], [3]):  # only rejected rows
        none.update(policy.apply(P[rows], labels=LABELS, model_id="m", sample_ids=rows), Y[rows])
    eager_none = policy.apply(P[[1, 3]], labels=LABELS, model_id="m", sample_ids=[1, 3]).report(Y[[1, 3]])
    assert none.report() == eager_none and eager_none.status == "no_accepted" and eager_none.risk is None
    assert CoverageAccumulator().report().status == "empty"


# ------------------------------------------------------------ threshold selection (AC5 / AC6)


def test_selection_runs_on_a_named_validation_split_and_reports_candidates():
    selection = select_threshold(
        P,
        Y,
        kind="max_probability",
        risk_ceiling=0.1,
        split_id="val",
        test_split_id="test",
        labels=LABELS,
        model_id="m",
    )
    assert selection.status == "ok" and selection.policy is not None
    assert selection.policy.threshold == 0.6 and selection.policy.tuning_split_id == "val"
    points = [(p.threshold, p.accepted, p.incorrect) for p in selection.points]
    assert points == [(0.34, 4, 2), (0.4, 3, 1), (0.6, 2, 0), (0.7, 1, 0), (None, 0, 0)]
    assert selection.points[-1].status == "accept_none" and selection.points[-1].risk is None
    assert selection.basis == "empirical"
    loose = select_threshold(
        P, Y, kind="max_probability", risk_ceiling=0.4, split_id="val", labels=LABELS, model_id="m"
    )
    assert loose.policy.threshold == 0.4


def test_selection_returns_no_feasible_policy_and_refuses_the_test_split():
    wrong = np.array([[0.9, 0.1], [0.8, 0.2]])
    selection = select_threshold(
        wrong,
        np.array([1, 1]),
        kind="max_probability",
        risk_ceiling=0.0,
        split_id="val",
        labels=("x", "y"),
        model_id="m",
    )
    assert selection.status == "no_feasible_policy" and selection.policy is None
    with pytest.raises(AbstentionError, match="no feasible"):
        selection.require()
    with pytest.raises(AbstentionError, match="test split"):
        select_threshold(
            P, Y, kind="margin", risk_ceiling=0.2, split_id="test", test_split_id="test", labels=LABELS, model_id="m"
        )
    with pytest.raises(AbstentionError, match="split_id"):
        select_threshold(P, Y, kind="margin", risk_ceiling=0.2, split_id="", labels=LABELS, model_id="m")
    with pytest.raises(AbstentionError, match="risk_ceiling"):
        select_threshold(P, Y, kind="margin", risk_ceiling=1.5, split_id="val", labels=LABELS, model_id="m")


def test_coverage_is_monotone_while_risk_is_measured():
    top = np.array([0.55, 0.65, 0.75, 0.85, 0.95])
    probabilities = np.stack([top, 1 - top], axis=1)
    targets = np.array([0, 1, 0, 0, 1])  # the 0.65 and 0.95 rows are wrong
    curve = risk_coverage_curve(probabilities, targets, kind="max_probability")
    coverages = [point.coverage for point in curve]
    assert coverages == sorted(coverages, reverse=True)  # never rises as the threshold rises
    risks = [point.risk for point in curve[:-1]]
    assert risks == pytest.approx([0.4, 0.5, 1 / 3, 0.5, 1.0])  # measured: not monotone
    assert curve[-1].threshold is None and curve[-1].coverage == 0.0 and not curve[-1].deployable


def test_the_accept_none_endpoint_is_curve_only():
    curve = risk_coverage_curve(P, Y, kind="margin")
    endpoint = curve[-1]
    assert endpoint.status == "accept_none" and endpoint.threshold is None
    with pytest.raises(AbstentionError, match="threshold"):
        AbstentionPolicy.from_state({**_policy().state(), "threshold": endpoint.threshold})


def test_selection_accepts_explicit_candidates_and_serializes():
    selection = select_threshold(
        P,
        Y,
        kind="max_probability",
        risk_ceiling=0.1,
        split_id="val",
        labels=LABELS,
        model_id="m",
        candidates=[0.5, 0.65],
    )
    assert [p.threshold for p in selection.points] == [0.5, 0.65, None]
    assert selection.policy.threshold == 0.5  # both are feasible; 0.5 covers more
    state = json.loads(selection.to_json())
    assert state["status"] == "ok" and state["risk_ceiling"] == 0.1 and state["split_id"] == "val"
    with pytest.raises(AbstentionError, match="candidates"):
        select_threshold(
            P, Y, kind="max_probability", risk_ceiling=0.1, split_id="v", labels=LABELS, model_id="m", candidates=[2.0]
        )


def test_the_duck_typed_prediction_path_needs_a_model_id():
    duck = SimpleNamespace(
        probabilities=P, spec=SimpleNamespace(kind="categorical", labels=LABELS), sample_ids=np.arange(4)
    )
    with pytest.raises(AbstentionError, match="model_id"):
        _policy().apply(duck)


def test_public_surface_is_complete():
    import nnx

    assert nnx.abstention is abstention and "abstention" in nnx.__all__
    for name in abstention.__all__:
        assert getattr(abstention, name, None) is not None, name


def test_numpy_label_arrays_are_accepted():
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    raw = prediction_from_logits(np.log(P), ProbabilitySpec("categorical"))
    result = _policy().apply(raw, labels=np.array(LABELS), model_id="m")
    assert result.labels == LABELS and all(type(label) is str for label in result.labels)
    with pytest.raises(AbstentionError, match="labels"):
        _policy().apply(raw, model_id="m")  # an unlabelled prediction must name its labels


def test_scores_helper_and_empty_input():
    prediction, score = abstention.scores(P, kind="margin")
    assert prediction.tolist() == [0, 0, 2, 0] and score[1] == 0.0
    empty = _policy().apply(np.zeros((0, 3)), labels=LABELS, model_id="m")
    report = empty.report(np.zeros(0, dtype=int))
    assert report.status == "empty" and report.coverage is None and report.risk is None


# ------------------------------------------------------------------------ review regressions


def test_accumulator_checks_the_policy_from_the_first_chunk_even_when_empty():
    accumulator = CoverageAccumulator()
    accumulator.update(_policy().apply(P[:0], labels=LABELS, model_id="m"), Y[:0])
    with pytest.raises(AbstentionError, match="policy"):
        accumulator.update(_policy(threshold=0.4).apply(P, labels=LABELS, model_id="m"), Y)


def test_accumulated_chunks_need_unique_sample_ids():
    policy = _policy()
    accumulator = CoverageAccumulator()
    accumulator.update(policy.apply(P[[0, 1]], labels=LABELS, model_id="m"), Y[[0, 1]])
    with pytest.raises(AbstentionError, match="sample_ids"):  # default ids 0..1 again: refused at update
        accumulator.update(policy.apply(P[[2, 3]], labels=LABELS, model_id="m"), Y[[2, 3]])
    accumulator.update(policy.apply(P[[2, 3]], labels=LABELS, model_id="m", sample_ids=[2, 3]), Y[[2, 3]])
    assert accumulator.report() == _apply(policy, sample_ids=[0, 1, 2, 3]).report(Y)  # still usable


def test_a_raw_prediction_cannot_pass_as_calibrated():
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    raw = prediction_from_logits(np.log(P), ProbabilitySpec("categorical", labels=LABELS))
    policy = _policy(input_field="calibrated_probabilities", calibrator_id="sha256:x")
    with pytest.raises(AbstentionSchemaError, match="raw probabilities only"):
        policy.apply(raw, model_id="m", input_field="calibrated_probabilities", calibrator_id="sha256:x")


def test_a_calibrated_prediction_refuses_contradicting_keywords():
    from nnx.calibration import TemperatureCalibrator

    calibrator = TemperatureCalibrator(temperature=2.0, labels=LABELS, model_id="m", split_id="calib")
    calibrated = calibrator.transform(np.log(P), labels=LABELS, model_id="m")
    policy = _policy(input_field="calibrated_probabilities", calibrator_id=calibrator.id)
    with pytest.raises(AbstentionSchemaError, match="sample_ids"):
        policy.apply(calibrated, sample_ids=[100, 101, 102, 103])
    with pytest.raises(AbstentionSchemaError, match="calibrator_id"):
        policy.apply(calibrated, calibrator_id="sha256:other")
    assert policy.apply(calibrated, sample_ids=[0, 1, 2, 3]).calibrator_id == calibrator.id


def test_selection_on_a_calibrated_prediction_names_its_view():
    from nnx.calibration import TemperatureCalibrator

    calibrator = TemperatureCalibrator(temperature=0.5, labels=LABELS, model_id="m", split_id="calib")
    calibrated = calibrator.transform(np.log(P), labels=LABELS, model_id="m")
    with pytest.raises(AbstentionError, match="input_field"):
        select_threshold(calibrated, Y, kind="max_probability", risk_ceiling=0.1, split_id="val")
    selection = select_threshold(
        calibrated, Y, kind="max_probability", risk_ceiling=0.1, split_id="val", input_field="calibrated_probabilities"
    )
    policy = selection.require()
    assert policy.input_field == "calibrated_probabilities" and policy.calibrator_id == calibrator.id
    assert policy.apply(calibrated).accepted.any()


def test_selections_reload_and_stay_consistent(tmp_path):
    selection = select_threshold(P, Y, kind="margin", risk_ceiling=0.2, split_id="val", labels=LABELS, model_id="m")
    selection.save(tmp_path / "selection.json")
    reloaded = abstention.ThresholdSelection.load(tmp_path / "selection.json")
    assert reloaded == selection and reloaded.policy == selection.policy
    import dataclasses

    with pytest.raises(AbstentionError, match="policy"):
        dataclasses.replace(selection, status="ok", policy=None)
    with pytest.raises(AbstentionError, match="policy"):
        dataclasses.replace(selection, status="no_feasible_policy")
    with pytest.raises(AbstentionError):
        dataclasses.replace(selection, points=selection.points[:-1])  # the accept-none endpoint is required


@pytest.mark.parametrize("ids", [[0.7, 1.2], None, [2**64 - 1]])
def test_reports_reject_non_integer_or_overflowing_ids(ids):
    state = _apply(_policy(), sample_ids=[0, 1, 2, 3]).report(Y).state()
    state["sample_ids"] = ids
    with pytest.raises(AbstentionError):
        CoverageReport.from_state(state)
    with pytest.raises(AbstentionError):
        CoverageReport(
            policy_id=None,
            total=1,
            accepted=0,
            incorrect=0,
            sample_ids=np.array([2**64 - 1], dtype=np.uint64),
            accepted_ids=[],
        )


def test_prediction_results_keep_their_own_sample_ids():
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    raw = prediction_from_logits(np.log(P), ProbabilitySpec("categorical", labels=LABELS), sample_ids=[5, 6, 7, 8])
    with pytest.raises(AbstentionSchemaError, match="sample_ids"):
        _policy().apply(raw, model_id="m", sample_ids=[10, 11, 12, 13])
    assert _policy().apply(raw, model_id="m", sample_ids=[5, 6, 7, 8]).sample_ids.tolist() == [5, 6, 7, 8]


def test_empty_id_lists_duplicates_and_row_identity():
    empty = _policy().apply(np.zeros((0, 3)), labels=LABELS, model_id="m", sample_ids=[])
    assert empty.report(np.zeros(0, dtype=int)).status == "empty"
    assert (
        CoverageReport(policy_id=None, total=0, accepted=0, incorrect=0, sample_ids=[], accepted_ids=[]).status
        == "empty"
    )
    repeated = _apply(_policy(), sample_ids=[3, 3, 4, 5])  # an oversampled split: each row decided on its own
    assert repeated.sample_ids.tolist() == [3, 3, 4, 5] and repeated.report(Y).sample_ids.tolist() == [3, 3, 4, 5]
    rows = _apply(_policy()).accepted_rows(Y)
    assert rows == rows and rows != _apply(_policy()).accepted_rows(Y)  # identity equality, never an array truth value
    assert hash(rows) == hash(rows)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["points"][0].update(coverage=0.123),
        lambda s: s["points"][0].update(bogus=1),
        lambda s: s["policy"].update(threshold=0.34),  # a threshold that is not the best feasible one
        lambda s: s.update(status="no_feasible_policy", policy=None),
    ],
)
def test_a_tampered_selection_does_not_load(mutate):
    selection = select_threshold(
        P, Y, kind="max_probability", risk_ceiling=0.1, split_id="val", labels=LABELS, model_id="m"
    )
    state = json.loads(selection.to_json())
    mutate(state)
    with pytest.raises(AbstentionError):
        abstention.ThresholdSelection.from_state(state)


def test_results_are_read_only_and_the_accumulator_keeps_copies():
    result = _apply(_policy(), sample_ids=[0, 1, 2, 3])
    for array in (result.sample_ids, result.probabilities, result.score, result.accepted, result.prediction):
        with pytest.raises(ValueError):
            array[...] = 0
    accumulator = CoverageAccumulator()
    ids = np.array([0, 1])
    accumulator.update(_policy().apply(P[:2], labels=LABELS, model_id="m", sample_ids=ids), Y[:2])
    ids[:] = [7, 8]  # the caller's buffer changes afterwards
    assert accumulator.report().sample_ids.tolist() == [0, 1]


@pytest.mark.parametrize(
    "points",
    [
        lambda pts: (abstention.CurvePoint(None, 0, 0, 4), *pts),  # accept-none in the middle
        lambda pts: (pts[1], pts[0], *pts[2:]),  # thresholds out of order
        lambda pts: (abstention.CurvePoint(0.1, 5, 2, 5), *pts[1:]),  # another total
    ],
)
def test_a_selection_curve_must_be_well_formed(points):
    import dataclasses

    selection = select_threshold(
        P, Y, kind="max_probability", risk_ceiling=0.5, split_id="val", labels=LABELS, model_id="m"
    )
    with pytest.raises(AbstentionError):
        dataclasses.replace(selection, points=points(selection.points))


def test_arrays_default_to_the_policy_field():
    policy = _policy(input_field="calibrated_probabilities", calibrator_id="sha256:c")
    result = policy.apply(P, labels=LABELS, model_id="m", calibrator_id="sha256:c")
    assert result.input_field == "calibrated_probabilities"
    with pytest.raises(AbstentionError, match="calibrator"):
        policy.apply(P, labels=LABELS, model_id="m")  # a calibrated field names its calibrator


def test_calibrated_predictions_keep_their_decoded_class_and_override():
    from nnx.calibration import TemperatureCalibrator

    calibrator = TemperatureCalibrator(temperature=1.0, labels=("a", "b"), model_id="m", split_id="calib")
    tied = calibrator.transform(np.array([[0.0, 1e-17], [3.0, 0.0]]), labels=("a", "b"), model_id="m")
    assert tied.decoded.tolist() == [1, 0] and tied.probabilities[0, 0] == tied.probabilities[0, 1]
    policy = AbstentionPolicy("max_probability", 0.4, labels=("a", "b"), model_id="m", tuning_split_id="v")
    result = policy.apply(tied)
    assert result.prediction.tolist() == [1, 0]  # the prediction's own decoded class, not a rounding tie's
    assert [o.prediction for o in result.outcomes()] == ["b", "a"]
    overridden = calibrator.transform(
        np.log(P[:, :2] / P[:, :2].sum(1, keepdims=True)), labels=("a", "b"), model_id="n", override="audit"
    )
    calibrated_policy = AbstentionPolicy(
        "max_probability",
        0.5,
        labels=("a", "b"),
        model_id="n",
        tuning_split_id="v",
        input_field="calibrated_probabilities",
        calibrator_id=calibrator.id,
    )
    kept = calibrated_policy.apply(overridden)
    assert kept.calibration_override is not None and kept.calibration_override["name"] == "audit"


def test_round_four_edge_cases():
    from nnx.decisions import ChoiceResult

    with pytest.raises(AbstentionError, match="candidates"):
        risk_coverage_curve(P, Y, kind="max_probability", candidates=np.array(0.5))
    with pytest.raises(AbstentionSchemaError, match="calibrator"):
        _policy().apply(P, labels=LABELS, model_id="m", calibrator_id="sha256:x")  # raw arrays have no calibrator
    near = ChoiceResult(question_digest="q", distribution=(("a", 0.7 + 9.9e-7), ("b", 0.2), ("c", 0.1)))
    assert decide(near, _policy(), model_id="m").accepted  # validated by nnx.decisions: scored as is


def test_round_five_candidates_ids_and_types():
    points = risk_coverage_curve(P, Y, kind="max_probability", candidates=(t for t in (0.6, 0.4)))
    assert [p.threshold for p in points] == [0.4, 0.6, None]  # a generator is read once, then sorted
    from_set = risk_coverage_curve(P, Y, kind="max_probability", candidates={0.4, 0.6})
    assert [p.threshold for p in from_set] == [0.4, 0.6, None]
    with pytest.raises(AbstentionError, match="candidates"):
        risk_coverage_curve(P, Y, kind="max_probability", candidates=[[0.4, 0.6]])
    with pytest.raises(AbstentionError, match="sample_ids"):
        select_threshold(
            P,
            Y,
            kind="max_probability",
            risk_ceiling=0.1,
            split_id="val",
            labels=LABELS,
            model_id="m",
            sample_ids=[1, 2],
        )
    with pytest.raises(TypeError):
        decide(SimpleNamespace(kind="choice", distribution=(("a", 1.0),)), _policy(), model_id="m")


def test_round_five_threshold_messages():
    with pytest.raises(AbstentionError, match="accept-none"):
        _policy(threshold=None)
    with pytest.raises(AbstentionError) as caught:
        _policy(threshold="abc")
    assert "accept-none" not in str(caught.value)
    with pytest.raises(AbstentionError, match=r"\[0, 1\]"):
        _policy(threshold=1.5)


def test_round_five_prediction_ids_of_another_length_contradict():
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    prediction = prediction_from_logits(
        np.log(P), ProbabilitySpec("categorical", labels=LABELS), sample_ids=[5, 6, 7, 8]
    )
    with pytest.raises(AbstentionSchemaError, match="sample_ids"):
        _policy().apply(prediction, model_id="m", sample_ids=[5, 6])


def _duck(decoded, probabilities=P):
    from nnx.prediction import ProbabilitySpec

    return SimpleNamespace(
        logits=np.log(probabilities),
        probabilities=probabilities,
        decoded=np.asarray(decoded),
        spec=ProbabilitySpec("categorical", labels=LABELS),
        sample_ids=np.arange(len(probabilities)),
    )


@pytest.mark.parametrize(
    ("decoded", "match"),
    [
        ([0, -1, 2, 0], r"in \[0, 3\)"),
        ([0, 5, 2, 0], r"in \[0, 3\)"),
        ([0.0, 0.0, 2.0, 0.0], "integers"),
        ([0, 0, 2], "integers"),
        ([0, 0, 0, 0], "maximal probability"),  # row 2's class 0 is not its top class
    ],
)
def test_round_six_a_prediction_decoded_class_is_validated(decoded, match):
    with pytest.raises(AbstentionError, match=match):
        _policy().apply(_duck(decoded), model_id="m")
    with pytest.raises(AbstentionError, match=match):
        select_threshold(_duck(decoded), Y, kind="margin", risk_ceiling=0.5, split_id="val", model_id="m")


def test_round_six_a_tied_decoded_class_keeps_a_zero_margin():
    result = _policy("margin", 0.0).apply(_duck([0, 1, 2, 0]), model_id="m")  # row 1: the tied second class
    assert result.prediction.tolist() == [0, 1, 2, 0]
    assert result.score[1] == 0.0 and (result.score >= 0).all()
    selection = select_threshold(_duck([0, 1, 2, 0]), Y, kind="margin", risk_ceiling=0.5, split_id="val", model_id="m")
    assert selection.points[0].threshold == 0.0 and selection.points[0].incorrect == 1  # row 1 decoded b: correct


def test_round_six_a_prediction_duck_needs_its_probabilities():
    duck = _duck([0, 0, 2, 0])
    del duck.decoded
    with pytest.raises(AbstentionError, match="decoded"):
        _policy().apply(duck, model_id="m")


@pytest.mark.parametrize("candidates", [[[0.5], [0.1, 0.2]], [[0.5]], ["0.5"], np.array([[0.5]])])
def test_round_six_nested_or_ragged_candidates_raise_abstention_errors(candidates):
    with pytest.raises(AbstentionError, match="flat list"):
        risk_coverage_curve(P, Y, kind="max_probability", candidates=candidates)


def test_round_six_the_accumulator_counts_as_the_eager_report():
    policy = _policy(threshold=0.5)
    accumulator = CoverageAccumulator()
    for rows, ids in ((slice(0, 2), [10, 11]), (slice(2, 4), [12, 13])):
        accumulator.update(_apply(policy, P[rows], sample_ids=ids), Y[rows])
    eager = _apply(policy, sample_ids=[10, 11, 12, 13]).report(Y)
    assert accumulator.report() == eager


def test_round_seven_negative_zero_is_one_threshold():
    assert _policy(threshold=-0.0) == _policy(threshold=0.0)
    assert '"threshold":0.0' in _policy(threshold=-0.0).canonical_bytes().decode()
    selection = select_threshold(P, Y, kind="margin", risk_ceiling=-0.0, split_id="val", labels=LABELS, model_id="m")
    assert math.copysign(1.0, selection.risk_ceiling) == 1.0


def test_round_seven_selection_counts_never_rise_with_the_threshold():
    from nnx.abstention import CurvePoint

    selection = select_threshold(
        P, Y, kind="max_probability", risk_ceiling=0.0, split_id="val", labels=LABELS, model_id="m"
    )
    rising = (CurvePoint(0.2, 1, 0, 4), CurvePoint(0.5, 3, 0, 4), CurvePoint(None, 0, 0, 4))
    with pytest.raises(AbstentionError, match="never rise"):
        dataclasses.replace(selection, points=rising, policy=_policy(threshold=0.5), status="ok")
    state = selection.state()
    state["points"][2].update(accepted=3, incorrect=2)  # more errors at 0.6 than at 0.4 (3 / 1)
    with pytest.raises(AbstentionError):
        abstention.ThresholdSelection.from_state(state)


def test_round_seven_decide_needs_a_policy():
    from nnx.decisions import ChoiceResult

    choice = ChoiceResult(question_digest="q", distribution=(("a", 0.7), ("b", 0.2), ("c", 0.1)))
    selection = select_threshold(
        P, Y, kind="max_probability", risk_ceiling=0.0, split_id="val", labels=LABELS, model_id="m"
    )
    with pytest.raises(TypeError, match="selection.require"):
        decide(choice, selection, model_id="m")


def test_round_seven_bare_calibrated_arrays_name_the_missing_calibrator():
    policy = _policy(input_field="calibrated_probabilities", calibrator_id="sha256:c")
    with pytest.raises(AbstentionSchemaError, match="policy's field, as input_field= was not given"):
        policy.apply(P, labels=LABELS, model_id="m")
    with pytest.raises(AbstentionSchemaError, match="must declare calibrator_id"):
        policy.apply(P, labels=LABELS, model_id="m", input_field="calibrated_probabilities")


def test_round_seven_decoded_ties_allow_the_inputs_own_rounding():
    near_tie = np.array([[0.5, 0.4995, 0.0005], [0.25, 0.7, 0.05]], dtype=np.float16)  # row 0: 4.9e-4 apart
    duck = _duck([1, 1], near_tie)
    result = _policy("margin", 0.0).apply(duck, model_id="m")
    assert result.prediction.tolist() == [1, 1] and result.score[0] == 0.0
    with pytest.raises(AbstentionError, match="maximal probability"):
        _policy().apply(_duck([1, 1], np.array([[0.5, 0.499, 0.001], [0.25, 0.7, 0.05]])), model_id="m")


def test_round_seven_the_accumulator_finds_repeats_across_many_chunks():
    policy = _policy(threshold=0.5)
    accumulator = CoverageAccumulator()
    for start in range(0, 40, 4):  # ten chunks: the id runs merge as they grow
        accumulator.update(_apply(policy, sample_ids=list(range(start, start + 4))), Y)
    for repeated in ([0, 100, 101, 102], [100, 101, 102, 39], [100, 17, 101, 102]):
        with pytest.raises(AbstentionError, match="repeats sample ids"):
            accumulator.update(_apply(policy, sample_ids=repeated), Y)
    accumulator.update(_apply(policy, sample_ids=[40, 41, 42, 43]), Y)
    report = accumulator.report()
    assert report.total == 44 and report.sample_ids.tolist() == list(range(44))


def test_round_eight_a_selection_records_the_calibration_override():
    from nnx.calibration import TemperatureCalibrator

    calibrator = TemperatureCalibrator(temperature=1.0, labels=("a", "b"), model_id="m", split_id="calib")
    two = P[:, :2] / P[:, :2].sum(1, keepdims=True)
    overridden = calibrator.transform(np.log(two), labels=("a", "b"), model_id="n", override="audit")
    selection = select_threshold(
        overridden, [0, 1, 1, 1], kind="max_probability", risk_ceiling=0.5, split_id="val",
        input_field="calibrated_probabilities",
    )  # fmt: skip
    assert selection.calibration_override["name"] == "audit"
    assert set(selection.calibration_override["mismatches"]) == {"model_id"}
    assert abstention.ThresholdSelection.from_json(selection.to_json()) == selection
    raw = select_threshold(
        overridden, [0, 1, 1, 1], kind="max_probability", risk_ceiling=0.5, split_id="val", input_field="probabilities"
    )
    assert raw.calibration_override is None  # the raw view was not produced under the override
    state = selection.state()
    state["calibration_override"] = {"name": "audit"}
    with pytest.raises(AbstentionError, match="calibration_override"):
        abstention.ThresholdSelection.from_state(state)


def test_round_eight_ids_and_fields_raise_the_right_errors():
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    prediction = prediction_from_logits(np.log(P), ProbabilitySpec("categorical", labels=LABELS))
    with pytest.raises(AbstentionError) as caught:
        _policy().apply(prediction, model_id="m", sample_ids=[0.0, 1.0, 2.0, 3.0])  # malformed, not a contradiction
    assert not isinstance(caught.value, AbstentionSchemaError)
    with pytest.raises(AbstentionError, match="input_field must be one of") as caught:
        _policy().apply(prediction, model_id="m", input_field="probabilites")
    assert not isinstance(caught.value, AbstentionSchemaError)
    with pytest.raises(AbstentionError, match="sample_ids") as caught:
        _apply(_policy(), sample_ids=[5, 6, 7])  # one call has no chunks to blame
    assert "chunk" not in str(caught.value)


def test_round_eight_selection_allows_repeated_ids_it_never_uses():
    repeated = select_threshold(
        P, Y, kind="max_probability", risk_ceiling=0.0, split_id="val", labels=LABELS, model_id="m",
        sample_ids=[7, 7, 8, 8],
    )  # fmt: skip
    plain = select_threshold(
        P, Y, kind="max_probability", risk_ceiling=0.0, split_id="val", labels=LABELS, model_id="m"
    )
    assert repeated == plain
    with pytest.raises(AbstentionError, match="sample_ids"):
        select_threshold(
            P, Y, kind="max_probability", risk_ceiling=0.0, split_id="val", labels=LABELS, model_id="m",
            sample_ids=[0.5, 1, 2, 3],
        )  # fmt: skip


def test_round_nine_a_decoded_class_clearly_below_the_top_is_refused():
    duck = _duck([1], np.array([[0.503, 0.497]], dtype=np.float16))
    duck.spec = type(duck.spec)("categorical", labels=("a", "b"))
    with pytest.raises(AbstentionError, match="maximal probability"):
        AbstentionPolicy("max_probability", 0.49, labels=("a", "b"), model_id="m", tuning_split_id="v").apply(
            duck, model_id="m"
        )


def test_round_nine_default_candidates_are_bounded():
    rng = np.random.default_rng(0)
    raw = rng.random((5000, 3))
    p = raw / raw.sum(axis=1, keepdims=True)
    y = rng.integers(0, 3, 5000)
    points = risk_coverage_curve(p, y, kind="max_probability")
    assert len(points) <= abstention.MAX_DEFAULT_CANDIDATES + 1
    assert points[0].accepted == 5000 and points[-2].accepted >= 1  # accept-all through the top score
    exhaustive = np.unique(abstention.scores(p, kind="max_probability")[1])
    full = risk_coverage_curve(p, y, kind="max_probability", candidates=exhaustive)
    assert len(full) == exhaustive.size + 1
    assert {pt.threshold for pt in points[:-1]} <= set(exhaustive.tolist())  # every point is an actual score
    small = risk_coverage_curve(P, Y, kind="max_probability")
    assert [pt.threshold for pt in small[:-1]] == sorted(set(P.max(axis=1).tolist()))


def test_round_nine_a_selection_records_its_schema():
    failed = select_threshold(
        P, Y, kind="max_probability", risk_ceiling=0.0, split_id="val", labels=LABELS, model_id="m",
        candidates=[0.3],
    )  # fmt: skip
    assert failed.status == "no_feasible_policy"
    assert (failed.labels, failed.model_id, failed.input_field, failed.calibrator_id) == (
        LABELS, "m", "probabilities", None,
    )  # fmt: skip
    other = select_threshold(
        P, Y, kind="max_probability", risk_ceiling=0.0, split_id="val", labels=LABELS, model_id="n",
        candidates=[0.3],
    )  # fmt: skip
    assert failed != other and failed.canonical_bytes() != other.canonical_bytes()  # models differ
    ok = select_threshold(P, Y, kind="max_probability", risk_ceiling=0.0, split_id="val", labels=LABELS, model_id="m")
    with pytest.raises(AbstentionError, match="schema"):
        dataclasses.replace(ok, model_id="n")
    with pytest.raises(AbstentionError, match="calibrated_probabilities"):
        dataclasses.replace(
            failed, calibration_override={"name": "x", "mismatches": {"model_id": {"expected": "a", "actual": "b"}}}
        )
    assert abstention.ThresholdSelection.from_json(failed.to_json()) == failed


def test_round_nine_reasons_are_read_only():
    with pytest.raises(TypeError):
        abstention.REASONS["margin"] = "x"  # type: ignore[index]


def test_round_ten_calibrated_chunks_accumulate_with_their_own_ids():
    from nnx.calibration import TemperatureCalibrator

    calibrator = TemperatureCalibrator(temperature=1.5, labels=LABELS, model_id="m", split_id="calib")
    policy = _policy(threshold=0.4, input_field="calibrated_probabilities", calibrator_id=calibrator.id)
    logits = np.log(P)
    accumulator = CoverageAccumulator()
    for rows, ids in ((slice(0, 2), [10, 11]), (slice(2, 4), [12, 13])):
        chunk = calibrator.transform(logits[rows], labels=LABELS, model_id="m", sample_ids=ids)
        accumulator.update(policy.apply(chunk), Y[rows])
    whole = calibrator.transform(logits, labels=LABELS, model_id="m", sample_ids=[10, 11, 12, 13])
    assert accumulator.report() == policy.apply(whole).report(Y)
    repeat = calibrator.transform(logits[:2], labels=LABELS, model_id="m")  # 0..1 again
    accumulator.update(policy.apply(repeat), Y[:2])
    with pytest.raises(AbstentionError, match=r"transform\(sample_ids"):
        accumulator.update(policy.apply(repeat), Y[:2])


def test_round_ten_a_float64_decoded_class_gets_float64_rounding_only():
    close = np.array([[0.5000004, 0.4999994, 2e-7], [0.2, 0.7, 0.1]])  # row 0: 1e-6 apart
    assert _policy().apply(_duck([0, 1], close), model_id="m").prediction.tolist() == [0, 1]
    with pytest.raises(AbstentionError, match="maximal probability"):
        _policy().apply(_duck([1, 1], close), model_id="m")


def test_round_ten_strict_loading_and_keyword_errors():
    state = _apply(_policy(threshold=0.0), sample_ids=[1, 2, 3, 4]).report(Y).state()  # coverage 1.0, risk 0.5
    assert abstention.CoverageReport.from_state(state).coverage == 1.0
    for key, value in (("coverage", True), ("risk", True), ("risk", 0.25)):  # True == 1 in Python, not in JSON
        with pytest.raises(AbstentionError, match="contradict"):
            abstention.CoverageReport.from_state(dict(state, **{key: value}))
    assert abstention.CoverageReport.from_state(dict(state, coverage=1)).coverage == 1.0  # a writer's 1 for 1.0
    selection = select_threshold(
        P, Y, kind="max_probability", risk_ceiling=0.5, split_id="val", labels=LABELS, model_id="m"
    )
    points = selection.state()
    points["points"][0]["coverage"] = True  # 1.0 stored as a boolean
    with pytest.raises(AbstentionError, match="contradict"):
        abstention.ThresholdSelection.from_state(points)
    points["points"][0]["coverage"] = 1  # a writer's 1 for 1.0 loads
    assert abstention.ThresholdSelection.from_state(points) == selection
    grid = select_threshold(
        P, Y, kind="max_probability", risk_ceiling=0.5, split_id="val", labels=LABELS, model_id="m",
        candidates=[0.0, 1.0],
    )  # fmt: skip
    integral = grid.state()
    integral["points"][0]["threshold"], integral["points"][1]["threshold"] = 0, 1  # integer thresholds load too
    assert abstention.ThresholdSelection.from_state(integral) == grid
    for text in ('{"coverage": NaN}', '{"risk": Infinity}'):
        with pytest.raises(AbstentionError, match="strict JSON"):
            abstention.CoverageReport.from_json(text)
    from nnx.calibration import TemperatureCalibrator

    calibrator = TemperatureCalibrator(temperature=1.0, labels=LABELS, model_id="m", split_id="calib")
    calibrated = calibrator.transform(np.log(P), labels=LABELS, model_id="m")
    for keywords in ({"model_id": ""}, {"model_id": 5}, {"calibrator_id": 7}):
        with pytest.raises(AbstentionError) as caught:
            _policy().apply(calibrated, **keywords)
        assert not isinstance(caught.value, AbstentionSchemaError)
    malformed = dataclasses.replace(calibrated, override={"name": "x"})
    calibrated_policy = _policy(input_field="calibrated_probabilities", calibrator_id=calibrator.id)
    with pytest.raises(AbstentionError, match="calibration override"):
        calibrated_policy.apply(malformed)


def test_round_twelve_repeated_ids_across_chunks_are_opt_in():
    policy = _policy(threshold=0.5)
    strict, lenient = CoverageAccumulator(), CoverageAccumulator(allow_repeated_ids=True)
    for accumulator in (strict, lenient):
        accumulator.update(_apply(policy, sample_ids=[1, 2, 3, 4]), Y)
    with pytest.raises(AbstentionError, match="allow_repeated_ids"):
        strict.update(_apply(policy, sample_ids=[4, 5, 6, 7]), Y)
    lenient.update(_apply(policy, sample_ids=[4, 5, 6, 7]), Y)  # an oversampled split: id 4 twice
    eager = _apply(policy, np.vstack([P, P]), sample_ids=[1, 2, 3, 4, 4, 5, 6, 7]).report(np.concatenate([Y, Y]))
    assert lenient.report() == eager
    with pytest.raises(AbstentionError, match="bool"):
        CoverageAccumulator(allow_repeated_ids="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AbstentionResult"):
        strict.update(eager, Y)  # a report, not a result


def test_round_twelve_kind_is_checked_before_probabilities():
    with pytest.raises(AbstentionError, match="policy kind"):
        abstention.scores(np.ones((2, 3)), kind="maxprob")  # rows summing to 3 would fail later
    with pytest.raises(AbstentionError, match="policy kind"):
        risk_coverage_curve(np.ones((2, 3)), [0, 1], kind="maxprob")


def test_round_twelve_margins_are_float64_differences():
    # 0.7 - 0.2 is 0.49999999999999994 in binary floating point: the stored
    # values differ by slightly less than 0.5, so a threshold of 0.5 abstains.
    rows = np.array([[0.7, 0.2, 0.1]])
    _, margin = abstention.scores(rows, kind="margin")
    assert margin[0] == 0.7 - 0.2 < 0.5
    assert not _policy("margin", 0.5).apply(rows, labels=LABELS, model_id="m").accepted[0]
    chosen = select_threshold(rows, [0], kind="margin", risk_ceiling=0.0, split_id="val", labels=LABELS, model_id="m")
    assert chosen.require().apply(rows, labels=LABELS, model_id="m").accepted[0]  # selection uses actual scores


def test_round_thirteen_malformed_files_and_states_raise_module_errors():
    for text in ('{"a": 1' + "0" * 5000 + "}", "[" * 100000 + "]" * 100000):
        with pytest.raises(AbstentionError, match="JSON"):
            AbstentionPolicy.from_json(text)
    state = _policy().state()
    state[1] = 2  # a non-string key beside string ones
    with pytest.raises(AbstentionError, match="unknown keys"):
        AbstentionPolicy.from_state(state)


def test_round_thirteen_accepted_ids_are_a_sub_multiset():
    with pytest.raises(AbstentionError, match="at most as often"):
        CoverageReport(policy_id=None, total=2, accepted=2, incorrect=0, sample_ids=[1, 2], accepted_ids=[1, 1])
    with pytest.raises(AbstentionError, match="among its sample ids"):
        CoverageReport(policy_id=None, total=1, accepted=1, incorrect=0, sample_ids=[1], accepted_ids=[9])
    kept = CoverageReport(policy_id=None, total=3, accepted=2, incorrect=0, sample_ids=[1, 1, 2], accepted_ids=[1, 1])
    assert kept.coverage == 2 / 3


def test_round_thirteen_selection_declarations_are_schema_errors():
    with pytest.raises(AbstentionSchemaError, match="calibrator_id"):
        select_threshold(
            P, Y, kind="margin", risk_ceiling=0.1, split_id="v", labels=LABELS, model_id="m",
            input_field="calibrated_probabilities",
        )  # fmt: skip
    assert "contradicts" in AbstentionSchemaError.__doc__ and "itself" in AbstentionSchemaError.__doc__


def test_round_fourteen_the_curve_reads_predictions_as_selection_does():
    from nnx.calibration import TemperatureCalibrator

    calibrator = TemperatureCalibrator(temperature=1.0, labels=("a", "b"), model_id="m", split_id="calib")
    tied = calibrator.transform(np.array([[0.0, 1e-17], [3.0, 0.0]]), labels=("a", "b"), model_id="m")
    y = [1, 0]  # the decoded classes are right; the probabilities' first argmax is wrong on row 0
    for field in ("probabilities", "calibrated_probabilities"):
        curve = risk_coverage_curve(tied, y, kind="max_probability", input_field=field)
        selection = select_threshold(tied, y, kind="max_probability", risk_ceiling=0.0, split_id="v", input_field=field)
        assert curve == selection.points and curve[0].incorrect == 0
    assert risk_coverage_curve(tied.probabilities, y, kind="max_probability")[0].incorrect == 1  # bare arrays
    with pytest.raises(AbstentionError, match="input_field"):
        risk_coverage_curve(tied, y, kind="max_probability")
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    prediction = prediction_from_logits(np.log(P), ProbabilitySpec("categorical", labels=LABELS))
    assert risk_coverage_curve(prediction, Y, kind="margin") == risk_coverage_curve(
        prediction.probabilities, Y, kind="margin"
    )


def test_round_fourteen_the_accumulator_default_refuses_any_repeat():
    policy = _policy(threshold=0.5)
    with pytest.raises(AbstentionError, match="repeats a sample id"):
        CoverageAccumulator().update(_apply(policy, sample_ids=[7, 7, 8, 9]), Y)
    lenient = CoverageAccumulator(allow_repeated_ids=True)
    lenient.update(_apply(policy, sample_ids=[7, 7, 8, 9]), Y)
    assert lenient.report().total == 4


def test_round_fourteen_a_nan_file_names_the_constant_once():
    with pytest.raises(AbstentionError) as caught:
        AbstentionPolicy.from_json('{"threshold": NaN}')
    assert str(caught.value) == "the abstention policy file is strict JSON; NaN is not a JSON number"


def test_round_fifteen_the_curve_checks_a_prediction_width():
    from nnx.prediction import ProbabilitySpec, prediction_from_logits

    prediction = prediction_from_logits(np.log(P), ProbabilitySpec("categorical", labels=LABELS))
    narrow = SimpleNamespace(
        logits=prediction.logits[:, :2],
        probabilities=np.array([[0.6, 0.4]] * 4),
        decoded=np.zeros(4, dtype=np.int64),
        spec=prediction.spec,  # three labels over two columns
        sample_ids=prediction.sample_ids,
    )
    for call in (
        lambda: risk_coverage_curve(narrow, [0, 0, 0, 0], kind="margin"),
        lambda: select_threshold(narrow, [0, 0, 0, 0], kind="margin", risk_ceiling=0.1, split_id="v", model_id="m"),
    ):
        with pytest.raises(AbstentionError, match="one label per class"):
            call()


def test_round_fifteen_predictions_refuse_ids_beyond_int64():
    from nnx.prediction import PredictionValidationError, ProbabilitySpec, prediction_from_logits

    with pytest.raises(PredictionValidationError, match="2\\*\\*63"):
        prediction_from_logits(
            np.zeros((1, 2)), ProbabilitySpec("categorical"), sample_ids=np.array([2**63], np.uint64)
        )


def test_round_seventeen_curve_declarations_candidates_and_ids():
    import torch

    with pytest.raises(AbstentionError, match="bare probabilities are read as given"):
        risk_coverage_curve(P, Y, kind="margin", input_field="calibrated_probabilities")
    grid = risk_coverage_curve(P, Y, kind="max_probability", candidates=torch.linspace(0, 1, 5))
    assert [point.threshold for point in grid[:-1]] == [0.0, 0.25, 0.5, 0.75, 1.0]
    duck = _duck([0, 0, 2, 0])
    duck.sample_ids = np.array([[0, 1, 2, 3]])  # malformed own ids
    with pytest.raises(AbstentionError, match="one integer id per row"):
        _policy().apply(duck, model_id="m", sample_ids=[0, 1, 2, 3])


def test_round_eighteen_overflowing_numbers_and_one_shot_keys():
    from nnx._artifacts import check_keys

    with pytest.raises(AbstentionError, match="overflows a float"):
        AbstentionPolicy.from_json('{"threshold": 1e999}')
    check_keys({"a": 1}, required=(key for key in ["a"]), what="state", error=AbstentionError)  # read once
    assert risk_coverage_curve(P, Y, kind="margin", input_field="probabilities") == risk_coverage_curve(
        P, Y, kind="margin"
    )


def test_round_nineteen_integers_too_large_for_a_float_raise_module_errors():
    from nnx.calibration import CalibrationError, TemperatureCalibrator

    state = _policy().state()
    state["threshold"] = 10**400
    with pytest.raises(AbstentionError, match="too large for a float"):
        AbstentionPolicy.from_state(state)
    with pytest.raises(AbstentionError, match="candidates"):
        risk_coverage_curve(P, Y, kind="margin", candidates=[10**400])
    with pytest.raises(CalibrationError, match="too large for a float"):
        TemperatureCalibrator(temperature=10**400, labels=("a", "b"), model_id="m", split_id="calib")


def test_round_twenty_malformed_ids_fail_before_the_probabilities_are_read():
    class Probabilities:
        reads = 0

        def __array__(self, dtype=None, copy=None):
            Probabilities.reads += 1
            return P

    with pytest.raises(AbstentionError, match="sample_ids"):
        _policy().apply(Probabilities(), labels=LABELS, model_id="m", sample_ids=[[1, 2]])
    assert Probabilities.reads == 0
    with pytest.raises(AbstentionError, match="one integer id per row \\(4\\)"):
        _policy().apply(Probabilities(), labels=LABELS, model_id="m", sample_ids=[1, 2])  # the count, after
