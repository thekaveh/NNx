"""FEAT-009: typed decision questions, results and the response validator.

Choice needs 2+ unique option ids and a distribution summing to 1 within
1e-6; Boolean holds one p_true in [0, 1]; Score keeps 2+ ordered levels and
their distribution. The validator reorders keyed output into request order,
rejects missing / duplicate / unknown / unlabeled ids and never
renormalizes; the digest changes when options are reordered or reworded.
"""

from __future__ import annotations

import math
import pickle

import pytest

from nnx import decisions
from nnx.decisions import (
    PROBABILITY_TOLERANCE,
    Boolean,
    BooleanResult,
    Capabilities,
    Choice,
    ChoiceResult,
    DecisionError,
    InvalidDecisionRequest,
    InvalidDecisionResponse,
    Option,
    ProviderFailure,
    Score,
    ScoreResult,
    UnsupportedCapability,
    question_from_state,
    validate_response,
)

COLORS = Choice("Which color is the sky?", (Option("red", "Red"), Option("green", "Green"), Option("blue", "Blue")))
SEVERITY = Score("How severe is the bug?", (("low", "Cosmetic"), ("mid", "Degraded"), ("high", "Outage")))


# --- the three primitives ---------------------------------------------------------------


def test_choice_needs_two_or_more_unique_ids_and_a_normalized_distribution():
    with pytest.raises(InvalidDecisionRequest, match="at least 2 options"):
        Choice("q?", (Option("a", "A"),))
    with pytest.raises(InvalidDecisionRequest, match="unique; duplicated: \\['a'\\]"):
        Choice("q?", (Option("a", "A"), Option("a", "Again")))
    with pytest.raises(InvalidDecisionRequest, match="non-empty prompt"):
        Choice(" ", (Option("a", "A"), Option("b", "B")))
    with pytest.raises(InvalidDecisionRequest, match="non-empty string"):
        Option("", "empty id")
    with pytest.raises(InvalidDecisionRequest, match="description"):
        Option("a", "  ")
    digest = COLORS.digest()
    ChoiceResult(digest, (("red", 0.2), ("green", 0.3), ("blue", 0.5)))
    ChoiceResult(digest, (("red", 0.2), ("green", 0.3), ("blue", 0.5 + PROBABILITY_TOLERANCE / 2)))  # within 1e-6
    with pytest.raises(InvalidDecisionResponse, match="never renormalized"):
        ChoiceResult(digest, (("red", 0.2), ("green", 0.3), ("blue", 0.5 + 2 * PROBABILITY_TOLERANCE)))
    with pytest.raises(InvalidDecisionResponse, match="at least 2"):
        ChoiceResult(digest, (("red", 1.0),))
    with pytest.raises(InvalidDecisionResponse, match="repeats ids"):
        ChoiceResult(digest, (("red", 0.5), ("red", 0.5)))


def test_boolean_holds_one_probability_in_the_unit_interval():
    question = Boolean("Is it raining?")
    result = BooleanResult(question.digest(), 0.25)
    assert result.p_true == 0.25 and result.p_false == 0.75
    for bad in (-0.01, 1.01, math.nan, math.inf, True, "0.5"):
        with pytest.raises(InvalidDecisionResponse):
            BooleanResult(question.digest(), bad)  # type: ignore[arg-type]


def test_score_keeps_ordered_levels_and_an_ordinal_expected_index():
    assert SEVERITY.option_ids == ("low", "mid", "high")
    assert SEVERITY.levels[2] == Option("high", "Outage")
    result = ScoreResult(SEVERITY.digest(), (("low", 0.2), ("mid", 0.5), ("high", 0.3)), vendor_score=7.5)
    assert result.expected_index == pytest.approx(0 * 0.2 + 1 * 0.5 + 2 * 0.3)  # sum(i * p_i), zero-based
    assert result.vendor_score == 7.5  # the vendor's own number stays separate
    with pytest.raises(InvalidDecisionRequest, match="at least 2 levels"):
        Score("q?", (("only", "Only"),))


# --- the validator --------------------------------------------------------------------------


def test_validate_response_reorders_keyed_output_into_request_order():
    result = validate_response(COLORS, {"blue": 0.5, "red": 0.2, "green": 0.3}, provider="p", raw={"id": 9})
    assert isinstance(result, ChoiceResult)
    assert [option_id for option_id, _ in result.distribution] == ["red", "green", "blue"]
    assert result.probabilities() == {"red": 0.2, "green": 0.3, "blue": 0.5} and result.top == "blue"
    assert result.raw == {"id": 9} and result.question_digest == COLORS.digest()
    pairs = validate_response(COLORS, [("green", 0.3), ("blue", 0.5), ("red", 0.2)])
    assert pairs.distribution == result.distribution  # pairs in any order land in request order
    assert pairs == validate_response(COLORS, {"red": 0.2, "green": 0.3, "blue": 0.5}, raw="other")  # raw ignored


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"red": 0.5, "green": 0.5}, "missing ids \\['blue'\\]"),
        ([("red", 0.2), ("red", 0.3), ("blue", 0.5)], "duplicate id 'red'"),
        ({"red": 0.2, "green": 0.3, "purple": 0.5}, "unknown id 'purple'"),
        ({"red": 0.2, "": 0.3, "blue": 0.5}, "unlabeled entry"),
        ({"red": 0.2, None: 0.3, "blue": 0.5}, "unlabeled entry"),
        ({"red": 0.2, "green": 0.2, "blue": 0.2}, "never renormalized"),
        ({"red": -0.1, "green": 0.6, "blue": 0.5}, "in \\[0, 1\\]"),
        ("red", "mapping or a sequence"),
        ([("red", 0.2, "extra")], "pairs"),
    ],
)
def test_validate_response_rejects_malformed_keyed_output(response, message):
    with pytest.raises(InvalidDecisionResponse, match=message):
        validate_response(COLORS, response)


def test_validate_response_never_renormalizes_and_keeps_vendor_scores_apart():
    unnormalized = {"red": 2.0, "green": 3.0, "blue": 5.0}
    with pytest.raises(InvalidDecisionResponse):
        validate_response(COLORS, unnormalized)
    result = validate_response(SEVERITY, {"high": 0.1, "low": 0.6, "mid": 0.3}, vendor_score=42)
    assert isinstance(result, ScoreResult) and result.vendor_score == 42.0
    assert result.probabilities() == {"low": 0.6, "mid": 0.3, "high": 0.1}
    with pytest.raises(InvalidDecisionResponse, match="vendor_score"):
        validate_response(SEVERITY, {"high": 0.1, "low": 0.6, "mid": 0.3}, vendor_score=math.nan)
    rain = Boolean("Is it raining?")
    assert validate_response(rain, 0.8).p_true == 0.8
    assert validate_response(rain, {"true": 0.8, "false": 0.2}).p_true == 0.8
    with pytest.raises(InvalidDecisionResponse, match="never renormalized"):
        validate_response(rain, {"true": 0.8, "false": 0.8})
    with pytest.raises(InvalidDecisionResponse, match="'true' and 'false'"):
        validate_response(rain, {"yes": 0.8, "no": 0.2})


# --- identity: digests and model-facing text ---------------------------------------------------


def test_digest_changes_on_option_order_or_description_changes():
    reordered = Choice(COLORS.prompt, (COLORS.options[2], COLORS.options[0], COLORS.options[1]))
    reworded = Choice(COLORS.prompt, (COLORS.options[0], COLORS.options[1], Option("blue", "Azure")))
    same = Choice(COLORS.prompt, tuple(Option(o.id, o.description) for o in COLORS.options))
    assert COLORS.digest() == same.digest()
    assert len({COLORS.digest(), reordered.digest(), reworded.digest()}) == 3
    assert Score(SEVERITY.prompt, tuple(reversed(SEVERITY.levels))).digest() != SEVERITY.digest()
    assert Boolean("a?").digest() != Boolean("b?").digest()
    # A result is bound to the digest of the exact question it answers.
    assert validate_response(reordered, {"red": 0.2, "green": 0.3, "blue": 0.5}).question_digest != COLORS.digest()


def test_bookkeeping_ids_never_reach_the_model_facing_text():
    question = Choice("Route this ticket", (Option("team-7f3a", "Billing"), Option("team-91c2", "Networking")))
    view = question.model_view()
    assert view == {"prompt": "Route this ticket", "options": ["Billing", "Networking"]}
    assert "team-7f3a" not in repr(view) and "team-91c2" not in repr(view)
    assert SEVERITY.model_view() == {"prompt": SEVERITY.prompt, "levels": ["Cosmetic", "Degraded", "Outage"]}


def test_questions_round_trip_through_state_and_pickle():
    for question in (COLORS, SEVERITY, Boolean("Is it raining?")):
        assert question_from_state(question.state()) == question
        assert pickle.loads(pickle.dumps(question)) == question
        assert question_from_state(question.state()).digest() == question.digest()
    with pytest.raises(InvalidDecisionRequest, match="unknown decision kind"):
        question_from_state({"kind": "ranking", "prompt": "?"})


# --- errors and capabilities ---------------------------------------------------------------------


def test_typed_errors_are_public_and_share_one_base():
    for error in (UnsupportedCapability, InvalidDecisionRequest, InvalidDecisionResponse, ProviderFailure):
        assert issubclass(error, DecisionError) and getattr(decisions, error.__name__) is error
    assert issubclass(InvalidDecisionRequest, ValueError) and issubclass(InvalidDecisionResponse, ValueError)
    assert issubclass(ProviderFailure, RuntimeError)


def test_capabilities_reject_unsupported_requests():
    caps = Capabilities(primitives={"choice"}, modalities={"tensor"}, max_batch=4)
    caps.check(COLORS, modality="tensor", batch_size=4)
    with pytest.raises(UnsupportedCapability, match="not 'boolean'"):
        caps.check(Boolean("?"), modality="tensor", batch_size=1)
    with pytest.raises(UnsupportedCapability, match="not 'text'"):
        caps.check(COLORS, modality="text", batch_size=1)
    with pytest.raises(UnsupportedCapability, match="max_batch=4"):
        caps.check(COLORS, modality="tensor", batch_size=5)
    with pytest.raises(UnsupportedCapability, match="no inference"):
        Capabilities(primitives={"choice"}, modalities={"tensor"}, inference=False).check(
            COLORS, modality="tensor", batch_size=1
        )
    assert (caps.inference, caps.training, caps.export, caps.dynamic_labels) == (True, False, False, False)


# --- review hardening ------------------------------------------------------------------------


def test_options_must_be_pairs_never_split_strings():
    with pytest.raises(InvalidDecisionRequest, match="\\(id, description\\) pairs, got 'ab'"):
        Choice("q?", ["ab", "cd"])
    with pytest.raises(InvalidDecisionRequest, match="sequence of options"):
        Choice("q?", "ab")  # type: ignore[arg-type]
    assert Choice("q?", [["a", "A"], ("b", "B")]).option_ids == ("a", "b")


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"kind": "choice", "prompt": "x"}, "missing \\['options'\\]"),
        ({"kind": "score", "levels": [["a", "A"], ["b", "B"]]}, "missing \\['prompt'\\]"),
        (["choice"], "must be a mapping"),
        ({"kind": 3, "prompt": "x"}, "unknown decision kind"),
        ({"kind": "choice", "prompt": "x", "options": ["ab", "cd"]}, "pairs"),
    ],
)
def test_malformed_question_state_raises_the_typed_error(state, message):
    with pytest.raises(InvalidDecisionRequest, match=message):
        question_from_state(state)


def test_capabilities_enforce_a_declared_label_space():
    caps = Capabilities(primitives={"choice"}, modalities={"text"}, labels=("red", "green", "blue"))
    caps.check(COLORS, modality="text", batch_size=1)
    renamed = Choice("?", (("r", "Red"), ("g", "Green"), ("b", "Blue")))
    with pytest.raises(UnsupportedCapability, match="unseen labels \\['r', 'g', 'b'\\]"):
        caps.check(renamed, modality="text", batch_size=1)
    caps.check(renamed, modality="text", batch_size=1, label_ids=("red", "green", "blue"))  # the provider's mapping
    with pytest.raises(UnsupportedCapability, match="missing \\['blue'\\]"):
        caps.check(Choice("?", (("red", "Red"), ("green", "Green"))), modality="text", batch_size=1)
    dynamic = Capabilities(primitives={"choice"}, modalities={"text"}, dynamic_labels=True, labels=("x", "y"))
    dynamic.check(renamed, modality="text", batch_size=1)
    with pytest.raises(ValueError, match="max_batch"):
        Capabilities(primitives={"choice"}, modalities={"text"}, max_batch=2.5)  # type: ignore[arg-type]
