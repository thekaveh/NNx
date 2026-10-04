"""FEAT-021: a reproducible, offline-first decision-provider benchmark.

Records join samples by id and question digest (never row position) and
round-trip through JSONL; the report gives accuracy, macro-F1, exact NLL,
Brier, reliability bins and selective risk with reasons and denominators;
resources stay null unless declared; perturbations, unsupported requests and
held-out families report apart; intervals resample grouping units with a
recorded seed; exports agree; replay calls no provider and fits nothing.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random

import numpy as np
import pytest
import torch

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams, TaskSpec
from nnx.abstention import AbstentionPolicy
from nnx.calibration import brier_score, negative_log_likelihood
from nnx.decisions import Boolean, Capabilities, Choice, FixedHeadProvider, UnsupportedCapability, validate_response
from nnx.decisions.benchmark import (
    BenchmarkError,
    BenchmarkReport,
    Budget,
    Record,
    Resources,
    Sample,
    add_context,
    add_distractors,
    add_none_of_the_above,
    bootstrap_interval,
    collect,
    compare_reports,
    evaluate,
    permute_options,
    read_records,
    redescribe,
    rewrite_input,
    write_records,
)

AB = Choice("Which class?", (("a", "class a"), ("b", "class b")))
LABELS = ["a", "b", "a", "b"]
PROBS = [[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.7, 0.3]]


def fixture(question=AB, **overrides):
    samples = [Sample(f"s{i}", question, f"text {i}", label, **overrides) for i, label in enumerate(LABELS)]
    records = [
        Record(
            sample_id=f"s{i}",
            question_digest=question.digest(),
            provider="stub",
            status="answered",
            distribution=tuple(zip(question.option_ids, p, strict=True)),
        )
        for i, p in enumerate(PROBS)
    ]
    return samples, records


def metric(report, name, slice_="in_family"):
    return report.slices[slice_].metrics[name]


# --- AC1: joining by id and digest; JSONL round trip ---------------------------------------------------------


def test_reversed_rows_match():
    samples, records = fixture()
    forward = evaluate(samples, records, split="test-v1")
    backward = evaluate(list(reversed(samples)), list(reversed(records)), split="test-v1")
    assert forward.slices["in_family"].state() == backward.slices["in_family"].state()
    # Duplicate, missing, extra and mismatched-digest records become coverage counts.
    other = Choice("Another question?", (("a", "x"), ("b", "y")))
    messy = [
        records[0],
        records[0],  # duplicate for s0
        records[1],  # s2 missing below
        Record("ghost", AB.digest(), "stub", "answered", distribution=(("a", 0.5), ("b", 0.5))),  # extra
        Record("s3", other.digest(), "stub", "answered", distribution=(("a", 0.5), ("b", 0.5))),  # mismatched
    ]
    report = evaluate(samples, messy, split="test-v1")
    coverage = report.slices["in_family"].coverage
    assert (coverage.eligible, coverage.duplicate, coverage.missing, coverage.mismatched, report.extra) == (
        1,
        1,
        1,
        1,
        1,
    )


def test_record_fields_round_trip(tmp_path):
    record = Record(
        sample_id="s0",
        question_digest=AB.digest(),
        provider="nli-baseline",
        status="answered",
        distribution=(("a", 0.25), ("b", 0.75)),
        revision="model@abc123",
        prompt_identity="sha256:feed",
        execution={"batch": 0, "batch_size": 4, "partial_batch": False, "seconds": 0.012, "attempt": 1},
    )
    failed = Record(
        "s1", AB.digest(), "nli-baseline", "failed", reason="ProviderFailure: timeout", revision="model@abc123"
    )
    path = tmp_path / "records.jsonl"
    write_records(path, [record, failed])
    assert read_records(path) == [record, failed]
    assert all(json.loads(line)["format"] == "nnx.decision-record/1" for line in path.read_text().splitlines())
    path.write_text('{"format": "nnx.decision-record/1", "sample_id": "x", "p_true": NaN}\n')
    with pytest.raises(BenchmarkError, match="strict JSON"):
        read_records(path)


# --- AC2: metrics -----------------------------------------------------------------------------------------


def test_fixture_metrics():
    samples, records = fixture()
    report = evaluate(samples, records, split="test-v1")
    assert metric(report, "accuracy").value == pytest.approx(0.75)
    assert metric(report, "macro_f1").value == pytest.approx((0.8 + 2 / 3) / 2)
    assert metric(report, "brier").value == pytest.approx(0.35)
    expected_nll = -(math.log(0.8) + math.log(0.9) + math.log(0.6) + math.log(0.3)) / 4
    assert metric(report, "nll").value == pytest.approx(expected_nll)
    # The same terms as nnx.calibration (the named nll / brier metrics).
    targets = [0, 1, 0, 1]
    assert metric(report, "nll").value == pytest.approx(negative_log_likelihood(np.array(PROBS), targets, epsilon=None))
    assert metric(report, "brier").value == pytest.approx(brier_score(np.array(PROBS), targets))
    bins = report.slices["in_family"].bins
    assert sum(b.count for b in bins) == 4 and len(bins) == 10
    assert metric(report, "ece").denominator == 4
    unavailable = metric(report, "selective_risk")
    assert unavailable.value is None and unavailable.reason == "no abstention policy given"


def test_zero_true_class():
    samples, records = fixture()
    records[1] = Record("s1", AB.digest(), "stub", "answered", distribution=(("a", 1.0), ("b", 0.0)))
    exact = evaluate(samples, records, split="test-v1")
    assert metric(exact, "nll").value == math.inf
    assert json.loads(exact.to_json())["slices"]["in_family"]["metrics"]["nll"]["value"] == "Infinity"
    floored = evaluate(samples, records, split="test-v1", epsilon=1e-12)
    assert math.isfinite(metric(floored, "nll").value)


def test_selective_risk_applies_a_policy_as_given():
    samples, records = fixture()
    policy = AbstentionPolicy("max_probability", 0.75, labels=("a", "b"), model_id="stub-v1", tuning_split_id="val")
    report = evaluate(samples, records, split="test-v1", policy=policy, model_id="stub-v1")
    coverage, risk = metric(report, "selective_coverage"), metric(report, "selective_risk")
    assert (coverage.value, coverage.denominator) == (0.5, 4)  # 0.8 and 0.9 accepted
    assert (risk.value, risk.denominator) == (0.0, 2)
    strict = AbstentionPolicy("max_probability", 0.99, labels=("a", "b"), model_id="stub-v1", tuning_split_id="val")
    none = metric(evaluate(samples, records, split="test-v1", policy=strict, model_id="stub-v1"), "selective_risk")
    assert none.value is None and none.reason == "no accepted rows" and none.denominator == 0


# --- AC3: resources -----------------------------------------------------------------------------------------


def test_resource_fields_null():
    samples, records = fixture()
    report = evaluate(samples, records, split="test-v1")
    assert json.loads(report.to_json())["resources"] == {
        "warmup": None,
        "hardware": None,
        "timing_boundary": None,
        "concurrency": None,
        "batch_count": None,
        "seconds": None,
        "source": None,
    }
    with pytest.raises(BenchmarkError, match="hardware"):
        Resources(seconds=1.5, source="measured")  # measured numbers need hardware
    with pytest.raises(BenchmarkError, match="source"):
        Resources(seconds=1.5)
    supplied = Resources(seconds=1.5, source="supplied", timing_boundary="vendor-reported latency")
    assert supplied.state()["hardware"] is None


# --- AC4: perturbations, unsupported requests, held-out families ---------------------------------------------


class KeywordProvider:
    """Text provider: mass on options whose description shares a word with the text."""

    def __init__(self):
        self.calls = 0

    def capabilities(self):
        return Capabilities(
            primitives=frozenset({"choice", "boolean"}), modalities=frozenset({"text"}), dynamic_labels=True
        )

    def decide(self, question, texts):
        self.calls += 1
        out = []
        for text in texts:
            words = set(text.lower().split())
            if isinstance(question, Boolean):
                out.append(validate_response(question, 0.9 if question.prompt.split()[-1] in words else 0.2))
                continue
            weights = [1.0 + 4.0 * len(words & set(o.description.lower().split())) for o in question.options]
            total = sum(weights)
            out.append(
                validate_response(question, {o.id: w / total for o, w in zip(question.options, weights, strict=True)})
            )
        return out


TOPIC = Choice("Topic?", (("sport", "goal match striker"), ("economy", "bank rates inflation")))


def test_perturbations():
    base = Sample("t1", TOPIC, "the striker scored a goal", "sport", family="news", group="article-1")
    variants = [
        base,
        permute_options(base, ["economy", "sport"]),
        redescribe(base, {"sport": "football", "economy": "finance"}),
        add_distractors(base, [("weather", "rain sun")]),
        add_none_of_the_above(base, ("none", "None of these")),
        add_context(base, "unrelated filler text " * 50),
        rewrite_input(base, "le buteur a marqué un but", kind="multilingual"),
        rewrite_input(base, "the striker? no: the bank raised rates", kind="adversarial"),
    ]
    assert {v.perturbation for v in variants[1:]} == {
        "permutation",
        "new_descriptions",
        "distractors",
        "none_of_the_above",
        "long_context",
        "multilingual",
        "adversarial",
    }
    assert all(v.unit == "article-1" for v in variants)  # resampled together
    assert variants[4].label == "none" and "sport" not in variants[4].question.option_ids
    collection = collect(KeywordProvider(), variants, provider_id="keywords", budget=Budget(max_calls=20))
    report = evaluate(variants, collection.records, split="perturbations-v1")
    names = {name for name in report.slices if name.startswith("perturbation:")}
    assert names == {f"perturbation:{v.perturbation or 'original'}" for v in variants}
    assert metric(report, "accuracy", "perturbation:original").value == 1.0
    assert metric(report, "accuracy", "perturbation:none_of_the_above").value == 0.0  # the stub cannot abstain
    assert metric(report, "accuracy", "perturbation:permutation").value == 1.0  # ids move with the options
    with pytest.raises(BenchmarkError, match="ordered"):
        permute_options(
            Sample("s", __import__("nnx").decisions.Score("Level?", (("lo", "low"), ("hi", "high"))), "x", "lo"),
            ["hi", "lo"],
        )


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


def test_unsupported_and_heldout():
    species = Choice("Which animal?", (("cat", "A cat"), ("dog", "A dog"), ("fox", "A fox")))
    cows = Choice("Which farm animal?", (("cat", "A cat"), ("dog", "A dog"), ("cow", "A cow")))
    photos = torch.tensor([[1.5, 0.0], [0.5, 2.0], [-1.0, 1.0], [1.0, 0.0]])
    samples = [
        Sample("p0", species, photos[0], "cat", family="pets"),
        Sample("p1", species, photos[1], "dog", family="pets"),
        Sample("p2", species, photos[2], "fox", family="pets"),
        Sample("f0", cows, photos[3], "cow", family="farm", heldout=True),  # outside the head's label space
    ]
    provider = FixedHeadProvider(_species_head())
    collection = collect(provider, samples, provider_id="fixed-head", budget=Budget(max_calls=5))
    assert provider.model_calls == 1  # the farm batch never reached the model
    report = evaluate(samples, collection.records, split="animals-v1")
    pets, farm = report.slices["in_family"], report.slices["heldout"]
    assert (pets.coverage.eligible, pets.coverage.unsupported) == (3, 0)
    assert metric(report, "accuracy").value == 1.0
    assert (farm.coverage.samples, farm.coverage.eligible, farm.coverage.unsupported) == (1, 0, 1)
    (reason,) = farm.coverage.reasons
    assert reason.startswith("unsupported: UnsupportedCapability") and "cow" in reason
    unavailable = metric(report, "accuracy", "heldout")
    assert unavailable.value is None and unavailable.denominator == 0 and unavailable.reason == "no eligible rows"
    assert report.slices["family:farm"].coverage.unsupported == 1


def test_a_bernoulli_head_records_another_boolean_as_unsupported():
    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=1, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.BINARY_CROSS_ENTROPY),
    )
    cat, dog = Boolean("Is it a cat?"), Boolean("Is it a dog?")
    head = FixedHeadProvider(model, question=cat)
    photos = torch.tensor([[1.5, 0.0], [0.5, 2.0]])
    samples = [Sample("c0", cat, photos[0], True), Sample("d0", dog, photos[1], False)]
    collection = collect(head, samples, provider_id="head", budget=Budget(max_calls=5))
    assert [r.status for r in collection.records] == ["answered", "unsupported"] and head.model_calls == 1
    assert "only the Boolean it was trained for" in (collection.records[1].reason or "")


# --- AC5: replay and the live collector -------------------------------------------------------------------------


def test_the_collector_needs_an_explicit_provider_and_budget_and_marks_partial_batches():
    texts = [f"goal number {i}" for i in range(5)]
    samples = [Sample(f"g{i}", TOPIC, text, "sport") for i, text in enumerate(texts)]
    with pytest.raises(BenchmarkError, match="explicit provider"):
        collect(None, samples, provider_id="x", budget=Budget(max_calls=1))
    with pytest.raises(BenchmarkError, match="explicit Budget"):
        collect(KeywordProvider(), samples, provider_id="x", budget=None)  # type: ignore[arg-type]
    provider = KeywordProvider()
    partial = collect(provider, samples, provider_id="kw", budget=Budget(max_calls=5, max_samples=3), batch_size=2)
    assert provider.calls == 2 and not partial.complete and "max_samples=3" in partial.stopped
    assert [(r.sample_id, r.execution["partial_batch"]) for r in partial.records] == [
        ("g0", False),
        ("g1", False),
        ("g2", True),
    ]
    capped = collect(KeywordProvider(), samples, provider_id="kw", budget=Budget(max_calls=1), batch_size=2)
    assert len(capped.records) == 2 and not capped.complete and "max_calls=1" in capped.stopped
    coverage = evaluate(samples, capped.records, split="goals").slices["in_family"].coverage
    assert (coverage.eligible, coverage.missing) == (2, 3)

    class Flaky(KeywordProvider):
        def decide(self, question, texts):
            self.calls += 1
            raise RuntimeError("upstream 503")

    flaky = Flaky()
    failed = collect(flaky, samples[:2], provider_id="flaky", budget=Budget(max_calls=3))
    assert flaky.calls == 1  # one attempt; retries belong to the provider
    assert [r.status for r in failed.records] == ["failed", "failed"] and "upstream 503" in failed.records[0].reason


# --- AC6: intervals ------------------------------------------------------------------------------------------


def test_bootstrap_groups():
    samples, records = fixture()
    samples = [
        Sample(s.id, s.question, s.input, s.label, group=g)
        for s, g in zip(samples, ["g1", "g1", "g2", "g2"], strict=True)
    ]
    interval = bootstrap_interval(samples, records, metric="accuracy", seed=11, resamples=400)
    assert (interval.unit, interval.seed, interval.units, interval.resamples) == ("group", 11, 2, 400)
    assert (
        interval.estimate == pytest.approx(0.75)
        and interval.low == pytest.approx(0.5)
        and interval.high == pytest.approx(1.0)
    )
    again = bootstrap_interval(samples, records, metric="accuracy", seed=11, resamples=400)
    assert again == interval  # the recorded seed reproduces it
    # Whole groups are drawn: g1 has accuracy 1.0, g2 has 0.5, so only 1.0, 0.75 and 0.5 can occur.
    single = [Sample(s.id, s.question, s.input, s.label, group="only") for s in samples]
    flagged = bootstrap_interval(single, records, metric="nll", seed=0)
    assert flagged.degenerate and flagged.low is None and flagged.reason == "fewer than two grouping units"
    same = [
        Sample(s.id, s.question, s.input, s.label, group=g)
        for s, g in zip(samples, ["g1", "g2", "g1", "g2"], strict=True)
    ]
    tied = list(records)
    tied[3] = Record("s3", AB.digest(), "stub", "answered", distribution=(("a", 0.3), ("b", 0.7)))
    constant = bootstrap_interval(same, tied, metric="accuracy", seed=0, resamples=50)
    assert constant.degenerate and constant.low == constant.high == 1.0


# --- AC7: exports and comparison -----------------------------------------------------------------------------


def test_export_agreement(tmp_path):
    samples, records = fixture()
    records = records[:3] + [Record("s3", AB.digest(), "stub", "failed", reason="ProviderFailure: boom")]
    report = evaluate(samples, records, split="test-v1")
    state = json.loads(report.to_json())
    table = list(csv.DictReader(io.StringIO(report.to_csv())))
    text = report.text()
    for row in table:
        stored = state["slices"][row["slice"]]
        value = stored["metrics"][row["metric"]]
        assert row["unit"] == value["unit"]
        assert int(row["denominator"]) == value["denominator"]
        assert (
            int(row["eligible"]) == stored["coverage"]["eligible"]
            and int(row["failed"]) == stored["coverage"]["failed"]
        )
        assert (row["value"] == "") == (value["value"] is None) and row["unavailable"] == (value["reason"] or "")
        if value["value"] is None:
            assert (
                f"{row['metric']}: unavailable ({value['reason']}; n={value['denominator']}) [{value['unit']}]" in text
            )
        else:
            assert f"{row['metric']}: {float(row['value'])!r} (n={value['denominator']}) [{value['unit']}]" in text
    assert "[in_family] samples=4 eligible=3 failed=1" in text

    path = tmp_path / "report.json"
    report.save(path)
    saved = BenchmarkReport.load_state(path)
    other = evaluate(samples, fixture()[1], split="test-v1")
    deltas = compare_reports(saved, other)
    assert deltas["in_family"]["accuracy"] == pytest.approx(0.75 - 1.0)  # b - a: 3 of 4 versus 3 of 3
    with pytest.raises(BenchmarkError, match="different splits"):
        compare_reports(report, evaluate(samples, records, split="test-v2"))
    with pytest.raises(BenchmarkError, match="metric identities"):
        compare_reports(report, evaluate(samples, records, split="test-v1", epsilon=1e-12))


# --- AC8: replay fits nothing -----------------------------------------------------------------------------------


def test_no_fitting_during_replay(monkeypatch, tmp_path):
    import socket

    import nnx.abstention
    import nnx.calibration

    samples, records = fixture()
    path = tmp_path / "records.jsonl"
    write_records(path, records)

    def refuse(*args, **kwargs):
        raise AssertionError("replay must not fit, tune or call anything")

    monkeypatch.setattr(nnx.calibration, "fit_temperature", refuse)
    monkeypatch.setattr(nnx.abstention, "select_threshold", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    policy = AbstentionPolicy("max_probability", 0.75, labels=("a", "b"), model_id="stub-v1", tuning_split_id="val")
    before = policy.state()
    report = evaluate(samples, read_records(path), split="test-v1", policy=policy, model_id="stub-v1")
    assert policy.state() == before and metric(report, "selective_coverage").value == 0.5
    assert metric(report, "nll").value == pytest.approx(
        -(math.log(0.8) + math.log(0.9) + math.log(0.6) + math.log(0.3)) / 4
    )


def test_records_must_come_from_one_provider_and_sample_ids_are_unique():
    samples, records = fixture()
    other = [Record(r.sample_id, r.question_digest, "other", r.status, distribution=r.distribution) for r in records]
    with pytest.raises(BenchmarkError, match="several providers"):
        evaluate(samples, records + other, split="x")
    assert metric(evaluate(samples, records + other, split="x", provider="other"), "accuracy").value == 0.75
    with pytest.raises(BenchmarkError, match="duplicate sample ids"):
        evaluate(samples + samples[:1], records, split="x")
    shuffled = list(records)
    random.Random(0).shuffle(shuffled)
    assert evaluate(samples, shuffled, split="x").to_json() == evaluate(samples, records, split="x").to_json()


def test_unsupported_from_decide_is_recorded_as_unsupported():
    class Refusing(KeywordProvider):
        def decide(self, question, texts):
            raise UnsupportedCapability("no booleans here")

    samples = [Sample("b0", Boolean("Mentions goal"), "a goal", True)]
    collection = collect(Refusing(), samples, provider_id="refusing", budget=Budget(max_calls=2))
    assert collection.records[0].status == "unsupported" and collection.calls == 1


# --- review hardening -----------------------------------------------------------------------------------------


def test_intervals_flag_non_finite_bounds_and_select_one_provider():
    samples, records = fixture()
    samples = [Sample(s.id, s.question, s.input, s.label, group=f"g{i}") for i, s in enumerate(samples)]
    zero = list(records)
    zero[1] = Record("s1", AB.digest(), "stub", "answered", distribution=(("a", 1.0), ("b", 0.0)))
    interval = bootstrap_interval(samples, zero, metric="nll", seed=0, resamples=200)
    assert interval.degenerate and interval.reason == "a resample gave a non-finite value" and interval.high == math.inf
    other = [Record(r.sample_id, r.question_digest, "other", r.status, distribution=r.distribution) for r in records]
    with pytest.raises(BenchmarkError, match="several providers"):
        bootstrap_interval(samples, records + other, metric="accuracy", seed=0)
    picked = bootstrap_interval(samples, records + other, metric="accuracy", seed=0, provider="other")
    assert picked.units == 4 and picked.estimate == 0.75
    with pytest.raises(BenchmarkError, match="epsilon"):
        bootstrap_interval(samples, records, metric="nll", seed=0, epsilon=2)


def test_collected_answers_are_validated_reordered_and_never_end_the_collection():
    from nnx.decisions import ChoiceResult

    question = AB
    samples = [Sample(f"s{i}", question, f"text {i}", "a") for i in range(2)]

    class Reordered(KeywordProvider):
        def decide(self, q, texts):
            return [ChoiceResult(q.digest(), (("b", 0.3), ("a", 0.7))) for _ in texts]

    collection = collect(Reordered(), samples, provider_id="r", budget=Budget(max_calls=2))
    assert [r.distribution for r in collection.records] == [(("a", 0.7), ("b", 0.3))] * 2  # the question's order

    class WrongQuestion(KeywordProvider):
        def decide(self, q, texts):
            return [ChoiceResult(TOPIC.digest(), (("sport", 0.5), ("economy", 0.5))) for _ in texts]

    class NotNormalized(KeywordProvider):
        def decide(self, q, texts):
            return [
                type("R", (), {"question_digest": q.digest(), "distribution": (("a", float("nan")), ("b", 0.5))})()
                for _ in texts
            ]

    for provider in (WrongQuestion(), NotNormalized()):
        failed = collect(provider, samples, provider_id="x", budget=Budget(max_calls=2))
        assert [r.status for r in failed.records] == ["failed", "failed"] and failed.calls == 1

    tensors = [Sample("t0", question, torch.zeros(3), "a"), Sample("t1", question, torch.zeros(4), "a")]
    unbatchable = collect(KeywordProvider(), tensors, provider_id="x", budget=Budget(max_calls=2))
    assert [r.status for r in unbatchable.records] == ["failed", "failed"] and unbatchable.calls == 0
    with pytest.raises(BenchmarkError, match="revision"):
        collect(KeywordProvider(), samples, provider_id="x", budget=Budget(max_calls=1), revision="")


def test_records_must_sum_to_one_and_malformed_answers_are_invalid_coverage():
    with pytest.raises(BenchmarkError, match="sums to"):
        Record("s0", AB.digest(), "stub", "answered", distribution=(("a", 0.9), ("b", 0.9)))
    samples, records = fixture()
    records[2] = Record("s2", AB.digest(), "stub", "answered", distribution=(("b", 0.4), ("a", 0.6)))  # other order
    records[3] = Record("s3", AB.digest(), "stub", "answered", p_true=0.5)  # a Boolean answer for a Choice
    report = evaluate(samples, records, split="x")
    coverage = report.slices["in_family"].coverage
    assert (coverage.eligible, coverage.invalid) == (2, 2)
    assert any(reason.startswith("invalid:") for reason in coverage.reasons)


def test_a_cut_short_batch_keeps_the_stop_reason_and_refusals_spend_no_budget():
    texts = [f"goal number {i}" for i in range(5)]
    samples = [Sample(f"g{i}", TOPIC, text, "sport") for i, text in enumerate(texts)]

    class Failing(KeywordProvider):
        def decide(self, question, texts):
            self.calls += 1
            raise RuntimeError("503")

    failed = collect(Failing(), samples, provider_id="f", budget=Budget(max_calls=5, max_samples=3), batch_size=5)
    assert len(failed.records) == 3 and not failed.complete and "max_samples=3" in failed.stopped

    class Refusing(KeywordProvider):
        def check(self, question, inputs):
            raise UnsupportedCapability("not today")

    refused = collect(Refusing(), samples, provider_id="r", budget=Budget(max_calls=1, max_samples=2), batch_size=5)
    assert len(refused.records) == 5 and refused.calls == 0  # the whole planned batch, no budget spent

    species = Choice("Which animal?", (("cat", "A cat"), ("dog", "A dog"), ("fox", "A fox")))
    cows = Choice("Which farm animal?", (("cat", "A cat"), ("dog", "A dog"), ("cow", "A cow")))
    rows = torch.tensor([[1.5, 0.0], [0.5, 2.0]])
    head = FixedHeadProvider(_species_head())
    mixed = [
        Sample("c0", cows, rows[0], "cow"),
        Sample("p0", species, rows[0], "cat"),
        Sample("p1", species, rows[1], "dog"),
    ]
    collection = collect(head, mixed, provider_id="head", budget=Budget(max_calls=1))
    assert collection.complete and collection.calls == 1  # the refused farm batch spent nothing
    assert [r.status for r in collection.records] == ["unsupported", "answered", "answered"]


def test_perturbation_ids_never_collide():
    base = Sample("t1", TOPIC, "the striker scored a goal", "sport")
    a = permute_options(base, ["economy", "sport"])
    b = permute_options(add_distractors(base, [("x", "y z")]), ["x", "economy", "sport"])
    c, d = rewrite_input(base, "le but", kind="multilingual"), rewrite_input(base, "das Tor", kind="multilingual")
    assert len({a.id, b.id, c.id, d.id}) == 4
    assert rewrite_input(base, "x", kind="multilingual", id="t1-fr").id == "t1-fr"
    evaluate([base, c, d], [], split="x")  # no duplicate-id refusal


def test_exports_carry_every_count_and_inputs_are_bounded():
    samples, records = fixture()
    held = [Sample(s.id, s.question, s.input, s.label, heldout=True) for s in samples]
    ghost = Record("ghost", AB.digest(), "stub", "answered", distribution=(("a", 0.5), ("b", 0.5)))
    report = evaluate(held, records + [ghost], split="x")
    assert report.extra == 1 and json.loads(report.to_json())["extra"] == 1 and "extra_records=1" in report.text()
    for row in csv.DictReader(io.StringIO(report.to_csv())):
        counts = sum(
            int(row[k]) for k in ("eligible", "failed", "unsupported", "missing", "duplicate", "mismatched", "invalid")
        )
        assert counts == int(row["samples"])
    with pytest.raises(BenchmarkError, match="n_bins"):
        evaluate(samples, records, split="x", n_bins=10**7)
    with pytest.raises(BenchmarkError, match="seconds"):
        Resources(seconds=float("nan"), source="supplied")
    with pytest.raises(BenchmarkError, match="seconds"):
        Resources(seconds=-1.0, source="supplied")


# --- review hardening ---------------------------------------------------------------------------------------


def _pets(n: int, question=None, wrap=lambda row: row):
    question = question or Choice("Which animal?", (("cat", "A cat"), ("dog", "A dog"), ("fox", "A fox")))
    rows = torch.tensor([[1.5, 0.0], [0.5, 2.0], [-1.0, 1.0]])
    labels = [o.id for o in question.options]
    return [Sample(f"p{i}", question, wrap(rows[i % 3]), labels[i % 3]) for i in range(n)]


def test_an_option_map_and_the_declared_max_batch_are_honoured():
    renamed = Choice("Which animal?", (("feline", "A cat"), ("canine", "A dog"), ("vulpine", "A fox")))
    head = FixedHeadProvider(_species_head(), option_map={"feline": "cat", "canine": "dog", "vulpine": "fox"})
    mapped = collect(head, _pets(3, renamed), provider_id="head", budget=Budget(max_calls=1))
    assert [r.status for r in mapped.records] == ["answered"] * 3 and mapped.calls == 1
    small = FixedHeadProvider(_species_head(), max_batch=8)
    capped = collect(small, _pets(40), provider_id="head", budget=Budget(max_calls=10))  # batch_size=16
    assert capped.complete and capped.calls == 5 and {r.status for r in capped.records} == {"answered"}
    assert {r.execution["batch_size"] for r in capped.records} == {8}


def test_a_sample_budget_cut_is_checked_as_the_batch_actually_sent():
    class AtMostTwo(KeywordProvider):
        def check(self, question, inputs):
            if len(inputs) > 2:
                raise UnsupportedCapability(f"batch of {len(inputs)} exceeds 2")

    texts = [f"goal {i}" for i in range(5)]
    samples = [Sample(f"g{i}", TOPIC, text, "sport") for i, text in enumerate(texts)]
    cut = collect(AtMostTwo(), samples, provider_id="kw", budget=Budget(max_calls=3, max_samples=2), batch_size=5)
    assert [r.status for r in cut.records] == ["answered", "answered"] and cut.calls == 1
    assert cut.records[0].execution["partial_batch"] and "max_samples=2" in cut.stopped


def test_a_failing_check_keeps_the_paid_records_and_spends_nothing():
    class Transient(KeywordProvider):
        checks = 0

        def check(self, question, inputs):
            Transient.checks += 1
            if Transient.checks == 2:
                raise RuntimeError("transient")

    other = Choice("Other?", (("x", "x"), ("y", "y")))
    samples = [Sample("g0", TOPIC, "goal", "sport"), Sample("o0", other, "x", "x")]
    provider = Transient()
    collection = collect(provider, samples, provider_id="t", budget=Budget(max_calls=5))
    assert [r.status for r in collection.records] == ["answered", "failed"] and collection.calls == 1
    assert collection.records[1].reason == "RuntimeError: transient" and collection.complete


def test_samples_with_several_inputs_are_batched_part_by_part():
    head = FixedHeadProvider(_species_head())
    collection = collect(head, _pets(3, wrap=lambda row: (row,)), provider_id="head", budget=Budget(max_calls=1))
    assert [r.status for r in collection.records] == ["answered"] * 3 and head.model_calls == 1
    report = evaluate(_pets(3, wrap=lambda row: (row,)), collection.records, split="x")
    assert metric(report, "accuracy").value == 1.0


def test_one_malformed_result_fails_only_its_own_sample():
    other = Choice("Other?", (("x", "x"), ("y", "y")))

    class OneWrong(KeywordProvider):
        def decide(self, question, texts):
            results = super().decide(question, texts)
            results[-1] = validate_response(other, {"x": 0.5, "y": 0.5})
            return results

    samples = [Sample(f"g{i}", TOPIC, f"goal {i}", "sport") for i in range(4)]
    collection = collect(OneWrong(), samples, provider_id="kw", budget=Budget(max_calls=1))
    assert [r.status for r in collection.records] == ["answered", "answered", "answered", "failed"]
    assert "answers question" in collection.records[-1].reason


STABLE_INPUTS = (
    "{'text': 'x', 'lang': 'en'}",
    "np.array([3.5, 'red'], dtype=object)",
    "torch.tensor([1.0, 2.0])",
    "(torch.tensor([1, 2]), 'caption')",
    "'plain text'",
)


def _derived_ids_in_a_new_process() -> list[str]:
    import subprocess
    import sys

    code = (
        "import numpy as np, torch\n"
        "from nnx.decisions import Choice\n"
        "from nnx.decisions.benchmark import Sample, permute_options\n"
        "q = Choice('Topic?', (('sport', 'goal'), ('economy', 'bank')))\n"
        f"for value in ({', '.join(STABLE_INPUTS)},):\n"
        "    print(permute_options(Sample('s1', q, value, 'sport'), ['economy', 'sport']).id)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    return out.split()


def test_derived_ids_are_the_same_in_every_process():
    first, second = _derived_ids_in_a_new_process(), _derived_ids_in_a_new_process()
    assert first == second and len(set(first)) == len(STABLE_INPUTS)


def test_inputs_without_a_stable_digest_need_an_id():
    sample = Sample("s1", TOPIC, object(), "sport")
    with pytest.raises(BenchmarkError, match="pass id="):
        permute_options(sample, ["economy", "sport"])
    assert permute_options(sample, ["economy", "sport"], id="s1-swapped").id == "s1-swapped"
    # dtype and shape are part of a tensor's digest
    a = permute_options(Sample("t", TOPIC, torch.zeros(4, dtype=torch.int32), "sport"), ["economy", "sport"])
    b = permute_options(Sample("t", TOPIC, torch.zeros(2, dtype=torch.int64), "sport"), ["economy", "sport"])
    assert a.id != b.id


def test_malformed_records_and_numbers_raise_benchmark_errors(tmp_path):
    base = {"sample_id": "s", "question_digest": "d", "provider": "p", "status": "answered"}
    for distribution in (5, [["a", 0.5, "x"], ["b", 0.5]], [[1, 0.5], ["b", 0.5]]):
        with pytest.raises(BenchmarkError, match="distribution|option ids"):
            Record.from_state({"format": "nnx.decision-record/1", **base, "distribution": distribution})
    with pytest.raises(BenchmarkError, match="p_true"):
        Record(**base, p_true=10**400)
    with pytest.raises(BenchmarkError, match="seconds"):
        Resources(seconds=10**400, source="supplied")
    samples, records = fixture()
    with pytest.raises(BenchmarkError, match="epsilon"):
        evaluate(samples, records, split="x", epsilon=10**400)
    with pytest.raises(BenchmarkError, match="resources must be a Resources"):
        evaluate(samples, records, split="x", resources={"seconds": 1.0})  # type: ignore[arg-type]
    path = tmp_path / "records.jsonl"
    write_records(path, records)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"format": "nnx.decision-record/1", **base, "distribution": 5}) + "\n")
    with pytest.raises(BenchmarkError, match=r"records\.jsonl:5: "):
        read_records(path)
    report = evaluate(samples, records, split="x")
    with pytest.raises(BenchmarkError, match="needs split, metric_identity and slices"):
        compare_reports(report, {"format": report.state()["format"]})


def test_the_default_prompt_identity_is_stable_or_absent():
    class Opaque(KeywordProvider):
        def record(self):
            return {"tokenizer": object()}

    class Plain(KeywordProvider):
        def record(self):
            return {"template": "{}", "revision": "r1"}

    samples = [Sample("g0", TOPIC, "goal", "sport")]
    opaque = collect(Opaque(), samples, provider_id="o", budget=Budget(max_calls=1))
    plain = [collect(Plain(), samples, provider_id="p", budget=Budget(max_calls=1)) for _ in range(2)]
    assert opaque.records[0].prompt_identity is None
    assert plain[0].records[0].prompt_identity == plain[1].records[0].prompt_identity is not None


def test_generators_exports_and_sample_sets():
    samples, records = fixture()
    collection = collect(
        KeywordProvider(),
        (Sample(f"g{i}", TOPIC, f"goal {i}", "sport") for i in range(4)),
        provider_id="kw",
        budget=Budget(max_calls=1),
    )
    assert collection.complete and len(collection.records) == 4
    infinite = [
        Record("s0", AB.digest(), "stub", "answered", distribution=(("a", 0.0), ("b", 1.0))),
        *records[1:],
    ]
    report = evaluate(samples, infinite, split="x")
    rows = {row["metric"]: row for row in csv.DictReader(io.StringIO(report.to_csv()))}
    assert (
        rows["nll"]["value"]
        == "Infinity"
        == json.loads(report.to_json())["slices"]["in_family"]["metrics"]["nll"]["value"]
    )
    assert all(row["extra"] == "0" for row in rows.values())
    interval = bootstrap_interval(
        [Sample(s.id, s.question, s.input, s.label, group=s.id) for s in samples], infinite, metric="nll", seed=0
    )
    json.dumps(interval.state(), allow_nan=False)  # strict JSON with an infinite bound
    # Reports over different samples of one split are never compared.
    fewer = evaluate(samples[:3], records[:3], split="x")
    with pytest.raises(BenchmarkError, match="different samples"):
        compare_reports(report, fewer)
    base = evaluate(samples, records, split="x")
    assert compare_reports(base, evaluate(list(reversed(samples)), records, split="x"))["in_family"]["accuracy"] == 0


def test_round_three_collector_edges():
    class DuplicatePairs(KeywordProvider):
        def decide(self, question, texts):
            self.calls += 1

            class Loose:  # not a validated ChoiceResult
                question_digest = question.digest()
                distribution = (("sport", 0.3), ("sport", 0.5), ("economy", 0.5))

            return [Loose() for _ in texts]

    samples = [Sample("g0", TOPIC, "goal", "sport")]
    collection = collect(DuplicatePairs(), samples, provider_id="d", budget=Budget(max_calls=1))
    assert collection.records[0].status == "failed" and "duplicate" in collection.records[0].reason

    class PairText(KeywordProvider):  # no check(): the declared capabilities decide
        def decide(self, question, inputs):
            premises, hypotheses = inputs
            return super().decide(question, [f"{p} {h}" for p, h in zip(premises, hypotheses, strict=True)])

    pairs = [Sample(f"p{i}", TOPIC, (f"goal {i}", "match"), "sport") for i in range(2)]
    answered = collect(PairText(), pairs, provider_id="pt", budget=Budget(max_calls=1))
    assert [r.status for r in answered.records] == ["answered", "answered"]
    provider = KeywordProvider()
    with pytest.raises(BenchmarkError, match="duplicate sample ids"):
        collect(provider, [samples[0], samples[0]], provider_id="k", budget=Budget(max_calls=1))
    assert provider.calls == 0


def test_round_three_report_edges(tmp_path):
    samples, records = fixture()
    grouped = [Sample(s.id, s.question, s.input, s.label, group=f"u{i % 2}") for i, s in enumerate(samples)]
    interval = bootstrap_interval((s for s in grouped), records, metric="accuracy", seed=0)
    assert interval.units == 2
    report = evaluate(samples, records, split="x")
    moved = [
        Sample(s.id, s.question, s.input, s.label, heldout=i % 2 == 0, family="farm") for i, s in enumerate(samples)
    ]
    with pytest.raises(BenchmarkError, match="different samples"):
        compare_reports(report, evaluate(moved, records, split="x"))
    resources = Resources(warmup=np.int64(3), seconds=np.float32(1.5), source="supplied", batch_count=np.int32(2))
    assert type(resources.warmup) is int and type(resources.seconds) is float
    with pytest.raises(BenchmarkError, match="hardware"):
        Resources(hardware=123)  # type: ignore[arg-type]
    path = tmp_path / "report.json"
    evaluate(samples, records, split="x", resources=resources).save(path)
    before = path.read_bytes()
    broken = evaluate(samples, records, split="x")
    object.__setattr__(broken, "extra", np.int64(1))  # not JSON: the save fails before touching the file
    with pytest.raises(TypeError):
        broken.save(path)
    assert path.read_bytes() == before
    for text, match in (("{", "JSON"), ('{"format": 1e999}', "overflows"), ("[" * 100000, "JSON")):
        bad = tmp_path / "bad.json"
        bad.write_text(text)
        with pytest.raises(BenchmarkError, match=match):
            BenchmarkReport.load_state(bad)
    (tmp_path / "latin.json").write_bytes(b"\xff\xfe")
    with pytest.raises(BenchmarkError, match="not text"):
        BenchmarkReport.load_state(tmp_path / "latin.json")
    lines = tmp_path / "records.jsonl"
    for raw in ("9" * 5000, "1e999"):
        write_records(lines, records)
        with open(lines, "a", encoding="utf-8") as handle:
            handle.write('{"format": "nnx.decision-record/1", "p_true": ' + raw + "}\n")
        with pytest.raises(BenchmarkError, match=r"records\.jsonl:5: "):
            read_records(lines)
    for execution in ([], 0, ""):
        with pytest.raises(BenchmarkError, match="execution"):
            Record.from_state({**records[0].state(), "execution": execution})


def test_a_saved_number_too_large_for_a_float_is_a_benchmark_error(tmp_path):
    samples, records = fixture()
    lines = tmp_path / "records.jsonl"
    write_records(lines, records)
    huge = json.dumps({**records[0].state(), "sample_id": "s9", "distribution": None, "p_true": 10**400})
    with open(lines, "a", encoding="utf-8") as handle:
        handle.write(huge + "\n")
    with pytest.raises(BenchmarkError, match=r"records\.jsonl:5: .*p_true"):
        read_records(lines)
    report = evaluate(samples, records, split="x")
    path = tmp_path / "report.json"
    report.save(path)
    state = json.loads(path.read_text())
    state["slices"]["in_family"]["metrics"]["accuracy"]["value"] = 10**400
    path.write_text(json.dumps(state))
    with pytest.raises(BenchmarkError, match="'accuracy' value .*too large for a float"):
        compare_reports(BenchmarkReport.load_state(path), report)


def test_round_four_edges(tmp_path):
    class ImageOnly(KeywordProvider):  # no check(): its declared capabilities decide what NNx can tell
        def capabilities(self):
            return Capabilities(primitives=frozenset({"choice"}), modalities=frozenset({"image"}), dynamic_labels=True)

        def decide(self, question, inputs):
            return super().decide(question, [str(item["caption"]) for item in inputs])

    samples = [Sample(f"i{i}", TOPIC, {"caption": f"goal {i}"}, "sport") for i in range(2)]
    collection = collect(ImageOnly(), samples, provider_id="img", budget=Budget(max_calls=1))
    assert [r.status for r in collection.records] == ["answered", "answered"]  # never refused on a guess

    class MappingAnswer(KeywordProvider):
        def decide(self, question, texts):
            class Loose:
                question_digest = question.digest()
                distribution = {"sport": 0.7, "economy": 0.3}

            return [Loose() for _ in texts]

    mapped = collect(
        MappingAnswer(), [Sample("m", TOPIC, "goal", "sport")], provider_id="m", budget=Budget(max_calls=1)
    )
    assert mapped.records[0].status == "answered"
    samples, records = fixture()
    lines = tmp_path / "records.jsonl"
    write_records(lines, records)
    text = lines.read_text().replace('"reason": null', '"reason": null, "x": "a\\u2028b"', 1)
    lines.write_text(text.replace("\\u2028", " "), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="unknown keys"):  # the line is parsed whole, not split at U+2028
        read_records(lines)
    infinite = [Record("s0", AB.digest(), "stub", "answered", distribution=(("a", 0.0), ("b", 1.0))), *records[1:]]
    assert "nll: Infinity" in evaluate(samples, infinite, split="x").text()
