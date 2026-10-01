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
    coverage = evaluate(samples, messy, split="test-v1").slices["in_family"].coverage
    assert (coverage.eligible, coverage.duplicate, coverage.missing, coverage.extra, coverage.mismatched) == (
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
