"""FEAT-032: multi-seed experiment summaries and paired comparisons.

Observations declare what a comparison depends on and leave unknown facts
unknown; incompatible observations are stratified, never pooled. Groups
report count, mean and the sample standard deviation (n - 1), keeping failed
and non-finite attempts visible. Pairing needs unique replicate keys and
declared semantics and reports signed B - A deltas. Reports are
order-independent, round-trip as strict JSON and are built from saved runs
without loading a model or touching a run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
    comparison,
)
from nnx.comparison import (
    Bootstrap,
    ComparisonError,
    ComparisonReport,
    IncompatibleObservations,
    Metric,
    Observation,
    compare,
    observations_from_runs,
    summarize,
)
from nnx.nn.callbacks import Callback
from nnx.provenance import ExperimentManifest, hash_bytes

ACC = Metric("accuracy", "maximize")
LOSS = Metric("loss", "minimize", unit="nats")


def obs(value, *, config="cfg-a", replicate="seed=0", attempt=None, status="completed", **overrides):
    seed = replicate.split("=")[-1] if replicate else "x"
    fields = dict(
        run_id=f"run-{config}-{seed}",
        attempt_id=attempt or f"att-{config}-{seed}",
        metric=ACC,
        value=value,
        status=status,
        split="validation",
        selection="last",
        config=config,
        data="data:v1",
        split_id="split:v1",
        replicate=replicate,
        evaluation="epoch 2 validation record",
    )
    fields.update(overrides)
    return Observation(**fields)


def replicates(values, *, config="cfg-a", **overrides):
    return [obs(v, config=config, replicate=f"seed={i}", **overrides) for i, v in enumerate(values)]


# --- AC1: declared records; incompatible ones stratify, never pool ------------------------------------------


def test_observations_declare_every_fact_and_leave_unknowns_unknown():
    item = Observation(run_id="r1", metric=LOSS, value=0.5, split="validation", selection="last")
    assert (item.config, item.data, item.split_id, item.replicate, item.attempt_id, item.evaluation) == (None,) * 6
    assert item.id == "run:r1" and item.metric.state() == {"name": "loss", "direction": "minimize", "unit": "nats"}
    group = summarize([item]).pooled()
    assert group.unknown == ("config", "data", "split_id")
    with pytest.raises(ComparisonError, match="direction"):
        Metric("loss", "lower")
    with pytest.raises(ComparisonError, match="status"):
        obs(0.5, status="done")
    with pytest.raises(ComparisonError, match="selection"):
        obs(0.5, selection="")


def test_altering_a_split_identity_stratifies_instead_of_pooling():
    same = replicates([0.8, 0.9])
    moved = replicates([0.5], split_id="split:v2", attempt="att-moved")
    summary = summarize([*same, *moved])
    assert len(summary.groups) == 2 and summary.differing == ("split_id",)
    with pytest.raises(IncompatibleObservations, match="split_id") as caught:
        summary.pooled()
    assert caught.value.fields == ("split_id",)
    text = ComparisonReport(summary).text()
    assert "groups differ in: split_id (stratified, not pooled)" in text and "split_id=split:v2" in text
    # A known identity never pools with an unknown one.
    unknown = replicates([0.7], data=None, attempt="att-unknown")
    assert summarize([*same, *unknown]).differing == ("data",)
    assert summarize([*same, *replicates([0.1], metric=LOSS, attempt="att-loss")]).differing == ("metric",)


# --- AC2: statistics ----------------------------------------------------------------------------------------


def test_known_fixtures_give_count_mean_and_sample_deviation():
    group = summarize(replicates([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])).pooled()
    assert (group.n, group.mean) == (8, 5.0)
    assert group.std == pytest.approx(statistics.stdev([2, 4, 4, 4, 5, 5, 7, 9]))  # n - 1
    assert group.std == pytest.approx(math.sqrt(32 / 7))
    one = summarize(replicates([0.75])).pooled()
    assert (one.n, one.mean, one.std) == (1, 0.75, None)  # n = 1: no deviation


def test_failed_and_nonfinite_attempts_stay_visible_and_counted():
    items = [
        *replicates([0.8, 0.9]),
        obs(None, replicate="seed=2", status="failed"),
        obs(math.nan, replicate="seed=3"),
        obs(math.inf, replicate="seed=4"),
    ]
    group = summarize(items).pooled()
    assert (group.n, group.n_attempts, group.n_failed, group.n_nonfinite) == (2, 5, 1, 2)
    assert group.mean == pytest.approx(0.85) and len(group.observations) == 5
    nothing = summarize([obs(None, status="failed")]).pooled()
    assert (nothing.n, nothing.mean, nothing.std) == (0, None, None)  # n = 0: neither
    rows = ComparisonReport(summarize(items)).table()
    assert [row["status"] for row in rows if row["row"] == "observation"].count("failed") == 1
    assert {row["value"] for row in rows if row["row"] == "observation"} >= {"NaN", "Infinity", None}


def test_duplicate_attempt_ids_are_rejected():
    with pytest.raises(ComparisonError, match="duplicate attempt ids.*att-x"):
        summarize([obs(0.5, attempt="att-x"), obs(0.6, replicate="seed=1", attempt="att-x")])
    with pytest.raises(ComparisonError, match=r"run:r"):
        summarize([obs(0.5, run_id="r", attempt_id=None), obs(0.6, run_id="r", attempt_id=None, replicate="seed=1")])
    # The same attempt measured on two metrics is two observations, not a duplicate.
    assert len(summarize([obs(0.5), obs(0.4, metric=LOSS)]).groups) == 2


# --- AC3: pairing -------------------------------------------------------------------------------------------


def test_paired_deltas_are_signed_b_minus_a_and_never_flipped_for_minimize():
    a = replicates([0.30, 0.40, 0.50], config="cfg-a", metric=LOSS)
    b = replicates([0.25, 0.45, 0.40], config="cfg-b", metric=LOSS)
    result = compare(a, b, pairing="same seed for initialisation and data order")
    assert [(pair.replicate, pair.delta) for pair in result.pairs] == [
        ("seed=0", pytest.approx(-0.05)),
        ("seed=1", pytest.approx(0.05)),
        ("seed=2", pytest.approx(-0.10)),
    ]
    assert result.mean == pytest.approx(-1 / 30) and result.std == pytest.approx(statistics.stdev([-0.05, 0.05, -0.1]))
    assert result.metric.direction == "minimize"  # negative = B lower; the sign is not flipped
    assert result.state()["delta"] == "b - a" and result.pairing.startswith("same seed")


def test_pairing_needs_unique_compatible_replicate_keys_and_declared_semantics():
    a, b = replicates([0.1, 0.2]), replicates([0.3, 0.4], config="cfg-b")
    with pytest.raises(ComparisonError, match="pairing"):
        compare(a, b, pairing="")
    with pytest.raises(ComparisonError, match="replicate key on every observation"):
        compare([*a, obs(0.3, replicate=None, attempt="att-none")], b, pairing="same seed")
    with pytest.raises(ComparisonError, match="repeats replicate keys"):
        compare([*a, obs(0.3, replicate="seed=0", attempt="att-again")], b, pairing="same seed")
    with pytest.raises(IncompatibleObservations, match="split_id"):
        compare(a, replicates([0.3, 0.4], config="cfg-b", split_id="split:v2"), pairing="same seed")
    with pytest.raises(IncompatibleObservations, match="side a"):
        compare([*a, obs(0.3, replicate="seed=9", config="cfg-c")], b, pairing="same seed")


def test_unmatched_and_incomplete_replicates_are_listed():
    a = replicates([0.1, 0.2, 0.3])
    b = [*replicates([0.15, 0.25], config="cfg-b"), obs(0.9, config="cfg-b", replicate="seed=7")]
    b[1] = obs(None, config="cfg-b", replicate="seed=1", status="failed")
    result = compare(a, b, pairing="same seed")
    assert [(p.replicate, p.delta) for p in result.pairs] == [("seed=0", pytest.approx(0.05)), ("seed=1", None)]
    assert result.unmatched_a == (("seed=2", "att-cfg-a-2"),) and result.unmatched_b == (("seed=7", "att-cfg-b-7"),)
    assert result.n == 1 and result.std is None


def test_a_bootstrap_records_its_unit_seed_and_method_and_is_reproducible():
    a = replicates([0.1, 0.2, 0.3, 0.4])
    b = replicates([0.2, 0.25, 0.45, 0.5], config="cfg-b")
    first = compare(a, b, pairing="same seed", bootstrap=Bootstrap(seed=7, resamples=500))
    again = compare(a, b, pairing="same seed", bootstrap=Bootstrap(seed=7, resamples=500))
    assert first.interval == again.interval and first.interval is not None
    low, high = first.interval
    assert low <= first.mean <= high
    record = first.state()["bootstrap"]
    assert record == {
        "method": "percentile bootstrap of the mean paired delta",
        "unit": "replicate pair",
        "seed": 7,
        "resamples": 500,
        "level": 0.95,
        "label": "seed variability",
    }
    report = ComparisonReport.build([*a, *b], [(a, b, "same seed")], bootstrap=Bootstrap(seed=7, resamples=500))
    assert "seed-variability interval" in report.text()


# --- AC4: exports, order independence, read-only run reading ------------------------------------------------


def _report(order):
    a = replicates([0.30, 0.40, 0.50], config="cfg-a")
    b = [*replicates([0.35, 0.38], config="cfg-b"), obs(None, config="cfg-b", replicate="seed=5", status="failed")]
    items = [*a, *b]
    random.Random(order).shuffle(items)
    return ComparisonReport.build(items, [(list(reversed(a)), b, "same seed")])


def test_table_and_text_carry_every_id_status_selection_unmatched_and_delta():
    report = _report(0)
    rows = report.table()
    kinds = {row["row"] for row in rows}
    assert kinds == {"group", "observation", "pair", "unmatched", "comparison"}
    observed = [
        (row["run_id"], row["attempt_id"], row["status"], row["selection"])
        for row in rows
        if row["row"] == "observation"
    ]
    assert len(observed) == 6 and ("run-cfg-b-5", "att-cfg-b-5", "failed", "last") in observed
    assert [row["delta"] for row in rows if row["row"] == "pair"] == [pytest.approx(0.05), pytest.approx(-0.02)]
    assert [(row["side"], row["replicate"]) for row in rows if row["row"] == "unmatched"] == [
        ("a", "seed=2"),
        ("b", "seed=5"),
    ]
    json.dumps(rows, allow_nan=False)  # machine-readable, strict JSON
    text = report.text()
    for needle in (
        "att-cfg-a-0",
        "run-cfg-b-5",
        "status=failed",
        "@ last",
        "unmatched A: seed=2",
        "= +0.05",
        "delta = B - A",
    ):
        assert needle in text


def test_reports_are_unchanged_by_input_order_and_round_trip(tmp_path):
    reports = [_report(order) for order in range(5)]
    assert len({report.to_json() for report in reports}) == 1 and len({report.text() for report in reports}) == 1
    path = tmp_path / "report.json"
    reports[0].save(path)
    loaded = ComparisonReport.load(path)
    assert loaded.to_json() == reports[0].to_json() and loaded.text() == reports[0].text()
    state = json.loads(path.read_text())
    state["comparisons"][0]["pairs"][0]["delta"] = 0.5  # an edited result no longer matches its observations
    path.write_text(json.dumps(state))
    with pytest.raises(ComparisonError, match="do not match"):
        ComparisonReport.load(path)
    path.write_text('{"format": "nnx.comparison/1", "observations": [], "summary": NaN, "comparisons": []}')
    with pytest.raises(ComparisonError, match="strict JSON"):
        ComparisonReport.load(path)


class _FailAt(Callback):
    def on_epoch_end(self, ctx):
        if ctx.epoch == 1:
            raise RuntimeError("boom in epoch 1")


def _fit(lr: float, seed: int, *, fail: bool = False):
    X = torch.randn(32, 4, generator=torch.Generator().manual_seed(0))
    y = (X[:, 0] > 0).long()
    train = DataLoader(TensorDataset(X[:24], y[:24]), batch_size=8, generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TensorDataset(X[24:], y[24:]), batch_size=8)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    params = NNTrainParams(
        n_epochs=2,
        train_loader=train,
        val_loader=val,
        optim=NNOptimParams.builder().sgd(max_lr=lr).build(),
        seed=seed,
        data_id="toy-32",
    )
    manifest = ExperimentManifest(data={"train": hash_bytes(b"toy-32")}, splits={"val": hash_bytes(b"24:32")})
    if fail:
        with pytest.raises(RuntimeError, match="boom"):
            model.train(params=params, provenance=manifest, callbacks=[_FailAt()])
        return None
    return model.train(params=params, provenance=manifest)


def _tree_digest(root: str) -> dict[str, str]:
    digests = {}
    for folder, _, files in os.walk(root):
        for name in files:
            path = os.path.join(folder, name)
            with open(path, "rb") as handle:
                digests[os.path.relpath(path, root)] = hashlib.sha256(handle.read()).hexdigest()
    return digests


def test_reading_runs_loads_no_model_elects_no_best_and_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    a_runs = [_fit(0.1, seed) for seed in (1, 2)]
    b_runs = [_fit(0.05, seed) for seed in (1, 2)]
    _fit(0.05, 3, fail=True)
    ids = sorted(name for name in os.listdir("runs") if name != "best" and not name.startswith("."))
    assert len(ids) == 5
    before = _tree_digest("runs")

    def no_load(*args, **kwargs):
        raise AssertionError("reading observations must not load a model or a checkpoint")

    monkeypatch.setattr(torch, "load", no_load)
    monkeypatch.setattr(NNModel, "from_checkpoint", no_load)
    monkeypatch.setattr(NNCheckpoint, "load", no_load)
    reads: list[str] = []
    read_csv = pd.read_csv

    def counted(path, *args, **kwargs):
        reads.append(os.fspath(path))
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", counted)
    found = observations_from_runs(ids, metric=Metric("loss", "minimize"))
    assert sorted(os.path.relpath(path) for path in reads) == sorted(
        os.path.join("runs", run_id, "idps.csv") for run_id in ids
    )  # one read each
    by_run = {item.run_id: item for item in found}
    a_ids, b_ids = [run.id for run in a_runs], [run.id for run in b_runs]
    failed = [item for item in found if item.status == "failed"]
    assert len(failed) == 1 and failed[0].replicate == "seed=3" and failed[0].value is not None  # epoch 0 committed
    assert all(by_run[run_id].status == "completed" and by_run[run_id].finite for run_id in a_ids + b_ids)
    configs = {by_run[run_id].config for run_id in a_ids} | {by_run[run_id].config for run_id in b_ids}
    assert len(configs) == 2 and by_run[a_ids[0]].config == by_run[a_ids[1]].config  # the seed is not configuration
    first = by_run[a_ids[0]]
    assert first.attempt_id == a_runs[0].provenance.attempt.attempt_id and first.replicate == "seed=1"
    assert first.data is not None and "digest" in first.data and first.split_id is not None
    assert first.evaluation.startswith("epoch 1 validation record") and "last generation" in first.evaluation
    expected = a_runs[0].idps[-1].val_edp.loss
    assert first.value == pytest.approx(expected)

    report = ComparisonReport.build(
        found, [([by_run[i] for i in a_ids], [by_run[i] for i in b_ids], "same seed, same split")]
    )
    report.save(tmp_path / "comparison.json")
    assert ComparisonReport.load(tmp_path / "comparison.json").to_json() == report.to_json()
    assert _tree_digest("runs") == before  # nothing written, no checkpoint, no best pointer elected
    assert len(report.summary.groups) == 2 and report.comparisons[0].n == 2


def test_selection_best_reads_the_monitor_election_or_stays_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    run = _fit(0.1, 4)
    (best,) = observations_from_runs([run.id], metric=Metric("loss", "minimize"), selection="best")
    assert best.value is None and best.selection == "best"  # no monitor elected anything: unknown, not guessed
    (last,) = observations_from_runs([run.id], metric=Metric("accuracy", "maximize"), replicate=None)
    assert last.replicate is None and last.value is not None
    (missing,) = observations_from_runs([run.id], metric=Metric("no_such_metric", "maximize"))
    assert missing.value is None and missing.status == "completed" and not missing.finite
    with pytest.raises(ComparisonError, match="split"):
        observations_from_runs([run.id], metric=LOSS, split="test")
    with pytest.raises(ValueError, match="path separators"):
        observations_from_runs(["../x"], metric=LOSS)


def test_the_module_is_public_and_complete():
    assert set(comparison.__all__) >= {
        "Bootstrap",
        "ComparisonReport",
        "Metric",
        "Observation",
        "compare",
        "observations_from_runs",
        "summarize",
    }
    assert all(getattr(comparison, name, None) is not None for name in comparison.__all__)
    np.testing.assert_allclose(summarize(replicates([1.0, 3.0])).pooled().std, math.sqrt(2.0))
