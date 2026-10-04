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
import pickle
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


def _fit(lr: float, seed: int, *, fail: bool = False, history=None):
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
    return model.train(params=params, provenance=manifest, history=history)


def _tree_digest(root: str) -> dict[str, str]:
    """Every entry under ``root``: file digests, symlink targets (the
    ``runs/best`` pointer) and directories."""
    digests = {}
    for folder, dirs, files in os.walk(root):
        for name in [*dirs, *files]:
            path = os.path.join(folder, name)
            key = os.path.relpath(path, root)
            if os.path.islink(path):
                digests[key] = "link:" + os.readlink(path)
            elif os.path.isdir(path):
                digests[key] = "dir"
            else:
                with open(path, "rb") as handle:
                    digests[key] = hashlib.sha256(handle.read()).hexdigest()
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


def test_a_journaled_run_is_read_from_its_history_journal(tmp_path, monkeypatch):
    from nnx.history import HistoryJournal

    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    (tmp_path / "csv").mkdir()
    (tmp_path / "journal").mkdir()
    monkeypatch.chdir(tmp_path / "csv")
    torch.manual_seed(0)  # the same initial weights for both runs
    plain = _fit(0.1, 5)
    (a,) = observations_from_runs([plain.id], metric=LOSS)
    (a_train,) = observations_from_runs([plain.id], metric=LOSS, split="train")
    monkeypatch.chdir(tmp_path / "journal")
    torch.manual_seed(0)
    journaled = _fit(0.1, 5, history=HistoryJournal(retention=2, chunk_size=2))  # 6 records, 2 kept in memory
    assert not os.path.exists(os.path.join("runs", journaled.id, "idps.csv"))
    before = _tree_digest("runs")
    (b,) = observations_from_runs([journaled.id], metric=LOSS)
    (b_train,) = observations_from_runs([journaled.id], metric=LOSS, split="train")
    assert (
        b.status == "completed" and b.finite and b.evaluation.startswith("epoch 1 validation record (history journal)")
    )
    assert b.value == pytest.approx(a.value) and b_train.value == pytest.approx(a_train.value)
    assert _tree_digest("runs") == before


def _forbid_unpickling(monkeypatch) -> None:
    """Every way to unpickle a checkpoint fails the test."""

    def no_load(*args, **kwargs):
        raise AssertionError("reading observations must not unpickle a checkpoint")

    monkeypatch.setattr(torch, "load", no_load)
    monkeypatch.setattr(pickle, "load", no_load)
    monkeypatch.setattr(pickle, "loads", no_load)
    monkeypatch.setattr(NNModel, "from_checkpoint", no_load)
    for loader in ("load", "from_file", "load_training_state", "load_with_training_state", "load_optimizer_state"):
        monkeypatch.setattr(NNCheckpoint, loader, no_load)


def test_a_journaled_run_is_read_without_unpickling_a_checkpoint(tmp_path, monkeypatch):
    from nnx.history import HistoryJournal, iter_history

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    run = _fit(0.1, 5, history=HistoryJournal(retention=2, chunk_size=2))
    last = list(iter_history(run.id))[-1]  # the last committed record, as readers show it
    readings = ((LOSS, "validation", "last"), (LOSS, "train", "last"), (ACC, "validation", "best"))
    expected = [observations_from_runs([run.id], metric=m, split=s, selection=r) for m, s, r in readings]
    _forbid_unpickling(monkeypatch)
    assert [observations_from_runs([run.id], metric=m, split=s, selection=r) for m, s, r in readings] == expected
    (item,) = expected[0]
    assert item.value == last.val_edp.loss
    assert item.evaluation.startswith(f"epoch {last.epoch_idx} validation record (history journal)")


@pytest.mark.parametrize("journal", [False, True], ids=["idps.csv", "history journal"])
def test_an_epoch_written_before_its_last_checkpoint_landed_is_never_read_as_committed(journal, tmp_path, monkeypatch):
    import nnx.history as history_module
    from nnx.history import HistoryJournal, iter_history

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    save = NNCheckpoint.save

    def killed(self, *args, **kwargs):
        if self.idp.epoch_idx == 1:
            raise RuntimeError("killed before LAST")
        return save(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        # Epoch 1's history is written, its LAST never lands and nothing is rolled back.
        patch.setattr(NNCheckpoint, "save", killed)
        patch.setattr(history_module._EagerHistory, "rollback_epoch", lambda self, run: None)
        patch.setattr(history_module._JournalWriter, "rollback", lambda self: None)
        with pytest.raises(RuntimeError, match="killed before LAST"):
            _fit(0.1, 5, history=HistoryJournal(retention=2, chunk_size=2) if journal else None)
    (run_id,) = [name for name in os.listdir("runs") if name != "best" and not name.startswith(".")]
    run_path = os.path.join("runs", run_id)
    if journal:
        assert json.load(open(os.path.join(run_path, "history", "journal.json")))["epochs"] == 2  # epoch 1 published
    else:
        assert set(pd.read_csv(os.path.join(run_path, "idps.csv"))["epoch_idx"]) == {0, 1}
    committed = list(iter_history(run_id))[-1]  # up to LAST's epoch
    assert committed.epoch_idx == 0
    _forbid_unpickling(monkeypatch)
    (item,) = observations_from_runs([run_id], metric=LOSS)
    assert item.status == "failed" and item.value == pytest.approx(committed.val_edp.loss)
    assert item.evaluation.startswith("epoch 0 validation record")
    # A run killed before recording its end names no committed epoch: its value is unknown.
    attempt = os.path.join(run_path, "attempt.json")
    record = json.loads(open(attempt).read())
    with open(attempt, "w") as handle:
        json.dump({**record, "status": "running", "finished_at": None, "last_committed": None}, handle)
    (running,) = observations_from_runs([run_id], metric=LOSS)
    assert running.status == "running" and running.value is None
    assert running.evaluation == "unknown: the committed epoch is not recorded"
    # A legacy run (no commit marker) keeps its whole history, unfiltered, as before.
    os.remove(os.path.join(run_path, ".history-committed-by-last"))
    (legacy,) = observations_from_runs([run_id], metric=LOSS)
    assert legacy.evaluation.startswith("epoch 1 validation record")


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


# --- review hardening -----------------------------------------------------------------------------------


def _fit_params(seed: int, **overrides):
    X = torch.randn(32, 4, generator=torch.Generator().manual_seed(0))
    y = (X[:, 0] > 0).long()
    fields = dict(
        n_epochs=2,
        train_loader=DataLoader(TensorDataset(X[:24], y[:24]), batch_size=8),
        val_loader=DataLoader(TensorDataset(X[24:], y[24:]), batch_size=8),
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        seed=seed,
    )
    fields.update(overrides)
    return NNTrainParams(**fields)


def _model():
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


MANIFEST = ExperimentManifest(data={"train": hash_bytes(b"toy-32")})


def test_trainer_and_resumed_runs_pool_across_seeds(tmp_path, monkeypatch):
    from nnx.objectives import supervised_objective
    from nnx.trainer import NNTrainerParams, Trainer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    X = torch.randn(16, 4, generator=torch.Generator().manual_seed(0))
    loader = DataLoader(TensorDataset(X, (X[:, 0] > 0).long()), batch_size=8)
    runs = []
    for seed in (0, 1):
        params = (
            NNTrainerParams.builder()
            .n_epochs(1)
            .seed(seed)
            .train_loader(loader)
            .optimizer("default", NNOptimParams.builder().sgd(max_lr=0.1).build())
            .save_phase_checkpoints(False)
            .build()
        )
        runs.append(Trainer(_model()).train(params=params, objective=supervised_objective(), provenance=MANIFEST))
    found = observations_from_runs([run.id for run in runs], metric=Metric("loss", "minimize"), split="train")
    assert len({item.config for item in found}) == 1 and [item.replicate for item in found] == ["seed=0", "seed=1"]

    parents = [_model().train(params=_fit_params(seed, n_epochs=1), provenance=MANIFEST) for seed in (3, 4)]
    children = [
        _model().train(params=_fit_params(seed, n_epochs=1, resume_from_run_id=parent.id), provenance=MANIFEST)
        for seed, parent in zip((3, 4), parents, strict=True)
    ]
    resumed = observations_from_runs([run.id for run in children], metric=Metric("loss", "minimize"))
    assert len(summarize(resumed).groups) == 1  # the resume lineage is per replicate, not configuration


def test_a_report_state_that_is_not_json_is_a_comparison_error():
    state = _report(0).state()
    state["summary"] = {"groups": {1, 2}}  # a set is not JSON
    with pytest.raises(ComparisonError, match="malformed comparison report"):
        ComparisonReport.from_state(state)


def test_a_continuation_never_pools_with_fresh_runs_or_its_parent(tmp_path, monkeypatch):
    """``n_epochs`` counts the epochs a run adds: a stateful continuation
    trained longer than a fresh run, and a weights-only warm start began
    from other weights, so neither is a replicate of a fresh run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    fresh = [_model().train(params=_fit_params(seed), provenance=MANIFEST) for seed in (0, 1, 2)]
    stateful = _model().train(params=_fit_params(3, resume_from_run_id=fresh[0].id), provenance=MANIFEST)
    warm = _model().train(
        params=_fit_params(4, resume_from_run_id=fresh[0].id, resume_mode="weights_only"), provenance=MANIFEST
    )
    found = observations_from_runs([run.id for run in [*fresh, stateful, warm]], metric=Metric("loss", "minimize"))
    groups = summarize(found).groups
    assert sorted(group.n for group in groups) == [1, 1, 3]
    assert len({item.config for item in found[:3]}) == 1 and len({item.config for item in found}) == 3

    # A run plus its own same-seed continuation are two configurations, not two seed=1 replicates.
    same_seed = _model().train(params=_fit_params(1, resume_from_run_id=fresh[1].id), provenance=MANIFEST)
    pair = observations_from_runs([fresh[1].id, same_seed.id], metric=Metric("loss", "minimize"))
    assert pair[0].config != pair[1].config
    assert pair[1].config == found[3].config  # continuations with one schedule still pool


def test_a_child_pools_only_with_children_of_identically_configured_parents(tmp_path, monkeypatch):
    import shutil

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    loss = Metric("loss", "minimize")
    slow = NNOptimParams.builder().sgd(max_lr=0.001).build()
    parents_a = [_model().train(params=_fit_params(seed), provenance=MANIFEST) for seed in (0, 1)]
    parents_b = [_model().train(params=_fit_params(seed, optim=slow), provenance=MANIFEST) for seed in (0, 1)]
    children = [
        _model().train(
            params=_fit_params(10 + i, resume_from_run_id=parent.id, resume_mode="weights_only"), provenance=MANIFEST
        )
        for i, parent in enumerate([*parents_a, *parents_b])
    ]
    found = observations_from_runs([run.id for run in children], metric=loss)
    assert found[0].config == found[1].config and found[2].config == found[3].config
    assert found[0].config != found[2].config  # fine-tunes of different pretraining never pool
    assert sorted(group.n for group in summarize(found).groups) == [2, 2]

    # Born-again generations (lineage only, no resume): each generation is its own configuration.
    gen0 = _model().train(params=_fit_params(20), provenance=MANIFEST)
    gen1 = _model().train(params=_fit_params(21, parent_run_id=gen0.id), provenance=MANIFEST)
    gen2 = _model().train(params=_fit_params(22, parent_run_id=gen1.id), provenance=MANIFEST)
    generations = observations_from_runs([gen0.id, gen1.id, gen2.id], metric=loss)
    assert len({item.config for item in generations}) == 3

    # A deleted parent leaves an unknown lineage: its children no longer pool, not even with each other.
    orphans = [_model().train(params=_fit_params(23 + i, parent_run_id=gen0.id), provenance=MANIFEST) for i in (0, 1)]
    assert {item.config for item in observations_from_runs([o.id for o in orphans], metric=loss)} == {
        generations[1].config
    }
    shutil.rmtree(os.path.join("runs", gen0.id))
    alone = observations_from_runs([o.id for o in orphans], metric=loss)
    assert len({item.config for item in alone} | {generations[1].config}) == 3

    # A resumed run whose metadata.yaml cannot be read is refused; a fresh one never reads it.
    with open(os.path.join("runs", children[0].id, "metadata.yaml"), "wb") as handle:
        handle.write(b"\xff\xfe not utf-8")
    with pytest.raises(ComparisonError, match="metadata.yaml is unreadable"):
        observations_from_runs([children[0].id], metric=loss)
    with open(os.path.join("runs", parents_a[0].id, "metadata.yaml"), "w") as handle:
        handle.write("when: 2020-13-45\n")
    assert observations_from_runs([parents_a[0].id], metric=loss)[0].value is not None


def test_a_parents_declared_data_and_lineage_errors_are_part_of_the_identity(tmp_path, monkeypatch):
    import shutil

    import yaml

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    loss = Metric("loss", "minimize")
    on_x = ExperimentManifest(data={"train": hash_bytes(b"pretrain-x")})
    on_y = ExperimentManifest(data={"train": hash_bytes(b"pretrain-y")})
    parents = [
        _model().train(params=_fit_params(0), salt=f"{name}{i}", provenance=manifest)
        for name, manifest in (("x", on_x), ("y", on_y))
        for i in range(2)
    ]
    children = [
        _model().train(
            params=_fit_params(10 + i, resume_from_run_id=parent.id, resume_mode="weights_only"), provenance=MANIFEST
        )
        for i, parent in enumerate(parents)
    ]
    found = observations_from_runs([run.id for run in children], metric=loss)
    assert found[0].config == found[1].config and found[2].config == found[3].config
    assert found[0].config != found[2].config  # pretraining declared only through provenance= still splits
    assert sorted(group.n for group in summarize(found).groups) == [2, 2]

    # Lineages are resolved the same whatever order the runs are listed in.
    def fake(run_id, parent_id, recorded=True):
        source = os.path.join("runs", children[0].id)
        state = yaml.safe_load(open(os.path.join(source, "run.yaml")))
        state["train"]["parent_run_id"] = parent_id
        os.makedirs(os.path.join("runs", run_id))
        with open(os.path.join("runs", run_id, "run.yaml"), "w") as handle:
            yaml.safe_dump(state, handle)
        if recorded:  # each run records the parent attempt it started from
            shutil.copy(os.path.join(source, "provenance.json"), os.path.join("runs", run_id))
            attempt = json.loads(open(os.path.join(source, "attempt.json")).read())
            parent = {"attempt_id": f"a{parent_id}", "checkpoint": "last", "epoch": 1, "run_id": parent_id}
            attempt.update(attempt_id=f"a{run_id}", run_id=run_id, parent=parent if parent_id else None)
            with open(os.path.join("runs", run_id, "attempt.json"), "w") as handle:
                json.dump(attempt, handle)

    from nnx.comparison import MAX_LINEAGE

    chain = [f"{i:032x}" for i in range(MAX_LINEAGE + 2)]
    fake(chain[0], None)
    for parent_id, run_id in zip(chain[:-1], chain[1:], strict=True):
        fake(run_id, parent_id)
    for order in (chain, chain[::-1], chain[-1:]):
        with pytest.raises(ComparisonError, match=f"more than {MAX_LINEAGE} ancestors"):
            observations_from_runs(order, metric=loss)
    assert len(observations_from_runs(chain[: MAX_LINEAGE + 1][::-1], metric=loss)) == MAX_LINEAGE + 1

    # A cycle is named; the moving "best" alias is never followed, and without
    # an attempt record its children share nothing.
    fake("a" * 32, "b" * 32)
    fake("b" * 32, "a" * 32)
    with pytest.raises(ComparisonError, match="cycle"):
        observations_from_runs(["a" * 32], metric=loss)
    fake("c" * 32, "best", recorded=False)
    fake("d" * 32, "best", recorded=False)
    pair = observations_from_runs(["c" * 32, "d" * 32], metric=loss)
    assert len({pair[0].config, pair[1].config, found[0].config}) == 3
    # An ancestor's unreadable provenance is raised, not swallowed.
    with open(os.path.join("runs", parents[0].id, "provenance.json"), "w") as handle:
        handle.write("{not json")
    with pytest.raises(ComparisonError, match="provenance"):
        observations_from_runs([children[0].id], metric=loss)


def test_children_of_the_best_alias_or_a_retrained_parent_are_named_by_the_attempt_they_started_from(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    loss = Metric("loss", "minimize")
    best = os.path.join("runs", "best")

    def point_best(run):
        if os.path.islink(best) or os.path.isfile(best):
            os.remove(best)
        elif os.path.isdir(best):
            import shutil

            shutil.rmtree(best)
        os.symlink(os.path.abspath(os.path.join("runs", run.id)), best)

    p1 = _model().train(params=_fit_params(0), provenance=MANIFEST)
    point_best(p1)
    c1 = _model().train(params=_fit_params(10, resume_from_run_id="best"), provenance=MANIFEST)
    point_best(c1)
    c2 = _model().train(params=_fit_params(11, resume_from_run_id="best"), provenance=MANIFEST)  # P1's grandchild
    point_best(p1)
    c3 = _model().train(params=_fit_params(12, resume_from_run_id="best"), provenance=MANIFEST)  # P1's child
    found = observations_from_runs([c1.id, c2.id, c3.id], metric=loss)
    assert found[0].config == found[2].config != found[1].config

    # A parent retrained in place since: its old child no longer pools with a new one.
    on_x = ExperimentManifest(data={"train": hash_bytes(b"x")})
    on_y = ExperimentManifest(data={"train": hash_bytes(b"y")})
    parent = _model().train(params=_fit_params(20), provenance=on_x)
    old = _model().train(params=_fit_params(21, resume_from_run_id=parent.id), provenance=MANIFEST)
    _model().train(params=_fit_params(20, overwrite_existing=True), provenance=on_y)
    new = _model().train(params=_fit_params(22, resume_from_run_id=parent.id), provenance=MANIFEST)
    pair = observations_from_runs([old.id, new.id], metric=loss)
    assert pair[0].config != pair[1].config

    # A parent directory without its run.yaml is unreadable, not deleted.
    os.remove(os.path.join("runs", parent.id, "run.yaml"))
    with pytest.raises(ComparisonError, match="run.yaml"):
        observations_from_runs([new.id], metric=loss)


def test_a_child_of_a_parent_still_training_or_retrained_since_is_named_by_what_it_started_from(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    loss = Metric("loss", "minimize")
    warm = {"resume_mode": "weights_only"}
    spawned = {}

    class Spawn(Callback):  # fine-tunes from the parent's first epoch while the parent still trains
        def on_epoch_end(self, ctx):
            last = os.path.join("runs", ctx.run.id, "checkpoints", "last.pt")
            if "early" not in spawned and os.path.isfile(last):
                spawned["early"] = _model().train(
                    params=_fit_params(10, resume_from_run_id=ctx.run.id, **warm), provenance=MANIFEST
                )
                spawned["while_running"] = observations_from_runs([spawned["early"].id], metric=loss)[0].config

    parent = _model().train(params=_fit_params(0, n_epochs=4), provenance=MANIFEST, callbacks=[Spawn()])
    late = _model().train(params=_fit_params(11, resume_from_run_id=parent.id, **warm), provenance=MANIFEST)
    later = _model().train(params=_fit_params(12, resume_from_run_id=parent.id, **warm), provenance=MANIFEST)
    early, final = observations_from_runs([spawned["early"].id, late.id], metric=loss)
    assert early.config != final.config  # a 1-epoch-pretrained fine-tune is not a replicate of a 4-epoch one
    assert early.config == spawned["while_running"]  # and its identity does not change when the parent finishes

    # Once the parent is retrained, a record without its checkpoint generation still names the epoch.
    _model().train(params=_fit_params(0, n_epochs=4, overwrite_existing=True), provenance=MANIFEST)
    for child in (spawned["early"], late, later):
        path = os.path.join("runs", child.id, "attempt.json")
        state = json.loads(open(path).read())
        state["parent"]["generation"] = None
        with open(path, "w") as handle:
            json.dump(state, handle)
    early, final, sibling = observations_from_runs([spawned["early"].id, late.id, later.id], metric=loss)
    assert early.config != final.config == sibling.config  # siblings at one epoch of that attempt still pool

    # Lineage without resume (parent_run_id) records no parent attempt: a child that started before the
    # parent's current attempt finished is not followed into the retrained parent.
    on_x = ExperimentManifest(data={"train": hash_bytes(b"x")})
    on_y = ExperimentManifest(data={"train": hash_bytes(b"y")})
    source = _model().train(params=_fit_params(5), provenance=on_x)
    old = _model().train(params=_fit_params(50, parent_run_id=source.id), provenance=MANIFEST)
    _model().train(params=_fit_params(5, overwrite_existing=True), provenance=on_y)
    new = _model().train(params=_fit_params(51, parent_run_id=source.id), provenance=MANIFEST)
    pair = observations_from_runs([old.id, new.id], metric=loss)
    assert pair[0].config != pair[1].config


def test_continuations_of_killed_replicate_parents_pool_by_the_epoch_they_started_from(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    loss = Metric("loss", "minimize")

    class Kill(Callback):
        def __init__(self, at):
            self.at = at

        def on_epoch_end(self, ctx):
            if ctx.epoch == self.at:
                raise RuntimeError("preempted")

    def saved_runs():
        entries = os.listdir("runs") if os.path.isdir("runs") else []
        return {e for e in entries if os.path.isfile(os.path.join("runs", e, "run.yaml")) and e != "best"}

    parents = []
    for seed, at in ((0, 2), (1, 2), (2, 1)):
        before = saved_runs()
        with pytest.raises(RuntimeError, match="preempted"):
            _model().train(params=_fit_params(seed, n_epochs=4), provenance=MANIFEST, callbacks=[Kill(at)])
        (run_id,) = saved_runs() - before
        path = os.path.join("runs", run_id, "attempt.json")
        state = json.loads(open(path).read())
        state.update(status="running", finished_at=None)  # what a SIGKILL leaves behind
        with open(path, "w") as handle:
            json.dump(state, handle)
        parents.append(run_id)
    children = [
        _model().train(params=_fit_params(10 + i, n_epochs=4, resume_from_run_id=parent), provenance=MANIFEST)
        for i, parent in enumerate(parents)
    ]
    found = observations_from_runs([run.id for run in children], metric=loss)
    assert found[0].config == found[1].config  # replicate parents killed at one epoch: their continuations pool
    assert found[2].config != found[0].config  # one killed earlier started from another epoch
    for child in children:  # a run whose resume mode predates metadata.yaml may have resumed: its epoch counts
        os.rename(os.path.join("runs", child.id, "metadata.yaml"), os.path.join("runs", child.id, "metadata.bak"))
    older = observations_from_runs([run.id for run in children], metric=loss)
    assert older[0].config == older[1].config != older[2].config
    for child in children:
        os.rename(os.path.join("runs", child.id, "metadata.bak"), os.path.join("runs", child.id, "metadata.yaml"))

    def edit_record(run, **changes):
        path = os.path.join("runs", run.id, "attempt.json")
        state = json.loads(open(path).read())
        state["parent"].update(changes)
        with open(path, "w") as handle:
            json.dump(state, handle)

    # The start epoch of a child of an unfinished parent is needed: unrecorded, the child pools with
    # nothing — not even with the continuation of a completed replicate, whose start epoch is an outcome.
    done = _model().train(params=_fit_params(3, n_epochs=4), provenance=MANIFEST)
    after = _model().train(params=_fit_params(13, n_epochs=4, resume_from_run_id=done.id), provenance=MANIFEST)
    edit_record(children[0], epoch=None)
    unknown, sibling, completed = observations_from_runs([children[0].id, children[1].id, after.id], metric=loss)
    assert len({unknown.config, sibling.config, completed.config}) == 3
    edit_record(children[1], epoch=None)  # two unknown start epochs are not one
    assert len({item.config for item in observations_from_runs([c.id for c in children[:2]], metric=loss)}) == 2


def test_a_completed_parents_stopping_epoch_is_an_outcome_and_an_unknown_lineage_never_pools(tmp_path, monkeypatch):
    from nnx import EarlyStopping, MonitorSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    loss = Metric("loss", "minimize")
    monitor = MonitorSpec("loss", split="val")
    # Early-stopped replicate parents complete at different epochs: their `last` continuations still pool.
    parents = [
        _model().train(
            params=_fit_params(seed, n_epochs=6, monitor=monitor),
            provenance=MANIFEST,
            callbacks=[EarlyStopping(patience=0)],
        )
        for seed in (0, 1, 2)
    ]
    epochs = {len({idp.epoch_idx for idp in parent.idps}) for parent in parents}
    children = [
        _model().train(
            params=_fit_params(10 + i, resume_from_run_id=p.id, resume_mode="weights_only"), provenance=MANIFEST
        )
        for i, p in enumerate(parents)
    ]
    found = observations_from_runs([c.id for c in children], metric=loss)
    assert len({item.config for item in found}) == 1, epochs

    # A child that recorded nothing (lineage only) of a parent still training never pools.
    spawned = {}

    class Spawn(Callback):
        def on_epoch_end(self, ctx):
            if "early" not in spawned and ctx.epoch == 0:
                spawned["early"] = _model().train(params=_fit_params(30, parent_run_id=ctx.run.id), provenance=MANIFEST)

    teacher = _model().train(params=_fit_params(20, n_epochs=3), provenance=MANIFEST, callbacks=[Spawn()])
    late = _model().train(params=_fit_params(31, parent_run_id=teacher.id), provenance=MANIFEST)
    early, final = observations_from_runs([spawned["early"].id, late.id], metric=loss)
    assert early.config != final.config


def test_with_nothing_recorded_only_a_parent_known_to_have_completed_first_is_followed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    loss = Metric("loss", "minimize")
    students = {}

    class Spawn(Callback):  # a lineage-only student that starts while its teacher still trains
        def on_epoch_end(self, ctx):
            if ctx.run.id not in students:
                students[ctx.run.id] = _model().train(
                    params=_fit_params(40 + len(students), parent_run_id=ctx.run.id), provenance=MANIFEST
                )

    def configs(*runs):
        return [item.config for item in observations_from_runs([run.id for run in runs], metric=loss)]

    # A teacher with provenance: students that started after it completed pool, an early one does not.
    teacher = _model().train(params=_fit_params(0, n_epochs=3), provenance=MANIFEST, callbacks=[Spawn()])
    late = [_model().train(params=_fit_params(50 + i, parent_run_id=teacher.id), provenance=MANIFEST) for i in (0, 1)]
    early, *after = configs(students[teacher.id], *late)
    assert after[0] == after[1] != early

    # A teacher without provenance has no recorded order with its students: none of them pools.
    bare = _model().train(params=_fit_params(1, n_epochs=3), callbacks=[Spawn()])
    late = [_model().train(params=_fit_params(60 + i, parent_run_id=bare.id), provenance=MANIFEST) for i in (0, 1)]
    assert len(set(configs(students[bare.id], *late))) == 3

    # Students without provenance never recorded when they started: they pool with nothing.
    blind = [_model().train(params=_fit_params(70 + i, parent_run_id=teacher.id)) for i in (0, 1)]
    assert len({*configs(*blind), after[0]}) == 3

    # A resumed child of a teacher without provenance carries the start epoch it recorded.
    resumed = [
        _model().train(
            params=_fit_params(80 + i, resume_from_run_id=bare.id, resume_mode="weights_only"), provenance=MANIFEST
        )
        for i in (0, 1)
    ]
    assert len(set(configs(*resumed))) == 1
    for run in resumed:  # followed into the teacher, not named by the checkpoint they recorded
        path = os.path.join("runs", run.id, "attempt.json")
        state = json.loads(open(path).read())
        state["parent"]["generation"] = None
        with open(path, "w") as handle:
            json.dump(state, handle)
    assert len(set(configs(*resumed))) == 1


def test_recorded_timestamps_are_compared_as_instants():
    from types import SimpleNamespace

    from nnx.comparison import _completed_before

    def run(status="completed", started=None, finished=None):
        return SimpleNamespace(attempt=SimpleNamespace(status=status, started_at=started, finished_at=finished))

    # 01:00 at UTC+2 is 23:00 UTC the day before: completed before a midnight UTC start ("Z").
    assert _completed_before(run(started="2026-01-01T00:00:00Z"), run(finished="2026-01-01T01:00:00+02:00"))
    assert not _completed_before(run(started="2026-01-01T00:00:00+00:00"), run(finished="2026-01-01T00:30:00+00:00"))
    for started, finished in (
        ("not a time", "2026-01-01T00:00:00+00:00"),
        ("2026-01-02T00:00:00", "2026-01-01T00:00:00"),
    ):
        assert not _completed_before(run(started=started), run(finished=finished))  # unknown or naive: no order
    assert not _completed_before(run(started="2026-01-02T00:00:00Z"), run("failed", finished="2026-01-01T00:00:00Z"))
    assert not _completed_before(SimpleNamespace(attempt=None), run(finished="2026-01-01T00:00:00Z"))


def test_best_selection_names_the_declared_monitor_even_without_an_election(tmp_path, monkeypatch):
    from nnx import MonitorSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monitor = MonitorSpec("loss", split="val")
    runs = [_model().train(params=_fit_params(seed, monitor=monitor), provenance=MANIFEST) for seed in (0, 1)]
    attempt = os.path.join("runs", runs[1].id, "attempt.json")
    state = json.loads(open(attempt).read())
    state.update(status="running", last_committed=None)  # as a killed run leaves it
    with open(attempt, "w") as handle:
        json.dump(state, handle)
    found = observations_from_runs([run.id for run in runs], metric=Metric("loss", "minimize"), selection="best")
    assert {item.selection for item in found} == {"best:loss(val,min)"}
    assert len(summarize(found).groups) == 1 and found[1].value is None and found[1].status == "running"
    plain = _model().train(params=_fit_params(5), provenance=MANIFEST)
    (no_monitor,) = observations_from_runs([plain.id], metric=Metric("loss", "minimize"), selection="best")
    assert no_monitor.selection == "best" and no_monitor.value is None
    assert no_monitor.evaluation == "unknown: the run declares no monitor"


def test_train_split_reads_only_a_whole_epoch_summary(tmp_path, monkeypatch):
    from nnx import MonitorSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    plain = _model().train(params=_fit_params(0), provenance=MANIFEST)
    summarized = _model().train(params=_fit_params(1, monitor=MonitorSpec("loss", split="train")), provenance=MANIFEST)
    without, with_summary = observations_from_runs(
        [plain.id, summarized.id],
        metric=Metric("loss", "minimize"),
        split="train",
        config={plain.id: "c", summarized.id: "c"},
    )
    assert without.value is None and "no whole-epoch training summary" in without.evaluation  # never a last batch
    assert with_summary.value == pytest.approx(summarized.idps[-1].train_summary.loss)


def test_a_report_compares_only_its_own_observations():
    a = replicates([0.3, 0.4], config="cfg-a")
    b = replicates([0.5, 0.6], config="cfg-b")
    altered = replicates([0.9, 0.9], config="cfg-b")  # same attempt ids, other values
    with pytest.raises(ComparisonError, match="report's own observations"):
        ComparisonReport.build([*a, *b], [(a, altered, "same seed")])
    elsewhere = [obs(item.value, config="cfg-b", replicate=item.replicate, split="train") for item in b]
    with pytest.raises(ComparisonError, match="report's own observations"):
        ComparisonReport.build([*a, *b], [(a, elsewhere, "same seed")])


def test_malformed_report_files_are_comparison_errors(tmp_path):
    good = json.loads(_report(0).to_json())
    path = tmp_path / "report.json"
    cases = []
    broken = json.loads(json.dumps(good))
    del broken["comparisons"][0]["metric"]
    cases.append(broken)
    broken = json.loads(json.dumps(good))
    broken["comparisons"] = ["x"]
    cases.append(broken)
    broken = json.loads(json.dumps(good))
    broken["observations"] = [1]
    cases.append(broken)
    for state in cases:
        path.write_text(json.dumps(state))
        with pytest.raises(ComparisonError):
            ComparisonReport.load(path)


def test_numpy_scalars_huge_values_and_declared_unknown_identities():
    item = obs(np.float32(0.25), replicate="seed=0")
    assert item.value == 0.25 and isinstance(item.value, float)
    assert obs(np.int64(1)).value == 1.0
    huge = summarize(replicates([1e200, -1e200, 1e200])).pooled()
    assert huge.std == pytest.approx(statistics.stdev([1.0, -1.0, 1.0]) * 1e200)  # finite: never overflowed
    assert summarize(replicates([1e308, 1e308])).pooled().std == 0.0
    a = replicates([-1e308, 0.0], config="cfg-a")
    b = replicates([1e308, 1.0], config="cfg-b")
    report = ComparisonReport.build([*a, *b], [(a, b, "same seed")])
    assert report.comparisons[0].pairs[0].delta == math.inf and report.comparisons[0].n == 1
    assert json.loads(report.to_json())["comparisons"][0]["pairs"][0]["delta"] == "Infinity"
    from nnx.comparison import _identities
    from nnx.provenance import IdentityRef

    assert _identities({"train": IdentityRef.unknown()}) is None


def test_registered_model_seeds_are_replicates_not_configuration(tmp_path, monkeypatch):
    from torch import nn

    from nnx import ModelSpec, register_model_factory, unregister_model_factory

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    register_model_factory(
        "cmp-mlp", 1, lambda config: nn.Sequential(nn.Linear(4, config["h"]), nn.ReLU(), nn.Linear(config["h"], 2))
    )
    try:
        runs = []
        for seed in (0, 1, 2):
            model = NNModel(
                params=NNModelParams(net=ModelSpec("cmp-mlp", 1, {"h": 8}, seed=seed + 10), loss=Losses.CROSS_ENTROPY)
            )
            runs.append(model.train(params=_fit_params(seed), provenance=MANIFEST))
        found = observations_from_runs([run.id for run in runs], metric=Metric("loss", "minimize"))
        assert len(summarize(found).groups) == 1  # the ModelSpec seed is a replicate, not configuration
        assert [item.replicate for item in found] == [
            "seed=0,init_seed=10",
            "seed=1,init_seed=11",
            "seed=2,init_seed=12",
        ]
    finally:
        unregister_model_factory("cmp-mlp", 1)
    from nnx.comparison import _config_identity

    state = {"model": {"device": "cpu", "loss": "x"}, "train": {"seed": 1}}
    assert _config_identity(state) == _config_identity({**state, "model": {"device": "cuda", "loss": "x"}})


def test_round_two_edges():
    a = replicates([0.3, math.nan], config="cfg-a")
    b = replicates([0.5, 0.6], config="cfg-b")
    copy = [Observation(**{**item.__dict__}) for item in a]  # equal in every field, NaN included
    ComparisonReport.build([*a, *b], [(copy, b, "same seed")])
    with pytest.raises(ComparisonError, match="does not fit a float"):
        obs(10**400)
    from nnx.comparison import PairedComparison

    twice = summarize([obs(0.1, attempt="x1", replicate="seed=1"), obs(0.2, attempt="x2", replicate="seed=1")]).pooled()
    with pytest.raises(ComparisonError, match="repeats replicate keys"):
        PairedComparison(twice, summarize(b).pooled(), "same seed")
    with pytest.raises(ComparisonError, match="report's own observations"):
        ComparisonReport(summarize(a), (compare(a, b, pairing="same seed"),))
    finite_a = replicates([0.3, 0.4, 0.2], config="cfg-a")
    finite_b = replicates([0.5, 0.6, 0.7], config="cfg-b")
    report = ComparisonReport.build(
        [*finite_a, *finite_b], [(finite_a, finite_b, "same seed")], bootstrap=Bootstrap(seed=1, level=0.975)
    )
    assert report.comparisons[0].interval is not None and "97.5% seed-variability interval" in report.text()
    with pytest.raises(ComparisonError, match="not one string"):
        observations_from_runs("abc", metric=LOSS)


def test_reader_edges(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    runs = [_model().train(params=_fit_params(seed), provenance=MANIFEST) for seed in (0, 1)]
    with pytest.raises(ComparisonError, match="label every run or none"):
        observations_from_runs([run.id for run in runs], metric=LOSS, config={runs[0].id: "a"})
    (plain,) = observations_from_runs([runs[0].id], metric=LOSS, selection="best")
    assert plain.evaluation == "unknown: the run declares no monitor"
    attempt = os.path.join("runs", runs[1].id, "attempt.json")
    state = json.loads(open(attempt).read())
    del state["started_at"]
    with open(attempt, "w") as handle:
        json.dump(state, handle)
    with pytest.raises(ComparisonError, match="provenance files"):
        observations_from_runs([runs[1].id], metric=LOSS)


def test_round_three_edges():
    from nnx.comparison import GroupSummary, PairedComparison, Summary

    a = replicates([0.3, 0.4], config="cfg-a")
    same = summarize(a).pooled()
    with pytest.raises(ComparisonError, match="on both sides"):
        PairedComparison(same, same, "same seed")
    mixed = (a[0], obs(0.9, config="cfg-b", replicate="seed=5"))
    with pytest.raises(IncompatibleObservations, match="do not"):
        GroupSummary(ACC, "validation", "last", "cfg-a", "data:v1", "split:v1", mixed)
    group = summarize(a).groups[0]
    with pytest.raises(ComparisonError, match="distinct poolable groups"):
        Summary((group, group))
    # A delta beyond float range is counted and reported, not silently dropped.
    left = [obs(-1e308, replicate="seed=0"), obs(0.0, replicate="seed=1")]
    right = [obs(1e308, config="cfg-b", replicate="seed=0"), obs(1.0, config="cfg-b", replicate="seed=1")]
    report = ComparisonReport.build([*left, *right], [(left, right, "same seed")])
    result = report.comparisons[0]
    assert (result.n, result.n_nonfinite, result.mean) == (1, 1, 1.0)
    assert "1 matched pair(s) with a delta beyond float range excluded" in report.text()
    assert ComparisonReport.from_state(json.loads(report.to_json())) == report
    # Interval levels print exactly.
    for level, text in ((0.9999999, "99.99999%"), (0.12345678, "12.345678%"), (0.95, "95%")):
        finite_a = replicates([0.3, 0.4, 0.2], config="cfg-a")
        finite_b = replicates([0.5, 0.6, 0.7], config="cfg-b")
        built = ComparisonReport.build(
            [*finite_a, *finite_b], [(finite_a, finite_b, "same seed")], bootstrap=Bootstrap(seed=1, level=level)
        )
        assert f"{text} seed-variability interval" in built.text()


def test_round_three_reader_edges(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    runs = [_model().train(params=_fit_params(seed), provenance=MANIFEST) for seed in (0, 1, 2, 3)]
    ids = [run.id for run in runs]
    # An empty label is refused, never replaced by the digest.
    with pytest.raises(ComparisonError, match="config"):
        observations_from_runs(ids[:2], metric=LOSS, config={ids[0]: "narrow", ids[1]: ""})
    # A metric the record does not hold says so.
    (typo,) = observations_from_runs(ids[:1], metric=Metric("acc", "maximize"))
    assert typo.value is None and typo.evaluation.startswith("unknown: the epoch 1 validation record has no 'acc'")
    # Damaged run files are comparison errors.
    with open(os.path.join("runs", ids[0], "provenance.json"), "w") as handle:
        handle.write("[]")
    with pytest.raises(ComparisonError, match="provenance files"):
        observations_from_runs(ids[:1], metric=LOSS)
    open(os.path.join("runs", ids[1], "idps.csv"), "w").close()
    with pytest.raises(ComparisonError, match="malformed idps.csv"):
        observations_from_runs(ids[1:2], metric=LOSS)
    frame = pd.read_csv(os.path.join("runs", ids[2], "idps.csv"))
    frame.drop(columns=["epoch_idx"]).to_csv(os.path.join("runs", ids[2], "idps.csv"), index=False)
    with pytest.raises(ComparisonError, match="malformed idps.csv"):
        observations_from_runs(ids[2:3], metric=LOSS)
    os.remove(os.path.join("runs", ids[3], "run.yaml"))
    with pytest.raises(ComparisonError, match="run.yaml is missing"):
        observations_from_runs(ids[3:4], metric=LOSS)
    # A run without a training seed has no replicate key.
    from nnx.comparison import _replicate_key

    assert _replicate_key({"model": {"net": {"kind": "registered", "seed": 0}}, "train": {}}) is None


def test_round_four_edges(tmp_path):
    from nnx.comparison import GroupSummary, PairedComparison, Summary

    a = replicates([0.3, 0.4, 0.2], config="cfg-a")
    b = replicates([0.5, 0.6, 0.7], config="cfg-b")
    forward = ComparisonReport.build([*a, *b], [(a, b, "same seed")])
    # Direct construction in any order gives the canonical report, which reloads.
    shuffled_a = GroupSummary(ACC, "validation", "last", "cfg-a", "data:v1", "split:v1", tuple(reversed(a)))
    shuffled_b = GroupSummary(ACC, "validation", "last", "cfg-b", "data:v1", "split:v1", tuple(reversed(b)))
    direct = ComparisonReport(
        Summary((shuffled_b, shuffled_a)), (PairedComparison(shuffled_a, shuffled_b, "same seed"),)
    )
    assert direct.to_json() == forward.to_json()
    path = tmp_path / "report.json"
    direct.save(path)
    assert ComparisonReport.load(path) == forward
    with pytest.raises(ComparisonError, match="at least one"):
        Summary(())
    with pytest.raises(ComparisonError, match="at least one observation"):
        GroupSummary(ACC, "validation", "last", None, None, None, ())
    with pytest.raises(ComparisonError, match="must be a Metric"):
        GroupSummary("acc", "validation", "last", None, None, None, tuple(a))  # type: ignore[arg-type]
    with pytest.raises(ComparisonError, match="GroupSummary"):
        PairedComparison("x", shuffled_b, "same seed")  # type: ignore[arg-type]
    with pytest.raises(ComparisonError, match="Bootstrap"):
        PairedComparison(shuffled_a, shuffled_b, "same seed", bootstrap="x")  # type: ignore[arg-type]
    with pytest.raises(ComparisonError, match="Summary"):
        ComparisonReport("x")  # type: ignore[arg-type]
    with pytest.raises(ComparisonError, match="resamples"):
        Bootstrap(seed=0, resamples=10**13)
    # The bootstrap interval stays finite where the mean does.
    huge_a = replicates([0.0, 0.0], config="cfg-a")
    huge_b = replicates([1e308, 1.5e308], config="cfg-b")
    result = compare(huge_a, huge_b, pairing="same seed", bootstrap=Bootstrap(seed=0, resamples=50))
    assert result.mean == pytest.approx(1.25e308) and all(math.isfinite(v) for v in result.interval)


def test_round_four_reader_edges(tmp_path, monkeypatch):
    import yaml

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    runs = [_model().train(params=_fit_params(seed), provenance=MANIFEST) for seed in (0, 1, 2)]
    ids = [run.id for run in runs]
    with pytest.raises(ComparisonError, match="not a run id"):
        observations_from_runs(["../x"], metric=LOSS)
    path = os.path.join("runs", ids[0], "run.yaml")
    state = yaml.safe_load(open(path))
    with open(path, "w") as handle:
        yaml.safe_dump({**state, "train": [1]}, handle)
    with pytest.raises(ComparisonError, match="'train' is not a mapping"):
        observations_from_runs(ids[:1], metric=LOSS)
    for bad, match in (([1], "not a mapping"), ({"epoch": "x"}, "not an epoch")):
        attempt = os.path.join("runs", ids[1], "attempt.json")
        record = json.loads(open(attempt).read())
        record["last_committed"] = bad
        with open(attempt, "w") as handle:
            json.dump(record, handle)
        with pytest.raises(ComparisonError, match=match):
            observations_from_runs(ids[1:2], metric=LOSS)
    (item,) = observations_from_runs(ids[2:], metric=Metric("acc", "maximize"))
    assert "missing, or NaN" in item.evaluation


def test_round_five_edges(monkeypatch):
    import nnx.comparison as module
    from nnx.comparison import GroupSummary, PairedComparison, Summary

    a = replicates([0.3, 0.4, 0.2, 0.5], config="cfg-a")
    b = replicates([0.5, 0.6, 0.7, 0.4], config="cfg-b")
    whole = compare(a, b, pairing="same seed", bootstrap=Bootstrap(seed=3, resamples=500)).interval
    sizes = []
    real = np.random.default_rng

    class Recording:
        def __init__(self, seed):
            self.rng = real(seed)

        def integers(self, *args, size=None, **kwargs):
            sizes.append(size)
            return self.rng.integers(*args, size=size, **kwargs)

    monkeypatch.setattr(module, "BOOTSTRAP_BLOCK", 9)  # two resamples of four pairs per block
    monkeypatch.setattr(np.random, "default_rng", Recording)
    blocked = compare(a, b, pairing="same seed", bootstrap=Bootstrap(seed=3, resamples=500)).interval
    assert blocked == whole and max(rows for rows, _ in sizes) == 2  # bounded memory, same interval
    monkeypatch.undo()
    # One-shot iterables are kept, not consumed by validation.
    group = GroupSummary(ACC, "validation", "last", "cfg-a", "data:v1", "split:v1", (o for o in a))
    assert group.n_attempts == 4
    summary = Summary(iter([group]))
    assert len(summary.groups) == 1
    other = summarize(b).pooled()
    pc = PairedComparison(group, other, "same seed")
    report = ComparisonReport(summarize([*a, *b]), (x for x in [pc]))
    assert len(report.comparisons) == 1 and hash(ComparisonReport(summarize([*a, *b]), [pc]))
    state = json.loads(report.to_json())
    del state["comparisons"][0]["b"]
    with pytest.raises(ComparisonError, match="malformed|'b'") as caught:
        ComparisonReport.from_state(state)
    assert "does not hold" not in str(caught.value)
