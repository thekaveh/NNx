"""FEAT-036: bounded training history with an append journal.

``history=HistoryJournal(retention, chunk_size)`` keeps the last
``retention`` records in memory and appends every record once to
``runs/<id>/history/``; the default (no journal) is the eager list and
``idps.csv`` it always was.
"""

from __future__ import annotations

import filecmp
import gc
import hashlib
import json
import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import nnx.history as history_module
import nnx.nn.params.nn_run as nn_run_module
from nnx import Callback, EarlyStopping, LRMonitor
from nnx.history import (
    HistoryCorruptionError,
    HistoryJournal,
    export_history_csv,
    iter_history,
    migrate_history,
)
from nnx.nn.enum.activations import Activations
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.enum.optims import Optims
from nnx.nn.nn_model import NNModel
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from nnx.nn.params.nn_iteration_data_point import NNIterationDataPoint
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_run import NNRun, _elect_best
from nnx.nn.params.nn_train_params import NNTrainParams
from nnx.trainer import NNTrainerParams, Trainer

_OPTIM = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)


def _model() -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _loader(n_batches: int = 5, batch_size: int = 4) -> DataLoader:
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(n_batches * batch_size, 4, generator=generator)
    labels = torch.randint(0, 2, (n_batches * batch_size,), generator=generator)
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)


def _params(n_epochs: int = 3, *, validate: bool = True, **overrides) -> NNTrainParams:
    loader = _loader()
    return NNTrainParams(
        n_epochs=n_epochs,
        seed=0,
        train_loader=loader,
        val_loader=loader if validate else None,
        extra_metrics={"half": lambda y_true, y_pred: 0.5},
        optim=_OPTIM,
        **overrides,
    )


def _trainer_step(ctx) -> NNEvaluationDataPoint:
    model = ctx.model
    optimizer = ctx.optimizers["main"]
    model.net.train()
    optimizer.zero_grad()
    features, labels = ctx.batch
    logits = model.net(features)
    loss = model.loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    error = float((logits.argmax(dim=1) != labels).float().mean())
    return NNEvaluationDataPoint(f1=0.0, recall=0.0, accuracy=1 - error, precision=0.0, loss=float(loss), error=error)


def _fit(entry: str = "model", history=None, *, n_epochs: int = 3, validate: bool = True, callbacks=None, **train):
    if entry == "model":
        return _model().train(_params(n_epochs, validate=validate, **train), callbacks=callbacks, history=history)
    loader = _loader()
    params = NNTrainerParams(
        n_epochs=n_epochs,
        seed=0,
        train_loader=loader,
        val_loader=loader if validate else None,
        optims={"main": _OPTIM},
        extra_metrics={"half": lambda y_true, y_pred: 0.5},
        **train,
    )
    return Trainer(_model()).train(params, trainer_step_fn=_trainer_step, callbacks=callbacks, history=history)


def _states(records) -> list[dict]:
    return [record.state() for record in records]


def _journal(run: NNRun) -> str:
    return os.path.join("runs", run.id, "history")


# --- configuration ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"retention": 0}, "retention must be a positive integer"),
        ({"chunk_size": 0}, "chunk_size must be a positive integer"),
        ({"retention": True}, "retention must be a positive integer"),
        ({"retention": 3, "chunk_size": 4}, "must not exceed retention"),
    ],
)
def test_journal_spec_is_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HistoryJournal(**kwargs)


@pytest.mark.parametrize("entry", ["model", "trainer"])
def test_the_default_run_is_unchanged(entry):
    run = _fit(entry)
    assert run.history is None
    assert len(run.idps) == 15
    assert os.path.isfile(os.path.join("runs", run.id, "idps.csv"))
    assert not os.path.exists(_journal(run))


@pytest.mark.parametrize("entry", ["model", "trainer"])
def test_the_run_id_ignores_the_journal(entry, tmp_path):
    legacy = _fit(entry)
    os.makedirs(tmp_path / "other")
    os.chdir(tmp_path / "other")
    journal = _fit(entry, HistoryJournal(retention=3, chunk_size=2))
    assert journal.id == legacy.id


# --- AC1: bounded memory, linear bytes -------------------------------------------------------------


class _Counter(Callback):
    """Counts live NNIterationDataPoint objects at the end of sampled epochs."""

    def __init__(self, every: int) -> None:
        self.every = every
        self.live: list[int] = []
        self.window: list[int] = []

    def on_epoch_end(self, ctx) -> None:
        self.window.append(len(ctx.idps))
        if ctx.epoch % self.every == 0:
            gc.collect()
            self.live.append(sum(isinstance(item, NNIterationDataPoint) for item in gc.get_objects()))


def _fast_step(ctx) -> NNEvaluationDataPoint:
    return NNEvaluationDataPoint(loss=0.5, error=0.25, accuracy=0.75, f1=0.0, precision=0.0, recall=0.0)


def _history_writer_bytes(monkeypatch) -> list[int]:
    """Bytes written for the run's history (run.yaml, metadata, idps.csv, the
    journal) — checkpoints excluded."""
    written: list[int] = []
    atomic, chunk, append = (
        nn_run_module._atomic_write_text,
        history_module._write_chunk_file,
        history_module._append_bytes,
    )

    def counted_atomic(path, content):
        if os.sep + "checkpoints" + os.sep not in path:
            written.append(len(content.encode("utf-8")))
        return atomic(path, content)

    def counted_chunk(path, data):
        written.append(len(data))
        return chunk(path, data)

    def counted_append(path, data):
        written.append(len(data))
        return append(path, data)

    monkeypatch.setattr(nn_run_module, "_atomic_write_text", counted_atomic)
    monkeypatch.setattr(history_module, "_write_chunk_file", counted_chunk)
    monkeypatch.setattr(history_module, "_append_bytes", counted_append)
    return written


def _long_fit(entry: str, n_epochs: int, *, history, callbacks, **train):
    batches = [(torch.zeros(1, 4), torch.zeros(1, dtype=torch.long))] * 100
    if entry == "model":
        params = NNTrainParams(n_epochs=n_epochs, train_loader=batches, optim=_OPTIM, **train)
        return _model().train(params, train_step_fn=_fast_step, callbacks=callbacks, history=history)
    params = NNTrainerParams(
        n_epochs=n_epochs, train_loader=batches, optims={"main": _OPTIM}, save_phase_checkpoints=False, **train
    )
    step = lambda ctx: _fast_step(ctx)  # noqa: E731 — the trainer needs its own step object
    return Trainer(_model()).train(params, trainer_step_fn=step, callbacks=callbacks, history=history)


@pytest.mark.parametrize("entry", ["model", "trainer"])
def test_retained_records_and_bytes_stay_bounded_at_1000_and_10000_records(entry, monkeypatch):
    spec = HistoryJournal(retention=64, chunk_size=32)
    measured = {}
    for n_epochs in (10, 100):  # 1,000 and 10,000 records
        counter = _Counter(every=max(1, n_epochs // 5))
        lr = LRMonitor()
        with pytest.MonkeyPatch.context() as patch:
            written = _history_writer_bytes(patch)
            run = _long_fit(entry, n_epochs, history=spec, callbacks=[counter, lr], data_id=f"n{n_epochs}")
        records = sum(1 for _ in iter_history(run.id))
        assert records == n_epochs * 100
        assert max(counter.window) <= spec.retention  # ctx.idps is the window
        assert len(run.idps) == spec.retention
        assert len(lr.history) <= spec.retention  # the built-in log is bounded too
        measured[n_epochs] = (max(counter.live), sum(written))
        del run
    small, large = measured[10], measured[100]
    # Live records: the window plus a constant (the BEST checkpoint's record
    # and the epoch's replaced last record), the same at 10x the records.
    assert small[0] <= spec.retention + 4 and large[0] <= spec.retention + 4
    # Bytes: linear in the records — 10x the records, ~10x the bytes (a little
    # more: the indices written grow a digit); rewriting the eager CSV each
    # epoch, as a run without a journal does, costs ~90x here.
    assert large[1] <= 11 * small[1]


@pytest.mark.parametrize("entry", ["model", "trainer"])
def test_a_lazy_resume_reads_no_parent_history_and_stays_bounded(entry, monkeypatch):
    spec = HistoryJournal(retention=64, chunk_size=32)
    parent_id = _long_fit(entry, 20, history=spec, callbacks=None, data_id="parent").id  # its window is not kept
    reads: list[str] = []
    original = history_module._JournalReader.read_chunk

    def spy(self, chunk):
        reads.append(os.path.join(self.directory, chunk.file))
        return original(self, chunk)

    monkeypatch.setattr(history_module._JournalReader, "read_chunk", spy)
    counter = _Counter(every=5)
    child = _long_fit(entry, 30, history=spec, callbacks=[counter], data_id="parent", resume_from_run_id=parent_id)
    assert not reads  # neither the parent's history nor the child's is read back
    # The window plus a constant (the BEST and resume-source checkpoints' records).
    assert max(counter.window) <= spec.retention and max(counter.live) <= spec.retention + 4
    assert child.idps[0].epoch_idx >= 20 and len(child.idps) == spec.retention


# --- AC2: crash recovery ---------------------------------------------------------------------------


class _Boom(RuntimeError):
    pass


def _crash_at_chunk_write(monkeypatch, epoch: int):
    original = history_module._write_chunk_file

    def fail(path, data):
        if b'"epoch_idx":%d' % epoch in data:
            raise _Boom("chunk write")
        return original(path, data)

    monkeypatch.setattr(history_module, "_write_chunk_file", fail)


def _crash_at_manifest(monkeypatch, epoch: int):
    original = nn_run_module._atomic_write_text

    def fail(path, content):
        if path.endswith("journal.json") and json.loads(content)["epochs"] == epoch + 1:
            raise _Boom("manifest")
        return original(path, content)

    monkeypatch.setattr(nn_run_module, "_atomic_write_text", fail)


def _crash_before_last(monkeypatch, epoch: int, *, rollback: bool):
    original = NNCheckpoint.save

    def fail(self, *args, **kwargs):
        if self.idp.epoch_idx == epoch:
            raise _Boom("LAST")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NNCheckpoint, "save", fail)
    if not rollback:  # a hard kill: the previous manifest is never republished
        monkeypatch.setattr(history_module._JournalWriter, "rollback", lambda self: None)


def _crash_after_last(monkeypatch, epoch: int):
    original = NNModel._update_tqdm_postfix
    calls = []

    def fail(self, *args, **kwargs):
        calls.append(None)
        if len(calls) == epoch + 1:  # runs once per epoch, after its LAST was replaced
            raise _Boom("after LAST")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NNModel, "_update_tqdm_postfix", fail)


CRASHES = {
    "chunk write": (lambda mp: _crash_at_chunk_write(mp, 2), 1),
    "manifest publication": (lambda mp: _crash_at_manifest(mp, 2), 1),
    "before LAST (rolled back)": (lambda mp: _crash_before_last(mp, 2, rollback=True), 1),
    "before LAST (hard kill)": (lambda mp: _crash_before_last(mp, 2, rollback=False), 1),
    "after LAST": (lambda mp: _crash_after_last(mp, 2), 2),
}


@pytest.mark.parametrize("case", sorted(CRASHES))
def test_a_crash_recovers_the_committed_epoch_exactly(case, monkeypatch, tmp_path):
    reference = _fit("model", n_epochs=4, data_id="reference")  # uninterrupted, eager
    inject, committed = CRASHES[case]
    spec = HistoryJournal(retention=3, chunk_size=2)
    with monkeypatch.context() as patch:
        inject(patch)
        with pytest.raises(_Boom):
            _fit("model", spec, n_epochs=4, data_id="crashed")
    crashed_id = NNRun(train=_params(4, data_id="crashed"), model=_model().params, net=_model().net_params).id
    expected = [record for record in reference.idps if record.epoch_idx <= committed]

    # Readers see exactly the committed epochs — the uncommitted tail on disk is ignored.
    assert _states(iter_history(crashed_id)) == _states(expected)
    assert _states(NNRun.load(crashed_id).idps) == _states(expected[-3:])
    assert export_history_csv(crashed_id, tmp_path / "crashed.csv") == len(expected)
    # Resuming continues the committed epochs without duplicates or gaps.
    child = _fit("model", spec, n_epochs=4 - (committed + 1), data_id="crashed", resume_from_run_id=crashed_id)
    lineage = list(iter_history(child.id, lineage=True))
    assert [(r.epoch_idx, r.batch_idx) for r in lineage] == [(r.epoch_idx, r.batch_idx) for r in reference.idps]
    assert [r.train_edp.loss for r in lineage] == [r.train_edp.loss for r in reference.idps]


def test_a_crash_in_the_first_epoch_releases_the_reservation(monkeypatch):
    spec = HistoryJournal(retention=3, chunk_size=2)
    with monkeypatch.context() as patch:
        _crash_at_chunk_write(patch, 0)
        with pytest.raises(_Boom):
            _fit("model", spec)
    assert not [entry for entry in os.listdir("runs") if not entry.startswith(".")]
    run = _fit("model", spec)  # the same run can be retried
    assert len(list(iter_history(run.id))) == 15


@pytest.mark.parametrize("entry", ["model", "trainer"])
def test_a_first_epoch_that_cannot_commit_leaves_an_empty_history_like_the_eager_run(entry, monkeypatch, tmp_path):
    with monkeypatch.context() as patch:
        _crash_before_last(patch, 0, rollback=True)
        with pytest.raises(_Boom):
            _fit(entry, n_epochs=2)
    os.makedirs(tmp_path / "journal")
    os.chdir(tmp_path / "journal")
    with monkeypatch.context() as patch:
        _crash_before_last(patch, 0, rollback=True)
        with pytest.raises(_Boom):
            _fit(entry, HistoryJournal(retention=3, chunk_size=2), n_epochs=2)
    (eager,) = NNRun.all(root=str(tmp_path))
    (journal,) = NNRun.all()
    assert eager.id == journal.id and eager.idps == journal.idps == []
    assert list(iter_history(journal.id)) == [] and journal._repr_html_()


def test_a_corrupt_committed_chunk_is_flagged():
    run = _fit("model", HistoryJournal(retention=3, chunk_size=2))
    first = os.path.join(_journal(run), "chunk-00000000.jsonl")
    last = sorted(name for name in os.listdir(_journal(run)) if name.startswith("chunk-"))[-1]
    with open(first, "r+b") as handle:
        data = handle.read()
        handle.seek(0)
        handle.write(data.replace(b'"batch_idx":0', b'"batch_idx":7', 1))
    with pytest.raises(HistoryCorruptionError, match="chunk 0 .* is corrupt"):
        list(iter_history(run.id))
    NNRun.load(run.id)  # the tail does not hold chunk 0
    os.remove(os.path.join(_journal(run), last))
    with pytest.raises(HistoryCorruptionError, match="is missing"):
        NNRun.load(run.id)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (b'"records":2', b'"records":3', "does not match its manifest"),
        (b'{"file"', b'{"file" ', "does not match its manifest"),
        (b'"index_bytes"', b'"index_byte"', "manifest .* is malformed"),
    ],
    ids=["count", "whitespace", "manifest key"],
)
def test_an_altered_index_or_manifest_is_flagged(old, new, message):
    run = _fit("model", HistoryJournal(retention=3, chunk_size=2))
    name = "journal.json" if b"index_byte" in old else "index.jsonl"
    path = os.path.join(_journal(run), name)
    with open(path, "rb") as handle:
        data = handle.read()
    with open(path, "wb") as handle:
        handle.write(data.replace(old, new, 1))
    with pytest.raises(HistoryCorruptionError, match=message):
        NNRun.load(run.id)
    with pytest.raises(HistoryCorruptionError, match=message):
        list(iter_history(run.id))


# --- AC3: parity, legacy and mixed directories, migration --------------------------------------------


@pytest.mark.parametrize("entry", ["model", "trainer"])
@pytest.mark.parametrize("validate", [True, False], ids=["validation", "no-validation"])
def test_the_journal_reproduces_the_legacy_history(entry, validate, tmp_path):
    os.makedirs("legacy")
    os.chdir("legacy")
    legacy = _fit(entry, validate=validate)
    os.chdir(tmp_path)
    journal = _fit(entry, HistoryJournal(retention=3, chunk_size=2), validate=validate)
    legacy_csv = os.path.join("legacy", "runs", legacy.id, "idps.csv")

    assert not os.path.exists(os.path.join("runs", journal.id, "idps.csv"))
    assert export_history_csv(journal.id, tmp_path / "export.csv") == 15
    assert filecmp.cmp(tmp_path / "export.csv", legacy_csv, shallow=False)  # order, indices, extras, None val
    assert _states(iter_history(journal.id)) == _states(legacy.idps)
    assert _states(journal.idps) == _states(legacy.idps[-3:])
    loaded = NNRun.load(journal.id)
    assert _states(loaded.idps) == _states(legacy.idps[-3:]) and loaded.history is not None
    assert json.dumps(loaded._epoch_series()) == json.dumps(legacy._epoch_series())
    if validate:
        assert loaded.idps[-1].val_edp is not None and loaded.idps[-1].val_edp.extra == {"half": 0.5}
    else:
        assert all(record.val_edp is None for record in iter_history(journal.id))


def test_mixed_directories_and_explicit_migration(tmp_path):
    legacy = _fit("model", data_id="legacy")
    journal = _fit("model", HistoryJournal(retention=3, chunk_size=2), data_id="journal")
    by_id = {run.id: run for run in NNRun.all()}
    assert set(by_id) == {legacy.id, journal.id}
    assert len(by_id[legacy.id].idps) == 15 and by_id[legacy.id].history is None
    assert len(by_id[journal.id].idps) == 3 and by_id[journal.id].history is not None

    run_yaml = os.path.join("runs", legacy.id, "run.yaml")
    before_yaml = open(run_yaml, "rb").read()
    csv_before = tmp_path / "before.csv"
    csv_before.write_bytes(open(os.path.join("runs", legacy.id, "idps.csv"), "rb").read())
    export_history_csv(legacy.id, tmp_path / "legacy.csv")  # a CSV run exports its own file exactly
    assert filecmp.cmp(tmp_path / "legacy.csv", csv_before, shallow=False)

    from filelock import FileLock

    with FileLock(os.path.join("runs", ".leases", f"{legacy.id}.lock")):  # as a training run holds it
        with pytest.raises(RuntimeError, match="is being trained"):
            migrate_history(legacy.id)
    assert os.path.isfile(os.path.join("runs", legacy.id, "idps.csv"))
    migrate_history(legacy.id, spec=HistoryJournal(retention=4, chunk_size=2))

    migrated = NNRun.load(legacy.id)
    assert migrated.id == legacy.id and open(run_yaml, "rb").read() == before_yaml
    assert not os.path.exists(os.path.join("runs", legacy.id, "idps.csv"))
    assert len(migrated.idps) == 4 and migrated.history is not None
    export_history_csv(legacy.id, tmp_path / "after.csv")
    assert filecmp.cmp(tmp_path / "after.csv", csv_before, shallow=False)
    assert json.dumps(migrated._epoch_series()) == json.dumps(legacy._epoch_series())
    with pytest.raises(ValueError, match="already keeps its history in a journal"):
        migrate_history(legacy.id)


# --- AC4: readers never scan the whole history; callbacks ------------------------------------------------


def _chunk_reads(monkeypatch) -> list[str]:
    reads: list[str] = []
    original = history_module._JournalReader.read_chunk

    def spy(self, chunk):
        reads.append(chunk.file)
        return original(self, chunk)

    monkeypatch.setattr(history_module._JournalReader, "read_chunk", spy)
    return reads


def test_summaries_best_and_builtin_callbacks_never_scan_the_history(monkeypatch):
    reads = _chunk_reads(monkeypatch)
    stopper = EarlyStopping(patience=10)
    lr = LRMonitor()
    run = _fit("model", HistoryJournal(retention=3, chunk_size=2), n_epochs=5, callbacks=[stopper, lr])
    assert not reads  # training (built-in callbacks, BEST, runs/best election) reads nothing back
    assert len(lr.history) == 3

    assert "<table" in run._repr_html_() and not reads  # the chart reads per-epoch rows
    assert _elect_best(os.path.join(os.getcwd(), "runs"), None) == run.id and not reads
    assert NNCheckpoint.load(run=run.id, type=Checkpoints.BEST) is not None and not reads
    loaded = NNRun.load(run.id)
    assert len(loaded.idps) == 3
    assert reads == ["chunk-00000013.jsonl", "chunk-00000014.jsonl"]  # the tail's chunks only
    reads.clear()
    loaded._repr_html_()
    loaded.save()
    assert not reads


def test_readers_never_see_an_uncommitted_tail(monkeypatch):
    spec = HistoryJournal(retention=10, chunk_size=2)
    with monkeypatch.context() as patch:
        _crash_before_last(patch, 2, rollback=False)
        with pytest.raises(_Boom):
            _fit("model", spec, n_epochs=4)
    run_id = NNRun(train=_params(4), model=_model().params, net=_model().net_params).id
    manifest = json.load(open(os.path.join("runs", run_id, "history", "journal.json")))
    assert manifest["epochs"] == 3  # epoch 2 was published, then LAST never landed
    loaded = NNRun.load(run_id)
    assert {record.epoch_idx for record in loaded.idps} == {0, 1}
    assert loaded._epoch_series()["epochs"] == [0, 1]
    assert {record.epoch_idx for record in iter_history(run_id)} == {0, 1}


class _Full(Callback):
    history_access = "full"

    def __init__(self) -> None:
        self.lengths: list[int] = []
        self.first: list[tuple[int, int]] = []

    def on_epoch_end(self, ctx) -> None:
        self.lengths.append(len(ctx.idps))
        self.first.append((ctx.idps[0].epoch_idx, ctx.idps[0].batch_idx))

    def on_train_end(self, ctx) -> None:
        self.lengths.append(len(ctx.idps))


class _Window(Callback):
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def on_epoch_end(self, ctx) -> None:
        self.lengths.append(len(ctx.idps))

    def on_train_end(self, ctx) -> None:
        self.lengths.append(len(ctx.idps))


@pytest.mark.parametrize("entry", ["model", "trainer"])
def test_a_callback_needing_the_full_history_requests_it(entry):
    full, window = _Full(), _Window()
    run = _fit(entry, HistoryJournal(retention=3, chunk_size=2), callbacks=[full, window])
    assert full.lengths == [5, 10, 15, 15] and full.first == [(0, 0)] * 3  # on_train_end too
    assert window.lengths == [3, 3, 3, 3]
    assert len(run.idps) == 3


@pytest.mark.parametrize("entry", ["model", "trainer"])
def test_a_function_callback_is_refused_before_training(entry):
    with pytest.raises(ValueError, match="receives the full idps list"):
        _fit(entry, HistoryJournal(retention=3, chunk_size=2), callbacks=[lambda idps: None])
    assert not os.path.exists("runs") or not [e for e in os.listdir("runs") if not e.startswith(".")]


def test_an_unknown_history_access_and_a_wrong_spec_are_refused():
    class _Odd(Callback):
        history_access = "some"

    with pytest.raises(ValueError, match="history_access must be one of"):
        _fit("model", HistoryJournal(retention=3, chunk_size=2), callbacks=[_Odd()])
    with pytest.raises(TypeError, match="HistoryJournal"):
        _fit("model", {"retention": 3})


# --- AC5: continuation and lineage -----------------------------------------------------------------------


def _hashes(path: str) -> dict[str, str]:
    result = {}
    for directory, _, files in os.walk(path):
        for name in files:
            if name.endswith(".lock"):
                continue
            full = os.path.join(directory, name)
            result[os.path.relpath(full, path)] = hashlib.sha256(open(full, "rb").read()).hexdigest()
    return result


@pytest.mark.parametrize("checkpoint", ["last", "first"])
def test_a_continuation_owns_its_files_and_exports_its_lineage_once(checkpoint):
    spec = HistoryJournal(retention=3, chunk_size=2)
    parent = _fit("model", spec, n_epochs=3)
    parent_path = os.path.join("runs", parent.id)
    before = _hashes(parent_path)

    child = _fit("model", spec, n_epochs=2, resume_from_run_id=parent.id, resume_from_checkpoint=checkpoint)
    assert _hashes(parent_path) == before  # the source run is byte-for-byte unchanged
    assert child.id != parent.id and child.history == os.path.abspath(_journal(child))
    assert os.listdir(_journal(child))

    start = 3 if checkpoint == "last" else 1
    own = list(iter_history(child.id))
    assert {r.epoch_idx for r in own} == set(range(start, start + 2))
    lineage = list(iter_history(child.id, lineage=True))
    epochs = [r.epoch_idx for r in lineage]
    assert epochs == sorted(epochs) and set(epochs) == set(range(start + 2))
    assert [(r.epoch_idx, r.batch_idx) for r in lineage] == sorted({(r.epoch_idx, r.batch_idx) for r in lineage})
    assert export_history_csv(child.id, "lineage.csv", lineage=True) == len(lineage) == (start + 2) * 5
