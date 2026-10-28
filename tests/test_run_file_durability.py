"""#552: a run's own files are durable before LAST, the epoch's commit marker.

LAST is published only once the epoch's history is on stable storage: the
renames of ``run.yaml``, ``metadata.yaml``, ``idps.csv`` (or the journal's
``history/``), the history marker and ``checkpoints/`` are flushed into
``runs/<id>/``, and ``<id>`` into ``runs/``. A real power loss cannot be
scripted, so the order of the renames and the flushes — traced at the
flush primitive — stands in for it. A real flush I/O error on
``idps.csv`` or the journal's manifest fails the epoch before LAST; one
after their rename also puts back the file the rename replaced.
"""

from __future__ import annotations

import errno
import json
import os

import pytest
import torch

import nnx.nn.params.nn_checkpoint as nn_checkpoint
from nnx import (
    Activations,
    Callback,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNParams,
    NNRun,
    NNTrainParams,
    Optims,
)
from nnx.history import HistoryJournal
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.trainer import NNTrainerParams, Trainer

_OPTIM = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0)
_BATCHES = [(torch.randn(4, 4, generator=torch.Generator().manual_seed(1)), torch.tensor([0, 1, 0, 1]))]
_LAST = os.path.join("checkpoints", "last.pt")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)


def _model() -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _trainer_step(ctx) -> NNEvaluationDataPoint:
    optimizer = ctx.optimizers["main"]
    ctx.model.net.train()
    optimizer.zero_grad()
    features, labels = ctx.batch
    loss = ctx.model.loss_fn(ctx.model.net(features), labels)
    loss.backward()
    optimizer.step()
    return NNEvaluationDataPoint(loss=float(loss.detach()))


def _fit(entry: str, history=None, *, n_epochs: int = 3, callbacks=None) -> NNRun:
    if entry == "model":
        params = NNTrainParams(n_epochs=n_epochs, seed=0, train_loader=_BATCHES, optim=_OPTIM)
        return _model().train(params, callbacks=callbacks, history=history)
    params = NNTrainerParams(n_epochs=n_epochs, seed=0, train_loader=_BATCHES, optims={"main": _OPTIM})
    return Trainer(_model()).train(params, trainer_step_fn=_trainer_step, callbacks=callbacks, history=history)


def _key(stat) -> tuple[int, int]:
    return stat.st_dev, stat.st_ino


def _the_run_id() -> str:
    (run_id,) = [entry for entry in os.listdir("runs") if entry != "best" and not entry.startswith(".")]
    return run_id


# --- the trace: renames, new directories and flushes, in order -----------------------------------


def _trace(patch) -> list[tuple[str, object]]:
    """Record every rename (its destination), new directory and flush (the
    flushed inode) — flushes at the primitive every run directory flush
    goes through, ``_full_fsync``."""
    events: list[tuple[str, object]] = []
    real_flush, real_replace, real_mkdir = nn_checkpoint._full_fsync, os.replace, os.mkdir

    def flush(fd):
        events.append(("flush", _key(os.fstat(fd))))
        return real_flush(fd)

    def replace(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        events.append(("rename", os.path.relpath(os.path.abspath(dst), os.path.abspath("runs"))))

    def mkdir(path, *args, **kwargs):
        real_mkdir(path, *args, **kwargs)
        events.append(("mkdir", os.path.relpath(os.path.abspath(path), os.path.abspath("runs"))))

    patch.setattr(nn_checkpoint, "_full_fsync", flush)
    patch.setattr(os, "replace", replace)
    patch.setattr(os, "mkdir", mkdir)
    return events


def _flushed_between(events, start: int, end: int, directory: str) -> bool:
    key = _key(os.stat(directory))
    return ("flush", key) in events[start + 1 : end]


def _last_published(events, run_id: str) -> list[int]:
    return [i for i, event in enumerate(events) if event == ("rename", os.path.join(run_id, _LAST))]


@pytest.mark.parametrize("entry", ["model", "trainer"])
def test_each_eager_epoch_flushes_the_run_directory_between_idps_csv_and_last(entry, monkeypatch):
    with monkeypatch.context() as patch:
        events = _trace(patch)
        run = _fit(entry, n_epochs=3)
    run_dir = os.path.join("runs", run.id)
    csv = ("rename", os.path.join(run.id, "idps.csv"))
    published = _last_published(events, run.id)
    assert len(published) >= 3
    for epoch, at in enumerate(published[:3]):
        renamed = max(i for i in range(at) if events[i] == csv)
        assert _flushed_between(events, renamed, at, run_dir), f"epoch {epoch}: idps.csv's rename is not flushed"


@pytest.mark.parametrize("entry", ["model", "trainer"])
@pytest.mark.parametrize("journal", [False, True], ids=["eager", "journal"])
def test_the_run_files_and_directory_are_flushed_before_the_first_last(entry, journal, monkeypatch):
    with monkeypatch.context() as patch:
        events = _trace(patch)
        run = _fit(entry, HistoryJournal(retention=3, chunk_size=2) if journal else None, n_epochs=1)
    run_dir = os.path.join("runs", run.id)
    first = _last_published(events, run.id)[0]

    def latest(event) -> int:
        return max(i for i in range(first) if events[i] == event)

    # <id> in runs/
    assert _flushed_between(events, latest(("mkdir", run.id)), first, "runs")
    # every entry of runs/<id>/ the first epoch renamed or created
    entries = ["run.yaml", "metadata.yaml", ".history-committed-by-last"]
    entries.append(os.path.join("history", "journal.json") if journal else "idps.csv")
    for name in entries:
        assert _flushed_between(events, latest(("rename", os.path.join(run.id, name))), first, run_dir), name
    assert _flushed_between(events, latest(("mkdir", os.path.join(run.id, "checkpoints"))), first, run_dir)
    if journal:
        history_made = latest(("mkdir", os.path.join(run.id, "history")))
        journal_renamed = latest(("rename", os.path.join(run.id, "history", "journal.json")))
        # history/ is flushed into runs/<id>/ by the history step itself, before
        # its manifest: NNCheckpoint.save skips its flush if checkpoints/ exists
        assert _flushed_between(events, history_made, journal_renamed, run_dir)
        assert _flushed_between(events, journal_renamed, first, os.path.join(run_dir, "history"))


# --- strict history files: an EIO on idps.csv or journal.json fails the epoch ----------------------


class _FailTheFlushOf(Callback):
    """From ``epoch``'s end on — before its history is saved — fail
    ``os.fsync`` with EIO on the temporary that ``_atomic_write_text``
    writes for ``name`` (``.<name>.XXXXXX``)."""

    def __init__(self, patch, name: str, epoch: int = 1) -> None:
        self.epoch, self.failing = epoch, False
        real_fsync = os.fsync

        def fsync(fd):
            if self.failing and self._is_temporary_of(fd, name):
                raise OSError(errno.EIO, "Input/output error")
            return real_fsync(fd)

        patch.setattr(os, "fsync", fsync)

    @staticmethod
    def _is_temporary_of(fd: int, name: str) -> bool:
        key = _key(os.fstat(fd))
        directory = os.path.join("runs", _the_run_id(), os.path.dirname(name))
        prefix = f".{os.path.basename(name)}."
        return any(entry.name.startswith(prefix) and _key(entry.stat()) == key for entry in os.scandir(directory))

    def on_epoch_end(self, ctx) -> None:
        self.failing = self.failing or ctx.epoch == self.epoch


@pytest.mark.parametrize("entry", ["model", "trainer"])
@pytest.mark.parametrize("journal", [False, True], ids=["idps.csv", "journal.json"])
def test_an_io_error_flushing_the_history_fails_the_epoch_before_last(entry, journal, monkeypatch):
    name = os.path.join("history", "journal.json") if journal else "idps.csv"
    with monkeypatch.context() as patch:
        failing = _FailTheFlushOf(patch, name)
        with pytest.raises(OSError) as caught:
            _fit(entry, HistoryJournal(retention=3, chunk_size=2) if journal else None, callbacks=[failing])
    assert caught.value.errno == errno.EIO
    run_id = _the_run_id()
    assert NNCheckpoint.load(run_id, Checkpoints.LAST).idp.epoch_idx == 0
    assert {record.epoch_idx for record in NNRun.load(run_id).idps} == {0}
    if journal:
        with open(os.path.join("runs", run_id, name), encoding="utf-8") as handle:
            assert json.load(handle)["epochs"] == 1
    assert not [entry for entry in os.listdir(os.path.join("runs", run_id)) if entry.startswith(".idps.csv.")]


# --- a failed flush after the manifest's rename republishes the previous one -----------------------


class _FailTheRenameFlush(Callback):
    """From ``epoch``'s end on — before its history is saved — fail, with
    EIO at ``_full_fsync``, every flush of the directory holding ``name``
    once ``name`` has been renamed: only the flush that makes the history
    step's own rename durable, never a flush before it. Each injected error
    is numbered, so the one the epoch raises can be told apart."""

    def __init__(self, patch, name: str, epoch: int) -> None:
        self.name, self.epoch, self.armed, self.renamed, self.raised = name, epoch, False, False, 0
        self.previous: bytes | None = None
        real_flush, real_replace = nn_checkpoint._full_fsync, os.replace

        def replace(src, dst, *args, **kwargs):
            real_replace(src, dst, *args, **kwargs)
            if self.armed and os.path.abspath(dst) == os.path.abspath(self._path(name)):
                self.renamed = True

        def flush(fd):
            if self.renamed and _key(os.fstat(fd)) == _key(os.stat(os.path.dirname(self._path(name)))):
                self.raised += 1
                raise OSError(errno.EIO, f"Input/output error (injected #{self.raised})")
            return real_flush(fd)

        patch.setattr(os, "replace", replace)
        patch.setattr(nn_checkpoint, "_full_fsync", flush)

    @staticmethod
    def _path(name: str) -> str:
        return os.path.join("runs", _the_run_id(), name)

    def on_epoch_end(self, ctx) -> None:
        if ctx.epoch == self.epoch:
            self.armed, self.run_id = True, _the_run_id()
            if os.path.exists(self._path(self.name)):
                with open(self._path(self.name), "rb") as handle:
                    self.previous = handle.read()  # the file the previous epoch committed


@pytest.mark.parametrize("epoch", [0, 1], ids=["first-epoch", "later-epoch"])
@pytest.mark.parametrize("entry", ["model", "trainer"])
@pytest.mark.parametrize("journal", [False, True], ids=["idps.csv", "journal.json"])
def test_a_failed_flush_of_the_history_rename_republishes_the_previous_file(entry, journal, epoch, monkeypatch):
    """The history step's rename is made durable before LAST. If that flush
    fails, the rename is undone — the previous epoch's file is put back
    byte for byte, or removed before the first epoch — and the flush's own
    error is raised, never one from the restore."""
    name = os.path.join("history", "journal.json") if journal else "idps.csv"
    with monkeypatch.context() as patch:
        failing = _FailTheRenameFlush(patch, name, epoch)
        with pytest.raises(OSError) as caught:
            _fit(entry, HistoryJournal(retention=3, chunk_size=2) if journal else None, callbacks=[failing])
    assert caught.value.errno == errno.EIO
    assert "injected #1" in str(caught.value)  # the flush's own error, not the restore's
    run_id = failing.run_id
    path = os.path.join("runs", run_id, name)
    if epoch == 0:
        assert failing.previous is None
        assert not os.path.exists(path)  # the run reads as uncommitted
        if journal:
            # Nothing but the journal was written: the reservation is released
            # (its release, not the manifest's removal, is what this asserts;
            # the metadata.yaml first-epoch case below pins the removal).
            assert not os.path.exists(os.path.join("runs", run_id))
            return
        assert NNCheckpoint.load(run_id, Checkpoints.LAST) is None
        assert not NNRun.load(run_id).idps
    else:
        assert failing.previous is not None
        with open(path, "rb") as handle:
            assert handle.read() == failing.previous
        assert NNCheckpoint.load(run_id, Checkpoints.LAST).idp.epoch_idx == 0
        assert {record.epoch_idx for record in NNRun.load(run_id).idps} == {0}
    leftovers = [
        entry for entry in os.listdir(os.path.dirname(path)) if entry.startswith(f".{os.path.basename(name)}.")
    ]
    assert not leftovers


# --- the journal flushes history/ itself before its manifest is renamed ---------------------------


@pytest.mark.parametrize("entry", ["model", "trainer"])
def test_each_journal_epoch_flushes_history_after_its_writes_and_before_its_manifest(entry, monkeypatch):
    """Every chunk rename and every new index or epoch-rows entry of an epoch
    is flushed into ``history/`` before the manifest naming them is renamed
    — by the epoch's own flush, not by the flush after the manifest."""
    import nnx.history as history_module

    with monkeypatch.context() as patch:
        events = _trace(patch)
        real_append = history_module._append_bytes

        def append(path, data):
            real_append(path, data)
            events.append(("write", os.path.relpath(os.path.abspath(path), os.path.abspath("runs"))))

        patch.setattr(history_module, "_append_bytes", append)
        run = _fit(entry, HistoryJournal(retention=3, chunk_size=2), n_epochs=3)
    history = os.path.join(run.id, "history")
    manifest = ("rename", os.path.join(history, "journal.json"))
    published = [i for i, event in enumerate(events) if event == manifest]
    assert len(published) >= 3
    for epoch, at in enumerate(published[:3]):
        writes = [
            i
            for i in range(at)
            if events[i][0] in ("rename", "write")
            and os.path.dirname(str(events[i][1])) == history
            and events[i] != manifest
        ]
        assert writes, f"epoch {epoch}: no journal write traced"
        assert _flushed_between(events, writes[-1], at, os.path.join("runs", history)), f"epoch {epoch}"


class _KeepTheFile(Callback):
    """The bytes of the run's ``name`` at ``epoch``'s end, before its history
    is saved (``None``: there is no file yet)."""

    def __init__(self, name: str, epoch: int = 1) -> None:
        self.name, self.epoch, self.kept = name, epoch, None

    def on_epoch_end(self, ctx) -> None:
        if ctx.epoch == self.epoch:
            self.run_id = _the_run_id()
            path = os.path.join("runs", self.run_id, self.name)
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    self.kept = handle.read()


@pytest.mark.parametrize("epoch", [0, 1], ids=["first-epoch", "later-epoch"])
@pytest.mark.parametrize("entry", ["model", "trainer"])
@pytest.mark.parametrize("name", ["run.yaml", "metadata.yaml"])
def test_a_failed_run_file_save_after_the_journal_manifest_republishes_it(entry, name, epoch, monkeypatch):
    """A journal epoch saves ``run.yaml`` and ``metadata.yaml`` after its
    manifest's rename; if that save fails, the previous manifest is put back
    as for a failed flush of the rename itself — or, before the first epoch,
    the new manifest is removed."""
    manifest = os.path.join("history", "journal.json")
    kept = _KeepTheFile(manifest, epoch)
    with monkeypatch.context() as patch:
        failing = _FailTheFlushOf(patch, name, epoch)
        with pytest.raises(OSError) as caught:
            _fit(entry, HistoryJournal(retention=3, chunk_size=2), callbacks=[kept, failing])
    assert caught.value.errno == errno.EIO
    run_id = kept.run_id
    path = os.path.join("runs", run_id, manifest)
    if epoch == 0:
        assert kept.kept is None
        if name == "run.yaml":
            # No run file was saved: the reservation is released with its journal.
            assert not os.path.exists(os.path.join("runs", run_id))
            return
        # run.yaml was saved, so the reservation stays: only the manifest's
        # removal leaves the run uncommitted.
        assert os.path.exists(os.path.join("runs", run_id, "run.yaml"))
        assert not os.path.exists(path)
        assert NNCheckpoint.load(run_id, Checkpoints.LAST) is None
        assert not NNRun.load(run_id).idps
        return
    with open(path, "rb") as handle:
        assert handle.read() == kept.kept
    assert NNCheckpoint.load(run_id, Checkpoints.LAST).idp.epoch_idx == 0
    assert {record.epoch_idx for record in NNRun.load(run_id).idps} == {0}
