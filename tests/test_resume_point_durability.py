"""FIX-030: resume points reach stable storage, not only the drive's cache.

On macOS ``fsync`` only hands the data to the drive, whose cache a power loss
can drop; Apple documents ``fcntl(fd, F_FULLFSYNC)`` for durability. Every
resume-point flush (checkpoint, training state, manifest, directory) uses it
where the platform has it, falls back to ``os.fsync`` elsewhere, and never
fails a save on a filesystem that refuses either."""

from __future__ import annotations

import os
import types

import pytest
import torch

import nnx.nn.params.nn_checkpoint as nn_checkpoint
from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams, NNTrainParams
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import NNCheckpoint


class Recorder:
    def __init__(
        self, monkeypatch, *, full: bool, full_fails: bool = False, fsync_fails: bool = False, code: int = 45
    ) -> None:
        self.full_calls: list[int] = []
        self.fsync_calls: list[int] = []
        fake = types.SimpleNamespace()
        if full:
            fake.F_FULLFSYNC = 51

            def fcntl(fd, command, *args):
                assert command == 51
                self.full_calls.append(fd)
                if full_fails:
                    raise OSError(code, os.strerror(code))
                return 0

            fake.fcntl = fcntl
        monkeypatch.setattr(nn_checkpoint, "fcntl", fake)

        def fsync(fd):
            self.fsync_calls.append(fd)
            if fsync_fails:
                raise OSError(code if code != 45 else 22, "flush failed")

        monkeypatch.setattr(nn_checkpoint.os, "fsync", fsync)


def test_full_fsync_is_used_where_the_platform_has_it(tmp_path, monkeypatch):
    calls = Recorder(monkeypatch, full=True)
    with open(tmp_path / "f", "wb") as handle:
        nn_checkpoint._full_fsync(handle.fileno())
        assert calls.full_calls == [handle.fileno()] and calls.fsync_calls == []


def test_os_fsync_is_the_fallback(tmp_path, monkeypatch):
    calls = Recorder(monkeypatch, full=False)
    with open(tmp_path / "f", "wb") as handle:
        nn_checkpoint._full_fsync(handle.fileno())
        assert calls.fsync_calls == [handle.fileno()]


def test_without_fcntl_at_all_os_fsync_is_used(tmp_path, monkeypatch):
    calls = Recorder(monkeypatch, full=False)
    monkeypatch.setattr(nn_checkpoint, "fcntl", None)  # Windows
    with open(tmp_path / "f", "wb") as handle:
        nn_checkpoint._full_fsync(handle.fileno())
        assert calls.fsync_calls == [handle.fileno()]


def test_a_filesystem_refusing_full_fsync_falls_back_and_refusing_both_does_not_fail(tmp_path, monkeypatch):
    calls = Recorder(monkeypatch, full=True, full_fails=True)
    with open(tmp_path / "f", "wb") as handle:
        nn_checkpoint._full_fsync(handle.fileno())
        assert calls.full_calls == [handle.fileno()] and calls.fsync_calls == [handle.fileno()]
    Recorder(monkeypatch, full=True, full_fails=True, fsync_fails=True)
    with open(tmp_path / "g", "wb") as handle:
        nn_checkpoint._full_fsync(handle.fileno())  # best effort: no error


@pytest.mark.parametrize("full", [True, False])
def test_every_resume_point_flush_goes_through_it(tmp_path, monkeypatch, full):
    """A stateful save flushes the checkpoint, its training state, the staged
    and live manifests and the directory — each with F_FULLFSYNC where it
    exists, else os.fsync — and still saves on a filesystem refusing both."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    loader = [(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))]
    run = model.train(params=NNTrainParams(n_epochs=1, train_loader=loader))
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    state = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)
    with monkeypatch.context() as scoped:
        calls = Recorder(scoped, full=full, full_fails=not full, fsync_fails=not full)
        checkpoint.save(
            run.id, Checkpoints.LAST, optimizer_state=state["optimizer"], scheduler_state=state["scheduler"]
        )
    flushed = calls.full_calls if full else calls.fsync_calls
    # checkpoint + generation sidecar + staged and live manifests + the directory (not the .opt.pt alias)
    assert len(flushed) == 5
    assert NNCheckpoint.verify(run.id, Checkpoints.LAST) is not None


def test_the_real_platform_call_succeeds(tmp_path):
    with open(tmp_path / "f", "wb") as handle:
        handle.write(b"x")
        handle.flush()
        nn_checkpoint._full_fsync(handle.fileno())
    assert os.path.getsize(tmp_path / "f") == 1


def test_a_real_io_error_fails_the_save_and_keeps_the_previous_point(tmp_path, monkeypatch):
    """Only an unsupported flush is tolerated: an EIO raises, and the live
    manifest still names the previous point."""
    import errno

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    run = model.train(params=NNTrainParams(n_epochs=1, train_loader=[(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))]))
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    state = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)
    before = NNCheckpoint.resume_point(run.id, Checkpoints.LAST)
    for full in (True, False):
        with monkeypatch.context() as scoped:
            Recorder(scoped, full=full, full_fails=True, fsync_fails=True, code=errno.EIO)
            with pytest.raises(OSError) as caught:
                checkpoint.save(run.id, Checkpoints.LAST, optimizer_state=state["optimizer"])
        assert caught.value.errno == errno.EIO
        assert NNCheckpoint.resume_point(run.id, Checkpoints.LAST) == before
        assert NNCheckpoint.verify(run.id, Checkpoints.LAST) == before


def test_the_real_platform_call_succeeds_on_a_directory(tmp_path):
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        nn_checkpoint._full_fsync(fd)
    finally:
        os.close(fd)
