"""FEAT-045: a resume point's SHA-256 digests are computed while its files
are written — no file is read back to hash it — and still equal
``hashlib.sha256`` of the files on disk, so ``verify`` accepts them."""

from __future__ import annotations

import builtins
import hashlib
import io
import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import nnx.nn.params.nn_checkpoint as nn_checkpoint
from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
)
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import NNCheckpoint


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    return _train()


def _train():
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=6, output_dim=3, hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    data = torch.Generator().manual_seed(1)
    loader = DataLoader(TensorDataset(torch.randn(12, 6, generator=data), torch.arange(12) % 3), batch_size=6)
    return model.train(
        params=NNTrainParams(
            n_epochs=2, train_loader=loader, val_loader=loader, optim=NNOptimParams.builder().adam(max_lr=0.01).build()
        )
    )


def _on_disk(directory: str, manifest) -> dict[str, str]:
    found = {}
    for name in manifest["files"]:
        with open(os.path.join(directory, name), "rb") as handle:
            found[name] = hashlib.sha256(handle.read()).hexdigest()
    return found


@pytest.mark.parametrize("tag", [Checkpoints.LAST, Checkpoints.BEST])
def test_the_training_saves_digests_equal_the_files_on_disk(run, tag):
    manifest = NNCheckpoint.resume_point(run.id, tag)
    assert manifest is not None and len(manifest["files"]) == 2  # the checkpoint and its training state
    directory = os.path.dirname(nn_checkpoint._checkpoint_path(run.id, tag))
    assert manifest["files"] == _on_disk(directory, manifest)
    assert NNCheckpoint.verify(run.id, tag) == manifest


class _ReadSpy:
    """Every file opened for reading, by any of open / io.open / os.open,
    while installed."""

    def __init__(self, monkeypatch) -> None:
        self.reads: list[str] = []
        real_open, real_os_open = builtins.open, os.open

        def spy_open(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, os.PathLike)) and ("r" in mode or "+" in mode):
                self.reads.append(os.fspath(file))
            return real_open(file, mode, *args, **kwargs)

        def spy_os_open(path, flags, *args, **kwargs):
            if not flags & (os.O_WRONLY | os.O_CREAT | os.O_APPEND):
                self.reads.append(os.fspath(path))
            return real_os_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr(io, "open", spy_open)
        monkeypatch.setattr(os, "open", spy_os_open)

        def no_read_back(path):
            raise AssertionError(f"read back to hash: {path}")

        monkeypatch.setattr(nn_checkpoint, "_sha256", no_read_back)

    def payloads(self) -> list[str]:
        """Reads of anything but a manifest (read to number the generation)
        or a directory (fsynced to persist a rename)."""
        return [path for path in self.reads if not path.endswith(".json") and not os.path.isdir(path)]


@pytest.mark.parametrize("stateful", [True, False])
def test_a_save_reads_no_file_back(run, monkeypatch, stateful):
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    state = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)
    with monkeypatch.context() as scoped:
        spy = _ReadSpy(scoped)
        if stateful:
            checkpoint.save(
                run.id,
                Checkpoints.LAST,
                optimizer_state=state["optimizer"],
                scheduler_state=state["scheduler"],
                rng_state=state["rng"],
            )
        else:
            checkpoint.save(run.id, Checkpoints.LAST)  # weights only
    assert spy.reads and spy.payloads() == []  # the spy saw the manifest read, and nothing else
    manifest = NNCheckpoint.resume_point(run.id, Checkpoints.LAST)
    assert len(manifest["files"]) == (2 if stateful else 1)
    directory = os.path.dirname(nn_checkpoint._checkpoint_path(run.id, Checkpoints.LAST))
    assert manifest["files"] == _on_disk(directory, manifest)
    assert NNCheckpoint.verify(run.id, Checkpoints.LAST) == manifest


def test_a_training_run_reads_no_resume_point_back_while_saving(tmp_path, monkeypatch):
    """LAST, the phase tags and BEST are all saved while training; none is
    hashed by reading it back."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    def no_read_back(path):
        raise AssertionError(f"read back to hash: {path}")

    with monkeypatch.context() as scoped:
        scoped.setattr(nn_checkpoint, "_sha256", no_read_back)
        trained = _train()
    for tag in (Checkpoints.LAST, Checkpoints.BEST):
        assert NNCheckpoint.verify(trained.id, tag) is not None
