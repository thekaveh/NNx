"""FIX-025: ``_atomic_write_text`` cleans up its own temporary file.

The helper writes a same-directory ``mkstemp`` temp and renames it over the
destination. It used to unlink the temp only around ``os.replace``, so a
failure in ``fdopen`` / ``write`` / ``flush`` / close leaked a
``.<name>.XXXXXX`` file beside every run text file. Faults are injected at
the helper's own boundary (``os.fdopen`` / ``os.replace`` / ``os.unlink`` as
seen from ``nnx.nn.params.nn_run``).
"""

from __future__ import annotations

import errno
import os
import tempfile
from unittest import mock

import pytest

import nnx.nn.params.nn_run as nn_run
from nnx.nn.params.nn_run import _atomic_write_text


class Injected(Exception):
    pass


class _FileProxy:
    """Wrap the helper's real text file and fail one operation on demand."""

    def __init__(self, real, fail, error=None):
        self._real = real
        self._fail = fail
        self._error = error

    def _raise(self, op):
        raise self._error if self._error is not None else Injected(op)

    def write(self, text):
        if self._fail == "write":
            self._raise("write")
        return self._real.write(text)

    def flush(self):
        if self._fail == "flush":
            self._raise("flush")
        return self._real.flush()

    def close(self):
        if self._fail == "close":
            self._real.close()  # an OS-level close error still releases the fd
            self._raise("close")
        if self._fail == "close-unreleased":
            self._real.flush()
            self._raise("close-unreleased")  # the file object is left open
        return self._real.close()


def _inject(monkeypatch, fail, record, error=None):
    """Patch only the helper's boundary: ``tempfile.mkstemp`` (to observe the
    owned temp / descriptor), the ``io`` constructors it uses and
    ``os.replace``, as seen from ``nnx.nn.params.nn_run``."""
    import io
    from types import SimpleNamespace

    real_mkstemp = tempfile.mkstemp
    real_replace = os.replace

    def mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        record.update(fd=fd, tmp=path, dir=kwargs.get("dir"))
        record.setdefault("tmps", []).append(path)
        return fd, path

    def file_io(*args, **kwargs):
        if fail == "fileio":
            raise Injected("fileio")
        return io.FileIO(*args, **kwargs)

    def text_wrapper(*args, **kwargs):
        if fail == "wrapper":
            raise Injected("wrapper")
        return _FileProxy(io.TextIOWrapper(*args, **kwargs), fail, error)

    def replace(src, dst):
        if fail == "replace":
            raise Injected("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(nn_run.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(
        nn_run, "io", SimpleNamespace(FileIO=file_io, TextIOWrapper=text_wrapper, BufferedWriter=io.BufferedWriter)
    )
    monkeypatch.setattr(nn_run.os, "replace", replace)


_FAILURES = ["fileio", "wrapper", "write", "flush", "close", "close-unreleased", "replace"]


@pytest.mark.parametrize("existing", [True, False], ids=["existing", "absent"])
@pytest.mark.parametrize("fail", _FAILURES)
def test_failure_leaves_no_owned_temp_or_descriptor(tmp_path, monkeypatch, fail, existing):
    target = tmp_path / "run.yaml"
    if existing:
        target.write_bytes(b"old")
    sentinel = tmp_path / ".run.yaml.sentinel"  # somebody else's temp
    sentinel.write_bytes(b"unowned")
    record: dict = {}
    _inject(monkeypatch, fail, record)

    with pytest.raises(Injected, match=fail):
        _atomic_write_text(str(target), "new")

    assert not os.path.exists(record["tmp"])
    with pytest.raises(OSError) as info:
        os.fstat(record["fd"])
    assert info.value.errno == errno.EBADF
    assert sentinel.read_bytes() == b"unowned"
    if existing:
        assert target.read_bytes() == b"old"
    else:
        assert not target.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [".run.yaml.sentinel"] + (["run.yaml"] if existing else [])
    )


def test_atomic_text_success_utf8(tmp_path, monkeypatch):
    target = tmp_path / "run.yaml"
    target.write_bytes(b"old")
    monkeypatch.setattr(nn_run.os, "fsync", mock.Mock(side_effect=OSError("fsync unsupported")))
    _atomic_write_text(str(target), "label: café\n")
    assert target.read_bytes() == "label: café\n".encode()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.yaml"]


def test_success_never_unlinks_the_renamed_temp_path(tmp_path, monkeypatch):
    unlink = mock.Mock(wraps=os.unlink)
    remove = mock.Mock(wraps=os.remove)
    monkeypatch.setattr(nn_run.os, "unlink", unlink)
    monkeypatch.setattr(nn_run.os, "remove", remove)
    _atomic_write_text(str(tmp_path / "run.yaml"), "x")
    unlink.assert_not_called()
    remove.assert_not_called()


@pytest.mark.parametrize("error", [KeyboardInterrupt(), Injected("write")], ids=["keyboard", "exception"])
@pytest.mark.parametrize("unlink_error", [OSError("unlink failed"), KeyboardInterrupt()], ids=["oserror", "interrupt"])
def test_cleanup_never_masks_the_first_exception(tmp_path, monkeypatch, error, unlink_error):
    record: dict = {}
    _inject(monkeypatch, "write", record, error=error)
    monkeypatch.setattr(nn_run.os, "unlink", mock.Mock(side_effect=unlink_error))
    with pytest.raises(type(error)) as info:
        _atomic_write_text(str(tmp_path / "run.yaml"), "new")
    assert info.value is error


def test_bare_filename_keeps_temp_in_destination_directory(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    before_system_tmp = set(os.listdir(tempfile.gettempdir()))
    record: dict = {}
    _inject(monkeypatch, "none", record)
    _atomic_write_text("run.yaml", "bare")
    assert os.path.realpath(record["dir"]) == os.path.realpath(workdir)
    assert (workdir / "run.yaml").read_text(encoding="utf-8") == "bare"
    assert set(os.listdir(tempfile.gettempdir())) - before_system_tmp == set()

    _inject(monkeypatch, "write", record)
    with pytest.raises(Injected):
        _atomic_write_text("run.yaml", "again")
    assert sorted(p.name for p in workdir.iterdir()) == ["run.yaml"]
    assert (workdir / "run.yaml").read_text(encoding="utf-8") == "bare"


# --- caller level --------------------------------------------------------------


def _tiny_run(tmp_path, monkeypatch, n_epochs=1):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

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
        Optims,
    )

    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))), batch_size=4)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    params = NNTrainParams(
        n_epochs=n_epochs,
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
    )
    return model, params


def _owned_temps_left(record):
    return [path for path in record.get("tmps", []) if os.path.exists(path)]


def test_nnrun_save_failure_preserves_files_and_leaves_no_temp(tmp_path, monkeypatch):
    model, params = _tiny_run(tmp_path, monkeypatch)
    run = model.train(params=params)
    run_dir = tmp_path / "runs" / run.id
    before = {name: (run_dir / name).read_bytes() for name in ("run.yaml", "metadata.yaml", "idps.csv")}
    record: dict = {}
    _inject(monkeypatch, "write", record)
    with pytest.raises(Injected):
        run.save()
    assert {name: (run_dir / name).read_bytes() for name in before} == before
    assert record["tmps"] and _owned_temps_left(record) == []


def test_writable_lease_failure_leaves_no_temp(tmp_path, monkeypatch):
    model, params = _tiny_run(tmp_path, monkeypatch)
    record: dict = {}
    _inject(monkeypatch, "flush", record)
    with pytest.raises(Injected):
        model.train(params=params)
    # The history-protocol marker is the lease's first text write.
    assert record["tmps"] and os.path.basename(record["tmps"][0]).startswith(".")
    assert _owned_temps_left(record) == []


def test_pointer_fallback_failure_preserves_pointer_and_leaves_no_temp(tmp_path, monkeypatch):
    best = tmp_path / "best"
    best.mkdir()
    (best / "POINTER.txt").write_text("runs/previous", encoding="utf-8")
    monkeypatch.setattr(nn_run.os, "symlink", mock.Mock(side_effect=OSError("no symlinks here")))
    record: dict = {}
    _inject(monkeypatch, "write", record)
    with pytest.raises(Injected):
        nn_run._point_best(str(best), str(tmp_path / "runs" / "next"))
    assert (best / "POINTER.txt").read_text(encoding="utf-8") == "runs/previous"
    assert sorted(p.name for p in best.iterdir()) == ["POINTER.txt"]
    assert not (tmp_path / "best.tmp").exists()


def test_resume_sample_text_save_failure(tmp_path, monkeypatch):
    """A text-save fault in a resumed run (examples/02 setup) leaves the
    source run loadable and resumable."""
    import runpy
    from dataclasses import replace
    from pathlib import Path

    from nnx import NNRun, NNTrainParams, set_seed

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "02_resume_training.py"),
        run_name="__nnx_fault_injection__",
    )  # import stays fault-free
    set_seed(0)
    model, loader = example["_make_model_and_loader"]()
    first = model.train(
        params=NNTrainParams(
            n_epochs=2, train_loader=loader, optim=example["_base_optim"](), scheduler=example["_base_sched"]()
        )
    )
    resumed_params = NNTrainParams(
        n_epochs=3,
        train_loader=loader,
        optim=example["_base_optim"](),
        scheduler=example["_base_sched"](),
        resume_from_run_id=first.id,
    )
    record: dict = {}
    _inject(monkeypatch, "write", record)
    with pytest.raises(Injected):
        model.train(params=resumed_params)
    assert record["tmps"] and _owned_temps_left(record) == []
    monkeypatch.undo()  # drop the fault injection before the recovery half
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    source = NNRun.load(first.id)
    assert source is not None and len({idp.epoch_idx for idp in source.idps}) == 2
    set_seed(0)
    model2, loader2 = example["_make_model_and_loader"]()
    resumed = model2.train(params=replace(resumed_params, train_loader=loader2))
    assert NNRun.load(resumed.id) is not None


def test_failed_lease_marker_releases_the_reservation(tmp_path, monkeypatch):
    """A failed history-marker write releases the empty reservation so the
    same run can be retried (no stranded ``runs/<id>/``)."""
    model, params = _tiny_run(tmp_path, monkeypatch)
    record: dict = {}
    _inject(monkeypatch, "write", record)
    with pytest.raises(Injected):
        model.train(params=params)
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    run_dirs = [p for p in (tmp_path / "runs").iterdir() if p.is_dir() and not p.name.startswith(".")]
    assert run_dirs == []
    assert model.train(params=params) is not None


def test_reservation_release_also_clears_a_stranded_marker_temp(tmp_path):
    """If a failed marker write could not unlink its temp, the lease's release
    rule still frees the otherwise-empty reservation."""
    run_path = tmp_path / "runs" / "abc"
    run_path.mkdir(parents=True)
    (run_path / f".{nn_run._HISTORY_PROTOCOL_FILE}.x1y2z3").write_text("1\n")
    nn_run._release_empty_reservation(str(run_path))
    assert not run_path.exists()

    kept = tmp_path / "runs" / "kept"
    kept.mkdir()
    (kept / nn_run._HISTORY_PROTOCOL_FILE).write_text("1\n")
    (kept / "run.yaml").write_text("x")
    nn_run._release_empty_reservation(str(kept))
    assert sorted(p.name for p in kept.iterdir()) == [nn_run._HISTORY_PROTOCOL_FILE, "run.yaml"]
