"""Every reader that accepts the run id ``"best"`` reads the run ``runs/best``
points at, in the symlink layout and in the ``best/POINTER.txt`` directory
written where symlinks fail (Windows without developer mode). A pointer that
names no run reads as an absent run, and one to ``..`` is never followed out
of the runs root. Saves to ``run="best"`` are unchanged."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNOptimParams, NNParams, NNTrainParams
from nnx.comparison import ComparisonError, Metric, observations_from_runs
from nnx.history import export_history_csv, iter_history
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_run import NNRun
from nnx.provenance import ExperimentManifest, hash_bytes, load_provenance

MANIFEST = ExperimentManifest(data={"train": hash_bytes(b"toy-32")})
OPTIM = NNOptimParams.builder().sgd(max_lr=0.1).build()
LAYOUTS = ["symlink", "pointer_file"]


def _loader():
    X = torch.randn(24, 4, generator=torch.Generator().manual_seed(0))
    return DataLoader(TensorDataset(X, (X[:, 0] > 0).long()), batch_size=8)


def _model():
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _fit_params(**overrides):
    fields = dict(n_epochs=2, seed=0, train_loader=_loader(), val_loader=_loader(), optim=OPTIM, data_id="toy")
    fields.update(overrides)
    return NNTrainParams(**fields)


def _trainer_params(**overrides):
    from nnx.trainer import NNTrainerParams

    fields = dict(n_epochs=2, seed=0, train_loader=_loader(), optims={"default": OPTIM}, data_id="toy")
    fields.update(overrides)
    return NNTrainerParams(**fields)


def _trainer_fit(**overrides):
    from nnx.objectives import supervised_objective
    from nnx.trainer import Trainer

    return Trainer(_model()).train(
        params=_trainer_params(**overrides), objective=supervised_objective(), provenance=MANIFEST
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One NNModel run and one Trainer run under a template root, trained
    once for the module; each test copies the tree."""
    root = tmp_path_factory.mktemp("best_pointer_template")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        patch.setenv("NNX_TQDM_DISABLE", "1")
        model_run = _model().train(params=_fit_params(), provenance=MANIFEST)
        trainer_run = _trainer_fit()
    return root, model_run.id, trainer_run.id


def _point(root: Path, layout: str, target: str) -> None:
    """``runs/best`` naming ``target``: the symlink ``_point_best`` writes (to
    the run's basename), or the ``POINTER.txt`` directory of its fallback
    (holding the run's path)."""
    best = root / "runs" / "best"
    if best.is_symlink() or best.is_file():
        best.unlink()
    elif best.is_dir():
        shutil.rmtree(best)
    if layout == "symlink":
        os.symlink(target, best, target_is_directory=True)
    else:
        best.mkdir()
        (best / "POINTER.txt").write_text(str(root / "runs" / target), encoding="utf-8")


@pytest.fixture
def runs(trained, tmp_path, monkeypatch):
    """The template's runs under ``tmp_path`` (the cwd), and ``point(layout,
    target)``, which repoints ``runs/best``."""
    template, model_id, trainer_id = trained
    shutil.copytree(template / "runs", tmp_path / "runs", symlinks=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    return model_id, trainer_id, lambda layout, target: _point(tmp_path, layout, target)


def _manifest_of(run_id, root):
    return NNCheckpoint.verify(run=run_id, type=Checkpoints.LAST, root=root)


# Each reader as (call, the comparable part of its result).
READERS = {
    "resume_point": lambda run, root: NNCheckpoint.resume_point(run, Checkpoints.LAST, root=root),
    "verify": lambda run, root: NNCheckpoint.verify(run, Checkpoints.LAST, root=root),
    "load_training_state": lambda run, root: _state_id(NNCheckpoint.load_training_state(run, Checkpoints.LAST, root)),
    "load_with_training_state": lambda run, root: _pair_ids(
        NNCheckpoint.load_with_training_state(run, Checkpoints.LAST, root=root)
    ),
    "_load_resume_point": lambda run, root: _pair_ids(
        NNCheckpoint._load_resume_point(run, Checkpoints.LAST, _manifest_of(NNRun.best_id(root) or "x", root), root)
    ),
    "load_optimizer_state": lambda run, root: _optimizer_steps(
        NNCheckpoint.load_optimizer_state(run, Checkpoints.LAST, root=root)
    ),
    "iter_history": lambda run, root: [record.state() for record in iter_history(run, root)],
    "export_history_csv": lambda run, root: _exported(run, root),
    "load_provenance": lambda run, root: _attempt_id(load_provenance(run, root)),
    "observations_from_runs": lambda run, root: [
        (item.run_id, item.value, item.attempt_id)
        for item in observations_from_runs([run], metric=Metric("loss", "minimize"), root=root)
    ],
}
# What each reader returns, or raises, for a run that is not there.
ABSENT = {
    "resume_point": None,
    "verify": None,
    "load_training_state": None,
    "load_with_training_state": (None, None),
    "_load_resume_point": (None, None),
    "load_optimizer_state": None,
    "iter_history": ValueError,
    "export_history_csv": ValueError,
    "load_provenance": None,
    "observations_from_runs": ComparisonError,
}


def _state_id(state):
    return None if state is None else state["checkpoint_id"]


def _pair_ids(pair):
    checkpoint, state = pair
    return (None if checkpoint is None else checkpoint.training_state_id), _state_id(state)


def _optimizer_steps(state):
    return None if state is None else sorted(state["state"])


def _exported(run, root):
    path = Path(root or ".") / "export.csv"
    export_history_csv(run, path, root)
    return path.read_text(encoding="utf-8")


def _attempt_id(record):
    return None if record is None or record.attempt is None else record.attempt.attempt_id


def _absent(reader, run, root):
    expected = ABSENT[reader]
    if isinstance(expected, type):
        with pytest.raises(expected):
            READERS[reader](run, root)
    else:
        assert READERS[reader](run, root) == expected


@pytest.mark.parametrize("reader", sorted(READERS))
@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_reader_of_best_reads_the_run_the_pointer_names(runs, tmp_path, layout, reader):
    """In both layouts, a read of ``"best"`` (and of a case variant, which a
    case-insensitive filesystem opens as ``runs/best``) returns what a read
    of the run's own id returns, with and without ``root=``. On the old code
    every one of them read ``runs/best/`` directly, which in the POINTER.txt
    layout holds no run."""
    model_id, _, point = runs
    point(layout, model_id)
    assert NNRun.best_id() == model_id
    for root in (None, str(tmp_path)):
        own = READERS[reader](model_id, root)
        assert own is not None and own != (None, None) and own != []
        assert READERS[reader]("best", root) == own
    assert READERS[reader]("BEST", str(tmp_path)) == own


@pytest.mark.parametrize("reader", sorted(READERS))
@pytest.mark.parametrize("target", ["..", "deleted"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_reader_of_best_reads_a_pointer_naming_no_run_as_an_absent_run(runs, tmp_path, layout, target, reader):
    """A pointer naming no run (a deleted run, or ``..``) reads as an absent
    run, as it does today. A symlink to ``..`` used to be followed out of
    the runs root, to a run's files beside ``runs/``."""
    model_id, _, point = runs
    for entry in (tmp_path / "runs" / model_id).iterdir():  # a whole run beside runs/
        (shutil.copytree if entry.is_dir() else shutil.copy2)(entry, tmp_path / entry.name)
    point(layout, target)
    assert NNRun.best_id() is None
    _absent(reader, "best", str(tmp_path))
    _absent(reader, "Best", None)


def test_without_any_pointer_every_reader_of_best_reads_an_absent_run(runs, tmp_path):
    shutil.rmtree(tmp_path / "runs" / "best", ignore_errors=True)
    if os.path.lexists(tmp_path / "runs" / "best"):
        os.remove(tmp_path / "runs" / "best")
    for reader in READERS:
        _absent(reader, "best", str(tmp_path))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("entry", ["model", "trainer"])
def test_resume_from_best_reads_the_run_the_pointer_names(runs, layout, entry):
    """``resume_from_run_id="best"`` (``NNModel.train`` and ``Trainer.train``)
    resumes the run ``runs/best`` names in both layouts: its attempt, its
    checkpoint and its provenance. The recorded lineage stays ``"best"``.
    On the old code the POINTER.txt layout raised "not found on disk"."""
    model_id, trainer_id, point = runs
    source = model_id if entry == "model" else trainer_id
    point(layout, source)
    source_run = NNRun.load(source)
    if entry == "model":
        resumed = _model().train(params=_fit_params(n_epochs=1, resume_from_run_id="best"), provenance=MANIFEST)
    else:
        resumed = _trainer_fit(n_epochs=1, resume_from_run_id="best")
    status = NNRun.load(resumed.id).resume_status
    assert status is not None and status.mode == "stateful"
    assert status.source_run_id == "best" and status.source_epoch == 1
    assert status.source_attempt_token == source_run.resume_status.attempt_token
    parent = load_provenance(resumed.id).attempt.parent
    assert parent["run_id"] == "best" and parent["attempt_id"] == load_provenance(source).attempt.attempt_id
    assert parent["epoch"] == 1 and parent["generation"] is not None


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("target", ["..", "deleted"])
def test_resume_from_a_best_pointer_naming_no_run_is_not_found(runs, tmp_path, layout, target):
    model_id, _, point = runs
    for entry in (tmp_path / "runs" / model_id).iterdir():  # a whole run beside runs/
        (shutil.copytree if entry.is_dir() else shutil.copy2)(entry, tmp_path / entry.name)
    point(layout, target)
    with pytest.raises(ValueError, match=r"resume_from_run_id='best'/last not found on disk"):
        _model().train(params=_fit_params(n_epochs=1, resume_from_run_id="best"))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_legacy_sidecar_resumed_through_best_is_checked_against_the_run_it_names(runs, tmp_path, layout):
    """#502 for a sidecar written before it: the hyperparameters come from
    the ``run.yaml`` of the run ``runs/best`` names, so a changed learning
    rate is still refused. Reading ``runs/best/run.yaml`` found nothing in
    the POINTER.txt layout, and the check was skipped silently."""
    from dataclasses import replace

    from tests._checkpoint_files import restamp

    model_id, _, point = runs
    point(layout, model_id)
    checkpoint = NNCheckpoint.load(model_id, Checkpoints.LAST)
    sidecar = tmp_path / "runs" / model_id / "checkpoints" / f"last.pt.opt.{checkpoint.training_state_id}.pt"
    state = torch.load(sidecar, weights_only=True)
    for key in ("optimizer_config", "scheduler_config"):
        del state[key]
    torch.save(state, sidecar)
    restamp(sidecar)
    with pytest.raises(ValueError, match=r"optimizer max_lr \(checkpoint 0.1, configuration 0.0001\)"):
        _model().train(params=_fit_params(n_epochs=1, resume_from_run_id="best", optim=replace(OPTIM, max_lr=1e-4)))


class _Ranks:
    """A DDP session stand-in: rank ``rank`` of a world whose writer is rank 0,
    whose gather returns ``values``."""

    def __init__(self, rank, values):
        from types import SimpleNamespace

        self.writer = rank == 0
        self.spec = SimpleNamespace(writer_rank=0)
        self.values = values
        self.sent = []

    def gather(self, value):
        self.sent.append(value)
        return self.values


def test_under_ddp_the_writer_resolves_best_and_every_rank_reads_its_run(runs, monkeypatch):
    """A rank other than the writer never reads the pointer: it reads the run
    the writer resolved, so a repoint between two ranks' reads cannot split
    them. A run's own id is never gathered."""
    from nnx.distributed import DistributedFailure
    from nnx.nn import nn_model
    from nnx.nn.params import nn_run

    model_id, trainer_id, point = runs
    point("pointer_file", model_id)
    writer = _Ranks(0, [("resolved", model_id), None])
    assert nn_model._resume_source_id("best", writer) == model_id and writer.sent == [("resolved", model_id)]
    monkeypatch.setattr(nn_run.NNRun, "best_id", staticmethod(lambda root=None: pytest.fail("read the pointer")))
    other = _Ranks(1, [("resolved", trainer_id), None])
    assert nn_model._resume_source_id("best", other) == trainer_id and other.sent == [None]
    assert nn_model._resume_source_id(model_id, other) == model_id and other.sent == [None]
    refused = _Ranks(1, [("refused", "PermissionError", "denied"), None])
    with pytest.raises(DistributedFailure, match="rank 0: PermissionError: denied"):
        nn_model._resume_source_id("best", refused)


def test_a_resume_records_the_attempt_of_the_run_it_read_when_best_was_repointed(runs, tmp_path):
    """The attempt record links its parent when it starts, before the resume
    reads its source: when ``runs/best`` is repointed in between, the parent
    names the attempt and checkpoint of the run the resume read."""
    from types import SimpleNamespace

    from nnx.provenance import _AttemptRecorder

    model_id, trainer_id, point = runs
    point("pointer_file", model_id)
    child = SimpleNamespace(id="c" * 32)
    (tmp_path / "runs" / child.id).mkdir()
    recorder = _AttemptRecorder(child, MANIFEST, parent_run_id="best", parent_checkpoint="last")
    assert recorder.attempt.parent["attempt_id"] == load_provenance(model_id).attempt.attempt_id
    read = NNCheckpoint.load(trainer_id, Checkpoints.LAST)
    recorder.resumed_from("best", read, trainer_id)
    parent = recorder.attempt.parent
    assert parent["run_id"] == "best" and parent["attempt_id"] == load_provenance(trainer_id).attempt.attempt_id
    assert parent["generation"] == read.training_state_id


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_pointer_to_a_case_variant_of_best_names_no_run(runs, tmp_path, layout):
    """A pointer whose basename is ``BEST`` names the pointer itself: on a
    case-insensitive filesystem ``runs/BEST`` opens ``runs/best``, so it
    would follow the symlink to a run outside the runs root."""
    model_id, _, point = runs
    outside = tmp_path / "outside" / "BEST"
    shutil.copytree(tmp_path / "runs" / model_id, outside)
    point(layout, str(outside))
    assert NNRun.best_id() is None
    with pytest.raises(FileNotFoundError, match="runs/best"):
        NNRun.load("best")
    assert NNCheckpoint.load(run="best", type=Checkpoints.LAST) is None
