"""#394: exact epoch-boundary resume to the planned horizon.

A stateful resume with ``resume_epochs="planned"`` continues a run's own plan
— up to its ``n_epochs`` — with logical epochs, steps and committed updates;
every resume point is generation-stamped and digested, so a torn, mixed or
corrupted one is refused with its reason; the source run is never written."""

from __future__ import annotations

import hashlib
import json
import os

import pytest
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
    NNSchedulerParams,
    NNTrainParams,
)
from nnx.nn.callbacks import Callback
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import NNCheckpoint, ResumePointError
from nnx.nn.params.nn_run import NNRun

DATA = torch.Generator().manual_seed(5)
X, Y = torch.randn(24, 6, generator=DATA), torch.arange(24) % 3
X_VAL, Y_VAL = torch.randn(9, 6, generator=DATA), torch.arange(9) % 3
N_BATCHES = 4  # 24 rows / batch 6


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")


def _model() -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(input_dim=6, output_dim=3, hidden_dims=[16], dropout_prob=0.2, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _optim(kind: str) -> NNOptimParams:
    builder = NNOptimParams.builder()
    return (builder.adam(max_lr=0.01) if kind == "adam" else builder.sgd(max_lr=0.05)).build()


def _params(kind: str, **changes) -> NNTrainParams:
    loader_rng = torch.Generator().manual_seed(7)
    fields = dict(
        n_epochs=4,
        seed=3,
        train_loader=DataLoader(TensorDataset(X, Y), batch_size=6, shuffle=True, generator=loader_rng),
        val_loader=DataLoader(TensorDataset(X_VAL, Y_VAL), batch_size=9),
        optim=_optim(kind),
        # ReduceLROnPlateau acting every epoch: patience 0 and a threshold no epoch meets.
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=0, cooldown=0, threshold=10.0),
    )
    fields.update(changes)
    return NNTrainParams(**fields)


class StopAfter(Callback):
    """Stop once epoch ``epoch`` is committed; record every committed epoch."""

    def __init__(self, epoch, seen):
        self.epoch, self.seen = epoch, seen

    def on_epoch_end(self, ctx):
        self.seen.append(ctx.epoch)
        if self.epoch is not None and ctx.epoch == self.epoch:
            ctx.should_stop = True


def _resume(kind, source_id, **changes):
    return _params(kind, resume_from_run_id=source_id, resume_mode="stateful", resume_epochs="planned", **changes)


def _digest(directory) -> dict[str, str]:
    found = {}
    for base, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(base, name)
            with open(path, "rb") as handle:
                found[os.path.relpath(path, directory)] = hashlib.sha256(handle.read()).hexdigest()
    return found


# --- AC1: 2 + 2 epochs equal 4, bit for bit -----------------------------------------------------


@pytest.mark.parametrize("kind", ["adam", "sgd"])
def test_a_planned_resume_equals_the_uninterrupted_run_bit_for_bit(kind):
    whole_model = _model()
    whole = whole_model.train(params=_params(kind), salt="whole")

    first_model = _model()
    first = first_model.train(params=_params(kind), callbacks=[StopAfter(1, [])], salt="first")
    assert sorted({idp.epoch_idx for idp in first.idps}) == [0, 1]

    resumed_model = _model()
    resumed = resumed_model.train(params=_resume(kind, first.id))
    assert sorted({idp.epoch_idx for idp in resumed.idps}) == [2, 3]  # up to the plan, not 4 more
    for name, weight in whole_model.net.state_dict().items():
        assert torch.equal(weight, resumed_model.net.state_dict()[name]), name
    # The plateau scheduler acted every epoch and continued across the resume.
    lrs = [idp.lr for idp in whole.idps if idp.val_edp is not None]
    assert lrs == sorted(lrs, reverse=True) and lrs[0] > lrs[-1]
    assert [idp.lr for idp in resumed.idps if idp.val_edp is not None] == lrs[2:]


def test_steps_and_updates_continue_logically_and_phase_tags_use_logical_epochs():
    first = _model().train(params=_params("sgd"), callbacks=[StopAfter(1, [])], salt="first")
    updates = []

    class Updates(Callback):
        def on_epoch_end(self, ctx):
            updates.append(ctx.committed_updates)

    resumed = _model().train(params=_resume("sgd", first.id), callbacks=[Updates()])
    assert [idp.iter_idx for idp in resumed.idps][0] == 2 * N_BATCHES  # the step counter continues
    assert updates == [3 * N_BATCHES, 4 * N_BATCHES]
    # Phase tags follow the plan's logical epochs: the resumed epochs 2-3 of 4
    # write the third-quarter tag, never the first one a fresh session would.
    tags = sorted(
        name
        for name in os.listdir(os.path.join("runs", resumed.id, "checkpoints"))
        if name.endswith(".pt") and "." not in name[:-3]
    )
    assert "q3.pt" in tags and "first.pt" not in tags


def test_a_finished_plan_and_a_changed_plan_are_refused_before_anything_is_restored():
    done = _model().train(params=_params("sgd", n_epochs=2), salt="done")
    with pytest.raises(ValueError, match="already completed"):
        _model().train(params=_resume("sgd", done.id, n_epochs=2))
    first = _model().train(params=_params("sgd"), callbacks=[StopAfter(1, [])], salt="first")
    model = _model()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    with pytest.raises(ValueError, match="planned 4 epochs"):
        model.train(params=_resume("sgd", first.id, n_epochs=6))
    assert all(torch.equal(before[k], v) for k, v in model.net.state_dict().items())


def test_the_default_still_trains_n_epochs_more():
    first = _model().train(params=_params("sgd"), callbacks=[StopAfter(1, [])], salt="first")
    more = _model().train(params=_params("sgd", resume_from_run_id=first.id, resume_mode="stateful", n_epochs=2))
    assert sorted({idp.epoch_idx for idp in more.idps}) == [2, 3]
    assert "resume_epochs" not in _params("sgd").state()  # omit-when-default: no run id moves
    assert _resume("sgd", "x").state()["resume_epochs"] == "planned"


# --- AC2: interrupting twice runs each epoch exactly once ---------------------------------------


def test_interrupting_twice_runs_each_epoch_exactly_once():
    whole_model = _model()
    whole_model.train(params=_params("adam"), salt="whole")
    seen: list[int] = []
    first = _model().train(params=_params("adam"), callbacks=[StopAfter(0, seen)], salt="first")
    second = _model().train(params=_resume("adam", first.id), callbacks=[StopAfter(2, seen)])
    last_model = _model()
    last_model.train(params=_resume("adam", second.id), callbacks=[StopAfter(None, seen)])
    assert seen == [0, 1, 2, 3]
    for name, weight in whole_model.net.state_dict().items():
        assert torch.equal(weight, last_model.net.state_dict()[name]), name


# --- AC3: torn, mixed or corrupted resume points are refused --------------------------------------


def _first():
    return _model().train(params=_params("sgd"), callbacks=[StopAfter(1, [])], salt="first")


def test_every_resume_point_is_generation_stamped_and_digested():
    run = _first()
    manifest = NNCheckpoint.resume_point(run.id, Checkpoints.LAST)
    assert manifest["completed_epoch"] == 1 and manifest["global_step"] == 2 * N_BATCHES
    # LAST was written at each committed epoch and once more after on_train_end.
    assert manifest["generation"] == 3
    assert len(manifest["files"]) == 2 and all(len(digest) == 64 for digest in manifest["files"].values())
    assert NNCheckpoint.verify(run.id, Checkpoints.LAST) == manifest


def test_a_flipped_bit_in_a_weight_tensor_is_refused_naming_the_file():
    run = _first()
    path = os.path.join("runs", run.id, "checkpoints", "last.pt")
    data = bytearray(open(path, "rb").read())
    data[len(data) // 2] ^= 0x01
    open(path, "wb").write(bytes(data))
    with pytest.raises(ResumePointError, match="digest mismatch.*last.pt"):
        _model().train(params=_resume("sgd", run.id))


def test_mixed_generations_and_a_torn_manifest_are_refused_with_their_reason():
    run = _first()
    manifest_path = os.path.join("runs", run.id, "checkpoints", "last.pt.manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    torn = dict(manifest, checkpoint_id="0" * 32)
    json.dump(torn, open(manifest_path, "w", encoding="utf-8"))
    with pytest.raises(ResumePointError, match="torn"):
        _model().train(params=_resume("sgd", run.id))
    # A manifest from another generation of the same run: mixed.
    json.dump(dict(manifest, generation=manifest["generation"] - 1), open(manifest_path, "w", encoding="utf-8"))
    with pytest.raises(ResumePointError, match="generation"):
        _model().train(params=_resume("sgd", run.id))


# --- AC4: the source run is never written ----------------------------------------------------------


def test_the_source_run_is_byte_identical_after_a_resume_and_lineage_is_recorded():
    run = _first()
    before = _digest(os.path.join("runs", run.id))
    resumed = _model().train(params=_resume("sgd", run.id))
    assert _digest(os.path.join("runs", run.id)) == before
    assert resumed.id != run.id
    status = NNRun.load(resumed.id).resume_status
    assert (status.source_run_id, status.source_generation) == (run.id, 3)
    assert status.source_checkpoint_id == NNCheckpoint.resume_point(run.id, Checkpoints.LAST)["checkpoint_id"]


def test_callbacks_read_the_scheduler():
    seen = []

    class Reader(Callback):
        def on_epoch_end(self, ctx):
            seen.append(ctx.scheduler.state_dict()["num_bad_epochs"] is not None)

    _model().train(params=_params("sgd", n_epochs=1), callbacks=[Reader()], salt="r")
    assert seen == [True]


# --- review: clocks, legacy sources, crashes --------------------------------------------------------


def test_a_planned_resume_on_the_update_clock_matches_the_uninterrupted_run():
    from nnx import Schedulers

    one_cycle = NNSchedulerParams(
        kind=Schedulers.ONE_CYCLE,
        clock="optimizer_update",
        max_lr=0.05,
        total_steps=4 * N_BATCHES,
        min_lr=0.0,
        factor=0.5,
        patience=0,
        cooldown=0,
        threshold=0.0,
    )
    whole_model = _model()
    whole = whole_model.train(params=_params("sgd", scheduler=one_cycle), salt="whole")
    first = _model().train(params=_params("sgd", scheduler=one_cycle), callbacks=[StopAfter(1, [])], salt="first")
    resumed_model = _model()
    resumed = resumed_model.train(params=_resume("sgd", first.id, scheduler=one_cycle))
    assert [idp.lr for idp in resumed.idps] == [idp.lr for idp in whole.idps][2 * N_BATCHES :]
    for name, weight in whole_model.net.state_dict().items():
        assert torch.equal(weight, resumed_model.net.state_dict()[name]), name


def test_a_corrupted_training_state_is_refused_naming_its_file():
    run = _first()
    manifest = NNCheckpoint.resume_point(run.id, Checkpoints.LAST)
    sidecar = next(name for name in manifest["files"] if ".opt." in name)
    path = os.path.join("runs", run.id, "checkpoints", sidecar)
    data = bytearray(open(path, "rb").read())
    data[len(data) // 3] ^= 0x01
    open(path, "wb").write(bytes(data))
    with pytest.raises(ResumePointError, match=f"digest mismatch in {sidecar}"):
        _model().train(params=_resume("sgd", run.id))


def test_a_planned_resume_needs_the_training_state_and_its_counters():
    run = _first()
    # Weights only: no plan to continue.
    with pytest.raises(ValueError, match="restores the weights only"):
        _model().train(
            params=_params("sgd", resume_from_run_id=run.id, resume_mode="weights_only", resume_epochs="planned")
        )
    # A checkpoint written before resume counters: plan and step unknown.
    checkpoints = os.path.join("runs", run.id, "checkpoints")
    manifest = NNCheckpoint.resume_point(run.id, Checkpoints.LAST)
    sidecar = os.path.join(checkpoints, next(name for name in manifest["files"] if ".opt." in name))
    state = torch.load(sidecar, weights_only=True)
    state.pop("counters")
    torch.save(state, sidecar)
    os.remove(os.path.join(checkpoints, "last.pt.manifest.json"))  # written before manifests, too
    with pytest.raises(ValueError, match="predates resume counters"):
        _model().train(params=_resume("sgd", run.id))
    with pytest.raises(ValueError, match="set resume_from_run_id"):
        _model().train(params=_params("sgd", resume_epochs="planned"), salt="no-source")


def test_a_crash_between_publishing_the_checkpoint_and_its_manifest_still_resumes(monkeypatch):
    """The writer dies after LAST is renamed into place but before its live
    manifest: the staged manifest, every digest matching, completes the point."""
    from nnx.nn.params import nn_checkpoint

    original = nn_checkpoint._write_manifest
    calls = {"live": 0}

    def flaky(path, manifest, *, staged=False):
        if not staged and path.endswith("last.pt"):
            calls["live"] += 1
            if calls["live"] == 2:  # epoch 1's LAST
                raise OSError("disk full (injected)")
        return original(path, manifest, staged=staged)

    monkeypatch.setattr(nn_checkpoint, "_write_manifest", flaky)
    with pytest.raises(OSError, match="injected"):
        _model().train(params=_params("sgd"), callbacks=[StopAfter(1, [])], salt="first")
    monkeypatch.setattr(nn_checkpoint, "_write_manifest", original)
    run_id = next(name for name in os.listdir("runs") if len(name) == 32)
    before = _digest(os.path.join("runs", run_id))
    manifest = NNCheckpoint.verify(run_id, Checkpoints.LAST)  # read from the staged manifest
    assert manifest is not None and manifest["completed_epoch"] == 1 and manifest["generation"] == 2
    resumed = _model().train(params=_resume("sgd", run_id))
    assert sorted({idp.epoch_idx for idp in resumed.idps}) == [2, 3]
    assert _digest(os.path.join("runs", run_id)) == before  # verifying and resuming wrote nothing to the source
    assert NNRun.load(resumed.id).resume_status.source_generation == 2
