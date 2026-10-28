"""FEAT-019: every fit is a recorded attempt.

A fresh attempt id per fit; resume links the parent attempt and checkpoint
generation; repeated saves keep the attempt and the fingerprint; two
attempts at one plan share the fingerprint; a run without a manifest has
no provenance and the same run id. Failed and cancelled fits record their
status and last committed checkpoint without masking the error, and an
interrupted write leaves the previous record intact.
"""

from __future__ import annotations

import json
import os
import warnings

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Callback,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNTrainerParams,
    NNTrainParams,
    Trainer,
)
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.provenance import ExperimentManifest, hash_bytes, load_provenance

PLAN = ExperimentManifest(
    data={"train": hash_bytes(b"synthetic-8x4")}, objective={"id": "supervised", "version": 1}, config={"lr": 0.1}
)


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")


def _loader() -> DataLoader:
    X = torch.randn(8, 4, generator=torch.Generator().manual_seed(0))
    return DataLoader(TensorDataset(X, (X[:, 0] > 0).long()), batch_size=4)


def _model() -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _params(epochs: int = 2, **kwargs) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=epochs, train_loader=_loader(), optim=NNOptimParams.builder().sgd(max_lr=0.1).build(), **kwargs
    )


def _files(run_id: str) -> tuple[bytes, bytes]:
    base = os.path.join("runs", run_id)
    with (
        open(os.path.join(base, "provenance.json"), "rb") as manifest,
        open(os.path.join(base, "attempt.json"), "rb") as attempt,
    ):
        return manifest.read(), attempt.read()


def test_each_fit_is_a_fresh_attempt_and_resume_links_its_parent():
    first = _model().train(params=_params(), provenance=PLAN)
    record = first.provenance
    assert record is not None and record.fingerprint == PLAN.fingerprint()
    attempt = record.attempt
    assert attempt is not None and attempt.status == "completed" and attempt.parent is None
    last = NNCheckpoint.load(run=first.id, type=Checkpoints.LAST)
    assert last is not None
    assert attempt.last_committed == {"checkpoint": "last", "epoch": 1, "generation": last.training_state_id}

    manifest_bytes = _files(first.id)[0]
    child = _model().train(params=_params(1, resume_from_run_id=first.id), provenance=PLAN)
    assert _files(first.id)[0] == manifest_bytes  # the parent's manifest is untouched by the resume
    assert _files(child.id)[0] == manifest_bytes  # same plan, byte-identical manifest
    child_attempt = child.provenance.attempt
    assert child_attempt.attempt_id != attempt.attempt_id and child_attempt.fingerprint == attempt.fingerprint
    assert child_attempt.parent == {
        "run_id": first.id,
        "attempt_id": attempt.attempt_id,
        "checkpoint": "last",
        "epoch": 1,
        "generation": last.training_state_id,
    }


def test_repeated_saves_keep_the_attempt_and_fingerprint_and_plans_share_it():
    run = _model().train(params=_params(1, data_id="one"), provenance=PLAN)
    before = _files(run.id)
    run.save()
    run.save()
    assert _files(run.id) == before
    loaded = NNRun.load(run.id)
    assert loaded.provenance == run.provenance
    other = _model().train(params=_params(1, data_id="two"), provenance=PLAN)  # a second attempt at one plan
    assert other.provenance.fingerprint == run.provenance.fingerprint
    assert other.provenance.attempt.attempt_id != run.provenance.attempt.attempt_id


def test_runs_without_a_manifest_have_no_provenance_and_the_same_run_id():
    plain = _model().train(params=_params(1, data_id="same"))
    assert plain.provenance is None and NNRun.load(plain.id).provenance is None
    assert load_provenance(plain.id) is None
    recorded = _model().train(params=_params(1, data_id="same", overwrite_existing=True), provenance=PLAN)
    assert recorded.id == plain.id  # provenance never enters the run id
    assert "provenance" not in json.dumps(recorded.state())
    with pytest.raises(TypeError, match="ExperimentManifest"):
        _model().train(params=_params(1, data_id="bad"), provenance={"task": "x"})  # type: ignore[arg-type]
    assert not os.path.exists(os.path.join("runs", "bad"))


class _FailAt(Callback):
    def __init__(self, *, epoch_end: int | None = None, train_end: bool = False, error=RuntimeError):
        self.epoch_end, self.train_end, self.error = epoch_end, train_end, error

    def on_epoch_end(self, ctx):
        if self.epoch_end is not None and ctx.epoch == self.epoch_end:
            raise self.error("boom in epoch")

    def on_train_end(self, ctx):
        if self.train_end:
            raise self.error("boom at train end")


@pytest.mark.parametrize(
    ("callback", "status", "epoch"),
    [
        (_FailAt(epoch_end=1), "failed", 0),  # epoch 0 committed, epoch 1 fails
        (_FailAt(train_end=True), "failed", 2),  # every epoch committed, on_train_end fails
        (_FailAt(epoch_end=1, error=KeyboardInterrupt), "cancelled", 0),
    ],
    ids=["validation", "on_train_end", "cancelled"],
)
def test_failed_and_cancelled_fits_record_status_without_masking_the_error(callback, status, epoch):
    with pytest.raises((RuntimeError, KeyboardInterrupt), match="boom"):
        _model().train(params=_params(3), provenance=PLAN, callbacks=[callback])
    (run_id,) = [name for name in os.listdir("runs") if not name.startswith(".") and name != "best"]
    attempt = load_provenance(run_id).attempt
    assert attempt.status == status and attempt.error["message"].startswith("boom")
    assert attempt.last_committed is not None and attempt.last_committed["epoch"] == epoch


def test_a_fit_that_fails_before_any_commit_releases_its_reservation():
    with pytest.raises(RuntimeError, match="boom in epoch"):
        _model().train(params=_params(2, data_id="early"), provenance=PLAN, callbacks=[_FailAt(epoch_end=0)])
    assert [name for name in os.listdir("runs") if not name.startswith(".")] == []  # retry-able, nothing left
    rerun = _model().train(params=_params(2, data_id="early"), provenance=PLAN)
    assert rerun.provenance.attempt.status == "completed"


def test_a_fit_whose_first_run_save_fails_releases_its_reservation(monkeypatch):
    """A first run.yaml write that fails leaves only NNRun.save's lock: the
    reservation is released, so a retry of the same run is not refused with
    "already exists". (A failed first LAST commit keeps an empty history,
    like the eager run; tests/test_history_journal.py pins that.)"""
    from nnx.nn.params import nn_run

    def dump(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(nn_run.yaml, "safe_dump", dump)
        with pytest.raises(OSError, match="disk full"):
            _model().train(params=_params(2, data_id="commit"), provenance=PLAN)
    assert [name for name in os.listdir("runs") if not name.startswith(".")] == []
    rerun = _model().train(params=_params(2, data_id="commit"), provenance=PLAN)
    assert rerun.provenance.attempt.status == "completed"


def test_a_failed_status_write_never_masks_the_training_error(monkeypatch):
    from nnx import provenance

    real = provenance._write_json
    calls = []

    def flaky(path, value):
        calls.append(path)
        if value.get("status") == "failed":
            raise OSError("disk full")
        real(path, value)

    monkeypatch.setattr(provenance, "_write_json", flaky)
    with pytest.warns(RuntimeWarning, match="could not record the failed attempt"):
        with pytest.raises(RuntimeError, match="boom in epoch"):
            _model().train(params=_params(3), provenance=PLAN, callbacks=[_FailAt(epoch_end=1)])


def test_an_interrupted_manifest_write_leaves_the_previous_one_intact(monkeypatch):
    run = _model().train(params=_params(1), provenance=PLAN)
    before = _files(run.id)
    from nnx import provenance

    def interrupted_replace(src, dst):
        raise KeyboardInterrupt("power cut during rename")

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", interrupted_replace)
        with pytest.raises(KeyboardInterrupt):
            provenance._write_json(os.path.join("runs", run.id, "attempt.json"), {"status": "garbage"})
    assert _files(run.id) == before
    assert not [name for name in os.listdir(os.path.join("runs", run.id)) if name.startswith(".attempt.json.")]


def test_trainer_fits_are_recorded_attempts_too():
    params = (
        NNTrainerParams.builder()
        .n_epochs(1)
        .train_loader(_loader())
        .optimizer("default", NNOptimParams.builder().sgd(max_lr=0.1).build())
        .save_phase_checkpoints(False)
        .build()
    )
    from nnx.objectives import supervised_objective

    run = Trainer(_model()).train(params=params, objective=supervised_objective(), provenance=PLAN)
    assert run.provenance.attempt.status == "completed" and run.provenance.fingerprint == PLAN.fingerprint()
    assert NNRun.load(run.id).provenance.attempt.attempt_id == run.provenance.attempt.attempt_id
    with pytest.raises(RuntimeError, match="boom at train end"):
        Trainer(_model()).train(
            params=params,
            objective=supervised_objective(),
            provenance=PLAN,
            callbacks=[_FailAt(train_end=True)],
            salt="failing",
        )
    statuses = sorted(
        record.attempt.status
        for name in os.listdir("runs")
        if not name.startswith(".") and name != "best" and (record := load_provenance(name)) is not None
    )
    assert statuses == ["completed", "failed"]


# --- review hardening -------------------------------------------------------------------------


@pytest.mark.parametrize("content", [[], "text", {"format": "x", "manifest": []}])
def test_provenance_that_is_not_a_json_object_is_unreadable_not_a_crash(tmp_path, content):
    """Valid JSON of the wrong shape is a ValueError (which run loading
    tolerates), not an AttributeError that escapes it."""
    run = _model().train(params=_params(1, data_id=f"shape-{type(content).__name__}"), provenance=PLAN)
    with open(os.path.join("runs", run.id, "provenance.json"), "w", encoding="utf-8") as handle:
        json.dump(content, handle)
    with pytest.raises(ValueError, match="JSON object"):
        load_provenance(run.id)
    with pytest.warns(RuntimeWarning, match="unreadable provenance ignored"):
        assert NNRun.load(run.id).provenance is None


@pytest.mark.parametrize("field", ["data", "splits", "config"])
def test_a_nested_manifest_field_of_the_wrong_shape_is_unreadable_not_a_crash(tmp_path, field):
    run = _model().train(params=_params(1, data_id=f"nested-{field}"), provenance=PLAN)
    path = os.path.join("runs", run.id, "provenance.json")
    stored = json.load(open(path, encoding="utf-8"))
    stored["manifest"][field] = ["not", "an", "object"]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle)
    with pytest.raises(ValueError, match="JSON object"):
        load_provenance(run.id)
    with pytest.warns(RuntimeWarning, match="unreadable provenance ignored"):
        assert NNRun.load(run.id).provenance is None


def test_unreadable_provenance_never_breaks_loading_or_resuming(tmp_path):
    run = _model().train(params=_params(1, data_id="corrupt"), provenance=PLAN)
    path = os.path.join("runs", run.id, "provenance.json")
    stored = json.load(open(path, encoding="utf-8"))
    stored["manifest"]["config"]["lr"] = 0.2  # edited: the stored fingerprint no longer matches
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle)
    with pytest.raises(ValueError, match="does not match its manifest"):
        load_provenance(run.id)
    with pytest.warns(RuntimeWarning, match="unreadable provenance ignored"):
        assert NNRun.load(run.id).provenance is None
    with pytest.warns(RuntimeWarning, match="its attempt is not linked") as linked:
        child = _model().train(params=_params(1, resume_from_run_id=run.id), provenance=PLAN)
    assert [w.filename for w in linked if "not linked" in str(w.message)] == [__file__]  # the caller's train()
    assert child.provenance.attempt.parent["attempt_id"] is None
    assert child.provenance.attempt.parent["generation"] is not None  # the checkpoint is still linked


def test_a_failed_completion_write_keeps_the_trained_run(monkeypatch):
    from nnx import provenance

    real = provenance._write_json

    def flaky(path, value):
        if value.get("status") == "completed":
            raise OSError("disk full")
        real(path, value)

    monkeypatch.setattr(provenance, "_write_json", flaky)
    with pytest.warns(RuntimeWarning, match="could not be recorded as completed") as caught:
        run = _model().train(params=_params(1, data_id="flaky"), provenance=PLAN)
    assert run.idps and load_provenance(run.id).attempt.status == "running"
    assert [w.filename for w in caught if "recorded as completed" in str(w.message)] == [__file__]


def test_a_failed_completion_write_keeps_the_trained_run_under_an_error_filter(monkeypatch):
    import warnings

    from nnx import provenance

    real = provenance._write_json

    def flaky(path, value):
        if value.get("status") == "completed":
            raise OSError("disk full")
        real(path, value)

    monkeypatch.setattr(provenance, "_write_json", flaky)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # -W error: the warning must not lose the saved run
        run = _model().train(params=_params(1, data_id="flaky-error"), provenance=PLAN)
    assert run.idps


def test_checkpoint_summaries_never_load_the_weights():
    from nnx.provenance import _checkpoint_summary

    def no_full_load(*args, **kwargs):
        raise AssertionError("provenance must not deserialize whole checkpoints")

    run = _model().train(params=_params(1, data_id="mmap"), provenance=PLAN)
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(NNCheckpoint, "from_file", no_full_load)
        patched.setattr(NNCheckpoint, "load", no_full_load)
        summary = _checkpoint_summary(run.id, "last", None)
    assert summary == run.provenance.attempt.last_committed


# --- #537: the parent is the generation the resume restored ------------------------------------


def _trainer_params(epochs: int, **resume) -> NNTrainerParams:
    builder = (
        NNTrainerParams.builder()
        .n_epochs(epochs)
        .train_loader(_loader())
        .optimizer("default", NNOptimParams.builder().sgd(max_lr=0.1).build())
    )
    if resume:
        builder.resume_from(**resume)
    return builder.build()


@pytest.mark.parametrize("loop", ["nnmodel", "trainer"])
def test_a_resumed_attempt_links_the_generation_it_restored(loop):
    """LAST is published again (epoch 5) after the attempt started but
    before the resume verified it: the resume restores that generation, and
    the parent and ResumeStatus name it, not the one read at the start."""
    from nnx.objectives import supervised_objective
    from tests._checkpoint_files import republished_on_verify

    def fit(epochs, **resume):
        if loop == "nnmodel":
            params = _params(epochs, resume_from_run_id=resume["run_id"]) if resume else _params(epochs)
            return _model().train(params=params, provenance=PLAN)
        return Trainer(_model()).train(
            params=_trainer_params(epochs, **resume), objective=supervised_objective(), provenance=PLAN
        )

    first = fit(2)
    with republished_on_verify(first.id, before=True):
        child = fit(1, run_id=first.id)
    restored = NNCheckpoint.resume_point(first.id, Checkpoints.LAST)
    assert restored["generation"] == 4 and restored["completed_epoch"] == 5
    status = NNRun.load(child.id).resume_status
    assert (status.source_generation, status.source_checkpoint_id, status.source_epoch) == (
        4,
        restored["checkpoint_id"],
        5,
    )
    for record in (child.provenance, load_provenance(child.id)):
        parent = record.attempt.parent
        assert (parent["epoch"], parent["generation"]) == (5, restored["checkpoint_id"]), parent
        assert parent["attempt_id"] == first.provenance.attempt.attempt_id


def test_a_nested_fit_without_a_manifest_leaves_the_enclosing_attempts_parent():
    """A callback's train() without provenance resumes the same source run
    from another checkpoint: its resume read is not the enclosing attempt's,
    whose parent still names the generation the outer resume restored."""
    first = _model().train(params=_params(2), provenance=PLAN)
    NNCheckpoint.load(run=first.id, type=Checkpoints.LAST).save(run=first.id, type="weights_e1")
    assert NNCheckpoint.resume_point(first.id, "weights_e1")["checkpoint_id"] is None
    inner_runs = []

    class _NestedFit(Callback):
        def on_epoch_end(self, ctx):
            params = _params(
                1, resume_from_run_id=first.id, resume_from_checkpoint="weights_e1", resume_mode="weights_only"
            )
            inner_runs.append(_model().train(params=params))

    child = _model().train(params=_params(1, resume_from_run_id=first.id), provenance=PLAN, callbacks=[_NestedFit()])
    assert len(inner_runs) == 1 and inner_runs[0].provenance is None
    restored = NNCheckpoint.resume_point(first.id, Checkpoints.LAST)
    expected = (restored["completed_epoch"], restored["checkpoint_id"])
    for record in (child.provenance, load_provenance(child.id)):
        parent = record.attempt.parent
        assert (parent["epoch"], parent["generation"]) == expected, parent


# --- #551: every manifest that constructs reads back -----------------------------------------------


def _nested(levels: int) -> dict:
    """A mapping nested ``levels`` deep, built without recursion."""
    value: dict = {}
    for _ in range(levels - 1):
        value = {"k": value}
    return value


@pytest.mark.parametrize("loop", ["nnmodel", "trainer"])
def test_the_deepest_manifest_round_trips_and_a_deeper_one_never_trains(loop):
    """FIX-083: a config nested 511 levels constructed and trained, but
    provenance.json then read back as too deep, so NNRun.load dropped
    run.provenance with only a RuntimeWarning."""
    from nnx.objectives import supervised_objective

    def fit(manifest):
        if loop == "nnmodel":
            return _model().train(params=_params(1), provenance=manifest)
        return Trainer(_model()).train(params=_trainer_params(1), objective=supervised_objective(), provenance=manifest)

    for levels in (511, 1500):
        with pytest.raises(ValueError, match=r"\$\.config: nested more than 254 levels"):
            ExperimentManifest(config=_nested(levels))
    deepest = ExperimentManifest(
        task=_nested(254), model=_nested(254), objective=_nested(254), config=_nested(254), data=PLAN.data
    )
    run = fit(deepest)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = NNRun.load(run.id)
    assert not [w for w in caught if "provenance" in str(w.message)]  # an unreadable file warns and is dropped
    assert loaded.provenance is not None and loaded.provenance.fingerprint == deepest.fingerprint()
    assert loaded.provenance.manifest == deepest and load_provenance(run.id).manifest == deepest
