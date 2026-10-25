"""FEAT-030: one writer, rank-aware callbacks and same-world resume under DDP."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ddp_adapter import assert_close_states, launch, results  # noqa: E402

# Every torchrun launch shares one worker (FIX-031): its module-scoped launch
# runs once, and no two 2-rank launches compete for a CI runner's 4 vCPUs.
pytestmark = pytest.mark.xdist_group("torchrun")


@pytest.fixture(scope="module")
def writer(tmp_path_factory) -> tuple[Path, list[dict]]:
    out = tmp_path_factory.mktemp("ddp-writer")
    launch("writer", out)
    return out, results(out)


def test_every_rank_returns_the_writers_provenance(writer):
    _, (first, second) = writer
    assert first["provenance"] is not None and first["provenance"] == second["provenance"]


def test_only_the_writer_rank_leases_and_writes(writer):
    out, (first, second) = writer
    assert first["calls"]["lease"] == 1 and second["calls"]["lease"] == 0
    assert first["calls"]["checkpoint_save"] > 0 and second["calls"]["checkpoint_save"] == 0
    assert first["calls"]["run_save"] > 0 and second["calls"]["run_save"] == 0
    runs = [name for name in os.listdir(out / "runs") if not name.startswith(".") and name != "best"]
    assert runs == [first["run_id"]] and first["run_id"] == second["run_id"]
    # the writer's history holds the same global records every rank returned
    assert first["train_loss"] == second["train_loss"]


def test_execution_owned_callbacks_are_built_on_the_writer_only(writer):
    out, (first, second) = writer
    assert first["calls"]["factory"] == 1 and second["calls"]["factory"] == 0
    events = [name for name in os.listdir(out / "tb-writer") if name.startswith("events")]
    assert len(events) == 1  # one writer, one event file
    # the writer-only ModelCheckpoint wrote its deferred checkpoint once
    assert (out / "runs" / first["run_id"] / "checkpoints").exists()


def test_canonical_keys_in_checkpoint(writer):
    _, (first, second) = writer
    assert first["last_keys"] == second["last_keys"]
    assert not [key for key in first["last_keys"] if key.startswith("module.")]
    assert set(first["state"]) == set(first["last_keys"])


def test_canonical_keys_in_hub_and_onnx_export(writer):
    for extra in ("safetensors", "huggingface_hub", "onnx"):
        pytest.importorskip(extra)
    out, (first, _) = writer
    assert first["exported"]
    config = json.loads((out / "hub" / "config.json").read_text(encoding="utf-8"))
    assert config
    from safetensors.torch import load_file

    assert not [key for key in load_file(str(out / "hub" / "model.safetensors")) if key.startswith("module.")]
    assert (out / "net.onnx").stat().st_size > 0


def test_checkpoints_carry_world_partition_and_every_ranks_rng(writer):
    _, (first, _) = writer
    distributed = first["last"]
    assert distributed["world_size"] == 2 and distributed["writer_rank"] == 0
    assert distributed["train"]["policy"] == "pad" and distributed["train"]["seed"] == 3
    assert len(distributed["rng_by_rank"]) == 2


@pytest.fixture(scope="module")
def resumed(tmp_path_factory) -> dict:
    full = tmp_path_factory.mktemp("ddp-full")
    launch("resume_full", full)
    split = tmp_path_factory.mktemp("ddp-split")
    launch("resume_first", split)
    for path in split.glob("rank*.pt"):
        path.rename(path.with_suffix(".first"))
    changed = {}
    launch("resume_changed", split)
    changed["partition"] = results(split)
    launch("resume_second", split, nproc=1)
    changed["world"] = results(split, 1)
    launch("resume_second", split)
    return {"full": results(full), "split": results(split), "changed": changed, "dir": split}


def test_two_epochs_equal_one_plus_a_resume(resumed):
    for full, split in zip(resumed["full"], resumed["split"], strict=True):
        assert_close_states(split["state"], full["state"])
        assert split["train_loss"] == pytest.approx(full["train_loss"][len(full["train_loss"]) // 2 :], rel=1e-6)
        assert torch.equal(split["rng_after"], full["rng_after"])  # each rank's own streams restored


def test_a_planned_resume_continues_the_plan_on_every_rank(resumed, tmp_path_factory):
    """#394 under DDP: stopped after epoch 0 of a 2-epoch plan, a planned
    resume trains epoch 1 only and equals the uninterrupted run."""
    planned = tmp_path_factory.mktemp("ddp-planned")
    launch("resume_planned_first", planned)
    launch("resume_planned_second", planned)
    for full, split in zip(resumed["full"], results(planned), strict=True):
        assert split["epochs"] == [1]
        assert_close_states(split["state"], full["state"])
        assert torch.equal(split["rng_after"], full["rng_after"])


def test_an_on_train_end_draw_keeps_one_plus_a_resume_equal_to_two_epochs(tmp_path_factory):
    """#538: every rank draws in on_train_begin and on_train_end; the final
    LAST holds each rank's streams from before on_train_end, the resume
    re-applies this rank's own after on_train_begin, so it equals the
    straight run."""
    full = tmp_path_factory.mktemp("ddp-draw-full")
    launch("train_end_draw_full", full)
    split = tmp_path_factory.mktemp("ddp-draw-split")
    launch("train_end_draw_first", split)
    first = results(split)
    assert not torch.equal(first[0]["rng_before"], first[1]["rng_before"])  # the ranks' streams differ
    for rank in first:
        for saved, ran in zip(rank["rng_by_rank"], first, strict=True):
            assert torch.equal(saved, ran["rng_before"])  # each rank's own pre-on_train_end streams
    launch("train_end_draw_second", split)
    for full_rank, split_rank in zip(results(full), results(split), strict=True):
        assert split_rank["epochs"] == [1]
        assert_close_states(split_rank["state"], full_rank["state"])
        assert torch.equal(split_rank["rng_before"], full_rank["rng_before"])


def test_a_changed_world_or_partition_fails_before_anything_is_restored(resumed):
    for rank in resumed["changed"]["partition"]:
        assert "train partition changed" in rank["error"] and rank["unchanged"]
    (single,) = resumed["changed"]["world"]
    assert "trained on 2 ranks" in single["error"] and single["unchanged"]


def test_a_single_process_stateful_resume_of_a_distributed_checkpoint_is_refused(resumed, tmp_path_factory):
    from ddp_scenarios import model, params, train_set

    split = resumed["dir"]
    first_id = (split / "first_run_id.txt").read_text(encoding="utf-8").strip()
    previous = os.getcwd()
    os.chdir(split)
    try:
        data = train_set()
        with pytest.raises(ValueError, match="distributed run"):
            model().train(params=params([(data.X, data.y)], n_epochs=1, data_id="split", resume_from_run_id=first_id))
    finally:
        os.chdir(previous)


# --- FEAT-046: one rank verifies the resume point -----------------------------------------------


def test_only_the_writer_rank_hashes_the_resume_point(resumed):
    writer, other = resumed["split"]
    assert (writer["verified"], other["verified"]) == (1, 0)  # the writer's verdict was shared
    assert other["hashed"] == 0  # (the writer also hashes its own saves while training)
    writer, other = resumed["changed"]["partition"]  # verified once, then refused for its partition
    assert (writer["verified"], other["verified"]) == (1, 0)
    assert writer["hashed"] == 2 and other["hashed"] == 0  # the checkpoint and its training state


@pytest.fixture(scope="module")
def corrupted(tmp_path_factory) -> list[dict]:
    out = tmp_path_factory.mktemp("ddp-corrupt")
    launch("resume_first", out)
    launch("resume_corrupt", out)
    return results(out)


def test_a_corrupted_resume_point_is_refused_on_every_rank_naming_the_reason(corrupted):
    for rank in corrupted:
        assert rank["error"].startswith("ResumePointError: "), rank["error"]
        assert "digest mismatch in last.pt.opt." in rank["error"] and rank["unchanged"]
    writer, other = corrupted
    assert writer["error"] == other["error"]  # the same reason on every rank
    assert (writer["verified"], other["verified"]) == (1, 0)
    assert writer["hashed"] >= 1 and other["hashed"] == 0


def test_a_resume_point_republished_after_the_writers_verification_is_refused_on_every_rank(tmp_path):
    """#537: LAST is published again (generation 4) after the writer rank
    verified generation 3 and shared it: a weights-only resume refuses on
    every rank before anything is restored, and no child run is written."""
    launch("resume_first", tmp_path)
    first_id = (tmp_path / "first_run_id.txt").read_text(encoding="utf-8").strip()
    launch("resume_republished", tmp_path)
    writer, other = results(tmp_path)
    for rank in (writer, other):
        assert rank["error"].startswith(f"ResumePointError: {first_id}/last: the resume point changed"), rank["error"]
        assert rank["unchanged"] and rank["rng_unchanged"], rank
    assert "generation 2 was verified, generation 3 is published" in writer["error"], writer["error"]
    assert (writer["verified"], other["verified"]) == (1, 0) and other["hashed"] == 0
    runs = [name for name in os.listdir(tmp_path / "runs") if not name.startswith(".") and name != "best"]
    assert runs == [first_id]


def test_a_resume_point_republished_between_the_ranks_loads_is_refused_alike_on_every_rank(tmp_path):
    """#537: the writer rank loads the verified point cleanly, then LAST is
    published again before rank 1 loads it: rank 1's refusal is shared, so
    every rank raises the same ResumePointError, not a DistributedFailure."""
    launch("resume_first", tmp_path)
    first_id = (tmp_path / "first_run_id.txt").read_text(encoding="utf-8").strip()
    launch("resume_republished_between_loads", tmp_path)
    writer, other = results(tmp_path)
    assert other["error"].startswith(f"ResumePointError: {first_id}/last: the resume point changed"), other["error"]
    assert writer["error"] == other["error"], (writer["error"], other["error"])
    assert "generation 2 was verified, generation 3 is published" in writer["error"], writer["error"]
    for rank in (writer, other):
        assert rank["unchanged"] and rank["rng_unchanged"], rank
    assert other["hashed"] == 0  # the load compares the checkpoint id: nothing is hashed again
    runs = [name for name in os.listdir(tmp_path / "runs") if not name.startswith(".") and name != "best"]
    assert runs == [first_id]


def test_an_invalid_run_id_fails_alike_on_every_rank(tmp_path):
    launch("resume_bad_id", tmp_path)
    for rank in results(tmp_path):
        assert rank["error"].startswith("ValueError: "), rank["error"]  # not a DistributedFailure elsewhere
        assert rank["verified"] == 0 and rank["hashed"] == 0


# --- #502: changed optimizer hyperparameters ----------------------------------------------------


def test_a_stateful_ddp_resume_with_changed_optimizer_hyperparameters_is_refused_on_every_rank(tmp_path):
    """A two-rank resume of a two-rank parent with a new lr / weight_decay /
    momentum would train on every rank with the parent's (the checkpoint's
    param groups) while recording the new ones: every rank refuses before
    the net or its RNG changes, and no child run is written."""
    launch("resume_first", tmp_path)
    first_id = (tmp_path / "first_run_id.txt").read_text(encoding="utf-8").strip()
    launch("resume_hyperparameters", tmp_path)
    for rank in results(tmp_path):
        assert rank["error"].startswith("ValueError: resume hyperparameter mismatch"), rank["error"]
        for fragment in (
            "max_lr (checkpoint 0.1, configuration 0.0001)",
            "weight_decay (checkpoint 0.0, configuration 0.1)",
            "momentum (checkpoint 0.9, configuration 0.0)",
            "resume_mode='weights_only'",
        ):
            assert fragment in rank["error"], rank["error"]
        assert rank["unchanged"] and rank["rng_unchanged"]
    runs = [name for name in os.listdir(tmp_path / "runs") if not name.startswith(".") and name != "best"]
    assert runs == [first_id]


# --- #507: a resume failing before its first update is rolled back on every rank ---------------


@pytest.fixture(scope="module")
def first_split_run(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("ddp-rollback")
    launch("resume_first", out)
    return out


def test_a_resumes_step_count_refusal_hands_every_ranks_model_and_rng_back(first_split_run):
    """The step-count check runs inside the resume's rollback window: with
    unequal per-rank step counts every rank refuses, its weights and its
    Python, NumPy and torch streams as before the call."""
    launch("resume_steps_mismatch", first_split_run, timeout=180)
    for rank in results(first_split_run):
        assert rank["error"].startswith("DistributedFailure: "), rank["error"]
        assert "different numbers of steps" in rank["error"], rank["error"]
        assert rank["unchanged"] and rank["python_rng_unchanged"], rank
        assert rank["numpy_rng_unchanged"] and rank["rng_unchanged"], rank


def test_a_resume_whose_first_fetch_fails_on_one_rank_is_rolled_back_on_both(first_split_run, monkeypatch):
    """Rank 1's first fetch fails; rank 0 waits in its first step until the
    process group's timeout fails it. Neither has committed an update, so
    both hand their model and RNG back."""
    monkeypatch.setenv("NNX_DDP_TIMEOUT", "15")
    launch("resume_unreadable_rank", first_split_run, timeout=180)
    waiting, failing = results(first_split_run)
    assert failing["error"].startswith("OSError: cannot read sample"), failing
    assert waiting["error"] is not None and "OSError" not in waiting["error"], waiting  # the group's timeout
    for rank in (waiting, failing):
        assert rank["unchanged"] and rank["python_rng_unchanged"], rank
        assert rank["numpy_rng_unchanged"] and rank["rng_unchanged"], rank
