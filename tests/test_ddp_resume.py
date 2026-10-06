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


def test_canonical_keys_in_checkpoint_hub_and_export(writer):
    out, (first, second) = writer
    assert first["last_keys"] == second["last_keys"]
    assert not [key for key in first["last_keys"] if key.startswith("module.")]
    assert set(first["state"]) == set(first["last_keys"])
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
