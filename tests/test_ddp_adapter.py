"""FEAT-030: the DDP adapter matches a single-process global-batch reference.

Scenarios run under ``torchrun`` (two CPU ranks over Gloo) through
``tests/ddp_scenarios.py``; each rank saves what it saw and these tests
compare it with a reference computed here in one process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nnx import NNTrainParams
from nnx.distributed import RankPartition

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ddp_scenarios as scenarios  # noqa: E402

RTOL, ATOL = 1e-6, 1e-7

# One worker runs this module, so its module-scoped torchrun launch runs once (FIX-031).
pytestmark = pytest.mark.xdist_group("ddp_adapter")


def launch(
    scenario: str, out: Path, *, nproc: int = 2, check: bool = True, timeout: float = 240
) -> subprocess.CompletedProcess:
    """``torchrun --standalone --nproc_per_node=N tests/ddp_scenarios.py SCENARIO OUT``."""
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NNX_TQDM_DISABLE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    command = [sys.executable, "-m", "torch.distributed.run", *_single_node_flags(env), f"--nproc_per_node={nproc}"]
    command += [str(ROOT / "tests" / "ddp_scenarios.py"), scenario, str(out)]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
    if check and completed.returncode != 0:
        raise AssertionError(f"torchrun {scenario} failed:\n{completed.stdout[-4000:]}\n{completed.stderr[-6000:]}")
    return completed


def _single_node_flags(env: dict) -> list[str]:
    """``--standalone``; on macOS, whose hosts often resolve their own name
    to an unreachable address, a static single-node rendezvous on loopback
    instead (with Gloo on loopback too) — the same single node either way."""
    if sys.platform != "darwin":
        return ["--standalone"]
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    return ["--nnodes=1", "--master_addr=127.0.0.1", f"--master_port={port}"]


def results(out: Path, nproc: int = 2) -> list[dict]:
    return [torch.load(out / f"rank{rank}.pt", weights_only=False) for rank in range(nproc)]


class GlobalBatches:
    """The single-process reference: each global batch is the union of the
    ranks' batches (in rank order), from the same seeded partitions."""

    def __init__(self, dataset, *, world_size: int, batch_size: int, seed: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.partitions = [
            RankPartition(len(dataset), rank=rank, world_size=world_size, seed=seed, shuffle=shuffle)
            for rank in range(world_size)
        ]
        self.batch_size = batch_size

    def set_epoch(self, epoch: int) -> None:
        for partition in self.partitions:
            partition.set_epoch(epoch)

    def __len__(self) -> int:
        rows = len(self.partitions[0])
        return (rows + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        per_rank = [partition.indices() for partition in self.partitions]
        for start in range(0, len(per_rank[0]), self.batch_size):
            rows = [row for indices in per_rank for row in indices[start : start + self.batch_size]]
            yield self.dataset.X[rows], self.dataset.y[rows]


def reference(tmp_path: Path, *, world_size: int = 2, n_epochs: int = 2, val=None):
    data = scenarios.train_set()
    model = scenarios.model()
    previous = os.getcwd()
    os.chdir(tmp_path)
    try:
        val_set = scenarios.val_set() if val is None else val
        run = model.train(
            params=scenarios.params(
                GlobalBatches(data, world_size=world_size, batch_size=2, seed=3),
                [(val_set.X, val_set.y)],
                n_epochs=n_epochs,
            )
        )
    finally:
        os.chdir(previous)
    return model, run


def assert_close_states(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=RTOL, atol=ATOL)


def assert_close_lists(a, b) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        if x is None or y is None:
            assert x is None and y is None
        else:
            assert x == pytest.approx(y, rel=RTOL, abs=ATOL)


@pytest.fixture(scope="module")
def parity(tmp_path_factory) -> list[dict]:
    out = tmp_path_factory.mktemp("ddp-parity")
    launch("parity", out)
    return results(out)


def test_two_ranks_match_the_global_batch_reference(parity, tmp_path):
    ref_model, ref_run = reference(tmp_path)
    for rank in parity:
        # updates (accumulation windows of uneven valid counts) ...
        assert_close_states(rank["state"], ref_model.net.state_dict())
        # ... and globally weighted per-batch and validation records
        assert_close_lists(rank["train_loss"], [idp.train_edp.loss for idp in ref_run.idps])
        assert_close_lists(rank["train_accuracy"], [idp.train_edp.accuracy for idp in ref_run.idps])
        assert_close_lists(rank["val_loss"], [None if i.val_edp is None else i.val_edp.loss for i in ref_run.idps])
        assert_close_lists(
            rank["val_accuracy"], [None if i.val_edp is None else i.val_edp.accuracy for i in ref_run.idps]
        )


def test_every_rank_returns_the_same_run_and_records(parity, tmp_path):
    first, second = parity
    assert first["run_id"] == second["run_id"]
    for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy"):
        assert first[key] == second[key]  # identical, not just close
    assert_close_states(first["state"], second["state"])
    _, ref_run = reference(tmp_path)
    assert first["run_id"] == ref_run.id  # the run identity ignores the process layout


def test_validation_scores_every_row_exactly_once_unpadded(parity):
    ids = [row for rank in parity for row in rank["validation_ids"]]
    assert sorted(ids) == list(range(len(scenarios.val_set()))) and len(ids) == len(set(ids))
    assert [len(rank["validation_ids"]) for rank in parity] == [3, 2]  # unequal, never padded


def test_train_partitions_pad_or_drop_and_reseed_per_epoch():
    pad = [RankPartition(11, rank=r, world_size=2, policy="pad", seed=3) for r in range(2)]
    drop = [RankPartition(11, rank=r, world_size=2, policy="drop", seed=3) for r in range(2)]
    assert [len(p) for p in pad] == [6, 6] and [len(p) for p in drop] == [5, 5]
    assert sorted(set(pad[0].indices()) | set(pad[1].indices())) == list(range(11))
    assert len(set(drop[0].indices()) | set(drop[1].indices())) == 10
    epoch0 = pad[0].indices()
    pad[0].set_epoch(1)
    assert pad[0].indices() != epoch0  # seed + epoch
    pad[0].set_epoch(0)
    assert pad[0].indices() == epoch0
    assert pad[0].descriptor() == {
        "n": 11,
        "world_size": 2,
        "kind": "train",
        "policy": "pad",
        "shuffle": True,
        "seed": 3,
    }
    with pytest.raises(ValueError):
        RankPartition(11, rank=2, world_size=2)
    with pytest.raises(ValueError):
        RankPartition(11, rank=0, world_size=2, policy="wrap")


def test_one_rank_matches_the_local_path(tmp_path):
    out = tmp_path / "one"
    launch("local_equivalent", out, nproc=1)
    (single,) = results(out, 1)
    ref_model, ref_run = reference(tmp_path, world_size=1)
    assert_close_states(single["state"], ref_model.net.state_dict())
    assert single["train_loss"] == pytest.approx([idp.train_edp.loss for idp in ref_run.idps], rel=RTOL, abs=ATOL)
    assert single["run_id"] == ref_run.id


def test_a_non_finite_loss_on_one_rank_stops_every_rank(tmp_path):
    out = tmp_path / "nonfinite"
    launch("nonfinite", out)
    errors = results(out)
    assert [rank["error"] for rank in errors] == ["FloatingPointError", "FloatingPointError"]
    assert any("non-finite training loss on rank(s)" in rank["message"] for rank in errors)


def test_unequal_step_counts_fail_collectively_not_hang(tmp_path):
    out = tmp_path / "steps"
    completed = launch("steps_mismatch", out, timeout=180)
    assert completed.returncode == 0
    errors = results(out)
    assert [rank["error"] for rank in errors] == ["DistributedFailure", "DistributedFailure"]
    assert all("different numbers of steps" in rank["message"] for rank in errors)


def test_a_rank_without_validation_rows_reaches_the_same_stop_decision(tmp_path):
    out = tmp_path / "empty"
    launch("empty_validation_rank", out)
    first, second = results(out)
    assert second["validation_rows_here"] == 0
    assert first["epochs"] == second["epochs"] and len(first["epochs"]) < 5  # both stopped, at the same epoch
    assert first["val_loss"] == second["val_loss"]


def test_preflight_refusals_fail_on_every_rank_before_any_run(tmp_path):
    out = tmp_path / "preflight"
    launch("preflight", out)
    first, second = results(out)
    for label in (
        "undeclared_callback",
        "borrowed_tensorboard",
        "plain_loader",
        "partition_mismatch",
        "writer_component",
        "writer_only_component",
        "batchnorm",
        "compile",
        "history_journal",
    ):
        assert first["errors"][label] is not None and second["errors"][label] is not None, label
        assert (first["errors"][label] == "unavailable") == (second["errors"][label] == "unavailable"), label
    assert "declare its rank behaviour" in first["errors"]["undeclared_callback"]
    if first["errors"]["borrowed_tensorboard"] != "unavailable":  # needs the tensorboard extra
        assert "writer_only" in first["errors"]["borrowed_tensorboard"]
    assert "train_loader" in first["errors"]["plain_loader"]
    assert "component state" in first["errors"]["writer_component"]
    assert "component state" in first["errors"]["writer_only_component"]
    assert (
        "component state" in second["errors"]["writer_only_component"]
        or "rank 0" in second["errors"]["writer_only_component"]
    )
    assert "batch normalization" in first["errors"]["batchnorm"]
    for errors in (first["errors"], second["errors"]):
        assert errors["compile"] == "ValueError: distributed= and compile= cannot be combined"
        assert "HistoryJournal is not supported" in errors["history_journal"]
    assert (
        "partitions differ" in first["errors"]["partition_mismatch"]
        or "partition" in first["errors"]["partition_mismatch"]
    )
    assert first["runs_after_preflight"] == [] and second["runs_after_preflight"] == []


def test_ddp_scope_is_refused_without_a_process_group():
    from nnx.distributed import DDP

    model = scenarios.model()
    with pytest.raises(RuntimeError, match="torchrun"):
        model.train(
            params=NNTrainParams(n_epochs=1, train_loader=[], optim=scenarios.params([]).optim),
            distributed=DDP(),
        )
    with pytest.raises(TypeError):
        model.train(params=scenarios.params([]), distributed="ddp")  # type: ignore[arg-type]


def test_a_rank_batch_with_no_valid_target_trains_like_the_union_batch(tmp_path):
    """Rows 1 and 3 ignored: rank 1's first batch has no valid target, the
    global batch does — one process on the union batch trains normally."""
    out = tmp_path / "all-ignored"
    launch("all_ignored_rank", out)
    data = scenarios.Rows(8, seed=1, ignore=(1, 3))
    reference = scenarios.model()
    previous = os.getcwd()
    os.chdir(tmp_path)
    try:
        ref_run = reference.train(
            params=scenarios.params(
                GlobalBatches(data, world_size=2, batch_size=2, seed=0, shuffle=False), accumulate=1, n_epochs=1
            )
        )
    finally:
        os.chdir(previous)
    for rank in results(out):
        assert_close_states(rank["state"], reference.net.state_dict())
        assert_close_lists(rank["train_loss"], [idp.train_edp.loss for idp in ref_run.idps])


def test_a_global_window_with_no_valid_target_fails_on_every_rank(tmp_path):
    out = tmp_path / "all-ignored-window"
    launch("all_ignored_window", out)
    assert [rank["error"] for rank in results(out)] == ["FloatingPointError", "FloatingPointError"]


def test_a_failing_writer_callback_stops_every_rank_without_waiting_for_the_timeout(tmp_path):
    out = tmp_path / "writer-callback"
    launch("writer_callback_fails", out)
    first, second = results(out)
    assert first["error"].startswith("OSError: disk full")
    assert second["error"].startswith("DistributedFailure") and "rank 0" in second["error"]
    assert second["seconds"] < 30  # agreed, not a 60 s collective timeout


def test_the_replay_never_moves_the_real_loss():
    from nnx.nn.nn_model import _Replay

    model = scenarios.model()
    model.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.ones(scenarios.CLASSES))
    replay = _Replay(model)
    assert replay.loss_fn is not model.loss_fn and replay.loss_fn.weight is not model.loss_fn.weight


def test_shutdown_without_a_process_group_is_a_no_op():
    """FEAT-042: nothing to leave outside torchrun; the timeout is checked first."""
    from nnx import distributed as nnx_dist

    assert not torch.distributed.is_initialized()
    nnx_dist.shutdown()
    nnx_dist.shutdown(timeout_seconds=1)
    for bad in (0, -1.0, float("inf"), float("nan"), True, "5"):
        with pytest.raises(ValueError, match="timeout_seconds"):
            nnx_dist.shutdown(timeout_seconds=bad)  # type: ignore[arg-type]


def test_shutdown_leaves_the_group_together_and_a_second_call_is_a_no_op(tmp_path):
    out = tmp_path / "shutdown"
    launch("shutdown", out)
    for result in results(out):
        assert result["initialized_after"] is False and result["second_call"] == "no-op"


def test_shutdown_is_bounded_when_a_rank_never_arrives(tmp_path):
    """Rank 1 reaches the teardown only after rank 0's wait timed out: rank 0's
    shutdown raises within its timeout instead of blocking (leaving the group
    to process exit), and the late rank's barrier completes against the one
    rank 0 left queued, so it leaves cleanly; the launch neither hangs nor
    aborts."""
    import time

    out = tmp_path / "shutdown-missing"
    start = time.monotonic()
    completed = launch("shutdown_missing", out, check=False, timeout=120)
    assert time.monotonic() - start < 90
    assert completed.returncode == 0 and "terminate called" not in completed.stderr
    first, late = results(out)
    assert first["error"].startswith("RuntimeError: ") and "teardown barrier within 3s" in first["error"], first
    assert first["seconds"] < 6 and first["initialized_after"] is True
    assert late["error"] is None and late["seconds"] < 6 and late["initialized_after"] is False


def test_single_process_entry_points_refuse_a_multi_rank_group(tmp_path):
    """FIX-028: ``ExperimentPlan.fit`` and ``nnx.search.search`` train in one
    process; under a two-rank torchrun both refuse on every rank before any
    factory call, study or run exists."""
    out = tmp_path / "entries"
    launch("single_process_entries", out)
    for result in results(out):
        for entry in ("fit", "search"):
            assert result["errors"][entry].startswith("RuntimeError: "), result["errors"]
            assert "single process" in result["errors"][entry] and "world size 2" in result["errors"][entry]
        assert result["factory_calls"] == 0 and not result["study_written"] and result["runs"] == []


def test_single_process_entry_points_run_in_a_one_rank_group(tmp_path):
    """World size 1 under torchrun behaves as without a process group."""
    import importlib.util

    out = tmp_path / "entries-1"
    launch("single_process_entries", out, nproc=1)
    (result,) = results(out, nproc=1)
    assert result["errors"]["fit"] is None and result["fit_run"] in result["runs"]
    if importlib.util.find_spec("optuna") is not None:
        assert result["errors"]["search"] is None and result["search_states"] == ["completed"]
        assert result["study_written"]
    else:  # without the optuna extra the search fails on its import, not on the guard
        assert "optuna" in result["errors"]["search"] and "single process" not in result["errors"]["search"]
