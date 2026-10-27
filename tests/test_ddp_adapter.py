"""FEAT-030: the DDP adapter matches a single-process global-batch reference.

Scenarios run under ``torchrun`` (two CPU ranks over Gloo) through
``tests/ddp_scenarios.py``; each rank saves what it saw and these tests
compare it with a reference computed here in one process.
"""

from __future__ import annotations

import datetime
import os
import re
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

# Every torchrun launch shares one worker (FIX-031): its module-scoped launch
# runs once, and no two 2-rank launches compete for a CI runner's 4 vCPUs.
pytestmark = pytest.mark.xdist_group("torchrun")


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
    try:
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as expired:  # say where it hung
        raise AssertionError(
            f"torchrun {scenario} timed out after {timeout:.0f} s:\n{_text(expired.stdout)[-4000:]}\n"
            f"{_text(expired.stderr)[-6000:]}{_rank_stacks(out)}"
        ) from None
    if check and completed.returncode != 0:
        raise AssertionError(
            f"torchrun {scenario} failed:\n{completed.stdout[-4000:]}\n{completed.stderr[-6000:]}{_rank_stacks(out)}"
        )
    return completed


def _rank_stacks(out: Path) -> str:
    """#478: the stacks each rank's faulthandler wrote (on an abort, or before
    the launch timeout), for a failure message."""
    dumps = [
        f"\n--- {path.name} ---\n{path.read_text(encoding='utf-8', errors='replace')[-6000:]}"
        for path in sorted(out.glob("faults-*.txt"))
        if path.stat().st_size
    ]
    return "\n\nrank stacks:" + "".join(dumps) if dumps else ""


def _text(output: object) -> str:
    """A timed-out run's partial output (bytes or str, or None)."""
    if isinstance(output, bytes):
        return output.decode("utf-8", "replace")
    return output or ""


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


def test_numpy_counts_are_stored_as_ints_with_the_plain_descriptor():
    """#544: NumPy integers (sweep grids, rng.integers seeds) are the plain
    ints they equal; bools, floats and strings stay refused."""
    from unittest import mock

    import numpy as np

    from nnx import distributed as nnx_dist

    plain = RankPartition(8, rank=1, world_size=2, seed=3)
    numpy = RankPartition(np.int64(8), rank=np.int64(1), world_size=np.int32(2), seed=np.int64(3))
    assert numpy.descriptor() == plain.descriptor() and numpy.indices() == plain.indices()
    assert all(type(getattr(numpy, field)) is int for field in ("n", "rank", "world_size", "seed"))
    numpy.set_epoch(np.int64(2))
    plain.set_epoch(2)
    assert type(numpy.epoch) is int and numpy.indices() == plain.indices()
    data = torch.utils.data.TensorDataset(torch.arange(8.0))
    sharded = nnx_dist.ShardedLoader(data, np.int64(2), RankPartition(8, rank=0, world_size=2))
    assert (
        type(sharded.batch_size) is int
        and sharded.descriptor() == nnx_dist.ShardedLoader(data, 2, RankPartition(8, rank=0, world_size=2)).descriptor()
    )
    with mock.patch.object(nnx_dist, "_world", return_value=(0, 2)):
        train = nnx_dist.train_loader(data, np.int64(2), seed=np.int64(5))
        assert train.descriptor() == nnx_dist.train_loader(data, 2, seed=5).descriptor()
        validation = nnx_dist.validation_loader(data, np.int32(3))
        assert validation.descriptor() == nnx_dist.validation_loader(data, 3).descriptor()
    assert type(train.batch_size) is int and type(train.partition.seed) is int and type(validation.batch_size) is int
    writer = nnx_dist.DDP(writer_rank=np.int64(1))
    assert writer == nnx_dist.DDP(writer_rank=1) and type(writer.writer_rank) is int
    refusing = (
        lambda v: RankPartition(v, rank=0, world_size=2),
        lambda v: RankPartition(8, rank=v, world_size=2),
        lambda v: RankPartition(8, rank=0, world_size=v),
        lambda v: RankPartition(8, rank=0, world_size=2, seed=v),
        lambda v: RankPartition(8, rank=0, world_size=2).set_epoch(v),
        lambda v: nnx_dist.ShardedLoader(data, v, RankPartition(8, rank=0, world_size=2)),
        lambda v: nnx_dist.DDP(writer_rank=v),
    )
    for bad in (True, 1.0, "1", np.float64(1.0)):
        for make in refusing:
            with pytest.raises(ValueError):
                make(bad)


def test_the_partition_seed_domain_is_torchs_generator_range(tmp_path, monkeypatch):
    """#544: seed + epoch overflowed torch's generator ("Overflow when
    unpacking long long") at epoch 1 for 2**64 - 1 and at epoch 0 for
    2**64, after the run had started. The largest seed now draws at every
    epoch, and a larger one is refused by name at construction."""
    from unittest import mock

    from nnx import distributed as nnx_dist

    largest = RankPartition(8, rank=0, world_size=2, seed=2**64 - 1)
    orders = []
    for epoch in (0, 1, 2):
        largest.set_epoch(epoch)
        orders.append(largest.global_order())
        assert sorted(orders[-1]) == list(range(8))
    assert orders[1] == RankPartition(8, rank=0, world_size=2, seed=0).global_order()  # (seed + 1) % 2**64
    for bad in (2**64, 2**65, -1):
        with pytest.raises(ValueError, match="seed"):
            RankPartition(8, rank=0, world_size=2, seed=bad)
    monkeypatch.chdir(tmp_path)
    data = torch.utils.data.TensorDataset(torch.arange(8.0))
    with mock.patch.object(nnx_dist, "_world", return_value=(0, 2)), pytest.raises(ValueError, match="seed"):
        nnx_dist.train_loader(data, 2, seed=2**64)
    assert not (tmp_path / "runs").exists()
    doc = " ".join((nnx_dist.train_loader.__doc__ or "").split()) + " ".join((RankPartition.__doc__ or "").split())
    assert "[0, 2**64 - 1]" in doc and "(seed + epoch) % 2**64" in doc


def test_the_largest_partition_seed_trains_both_epochs_on_every_rank(tmp_path):
    """#544, the ticket's 2-rank script: seed=2**64 - 1 failed on every rank
    at epoch 1, after the run's checkpoints were written; 2**64 is refused
    on every rank before any run directory exists."""
    out = tmp_path / "largest-seed"
    launch("largest_seed", out)
    for result in results(out):
        assert result["epochs"] == [0, 1], result
        assert result["too_large"].startswith("ValueError: ") and "seed" in result["too_large"]
        assert result["runs_after_refusal"] == []


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
    assert [rank["live_ddp"] for rank in errors] == [0, 0]  # released on every rank (#478)


def test_a_fit_failing_before_the_loop_or_in_ddps_forward_releases_the_wrapper(tmp_path):
    # #539: counted in each rank's handler and when shutdown() called there
    # reaches destroy_process_group.
    out = tmp_path / "release"
    launch("release_on_failure", out)
    first, second = (rank["cases"] for rank in results(out))
    assert first["reservation"]["error"].startswith("FileExistsError")
    assert second["reservation"]["error"].startswith("DistributedFailure: reserving the run failed on rank 0")
    assert first["build"]["error"].startswith("DistributedFailure: building the DDP wrapper failed on rank 1")
    assert second["build"]["error"] == "RuntimeError: Reducer failed (injected on rank 1)"
    for label, what in (("nnrun", "NNRun"), ("compiling", "_compiling")):
        assert first[label]["error"] == f"RuntimeError: {what} failed (injected on rank 0)"
        assert second[label]["error"] == f"RuntimeError: {what} failed (injected on rank 1)"
    assert first["forward"]["error"].startswith("RuntimeError") and second["forward"]["error"].startswith(
        "RuntimeError"
    )
    for cases in (first, second):
        for label, case in cases.items():
            assert case["live_ddp"] == 0, label
            assert case["live_ddp_at_destroy"] == [0], label
        # the net's own forward frames keep their locals; only DDP's are cleared
        assert cases["forward"]["net_forward_locals"]
        assert all(names for names in cases["forward"]["net_forward_locals"])


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
        "topology_callback",
        "writer_only_topology_callback",
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
    # #510: a topology-changing callback is refused on every rank, wrapped in
    # writer_only or not; the writer names the built class, the others the writer.
    topology = (
        "ValueError: callbacks ['TopologyChanging'] change model.net's topology, which distributed= does not cover"
    )
    assert first["errors"]["topology_callback"] == second["errors"]["topology_callback"] == topology
    assert first["errors"]["writer_only_topology_callback"].startswith(
        "ValueError: writer_only built TopologyChanging, which changes model.net's topology"
    )
    assert second["errors"]["writer_only_topology_callback"].startswith(
        "DistributedFailure: the distributed preflight failed on rank 0: ValueError: writer_only built TopologyChanging"
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


@pytest.fixture(scope="module")
def stock_forward_masked_rank(tmp_path_factory) -> list[dict]:
    out = tmp_path_factory.mktemp("ddp-stock-forward")
    launch("stock_forward_masked_rank", out)
    return results(out)


@pytest.mark.parametrize("loss", list(scenarios.STOCK_FORWARD_LOSSES))
def test_a_stock_forward_subclass_masked_on_one_rank_trains_like_the_union_batch(
    stock_forward_masked_rank, loss, tmp_path
):
    """#512: rank 0's batch is unmasked, rank 1's has one NaN target. A
    stock-forward ``MSELoss`` subclass weighted rank 0 by its rows, so the
    update was 2.679e-02 off the union batch's and the recorded loss
    0.783549; both now match one process on the union batch (0.981019)."""
    data = scenarios.stock_forward_rows()
    reference = scenarios.stock_forward_model(scenarios.STOCK_FORWARD_LOSSES[loss]())
    previous = os.getcwd()
    os.chdir(tmp_path)
    try:
        ref_run = reference.train(
            params=scenarios.params(
                GlobalBatches(data, world_size=2, batch_size=4, seed=0, shuffle=False),
                accumulate=1,
                n_epochs=1,
                data_id=f"stock-forward-{loss}-reference",
            )
        )
    finally:
        os.chdir(previous)
    assert [idp.train_edp.loss for idp in ref_run.idps] == [pytest.approx(0.981019, abs=1e-6)]
    for rank in stock_forward_masked_rank:
        got = rank["cases"][loss]
        assert_close_states(got["state"], reference.net.state_dict())
        assert_close_lists(got["train_loss"], [idp.train_edp.loss for idp in ref_run.idps])


@pytest.fixture(scope="module")
def sum_masked_windows(tmp_path_factory) -> list[dict]:
    out = tmp_path_factory.mktemp("ddp-sum-masked")
    launch("sum_masked_windows", out)
    return results(out)


def sum_task_reference(tmp_path: Path, loss: str, case: str) -> tuple[torch.nn.Module, dict]:
    """One process training on the union batches of the same case."""
    data = scenarios.sum_task_rows(loss, scenarios.SUM_MASK_CASES[case][1])
    model = scenarios.sum_task_model(loss)
    previous = os.getcwd()
    os.chdir(tmp_path)
    try:
        loader = GlobalBatches(data, world_size=2, batch_size=2, seed=0, shuffle=False)
        with scenarios.CountedSteps() as steps:
            run = model.train(params=scenarios.sum_task_params(loader, loss, case))
        return model.net, scenarios.sum_task_summary(run, steps.count)
    finally:
        os.chdir(previous)


@pytest.mark.parametrize("case", list(scenarios.SUM_MASK_CASES))
@pytest.mark.parametrize("loss", scenarios.SUM_LOSSES)
def test_a_sum_loss_window_masked_on_one_rank_trains_like_the_union_batch(sum_masked_windows, loss, case, tmp_path):
    """#503: a sum-reduction task loss whose window is all-masked on one rank
    (the writer or not) commits the global batch's update on every rank, and
    records the global batch's summed loss; a mean task loss and a task-free
    sum loss are unchanged."""
    ref_net, ref = sum_task_reference(tmp_path, loss, case)
    for rank in sum_masked_windows:
        got = rank["cases"][loss, case]
        assert got["steps"] == ref["steps"]
        assert_close_states(got["state"], ref_net.state_dict())
        assert_close_lists(got["train_loss"], ref["train_loss"])
        assert_close_lists(got["epoch_loss"], ref["epoch_loss"])
        assert got["status"] == ref["status"]
        assert (got["best"] is None) == (ref["best"] is None)
        if ref["best"] is not None:
            assert_close_states(got["best"], ref["best"])
    first, second = (rank["cases"][loss, case] for rank in sum_masked_windows)
    assert first["train_loss"] == second["train_loss"] and first["epoch_loss"] == second["epoch_loss"]
    assert_close_states(first["state"], second["state"])


@pytest.mark.parametrize("case", ["writer_window_acc2", "nonwriter_window_acc2"])
@pytest.mark.parametrize("loss", ["cross_entropy", "nll", "mse", "bce"])
def test_a_sum_loss_epoch_with_one_rank_window_masked_has_a_loss_and_a_best(sum_masked_windows, loss, case):
    for rank in sum_masked_windows:
        got = rank["cases"][loss, case]
        assert all(value is not None for value in got["train_loss"])
        assert all(value is not None for value in got["epoch_loss"]) and len(got["epoch_loss"]) == 2
        assert got["status"] == ["ok", "ok"]
        assert got["best"] is not None


@pytest.mark.parametrize(
    ("case", "skipped", "steps"),
    [("every_rank_acc1", [0, 3], 4), ("every_rank_acc2", [0, 1, 3, 4], 2)],
)
@pytest.mark.parametrize("loss", ["cross_entropy", "nll", "mse", "bce"])
def test_a_sum_loss_window_masked_on_every_rank_is_still_skipped(sum_masked_windows, loss, case, skipped, steps):
    """No rank has a valid target in the window: no update and no loss on any rank."""
    for rank in sum_masked_windows:
        got = rank["cases"][loss, case]
        assert got["steps"] == steps
        assert [index for index, value in enumerate(got["train_loss"]) if value is None] == skipped


def test_a_failing_writer_callback_stops_every_rank_without_waiting_for_the_timeout(tmp_path):
    out = tmp_path / "writer-callback"
    launch("writer_callback_fails", out)
    first, second = results(out)
    assert first["error"].startswith("OSError: disk full")
    assert second["error"].startswith("DistributedFailure") and "rank 0" in second["error"]
    assert second["seconds"] < 30  # agreed, not a 60 s collective timeout


def test_a_writer_only_failure_in_a_hook_listener_or_attempt_start_stops_every_rank(tmp_path, monkeypatch):
    """#540: a writer-only callback failing in on_train_begin, in on_train_end
    or in the update listener it registered, and the writer failing to write
    provenance.json, are agreed: the writer raises its own error, the other
    rank a DistributedFailure naming it, both well before the group's 30 s
    timeout, and then every rank leaves together through shutdown()."""
    monkeypatch.setenv("NNX_DDP_TIMEOUT", "30")
    out = tmp_path / "writer-hooks"
    launch("writer_hooks_fail", out)
    first, second = (rank["cases"] for rank in results(out))
    agreed = {
        "on_train_begin": "the callbacks' on_train_begin",
        "on_train_end": "the callbacks' on_train_end",
        "listener": "epoch 0's update listeners",
        "attempt": "starting the provenance attempt",
    }
    for hook, what in agreed.items():
        own = f"OSError: {scenarios.disk_full(hook)}"
        assert first[hook]["error"] == own, first[hook]
        assert second[hook]["error"] == f"DistributedFailure: {what} failed on rank 0: {own}", second[hook]
        for case in (first[hook], second[hook]):
            assert case["seconds"] < 15, (hook, case)  # agreed, not the 30 s collective timeout
            assert case["shutdown"] is None, (hook, case)


@pytest.mark.parametrize("hook", ["on_train_begin", "on_train_end", "listener", "attempt"])
def test_without_distributed_a_failing_hook_or_attempt_start_raises_as_before(hook, tmp_path, monkeypatch):
    """#540 leaves the single-process path alone: the failure is the
    caller's error, raised where it happens."""
    from nnx.provenance import ExperimentManifest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    data = scenarios.train_set()
    undo = scenarios.unwritable_provenance() if hook == "attempt" else (lambda: None)
    try:
        with pytest.raises(OSError, match=re.escape(scenarios.disk_full(hook))):
            scenarios.model().train(
                params=scenarios.params([(data.X, data.y)], n_epochs=1),
                callbacks=[] if hook == "attempt" else [scenarios.FailingWriterHook(hook)],
                provenance=ExperimentManifest(config={"case": hook}) if hook == "attempt" else None,
            )
    finally:
        undo()


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


@pytest.mark.parametrize("backend", [None, "gloo"])
def test_init_process_group_selects_cuda_local_rank_only_under_nccl(backend, monkeypatch):
    """#540, mocked (no GPU needed; real-GPU evidence stays with #420): rank 1
    of 2 on a CUDA host takes cuda:1 under the default nccl, and no device
    under gloo, as the docstring says."""
    from unittest import mock

    import torch.distributed as dist

    from nnx import distributed as nnx_dist

    for name, value in dict(RANK="1", LOCAL_RANK="1", WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT="0").items():
        monkeypatch.setenv(name, value)
    calls: list[str] = []
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.set_device", side_effect=lambda index: calls.append(f"set_device({index})")),
        mock.patch.object(dist, "is_initialized", return_value=False),
        mock.patch.object(dist, "init_process_group", side_effect=lambda backend, timeout: calls.append(backend)),
        mock.patch.object(dist, "get_rank", return_value=1),
        mock.patch.object(dist, "get_world_size", return_value=2),
    ):
        assert nnx_dist.init_process_group(backend) == (1, 2)
    assert calls == (["set_device(1)", "nccl"] if backend is None else ["gloo"])
    doc = " ".join((nnx_dist.init_process_group.__doc__ or "").split())
    assert "Under ``nccl`` each process takes ``cuda:LOCAL_RANK``" in doc
    assert "any other backend selects no device" in doc and "torch.cuda.set_device(local_rank)" in doc


@pytest.mark.parametrize("backend", ["nccl", "gloo"])
def test_the_cuda_device_refusal_gives_the_advice_of_the_backend_in_use(backend, monkeypatch):
    """#540, mocked: a CUDA model on cuda:0 on LOCAL_RANK 1 is refused, with
    advice that fixes it under the backend in use (gloo selects no device)."""
    from types import SimpleNamespace
    from unittest import mock

    from nnx.nn.nn_model import _distributed_scope

    monkeypatch.setenv("LOCAL_RANK", "1")
    model = SimpleNamespace(net=torch.nn.Linear(2, 2), device=torch.device("cuda", 0))
    with (
        mock.patch("torch.distributed.get_backend", return_value=backend),
        pytest.raises(ValueError, match=r"LOCAL_RANK=1\) holds its model on cuda:0") as refused,
    ):
        _distributed_scope(
            model,
            None,
            None,
            train_step_fn=None,
            eval_step_fn=None,
            objective=None,
            precision=SimpleNamespace(effective="fp32"),  # type: ignore[arg-type]
            history=None,
            compile=None,
            writer=False,
        )
    if backend == "nccl":
        assert "build the model after nnx.distributed.init_process_group()" in str(refused.value)
    else:
        assert "selects no device under gloo: call torch.cuda.set_device(1)" in str(refused.value)


def test_init_process_group_checks_its_timeout_before_the_environment(monkeypatch):
    """The same timeout domain as shutdown(), checked first: a bool or a
    non-finite value never reaches the process group."""
    from nnx import distributed as nnx_dist

    for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    for bad in (0, -1.0, float("inf"), float("nan"), True, "5"):
        with pytest.raises(ValueError, match="timeout_seconds"):
            nnx_dist.init_process_group(timeout_seconds=bad)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no torchrun environment"):
        nnx_dist.init_process_group(timeout_seconds=5)


def test_timeouts_take_numpy_numbers_and_refuse_what_timedelta_cannot_hold(monkeypatch):
    """#544: np.float32(60) / np.int64(60) are timeouts; 10**400 (an
    OverflowError), 1e300 (accepted, then datetime.timedelta's
    OverflowError) and anything past the cap torch's nanosecond clock holds
    are refused by name before the environment is read."""
    from unittest import mock

    import numpy as np
    import torch.distributed as dist

    from nnx import distributed as nnx_dist

    for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    for good in (np.float32(60), np.int64(60), nnx_dist._MAX_TIMEOUT_SECONDS):
        nnx_dist.shutdown(timeout_seconds=good)
        with pytest.raises(RuntimeError, match="no torchrun environment"):
            nnx_dist.init_process_group(timeout_seconds=good)
    for bad in (10**400, -(10**400), 1e300, 999_999_999 * 86_400, nnx_dist._MAX_TIMEOUT_SECONDS + 1):
        with pytest.raises(ValueError, match="timeout_seconds"):
            nnx_dist.shutdown(timeout_seconds=bad)
        with pytest.raises(ValueError, match="timeout_seconds"):
            nnx_dist.init_process_group(timeout_seconds=bad)
    for name, value in dict(RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="0").items():
        monkeypatch.setenv(name, value)
    timeouts: list = []
    with (
        mock.patch.object(dist, "is_initialized", return_value=False),
        mock.patch.object(dist, "init_process_group", side_effect=lambda backend, timeout: timeouts.append(timeout)),
        mock.patch.object(dist, "get_rank", return_value=0),
        mock.patch.object(dist, "get_world_size", return_value=1),
    ):
        nnx_dist.init_process_group("gloo", timeout_seconds=np.float32(60))
    assert timeouts == [datetime.timedelta(seconds=60)]


def test_shutdown_leaves_the_group_together_and_a_second_call_is_a_no_op(tmp_path):
    out = tmp_path / "shutdown"
    launch("shutdown", out)
    for result in results(out):
        assert result["initialized_after"] is False and result["second_call"] == "no-op"


def test_the_longest_timeout_forms_a_real_gloo_group(tmp_path, monkeypatch):
    """#544: the largest accepted timeout_seconds forms a Gloo group and
    leaves it; longer ones overflowed torch's int64-nanosecond deadlines
    (a DistNetworkError on macOS, an at-once barrier timeout on Linux),
    naming no field."""
    from nnx import distributed as nnx_dist

    monkeypatch.setenv("NNX_DDP_TIMEOUT", str(nnx_dist._MAX_TIMEOUT_SECONDS))
    out = tmp_path / "longest-timeout"
    launch("shutdown", out, timeout=120)
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
    """FIX-028: ``ExperimentPlan.fit``, ``nnx.search.search`` and
    ``Trainer.train`` train in one process; under a two-rank torchrun each
    refuses on every rank before any factory call, study or run exists."""
    out = tmp_path / "entries"
    launch("single_process_entries", out)
    for result in results(out):
        for entry in ("fit", "search", "trainer", "plain"):
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
    assert result["errors"]["trainer"] is None and result["trainer_run"] in result["runs"]
    assert result["errors"]["plain"] is None and result["plain_run"] in result["runs"]
    if importlib.util.find_spec("optuna") is not None:
        assert result["errors"]["search"] is None and result["search_states"] == ["completed"]
        assert result["study_written"]
    else:  # without the optuna extra the search fails on its import, not on the guard
        assert "optuna" in result["errors"]["search"] and "single process" not in result["errors"]["search"]


def test_a_hung_scenario_reports_where_each_rank_was(tmp_path, monkeypatch):
    """#478: a scenario that hangs or aborts says where each rank was. Every
    rank's faulthandler dumps its stacks before the launch timeout, and
    launch() attaches them to the failure."""
    # The dump is timed from the rank's start; torchrun startup and imports
    # take several seconds on a CI runner, so the launch allows ample time.
    monkeypatch.setenv("NNX_DDP_DUMP_AFTER", "3")
    monkeypatch.setenv("NNX_DDP_PARK_SECONDS", "40")
    monkeypatch.setenv("NNX_DDP_TIMEOUT", "35")  # the ranks also end on their own afterwards
    with pytest.raises(AssertionError) as caught:
        launch("parked", tmp_path / "parked", timeout=30)
    message = str(caught.value)
    assert "timed out after 30 s" in message and "rank stacks:" in message
    assert "_parked_for_diagnostics" in message  # the parked rank's own frame
    assert message.count("joined the group: parked") == 2  # each rank's last phase


def test_a_batch_one_rank_cannot_score_stops_every_rank(tmp_path):
    out = tmp_path / "unscorable"
    launch("unscorable", out)
    errors = results(out)
    assert [rank["error"] for rank in errors] == ["ValueError", "ValueError"]
    messages = [rank["message"] for rank in errors]
    assert _DDP_DEFAULT_STEP_REFUSAL in messages  # the refusing rank
    assert any("cannot score their batch's targets" in message for message in messages)  # the other rank


_DECLARE_TASK = "declare the task, e.g. NNModelParams(task=TaskSpec.regression(...))"
# distributed= refuses train_step_fn / objective / eval_step_fn: a task is the one remedy (#565).
_DDP_DEFAULT_STEP_REFUSAL = f"the default steps score classification and cannot score MSELoss targets: {_DECLARE_TASK}"


def test_distributed_regression_refusals_advise_only_a_task_declaration(tmp_path):
    """#565: under DDP the default training step and validation score
    classification without a task, and custom steps are refused, so every
    rank's refusal names only the task declaration — which then trains."""
    out = tmp_path / "unscorable_validation"
    launch("unscorable_validation", out)
    for rank in results(out):
        assert rank["messages"] == {
            "train": _DDP_DEFAULT_STEP_REFUSAL,
            "validation": "distributed validation without a task scores classification and cannot score MSELoss "
            f"targets: {_DECLARE_TASK}",
            "task": None,
        }
