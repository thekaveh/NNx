"""FEAT-050: CPU pins of the MPS fixes, so a Linux lane fails when one is reverted.

MPS has no float64: NNx widens a tensor that may live on the model's device
on the CPU. On a CPU-only host the ``.cpu()`` hop is a no-op, and the MPS
tests in ``test_mps_float64_paths.py`` skip. Each test here hands the fixed
code an ``OffCPU`` tensor (``tests/_mps.py``), which refuses float64 and NumPy
as an MPS tensor does, so removing the hop raises the MPS ``TypeError`` on
any host. The MPS random-number-generator branches are pinned in
``test_compile_execution.py``, ``test_viz_summary.py``, ``test_lr_finder.py``
and ``test_seeding.py``.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tests._mps import FLOAT64_ERROR, mps_or_skip, off_cpu, reports_mps

ROOT = Path(__file__).resolve().parents[1]


def _gate_outcome() -> tuple[object, str]:
    # Caught here, not by pytest.raises: an unexpected skip would skip this test, not fail it.
    try:
        mps_or_skip()
    except BaseException as outcome:  # pytest's skip and fail outcomes are BaseExceptions
        return type(outcome), str(outcome)
    return None, ""


def test_an_mps_test_skips_without_mps_and_fails_under_nnx_require_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.delenv("NNX_REQUIRE_MPS", raising=False)
    kind, message = _gate_outcome()
    assert kind is pytest.skip.Exception and "needs an MPS device" in message
    monkeypatch.setenv("NNX_REQUIRE_MPS", "1")
    kind, message = _gate_outcome()
    assert kind is pytest.fail.Exception and "NNX_REQUIRE_MPS=1 and needs an MPS device" in message
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _gate_outcome() == (None, "")  # a device: the test runs


@pytest.mark.parametrize("required", [False, True], ids=["skips", "fails-under-NNX_REQUIRE_MPS"])
def test_the_mps_marker_skips_or_fails_a_collected_test_without_mps(required):
    """The conftest hook, end to end: a Linux lane (MPS reported unavailable)
    skips an ``mps`` test; the ``mps-evidence`` job's flag fails it."""
    node = "tests/test_mps_float64_paths.py::test_the_diffusion_schedule_fingerprint_is_device_independent"
    run = "import sys, pytest; sys.exit(pytest.main(['-q', '-rs', '-p', 'no:cacheprovider', '-o', 'addopts=', sys.argv[1]]))"
    env = {key: value for key, value in os.environ.items() if key != "NNX_REQUIRE_MPS"}
    if required:
        env["NNX_REQUIRE_MPS"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", "import torch; torch.backends.mps.is_available = lambda: False; " + run, node],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if required:
        assert result.returncode == 1 and "NNX_REQUIRE_MPS=1 and needs an MPS device" in result.stdout, result.stdout
    else:
        assert result.returncode == 0 and "1 skipped" in result.stdout, result.stdout


def test_every_mps_test_is_gated_by_the_marker():
    """A bare ``skipif`` on MPS would skip in the ``mps-evidence`` job too."""
    bare = re.compile(r"skipif\(\s*not\s+torch\.backends\.mps\.is_available\(\)")
    offenders = [
        path.name for path in sorted((ROOT / "tests").glob("*.py")) if bare.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
    marked = [
        path.name
        for path in sorted((ROOT / "tests").glob("test_*.py"))
        if "pytest.mark.mps" in path.read_text(encoding="utf-8")
    ]
    assert {"test_mps_float64_paths.py", "test_compile_execution.py", "test_ranking_task.py"} <= set(marked)


def test_the_stand_in_refuses_float64_and_numpy_until_it_reaches_the_cpu():
    tensor = off_cpu(torch.arange(4, dtype=torch.float32))
    for widen in (
        lambda: tensor.double(),
        lambda: tensor.to(torch.float64),
        lambda: tensor.sum(dtype=torch.float64),
        lambda: torch.ones(4, dtype=torch.float64)[tensor > 0],
    ):
        with pytest.raises(TypeError, match="MPS framework doesn't support float64"):
            widen()
    with pytest.raises(TypeError, match="numpy"):
        np.asarray(tensor)
    assert type(tensor.cpu()) is torch.Tensor and type(tensor.to("cpu")) is torch.Tensor
    assert tensor.detach().cpu().double().sum().item() == 6.0
    assert reports_mps(torch.zeros(1)).device.type == "mps"
    assert FLOAT64_ERROR.startswith("Cannot convert a MPS Tensor to float64 dtype")


@pytest.mark.parametrize("keep_arrays", [False, True])
def test_a_regression_task_accumulates_an_mps_batch_on_the_cpu(keep_arrays):
    from nnx import TaskSpec
    from nnx.tasks import task_adapter

    accumulator = task_adapter(TaskSpec.regression(1)).accumulator(keep_arrays=keep_arrays)
    output = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    target = torch.tensor([[1.5], [2.0], [2.0], [9.0]])
    valid = torch.tensor([[True], [True], [True], [False]])
    accumulator.update(off_cpu(output), off_cpu(target), off_cpu(valid))
    record = accumulator.result(loss=None)
    assert accumulator.count == 3
    assert record.metrics["mse"] == pytest.approx((0.25 + 0 + 1) / 3)
    assert record.metrics["mae"] == pytest.approx((0.5 + 0 + 1) / 3)


@pytest.mark.parametrize("with_ce", [True, False], ids=["train", "evaluate"])
def test_causal_lm_token_sums_widen_an_mps_nll_on_the_cpu(with_ce):
    from nnx.lm_tasks import CausalLMTask

    torch.manual_seed(0)
    task = CausalLMTask(vocab_size=11, alignment="pre_shifted")
    logits = torch.randn(6, 11)
    targets = torch.randint(0, 11, (6,))
    valid = torch.ones(6, dtype=torch.bool)
    _, nll, correct, n = task.token_sums(off_cpu(logits), off_cpu(targets), off_cpu(valid), with_ce=with_ce)
    _, expected, expected_correct, _ = task.token_sums(logits, targets, valid, with_ce=with_ce)
    assert n == 6 and math.isfinite(nll)
    assert nll == pytest.approx(expected, rel=1e-12) and correct == expected_correct


def test_a_standardizer_fits_an_mps_tensor_on_the_cpu():
    from nnx.preprocessing import Standardizer

    data = torch.randn(10, 3, generator=torch.Generator().manual_seed(0))
    fitted = Standardizer.fit(off_cpu(data), rows=range(8))
    reference = Standardizer.fit(data, rows=range(8))
    assert list(fitted.mean) == list(reference.mean) and list(fitted.scale) == list(reference.scale)


def test_the_diffusion_schedule_fingerprint_reads_mps_betas_on_the_cpu():
    import dataclasses

    from nnx import NoiseSchedulers
    from nnx.diffusion.objective import _schedule_fingerprint

    schedule = NoiseSchedulers.LINEAR(10)
    on_mps = dataclasses.replace(schedule, betas=off_cpu(schedule.betas))
    assert _schedule_fingerprint(on_mps) == _schedule_fingerprint(schedule)


def test_low_rank_factorize_decomposes_an_mps_weight_on_the_cpu(monkeypatch):
    # The hop is gated on device.type == "mps"; the factors are allocated on
    # the CPU here (a CPU-only host cannot allocate them on MPS).
    import nnx.surgery.low_rank as low_rank

    torch.manual_seed(0)
    reference = torch.nn.Linear(8, 6)
    layer = torch.nn.Linear(8, 6)
    layer.load_state_dict(reference.state_dict())
    layer.weight = torch.nn.Parameter(reports_mps(layer.weight.detach().clone()))
    allocate = low_rank._allocate_factors
    monkeypatch.setattr(low_rank, "_allocate_factors", lambda linear, rank: allocate(reference, rank))
    factors = low_rank.low_rank_factorize(layer, rank=6)
    x = torch.randn(3, 8)
    torch.testing.assert_close(factors(x), reference(x), atol=1e-5, rtol=1e-5)


def test_offline_distillation_sums_an_mps_student_on_the_cpu():
    from nnx.paradigms.offline_distillation import TeacherBatch, _Tally

    log_p = torch.log_softmax(torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.5, 0.5, 3.0]]), dim=-1)
    batch = TeacherBatch(
        inputs=torch.zeros(3, 1),
        teacher=torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]),
        target=torch.tensor([0, -1, 1]),
        labelled=torch.tensor([True, False, True]),
        sample_ids=("a", "b", "c"),
        schema_digest="digest",
        candidates=("x", "y", "z"),
    )
    on_mps, reference = _Tally(), _Tally()
    on_mps.update(off_cpu(log_p), batch)
    reference.update(log_p, batch)
    assert (on_mps.rows, on_mps.agree, on_mps.labelled, on_mps.correct) == (3, 3, 2, 1)
    assert on_mps.kl == pytest.approx(reference.kl, rel=1e-12) and on_mps.kl > 0
    assert on_mps.nll == pytest.approx(reference.nll, rel=1e-12) and on_mps.nll > 0
