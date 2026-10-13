"""MPS has no float64: every float64 sum or copy NNx makes from a tensor that
may live on the model's device widens on the CPU. Each test covers one path
that crashed with "Cannot convert a MPS Tensor to float64 dtype"."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")


def test_low_rank_factorize_keeps_the_factors_on_mps():
    from nnx.surgery.low_rank import low_rank_factorize

    layer = torch.nn.Linear(8, 6).to("mps")
    factors = low_rank_factorize(layer, rank=6)
    x = torch.randn(3, 8, device="mps")
    assert all(p.device.type == "mps" for p in factors.parameters())
    torch.testing.assert_close(factors(x), layer(x), atol=1e-4, rtol=1e-4)


def test_a_regression_task_accumulates_on_mps():
    from nnx import TaskSpec
    from nnx.tasks import task_adapter

    adapter = task_adapter(TaskSpec.regression(1))
    accumulator = adapter.accumulator()
    output = torch.randn(4, 1, device="mps")
    target = torch.randn(4, 1, device="mps")
    accumulator.update(output, target, torch.ones(4, 1, dtype=torch.bool, device="mps"))
    assert accumulator.count == 4


def test_causal_lm_token_sums_on_mps():
    from nnx.lm_tasks import CausalLMTask

    task = CausalLMTask(vocab_size=11, alignment="pre_shifted")
    logits = torch.randn(2, 3, 11, device="mps")
    targets = torch.randint(0, 11, (2, 3), device="mps")
    _, nll, _, n = task.token_sums(
        logits.reshape(-1, 11), targets.reshape(-1), torch.ones(6, dtype=torch.bool, device="mps")
    )
    assert n == 6 and math.isfinite(nll)


def test_standardizer_fits_an_mps_tensor():
    from nnx.preprocessing import Standardizer

    data = torch.randn(10, 3, device="mps")
    fitted = Standardizer.fit(data, rows=range(8))
    reference = Standardizer.fit(data.cpu(), rows=range(8))
    assert list(fitted.mean) == list(reference.mean) and list(fitted.scale) == list(reference.scale)


def test_the_diffusion_schedule_fingerprint_is_device_independent():
    from nnx import NoiseSchedulers
    from nnx.diffusion.objective import _schedule_fingerprint

    schedule = NoiseSchedulers.LINEAR(10)
    assert _schedule_fingerprint(schedule.to("mps")) == _schedule_fingerprint(schedule)
