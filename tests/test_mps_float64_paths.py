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


def test_a_summary_by_input_size_leaves_an_mps_model_on_its_device():
    pytest.importorskip("torchinfo")
    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams
    from nnx.viz import summary

    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.MPS, loss=Losses.CROSS_ENTROPY),
    )
    summary(model, input_size=(2, 4))
    assert all(p.device.type == "mps" for p in model.net.parameters())


def test_compare_compile_diffs_an_mps_model_in_float64_on_the_cpu():
    from nnx import CompileSpec
    from nnx.benchmarking import compare_compile

    torch.manual_seed(0)
    net = torch.nn.Linear(4, 2).to("mps")
    comparison = compare_compile(
        net, torch.randn(3, 4, device="mps"), compile=CompileSpec(backend="aot_eager"), warmup=0, repeats=1
    )
    assert comparison.compiled.compile.effective == "compiled" and comparison.max_abs_diff == 0.0


def test_max_abs_diff_widens_every_mps_chunk_on_the_cpu(monkeypatch):
    import nnx.benchmarking as benchmarking

    monkeypatch.setattr(benchmarking, "_DIFF_CHUNK", 4)
    a = torch.arange(10, dtype=torch.float32, device="mps")
    b = a.clone()
    b[-1] += 2
    assert benchmarking._abs_diff(a, b) == 2.0
