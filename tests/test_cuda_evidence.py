"""FEAT-043: the CUDA paths, executed on CUDA.

These tests run only where CUDA is present (a GPU runner, or a maintainer's
GPU machine; see CONTRIBUTING, "GPU evidence"); elsewhere they skip, and a
skip is never evidence. Feature-composition rows TC-06, TC-10 and TC-11 list
them, so a run on the registry's ``cuda`` evidence profile verifies those
rows; the CPU profile renders them unverified."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    PrecisionPolicy,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

sys.path.insert(0, str(Path(__file__).resolve().parent))


class _ScalerUpdates:
    """Counts ``GradScaler.update`` calls and the committed ones among them.

    An update after an unskipped step advances the growth tracker (or, at
    the growth interval, grows the scale); one after a skipped step backs
    the scale off. A fresh, never-stepped scaler already carries the initial
    scale and a ``_growth_tracker`` of 0 (#571), so only a committed update
    is evidence that the scaler stepped the optimizer."""

    def __init__(self, monkeypatch) -> None:
        self.calls = self.committed = 0
        update = torch.amp.GradScaler.update

        def counting(scaler, *args, **kwargs):
            before = scaler.state_dict()
            update(scaler, *args, **kwargs)
            after = scaler.state_dict()
            self.calls += 1
            grew = after["scale"] > before["scale"]
            advanced = after["scale"] == before["scale"] and after["_growth_tracker"] > before["_growth_tracker"]
            self.committed += grew or advanced

        monkeypatch.setattr(torch.amp.GradScaler, "update", counting)


def _train(tmp_path, monkeypatch, precision: str):
    """Train on CUDA; returns the run, its optimizer-step count, whether any
    weight moved and the output dtypes of every training-mode ``Linear``
    forward (the observed autocast, not the precision record)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(
            input_dim=16, output_dim=4, hidden_dims=[64], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CUDA, loss=Losses.CROSS_ENTROPY, precision=PrecisionPolicy(precision)
        ),
    )
    initial = {name: value.detach().clone() for name, value in model.net.state_dict().items()}
    forward: set[torch.dtype] = set()

    def record(module, inputs, output) -> None:
        if module.training:
            forward.add(output.dtype)

    hooks = [
        module.register_forward_hook(record) for module in model.net.modules() if isinstance(module, torch.nn.Linear)
    ]
    assert hooks, "the evidence net has no Linear layer to observe"
    data = torch.Generator().manual_seed(1)
    loader = DataLoader(TensorDataset(torch.randn(256, 16, generator=data), torch.arange(256) % 4), batch_size=32)
    try:
        run = model.train(
            params=NNTrainParams(
                n_epochs=3,
                train_loader=loader,
                optim=NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.0, weight_decay=0.0),
                scheduler=NNSchedulerParams(min_lr=0.0, factor=0.5, patience=0, cooldown=0, threshold=0.0),
                overwrite_existing=True,
            )
        )
    finally:
        for hook in hooks:
            hook.remove()
    moved = any(not torch.equal(initial[name], value) for name, value in model.net.state_dict().items())
    return run, len(loader) * 3, moved, forward


def test_fp16_amp_trains_through_a_grad_scaler_on_cuda(tmp_path, monkeypatch):
    updates = _ScalerUpdates(monkeypatch)
    run, steps, moved, forward = _train(tmp_path, monkeypatch, "fp16")
    record = run.precision.record()
    assert (record["requested"], record["effective"]) == ("fp16", "fp16")
    assert record["autocast_dtype"] == "float16" and record["grad_scaler"] is True
    assert forward == {torch.float16}  # autocast ran every training forward in float16
    losses = [idp.train_edp.loss for idp in run.idps]
    assert losses and all(math.isfinite(loss) for loss in losses)
    assert updates.calls == steps  # one scaler update per optimizer step
    assert updates.committed > 0, "every scaler update backed off or changed nothing: no step was taken"
    assert moved  # the committed steps reached the weights
    scaler = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)["scaler"]
    assert scaler is not None and math.isfinite(scaler["scale"]) and scaler["scale"] > 0


def test_bf16_trains_without_a_scaler_on_cuda(tmp_path, monkeypatch):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("this CUDA device has no bf16 support")
    run, _, moved, forward = _train(tmp_path, monkeypatch, "bf16")
    record = run.precision.record()
    assert (record["requested"], record["effective"]) == ("bf16", "bf16")
    assert record["autocast_dtype"] == "bfloat16" and record["grad_scaler"] is False
    assert forward == {torch.bfloat16}  # autocast ran every training forward in bfloat16
    assert all(math.isfinite(idp.train_edp.loss) for idp in run.idps)
    assert moved
    assert NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)["scaler"] is None


@pytest.mark.xdist_group("torchrun")  # one torchrun launch at a time (FIX-031)
def test_two_gpu_nccl_parity_matches_the_single_process_reference(tmp_path, monkeypatch):
    """The FEAT-030 parity scenario on two GPUs over NCCL equals the
    single-process CPU reference within FP32 device tolerance."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    from test_ddp_adapter import launch, reference, results

    monkeypatch.setenv("NNX_DDP_DEVICE", "cuda")
    out = tmp_path / "parity-nccl"
    launch("parity", out)
    ranks = results(out)
    monkeypatch.delenv("NNX_DDP_DEVICE")
    ref_model, ref_run = reference(tmp_path)
    expected = {name: value.cpu() for name, value in ref_model.net.state_dict().items()}
    for rank in ranks:
        state = {name: value.cpu() for name, value in rank["state"].items()}
        assert set(state) == set(expected)
        for name in state:
            torch.testing.assert_close(state[name], expected[name], rtol=1e-4, atol=1e-5)
        assert len(rank["train_loss"]) == len(ref_run.idps)
        for got, want in zip(rank["train_loss"], [idp.train_edp.loss for idp in ref_run.idps], strict=True):
            assert got == pytest.approx(want, rel=1e-4, abs=1e-5)
