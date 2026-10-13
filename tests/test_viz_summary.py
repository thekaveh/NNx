"""Tests for nnx.viz.summary — Keras-style parameter table.

Thin wrapper around torchinfo.summary. Two contracts:
- Accepts an NNModel and unwraps to .net.
- Accepts an nn.Module directly (no NNModel detour required).
"""

from __future__ import annotations

import pytest

# Same optional-extra convention every other gated test file follows:
# skip gracefully when the [viz] extra isn't installed (the shipped
# sdist's suite must not hard-fail without it).
pytest.importorskip("torchinfo")

from nnx import (  # noqa: E402
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
)
from nnx.viz import summary


def _tiny() -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=4,
            output_dim=2,
            hidden_dims=[8],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def test_summary_returns_torchinfo_modelstatistics():
    from torchinfo.model_statistics import ModelStatistics

    m = _tiny()
    s = summary(m, input_size=(1, 4))
    assert isinstance(s, ModelStatistics)
    assert s.total_params > 0


def test_summary_accepts_nn_module_directly():
    m = _tiny()
    s = summary(m.net, input_size=(1, 4))
    assert s.total_params > 0


def test_summary_input_size_does_not_advance_global_rng():
    """torchinfo synthesizes the input_size= dummy via torch.rand — the
    wrapper must snapshot/restore RNG so a mid-pipeline summary probe
    (the concepts.md idiom) doesn't shift a seeded run. The values are
    immaterial to the statistics, so restoring is contract-safe."""
    import torch

    m = _tiny()
    torch.manual_seed(11)
    state = torch.get_rng_state()
    summary(m, input_size=(1, 4))
    assert torch.equal(torch.get_rng_state(), state)


def test_a_summary_by_input_size_never_moves_a_cpu_model_on_a_cuda_host(monkeypatch):
    # torchinfo's own inference would pick CUDA for a CPU model; the dummy goes to the model's device.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = _tiny()
    summary(model, input_size=(2, 4))
    assert all(p.device.type == "cpu" for p in model.net.parameters())


def test_a_summary_by_input_size_builds_a_double_dummy_for_a_double_model():
    import torch

    net = torch.nn.Sequential(torch.nn.Linear(3, 4)).double()
    assert summary(net, input_size=(2, 3)) is not None
