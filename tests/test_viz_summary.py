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


def test_a_parameterless_model_stays_on_cpu_and_two_double_inputs_get_two_dummies(monkeypatch):
    import torch

    class Pair(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(3, 2)
            self.b = torch.nn.Linear(4, 2)

        def forward(self, x, y):
            return self.a(x) + self.b(y)

    assert summary(Pair().double(), input_size=[(2, 3), (2, 4)]) is not None  # one dtype per input
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    buffered = torch.nn.BatchNorm1d(3, affine=False)  # buffers only
    assert summary(buffered, input_size=(2, 3)) is not None
    assert summary(torch.nn.ReLU(), input_size=(2, 3)) is not None  # nothing to read a device from


def _dora_net():
    import torch

    from nnx.peft import apply_dora_to

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.ReLU(), torch.nn.Linear(6, 2))
    apply_dora_to(net, "0", r=2)
    return net


def test_summary_runs_on_a_dora_net():
    # torchinfo hooks every submodule, DoRA's base included; DoRA refuses a user's hook on
    # its base at forward (FIX-056), never the ones summary() installs for its one pass.
    import torch

    net = _dora_net()
    x = torch.randn(4, 6)
    assert summary(net, input_size=(4, 6)).total_params == 86
    assert summary(net, input_data=[x]).total_params == 86
    assert not net[0].base._forward_hooks and not net[0].base._forward_pre_hooks  # torchinfo removed its hooks


def test_a_user_hook_on_a_dora_base_is_still_refused_after_and_inside_summary():
    import torch

    net = _dora_net()
    x = torch.randn(4, 6)
    summary(net, input_size=(4, 6))
    handle = net[0].base.register_forward_hook(lambda m, i, o: o * 2)
    with pytest.raises(TypeError, match="forward hook, added after wrapping"):
        net(x)  # no exemption outlives summary()
    with pytest.raises(RuntimeError, match="torchinfo") as raised:
        summary(net, input_size=(4, 6))  # nor covers a hook that summary() did not install
    assert isinstance(raised.value.__cause__, TypeError)
    handle.remove()
    net(x)
