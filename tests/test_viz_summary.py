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


def test_a_summary_by_input_size_restores_an_mps_model_s_generator(monkeypatch):
    """FEAT-050: a CPU pin of the MPS branch. The weight reports device mps,
    the MPS generator is fake, and torchinfo (stubbed: a CPU-only host cannot
    build the dummy on MPS) draws from it as its MPS dummy would."""
    import torch
    import torchinfo

    from tests._mps import fake_mps_generator, reports_mps

    mps = fake_mps_generator(monkeypatch)
    seen = []

    def stub(model, **kwargs):
        seen.append(kwargs["device"])
        mps.draw()
        return "statistics"

    monkeypatch.setattr(torchinfo, "summary", stub)
    net = torch.nn.Linear(4, 2)
    net.weight = torch.nn.Parameter(reports_mps(net.weight.detach().clone()))
    before = mps.state.clone()
    assert summary(net, input_size=(2, 4)) == "statistics"
    assert seen == [torch.device("mps")] and torch.equal(mps.state, before)


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


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
def test_summary_places_batch_and_input_data_on_the_module_device(monkeypatch, keyword):
    """#556: summary(batch=) gave torchinfo the loader's CPU batch, which an MPS or CUDA
    module refused. A meta-device module stands in for the accelerator, and a spy on
    torchinfo's summary records where the inputs it is given live."""
    import torch
    import torchinfo

    from nnx.models import KeywordInputs

    class Keyword(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, *, features):
            return self.linear(features)

    seen = []

    def spy(model, *, input_data=None, **kwargs):
        seen.append((input_data, {name: value for name, value in kwargs.items() if isinstance(value, torch.Tensor)}))
        return "statistics"

    monkeypatch.setattr(torchinfo, "summary", spy)
    if keyword:
        model = NNModel(
            module=Keyword(), params=NNModelParams(), batch_adapter=KeywordInputs(("features",), target="labels")
        )
        batch = {"features": torch.randn(3, 4), "labels": torch.tensor([0, 1, 0])}
        given = {"features": torch.randn(3, 4)}
    else:
        model = _tiny()
        batch = (torch.randn(3, 4), torch.tensor([0, 1, 0]))
        given = [torch.randn(3, 4)]
    model.net.to("meta")
    assert summary(model, batch=batch) == "statistics"
    assert summary(model, input_data=given) == "statistics"
    assert summary(model.net, input_data=torch.randn(3, 4)) == "statistics"  # a bare tensor too
    tensors = []
    for input_data, keywords in seen:
        if isinstance(input_data, torch.Tensor):
            input_data = [input_data]
        tensors += [*(input_data.values() if isinstance(input_data, dict) else input_data), *keywords.values()]
    assert len(seen) == 3 and len(tensors) == 3 and all(t.device.type == "meta" for t in tensors)
    assert isinstance(seen[0][0], dict) is keyword and isinstance(seen[1][0], dict) is keyword


def test_summary_places_mixed_adapter_keywords_and_nested_namedtuples_on_the_module_device(monkeypatch):
    """#556: an adapter that returns positional and keyword inputs leaves the keywords
    in torchinfo's kwargs, and a namedtuple nested in input_data keeps its type; both
    must reach the module's device too."""
    from collections import namedtuple

    import torch
    import torchinfo

    from nnx.models import BatchAdapter

    class Scaled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, x, *, scale):
            return self.linear(x) * scale

    class Mixed(BatchAdapter):
        def split(self, batch):
            x, scale, y = batch
            return (x,), {"scale": scale}, y

    seen = []

    def spy(model, *, input_data=None, **kwargs):
        seen.append((input_data, kwargs))
        return "statistics"

    monkeypatch.setattr(torchinfo, "summary", spy)
    model = NNModel(module=Scaled(), params=NNModelParams(), batch_adapter=Mixed())
    model.net.to("meta")
    assert summary(model, batch=(torch.randn(3, 4), torch.ones(3, 1), torch.tensor([0, 1, 0]))) == "statistics"
    (x,), keywords = seen[0]
    assert x.device.type == "meta" and keywords["scale"].device.type == "meta"

    Pair = namedtuple("Pair", "a b")
    assert summary(model.net, input_data=[Pair(torch.randn(3, 4), torch.randn(3, 4))]) == "statistics"
    (pair,), _ = seen[1]
    assert type(pair) is Pair and all(t.device.type == "meta" for t in pair)
