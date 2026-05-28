"""save_pretrained / from_pretrained round-trip via local directory.

Hub upload is exercised manually against the real Hub — tests only cover
the local serialization path so they run offline in CI. The on-disk
layout matches what PyTorchModelHubMixin produces (a flat directory
with `model.safetensors`, `config.json`, and a README), so a
``HfApi.upload_folder(local_dir)`` would push the same bits the test
asserts on.
"""

from __future__ import annotations

import json

import torch

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
)


def _tiny_model() -> NNModel:
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


def test_hub_save_pretrained_emits_safetensors_and_config(tmp_path):
    """save_pretrained writes the canonical PyTorchModelHubMixin layout —
    `model.safetensors` (weights) and `config.json` (NNParams + NNModelParams).
    A `README.md` is also emitted by the mixin's model-card generator;
    we only assert on the load-critical files.
    """
    m = _tiny_model()
    m.save_pretrained(str(tmp_path))
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "config.json").exists()


def test_hub_save_pretrained_config_contains_params(tmp_path):
    """config.json must carry the round-trippable form of both params
    dataclasses so `from_pretrained` can rebuild the model identically.
    Use the `state()` form (not asdict) — that's the public, stable,
    hash-grouping serialization NNRun uses on disk.
    """
    m = _tiny_model()
    m.save_pretrained(str(tmp_path))

    with open(tmp_path / "config.json") as f:
        cfg = json.load(f)
    assert "net_params" in cfg
    assert "params" in cfg
    assert cfg["net_params"] == m.net_params.state()
    assert cfg["params"] == m.params.state()


def test_hub_from_pretrained_round_trip(tmp_path):
    """A model loaded from a save_pretrained'd directory must have:
    - bit-exact weights in `.net`,
    - identical params + net_params state() dicts,
    - a working forward pass.
    """
    m = _tiny_model()
    m.save_pretrained(str(tmp_path))

    rt = NNModel.from_pretrained(str(tmp_path))

    for k in m.net.state_dict():
        assert torch.equal(m.net.state_dict()[k], rt.net.state_dict()[k])
    assert rt.params.state() == m.params.state()
    assert rt.net_params.state() == m.net_params.state()

    # Forward pass must work — exercises the rebuilt net + device + loss_fn.
    x = torch.randn(2, 4)
    with torch.no_grad():
        out_m = m.net(x)
        out_rt = rt.net(x)
    assert torch.equal(out_m, out_rt)


def test_hub_save_pretrained_overwrite(tmp_path):
    """A second save_pretrained into the same directory must rewrite,
    not error. Round-trip after overwrite still works.
    """
    m1 = _tiny_model()
    m1.save_pretrained(str(tmp_path))
    m2 = _tiny_model()  # fresh init → different weights
    m2.save_pretrained(str(tmp_path))

    rt = NNModel.from_pretrained(str(tmp_path))
    # Round-trip recovers the *second* save's weights, not the first.
    for k in m2.net.state_dict():
        assert torch.equal(m2.net.state_dict()[k], rt.net.state_dict()[k])


def test_hub_mixin_inheritance_is_visible():
    """NNModel must surface the PyTorchModelHubMixin API publicly — the
    three methods that distribution workflows depend on.
    """
    assert hasattr(NNModel, "save_pretrained")
    assert hasattr(NNModel, "from_pretrained")
    assert hasattr(NNModel, "push_to_hub")
