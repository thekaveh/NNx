"""A Hub artifact whose config declares tensors its weights do not hold is refused before any network is allocated."""

from __future__ import annotations

import json

import pytest
import torch

from nnx import NNModel
from nnx.nn.enum import nets as nets_module
from tests.test_hub_mixin import _resign, _tiny_model  # skips this module when the hub extra is absent


def _declare_huge_feed_forward(directory) -> None:
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["net_params"]["input_dim"] = 100_000
    config["net_params"]["hidden_dims"] = "[100000]"  # the state stores the list's repr
    config_path.write_text(json.dumps(config, sort_keys=True, indent=2), encoding="utf-8")
    _resign(directory)


def test_a_config_declaring_huge_dimensions_is_refused_before_the_network_is_allocated(tmp_path, monkeypatch):
    _tiny_model().save_pretrained(str(tmp_path))
    _declare_huge_feed_forward(tmp_path)

    # The real constructor is a guard: a build on any real device fails here, before a 100k x 100k
    # tensor is ever allocated, so a regression fails fast instead of exhausting memory.
    constructed_on: list[str] = []
    real_feed_forward = nets_module.FeedFwdNN

    def guarded_feed_forward(params):
        device = torch.get_default_device().type
        constructed_on.append(device)
        if device != "meta":
            raise AssertionError("the network was constructed on a real device before the weights were checked")
        return real_feed_forward(params=params)

    monkeypatch.setattr(nets_module, "FeedFwdNN", guarded_feed_forward)
    with pytest.raises(RuntimeError, match="refused before the network was built") as caught:
        NNModel.from_pretrained(str(tmp_path))
    assert "layers.0.weight" in str(caught.value)
    assert constructed_on == ["meta"]


def test_a_mismatch_is_refused_before_the_network_is_built_on_a_real_device(tmp_path, monkeypatch):
    """A mismatch the config declares on a small net is named before the weights load."""
    _tiny_model().save_pretrained(str(tmp_path))
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["net_params"]["hidden_dims"] = "[9]"
    config_path.write_text(json.dumps(config, sort_keys=True, indent=2), encoding="utf-8")
    _resign(tmp_path)

    constructed_on: list[str] = []
    real_feed_forward = nets_module.FeedFwdNN

    def recording_feed_forward(params):
        constructed_on.append(torch.get_default_device().type)
        return real_feed_forward(params=params)

    monkeypatch.setattr(nets_module, "FeedFwdNN", recording_feed_forward)
    with pytest.raises(RuntimeError, match="size mismatch for layers.0.weight"):
        NNModel.from_pretrained(str(tmp_path))
    assert constructed_on == ["meta"]


def test_a_normal_save_and_load_round_trip_still_loads(tmp_path):
    model = _tiny_model()
    model.save_pretrained(str(tmp_path))
    restored = NNModel.from_pretrained(str(tmp_path))
    for key, value in model.net.state_dict().items():
        assert torch.equal(restored.net.state_dict()[key], value), key
