"""Tests for nnx.finetune.loading — external pretrained-weight loading."""

from __future__ import annotations

import pytest
import torch

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams
from nnx.finetune import LoadPretrainedResult, load_pretrained


def _model() -> NNModel:
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


def test_load_pretrained_from_dict_loads_overlapping_keys():
    model = _model()
    src = {k: torch.zeros_like(v) for k, v in model.net.state_dict().items()}
    result = load_pretrained(model.net, src)

    assert isinstance(result, LoadPretrainedResult)
    assert set(result.loaded_keys) == set(model.net.state_dict().keys())
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    # Zeros actually loaded.
    for v in model.net.state_dict().values():
        assert torch.all(v == 0)


def test_load_pretrained_strict_false_tolerates_missing(tmp_path):
    model = _model()
    # Source has only ONE of the expected keys.
    src = {"layers.0.weight": torch.zeros(8, 4)}
    result = load_pretrained(model.net, src, strict=False)

    assert result.loaded_keys == ["layers.0.weight"]
    assert "layers.0.bias" in result.missing_keys
    assert "layers.1.weight" in result.missing_keys
    assert result.unexpected_keys == []


def test_load_pretrained_strict_true_raises_on_mismatch():
    model = _model()
    src = {"layers.0.weight": torch.zeros(8, 4)}
    with pytest.raises(RuntimeError, match="mismatches"):
        load_pretrained(model.net, src, strict=True)


def test_load_pretrained_unexpected_keys_reported():
    model = _model()
    src = {k: torch.zeros_like(v) for k, v in model.net.state_dict().items()}
    src["foreign.extra"] = torch.zeros(3)
    result = load_pretrained(model.net, src)
    assert "foreign.extra" in result.unexpected_keys


def test_load_pretrained_key_map_remaps_prefix():
    """A common torchvision idiom: 'backbone.conv1.weight' → 'features.0.weight'."""
    model = _model()
    src = {
        "foreign.layers.0.weight": torch.ones(8, 4),
        "foreign.layers.0.bias": torch.ones(8),
        "foreign.layers.1.weight": torch.ones(2, 8),
        "foreign.layers.1.bias": torch.ones(2),
    }
    result = load_pretrained(model.net, src, key_map={"foreign.": ""})

    # All four keys remapped + applied.
    assert len(result.loaded_keys) == 4
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    # Weights are now 1s (from the source).
    assert torch.all(model.net.layers[0].weight == 1.0)


def test_load_pretrained_prefix_strip():
    model = _model()
    src = {
        "model.layers.0.weight": torch.full((8, 4), 7.0),
        "model.layers.0.bias": torch.zeros(8),
        "model.layers.1.weight": torch.zeros(2, 8),
        "model.layers.1.bias": torch.zeros(2),
    }
    result = load_pretrained(model.net, src, prefix="model.")
    assert len(result.loaded_keys) == 4
    assert torch.all(model.net.layers[0].weight == 7.0)


def test_load_pretrained_from_path_round_trip(tmp_path):
    """Export → load: same weights flow back in via the file-path source."""
    model_a = _model()
    pt_path = tmp_path / "weights.pt"
    torch.save(model_a.net.state_dict(), pt_path)

    model_b = _model()
    # Pre-load: parameters differ.
    assert not torch.equal(
        model_a.net.layers[0].weight,
        model_b.net.layers[0].weight,
    )

    result = load_pretrained(model_b.net, pt_path)
    assert result.missing_keys == [] and result.unexpected_keys == []
    # Post-load: every parameter matches.
    for (ka, va), (kb, vb) in zip(
        model_a.net.state_dict().items(),
        model_b.net.state_dict().items(),
        strict=True,
    ):
        assert ka == kb
        assert torch.equal(va, vb)


def test_load_pretrained_from_module_uses_state_dict():
    """Passing an nn.Module directly should equal passing its state_dict."""
    model_a = _model()
    model_b = _model()
    result = load_pretrained(model_b.net, model_a.net)
    assert result.missing_keys == [] and result.unexpected_keys == []
    for va, vb in zip(
        model_a.net.state_dict().values(),
        model_b.net.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(va, vb)


def test_load_pretrained_rejects_bad_source_type():
    model = _model()
    with pytest.raises(TypeError, match="path, dict, or nn.Module"):
        load_pretrained(model.net, 12345)


def test_nnmodel_export_state_dict_round_trip(tmp_path):
    """NNModel.export_state_dict + load_pretrained = round-trip identity."""
    model_a = _model()
    out_path = model_a.export_state_dict(str(tmp_path / "exported.pt"))
    assert out_path.endswith("exported.pt")

    model_b = _model()
    load_pretrained(model_b.net, out_path)
    for va, vb in zip(
        model_a.net.state_dict().values(),
        model_b.net.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(va, vb)


def test_load_pretrained_rejects_remap_collisions():
    """Prefix-stripping can collapse two source keys onto one target
    ('model.x' and 'x' both -> 'x') — silently letting the later one
    win loads unpredictable weights; it must raise instead."""
    import pytest
    import torch
    from torch import nn

    from nnx.finetune import load_pretrained

    target = nn.Linear(2, 2)
    src = {
        "weight": torch.zeros(2, 2),
        "model.weight": torch.ones(2, 2),
        "model.bias": torch.zeros(2),
    }
    with pytest.raises(ValueError, match="collapses"):
        load_pretrained(target, src, prefix="model.")
