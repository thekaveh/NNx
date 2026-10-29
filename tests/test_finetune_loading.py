"""Tests for nnx.finetune.loading — external pretrained-weight loading."""

from __future__ import annotations

import pytest
import torch
from torch import nn

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


def test_a_shape_mismatch_refuses_the_load_before_copying_anything():
    """A head of another width (1000 classes into 3) used to raise after the
    body had already been copied; now the module is untouched."""
    import torch
    from torch import nn

    from nnx.finetune import load_pretrained

    torch.manual_seed(0)
    source = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 10))
    target = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    before = {k: v.clone() for k, v in target.state_dict().items()}
    with pytest.raises(RuntimeError, match=r"different shapes.*2\.weight"):
        load_pretrained(target, source)
    assert all(torch.equal(before[k], v) for k, v in target.state_dict().items())
    head = {k: v for k, v in source.state_dict().items() if not k.startswith("2.")}
    assert load_pretrained(target, head).loaded_keys == ["0.bias", "0.weight"]


# --- uninitialized lazy layers (#528) ------------------------------------------

_LAZY_LAYERS = {
    "linear": (lambda: nn.LazyLinear(3), lambda: nn.Linear(4, 3)),
    "conv1d": (lambda: nn.LazyConv1d(3, 2), lambda: nn.Conv1d(4, 3, 2)),
    "conv2d": (lambda: nn.LazyConv2d(3, 2), lambda: nn.Conv2d(4, 3, 2)),
    "conv3d": (lambda: nn.LazyConv3d(3, 2), lambda: nn.Conv3d(4, 3, 2)),
    "batchnorm1d": (lambda: nn.LazyBatchNorm1d(), lambda: nn.BatchNorm1d(4)),
}


def _lazy_pair(kind: str) -> tuple[nn.Module, nn.Module]:
    """A target whose first layer is unforwarded-lazy, and a materialized
    source of the same topology with non-default buffers."""
    lazy, eager = _LAZY_LAYERS[kind]
    torch.manual_seed(0)
    source = nn.Sequential(eager(), nn.ReLU(), nn.Linear(3, 2))
    for buf in source[0].buffers():  # running stats, so a copy is visible
        if buf.is_floating_point():
            buf.uniform_(0.5, 1.5)
    return nn.Sequential(lazy(), nn.ReLU(), nn.Linear(3, 2)), source


def _assert_loaded_from(target: nn.Module, source: nn.Module) -> None:
    got, want = target.state_dict(), source.state_dict()
    assert list(got) == list(want)
    for key, value in want.items():
        assert not nn.parameter.is_lazy(got[key]), key
        assert torch.equal(got[key], value), key


@pytest.mark.filterwarnings("ignore:Lazy modules are a new feature")
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("source_kind", ["dict", "module", "path"])
@pytest.mark.parametrize("kind", list(_LAZY_LAYERS))
def test_load_pretrained_materializes_an_unforwarded_lazy_target(kind, source_kind, strict, tmp_path):
    target, source = _lazy_pair(kind)
    assert any(nn.parameter.is_lazy(v) for v in target.state_dict().values())
    if source_kind == "dict":
        src = source.state_dict()
    elif source_kind == "module":
        src = source
    else:
        src = tmp_path / "pretrained.pt"
        torch.save(source.state_dict(), src)

    result = load_pretrained(target, src, strict=strict)

    assert result.loaded_keys == sorted(source.state_dict())
    assert result.missing_keys == [] and result.unexpected_keys == []
    _assert_loaded_from(target, source)


@pytest.mark.filterwarnings("ignore:Lazy modules are a new feature")
def test_load_pretrained_materializes_a_lazy_runtime_module_net(tmp_path):
    """The documented NNModel(module=...) recipe: a lazy module wrapped and
    warm-started before any forward."""
    target, source = _lazy_pair("linear")
    path = tmp_path / "pretrained.pt"
    torch.save(source.state_dict(), path)
    model = NNModel(module=target, params=NNModelParams(loss=Losses.CROSS_ENTROPY))

    result = load_pretrained(model.net, path)

    assert result.loaded_keys == ["0.bias", "0.weight", "2.bias", "2.weight"]
    assert tuple(model.net[0].weight.shape) == (3, 4)
    _assert_loaded_from(model.net, source)


@pytest.mark.filterwarnings("ignore:Lazy modules are a new feature")
@pytest.mark.parametrize("lazy_side", ["target", "source"])
def test_a_lazy_tensor_matches_any_shape_but_a_known_mismatch_still_refuses(lazy_side):
    """An uninitialized lazy tensor on either side has no shape to compare;
    a head of another width on a known-shape key still refuses before any
    copy, and a lazy target stays lazy."""
    torch.manual_seed(0)
    first = {side: nn.LazyLinear(3) if side == lazy_side else nn.Linear(4, 3) for side in ("target", "source")}
    target = nn.Sequential(first["target"], nn.ReLU(), nn.Linear(3, 2))
    source = nn.Sequential(first["source"], nn.ReLU(), nn.Linear(3, 10))  # a 10-class head into 2
    before = {k: v.clone() for k, v in target.state_dict().items() if not nn.parameter.is_lazy(v)}

    with pytest.raises(RuntimeError, match=r"different shapes") as excinfo:
        load_pretrained(target, source)

    assert "2.weight" in str(excinfo.value) and "0.weight" not in str(excinfo.value)
    after = target.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    if lazy_side == "target":
        assert nn.parameter.is_lazy(target[0].weight) and nn.parameter.is_lazy(target[0].bias)


@pytest.mark.filterwarnings("ignore:Lazy modules are a new feature")
def test_load_pretrained_from_a_lazy_source_into_a_lazy_target():
    """Neither side has materialized the lazy layer: the other keys load and
    the lazy layer stays lazy until its first forward."""
    torch.manual_seed(0)
    source = nn.Sequential(nn.LazyLinear(3), nn.ReLU(), nn.Linear(3, 2))
    target = nn.Sequential(nn.LazyLinear(3), nn.ReLU(), nn.Linear(3, 2))

    result = load_pretrained(target, source, strict=True)

    assert result.loaded_keys == ["0.bias", "0.weight", "2.bias", "2.weight"]
    assert nn.parameter.is_lazy(target[0].weight) and nn.parameter.is_lazy(target[0].bias)
    assert torch.equal(target[2].weight, source[2].weight) and torch.equal(target[2].bias, source[2].bias)
    assert target(torch.randn(5, 4)).shape == (5, 2)
