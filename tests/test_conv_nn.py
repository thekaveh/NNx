"""ConvNN + Nets.CONV + NNConvParams (#89).

LeNet-style Conv→Pool→…→Flatten→FC classifier for small square images.
``NNConvParams`` subclasses ``NNParams`` (the NNTransformerParams/NNMoEParams
lift-via-subclassing pattern): ``conv_channels`` is required and ALWAYS in
``state()`` (the ``resolve_from_state`` discriminator + hash distinctness);
the LeNet-default knobs (kernel_size=5, stride=1, padding=0, pool_size=2,
in_channels=1) are omit-when-default.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNConvParams,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
)
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.net.conv_nn import ConvNN
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_params import NNParams


def _conv_params(**kw) -> NNConvParams:
    # 28×28 grayscale (MNIST shape) → LeNet-5 head sizes by default.
    base = dict(
        input_dim=784,
        output_dim=10,
        hidden_dims=[32],
        dropout_prob=0.0,
        activation=Activations.RELU,
        conv_channels=[6, 16],
    )
    base.update(kw)
    return NNConvParams(**base)


def _model(params: NNConvParams) -> NNModel:
    return NNModel(
        net_params=params,
        params=NNModelParams(net=Nets.CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


# ---------------------------------------------------------------------------
# Params: validation + state round-trip + hash distinctness
# ---------------------------------------------------------------------------


def test_conv_params_validation():
    with pytest.raises(ValueError, match="conv_channels"):
        _conv_params(conv_channels=[])
    with pytest.raises(ValueError, match="conv_channels"):
        _conv_params(conv_channels=[6, 0])
    with pytest.raises(ValueError, match="kernel_size"):
        _conv_params(kernel_size=0)
    with pytest.raises(ValueError, match="stride"):
        _conv_params(stride=0)
    with pytest.raises(ValueError, match="padding"):
        _conv_params(padding=-1)
    with pytest.raises(ValueError, match="pool_size"):
        _conv_params(pool_size=0)
    with pytest.raises(ValueError, match="in_channels"):
        _conv_params(in_channels=0)


def test_conv_params_requires_square_image():
    # input_dim / in_channels must be a perfect square (v1 square-image contract)
    with pytest.raises(ValueError, match="square"):
        _conv_params(input_dim=782)


def test_conv_params_rejects_spatial_collapse():
    """A stack whose conv/pool arithmetic shrinks the feature map below 1×1
    must fail at construction, not deep inside the first forward."""
    with pytest.raises(ValueError, match="spatial"):
        # 28 → conv5 → 24 → pool2 → 12 → conv5 → 8 → pool2 → 4 → conv5 fails (4 < 5)
        _conv_params(conv_channels=[6, 16, 32])


def test_conv_params_spatial_sizes_lenet():
    p = _conv_params()
    assert p.image_side() == 28
    # 28 → conv5 → 24 → pool2 → 12 → conv5 → 8 → pool2 → 4
    assert p.spatial_sizes() == [12, 4]
    assert p.flatten_dim() == 16 * 4 * 4


def test_conv_params_state_round_trip():
    p = _conv_params(conv_channels=[4, 8], kernel_size=3, padding=1, pool_size=2, in_channels=1)
    s = p.state()
    back = NNParams.resolve_from_state(s)
    assert isinstance(back, NNConvParams)
    assert back.conv_channels == [4, 8]
    assert back.kernel_size == 3
    assert back.padding == 1
    assert back.state() == s


def test_conv_state_differs_from_plain_config():
    """NNRun.id hashes net.state() — conv fields must make it distinct from
    the equivalent plain-NNParams config (the silent-collision guard)."""
    plain = NNParams(input_dim=784, output_dim=10, hidden_dims=[32], dropout_prob=0.0, activation=Activations.RELU)
    assert _conv_params().state() != plain.state()
    # distinct conv configs hash distinctly too
    assert _conv_params(conv_channels=[6, 16]).state() != _conv_params(conv_channels=[8, 16]).state()
    assert _conv_params().state() != _conv_params(kernel_size=3, padding=1).state()


def test_conv_defaults_omitted_from_state():
    """LeNet-default knobs are omit-when-default; conv_channels is ALWAYS
    emitted (resolve_from_state discriminator)."""
    s = _conv_params().state()
    assert "conv_channels" in s
    for key in ("kernel_size", "stride", "padding", "pool_size", "in_channels"):
        assert key not in s
    back = NNParams.resolve_from_state(s)
    assert (back.kernel_size, back.stride, back.padding, back.pool_size, back.in_channels) == (5, 1, 0, 2, 1)


# ---------------------------------------------------------------------------
# Net: Conv2d+pool blocks, Linear head, forward shapes
# ---------------------------------------------------------------------------


def test_net_structure():
    model = _model(_conv_params())
    assert isinstance(model.net, ConvNN)
    convs = list(model.net.convs)
    assert len(convs) == 2
    assert all(isinstance(c, torch.nn.Conv2d) for c in convs)
    assert convs[0].in_channels == 1 and convs[0].out_channels == 6
    assert convs[1].in_channels == 6 and convs[1].out_channels == 16
    fcs = list(model.net.fcs)
    assert all(isinstance(f, torch.nn.Linear) for f in fcs)
    # flatten(256) → 32 → 10
    assert fcs[0].in_features == 256 and fcs[-1].out_features == 10


def test_forward_accepts_image_and_flat_input():
    model = _model(_conv_params())
    model.net.eval()
    imgs = torch.randn(5, 1, 28, 28)
    flat = imgs.view(5, -1)
    with torch.no_grad():
        out_img = model.net(imgs)
        out_flat = model.net(flat)
    assert out_img.shape == (5, 10)
    # flat input reshapes to the same images → identical logits
    assert torch.equal(out_img, out_flat)


# ---------------------------------------------------------------------------
# Training + checkpoint round-trip
# ---------------------------------------------------------------------------


def _tiny_train_params(loader: DataLoader) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=1,
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
    )


def test_conv_trains_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    # 12×12 single-channel toy images, one conv block (12 → conv5 → 8 → pool2 → 4)
    X = torch.randn(32, 1, 12, 12)
    y = torch.randint(0, 3, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)
    model = _model(_conv_params(input_dim=144, output_dim=3, hidden_dims=[16], conv_channels=[4]))

    run = model.train(params=_tiny_train_params(loader))
    assert run.idps
    last = run.idps[-1].train_edp
    assert last.loss is not None and torch.isfinite(torch.tensor(last.loss))


def test_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    X = torch.randn(16, 1, 12, 12)
    y = torch.randint(0, 3, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)
    model = _model(_conv_params(input_dim=144, output_dim=3, hidden_dims=[16], conv_channels=[4]))
    run = model.train(params=_tiny_train_params(loader))

    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert ckpt is not None
    assert isinstance(ckpt.net_params, NNConvParams)  # resolve_from_state dispatch
    reloaded = NNModel.from_checkpoint(ckpt)
    assert isinstance(reloaded.net, ConvNN)
    assert set(reloaded.net.state_dict().keys()) == set(ckpt.net_state.keys())
