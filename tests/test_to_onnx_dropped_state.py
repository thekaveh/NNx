"""``NNModel.to_onnx`` names the attached tensors the exporter dropped, instead of leaving them silent."""

from __future__ import annotations

import warnings

import pytest
import torch

pytest.importorskip("onnx")

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams, NNTransformerParams  # noqa: E402


def _feed_forward() -> NNModel:
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _export(model: NNModel, tmp_path) -> None:
    model.to_onnx(str(tmp_path / "model.onnx"), torch.randn(2, 4), dynamic_batch=False)


def test_an_attached_buffer_the_forward_pass_never_reads_is_named_in_a_warning(tmp_path):
    model = _feed_forward()
    model.net.register_buffer("class_prior", torch.tensor([0.25, 0.75]))
    with pytest.warns(UserWarning, match=r"no initializer for 1 tensor.*class_prior"):
        _export(model, tmp_path)


def test_an_attached_weight_the_forward_pass_never_reads_is_named_in_a_warning(tmp_path):
    model = _feed_forward()
    model.net.register_parameter("unused_scale", torch.nn.Parameter(torch.randn(3)))
    with pytest.warns(UserWarning, match="unused_scale"):
        _export(model, tmp_path)


def test_a_plain_export_names_no_dropped_tensor(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _export(_feed_forward(), tmp_path)
    assert not [w for w in caught if "initializer" in str(w.message)]


def test_a_transformer_export_names_no_tensor_it_carries(tmp_path):
    """The legacy exporter renames the transposed MatMul weights and shares the tied embedding:
    those tensors are carried, so the check reports nothing for them."""
    model = NNModel(
        net_params=NNTransformerParams(
            input_dim=64,
            output_dim=64,
            dropout_prob=0.0,
            vocab_size=64,
            n_layers=1,
            n_heads=2,
            d_model=16,
            ffn_mult=2,
            max_seq_len=16,
        ),
        params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.to_onnx(str(tmp_path / "lm.onnx"), torch.randint(0, 64, (2, 8)), dynamic_batch=False)
    assert not [w for w in caught if "initializer" in str(w.message)]
