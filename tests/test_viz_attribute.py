"""Tests for nnx.viz.attribute — Captum attribution wrapper.

Five contracts:
- Unknown method strings raise ValueError listing the supported keys.
- Integrated-gradients call returns (Tensor, plotly.Figure) and the
  attribution tensor's shape matches the input shape (Captum's standard
  return contract for the gradient-family methods).
- Saliency works through the same call site (no baseline required).
- Missing-captum path raises a clear ImportError pointing the user at
  `pip install captum`.
- All six supported method strings are individually callable end-to-end
  (parametrized to lock the public method list down).
"""

from __future__ import annotations

import sys

import plotly.graph_objects as go
import pytest
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
from nnx.viz import attribute
from nnx.viz.attribute import SUPPORTED_METHODS


@pytest.fixture
def tiny_model() -> NNModel:
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


@pytest.fixture
def tiny_input() -> torch.Tensor:
    # (B=2, D=4) tabular input matching the tiny_model's input_dim.
    torch.manual_seed(0)
    return torch.randn(2, 4)


def test_attribute_unknown_method_raises(tiny_model, tiny_input):
    with pytest.raises(ValueError, match="unknown method 'not_real'"):
        attribute(tiny_model, tiny_input, method="not_real", target=0)


def test_attribute_integrated_gradients_returns_tensor_and_figure(tiny_model, tiny_input):
    attr, fig = attribute(tiny_model, tiny_input, method="integrated_gradients", target=0)
    assert isinstance(attr, torch.Tensor)
    assert isinstance(fig, go.Figure)
    # Captum's gradient-family methods return an attribution tensor
    # whose shape matches the input tensor's shape.
    assert attr.shape == tiny_input.shape


def test_attribute_saliency_works(tiny_model, tiny_input):
    attr, fig = attribute(tiny_model, tiny_input, method="saliency", target=0)
    assert isinstance(attr, torch.Tensor)
    assert isinstance(fig, go.Figure)
    assert attr.shape == tiny_input.shape


def test_attribute_missing_captum_raises_importerror(tiny_model, tiny_input, monkeypatch):
    # Force the lazy `from captum.attr import ...` inside `_build_attributor`
    # to fail by stubbing out the parent `captum` module entry to None.
    # This is the standard way to simulate a missing optional dep
    # without uninstalling the package from the test environment.
    monkeypatch.setitem(sys.modules, "captum", None)
    monkeypatch.setitem(sys.modules, "captum.attr", None)
    with pytest.raises(ImportError, match="pip install captum"):
        attribute(tiny_model, tiny_input, method="saliency", target=0)


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_attribute_each_supported_method_callable(method, tiny_model, tiny_input):
    # Each method should run end-to-end with the wrapper's per-method
    # default kwargs (the wrapper supplies `baselines` for gradient_shap
    # and `sliding_window_shapes` for occlusion). Locks the public method
    # list down — adding a new key without wiring it through will fail
    # here loudly.
    attr, fig = attribute(tiny_model, tiny_input, method=method, target=0)
    assert isinstance(attr, torch.Tensor)
    assert isinstance(fig, go.Figure)
    assert attr.shape == tiny_input.shape
