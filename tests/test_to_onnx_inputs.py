"""Input-coercion contracts for ``NNModel.to_onnx``.

The exporter accepts three shapes for ``example_input``:

  * ``torch.Tensor`` (single input)         — historical Python API.
  * ``np.ndarray``   (single input)         — convenience for numpy callers.
  * ``tuple[Tensor | ndarray, ...]``        — multi-input nets.

The bug this file regresses against: the np.ndarray singleton case used
to fall into the iterable branch of the normalizer and get unpacked
row-by-row, so a single ``np.zeros((2, 4))`` produced an ONNX model with
two inputs of shape ``(4,)`` instead of one input of shape ``(2, 4)``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from nnx.nn.enum.activations import Activations
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.nn_model import NNModel
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_params import NNParams


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


def _onnx_input_count(path: str) -> int:
    """Return the number of declared inputs on the exported ONNX graph."""
    import onnx

    model = onnx.load(path)
    return len(model.graph.input)


def test_to_onnx_accepts_torch_tensor_single_input(tmp_path):
    """Baseline — torch.Tensor singleton produces exactly one ONNX input.
    Pairs with the np.ndarray regression below; locks the two paths to
    the same observable contract."""
    pytest.importorskip("onnx")

    model = _model()
    onnx_path = tmp_path / "torch_singleton.onnx"
    out = model.to_onnx(str(onnx_path), example_input=torch.randn(2, 4))

    assert Path(out).exists()
    assert os.path.getsize(out) > 0
    assert _onnx_input_count(out) == 1


def test_to_onnx_accepts_numpy_ndarray_single_input(tmp_path):
    """Regression — a single 2-D np.ndarray must produce ONE ONNX input,
    not ``first_dim`` of them. The buggy normalizer treated the array
    as iterable (np.ndarray IS iterable — it iterates rows) and unpacked
    ``np.zeros((2, 4))`` into two rank-1 inputs of shape ``(4,)``."""
    pytest.importorskip("onnx")

    model = _model()
    onnx_path = tmp_path / "numpy_singleton.onnx"
    out = model.to_onnx(str(onnx_path), example_input=np.zeros((2, 4), dtype=np.float32))

    assert Path(out).exists()
    assert os.path.getsize(out) > 0
    assert _onnx_input_count(out) == 1, (
        f"single np.ndarray must produce 1 ONNX input, got {_onnx_input_count(out)} "
        "— normalizer is unpacking the array row-by-row"
    )

    import onnx

    onnx.checker.check_model(str(onnx_path))


def test_to_onnx_accepts_tuple_of_tensors(tmp_path):
    """Tuple-of-tensors path — multi-input nets stay supported. Reaches
    the tuple branch by construction; FeedFwdNN only has one input so we
    re-use the singleton coercion through a length-1 tuple."""
    pytest.importorskip("onnx")

    model = _model()
    onnx_path = tmp_path / "tuple_singleton.onnx"
    out = model.to_onnx(str(onnx_path), example_input=(torch.randn(2, 4),))

    assert Path(out).exists()
    assert os.path.getsize(out) > 0
    assert _onnx_input_count(out) == 1


def test_to_onnx_accepts_tuple_with_numpy_element(tmp_path):
    """Mixed tuple — one numpy element gets coerced to tensor at the
    boundary; no row-unpacking. Same input-count contract."""
    pytest.importorskip("onnx")

    model = _model()
    onnx_path = tmp_path / "tuple_with_numpy.onnx"
    out = model.to_onnx(str(onnx_path), example_input=(np.zeros((2, 4), dtype=np.float32),))

    assert Path(out).exists()
    assert _onnx_input_count(out) == 1
