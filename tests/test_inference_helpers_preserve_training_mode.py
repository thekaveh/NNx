"""Regression tests for the train → inference → train pattern.

NNx's inference-shaped helpers all switch the underlying ``nn.Module``
to ``eval()`` mode before running, since BatchNorm / Dropout layers
behave differently in train vs eval. Without an explicit restore, the
common train → inference → train pattern silently strands the caller's
model in ``.eval()`` mode after the helper returns. ``BatchNorm``'s
running-stats update and ``Dropout``'s masking are then disabled on
the next training step.

This file covers FOUR of the five inference helpers:

  * ``NNModel.predict``
  * ``NNModel.evaluate``
  * ``nnx.diffusion.sample``
  * ``nnx.embeddings.embed_texts``

The fifth — ``GenerativeNNModel.generate`` — also implements the same
non-destructive contract, but its round-trip test lives in
``tests/test_generative_nn_model.py::test_generate_restores_training_mode_after_call``
because that file already carries the ``pytest.importorskip("tokenizers")``
guard ``generate()`` needs (the ``lm`` optional extra). Keeping the
generate test there means this file stays runnable on every CI matrix
row regardless of the ``lm`` extra install.

The codebase carries two prior precedents for the non-destructive
restore pattern:

  * ``nnx.viz.activation_map`` (src/nnx/viz/activation.py:124) —
    snapshots ``model.training``, calls ``eval()``, restores in
    ``finally``.
  * ``nnx.lr_finder`` (src/nnx/lr_finder.py) — same pattern, with the
    explicit docstring promise "non-destructive".
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
)


def _tiny_nnmodel() -> NNModel:
    """3-class FFN with BN/Dropout-light architecture, suitable for
    asserting the .training flag round-trips."""
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(
            input_dim=4,
            output_dim=3,
            hidden_dims=[8],
            dropout_prob=0.1,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def _tiny_loader() -> DataLoader:
    X = torch.randn(32, 4)
    y = torch.randint(0, 3, (32,))
    return DataLoader(TensorDataset(X, y), batch_size=8)


def test_predict_restores_training_mode_after_call():
    """``NNModel.predict`` must leave ``self.net.training`` exactly as
    it found it. Before the fix, predict() left every model in eval()
    mode; a caller doing the common train → predict → train-more
    pattern silently disabled Dropout / BatchNorm-running-stats on the
    next training step."""
    model = _tiny_nnmodel()
    model.net.train()
    assert model.net.training is True

    model.predict(X=torch.randn(8, 4).numpy())

    assert model.net.training is True, "predict() did not restore training-mode"


def test_predict_preserves_eval_mode_caller():
    """Symmetric: if the caller is in eval() mode, predict() must not
    flip them into train()."""
    model = _tiny_nnmodel()
    model.net.eval()
    assert model.net.training is False

    model.predict(X=torch.randn(8, 4).numpy())

    assert model.net.training is False


def test_evaluate_restores_training_mode_after_call():
    """``NNModel.evaluate`` must leave ``self.net.training`` exactly as
    it found it — the canonical train → evaluate → train-more loop."""
    model = _tiny_nnmodel()
    model.net.train()
    assert model.net.training is True

    model.evaluate(loader=_tiny_loader())

    assert model.net.training is True, "evaluate() did not restore training-mode"


def test_evaluate_preserves_eval_mode_caller():
    model = _tiny_nnmodel()
    model.net.eval()

    model.evaluate(loader=_tiny_loader())

    assert model.net.training is False


def test_predict_restores_training_mode_after_exception():
    """Non-destructive restore must hold on the exception path too.
    Use a non-tensor input that triggers the numpy-coercion failure
    INSIDE the try block; the finally must still restore training-mode."""
    model = _tiny_nnmodel()
    model.net.train()
    assert model.net.training is True

    # A list of mixed Python objects → np.asarray produces dtype=object →
    # torch.from_numpy raises TypeError (the failure path observed in
    # the wild). Any exception inside the predict body would do; this
    # one is reproducible without monkeypatching.
    with pytest.raises(TypeError):
        model.predict(X=[object(), object()])

    assert model.net.training is True, (
        "predict() left model in eval() after exception — non-destructive contract broken on the exception path"
    )


def test_predict_rejects_empty_loader_and_restores_training_mode():
    model = _tiny_nnmodel()
    model.net.train()
    empty = DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)), batch_size=8)

    with pytest.raises(ValueError, match=r"predict\(\) loader produced zero batches"):
        model.predict(empty)

    assert model.net.training is True


def _make_diffusion_model_with_schedule():
    """Shared fixture-factory for the two diffusion.sample mode-restore
    tests. Returns (model, schedule) primed for a 4-step reverse-diffusion
    sample call."""
    from nnx.diffusion import DiffusionMLP, NoiseSchedulers

    torch.manual_seed(0)
    net = DiffusionMLP(input_dim=2, hidden_dims=[16], time_embed_dim=8)
    model = NNModel(
        net_params=NNParams(
            input_dim=2,
            output_dim=2,
            hidden_dims=[16],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR),
    )
    # Swap in the diffusion net (the sampler operates on model.net).
    model.net = net
    schedule = NoiseSchedulers.LINEAR(T=4)
    return model, schedule


def test_diffusion_sample_restores_training_mode_after_call():
    """``nnx.diffusion.sample`` from a ``train()`` caller leaves the net
    in ``train()``."""
    from nnx.diffusion import sample

    model, schedule = _make_diffusion_model_with_schedule()
    model.net.train()
    assert model.net.training is True

    _ = sample(model, schedule, shape=(2, 2))

    assert model.net.training is True, "diffusion.sample did not restore training-mode"


def test_diffusion_sample_preserves_eval_mode_caller():
    """Symmetric: a caller in ``eval()`` must stay in ``eval()`` after
    ``nnx.diffusion.sample`` — the non-destructive contract preserves
    whichever mode the caller chose, not just ``train``."""
    from nnx.diffusion import sample

    model, schedule = _make_diffusion_model_with_schedule()
    model.net.eval()
    assert model.net.training is False

    _ = sample(model, schedule, shape=(2, 2))

    assert model.net.training is False, "diffusion.sample flipped eval() caller into train()"


class _FakeBackbone(nn.Module):
    """Tiny stand-in for a sentence-transformers / nn.Module backbone.

    Sized just enough for ``nnx.embeddings.embed_texts`` to invoke
    ``forward(list[str]) -> Tensor[B, D]`` and exit cleanly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.proj(torch.zeros(len(texts), 4))


def test_embed_texts_restores_training_mode_after_call():
    """``nnx.embeddings.embed_texts`` from a ``train()`` caller leaves
    the backbone in ``train()``."""
    from nnx.embeddings import embed_texts

    backbone = _FakeBackbone()
    backbone.train()
    assert backbone.training is True

    _ = embed_texts(backbone, texts=["a", "b"], normalize=False)

    assert backbone.training is True, "embed_texts did not restore training-mode"


def test_embed_texts_preserves_eval_mode_caller():
    """Symmetric: a caller in ``eval()`` must stay in ``eval()`` after
    ``nnx.embeddings.embed_texts`` — the non-destructive contract
    preserves whichever mode the caller chose, not just ``train``."""
    from nnx.embeddings import embed_texts

    backbone = _FakeBackbone()
    backbone.eval()
    assert backbone.training is False

    _ = embed_texts(backbone, texts=["a", "b"], normalize=False)

    assert backbone.training is False, "embed_texts flipped eval() caller into train()"
