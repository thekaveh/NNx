"""Arbitrary modules and registered model factories (FEAT-006).

``NNModel`` is not limited to the built-in ``Nets`` enum:

  1. **Register a factory** for your own ``nn.Module`` under a stable
     ``(id, version)`` — ``register_model_factory("examples.encoder", 1,
     factory)`` — and describe the model with
     ``ModelSpec(id, version, config, seed)``. The module is built under
     ``torch.manual_seed(seed)`` (the ambient RNG is restored), trains
     through the standard loop, and every artifact records the spec, never
     code: the checkpoint rebuilds it through the registry
     (``NNModel.from_checkpoint``), an unknown factory fails before weights
     load, and ``print(run)`` shows the descriptor instead of built-in
     dimensions.
  2. **Wrap an instance** you already have — ``NNModel(module=encoder,
     params=NNModelParams(loss=...))``. The object itself is trained
     (``model.net is encoder``); a ``KeywordInputs`` batch adapter feeds
     mapping batches to a keyword-only ``forward``. Its descriptor is
     runtime-only (``reconstructible=False``): checkpoints hold its weights,
     and reloading them needs the module again (``module=``).

Fully offline, CPU only.

Run:
    python examples/custom_module.py

The bounded ``custom_module_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Checkpoints,
    KeywordInputs,
    Losses,
    MissingModelFactoryError,
    ModelSpec,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNRun,
    NNTrainParams,
    register_model_factory,
    set_seed,
    unregister_model_factory,
)

FACTORY_ID = "examples.encoder"


class Encoder(nn.Module):
    """A small MLP encoder with a classification head — plain PyTorch, no
    NNx hooks (no ``unpack_batch``): batches ``(x, y)`` reach it through the
    default positional adapter."""

    def __init__(self, in_dim: int = 4, width: int = 16, classes: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.head = nn.Linear(width, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class KeywordEncoder(nn.Module):
    """A keyword-only ``forward(features=..., mask=...)`` like many
    third-party encoders."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 16)
        self.head = nn.Linear(16, 2)

    def forward(self, *, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.proj(features * mask)))


def _data(seed: int = 0, n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 4, generator=generator)
    return X, (X[:, 0] + 0.5 * X[:, 1] > 0).long()


def _train_params(train_loader, val_loader=None, epochs: int = 3) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams.builder().adam(max_lr=1e-2).build(),
        save_phase_checkpoints=False,
    )


def custom_module_workflow() -> None:
    set_seed(0)
    X, y = _data()
    train = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(0))
    val = DataLoader(TensorDataset(*_data(1, 32)), batch_size=16)

    # 1. A registered factory: portable, rebuilt from its spec.
    register_model_factory(FACTORY_ID, 1, lambda config: Encoder(**config))
    try:
        spec = ModelSpec(FACTORY_ID, 1, {"width": 16}, seed=7)
        model = NNModel(params=NNModelParams(net=spec, loss=Losses.CROSS_ENTROPY))
        run = model.train(params=_train_params(train, val))
        print(run)  # the descriptor (kind, config, seed) — no built-in dims
        assert "kind=registered" in str(run) and "dims=" not in str(run)
        assert NNRun.load(run.id).model.net == spec  # reloads offline, as data

        checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
        assert checkpoint is not None and checkpoint.reconstructible
        rebuilt = NNModel.from_checkpoint(checkpoint)  # through the registry
        probe = _data(2, 8)[0]
        np.testing.assert_array_equal(rebuilt.predict(probe).logits, model.predict(probe).logits)
        print(f"rebuilt {spec} from run {run.id}: identical predictions")
    finally:
        unregister_model_factory(FACTORY_ID, 1)
    try:
        NNModel.from_checkpoint(checkpoint)
        raise AssertionError("an unregistered factory must not rebuild")
    except MissingModelFactoryError as error:
        print(f"without the factory: {error}")

    # 2. A caller-owned instance with keyword inputs: trained in place.
    encoder = KeywordEncoder()
    records = [{"features": x, "mask": torch.ones(4), "labels": label} for x, label in zip(*_data(3, 48), strict=True)]
    wrapped = NNModel(
        module=encoder,
        params=NNModelParams(loss=Losses.CROSS_ENTROPY),
        batch_adapter=KeywordInputs(("features", "mask"), target="labels"),
    )
    assert wrapped.net is encoder
    wrapped_run = wrapped.train(params=_train_params(DataLoader(records, batch_size=16), epochs=2))
    print(wrapped_run)
    checkpoint = NNCheckpoint.load(run=wrapped_run.id, type=Checkpoints.LAST)
    assert checkpoint is not None and not checkpoint.reconstructible  # runtime-only weights
    reloaded = NNModel.from_checkpoint(
        checkpoint, module=KeywordEncoder(), batch_adapter=KeywordInputs(("features", "mask"))
    )
    probe = {"features": _data(4, 6)[0], "mask": torch.ones(6, 4)}
    np.testing.assert_array_equal(reloaded.predict(probe).logits, wrapped.predict(probe).logits)
    print("runtime module reloaded into a supplied KeywordEncoder: identical predictions")


def main() -> None:
    custom_module_workflow()


if __name__ == "__main__":
    main()
