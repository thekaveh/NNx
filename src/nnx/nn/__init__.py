"""nnx.nn — foundational NN layer.

This subpackage groups the core building blocks the rest of nnx
composes:

  - `nnx.nn.params` — frozen-dataclass configuration types (NNParams,
    NNModelParams, NNOptimParams, NNSchedulerParams, NNTransformerParams,
    NNTrainParams, NNRun, NNCheckpoint, …) plus their Builder classes.
  - `nnx.nn.net` — concrete nn.Module subclasses (FeedFwdNN, the graph
    family, TransformerNN, ViTNN, …).
  - `nnx.nn.dataset` — DataLoader-producing wrappers (NNDataset,
    NNTabularDataset, NNGraphDataset, NNPreferenceDataset).
  - `nnx.nn.enum` — typed enums for activations, optimizers, schedulers,
    losses, devices, nets, checkpoints.
  - `nnx.nn.callbacks` — the Callback abstract base + the four built-ins
    shipped with NNx (EarlyStopping / LRMonitor / ModelCheckpoint /
    TensorBoardCallback / WandbCallback).
  - `nnx.nn.nn_model` / `nnx.nn.generative_nn_model` — the NNModel
    orchestrator + the LM-path GenerativeNNModel subclass.
  - `nnx.nn.moe` — Mixture-of-Experts linear layer.

The intentional convention is to reach the public surface via the
top-level `nnx` namespace — `from nnx import NNModel, NNParams, …`,
not `from nnx.nn.nn_model import NNModel`. The top-level `__init__.py`
re-exports the curated subset; `__all__` here is intentionally empty so
`from nnx.nn import *` is a no-op rather than a way to bypass the
top-level surface.
"""

from __future__ import annotations

__all__: list[str] = []
