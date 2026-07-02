"""Regression tests for the curated public import facade."""

from __future__ import annotations

import nnx


def test_core_public_exports_are_available_from_top_level():
    names = [
        "Activations",
        "Devices",
        "FeedFwdNN",
        "Losses",
        "NNModel",
        "NNModelParams",
        "NNOptimParams",
        "NNParams",
        "NNRun",
        "NNSchedulerParams",
        "NNTrainParams",
        "Nets",
        "Optims",
        "Schedulers",
        "Trainer",
        "Utils",
        "VisUtils",
        "default_train_step",
        "drop_layer",
        "freeze",
        "lr_finder",
        "sample_next_token",
        "set_seed",
        "widen",
    ]

    missing = [name for name in names if name not in nnx.__all__ or getattr(nnx, name, None) is None]
    assert missing == []


def test_specialized_public_facades_are_available_from_top_level():
    facades = [
        "diffusion",
        "embeddings",
        "finetune",
        "generation",
        "interop",
        "paradigms",
        "peft",
        "prune",
        "quantize",
        "surgery",
        "trainer",
        "viz",
    ]

    missing = [name for name in facades if name not in nnx.__all__ or getattr(nnx, name, None) is None]
    assert missing == []
