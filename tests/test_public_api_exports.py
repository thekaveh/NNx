"""Regression tests for the curated public import facade."""

from __future__ import annotations

import nnx


def test_core_public_exports_are_available_from_top_level():
    names = [
        "Activations",
        "Devices",
        "ConvNN",
        "FeedFwdNN",
        "FeedFwdMoENN",
        "Losses",
        "NNModel",
        "NNModelParams",
        "NNConvParams",
        "NNMoEParams",
        "NNOptimFactoryParams",
        "NNOptimParams",
        "NNParams",
        "NNRun",
        "PredictionResult",
        "PredictionValidationError",
        "ProbabilitySpec",
        "NNSchedulerParams",
        "NNTrainParams",
        "Nets",
        "OptimizerFactorySpec",
        "Optims",
        "Schedulers",
        "Trainer",
        "Utils",
        "VisUtils",
        "build_optimizer",
        "default_train_step",
        "drop_layer",
        "freeze",
        "lr_finder",
        "prediction_from_logits",
        "register_optimizer_factory",
        "registered_optimizer_factories",
        "sample_next_token",
        "set_seed",
        "unregister_optimizer_factory",
        "widen",
    ]

    missing = [name for name in names if name not in nnx.__all__ or getattr(nnx, name, None) is None]
    assert missing == []


def test_specialized_public_facades_are_available_from_top_level():
    facades = [
        "decisions",
        "diffusion",
        "embeddings",
        "finetune",
        "generation",
        "interop",
        "optimizers",
        "paradigms",
        "prediction",
        "peft",
        "prune",
        "quantize",
        "surgery",
        "trainer",
        "viz",
    ]

    missing = [name for name in facades if name not in nnx.__all__ or getattr(nnx, name, None) is None]
    assert missing == []


def test_decision_api_is_public_and_complete():
    from nnx import decisions

    expected = {
        "Choice",
        "Boolean",
        "Score",
        "Option",
        "ChoiceResult",
        "BooleanResult",
        "ScoreResult",
        "Capabilities",
        "DecisionProvider",
        "FixedHeadProvider",
        "validate_response",
        "UnsupportedCapability",
        "InvalidDecisionRequest",
        "InvalidDecisionResponse",
        "ProviderFailure",
    }
    assert expected <= set(decisions.__all__)
    assert all(getattr(decisions, name, None) is not None for name in decisions.__all__)
