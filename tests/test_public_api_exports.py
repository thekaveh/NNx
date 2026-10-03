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
        "abstention",
        "bundles",
        "calibration",
        "comparison",
        "data_splits",
        "decisions",
        "diffusion",
        "embeddings",
        "finetune",
        "generation",
        "history",
        "interop",
        "optimizers",
        "paradigms",
        "plans",
        "prediction",
        "preprocessing",
        "provenance",
        "peft",
        "prune",
        "quantize",
        "streaming",
        "surgery",
        "trainer",
        "transforms",
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


def test_data_splits_api_is_public_and_complete():
    from nnx import data_splits

    assert {"plan_split", "SplitManifest", "SplitIndices", "SplitError", "FORMAT"} <= set(data_splits.__all__)
    assert all(getattr(data_splits, name, None) is not None for name in data_splits.__all__)


def test_preprocessing_api_is_public_and_complete():
    from nnx import preprocessing

    expected = {"Standardizer", "SplitView", "describe_transform", "PreprocessingError", "FORMAT"}
    assert expected <= set(preprocessing.__all__)
    assert all(getattr(preprocessing, name, None) is not None for name in preprocessing.__all__)


def test_bundles_api_is_public_and_complete():
    from nnx import bundles

    expected = {
        "BUNDLE_FORMAT",
        "BUNDLE_VERSION",
        "BundleCapabilityError",
        "BundleError",
        "BundleInfo",
        "BundleIntegrityError",
        "BundleReconstructionError",
        "ReconstructedBundle",
        "export_bundle",
        "inspect_bundle",
        "reconstruct_bundle",
        "validate_bundle",
    }
    assert set(bundles.__all__) == expected
    assert all(getattr(bundles, name, None) is not None for name in bundles.__all__)
    assert "bundles" in nnx.__all__


def test_history_api_is_public_and_complete():
    from nnx import history

    expected = {
        "JOURNAL_FORMAT",
        "JOURNAL_VERSION",
        "HistoryCorruptionError",
        "HistoryJournal",
        "export_history_csv",
        "iter_history",
        "migrate_history",
    }
    assert set(history.__all__) == expected
    assert all(getattr(history, name, None) is not None for name in history.__all__)
    assert "history" in nnx.__all__


def test_transforms_api_is_public_and_complete():
    from nnx import transforms

    expected = {"RecipeError", "TransformOp", "TransformRecipe", "check_optimizer", "lora", "low_rank"}
    assert set(transforms.__all__) == expected
    assert all(getattr(transforms, name, None) is not None for name in transforms.__all__)
    assert "transforms" in nnx.__all__


def test_calibration_api_is_public_and_complete():
    from nnx import calibration

    expected = {
        "fit_temperature",
        "TemperatureCalibrator",
        "CalibrationFit",
        "CalibratedPrediction",
        "CalibrationReport",
        "CalibrationMetrics",
        "ReliabilityBin",
        "negative_log_likelihood",
        "brier_score",
        "reliability_bins",
        "expected_calibration_error",
        "model_fingerprint",
        "CalibrationError",
        "CalibrationFitError",
        "CalibrationMismatchError",
        "DEFAULT_EPSILON",
        "FORMAT",
    }
    assert expected <= set(calibration.__all__)
    assert all(getattr(calibration, name, None) is not None for name in calibration.__all__)


def test_abstention_api_is_public_and_complete():
    from nnx import abstention

    expected = {
        "AbstentionPolicy",
        "AbstentionResult",
        "Outcome",
        "AcceptedRows",
        "CoverageReport",
        "CoverageAccumulator",
        "CurvePoint",
        "ThresholdSelection",
        "SelectiveDecision",
        "select_threshold",
        "risk_coverage_curve",
        "scores",
        "decide",
        "AbstentionError",
        "AbstentionSchemaError",
        "FORMAT",
        "POLICY_KINDS",
        "REASONS",
        "INPUT_FIELDS",
    }
    assert expected <= set(abstention.__all__)
    assert all(getattr(abstention, name, None) is not None for name in abstention.__all__)


def test_plans_api_is_public_and_complete():
    import nnx
    from nnx import plans

    expected = {
        "ATTEMPT_SALT_PREFIX",
        "Diagnostic",
        "ExperimentPlan",
        "FitResult",
        "PlanError",
        "PlanValidation",
        "ProbeResult",
        "SplitMetrics",
    }
    assert set(plans.__all__) == expected
    assert all(getattr(plans, name, None) is not None for name in plans.__all__)
    assert nnx.ExperimentPlan is plans.ExperimentPlan and nnx.FitResult is plans.FitResult
    assert {"plans", "ExperimentPlan", "FitResult"} <= set(nnx.__all__)


def test_streaming_api_is_public_and_complete():
    import nnx
    from nnx import streaming

    expected = {
        "MetricMergeError",
        "MetricSnapshot",
        "PredictionBatch",
        "PredictionStream",
        "StreamClosedError",
        "StreamingMetrics",
        "concatenate_predictions",
        "streaming_eval_step",
    }
    assert set(streaming.__all__) == expected
    assert all(getattr(streaming, name, None) is not None for name in streaming.__all__)
    for name in ("PredictionStream", "PredictionBatch", "StreamingMetrics", "MetricSnapshot", "streaming_eval_step"):
        assert getattr(nnx, name) is getattr(streaming, name) and name in nnx.__all__
    assert "streaming" in nnx.__all__ and callable(nnx.NNModel.iter_predict)
