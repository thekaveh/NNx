from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Unnumbered scripts are outside the numbered glob: register them here so
# they are imported (without running main) and, below, executed.
UNNUMBERED_EXAMPLES = [
    "abstention_offline.py",
    "builder_branching.py",
    "calibration_offline.py",
    "compare_seeds.py",
    "custom_module.py",
    "decision_benchmark_offline.py",
    "decision_fixed_head.py",
    "decision_jobs.py",
    "decision_nli.py",
    "experiment_plan.py",
    "graph_classification_offline.py",
    "graph_optional_splits.py",
    "history_journal.py",
    "optimizer_factories.py",
    "prediction_stream.py",
    "preprocessing_offline.py",
    "ranking_offline.py",
    "regression_task.py",
    "run_bundle.py",
    "scheduler_clocks.py",
    "split_replay.py",
    "tabular_validation.py",
]
EXAMPLES = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")) + [
    ROOT / "examples" / name for name in UNNUMBERED_EXAMPLES
]


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_example_imports_without_running_main(example):
    runpy.run_path(str(example), run_name="__nnx_example_smoke__")


@pytest.mark.parametrize(
    "name",
    [
        "01_synthetic_classification.py",
        "03_custom_metrics.py",
        "05_custom_train_step_autoencoder.py",
        "abstention_offline.py",
        "calibration_offline.py",
        "compare_seeds.py",
        "10_knowledge_distillation.py",
        "custom_module.py",
        "decision_benchmark_offline.py",
        "decision_fixed_head.py",
        "decision_jobs.py",
        "decision_nli.py",
        "experiment_plan.py",
        "graph_classification_offline.py",
        "history_journal.py",
        "26_custom_eval_step.py",
        "prediction_stream.py",
        "preprocessing_offline.py",
        "ranking_offline.py",
        "regression_task.py",
        "run_bundle.py",
        "scheduler_clocks.py",
        "split_replay.py",
    ],
)
def test_representative_examples_run_end_to_end(name, tmp_path):
    env = os.environ.copy()
    env["NNX_TQDM_DISABLE"] = "1"
    subprocess.run(
        [sys.executable, str(ROOT / "examples" / name)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )


BOUNDED_EXAMPLE_HELPERS = [
    ("01_synthetic_classification.py", "native_nll_workflow"),
    ("01_synthetic_classification.py", "precision_workflow"),
    ("02_resume_training.py", "amp_resume_compatibility"),
    ("03_custom_metrics.py", "named_monitor_workflow"),
    ("04_onnx_export.py", "registered_module_variant"),
    ("10_knowledge_distillation.py", "objective_mode"),
    ("08_diffusion_2d_mixture.py", "objective_mode"),
    ("16_ijepa_image_plumbing.py", "objective_mode"),
    ("02_resume_training.py", "callback_continuation"),
    ("02_resume_training.py", "provenance_mode"),
    ("02_resume_training.py", "checkpoint_probe_first_fit"),
    ("02_resume_training.py", "iterable_graph_resume"),
    ("02_resume_training.py", "overwrite_best_recovery"),
    ("07_lora_finetuning.py", "dora_zero_row_composition"),
    ("07_lora_finetuning.py", "lora_artifact_roundtrip"),
    ("07_lora_finetuning.py", "peft_alias_preflight"),
    ("07_lora_finetuning.py", "peft_eval_injection"),
    ("07_lora_finetuning.py", "peft_preconverted_base"),
    ("09_gan_with_trainer.py", "trainer_builder_snapshot"),
    ("09_gan_with_trainer.py", "two_optimizer_resume"),
    ("09_gan_with_trainer.py", "manual_scheduler_ownership"),
    ("11_tinystories_lm.py", "manual_attention_lm_example"),
    ("11_tinystories_lm.py", "lm_config_rejects_nonfinite"),
    ("11_tinystories_lm.py", "causal_lm_task_workflow"),
    ("12_quantize_int8.py", "quantized_generative_subtype"),
    ("16_ijepa_image_plumbing.py", "float64_vit_predictor_step"),
    ("20_low_rank_surgery_ffn.py", "surgery_freeze_roles"),
    ("20_low_rank_surgery_ffn.py", "widen_supported_workflow"),
    ("20_low_rank_surgery_ffn.py", "deepen_override_workflow"),
    ("20_low_rank_surgery_ffn.py", "named_deepen_workflow"),
    ("20_low_rank_surgery_ffn.py", "recipe_reconstruction_workflow"),
    ("22_dpo_synthetic_preferences.py", "dpo_sample_batch_sizes"),
    ("abstention_offline.py", "abstention_offline_workflow"),
    ("builder_branching.py", "builder_branching_workflow"),
    ("calibration_offline.py", "calibration_offline_workflow"),
    ("compare_seeds.py", "compare_seeds_workflow"),
    ("custom_module.py", "custom_module_workflow"),
    ("decision_benchmark_offline.py", "decision_benchmark_offline_workflow"),
    ("decision_fixed_head.py", "decision_fixed_head_workflow"),
    ("decision_jobs.py", "decision_jobs_workflow"),
    ("decision_nli.py", "decision_nli_workflow"),
    ("experiment_plan.py", "experiment_plan_workflow"),
    ("graph_classification_offline.py", "graph_classification_workflow"),
    ("graph_optional_splits.py", "graph_optional_splits_workflow"),
    ("history_journal.py", "history_journal_workflow"),
    ("optimizer_factories.py", "optimizer_factories_workflow"),
    ("prediction_stream.py", "prediction_stream_workflow"),
    ("preprocessing_offline.py", "preprocessing_offline_workflow"),
    ("ranking_offline.py", "ranking_offline_workflow"),
    ("regression_task.py", "regression_task_workflow"),
    ("run_bundle.py", "run_bundle_workflow"),
    ("scheduler_clocks.py", "scheduler_clocks_workflow"),
    ("split_replay.py", "split_replay_workflow"),
    ("tabular_validation.py", "tabular_validation_workflow"),
    ("25_conv_classifier.py", "conv_integral_schema_roundtrip"),
    ("26_custom_eval_step.py", "nonfinite_metric_workflow"),
]


@pytest.mark.parametrize(("name", "helper"), BOUNDED_EXAMPLE_HELPERS, ids=lambda value: value)
def test_bounded_example_helpers_execute(name, helper, tmp_path, monkeypatch):
    """Execution coverage for bounded example helpers: import the script
    without running ``main`` and call the named helper in a temporary
    working directory. Import success alone is insufficient — the
    helper's own assertions are the contract under test."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    namespace = runpy.run_path(str(ROOT / "examples" / name), run_name="__nnx_example_smoke__")
    try:
        fn = namespace[helper]
        fn()
        if helper in {"dora_zero_row_composition", "manual_attention_lm_example"}:
            import torch

            fn(dtype=torch.float32)
    except ImportError as exc:
        # Helpers that need an optional extra raise ImportError naming it:
        # an explicit skip in a core-only environment, never a silent pass.
        pytest.skip(f"{name}:{helper} needs an optional extra: {exc}")


def test_the_precision_example_runs_bf16_where_the_host_supports_it(tmp_path, monkeypatch):
    """FEAT-028: example 01's precision option — bounded CPU FP32 runs in the
    helper list above; BF16 is gated on ``precision_support`` (CPU qualifies),
    and a mode the device cannot run is skipped, not claimed."""
    from nnx import Devices

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    namespace = runpy.run_path(
        str(ROOT / "examples" / "01_synthetic_classification.py"), run_name="__nnx_example_smoke__"
    )
    summary = namespace["precision_workflow"]("bf16")
    assert summary is not None and (summary["requested"], summary["effective"]) == ("bf16", "bf16")
    assert namespace["precision_workflow"]("bf16", Devices.MPS) is None  # unsupported there: gated off
