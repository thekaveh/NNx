from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_example_imports_without_running_main(example):
    runpy.run_path(str(example), run_name="__nnx_example_smoke__")


@pytest.mark.parametrize(
    "name",
    [
        "01_synthetic_classification.py",
        "05_custom_train_step_autoencoder.py",
        "26_custom_eval_step.py",
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
    ("02_resume_training.py", "checkpoint_probe_first_fit"),
    ("02_resume_training.py", "iterable_graph_resume"),
    ("02_resume_training.py", "overwrite_best_recovery"),
    ("07_lora_finetuning.py", "dora_zero_row_composition"),
    ("07_lora_finetuning.py", "lora_artifact_roundtrip"),
    ("07_lora_finetuning.py", "peft_preconverted_base"),
    ("09_gan_with_trainer.py", "trainer_builder_snapshot"),
    ("11_tinystories_lm.py", "manual_attention_lm_example"),
    ("12_quantize_int8.py", "quantized_generative_subtype"),
    ("20_low_rank_surgery_ffn.py", "surgery_freeze_roles"),
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
