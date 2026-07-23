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
