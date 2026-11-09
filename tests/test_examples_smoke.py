from __future__ import annotations

import importlib.util
import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from tests._process_group import run_group
from tests._repo_files import repo_file

ROOT = Path(__file__).resolve().parents[1]
# Every test here runs a script under examples/, which the sdist does not ship.
repo_file("examples")
# Unnumbered scripts are outside the numbered glob: register them here so
# they are imported (without running main) and, below, executed.
UNNUMBERED_EXAMPLES = [
    "abstention_offline.py",
    "benchmark_compile.py",
    "builder_branching.py",
    "calibration_offline.py",
    "compare_seeds.py",
    "custom_module.py",
    "ddp_supervised.py",
    "decision_benchmark_offline.py",
    "decision_model_pilot.py",
    "decision_fixed_head.py",
    "decision_jev.py",
    "decision_jobs.py",
    "decision_nli.py",
    "experiment_plan.py",
    "graph_classification_offline.py",
    "graph_optional_splits.py",
    "link_prediction_offline.py",
    "history_journal.py",
    "offline_teacher_distillation.py",
    "optimizer_factories.py",
    "ordered_logits_pipeline.py",
    "prediction_stream.py",
    "preprocessing_offline.py",
    "ranking_offline.py",
    "regression_task.py",
    "result_boundaries.py",
    "run_bundle.py",
    "scheduler_clocks.py",
    "search_offline.py",
    "split_replay.py",
    "tabular_validation.py",
]
EXAMPLES = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")) + [
    ROOT / "examples" / name for name in UNNUMBERED_EXAMPLES
]


# Distributions whose import name is not their normalised distribution name.
_IMPORT_NAMES = {"scikit-learn": "sklearn", "pyyaml": "yaml"}


def _core_import_names() -> frozenset[str]:
    """Top-level import names of pyproject's core ``[project] dependencies``
    (regex over the raw text: ``tomllib`` is 3.11+ and NNx supports 3.10)."""
    import re

    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)^\]", text, re.M | re.S)
    assert block is not None, "pyproject.toml has no [project] dependencies list"
    lines = (line.split("#", 1)[0] for line in block.group(1).splitlines())
    requirements = re.findall(r'"([^"]+)"', "\n".join(lines))
    distributions = (re.match(r"[A-Za-z0-9._-]+", req).group(0).lower() for req in requirements)
    return frozenset(_IMPORT_NAMES.get(dist, dist.replace("-", "_").replace(".", "_")) for dist in distributions)


def _skip_if_optional_extra_missing(exc: ImportError, what: str) -> None:
    """Skip when ``exc`` is a missing optional extra: a third-party module
    that is not installed and not a core dependency, or NNx's own
    "pip install thekaveh-nnx[...]" error. Re-raise anything else (a broken
    ``nnx`` import or a missing core dependency fails, never skips)."""
    missing = exc.name or getattr(exc.__cause__, "name", None) or ""
    top = missing.split(".", 1)[0]
    if top == "nnx":
        raise exc
    if "thekaveh-nnx[" not in str(exc) and (not top or top in _core_import_names()):
        raise exc
    pytest.skip(f"{what} needs an optional extra: {exc}")


def test_core_dependencies_never_count_as_optional_extras():
    assert {"torch", "numpy", "pandas", "sklearn", "prettytable", "yaml", "tqdm", "filelock"} <= _core_import_names()
    for name in ("torch", "sklearn", "yaml", "numpy.linalg", "nnx.vision"):
        with pytest.raises(ImportError):
            _skip_if_optional_extra_missing(ModuleNotFoundError(f"No module named {name!r}", name=name), "x")
    with pytest.raises(ImportError):
        _skip_if_optional_extra_missing(ImportError("cannot import name 'x'"), "x")  # no module named: a real bug
    with pytest.raises(pytest.skip.Exception):
        _skip_if_optional_extra_missing(ModuleNotFoundError("No module named 'plotly'", name="plotly"), "x")
    with pytest.raises(pytest.skip.Exception):
        _skip_if_optional_extra_missing(ImportError("pip install thekaveh-nnx[viz]"), "x")


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_example_imports_without_running_main(example):
    try:
        runpy.run_path(str(example), run_name="__nnx_example_smoke__")
    except ImportError as exc:
        _skip_if_optional_extra_missing(exc, example.name)


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
        "link_prediction_offline.py",
        "history_journal.py",
        "offline_teacher_distillation.py",
        "26_custom_eval_step.py",
        "prediction_stream.py",
        "preprocessing_offline.py",
        "ranking_offline.py",
        "regression_task.py",
        "result_boundaries.py",
        "run_bundle.py",
        "scheduler_clocks.py",
        "split_replay.py",
    ],
)
def test_representative_examples_run_end_to_end(name, tmp_path):
    env = os.environ.copy()
    env["NNX_TQDM_DISABLE"] = "1"
    # conftest sets these for the in-process FAISS tests; a user's shell
    # does not, so an example that needs them must set them itself.
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    env.pop("OMP_NUM_THREADS", None)
    done = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if done.returncode != 0:
        # A missing optional extra (graph / vision) skips, as the import tests do.
        missing = re.search(r"ModuleNotFoundError: No module named '([^']+)'", done.stderr)
        if missing is not None:
            _skip_if_optional_extra_missing(ImportError(missing.group(0), name=missing.group(1)), name)
        if "thekaveh-nnx[" in done.stderr:
            pytest.skip(f"{name} needs an optional extra")
        raise AssertionError(f"{name} exited {done.returncode}:\n{done.stderr[-4000:]}")


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
    ("15_qat_classifier.py", "qat_reload_workflow"),
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
    ("decision_jev.py", "decision_jev_workflow"),
    ("decision_nli.py", "decision_nli_workflow"),
    ("search_offline.py", "search_workflow"),
    ("decision_model_pilot.py", "pilot_workflow"),
    ("result_boundaries.py", "result_boundaries_workflow"),
    ("ordered_logits_pipeline.py", "ordered_logits_pipeline_workflow"),
    ("experiment_plan.py", "experiment_plan_workflow"),
    ("graph_classification_offline.py", "graph_classification_workflow"),
    ("graph_optional_splits.py", "graph_optional_splits_workflow"),
    ("link_prediction_offline.py", "link_prediction_workflow"),
    ("history_journal.py", "history_journal_workflow"),
    ("offline_teacher_distillation.py", "offline_teacher_distillation_workflow"),
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
    try:
        namespace = runpy.run_path(str(ROOT / "examples" / name), run_name="__nnx_example_smoke__")
        fn = namespace[helper]
        fn()
        if helper in {"dora_zero_row_composition", "manual_attention_lm_example"}:
            import torch

            fn(dtype=torch.float32)
    except ImportError as exc:
        # Helpers that need an optional extra raise ImportError naming it:
        # an explicit skip in a core-only environment, never a silent pass.
        _skip_if_optional_extra_missing(exc, f"{name}:{helper}")


def test_overwrite_best_recovery_reads_the_pointer_file_layout(tmp_path, monkeypatch):
    """#570: example 02 reads ``runs/best`` through ``NNRun.best_id``, so
    its re-election demo also runs where symlinks fail and ``runs/best``
    is a directory holding POINTER.txt (Windows without developer mode)."""
    from nnx.nn.params import nn_run as nn_run_mod

    def _raise(*a, **kw):
        raise OSError("symlink not supported (simulated)")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.setattr(nn_run_mod.os, "symlink", _raise)
    namespace = runpy.run_path(str(ROOT / "examples" / "02_resume_training.py"), run_name="__nnx_example_smoke__")
    summary = namespace["overwrite_best_recovery"]()
    assert summary["winner_is_survivor"]
    assert (tmp_path / "runs" / "best" / "POINTER.txt").is_file()


# #570: helpers build models from the caller's RNG state; only main() and
# the workflow or phase entry points below seed.
SEEDING_HELPERS = [
    ("prediction_stream.py", "_model"),
    ("history_journal.py", "_model"),
    ("scheduler_clocks.py", "_model"),
    ("offline_teacher_distillation.py", "_student"),
    ("ranking_offline.py", "_scorer"),
    ("run_bundle.py", "_classifier"),
    ("result_boundaries.py", "_bundle"),
    ("ordered_logits_pipeline.py", "_generate"),
]
# Phase entry points that seed, besides main() and BOUNDED_EXAMPLE_HELPERS.
SEEDING_PHASES = {
    "benchmark_compile.py": {"run"},
    "decision_model_pilot.py": {"run_mechanics"},
    "run_bundle.py": {"built_in_classifier", "registered_module"},
}
_SEED_CALLS = {"set_seed", "torch.manual_seed", "torch.cuda.manual_seed", "torch.cuda.manual_seed_all"}
_SEED_CALLS |= {"np.random.seed", "numpy.random.seed", "random.seed"}


def _call_helper(name: str, helper: str, root: Path):
    fn = runpy.run_path(str(ROOT / "examples" / name), run_name="__nnx_example_smoke__")[helper]
    if helper == "_model" and name == "prediction_stream.py":
        from nnx import Losses, TaskSpec

        return fn(TaskSpec.regression(2), Losses.MEAN_SQUARED_ERROR)
    return fn(str(root)) if helper == "_bundle" else fn()


@pytest.mark.parametrize(("name", "helper"), SEEDING_HELPERS, ids=lambda value: value)
def test_example_helpers_leave_the_callers_random_streams_alone(name, helper, tmp_path, monkeypatch):
    """#570: after the caller's ``set_seed(42)`` a helper leaves torch's
    seed at 42 and the NumPy and Python streams where the caller put them."""
    import random

    import numpy as np
    import torch

    from nnx import set_seed

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    set_seed(42)
    numpy_state, python_state = np.random.get_state()[1].copy(), random.getstate()
    try:
        _call_helper(name, helper, tmp_path)
    except ImportError as exc:
        _skip_if_optional_extra_missing(exc, f"{name}:{helper}")
    assert torch.initial_seed() == 42
    assert np.array_equal(np.random.get_state()[1], numpy_state)
    assert random.getstate() == python_state


@pytest.mark.parametrize(("name", "helper"), SEEDING_HELPERS[:6], ids=lambda value: value)
def test_example_model_helpers_build_from_the_callers_seed(name, helper, tmp_path, monkeypatch):
    """#570: the caller's seed decides a helper's initial weights."""
    import torch

    from nnx import set_seed

    monkeypatch.chdir(tmp_path)
    states = []
    for seed in (1, 2):
        set_seed(seed)
        states.append(_call_helper(name, helper, tmp_path).net.state_dict())
    assert any(not torch.equal(tensor, states[1][key]) for key, tensor in states[0].items())


def test_examples_seed_only_in_main_and_registered_entry_points():
    """#570: no ``set_seed`` / ``torch.manual_seed`` / ``np.random.seed`` /
    ``random.seed`` call outside ``main()``, a registered bounded helper or a
    phase entry point (a local ``torch.Generator().manual_seed`` is fine)."""
    import ast

    registered: dict[str, set[str]] = {}
    for name, helper in BOUNDED_EXAMPLE_HELPERS:
        registered.setdefault(name, set()).add(helper)
    offenders = []
    for path in sorted((ROOT / "examples").glob("*.py")):
        allowed = {"main"} | registered.get(path.name, set()) | SEEDING_PHASES.get(path.name, set())

        def visit(node: ast.AST, function: str, path: Path = path, allowed: set[str] = allowed) -> None:
            for child in ast.iter_child_nodes(node):
                inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
                if isinstance(child, ast.Call):
                    called = ast.unparse(child.func)
                    if (called in _SEED_CALLS or called.endswith(".set_seed")) and function not in allowed:
                        offenders.append(f"{path.name}:{child.lineno}: {called} in {function}")
                visit(child, inner)

        visit(ast.parse(path.read_text(encoding="utf-8")), "<module>")
    assert offenders == []


def test_qat_reload_check_compares_logits_not_accuracy(tmp_path, monkeypatch):
    """#570: example 15's reload check passes on the real LAST checkpoint and
    fails when every ``scales`` tensor is scaled by 1.5 (on the old,
    separable data that corruption kept the accuracy at 100%)."""
    import dataclasses

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    namespace = runpy.run_path(str(ROOT / "examples" / "15_qat_classifier.py"), run_name="__nnx_example_smoke__")
    try:
        out = namespace["qat_reload_workflow"]()
    except ImportError as exc:
        _skip_if_optional_extra_missing(exc, "15_qat_classifier.py:qat_reload_workflow")
        raise
    check_reload, checkpoint = namespace["check_reload"], out["checkpoint"]
    assert check_reload(out["model"], checkpoint, out["inputs"]) <= 1e-5
    scaled = {key: value * 1.5 if key.endswith("scales") else value for key, value in checkpoint.net_state.items()}
    assert scaled.keys() == checkpoint.net_state.keys() and any(key.endswith("scales") for key in scaled)
    with pytest.raises(AssertionError, match=r"max \|logit diff\|"):
        check_reload(out["model"], dataclasses.replace(checkpoint, net_state=scaled), out["inputs"])


def test_quantization_examples_share_overlapping_class_data():
    """#570: examples 12 and 15 share ``_loaders`` verbatim, and its classes
    overlap: even the nearest-class-mean rule misses validation rows, so a
    quantization delta can be non-zero."""
    import ast

    import torch

    sources = []
    for name in ("12_quantize_int8.py", "15_qat_classifier.py"):
        text = (ROOT / "examples" / name).read_text(encoding="utf-8")
        (node,) = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_loaders"]
        sources.append(ast.get_source_segment(text, node))
    assert sources[0] == sources[1]
    loaders = runpy.run_path(str(ROOT / "examples" / "12_quantize_int8.py"), run_name="__nnx_example_smoke__")
    train, val = loaders["_loaders"](seed=0)
    X, y = train.dataset.tensors
    means = torch.stack([X[y == c].mean(dim=0) for c in range(3)])
    X_val, y_val = val.dataset.tensors
    nearest = (torch.cdist(X_val, means).argmin(dim=1) == y_val).float().mean().item()
    assert 0.8 < nearest < 0.97, nearest


def test_onnx_example_skips_its_exports_without_onnx(tmp_path, monkeypatch, capsys):
    """#570: without the onnx packages, ``04_onnx_export.py``'s ``main()``
    prints its skip messages and returns (``to_onnx`` raises an ImportError
    naming the extra, which the example catches), and its docstring says
    that export needs ``onnx``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    for module in ("onnx", "onnxruntime", "onnxscript"):
        monkeypatch.setitem(sys.modules, module, None)  # not installed
    namespace = runpy.run_path(str(ROOT / "examples" / "04_onnx_export.py"), run_name="__nnx_example_smoke__")
    assert "Requires ``onnx`` to export the model" in namespace["__doc__"]
    namespace["main"]()
    skipped = [line for line in capsys.readouterr().out.splitlines() if "(install `onnx`" in line]
    assert len(skipped) == 2, skipped


def test_the_compile_benchmark_example_runs_with_its_documented_arguments(tmp_path):
    """FEAT-029: ``examples/benchmark_compile.py --device cpu --warmup 3
    --repeats 10`` trains with the advertised inductor backend, reloads its
    checkpoint eagerly and prints separate compile, warmed and profile
    output — real numbers from this host, no speedup asserted."""
    env = os.environ.copy()
    env["NNX_TQDM_DISABLE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "benchmark_compile.py"),
            "--device",
            "cpu",
            "--warmup",
            "3",
            "--repeats",
            "10",
            "--output-dir",
            str(tmp_path / "profile"),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    report = json.loads(completed.stdout[completed.stdout.index("{\n") :])
    assert set(report) == {"compile", "warmed", "profile"}
    assert report["compile"]["training"]["effective"] == "compiled"
    assert report["compile"]["training"]["backend"] == "inductor"
    assert report["compile"]["checkpoint_reloaded_eagerly"] is True
    assert report["compile"]["first_call_seconds"]["compiled"] > 0
    warmed = report["warmed"]
    assert (warmed["warmup"], warmed["repeats"], warmed["device"], warmed["backend"]) == (3, 10, "cpu", "inductor")
    assert warmed["effective"] == "compiled"  # after the compiled fit of the same class
    assert warmed["weights_digest_match"] and warmed["max_abs_diff"] <= 1e-5
    assert warmed["eager"]["mean_seconds"] > 0 and warmed["compiled"]["mean_seconds"] > 0
    assert report["profile"]["steps"] == 5 and report["profile"]["trace_files"]
    assert os.listdir(tmp_path / "profile")


def _torchrun(
    script: str, args: list[str], cwd: Path, timeout: float, launcher: Path | None = None
) -> subprocess.CompletedProcess:
    """``torchrun --standalone --nproc_per_node=2 <script> <args>`` (on macOS
    the rendezvous and Gloo stay on loopback: hosts often resolve their own
    name to an unreachable address). A ``launcher`` runs first on each rank
    and then runs the example."""
    env = os.environ.copy()
    env["NNX_TQDM_DISABLE"] = "1"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    if sys.platform == "darwin":
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--master_addr=127.0.0.1"]
        command += [f"--master_port={port}", "--nproc_per_node=2"]
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    return run_group(  # a timeout ends every rank too, not only torchrun
        [*command, *([str(launcher)] if launcher is not None else []), str(ROOT / "examples" / script), *args],
        cwd=cwd,
        env=env,
        timeout=timeout,
    )


# Runs the DDP example on one rank: it records each rank that calls
# NNRun.save, and with NNX_SMOKE_WRITER_RANK set it moves the writer there.
_DDP_LAUNCHER = """
import dataclasses, os, runpy, sys
from nnx import NNModel, NNRun

def save(self, *args, _save=NNRun.save, **kwargs):
    with open(os.path.join(os.environ["NNX_SMOKE_SAVERS"], f"rank{os.environ['RANK']}"), "a", encoding="utf-8") as fh:
        fh.write("NNRun.save\\n")
    return _save(self, *args, **kwargs)

def train(self, *args, _train=NNModel.train, distributed=None, **kwargs):
    if distributed is not None and "NNX_SMOKE_WRITER_RANK" in os.environ:
        distributed = dataclasses.replace(distributed, writer_rank=int(os.environ["NNX_SMOKE_WRITER_RANK"]))
    return _train(self, *args, distributed=distributed, **kwargs)

NNRun.save, NNModel.train = save, train
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


@pytest.mark.parametrize("writer", [0, 1], ids=["writer-rank-0", "writer-rank-1"])
@pytest.mark.xdist_group("torchrun")  # one torchrun launch at a time (FIX-031)
def test_the_ddp_example_runs_under_torchrun_with_one_artifact_owner(tmp_path, monkeypatch, writer):
    """FEAT-030: two CPU ranks under ``torchrun --standalone``; exactly-once
    validation ids, one run id, one artifact owner. #571: the owner the
    example reports is the rank that saved the run, wherever the writer is."""
    out = tmp_path / "ddp-out"
    savers = tmp_path / "savers"
    savers.mkdir()
    launcher = tmp_path / "launcher.py"
    launcher.write_text(_DDP_LAUNCHER, encoding="utf-8")
    monkeypatch.setenv("NNX_SMOKE_SAVERS", str(savers))
    if writer:
        monkeypatch.setenv("NNX_SMOKE_WRITER_RANK", str(writer))
    completed = _torchrun(
        "ddp_supervised.py", ["--device", "cpu", "--epochs", "2", "--output", str(out)], tmp_path, 240, launcher
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    reports = [json.loads((out / f"report-rank{rank}.json").read_text(encoding="utf-8")) for rank in (0, 1)]
    assert sorted(report["rank"] for report in reports) == [0, 1]
    ids = [row for report in reports for row in report["validation_ids"]]
    assert sorted(ids) == list(range(6)) and len(ids) == 6  # every id exactly once
    assert len({report["run_id"] for report in reports}) == 1
    reports.sort(key=lambda report: report["rank"])
    assert sorted(path.name for path in savers.iterdir()) == [f"rank{writer}"]  # only the writer saved the run
    assert [report["owns_artifacts"] for report in reports] == [rank == writer for rank in (0, 1)]
    assert [report["artifacts"] is not None for report in reports] == [rank == writer for rank in (0, 1)]
    # #570: the validation pass scores the training rows, and the report says so.
    assert all("val_accuracy" not in report and "train_rows_accuracy" in report for report in reports)
    runs = [name for name in os.listdir(out / "runs") if not name.startswith(".") and name != "best"]
    assert runs == [reports[0]["run_id"]]


@pytest.mark.xdist_group("torchrun")  # one torchrun launch at a time (FIX-031)
def test_the_ddp_example_terminates_after_an_injected_failure(tmp_path):
    """``--inject-failure``: rank 1 raises; the launch ends non-zero within
    a bounded time and leaves no worker behind."""
    out = tmp_path / "ddp-fail"
    completed = _torchrun(
        "ddp_supervised.py",
        ["--device", "cpu", "--epochs", "3", "--output", str(out), "--inject-failure"],
        tmp_path,
        180,
    )
    assert completed.returncode != 0
    assert "injected failure on rank 1" in completed.stdout + completed.stderr
    leftover = subprocess.run(["pgrep", "-f", str(out)], capture_output=True, text=True)
    assert leftover.stdout.strip() == "", f"orphaned workers: {leftover.stdout}"


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


def test_onnx_sample_runtime_conformance(tmp_path, monkeypatch):
    """``examples/04_onnx_export.py`` labels what it showed (FEAT-038):
    ``executed`` (ONNX Runtime ran the graph on CPU and matched) only with
    the runtime installed; ``checker-only`` — with no parity claim —
    without it."""
    pytest.importorskip("onnx")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    namespace = runpy.run_path(str(ROOT / "examples" / "04_onnx_export.py"), run_name="__nnx_example_smoke__")
    if importlib.util.find_spec("onnxruntime") is not None:
        executed = namespace["registered_module_variant"]()
        assert executed["level"] == "executed" and executed["parity"] is True
        assert executed["provider"] == "CPUExecutionProvider" and executed["max_abs_error"] < 1e-5
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # not installed
    checker_only = namespace["registered_module_variant"]()
    assert checker_only == {
        "level": "checker-only",
        "parity": None,
        "max_abs_error": None,
        "provider": None,
        "rtol": 1e-4,
        "atol": 1e-5,
    }


def test_onnx_sample_never_claims_parity_without_the_runtime(tmp_path, monkeypatch, capsys):
    """The output contract: without ONNX Runtime every result line says
    checker-only and none claims a match; with it, every line says executed."""
    pytest.importorskip("onnx")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    namespace = runpy.run_path(str(ROOT / "examples" / "04_onnx_export.py"), run_name="__nnx_example_smoke__")
    namespace["main"]()
    lines = [line for line in capsys.readouterr().out.splitlines() if "checker-only" in line or "executed" in line]
    assert len(lines) == 2 and all("checker-only" in line and "no parity is claimed" in line for line in lines)
    assert not any("match" in line for line in lines)
    describe = namespace["describe"]
    executed = {
        "level": "executed",
        "parity": True,
        "max_abs_error": 0.0,
        "provider": "CPU",
        "rtol": 1e-4,
        "atol": 1e-5,
    }
    assert "executed" in describe(executed) and "rtol=0.0001" in describe(executed)
    assert "DO NOT match" in describe({**executed, "parity": False, "max_abs_error": None})


def test_structural_only_examples_never_claim_runtime_parity():
    """Quantized exports, Netron files and GGUF / Ollama artifacts stay
    structural or container-only (FEAT-038): their samples never claim a
    runtime match."""
    quantized = (ROOT / "examples" / "12_quantize_int8.py").read_text(encoding="utf-8")
    assert "structural only" in quantized and "no parity claimed" in quantized
    from nnx.export_conformance import PROFILES

    assert sorted(PROFILES) == ["feedfwd-fp32-dynamo", "feedfwd-fp32-torchscript"]
    assert all(profile.dtype == "float32" and profile.model["net"] == "feed_fwd" for profile in PROFILES.values())
