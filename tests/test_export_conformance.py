"""Executable ONNX export conformance profiles (FEAT-038).

Structural validity (``onnx.checker``) and numerical parity (ONNX Runtime
on CPU against the native model) are separate, classified outcomes. The
runtime-backed tests need ``onnxruntime`` (and ``onnxscript`` for the
dynamo profile); with ``NNX_REQUIRE_ONNX_RUNTIME=1`` — set by the CI
``export-conformance`` job — a missing runtime fails them instead of
skipping, so that job cannot pass by skipping.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import nnx.export_conformance as conformance
from nnx.export_conformance import (
    EXIT_CODES,
    FORMAT,
    PROFILES,
    ConformanceError,
    Profile,
    execute,
    exit_code,
    run_profile,
    run_profiles,
    save_report,
    validate_record,
    verify_artifacts,
)
from nnx.nn.nn_model import NNModel
from tests._repo_files import repo_file

ROOT = Path(__file__).resolve().parents[1]


def _require(*modules: str) -> None:
    """Import ``modules``; a missing one skips — or fails when the CI
    conformance job sets ``NNX_REQUIRE_ONNX_RUNTIME=1``."""
    for name in modules:
        if os.environ.get("NNX_REQUIRE_ONNX_RUNTIME") == "1":
            __import__(name)  # an ImportError fails the test: no skip
        else:
            pytest.importorskip(name)


def _runtime(profile: str = "feedfwd-fp32-torchscript") -> None:
    _require(*PROFILES[profile].requires)


# --- AC2 / AC3: parity under both exporters, shape cases -------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_both_exporters_match_native_logits_in_onnx_runtime(name, tmp_path):
    _runtime(name)
    record = run_profile(name, tmp_path)
    validate_record(record)
    assert record["status"] == "passed" and record["failure"] is None and record["level"] == "executed", record
    assert record["tolerances"] == {"rtol": 1e-4, "atol": 1e-5}
    assert record["runtime"]["provider"] == "CPUExecutionProvider" and record["dtype"] == "float32"
    assert record["stages"]["load"]["detail"]["providers"][0] == "CPUExecutionProvider"
    matched = [case for case in record["input_cases"] if case["expect"] == "match"]
    assert [case["shape"][0] for case in matched] == [1, 2, 3, 7]  # exported at batch 2, dynamic batch
    assert all(case["outcome"] == "match" and case["max_abs_error"] < 1e-5 for case in matched)
    assert record["config"]["input_dim"] == 4 and record["config"]["hidden_dims"] == [8]
    assert record["config"]["output_dim"] == 2 and record["config"]["dropout"] == 0.0
    # The weights are deterministic and nontrivial.
    net = conformance.build_model().net
    weights = torch.cat([p.detach().reshape(-1) for p in net.parameters()])
    assert weights.abs().min() > 0 and weights.unique().numel() == weights.numel()
    assert conformance._weights_digest(net) == record["config"]["weights_sha256"]


def test_shape_cases_and_a_wrong_width_is_a_classified_refusal(tmp_path):
    _runtime()
    record = run_profile("feedfwd-fp32-torchscript", tmp_path)
    (wrong,) = [case for case in record["input_cases"] if case["expect"] == "rejected"]
    assert wrong["shape"] == [3, 5] and wrong["outcome"] == "rejected" and wrong["passed"]
    assert "INVALID_ARGUMENT" in wrong["detail"] or "dimension" in wrong["detail"].lower()
    signature = record["stages"]["check"]["detail"]
    assert signature["inputs"] == [{"name": "features", "dtype": "float", "shape": ["batch", 4]}]
    assert signature["outputs"][0]["name"] == "logits" and signature["outputs"][0]["shape"][1] == 2
    assert record["opset"] == {"requested": 17, "model": {"ai.onnx": 17}}

    # A static batch dimension fails the other batches as an input contract, not a mismatch.
    static = run_profile(Profile("static-batch", "torchscript", dynamic_batch=False), tmp_path)
    assert static["failure"] == "input-contract" and static["level"] == "structural"
    failing = {case["name"] for case in static["input_cases"] if not case["passed"]}
    assert failing == {"batch-1", "batch-3", "batch-7"}
    assert exit_code({"profiles": [static]}) == EXIT_CODES["input-contract"]


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_export_keeps_parameters_gradients_and_mixed_modes(name, tmp_path):
    _runtime(name)
    record = run_profile(name, tmp_path)
    assert record["model_state"] == {"parameters": True, "gradients": True, "modes": True}


# --- AC1: the record, its schema and its artifacts ---------------------------------------------------------------


def test_the_record_carries_every_fact_a_parity_claim_depends_on(tmp_path):
    _runtime()
    record = run_profile("feedfwd-fp32-torchscript", tmp_path, source_revision="abc123")
    assert record["format"] == FORMAT and record["profile"] == "feedfwd-fp32-torchscript"
    assert record["source"]["revision"] == "abc123" and record["source"]["origin"] == "given"
    assert record["exporter"] == {
        "name": "torchscript",
        "options": {
            "exporter": "torchscript",
            "dynamo": False,
            "opset_version": 17,
            "dynamic_batch": True,
            "export_batch": 2,
            "input_names": ["features"],
            "output_names": ["logits"],
        },
    }
    for package in ("python", "nnx", "torch", "numpy", "onnx", "onnxruntime"):
        assert record["versions"][package], package
    assert record["runtime"]["version"] == record["versions"]["onnxruntime"]
    (model,) = record["artifacts"]["files"]
    assert model["path"] == "model.onnx" and model["role"] == "model" and len(model["sha256"]) == 64
    assert set(record["stages"]) == {"dependencies", "export", "check", "load", "parity"}
    assert all(stage["status"] == "passed" for stage in record["stages"].values())
    # Strict JSON round trip.
    text = json.dumps(record, allow_nan=False)
    validate_record(json.loads(text))
    report = run_profiles(tmp_path / "again", ["feedfwd-fp32-torchscript"], source_revision="abc123")
    save_report(report, tmp_path / "conformance.json")
    assert json.loads((tmp_path / "conformance.json").read_text())["status"] == "passed"


def _valid_record(tmp_path):
    _runtime()
    return run_profile("feedfwd-fp32-torchscript", tmp_path)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: r.pop("tolerances"), "lacks \\['tolerances'\\]"),
        (lambda r: r.update(extra=1), "unknown keys"),
        (lambda r: r.update(format="nnx.export-conformance/0"), "format"),
        (lambda r: r["stages"]["load"].update(status="skipped"), "status 'skipped'"),
        (lambda r: r["stages"]["parity"].update(status="failed", failure="hash-mismatch"), "not one of"),
        (lambda r: r.update(level="structural"), "level 'structural' contradicts"),
        (lambda r: r.update(status="failed"), "status 'failed' contradicts"),
        (lambda r: r["artifacts"]["files"][0].update(sha256="xyz"), "malformed sha256"),
        (lambda r: r["config"].update(weights_sha256="x"), "weights_sha256 is malformed"),
        (lambda r: r["config"].update(weights_sha256=None), "weights_sha256 is malformed"),
        (lambda r: r["config"].update(weights_sha256="a" * 63), "weights_sha256 is malformed"),
        (lambda r: r["config"].update(weights_sha256="A" * 64), "weights_sha256 is malformed"),
        (lambda r: r["tolerances"].update(rtol=float("nan")), "finite"),
        (
            lambda r: (
                r["stages"]["check"].update(status="failed", failure="checker-error"),
                r.update(status="failed", failure="checker-error", level="none"),
            ),
            "ran after an earlier stage",
        ),
    ],
)
def test_malformed_records_are_refused_by_name(tmp_path, edit, message):
    record = _valid_record(tmp_path)
    edit(record)
    with pytest.raises(ConformanceError, match=message):
        validate_record(record)


def test_a_tampered_artifact_is_rejected_before_it_is_loaded(tmp_path, monkeypatch):
    record = _valid_record(tmp_path)  # requires (or skips without) the runtime before any import
    import onnx
    import onnxruntime

    verify_artifacts(record, tmp_path)
    path = tmp_path / "feedfwd-fp32-torchscript" / "model.onnx"
    model = onnx.load(str(path))
    weights = onnx.numpy_helper.to_array(model.graph.initializer[0]).copy()
    weights.flat[0] += 1.0
    model.graph.initializer[0].CopyFrom(onnx.numpy_helper.from_array(weights, model.graph.initializer[0].name))
    onnx.save(model, str(path))
    with pytest.raises(ConformanceError, match="model.onnx: hash mismatch"):
        verify_artifacts(record, tmp_path)

    sessions = []
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: sessions.append(a) or None)
    replay = execute(record, tmp_path)
    assert replay["stages"]["load"]["status"] == "failed" and replay["stages"]["load"]["failure"] == "hash-mismatch"
    assert replay["stages"]["parity"]["status"] == "not_run" and replay["level"] == "structural"
    assert all(case["outcome"] is None and case["passed"] is False for case in replay["input_cases"])  # none stale
    validate_record(replay)
    assert replay["failure"] == "hash-mismatch" and sessions == []  # nothing was loaded
    assert exit_code({"profiles": [replay]}) == EXIT_CODES["hash-mismatch"]


def test_an_untampered_record_executes_again_and_extra_or_missing_files_are_refused(tmp_path):
    record = _valid_record(tmp_path)
    again = execute(record, tmp_path)
    assert again["status"] == "passed" and again["level"] == "executed"
    folder = tmp_path / "feedfwd-fp32-torchscript"
    (folder / "weights.bin").write_bytes(b"\0")
    with pytest.raises(ConformanceError, match="weights.bin: not in the record"):
        verify_artifacts(record, tmp_path)
    (folder / "weights.bin").unlink()
    (folder / "model.onnx").unlink()
    with pytest.raises(ConformanceError, match="model.onnx: missing"):
        verify_artifacts(record, tmp_path)
    escaped = json.loads(json.dumps(record))
    escaped["artifacts"]["directory"] = "../elsewhere"
    with pytest.raises(ConformanceError, match="plain name"):
        verify_artifacts(escaped, tmp_path)


def test_external_tensors_are_hashed_and_checked(tmp_path, monkeypatch):
    _runtime()
    import onnx

    original = NNModel.to_onnx

    def external(self, path, *args, **kwargs):
        written = original(self, path, *args, **kwargs)
        model = onnx.load(written)
        onnx.save_model(model, written, save_as_external_data=True, location="weights.bin", size_threshold=0)
        return written

    monkeypatch.setattr(NNModel, "to_onnx", external)
    record = run_profile("feedfwd-fp32-torchscript", tmp_path)
    assert record["status"] == "passed", record
    roles = {entry["path"]: entry["role"] for entry in record["artifacts"]["files"]}
    assert roles == {"model.onnx": "model", "weights.bin": "external-data"}
    data = tmp_path / "feedfwd-fp32-torchscript" / "weights.bin"
    data.write_bytes(data.read_bytes()[:-4] + b"\x00\x00\x80\x3f")  # one float changed: same size
    replay = execute(record, tmp_path)
    assert replay["failure"] == "hash-mismatch" and "weights.bin: hash mismatch" in replay["stages"]["load"]["detail"]


# --- AC4: separated, classified failures --------------------------------------------------------------------------


def test_a_missing_runtime_is_a_failure_never_a_skip(tmp_path, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        conformance.importlib.util, "find_spec", lambda name, *a: None if name == "onnxruntime" else real(name, *a)
    )
    record = run_profile("feedfwd-fp32-torchscript", tmp_path)
    validate_record(record)
    assert record["stages"]["dependencies"]["failure"] == "missing-dependency"
    assert "onnxruntime" in record["stages"]["dependencies"]["detail"]
    assert all(record["stages"][s]["status"] == "not_run" for s in ("export", "check", "load", "parity"))
    assert record["status"] == "failed" and record["level"] == "none"
    assert exit_code({"profiles": [record]}) == EXIT_CODES["missing-dependency"] == 10


def test_export_check_load_and_parity_failures_are_classified(tmp_path, monkeypatch):
    _runtime()
    import onnx
    import onnxruntime

    def raise_(*args, **kwargs):
        raise RuntimeError("exporter dispatch failed")

    with monkeypatch.context() as patch:
        patch.setattr(NNModel, "to_onnx", raise_)
        record = run_profile("feedfwd-fp32-torchscript", tmp_path / "export")
    assert record["failure"] == "export-error" and "dispatch failed" in record["stages"]["export"]["detail"]["error"]
    assert record["stages"]["check"]["status"] == "not_run" and exit_code({"profiles": [record]}) == 11

    original = NNModel.to_onnx

    def mutate(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        with torch.no_grad():
            next(self.net.parameters()).add_(1.0)
        return out

    with monkeypatch.context() as patch:
        patch.setattr(NNModel, "to_onnx", mutate)
        record = run_profile("feedfwd-fp32-torchscript", tmp_path / "state")
    assert record["failure"] == "state-changed" and record["model_state"]["parameters"] is False

    with monkeypatch.context() as patch:
        patch.setattr(onnx.checker, "check_model", raise_)
        record = run_profile("feedfwd-fp32-torchscript", tmp_path / "check")
    assert record["failure"] == "checker-error" and record["stages"]["load"]["status"] == "not_run"
    assert exit_code({"profiles": [record]}) == 13

    with monkeypatch.context() as patch:
        patch.setattr(onnxruntime, "InferenceSession", raise_)
        record = run_profile("feedfwd-fp32-torchscript", tmp_path / "load")
    assert record["failure"] == "load-error" and record["level"] == "structural"
    assert exit_code({"profiles": [record]}) == 15

    real_session = onnxruntime.InferenceSession

    class Drifting:
        def __init__(self, *args, **kwargs):
            self.inner = real_session(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def run(self, *args, **kwargs):
            return [out + np.float32(1e-3) for out in self.inner.run(*args, **kwargs)]

    with monkeypatch.context() as patch:
        patch.setattr(onnxruntime, "InferenceSession", Drifting)
        record = run_profile("feedfwd-fp32-torchscript", tmp_path / "parity")
    assert record["failure"] == "mismatch" and record["level"] == "structural"
    assert all(case["outcome"] == "mismatch" for case in record["input_cases"] if case["expect"] == "match")
    assert exit_code({"profiles": [record]}) == 16
    for failure in ("export-error", "state-changed", "checker-error", "load-error", "mismatch"):
        assert failure in EXIT_CODES
    assert len(set(EXIT_CODES.values())) == len(EXIT_CODES) and min(EXIT_CODES.values()) > 2  # clear of 1 / 2


def test_a_profile_downloads_nothing(tmp_path, monkeypatch):
    _runtime()

    def offline(*args, **kwargs):
        raise AssertionError("a conformance profile opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", offline)
    monkeypatch.setattr(socket, "create_connection", offline)
    assert run_profile("feedfwd-fp32-torchscript", tmp_path)["status"] == "passed"


def test_a_profile_owns_its_artifact_directory(tmp_path):
    (tmp_path / "feedfwd-fp32-torchscript").mkdir()
    (tmp_path / "feedfwd-fp32-torchscript" / "stale.onnx").write_bytes(b"x")
    with pytest.raises(ConformanceError, match="new or empty directory"):
        run_profile("feedfwd-fp32-torchscript", tmp_path)
    with pytest.raises(ConformanceError, match="unknown profile"):
        run_profile("resnet-int8", tmp_path)


def test_core_import_does_not_need_the_runtime_extra():
    """With onnx / onnxruntime / onnxscript absent, ``import nnx`` and the
    conformance module import, and no NNx module even tries to import them
    (a spy on ``__import__`` that names the importing module). Torch's own
    guarded probes are not NNx's: torch 2.4's ``torch.onnx`` tries
    ``import onnx`` and ``import onnxscript`` and carries on without them.
    Two controls check the spy: an import from an NNx module is seen, one
    from a third-party module is not."""
    code = (
        "import builtins, sys\n"
        "blocked = ('onnx', 'onnxruntime', 'onnxscript')\n"
        "for name in blocked:\n"
        "    sys.modules[name] = None  # not installed\n"
        "tried, real = [], builtins.__import__\n"
        "def spy(name, globals=None, *args, **kwargs):\n"
        "    importer = str((globals or {}).get('__name__', ''))\n"
        "    if name.split('.')[0] in blocked and importer.split('.')[0] == 'nnx':\n"
        "        tried.append(name)\n"
        "    return real(name, globals, *args, **kwargs)\n"
        "builtins.__import__ = spy\n"
        "import nnx, nnx.export_conformance as c\n"
        "assert c.PROFILES and tried == [], tried\n"
        "probe = 'try:\\n    import {}\\nexcept ImportError:\\n    pass\\n'\n"
        "exec(probe.format('onnx'), {'__name__': 'torch.onnx._internal.onnxruntime'})\n"
        "assert tried == [], tried  # a dependency's guarded probe is not NNx's\n"
        "exec(probe.format('onnxruntime'), {'__name__': 'nnx._probe'})\n"
        "assert tried == ['onnxruntime'], tried  # the spy sees NNx's own imports\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=120)


# --- AC5: the script ------------------------------------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location(
        "check_export_conformance", repo_file("scripts", "check_export_conformance.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_the_script_runs_a_profile_and_keeps_its_evidence(name, tmp_path, capsys):
    _runtime(name)
    output = tmp_path / "conformance.json"
    assert _script().main(["--output", str(output), "--profile", name, "--source-revision", "deadbeef"]) == 0
    report = json.loads(output.read_text())
    (record,) = report["profiles"]
    assert report["status"] == "passed" and record["source"]["revision"] == "deadbeef"
    folder = tmp_path / "conformance-artifacts" / name
    assert sorted(os.listdir(folder)) == sorted(entry["path"] for entry in record["artifacts"]["files"])
    verify_artifacts(record, tmp_path / "conformance-artifacts")
    assert f"{name}: passed; level=executed" in capsys.readouterr().out


def test_the_script_fails_without_the_runtime(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    output = tmp_path / "missing.json"
    assert _script().main(["--output", str(output)]) == EXIT_CODES["missing-dependency"]
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert {r["failure"] for r in report["profiles"]} == {"missing-dependency"}
    assert "missing-dependency" in capsys.readouterr().out


# --- review round 1 ------------------------------------------------------------------------------------------


class _Session:
    """Wraps a real session, rewriting what ``run`` gives back."""

    def __init__(self, inner, run):
        self.inner, self._run = inner, run

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def run(self, *args, **kwargs):
        return self._run(self.inner, *args, **kwargs)


def _patched_session(monkeypatch, run):
    import onnxruntime

    real = onnxruntime.InferenceSession
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: _Session(real(*a, **k), run))


def test_a_non_finite_output_is_a_mismatch_and_the_report_is_kept(tmp_path, monkeypatch, capsys):
    _runtime()
    _patched_session(monkeypatch, lambda inner, *a, **k: [out * np.float32("nan") for out in inner.run(*a, **k)])
    output = tmp_path / "nan.json"
    assert _script().main(["--output", str(output), "--profile", "feedfwd-fp32-torchscript"]) == EXIT_CODES["mismatch"]
    (record,) = json.loads(output.read_text())["profiles"]
    bad = [case for case in record["input_cases"] if case["expect"] == "match"]
    assert all(case["failure"] == "mismatch" and "non-finite" in case["detail"] for case in bad)
    assert all(case["max_abs_error"] is None for case in bad)


def test_only_an_invalid_argument_counts_as_the_runtime_refusing_an_input(tmp_path, monkeypatch):
    _runtime()

    def crash(inner, *args, **kwargs):
        raise RuntimeError("kernel crashed")

    with monkeypatch.context() as patch:
        _patched_session(patch, crash)
        record = run_profile("feedfwd-fp32-torchscript", tmp_path / "crash")
    assert record["failure"] == "runtime-error" and exit_code({"profiles": [record]}) == EXIT_CODES["runtime-error"]
    assert not any(case["passed"] for case in record["input_cases"])  # the width case is not "refused" either

    def two_outputs_any_width(inner, output_names, feed):
        (name,) = feed
        rows = feed[name].shape[0]
        return [np.zeros((rows, 2), np.float32), np.zeros((rows, 2), np.float32)]

    with monkeypatch.context() as patch:
        _patched_session(patch, two_outputs_any_width)
        record = run_profile("feedfwd-fp32-torchscript", tmp_path / "two")
    width = next(case for case in record["input_cases"] if case["expect"] == "rejected")
    assert width["outcome"] == "accepted" and width["failure"] == "input-contract" and not width["passed"]
    assert record["failure"] == "mismatch"


def test_the_script_refuses_to_mix_old_artifacts_unless_forced(tmp_path, capsys):
    _runtime()
    script = _script()
    output = tmp_path / "out" / "nested" / "conformance.json"  # the parent is created
    args = ["--output", str(output), "--artifacts", str(tmp_path / "art"), "--profile", "feedfwd-fp32-torchscript"]
    assert script.main([*args, "--profile", "feedfwd-fp32-torchscript"]) == 0  # a repeated profile runs once
    assert len(json.loads(output.read_text())["profiles"]) == 1
    with pytest.raises(SystemExit) as refused:
        script.main(args)
    assert refused.value.code == 2 and "--force" in capsys.readouterr().err
    assert script.main([*args, "--force"]) == 0


def test_a_dot_relative_external_location_and_empty_warnings_are_handled(tmp_path, monkeypatch):
    _runtime()
    import warnings as warnings_

    import onnx

    original = NNModel.to_onnx

    def relocated(self, path, *args, **kwargs):
        warnings_.warn("", stacklevel=2)  # an empty warning message
        written = original(self, path, *args, **kwargs)
        model = onnx.load(written)
        onnx.save_model(model, written, save_as_external_data=True, location="./w.bin", size_threshold=0)
        return written

    monkeypatch.setattr(NNModel, "to_onnx", relocated)
    record = run_profile("feedfwd-fp32-torchscript", tmp_path)
    assert record["status"] == "passed", record["stages"]
    roles = {entry["path"]: entry["role"] for entry in record["artifacts"]["files"]}
    assert roles == {"model.onnx": "model", "w.bin": "external-data"}  # "./w.bin" is w.bin
    (tmp_path / "feedfwd-fp32-torchscript" / "sub").mkdir()
    (tmp_path / "feedfwd-fp32-torchscript" / "sub" / "extra.bin").write_bytes(b"x")
    with pytest.raises(ConformanceError, match="sub/extra.bin: not in the record"):
        verify_artifacts(record, tmp_path)


def test_the_validator_refuses_contradictions_without_crashing(tmp_path):
    record = _valid_record(tmp_path)

    def broken(edit):
        copy = json.loads(json.dumps(record))
        edit(copy)
        return copy

    cases = {
        "must hold status, failure and detail": lambda r: r["stages"].update(load="passed"),
        "a list of files": lambda r: r["artifacts"].update(files=None),
        "an input case did not": lambda r: r["input_cases"][0].update(passed=False, outcome="mismatch"),
        "model state was not kept": lambda r: r["model_state"].update(parameters=False),
        "carries the model's opset": lambda r: r["opset"].update(model=None),
        "lists its model file": lambda r: r["artifacts"].update(model="other.onnx"),
        "at least one valid case": lambda r: r.update(
            input_cases=[c for c in r["input_cases"] if c["expect"] != "match"]
        ),
        "leaves its directory": lambda r: r["artifacts"]["files"][0].update(path="../model.onnx"),
    }
    for message, edit in cases.items():
        with pytest.raises(ConformanceError, match=message):
            validate_record(broken(edit))
    with pytest.raises(ConformanceError, match="plain directory name"):
        Profile("a/b", "torchscript")


# --- review round 2 ------------------------------------------------------------------------------------------


def test_a_profile_declares_only_what_it_runs_and_tested_names_keep_their_settings(tmp_path):
    for settings in ({"dtype": "float16"}, {"runtime": "tensorrt"}, {"batches": ()}, {"rtol": float("nan")}):
        with pytest.raises(ConformanceError):
            Profile("custom", "torchscript", **settings)
    loose = Profile("feedfwd-fp32-torchscript", "torchscript", rtol=1.0, atol=1.0)
    with pytest.raises(ConformanceError, match="names a tested profile"):
        run_profile(loose, tmp_path)


def test_a_hand_edited_record_of_a_tested_profile_is_refused(tmp_path):
    record = _valid_record(tmp_path)
    loosened = json.loads(json.dumps(record))
    loosened["tolerances"] = {"rtol": 1e9, "atol": 1e9}
    loosened["input_cases"] = loosened["input_cases"][:1]
    with pytest.raises(ConformanceError, match="tested profile, but its settings differ"):
        validate_record(loosened)
    with pytest.raises(ConformanceError, match="tested profile"):
        execute(loosened, tmp_path)
    stale = json.loads(json.dumps(record))
    stale["stages"]["load"] = {"status": "failed", "failure": "load-error", "detail": "x"}
    stale["stages"]["parity"] = {"status": "not_run", "failure": None, "detail": None}
    stale.update(status="failed", failure="load-error", level="structural")
    with pytest.raises(ConformanceError, match="no input case may carry an outcome"):
        validate_record(stale)
    doubled = json.loads(json.dumps(record))
    doubled["artifacts"]["files"].append(dict(doubled["artifacts"]["files"][0]))
    with pytest.raises(ConformanceError, match="more than once"):
        validate_record(doubled)


def test_an_unreadable_artifact_and_a_broken_install_are_classified(tmp_path, monkeypatch):
    record = _valid_record(tmp_path)

    def unreadable(*args, **kwargs):
        raise PermissionError("denied")

    with monkeypatch.context() as patch:
        patch.setattr(conformance, "_sha256", unreadable)
        replay = execute(record, tmp_path)
    assert replay["failure"] == "hash-mismatch" and "PermissionError" in replay["stages"]["load"]["detail"]

    real = conformance.importlib.import_module

    def broken(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("DLL load failed")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(conformance.importlib, "import_module", broken)
    broke = run_profile("feedfwd-fp32-torchscript", tmp_path / "broken")
    assert broke["failure"] == "missing-dependency" and "onnxruntime" in broke["stages"]["dependencies"]["detail"]


# --- review round 3 ------------------------------------------------------------------------------------------


def test_every_record_states_only_what_this_module_runs(tmp_path):
    record = _valid_record(tmp_path)

    def edited(edit):
        copy_ = json.loads(json.dumps(record))
        edit(copy_)
        return copy_

    custom = edited(lambda r: r.update(profile="custom", dtype="float16"))
    custom["runtime"]["name"] = "tensorrt"
    custom["artifacts"]["directory"] = "custom"
    for broken, message in (
        (custom, "profiles run FP32 inputs in ONNX Runtime"),
        (edited(lambda r: r["runtime"]["session"].update(intra_op_num_threads=8)), "session options"),
        (edited(lambda r: r["exporter"].update(name="dynamo")), "exporter.name"),
        (edited(lambda r: r["opset"].update(requested=18)), "opset.requested"),
        (edited(lambda r: r["artifacts"].update(directory="elsewhere")), "settings differ"),
        (edited(lambda r: r["input_cases"][0].update(seed=2**70)), "malformed name, seed or shape"),
    ):
        with pytest.raises(ConformanceError, match=message):
            validate_record(broken)
    stale = edited(lambda r: None)
    stale["stages"]["load"] = {"status": "failed", "failure": "load-error", "detail": "x"}
    stale["stages"]["parity"] = {"status": "not_run", "failure": None, "detail": None}
    stale.update(status="failed", failure="load-error", level="structural")
    for case in stale["input_cases"]:
        case.update(outcome=None, passed=False)  # max_abs_error and detail left over
    with pytest.raises(ConformanceError, match="no input case may carry an outcome"):
        validate_record(stale)


def test_a_profile_is_compared_canonically_and_checked_before_it_runs(tmp_path):
    from nnx.export_conformance import MODEL

    with pytest.raises(ConformanceError, match="names a tested profile"):
        run_profile(Profile("feedfwd-fp32-torchscript", "torchscript", rtol=1e-3), tmp_path)
    assert not (tmp_path / "feedfwd-fp32-torchscript").exists()  # refused before anything was written
    same = Profile(
        "feedfwd-fp32-torchscript", "torchscript", model={**MODEL, "dropout": 0}
    )  # 0 and 0.0 are one setting
    assert conformance._profile_settings(same) == conformance._profile_settings(PROFILES["feedfwd-fp32-torchscript"])
    for settings in (
        {"export_batch": 0},
        {"export_batch": 2.0},
        {"dynamic_batch": 1},
        {"opset_version": "17"},
        {"provider": ""},
        {"rtol": 10**400},
        {"model": {**MODEL, "activation": "bogus"}},
        {"model": {**MODEL, "activation": []}},
        {"model": {**MODEL, "input_dim": 2**62}},
        {"model": {**MODEL, "weights": {**MODEL["weights"], "std": 10**400}}},
    ):
        with pytest.raises(ConformanceError):
            Profile("custom", "torchscript", **settings)  # type: ignore[arg-type]


def test_a_symlinked_artifact_never_verifies(tmp_path):
    record = _valid_record(tmp_path)
    folder = tmp_path / "feedfwd-fp32-torchscript"
    outside = tmp_path / "outside.onnx"
    (folder / "model.onnx").rename(outside)
    (folder / "model.onnx").symlink_to(outside)
    with pytest.raises(ConformanceError, match="a symlink"):
        verify_artifacts(record, tmp_path)


# --- review round 4 ------------------------------------------------------------------------------------------


def test_a_forged_custom_record_cannot_replay_as_executed(tmp_path):
    record = _valid_record(tmp_path)

    def custom(edit):
        forged = json.loads(json.dumps(record))
        forged["profile"] = forged["artifacts"]["directory"] = "custom"
        edit(forged)
        return forged

    for edit in (
        lambda r: r["exporter"].update(name="tensorrt") or r["exporter"]["options"].update(exporter="tensorrt"),
        lambda r: r["exporter"]["options"].update(dynamo=True),
        lambda r: r["exporter"]["options"].update(quantize="int8"),
        lambda r: r["exporter"]["options"].update(input_names=["x"]),
        lambda r: r["input_cases"][0].update(seed=1),
        lambda r: r["artifacts"].update(model="net.onnx"),
        lambda r: r["config"].update(activation=[]),
        lambda r: r["tolerances"].update(rtol=10**400),
    ):
        with pytest.raises(ConformanceError):
            validate_record(custom(edit))
    renamed = custom(lambda r: None)
    (tmp_path / "feedfwd-fp32-torchscript").rename(tmp_path / "custom")
    validate_record(renamed)  # a custom profile with the same settings is a profile this module runs
    assert execute(renamed, tmp_path)["level"] == "executed"
    (tmp_path / "linked").symlink_to(tmp_path / "custom")
    linked = json.loads(json.dumps(renamed))
    linked["profile"] = linked["artifacts"]["directory"] = "linked"
    with pytest.raises(ConformanceError, match="symlink"):
        verify_artifacts(linked, tmp_path)


# --- review round 5 ------------------------------------------------------------------------------------------


def _parity_failed_record(tmp_path, monkeypatch):
    """A genuine record whose parity failed: the runtime's logits drift."""
    _runtime()
    import onnxruntime

    real_session = onnxruntime.InferenceSession

    class Drifting:
        def __init__(self, *args, **kwargs):
            self.inner = real_session(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def run(self, *args, **kwargs):
            return [out + np.float32(1e-3) for out in self.inner.run(*args, **kwargs)]

    with monkeypatch.context() as patch:
        patch.setattr(onnxruntime, "InferenceSession", Drifting)
        return run_profile("feedfwd-fp32-torchscript", tmp_path)


def test_every_input_case_states_an_outcome_a_run_gives(tmp_path, monkeypatch):
    failed = _parity_failed_record(tmp_path / "failed", monkeypatch)
    validate_record(failed)
    assert failed["failure"] == "mismatch"
    passed = _valid_record(tmp_path / "passed")

    def edited(record, edit):
        copy_ = json.loads(json.dumps(record))
        edit(copy_)
        return copy_

    for record, edit in (
        (failed, lambda r: r["input_cases"][0].update(failure=["x"])),
        (failed, lambda r: r["input_cases"][0].update(failure={})),
        (failed, lambda r: r["stages"]["parity"].update(failure=[])),
        (failed, lambda r: r["input_cases"][0].update(failure="bogus", outcome="teleported")),
        (failed, lambda r: r["input_cases"][0].update(failure="hash-mismatch")),  # a load-stage class
        (failed, lambda r: r["input_cases"][0].update(failure="runtime-error")),  # not what a mismatch names
        (failed, lambda r: r["input_cases"][-1].update(outcome="mismatch")),  # a refused input never mismatches
        (passed, lambda r: r["input_cases"][0].update(max_abs_error=-3)),
        (passed, lambda r: r["input_cases"][0].update(max_rel_error="big")),
        (passed, lambda r: r["input_cases"][0].update(max_abs_error=None)),
        (passed, lambda r: r["input_cases"][0].update(detail=7)),
        (passed, lambda r: r["input_cases"][0].update(detail="forged")),  # a compared case has no detail
        (passed, lambda r: r["input_cases"][-1].update(max_abs_error=0.5)),  # a refused input compares nothing
        (  # the stage names the highest-ranked class its failing cases name (mismatch before input-contract)
            failed,
            lambda r: (
                r["input_cases"][-1].update(outcome="accepted", passed=False, failure="input-contract")
                or r["stages"]["parity"].update(failure="input-contract")
                or r.update(failure="input-contract")
            ),
        ),
    ):
        broken = edited(record, edit)
        with pytest.raises(ConformanceError):
            validate_record(broken)
        with pytest.raises(ConformanceError):
            save_report({"format": FORMAT, "status": "failed", "profiles": [broken]}, tmp_path / "report.json")


def test_a_profile_and_a_record_are_bounded_in_size(tmp_path):
    from nnx.export_conformance import MODEL

    for settings in (
        {"batches": (1, 2**40)},
        {"batches": 5},
        {"batches": tuple(range(1, 66))},
        {"model": {**MODEL, "input_dim": 4096}, "batches": (4097,)},  # one batch of more than 2**24 values
        {"model": {**MODEL, "input_dim": 65536, "hidden_dims": [16384]}},  # 2**30 weights
        {"model": {**MODEL, "hidden_dims": [1] * 65}},
    ):
        with pytest.raises(ConformanceError):
            Profile("custom", "torchscript", **settings)  # type: ignore[arg-type]
    with pytest.raises(ConformanceError, match="integer in 1"):
        Profile("custom", "torchscript", opset_version=70000)
    Profile("custom", "torchscript", model={**MODEL, "hidden_dims": [8] * 64}, batches=tuple(range(1, 65)))

    record = _valid_record(tmp_path)

    def custom(edit):
        forged = json.loads(json.dumps(record))
        forged["profile"] = forged["artifacts"]["directory"] = "custom"
        edit(forged)
        return forged

    huge_batch = custom(lambda r: r["input_cases"][0].update(shape=[2**40, 4], seed=1000 + 2**40))
    huge_model = custom(lambda r: r["config"].update(input_dim=65536, hidden_dims=[16384]))
    for forged in (huge_batch, huge_model):
        with pytest.raises(ConformanceError):
            validate_record(forged)
        with pytest.raises(ConformanceError):
            execute(forged, tmp_path)  # refused before anything is built or run
    with pytest.raises(ConformanceError, match="malformed"):
        verify_artifacts({}, tmp_path)
    wide = {**MODEL, "input_dim": 1, "hidden_dims": [65536], "output_dim": 1}  # 2**32 hidden activations
    with pytest.raises(ConformanceError, match="through a layer of 65536 units"):
        Profile("wide", "torchscript", batches=(65536,), model=wide)
    grown = Profile("grown", "torchscript", batches=(65536,), model={**MODEL, "input_dim": 256, "hidden_dims": [8]})
    grown.model["input_dim"] = 257  # changed in place after it was checked
    with pytest.raises(ConformanceError, match="holds more than"):
        run_profile(grown, tmp_path / "grown")
    nan = json.loads(json.dumps(record))
    nan["stages"]["check"]["detail"]["producer"] = float("nan")
    with pytest.raises(ConformanceError, match="strict JSON"):
        validate_record(nan)
    with pytest.raises(ConformanceError, match="strict JSON"):
        save_report({"format": FORMAT, "status": "passed", "profiles": [nan]}, tmp_path / "report.json")


def test_a_record_holds_only_what_run_profile_writes(tmp_path, monkeypatch):
    record = _valid_record(tmp_path / "valid")

    def edited(edit):
        copy_ = json.loads(json.dumps(record))
        edit(copy_)
        return copy_

    for edit, message in (
        (lambda r: r.update(source={1: "a"}), "round trip"),
        (lambda r: r["source"].update(revision=("a", "b")), "round trip"),
        (lambda r: r["exporter"].update(quantization="int8"), "exporter must hold exactly"),
        (lambda r: r["runtime"].update(provider_options={"device_id": 1}), "runtime must hold exactly"),
        (lambda r: r["opset"].update(converted_to=18), "opset must hold exactly"),
        (lambda r: r["opset"].update(requested=17.0), "opset.requested"),
        (lambda r: r["model_state"].update(modes="yes"), "model_state holds bools"),
    ):
        with pytest.raises(ConformanceError, match=message):
            validate_record(edited(edit))

    real = importlib.util.find_spec
    with monkeypatch.context() as patch:
        patch.setattr(
            conformance.importlib.util, "find_spec", lambda name, *a: None if name == "onnxruntime" else real(name, *a)
        )
        missing = run_profile("feedfwd-fp32-torchscript", tmp_path / "missing")
    validate_record(missing)
    missing["model_state"] = {"parameters": True, "gradients": True, "modes": True}
    with pytest.raises(ConformanceError, match="no model_state"):
        validate_record(missing)

    original = NNModel.to_onnx

    def mutate(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        with torch.no_grad():
            next(self.net.parameters()).add_(1.0)
        return out

    with monkeypatch.context() as patch:
        patch.setattr(NNModel, "to_onnx", mutate)
        changed = run_profile("feedfwd-fp32-torchscript", tmp_path / "changed")
    validate_record(changed)
    changed["model_state"]["parameters"] = True
    with pytest.raises(ConformanceError, match="model_state says otherwise"):
        validate_record(changed)

    def raise_(*args, **kwargs):
        raise RuntimeError("exporter dispatch failed")

    with monkeypatch.context() as patch:
        patch.setattr(NNModel, "to_onnx", raise_)
        failed = run_profile("feedfwd-fp32-torchscript", tmp_path / "failed")
    validate_record(failed)
    failed["model_state"] = {"parameters": False, "gradients": True, "modes": True}
    with pytest.raises(ConformanceError, match="failed as state-changed"):
        validate_record(failed)  # a changed model state is reported as state-changed, never export-error

    # A tested profile changed in place is no longer what its name stands for.
    monkeypatch.setitem(PROFILES["feedfwd-fp32-torchscript"].model, "dropout", 0.5)
    with pytest.raises(ConformanceError, match="names a tested profile"):
        run_profile("feedfwd-fp32-torchscript", tmp_path / "mutated")
    assert not (tmp_path / "mutated").exists()


def test_a_non_finite_native_reference_is_a_classified_mismatch(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nnx.export_conformance import MODEL

    _runtime()
    real = NNModel.predict

    def overflowing(self, *args, **kwargs):
        logits = torch.as_tensor(real(self, *args, **kwargs).logits)
        return SimpleNamespace(logits=torch.full_like(logits, float("inf")))

    with monkeypatch.context() as patch:
        patch.setattr(NNModel, "predict", overflowing)
        record = run_profile("feedfwd-fp32-torchscript", tmp_path / "native")
    validate_record(record)  # every outcome is classified, as strict JSON
    assert record["failure"] == "mismatch"
    compared = [case for case in record["input_cases"] if case["expect"] == "match"]
    assert all(case["detail"] == "the native model gave non-finite logits" for case in compared)
    save_report({"format": FORMAT, "status": "failed", "profiles": [record]}, tmp_path / "report.json")

    # Weights so large that gradients overflow to NaN: NaN stays NaN, so the export changed nothing.
    huge = Profile("huge", "torchscript", model={**MODEL, "weights": {**MODEL["weights"], "std": 1e30}})
    record = run_profile(huge, tmp_path / "huge")
    validate_record(record)
    assert record["stages"]["export"]["status"] == "passed" and record["failure"] == "mismatch"


def test_a_model_changed_in_place_or_by_a_failing_exporter_is_recorded_as_it_ran(tmp_path, monkeypatch):
    _runtime()
    tuple_dims = Profile("tuple-dims", "torchscript")
    tuple_dims.model["hidden_dims"] = (8,)  # changed in place: still the same model
    record = run_profile(tuple_dims, tmp_path)
    validate_record(record)
    assert record["config"]["hidden_dims"] == [8] and record["level"] == "executed"

    original = NNModel.to_onnx

    def mutate_then_raise(self, *args, **kwargs):
        original(self, *args, **kwargs)
        with torch.no_grad():
            next(self.net.parameters()).add_(1.0)
        raise RuntimeError("exporter gave up")

    with monkeypatch.context() as patch:
        patch.setattr(NNModel, "to_onnx", mutate_then_raise)
        record = run_profile("feedfwd-fp32-torchscript", tmp_path / "raised")
    validate_record(record)
    assert record["failure"] == "state-changed" and record["model_state"]["parameters"] is False
    assert "exporter gave up" in record["stages"]["export"]["detail"]["export"]["error"]
    assert exit_code({"profiles": [record]}) == EXIT_CODES["state-changed"]


def test_the_profile_model_and_inputs_are_fp32_under_any_default_dtype():
    """The records say FP32: the model and the case inputs are built in
    float32 even when torch's default dtype is float64."""
    from nnx.export_conformance import _inputs, build_model

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = build_model()
        assert {p.dtype for p in model.net.parameters()} == {torch.float32}
        assert _inputs({"shape": [2, 4], "seed": 0}).dtype == torch.float32
    finally:
        torch.set_default_dtype(previous)
    fp32 = build_model()
    assert all(torch.equal(a, b) for a, b in zip(model.net.parameters(), fp32.net.parameters(), strict=True))


# --- FIX-098 (#566): every name and profile folder is checked before anything is written ------------------------

_HELD = {"empty": [], "full": ["old.onnx"], "dangling": None}  # what each link target holds


def _linked_folder(tmp_path, target, name="feedfwd-fp32-torchscript"):
    """``<tmp>/artifacts/<name>`` as a symlink to an empty folder, a folder
    holding ``old.onnx``, or a path that does not exist."""
    elsewhere = tmp_path / "elsewhere"
    if target != "dangling":
        elsewhere.mkdir(parents=True)
    if target == "full":
        (elsewhere / "old.onnx").write_bytes(b"old")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / name).symlink_to(elsewhere, target_is_directory=True)
    return artifacts, elsewhere


def _untouched(link, elsewhere, target):
    held = sorted(os.listdir(elsewhere)) if os.path.lexists(elsewhere) else None
    return link.is_symlink() and os.readlink(link) == str(elsewhere) and held == _HELD[target]


def test_an_unknown_later_name_is_refused_before_any_profile_runs(tmp_path):
    with pytest.raises(ConformanceError, match="unknown profile 'feedfwd-fp32-dynamoo'"):
        run_profiles(tmp_path, ["feedfwd-fp32-torchscript", "feedfwd-fp32-dynamoo"], source_revision="test")
    assert not (tmp_path / "feedfwd-fp32-torchscript").exists()  # nothing was written, so the retry is not blocked
    _runtime("feedfwd-fp32-dynamo")
    report = run_profiles(tmp_path, ["feedfwd-fp32-torchscript", "feedfwd-fp32-dynamo"], source_revision="test")
    assert report["status"] == "passed"


def test_a_later_profile_folder_is_checked_before_any_profile_runs(tmp_path, monkeypatch):
    held = tmp_path / "feedfwd-fp32-dynamo"
    held.mkdir()
    (held / "stale.txt").write_text("", encoding="utf-8")
    with pytest.raises(ConformanceError, match="new or empty directory"):
        run_profiles(tmp_path, source_revision="test")  # the default names: the torchscript profile comes first
    assert not (tmp_path / "feedfwd-fp32-torchscript").exists() and os.listdir(held) == ["stale.txt"]

    artifacts, elsewhere = _linked_folder(tmp_path / "linked", "empty", name="feedfwd-fp32-dynamo")
    with pytest.raises(ConformanceError, match="is a symlink"):
        run_profiles(artifacts, source_revision="test")
    assert os.listdir(artifacts) == ["feedfwd-fp32-dynamo"]
    assert _untouched(artifacts / "feedfwd-fp32-dynamo", elsewhere, "empty")

    # Every check run_profile makes before it writes runs for each name first: a later tested profile changed in place.
    monkeypatch.setitem(PROFILES["feedfwd-fp32-dynamo"].model, "dropout", 0.5)
    with pytest.raises(ConformanceError, match="'feedfwd-fp32-dynamo' names a tested profile"):
        run_profiles(tmp_path / "fresh", source_revision="test")
    assert not (tmp_path / "fresh").exists()


@pytest.mark.parametrize("target", sorted(_HELD))
def test_a_symlinked_profile_folder_is_refused_before_anything_is_written(tmp_path, target):
    """A link to an empty or a filled folder, or a dangling one, is refused
    (never followed or cleared), by ``run_profile`` and ``run_profiles``."""
    name = "feedfwd-fp32-torchscript"
    artifacts, elsewhere = _linked_folder(tmp_path, target)
    with pytest.raises(ConformanceError, match="is a symlink"):
        run_profile(name, artifacts, source_revision="test")
    with pytest.raises(ConformanceError, match="is a symlink"):
        run_profiles(artifacts, [name], source_revision="test")
    assert _untouched(artifacts / name, elsewhere, target)


@pytest.mark.parametrize("call", ["listdir", "lstat"])  # it cannot be listed, or looked up (its parent unsearchable)
def test_an_unreadable_profile_folder_is_a_refusal_not_a_raw_error(tmp_path, monkeypatch, call):
    name = "feedfwd-fp32-torchscript"
    folder = tmp_path / name
    folder.mkdir()
    real = getattr(os, call)

    def denied(path=".", **kwargs):
        if os.fspath(path) == str(folder):
            raise PermissionError(13, "Permission denied", str(folder))
        return real(path, **kwargs)

    monkeypatch.setattr(conformance.os, call, denied)
    with pytest.raises(ConformanceError, match="cannot be read: PermissionError"):
        run_profile(name, tmp_path, source_revision="test")
    with pytest.raises(ConformanceError, match="cannot be read: PermissionError"):
        run_profiles(tmp_path, [name], source_revision="test")
    monkeypatch.undo()
    assert os.listdir(tmp_path) == [name] and os.listdir(folder) == []  # nothing was written


def test_a_symlinked_artifacts_root_still_runs_and_verifies(tmp_path):
    _runtime()
    real = tmp_path / "real"
    real.mkdir()
    root = tmp_path / "root"
    root.symlink_to(real, target_is_directory=True)  # only the profile's own folder must be no link
    (record,) = run_profiles(root, ["feedfwd-fp32-torchscript"], source_revision="test")["profiles"]
    assert record["status"] == "passed" and record["level"] == "executed"
    verify_artifacts(record, root)
    assert (real / "feedfwd-fp32-torchscript" / "model.onnx").is_file()


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("target", sorted(_HELD))
def test_the_script_refuses_a_symlinked_profile_folder_even_when_forced(tmp_path, capsys, target, force):
    name = "feedfwd-fp32-torchscript"
    artifacts, elsewhere = _linked_folder(tmp_path, target)
    output = tmp_path / "conformance.json"
    args = ["--output", str(output), "--artifacts", str(artifacts), "--profile", name, "--source-revision", "test"]
    with pytest.raises(SystemExit) as refused:
        _script().main([*args, "--force"] if force else args)
    assert refused.value.code == 2 and "is a symlink" in capsys.readouterr().err  # a usage error
    assert _untouched(artifacts / name, elsewhere, target) and not output.exists()


@pytest.mark.parametrize("call", ["iterdir", "lstat"])  # it cannot be listed, or looked up (its parent unsearchable)
def test_the_script_refuses_an_unreadable_profile_folder_even_when_forced(tmp_path, capsys, monkeypatch, call):
    artifacts = tmp_path / "artifacts"
    folder = artifacts / "feedfwd-fp32-torchscript"
    folder.mkdir(parents=True)
    (folder / "old.onnx").write_bytes(b"old")
    real = getattr(Path, call)

    def denied(self, *args, **kwargs):
        if self == folder:
            raise PermissionError(13, "Permission denied", str(folder))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, call, denied)
    output = tmp_path / "c.json"
    args = ["--output", str(output), "--artifacts", str(artifacts), "--profile", "feedfwd-fp32-torchscript"]
    with pytest.raises(SystemExit) as refused:
        _script().main([*args, "--source-revision", "test", "--force"])
    monkeypatch.undo()
    assert refused.value.code == 2 and "cannot be read: PermissionError" in capsys.readouterr().err  # a usage error
    assert os.listdir(folder) == ["old.onnx"] and not output.exists()  # nothing was cleared or written


def test_the_script_checks_every_folder_before_force_clears_any(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    (artifacts / "feedfwd-fp32-torchscript").mkdir(parents=True)
    (artifacts / "feedfwd-fp32-torchscript" / "old.onnx").write_bytes(b"old")
    (artifacts / "feedfwd-fp32-dynamo").symlink_to(tmp_path / "gone", target_is_directory=True)
    with pytest.raises(SystemExit) as refused:
        _script().main(["--output", str(tmp_path / "c.json"), "--artifacts", str(artifacts), "--force"])
    assert refused.value.code == 2 and "is a symlink" in capsys.readouterr().err
    assert os.listdir(artifacts / "feedfwd-fp32-torchscript") == ["old.onnx"]  # nothing was cleared


# --- an artifacts root that cannot be created or written is refused, not a raw OSError ----------------------------

_POSIX_USER = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="a read-only folder is writable by root"
)


@pytest.fixture
def unusable_root(tmp_path, request):
    """An artifacts root that is a dangling symlink, or a read-only folder
    (its mode restored afterwards, so ``tmp_path`` can be cleaned)."""
    root = tmp_path / "artifacts"
    if request.param == "dangling":
        root.symlink_to(tmp_path / "gone", target_is_directory=True)
        yield root, "FileNotFoundError"
        assert root.is_symlink() and not os.path.lexists(tmp_path / "gone")  # nothing was written
        return
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        pytest.skip("a read-only folder is writable by root")
    root.mkdir()
    root.chmod(0o555)
    try:
        yield root, "PermissionError"
        assert os.listdir(root) == []  # nothing was written
    finally:
        root.chmod(0o755)


@pytest.mark.parametrize("unusable_root", ["dangling", "read-only"], indirect=True)
def test_an_artifacts_root_that_cannot_be_created_is_a_refusal_not_a_raw_error(unusable_root):
    root, cause = unusable_root
    name = "feedfwd-fp32-torchscript"
    with pytest.raises(ConformanceError, match=f"{name}' cannot be created: {cause}") as refused:
        run_profile(name, root, source_revision="test")
    assert isinstance(refused.value.__cause__, OSError)
    with pytest.raises(ConformanceError, match=f"{name}' cannot be created: {cause}"):
        run_profiles(root, [name], source_revision="test")


@_POSIX_USER
def test_a_later_profile_folder_that_cannot_be_created_is_refused_before_any_profile_runs(tmp_path):
    """An empty first folder in a read-only root passes every check; the
    later folder that cannot be created is refused before the first runs."""
    first = tmp_path / "feedfwd-fp32-torchscript"
    first.mkdir()
    tmp_path.chmod(0o555)
    try:
        with pytest.raises(ConformanceError, match="feedfwd-fp32-dynamo' cannot be created: PermissionError"):
            run_profiles(tmp_path, source_revision="test")  # the default names: the torchscript profile comes first
    finally:
        tmp_path.chmod(0o755)
    assert os.listdir(tmp_path) == [first.name] and os.listdir(first) == []  # no artifact was written


@pytest.mark.parametrize("unusable_root", ["dangling", "read-only"], indirect=True)
def test_the_script_refuses_an_artifacts_root_that_cannot_be_created(tmp_path, capsys, unusable_root):
    root, cause = unusable_root
    output = tmp_path / "out" / "c.json"
    args = ["--output", str(output), "--artifacts", str(root), "--profile", "feedfwd-fp32-torchscript"]
    with pytest.raises(SystemExit) as refused:
        _script().main([*args, "--source-revision", "test"])
    assert refused.value.code == 2 and f"cannot be created: {cause}" in capsys.readouterr().err  # a usage error
    assert not output.exists()
