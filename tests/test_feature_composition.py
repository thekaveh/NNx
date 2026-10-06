"""FEAT-039: the feature-composition registry, its checker and its page.

The home of the matrix's checks: the registry's schema, stale references,
how each test phase counts as evidence, stale reports, rendering and the
publication path."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts import check_feature_composition as fc

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registry():
    return fc.load_registry()


def _fresh(registry, nodes, **profile):
    """A current report: the registry's sources and published profile."""
    expected = registry["evidence_profile"]
    full = {
        "os": expected["os"],
        "python": expected["python"],
        "device": expected["device"],
        "devices": [expected["device"]],
        "torch": "2.13.0",
        "dependencies": list(fc.KNOWN_DEPENDENCIES),
        **profile,
    }
    return fc.evidence(registry, nodes, profile=full)


PASSED = {"setup": "passed", "call": "passed", "teardown": "passed", "xfail": False}


# ---------------- the registry ----------------


# The features this matrix composes, each by the test module that owns its
# behaviour: every one must be exercised in at least one registered scenario.
FEATURE_TEST_MODULES = {
    "FEAT-010 Jev adapter": "tests/test_decision_jev.py",
    "FEAT-023 decision-model pilot": "tests/test_decision_model_pilot.py",
    "FEAT-025 result boundaries": "tests/test_result_boundaries.py",
    "FEAT-029 compile": "tests/test_compile_execution.py",
    "FEAT-029 benchmarks": "tests/test_benchmarking.py",
    "FEAT-030 DDP": "tests/test_ddp_adapter.py",
    "FEAT-031 lazy domains": "tests/test_core_install.py",
    "FEAT-033 search": "tests/test_search.py",
    "FEAT-037 ordered logits": "tests/test_ordered_logits_pipeline.py",
}


def test_every_composed_feature_appears_in_a_scenario(registry):
    cited = {test.split("::")[0] for scenario in registry["scenarios"] for test in scenario["tests"]}
    missing = [feature for feature, module in FEATURE_TEST_MODULES.items() if module not in cited]
    assert missing == []


def test_the_registry_is_valid_with_unique_ids(registry):
    assert fc.validate_registry(registry) == []
    ids = [scenario["id"] for scenario in registry["scenarios"]]
    assert len(ids) == len(set(ids))


def test_twelve_or_more_scenarios_span_every_interaction_group(registry):
    assert len(registry["scenarios"]) >= 12
    groups = {scenario["group"] for scenario in registry["scenarios"]}
    assert groups == {
        "training-controls",
        "checkpoint-resume",
        "model-transformations",
        "inference-export",
        "decisions-results",
        "packaging",
    }
    statuses = {scenario["status"] for scenario in registry["scenarios"]}
    assert statuses == {"verified", "unsupported", "unverified"}
    for scenario in registry["scenarios"]:
        if scenario["status"] == "verified":  # behaviour, never an import
            assert scenario["tests"] and not any(t.startswith(fc.IMPORT_ONLY) for t in scenario["tests"])


def test_cuda_amp_and_onnx_runtime_rows_stay_unverified_on_cpu(registry):
    by_features = {" + ".join(s["features"]): s for s in registry["scenarios"]}
    amp = next(s for name, s in by_features.items() if "FP16 automatic mixed precision" in name)
    ort = next(s for name, s in by_features.items() if "ONNX Runtime" in name)
    assert amp["status"] == ort["status"] == "unverified"
    report = _fresh(registry, {t: dict(PASSED) for s in registry["scenarios"] for t in s["tests"]})
    assert fc.judge(amp, report, None)[0] == fc.judge(ort, report, None)[0] == "unverified"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r["scenarios"].append(copy.deepcopy(r["scenarios"][0])), "duplicate id"),
        (lambda r: r["scenarios"][0].update(id="bad"), "id must look like"),
        (lambda r: r["scenarios"][0].update(group="nope"), "is not declared"),
        (lambda r: r["scenarios"][0].update(status="probably"), "status must be one of"),
        (lambda r: r["scenarios"][0].update(tests=[]), "lists the tests"),
        (lambda r: r["scenarios"][0].update(features=["one"]), "at least two"),
        (lambda r: r["scenarios"][0].update(profile={"device": "tpu", "dtype": "fp32", "dependencies": []}), "profile"),
        (
            lambda r: r["scenarios"][0].update(tests=[fc.IMPORT_ONLY + "[01_synthetic_classification.py]"]),
            "import-only",
        ),
        (lambda r: r["scenarios"][0].update(tests=["no-node-separator"]), "node ids"),
        (lambda r: r.pop("evidence_profile"), "evidence_profile"),
    ],
)
def test_schema_problems_are_named(registry, mutate, message):
    broken = copy.deepcopy(registry)
    mutate(broken)
    assert any(message in problem for problem in fc.validate_registry(broken))


def test_an_unsupported_row_needs_a_negative_test_or_an_exact_limitation(registry):
    broken = copy.deepcopy(registry)
    unsupported = next(s for s in broken["scenarios"] if s["status"] == "unsupported")
    unsupported["tests"] = []
    assert any("negative test or a limitation" in p for p in fc.validate_registry(broken))
    unsupported["limitation"] = "docs/jepa.md#7-resume: the EMA target is not persisted"
    assert fc.validate_registry(broken) == []


# ---------------- stale references ----------------


def test_a_removed_test_reference_fails_naming_its_scenario(registry):
    broken = copy.deepcopy(registry)
    broken["scenarios"][0]["tests"].append("tests/test_schedulers.py::removed_test")
    collected = fc.collect(fc.scenario_tests(registry))  # the real collection
    assert fc.stale_references(registry, collected) == []
    problems = fc.stale_references(broken, collected)
    assert problems == [
        f"{broken['scenarios'][0]['id']}: tests/test_schedulers.py::removed_test is not collected (renamed or removed?)"
    ]


def test_a_function_reference_covers_its_parametrizations():
    assert fc._matches("tests/a.py::t", "tests/a.py::t[1-2]")
    assert not fc._matches("tests/a.py::t", "tests/a.py::t_other")
    assert not fc._matches("tests/a.py::t[1]", "tests/a.py::t[1]x")


# ---------------- what counts as evidence ----------------

SCENARIO = {
    "id": "TC-99",
    "group": "training-controls",
    "features": ["a", "b"],
    "entry_point": "x",
    "profile": {"device": "cpu", "dtype": "fp32", "dependencies": []},
    "expected": "y",
    "tests": ["tests/a.py::t"],
    "rationale": "z",
    "status": "verified",
}


@pytest.mark.parametrize(
    ("phases", "cell"),
    [
        ({"setup": "passed", "call": "passed", "teardown": "passed"}, "verified"),
        ({"setup": "skipped", "call": None, "teardown": "passed"}, "failed"),
        ({"setup": "passed", "call": "failed", "teardown": "passed"}, "failed"),
        ({"setup": "passed", "call": "passed", "teardown": "failed"}, "failed"),
        ({"setup": "passed", "call": "skipped", "teardown": "passed", "xfail": True}, "failed"),
    ],
    ids=["passed", "setup-skipped", "call-failed", "teardown-failed", "xfailed"],
)
def test_only_a_test_passing_every_phase_verifies(registry, phases, cell):
    report = _fresh(registry, {"tests/a.py::t": {"xfail": False, **phases}})
    assert fc.judge(SCENARIO, report, None)[0] == cell


def test_a_row_whose_test_did_not_run_has_lost_coverage(registry):
    report = _fresh(registry, {})
    cell, note = fc.judge(SCENARIO, report, None)
    assert cell == "failed" and "lost coverage" in note
    assert fc.gate({"scenarios": [SCENARIO]}, report) == ["TC-99: lost coverage: tests/a.py::t did not run"]


def test_a_failing_negative_test_fails_the_run(registry):
    unsupported = {**SCENARIO, "status": "unsupported"}
    report = _fresh(registry, {"tests/a.py::t": {**PASSED, "call": "failed"}})
    assert fc.gate({"scenarios": [unsupported]}, report)


def test_a_missing_dependency_leaves_the_row_unjudged_not_failed(registry):
    needs_onnx = {**SCENARIO, "profile": {"device": "cpu", "dtype": "fp32", "dependencies": ["onnx"]}}
    report = _fresh(registry, {}, dependencies=[])
    assert fc.judge(needs_onnx, report, None)[0] == "unverified"
    assert fc.gate({"scenarios": [needs_onnx]}, report) == []


# ---------------- stale reports and rendering ----------------


def test_evidence_from_another_commit_or_profile_is_stale(registry):
    nodes = {t: dict(PASSED) for s in registry["scenarios"] for t in s["tests"]}
    assert fc.staleness(registry, _fresh(registry, nodes)) is None
    other_commit = fc.evidence(
        registry, nodes, profile=_fresh(registry, nodes)["profile"], source={"content_digest": "0" * 64}
    )
    assert "another commit" in str(fc.staleness(registry, other_commit))
    assert "another profile" in str(fc.staleness(registry, _fresh(registry, nodes, os="darwin")))
    tampered = _fresh(registry, nodes)
    first = next(iter(tampered["nodes"]))
    tampered["nodes"][first] = {**tampered["nodes"][first], "call": "failed"}
    assert "digest does not match" in str(fc.staleness(registry, tampered))
    assert fc.staleness(registry, None) == "no evidence report"


def test_a_missing_or_stale_report_renders_every_cell_unverified(registry):
    nodes = {t: dict(PASSED) for s in registry["scenarios"] for t in s["tests"]}
    for report in (None, _fresh(registry, nodes, python="3.10")):
        page = fc.render(registry, report)
        assert "**verified**" not in page and "**unsupported**" not in page
        assert page.count("**unverified**") == len(registry["scenarios"])


def test_a_current_report_renders_its_digest_identically_for_site_and_wiki(registry):
    nodes = {t: dict(PASSED) for s in registry["scenarios"] for t in s["tests"]}
    report = _fresh(registry, nodes)
    site, wiki = fc.render(registry, report), fc.render(registry, report)
    assert site == wiki and f"`{report['digest'][:16]}`" in site
    verified = sum(1 for s in registry["scenarios"] if s["status"] == "verified")
    assert site.count("**verified**") == verified


def test_the_committed_page_is_the_page_rendered_without_evidence(registry):
    assert (ROOT / "docs" / "feature-composition.md").read_text(encoding="utf-8") == fc.render(registry, None)


# ---------------- publication ----------------


def test_the_page_is_registered_for_site_and_wiki():
    manifest = yaml.safe_load((ROOT / "docs" / "manifest.yaml").read_text(encoding="utf-8"))
    sources = [
        entry.get("source")
        for section in manifest["sections"]
        for entry in ([section] + list(section.get("children", [])))
    ]
    assert "docs/feature-composition.md" in sources and manifest["surfaces"] == ["repo", "site", "wiki"]


def test_ci_runs_the_scenarios_and_docs_render_both_surfaces_from_one_report():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    docs = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    assert "check_feature_composition.py --check --run --report" in ci and "upload-artifact" in ci
    assert docs.count("check_feature_composition.py --render --report") == 2  # site job and wiki job
    assert "download-artifact" in docs


def test_composition_example_evidence(registry):
    """examples/README shows the checker on a registered 01 / 05 / 26
    scenario, and treats import-only smoke tests as references."""
    readme = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    assert "scripts/check_feature_composition.py" in readme and "EX-01" in readme
    assert "references, not evidence" in readme
    by_id = {s["id"]: s for s in registry["scenarios"]}
    for sid, example in (("EX-01", "01_"), ("EX-05", "05_"), ("EX-26", "26_")):
        tests = by_id[sid]["tests"]
        assert by_id[sid]["status"] == "verified"
        assert any("run_end_to_end" in t and example in t for t in tests)


def test_contributing_names_the_scenarios_to_review():
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "docs/feature-composition.yaml" in contributing and "scenario ID" in contributing
    assert "never proof" in contributing


def test_the_digest_covers_scenario_helpers_conftest_pins_and_the_checker(registry, tmp_path):
    import shutil

    for name in ("uv.lock", "pyproject.toml", "scripts/composition_plugin.py", "scripts/check_feature_composition.py"):
        assert name in fc.DIGESTED
    digest = fc.content_digest(registry, root=ROOT)
    copy_root = tmp_path / "copy"
    for name in ("src", "examples", "tests", "docs", "scripts"):
        shutil.copytree(ROOT / name, copy_root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / name, copy_root / name)
    assert fc.content_digest(registry, root=copy_root) == digest
    for name in ("tests/conftest.py", "tests/ddp_scenarios.py", "uv.lock"):
        target = copy_root / name
        original = target.read_bytes()
        target.write_bytes(original + b"\n# changed\n")
        assert fc.content_digest(registry, root=copy_root) != digest, name
        target.write_bytes(original)


def test_a_collection_error_is_reported_as_one_not_as_stale_references(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_broken.py").write_text(
        "import nonexistent_module_xyz\n\n\ndef test_x():\n    pass\n", encoding="utf-8"
    )
    with pytest.raises(fc.CollectionError, match="could not collect"):
        fc.collect(["tests/test_broken.py::test_x"], root=tmp_path)


def test_a_pytest_crash_is_reported_plainly(registry, tmp_path, monkeypatch):
    monkeypatch.setattr(fc.subprocess, "run", lambda *args, **kwargs: None)  # no phase record written
    with pytest.raises(RuntimeError, match="no phase record"):
        fc.run(registry, root=tmp_path)


def test_the_run_fails_when_the_published_profile_lost_an_extra(registry):
    nodes = {t: dict(PASSED) for s in registry["scenarios"] for t in s["tests"]}
    assert any(
        "published profile's dependencies" in p
        for p in fc.gate(registry, _fresh(registry, nodes, dependencies=["onnx"]))
    )
    assert fc.gate(registry, _fresh(registry, nodes)) == []


def test_a_limitation_must_cite_its_documenting_file(registry):
    broken = copy.deepcopy(registry)
    broken["scenarios"][0]["limitation"] = "lol"
    assert any("limitation must start" in p for p in fc.validate_registry(broken))
    broken["scenarios"][0]["limitation"] = "docs/jepa.md#resume: the documented boundary"
    assert fc.validate_registry(broken) == []


def _tracked_copy(tmp_path, dirname="repo"):
    import shutil
    import subprocess

    listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    tracked = [name for name in listing.split("\0") if name]
    copy_root = tmp_path / dirname
    for name in tracked:
        source = ROOT / name
        if source.is_file():
            (copy_root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, copy_root / name)
    subprocess.run(["git", "init", "-q"], cwd=copy_root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=copy_root, check=True)
    return copy_root


def _build_both(copy_root):
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": f"{copy_root / 'src'}{os.pathsep}{copy_root}"}
    subprocess.run(
        [sys.executable, "-m", "scripts.docs.build_docs"], cwd=copy_root, env=env, check=True, capture_output=True
    )
    site = (copy_root / "generated" / "site" / "Feature-composition.md").read_text(encoding="utf-8")
    wiki = (copy_root / "generated" / "wiki" / "Feature-composition.md").read_text(encoding="utf-8")
    return site, wiki


def test_site_and_wiki_render_one_report_with_an_identical_evidence_digest(registry, tmp_path):
    """The published path: render the page from a report in a clean copy of
    the tracked files, build both surfaces, and find the same evidence line
    on each. A second fresh copy (the wiki job's checkout) judges the report
    current. With a stale report or none every cell is unverified: the unit
    tests above render that case, so it is not built a second time here."""
    copy_root = _tracked_copy(tmp_path)
    nodes = {t: dict(PASSED) for s in registry["scenarios"] for t in s["tests"]}
    expected = registry["evidence_profile"]
    profile = {
        **expected,
        "devices": [expected["device"]],
        "torch": "2.13.0",
        "dependencies": list(fc.KNOWN_DEPENDENCIES),
    }
    report = fc.evidence(registry, nodes, profile=profile, root=copy_root)
    page = copy_root / "docs" / "feature-composition.md"
    page.write_text(fc.render(registry, report, root=copy_root), encoding="utf-8")
    site, wiki = _build_both(copy_root)
    line = f"`{report['digest'][:16]}`"
    assert line in site and line in wiki
    assert site.count("**verified**") == wiki.count("**verified**") > 0
    assert fc.staleness(registry, report, root=_tracked_copy(tmp_path, "fresh")) is None
