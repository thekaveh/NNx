"""Check, run and render the feature-composition matrix (FEAT-039).

    python scripts/check_feature_composition.py --check
    python scripts/check_feature_composition.py --check --run --report composition.json
    python scripts/check_feature_composition.py --render --report composition.json

``docs/feature-composition.yaml`` is the expectation: which feature
combinations are verified, unsupported or unverified, on which profile, by
which pytest node ids. The evidence is a separate report that a run writes:
every listed node's setup / call / teardown outcome, the source it ran on
(a content digest of ``src/``, the examples, the referenced test files and
the registry, plus the git revision) and the profile it ran on (OS, Python,
torch, devices, installed optional dependencies).

- ``--check`` validates the registry's schema and unique ids, collects the
  listed node ids through pytest and fails on any stale reference, naming
  the scenario; with ``--report`` it also reports whether that evidence is
  current. It then checks that the committed ``docs/feature-composition.md``
  is the page rendered without evidence (CI renders the published page
  from its own report).
- ``--run`` executes every listed node id (one pytest run, a phase-recording
  plugin) and writes ``--report``. It exits 1 when a ``verified`` row
  fails, loses coverage or is only skipped / xfailed, or when an
  ``unsupported`` row's negative test does not pass.
- ``--render`` writes the page from the registry and ``--report``. Evidence
  from other sources (another commit's content) or another profile than the
  registry's ``evidence_profile`` is stale: its cells render
  ``unverified``, exactly as with no report. Site and wiki render the same
  report, so they carry the same evidence digest.

Never proof: an import, a skip, an xfail, or a setup or teardown failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "feature-composition.yaml"
PAGE = ROOT / "docs" / "feature-composition.md"
FORMAT = "nnx.feature-composition/1"
STATUSES = ("verified", "unsupported", "unverified")
DEVICES = ("cpu", "cuda", "mps")
DTYPES = ("fp32", "fp16", "bf16")
ID_PATTERN = re.compile(r"^[A-Z]{2}-\d{2}$")
IMPORT_ONLY = "tests/test_examples_smoke.py::test_example_imports_without_running_main"
# The optional packages a profile can name (import names); "cuda" and the
# Gloo backend are capabilities, not packages.
KNOWN_DEPENDENCIES = (
    "optuna",
    "onnx",
    "onnxscript",
    "onnxruntime",
    "torchao",
    "tensorboard",
    "safetensors",
    "typesafe_sdk",
)
RUN_TIMEOUT_SECONDS = 1800


# --- the registry -------------------------------------------------------------------------------


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_registry(registry: Any) -> list[str]:
    """Every schema problem, as ``"<scenario id>: <problem>"`` lines."""
    problems: list[str] = []
    if not isinstance(registry, Mapping) or registry.get("version") != 1:
        return ["the registry must be a mapping with version: 1"]
    groups = registry.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        problems.append("groups must be a non-empty mapping of id to title")
        groups = {}
    profile = registry.get("evidence_profile")
    if (
        not isinstance(profile, Mapping)
        or not {"os", "python", "device"} <= set(profile)
        or not isinstance(profile.get("dependencies", []), list)
    ):
        problems.append(
            "evidence_profile must name the os, python, device (and a dependencies list) the published evidence runs on"
        )
    scenarios = registry.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        return [*problems, "scenarios must be a non-empty list"]
    seen: set[str] = set()
    for index, scenario in enumerate(scenarios):
        where = scenario.get("id", f"#{index}") if isinstance(scenario, Mapping) else f"#{index}"
        if not isinstance(scenario, Mapping):
            problems.append(f"{where}: a scenario must be a mapping")
            continue
        sid = scenario.get("id")
        if not isinstance(sid, str) or not ID_PATTERN.match(sid):
            problems.append(f"{where}: id must look like AB-01")
        elif sid in seen:
            problems.append(f"{sid}: duplicate id")
        else:
            seen.add(sid)
        if scenario.get("group") not in groups:
            problems.append(f"{where}: group {scenario.get('group')!r} is not declared")
        features = scenario.get("features")
        if not isinstance(features, list) or len(features) < 2 or not all(isinstance(f, str) and f for f in features):
            problems.append(f"{where}: features must list at least two combined features")
        for field in ("entry_point", "expected", "rationale"):
            if not isinstance(scenario.get(field), str) or not scenario[field].strip():
                problems.append(f"{where}: {field} must be non-empty text")
        prof = scenario.get("profile")
        if (
            not isinstance(prof, Mapping)
            or prof.get("device") not in DEVICES
            or prof.get("dtype") not in DTYPES
            or not isinstance(prof.get("dependencies"), list)
        ):
            problems.append(f"{where}: profile needs device {DEVICES}, dtype {DTYPES} and a dependencies list")
        status = scenario.get("status")
        if status not in STATUSES:
            problems.append(f"{where}: status must be one of {STATUSES}")
        tests = scenario.get("tests", [])
        if not isinstance(tests, list) or not all(isinstance(t, str) and "::" in t for t in tests):
            problems.append(f"{where}: tests must be pytest node ids (path::name)")
            tests = []
        if status == "verified" and not tests:
            problems.append(f"{where}: a verified scenario lists the tests that verify it")
        if status == "unsupported" and not tests and not scenario.get("limitation"):
            problems.append(f"{where}: an unsupported scenario links an executed negative test or a limitation")
        limitation = scenario.get("limitation")
        if limitation is not None:
            target = str(limitation).split("#", 1)[0].split()[0] if str(limitation).strip() else ""
            if not (target.endswith(".md") and (ROOT / target).is_file()):
                problems.append(f"{where}: limitation must start with the documenting file (path.md#anchor: text)")
        if any(test.startswith(IMPORT_ONLY) for test in tests):
            problems.append(f"{where}: an import-only smoke test is a reference, never evidence")
    return problems


def scenario_tests(registry: Mapping[str, Any], *, statuses: Sequence[str] = STATUSES) -> list[str]:
    """The node ids the given statuses' scenarios list (a run executes the
    verified and unsupported ones; unverified rows are claimed by nothing)."""
    return sorted(
        {
            test
            for scenario in registry["scenarios"]
            if scenario["status"] in statuses
            for test in scenario.get("tests", [])
        }
    )


# --- collection and staleness -------------------------------------------------------------------


def collect(node_ids: Sequence[str], *, root: Path = ROOT) -> set[str]:
    """The node ids pytest collects from the files the registry names."""
    files = sorted({node.split("::", 1)[0] for node in node_ids})
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-o", "addopts=", *files],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 5):  # 5: nothing collected
        tail = "\n".join((completed.stdout + completed.stderr).strip().splitlines()[-15:])
        raise CollectionError(f"pytest could not collect the registry's test files:\n{tail}")
    return {line.strip() for line in completed.stdout.splitlines() if "::" in line}


class CollectionError(RuntimeError):
    """pytest failed to collect a referenced file (not a stale reference)."""


def _matches(reference: str, node: str) -> bool:
    """A reference names a node exactly, or a test function's every
    parametrization (``path::name`` for ``path::name[...]``)."""
    return node == reference or (node.startswith(reference + "[") and "[" not in reference)


def stale_references(registry: Mapping[str, Any], collected: Iterable[str]) -> list[str]:
    """``"<scenario id>: <node id> is not collected"`` for every reference
    no collected node matches."""
    nodes = set(collected)
    problems = []
    for scenario in registry["scenarios"]:
        if _missing_dependencies(scenario):
            continue  # its file skips at collection without the extra: judged where it is installed
        for test in scenario.get("tests", []):
            if not any(_matches(test, node) for node in nodes):
                problems.append(f"{scenario['id']}: {test} is not collected (renamed or removed?)")
    return problems


def _missing_dependencies(scenario: Mapping[str, Any]) -> list[str]:
    import importlib.util

    return [
        dep
        for dep in scenario["profile"]["dependencies"]
        if dep in KNOWN_DEPENDENCIES and importlib.util.find_spec(dep) is None
    ]


DIGESTED = (
    "pyproject.toml",
    "uv.lock",
    "docs/feature-composition.yaml",
    "scripts/check_feature_composition.py",
    "scripts/composition_plugin.py",
)


def content_digest(registry: Mapping[str, Any], *, root: Path = ROOT) -> str:
    """What the evidence is evidence of: the library, the examples, every
    test module (scenario helpers and conftest included), the dependency
    pins, the registry and this checker."""
    files = sorted(
        {
            *(p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")),
            *(p.relative_to(root).as_posix() for p in (root / "examples").glob("*.py")),
            *(p.relative_to(root).as_posix() for p in (root / "tests").rglob("*.py")),
            *(name for name in DIGESTED if (root / name).is_file()),
        }
    )
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def current_profile() -> dict[str, Any]:
    import importlib.util

    import torch

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    return {
        "os": platform.system().lower(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__.split("+")[0],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "devices": devices,
        "dependencies": sorted(name for name in KNOWN_DEPENDENCIES if importlib.util.find_spec(name) is not None),
    }


def _git_revision(root: Path) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --- running and judging ------------------------------------------------------------------------


def run(registry: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Execute every listed node id once and return the evidence report."""
    nodes = scenario_tests(registry, statuses=("verified", "unsupported"))
    with tempfile.TemporaryDirectory() as scratch:
        phases_path = os.path.join(scratch, "phases.json")
        env = {**os.environ, "NNX_COMPOSITION_PHASES": phases_path, "NNX_TQDM_DISABLE": "1"}
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                "-p",
                "scripts.composition_plugin",
                *nodes,
            ],
            cwd=root,
            env=env,
            timeout=RUN_TIMEOUT_SECONDS,
        )
        if not os.path.exists(phases_path):
            raise RuntimeError("pytest exited before its session finished: no phase record was written")
        with open(phases_path, encoding="utf-8") as handle:
            phases = json.load(handle)
    return evidence(registry, phases, profile=current_profile(), root=root)


def evidence(
    registry: Mapping[str, Any],
    phases: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    root: Path = ROOT,
    source: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The report: its source, profile and every node's phases, with a
    digest over them."""
    body = {
        "format": FORMAT,
        "source": dict(source)
        if source is not None
        else {"content_digest": content_digest(registry, root=root), "git_revision": _git_revision(root)},
        "profile": dict(profile),
        "nodes": {node: dict(entry) for node, entry in sorted(phases.items())},
    }
    body["digest"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    return body


def node_passed(entry: Mapping[str, Any]) -> bool:
    """Passed in setup, call and teardown, and not an expected failure."""
    return (
        entry.get("setup") == "passed"
        and entry.get("call") == "passed"
        and entry.get("teardown") == "passed"
        and not entry.get("xfail")
    )


def staleness(registry: Mapping[str, Any], report: Optional[Mapping[str, Any]], *, root: Path = ROOT) -> Optional[str]:
    """Why ``report`` cannot be shown as current evidence (``None`` if it can)."""
    if report is None:
        return "no evidence report"
    if report.get("format") != FORMAT:
        return f"unknown report format {report.get('format')!r}"
    body = {key: value for key, value in report.items() if key != "digest"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest() != report.get("digest"):
        return "invalid: the report's digest does not match its content"
    if report.get("source", {}).get("content_digest") != content_digest(registry, root=root):
        return "stale: the report ran on other sources (another commit)"
    expected = registry["evidence_profile"]
    profile = report.get("profile", {})
    mismatch = [key for key in ("os", "python", "device") if str(profile.get(key)) != str(expected[key])]
    if mismatch:
        return f"stale: the report ran on another profile ({', '.join(mismatch)} differ)"
    return None


def judge(scenario: Mapping[str, Any], report: Optional[Mapping[str, Any]], stale: Optional[str]) -> tuple[str, str]:
    """``(cell, note)`` for one scenario under one report."""
    status = scenario["status"]
    if status == "unverified":
        return "unverified", scenario["rationale"]
    if stale is not None or report is None:
        return "unverified", stale or "no evidence report"
    profile = report["profile"]
    needed = scenario["profile"]
    if needed["device"] not in profile.get("devices", [profile.get("device")]):
        return "unverified", f"profile not run: needs {needed['device']}"
    missing = [
        dep for dep in needed["dependencies"] if dep in KNOWN_DEPENDENCIES and dep not in profile["dependencies"]
    ]
    if missing:
        return "unverified", f"profile not run: needs {', '.join(missing)}"
    nodes = report["nodes"]
    for test in scenario.get("tests", []):
        matched = [entry for node, entry in nodes.items() if _matches(test, node)]
        if not matched:
            return "failed", f"lost coverage: {test} did not run"
        if not all(node_passed(entry) for entry in matched):
            return "failed", f"{test} did not pass in every phase (a skip or xfail never counts)"
    if status == "verified":
        return "verified", "every listed test passed"
    return "unsupported", "the negative test passes" if scenario.get("tests") else scenario.get("limitation", "")


def gate(registry: Mapping[str, Any], report: Mapping[str, Any]) -> list[str]:
    """What fails a fresh run: a verified or unsupported row whose listed
    tests did not all pass in every phase, or did not run (lost coverage).
    A row whose profile this machine does not provide (a device or an
    optional dependency) is not judged; the published page shows it
    unverified."""
    problems = []
    published = registry.get("evidence_profile", {})
    if report["profile"].get("os") == published.get("os"):
        # On the publishing profile a lost extra is lost coverage, not a skip.
        missing = sorted(set(published.get("dependencies", [])) - set(report["profile"].get("dependencies", [])))
        if missing:
            problems.append(f"the run lacks the published profile's dependencies {missing}")
    for scenario in registry["scenarios"]:
        if scenario["status"] == "unverified":
            continue
        cell, note = judge(scenario, report, None)
        if cell == "failed":
            problems.append(f"{scenario['id']}: {note}")
    return problems


# --- rendering ----------------------------------------------------------------------------------


def render(registry: Mapping[str, Any], report: Optional[Mapping[str, Any]], *, root: Path = ROOT) -> str:
    stale = staleness(registry, report, root=root)
    lines = [
        "# 25. Feature-composition matrix",
        "",
        "*Generated by `scripts/check_feature_composition.py --render` from",
        "`docs/feature-composition.yaml` (the expectation) and a run's evidence report",
        "— do not edit by hand.*",
        "",
        "Which feature **combinations** NNx verifies, refuses or leaves unverified, per",
        "device / dtype / dependency profile. A cell is *verified* only when every listed",
        "test passed its setup, call and teardown on the published profile at these exact",
        "sources; an import, a skip or an xfail never counts. Without a current report every",
        "cell reads *unverified*.",
        "",
    ]
    if stale is None and report is not None:
        lines.append(
            f"**Evidence:** `{report['digest'][:16]}` — sources `{report['source']['content_digest'][:12]}`"
            f" (git `{(report['source'].get('git_revision') or 'unknown')[:12]}`), profile "
            f"{report['profile']['os']} / Python {report['profile']['python']} / torch "
            f"{report['profile']['torch']} / {report['profile']['device']}."
        )
    else:
        lines.append(f"**Evidence:** none current ({stale}).")
    lines.append("")
    for group_id, title in registry["groups"].items():
        rows = [scenario for scenario in registry["scenarios"] if scenario["group"] == group_id]
        if not rows:
            continue
        lines += [
            f"## {title}",
            "",
            "| ID | Features | Profile | Expected | Status | Evidence |",
            "|---|---|---|---|---|---|",
        ]
        for scenario in rows:
            cell, note = judge(scenario, report, stale)
            profile = scenario["profile"]
            deps = ", ".join(profile["dependencies"]) or "core"
            tests = "<br>".join(f"`{test}`" for test in scenario.get("tests", [])) or scenario.get("limitation", "")
            lines.append(
                f"| {scenario['id']} | {' + '.join(scenario['features'])} | {profile['device']} / {profile['dtype']}"
                f" / {deps} | {scenario['expected']} | **{cell}** — {note} | {tests} |"
            )
        lines.append("")
    lines += [
        "Status meanings: *verified* — the listed tests assert the combination and passed;",
        "*unsupported* — refused, proven by an executed negative test (or an exact documented",
        "limitation); *unverified* — no executed evidence on the profiles CI runs (for example",
        "CUDA AMP on a CPU runner), or no current report; *failed* — a listed test did not",
        "pass or did not run. Contributors: see CONTRIBUTING (scenario IDs to review when a",
        "boundary changes).",
        "",
    ]
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="validate the registry, its references and the page")
    parser.add_argument("--run", action="store_true", help="execute the scenarios and write --report")
    parser.add_argument("--render", action="store_true", help="write docs/feature-composition.md from --report")
    parser.add_argument("--report", type=Path, help="the evidence report to write (--run) or read")
    args = parser.parse_args(argv)
    if not (args.check or args.run or args.render):
        parser.error("pass --check, --run and/or --render")
    if args.run and args.report is None:
        parser.error("--run needs --report")
    registry = load_registry()
    failures: list[str] = []
    if args.check:
        failures += validate_registry(registry)
        if not failures:
            try:
                failures += stale_references(registry, collect(scenario_tests(registry)))
            except CollectionError as error:
                failures.append(str(error))
        if PAGE.read_text(encoding="utf-8") != render(registry, None):
            failures.append("docs/feature-composition.md is not the page rendered without evidence: run --render")
    report: Optional[dict[str, Any]] = None
    if args.run and not failures:
        report = run(registry)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        failures += gate(registry, report)
    elif args.report is not None and args.report.exists():
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if args.check:
            stale = staleness(registry, report)
            print(f"evidence: {'current' if stale is None else stale}")
    if args.render and not failures:
        PAGE.write_text(render(registry, report), encoding="utf-8")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
