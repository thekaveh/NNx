"""FIX-032: every workflow job runs on one pinned, tested runner image.

``ubuntu-latest`` moves to Ubuntu 26 from October 19, 2026 (and to later
images after that) without a change in this repository. Jobs name the image
they were verified on instead; moving to a new one is a deliberate change
(CONTRIBUTING §7.1), checked by this test and a CI run on the new label. A
package-mirror stall in an apt step fails within its own timeout instead of
consuming the whole job budget (#433 lost 30 minutes to one)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tests._repo_files import repo_file

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted(repo_file(".github", "workflows").glob("*.y*ml"))
RUNNER = "ubuntu-26.04"
# The jobs that need hardware no hosted image has. Each runs only on a
# registered self-hosted runner, opted in by its repository variable:
# FEAT-043 CUDA (NNX_GPU_RUNNER), FEAT-050 Apple-silicon MPS (NNX_MPS_RUNNER).
SELF_HOSTED = {
    ("docs.yml", "gpu-evidence"): ["self-hosted", "linux", "x64", "gpu"],
    ("mps.yml", "mps-evidence"): ["self-hosted", "macos", "arm64"],
}
OPT_IN = {("docs.yml", "gpu-evidence"): "NNX_GPU_RUNNER", ("mps.yml", "mps-evidence"): "NNX_MPS_RUNNER"}
APT_STEP_MINUTES = 5


def _jobs():
    for path in WORKFLOWS:
        for name, job in (yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs") or {}).items():
            yield path.name, name, job


def test_the_workflows_exist():
    assert {path.name for path in WORKFLOWS} >= {
        "ci.yml",
        "docs.yml",
        "mps.yml",
        "release.yml",
        "release-please.yml",
        "security.yml",
    }


@pytest.mark.parametrize(("workflow", "job_name", "job"), list(_jobs()), ids=lambda value: str(value)[:40])
def test_every_job_runs_on_the_pinned_image(workflow, job_name, job):
    if "uses" in job:  # a reusable-workflow call runs on the called workflow's image
        return
    expected = SELF_HOSTED.get((workflow, job_name), RUNNER)
    assert job.get("runs-on") == expected, f"{workflow}:{job_name} runs on {job.get('runs-on')!r}"


def test_the_self_hosted_jobs_exist_and_are_opt_in():
    jobs = {(workflow, name): job for workflow, name, job in _jobs()}
    assert OPT_IN.keys() == SELF_HOSTED.keys()
    for key in SELF_HOSTED:
        assert key in jobs, f"{key} is listed as self-hosted but does not exist"
        # skipped unless its runner is registered, and never for a fork's pull request
        assert jobs[key]["if"].startswith(f"vars.{OPT_IN[key]} == 'true' && ")
        assert "head.repo.full_name == github.repository" in jobs[key]["if"]


def test_the_mps_job_runs_every_mps_test_and_cannot_pass_by_skipping():
    """FEAT-050: the probe fails without a usable MPS device, the tests run
    with NNX_REQUIRE_MPS=1 (an ``mps`` test then fails, not skips), and the
    JUnit report must show tests and no skip."""
    job = dict((name, job) for workflow, name, job in _jobs() if workflow == "mps.yml")["mps-evidence"]
    steps = {step.get("name"): step for step in job["steps"]}
    probe = steps["Probe the MPS device (fails without one)"]["run"]
    assert "torch.backends.mps.is_built()" in probe and "torch.backends.mps.is_available()" in probe
    assert "assert built and available" in probe
    run = steps["Run the MPS tests (a missing device fails, never skips)"]
    assert run["env"]["NNX_REQUIRE_MPS"] == "1"
    assert "pytest tests -m mps " in run["run"]
    assert "assert skipped == 0" in run["run"] and "assert tests > 0" in run["run"]


def test_a_stalled_mps_queue_never_holds_the_ci_run():
    """FEAT-050: a run waits for an offline self-hosted runner up to GitHub's
    24-hour queue limit. The MPS job has a workflow and concurrency group of
    its own, where a newer commit replaces the waiting run, so CI's group
    (which lets pushes finish) never waits behind it."""
    workflows = {path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in WORKFLOWS}
    assert set(workflows["mps.yml"]["jobs"]) == {"mps-evidence"}
    assert workflows["mps.yml"]["concurrency"] == {
        "group": "mps-${{ github.workflow }}-${{ github.ref }}",
        "cancel-in-progress": True,
    }
    assert not any("self-hosted" in str(job.get("runs-on")) for job in workflows["ci.yml"]["jobs"].values())


def test_contributing_says_a_stalled_gpu_queue_holds_the_docs_run():
    """FEAT-043: unlike the MPS job, the GPU job is part of the docs
    workflow. The build needs it, the Pages deploy and the wiki publish need
    the build, and the workflow's concurrency group lets a run finish. While
    the runner is offline that run waits, and the next push's run waits
    behind it; CONTRIBUTING's GPU runner note says so and names the remedy."""
    docs = yaml.safe_load(repo_file(".github", "workflows", "docs.yml").read_text(encoding="utf-8"))
    assert docs["concurrency"]["cancel-in-progress"] is False
    jobs = docs["jobs"]
    assert "gpu-evidence" in jobs["build"]["needs"]
    for name in ("deploy", "wiki"):
        assert "build" in jobs[name]["needs"], name
    contributing = repo_file("CONTRIBUTING.md").read_text(encoding="utf-8")
    section = contributing.split("#### GPU evidence", 1)[1].split("### 5.2.", 1)[0]
    note = " ".join(section.split())
    assert "While the runner is offline, the docs build, the Pages deploy and the wiki publish wait" in note
    assert "Unset `NNX_GPU_RUNNER` while the runner is offline" in note


def test_every_apt_step_has_its_own_short_timeout():
    apt_steps = [
        (workflow, job_name, step)
        for workflow, job_name, job in _jobs()
        for step in job.get("steps") or []
        if re.search(r"\bapt(-get)? ", str(step.get("run", "")))
    ]
    assert apt_steps, "the cairo runtime is installed with apt somewhere"
    for workflow, job_name, step in apt_steps:
        minutes = step.get("timeout-minutes")
        assert isinstance(minutes, int) and 0 < minutes <= APT_STEP_MINUTES, (workflow, job_name, step.get("name"))


def test_no_step_writes_under_tmp():
    """On ubuntu-26.04 /tmp is a size-capped tmpfs: the v0.4.0 release's
    artifact smoke test overflowed it installing torch ("Disk quota
    exceeded"). Steps use $RUNNER_TEMP, on the runner's work disk."""
    for workflow, job_name, job in _jobs():
        for step in job.get("steps") or []:
            assert "/tmp" not in str(step.get("run", "")), (workflow, job_name, step.get("name"))
