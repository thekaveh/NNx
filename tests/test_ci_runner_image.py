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
# FEAT-043: the one job that needs hardware no hosted image has (CUDA); it
# runs only on a registered self-hosted GPU runner, opted in by a variable.
SELF_HOSTED = {("docs.yml", "gpu-evidence"): ["self-hosted", "linux", "x64", "gpu"]}
APT_STEP_MINUTES = 5


def _jobs():
    for path in WORKFLOWS:
        for name, job in (yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs") or {}).items():
            yield path.name, name, job


def test_the_workflows_exist():
    assert {path.name for path in WORKFLOWS} >= {
        "ci.yml",
        "docs.yml",
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
    for key in SELF_HOSTED:
        assert key in jobs, f"{key} is listed as self-hosted but does not exist"
        # skipped unless a GPU runner is registered, and never for a fork's pull request
        assert jobs[key]["if"].startswith("vars.NNX_GPU_RUNNER == 'true' && ")
        assert "head.repo.full_name == github.repository" in jobs[key]["if"]


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
