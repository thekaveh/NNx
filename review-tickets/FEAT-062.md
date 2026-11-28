---
id: FEAT-062
title: "[FEAT-062] Drop the unused libcairo2 step from the hosted MPS gate"
labels: ["enhancement", "github_actions", "area: packaging", "priority: P3", "review: 2026-11-27"]
sources: ["CI-07"]
depends_on: []
related_issues: [445, 572]
status: READY
---

**Category:** Packaging · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop)

## Summary

FEAT-062: the hosted `mps-gate-without-mps` job in `.github/workflows/mps.yml` installs the Cairo runtime with apt, but the job only runs `pytest tests -m mps`, and no test it runs needs Cairo. Delete the step: the job gets faster and loses one network dependency (a stalled package mirror).

## Context

**What happens today.** The job's first step after checkout:

```yaml
      - name: Install diagram rendering runtime
        # A package-mirror stall fails here, not after the whole job budget.
        timeout-minutes: 5
        run: sudo apt-get -o Acquire::Retries=3 update && sudo apt-get -o Acquire::Retries=3 install --yes libcairo2
```

The job then syncs the extras, checks that the MPS probe fails, and runs only the `mps`-marked tests with `NNX_REQUIRE_MPS=1`.

**Why it matters.** `libcairo2` is needed only by `cairosvg`, which only `scripts/docs/extract_architecture_svg.py` imports. Its sole test users, in `tests/test_docs_projection.py` (L1393, L1402), import it inside the test functions, and those tests carry no `mps` mark, so `-m mps` deselects them; collection never imports `cairosvg`. The step looks copied from `ci.yml`, where the docs tests do need Cairo.

**History.** #572 (FEAT-050) added this hosted twin of `mps-evidence` to prove the MPS gate fails rather than skips. #445 (FIX-032) gave every apt step its own short `timeout-minutes`; `tests/test_ci_runner_image.py::test_every_apt_step_has_its_own_short_timeout` still finds the apt steps in `ci.yml`, `docs.yml` and `release.yml` after this change.

**Background for the worker.**
- *`mps-gate-without-mps`*: a hosted Linux job with no MPS device; every `@pytest.mark.mps` test must fail with the `NNX_REQUIRE_MPS` message there.
- *`libcairo2`*: the C library `cairosvg` loads when it is imported. `uv sync --all-extras` installs the `cairosvg` Python package without it; only importing `cairosvg` needs the library.

**Proposal.**

1. In `.github/workflows/mps.yml`, delete the whole step `- name: Install diagram rendering runtime` of the `mps-gate-without-mps` job (L168–L171: the `name`, the comment, `timeout-minutes: 5` and the `run:` line). Leave the same-named steps in `ci.yml`, `docs.yml` and `release.yml` alone.
2. Add this test to `tests/test_ci_runner_image.py`, below `test_the_no_mps_gate_job_fails_every_mps_test_on_a_runner_without_mps`:
   ```python
   def test_the_no_mps_gate_job_installs_no_apt_package():
       """FEAT-062: the hosted gate runs only `pytest -m mps`, and no mps test
       imports cairosvg, so the job installs nothing with apt."""
       job = dict((name, job) for workflow, name, job in _jobs() if workflow == "mps.yml")["mps-gate-without-mps"]
       assert not [step for step in job["steps"] if re.search(r"\bapt(-get)? ", str(step.get("run", "")))]
   ```

No library behaviour changes, so no CHANGELOG entry.

**Worker setup.** No torch needed (CI config only); `uv sync --frozen --all-extras` still installs what `tests/test_ci_runner_image.py` imports.

## Acceptance criteria

- [ ] The new test `tests/test_ci_runner_image.py::test_the_no_mps_gate_job_installs_no_apt_package` fails on `develop` and passes after the change. Check: `uv run pytest tests/test_ci_runner_image.py::test_the_no_mps_gate_job_installs_no_apt_package`
- [ ] `mps.yml` no longer installs `libcairo2`. Check: `grep -c libcairo2 .github/workflows/mps.yml` prints `0`
- [ ] The other workflows keep their apt step, and the runner-image tests pass. Check: `uv run pytest tests/test_ci_runner_image.py`
- [ ] On GitHub, the "MPS gate without MPS (hosted Linux, must fail)" job of the PR's MPS workflow run is green. Check: the PR's checks page
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0

## Evidence

| Location | What it shows |
|---|---|
| [mps.yml L152-L171](https://github.com/thekaveh/NNx/blob/9ad461e/.github/workflows/mps.yml#L152-L171) | The hosted gate job and its apt step |
| [mps.yml L192-L200](https://github.com/thekaveh/NNx/blob/9ad461e/.github/workflows/mps.yml#L192-L200) | The job runs `pytest tests -m mps` only |
| [extract_architecture_svg.py L13](https://github.com/thekaveh/NNx/blob/9ad461e/scripts/docs/extract_architecture_svg.py#L13) | The only `cairosvg` import |
| [test_docs_projection.py L1393](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_docs_projection.py#L1393), [L1402](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_docs_projection.py#L1402) | Its test users import it inside the test functions |
| [test_ci_runner_image.py L124-L134](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_ci_runner_image.py#L124-L134) | The apt-timeout test, which still finds the other apt steps |
| #572, #445 | The hosted gate; the apt-step timeouts |

## Scope

**In:** one step of `mps.yml`, one test.

**Out:** the self-hosted `mps-evidence` and `gpu-check` jobs (DEC-015); the Cairo steps of the other workflows.

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #572, #445, DEC-015
