# Three-Surface Documentation Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all eight documentation-audit findings and publish the result through develop and main.

**Architecture:** Canonical Markdown and HTML diagram masters remain authoritative. Projection tests enforce opener and numbering contracts, while CairoSVG adds deterministic PNG fallbacks beside existing SVG derivatives.

**Tech Stack:** Python 3.10+, pytest, Markdown, PyYAML, CairoSVG, MkDocs Material, GitHub Actions.

## Global Constraints

- Keep `generated/site`, `generated/wiki`, `site`, and root `mkdocs.yml` untracked.
- Preserve self-contained site and wiki outputs.
- Use PNG for repository/wiki diagrams and SVG for the generated site.
- Keep the shared executive summary between 100 and 150 words.
- Remove unsupported absolute comparison and GPU-performance claims.

---

### Task 1: Documentation contracts

**Files:**
- Modify: `tests/test_docs_projection.py`
- Modify: `scripts/docs/build_docs.py`
- Modify: canonical Markdown sources listed in `docs/manifest.yaml`

- [ ] Add failing tests for manifest-number/H1 agreement and opener parity.
- [ ] Run the focused tests and confirm they fail for the audited reasons.
- [ ] Add numbered H1s and projection-time License handling.
- [ ] Add the mirrored tagline, badges, poster placement, and executive summary.
- [ ] Run focused tests and confirm they pass.

### Task 2: Deterministic visual fallbacks

**Files:**
- Create: `docs/diagrams/nnx-poster.html`
- Create: `docs/assets/nnx-poster.svg`
- Create: `docs/assets/nnx-poster.png`
- Create: `docs/assets/architecture.png`
- Create: `docs/assets/training-lifecycle.png`
- Create: `docs/assets/docs-projection.png`
- Modify: `scripts/docs/extract_architecture_svg.py`
- Modify: `scripts/docs/build_docs.py`
- Modify: `tests/test_docs_projection.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.github/workflows/docs.yml`

- [ ] Add failing tests for paired deterministic assets and surface-specific formats.
- [ ] Run focused tests and confirm the missing PNG failure.
- [ ] Extend the extractor with CairoSVG rendering and check mode.
- [ ] Add the poster master and generate committed SVG/PNG derivatives.
- [ ] Add the Cairo dependency and CI system package.
- [ ] Run extractor and projection tests to green.

### Task 3: Accurate claims

**Files:**
- Modify: `README.md`
- Modify: `docs/concepts.md`
- Modify: `docs/lm.md`
- Modify: `docs/comparison.md`
- Modify: `tests/test_docs_projection.py`

- [ ] Add regression assertions that reject the audited unsupported phrases.
- [ ] Confirm the assertions fail on current copy.
- [ ] Correct lifecycle cadence and qualify or source comparative/performance claims.
- [ ] Run focused tests to green.

### Task 4: Full verification and publication

**Files:**
- Verify all changed files and generated projections.

- [ ] Run diagram checks, projection checks, wiki checks, strict MkDocs build, ruff, pyright, and pytest.
- [ ] Commit and push the implementation branch.
- [ ] Open and merge a PR into `develop` after checks pass.
- [ ] Open and merge a PR from `develop` into `main` after checks pass.
- [ ] Confirm the main documentation deployment succeeds.
