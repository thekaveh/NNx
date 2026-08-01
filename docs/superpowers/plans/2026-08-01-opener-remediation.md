# NNx Opener Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken numbered, left-aligned NNx opener and inaccurate workflow poster with a reference-quality shared opener across the repository, MkDocs, wiki, and PyPI surfaces.

**Architecture:** Extend the documentation manifest with an explicit H1-numbering override while preserving numbered navigation. Express the opener in GitHub-compatible HTML shared by the two canonical sources, project it through the existing generators, and protect its structure and stack coverage with focused tests.

**Tech Stack:** Python 3.10+, pytest, YAML manifest, Markdown/HTML, MkDocs Material, shields.io badges, CairoSVG documentation assets.

## Global Constraints

- The brand H1 is exactly `NNx`, centered, and unnumbered on README, Home, wiki, site, and PyPI.
- Navigation labels remain `1. Home` and `15. Repository overview`.
- The banner precedes the H1 and contains no baked text or architecture labels.
- Badge rows distinguish project status from core and optional technology stacks.
- The executive summary remains identical across canonical openers and remains 100-150 words.
- All surfaces remain self-contained and deterministic.

---

### Task 1: Manifest H1 Numbering Override

**Files:**
- Modify: `tests/test_docs_projection.py`
- Modify: `scripts/docs/build_docs.py`
- Modify: `docs/manifest.yaml`

**Interfaces:**
- Consumes: manifest page mappings.
- Produces: `Page.numbered_h1: bool`, defaulting to `True`.

- [ ] **Step 1: Write failing tests** asserting Home and Repository Overview may opt out while all other manifest pages retain numbered H1 validation.
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_docs_projection.py -k 'manifest or opener'` and confirm failure because `numbered_h1` is unsupported.
- [ ] **Step 3: Add** `numbered_h1: bool = True` to `Page`, accept only a boolean manifest value, and skip baked-number validation only when false.
- [ ] **Step 4: Set** `numbered_h1: false` for `docs/index.md` and `README.md`; change their H1s to unnumbered centered HTML in Task 2.
- [ ] **Step 5: Run the focused tests** and commit the schema behavior.

### Task 2: Shared Opener Contract

**Files:**
- Modify: `tests/test_docs_projection.py`
- Modify: `tests/test_pypi_publication.py`
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `scripts/docs/build_pypi_readme.py` only if projection requires adjustment.

**Interfaces:**
- Consumes: canonical opener HTML and existing executive-summary markers.
- Produces: identical semantic opener blocks with surface-local banner paths.

- [ ] **Step 1: Replace existing permissive opener assertions with failing tests** that require banner-before-H1 ordering, `<h1 align="center">NNx</h1>`, centered tagline/support copy, status badges, core badges, optional badges, and no numbered product H1.
- [ ] **Step 2: Run focused tests** and confirm failures against the existing Markdown opener.
- [ ] **Step 3: Implement the shared opener structure** in `README.md` and `docs/index.md`, using local image paths and honest shields for PyTorch, PyTorch Geometric, NumPy, pandas, scikit-learn, Plotly, TensorBoard, W&B, ONNX, torchao, Hugging Face, safetensors, and FAISS.
- [ ] **Step 4: Regenerate `PYPI_README.md`** and assert the banner URL is absolute while the unnumbered centered H1 and badges are preserved.
- [ ] **Step 5: Run focused tests** and commit the opener contract.

### Task 3: Brand Banner And Responsive Styling

**Files:**
- Replace: `docs/assets/nnx-poster.png`
- Remove: `docs/assets/nnx-poster.svg`
- Remove: `docs/diagrams/nnx-poster.html`
- Modify: `scripts/docs/extract_architecture_svg.py`
- Modify: `docs/stylesheets/extra.css`
- Modify: `tests/test_docs_projection.py`

**Interfaces:**
- Consumes: generated raster banner artwork.
- Produces: a committed local PNG projected to site/wiki/PyPI without SVG rewriting.

- [ ] **Step 1: Add failing tests** that the poster is a raster-only brand asset, is wider than tall, and its former SVG/HTML architecture master is absent from the diagram renderer.
- [ ] **Step 2: Run focused tests** and confirm the current SVG/HTML poster fails them.
- [ ] **Step 3: Generate and visually inspect** text-free brand artwork evoking transparent neural layers, graph connectivity, repeatable training, and durable checkpoints; install it as `docs/assets/nnx-poster.png`.
- [ ] **Step 4: Remove the old poster master/SVG mapping** and add site CSS for centered opener spacing, full-width banner containment, and wrapping badge rows.
- [ ] **Step 5: Regenerate projections, inspect PNG copies, run focused tests, and commit.**

### Task 4: CI Coverage And End-to-End Verification

**Files:**
- Modify: `.github/workflows/security.yml`
- Modify: `tests/test_docs_projection.py`
- Regenerate: `PYPI_README.md`

**Interfaces:**
- Consumes: completed opener and projection contracts.
- Produces: push and PR security coverage for `main` and `develop`.

- [ ] **Step 1: Add a failing workflow assertion** requiring security push branches `[main, develop]`.
- [ ] **Step 2: Run the assertion** and confirm failure because security currently has no push trigger.
- [ ] **Step 3: Add the push trigger**, then run all 245 documentation tests.
- [ ] **Step 4: Run** diagram staleness checks, deterministic docs/wiki/PyPI checks, `mkdocs build --strict`, Ruff, Pyright, and the full test suite.
- [ ] **Step 5: Inspect repository, generated site, generated wiki, and PyPI opener outputs; confirm clean git status except intended files; commit the remediation.**
