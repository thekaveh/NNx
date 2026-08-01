# NNx Poster Wordmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unexplained crystal cluster in the NNx banner with a prominent, integrated `NNx` wordmark.

**Architecture:** Edit the canonical raster banner directly with the existing PNG as the sole image reference. Preserve its neural-layer composition and dimensions, then rely on the existing deterministic documentation projection to carry the revised asset to MkDocs, wiki, and package surfaces.

**Tech Stack:** Built-in image generation edit mode, PNG, Pillow-backed asset tests, pytest, MkDocs Material

## Global Constraints

- The only in-image text is the exact case-sensitive string `NNx`.
- Remove every crystal, floating cube, and pedestal from the right third.
- Preserve the black panoramic backdrop, central flare, neural layers, and left-to-right data flow.
- Keep the output at least 1200 pixels wide with an aspect ratio of at least 2:1.
- Replace only `docs/assets/nnx-poster.png`; do not change opener copy or badge layout.

---

### Task 1: Edit and Validate the Canonical Banner

**Files:**
- Modify: `docs/assets/nnx-poster.png`
- Test: `tests/test_docs_projection.py`

**Interfaces:**
- Consumes: the existing `docs/assets/nnx-poster.png` composition and the opener's unchanged image path
- Produces: a replacement PNG consumed automatically by README, MkDocs, wiki, and PyPI projections

- [ ] **Step 1: Inspect the edit target**

Open `docs/assets/nnx-poster.png` at full size and confirm the neural layers to preserve and crystal cluster to remove.

- [ ] **Step 2: Edit the banner with the built-in image tool**

Use the existing PNG as the edit target with this exact production prompt:

```text
Use case: precise-object-edit
Asset type: wide open-source repository README banner
Primary request: Remove the entire crystal cluster, all floating cubes, and all pedestals from the right third. Replace that region with the exact wordmark "NNx".
Composition/framing: Preserve the 1983 x 793 panoramic framing, black backdrop, left and central neural layers, central flare, and left-to-right data flow. Put the wordmark in the cleared right third with generous breathing room.
Style/medium: polished cinematic 3D technology artwork matching the existing image
Text (verbatim): "NNx"
Wordmark treatment: large bold geometric letterforms, cyan-white internal illumination, restrained electric-blue edge glow, with only a few neural lines and nodes connecting the network into the letters so the wordmark reads as the network's output
Constraints: preserve all non-right-side visual structure; keep the wordmark immediately legible at reduced README width; no separate label, card, or logo panel
Avoid: any text other than NNx, misspelling, alternate capitalization, crystals, trophies, coins, blockchain imagery, pedestals, floating cubes, chips, robots, brains, icons, watermark, tagline, subtitle
```

- [ ] **Step 3: Install and inspect the generated PNG**

Copy the selected generated output to `docs/assets/nnx-poster.png`, then inspect it at full size and reduced width. Confirm the exact text, crystal removal, coherent composition, and readable contrast.

- [ ] **Step 4: Run focused asset and projection tests**

Run:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/cairo/lib .venv/bin/pytest -q \
  tests/test_docs_projection.py::test_product_banner_is_a_raster_only_brand_asset \
  tests/test_docs_projection.py::test_primary_openers_share_centered_brand_contract_and_executive_summary
```

Expected: both tests pass.

- [ ] **Step 5: Run deterministic documentation verification**

Run:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/cairo/lib .venv/bin/python -m scripts.docs.build_docs --check
.venv/bin/python -m scripts.docs.build_wiki --check
.venv/bin/python -m scripts.docs.build_pypi_readme --check
.venv/bin/mkdocs build --strict
```

Expected: all projection checks pass and MkDocs exits successfully.

- [ ] **Step 6: Commit the replacement asset**

```bash
git add docs/assets/nnx-poster.png
git commit -m "docs: integrate NNx wordmark into poster"
```
