---
id: FEAT-069
title: "[FEAT-069] Shrink the README banner under the 500 KB file limit"
labels: ["enhancement", "area: documentation", "priority: P3", "review: 2026-11-27"]
sources: ["CLUT-03"]
depends_on: []
related_issues: []
status: READY
---

**Category:** Documentation · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); the Pillow fallback was run on a copy of the banner

## Summary

FEAT-069: `docs/assets/nnx-poster.png` is 2,445,656 bytes (2304×921), five times the repository's own `check-added-large-files --maxkb=500` limit, which only checks newly added files, so updates slip past it. Recompress it in place to under 512,000 bytes as a PNG that keeps the width (≥ 1200) and aspect ratio (≥ 2) the docs test asserts, and add a test that keeps it under the limit.

## Context

**What happens today.** `.pre-commit-config.yaml` L26–L27:

```yaml
      - id: check-added-large-files
        args: [--maxkb=500]
```

The banner is 2,445,656 bytes, and the history already holds three revisions of it. `tests/test_docs_projection.py::test_product_banner_is_a_raster_only_brand_asset` pins the format:

```python
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", payload[16:24])
    assert width >= 1200
    assert width / height >= 2
```

**Why it matters.** Every clone and every README render on GitHub and PyPI downloads 2.4 MB for one image; the repository breaks its own size rule.

**History.** No board issue covers the banner size.

**Background for the worker.**
- *500 KB*: `--maxkb=500` means 500 × 1024 = 512,000 bytes.
- *`pngquant`*: a command-line PNG quantizer (palette of at most 256 colours, with dithering). Install with `brew install pngquant` (macOS) or `sudo apt-get install pngquant` (Debian/Ubuntu).
- *Pillow*: the Python imaging library; `uv sync --frozen --all-extras` installs it (matplotlib and torchvision depend on it), so `uv run python` can use it with nothing else installed.
- *Measured fallback*: on a copy of the `9ad461e` banner, the Pillow script below (Pillow 13.0.0) gave a 1920×768 PNG of 444,229 bytes with the fabric texture and the wordmark intact; quantizing at full width with Pillow's fast-octree method gave 479,793 bytes but blotchy dark areas, so the script resizes first.

**Proposal.**

1. Try `pngquant` first. If `pngquant --version` exits 0, run:
   ```bash
   pngquant --quality=70-90 --strip --force --output docs/assets/nnx-poster.png docs/assets/nnx-poster.png
   ```
   Keep its result only if the command exits 0 and `stat -c%s docs/assets/nnx-poster.png` (macOS: `stat -f%z`) is below 512000.
2. Otherwise (pngquant missing, a non-zero exit, or a file of 512,000 bytes or more), restore the original with `git checkout -- docs/assets/nnx-poster.png` and run the Pillow fallback:
   ```bash
   uv run python - <<'PY'
   from PIL import Image

   path = "docs/assets/nnx-poster.png"
   image = Image.open(path).convert("RGB").resize((1920, 768), Image.Resampling.LANCZOS)
   image.quantize(colors=256, method=Image.Quantize.MAXCOVERAGE).save(path, optimize=True)
   PY
   ```
   1920×768 keeps the 2.5 aspect ratio of the original.
3. Open the result and compare it with the original side by side: the "NNX" wordmark, the gold contour lines and the dark fabric texture must stay visible, with no large flat blotches. Name the tool and the final size in the PR body.
4. Add this test to `tests/test_docs_projection.py`, below `test_product_banner_is_a_raster_only_brand_asset`:
   ```python
   def test_product_banner_stays_under_the_large_file_limit():
       """FEAT-069: pre-commit's check-added-large-files (--maxkb=500) checks only
       new files, so an updated banner could grow past it unnoticed."""
       banner = build_docs.ROOT / "docs" / "assets" / "nnx-poster.png"
       assert banner.stat().st_size < 500 * 1024
   ```

`README.md`, `docs/index.md` and `PYPI_README.md` reference the file by path, so they need no change. No library behaviour changes, so no CHANGELOG entry.

**Worker setup.** No torch needed (docs asset only); `uv sync --frozen --all-extras` installs Pillow and what the docs tests import.

## Acceptance criteria

- [ ] The new test `tests/test_docs_projection.py::test_product_banner_stays_under_the_large_file_limit` fails on `develop` and passes after the change. Check: `uv run pytest tests/test_docs_projection.py::test_product_banner_stays_under_the_large_file_limit`
- [ ] The banner is a PNG under 512,000 bytes, at least 1200 px wide, with an aspect ratio of at least 2. Check: `test $(stat -c%s docs/assets/nnx-poster.png) -lt 512000 && uv run pytest tests/test_docs_projection.py -k banner`
- [ ] The PR body names the tool used (pngquant or the Pillow fallback) and the final size, and shows the new banner. Check: the PR body
- [ ] The docs build passes. Check: `uv run python -m scripts.docs.build_docs --check`

## Evidence

| Location | What it shows |
|---|---|
| [docs/assets/nnx-poster.png](https://github.com/thekaveh/NNx/blob/9ad461e/docs/assets/nnx-poster.png) | 2,445,656 bytes, 2304×921 |
| [.pre-commit-config.yaml L26-L27](https://github.com/thekaveh/NNx/blob/9ad461e/.pre-commit-config.yaml#L26-L27) | The 500 KB limit on added files |
| [test_docs_projection.py L197-L206](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_docs_projection.py#L197-L206) | The PNG, width and aspect-ratio assertions |
| [README.md L2](https://github.com/thekaveh/NNx/blob/9ad461e/README.md#L2), [docs/index.md L2](https://github.com/thekaveh/NNx/blob/9ad461e/docs/index.md#L2), [PYPI_README.md L2](https://github.com/thekaveh/NNx/blob/9ad461e/PYPI_README.md#L2) | The three places that show the banner |

## Scope

**In:** recompressing the one PNG, one size test.

**Out:** redesigning the banner; switching to another image format (the docs test requires PNG); rewriting git history to drop the old revisions.

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** none
