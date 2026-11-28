---
id: FEAT-065
title: "[FEAT-065] Remove the Python 3.8 version shim and its stale comment"
labels: ["enhancement", "area: packaging", "priority: P3", "review: 2026-11-27"]
sources: ["DEAD-04", "DOC-35"]
depends_on: []
related_issues: [418]
status: READY
---

**Category:** Packaging · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop)

## Summary

FEAT-065: `src/nnx/__init__.py` wraps its `__version__` lookup in a `try/except ImportError` for Python < 3.8, which cannot run under `requires-python = ">=3.11"`, and that dead branch holds a second hard-coded version string release-please has to keep bumping. Its comment also tells contributors to bump the version by hand, which contradicts release-please. Remove the outer `try`, keep one marked fallback line, and fix the comment.

## Context

**What happens today.**

```python
try:
    # Underscore-aliased so neither symbol leaks into `nnx.*` — these are
    # implementation details of `__version__` lookup, not public nnx API.
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        __version__ = _version("thekaveh-nnx")
    except _PackageNotFoundError:
        # Editable install before metadata exists, or run from the source
        # tree without installation. Keep in sync with pyproject.toml
        # [project] version on every bump.
        __version__ = "0.5.0"  # x-release-please-version
except ImportError:  # pragma: no cover — Python <3.8.
    __version__ = "0.5.0"  # x-release-please-version
```

`pyproject.toml` L20 says the opposite of the comment: `# Managed by release-please (CONTRIBUTING §7); never bump it by hand.`

**Why it matters.** Two marker lines for one value, one of them unreachable, and a comment that invites a manual bump.

**History.** #418 (FIX-029) made release PRs pass without a manual finalize commit: release-please bumps every line carrying `x-release-please-version` in the files `release-please-config.json` lists as `extra-files` (`src/nnx/__init__.py` is one, as a `generic` updater). #51 underscore-aliased the `importlib.metadata` names so they do not leak as `nnx.PackageNotFoundError`; `tests/test_imports.py::test_packagenotfounderror_is_not_a_public_nnx_symbol` guards that and must keep passing.

**Background for the worker.**
- *release-please marker*: a line containing `x-release-please-version` in a `generic` extra-file; on a release PR, release-please replaces the version string on that line. A comment line that contains the marker but no version would still count as a marker line, so the new comment must not contain the marker text.
- *`requires-python`*: `pyproject.toml` declares `>=3.11`; `importlib.metadata` exists from Python 3.8.

**Proposal.**

1. In `src/nnx/__init__.py`, replace L13–L32 (from `from importlib import import_module as _import_module` through the second `__version__ = "0.5.0"  # x-release-please-version`) with:
   ```python
   from importlib import import_module as _import_module

   # Underscore-aliased so neither symbol leaks into `nnx.*`: they are
   # implementation details of the `__version__` lookup, not public nnx API.
   from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
   from importlib.metadata import version as _version
   from typing import TYPE_CHECKING, Any

   from . import _optional

   try:
       __version__ = _version("thekaveh-nnx")
   except _PackageNotFoundError:
       # Editable install before metadata exists, or run from the source tree
       # without installation. release-please updates this line through its
       # marker on every release; never edit the version by hand.
       __version__ = "0.5.0"  # x-release-please-version
   ```
   Keep the version string exactly as it is on `develop` when you start (release-please may have bumped it). If `uv run ruff check src/nnx/__init__.py` reports an import-order (I001) finding, apply `uv run ruff check --fix src/nnx/__init__.py`.
2. Add this test to `tests/test_release_finalize.py`, after `test_release_please_bumps_the_readme_version_line_and_its_pypi_copy_alike`:
   ```python
   def test_the_package_init_has_one_marked_version_line_matching_pyproject():
       """FEAT-065: release-please bumps one fallback version line in
       src/nnx/__init__.py, and it equals pyproject.toml's version."""
       import tomllib

       init = (ROOT / "src" / "nnx" / "__init__.py").read_text(encoding="utf-8")
       marked = [line for line in init.split("\n") if "x-release-please-version" in line]
       version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
       assert marked == [f'    __version__ = "{version}"  # x-release-please-version']
       assert "pragma: no cover" not in init
   ```

No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_release_finalize.py::test_the_package_init_has_one_marked_version_line_matching_pyproject` fails on `develop` (two marked lines) and passes after the change. Check: `uv run pytest tests/test_release_finalize.py::test_the_package_init_has_one_marked_version_line_matching_pyproject`
- [ ] The comment no longer says to keep the version in sync by hand. Check: `grep -c "on every bump" src/nnx/__init__.py` prints `0`
- [ ] `nnx.__version__` still resolves and `nnx.PackageNotFoundError` still does not exist. Check: `uv run pytest tests/test_imports.py tests/test_release_finalize.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [__init__.py L13-L32](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/__init__.py#L13-L32) | The outer `try/except ImportError`, the two marker lines and the "on every bump" comment |
| [pyproject.toml L20-L21](https://github.com/thekaveh/NNx/blob/9ad461e/pyproject.toml#L20-L21), [L29](https://github.com/thekaveh/NNx/blob/9ad461e/pyproject.toml#L29) | "Managed by release-please … never bump it by hand"; `requires-python = ">=3.11"` |
| [release-please-config.json L13-L18](https://github.com/thekaveh/NNx/blob/9ad461e/release-please-config.json#L13-L18) | `src/nnx/__init__.py` is a `generic` extra-file |
| [test_imports.py L136-L152](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_imports.py#L136-L152) | The `PackageNotFoundError` leak guard (#51) |
| #418 | Release PRs pass without a manual finalize commit |

## Scope

**In:** `src/nnx/__init__.py` L13–L32, one test.

**Out:** any version bump (release-please owns it); the README and `docs/api.md` marker lines.

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #418
