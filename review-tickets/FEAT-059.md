---
id: FEAT-059
title: "[FEAT-059] Load nnx.decisions and nnx.export_conformance lazily on import nnx"
labels: ["enhancement", "area: packaging", "priority: P3", "review: 2026-11-27"]
sources: ["SCOPE-01", "SCOPE-03"]
depends_on: []
related_issues: [219, 220, 243, 245, 234, 263, 255, 249, 566]
status: READY
---

**Category:** Packaging · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-059: every `import nnx` imports `nnx.decisions` (about 4,900 lines) and `nnx.export_conformance` (1,362 lines), although nothing in the training core uses either. Serve both submodules on first access through the package's existing lazy `__getattr__`, so `nnx.decisions`, `from nnx import decisions` and `import nnx.export_conformance` keep working unchanged.

## Context

**What happens today.** Both modules are in the eager import list of `src/nnx/__init__.py`:

```python
from . import (
    abstention,
    benchmarking,
    bundles,
    calibration,
    comparison,
    compilation,
    components,
    data_splits,
    decisions,
    distributed,
    embeddings,
    export_conformance,
```

The lazy `__getattr__` only knows class names that need an extra; it returns an attribute of a module, never a module:

```python
def __getattr__(name: str) -> Any:
    entry = _LAZY_DOMAIN_NAMES.get(name)
    if entry is None:
        raise AttributeError(f"module 'nnx' has no attribute {name!r}")
    value = getattr(_import_module(entry[0], __name__), name)
    globals()[name] = value  # resolved once
    return value
```

Nothing else in the package imports either module at module level: `abstention.decide` and the `result.py` decision boundaries import `nnx.decisions` inside the functions that need it, and no module imports `nnx.export_conformance`. So removing the two names from the eager list really stops them loading.

**Why it matters.** A bare `import nnx` pays for two products the training core never calls. This is a load-time cost only; where the code lives is not in question.

**History.**
- Keeping `nnx.decisions` inside NNx is deliberate: #219 (FEAT-009) created it as part of NNx, and #220, #234, #243, #245 and #263 built on it. #219's acceptance criteria say "A fresh process that imports `nnx` and `nnx.decisions` starts no provider or model backend". Whether it moves to its own distribution is DEC-017; this ticket only changes when it loads.
- `nnx.export_conformance` is public by #255 (FEAT-038), which required "`python -c "import nnx"` succeeds in an environment without the new extra". Whether it leaves the runtime package is DEC-018.
- #249 (FEAT-031) added `_LAZY_DOMAIN_NAMES` and the `__getattr__` / `__dir__` pair this ticket extends.
- `tests/test_imports.py:530` asserts `"decisions" in nnx.__all__ and nnx.decisions is decisions`. That must keep passing unchanged.

**Background for the worker.**
- *Module `__getattr__`* (PEP 562): Python calls a module's `__getattr__(name)` only when normal attribute lookup fails. Returning a value makes `nnx.<name>` work.
- *`_LAZY_DOMAIN_NAMES`*: the table in `src/nnx/__init__.py` that maps a class name (`NNDataset`, the graph nets) to `(module, extra)`; `__getattr__` imports the module and returns the class.
- *Submodule binding*: importing `nnx.decisions` by any route (`import nnx.decisions`, `importlib.import_module(".decisions", "nnx")`) sets the attribute `decisions` on the `nnx` package, so `nnx.decisions is sys.modules["nnx.decisions"]` always holds afterwards.
- *`__dir__`*: the module-level function that `dir(nnx)`, pydoc and `inspect.getmembers` use; it must list only names `getattr` can return.
- *`TYPE_CHECKING` imports*: a `from . import x` under `if TYPE_CHECKING:` is seen by pyright, never executed at run time.

**Proposal.**

1. In `src/nnx/__init__.py`, delete the two lines `    decisions,` (L43) and `    export_conformance,` (L46) from the eager `from . import (...)` list. Leave `"decisions"` and `"export_conformance"` in `__all__` unchanged.
2. Directly after the `_LAZY_DOMAIN_NAMES = {...}` dict (ends L278), add:
   ```python
   # Submodules served on first access (FEAT-059): a bare `import nnx` loads
   # neither nnx.decisions nor nnx.export_conformance. Neither needs an extra.
   _LAZY_SUBMODULES = frozenset({"decisions", "export_conformance"})
   ```
3. In the `if TYPE_CHECKING:` block (L280–L285), add the line `    from . import decisions, export_conformance` as its first line, so pyright still knows the types of `nnx.decisions` and `nnx.export_conformance`.
4. Replace the body of `__getattr__` (L288–L294) with:
   ```python
   def __getattr__(name: str) -> Any:
       if name in _LAZY_SUBMODULES:
           module = _import_module(f".{name}", __name__)
           globals()[name] = module  # resolved once
           return module
       entry = _LAZY_DOMAIN_NAMES.get(name)
       if entry is None:
           raise AttributeError(f"module 'nnx' has no attribute {name!r}")
       value = getattr(_import_module(entry[0], __name__), name)
       globals()[name] = value  # resolved once
       return value
   ```
5. In `__dir__` (L297–L301), change the return line to:
   ```python
       return sorted(set(globals()) | usable | _LAZY_SUBMODULES)
   ```
6. Add this test at the end of `tests/test_imports.py`:
   ```python
   def test_import_nnx_loads_decisions_and_export_conformance_on_first_access():
       """FEAT-059: a bare `import nnx` imports neither nnx.decisions nor
       nnx.export_conformance; first access loads each, as the same module
       object `import nnx.<name>` gives, and dir(nnx) lists both."""
       code = r"""
   import sys
   import nnx
   assert "nnx.decisions" not in sys.modules, "import nnx loaded nnx.decisions"
   assert "nnx.export_conformance" not in sys.modules, "import nnx loaded nnx.export_conformance"
   assert {"decisions", "export_conformance"} <= set(dir(nnx))
   decisions = nnx.decisions
   assert decisions is sys.modules["nnx.decisions"]
   assert hasattr(decisions, "FixedHeadProvider")
   import nnx.export_conformance as conformance
   assert nnx.export_conformance is conformance and hasattr(conformance, "validate_record")
   try:
       nnx.no_such_name
   except AttributeError:
       pass
   else:
       raise AssertionError("an unknown name did not raise AttributeError")
   """
       subprocess.run([sys.executable, "-c", code], check=True)
   ```
7. Add this bullet at the end of `### Changed` under `## [Unreleased]` in `CHANGELOG.md`:
   ```markdown
   - **`import nnx` no longer imports `nnx.decisions` and `nnx.export_conformance`.** Each loads on first use (`nnx.decisions`, `from nnx import decisions`, `import nnx.export_conformance`) as the same module as before, so existing code keeps working, and a bare `import nnx` skips about 6,200 lines it did not use.
   ```

No docstring that `docs/api.md` renders changes, so the API reference needs no rebuild.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_imports.py::test_import_nnx_loads_decisions_and_export_conformance_on_first_access` fails on `develop` before the change (its first assert) and passes after it. Check: `uv run pytest tests/test_imports.py::test_import_nnx_loads_decisions_and_export_conformance_on_first_access`
- [ ] `tests/test_imports.py:530` still holds unchanged (`nnx.decisions is decisions`), and every import and public-API test passes. Check: `uv run pytest tests/test_imports.py tests/test_public_api_exports.py tests/test_core_install.py`
- [ ] `import nnx` still works without the `onnx-runtime` extra (#255). Check: `uv run pytest tests/test_export_conformance.py::test_core_import_does_not_need_the_runtime_extra`
- [ ] The decision and export-conformance suites pass. Check: `uv run pytest tests/test_decision_*.py tests/test_result_boundaries.py tests/test_abstention.py tests/test_export_conformance.py`
- [ ] `decisions` and `export_conformance` no longer appear in the eager import list. Check: `sed -n '/^from \. import (/,/^)/p' src/nnx/__init__.py | grep -cE "^\s+(decisions|export_conformance),"` prints `0`
- [ ] The CHANGELOG bullet from step 7 is under `## [Unreleased]` → `### Changed`. Check: `grep -c "no longer imports \`nnx.decisions\`" CHANGELOG.md` prints `1`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [__init__.py L34-L69](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/__init__.py#L34-L69) | The eager `from . import (...)` list, with `decisions` (L43) and `export_conformance` (L46) |
| [__init__.py L272-L301](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/__init__.py#L272-L301) | `_LAZY_DOMAIN_NAMES`, the `TYPE_CHECKING` block, `__getattr__` and `__dir__` |
| [abstention.py L1462](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/abstention.py#L1462) | `decide` imports `nnx.decisions.schema` inside the function |
| [result.py L223](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/result.py#L223), [L272](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/result.py#L272), [L296](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/result.py#L296) | The decision boundaries import `nnx.decisions` inside functions |
| [test_imports.py L516-L540](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_imports.py#L516-L540) | `nnx.decisions is decisions` (L530) and the no-hosted-SDK check |
| [test_export_conformance.py L348](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_export_conformance.py#L348) | `test_core_import_does_not_need_the_runtime_extra` (#255) |
| #219, #255, #249 | Decisions inside NNx by design; `import nnx` without the conformance extra; the lazy `__getattr__` |

## Scope

**In:** `src/nnx/__init__.py` (eager list, `__getattr__`, `__dir__`, `TYPE_CHECKING`), one new test, one CHANGELOG bullet.

**Out:**
- Moving `abstention.decide` into `nnx.decisions` and breaking the import cycle: FEAT-060.
- Splitting `export_conformance.validate_record`: FEAT-061.
- Moving `nnx.decisions` to its own distribution: DEC-017. Moving `nnx.export_conformance` out of the runtime package: DEC-018.
- Making other subpackages lazy.

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-060, DEC-017, DEC-018
**Related:** #219, #220, #234, #243, #245, #263, #255, #249, #566
