---
id: FEAT-066
title: "[FEAT-066] Remove torch-version guards below the torch 2.4 floor"
labels: ["enhancement", "area: numerics", "priority: P3", "review: 2026-11-27"]
sources: ["DEAD-05"]
depends_on: []
related_issues: [210, 572]
status: READY
---

**Category:** Numerics · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-066: four places guard against torch versions older than the declared `torch>=2.4` floor: three `hasattr(torch.backends, "mps")` checks and a `hasattr(torch, "float8_…")` loop. `torch.backends.mps` exists since torch 1.12 and both float8 dtypes since 2.1. Drop the guards, and add a comment to the two `getattr(nn.modules…, "_…", None)` guards in `models.py`, which protect against a *future* torch removing private names and stay.

## Context

**What happens today.**

```python
    @staticmethod
    def get() -> Devices:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return Devices.MPS
```

```python
for _name, _code in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")):
    if hasattr(torch, _name):
        _CODES[getattr(torch, _name)] = _code
```

The same MPS guard sits in `bundles._missing_device` (L1414) and, as `getattr(torch.backends, "mps", None)`, in `precision.precision_support` (L374).

**Why it matters.** The guards suggest NNx supports torch builds it does not, and the float8 loop hides two constant table entries behind run-time logic.

**History.** #210 (FIX-011) raised the floor to `torch>=2.4` after torch 2.3.1 failed 23 core tests; CI's `floor-deps` job runs the core suite at torch 2.4.1. #572's MPS pins monkeypatch only `torch.backends.mps.is_available` (`tests/test_mps_cpu_pins.py:41`, `:48`; `tests/test_train_integration.py:351`), which keeps working when the `hasattr` goes.

**Background for the worker.**
- *`_CODES`*: the dtype → safetensors dtype-code table in `bundles.py` that NNx's own safetensors writer and reader use; `_BY_CODE` is its inverse, built right after it.
- *Private-name guards*: `models.py` reads `nn.modules.conv._LazyConvXdMixin`, `nn.modules.batchnorm._LazyNormBase`, `nn.modules.conv._ConvNd` and `nn.modules.batchnorm._NormBase` with `getattr(..., None)`, because torch may rename private classes in any release.

**Proposal.**

1. `src/nnx/nn/enum/devices.py` L29: change the condition to `if torch.backends.mps.is_available():`.
2. `src/nnx/bundles.py`:
   - add the two float8 entries at the end of the `_CODES` dict (after `torch.bool: "BOOL",`, L654):
     ```python
         torch.float8_e4m3fn: "F8_E4M3",
         torch.float8_e5m2: "F8_E5M2",
     ```
   - delete the loop L656–L658 (`for _name, _code in ...` and its two lines). `_BY_CODE = {...}` (L659) stays.
   - L1414: change the condition to `if value == "mps" and not torch.backends.mps.is_available():`.
3. `src/nnx/precision.py` L374: change the entry to `        "mps": torch.backends.mps.is_available(),`.
4. `src/nnx/models.py`: replace the comment above `_LAZY_CONV` (L410) with
   ```python
   # The lazy layer families torch ships, whose dry-run inference NNx emulates.
   # getattr: these are private torch names a future release may remove (not a
   # guard for torch versions below the floor).
   ```
   and the comment above `_CONV` (L474–L475) with
   ```python
   # The layer families torch's lazy layers become, whose attributes a dry run
   # infers from the input. getattr: these are private torch names a future
   # release may remove; an absent one is skipped.
   ```
   The four `getattr(...)` lines stay as they are.
5. Add this test to `tests/test_mps_cpu_pins.py`:
   ```python
   def test_no_guard_checks_for_a_torch_older_than_the_floor():
       """FEAT-066: torch>=2.4 always has torch.backends.mps and both float8
       dtypes, so NNx never tests for them."""
       import pathlib

       import nnx
       from nnx.bundles import _BY_CODE, _CODES

       root = pathlib.Path(nnx.__file__).parent
       guards = [
           f"{path.relative_to(root)}:{number}"
           for path in root.rglob("*.py")
           for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
           if 'hasattr(torch.backends, "mps")' in line
           or 'getattr(torch.backends, "mps"' in line
           or 'hasattr(torch, "float8' in line
           or "hasattr(torch, _name)" in line
       ]
       assert guards == []
       assert _CODES[torch.float8_e4m3fn] == "F8_E4M3" and _BY_CODE["F8_E5M2"] is torch.float8_e5m2
   ```

No behaviour changes on any supported torch, so no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need. CI's `floor-deps` job proves the change on torch 2.4.1.

## Acceptance criteria

- [ ] The new test `tests/test_mps_cpu_pins.py::test_no_guard_checks_for_a_torch_older_than_the_floor` fails on `develop` and passes after the change. Check: `uv run pytest tests/test_mps_cpu_pins.py::test_no_guard_checks_for_a_torch_older_than_the_floor`
- [ ] The MPS pins, bundle, precision and lazy-module suites pass. Check: `uv run pytest tests/test_mps_cpu_pins.py tests/test_run_bundles.py tests/test_bundle_validation.py tests/test_precision_policy.py tests/test_external_module_runtime.py`
- [ ] The `models.py` private-name guards carry the new comments. Check: `grep -c "private torch names a future" src/nnx/models.py` prints `2`
- [ ] The `floor-deps` CI job (torch 2.4.1) is green on the PR. Check: the PR's checks page
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [devices.py L27-L33](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/enum/devices.py#L27-L33) | `Devices.get`'s MPS guard |
| [bundles.py L643-L659](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L643-L659) | `_CODES` and the float8 loop |
| [bundles.py L1409-L1416](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L1409-L1416) | `_missing_device`'s MPS guard |
| [precision.py L371-L375](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/precision.py#L371-L375) | `precision_support`'s MPS guard |
| [models.py L410-L412](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L410-L412), [L474-L477](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L474-L477) | The private-name guards that stay |
| [test_mps_cpu_pins.py L41](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_mps_cpu_pins.py#L41), [L48](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_mps_cpu_pins.py#L48) | #572's pins patch only `is_available` |
| #210 | The `torch>=2.4` floor |

## Scope

**In:** the four guards, the two comments, one test.

**Out:** raising the torch floor; the private-name `getattr` guards themselves.

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #210, #572
