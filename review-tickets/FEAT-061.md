---
id: FEAT-061
title: "[FEAT-061] Split export_conformance.validate_record and share one SHA-256 check"
labels: ["enhancement", "area: interoperability", "priority: P3", "review: 2026-11-27"]
sources: ["SCOPE-03"]
depends_on: []
related_issues: [255, 566]
status: READY
---

**Category:** Interoperability · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-061: `export_conformance.validate_record` is one 243-line function, and three modules each spell their own "is this a SHA-256 hex digest" check. Split `validate_record` along its existing sections into six private helpers called in the same order, and add one `_validation.is_sha256_hex` used by `export_conformance`, `bundles` and `nn_model`. Every refusal message and its order stay byte-identical.

## Context

**What happens today.** `validate_record` (L1105–L1347) runs every check inline, sharing one `problems` list and a nested `status` helper:

```python
    stages = record["stages"]
    if not isinstance(stages, Mapping) or set(stages) != set(STAGES):
        raise ConformanceError(f"stages must be exactly {list(STAGES)}, got {stages!r}")

    def status(name: str) -> Any:
        stage = stages[name]
        return stage.get("status") if isinstance(stage, Mapping) else None
```

The SHA-256 check is written out twice inside it:

```python
        if not (
            isinstance(entry["sha256"], str)
            and len(entry["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in entry["sha256"])
        ):
            problems.append(f"artifact {entry['path']!r} has a malformed sha256")
```

and two more modules keep their own regex: `bundles._SHA256 = re.compile(r"[0-9a-f]{64}")` (L110) and `nn_model._SHA256_HEX = re.compile(r"[0-9a-f]{64}")` (L191).

**Why it matters.** The function is hard to read and change safely, and three spellings of one rule can drift apart.

**History.** #255 (FEAT-038) made `nnx.export_conformance` public, scoped to the FP32 feed-forward profiles; #566 added the checks of names and profile folders before writing. Its placement in the runtime package is deliberate per #255; moving it is DEC-018. This ticket changes no behaviour.

**Background for the worker.**
- *Conformance record*: the JSON document `run_profile` writes for one export profile (format `nnx.export-conformance/1`); `validate_record` refuses any record that does not follow it, collecting every problem into one `ConformanceError` message joined by `"; "`.
- *Early raises*: three conditions raise at once instead of collecting a problem: a record that is not a mapping, missing keys, and `stages` that are not exactly `STAGES`. They must still raise at the same point.
- *Problem order*: tests and users read the joined message, so the helpers must append problems in exactly today's order. Calling them in source order guarantees it.
- *`_validation`*: `src/nnx/_validation.py`, the standard-library-only module of shared validators (`require_count`, `require_finite_real`, …).

**Proposal.**

1. **Pin the message first.** On `develop`, before any edit, add this test to `tests/test_export_conformance.py`, below `test_the_validator_refuses_contradictions_without_crashing`:
   ```python
   def _every_section_broken() -> dict:
       record = {key: {} for key in conformance._KEYS}
       record.update(
           extra=1,
           format="not-a-format",
           profile="no-such-profile",
           status="passed",
           failure=None,
           level="executed",
           stages={name: {"status": "passed", "failure": "x", "detail": None} for name in conformance.STAGES},
           input_cases=[],
           model_state="kept",
           artifacts={"directory": "d", "model": "m.onnx", "files": [{"path": "a", "role": "r", "bytes": -1, "sha256": "XYZ"}]},
           tolerances={"rtol": -1.0, "atol": 0.0},
           config={"weights_sha256": "not-hex"},
           dtype="float32",
       )
       return record


   def test_validate_record_reports_every_section_in_a_fixed_order():
       """FEAT-061: splitting validate_record keeps every problem and their order."""
       with pytest.raises(ConformanceError) as refused:
           conformance.validate_record(_every_section_broken())
       assert str(refused.value) == _EVERY_SECTION_REFUSAL
   ```
   Then print the current message:
   ```bash
   uv run python - <<'PY'
   import nnx.export_conformance as c
   from tests.test_export_conformance import _every_section_broken
   try:
       c.validate_record(_every_section_broken())
   except c.ConformanceError as error:
       print(repr(str(error)))
   PY
   ```
   and paste the printed string as `_EVERY_SECTION_REFUSAL = <printed string>` at module level in the test file (wrap it with implicit string concatenation if ruff format needs). Commit this test on its own first; it passes on `develop`.
2. In `src/nnx/_validation.py`, add `import re` to the imports and, after `required_id` (L246–L250), add:
   ```python
   _SHA256_HEX = re.compile(r"[0-9a-f]{64}")


   def is_sha256_hex(value: object) -> bool:
       """Whether ``value`` is a SHA-256 digest as NNx writes it: a ``str`` of
       exactly 64 lowercase hex digits."""
       return isinstance(value, str) and _SHA256_HEX.fullmatch(value) is not None
   ```
3. In `src/nnx/export_conformance.py`:
   - add `from ._validation import is_sha256_hex` next to `from ._artifacts import file_sha256` (L70);
   - add a module-level helper directly above `validate_record`:
     ```python
     def _stage_status(record: Mapping[str, Any], name: str) -> Any:
         stage = record["stages"][name]
         return stage.get("status") if isinstance(stage, Mapping) else None
     ```
   - move each section of `validate_record` into its own function with the signature `(record: Mapping[str, Any], problems: list[str]) -> None`, cutting the lines unchanged except as listed below:

     | New function | Lines moved (at `9ad461e`) | Section |
     |---|---|---|
     | `_record_keys_problems` | L1120–L1131 | strict-JSON round trip, unknown and missing keys (the missing-keys raise stays), `format` |
     | `_stage_problems` | L1132–L1167 | the `stages` shape (its raise stays), each stage, `status`, `failure`, `level` |
     | `_case_problems` | L1169–L1235 | `input_cases` and their outcomes |
     | `_model_state_problems` | L1236–L1250 | `model_state` against the export stage |
     | `_artifact_problems` | L1252–L1294 | `artifacts` entries and the checked-model rule |
     | `_config_problems` | L1295–L1345 | `tolerances`, `config.weights_sha256`, `source` / `exporter` / `opset` / `versions` / `runtime`, the profile settings, the session options, the exporter options and `artifacts.model` (L1344–L1345 stay last here, so the order does not change) |

     Inside the moved code: delete the nested `def status(...)` (L1136–L1138) and replace every `status(<name>)` call with `_stage_status(record, <name>)`. `_stage_problems` keeps its first line `stages = record["stages"]` (L1132). Add `stages = record["stages"]` as the first line of `_case_problems` (it reads `stages["parity"]` at L1202 and L1204) and of `_model_state_problems` (L1242), and `artifacts = record["artifacts"]` as the first line of `_config_problems` (it reads `artifacts` at L1344).
   - replace the two inline digest checks with the helper: L1278–L1282 becomes `if not is_sha256_hex(entry["sha256"]):`, and L1303–L1305 becomes `elif not is_sha256_hex(weights_sha):`. The appended messages stay as they are.
   - `validate_record` keeps its signature and docstring and becomes:
     ```python
         if not isinstance(record, Mapping):
             raise ConformanceError(f"a conformance record is a mapping, got {type(record).__name__}")
         problems: list[str] = []
         _record_keys_problems(record, problems)
         _stage_problems(record, problems)
         _case_problems(record, problems)
         _model_state_problems(record, problems)
         _artifact_problems(record, problems)
         _config_problems(record, problems)
         if problems:
             raise ConformanceError("malformed conformance record: " + "; ".join(problems))
     ```
4. In `src/nnx/bundles.py`, add `from ._validation import is_sha256_hex` to the module imports, replace the two lines `or not isinstance(entry["sha256"], str)` / `or not _SHA256.fullmatch(entry["sha256"])` (L555–L556) with the one line `or not is_sha256_hex(entry["sha256"])`, and delete `_SHA256 = re.compile(r"[0-9a-f]{64}")` (L110).
5. In `src/nnx/nn/nn_model.py`, add `from .._validation import is_sha256_hex` to the module imports, change L1388 to `        if not is_sha256_hex(digest) or type(size) is not int or size < 0:`, and delete `_SHA256_HEX = re.compile(r"[0-9a-f]{64}")` (L191). Nothing else references `bundles._SHA256` or `nn_model._SHA256_HEX` (`grep -rn "_SHA256\b\|_SHA256_HEX" src tests` lists only the two definitions and these two uses).
6. Add this test to `tests/test_export_conformance.py`:
   ```python
   def test_is_sha256_hex_accepts_only_64_lowercase_hex_digits():
       from nnx._validation import is_sha256_hex

       assert is_sha256_hex("0" * 64) and is_sha256_hex("0123456789abcdef" * 4)
       for bad in ("0" * 63, "0" * 65, "A" * 64, "g" * 64, "0" * 64 + "\n", b"0" * 64, None, 64):
           assert not is_sha256_hex(bad), bad
   ```

No behaviour changes, so no CHANGELOG entry. `validate_record`'s docstring is unchanged, so `docs/api.md` needs no rebuild.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `test_validate_record_reports_every_section_in_a_fixed_order` passes on `develop` before the split and, unchanged, after it. Check: `uv run pytest tests/test_export_conformance.py::test_validate_record_reports_every_section_in_a_fixed_order`
- [ ] `validate_record` is at most 15 lines of code after its docstring, and the six helpers exist. Check: `uv run python -c "import inspect, nnx.export_conformance as c; assert all(hasattr(c, n) for n in ('_record_keys_problems','_stage_problems','_case_problems','_model_state_problems','_artifact_problems','_config_problems','_stage_status')); print(len(inspect.getsource(c.validate_record).splitlines()))"` prints a number of at most 27
- [ ] `test_is_sha256_hex_accepts_only_64_lowercase_hex_digits` passes. Check: `uv run pytest tests/test_export_conformance.py::test_is_sha256_hex_accepts_only_64_lowercase_hex_digits`
- [ ] No inline hex-digit check or private SHA-256 regex is left. Check: `grep -rn '0123456789abcdef\|\[0-9a-f\]{64}' src/nnx` prints exactly one line, in `src/nnx/_validation.py`
- [ ] The conformance, bundle and Hub-manifest suites pass, and the script still starts. Check: `uv run pytest tests/test_export_conformance.py tests/test_bundle_validation.py tests/test_run_bundles.py tests/test_hub_manifest.py` and `uv run python scripts/check_export_conformance.py --help`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [export_conformance.py L1105-L1347](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/export_conformance.py#L1105-L1347) | `validate_record`, its sections and the nested `status` (L1136–L1138) |
| [export_conformance.py L1278-L1283](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/export_conformance.py#L1278-L1283), [L1300-L1306](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/export_conformance.py#L1300-L1306) | The two inline SHA-256 checks |
| [bundles.py L110](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L110), [L552-L560](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L552-L560) | `_SHA256` and the manifest-entry check |
| [nn_model.py L191](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L191), [L1388](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1388) | `_SHA256_HEX` and its use in `_manifest_files` |
| [_validation.py L246-L250](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_validation.py#L246-L250) | `required_id`, after which `is_sha256_hex` goes |
| #255, #566 | The feature and the pre-write checks |

## Scope

**In:** the split of `validate_record`, `_validation.is_sha256_hex` and its three callers, two tests.

**Out:**
- Hoisting `export_conformance`'s and `bundles`' function-local imports: FEAT-140.
- Moving the module out of the runtime package: DEC-018.
- Any change to what a record may contain.

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-140
**Related:** #255, #566
