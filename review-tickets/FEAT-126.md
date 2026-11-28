---
id: FEAT-126
title: "[FEAT-126] Share JSONL record read and write and the key checks"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-49"]
depends_on: []
related_issues: [521, 558, 243, 244]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-126: the decision benchmark (`write_records` / `read_records`) and offline distillation (`write_teacher_records` / `read_teacher_records`) carry the same JSONL writer and reader, and both `from_state` methods hand-roll their unknown / missing key checks although `_artifacts.check_keys` exists (the benchmark one needs four `# type: ignore[arg-type]` because of it). This ticket adds `_artifacts.write_jsonl` and `_artifacts.read_jsonl`, routes all four functions through them, and switches both `from_state` methods to `check_keys`. The file format does not change; the key-check messages do (a CHANGELOG line).

## Context

**What happens today.** `src/nnx/decisions/benchmark.py` (the `offline_distillation.py` pair is the same with `TeacherRecord` / `TeacherRecordError` / `"teacher records"`):

```python
def write_records(path: Union[str, os.PathLike[str]], records: Iterable[Record]) -> None:
    """One JSON object per line (strict JSON, sorted keys)."""
    from .._artifacts import atomic_write

    lines = [json.dumps(record.state(), sort_keys=True, allow_nan=False) for record in records]
    atomic_write(path, "".join(line + "\n" for line in lines))  # an interrupted write never leaves half a file
```

```python
    for number, line in enumerate(read_text(path, "records", BenchmarkError).split("\n"), start=1):
        if not line.strip():
            continue
        try:
            records.append(Record.from_state(parse_json(line, "records", BenchmarkError)))
        except BenchmarkError as error:
            raise BenchmarkError(f"{os.fspath(path)}:{number}: {error}") from error
```

and `Record.from_state`:

```python
        unknown = sorted(set(state) - known)
        if unknown:
            raise BenchmarkError(f"a record has unknown keys {unknown}")
        distribution = state.get("distribution")
        return Record(
            sample_id=state.get("sample_id"),  # type: ignore[arg-type]
```

**Why it matters.** Two copies of a file format's reader and writer, and a third and fourth key checker beside the shared one, which other artifacts (#521) already use.

**History.** #521 (FIX-054) made splits, standardizers and link splits refuse unknown or missing keys through `check_keys`. #558 froze the benchmark records. #243 defined `nnx.decision-record/1` (refuses unknown keys) and #244 `nnx.teacher-record/1`. The error behaviour stays equivalent: both formats still refuse every unknown key and name the file line; what changes is the wording, and a benchmark record that lacks a core field is now refused by name.

**Background for the worker.**
- JSONL: one JSON object per line. Both writers sort keys, refuse NaN / ±inf (`allow_nan=False`) and write atomically (`_artifacts.atomic_write`: a temporary file renamed onto the destination).
- `parse_json(text, what, error)` (`src/nnx/_artifacts.py` L80): strict JSON; `read_text(path, what, error)` (L70): UTF-8 text or `error`.
- `check_keys(state, *, required, optional, what, error)` (L117): raises `error("the <what> has unknown keys [...] (expected [...])")`, `error("the <what> lacks [...]")`, or both joined with `and`.
- `_FIELDS` (`offline_distillation.py` L110): the nine required teacher-record fields; `label` is optional.

**Proposal.**
1. In `src/nnx/_artifacts.py`, add `Callable` to the `collections.abc` import and `TypeVar` to the `typing` import, and after `check_keys` add:

   ```python
   _T = TypeVar("_T")


   def write_jsonl(path: Union[str, os.PathLike[str]], states: Iterable[Mapping[str, Any]]) -> None:
       """Write one strict-JSON object per line (sorted keys, no NaN / ±inf),
       atomically: every line is serialized first, and an interrupted write
       never leaves half a file."""
       lines = [json.dumps(state, sort_keys=True, allow_nan=False) for state in states]
       atomic_write(path, "".join(line + "\n" for line in lines))


   def read_jsonl(
       path: Union[str, os.PathLike[str]], what: str, error: type[Exception], from_state: Callable[[Any], _T]
   ) -> list[_T]:
       """The records of a UTF-8 JSONL file written by :func:`write_jsonl`:
       each non-blank line is parsed as strict JSON (:func:`parse_json`) and
       rebuilt by ``from_state``. A line that fails raises ``error`` prefixed
       with ``<path>:<line>: ``."""
       records: list[_T] = []
       for number, line in enumerate(read_text(path, what, error).split("\n"), start=1):
           if not line.strip():
               continue
           try:
               records.append(from_state(parse_json(line, what, error)))
           except error as exc:
               raise error(f"{os.fspath(path)}:{number}: {exc}") from exc
       return records
   ```

2. `src/nnx/decisions/benchmark.py`:
   - Add `from .._artifacts import check_keys, read_jsonl, write_jsonl` to the module-top imports.
   - `write_records` (L550-L555): replace the body after the docstring with `write_jsonl(path, (record.state() for record in records))  # an interrupted write never leaves half a file`.
   - `read_records` (L558-L571): replace the body after the docstring (the local import, the loop and the return) with `return read_jsonl(path, "records", BenchmarkError, Record.from_state)`.
   - `Record.from_state` (L526-L547): replace the `known` / `unknown` lines (L530-L534) with:

     ```python
             check_keys(
                 state,
                 required=("format", "sample_id", "question_digest", "provider", "status"),
                 optional=("distribution", "p_true", "reason", "revision", "prompt_identity", "execution"),
                 what="decision record",
                 error=BenchmarkError,
             )
     ```

     `optional=` holds every other key of the `known` set as it stands when you start (at `9ad461e` the six above; a field added by an earlier ticket goes there too). Then read the four required fields as `state["sample_id"]`, `state["question_digest"]`, `state["provider"]`, `state["status"]` and delete their four `# type: ignore[arg-type]` comments. The other arguments stay as they are.
3. `src/nnx/paradigms/offline_distillation.py`:
   - Add `from .._artifacts import check_keys, read_jsonl, write_jsonl` to the module-top imports.
   - `write_teacher_records` (L258-L264): body becomes `write_jsonl(path, (record.state() for record in records))` (the local `json` and `atomic_write` imports go).
   - `read_teacher_records` (L267-L282): body becomes `return read_jsonl(path, "teacher records", TeacherRecordError, TeacherRecord.from_state)`.
   - `TeacherRecord.from_state` (L245-L255): replace the `unknown` and `missing` blocks (L249-L254) with `check_keys(state, required=("format", *_FIELDS), optional=("label",), what="teacher record", error=TeacherRecordError)`.
4. Message changes (no test asserts the old full text):

   | Old | New |
   |---|---|
   | `a record has unknown keys ['x']` | `the decision record has unknown keys ['x'] (expected ['distribution', 'execution', 'format', …])` |
   | a missing `sample_id` / `question_digest` / `provider` / `status`: `Record`'s field error | `the decision record lacks ['sample_id']` |
   | `a teacher record has unknown keys ['logits']` | `the teacher record has unknown keys ['logits'] (expected [...])` |
   | `a teacher record lacks ['candidates']` | `the teacher record lacks ['candidates']` |

   Tests that match parts of these texts and keep passing: `tests/test_decision_benchmark.py` L929 (`"unknown keys"`), `tests/test_offline_distillation.py` L161 (the missing field's name), L171 (`"unknown keys"`) and L176 (`teacher\.jsonl:1`). `git grep -n "a record has unknown\|a teacher record" tests docs` returns 0 lines.
5. Add to `CHANGELOG.md` under `## [Unreleased]` → `### Changed`:

   ```
   - **Benchmark and teacher record files name every unknown and missing key in one message.** `Record.from_state` / `read_records` and `TeacherRecord.from_state` / `read_teacher_records` now refuse through the shared key check (`the decision record has unknown keys [...] (expected [...])`, `the teacher record lacks [...]`), and a benchmark record without `sample_id`, `question_digest`, `provider` or `status` is refused by that name. The file formats are unchanged.
   ```

6. Add to `tests/test_decision_benchmark.py`, after `test_malformed_records_and_numbers_raise_benchmark_errors`:

   ```python
   def test_a_record_missing_a_core_field_is_refused_by_name():
       state = {"format": "nnx.decision-record/1", "question_digest": "d", "provider": "p", "status": "answered"}
       with pytest.raises(BenchmarkError, match=r"the decision record lacks \['sample_id'\]"):
           Record.from_state(state)
   ```

7. Run `uv run ruff check --fix src/nnx/` and `uv run ruff format src/nnx/`.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `test_a_record_missing_a_core_field_is_refused_by_name` fails on `develop` before the change. Check: `uv run pytest tests/test_decision_benchmark.py::test_a_record_missing_a_core_field_is_refused_by_name`
- [ ] Both formats read and write through the shared helpers. Check: `grep -n "write_jsonl(\|read_jsonl(" src/nnx/decisions/benchmark.py src/nnx/paradigms/offline_distillation.py` returns 4 lines
- [ ] The benchmark's `type: ignore[arg-type]` comments in `Record.from_state` are gone. Check: `grep -c "type: ignore\[arg-type\]" src/nnx/decisions/benchmark.py` returns 4 fewer than on `develop`
- [ ] The record round-trip, pickling, line-number and key tests pass. Check: `uv run pytest tests/test_decision_benchmark.py tests/test_offline_distillation.py tests/test_decision_model_pilot.py`
- [ ] The CHANGELOG entry is under `[Unreleased]` → `### Changed`. Check: `grep -n "name every unknown and missing key in one message" CHANGELOG.md`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [benchmark.py L526-L547](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L526-L547) | `Record.from_state`: the hand-rolled unknown-key check and four `type: ignore`s |
| [benchmark.py L550-L571](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L550-L571) | `write_records` / `read_records` |
| [offline_distillation.py L245-L282](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L245-L282) | `TeacherRecord.from_state` and the teacher JSONL pair |
| [_artifacts.py L64-L132](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_artifacts.py#L64-L132) | `atomic_write`, `read_text`, `parse_json`, `check_keys` |
| [test_decision_benchmark.py L929](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_decision_benchmark.py#L929) | matches "unknown keys" |
| [test_offline_distillation.py L158-L177](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_offline_distillation.py#L158-L177) | the teacher-record key and line-number tests |
| #521, #558, #243, #244 | `check_keys` adoption; frozen records; the two record formats |

## Scope

**In:** `write_jsonl`, `read_jsonl`, the four record functions and both `from_state` key checks.

**Out:** the report JSON (`BenchmarkReport.save` / `load_state`; its local imports are FEAT-136). Persisting `p_false` in records (DEC-016). Splitting `benchmark.py` (FEAT-121).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-121, FEAT-136, FEAT-137
**Related:** #521, #558, #243, #244
