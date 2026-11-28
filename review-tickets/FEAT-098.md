---
id: FEAT-098
title: "[FEAT-098] Read run.yaml and idps.csv through one helper each"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-17"]
depends_on: ["FEAT-063", "FEAT-099"]
related_issues: [522]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-098: Four modules read a run's `run.yaml` each with their own `yaml.safe_load` and "is it a mapping" check, `nnx.comparison` re-implements the round-trip `idps.csv` reader, and the three run-file names are string literals in about twenty places. This ticket names the three files once (`_RUN_YAML`, `_METADATA_YAML`, `_IDPS_CSV`), adds `nn_run._read_run_yaml` and `nn_run._read_idps_frame`, and uses them everywhere. Every caller keeps its own error type and wording, and #522's missing-`idps.csv` rule stays.

## Context

**What happens today.** `NNRun._load` (`nn_run.py` L1329–L1346, comment omitted) and `history._parent` (`history.py` L1171–L1175) read `run.yaml` the same way with different messages. First:

```python
        yaml_path = os.path.join(run_path, "run.yaml")

        with open(yaml_path, encoding="utf-8") as f:
```

then, in `_parent`:

```python
    path = os.path.join(run_path, "run.yaml")
    with open(path, encoding="utf-8") as handle:
        state = yaml.safe_load(handle)
    if not isinstance(state, dict):
        raise ValueError(f"malformed run.yaml at {path}: expected a mapping")
```

`comparison._read_run_state` (L1439–L1445) wraps the same read in `ComparisonError`:

```python
    try:
        with open(os.path.join(run_path, "run.yaml"), encoding="utf-8") as handle:
            run_state = yaml.safe_load(handle)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ComparisonError(f"run {run_id}: run.yaml is missing or unreadable: {error}") from error
    if not isinstance(run_state, Mapping):
        raise ComparisonError(f"malformed run.yaml for run {run_id}")
```

and `nn_model._source_hyperparameters` (L2891–L2892) reads it a fourth time inside a broad `except`. `comparison._read_run` (L1494–L1499) repeats `nn_run._read_idps_csv`'s parse:

```python
        csv_path = os.path.join(run_path, "idps.csv")
        try:
            # round_trip: the default parser misreads about one float in five by one ULP.
            frame = pd.read_csv(csv_path, float_precision=cast(Any, "round_trip")) if os.path.isfile(csv_path) else None
            rows = [] if frame is None else frame.to_dict(orient="records")
```

**Why it matters.** The readers already disagree on the error (`ValueError` with or without the type, `ComparisonError`, a silent `{}`), and a change to how `run.yaml` is loaded (a stricter loader, say) would have to be made four times.

**History.** #522 kept two rules this ticket must not change: a run under the history protocol with neither LAST nor `idps.csv` reads as empty in `NNRun.load`, and "A committed or legacy run without `idps.csv` still fails to load, with a `ValueError` that names the file". Its evidence also notes that `comparison` "already treats a missing `idps.csv` as an empty history"; that stays too.

**Background for the worker.**
- Run directory: `runs/<id>/` holding `run.yaml` (`NNRun.state()`), `metadata.yaml` (environment, resume, compile and precision records), `idps.csv` (the training records as CSV) or a `history/` journal (`HISTORY_DIR`).
- `HISTORY_DIR = "history"` is defined in `history.py` (L95) and imported by `nn_run` at module level. The three new constants go beside it, because `nn_run` imports `history` at the top while `history` can import `nn_run` only inside functions.
- `yaml.safe_load`: parses YAML into plain dicts, lists and scalars; an empty file gives `None`. It never builds Python objects.
- `_read_idps_frame` returns `(rows, columns)`: `rows` is `frame.to_dict(orient="records")`, one dict per CSV row mapping a column name to its value (an empty cell is a float NaN); `columns` is the tuple of column names, as `str`, in file order (for example `("Unnamed: 0", "epoch_idx", "train_edp.loss", ...)`). It raises what `pandas.read_csv` raises: `FileNotFoundError`, `pandas.errors.EmptyDataError` for a zero-byte file, `pandas.errors.ParserError`.
- FEAT-063 removes `_read_idps_csv`'s unused `exact=` parameter first; after it, `_read_idps_csv(csv_path)` always parses round-trip.

**Proposal.**
1. In `src/nnx/history.py`, directly below `HISTORY_DIR = "history"` (L95), add:

   ```python
   # The run directory's own files (NNRun.save), named once for every reader.
   _RUN_YAML = "run.yaml"
   _METADATA_YAML = "metadata.yaml"
   _IDPS_CSV = "idps.csv"
   ```

   and replace the literal in each `os.path.join(..., "<name>")` with the constant at L837 (`_IDPS_CSV`), L1142 (`_IDPS_CSV`), L1165 (`_METADATA_YAML`), L1316 (`_RUN_YAML`), L1417 (`_RUN_YAML`) and L1449 (`_IDPS_CSV`). Leave the file names inside message strings as they are.
2. In `src/nnx/nn/params/nn_run.py`:
   - extend the top-level `from ...history import (...)` (L28) with `_IDPS_CSV, _METADATA_YAML, _RUN_YAML`;
   - replace the literals at L447 (`_RUN_YAML`), L456 (`_IDPS_CSV`), L769 (`_METADATA_YAML`), L1190–L1192 (`_IDPS_CSV`, `_RUN_YAML`, `_METADATA_YAML`), L1329 (`_RUN_YAML`) and L1375 (`_METADATA_YAML`);
   - directly above `_read_idps_csv` (L467), add:

     ```python
     class _MalformedRunYaml(ValueError):
         """A ``run.yaml`` that does not hold a mapping (:func:`_read_run_yaml`)."""


     def _read_run_yaml(run_path: str) -> dict[str, Any]:
         """The mapping the run at ``run_path`` records in ``run.yaml``. Read with
         ``yaml.safe_load``, never FullLoader: a tampered file must not build
         Python objects through ``!!python/object`` tags. A missing or unreadable
         file raises ``OSError``, malformed YAML ``yaml.YAMLError``, and anything
         but a mapping (an empty file loads as ``None``) :class:`_MalformedRunYaml`,
         naming the file."""
         yaml_path = os.path.join(run_path, _RUN_YAML)
         with open(yaml_path, encoding="utf-8") as f:
             rep = yaml.safe_load(f)
         if not isinstance(rep, dict):
             raise _MalformedRunYaml(f"malformed run.yaml at {yaml_path}: expected a mapping, got {type(rep).__name__}")
         return rep


     def _read_idps_frame(csv_path: str) -> tuple[list[dict[Any, Any]], tuple[str, ...]]:
         """The rows of an ``idps.csv`` (one dict per row) and its column names,
         floats parsed round-trip: ``to_csv`` writes each float's shortest
         round-trip repr, and pandas' default parser misreads about one in five
         by one ULP. Raises what ``pandas.read_csv`` raises."""
         # "round_trip" is accepted by every supported pandas, though pandas 2.x
         # leaves it out of the parameter's annotation.
         frame = pd.read_csv(csv_path, float_precision=cast(Any, "round_trip"))
         return frame.to_dict(orient="records"), tuple(str(column) for column in frame.columns)
     ```

   - in `_read_idps_csv`, replace the two lines that parse the file (`frame = pd.read_csv(csv_path, float_precision=cast(Any, "round_trip"))` and `raw_idps = frame.to_dict(orient="records")`, inside the first `try`, as FEAT-063 left them) with `raw_idps, _ = _read_idps_frame(csv_path)`, and delete the two-line `"round_trip"` comment above them (it moved into the helper). Both `except` clauses and their messages stay;
   - in `NNRun._load`, keep `yaml_path = os.path.join(run_path, _RUN_YAML)` (it names the file in the `missing key` error at L1397) and replace the `with open(yaml_path, ...)` block and the `if not isinstance(rep, dict): raise ValueError(...)` block (L1331–L1346) with `rep = _read_run_yaml(run_path)`.
3. In `src/nnx/history.py`, `_parent` (L1151): add `_read_run_yaml` to its local `from .nn.params.nn_run import _load_resume_status, _read_metadata`, delete its local `import yaml` (L1160, now unused), and replace L1171–L1175 (from `path = os.path.join(run_path, "run.yaml")` to the `raise ValueError(...)`) with `state = _read_run_yaml(run_path)`. The message for a non-mapping file becomes the one `NNRun.load` gives: `malformed run.yaml at <path>: expected a mapping, got <type>` (old: `... expected a mapping`). No test asserts the old text (`grep -rn "malformed run.yaml" tests` returns nothing).
4. In `src/nnx/comparison.py`:
   - in `_read_run_state` (L1429), extend the local import to `from .nn.params.nn_run import _MalformedRunYaml, _read_run_yaml, _runs_root, _validate_run_id` and replace L1439–L1445 with:

     ```python
         try:
             run_state = _read_run_yaml(run_path)
         except _MalformedRunYaml:
             raise ComparisonError(f"malformed run.yaml for run {run_id}") from None
         except (OSError, ValueError, yaml.YAMLError) as error:
             raise ComparisonError(f"run {run_id}: run.yaml is missing or unreadable: {error}") from error
     ```

     (`_MalformedRunYaml` is caught first; any other `ValueError`, a non-UTF-8 file for example, keeps the "missing or unreadable" wording.)
   - in `_resume_mode`, replace the literal at L1030 with `_METADATA_YAML`, imported locally from `.history` there (`from .history import _METADATA_YAML`);
   - in `_read_run` (L1452), add `_IDPS_CSV` to the local `from .history import _own_records, has_journal`, add `_read_idps_frame` to the local `from .nn.params.nn_run import _HISTORY_PROTOCOL_FILE`, delete the local `import pandas as pd`, and replace L1494–L1497 (from `csv_path = ...` through `rows = [] if frame is None else ...`) with:

     ```python
             csv_path = os.path.join(run_path, _IDPS_CSV)
             try:
                 # A run without idps.csv reads as an empty history (#522).
                 rows, columns = _read_idps_frame(csv_path) if os.path.isfile(csv_path) else ([], ())
     ```

     and in the line `recorded = _recorded_names(idps, () if frame is None else frame.columns)` (L1503) pass `columns` instead: `recorded = _recorded_names(idps, columns)`. The surrounding `except Exception` and its `malformed idps.csv` message stay.
5. In `src/nnx/nn/nn_model.py`, `_source_hyperparameters` (L2880): add `_read_run_yaml` to the top-level `from .params.nn_run import (...)` (L121) and replace L2891–L2892 (the `with open(...)` and `rep = yaml.safe_load(handle)` lines) with `rep = _read_run_yaml(os.path.join(_runs_root(), _validate_run_id(run_id)))`. The `except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError, AttributeError)` stays and catches `_MalformedRunYaml` (a `ValueError`), returning `{}` as a non-mapping file does today.
6. Tests:
   - in `tests/test_run_comparison.py`, add:

     ```python
     def test_a_run_yaml_that_is_not_a_mapping_is_a_malformed_run(tmp_path):
         run_dir = tmp_path / "runs" / ("a" * 32)
         run_dir.mkdir(parents=True)
         (run_dir / "run.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
         with pytest.raises(ComparisonError, match="malformed run.yaml for run a{32}"):
             observations_from_runs(["a" * 32], metric=LOSS, root=str(tmp_path))
     ```

   - in `tests/test_run_load_exact_history.py`, add:

     ```python
     def test_the_idps_frame_reader_returns_rows_and_column_names(tmp_path):
         from nnx.nn.params.nn_run import _read_idps_frame

         path = tmp_path / "idps.csv"
         path.write_text("epoch_idx,train_edp.loss\n0,0.1\n1,0.30000000000000004\n", encoding="utf-8")
         rows, columns = _read_idps_frame(str(path))
         assert columns == ("epoch_idx", "train_edp.loss")
         assert rows == [{"epoch_idx": 0, "train_edp.loss": 0.1}, {"epoch_idx": 1, "train_edp.loss": 0.30000000000000004}]
     ```

   Existing tests that pin the wording and keep passing: `tests/test_train_integration.py` L1662 (`expected a mapping`), `tests/test_run_comparison.py` L1202, L1206 (`malformed idps.csv`), L1210 and L1634 (`run.yaml is missing`), `tests/test_history_journal.py` L438 (`missing idps.csv at`).
7. No behaviour changes beyond `history._parent`'s message gaining the type name, which no caller or test reads, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FEAT-063 and FEAT-099 have merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] Both new tests pass. Check: `uv run pytest tests/test_run_comparison.py::test_a_run_yaml_that_is_not_a_mapping_is_a_malformed_run tests/test_run_load_exact_history.py::test_the_idps_frame_reader_returns_rows_and_column_names`
- [ ] No path literal for the three files remains. Check: `grep -rnE "os\.path\.join\([^)]*\"(run\.yaml|metadata\.yaml|idps\.csv)\"" src/nnx` returns 0 lines
- [ ] `run.yaml` is parsed in one place. Check: `grep -rn "yaml.safe_load(" src/nnx | grep -v metadata` returns 1 line, in `_read_run_yaml` (`nn_run.py`; the `metadata.yaml` readers are separate)
- [ ] `comparison` no longer calls `pd.read_csv`. Check: `grep -n "read_csv" src/nnx/comparison.py` returns 0 lines
- [ ] The run-reading suites pass, with the pinned messages unchanged. Check: `uv run pytest tests/test_run_comparison.py tests/test_history_journal.py tests/test_exact_resume.py tests/test_run_load_exact_history.py tests/test_train_integration.py tests/test_component_resume.py tests/test_best_pointer_readers.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [history.py L95](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L95) | `HISTORY_DIR`: the run-file name constant already shared with `nn_run` |
| [history.py L1151-L1179](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L1151-L1179) | `_parent`: its own `run.yaml` read and message |
| [nn_run.py L450-L491](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L450-L491) | `_read_run_idps` (#522's rule) and `_read_idps_csv` |
| [nn_run.py L1309-L1398](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L1309-L1398) | `NNRun._load`: the `run.yaml` read and `yaml_path` in the `missing key` error |
| [nn_run.py L1186-L1192](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L1186-L1192) | `_save_locked`: the three file names |
| [comparison.py L1429-L1449](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/comparison.py#L1429-L1449) | `_read_run_state`: the `ComparisonError` wrapping |
| [comparison.py L1452-L1507](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/comparison.py#L1452-L1507) | `_read_run`: the copied round-trip parse and the column names |
| [nn_model.py L2880-L2911](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2880-L2911) | `_source_hyperparameters`: the fourth read |
| [test_run_comparison.py L1185-L1210](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_run_comparison.py#L1185-L1210), [L1634](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_run_comparison.py#L1634) | the pinned comparison messages |
| [test_train_integration.py L1650-L1663](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_train_integration.py#L1650-L1663) | the pinned `expected a mapping` |
| #522 | the missing-`idps.csv` rules kept |

## Scope

**In:** the three constants, `_read_run_yaml`, `_MalformedRunYaml`, `_read_idps_frame`, their callers, two tests.

**Out:** the `metadata.yaml` record parsers (FEAT-099). Removing `_read_idps_csv(exact=)` (FEAT-063). Comparison's two "last record per epoch" loops (`_read_run` and `observations_from_runs`), which are left as they are. A stricter YAML loader.

## Dependencies

**Blocked by:** FEAT-063, FEAT-099
**Blocks:** none
**Related:** #522
