---
id: FEAT-072
title: "[FEAT-072] Move the checkpoint, run, history and bundle writers onto nnx._atomic_io (part 2 of 3)"
labels: ["enhancement", "area: persistence", "priority: P2", "review: 2026-11-27"]
sources: ["SMELL-01"]
depends_on: ["FEAT-071"]
related_issues: [552, 443, 260]
status: READY
---

**Category:** Persistence · **Priority:** P2 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-072: after FEAT-071, `nn_checkpoint`, `nn_run`, `history` and `bundles` still write through hand-rolled copies (`_write_manifest`, `_write_chunk_file`) and call the compatibility names. Part 2 rewrites those two writers onto `nnx._atomic_io`, makes every call in these four modules go through the `_atomic_io` module attribute, deletes `history.py`'s six function-local imports of the atomic helpers, and retargets the three tests that patch `nn_run._atomic_write_text`. Bytes, modes and flush points stay identical.

## Context

**What happens today.** `nn_checkpoint._write_manifest` repeats the atomic text writer by hand:

```python
def _write_manifest(checkpoint_path: str, manifest: dict[str, Any], *, staged: bool = False) -> None:
    path = _staged_manifest_path(checkpoint_path) if staged else _manifest_path(checkpoint_path)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=1)
            handle.flush()
            _full_fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
```

`history._write_chunk_file` repeats the bytes version without a flush:

```python
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
```

and `history.py` imports the helpers inside six functions (L327, L345, L561, L586, L830, L1384), because `nn_run` imports `history` at module level.

**Why it matters.** Each copy is one more place a durability fix can miss (BUG-10, BUG-11), and the local imports hide the dependency.

**History.** #552 made history and bundles raise flush errors and keeps `bundles._fsync_directory` as the bundle export's directory-flush seam (`tests/test_bundle_validation.py:595` patches it); it stays a function. #443 added `F_FULLFSYNC` to the resume-point writes, `_write_manifest` included. #260 made the atomic text writer clean its temporary on every failure.

**Background for the worker.**
- *`nnx._atomic_io`*: the standard-library-only module FEAT-071 created: `full_fsync`, `fsync_directory`, `fsync_path`, `fsync_file`, `atomic_write_text(path, content, *, plain_mode=False, full_flush=False)` (`full_flush=True` flushes with `full_fsync`, FIX-115), `atomic_write_bytes(path, data, *, durable=True)`, `reserve_partial`, `commit_partial`, `discard_partial`, `partial_file`, `quietly`.
- *Calling through the module attribute*: write `_atomic_io.atomic_write_text(...)`, never `from .._atomic_io import atomic_write_text`; a test that patches `nnx._atomic_io.atomic_write_text` then intercepts this call.
- *Resume-point manifest*: `<checkpoint>.manifest.json` (and `.manifest.staged.json`), written 0600 and flushed before the checkpoint is published; its text is `json.dumps(manifest, sort_keys=True, indent=1)`.
- *History chunk*: a write-once file of journal records, flushed later together with the epoch's other files.

**Proposal.**

1. `src/nnx/nn/params/nn_checkpoint.py`:
   - replace the body of `_write_manifest` (L361–L371) with:
     ```python
         path = _staged_manifest_path(checkpoint_path) if staged else _manifest_path(checkpoint_path)
         _atomic_io.atomic_write_text(path, json.dumps(manifest, sort_keys=True, indent=1), full_flush=True)
     ```
     Same text, same 0600 mode (`plain_mode` stays `False`), same flush (`full_flush=True` keeps today's `_full_fsync`, which #443 requires for resume points; the default would be plain `os.fsync`), same temporary-name prefix; the function keeps its name because `tests/test_exact_resume.py` L395 patches it.
   - change the remaining direct calls to the module attribute: L315 and L678 `_full_fsync(handle.fileno())` → `_atomic_io.full_fsync(handle.fileno())`; L423 and L759 `_fsync_directory(...)` → `_atomic_io.fsync_directory(...)`.
   - delete `import tempfile` only if ruff reports it unused (the `save()` reservations and `_atomic_torch_save` still use it).
2. `src/nnx/nn/params/nn_run.py`: change every call of a compatibility name to the module attribute: `_quietly(` → `_atomic_io.quietly(` (L353, L355, L1149); `_atomic_write_text(` → `_atomic_io.atomic_write_text(` (L540, L1142, L1214, L1229, L1234); `_fsync_directory(` → `_atomic_io.fsync_directory(` (L1146). Remove `_fsync_directory` from the `from ..params.nn_checkpoint import (...)` list (nothing imports it from `nn_run`). The forwarding functions FEAT-071 left stay.
3. `src/nnx/history.py` (it imports `_atomic_io` at module level since FEAT-071):
   - delete the local import lines L327 (`from .nn.params.nn_run import _atomic_write_text, _quietly`), L345 (`from .nn.params.nn_checkpoint import _fsync_directory`), L561 and L586 (`from .nn.params.nn_run import _atomic_write_text`) and L1384 (`from .nn.params.nn_run import _atomic_write_text`); change L830 to `            from .nn.params.nn_run import _idps_csv_text`;
   - change the calls: L333, L565, L589, L1387 `_atomic_write_text(` → `_atomic_io.atomic_write_text(`; L336, L838 `_quietly(` → `_atomic_io.quietly(`; L347 `_fsync_directory(` → `_atomic_io.fsync_directory(`;
   - replace the body of `_write_chunk_file` (L367–L377) with `    _atomic_io.atomic_write_bytes(path, data, durable=False)` and its docstring with `"""A chunk is written once, through an owned temporary file renamed into place (removed if anything fails). Not fsynced here — see :func:`_append_bytes`."""`. The function keeps its name because `tests/test_history_journal.py` L209 and L288 patch it;
   - keep `_fsync_path` (FEAT-071's forwarding function) and every call of it unchanged: `tests/test_history_journal.py` L914 and L1114 patch it;
   - delete `import tempfile` (L72), now unused.
4. `src/nnx/bundles.py`: add `from . import _atomic_io` to the module imports; in `_write` delete the local `from .nn.params.nn_checkpoint import _full_fsync` (L1138) and change L1151 to `        _atomic_io.full_fsync(handle.fileno())`; replace the body of `_fsync_directory` (L1156–L1160) with
   ```python
       """:func:`nnx._atomic_io.fsync_directory`, kept as this module's seam
       (tests intercept the export's directory flushes here)."""
       _atomic_io.fsync_directory(path)
   ```
   `_publish` stays as it is: its temporary is created like the payloads (plain mode), which `atomic_write_bytes` (0600) would change.
5. Retarget the tests that patch `nn_run._atomic_write_text`, capture and patch together, and give each replacement function a `**kwargs` it forwards (`plain_mode=` and `full_flush=` reach `atomic_write_text` from some callers):

   | Test (line at `9ad461e`) | Today | New target |
   |---|---|---|
   | `tests/test_history_journal.py` L190, L208 (`_history_writer_bytes`) | `nn_run_module._atomic_write_text` | `nnx._atomic_io.atomic_write_text`; `def counted_atomic(path, content, **kwargs)` returns `atomic(path, content, **kwargs)` |
   | `tests/test_history_journal.py` L292, L299 (`_crash_at_manifest`) | `nn_run_module._atomic_write_text` | `nnx._atomic_io.atomic_write_text`; `def fail(path, content, **kwargs)` returns `original(path, content, **kwargs)` |
   | `tests/test_train_end_checkpoint.py` L362, L372 | `nn_run._atomic_write_text` | `nnx._atomic_io.atomic_write_text`; `def flaky_write(path, content, **kwargs)` returns `real_write(path, content, **kwargs)` |

   Import the module as `import nnx._atomic_io as atomic_io` in those files. Every writer still reaches `atomic_write_text` (directly, or through `nn_run`'s forwarding `_atomic_write_text`), so these patches see the same writes as before. These keep working unchanged: `history._write_chunk_file`, `history._append_bytes`, `history._fsync_path`, `bundles._fsync_directory`, `nn_checkpoint._write_manifest`, `nn_checkpoint._atomic_torch_save` (`tests/test_train_integration.py` L520).
6. Add this test to `tests/test_atomic_io.py`:
   ```python
   def test_the_core_writers_call_atomic_io_through_its_module(tmp_path, monkeypatch):
       """FEAT-072: nn_checkpoint, nn_run, history and bundles write through
       nnx._atomic_io, and history imports no atomic helper in a function."""
       import inspect

       import nnx.bundles as bundles

       source = inspect.getsource(history)
       assert "import _atomic_write_text" not in source and "import _quietly" not in source
       assert "import _fsync_directory" not in source and "import _full_fsync" not in source
       assert "import _full_fsync" not in inspect.getsource(bundles)
       written = []
       real = atomic_io.atomic_write_text
       monkeypatch.setattr(atomic_io, "atomic_write_text", lambda path, text, **kw: (written.append(path), real(path, text, **kw))[1])
       nn_checkpoint._write_manifest(str(tmp_path / "LAST.pt"), {"format": "nnx.resume-point/1"})
       assert written == [str(tmp_path / "LAST.pt.manifest.json")]
       assert (tmp_path / "LAST.pt.manifest.json").read_text(encoding="utf-8") == '{\n "format": "nnx.resume-point/1"\n}'
   ```

No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FEAT-071 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_atomic_io.py::test_the_core_writers_call_atomic_io_through_its_module` fails before the change and passes after it. Check: `uv run pytest tests/test_atomic_io.py`
- [ ] `history.py` has no function-local import of an atomic helper. Check: `grep -n "import .*\(_atomic_write_text\|_quietly\|_fsync_directory\|_full_fsync\)" src/nnx/history.py` prints nothing
- [ ] The four modules call no compatibility name. Check: `grep -nE "(^|[^._a-z])(_full_fsync|_fsync_directory|_atomic_write_text|_quietly)\(" src/nnx/nn/params/nn_checkpoint.py src/nnx/nn/params/nn_run.py src/nnx/history.py src/nnx/bundles.py` lists only the `def` lines of the forwarding functions and of `bundles._fsync_directory`, and the calls of `bundles._fsync_directory` inside `bundles.py`
- [ ] Resume-point manifests are byte-identical and still 0600. Check: `uv run pytest tests/test_resume_point_hashing.py tests/test_resume_point_durability.py tests/test_exact_resume.py`
- [ ] #552's flush coverage and the history, run and bundle suites pass. Check: `uv run pytest tests/test_atomic_write_text.py tests/test_run_file_durability.py tests/test_history_journal.py tests/test_run_bundles.py tests/test_bundle_validation.py tests/test_checkpoint_safetensors.py tests/test_train_end_checkpoint.py tests/test_train_integration.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_checkpoint.py L361-L371](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L361-L371) | `_write_manifest`, a hand-rolled atomic text write |
| [nn_checkpoint.py L315](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L315), [L423](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L423), [L678](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L678), [L759](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L759) | Direct flush calls |
| [nn_run.py L353](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L353), [L540](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L540), [L1142-L1149](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L1142-L1149), [L1214-L1234](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L1214-L1234) | `nn_run`'s writer calls |
| [history.py L327](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L327), [L345](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L345), [L561](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L561), [L586](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L586), [L830](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L830), [L1384](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L1384) | The six local imports |
| [history.py L363-L377](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L363-L377) | `_write_chunk_file` |
| [bundles.py L1136-L1160](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L1136-L1160), [L1386-L1403](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L1386-L1403) | `_write`, the `_fsync_directory` seam and `_publish` |
| [test_history_journal.py L183-L211](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_history_journal.py#L183-L211), [L291-L299](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_history_journal.py#L291-L299), [test_train_end_checkpoint.py L362-L372](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_train_end_checkpoint.py#L362-L372) | The patches of `nn_run._atomic_write_text` |
| [test_bundle_validation.py L595](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_bundle_validation.py#L595) | The `bundles._fsync_directory` seam (#552) |
| #552, #443, #260 | Flush errors raised; `F_FULLFSYNC`; temporary cleanup |

## Scope

**In:** `_write_manifest`, `_write_chunk_file`, the call sites in the four modules, `history.py`'s local atomic imports, three test retargets, one test.

**Out:**
- `nn_checkpoint._to_safetensors_file`: it writes 0600 today and `partial_file` uses the plain mode, so moving it waits for DEC-021.
- `nn_checkpoint._atomic_torch_save` and the name reservations in `NNCheckpoint.save()`: hashed writes whose temporary names belong to the resume-point publish sequence. `_atomic_torch_save` does not merge with #859's `_artifacts.atomic_torch_save` (FEAT-071 step 2 says why); FEAT-073 moves that one onto `_atomic_io.partial_file`.
- `bundles._publish` and `bundles._write` (plain-mode files created like the payloads).
- `history.py`'s other local imports of `nn_run` (run-layout helpers, a real cycle).
- The remaining writers in other modules: FEAT-073.

## Dependencies

**Blocked by:** FEAT-071
**Blocks:** FEAT-073
**Related:** #552, #443, #260
