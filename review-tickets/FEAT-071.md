---
id: FEAT-071
title: "[FEAT-071] Add a shared atomic-I/O module (part 1 of 3)"
labels: ["enhancement", "area: persistence", "priority: P2", "review: 2026-11-27"]
sources: ["SMELL-01"]
depends_on: ["FIX-114", "FIX-115", "FIX-116", "FIX-117", "FIX-150"]
related_issues: [552, 443, 260]
status: READY
---

**Category:** Persistence · **Priority:** P2 · **Size:** M · **Evidence:** read at `9ad461e` (develop); `_artifacts.py` and `tests/test_atomic_torch_saves.py` re-read at `1402b49` (#859); not run: torch was unavailable in the review environment

## Summary

FEAT-071: NNx's flush primitive and atomic-write helpers live in two heavy modules (`nn/params/nn_checkpoint.py`, `nn/params/nn_run.py`) plus a copy in `history.py`, so every other writer imports them function-locally and copies drift (BUG-10 and BUG-11 came from that). Part 1 creates the standard-library-only module `src/nnx/_atomic_io.py`, moves the primitives into it unchanged, adds two helpers the later parts use (`atomic_write_bytes`, `partial_file`), leaves forwarding functions under every old name, and makes `nnx._atomic_io.full_fsync` the one place a test patches to see every flush. No caller changes in this part. It also records why `_artifacts.atomic_torch_save` (#859) stays in `_artifacts.py` and does not merge with `nn_checkpoint._atomic_torch_save`.

## Context

**What happens today.** The flush primitive is defined in `nn_checkpoint.py` and reached three ways: by import (`nn_run.py` L32–L38), by a function-local import at call time (`history._fsync_path`, `bundles._write`), and through its own module (`nn_checkpoint` L315, L368, L678):

```python
def _full_fsync(fd: int) -> None:
    """Flush ``fd`` to stable storage (FIX-030): ``fcntl(fd, F_FULLFSYNC)``
    where the platform has it — on macOS ``fsync`` only reaches the drive's
    cache, which a power loss can drop — else ``os.fsync``. A filesystem that
    does not support ``F_FULLFSYNC`` falls back to ``os.fsync``, and one that
    supports neither does not fail the save; a real I/O error (``EIO``,
    ``ENOSPC``, ...) is raised, so a save never reports success over it."""
```

`history.py` keeps its own file-or-directory flush that imports it at call time:

```python
def _fsync_path(path: str) -> None:
    """Flush a written file (or a directory entry) to stable storage as a
    resume point's are (#552): a filesystem that cannot flush is tolerated,
    a real I/O error (``EIO``, ...) raises before the epoch is published."""
    from .nn.params.nn_checkpoint import _full_fsync
```

**Why it matters.** Because the helpers sit in `nn_run` (which imports `history` at top level), `history.py`, `provenance.py`, `_artifacts.py`, `data_splits.py`, `preprocessing.py` and `interop/ollama.py` all import them inside functions, and each writer grew its own variant.

**History.** #552 (FIX-084) made bundles and history raise flush I/O errors and routed `bundles._write`, `bundles._fsync_directory` and `history._fsync_path` through `_full_fsync`; its rule "delete any flush and a test fails" must keep holding. #443 (FIX-030) introduced `F_FULLFSYNC`. #260 (FIX-025) made `_atomic_write_text` clean its temporary on every pre-rename failure. FIX-114 (writable descriptor in `_fsync_file`), FIX-115 (`_atomic_write_text(..., full_flush=True)`: `_full_fsync` for standalone artifacts), FIX-116 (`_create_exclusive`, `plain_mode`, mode kept on overwrite; `_CREATED_FILE_MODE` deleted) and FIX-117 (tokenizer through a partial file) fix the bugs in place first, and FIX-150 moves #859's `_artifacts.atomic_torch_save` onto the partial-file protocol and deletes its `_created_file_mode` copy; this ticket moves their result.

**Background for the worker.**
- *Flush primitive*: `_full_fsync(fd)`, F_FULLFSYNC on macOS, `os.fsync` elsewhere; an unsupported flush (`_UNSUPPORTED_FLUSH` errnos) is tolerated, a real I/O error raises.
- *Partial file*: a uniquely named, hidden file beside the destination (`.<name>.<random>.partial`) that a writer fills, then `_commit_partial` flushes and renames onto the destination; `_discard_partial` removes it after a failure.
- *Forwarding function*: a compatibility `def` under the old name whose body calls the new function through the module attribute (`_atomic_io.full_fsync(fd)`). Unlike a plain alias, it looks the target up at call time, so `monkeypatch.setattr(nnx._atomic_io, "full_fsync", ...)` intercepts callers that still use the old name, and a patch on the old name still intercepts that module's own callers.
- *Test seam*: a module attribute tests replace with `monkeypatch.setattr` to observe or fail a step.

**Proposal.**

0. Before starting, confirm FIX-114, FIX-115, FIX-116, FIX-117 and FIX-150 are merged into `develop` and that `grep -rn "_CREATED_FILE_MODE\|_created_file_mode" src/nnx` prints nothing (FIX-116 deletes `nn_run`'s copy, FIX-150 the copy #859 added to `_artifacts.py`). If it prints lines, stop and report on this issue.
1. Create `src/nnx/_atomic_io.py`. It imports only the standard library (`contextlib`, `errno`, `io`, `os`, `secrets`, `stat`, `tempfile`, `collections.abc`, `typing`, and `fcntl` behind its Windows fallback); it imports nothing from `nnx`, torch or numpy. Module docstring:
   ```python
   """Atomic, durable file writes (FEAT-071). Standard library only.

   The one home of NNx's flush primitive (:func:`full_fsync`: ``F_FULLFSYNC``
   where the platform has it, else ``os.fsync``), its directory and path
   variants, the atomic text and bytes writers, and the partial-file protocol
   (:func:`reserve_partial` / :func:`commit_partial` / :func:`discard_partial`,
   or the :func:`partial_file` context manager). Tests patch
   ``nnx._atomic_io.full_fsync`` to see every flush. Internal; not public API.
   """
   ```
2. Move these definitions into it, as they are on `develop` when you start (line numbers are at `9ad461e`; FIX-114 to FIX-117 change some bodies). Drop the leading underscore in the new name; keep bodies and messages unchanged except that references to other moved names use the new names (`_full_fsync` → `full_fsync`, `_quietly` → `quietly`, `_UNSUPPORTED_FLUSH` → `UNSUPPORTED_FLUSH`, `_fsync_file` → `fsync_file`, `_create_exclusive` → `create_exclusive`):

   | From | Lines at `9ad461e` | New name in `nnx._atomic_io` |
   |---|---|---|
   | `nn_checkpoint.py`: `try: import fcntl` / `except ImportError: fcntl = None` | L24–L27 | module-level `fcntl` (same code) |
   | `nn_checkpoint.py`: `_UNSUPPORTED_FLUSH` and its comment | L204–L210 | `UNSUPPORTED_FLUSH` |
   | `nn_checkpoint.py`: `_full_fsync` | L213–L232 | `full_fsync` |
   | `nn_checkpoint.py`: `_fsync_directory` | L235–L245 | `fsync_directory` |
   | `nn_run.py`: `_atomic_write_text` | L210–L276 | `atomic_write_text`; annotate `path: Union[str, os.PathLike[str]]` and make `path = os.fspath(path)` its first statement |
   | `nn_run.py`: `_create_exclusive` (added by FIX-116) | — | `create_exclusive` |
   | `nn_run.py`: `_reserve_partial` | L290–L297 | `reserve_partial` |
   | `nn_run.py`: `_fsync_file` | L300–L306 | `fsync_file` |
   | `nn_run.py`: `_commit_partial` | L309–L316 | `commit_partial` |
   | `nn_run.py`: `_discard_partial` | L319–L322 | `discard_partial` |
   | `nn_run.py`: `_quietly` | L325–L331 | `quietly` |
   | `history.py`: `_fsync_path` | L299–L316 | `fsync_path` (without its local `from .nn.params.nn_checkpoint import _full_fsync`) |

   `fsync_file` (strict: a missing file raises) and `fsync_path` (tolerant: a missing file or an unopenable directory is skipped) stay two functions.

   Not moved, and not merged (#859 added a second atomic `torch.save`):
   - `_artifacts.atomic_torch_save` (`src/nnx/_artifacts.py` L191–L208 at `1402b49`) stays in `_artifacts.py`: it imports torch, which `_atomic_io` must not. After FIX-150 its body is `_reserve_partial`, `torch.save`, `_commit_partial`, and `_discard_partial` on failure, so it already shares the moved create, mode, flush and rename steps through the forwarding functions; FEAT-073 step 1 rewrites it onto `_atomic_io.partial_file`.
   - It does not merge with `nn_checkpoint._atomic_torch_save` (L294–L320 at `9ad461e`), which stays as it is (FEAT-072, Scope: Out). That one hashes every byte while it writes (`_HashingWriter`) and returns the SHA-256 the resume-point manifest records, writes run-internal files with `mkstemp`'s 0600, skips the flush when `durable=False`, and creates temporary names that belong to the resume-point publish sequence. `atomic_torch_save` writes user exports (no digest, the plain mode, always flushed). One function would carry four options, each for one caller.
   - `_artifacts._created_file_mode` / `_CREATED_FILE_MODE` (L181–L188 at `1402b49`) is not moved: FIX-150 deletes it, and step 0 checks it is gone.
3. Add two new helpers at the end of `_atomic_io.py`:
   ```python
   def atomic_write_bytes(path: Union[str, os.PathLike[str]], data: bytes, *, durable: bool = True) -> None:
       """Write ``data`` to ``path`` atomically: an owned temporary file beside
       it (mode 0600, as ``tempfile.mkstemp`` creates it), flushed with
       :func:`full_fsync` when ``durable``, then renamed onto ``path``. The
       temporary is removed if anything fails before the rename."""
       path = os.fspath(path)
       fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(os.path.abspath(path)))
       try:
           with os.fdopen(fd, "wb") as handle:
               handle.write(data)
               if durable:
                   handle.flush()
                   full_fsync(handle.fileno())
           os.replace(temporary, path)
       except BaseException:
           quietly(os.remove, temporary)
           raise


   @contextlib.contextmanager
   def partial_file(path: Union[str, os.PathLike[str]]) -> Iterator[str]:
       """A reserved partial file beside ``path`` for the caller to write
       (:func:`reserve_partial`). It is committed onto ``path`` when the block
       ends normally (:func:`commit_partial`: flushed, then renamed) and
       discarded when the block raises (:func:`discard_partial`)."""
       path = os.fspath(path)
       partial = reserve_partial(path)
       try:
           yield partial
           commit_partial(partial, path)
       except BaseException:
           discard_partial(partial)
           raise
   ```
4. Leave forwarding functions under every old name. Each keeps the exact parameter list of the function it replaces.
   - `src/nnx/nn/params/nn_checkpoint.py`: add `from ... import _atomic_io` to the imports; delete the `fcntl` fallback (L24–L27) and `import errno` (no longer used); replace L204–L245 with:
     ```python
     # Moved to nnx._atomic_io (FEAT-071). The old names stay importable and
     # forward through the module attribute, so patching nnx._atomic_io.full_fsync
     # intercepts every caller.
     _UNSUPPORTED_FLUSH = _atomic_io.UNSUPPORTED_FLUSH


     def _full_fsync(fd: int) -> None:
         _atomic_io.full_fsync(fd)


     def _fsync_directory(path: str) -> None:
         _atomic_io.fsync_directory(path)
     ```
   - `src/nnx/nn/params/nn_run.py`: add `from ... import _atomic_io`; remove `_UNSUPPORTED_FLUSH` and `_full_fsync` from the `from ..params.nn_checkpoint import (...)` list (L32–L38; keep `NNCheckpoint`, `NNCheckpointTransform`, `_fsync_directory`); replace the moved definitions (L210–L331 at `9ad461e`) with forwarding functions, under the same comment as above:
     ```python
     def _atomic_write_text(path: str, content: str, *, plain_mode: bool = False, full_flush: bool = False) -> None:
         _atomic_io.atomic_write_text(path, content, plain_mode=plain_mode, full_flush=full_flush)


     def _create_exclusive(directory: str, prefix: str, suffix: str, mode: int) -> tuple[int, str]:
         return _atomic_io.create_exclusive(directory, prefix, suffix, mode)


     def _reserve_partial(path: str) -> str:
         return _atomic_io.reserve_partial(path)


     def _fsync_file(path: str) -> None:
         _atomic_io.fsync_file(path)


     def _commit_partial(partial: str, path: str) -> None:
         _atomic_io.commit_partial(partial, path)


     def _discard_partial(partial: str) -> None:
         _atomic_io.discard_partial(partial)


     def _quietly(fn: Callable[..., Any], *args: Any) -> None:
         _atomic_io.quietly(fn, *args)
     ```
     Then delete the imports ruff reports unused in `nn_run.py` (at `9ad461e` these become `contextlib`, `io` and `tempfile`).
   - `src/nnx/history.py`: add `from . import _atomic_io` to the module imports and replace `_fsync_path` (L299–L316) with:
     ```python
     def _fsync_path(path: str) -> None:
         """:func:`nnx._atomic_io.fsync_path`, kept as this module's seam (tests patch it)."""
         _atomic_io.fsync_path(path)
     ```
   No caller in `src/` changes in this part.
5. Retarget the tests that patch a moved name. Run `grep -rn '_full_fsync\|"fcntl"\|\.tempfile, ' tests` first: it must list exactly the lines below plus any FIX-114 to FIX-117 and FIX-150 added; retarget every hit the same way. Where a test captures the real function before patching (`real = …`), retarget the capture too, or the patched function would call itself through the forwarding wrapper.

   | Test (line at `9ad461e`) | Today | New target |
   |---|---|---|
   | `tests/test_checkpoint_safetensors.py` L766–L767, L779 | `nn_checkpoint._full_fsync` (capture and patch) | `nnx._atomic_io.full_fsync` |
   | `tests/test_history_journal.py` L1718, L1729 | `nn_checkpoint_module._full_fsync` (capture and patch) | `nnx._atomic_io.full_fsync` |
   | `tests/test_history_journal.py` L1623 | `nn_checkpoint_module.fcntl` | `nnx._atomic_io.fcntl` |
   | `tests/test_run_file_durability.py` L98, L112, L231, L245 | `nn_checkpoint._full_fsync` (captures and patches) | `nnx._atomic_io.full_fsync` |
   | `tests/test_run_bundles.py` L708, L715 | `nn_checkpoint._full_fsync` (capture and patch) | `nnx._atomic_io.full_fsync` |
   | `tests/test_run_bundles.py` L605 | `nn_checkpoint.fcntl` | `nnx._atomic_io.fcntl` |
   | `tests/test_hub_mixin.py` L851 | `nn_run._full_fsync` (`raising=False`) | `nnx._atomic_io.full_fsync` (drop `raising=False`) |
   | `tests/test_resume_point_durability.py` L44, L70 | `nn_checkpoint.fcntl` | `nnx._atomic_io.fcntl` |
   | `tests/test_atomic_write_text.py` L90 | `nn_run.tempfile.mkstemp` | `nnx._atomic_io.tempfile.mkstemp` |
   | `tests/test_atomic_torch_saves.py` `_fake_full_fsync` (added by FIX-150) | `nn_checkpoint.fcntl` | `nnx._atomic_io.fcntl`; then delete that file's `import nnx.nn.params.nn_checkpoint as nn_checkpoint`, which ruff reports unused |

   Import the module in each file as `import nnx._atomic_io as atomic_io` and use `atomic_io` in place of the old module object for these lines only. These keep working unchanged: patches of `os.*` through `nn_run.os` / `nn_checkpoint.os` (the global `os` module), `history._fsync_path` (`tests/test_history_journal.py` L914, L1114: history's own callers still look it up), `nn_run._atomic_write_text` (L208, L299 there; `tests/test_train_end_checkpoint.py` L372), `history._write_chunk_file`, `bundles._fsync_directory`, `nn_checkpoint._write_manifest` and `nn_checkpoint._atomic_torch_save`. The calls to `nn_checkpoint._full_fsync(...)` in `tests/test_resume_point_durability.py` stay: they now also prove the forwarding wrapper.
6. Add `tests/test_atomic_io.py`:
   ```python
   """FEAT-071: nnx._atomic_io, the standard-library-only home of NNx's atomic writes."""

   from __future__ import annotations

   import ast
   import os
   import sys

   import pytest

   import nnx._atomic_io as atomic_io
   import nnx.history as history
   import nnx.nn.params.nn_checkpoint as nn_checkpoint
   import nnx.nn.params.nn_run as nn_run


   def test_atomic_io_imports_only_the_standard_library():
       tree = ast.parse(open(atomic_io.__file__, encoding="utf-8").read())
       roots = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
       roots |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module and node.level == 0}
       assert not any(isinstance(node, ast.ImportFrom) and node.level for node in ast.walk(tree))
       assert roots - {"__future__"} <= set(sys.stdlib_module_names), roots


   def test_every_old_name_reaches_the_one_flush_seam(tmp_path, monkeypatch):
       flushed: list[int] = []
       real = atomic_io.full_fsync
       monkeypatch.setattr(atomic_io, "full_fsync", lambda fd: (flushed.append(fd), real(fd))[1])
       target = tmp_path / "f.txt"
       nn_run._atomic_write_text(str(target), "x", full_flush=True)  # a standalone artifact: full_fsync (FIX-115)
       nn_run._fsync_file(str(target))
       history._fsync_path(str(target))
       nn_checkpoint._fsync_directory(str(tmp_path))
       with open(target, "rb") as handle:
           nn_checkpoint._full_fsync(handle.fileno())
       assert len(flushed) == (5 if os.name != "nt" else 4)  # Windows cannot open a directory to flush it


   def test_atomic_write_bytes_replaces_or_keeps_the_previous_file(tmp_path, monkeypatch):
       target = tmp_path / "chunk.bin"
       atomic_io.atomic_write_bytes(target, b"one", durable=False)
       assert target.read_bytes() == b"one" and os.listdir(tmp_path) == ["chunk.bin"]
       if os.name != "nt":
           assert target.stat().st_mode & 0o777 == 0o600

       def refuse(*args):
           raise OSError("rename refused")

       monkeypatch.setattr(atomic_io.os, "replace", refuse)
       with pytest.raises(OSError, match="rename refused"):
           atomic_io.atomic_write_bytes(target, b"two")
       assert target.read_bytes() == b"one" and os.listdir(tmp_path) == ["chunk.bin"]


   def test_partial_file_commits_on_success_and_discards_on_error(tmp_path):
       target = tmp_path / "model.bin"
       with atomic_io.partial_file(target) as partial:
           assert os.path.dirname(partial) == str(tmp_path) and partial.endswith(".partial")
           with open(partial, "wb") as handle:
               handle.write(b"new")
       assert target.read_bytes() == b"new" and os.listdir(tmp_path) == ["model.bin"]
       with pytest.raises(RuntimeError, match="writer failed"):
           with atomic_io.partial_file(target) as partial:
               with open(partial, "wb") as handle:
                   handle.write(b"half")
               raise RuntimeError("writer failed")
       assert target.read_bytes() == b"new" and os.listdir(tmp_path) == ["model.bin"]
   ```

No behaviour changes, so no CHANGELOG entry. None of the moved names appears in `docs/api.md`.

**Worker setup.** Branch from `develop` after FIX-114, FIX-115, FIX-116, FIX-117 and FIX-150 have merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `src/nnx/_atomic_io.py` exists, imports only the standard library, and holds every name in step 2 plus `atomic_write_bytes` and `partial_file`. Check: `uv run pytest tests/test_atomic_io.py`
- [ ] Every old name still imports and forwards to the seam. Check: `uv run pytest tests/test_atomic_io.py::test_every_old_name_reaches_the_one_flush_seam`
- [ ] The flush primitive is defined once. Check: `grep -rn "def _full_fsync\|def full_fsync\|F_FULLFSYNC" src/nnx` lists `def full_fsync` and its `F_FULLFSYNC` lines in `src/nnx/_atomic_io.py` and only the forwarding `def _full_fsync` in `nn_checkpoint.py`
- [ ] No test patches the flush through `nn_checkpoint` or `nn_run` any more. Check: `grep -rn 'setattr(nn_checkpoint[a-z_]*, "_full_fsync"\|setattr(nn_run[a-z_]*, "_full_fsync"\|setattr(nn_checkpoint[a-z_]*, "fcntl"' tests` prints nothing
- [ ] #552's coverage holds: the durability, bundle and history suites pass. Check: `uv run pytest tests/test_atomic_write_text.py tests/test_run_file_durability.py tests/test_resume_point_durability.py tests/test_resume_point_hashing.py tests/test_history_journal.py tests/test_hub_export_atomic.py tests/test_hub_mixin.py tests/test_hub_manifest.py tests/test_to_onnx_dropped_state.py tests/test_to_onnx_inputs.py tests/test_interop_gguf_writer.py tests/test_checkpoint_safetensors.py tests/test_tokenizer_params.py tests/test_atomic_torch_saves.py tests/test_run_bundles.py tests/test_bundle_validation.py tests/test_provenance.py tests/test_exact_resume.py tests/test_train_end_checkpoint.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_checkpoint.py L24-L27](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L24-L27), [L204-L245](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L204-L245) | `fcntl` fallback, `_UNSUPPORTED_FLUSH`, `_full_fsync`, `_fsync_directory` |
| [nn_run.py L32-L38](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L32-L38), [L210-L331](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L210-L331) | The re-imported flush names; `_atomic_write_text` through `_quietly` |
| [history.py L299-L316](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L299-L316) | `_fsync_path` and its call-time import |
| [_artifacts.py L181-L208](https://github.com/thekaveh/NNx/blob/1402b49/src/nnx/_artifacts.py#L181-L208) | #859's `_created_file_mode` copy and `atomic_torch_save` (FIX-150 rewrites both) |
| [nn_checkpoint.py L294-L320](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L294-L320) | `_atomic_torch_save`: hashed as written, 0600, `durable=` |
| [bundles.py L1136-L1160](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/bundles.py#L1136-L1160) | `_write` and the `_fsync_directory` seam, both reaching `nn_checkpoint` at call time |
| [test_run_bundles.py L703-L727](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_run_bundles.py#L703-L727) | #552's "every bundle file and directory is flushed" test |
| [test_resume_point_durability.py L25-L83](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_resume_point_durability.py#L25-L83) | #443's `F_FULLFSYNC` tests that patch `nn_checkpoint.fcntl` |
| #552, #443, #260 | Flush errors raised; `F_FULLFSYNC`; temporary cleanup |

## Scope

**In:** the new module, the moves, the two new helpers, the forwarding functions, the test retargets, `tests/test_atomic_io.py`.

**Out:**
- Rewriting the checkpoint, run, history and bundle writers onto the module: FEAT-072.
- Rewriting the remaining writers and deleting their local imports: FEAT-073.
- Merging `fsync_file` and `fsync_path` (they differ on a missing file).
- Moving `_artifacts.atomic_torch_save` onto `_atomic_io.partial_file`: FEAT-073. Merging it with `nn_checkpoint._atomic_torch_save`: not done (step 2 says why).
- Which writers use the plain-open mode: DEC-021.

## Dependencies

**Blocked by:** FIX-114, FIX-115, FIX-116, FIX-117, FIX-150
**Blocks:** FEAT-072
**Related:** #552, #443, #260
