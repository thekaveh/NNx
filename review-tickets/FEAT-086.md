---
id: FEAT-086
title: "[FEAT-086] Share one file-digest helper"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-08"]
depends_on: []
related_issues: [520]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-086: three functions hash a file's bytes with the same 1 MiB read loop: `_artifacts.file_sha256`, `nn_model._file_digest` (which also returns the size, with its own `_HUB_DIGEST_CHUNK = 1 << 20`) and `provenance.hash_file`. Add one `_artifacts.file_digest(path, chunk_size=1 << 20) -> tuple[str, int]` and build the three on it. Every digest stays byte-identical, so nothing persisted changes.

## Context

**What happens today.**

```python
def file_sha256(path: str) -> str:
    """SHA-256 of a file's bytes, read in 1 MiB blocks. Export and checkpoint
    verification both use it, so one digest rule covers every artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
```

```python
def _file_digest(path: str) -> tuple[str, int]:
    """The SHA-256 hex digest and the byte size of the file at ``path``."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HUB_DIGEST_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size
```

`provenance.hash_file` (L210–L225) checks `chunk_size` with `require_count` and then runs the same loop.

**Why it matters.** `file_sha256`'s own docstring says "one digest rule covers every artifact", yet there are three copies of the rule.

**History.** #520 (FIX-053) made `hash_file` refuse an invalid `chunk_size` (a zero chunk hashed nothing) before the file is opened; that check stays in `hash_file`. The weight digests are BUG-27 (FIX-126) and are not touched here.

**Background for the worker.**
- *Digest*: SHA-256 of the file's bytes as lowercase hex; the block size changes only the memory used, never the digest.
- *Hub manifest*: `nnx_hub_manifest.json`, which lists each file of a `save_pretrained` artifact with its digest and size (`_write_hub_manifest`, `_verify_hub_manifest` use `_file_digest`).

**Proposal.**

1. In `src/nnx/_artifacts.py`, add before `file_sha256` (L170 at `9ad461e`, L171 at `1402b49`):
   ```python
   def file_digest(path: Union[str, os.PathLike[str]], chunk_size: int = 1 << 20) -> tuple[str, int]:
       """The SHA-256 hex digest and the byte size of the file at ``path``, read
       in ``chunk_size`` blocks (1 MiB by default; the block size changes the
       memory used, never the digest). ``chunk_size`` must be ``>= 1``: a caller
       that takes it from users checks it first (``nnx.provenance.hash_file``)."""
       digest = hashlib.sha256()
       size = 0
       with open(path, "rb") as handle:
           for block in iter(lambda: handle.read(chunk_size), b""):
               digest.update(block)
               size += len(block)
       return digest.hexdigest(), size
   ```
   and replace `file_sha256`'s body (after its docstring) with `    return file_digest(path)[0]`.
2. In `src/nnx/nn/nn_model.py`: change `from .._artifacts import parse_json` to `from .._artifacts import file_digest, parse_json`; replace `_file_digest` (L1359–L1367) with
   ```python
   # FEAT-086: the shared digest; the old name stays for the Hub manifest code.
   _file_digest = file_digest
   ```
   and delete `_HUB_DIGEST_CHUNK = 1 << 20` (L192), which nothing else reads.
3. In `src/nnx/provenance.py`: change `from ._artifacts import check_keys` to `from ._artifacts import check_keys, file_digest`, and replace the last five lines of `hash_file` (from `digest = hashlib.sha256()` through the `return`) with:
   ```python
       hexdigest, _ = file_digest(path, chunk_size)
       return IdentityRef("digest", f"sha256:{hexdigest}")
   ```
   The `require_count` check before it stays. Delete `import hashlib` from `provenance.py` only if ruff reports it unused.
4. Add to `tests/test_provenance.py`:
   ```python
   def test_one_file_digest_serves_hash_file_file_sha256_and_the_hub_manifest(tmp_path):
       """FEAT-086: one digest rule; the chunk size never changes it."""
       import hashlib

       import nnx.nn.nn_model as nn_model
       from nnx._artifacts import file_digest, file_sha256

       path = tmp_path / "data.bin"
       path.write_bytes(b"abcdefgh" * 1000)
       expected = (hashlib.sha256(path.read_bytes()).hexdigest(), 8000)
       assert file_digest(path) == file_digest(path, 3) == expected
       assert file_sha256(str(path)) == expected[0] and nn_model._file_digest is file_digest
       assert hash_file(path, chunk_size=7) == hash_bytes(path.read_bytes())
       empty = tmp_path / "empty"
       empty.write_bytes(b"")
       assert file_digest(empty) == (hashlib.sha256(b"").hexdigest(), 0)
   ```
   (`hash_file` and `hash_bytes` are already imported at the top of `tests/test_provenance.py`.)

No behaviour changes, so no CHANGELOG entry; `hash_file`'s docstring is unchanged, so `docs/api.md` needs no rebuild.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_provenance.py::test_one_file_digest_serves_hash_file_file_sha256_and_the_hub_manifest` passes (it fails on `develop`, where `file_digest` does not exist). Check: `uv run pytest tests/test_provenance.py::test_one_file_digest_serves_hash_file_file_sha256_and_the_hub_manifest`
- [ ] One read loop is left. Check: `grep -rn "iter(lambda: handle.read(" src/nnx/_artifacts.py src/nnx/nn/nn_model.py src/nnx/provenance.py` prints exactly one line, in `_artifacts.py`
- [ ] `_HUB_DIGEST_CHUNK` is gone. Check: `grep -rn "_HUB_DIGEST_CHUNK" src tests` prints nothing
- [ ] Hub manifests, provenance identities, resume-point hashes and conformance records are unchanged. Check: `uv run pytest tests -k "hub_manifest or provenance or resume_point_hashing or export_conformance"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [_artifacts.py L170-L177](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_artifacts.py#L170-L177) | `file_sha256` |
| [nn_model.py L192](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L192), [L1359-L1367](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1359-L1367) | `_HUB_DIGEST_CHUNK` and `_file_digest` |
| [nn_model.py L1411](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1411), [L1446](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1446) | The Hub manifest's uses of `_file_digest` |
| [provenance.py L210-L225](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/provenance.py#L210-L225) | `hash_file` and its `require_count` check (#520) |
| #520 | `chunk_size` refused before the file is opened |

## Scope

**In:** `file_digest` and the three helpers built on it; one test.

**Out:** the weight digests (FIX-126); moving the Hub code (FEAT-076, which keeps the `_file_digest` name).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-076
**Related:** #520
