---
id: FEAT-073
title: "[FEAT-073] Move the remaining atomic writers onto nnx._atomic_io and drop their local imports (part 3 of 3)"
labels: ["enhancement", "area: persistence", "priority: P2", "review: 2026-11-27"]
sources: ["SMELL-01"]
depends_on: ["FEAT-072"]
related_issues: [552, 260, 522]
status: READY
---

**Category:** Persistence · **Priority:** P2 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-073: after FEAT-072, seven modules outside the run core still import the atomic writer from `nnx.nn.params.nn_run` (most of them inside a function, because `nn_run` is heavy and cyclic), and four writers spell the partial-file protocol out by hand. Part 3 points all of them at `nnx._atomic_io`, imported at module top, and replaces the hand-written `reserve / commit / discard` blocks with `with _atomic_io.partial_file(path) as partial:`. Files, modes and flushes stay identical.

## Context

**What happens today.** Writers reach the atomic writer through `nn_run`, inside functions:

```python
def _write_json(path: str, value: Mapping[str, Any]) -> None:
    from .nn.params.nn_run import _atomic_write_text

    # Atomic: an interrupted write leaves the previous file intact.
    _atomic_write_text(path, canonical_bytes(value).decode("utf-8") + "\n")
```

```python
def atomic_write(path: Union[str, os.PathLike[str]], text: str) -> None:
    from .nn.params.nn_run import _atomic_write_text

    _atomic_write_text(os.fspath(path), text)
```

and `export_state_dict` writes the partial-file protocol by hand:

```python
        partial = _reserve_partial(path)
        try:
            torch.save(self.net.state_dict(), partial)
            _commit_partial(partial, path)
        except BaseException:
            _discard_partial(partial)
            raise
```

`_artifacts.py`'s docstring admits the inverted dependency: "the atomic writer is imported from ``nnx.nn.params.nn_run`` when saving".

**Why it matters.** Function-local imports of a heavy module for a stdlib helper, and four copies of the commit-or-discard dance.

**History.** #552 made every durable write raise its flush errors; #260 made the atomic text writer clean up its temporary; #522 kept `NNRun.all()` robust to interrupted saves. FIX-116 gave the plain-mode writers (`config.json`, the Hub manifest, the tokenizer, the Modelfile) `plain_mode=True`; this ticket keeps every call's mode exactly as FIX-116 left it.

**Background for the worker.**
- *`nnx._atomic_io`*: the standard-library-only module from FEAT-071; `atomic_write_text(path, content, *, plain_mode=False)` accepts `str` or `os.PathLike`; `partial_file(path)` yields a reserved partial path, commits it onto `path` when the block ends normally and discards it when the block raises.
- *Plain mode*: the permission bits a plain `open(path, "w")` gives under the current umask (FIX-116); the default (`plain_mode=False`) is 0600.
- *Staging directory*: `NNModel.save_pretrained` writes the whole Hub save into a fresh hidden directory (`tempfile.mkdtemp`) and moves the files into the target only after it succeeds, so `_save_pretrained` writes into an empty directory.

**Proposal.** In every file below, import the module at module top as `from <dots> import _atomic_io` (relative depth: `.` in `src/nnx/*.py`, `..` in `src/nnx/nn/*.py` and `src/nnx/interop/*.py`, `...` in `src/nnx/nn/params/*.py` and `src/nnx/interop/gguf/*.py`), and call it through the module attribute.

1. `src/nnx/_artifacts.py`:
   - change the module docstring's first sentence ending (L2–L3) from `atomic files, all derived from ``state()``. Internal; standard library only` + newline + `(the atomic writer is imported from ``nnx.nn.params.nn_run`` when saving).` to `atomic files, all derived from ``state()``. Internal; standard library only` + newline + `(atomic files go through ``nnx._atomic_io``).`;
   - replace `atomic_write` (L64–L67) with a forwarding function (it stays importable for `comparison`, `decisions/benchmark` and `paradigms/offline_distillation`):
     ```python
     def atomic_write(path: Union[str, os.PathLike[str]], text: str) -> None:
         """:func:`nnx._atomic_io.atomic_write_text`: 0600, flushed, renamed into place."""
         _atomic_io.atomic_write_text(path, text)
     ```
2. `src/nnx/provenance.py` `_write_json` (L563–L567): delete the local import and call `_atomic_io.atomic_write_text(path, canonical_bytes(value).decode("utf-8") + "\n")`. Keep the function (`tests/test_compile_execution.py` patches `provenance._write_json`). Leave `provenance.py`'s other local `nn_run` imports (run-layout helpers) alone.
3. `src/nnx/data_splits.py` `SplitManifest.save` (L321–L325) and `src/nnx/preprocessing.py` `Standardizer.save` (L467–L471): delete the local import and call `_atomic_io.atomic_write_text(path, self.to_json())`.
4. `src/nnx/nn/generative_nn_model.py`: delete the `from .params.nn_run import ...` line (L33) and change the call in `_hub_reconstruction_config` (L68–L72) to `_atomic_io.atomic_write_text(` with the same arguments and `plain_mode=True`.
5. `src/nnx/interop/ollama.py`: delete the local import (L171, with its "lazy: the run module is heavy" comment) and change the call (L173–L177) to `_atomic_io.atomic_write_text(` with the same arguments and `plain_mode=True`.
6. `src/nnx/nn/nn_model.py`:
   - `_write_hub_manifest` (L1414) and `_save_pretrained`'s config write (L4777): `_atomic_write_text(` → `_atomic_io.atomic_write_text(`, keeping `plain_mode=True`;
   - `_save_pretrained`'s weights (L4752–L4762): replace the `partial_path` block with
     ```python
             weights_path = save_dir / _HUB_MODEL_FILENAME
             with _atomic_io.partial_file(weights_path) as partial:
                 save_file(tensors, partial)
     ```
     keeping the two comment lines above it. In the fresh staging directory both forms give the file the plain-open mode, flush it and rename it; only the hidden partial name changes (`.model.safetensors.<random>.partial` instead of `model.safetensors.partial`), and it never survives a save.
   - `export_state_dict` (L5019–L5024): replace the block with
     ```python
             with _atomic_io.partial_file(path) as partial:
                 torch.save(self.net.state_dict(), partial)
     ```
     keeping the two comment lines above it.
   - in the `from .params.nn_run import (...)` list (L121–L135), remove `_atomic_write_text` and `_fsync_file`. Keep `_reserve_partial`, `_commit_partial` and `_discard_partial`: `to_onnx` (L4344–L4406) still uses them and is not touched here.
7. `src/nnx/interop/gguf/writer.py` `write_gguf`: delete the local import (L258) and replace `partial = _reserve_partial(out_path)` / `try:` … `_commit_partial(partial, out_path)` / `except BaseException:` / `_discard_partial(partial)` / `raise` (L260–L326) with `with _atomic_io.partial_file(out_path) as partial:` around the same body (the `GGUFWriter(partial, ...)` construction and its inner `try/finally: writer.close()`), dedenting nothing inside the inner `try`. The commit happens after `writer.close()`, as today.
8. `src/nnx/nn/params/nn_tokenizer_params.py` `NNTokenizerParams.of` (the static method that saves the tokenizer to `path`, L56–L84): replace FIX-117's `reserve / commit / discard` block with
   ```python
           with _atomic_io.partial_file(path) as partial:
               cast(Any, tokenizer).save(partial)
   ```
9. Add this test to `tests/test_atomic_io.py`:
   ```python
   def test_no_module_imports_the_atomic_writer_from_nn_run_any_more():
       """FEAT-073: every writer imports nnx._atomic_io at module top; only
       to_onnx (FIX-108 owns it) still uses nn_run's partial-file names."""
       import pathlib

       import nnx

       root = pathlib.Path(nnx.__file__).parent
       hits = [
           f"{path.relative_to(root)}:{number}"
           for path in root.rglob("*.py")
           for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
           if "nn_run import" in line and ("_atomic_write_text" in line or "_CREATED_FILE_MODE" in line or "_fsync_file" in line)
       ]
       assert hits == []
   ```

No test patches any of the edited call sites by an old name (`grep -rn "setattr(.*\(_reserve_partial\|_commit_partial\|_discard_partial\|_fsync_file\|atomic_write\)" tests` prints nothing at `9ad461e`); the patches of `safetensors.torch.save_file`, `torch.save` and `os.replace` in `tests/test_hub_export_atomic.py` and `tests/test_hub_mixin.py` patch library attributes and keep working. No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FEAT-072 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_atomic_io.py::test_no_module_imports_the_atomic_writer_from_nn_run_any_more` fails before the change and passes after it. Check: `uv run pytest tests/test_atomic_io.py`
- [ ] No module outside `nn_run.py` and `_atomic_io.py` imports an atomic helper inside a function. Check: `grep -rn "^\s\+from .*nn_run import .*\(_atomic_write_text\|_quietly\|_reserve_partial\|_commit_partial\|_discard_partial\)" src/nnx` prints nothing
- [ ] The hand-written partial-file blocks are gone except in `to_onnx`. Check: `grep -rn "_reserve_partial(" src/nnx` lists only `nn_run.py` (the forwarding function) and `nn_model.py` inside `to_onnx`
- [ ] Hub, export, GGUF, tokenizer, provenance, split and preprocessing files are written as before. Check: `uv run pytest tests/test_hub_export_atomic.py tests/test_hub_mixin.py tests/test_hub_manifest.py tests/test_atomic_write_text.py tests/test_interop_gguf_writer.py tests/test_tokenizer_params.py tests/test_provenance.py tests/test_attempt_provenance.py tests/test_data_splits.py tests/test_preprocessing.py tests/test_generative_nn_model.py tests/test_finetune_loading.py tests/test_model_recipes.py tests/test_compile_execution.py`
- [ ] The plain-mode writers keep FIX-116's mode tests green. Check: `uv run pytest tests/test_atomic_write_text.py -k mode`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [_artifacts.py L1-L3](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_artifacts.py#L1-L3), [L64-L67](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_artifacts.py#L64-L67) | The inverted dependency and `atomic_write` |
| [provenance.py L563-L567](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/provenance.py#L563-L567) | `_write_json` |
| [data_splits.py L321-L325](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/data_splits.py#L321-L325), [preprocessing.py L467-L471](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/preprocessing.py#L467-L471) | `save` methods with local imports |
| [generative_nn_model.py L33](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/generative_nn_model.py#L33), [L68-L72](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/generative_nn_model.py#L68-L72) | The tokenizer write |
| [ollama.py L171-L177](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/interop/ollama.py#L171-L177) | The Modelfile write and its "heavy module" local import |
| [nn_model.py L121-L135](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L121-L135), [L1414](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1414), [L4752-L4762](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4752-L4762), [L4777](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4777), [L5019-L5024](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5019-L5024) | `nn_model`'s writers |
| [nn_model.py L4614-L4636](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4614-L4636) | `save_pretrained` stages into a fresh directory |
| [writer.py L258-L326](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/interop/gguf/writer.py#L258-L326) | `write_gguf`'s partial-file block |
| [nn_tokenizer_params.py L77-L83](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_tokenizer_params.py#L77-L83) | `NNTokenizerParams.of`'s save, which FIX-117 rewrites |
| #552, #260, #522 | Durable writes, temporary cleanup, robust run listing |

## Scope

**In:** the writers in `_artifacts`, `provenance`, `data_splits`, `preprocessing`, `generative_nn_model`, `interop/ollama`, `nn_model` (Hub save, Hub manifest, `export_state_dict`), the GGUF writer and `NNTokenizerParams.of`; one test.

**Out:**
- `to_onnx`: FIX-108 rewrites how it stages its output, and FEAT-107 builds its export arguments in one place.
- `nn_checkpoint._to_safetensors_file` (0600 today): waits for DEC-021.
- Which files get the plain-open mode: DEC-021.
- Other function-local imports in these modules: FEAT-095 (`nn_model`), FEAT-137, FEAT-140.

## Dependencies

**Blocked by:** FEAT-072
**Blocks:** none
**Related:** #552, #260, #522, FIX-108, FEAT-101 (also edits `_save_pretrained`; whichever lands second rebases)
