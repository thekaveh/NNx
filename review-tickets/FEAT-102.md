---
id: FEAT-102
title: "[FEAT-102] Name the tokenizer file once and annotate the Hub stub NoReturn"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-21"]
depends_on: ["FIX-110"]
related_issues: [260]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-102: `GenerativeNNModel` writes its tokenizer under a hard-coded `"tokenizer.json"`, while the Hub manifest writer and the stale-file cleanup in `nn_model.py` use the constant `_HUB_TOKENIZER_FILENAME`; renaming one would break the other silently. The no-hub stub's `_hub_unavailable` is annotated `-> NNModel` although it always raises. This ticket uses the constant in `generative_nn_model.py` and annotates the stub `-> NoReturn`. Behaviour does not change.

## Context

**What happens today.** `GenerativeNNModel._hub_reconstruction_config` (`generative_nn_model.py` L61–L74):

```python
    def _hub_reconstruction_config(self, save_directory) -> dict[str, Any]:
        if self.tokenizer is None:
            return {}
        filename = "tokenizer.json"
```

`nn_model.py` defines the name once (L187, `_HUB_TOKENIZER_FILENAME = "tokenizer.json"`) and relies on it in `_write_hub_manifest` (L1409, the files the manifest lists) and in `save_pretrained`'s removal of a stale tokenizer (L4625–L4627). The stub installed without the `hub` extra (L164–L165):

```python
        def _hub_unavailable(self) -> NNModel:
            raise ImportError("HuggingFace Hub integration requires the `hub` extra: `pip install thekaveh-nnx[hub]`.")
```

**Why it matters.** If the generative model's file name and the constant drift, the manifest stops listing the tokenizer (so a load refuses the artifact), and the cleanup removes or keeps the wrong file. A wrong return annotation tells the type checker the stub returns a model.

**History.** #260 made the atomic text writer clean its temporaries; the tokenizer is written through it (`_atomic_write_text`, L68). The manifest and the stale-tokenizer cleanup came with ed93d51 (#855); FIX-110 reorders that cleanup, still through `_HUB_TOKENIZER_FILENAME`.

**Background for the worker.**
- `_HUB_TOKENIZER_FILENAME`: the file name NNx gives the tokenizer inside a `save_pretrained` directory.
- `NoReturn` (`typing`): the annotation of a function that never returns normally (it always raises).
- `generative_nn_model.py` already imports `NNModel` from `.nn_model` (L30), so importing the constant from there adds no cycle.

**Proposal.**
1. In `src/nnx/nn/generative_nn_model.py`, change `from .nn_model import NNModel` (L30) to `from .nn_model import _HUB_TOKENIZER_FILENAME, NNModel` (the order ruff's isort gives), and in `_hub_reconstruction_config` change `filename = "tokenizer.json"` (L64) to `filename = _HUB_TOKENIZER_FILENAME`.
2. In `src/nnx/nn/nn_model.py`, add `NoReturn` to `from typing import ...` (L22) and change `def _hub_unavailable(self) -> NNModel:` (L164) to `def _hub_unavailable(self) -> NoReturn:`.
3. Run `uv run ruff check --fix` and `uv run ruff format` on both files. No message or behaviour changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FIX-110 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The file name is written once in `src/nnx/nn`. Check: `grep -rn '"tokenizer.json"' src/nnx/nn` returns 1 line (`nn_model.py`, the constant)
- [ ] The stub is annotated `NoReturn`. Check: `grep -n "def _hub_unavailable(self) -> NoReturn:" src/nnx/nn/nn_model.py` returns 1 line
- [ ] The generative and manifest suites pass. Check: `uv run pytest tests/test_generative_nn_model.py tests/test_hub_manifest.py tests/test_hub_mixin.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [generative_nn_model.py L61-L74](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/generative_nn_model.py#L61-L74) | the hard-coded `"tokenizer.json"` |
| [nn_model.py L183-L190](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L183-L190) | `_HUB_TOKENIZER_FILENAME` and the other Hub file names |
| [nn_model.py L1409](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1409) | the manifest lists the tokenizer by the constant |
| [nn_model.py L4622-L4627](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4622-L4627) | the stale-tokenizer removal by the constant |
| [nn_model.py L156-L178](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L156-L178) | the no-hub stub and `_hub_unavailable` |
| #260 | the atomic text writer the tokenizer goes through |
| ed93d51 (#855) | the Hub manifest and the stale-tokenizer cleanup |

## Scope

**In:** the constant in `generative_nn_model.py`, the `NoReturn` annotation.

**Out:** where and when a stale tokenizer is removed (FIX-110). Writing the tokenizer through a unique partial file (FIX-117). The file modes of Hub files (DEC-021).

## Dependencies

**Blocked by:** FIX-110
**Blocks:** FEAT-076
**Related:** #260
