---
id: FEAT-076
title: "[FEAT-076] Move the Hub I/O helpers into nnx/nn/_hub.py (part 3 of 5)"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-02"]
depends_on: ["FIX-110", "FIX-112", "FIX-113", "FEAT-092", "FEAT-101", "FEAT-102", "FEAT-074", "FEAT-075", "FEAT-061", "FEAT-073", "FEAT-086"]
related_issues: [509, 441, 260, 767]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** L · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-076: the Hugging Face Hub save and load code (the `_HUB_*` constants, the safetensors shape preflight, the Hub manifest, and the 300-line bodies of `save_pretrained`, `_save_pretrained` and `_from_pretrained`) sits inside `nn/nn_model.py`. Move it, unchanged, into a new `src/nnx/nn/_hub.py`; the three `NNModel` methods keep their signatures and docstrings and become one-line delegations; `nn_model` re-exports every moved module-level name. Part 3 of 5 of the `nn_model.py` split; no behaviour changes.

## Context

**What happens today.** The constants open the file (L183–L193):

```python
_HUB_MODEL_FILENAME = "model.safetensors"
_HUB_CONFIG_FILENAME = "config.json"
```

the helpers sit at L1261–L1452 (`_SAFETENSORS_MAX_HEADER_BYTES` through `_verify_hub_manifest`), and the method bodies at L4592–L4979. `save_pretrained` calls the mixin through `super()`:

```python
        staging = tempfile.mkdtemp(prefix=".nnx-hub-", dir=str(target.parent))
        try:
            result = super().save_pretrained(staging, *args, **kwargs)
```

**Why it matters.** Hub I/O is a separate concern with its own optional dependency; inside `nn_model.py` it adds about 500 lines to every read of the training code.

**History.** #509 added the refusal of unreplayable Hub saves and let `exclude_submodules` drop attached tensors, #441 rebuilt JEPA-trained encoders through the Hub without their predictor (the attached-submodule explanation), #260 the atomic config write, #767 (DEC-013) the refusal of an unreplayable transform before writing. FIX-110, FIX-112, FIX-113, FEAT-061, FEAT-073, FEAT-086, FEAT-092, FEAT-101 and FEAT-102 all edit this code and land first. FEAT-074 and FEAT-075 must land first too: the Hub code calls `_attached_submodules_error`, `_excluded_names`, `_without_submodules` (moved to `nnx.nn._resume`) and `_replay_transforms`, `_refuse_unrecorded_recipe_state` (moved to `nnx.transforms`); importing them from there is what lets `_hub.py` avoid importing `nn_model`.

**Background for the worker.**
- *`PyTorchModelHubMixin`*: `huggingface_hub`'s mixin `NNModel` inherits (as `_HubMixinBase`, a no-op stub when the `hub` extra is missing). Its `save_pretrained` calls `self._save_pretrained(dir)`, and its `from_pretrained` calls `cls._from_pretrained(...)`; so both methods must stay on `NNModel` with their signatures.
- *Zero-argument `super()`*: works only inside a class body. The delegation therefore passes the bound base method (`super().save_pretrained`) to the moved function as an argument.
- *Re-export*: `from ._hub import _HUB_MODEL_FILENAME as _HUB_MODEL_FILENAME` in `nn_model.py`; the redundant alias keeps the name importable (`generative_nn_model.py` and `tests/test_hub_mixin.py` import Hub names from `nn_model`) and visible to call-time lookups.
- *Cycle rule*: `nn_model` imports `_hub` at module top, so `_hub` imports `nn_model` only under `TYPE_CHECKING` (for the `NNModel` annotation).

**Proposal.**

1. Create `src/nnx/nn/_hub.py` with the docstring `"""Hugging Face Hub save and load for NNModel: the file names, the safetensors shape preflight, the Hub manifest, and the bodies of save_pretrained / _save_pretrained / _from_pretrained. Moved from nn_model (FEAT-076). Internal."""` and `from __future__ import annotations`.
2. Move these module-level definitions from `nn_model.py` into it, in this order and unchanged, as they stand after the dependency tickets:
   - the Hub constants and their comments (at `9ad461e` L179–L193): `_HUB_MODEL_FILENAME`, `_HUB_CONFIG_FILENAME`, `_HUB_TOKENIZER_FILENAME`, `_HUB_MANIFEST_FILENAME`, `_HUB_MANIFEST_FORMAT`, `_HUB_MANIFEST_VERSION`, `_HUB_MOVE_ORDER` (FEAT-061 deletes `_SHA256_HEX` and FEAT-086 deletes `_HUB_DIGEST_CHUNK`; move whatever is left there);
   - everything from `_SAFETENSORS_MAX_HEADER_BYTES` through the end of `_verify_hub_manifest` (at `9ad461e` L1261–L1452): `_safetensors_shapes`, `_declared_module`, `_refuse_shapes_before_build`, `_file_digest`, `_manifest_files`, `_previous_hub_files`, `_write_hub_manifest`, `_verify_hub_manifest`, plus FEAT-092's `_load_hub_state`. `_looks_like_converted_qat_state` (L1454) and `_CallbackFinalizer` (L1458) stay in `nn_model.py`.
3. Add three functions to `_hub.py` holding the method bodies, with `self` renamed to `model`:
   - `def save_pretrained(model: NNModel, base_save_pretrained: Callable[..., Any], save_directory: Any, *args: Any, **kwargs: Any) -> Any:` — the body of `NNModel.save_pretrained` (L4608–L4636), with `self.` → `model.` and `super().save_pretrained(staging, *args, **kwargs)` → `base_save_pretrained(staging, *args, **kwargs)`;
   - `def write_pretrained(model: NNModel, save_directory: Any) -> None:` — the body of `NNModel._save_pretrained` (L4733–L4783), `self.` → `model.` (it still calls `model._require_portable(...)` and `model._hub_reconstruction_config(...)`, which subclasses override);
   - `def from_pretrained(cls: type[NNModel], *, model_id: str, revision=None, cache_dir=None, force_download: bool = False, local_files_only: bool = False, token=None, map_location: str = "cpu", strict: bool = True, precision: Optional[PrecisionPolicy] = None, exclude_submodules: Sequence[str] = (), **model_kwargs) -> NNModel:` — the body of `NNModel._from_pretrained` (L4842–L4979), keeping `cls` (copy the parameter list exactly as it is on `develop` when you start).
   Each gets a one-line docstring naming the `NNModel` method it implements.
4. Give `_hub.py` the imports the moved code uses: the standard-library and typing names, `torch`, `from .. import _atomic_io`, `from .._artifacts import parse_json`, `from .._json_depth import MAX_JSON_DEPTH`, `from .._validation import is_sha256_hex`, `from ..models import MissingModelFactoryError, ModelSpec, RuntimeModule, build_module, check_state_schema`, `from ..precision import PrecisionPolicy`, `from ..transforms import _canonical_transforms, _refuse_unrecorded_recipe_state, _replay_transforms`, `from ._resume import _attached_submodules_error, _excluded_names, _without_submodules`, `from .enum.devices import Devices`, `from .enum.nets import Nets`, `from .params.nn_checkpoint import NNCheckpointTransform, _tensor_state_dict`, `from .params.nn_model_params import NNModelParams`, `from .params.nn_params import NNParams`, and under `if TYPE_CHECKING:` `from .nn_model import NNModel`. Adjust the list until `uv run ruff check src/nnx/nn/_hub.py` and `uv run pyright --warnings` pass; it must not import `nn_model` outside `TYPE_CHECKING`.
5. In `nn_model.py`:
   - replace the three method bodies with delegations, keeping each signature, decorator and docstring exactly:
     ```python
         def save_pretrained(self, save_directory, *args: Any, **kwargs: Any) -> Any:
             """<unchanged docstring>"""
             return _hub.save_pretrained(self, super().save_pretrained, save_directory, *args, **kwargs)
     ```
     ```python
         def _save_pretrained(self, save_directory) -> None:
             """<unchanged docstring>"""
             _hub.write_pretrained(self, save_directory)
     ```
     and in `_from_pretrained` a single `return _hub.from_pretrained(cls, model_id=model_id, revision=revision, cache_dir=cache_dir, force_download=force_download, local_files_only=local_files_only, token=token, map_location=map_location, strict=strict, precision=precision, exclude_submodules=exclude_submodules, **model_kwargs)`;
   - add `from . import _hub` and a re-export of every module-level name from step 2 with the redundant alias (`from ._hub import _write_hub_manifest as _write_hub_manifest`, …);
   - keep the `_HubMixinBase` `try/except` block and `_HUB_AVAILABLE` (L149–L177) in `nn_model.py`: `NNModel` subclasses it and `tests/test_imports.py` reads `nn_model._HUB_AVAILABLE`;
   - run `uv run ruff check --fix src/nnx/nn/nn_model.py` to drop the imports the remaining code no longer uses.
6. No test patch needs a new target: no test patches a Hub name on `nn_model` (`grep -rn 'setattr(nn_model' tests | grep -i hub` prints nothing). The Hub tests patch library attributes (`safetensors.torch.save_file`, `huggingface_hub.snapshot_download`, `huggingface_hub.hub_mixin.HfApi`, `torch.save`, `os.replace`, `nets_module.FeedFwdNN`), which the moved code still reads at call time through its local imports. `tests/test_hub_mixin.py` imports `_write_hub_manifest` from `nn_model`; the re-export serves it.
7. Add this test to `tests/test_imports.py`:
   ```python
   def test_hub_io_lives_in_nnx_nn_hub_and_stays_importable_from_nn_model():
       """FEAT-076: NNModel's Hub methods delegate to nnx.nn._hub; nn_model re-exports its names."""
       import ast
       import inspect
       import pathlib

       import nnx.nn._hub as hub
       import nnx.nn.nn_model as nn_model

       for name in ("_HUB_MODEL_FILENAME", "_HUB_CONFIG_FILENAME", "_HUB_TOKENIZER_FILENAME", "_HUB_MANIFEST_FILENAME",
                    "_write_hub_manifest", "_verify_hub_manifest", "_refuse_shapes_before_build", "_safetensors_shapes"):
           assert getattr(nn_model, name) is getattr(hub, name), name
       for method in ("save_pretrained", "_save_pretrained", "_from_pretrained"):
           assert len(inspect.getsource(getattr(nn_model.NNModel, method)).splitlines()) < 60, method
       tree = ast.parse(pathlib.Path(hub.__file__).read_text(encoding="utf-8"))
       assert not [n for n in tree.body if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("nn_model")]
   ```

No behaviour changes, so no CHANGELOG entry. The three methods keep their docstrings, so `docs/api.md` is unchanged.

**Worker setup.** Branch from `develop` after every ticket in `depends_on` has merged. `uv sync --frozen --all-extras` installs torch, `huggingface_hub`, `safetensors` and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_imports.py::test_hub_io_lives_in_nnx_nn_hub_and_stays_importable_from_nn_model` fails before the move and passes after it. Check: `uv run pytest tests/test_imports.py`
- [ ] The API reference is unchanged. Check: `uv run python -m scripts.docs.build_api_reference --check`
- [ ] The Hub suites pass with and without the `hub` extra's import path. Check: `uv run pytest tests -k "hub or generative_nn_model or external_module_runtime or imports"`
- [ ] The full suite passes. Check: `uv run pytest -n logical --dist loadgroup`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L149-L193](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L149-L193) | `_HubMixinBase` / `_HUB_AVAILABLE` (stay) and the Hub constants (move) |
| [nn_model.py L1261-L1452](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1261-L1452) | The shape preflight and the Hub manifest helpers |
| [nn_model.py L4592-L4636](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4592-L4636) | `save_pretrained` and its `super()` call |
| [nn_model.py L4716-L4783](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4716-L4783) | `_save_pretrained` |
| [nn_model.py L4794-L4979](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4794-L4979) | `_from_pretrained` |
| [generative_nn_model.py L60-L74](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/generative_nn_model.py#L60-L74) | A subclass overriding `_hub_reconstruction_config`, which stays a method |
| [test_imports.py L80](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_imports.py#L80) | `nn_model._HUB_AVAILABLE` read by a test |
| #509, #441, #260, #767 | Unreplayable-save refusal and `exclude_submodules`; rebuild without attached submodules; atomic config; pre-write refusal |

## Scope

**In:** the Hub constants and helpers, the three method bodies, their re-exports and delegations, one test.

**Out:**
- `push_to_hub`, `_refuse_unloadable_hub_artifact`, `_only_attached_submodules`, `_require_portable`, `_hub_reconstruction_config` and `_hub_reconstruction_kwargs`: they stay `NNModel` methods (subclass hooks or small guards).
- Behaviour changes to the Hub path: FIX-110, FIX-112, FIX-113, FEAT-092, FEAT-101, FEAT-102.
- Resume (FEAT-074), transform replay (FEAT-075), DDP (FEAT-077), lazy modules (FEAT-078).

## Dependencies

**Blocked by:** FIX-110, FIX-112, FIX-113, FEAT-092, FEAT-101, FEAT-102, FEAT-074, FEAT-075, FEAT-061, FEAT-073, FEAT-086
**Blocks:** none
**Related:** #509, #441, #260, #767
