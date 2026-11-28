---
id: FEAT-101
title: "[FEAT-101] Build the state dict once per save_pretrained"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-20"]
depends_on: ["FIX-110"]
related_issues: [509, 767]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-101: Before `save_pretrained` writes anything, its refusal checks call `self.net.state_dict()` three times (quantized tensors, attached submodules, non-tensor entries), and `NNCheckpoint.to_file` runs `_require_portable` twice on one safetensors save. This ticket builds the state dict once for the refusal checks, drops the second `_require_portable` inside the private `_to_safetensors_file`, and documents why `_save_pretrained` keeps its own check. Every refusal and message stays.

## Context

**What happens today.** `_refuse_unloadable_hub_artifact` (`nn_model.py` L4648–L4688) reads the state dict for the quantized check (L4654–L4658):

```python
        quantized = sorted(
            name
            for name, tensor in self.net.state_dict().items()
            if isinstance(tensor, torch.Tensor) and type(tensor) not in (torch.Tensor, torch.nn.Parameter)
        )
```

again through `self._only_attached_submodules()` (L4683; it reads `self.net.state_dict()` at L4697), and a third time at the end (L4688):

```python
        _refuse_non_tensor_state(self.net.state_dict(), operation="Hugging Face Hub export")
```

`_save_pretrained` (L4716), the hook huggingface_hub calls, repeats `self._require_portable("save_pretrained")` (L4733) and reads the state a fourth time for `_tensor_state_dict` (L4750). In `nn_checkpoint.py`, `to_file` calls `self._require_portable()` before anything is created (L604), and `_to_safetensors_file`, reached only from `to_file` (L613), calls it again (L641).

**Why it matters.** Each `state_dict()` call walks every submodule and builds a new ordered dict; the checks need one snapshot. Duplicate guards also make it unclear which one is the contract.

**History.** #509 made `save_pretrained` and `push_to_hub` refuse what `from_pretrained` cannot rebuild, before anything is written. #767 (DEC-013) added the same refusal to `to_file(format="safetensors")`, before any directory is created. Neither guard moves.

**Background for the worker.**
- `save_pretrained` (L4592): NNx's override of huggingface_hub's `PyTorchModelHubMixin.save_pretrained`. It refuses first (`_refuse_unloadable_hub_artifact`), then calls the mixin, which calls `self._save_pretrained(directory)` with no other arguments; so `_save_pretrained` cannot receive the state from `save_pretrained`, and keeps its own read.
- `_require_portable` refuses a runtime-only module (`RuntimeModule` descriptor, `reconstructible=False`) with `MissingModelFactoryError`.
- `_refuse_non_tensor_state(state, *, operation)` (`nn_checkpoint.py` L187): `TypeError` for a non-tensor entry (module extra state); `_tensor_state_dict` (L198) runs it again before cloning.
- `self.net.state_dict()` returns references, not copies; nothing between the three reads changes the network, so one snapshot gives the same answers.

**Proposal.**
1. In `src/nnx/nn/nn_model.py`, `_refuse_unloadable_hub_artifact`:
   - directly after `self._require_portable(operation)` (L4652), add `state = self.net.state_dict()  # one snapshot for every check below`;
   - in the `quantized` expression, replace `self.net.state_dict().items()` with `state.items()`;
   - in the `drift = ...` line (L4683), replace `self._only_attached_submodules()` with `self._only_attached_submodules(state)`;
   - replace the last line (L4688) with `_refuse_non_tensor_state(state, operation="Hugging Face Hub export")`. Keep the comment above it.
2. Change `_only_attached_submodules` (L4690) to `def _only_attached_submodules(self, state: Optional[Mapping[str, Any]] = None) -> bool:`, add to its docstring "``state`` is the net's state dict when the caller has read it already.", and in the `live = ...` line (L4697) replace `self.net.state_dict().items()` with `(self.net.state_dict() if state is None else state).items()`.
3. In `_save_pretrained`, add on the line above `self._require_portable("save_pretrained")` (L4733):

   ```python
           # Repeated from save_pretrained on purpose: huggingface_hub calls this hook, and a caller can call it
           # directly, without save_pretrained's refusals.
   ```

4. In `src/nnx/nn/params/nn_checkpoint.py`, `_to_safetensors_file` (L639): delete `self._require_portable()` (L641) and change the docstring to `"""Atomic safetensors write. Requires the ``thekaveh-nnx[hub]`` extra. Reached only through :meth:`to_file`, which refuses a runtime-only module first."""`.
5. Add to `tests/test_hub_mixin.py`:

   ```python
   def test_the_hub_refusal_checks_read_the_state_dict_once(monkeypatch):
       model = _tiny_model()
       calls = []
       original = model.net.state_dict

       def counting(*args, **kwargs):
           calls.append(1)
           return original(*args, **kwargs)

       monkeypatch.setattr(model.net, "state_dict", counting)
       model._refuse_unloadable_hub_artifact("save_pretrained refused before writing anything")
       # One snapshot for the checks, and one inside the topology comparison (_topology_problems).
       assert len(calls) == 2
   ```

6. No message changes; no test patches `_only_attached_submodules`, `_refuse_unloadable_hub_artifact` or `_to_safetensors_file` (`grep -rn` over `tests/` finds none), so none is retargeted. No behaviour changes, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FIX-110 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test fails on `develop` before the change (4 reads) and passes after it. Check: `uv run pytest tests/test_hub_mixin.py::test_the_hub_refusal_checks_read_the_state_dict_once`
- [ ] The refusal checks share one snapshot. Check: `awk '/def _refuse_unloadable_hub_artifact/,/def _only_attached_submodules/' src/nnx/nn/nn_model.py | grep -c "self.net.state_dict()"` prints `1`
- [ ] `_to_safetensors_file` no longer checks portability. Check: `grep -c "self._require_portable()" src/nnx/nn/params/nn_checkpoint.py` prints `1`
- [ ] `_save_pretrained` keeps its check, with the comment. Check: `grep -n "Repeated from save_pretrained on purpose" src/nnx/nn/nn_model.py` returns 1 line
- [ ] Every Hub and safetensors refusal still holds. Check: `uv run pytest tests/test_hub_mixin.py tests/test_hub_manifest.py tests/test_hub_load_preflight.py tests/test_checkpoint_safetensors.py tests/test_model_recipes.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L4592-L4636](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4592-L4636) | `save_pretrained`: refuses, then the mixin calls `_save_pretrained` |
| [nn_model.py L4648-L4688](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4648-L4688) | `_refuse_unloadable_hub_artifact`: reads at L4656 and L4688 |
| [nn_model.py L4690-L4705](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4690-L4705) | `_only_attached_submodules`: the read at L4697 |
| [nn_model.py L4716-L4750](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4716-L4750) | `_save_pretrained`: the repeated check and the read for the clone |
| [nn_checkpoint.py L601-L613](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L601-L613) | `to_file`: `_require_portable` before anything is created |
| [nn_checkpoint.py L639-L641](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L639-L641) | `_to_safetensors_file`: the second check |
| [nn_checkpoint.py L187-L200](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L187-L200) | `_refuse_non_tensor_state`, `_tensor_state_dict` |
| #509, #767 | the refusals before anything is written |

## Scope

**In:** the shared snapshot in `_refuse_unloadable_hub_artifact`, the `state` parameter of `_only_attached_submodules`, the `_save_pretrained` comment, the dropped `_to_safetensors_file` check, one test.

**Out:** the read inside `_save_pretrained` (the mixin passes no state) and inside `nnx.transforms._topology_problems`. The tokenizer removal order (FIX-110). Moving the Hub I/O helpers out of `nn_model.py` (FEAT-076).

## Dependencies

**Blocked by:** FIX-110
**Blocks:** FEAT-076
**Related:** #509, #767
