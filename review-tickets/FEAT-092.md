---
id: FEAT-092
title: "[FEAT-092] Share the Hub strict-load sequence between the preflight and the load"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-12"]
depends_on: ["FIX-112"]
related_issues: [441, 509]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-092: `NNModel.from_pretrained` runs the same four-step strict-load check twice: once on a meta-device module before the network is allocated (`_refuse_shapes_before_build`) and once on the real network (`_from_pretrained`). The two copies differ only in how they re-raise. This ticket moves the sequence into one helper, `_load_hub_state`, with a keyword that selects the preflight's re-raise. Every message, exception type and `__cause__` stays as it is.

## Context

**What happens today.** The preflight, `_refuse_shapes_before_build` (`nn_model.py` L1347–L1356):

```python
    if not transforms and strict:
        _refuse_unrecorded_recipe_state(state, module.state_dict())
    try:
        if net_params is None and strict:
            check_state_schema(module, state, what=f"Hub artifact of {params.net}")
        module.load_state_dict(state, strict=strict)
    except (RuntimeError, ValueError) as error:
        explained = _attached_submodules_error(error, module, state, call, excluded) if strict else None
        refusal = explained if explained is not None else error
        raise type(refusal)(f"refused before the network was built: {refusal}") from error
```

The real load, in `_from_pretrained` (L4968–L4978):

```python
        if not transforms and strict:
            _refuse_unrecorded_recipe_state(state_dict, model.net.state_dict())
        try:
            if net_params is None and strict:
                check_state_schema(model.net, state_dict, what=f"Hub artifact of {params.net}")
            model.net.load_state_dict(state_dict, strict=strict)
        except (RuntimeError, ValueError) as error:
            explained = _attached_submodules_error(error, model.net, state_dict, call, excluded) if strict else None
            if explained is not None:
                raise explained from error
            raise
```

**Why it matters.** A change to one copy (a new check, another exception type) must be made twice, or the preflight and the load start refusing different artifacts.

**History.** #441 added `exclude_submodules` to Hub `from_pretrained` and required that "The unexpected-keys hint names the option of the entry point the caller used": that hint is `_attached_submodules_error` with the `call` template. #509 extended it to top-level tensors. The preflight came with ed93d51 (#855).

**Background for the worker.**
- Strict load: `module.load_state_dict(state, strict=True)`, which raises torch's `RuntimeError` ("Error(s) in loading state_dict for …: Missing key(s) …", "size mismatch for …").
- `check_state_schema(module, state, *, what)` (`models.py` L622): runs for a registered `ModelSpec` (`net_params is None`) and raises `ValueError(f"{what}: the rebuilt module's topology does not match the saved weights (...)")`.
- `_refuse_unrecorded_recipe_state(net_state, base_state)` (`nn_model.py` L1252): raises `ValueError("the weights come from a transformed topology ... record no transformation recipe ...")`. It runs **outside** the `try` in both copies, so the preflight never prefixes it.
- `call`: the caller's entry point as a format template with one field, `{names}`, for example `"NNModel.from_pretrained(model_id, exclude_submodules={names!r})"` (built in `_from_pretrained` L4958 and passed to the preflight). `_attached_submodules_error(error, net, state, call, excluded)` (L411) returns `None` when the weights hold no top-level submodule or tensor the net lacks; otherwise it returns `type(error)(f"{error}\nThe weights hold top-level submodules or tensors ... {call.format(names=(*excluded, *attached))} rebuilds the architecture without them")`.
- Exception types that reach the `except`: only `RuntimeError` (torch's strict load) and `ValueError` (`check_state_schema`). Both take one message argument, and `_attached_submodules_error` returns the same type, so `type(refusal)(message)` builds the same type with the prefixed text. The preflight already does this today.
- Meta device: `torch.device("meta")`, on which a module is built with no storage; the preflight builds the declared network there.

**Proposal.**
1. In `src/nnx/nn/nn_model.py`, directly above `_refuse_shapes_before_build` (L1324), add:

   ```python
   def _load_hub_state(
       module: torch.nn.Module,
       state: Mapping[str, Any],
       *,
       net_params: Optional[NNParams],
       params: NNModelParams,
       transforms: Sequence[Any],
       excluded: Any,
       strict: bool,
       call: str,
       before_build: bool,
   ) -> None:
       """Load a Hub artifact's ``state`` into ``module`` as ``from_pretrained``
       does: a transformed topology with no recorded recipe is refused first
       (strict loads only), a registered module's state is checked against its
       schema, then the state loads. A strict refusal that top-level submodules
       or tensors explain gets the ``exclude_submodules`` hint (``call``: the
       entry point, formatted with ``names``). ``before_build`` (the preflight,
       on the meta module) prefixes the strict load's refusals with "refused
       before the network was built: "; the recipe refusal is never prefixed."""
       if not transforms and strict:
           _refuse_unrecorded_recipe_state(state, module.state_dict())
       try:
           if net_params is None and strict:
               check_state_schema(module, state, what=f"Hub artifact of {params.net}")
           module.load_state_dict(state, strict=strict)
       except (RuntimeError, ValueError) as error:
           explained = _attached_submodules_error(error, module, state, call, excluded) if strict else None
           if before_build:
               refusal = explained if explained is not None else error
               raise type(refusal)(f"refused before the network was built: {refusal}") from error
           if explained is not None:
               raise explained from error
           raise
   ```

2. In `_refuse_shapes_before_build`, replace L1347–L1356 (from `if not transforms and strict:` to the `raise type(refusal)(...)` line) with:

   ```python
       _load_hub_state(
           module,
           state,
           net_params=net_params,
           params=params,
           transforms=transforms,
           excluded=excluded,
           strict=strict,
           call=call,
           before_build=True,
       )
   ```

3. In `_from_pretrained`, replace L4968–L4978 (from `if not transforms and strict:` to the bare `raise`) with:

   ```python
           _load_hub_state(
               model.net,
               state_dict,
               net_params=net_params,
               params=params,
               transforms=transforms,
               excluded=excluded,
               strict=strict,
               call=call,
               before_build=False,
           )
   ```

   `return model` (L4979) stays.
4. Add two pin tests to `tests/test_hub_load_preflight.py` (they pass before and after the change):

   ```python
   def test_the_preflight_prefixes_the_strict_load_refusal_and_chains_the_original(tmp_path):
       _tiny_model().save_pretrained(str(tmp_path))
       config_path = tmp_path / "config.json"
       config = json.loads(config_path.read_text(encoding="utf-8"))
       config["net_params"]["hidden_dims"] = "[9]"
       config_path.write_text(json.dumps(config, sort_keys=True, indent=2), encoding="utf-8")
       _resign(tmp_path)
       with pytest.raises(RuntimeError) as caught:
           NNModel.from_pretrained(str(tmp_path))
       assert str(caught.value).startswith("refused before the network was built: ")
       assert type(caught.value.__cause__) is RuntimeError
       assert not str(caught.value.__cause__).startswith("refused before")


   def test_the_no_recipe_refusal_is_not_prefixed(tmp_path):
       from nnx.transforms import TransformRecipe, lora

       model = _tiny_model()
       linear = next(name for name, module in model.net.named_modules() if type(module) is torch.nn.Linear)
       TransformRecipe([lora(linear, r=2, alpha=4.0)]).materialize(model)
       model.save_pretrained(str(tmp_path))
       config_path = tmp_path / "config.json"
       config = json.loads(config_path.read_text(encoding="utf-8"))
       config_path.write_text(json.dumps({key: value for key, value in config.items() if key != "transforms"}))
       _resign(tmp_path)
       with pytest.raises(ValueError, match="record no transformation recipe") as caught:
           NNModel.from_pretrained(str(tmp_path))
       assert str(caught.value).startswith("the weights come from a transformed topology")
   ```

5. No message changes, no test is retargeted (no test patches either function: `grep -rn "_refuse_shapes_before_build\|_load_hub_state" tests` returns nothing), and there is no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FIX-112 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The two new pin tests pass on `develop` before the source change, and after it. Check: `uv run pytest tests/test_hub_load_preflight.py -k "prefixes_the_strict_load_refusal or no_recipe_refusal_is_not_prefixed"`
- [ ] The strict-load sequence exists once. Check: `grep -c "_refuse_unrecorded_recipe_state(state" src/nnx/nn/nn_model.py` prints `1`, and `grep -c "check_state_schema(.*Hub artifact of" src/nnx/nn/nn_model.py` prints `1`
- [ ] Both callers use the helper. Check: `grep -c "_load_hub_state(" src/nnx/nn/nn_model.py` prints `3` (the definition and two calls)
- [ ] The Hub suites pass unchanged. Check: `uv run pytest tests/test_hub_load_preflight.py tests/test_hub_mixin.py tests/test_hub_manifest.py tests/test_external_module_runtime.py tests/test_model_recipes.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L1324-L1356](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1324-L1356) | `_refuse_shapes_before_build`: the preflight copy, with the prefixing re-raise |
| [nn_model.py L4955-L4979](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4955-L4979) | `_from_pretrained`: `call`, the preflight call and the load copy |
| [nn_model.py L411-L429](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L411-L429) | `_attached_submodules_error`: how `call` is formatted |
| [nn_model.py L1252-L1258](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1252-L1258) | `_refuse_unrecorded_recipe_state` |
| [models.py L622-L642](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L622-L642) | `check_state_schema`: raises `ValueError` |
| [test_hub_load_preflight.py L24-L66](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_hub_load_preflight.py#L24-L66) | the existing preflight tests (prefix and message asserted) |
| [test_hub_mixin.py L395-L425](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_hub_mixin.py#L395-L425) | the no-recipe refusal test the new pin test mirrors |
| #441, #509 | the `exclude_submodules` hint and its entry-point wording |
| ed93d51 (#855) | the commit that added the preflight |

## Scope

**In:** `_load_hub_state`, its two callers, two pin tests.

**Out:** building a registered factory once per `from_pretrained` (FIX-112). The fake `NNModel` in `_declared_module` (FEAT-108). Moving the Hub helpers out of `nn_model.py` (FEAT-076). The checkpoint load path in `from_checkpoint`, which has its own sequence and messages.

## Dependencies

**Blocked by:** FIX-112
**Blocks:** FEAT-076
**Related:** #441, #509
