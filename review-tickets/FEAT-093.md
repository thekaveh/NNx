---
id: FEAT-093
title: "[FEAT-093] Bundle NNModel's lazy-module tracking into one object"
labels: ["enhancement", "area: model-integration", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-13"]
depends_on: []
related_issues: [562, 764]
status: READY
---

**Category:** Model integration · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-093: `NNModel` follows a wrapped lazy module (#562) through five separate attributes (`_lazy_wrap_keys`, `_lazy_layers`, `_lazy_places`, `_lazy_stamp`, `_lazy_wrapped`), each read with `getattr(..., default)`, and `__init__` fingerprints a wrapped module twice. This ticket puts the five attributes into one `_LazyTracker` object (`self._lazy`), keeps the pickled form exactly as it is today (so pickles load in older and newer NNx alike), and fingerprints the module once. The refresh of `params` on read stays: it is #562's design.

## Context

**What happens today.** `__init__` sets the five attributes (`nn_model.py` L3935–L3952, comments omitted by quoting two fragments). First:

```python
            self._lazy_wrap_keys: Optional[set[str]] = (
                _tensor_keys(module.state_dict())
                if _has_pending_lazy(module) and _lazy_layers_emulable(module)
                else None
            )
```

then:

```python
            self._lazy_stamp: Optional[tuple[object, int]] = _module_registrations() if pending else None
            # The module those keys describe: a model given another net (a
            # copy made by quantize_int8, say) keeps its descriptor.
            self._lazy_wrapped: Optional[torch.nn.Module] = module if self._lazy_wrap_keys is not None else None
```

`_refresh_lazy_descriptor` (L4012–L4066) reads every one through `getattr`, for pickles written before each existed:

```python
        keys = getattr(self, "_lazy_wrap_keys", None)
        module = getattr(self, "net", None)
        if keys is None or module is None:
            return
        if module is not getattr(self, "_lazy_wrapped", module) or "params" not in self.__dict__:
```

The double fingerprint: `__init__` computes `RuntimeModule.of(module) != params.net` for `inexact` (L3893–L3898), and `_resolve_net_descriptor` computes `RuntimeModule.of(module)` again (L3615).

**Why it matters.** Five loosely coupled attributes with defaults scattered over two methods are easy to update inconsistently, and `RuntimeModule.of` hashes the whole state-dict layout and `repr` of the module, twice per wrap.

**History.** #562 made `NNModel` take a lazy module's descriptor afresh once it materializes, at the first `params` read; #764 (DEC-010) kept that default ("#562 makes `NNModel` take the module's `RuntimeModule` descriptor afresh after that forward"). The `params` property's comment states the pickle rule this ticket must keep: "Kept in the instance dict under its own name, as before it was a property, so pickles load across nnx versions in both directions." So the refresh on read is **not** changed here, and the pickled form keeps the five attribute names.

**Background for the worker.**
- Lazy module: a module with layers such as `nn.LazyLinear` that infer their input size at the first forward ("materialization").
- `RuntimeModule`: the descriptor of a caller-owned module (`NNModel(module=...)`): its qualified class name plus a `topology` hash (`models.module_topology`). The run id uses it.
- `_module_registrations()` (`models.py` L357): a `(token, count)` stamp that changes whenever a module bringing a pending lazy layer is registered anywhere in the process.
- What each attribute means (the tracker's fields):
  - `keys` (`_lazy_wrap_keys`): the module's tensor keys at the wrap; materialization keeps them, surgery does not.
  - `layers` (`_lazy_layers`): every lazy layer seen pending, at the wrap and at each walk.
  - `places` (`_lazy_places`): those layers by the dotted names a walk found them at.
  - `stamp` (`_lazy_stamp`): the registration stamp after that walk.
  - `wrapped` (`_lazy_wrapped`): the module the keys describe. `None` in the tracker means "absent from an old pickle": any net then counts as the wrapped one, as `getattr(self, "_lazy_wrapped", module)` does today.
- Refresh semantics (unchanged, now `_LazyTracker.refresh(model)`): at each `params` read while the tracker exists, (a) stop tracking when `model.net` is another module than `wrapped` or `params` is not in the instance dict; (b) return early while the stamp is unchanged, some layer in `places` is still pending and every place still holds its layer; (c) otherwise walk the module: if lazy layers are still pending, record them in `layers` / `places` / `stamp` and return; (d) when none is pending, stop tracking and, if the tensor keys still equal `keys`, settle the loaded layers and replace `params.net` with the fresh descriptor. "Stop tracking" is `model._lazy = None` (today `self._lazy_wrap_keys = None`).
- Pickling: Python calls `__getstate__` to get the dict it stores and `__setstate__` to restore it (also for `copy.copy` and `copy.deepcopy`). `NNModel` defines neither today.

**Proposal.**
1. In `src/nnx/nn/nn_model.py`, directly above `class _LazyReload` (L3822), add:

   ```python
   # The attributes a pickle stores for _LazyTracker, its fields before it existed (#562).
   _LAZY_LEGACY_ATTRIBUTES = ("_lazy_wrap_keys", "_lazy_layers", "_lazy_places", "_lazy_stamp", "_lazy_wrapped")


   class _LazyTracker:
       """How ``NNModel`` follows a wrapped lazy module until it materializes
       (#562): ``keys``, its tensor keys at the wrap; ``layers``, the lazy
       layers seen pending at the wrap and at each walk since; ``places``,
       those layers by the names a walk found them at; ``stamp``, the
       registration stamp after that walk (:func:`nnx.models._module_registrations`);
       ``wrapped``, the module the keys describe (``None``: not recorded, as in
       a pickle written before it was, so any net is the wrapped one)."""

       __slots__ = ("keys", "layers", "places", "stamp", "wrapped")

       def __init__(
           self,
           *,
           keys: set[str],
           layers: list[torch.nn.Module],
           places: dict[str, torch.nn.Module],
           stamp: Optional[tuple[object, int]],
           wrapped: Optional[torch.nn.Module],
       ) -> None:
           self.keys = keys
           self.layers = layers
           self.places = places
           self.stamp = stamp
           self.wrapped = wrapped

       @classmethod
       def of(cls, module: torch.nn.Module) -> Optional[_LazyTracker]:
           """The tracker of a module wrapped with ``module=``; ``None`` when there
           is nothing to follow: no pending lazy layer, or one torch does not ship
           (nothing could check a fresh instance against a later descriptor)."""
           if not (_has_pending_lazy(module) and _lazy_layers_emulable(module)):
               return None
           pending = _pending_lazy_modules(module)
           return cls(
               keys=_tensor_keys(module.state_dict()),
               layers=[layer for _, layer in pending],
               places=dict(pending),
               stamp=_module_registrations() if pending else None,
               wrapped=module,
           )

       @classmethod
       def from_legacy(cls, state: Mapping[str, Any]) -> Optional[_LazyTracker]:
           """The tracker a pickle's ``_lazy_*`` attributes describe; ``None``
           when it holds no wrap keys (nothing tracked)."""
           keys = state.get("_lazy_wrap_keys")
           if keys is None:
               return None
           return cls(
               keys=keys,
               layers=state.get("_lazy_layers", []),
               places=state.get("_lazy_places", {}),
               stamp=state.get("_lazy_stamp"),
               wrapped=state.get("_lazy_wrapped"),
           )

       def legacy_state(self) -> dict[str, Any]:
           """The ``_lazy_*`` attributes a pickle stores for this tracker."""
           state: dict[str, Any] = {
               "_lazy_wrap_keys": self.keys,
               "_lazy_layers": self.layers,
               "_lazy_places": self.places,
               "_lazy_stamp": self.stamp,
           }
           if self.wrapped is not None:
               state["_lazy_wrapped"] = self.wrapped
           return state

       def refresh(self, model: NNModel) -> None:
           """One ``params`` read of ``model``: see :meth:`NNModel._refresh_lazy_descriptor`.
           Tracking ends with ``model._lazy = None``."""
           module = getattr(model, "net", None)
           if module is None:
               return
           if (self.wrapped is not None and module is not self.wrapped) or "params" not in model.__dict__:
               # Another net, or a subclass keeping params elsewhere: nothing to
               # take afresh here.
               model._lazy = None
               return
           stamp = _module_registrations()
           if (
               self.stamp == stamp
               and any(_is_pending_lazy(layer) for layer in self.places.values())
               and all(_submodule_at(module, name) is layer for name, layer in self.places.items())
           ):
               return
           pending = _pending_lazy_modules(module)
           if pending:
               self.layers.extend(layer for _, layer in pending if all(layer is not other for other in self.layers))
               # The places still holding their layer, materialized ones
               # included, so a layer put in place of one is seen; then the
               # pending ones found now.
               kept = {name: layer for name, layer in self.places.items() if _submodule_at(module, name) is layer}
               self.places = {**kept, **dict(pending)}
               self.stamp = stamp
               return
           model._lazy = None
           net = model.__dict__["params"].net
           if isinstance(net, RuntimeModule) and _tensor_keys(module.state_dict()) == self.keys:
               reachable = {id(sub) for sub in module.modules()}
               _settle_loaded_lazy_layers(layer for layer in self.layers if id(layer) in reachable)
               descriptor = RuntimeModule.of(module)
               if _same_class_once_materialized(descriptor.module, net.module):
                   model.__dict__["params"] = replace(model.__dict__["params"], net=descriptor)
   ```

   `refresh` is the body of today's `_refresh_lazy_descriptor` from L4032 on, with `getattr(self, "_lazy_*", default)` replaced by the field and `self._lazy_wrap_keys = None` replaced by `model._lazy = None`, at the same two points.
2. In `NNModel.__init__`, replace the five assignments (L3935–L3952, from `self._lazy_wrap_keys: Optional[set[str]] = (` to the `self._lazy_wrapped: ...` line, with their comments) by:

   ```python
               self._lazy: Optional[_LazyTracker] = _LazyTracker.of(module)
   ```

   Keep the `# #562: ...` comment block above it (L3930–L3934), and change the next line, `if inexact and reload is None and self._lazy_wrap_keys is not None:` (L3953), to `if inexact and reload is None and self._lazy is not None:`.
3. Replace the body of `NNModel._refresh_lazy_descriptor` (L4028–L4066, after its docstring; keep the docstring) with:

   ```python
           lazy = getattr(self, "_lazy", None)  # getattr: stand-ins and subclasses that bypass __init__
           if lazy is not None:
               lazy.refresh(self)
   ```

4. In the `params` getter (L3993), change `if getattr(self, "_lazy_wrap_keys", None) is not None:` to `if getattr(self, "_lazy", None) is not None:`. In `from_checkpoint` (L4568), make the same change: `if getattr(model, "_lazy", None) is not None:`.
5. Add the pickle hooks to `NNModel`, directly after the `params` deleter (after L4010):

   ```python
       def __getstate__(self) -> dict[str, Any]:
           """The instance dict, with the lazy-module tracker stored as the five
           ``_lazy_*`` attributes it replaced (#562): a pickle loads in older and
           newer nnx alike."""
           state = dict(self.__dict__)
           lazy = state.pop("_lazy", None)
           if lazy is not None:
               state.update(lazy.legacy_state())
           return state

       def __setstate__(self, state: dict[str, Any]) -> None:
           state = dict(state)
           legacy = {name: state.pop(name) for name in _LAZY_LEGACY_ATTRIBUTES if name in state}
           state["_lazy"] = _LazyTracker.from_legacy(legacy)
           self.__dict__.update(state)
   ```

6. Fingerprint the module once. In `NNModel.__init__`, replace the `inexact = (...)` expression (L3893–L3898) with:

   ```python
           descriptor = (
               RuntimeModule.of(module)
               if module is not None and isinstance(params.net, RuntimeModule) and isinstance(module, torch.nn.Module)
               else None
           )
           inexact = descriptor is not None and descriptor != params.net
   ```

   and pass `descriptor=descriptor` as the last argument of the `_resolve_net_descriptor(...)` call (L3899–L3906). In `_resolve_net_descriptor` (L3592), add the keyword parameter `descriptor: Optional[RuntimeModule] = None` after `explain_state`, add to its docstring "``descriptor`` is ``RuntimeModule.of(module)`` when the caller took it already (``NNModel.__init__``), so the module is fingerprinted once.", and change L3615 to `descriptor = RuntimeModule.of(module) if descriptor is None else descriptor`. The other caller, `plans.py` L555, passes no `descriptor` and is unchanged.
7. Add three tests to `tests/test_external_module_runtime.py`, after `test_params_stays_in_the_instance_dict_for_pickles_across_versions` (L1295):

   ```python
   def test_lazy_tracking_pickles_as_the_attributes_older_nnx_reads():
       import pickle

       from nnx.nn.nn_model import _LazyTracker

       model = NNModel(module=_lazy(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
       assert isinstance(model._lazy, _LazyTracker)
       state = model.__getstate__()
       assert "_lazy" not in state
       assert state["_lazy_wrap_keys"] == model._lazy.keys and state["_lazy_wrapped"] is model.net
       loaded = pickle.loads(pickle.dumps(model))
       assert isinstance(loaded._lazy, _LazyTracker) and loaded._lazy.wrapped is loaded.net
       loaded.net(torch.zeros(1, 4))
       assert loaded.params.net == RuntimeModule.of(loaded.net) and loaded._lazy is None


   def test_a_pickle_without_the_later_lazy_attributes_still_refreshes():
       model = NNModel(module=_lazy(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
       state = model.__getstate__()
       for name in ("_lazy_places", "_lazy_stamp", "_lazy_wrapped"):  # absent from earlier pickles
           del state[name]
       restored = object.__new__(NNModel)
       restored.__setstate__(state)
       assert restored._lazy is not None and restored._lazy.places == {} and restored._lazy.wrapped is None
       restored.net(torch.zeros(1, 4))
       assert restored.params.net == RuntimeModule.of(restored.net) and restored._lazy is None


   def test_a_wrapped_module_is_fingerprinted_once(monkeypatch):
       params = NNModelParams(net=RuntimeModule.of(Classifier()), loss=Losses.CROSS_ENTROPY)
       calls = []
       original = RuntimeModule.of.__func__

       def counting(cls, module):
           calls.append(module)
           return original(cls, module)

       monkeypatch.setattr(RuntimeModule, "of", classmethod(counting))
       NNModel(module=Classifier(), params=params)
       assert len(calls) == 1
   ```

8. No test reads the `_lazy_*` attributes (`grep -rn "_lazy_wrap_keys\|_lazy_places\|_lazy_stamp\|_lazy_wrapped" tests` returns nothing), so none is rewritten. Behaviour and pickles are unchanged, so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `test_a_wrapped_module_is_fingerprinted_once` fails on `develop` before the change (two calls) and passes after it. Check: `uv run pytest tests/test_external_module_runtime.py::test_a_wrapped_module_is_fingerprinted_once`
- [ ] The two pickle tests pass. Check: `uv run pytest tests/test_external_module_runtime.py -k "older_nnx_reads or later_lazy_attributes"`
- [ ] No `_lazy_*` attribute is read or written outside `_LazyTracker` and the pickle hooks. Check: `grep -n "self\._lazy_\|model\._lazy_\|\"_lazy_wrap_keys\", None" src/nnx/nn/nn_model.py` returns 0 lines
- [ ] The #562 suites and the pickle suites pass unchanged, including `test_params_stays_in_the_instance_dict_for_pickles_across_versions` and DEC-010's `test_an_uninitialized_module_cannot_match_a_materialized_descriptor_without_weights`. Check: `uv run pytest tests/test_external_module_runtime.py tests/test_legacy_net_pickles.py tests/test_quantize_ptq.py tests/test_model_factories.py tests/test_compile_execution.py tests/test_experiment_plan.py`
- [ ] Copies still work. Check: `uv run pytest tests/test_quantize_ptq.py tests/test_external_module_runtime.py -k "deepcopy or pickle"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L3592-L3642](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L3592-L3642) | `_resolve_net_descriptor`: the second `RuntimeModule.of(module)` at L3615 |
| [nn_model.py L3888-L3906](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L3888-L3906) | `__init__`: the first fingerprint (`inexact`) and the call |
| [nn_model.py L3929-L3963](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L3929-L3963) | the five attributes set at the wrap |
| [nn_model.py L3983-L4010](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L3983-L4010) | the `params` property and its pickle comment |
| [nn_model.py L4012-L4066](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4012-L4066) | `_refresh_lazy_descriptor`: every `getattr` default |
| [nn_model.py L4563-L4572](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4563-L4572) | `from_checkpoint` reads `_lazy_wrap_keys` |
| [models.py L331-L370](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/models.py#L331-L370) | `_module_registrations`: the stamp |
| [test_external_module_runtime.py L1286-L1295](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_external_module_runtime.py#L1286-L1295) | the pin that `params` stays in the instance dict for pickles |
| #562 | the refresh-on-read design |
| #764 | DEC-010: the refresh default kept |

## Scope

**In:** `_LazyTracker`, the `NNModel` pickle hooks, the single fingerprint, three tests.

**Out:** the refresh-on-read itself (#562's design, kept). DEC-010's refusal of a fresh lazy module with a materialized run's params (kept). Moving the lazy-module descriptor helpers into `nnx.models` (FEAT-078). The dry-run inference rules (FEAT-094).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-078
**Related:** #562, #764
