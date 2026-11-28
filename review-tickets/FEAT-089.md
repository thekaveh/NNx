---
id: FEAT-089
title: "[FEAT-089] Pass NNCheckpoint.save's training state as one value (part 2 of 3)"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-10"]
depends_on: ["FEAT-088"]
related_issues: [536, 215, 502]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-089: the training loops hand their optimizers and schedulers to `NNCheckpoint.save` as one of two parallel keyword families: eight singular keywords for `NNModel.train`'s one optimizer, eight name-keyed plural ones for a `Trainer`, each built by its own code path. Add one frozen dataclass, `_TrainingStateParts`, keyed by optimizer name (`""` for `NNModel.train`), that `_Optimization.save_state` builds for both loops and that turns itself into the right keyword family with one method. `NNCheckpoint.save`'s public signature and the sidecar it writes stay exactly the same (FEAT-088's pin proves it). Part 2 of 3 of the SMELL-10 work.

## Context

**What happens today.** `_Optimization.save_state` (`nn_model.py` L786–L803) branches into two builders:

```python
        if self.named:
            return _named_training_state(
                net, self.optimizers, self.schedulers, self.factories, self.hyperparameters, topologies
            )
        optimizer, scheduler = self.optimizers[""], self.schedulers[""]
        return {
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
```

`_named_training_state` (L225–L252) builds the plural family (`optimizers_state`, `optimizer_types`, `optimizer_topologies`, `optimizer_factories`, `optimizer_configs`, `scheduler_configs`, `schedulers_state`, `scheduler_types`). Both callers splat the result into `NNCheckpoint.save`: `_save_final_last` (L6003) and `_save_checkpoints` (L6670).

**Why it matters.** Two code paths encode one rule ("one value per optimizer, keyed by name"), so a field added to one family has to be added to the other by hand.

**History.** #215 (FEAT-005) added the name-keyed Trainer family; #502 added the configured hyperparameters (`*_config`, `*_configs`); #536 made both loops save through `_Optimization`. `NNCheckpoint.save` is public and listed in `docs/api.md` with its 30 keywords, and tests write checkpoints through it with singular keywords (`tests/_checkpoint_files.py`, `tests/test_train_integration.py`); its signature therefore stays, and only the loops' internal plumbing changes.

**Background for the worker.**
- *Singular vs plural family*: the sidecar stores `NNModel.train`'s optimizer under `optimizer`, `optimizer_type`, … and a Trainer's under `optimizers`, `optimizer_types`, … (name → value); the other family is `None`. `NNCheckpoint.save` writes whichever keywords it receives, in a fixed key order.
- *`_Optimization`*: the frozen dataclass both loops build (#536) with `optimizers`, `schedulers`, `factories`, `hyperparameters` (name → `{"optimizer": …, "scheduler": …}`) and `named`.
- *Equivalence*: for one unnamed optimizer, `{"": x}.get("")` is `x`, so building name-keyed mappings first and reading key `""` back gives exactly today's singular values.

**Proposal.**

1. In `src/nnx/nn/params/nn_checkpoint.py`, add above `class NNCheckpoint` (and import `Mapping` if needed; it is imported at L14):
   ```python
   @dataclass(frozen=True)
   class _TrainingStateParts:
       """A fit's optimizers and schedulers for one checkpoint, keyed by optimizer
       name: ``""`` for NNModel.train's one optimizer, a Trainer's names
       otherwise (SMELL-10). :meth:`save_keywords` gives the
       :meth:`NNCheckpoint.save` keywords of the on-disk sidecar format."""

       optimizers: Mapping[str, Any]
       schedulers: Mapping[str, Any]
       optimizer_types: Mapping[str, Any]
       scheduler_types: Mapping[str, Any]
       topologies: Mapping[str, Any]
       factories: Mapping[str, Any]
       optimizer_configs: Optional[Mapping[str, Any]] = None
       scheduler_configs: Optional[Mapping[str, Any]] = None

       @property
       def single(self) -> bool:
           """NNModel.train's one unnamed optimizer (key ``""``)."""
           return set(self.optimizers) == {""}

       def save_keywords(self) -> dict[str, Any]:
           """The singular ``optimizer_*`` / ``scheduler_*`` keywords for one
           unnamed optimizer, the name-keyed plural ones otherwise."""
           if self.single:
               optimizer_configs = self.optimizer_configs or {}
               scheduler_configs = self.scheduler_configs or {}
               return {
                   "optimizer_state": self.optimizers[""],
                   "scheduler_state": self.schedulers[""],
                   "optimizer_type": self.optimizer_types[""],
                   "scheduler_type": self.scheduler_types[""],
                   "optimizer_topology": self.topologies[""],
                   "optimizer_factory": self.factories[""],
                   "optimizer_config": optimizer_configs.get(""),
                   "scheduler_config": scheduler_configs.get(""),
               }
           return {
               "optimizers_state": dict(self.optimizers),
               "optimizer_types": dict(self.optimizer_types),
               "optimizer_topologies": dict(self.topologies),
               "optimizer_factories": dict(self.factories),
               "optimizer_configs": None if self.optimizer_configs is None else dict(self.optimizer_configs),
               "scheduler_configs": None if self.scheduler_configs is None else dict(self.scheduler_configs),
               "schedulers_state": dict(self.schedulers),
               "scheduler_types": dict(self.scheduler_types),
           }
   ```
   `NNCheckpoint.save` itself does not change.
2. In `src/nnx/nn/nn_model.py`:
   - import `_TrainingStateParts` from `.params.nn_checkpoint` (add it to the existing import list, L106–L114);
   - replace the body of `_Optimization.save_state` (after its docstring) with one builder for both loops, and change its return annotation to `_TrainingStateParts`:
     ```python
             topologies = self.topologies(net) if topologies is None else topologies
             configured = self.hyperparameters
             return _TrainingStateParts(
                 optimizers={name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
                 schedulers={name: scheduler.state_dict() for name, scheduler in self.schedulers.items()},
                 optimizer_types={name: _component_type(optimizer) for name, optimizer in self.optimizers.items()},
                 scheduler_types={name: _component_type(scheduler) for name, scheduler in self.schedulers.items()},
                 topologies=dict(topologies),
                 factories={name: self.factories.get(name) for name in self.optimizers},
                 optimizer_configs={n: configured[n]["optimizer"] for n in self.optimizers if n in configured} or None,
                 scheduler_configs={n: configured[n]["scheduler"] for n in self.optimizers if n in configured} or None,
             )
     ```
     and change its docstring to `"""The optimizer and scheduler states, types, parameter ``topologies`` (now, unless given), factories and configured hyperparameters (#502), keyed by optimizer name, for ``NNCheckpoint.save`` (:meth:`_TrainingStateParts.save_keywords`)."""`;
   - delete `_named_training_state` (L225–L252); nothing else calls it (`grep -rn "_named_training_state" src tests` lists only its definition and the call removed above);
   - `_save_final_last` (L6003): `**optimization.save_state(self.net, topologies),` → `**optimization.save_state(self.net, topologies).save_keywords(),`;
   - `_save_checkpoints` (L6670): `**(optimization.save_state(self.net) if optimization is not None else {}),` → `**(optimization.save_state(self.net).save_keywords() if optimization is not None else {}),`.
   `_component_type` stays where it is (FEAT-103 may replace it with `models._qualified`; use whichever name exists).
3. Retarget `tests/test_resume_parity.py::test_a_named_save_walks_each_optimizer_topology_once`: L408's `state["optimizer_topologies"]` → `state.topologies`, and L412's `optimization.save_state(net, given)["optimizer_topologies"]` → `optimization.save_state(net, given).topologies`. (L383 reads a loaded sidecar dict and stays.) Its `_optimizer_topology` patch (L405) still intercepts: `_Optimization.topologies` calls it.
4. Add to `tests/test_checkpoint_sidecar_format.py`:
   ```python
   def test_training_state_parts_give_the_singular_family_for_one_optimizer_and_the_plural_one_otherwise():
       from nnx.nn.params.nn_checkpoint import _TrainingStateParts

       one = _TrainingStateParts({"": "o"}, {"": "s"}, {"": "ot"}, {"": "st"}, {"": "t"}, {"": None}, {"": "oc"}, {"": "sc"})
       assert one.single and one.save_keywords() == {
           "optimizer_state": "o", "scheduler_state": "s", "optimizer_type": "ot", "scheduler_type": "st",
           "optimizer_topology": "t", "optimizer_factory": None, "optimizer_config": "oc", "scheduler_config": "sc",
       }
       named = _TrainingStateParts({"a": "o"}, {}, {"a": "ot"}, {}, {"a": "t"}, {"a": None})
       assert not named.single and named.save_keywords() == {
           "optimizers_state": {"a": "o"}, "optimizer_types": {"a": "ot"}, "optimizer_topologies": {"a": "t"},
           "optimizer_factories": {"a": None}, "optimizer_configs": None, "scheduler_configs": None,
           "schedulers_state": {}, "scheduler_types": {},
       }
   ```

No behaviour changes, so no CHANGELOG entry; `NNCheckpoint.save`'s signature and docstring are unchanged, so `docs/api.md` needs no rebuild.

**Worker setup.** Branch from `develop` after FEAT-088 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] FEAT-088's sidecar pin passes unchanged. Check: `uv run pytest tests/test_checkpoint_sidecar_format.py`
- [ ] The new parts test passes (it fails on `develop`, where `_TrainingStateParts` does not exist). Check: `uv run pytest tests/test_checkpoint_sidecar_format.py::test_training_state_parts_give_the_singular_family_for_one_optimizer_and_the_plural_one_otherwise`
- [ ] One builder is left. Check: `grep -n "def _named_training_state\|if self.named:" src/nnx/nn/nn_model.py` shows no `_named_training_state` and no `if self.named:` inside `save_state`
- [ ] `NNCheckpoint.save` is unchanged. Check: `git diff origin/develop -- src/nnx/nn/params/nn_checkpoint.py | grep -c "^[-+].*def save("` prints `0`, and `uv run python -m scripts.docs.build_api_reference --check` passes
- [ ] Resume, trainer and checkpoint behaviour is unchanged. Check: `uv run pytest tests -k "exact_resume or resume_parity or trainer or component_resume or objective_resume or resume_point"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L225-L252](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L225-L252) | `_named_training_state`, the plural builder |
| [nn_model.py L765-L816](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L765-L816) | `_Optimization` and its two-branch `save_state` |
| [nn_model.py L5994-L6014](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5994-L6014), [L6669-L6679](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L6669-L6679) | The two callers that splat the keywords into `save` |
| [nn_checkpoint.py L684-L713](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L684-L713) | `NNCheckpoint.save`'s public keywords, which stay |
| [_checkpoint_files.py L20-L50](https://github.com/thekaveh/NNx/blob/9ad461e/tests/_checkpoint_files.py#L20-L50) | Tests that write through the public keywords |
| [test_resume_parity.py L386-L414](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_resume_parity.py#L386-L414) | The test reading `save_state`'s result |
| #215, #502, #536 | Named state, configured hyperparameters, the shared save |

## Scope

**In:** `_TrainingStateParts`, `_Optimization.save_state`, the two callers, one test retarget, one test.

**Out:**
- Shrinking `NNCheckpoint.save`'s public keyword list (a public API change that needs a deprecation; no ticket).
- The read side (`_Optimization.saved`) and the `named=` / `trainer=` flags: FEAT-090.
- Any change to the sidecar format.

## Dependencies

**Blocked by:** FEAT-088
**Blocks:** FEAT-090, FEAT-074
**Related:** #536, #215, #502
