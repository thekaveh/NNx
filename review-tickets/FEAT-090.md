---
id: FEAT-090
title: "[FEAT-090] Derive single vs named optimizers instead of passing trainer= and named= (part 3 of 3)"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-10"]
depends_on: ["FEAT-089"]
related_issues: [536, 215, 206, 502]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-090: whether a fit has `NNModel.train`'s one optimizer or a `Trainer`'s named ones is already encoded in the optimizer names (`""` versus non-empty names), yet six resume helpers take it again as a boolean (`named=` or `trainer=`), and `_Optimization` stores it as a field. Derive it from the names everywhere and drop the flags. Every refusal message stays word for word, including the `_owner` suffix (` for 'name'`). Part 3 of 3 of the SMELL-10 work.

## Context

**What happens today.** The suffix that names a Trainer's optimizer in a refusal is chosen by a flag:

```python
def _owner(name: str, *, named: bool) -> str:
    """The suffix naming a Trainer's optimizer in a refusal (none for
    ``NNModel.train``'s one)."""
    return f" for {name!r}" if named else ""
```

and the callers pass constants that the names already imply: `nn_model.py` L5403 `_check_resume_horizons(params, {"": params.scheduler}, named=False)`, `trainer.py` L599 `_check_resume_horizons(params, scheduler_params, named=True)`, `_Optimization(..., named=False)` (L5453–L5459) and `_Optimization(..., named=True)` (`trainer.py` L669–L675). `_resume_fit` forwards `optimization.named` as `trainer=` to `_saved_hyperparameters` (L998) and as `named=` to `_check_resume_hyperparameters` (L1002); `_saved_hyperparameters` passes `trainer=` on to `_source_hyperparameters` (L2972).

**Why it matters.** A flag that can disagree with the data it describes is a bug waiting to happen; the names are the single source of truth.

**History.** #536 shares the resume between the loops through `_Optimization`; #215 introduced the Trainer's named state; #502 added the hyperparameter check whose message names `optimizer 'head'` versus `optimizer`; #206 is the absent-state rule the resume keeps. The SMELL-10 cross-check requires the `_owner` messages to stay; they do. `NNTrainerParams` refuses an empty optimizer name ("names must be non-empty strings", `trainer/params.py` L182), so `name == ""` holds exactly for `NNModel.train`'s optimizer.

**Background for the worker.**
- *Single vs named*: `NNModel.train` keys its one optimizer and scheduler by `""`; a `Trainer` keys them by the user's names. A Trainer may have no schedulers at all, so the rule for a scheduler mapping is "named unless its keys are exactly `{""}`".
- *`_owner(name)`*: the refusal suffix: empty for `""`, ` for 'name'` otherwise.

**Proposal.** All edits are in `src/nnx/nn/nn_model.py` unless noted (line numbers at `9ad461e`; FEAT-089 shifts them, so find each function by name).

1. `_Optimization` (L765): delete the field `named: bool` (L775) and add, after the fields:
   ```python
       @property
       def named(self) -> bool:
           """A Trainer's named optimizers: any name but ``""`` (NNTrainerParams
           refuses an empty name)."""
           return set(self.optimizers) != {""}
   ```
   Change `owner` to `return _owner(name)`. `saved()`, `_resume_fit` (`if optimization.named:`) and `_load_resume_source(..., trainer=optimization.named)` keep reading `named`, now the property.
2. Remove the `named=` keyword from both constructions: `_Optimization(...)` in `NNModel._train_impl` (delete `named=False,`) and in `src/nnx/trainer/trainer.py` L669–L675 (delete `named=True,`).
3. `_owner`: change the signature to `def _owner(name: str) -> str:`, the body to `return f" for {name!r}" if name else ""`, and the docstring to `"""The suffix naming a Trainer's optimizer in a refusal: none for NNModel.train's one (named ``""``)."""`.
4. `_check_resume_horizons`: drop `*, named: bool` from the signature and call `_owner(name)` in its loop. Update the two callers: `_check_resume_horizons(params, {"": params.scheduler})` in `NNModel._train_impl` and `_check_resume_horizons(params, scheduler_params)` in `trainer.py` L599 (keep its end-of-line comment).
5. `_check_resume_hyperparameters`: drop `*, named: bool` and change the owner line to `owner = f"{component} {name!r}" if name else component`. In `_resume_fit`, delete the `named=optimization.named,` argument.
6. `_saved_hyperparameters`: drop `trainer: bool,` from its keyword parameters (keep `default_scheduler`), add `    named = set(schedulers) != {""}` as its first statement, change `if trainer:` to `if named:`, and end with `return _source_hyperparameters(run_id, schedulers, default_scheduler=default_scheduler)`. In `_resume_fit`, delete the `trainer=optimization.named,` argument.
7. `_source_hyperparameters`: drop `trainer: bool,` and change `if trainer:` to `if set(schedulers) != {""}:`.
8. `tests/test_resume_parity.py` L391–L397: delete `named=True,` from the `_Optimization(...)` call (its optimizers are named `"a"` and `"b"`).
9. Add to `tests/test_resume_parity.py`:
   ```python
   def test_single_or_named_is_derived_from_the_optimizer_names():
       """FEAT-090: no flag; "" is NNModel.train's optimizer, any other name a Trainer's."""
       net = _model().net
       optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
       one = nn_model._Optimization(optimizers={"": optimizer}, schedulers={}, factories={"": None}, hyperparameters={})
       named = nn_model._Optimization(optimizers={"head": optimizer}, schedulers={}, factories={"head": None}, hyperparameters={})
       assert not one.named and named.named
       assert one.owner("") == "" and named.owner("head") == " for 'head'"
       assert nn_model._owner("") == "" and nn_model._owner("head") == " for 'head'"
   ```
   (`nn_model` is `tests/test_resume_parity.py`'s module import of `nnx.nn.nn_model`, and `_model` its model builder.)

No test passes `named=` or `trainer=` to these helpers apart from step 8 (`grep -rn "named=\|_owner(\|_check_resume_horizons(\|_saved_hyperparameters(\|_source_hyperparameters(\|_check_resume_hyperparameters(" tests` lists only `tests/test_resume_parity.py` L396). `_load_resume_source(..., trainer=...)` and `_resume_training_state(..., trainer=...)` keep their `trainer=`: they say which loop is resuming, before an `_Optimization` exists, and `tests/test_model_recipes.py` L526–L528 call `_load_resume_source(..., trainer=False, ...)`. No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FEAT-089 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test `tests/test_resume_parity.py::test_single_or_named_is_derived_from_the_optimizer_names` fails before the change (`_Optimization` requires `named`) and passes after it. Check: `uv run pytest tests/test_resume_parity.py::test_single_or_named_is_derived_from_the_optimizer_names`
- [ ] No helper takes the flag any more. Check: `grep -nE "def (_owner|_check_resume_horizons|_check_resume_hyperparameters|_saved_hyperparameters|_source_hyperparameters)\(" -A3 src/nnx/nn/nn_model.py | grep -cE "named: bool|trainer: bool"` prints `0`, and `grep -rn "named=True\|named=False" src/nnx tests` prints nothing
- [ ] Every refusal keeps its text (the ` for 'name'` suffix included). Check: `uv run pytest tests -k "exact_resume or resume_parity or trainer or component_resume or objective_resume or resume_point or model_recipes"`
- [ ] FEAT-088's sidecar pin still passes. Check: `uv run pytest tests/test_checkpoint_sidecar_format.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L765-L779](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L765-L779) | `_Optimization` and its `named` field and `owner` |
| [nn_model.py L819-L835](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L819-L835) | `_owner(named=)` and `_check_resume_horizons(named=)` |
| [nn_model.py L994-L1003](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L994-L1003) | `_resume_fit` forwarding the flag twice |
| [nn_model.py L2880-L2972](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2880-L2972) | `_source_hyperparameters(trainer=)`, `_check_resume_hyperparameters(named=)`, `_saved_hyperparameters(trainer=)` |
| [nn_model.py L5403](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5403), [L5453-L5459](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L5453-L5459) | `NNModel.train`'s constant `named=False` |
| [trainer.py L599](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/trainer.py#L599), [L669-L675](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/trainer.py#L669-L675) | `Trainer.train`'s constant `named=True` |
| [params.py L180-L182](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/params.py#L180-L182) | Trainer optimizer names are non-empty |
| [test_resume_parity.py L391-L397](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_resume_parity.py#L391-L397) | The one test constructing `_Optimization(named=True)` |
| #536, #215, #206, #502 | Shared resume, named state, absent-state rule, hyperparameter check |

## Scope

**In:** `_Optimization.named`, `_owner`, `_check_resume_horizons`, `_check_resume_hyperparameters`, `_saved_hyperparameters`, `_source_hyperparameters`, their callers, one test edit, one test.

**Out:** `_load_resume_source` / `_resume_training_state`'s `trainer=` (the resuming loop's identity); the sidecar format; moving the resume code (FEAT-074).

## Dependencies

**Blocked by:** FEAT-089
**Blocks:** FEAT-074
**Related:** #536, #215, #206, #502
