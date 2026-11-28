---
id: FEAT-133
title: "[FEAT-133] Share freeze and unfreeze's pattern loop"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-57"]
depends_on: []
related_issues: [224, 554]
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-133: `nnx.finetune.freeze` and `unfreeze` are two copies of one loop that differ only in the flag they set and the function name in their refusal. This ticket moves the loop into `_set_requires_grad(module, patterns, flag, fn)` in `src/nnx/finetune/freezing.py`; both public functions become one-line calls. Behaviour, return values and messages are unchanged.

## Context

**What happens today.** `src/nnx/finetune/freezing.py`:

```python
    if not patterns:
        raise ValueError("freeze() requires at least one pattern; pass '*' to freeze every parameter")
    n = 0
    for name, param in module.named_parameters():
        if any(fnmatch.fnmatchcase(name, p) for p in patterns):
            if param.requires_grad:
                n += 1
            param.requires_grad = False
    return n
```

`unfreeze` (L58-L69) is the same with `unfreeze()`, `not param.requires_grad` and `True`.

**Why it matters.** The two must stay mirror images; one loop guarantees it.

**History.** SMELL-57 also asks for prune pattern targeting on every registration path and a validated `sparsity`; those change behaviour and are FIX-149 (which keeps the `layer_pattern` keyword #554's message names). #224 introduced every-path targeting for PEFT; `freeze` keeps matching `named_parameters()` names, as documented.

**Background for the worker.**
- `fnmatch.fnmatchcase(name, pattern)`: case-sensitive shell glob; `*` matches dots too.
- The return value counts parameters whose `requires_grad` actually changed.

**Proposal.**
1. In `src/nnx/finetune/freezing.py`, after the imports, add:

   ```python
   def _set_requires_grad(module: nn.Module, patterns: tuple[str, ...], flag: bool, fn: str) -> int:
       """Set ``requires_grad=flag`` on every parameter of ``module`` whose
       dotted name matches any fnmatch glob in ``patterns``; return how many
       changed. ``fn`` (``"freeze"`` / ``"unfreeze"``) names the caller in the
       refusal of an empty ``patterns``."""
       if not patterns:
           raise ValueError(f"{fn}() requires at least one pattern; pass '*' to {fn} every parameter")
       n = 0
       for name, param in module.named_parameters():
           if any(fnmatch.fnmatchcase(name, p) for p in patterns):
               if param.requires_grad != flag:
                   n += 1
               param.requires_grad = flag
       return n
   ```

2. Replace the body of `freeze` after its docstring (L47-L55) with `return _set_requires_grad(module, patterns, False, "freeze")`, and the body of `unfreeze` after its docstring (L61-L69) with `return _set_requires_grad(module, patterns, True, "unfreeze")`. Keep both docstrings.
3. Run `uv run ruff format src/nnx/finetune/freezing.py`.

The refusal texts come out as today: `freeze() requires at least one pattern; pass '*' to freeze every parameter` and `unfreeze() requires at least one pattern; pass '*' to unfreeze every parameter`. No test changes, no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] One loop. Check: `grep -c "for name, param in module.named_parameters()" src/nnx/finetune/freezing.py` returns 2 (the helper and `frozen`)
- [ ] A new test `tests/test_finetune_freezing.py::test_unfreeze_requires_at_least_one_pattern` asserts `unfreeze(nn.Linear(2, 2))` raises `ValueError` matching `r"unfreeze\(\) requires at least one pattern; pass '\*' to unfreeze every parameter"`. Check: `uv run pytest tests/test_finetune_freezing.py::test_unfreeze_requires_at_least_one_pattern`
- [ ] The freezing suite passes unchanged, including the counts and `NNModel.freeze` delegation. Check: `uv run pytest tests/test_finetune_freezing.py tests -k "freeze or frozen"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [freezing.py L25-L55](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/finetune/freezing.py#L25-L55) | `freeze` |
| [freezing.py L58-L69](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/finetune/freezing.py#L58-L69) | `unfreeze`: the mirrored copy |
| [test_finetune_freezing.py L60](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_finetune_freezing.py#L60) | `freeze`'s empty-pattern refusal test |
| #224, #554 | every-path PEFT targeting; the `layer_pattern` keyword |

## Scope

**In:** `_set_requires_grad` and the two public bodies.

**Out:** prune patterns on every registration path and `sparsity` validation (FIX-149). Varargs patterns for the prune functions (FIX-149). `freeze` matching every registration path (a behaviour change; not ticketed).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #224, #554
