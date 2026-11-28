---
id: FEAT-132
title: "[FEAT-132] Share DPO's empty-row lookup"
labels: ["enhancement", "area: training", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-56"]
depends_on: ["FIX-131"]
related_issues: []
status: READY
---

**Category:** Training · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-132: `_require_response_tokens` and `_require_prompt_tokens` in `src/nnx/paradigms/dpo.py` repeat the same eight lines that find the first batch row with no valid token and word the reason. This ticket moves those lines into `_first_empty_row(ids, pad_token_id) -> Optional[tuple[int, str]]`; each function keeps its own message.

## Context

**What happens today.** Both functions (L252-L268 and L271-L287) contain:

```python
    if pad_token_id is None:
        counts = response_ids.new_full((response_ids.shape[0],), response_ids.shape[1])
    else:
        counts = (response_ids != pad_token_id).sum(dim=-1)
    empty = (counts == 0).nonzero(as_tuple=False)
    if empty.numel():
        row = int(empty[0, 0])
        reason = "it is empty" if pad_token_id is None else f"every position is the pad id {pad_token_id}"
```

(with `prompt_ids` in the second), then raise their own `ValueError`.

**Why it matters.** The rule "a row with no valid token" is the refusal FIX-131 documents; one copy keeps the response and prompt checks from drifting.

**History.** No board issue covers these helpers. FIX-131 documents the refusal they raise; it lands first so this refactor does not collide with its docstring edits.

**Background for the worker.**
- Valid token: a position whose id is not `pad_token_id`. Without a `pad_token_id` every position counts, so only a zero-width tensor (`shape[1] == 0`) has a row with no valid token.
- **What the `str` holds:** the reason text that goes inside the parentheses of each refusal, exactly as today: `"it is empty"` when `pad_token_id` is `None`, else `"every position is the pad id <pad_token_id>"` (for example `"every position is the pad id 1"`). The `int` is the first such row's index in the batch.

**Proposal.**
1. In `src/nnx/paradigms/dpo.py`, directly above `_require_response_tokens` (L252), add:

   ```python
   def _first_empty_row(ids: torch.Tensor, pad_token_id: Optional[int]) -> Optional[tuple[int, str]]:
       """The first batch row of ``ids`` that holds no valid token, with the
       reason as the refusals word it — ``"it is empty"`` without a
       ``pad_token_id`` (every position counts, so only a zero-width batch
       qualifies), else ``"every position is the pad id <pad_token_id>"`` —
       or ``None`` when every row holds one."""
       if pad_token_id is None:
           counts = ids.new_full((ids.shape[0],), ids.shape[1])
       else:
           counts = (ids != pad_token_id).sum(dim=-1)
       empty = (counts == 0).nonzero(as_tuple=False)
       if not empty.numel():
           return None
       reason = "it is empty" if pad_token_id is None else f"every position is the pad id {pad_token_id}"
       return int(empty[0, 0]), reason
   ```

2. Replace the body of `_require_response_tokens` after its docstring with:

   ```python
       found = _first_empty_row(response_ids, pad_token_id)
       if found is not None:
           row, reason = found
           raise ValueError(
               f"DPO {name} response of batch row {row} has no valid tokens ({reason}): its log-prob would be 0 "
               "with no gradient yet still count as a preference pair. Drop that row from the dataset."
           )
   ```

3. Replace the body of `_require_prompt_tokens` after its docstring with:

   ```python
       found = _first_empty_row(prompt_ids, pad_token_id)
       if found is not None:
           row, reason = found
           raise ValueError(
               f"DPO prompt of batch row {row} has no valid tokens ({reason}): with prompt_packing the response "
               "is placed right after the prompt, so it needs a prompt token to be predicted from. "
               "Drop that row from the dataset."
           )
   ```

   Both message texts are copied unchanged. Keep both docstrings (as FIX-131 left them).
4. Run `uv run ruff format src/nnx/paradigms/dpo.py`.

No message or behaviour changes, no test changes, no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `_first_empty_row` exists and the counting appears once. Check: `grep -c "nonzero(as_tuple=False)" src/nnx/paradigms/dpo.py` returns 1
- [ ] A new test `tests/test_paradigms_dpo.py::test_first_empty_row_names_the_row_and_the_reason` asserts `_first_empty_row(torch.tensor([[2, 3], [1, 1]]), 1) == (1, "every position is the pad id 1")`, `_first_empty_row(torch.tensor([[2, 3]]), 1) is None` and `_first_empty_row(torch.zeros(2, 0, dtype=torch.long), None) == (0, "it is empty")` (it imports `from nnx.paradigms.dpo import _first_empty_row` in its body). Check: `uv run pytest tests/test_paradigms_dpo.py::test_first_empty_row_names_the_row_and_the_reason`
- [ ] The refusal tests pass unchanged. Check: `uv run pytest tests/test_paradigms_dpo.py::test_a_fully_padded_response_is_refused_before_any_update tests/test_paradigms_dpo.py::test_prompt_packing_refuses_a_prompt_with_no_valid_token`
- [ ] The DPO suite passes. Check: `uv run pytest tests/test_paradigms_dpo.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [dpo.py L252-L268](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/dpo.py#L252-L268) | `_require_response_tokens` |
| [dpo.py L271-L287](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/dpo.py#L271-L287) | `_require_prompt_tokens`: the same eight lines |
| [dpo.py L151-L154](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/dpo.py#L151-L154) | the call sites (unchanged) |
| [test_paradigms_dpo.py L709](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_paradigms_dpo.py#L709), [L842](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_paradigms_dpo.py#L842) | the pinned refusal texts |

## Scope

**In:** `_first_empty_row` and the two callers.

**Out:** documenting the refusal (FIX-131). Any change to what is refused.

## Dependencies

**Blocked by:** FIX-131
**Blocks:** none
**Related:** none
