---
id: FEAT-064
title: "[FEAT-064] Remove helpers that only tests use"
labels: ["enhancement", "area: packaging", "priority: P3", "review: 2026-11-27"]
sources: ["DEAD-03", "DEAD-11", "DEAD-12"]
depends_on: []
related_issues: [568]
status: READY
---

**Category:** Packaging · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); `_list_items_after` output checked without torch

## Summary

FEAT-064: three helpers survive only because a test calls them, or not even that: `nn_run._mean_field` (one test), `build_wiki._list_indent_after` (one test) and the test helper `_only_run_dir` (no caller). Rewrite the two tests onto the functions the code really uses, keeping every assertion they make, and delete the three helpers.

## Context

**What happens today.**

1. `nn_run._mean_field` wraps `_field_total`, which the chart code calls directly:
   ```python
   def _mean_field(records: list[Any], name: str) -> float:
       """The epoch mean of a per-batch field (:func:`_field_total`); NaN when
       no record has the field."""
       total, weight = _field_total(records, name)
       return total / weight if weight else float("nan")
   ```
   Its only caller is `tests/test_graph_classification.py::test_rendering_weights_batch_records_by_their_labeled_graph_count` (L282–L292).
2. `build_wiki._list_indent_after` is a wrapper whose every production caller moved to `_list_items_after`:
   ```python
   def _list_indent_after(text: str, active_list_indent: int = 0) -> int:
       """The deepest open list item's content indent after the lines of
       ``text``, from one item open at ``active_list_indent`` (0: none)."""
       open_items = _list_items_after(text, ((0, active_list_indent),) if active_list_indent else ())
       return open_items[-1][1] if open_items else 0
   ```
   Its only caller is `tests/test_wiki_projection.py::test_the_list_indent_after_a_nested_item_is_its_content_indent` (L833–L834).
3. `tests/test_compile_execution.py` L121–L123 defines `_only_run_dir()`, added in 83c7c3e and never called.

**Why it matters.** Code kept alive only for a test is dead weight, and a test of a wrapper stops testing what production runs.

**History.** #568 (FIX-100) added the nested-list indent rule; its check that a nested item's content indent is 6 must stay. The rewritten wiki assertion below keeps it: `_list_items_after("- Guides\n    - See [qs].\n", ())` returns `((0, 2), (0, 6))` at `9ad461e` (checked), so its last item's indent is 6.

**Background for the worker.**
- *`_field_total(records, name)`*: returns `(sum, weight)` of a per-batch field over an epoch's records, weighted by each record's labelled `count` when every record with the field has one. With no record carrying the field it returns `(0.0, 0)`: the running sum starts at `0.0` and the weight is the number of records that have the field.
- *Where the NaN comes from* (the question DEAD-03 leaves open): `_epoch_values(summary, losses, errors, val_edp, record)` turns a `(sum, weight)` pair into the plotted value with `total[0] / total[1] if total[1] else nan`. So a weight of 0 is what plots as NaN; `_mean_field` only repeated that rule.
- *`_list_items_after(text, open_items)`*: returns the open list items after `text` as `(quote_depth, content_indent)` pairs.

**Proposal.**

1. In `tests/test_graph_classification.py`, replace the body of `test_rendering_weights_batch_records_by_their_labeled_graph_count` (L283–L292) with:
   ```python
       from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
       from nnx.nn.params.nn_run import _epoch_values, _field_total

       records = [
           NNEvaluationDataPoint(loss=1.5, kind="categorical", count=2, status="ok"),
           NNEvaluationDataPoint(loss=3.0, kind="categorical", count=1, status="ok"),
       ]
       total, weight = _field_total(records, "loss")
       assert total / weight == pytest.approx(2.0)  # (1 + 2 + 3) / 3, not 2.25
       total, weight = _field_total([NNEvaluationDataPoint(loss=1.5), NNEvaluationDataPoint(loss=3.0)], "loss")
       assert total / weight == 2.25
       # No record has the field: weight 0, which the chart plots as NaN.
       missing = _field_total([NNEvaluationDataPoint()], "error")
       assert missing == (0.0, 0)
       assert math.isnan(_epoch_values(None, missing, missing, None, None)["train_err"])
   ```
2. Delete `_mean_field` from `src/nnx/nn/params/nn_run.py` (L85–L89 and the two blank lines after it).
3. In `tests/test_wiki_projection.py`, change the body of `test_the_list_indent_after_a_nested_item_is_its_content_indent` (L834) to:
   ```python
       assert build_wiki._list_items_after("- Guides\n    - See [qs].\n", ())[-1] == (0, 6)
   ```
4. Delete `_list_indent_after` from `scripts/docs/build_wiki.py` (L787–L791 and the two blank lines after it).
5. Delete `_only_run_dir` from `tests/test_compile_execution.py` (L121–L123 and the two blank lines after it). `os` stays imported; the file uses it elsewhere (L113, L150, L296).

No behaviour changes, so no CHANGELOG entry. `_mean_field` and `_list_indent_after` are private and appear in no docs.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The three helpers are gone. Check: `grep -rn "_mean_field\|_list_indent_after\|_only_run_dir" src scripts tests` prints nothing
- [ ] The rewritten graph test still asserts the weighted mean, the plain mean and the NaN for a missing field, and passes. Check: `uv run pytest tests/test_graph_classification.py::test_rendering_weights_batch_records_by_their_labeled_graph_count`
- [ ] The rewritten wiki test keeps #568's indent-6 check and passes. Check: `uv run pytest tests/test_wiki_projection.py -k list_indent`
- [ ] The compile tests still lint and pass. Check: `uv run ruff check tests/test_compile_execution.py` and `uv run pytest tests/test_compile_execution.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_run.py L70-L89](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L70-L89) | `_field_total` and `_mean_field` |
| [nn_run.py L371-L394](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L371-L394) | `_epoch_values`: weight 0 plots as NaN |
| [test_graph_classification.py L282-L292](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_graph_classification.py#L282-L292) | The only `_mean_field` caller |
| [build_wiki.py L779-L791](https://github.com/thekaveh/NNx/blob/9ad461e/scripts/docs/build_wiki.py#L779-L791) | `_list_items_after` and the wrapper `_list_indent_after` |
| [test_wiki_projection.py L833-L834](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_wiki_projection.py#L833-L834) | The only `_list_indent_after` caller |
| [test_compile_execution.py L121-L123](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_compile_execution.py#L121-L123) | `_only_run_dir`, never called |
| #568 | The nested-list indent rule the wiki test guards |

## Scope

**In:** the three deletions and the two test rewrites.

**Out:** the tombstone asserts for `pip-audit-py310-ignores.txt` (DEAD-13, kept: they guard #759's acceptance criterion); `bundles._save_tensors` (DEAD-10, resolved by FIX-121).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #568
