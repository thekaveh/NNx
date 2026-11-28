---
id: FEAT-141
title: "[FEAT-141] Add docstrings to the nnx.decisions members api.md lists without one"
labels: ["enhancement", "area: documentation", "priority: P3", "review: 2026-11-27"]
sources: ["DOC-09"]
depends_on: ["DEC-016"]
related_issues: [219, 521, 243]
status: READY
---

**Category:** Documentation · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); `docs/api.md` counted with the command in the acceptance criteria

## Summary

FEAT-141: `docs/api.md` shows "No public description is currently available." for 30 `nnx.decisions` members (367 across the whole reference). One of them, `BooleanResult.p_false`, is documented by DEC-016. This ticket writes a one-line docstring, derived from the code, for the other 29 and regenerates the API reference.

## Context

**What happens today.** For example, in `docs/api.md`:

````text
##### `nnx.decisions.Question.state`

```python
nnx.decisions.Question.state(self) -> 'dict[str, Any]'
```

No public description is currently available.
````

The generator (`scripts/docs/build_api_reference.py` L63) prints that line for a public method or property whose `inspect.getdoc` is empty.

**Why it matters.** `nnx.decisions` is documented as a public typed-decision API (#219); a reader of the reference cannot tell what `state()`, `probabilities()` or `capabilities()` return.

**History.** DOC-09 item 4 asks to document the most-used undocumented public members; this ticket takes the `nnx.decisions` slice. #219 (FEAT-009) and #243 (FEAT-021) defined these classes; #521 set the record-key rules the `from_state` text refers to. DOC-09 items 1–3 (`BooleanResult`'s class docstring and `p_false`) belong to DEC-016, which lands first because it may change `Record`.

**Background for the worker.**
- `docs/api.md` is generated: edit the docstrings, then run `uv run python -m scripts.docs.build_api_reference`. Never edit `docs/api.md` by hand.
- Each docstring below is one line (or one short paragraph), written as the first statement of the function body; a property's docstring goes in its getter. Sphinx roles (`:class:`, `:meth:`, `:func:`) render as code spans.
- The texts describe what the code at `9ad461e` returns; they are worded so DEC-016, FIX-105 and FEAT-126 do not make them false.

**Proposal.** Add exactly these docstrings.

1. `src/nnx/decisions/schema.py`
   - `Question.state` (L142): `"""The question as plain JSON data — its ``kind``, ``prompt`` and options in order — which :meth:`digest` hashes and :func:`question_from_state` rebuilds."""`
   - `Question.option_ids` (L161): `"""The option ids in the question's order (none for a :class:`Boolean`)."""`
   - `Choice.option_ids` (L178): `"""Each option's ``id``, in the question's order."""`
   - `Choice.state` (L181): `"""``{"kind": "choice", "prompt": ..., "options": [[id, description], ...]}``, the options in order."""`
   - `Boolean.state` (L202): `"""``{"kind": "boolean", "prompt": ...}``."""`
   - `Score.option_ids` (L224): `"""Each level's ``id``, lowest level first."""`
   - `Score.state` (L227): `"""``{"kind": "score", "prompt": ..., "levels": [[id, description], ...]}``, lowest level first."""`
   - `ChoiceResult.probabilities` (L337): `"""``{option id: probability}``, in the question's option order."""`
   - `ScoreResult.probabilities` (L398): `"""``{level id: probability}``, lowest level first."""`
   - `ScoreResult.expected_index` (L402): `"""``sum(i * p_i)`` over the zero-based levels: an ordinal position between levels, not an interval-scale score."""`
2. `src/nnx/decisions/providers.py`
   - `DecisionProvider.capabilities` (L173): change the one-line `def capabilities(self) -> Capabilities: ...` to a body of `"""What this provider declares it can answer (:class:`Capabilities`), checked before any model call or network I/O."""` followed by `...` on its own line.
   - `DecisionProvider.decide` (L175): the same change, with `"""Answer ``question`` for every row of ``inputs``: one validated result per input, in input order."""`
   - `FixedHeadProvider.capabilities` (L396): `"""Boolean for a Bernoulli head, Choice (and Score when ``ordinal=True``) for a categorical one, over tensor inputs, with the head's ``labels`` as the fixed label space and ``max_batch`` as given; training and export are available (the wrapped ``NNModel`` trains and exports)."""`
3. `src/nnx/decisions/nli.py`
   - `NLIProvider.capabilities` (L260): `"""Choice and Boolean over text, with the candidates supplied per request (``dynamic_labels``) and ``max_batch`` as given; inference only (the caller's model is used as is)."""`
4. `src/nnx/decisions/jev.py`
   - `JevProvider.capabilities` (L369): `"""Choice, Boolean and Score over text, with the options supplied per request and ``max_batch`` as given; inference only (a hosted model)."""`
5. `src/nnx/decisions/jobs.py`
   - `QuestionOutcome.abstained` (L347): `"""Whether any row of the question abstained under its policy (always ``False`` without a policy or without rows)."""`
6. `src/nnx/decisions/benchmark.py`
   - `Sample.digest` (L190): `"""The question's :meth:`~nnx.decisions.Question.digest`: the schema digest a record for this sample must carry."""`
   - `Sample.true_label` (L199): `"""The label as an option id: a Boolean label's ``True`` / ``False`` as ``"true"`` / ``"false"``, any other label as given."""`
   - `Record.state` (L511): `"""The record as plain JSON (``nnx.decision-record/1``), as :func:`write_records` writes it: ``distribution`` as ``[id, probability]`` pairs and ``execution`` as plain lists and dicts."""`
   - `Record.from_state` (L527): `"""Rebuild a record from :meth:`state` (one line of a :func:`read_records` file); a mapping that is not a ``nnx.decision-record/1`` record, or whose keys or fields do not validate, raises :class:`BenchmarkError`."""`
   - `MetricValue.unit` (L845): `"""The metric's unit (for example ``"nats per row"`` for ``nll``)."""`
   - `MetricValue.state` (L848): `"""``value`` (a non-finite one as the string ``"Infinity"``, ``"-Infinity"`` or ``"NaN"``), ``denominator``, ``reason`` and ``unit``, plus ``note`` only when it is set."""`
   - `Coverage.state` (L959): `"""Every count, and ``reasons`` sorted by reason, as plain JSON."""`
   - `SliceReport.state` (L986): `"""The slice as plain JSON: its coverage, every metric in the report's metric order and the reliability bins."""`
   - `Resources.state` (L1031): `"""Every field as plain JSON, ``None`` where it was not declared."""`
   - `BenchmarkReport.state` (L1478): `"""The report as plain JSON (``nnx.decision-benchmark/1``): the form :meth:`to_json` writes, :meth:`save` stores and :func:`compare_reports` compares."""`
   - `BenchmarkReport.to_json` (L1495): `""":meth:`state` as strict JSON text (sorted keys, two-space indent, no NaN), ending in a newline."""`
   - `BenchmarkReport.to_csv` (L1531): `""":meth:`csv_rows` as CSV text with a header row."""`
   - `BenchmarkReport.text` (L1541): `"""A plain-text summary: a header line, then per slice its coverage counts, failure reasons and every metric with its denominator and unit (``unavailable`` with the reason when it has no value), then the declared resources."""`
7. Wrap each docstring so no line passes 120 characters (the project's line length), keeping the text word for word.
8. Regenerate the API reference: `uv run python -m scripts.docs.build_api_reference`.

No behaviour changes, no CHANGELOG entry.

**Worker setup.** No torch needed to edit; `uv sync --frozen --all-extras` is needed to regenerate `docs/api.md` and run the checks.

## Acceptance criteria

- [ ] No `nnx.decisions` entry in `docs/api.md` lacks a description. Check: `awk '/^####/{h=$0} /No public description is currently available/{print h}' docs/api.md | grep -c "nnx.decisions"` prints `0` (30 on `develop`; DEC-016 removes the `p_false` one)
- [ ] The generated reference is current. Check: `uv run python -m scripts.docs.build_api_reference --check && uv run python -m scripts.docs.build_docs --check`
- [ ] The docstring snippets in the docs still run. Check: `uv run pytest tests/test_doc_snippets.py tests/test_docs_projection.py`
- [ ] The decision suites pass. Check: `uv run pytest tests -k decision`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [api.md L3010](https://github.com/thekaveh/NNx/blob/9ad461e/docs/api.md#L3010) | `Question.state`: the first of the 30 placeholders |
| [api.md L4193-L4486](https://github.com/thekaveh/NNx/blob/9ad461e/docs/api.md#L4193-L4486) | the benchmark members' placeholders |
| [build_api_reference.py L63](https://github.com/thekaveh/NNx/blob/9ad461e/scripts/docs/build_api_reference.py#L63) | where the placeholder comes from |
| [schema.py L128-L235](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/schema.py#L128-L235) | the question classes |
| [schema.py L323-L404](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/schema.py#L323-L404) | the result classes |
| [providers.py L168-L175](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/providers.py#L168-L175), [L396-L410](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/providers.py#L396-L410) | `DecisionProvider`, `FixedHeadProvider.capabilities` |
| [nli.py L260-L270](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/nli.py#L260-L270), [jev.py L357-L370](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jev.py#L357-L370) | the other two `capabilities` |
| [jobs.py L346-L348](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L346-L348) | `QuestionOutcome.abstained` |
| [benchmark.py L189-L202](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L189-L202), [L511-L547](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L511-L547), [L844-L866](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L844-L866), [L959-L1040](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L959-L1040), [L1478-L1562](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1478-L1562) | the benchmark members |
| #219, #243, #521 | the decision API, the benchmark, the record-key rules |

## Scope

**In:** the 29 docstrings and the regenerated `docs/api.md`.

**Out:** `BooleanResult`'s class docstring and `p_false` (DEC-016). The other 337 placeholders outside `nnx.decisions` (not ticketed). The cross-references between the `MetricValue` types (FEAT-128).

## Dependencies

**Blocked by:** DEC-016
**Blocks:** FEAT-121
**Related:** #219, #521, #243
