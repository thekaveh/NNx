---
id: FEAT-128
title: "[FEAT-128] Cross-reference duplicate digest forms and value types in docstrings"
labels: ["enhancement", "area: documentation", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-51", "SMELL-59"]
depends_on: []
related_issues: [557, 521]
status: READY
---

**Category:** Documentation · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-128: four persisted digests use canonical-JSON forms other than the shared `_config._canonical_json` + `sha256:` form, with nothing saying so; and three "value, or unavailable with a reason" types exist, two of them public and both named `MetricValue`. Changing either would break saved data or the public API, so this ticket only documents them: a comment at each odd digest, a rule in `_artifacts.py`, and cross-references between the three value types. No code changes.

## Context

**What happens today.** The odd digest forms:

```python
    def digest(self) -> str:
        text = json.dumps(self.state(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
```

(`LinkSplit.digest`, `src/nnx/link_tasks.py` L435-L437: compact, ASCII-escaped, prefixed)

```python
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.state(), sort_keys=True).encode("utf-8")).hexdigest()
```

(`SearchSpace.digest`, `src/nnx/search.py` L276-L277: default separators, ASCII-escaped, bare hex). `Question.digest` (`src/nnx/decisions/schema.py` L145-L152) is compact with Unicode kept and bare hex; the benchmark's `_identity` (`src/nnx/decisions/benchmark.py` L628-L636) is compact, ASCII-escaped and prefixed. `JsonArtifact` (calibrators, split manifests, standardizers through `provenance.canonical_bytes`) uses `_config._canonical_json` and `sha256:`.

The value types: `link_tasks.MetricValue(value, reason)`, `decisions.benchmark.MetricValue(name, value, denominator, reason, note)` and `paradigms.offline_distillation.Measure(value, denominator, reason)`, each documented alone.

**Why it matters.** Every one of these digests is saved: a `LinkTask`'s checkpointed state names its split by `LinkSplit.digest()` (L967), a study names its space by `SearchSpace.digest()` (`search.py` L576), records carry `Question.digest()` as `question_digest` and `_identity` as `prompt_identity`. A well-meant "fix" to one form would orphan saved data. The two `MetricValue`s invite a wrong import.

**History.** #557 (FIX-089) made the decision digests refuse unencodable text and fixed what a derived id hashes; #521 (FIX-054) moved splits and standardizers onto the shared artifact checks. Both keep their formats; this ticket records why the remaining forms stay.

**Background for the worker.**
- Canonical JSON: a deterministic `json.dumps` (sorted keys, fixed separators, fixed escaping) whose bytes are hashed.
- `_config._canonical_json` (`src/nnx/_config.py` L110): sorted keys, compact separators, Unicode kept, no NaN / ±inf. `JsonArtifact` (`src/nnx/_artifacts.py` L135) hashes it with a `sha256:` prefix.

**Proposal.** Comments and docstrings only.

1. `src/nnx/link_tasks.py`, `LinkSplit.digest`: insert above `text = json.dumps(...)` (L436):

   ```python
           # Saved (a LinkTask's checkpointed state names the split by it): compact separators,
           # ASCII-escaped, "sha256:" prefix. Never change the form; new artifacts use
           # nnx._config._canonical_json (see nnx._artifacts).
   ```

2. `src/nnx/search.py`, `SearchSpace.digest`: insert above the `return` (L277):

   ```python
           # Saved (a study's records name the space by it): json.dumps's default separators,
           # ASCII-escaped, bare hex. Never change the form; new artifacts use
           # nnx._config._canonical_json and a "sha256:" prefix (see nnx._artifacts).
   ```

3. `src/nnx/decisions/schema.py`, `Question.digest`: insert above `canonical = json.dumps(...)` (L150):

   ```python
               # Saved (records carry it as question_digest): compact separators, Unicode kept,
               # bare hex. Never change the form; new artifacts use nnx._config._canonical_json
               # and a "sha256:" prefix (see nnx._artifacts).
   ```

4. `src/nnx/decisions/benchmark.py`, `_identity`: insert above its `try:` (L632), at the function body's indent:

   ```python
       # Saved as a record's prompt_identity: compact separators, ASCII-escaped, "sha256:"
       # prefix. Never change the form; new artifacts use nnx._config._canonical_json
       # (see nnx._artifacts).
   ```

5. `src/nnx/_artifacts.py`: append this paragraph to the module docstring, before its closing `"""`:

   ```
   A new artifact type subclasses :class:`JsonArtifact`: its digest is
   ``sha256:`` plus the SHA-256 of ``_config._canonical_json(state())``
   (sorted keys, compact separators, Unicode kept). Older digests use other
   canonical forms — ``LinkSplit``, ``SearchSpace``, ``Question`` and the
   benchmark's provider identity — and are saved in checkpoints and records,
   so they keep their forms.
   ```

6. `src/nnx/link_tasks.py`, `MetricValue` (L596-L597): replace the docstring with:

   ```python
       """A metric's value, or ``None`` with the reason it is unavailable.

       Not :class:`nnx.decisions.benchmark.MetricValue` (which also carries the
       metric's name, denominator and a note) nor
       :class:`nnx.paradigms.offline_distillation.Measure` (value, denominator,
       reason): three separate public types."""
   ```

7. `src/nnx/decisions/benchmark.py`, `MetricValue` (L834-L836): replace the docstring with:

   ```python
       """A metric's ``value`` over ``denominator`` rows, or ``None`` with the
       ``reason`` it is unavailable. ``note`` qualifies an available value
       (rows it does not count, say).

       Not :class:`nnx.link_tasks.MetricValue` (value and reason only) nor
       :class:`nnx.paradigms.offline_distillation.Measure` (no name or note)."""
   ```

8. `src/nnx/paradigms/offline_distillation.py`, `Measure` (L641-L642): replace the docstring with:

   ```python
       """One reported quantity with its denominator, or ``value=None`` and the
       ``reason`` it is unavailable.

       :class:`nnx.decisions.benchmark.MetricValue` adds a name and a note;
       :class:`nnx.link_tasks.MetricValue` has no denominator."""
   ```

9. `src/nnx/_metrics.py`: append this paragraph to the module docstring, before `Internal; not part of the public API.`:

   ```
   A module that reports "a value, or unavailable with a reason" uses one of
   the existing types — ``nnx.link_tasks.MetricValue``,
   ``nnx.decisions.benchmark.MetricValue``,
   ``nnx.paradigms.offline_distillation.Measure`` — rather than a fourth;
   the three stay separate because each is public.
   ```

10. Regenerate the API reference (the three classes are catalogued): `uv run python -m scripts.docs.build_api_reference`.

No digest, value or behaviour changes, so no test changes and no CHANGELOG entry.

**Worker setup.** No torch needed to edit; `uv sync --frozen --all-extras` is needed to regenerate `docs/api.md` and run the checks.

## Acceptance criteria

- [ ] Each of the four odd digest sites carries its comment. Check: `grep -c "Never change the form" src/nnx/link_tasks.py src/nnx/search.py src/nnx/decisions/schema.py src/nnx/decisions/benchmark.py` returns 1 for each file
- [ ] `_artifacts.py` states the rule for new artifacts. Check: `grep -n "A new artifact type subclasses" src/nnx/_artifacts.py` returns 1 line
- [ ] `docs/api.md` shows the cross-references. Check: `grep -c "three separate public types" docs/api.md` returns 1, and `uv run python -m scripts.docs.build_api_reference --check && uv run python -m scripts.docs.build_docs --check`
- [ ] No digest changed: the digest-pinning suites pass. Check: `uv run pytest tests/test_link_tasks.py tests/test_decision_schema.py tests/test_decision_benchmark.py tests/test_search.py tests/test_offline_distillation.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [link_tasks.py L435-L437](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/link_tasks.py#L435-L437) | `LinkSplit.digest`: compact, ASCII-escaped, `sha256:` |
| [link_tasks.py L967](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/link_tasks.py#L967) | the split digest saved in the task state |
| [search.py L276-L277](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/search.py#L276-L277), [L576](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/search.py#L576) | `SearchSpace.digest` and where a study saves it |
| [schema.py L145-L152](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/schema.py#L145-L152) | `Question.digest`: compact, Unicode kept, bare hex |
| [benchmark.py L628-L636](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L628-L636) | `_identity`: the provider's prompt identity |
| [_config.py L110-L115](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_config.py#L110-L115) | `_canonical_json`, the shared form |
| [_artifacts.py L1-L7](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_artifacts.py#L1-L7) | the module docstring that gets the rule |
| [link_tasks.py L595-L600](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/link_tasks.py#L595-L600), [benchmark.py L832-L838](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L832-L838), [offline_distillation.py L639-L643](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/paradigms/offline_distillation.py#L639-L643) | the three value types |
| #557, #521 | the decision digests' and artifacts' recorded formats |

## Scope

**In:** comments at four digest sites, the `_artifacts.py` and `_metrics.py` notes, three docstrings, the regenerated API reference.

**Out:** changing any digest form (it would orphan saved checkpoints, studies and records). Merging the value types (a public API break). The undocumented decisions members in `docs/api.md` (FEAT-141).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-121
**Related:** #557, #521
