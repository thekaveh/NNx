---
id: FEAT-114
title: "[FEAT-114] Validate decision counts with require_count (part 1 of 4)"
labels: ["enhancement", "area: decisions", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-38"]
depends_on: []
related_issues: [544, 231]
status: READY
---

**Category:** Decisions · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-114: `nnx.decisions` checks its integer settings with eleven hand-written `isinstance(x, bool) or not isinstance(x, numbers.Integral) or x < N` tests, each with its own message, although `_validation.require_count` and `checked()` do exactly this for the rest of NNx. This ticket routes them through one small helper per module. Each module keeps its own error type (`BenchmarkError`, `InvalidJob`, `InvalidDecisionRequest`); the messages change to the shared wording. This is part 1 of SMELL-38 (the decisions sites); FEAT-115, FEAT-116 and FEAT-117 cover the rest.

## Context

**What happens today.** `Budget.__post_init__` (`decisions/benchmark.py` L587–L592):

```python
        for name in ("max_calls", "max_samples"):
            value = getattr(self, name)
            if value is None and name == "max_samples":
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise BenchmarkError(f"Budget.{name} must be a positive integer, got {value!r}")
```

`Limits.__post_init__` (`decisions/jobs.py` L296–L301):

```python
        if isinstance(depth, bool) or not isinstance(depth, numbers.Integral) or depth < 0:
            raise InvalidJob(f"max_depth must be a non-negative integer, got {depth!r}")
        concurrency = self.max_concurrency
        if isinstance(concurrency, bool) or not isinstance(concurrency, numbers.Integral) or concurrency < 1:
            raise InvalidJob(f"max_concurrency must be a positive integer, got {concurrency!r}")
        object.__setattr__(self, "max_concurrency", int(concurrency))
```

The other sites: `collect(batch_size=)` (benchmark L709), `Resources` (L1024–L1027), `evaluate(n_bins=)` (L1260), `bootstrap_interval(seed=, resamples=)` (L1390–L1393), `jobs._optional_positive` (L247–L252), `providers._check_max_batch` (L69–L73) and `nli._positive` (L149–L152).

**Why it matters.** Eleven copies of one rule with eleven wordings. `require_count` also names the owner and the accepted domain, and it is what #544 made every other validator use.

**History.** #231 (FIX-021) introduced `require_count` for counts across NNx; #544 (FIX-076) moved the hand-written validators to the shared helpers ("Accept NumPy numbers and refuse oversized values in hand-written validators") and left these decisions sites. No test pins the old decisions wording: the tests match field names only (`tests/test_decision_benchmark.py` L612 `match="n_bins"`, `tests/test_decision_jobs.py` L620 `match="max_concurrency"`, `tests/test_decision_schema.py` L240 `match="max_batch"`), which the new messages keep.

**Background for the worker.**
- `require_count(value, field, *, owner, minimum=None, maximum=None, ...)` (`_validation.py` L140): returns `int(value)` for any `numbers.Integral` except `bool`; otherwise raises `ValueError`. Its two messages:
  - not an integer: `"{owner} requires {field} to be an integer{domain}, got {value!r} of type {type}"`;
  - out of range: `"{owner} requires {field}{domain}, got {value!r}"`,
  where `{domain}` is ` >= 1`, ` >= 0` or ` >= 1 and <= 10000`.
- `checked(check, value, field, *, owner, error, **domain)` (`_validation.py` L232): runs `check` and re-raises its `ValueError` as `error(str(exc))`, so each module keeps its error type. `calibration._checked` (L169) is the house pattern for a per-module wrapper.
- Not changed: `nli._class_id` (L139–L140) accepts an `int` **or** a label name, so its "class id (int) or a label name (str)" message stays; a count message would drop the second form.
- Where a site stored `int(value)`, it keeps storing the returned `int`; where it stored nothing, the return value is discarded, so stored values do not change.

**Proposal.**
1. `src/nnx/decisions/benchmark.py`:
   - add `from .._validation import checked, require_count` to the first-party imports (after `from .._json_depth import ...`, L64);
   - after `class BenchmarkError` (L124–L126), add:

     ```python
     def _count(value: Any, field: str, **domain: Any) -> int:
         """``value`` as an integral count (:func:`nnx._validation.require_count`),
         refused as :class:`BenchmarkError`."""
         return checked(require_count, value, field, owner="nnx.decisions.benchmark", error=BenchmarkError, **domain)
     ```

   - `Budget.__post_init__`: replace the `if isinstance(value, bool) ...: raise ...` pair (L591–L592) with `_count(value, f"Budget.{name}", minimum=1)`;
   - `collect`: replace L709–L710 with `_count(batch_size, "batch_size", minimum=1)`;
   - `Resources.__post_init__`: replace L1024–L1027 (both `if` statements in the loop) with:

     ```python
                 if value is not None:
                     object.__setattr__(self, name, _count(value, f"Resources.{name}", minimum=0))
     ```

   - `evaluate`: replace L1260–L1261 with `_count(n_bins, "n_bins", minimum=1, maximum=10_000)`;
   - `bootstrap_interval`: replace L1390–L1393 with `_count(seed, "seed", minimum=0)` and `_count(resamples, "resamples", minimum=1)`.
2. `src/nnx/decisions/jobs.py`:
   - add `from .._validation import checked, require_count` to the first-party imports (before `from .providers import _synchronous`, L87);
   - replace `_optional_positive` (L247–L252) with:

     ```python
     def _count(value: Any, field: str, **domain: Any) -> int:
         """``value`` as an integral count (:func:`nnx._validation.require_count`),
         refused as :class:`InvalidJob`."""
         return checked(require_count, value, field, owner="nnx.decisions.jobs", error=InvalidJob, **domain)


     def _optional_positive(value: Any, what: str) -> Optional[int]:
         return None if value is None else _count(value, what, minimum=1)
     ```

   - `Limits.__post_init__`: replace L295–L301 (from `depth = self.max_depth` to `object.__setattr__(self, "max_concurrency", int(concurrency))`) with:

     ```python
             _count(self.max_depth, "max_depth", minimum=0)
             object.__setattr__(self, "max_concurrency", _count(self.max_concurrency, "max_concurrency", minimum=1))
     ```

3. `src/nnx/decisions/providers.py`: add `from .._validation import checked, require_count` (after `from .._probability import class_labels`, L16), and replace the body of `_check_max_batch` (L70–L73) with:

   ```python
       if max_batch is not None:
           checked(
               require_count, max_batch, "max_batch", owner="nnx.decisions.providers", error=InvalidDecisionRequest, minimum=1
           )
   ```

4. `src/nnx/decisions/nli.py`: add `from .._validation import checked, require_count` (before `from .providers import Capabilities, Question`, L79), and replace the body of `_positive` (L150–L152) with:

   ```python
       return checked(require_count, value, what, owner="nnx.decisions.nli", error=InvalidDecisionRequest, minimum=1)
   ```

5. Run `uv run ruff check --fix src/nnx/decisions/` (it removes `import numbers` from a module that no longer uses it) and `uv run ruff format src/nnx/decisions/`.
6. Add to `tests/test_decision_benchmark.py`:

   ```python
   @pytest.mark.parametrize(
       ("build", "error", "message"),
       [
           (lambda: Budget(max_calls=0), BenchmarkError, r"nnx\.decisions\.benchmark requires Budget\.max_calls >= 1, got 0"),
           (
               lambda: Budget(max_calls=True),
               BenchmarkError,
               r"nnx\.decisions\.benchmark requires Budget\.max_calls to be an integer >= 1, got True of type bool",
           ),
           (lambda: Resources(warmup=-1), BenchmarkError, r"requires Resources\.warmup >= 0, got -1"),
       ],
   )
   def test_benchmark_counts_use_the_shared_count_message(build, error, message):
       with pytest.raises(error, match=message):
           build()
       assert Resources(warmup=np.int64(3)).warmup == 3 and type(Resources(warmup=np.int64(3)).warmup) is int


   def test_job_provider_and_nli_counts_use_the_shared_count_message():
       from nnx.decisions import InvalidDecisionRequest, InvalidJob, Limits
       from nnx.decisions.nli import _positive

       with pytest.raises(InvalidJob, match=r"nnx\.decisions\.jobs requires max_depth >= 0, got -1"):
           Limits(max_depth=-1)
       assert Limits(max_concurrency=np.int64(2)).max_concurrency == 2
       with pytest.raises(InvalidDecisionRequest, match=r"nnx\.decisions\.providers requires max_batch >= 1, got 0"):
           Capabilities(primitives={"choice"}, modalities={"text"}, max_batch=0)
       with pytest.raises(InvalidDecisionRequest, match=r"nnx\.decisions\.nli requires max_length to be an integer"):
           _positive(2.5, "max_length")
   ```

   (`Budget`, `Resources`, `BenchmarkError`, `Capabilities`, `np` and `pytest` are imported there already.)
7. Add to `CHANGELOG.md` under `## [Unreleased]` → `### Changed`:

   ```
   - **Decision counts are validated like every other NNx count.** `Budget`, `collect(batch_size=)`, `Resources`, `evaluate(n_bins=)`, `bootstrap_interval(seed=, resamples=)`, `Limits`, `Capabilities(max_batch=)`, `FixedHeadProvider(max_batch=)` and `NLIProvider`'s `max_length`, `pair_batch_size` and `max_batch` refuse a non-integer or out-of-range value with the shared message (`nnx.decisions.<module> requires <field> >= 1, got …`). The error types are unchanged.
   ```

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new tests fail on `develop` before the change and pass after it. Check: `uv run pytest tests/test_decision_benchmark.py -k "shared_count_message"`
- [ ] No hand-written count test remains in the four modules except `nli._class_id`. Check: `grep -n "isinstance(.*, bool) or not isinstance(.*, numbers.Integral)" src/nnx/decisions/benchmark.py src/nnx/decisions/jobs.py src/nnx/decisions/providers.py src/nnx/decisions/nli.py` returns 1 line (`nli.py`, `_class_id`)
- [ ] The decisions suites pass, including the field-name matches. Check: `uv run pytest tests/test_decision_benchmark.py tests/test_decision_jobs.py tests/test_decision_schema.py tests/test_decision_nli.py tests/test_decision_nnx_adapter.py tests/test_decision_jev.py`
- [ ] The CHANGELOG entry is under `[Unreleased]` → `### Changed`. Check: `grep -n "Decision counts are validated like every other NNx count" CHANGELOG.md`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [benchmark.py L124-L126](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L124-L126) | `BenchmarkError` |
| [benchmark.py L586-L592](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L586-L592) | `Budget` |
| [benchmark.py L709-L710](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L709-L710) | `collect(batch_size=)` |
| [benchmark.py L1022-L1027](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1022-L1027) | `Resources` |
| [benchmark.py L1260-L1261](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1260-L1261), [L1390-L1393](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1390-L1393) | `evaluate(n_bins=)`, `bootstrap_interval(seed=, resamples=)` |
| [jobs.py L247-L252](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L247-L252), [L291-L301](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L291-L301) | `_optional_positive`, `Limits` |
| [providers.py L69-L73](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/providers.py#L69-L73) | `_check_max_batch` |
| [nli.py L131-L152](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/nli.py#L131-L152) | `_class_id` (kept) and `_positive` |
| [_validation.py L140-L176](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_validation.py#L140-L176), [L232-L243](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_validation.py#L232-L243) | `require_count` and `checked` |
| [calibration.py L169-L171](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/calibration.py#L169-L171) | the per-module wrapper pattern |
| #231, #544 | the shared count validator and the hand-written validators it replaced |

## Scope

**In:** the eleven decisions sites listed, their helpers, two tests, the CHANGELOG line.

**Out:** `nli._class_id` (int-or-name, kept). Generation, diffusion and plan counts (FEAT-115), viz (FEAT-116), the task modules' `_count` helpers (FEAT-117). The non-empty-text validators (FEAT-134). Splitting `benchmark.py` (FEAT-121).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-134, FEAT-121
**Related:** #544, #231
