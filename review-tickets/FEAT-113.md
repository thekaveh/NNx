---
id: FEAT-113
title: "[FEAT-113] Reuse the shared NLL, Brier and macro-F1 in the decision benchmark"
labels: ["enhancement", "area: decisions", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-37"]
depends_on: []
related_issues: [243, 518]
status: READY
---

**Category:** Decisions · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-113: The decision benchmark computes NLL, Brier and macro-F1 with its own pure-Python code, although its module docstring says they "follow the definitions of the named `nll` / `brier` metrics", whose terms live in `_probability.nll_terms` / `brier_terms`, and macro-F1 is `_confusion.ConfusionCounts.scores("macro")`. This ticket computes all three through the shared code. The benchmark keeps its own NLL floor, exact by default (`epsilon=None`, #243). Values agree with today's to within rounding (relative 1e-12). The `link_tasks` floor is decided in DEC-026 and is not touched here (steps 1–2 of SMELL-37).

## Context

**What happens today.** `decisions/benchmark.py` L892–L917:

```python
def _macro_f1(rows: Sequence[_Row]) -> float:
    true = [row.labels[row.truth] for row in rows]
    predicted = [row.labels[row.top] for row in rows]
    scores = []
    for label in sorted(set(true) | set(predicted)):
        tp = sum(t == label and p == label for t, p in zip(true, predicted, strict=True))
```

```python
def _nll(rows: Sequence[_Row], epsilon: Optional[float]) -> float:
    total = 0.0
    for row in rows:
        p = row.probabilities[row.truth]
        if epsilon is not None:
            p = max(p, epsilon)
        total += math.inf if p == 0.0 else -math.log(p)
    return total / len(rows)


def _brier(rows: Sequence[_Row]) -> float:
    return sum(sum((p - (i == row.truth)) ** 2 for i, p in enumerate(row.probabilities)) for row in rows) / len(rows)
```

`_macro_f1` loops over every label for every row (O(L·N)). The shared terms: `nll_terms(target, probabilities, epsilon)` (`_probability.py` L125) clips to `[epsilon, 1]` (`epsilon=None`: exact, `+inf` at 0) and `brier_terms(target, probabilities)` (L149) sums squared errors over the class axis; `ConfusionCounts` (`_confusion.py` L44) reproduces scikit-learn's macro precision/recall/F1 with `zero_division=0`.

**Why it matters.** Three metric definitions exist twice. A fix to the shared one (as #518 was for the Bernoulli NLL) does not reach the benchmark, and the docstring's claim is true only by coincidence.

**History.**
- #243 (FEAT-021) built the benchmark and fixed its NLL rule: "Exact NLL is `+inf` when the true class has zero probability and `epsilon=None`" (`test_zero_true_class`). That default stays: the benchmark passes its own `epsilon`, never `NLL_EPSILON`.
- #518 fixed the named Bernoulli `nll` metric's float32 saturation in the shared terms.
- The `link_tasks` floor (`1e-300`) is a separate decision, DEC-026.

**Background for the worker.**
- `_Row` (L868–L886): one answered sample: `labels` (its question's option ids), `probabilities` (aligned with `labels`), `truth` (the true option's index), `top` (the first option of maximal probability). Rows of one report may have different numbers of options.
- Zero padding: appending probability 0 options to a row adds `0` to its Brier sum (`(0 - 0)**2`) and does not change its true-option probability, so all rows fit one `(N, W)` array.
- Sum order: each mean is taken with Python's `sum(...) / len(rows)` over the per-row terms, in row order, as today. The per-row terms come from NumPy (`np.log`, `x * x`) instead of `math.log` and `x ** 2`, which can round the last bit differently; that is the only source of difference.
- `bootstrap_interval` (L1376) calls the same `_nll` / `_STATISTICS` functions on each resample, so it follows automatically.

**Proposal.**
1. In `src/nnx/decisions/benchmark.py`, add to the imports (first-party block, after `from .._json_depth import MAX_JSON_DEPTH, too_deep`, L64):

   ```python
   import numpy as np

   from .._confusion import ConfusionCounts
   from .._probability import brier_terms, nll_terms
   ```

   (`import numpy as np` goes with the third-party imports; `uv run ruff check --fix` places it.)
2. Directly above `_accuracy` (L888), add:

   ```python
   def _row_arrays(rows: Sequence[_Row]) -> tuple[np.ndarray, np.ndarray]:
       """Each row's true-option index and its probabilities, zero-padded to the
       widest row (rows may offer different numbers of options; a padded 0 adds
       nothing to a row's NLL or Brier term)."""
       probabilities = np.zeros((len(rows), max(len(row.probabilities) for row in rows)), dtype=np.float64)
       for index, row in enumerate(rows):
           probabilities[index, : len(row.probabilities)] = row.probabilities
       return np.array([row.truth for row in rows], dtype=np.int64), probabilities
   ```

3. Replace the bodies of `_macro_f1`, `_nll` and `_brier` (L892–L917; `_accuracy` stays) with:

   ```python
   def _macro_f1(rows: Sequence[_Row]) -> float:
       true = [row.labels[row.truth] for row in rows]
       predicted = [row.labels[row.top] for row in rows]
       code = {label: index for index, label in enumerate(sorted(set(true) | set(predicted)))}
       counts = ConfusionCounts()
       counts.update(np.array([code[label] for label in true]), np.array([code[label] for label in predicted]))
       scores = counts.scores("macro")
       assert scores is not None  # rows is never empty here
       return scores[2]


   def _nll(rows: Sequence[_Row], epsilon: Optional[float]) -> float:
       truth, probabilities = _row_arrays(rows)
       return sum(nll_terms(truth, probabilities, epsilon).tolist()) / len(rows)


   def _brier(rows: Sequence[_Row]) -> float:
       truth, probabilities = _row_arrays(rows)
       return sum(brier_terms(truth, probabilities).tolist()) / len(rows)
   ```

   `_nll` passes the benchmark's `epsilon` (default `None`, exact) explicitly; it must never use `nll_terms`' own default.
4. In the module docstring (L39–L46), replace "NLL and Brier follow the\ndefinitions of the named ``nll`` / ``brier`` metrics (``nnx.monitors``)," with "NLL, Brier and\nmacro-F1 are computed by the same code as the named ``nll`` / ``brier`` /\n``f1`` metrics (``nnx.monitors``),". `docs/api.md` does not render this module docstring, so nothing is regenerated.
5. Add a test to `tests/test_decision_benchmark.py`, comparing against the old pure-Python definitions on rows of different widths:

   ```python
   def test_benchmark_metrics_match_their_reference_definitions():
       from nnx.decisions.benchmark import _brier, _macro_f1, _nll, _Row

       three = Choice("Pick?", (("a", "A"), ("b", "B"), ("c", "C")))
       rows = [
           _Row(Sample("s0", AB, "x", "a"), ("a", "b"), (0.8, 0.2)),
           _Row(Sample("s1", AB, "x", "b"), ("a", "b"), (0.6, 0.4)),
           _Row(Sample("s2", three, "x", "c"), ("a", "b", "c"), (0.1, 0.2, 0.7)),
           _Row(Sample("s3", three, "x", "a"), ("a", "b", "c"), (0.3, 0.5, 0.2)),
       ]
       nll = sum(-math.log(row.probabilities[row.truth]) for row in rows) / len(rows)
       brier = sum(sum((p - (i == row.truth)) ** 2 for i, p in enumerate(row.probabilities)) for row in rows) / len(rows)
       assert _nll(rows, None) == pytest.approx(nll, rel=1e-12)
       assert _nll(rows, 0.5) == pytest.approx(sum(-math.log(max(r.probabilities[r.truth], 0.5)) for r in rows) / 4, rel=1e-12)
       assert _brier(rows) == pytest.approx(brier, rel=1e-12)
       # true a, b, c, a; predicted a, a, c, b: F1 a = 0.5, b = 0, c = 1.
       assert _macro_f1(rows) == pytest.approx(0.5, rel=1e-12)
   ```

   (`Choice`, `Sample`, `AB`, `math` and `pytest` are already imported or defined in that module: L27, L34, L49, L16.)
6. Add to `CHANGELOG.md` under `## [Unreleased]` → `### Changed`:

   ```
   - **The decision benchmark computes NLL, Brier and macro-F1 with the shared metric code.** `evaluate` and `bootstrap_interval` now use the same terms as the named `nll`, `brier` and `f1` metrics; NLL stays exact by default (`epsilon=None`). Values can differ from earlier releases in the last bits (below 1e-12 relative).
   ```

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The new test passes. Check: `uv run pytest tests/test_decision_benchmark.py::test_benchmark_metrics_match_their_reference_definitions`
- [ ] The exact default and the existing metric tests hold, including `test_zero_true_class` (`+inf` with `epsilon=None`). Check: `uv run pytest tests/test_decision_benchmark.py`
- [ ] No pure-Python NLL, Brier or F1 loop remains. Check: `grep -n "math.log(p)\|tp + fp\|\*\* 2 for i, p" src/nnx/decisions/benchmark.py` returns 0 lines
- [ ] The epsilon is passed explicitly. Check: `grep -n "nll_terms(truth, probabilities, epsilon)" src/nnx/decisions/benchmark.py` returns 1 line
- [ ] The decisions suites pass. Check: `uv run pytest tests/test_decision_benchmark.py tests/test_decision_jobs.py tests/test_decision_schema.py`
- [ ] The CHANGELOG entry is under `[Unreleased]` → `### Changed`. Check: `grep -n "computes NLL, Brier and macro-F1 with the shared metric code" CHANGELOG.md`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [benchmark.py L38-L46](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L38-L46) | the docstring's claim and the exact-NLL rule |
| [benchmark.py L868-L917](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L868-L917) | `_Row`, `_accuracy`, `_macro_f1`, `_nll`, `_brier` |
| [benchmark.py L933](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L933), [L1090-L1098](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1090-L1098), [L1408-L1409](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1408-L1409) | `_STATISTICS`, `_metrics`, the bootstrap statistic |
| [_probability.py L125-L157](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_probability.py#L125-L157) | `nll_terms`, `brier_terms` |
| [_confusion.py L44-L66](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_confusion.py#L44-L66), [L123-L141](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_confusion.py#L123-L141), [L242-L259](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_confusion.py#L242-L259) | `ConfusionCounts`, `scores`, `average_scores` |
| [test_decision_benchmark.py L125-L152](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_decision_benchmark.py#L125-L152) | the fixture metrics and `test_zero_true_class` |
| #243 | exact NLL by default |
| #518 | the shared NLL terms' saturation fix |

## Scope

**In:** `_row_arrays`, the three metric bodies, the docstring sentence, one test, the CHANGELOG line.

**Out:** the `link_tasks` NLL floor (DEC-026). `_accuracy` (already a one-line mean). Splitting `benchmark.py` (FEAT-121). The 1e-6 probability tolerance (FEAT-127).

## Dependencies

**Blocked by:** none
**Blocks:** DEC-026, FEAT-121
**Related:** #243, #518
