---
id: FEAT-068
title: "[FEAT-068] Rename the test_pass2_*_series modules after what they test"
labels: ["enhancement", "area: packaging", "priority: P3", "review: 2026-11-27"]
sources: ["CLUT-02"]
depends_on: ["FIX-109"]
related_issues: [206, 211, 755, 757]
status: READY
---

**Category:** Packaging · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop)

## Summary

FEAT-068: six test modules (`tests/test_pass2_{d,f,n,ou,r,v}_series.py`, about 4,100 lines and 163 tests) are named after an old audit pass, not after the code they test, and `CONTRIBUTING.md` recommends them as "good models". Rename them with `git mv` to names that say what they cover, update every reference, and replace the CONTRIBUTING advice with "put the test in the module named after the code it covers".

## Context

**What happens today.** `CONTRIBUTING.md` L62:

```markdown
3. **Write tests.** Every PR that changes behavior should land with a focused test that fails before the change and passes after it. The existing `tests/test_*_series.py` files (organized by audit pass) are good models.
```

The module docstrings show what each file covers, for example `test_pass2_v_series.py`: `"""Pass-2 catalog: V-series tests (reproducibility).`, and `test_pass2_d_series.py`: `- D6: Utils / VisUtils — module-level functions exist; the class API still works (back-compat shim)`.

References outside the six files (`git grep -n test_pass2_ -- . ':!CHANGELOG.md'`):
- `CONTRIBUTING.md:83`: `uv run pytest tests/test_pass2_n_series.py::test_n7_evaluate_aggregates_across_batches`
- `docs/concepts.md:65-66`: the `seed` and `save_phase_checkpoints` test pointers
- `docs/feature-composition.yaml:53` (and the generated `docs/feature-composition.md:20`): `tests/test_pass2_f_series.py::test_f2_accumulate_grad_batches_steps_trailing_partial_cycle`
- `tests/test_nn_dataset.py:86`: a docstring pointer to `test_pass2_f_series.py`

**Why it matters.** A contributor cannot tell where a test belongs or where to find one, and the docs point new tests at the wrong pattern.

**History.** About 25 board issues cite these files by their old names (for example #206, #211, #755, #757); closed issues keep their text, so the PR body carries the rename map for anyone following an old link. FIX-109 (BUG-07) edits `tests/test_pass2_r_series.py`; it lands first so this rename does not conflict with it.

**Background for the worker.**
- *`docs/feature-composition.md`*: generated from `docs/feature-composition.yaml` by `uv run python scripts/check_feature_composition.py --render`; never edit the `.md` by hand. `--check` verifies the two agree and that every listed test node id exists.
- *CHANGELOG mentions*: old names in released CHANGELOG sections are history and stay.

**Proposal.**

1. Rename with history kept:
   ```bash
   git mv tests/test_pass2_d_series.py tests/test_utils_function_aliases.py
   git mv tests/test_pass2_f_series.py tests/test_training_features.py
   git mv tests/test_pass2_n_series.py tests/test_evaluate_regressions.py
   git mv tests/test_pass2_ou_series.py tests/test_predict_api_and_ux.py
   git mv tests/test_pass2_r_series.py tests/test_training_reliability.py
   git mv tests/test_pass2_v_series.py tests/test_seed_reproducibility.py
   ```
   None of the new names exists at `9ad461e`. Test function names and module docstrings stay as they are.
2. `CONTRIBUTING.md`:
   - L62: replace `The existing \`tests/test_*_series.py\` files (organized by audit pass) are good models.` with `Put the test in the module named after the code it covers (for example \`tests/test_calibration.py\`).`
   - L83: replace `tests/test_pass2_n_series.py::test_n7_evaluate_aggregates_across_batches` with `tests/test_evaluate_regressions.py::test_n7_evaluate_aggregates_across_batches`.
3. `docs/concepts.md` L65: replace `tests/test_pass2_v_series.py` with `tests/test_seed_reproducibility.py`; L66: replace `tests/test_pass2_ou_series.py` with `tests/test_predict_api_and_ux.py`.
4. `docs/feature-composition.yaml` L53: replace `tests/test_pass2_f_series.py::` with `tests/test_training_features.py::`, then run `uv run python scripts/check_feature_composition.py --render`.
5. `tests/test_nn_dataset.py` L86: replace `test_pass2_f_series.py` with `test_training_features.py`.
6. Put this rename map in the PR body:

   | Old module | New module |
   |---|---|
   | `tests/test_pass2_d_series.py` | `tests/test_utils_function_aliases.py` |
   | `tests/test_pass2_f_series.py` | `tests/test_training_features.py` |
   | `tests/test_pass2_n_series.py` | `tests/test_evaluate_regressions.py` |
   | `tests/test_pass2_ou_series.py` | `tests/test_predict_api_and_ux.py` |
   | `tests/test_pass2_r_series.py` | `tests/test_training_reliability.py` |
   | `tests/test_pass2_v_series.py` | `tests/test_seed_reproducibility.py` |

No library behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FIX-109 has merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] No reference to the old names remains outside `CHANGELOG.md`. Check: `git grep -n "test_pass2_" -- . ':!CHANGELOG.md'` prints nothing
- [ ] The six new modules exist and git records them as renames. Check: `git diff --name-status -M origin/develop -- tests | grep -c "^R"` prints at least `6`
- [ ] The renamed suites pass and collect as many tests as the old modules did. Check: on `develop`, `uv run pytest --collect-only -q tests/test_pass2_*_series.py | tail -1` gives N tests; on the branch, `uv run pytest -q tests/test_utils_function_aliases.py tests/test_training_features.py tests/test_evaluate_regressions.py tests/test_predict_api_and_ux.py tests/test_training_reliability.py tests/test_seed_reproducibility.py` passes and reports the same N
- [ ] The feature-composition page is regenerated and consistent. Check: `uv run python scripts/check_feature_composition.py --check`
- [ ] The docs build passes. Check: `uv run python -m scripts.docs.build_docs --check`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0

## Evidence

| Location | What it shows |
|---|---|
| [CONTRIBUTING.md L62](https://github.com/thekaveh/NNx/blob/9ad461e/CONTRIBUTING.md#L62), [L83](https://github.com/thekaveh/NNx/blob/9ad461e/CONTRIBUTING.md#L83) | The "good models" advice and an old-name command |
| [concepts.md L65-L66](https://github.com/thekaveh/NNx/blob/9ad461e/docs/concepts.md#L65-L66) | Test pointers by old name |
| [feature-composition.yaml L53](https://github.com/thekaveh/NNx/blob/9ad461e/docs/feature-composition.yaml#L53) | A test node id by old name |
| [test_nn_dataset.py L86](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_nn_dataset.py#L86) | A docstring pointer by old name |
| [test_pass2_v_series.py L1](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_pass2_v_series.py#L1), [test_pass2_d_series.py L1-L4](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_pass2_d_series.py#L1-L4) | Docstrings that show what the files cover |
| #206, #211, #755, #757 | Board issues citing the old names |

## Scope

**In:** six `git mv`s, the five reference updates, the regenerated feature-composition page, the CONTRIBUTING sentence.

**Out:** renaming test functions or rewriting module docstrings; splitting the files by module.

## Dependencies

**Blocked by:** FIX-109
**Blocks:** none
**Related:** #206, #211, #755, #757
