# Haiku readiness, part 2 (SYNC, SCOPE, DEAD, CLUT, SMELL): key results

Totals: READY 51, NEEDS-DETAIL 16, NEEDS-DECISION 15, TOO-BIG 9.
Combined with part 1: READY 98, NEEDS-DETAIL 23, NEEDS-DECISION 29, TOO-BIG 11 (161).

## Verdicts
- READY:
  - SYNC-01..07
  - DEAD-01, 02, 04, 05, 06, 07, 08, 09, 11, 12, 13
  - CLUT-02, 03, 05
  - SMELL-05, 08, 09, 11, 14, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 33, 34, 40, 47, 48, 49, 50, 51, 52, 54, 59, 60
- NEEDS-DETAIL:
  - DEAD-03: what gives NaN when no record has the field
  - DEAD-10: close it if BUG-21 landed?
  - SMELL-04: include BUG-18 skip=?
  - SMELL-06: contract of `_close_epoch_records`; tqdm arguments
  - SMELL-12: the `call` parameter; constructing `type(e)`
  - SMELL-13: `refresh` semantics; mapping in `__setstate__`
  - SMELL-17: columns, wording
  - SMELL-19: tuple contents, root
  - SMELL-29: depends on BUG-03/04; the helper is not named
  - SMELL-35: **invalid signature** (default before non-default `ctx`)
  - SMELL-39: per-module messages
  - SMELL-44: `cuda=` placeholder; CUDA restore is a behaviour change
  - SMELL-55: `paradigm=` purpose
  - SMELL-56: `str` meaning
  - SMELL-57: union or error; behaviour change
  - SMELL-58: `optional=` semantics
- NEEDS-DECISION:
  - SCOPE-01..04
  - DEAD-14 (delete markers or enable BLE)
  - CLUT-01, CLUT-04
  - SMELL-07 (stricter parsing OK?)
  - SMELL-31
  - SMELL-37 (NLL floor)
  - SMELL-42 and SMELL-46 (G11)
  - SMELL-43 (jev blank text behaviour)
  - SMELL-45 (ImportError → ValueError)
  - SMELL-53 (G11)
- TOO-BIG:
  - SMELL-01: (1) `_atomic_io` + re-exports, (2) nn_run/nn_checkpoint callers, (3) other callers
  - SMELL-02: one PR per step
  - SMELL-03: one PR per function
  - SMELL-10: pin bytes → dataclass → flags
  - SMELL-16: one PR per module; which semantics
  - SMELL-36: 3 PRs
  - SMELL-38: per module
  - SMELL-41: arun, then split
  - SMELL-61: per module; needs the full set

## Missing Check fields
DEAD-06/07/08/10/13/14, CLUT-01/05, SMELL-51/52/59, SCOPE-04.

## Ambiguities in sections 0-3 (in addition to part 1's)
1. Conditional and optional steps are treated inconsistently ("optional", "if X is done").
2. Rule 9's list of decision items is incomplete. Also needing a decision: SCOPE-02, CLUT-04, DEAD-14, SMELL-07, 37, 43, 45, and all of G11.
3. Tests named per item: 14 items name none, and torch is not installed.
4. "Refactors must not change behaviour", yet SMELL-07, 43, 44, 45, 47, 49 and 57 change behaviour: reclassify them as behaviour changes with a CHANGELOG line.
5. "Keep message text" vs SMELL-38 and SMELL-49.
6. One item per PR vs one branch per group.
7. Rule 4 aliases vs deletions; rule 10 (no test deletion) vs DEAD-13 and DEAD-03.
8. "Smallest first" is undefined.
9. Cross-group ordering (DEAD-04 must go with DOC-35).
10. torch cannot be installed in this environment.

## Overlaps and conflicts (ordering needed)
- SMELL-03 vs SMELL-06 (same `_train_impl` ranges)
- SMELL-04 vs BUG-18
- SMELL-41 vs SMELL-37, 38, 43, 49, 50, 58 (benchmark.py)
- SMELL-15 vs SMELL-61 (local imports)
- SMELL-05 vs DEAD-14
- SMELL-11 vs SMELL-30 vs SMELL-02 step 2 (replay helpers)
- SMELL-17 vs SMELL-18; DEAD-01 vs SMELL-17
- SMELL-58 vs SMELL-38
- SMELL-02 vs SMELL-12, 13, 20, 21, 30 (nn_model)
- SMELL-35 vs SCOPE-01 and SMELL-42 (cycles)

## Glossary needed
Hub, durability, run layout, records, lazy modules, warnings, paradigms, decisions, PEFT, transforms, digests, validators, release tooling, test rules, build tooling, ruff/BLE, external tools (pngquant, repo variables), group names.
