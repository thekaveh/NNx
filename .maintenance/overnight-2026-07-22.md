# 1. Overnight Maintenance Ledger: 2026-07-22

**Status:** Complete. The run reached 10 consecutive zero-issue verification
passes after two issue-bearing audit passes.

## 1.1. Run Parameters

- Branch: `overnight-maintenance-2026-07-22`
- Upstream: `origin/overnight-maintenance-2026-07-22`
- Required zero-issue passes: 10
- Maximum verification passes: 50
- Numbered docs: yes
- Diagrams: yes
- Push at completion: yes

## 1.2. Verification Pass History

| Pass | Result | Coverage evidence |
| ---: | --- | --- |
| 1 | Issues fixed | Broad code, test, dependency, workflow, release, examples, docs, diagram, GitHub settings, and published-surface audit. |
| 2 | Issues fixed | Independent correctness, concurrency, training-resume, build, dependency, documentation, and publication review. |
| 3 | Zero issues | Documentation inventory, links, numbering, terminology, examples, MkDocs/wiki projection, architecture source parity, accessibility, and responsive rendering. |
| 4 | Zero issues | Training lifecycle, warm resume, transformed checkpoints, finite-horizon schedulers, overwrite behavior, and phase checkpoints. |
| 5 | Zero issues | Process concurrency, atomic persistence, run admission, global best selection, immutable state, and defensive serialization. |
| 6 | Zero issues | Dataset boundaries, callbacks, learning-rate finder, custom steps, partial accumulation, and Trainer multi-optimizer behavior. |
| 7 | Zero issues | Public API/import surface, all numbered examples, external contracts, GGUF/Ollama claims, and optional dependencies. |
| 8 | Zero issues | Workflow permissions, immutable action pins, CodeQL/security settings, frozen lock, vulnerability audit, and release controls. |
| 9 | Zero issues | Reproducible sdist/wheel behavior, package metadata, strict documentation builds, wiki generation, and repository hygiene. |
| 10 | Zero issues | Full regression suite, Ruff lint/format, Pyright, lock drift, zizmor, build/twine, and diff whitespace. |
| 11 | Zero issues | Fresh cross-link, stale-name, unsafe-option, secret-pattern, generated-artifact, and heading-hierarchy scans. |
| 12 | Zero issues | Final exact-tree rerun of all applicable automated gates and maintenance acceptance checks. |

## 1.3. Issue Log

| ID | Severity | Area | Resolution | Verification |
| --- | --- | --- | --- | --- |
| OM-2026-001 | High | Gradient accumulation | Flush the final partial accumulation window and expose last-batch context. | Focused accumulation tests; full suite. |
| OM-2026-002 | High | Warm resume | Persist optimizer, scheduler, scaler, completed epoch, model resume state, and all RNG states in a versioned sidecar. | Resume parity and QAT tests. |
| OM-2026-003 | High | Checkpoint integrity | Add generation stamps, process locks, unique temporary files, and mismatch rejection. | Concurrent writer and interrupted-pair tests. |
| OM-2026-004 | High | Run integrity | Add process-safe admission, full-lifecycle leases, overwrite cleanup, lineage-aware identities, and root-level best locking. | Concurrent admission/overwrite tests. |
| OM-2026-005 | High | Born-Again Networks | Reset each student to the original initialization and record parent lineage. | Generation reset/teacher isolation tests. |
| OM-2026-006 | High | QAT resume | Store pre-transform model state with transformed inference checkpoints. | End-to-end QAT checkpoint tests. |
| OM-2026-007 | Medium | Scheduler continuation | Require an explicit shared horizon for finite-horizon warm resume. | OneCycle rejection and resume tests. |
| OM-2026-008 | Medium | Input contracts | Reject invalid tabular labels, graph sampler combinations, callback controls, LR-finder controls, and unsafe checkpoint tags. | Boundary regression tests. |
| OM-2026-009 | Medium | Ollama export | Validate the audited option schema, numeric domains, unknown keys, sequences, and control characters. | Documented-option and rejection matrix tests. |
| OM-2026-010 | Medium | Immutable state | Defensively freeze nested parameter collections and snapshot run state. | Mutation and round-trip tests. |
| OM-2026-011 | Medium | Optional dependencies | Move IPython to the notebook extra; add explicit file locking and pre-commit development dependencies. | Import fallback and frozen lock checks. |
| OM-2026-012 | Medium | Examples | Rename misleading examples, correct custom evaluation/AMP accumulation, and add import plus representative execution smoke coverage. | 29 example smoke tests. |
| OM-2026-013 | Medium | Documentation | Correct resume, evaluation, GGUF/Ollama, persistence, and architecture claims across canonical docs. | Strict MkDocs and link scans. |
| OM-2026-014 | Medium | Three surfaces | Project README, examples, contributing, security, tests, changelog, and license into both MkDocs and wiki outputs. | Inventory and deterministic projection checks. |
| OM-2026-015 | Medium | Wiki integrity | Fail on existing unmapped documentation links instead of silently discarding them. | Wiki projection regressions. |
| OM-2026-016 | Medium | Diagram integrity | Extract the standalone SVG from one accessible HTML source and verify desktop/mobile containment. | Parity check plus visual inspection at 1440x900 and 390x844. |
| OM-2026-017 | Medium | Reproducible builds | Normalize sdist metadata and compare two independently built release artifacts. | Reproducibility tests and local double build. |
| OM-2026-018 | Medium | Supply chain | Update immutable action pins, minimize permissions, freeze dependency resolution, and audit the production export. | Zizmor clean; no known production vulnerabilities. |
| OM-2026-019 | Medium | Repository security | Enable alerts, security updates, private reporting, secret scanning, push protection, CodeQL, immutable releases, and merge-only Gitflow rules. | GitHub API readback. |
| OM-2026-020 | Low | Contributor tooling | Align pre-commit Ruff with CI and include pre-commit in the dev extra. | Hook configuration and frozen lock checks. |
| OM-2026-021 | Low | Hygiene | Update stale example references and document the superseded 2026-07-01 run. | Repository-wide stale-name scan. |

## 1.4. Accepted Complexity

`NNModel._train_impl` and `Trainer._train_impl` remain long orchestration
methods. Their public validation/admission layers were extracted, but further
fragmentation was intentionally deferred: lifecycle order is the primary
contract, the loops are comprehensively covered, and a broader decomposition
would add regression risk without removing domain complexity.

## 1.5. Final Verification

- `pytest`: 1,163 passed, 1 skipped.
- Ruff lint and format: clean.
- Pyright: 0 errors, 0 warnings.
- MkDocs strict build, wiki projection, and architecture extraction: clean.
- Frozen production dependency audit: no known vulnerabilities.
- `uv lock --check`, package build, Twine metadata, zizmor, and
  `git diff --check`: clean.
- Live site/wiki deployment is intentionally deferred until this branch is
  merged to `main`; the source and generated projections are synchronized.
