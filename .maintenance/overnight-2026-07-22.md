# 1. Overnight Maintenance Ledger: 2026-07-22

**Status:** Capped without convergence. All findings discovered through Pass 50
were repaired and final automated verification passed, but the run did not
reach 10 consecutive global zero-finding passes before `--max-passes 50`.

## 1.1. Run Parameters

- Branch/upstream: `overnight-maintenance-2026-07-22`
- Required consecutive clean passes: 10
- Maximum passes: 50
- Numbered documentation and diagrams: enabled
- Completion push: enabled

## 1.2. Pass History

| Passes | Result | Principal coverage |
| ---: | --- | --- |
| 1-18 | Findings repaired | Training, persistence, callbacks, resume, serialization, inference, examples, packaging, workflows, docs, wiki/site projection, and diagrams. |
| 19 | Findings repaired; recovery event | An unsafe regression probe deleted the checkout. The incident was disclosed immediately; the repository was recloned, committed work and the source distribution were overlaid, and the maintenance tree was reconstructed before verification resumed. |
| 20-35 | Findings repaired | Process safety, transformed checkpoints, device loading, Transformer targets/KV cache, subclass reconstruction, Hub/safetensors, release locking, action pinning, raw HTML/CSS/SVG parsing, and publication determinism. |
| 36 | Findings repaired | Multidimensional soft targets; tokenizer/QAT Hub reconstruction; locked release interpreter; all-groups audit; ruleset integration binding; CSS/SVG/list parsing. |
| 37 | Findings repaired | Classification error semantics, Trainer transforms, atomic Hub snapshots, CE subclasses, BCE logits, PyPI README projection, and parser edge cases. |
| 38 | Findings repaired | Native weighted-loss denominators, transform composition, tab-aware fences/references, directional images, PI attributes, temp aliases, and empty-wiki bootstrap. |
| 39 | Findings repaired | Soft BCE metric labels, tab residual columns, invalid CSS code points, and strict PI pseudo-attributes. |
| 40 | Findings repaired | Dedented list-fence closure, unreconstructible surgery persistence, and incomplete historical release assets. |
| 41-50 | Not globally clean | Supply-chain and documentation scopes reported zero throughout. Runtime found custom-step guard compatibility and mutable evaluation extras before the cap; both were repaired. No later passes were run beyond the configured cap. |

## 1.3. Major Resolutions

- Corrected gradient accumulation, ignored/class-weighted loss normalization,
  partial windows, finite-loss handling, evaluation aggregation, BCE logits and
  soft-target metrics, scheduler ownership, and callback lifecycle ordering.
- Hardened run/checkpoint persistence with locks, atomic generations, rollback,
  RNG/loader state, resume compatibility checks, topology recipes, portable
  devices, and explicit rejection of unreconstructible persistent surgery.
- Preserved subclass reconstruction, generative tokenizers, QAT topology,
  safetensors placement, atomic Hub snapshots, inference training modes, and
  complete Transformer KV-cache contracts.
- Added deterministic repository/MkDocs/wiki/PyPI projections, synchronized
  diagrams, strict link/anchor/manifest checks, and robust Markdown, HTML, CSS,
  SVG, and XML processing across adversarial container/escape cases.
- Hardened CI, release, security, and docs workflows with immutable action SHAs,
  least privilege, frozen tool groups, complete dependency audits, reproducible
  locked-interpreter builds, hashed smoke installs, and source-bound required
  checks.
- Bound all required gitflow checks to GitHub Actions integration `15368`.
- Attached the exact PyPI wheel and sdist to GitHub release `v0.2.1` and enabled
  immutable releases prospectively at repository level.

## 1.4. Accepted Conditions

- Python 3.13/3.14 required contexts become enforceable only after the candidate
  workflow reaches `main`.
- Release PR #133 lock refresh depends on that candidate workflow reaching
  `main`.
- GitHub does not retroactively make the already-published `v0.2.1` release
  immutable; its exact assets are attached and future releases are protected.
- Live site/wiki deployment remains deferred until merge to `main`; source and
  deterministic projections are synchronized.
- Long training orchestration methods, bounded one-batch lookahead, custom-loss
  fallback behavior, and historical version `v0.2.1` were retained where their
  complexity is contractual and covered.

## 1.5. Final Verification

- Full pytest suite, Ruff lint/format, and Pyright warnings gate.
- Deterministic docs, wiki, PyPI README, and diagram checks; strict MkDocs build.
- Frozen lock and dependency exports, reproducible wheel/sdist, Twine metadata,
  action/workflow security scans, and `git diff --check`.
- Live GitHub ruleset, release assets, and immutable-release repository setting.

The exact final command results are recorded in the completion commit context.
