# 1. Overnight Maintenance Ledger: 2026-07-01

## 1.1. Run Parameters

- Branch: `codex/overnight-maintenance`
- Upstream: `origin/codex/overnight-maintenance`
- Required zero-issue passes: 25
- Maximum verification passes: 75
- Numbered docs: yes
- Push at completion: yes

## 1.2. Verification Pass History

| Pass | Type | Issue count | Status | Coverage evidence |
| --- | --- | ---: | --- | --- |
| 1 | Genuine | 13 | Fixed or disposition recorded | Scanned package entry points, training loops, ONNX export paths, visualization helpers, surgery API, docs navigation, numbering, examples, CI, release hygiene, supply-chain automation, and architecture docs. |

## 1.3. Issue Log

| ID | Category | Location | Severity | Description | Why it matters | Proposed fix | Status | Validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OM-001 | Correctness / error handling | `src/nnx/nn/nn_model.py` | High | Non-finite training loss was detected after backward and optimizer step. | A NaN/Inf loss could mutate model weights before the guard raised. | Check finite loss before backward/optimizer mutation in AMP and non-AMP paths. | Fixed | Regression test plus full pytest suite. |
| OM-002 | Resource cleanup / lifecycle | `src/nnx/nn/nn_model.py`, `src/nnx/trainer/trainer.py` | High | `on_train_end` callbacks ran only on the happy path. | Callback-owned resources could leak or miss finalization after training failures. | Add shared callback finalizer used by both training loops. | Fixed | Exception-path regression tests plus full pytest suite. |
| OM-003 | API contract / error handling | `src/nnx/nn/nn_model.py`, `src/nnx/viz/netron.py` | Medium | ONNX export used broad `TypeError` fallback for the `dynamo` argument. | Real exporter/model `TypeError`s could be masked as compatibility fallback. | Inspect `torch.onnx.export` signature before passing `dynamo`. | Fixed | Regression test preserves exporter `TypeError`; full pytest suite. |
| OM-004 | Correctness / examples | `src/nnx/vis_utils.py` | Medium | t-SNE visualization used only the first test batch, had unsafe small-sample perplexity, and lacked deterministic defaults. | Visualizations could silently ignore requested samples or fail on small datasets. | Collect logits across batches, validate sample count, cap perplexity, and default `random_state=0`. | Fixed | Regression test plus full pytest suite. |
| OM-005 | Public API consistency | `src/nnx/surgery/drop_layer.py` | Medium | `drop_layer(..., importance=...)` silently ignored scorer input for a single string target. | Caller intent was discarded without warning. | Reject `importance` with a single string target and explain valid usage. | Fixed | Regression test plus full pytest suite. |
| OM-006 | Documentation accuracy | `README.md`, `docs/architecture.md` | Medium | Architecture docs implied interactive hover behavior for static checked-in artifacts. | Documentation overstated the artifact capability. | Reword static artifact description and keep diagram references accurate. | Fixed | MkDocs strict build. |
| OM-007 | Numbered documentation | `docs/surgery.md` | Medium | A section heading was unnumbered while numbered docs enforcement was enabled. | Heading hierarchy was inconsistent with the run standard. | Number the section and renumber the following heading. | Fixed | MkDocs strict build; manual heading scan. |
| OM-008 | Documentation indexing | `docs/external-contracts.md`, `README.md`, `mkdocs.yml` | Medium | External dependency contract assumptions were not centralized. | Dependency API checks were scattered and hard to audit. | Add a contract ledger and link it from README and MkDocs nav. | Fixed | MkDocs strict build. |
| OM-009 | Supply-chain security | `.github/workflows/security.yml`, `.github/dependabot.yml` | Medium | Dependency audit automation and update policy were absent. | Known vulnerable dependencies could go unnoticed until manual review. | Add pip-audit workflow and Dependabot coverage for pip and GitHub Actions. | Fixed | `pip-audit` local run; YAML validation. |
| OM-010 | Tooling accuracy | `.github/workflows/ci.yml` | Medium | Pyright advisory comment cited a stale approximate diagnostic count. | Stale CI comments reduce trust in advisory gates. | Make the advisory rationale count-independent. | Fixed | Pyright advisory rerun; YAML validation. |
| OM-011 | Example hygiene | `examples/03_custom_metrics.py` | Low | Example metric accepted an unused `Y` parameter. | It encouraged noisy unused-argument patterns. | Rename the parameter to `_Y`. | Fixed | Ruff and full pytest suite. |
| OM-012 | Build reproducibility | Packaging / CI | Medium | The project has no committed lockfile or constraints file for release dependency resolution. | Reproducible release environments remain dependent on resolver state. | Choose a lock/constraints strategy before pinning generated dependency state. | Deferred | Mitigated by Dependabot and pip-audit; requires project-level policy choice. |
| OM-013 | CI hardening | `.github/workflows/*.yml` | Low | GitHub Actions are pinned to moving tags rather than immutable SHAs. | Tag movement is a supply-chain risk. | Pin action references to SHAs or adopt an organization-wide policy. | Deferred | Dependabot added for visibility; SHA pinning deferred for policy/readability trade-off. |
| OM-014 | Test architecture | `tests/` | Low | Some behavior tests import deep implementation modules directly. | Deep imports can make public contract tests more brittle. | Split public-contract tests from implementation-unit tests over future passes. | Deferred | Existing tests remain passing; future pass should classify each deep import. |

## 1.4. Deferred Decisions

- Dependency lock strategy is deferred because the repository supports optional extras and a Python-version matrix; generating a single committed lock without selecting tooling such as pip-tools or uv would be a policy change.
- GitHub Action SHA pinning is deferred until the repository owner chooses between immutable-pinning strictness and tag readability with Dependabot monitoring.
- Deep test imports are deferred for classification so implementation-unit coverage is not accidentally weakened while moving public-contract assertions.
