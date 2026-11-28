# Cross-check of DOC, SYNC, SCOPE, DEAD and CLUT findings against the issue history

Totals: 39 NEW, 17 RELATED, 4 INCOMPLETE, 5 DELIBERATE (full or partial), 0 REGRESSION, 0 DUPLICATE.

## Related issues per finding (add as "Related:" lines)

| Finding | Related issues | Notes |
|---|---|---|
| DOC-03 | #557, #243 | #557 makes lone-surrogate refusal deliberate for derived ids, not for evaluate |
| DOC-04 | #525 | same gguf.md paragraph |
| DOC-05 | #217, #218 | |
| DOC-06 | #513 | owns §6.3 and the KDObjective alpha=1 doc contract |
| DOC-07 | #572 | |
| DOC-08 | #219 | |
| DOC-09 | #219, #521 | |
| DOC-11 | #518 | precedent: concepts §21 and the StreamingMetrics.update docstring |
| DOC-15 | #537 | its "Scope: out" covered export_bundle; 3ed0133 reversed that |
| DOC-16 | #570 | |
| DOC-18 | #472 | pilot link stays out per #262 (not_planned) |
| DOC-19 | #220, #235 | api.md:3612 already states the SDK range in prose |
| DOC-22 | **INCOMPLETE** of #767 (DEC-013, option C landed in 0.4.4; no guide update); #509 | |
| DOC-24 | #472 | |
| DOC-25 | **INCOMPLETE** of #418 (dedupe alternative never done for the no-curated branch); #535 | |
| DOC-26 | #243 | |
| DOC-27 | #567 | prefer "see uv.lock" for non-tool rows |
| DOC-31 | #472, #570, #575 | the label scan catches only `[[...]]`; the bare memo name slipped |
| DOC-33 | **INCOMPLETE** of #759 | reword to "Python 3.10 (no longer supported) replaced class-body hooks…" |
| DOC-34 | #757, #567 | |
| DOC-35 | #418 | |
| SYNC-01 | #217 | |
| SYNC-02 | #571 | same "test passes when the behaviour breaks" class |
| SYNC-03 | #217, #218 | |
| SYNC-04 | **DELIBERATE** per #572 Scope Out ("Running import-only examples … policy is unchanged"); #570 | Revise: README listing is the default fix; running them needs owner approval; downgrade to Low |
| SYNC-05 | | the #572 policy is why no test catches it |
| SYNC-06 | #571 | |
| SYNC-07 | #571 | |
| SCOPE-01 | **DELIBERATE**: #219, #220, #243, #245, #234, #263; #262 not_planned | Revise: drop "bolted on" and High; Medium "eager import of ~5k lines"; extraction becomes a DEC; the lazy `__getattr__` needs a module-returning branch (`_LAZY_DOMAIN_NAMES` maps name → (module, extra) and returns getattr; `tests/test_imports.py:530` asserts `nnx.decisions is decisions`) |
| SCOPE-02 | #572, #755 | the lane is deliberate; the GPU coordination is not on record (a31db9d, d429011); removing the busy-skip restores #572's fail-not-skip intent |
| SCOPE-03 | **DELIBERATE** placement per #255 (FP32 feed-forward-only scope; `import nnx` works without the extra); #566, #573 | Retitle as eager-import + complexity smell; moving to tools/ becomes a DEC citing #255; same `__getattr__` fix |
| SCOPE-04 | #255, #525, #517 | maintained recently; option "document a custom runtime is required" matches #255 |
| DEAD-02 | | do not touch the `validation_ids` report key pinned by #570/#571 |
| DEAD-04 | #418 | |
| DEAD-05 | #210 | safe: #572 pins only monkeypatch `is_available` |
| DEAD-11 | #568 | the rewritten assert must keep the indent-6 check |
| DEAD-13 | #759 | **DROP**: the asserts guard #759's acceptance criterion |
| CLUT-02 | ~25 issues cite these files (#206, #211, #755, #757, …) | list the rename map in the PR body / CHANGELOG |
| CLUT-04 | #220, #225 | `uv sync --frozen --all-extras` is the documented CI-equivalent; need `--all-groups` everywhere |
| CLUT-05 | **DELIBERATE (partial)** per #573 (comment block naming the modules is required) and #575 (ids in comments stay) | Revise: delete only the narration (Megamerge/thekaveh/ml, overnight audit lines); keep the convention sentence, module→keyword comments and FEAT-0xx refs |

## NEW, with no related issue

DOC-01, DOC-02, DOC-10, DOC-12, DOC-13, DOC-14, DOC-17, DOC-20, DOC-21, DOC-23, DOC-28, DOC-29, DOC-30, DOC-32, DEAD-01, DEAD-03, DEAD-06, DEAD-07, DEAD-08, DEAD-09, DEAD-10, DEAD-12, DEAD-14, CLUT-01, CLUT-03.
