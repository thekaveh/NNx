# Haiku readiness, part 1 (BUG, CI, DOC): key results

Totals: READY 47, NEEDS-DETAIL 7, NEEDS-DECISION 14, TOO-BIG 2.

## NEEDS-DETAIL
- BUG-03: what the post-export check calls after BUG-04; is optional step 3 in scope
- BUG-05: if BUG-04 deletes the check, is the counter test dropped
- BUG-07: which other new-run entry points need `_validate_new_run_id`
- BUG-13: where the existing mode is read and applied after BUG-12
- BUG-21: `tests/test_embeddings_faiss_export.py:247 test_export_to_safetensors_roundtrip_torch_save_fallback` enshrines the fallback; the docstring change is not listed
- BUG-25: MappingProxyType is not picklable; does TeacherRecord need to pickle
- BUG-27: bf16 digest format unspecified; "if bytes differ" needs torch

## NEEDS-DECISION
- BUG-09 (a/b), BUG-16 (A/B), BUG-20 (metrics= vs extra=; run.yaml impact)
- CI-02..06 and DOC-07 (gated on SCOPE-02)
- DOC-08, DOC-09, DOC-10 (depend on BUG-16)
- DOC-17 (a/b), DOC-30 (a/b)

## TOO-BIG
- BUG-28: split into (1) mapping + require_onnxscript, (2) per-module conversions, (3) hint text
- DOC-31: split into a comments PR and an imports PR

## Ambiguities in sections 0-3
1. "Pick the first option" vs items saying "pick one"
2. Rule 9 omits SCOPE-02
3. Branch per item vs per group
4. Alias rule for deleted helpers
5. G7 used twice
6. Abbreviated quotes with "..."
7. BUG-21 enshrined test not named
8. torch availability not stated
9. Which tickets need CHANGELOG
10. Full suite before PR?
11. Inconsistent Tests/Check labels; some tickets have no proof command (CI-04, DOC-03, 05, 06, 11, 13, 14, 15, 21, 22)

## Conflicts
- BUG-03 vs BUG-04 (function name)
- BUG-12 vs BUG-13 (chmod location)
- CI-03 vs DOC-07 "otherwise" text
- BUG-02 vs DOC-26 (version exception)
- BUG-08 vs BUG-09b (tokenizer not in backups)
- BUG-05 vs BUG-04
- BUG-12 vs SMELL-31 (policy not decided)
- BUG-17 optional meta_safe vs DOC-02 text
- SCOPE-02 pending or made

## Unexplained terms
A glossary is needed: partial files, fsync, ONNX exporters, Hub manifest, runs/best, decisions terms, training terms, helpers, CI variables, docs tooling.
