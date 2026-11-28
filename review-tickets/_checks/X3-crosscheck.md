# Cross-check of the SMELL findings against the issue history

Totals: 20 NEW, 25 RELATED, 8 INCOMPLETE, 8 DELIBERATE (full or partial).

| Finding | Relationship and issues | Required change |
|---|---|---|
| SMELL-01 | INCOMPLETE of #552; #443, #260 | bundles `_write` / `_fsync_directory` and `history._fsync_path` already delegate to `_full_fsync` (#552). Keep `bundles._fsync_directory` as a test seam. Leftovers: `_fsync_file` vs `_fsync_path`, `os.fsync` in `_atomic_write_text`, the partial-file patterns, the missing shared module. Keep #552's "delete any flush → a test fails" coverage and the #443 tests. |
| SMELL-02 | RELATED #536 | step 1 moves the #536 shared-resume code as a unit; the NNModel-only DDP checks stay hooks |
| SMELL-03 | RELATED #536, #576, #765 | extractions from `UpdateEngine.commit` must keep the in-place path; the two loops stay separate by design |
| SMELL-05 | RELATED #507 | #507's part-2 tests (`-W error`) |
| SMELL-06 | INCOMPLETE of #536; #541 | #536 already shares `_resume_fit`, `_commit_epoch`, `_save_final_last` and `_update_tqdm_postfix`; put the new helpers beside them; do not merge the loops |
| SMELL-07 | INCOMPLETE of #551; #762, #521 | the depth walk is already shared; only the parse wrappers remain; keep `bundles._MAX_DEPTH = 64` |
| SMELL-08 | RELATED #520 | keep `require_count` on `chunk_size` |
| SMELL-09 | RELATED #515 | |
| SMELL-10 | RELATED #536, #215, #206 | `named=` drives the `_owner` message suffix; keep those messages; pin the sidecar bytes |
| SMELL-11 | RELATED #509, #767 | keep #767's refusal before anything is written |
| SMELL-12 | RELATED #441, #509 | |
| SMELL-13 | **DELIBERATE (partial)** #562, #764 | the refresh on reading `params` is #562's chosen design; limit the finding to bundling the 5 attributes into a tracker and to the double fingerprint |
| SMELL-14 | RELATED #562 (origin), #528 | |
| SMELL-16 | RELATED #563, #558, #209, #551 | `benchmark._FrozenDict` already subclasses `_FrozenMapping`; any shared class must keep the non-recursive `__reduce__` (depth bound) |
| SMELL-17 | RELATED #522 | keep #522's missing-idps.csv rule |
| SMELL-19 | RELATED #206 | keep each reader's own absent result (`None` / `(None, None)`); never create `runs/<id>/` |
| SMELL-20 | RELATED #509, #767 | |
| SMELL-23 | INCOMPLETE of #564; #561 | the leftover is #564's optional "one reader" step, plus the graph `seed_count` slice |
| SMELL-27 | RELATED #522 | precedent |
| SMELL-29 | RELATED #556 | the input-placement copies already drifted once (MPS); keep #556's MPS tests |
| SMELL-32 | RELATED #532 | keep test_legacy_net_pickles |
| SMELL-33 | RELATED #545, #544 | the docstring already explains; keep the rename only |
| SMELL-34 | INCOMPLETE of #521 | readers already use `check_keys` / `parse_json` / `read_text`; `JsonArtifact.from_file` is optional; the leftover is subclassing |
| SMELL-35 | RELATED #214, #513, #542, #544 | **drop step 4** (the `_check_alpha` TypeError is deliberate per #542/#544) |
| SMELL-36 | **DELIBERATE (partial)** #782, #257, #560, #561 | #782 shared `placement_in`; the messages must differ; the component-state checks validate different schemas. Keep only `q_sample` and `freeze()`. |
| SMELL-37 | **DELIBERATE (partial)** #243, #518 | the benchmark's exact NLL (`epsilon=None`) is by design; pass the benchmark's own epsilon; keep the link_tasks floor as a decision |
| SMELL-38 | INCOMPLETE of #544; #231 | the viz sites are #544 leftovers; keep the messages pinned by test_viz_weight_histogram.py:84 and test_viz_activation.py:149 |
| SMELL-39 | RELATED #548 | refuse before any update |
| SMELL-40 | RELATED #224, #523, #524, #526 | do not change the members of `WEIGHT_READING_PARENTS` |
| SMELL-43 | INCOMPLETE of #557; #763 | #557 aligned jobs with benchmark. 3 differences remain: tuple of str; 0-d tensor; Sequence vs list. #557 treated benchmark as the reference. Keep "unknown → provider judges" and DEC-009's empty-batch outcome. |
| SMELL-44 | RELATED #572, #555, #576 | retarget #572's MPS RNG tests; compilation's `_Snapshot` is out of scope (hot path) |
| SMELL-45 | RELATED #261, #525, #517 | eps 1e-6 is pinned by #261 (test_generative_nn_model.py:387); low value |
| SMELL-46 | **DELIBERATE** #254, #578 | `LogitsChain` coexists by design; deprecating it is a DEC that reverses #254 |
| SMELL-47 | RELATED #528 (Problem contradicted) | PEFT checks only owned keys, so the lazy-param problem is unreachable; keep only the de-duplication |
| SMELL-48 | RELATED #201, #224 | keep #201's per-family ownership and errors |
| SMELL-49 | RELATED #521, #558 | error behaviour equivalent |
| SMELL-50 | RELATED #211, #219, #244 | equal by spec |
| SMELL-51 | RELATED #557, #521 | |
| SMELL-52 | **DELIBERATE** error types: #525, #554, #524, #523 | drop step 2's "rule"; record the per-API types as decided; keep step 1 (the walk) |
| SMELL-53 | RELATED #542, #544 | |
| SMELL-55 | RELATED #513, #542 | keep the Mixup-only token-id refusal outside the helper |
| SMELL-57 | RELATED #224, #554 | keep the `layer_pattern` keyword named by #554's message |
| SMELL-60 | RELATED #516, #577, #766, #225 | prefer a torch-only `_dtypes.py` |
| SMELL-61 | RELATED #249 | |

## NEW

SMELL-04, 15, 18, 21, 22, 24, 25, 26, 28, 30, 31, 41, 42, 54, 56, 58, 59.
