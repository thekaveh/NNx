# Cross-check of the BUG and CI findings against the issue history

All 196 issues predate PRs #849–#857, so no reviewed fix has its own ticket.

Totals: 4 DELIBERATE (BUG-11, BUG-19, and BUG-22 part 1, plus the BUG-20 step-2 contract), 1 REGRESSION (CI-03), 6 INCOMPLETE, 14 RELATED, 10 NEW.

| Finding | Relationship | Issues | Required change |
|---|---|---|---|
| BUG-01 | RELATED | #557, #243 | #557 makes custom inputs a served case elsewhere, which supports the fix |
| BUG-02 | RELATED | #243, #762 | #762 Option B is a precedent for documenting a break; `report_version` is justified; a CHANGELOG-only note is the fallback |
| BUG-03 | RELATED | #255 | add: the CI conformance upload (`upload-artifact`, no `include-hidden-files`) silently omits the hidden `.partial.data`, which breaks #255's "retains external-data files" |
| BUG-04/05/06 | NEW | | |
| BUG-07 | RELATED | #546, #522 | same class as #546, which fixed it by scoping the rule; #522's shared `_is_run_entry` predicate; its `test_pass2_n_series` cases must still pass; if the refusal is kept, the CHANGELOG follows #546's wording |
| BUG-08 | NEW | #260 principle | "Cleanup never unlinks a path the helper does not own" |
| BUG-09 | RELATED | #260, #522, #509 | **option (a) is the project's recorded stance** (no multi-file transaction; concepts §4.2); (b) needs an owner decision |
| BUG-10 | RELATED | #552, #443 | the guarantee it breaks is concepts §14.3: "On Windows the files are flushed with `os.fsync`" |
| BUG-11 | **DELIBERATE** | #552 | #552 chose plain `os.fsync` for run files and documented it (§4.2, §14.3). Reframe: run files are followed by a directory `F_FULLFSYNC` before LAST, so they are covered; the gap is standalone artifacts (split manifests, standardizers, calibration, provenance, `export_history_csv`, config.json, Modelfile). **Downgrade to Low.** If changed, edit §4.2 and §14.3. |
| BUG-12 | NEW | #260 | `_create_exclusive` must keep #260's guarantees (`tests/test_atomic_write_text.py`) |
| BUG-13 | NEW | | |
| BUG-14 | RELATED | #243 | FEAT-021's contract compared only split and metric identity; the owner may want `same_records=False` as the default |
| BUG-15 | RELATED | #513, #542 | the test must show #513's placeholder alpha=1 runs without `extra_metrics` still train |
| BUG-16 | RELATED | #219, #243 | B restores #219's contract; A changes the `nnx.decision-record/1` format defined by #243, so it needs a format note |
| BUG-17 | RELATED | #216 | the post-build shape check must still run before `load_state_dict` (`tests/test_model_factories.py`) |
| BUG-18 | RELATED | #522 | the same hand-picked stacklevel failure, fixed before |
| BUG-19 | **DELIBERATE** | #201, #225 | #201 Scope: "Out: A separate DoRA-magnitude artifact API"; the docs and lora.py:270 already warn. **Revise:** title "save_lora_weights on a DoRA net omits magnitude without a runtime warning", Low. Step 1 (`save_dora_weights`) becomes an owner decision. Keep step 2 (the warning). Step 3 is a nit (one tensor cannot half-load). |
| BUG-20 | INCOMPLETE of #257 + RELATED #82, #212, #251 | | **Revise step 2:** keep the DPO diagnostics in `extra` (#82's contract; `test_dpo_step_emits_reward_metrics_in_extra`; read by NNx Studio). Add a CHANGELOG "Changed" entry for 0.0 → None. |
| BUG-21 | NEW | #237 | "no pickle fallback" principle |
| BUG-22 | **DELIBERATE (part 1)** + NEW (part 2) | #551, #218 | #551 Scope Out: "The bundle limit of 64"; documented in `_json_depth.py`. Restate the problem: the bound's premise "deeper than any NNx state" is false, because spec configs may be up to 256 levels deep. Owner choice: refuse up front with a message that names the bound, or raise the bound. Keep the `1e999` part. |
| BUG-23 | INCOMPLETE of #544 | #254 | `LogitsStage._real` is the in-repo precedent |
| BUG-24 | INCOMPLETE of #542/#544 | rule from #231 | |
| BUG-25 | RELATED | #558, #244 | **Revise the fix:** `MappingProxyType` cannot be pickled or deep-copied (checked). Use `trainer.params._FrozenMapping` as #558 did. Add pickle and deepcopy tests. |
| BUG-26 | NEW | #260, #522 | |
| BUG-27 | NEW | #255 | |
| BUG-28 | INCOMPLETE of #249 | #220, #234, #570 | hint wording pinned by these |
| CI-01 | RELATED | #572, #420, #757 | stays High |
| CI-02 | NEW | #572 | |
| CI-03 | **REGRESSION** of #572 | | "fails, not skips"; can be downgraded to Low because the CPU pins still catch reverts |
| CI-04/05/06 | NEW | | |
| CI-07 | RELATED | #445 | the apt stall risk |
