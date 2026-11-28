# Ticket plan: review of develop f486c85..9ad461e

**How to read this plan:**
- The numbering continues the board: FIX-104 onward, FEAT-059 onward, DEC-015 onward.
- Every ticket also gets the label `review: 2026-11-27`.
- "Sources" are finding IDs in `/home/user/NNx/REVIEW-develop-2026-11-27.md`.
- Check results live in `/tmp/claude-0/-home-user-NNx/11f05a29-8365-5138-9342-787fd28b7c18/scratchpad/check/`:
  - `X1-crosscheck.md`: BUG and CI findings against the board
  - `X2-crosscheck.md`: DOC, SYNC, SCOPE, DEAD and CLUT findings against the board
  - `X3-crosscheck.md`: SMELL findings against the board
  - `H1-summary.md` and `H2-summary.md`: the worker's blocking questions and conflicts
- **Every writer must:**
  - apply the cross-check's "Required change" for each source;
  - answer the worker's blocking questions for each source, by reading the code at 9ad461e.

Dropped or merged findings (no ticket of their own):
- **DEAD-13:** dropped. The asserts guard #759's acceptance criterion.
- **DEAD-10:** resolved by FIX-121, because `_save_tensors` gains a real caller.
- **SMELL-42:** merged into FEAT-060.
- **SMELL-03:** the hotspots with no concrete fix are not ticketed. They are `Trainer._train_impl`, `NNTabularDataset.__post_init__`, `generate`, `_resume_fit`, `default_train_step`, `_match_lazy_module`, `StreamingMetrics.update` and `UpdateEngine.commit`.
- **CI-02 to CI-06, and DOC-07:** they live inside DEC-015's option B and option A acceptance criteria.
- **DOC-08, DOC-09 items 1–3, DOC-10:** inside DEC-016.

Conflict resolutions (binding on writers):
1. **FIX-107 vs FIX-108.**
   - FIX-107 adds `_unread_state_keys(net, example_input)` in `src/nnx/nn/nn_model.py`, replacing `_unexported_state_keys`. It takes no file path.
   - FIX-108 (staging directory) only changes where torch writes. It calls `_unread_state_keys` exactly as FIX-107 left it.
   - BUG-05 and BUG-06 become acceptance criteria of FIX-107: no per-key NumPy conversion remains, and the check is the same for both exporters.
2. **BUG-12 + BUG-13** are one ticket (FIX-116).
   - It defines `_create_exclusive(directory, prefix, suffix, mode)` and `_atomic_write_text(..., plain_mode: bool = False)`.
   - It replaces the chmod in `_commit_partial` with "keep the existing destination's mode if `path` exists, else rely on the umask applied at create time".
   - It does NOT change which call sites use plain mode. That policy is DEC-021.
3. **FIX-111 (BUG-09)** implements docs-only option (a), the project's recorded stance (#260; concepts §4.2). It describes the ordering after FIX-110, which moves the tokenizer removal after the manifest.
4. **FIX-112 (BUG-17)** takes the non-optional path: the ModelSpec branch returns `None`, and there is no `meta_safe` flag. The DOC-02 text is written for that outcome.
5. **FIX-113 (BUG-18)** adds the `skip=` keyword to `_warn_at_user_frame` in `nn/callbacks.py`. FEAT-082 (SMELL-04) moves the function later, with `skip=` included.
6. **FIX-105 / FIX-106** module docstring (DOC-26): FIX-106 writes the final text, which mentions the report-version exception from FIX-105 and the `same_records` keyword.
7. **SMELL-06 before SMELL-03:** FEAT-081 lands before FEAT-079.
8. **SMELL-11 → SMELL-30 → SMELL-02 step 2:** FEAT-091 → FEAT-108 → FEAT-075.
9. **DEAD-01 → SMELL-18 → SMELL-17:** FEAT-063 → FEAT-099 → FEAT-098.
10. **SMELL-38 (decisions part) before SMELL-58:** FEAT-114 → FEAT-134.
11. **SMELL-02 moves (FEAT-074 to FEAT-078)** come last among `nn_model.py` refactors. Each depends on the other `nn_model` FEAT tickets that touch the same code (listed per ticket).
12. **SMELL-15 (FEAT-095)** hoists `nn_model.py` local imports EXCEPT `..distributed`. Those are removed by FEAT-077 (the DDP move).
13. **DEAD-14 (FEAT-067)** deletes all 11 inert `noqa: BLE001` markers. FEAT-083 (SMELL-05) must not add any back.
14. **SMELL-35 signature:** `_kd_loss(ctx, m, X, Y, student_logits, teacher_logits, *, alpha, temperature, extra_loss=0.0)`. The report's version had a default argument before a non-default one, which is invalid Python.
15. **SMELL-41 part b (FEAT-121)** is blocked by DEC-017 and runs after every other `decisions/benchmark.py` ticket.

## DEC tickets

Labels for all DEC tickets: `question` plus the listed areas.

| ID | Title | Areas | Pri | Size | Sources | Depends | Writer notes |
|---|---|---|---|---|---|---|---|
| DEC-015 | Decide whether the MPS workflow keeps its ComfyUI and Ollama GPU coordination | github_actions, area: packaging | P2 | M | SCOPE-02, CI-02, CI-03, CI-04, CI-05, CI-06, DOC-07 | FIX-130 | A (recommended): remove `gpu-check` and the lock script, add the `NNX_MPS_GPU_WRAPPER` repository variable (SCOPE-02 steps), rewrite CONTRIBUTING. B: keep it and fix the probe (CI-02), fail red when busy (CI-03), probe under the lock (CI-04), `importorskip` fcntl (CI-05), pyright entry (CI-06), with DOC-07 text that matches. Both options restore #572's "fails, not skips" (CI-03 is a regression of #572). Each option gets its own exact acceptance criteria. Related #572, #755. |
| DEC-016 | Decide whether benchmark records keep a provider's own false probability | area: decisions | P3 | M | BUG-16, DOC-08, DOC-09 (items 1–3), DOC-10 | — | A (recommended): persist `p_false` in `Record` (emit only when given) and drop `compare=False`; give the exact doc texts. B: revert #857 and make the docs say `1 - p_true`. Note #243's record format (`nnx.decision-record/1`): 0.5.0 refuses unknown keys. Related #219, #243. |
| DEC-017 | Decide whether nnx.decisions moves to its own distribution | area: decisions, area: packaging | P3 | L | SCOPE-01 (long-term) | FEAT-059, FEAT-060 | Deliberate per #219, #220, #243, #245, #234, #263. A (recommended): keep it in NNx, loaded lazily (FEAT-059). B: extract it to a separate package with a shim. List what B moves (docs, examples, extra). |
| DEC-018 | Decide whether nnx.export_conformance moves out of the runtime package | area: interoperability, area: packaging | P3 | S | SCOPE-03 (step 4) | FEAT-059 | #255 made it public. A (recommended): keep it, loaded lazily. B: move it to `tools/` in the next major release. |
| DEC-019 | Decide how export_bundle treats run configs nested deeper than its 64-level bound | area: persistence | P3 | S | BUG-22 (part 1) | FIX-122 | #551 kept 64 on purpose, but the premise "deeper than any NNx state" is false (spec configs allow 256). A (recommended): refuse in `export_bundle` before staging, with a message naming the bound. B: raise `_MAX_DEPTH` to `MAX_JSON_DEPTH` plus the wrapper levels. C: keep it and document it in the `export_bundle` docstring. |
| DEC-020 | Decide whether NNx adds save and load helpers for DoRA adapters | area: peft | P3 | S | BUG-19 (step 1) | FIX-119 | #201 scoped this out. A (recommended): keep the full `state_dict` round trip. B: add `save_dora_weights` / `load_dora_weights`, with exact ACs. |
| DEC-021 | Decide which permission bits NNx gives the files it writes | area: persistence | P3 | S | SMELL-31 | FIX-116 | A (recommended): user exports get the plain-open mode, run-internal files stay 0600. B: everything 0600. C: everything plain mode. List the call sites per option. |
| DEC-022 | Decide whether the .maintenance/ ledgers stay in the repository | area: documentation | P3 | S | CLUT-01 | — | A (recommended): remove them. B: move them to the wiki. C: keep them with a "historical record" header. |
| DEC-023 | Decide whether dev and docs tooling leave the published PyPI extras | dependencies, area: packaging | P3 | M | CLUT-04 | — | A: move to `[dependency-groups]` and add `--all-groups` everywhere `--all-extras` is documented (#220, #225 convention). B (recommended): keep them. Exact ACs for A. |
| DEC-024 | Decide whether LogitsChain is deprecated in favour of OrderedLogitsPipeline | area: inference | P3 | S | SMELL-46 | — | #254 kept both by design; #578. A (recommended): keep both and add docstring cross-links. B: deprecate (SMELL-46 steps). |
| DEC-025 | Decide which input-modality rule nnx.decisions shares | area: decisions | P3 | S | SMELL-43 | — | Three differences remain after #557. A: jobs' rule. B (recommended): benchmark's rule (#557's reference). Both keep "unknown → the provider judges" and DEC-009's empty-batch outcome. List per-option behaviour changes and tests. |
| DEC-026 | Decide one probability floor for link-task NLL | area: evaluation, area: graph | P3 | S | SMELL-37 (step 3) | FEAT-113 | Floors today: 1e-300 (`link_tasks`), 1e-12 (the named metric), exact (benchmark, #243). Changing it changes reported values. A (recommended): keep 1e-300 and document it. B: switch to `NLL_EPSILON` (1e-12). |
| DEC-027 | Decide whether train_contrastive becomes a thin wrapper over NNModel.train | area: training | P3 | M | SMELL-53 | FIX-124 | A (recommended): keep it. B: rewrite it on top of `NNModel.train` with `text_contrastive_train_step_factory`. Risk: a different RNG trajectory. |

## FIX tickets

| ID | Title | Labels | Pri | Size | Sources | Depends | Writer notes |
|---|---|---|---|---|---|---|---|
| FIX-104 | Score samples whose input has no stable digest instead of failing evaluate() | bug, area: decisions | P1 | S | BUG-01, DOC-03 (sample part) | — | Rewrite `test_evaluate_names_a_sample_whose_input_has_no_stable_digest`. Related #557, #243. |
| FIX-105 | Version benchmark reports so reports saved by 0.5.0 still compare | bug, area: decisions | P1 | M | BUG-02, DOC-03 (comparison part) | FIX-104 | Covers revisions and prompt identities too. Related #243, #762. |
| FIX-106 | Let compare_reports compare reports of different revisions on request | bug, area: decisions | P2 | S | BUG-14, DOC-26 | FIX-105 | `same_records: bool = True`, keyword-only. Write the final module docstring text. Related #243. |
| FIX-107 | Report only state the forward pass never reads in to_onnx, and warn after the file is written | bug, area: interoperability | P1 | M | BUG-04, BUG-05, BUG-06 | — | See conflict 1. Write the test first; it needs torch. |
| FIX-108 | Write to_onnx's external data under the final file name | bug, area: interoperability | P1 | M | BUG-03, DOC-16 | FIX-107 | Staging directory. The optional cleanup of old external files is Scope: Out. Add the CI-upload hidden-file effect (#255). Related #255. |
| FIX-109 | Keep reading run directories whose id ends in a dot or a space | bug, area: persistence | P1 | M | BUG-07 | — | The writer must list every `_validate_run_id` caller and classify it as read or create; the create sites use `_validate_new_run_id`. Related #546, #522. |
| FIX-110 | Remove a stale tokenizer.json only after the new Hub save is in place, and only if NNx wrote it | bug, area: persistence | P1 | S | BUG-08 | — | Related #260. |
| FIX-111 | State what a failed save_pretrained leaves behind | documentation, area: persistence | P2 | S | BUG-09 (option a) | FIX-110 | Exact CHANGELOG and docstring texts. Related #260, #509. |
| FIX-112 | Build a registered model factory once per from_pretrained | bug, area: model-integration | P2 | S | BUG-17, DOC-02 | — | See conflict 4. Related #216. |
| FIX-113 | Attribute the Hub "no manifest" warning to the caller's line | bug, area: persistence | P3 | S | BUG-18 | — | See conflict 5. Related #522. |
| FIX-114 | Flush committed export files through a writable descriptor | bug, area: persistence | P1 | S | BUG-10 | — | Related #552, #443, concepts §14.3. |
| FIX-115 | Flush standalone artifact files with F_FULLFSYNC on macOS | bug, area: persistence | P3 | S | BUG-11 (revised) | FIX-114 | Revised per #552 (X1). Update concepts §4.2 and §14.3. |
| FIX-116 | Apply the current umask at write time and keep an existing file's mode | bug, area: persistence | P2 | M | BUG-12, BUG-13 | FIX-114 | See conflict 2. Related #260. |
| FIX-117 | Write the tokenizer file through a unique partial file | bug, area: persistence | P3 | S | BUG-26 | FIX-116 | Related #260, #522. |
| FIX-118 | Refuse extra_metrics in task-less alpha=1 distillation and document the loss-only record | bug, area: training | P2 | S | BUG-15, DOC-06 | — | Related #513, #542. |
| FIX-119 | Warn when save_lora_weights skips a DoRA magnitude | bug, area: peft | P3 | S | BUG-19 (revised, step 2 only) | — | Related #201, #225, DEC-020. |
| FIX-120 | Leave classification metrics unset in paradigm steps that do not compute them | bug, area: training | P2 | S | BUG-20 (revised), DOC-31 (diffusion bullet) | — | Keep DPO's `extra` (#82). Add a CHANGELOG "Changed" entry. Related #257, #82, #212, #251. |
| FIX-121 | Write real safetensors from export_to_safetensors without the safetensors package | bug, area: interoperability | P2 | M | BUG-21, DEAD-10 | — | Rewrite `test_export_to_safetensors_roundtrip_torch_save_fallback`. Related #237. |
| FIX-122 | Parse bundle JSON with the shared strict parser | bug, area: persistence | P3 | S | BUG-22 (part 2), SMELL-07 (bundles step) | — | Keep `_MAX_DEPTH = 64`. Related #551, #218. |
| FIX-123 | Validate top_p as a finite real number | bug, area: inference | P3 | S | BUG-23 | — | Leftover of #544. Related #254. |
| FIX-124 | Validate train_contrastive's n_epochs and batch_size as counts | bug, area: training | P3 | S | BUG-24 | — | Leftover of #542 and #544. |
| FIX-125 | Freeze TeacherRecord's provenance mapping | bug, area: training | P3 | S | BUG-25 (revised: `_FrozenMapping`) | — | Pickle and deepcopy tests. Related #558, #244. |
| FIX-126 | Hash bfloat16 weights in export-conformance digests | bug, area: interoperability | P3 | S | BUG-27 | — | fp32 digests stay byte-identical; for bf16, view as int16. Related #255. |
| FIX-127 | Register every optional extra in nnx._optional | bug, area: packaging | P2 | S | BUG-28 (part 1) | — | `DOMAIN_EXTRAS` entries plus `require_onnxscript`; no caller changes. Leftover of #249. |
| FIX-128 | Route quantize, prune, GGUF, Jev and search imports through nnx._optional.require | bug, area: packaging | P2 | M | BUG-28 (part 2) | FIX-127 | Keep every hint text that tests assert (#220, #234). |
| FIX-129 | Route embeddings, viz, onnxscript and Hub imports through nnx._optional.require | bug, area: packaging | P2 | M | BUG-28 (part 3) | FIX-127 | Same rule on hint text. Related #570. |
| FIX-130 | Keep Dependabot pull requests off the self-hosted runners | bug, github_actions, area: packaging | P1 | S | CI-01 | — | Add the operational note for the owner (runner account). Related #572, #420, #757. |
| FIX-131 | Document DPO's refusal of responses with no valid token | documentation, area: training | P1 | S | DOC-01 | — | |
| FIX-132 | Document GGUF export's refusal of state no tensor carries | documentation, area: interoperability | P2 | S | DOC-04 | — | Related #525. |
| FIX-133 | Document the test_split_id deprecation and the new threshold refusals | documentation, area: evaluation | P2 | S | DOC-05 | — | Related #217, #218. |
| FIX-134 | Document StreamingMetrics' refusal of unnormalized probability rows | documentation, area: evaluation | P3 | S | DOC-11 | — | Related #518. |
| FIX-135 | Bring the Hub guide and docstrings up to date with manifests and refusals | documentation, area: persistence | P3 | S | DOC-12, DOC-13, DOC-21, DOC-22 | — | DOC-22 is a leftover of #767. Related #509. |
| FIX-136 | Add the missing CHANGELOG entries and fix two train() shorthands | documentation, area: documentation | P3 | S | DOC-14, DOC-20 | — | |
| FIX-137 | Document export_bundle's resume-point verification and "best" refusal | documentation, area: persistence | P3 | S | DOC-15 | — | Related #537. |
| FIX-138 | Re-sync the README with the quickstart split | documentation, area: documentation | P3 | S | DOC-17 (option a: a note under the H1), DOC-18, DOC-24 | — | Related #472. |
| FIX-139 | List SDK_RANGE and SUPPORTED_QUANTIZATIONS in the API reference | documentation, area: documentation | P3 | S | DOC-19 | — | The coverage test is Scope: Out. Related #220, #235. |
| FIX-140 | Correct stale tooling comments and versions in config and contributor docs | documentation, area: packaging | P3 | S | DOC-23, DOC-27, DOC-29, DOC-34 | — | Related #567, #757. |
| FIX-141 | Deduplicate release-please's changelog list when nothing is curated | bug, area: packaging | P3 | M | DOC-25 | — | Leftover of #418. Related #535. |
| FIX-142 | Say what the embeddings extra installs and why | documentation, area: documentation | P3 | S | DOC-28 (step 1) | — | Dropping the dependency is Scope: Out. |
| FIX-143 | Correct stale code comments and docstrings | documentation, area: documentation | P3 | S | DOC-30 (option a), DOC-31 (comment bullets: qat, `_targets`, ollama memo, gguf writer comments), DOC-32, DOC-33 (revised wording) | — | Related #472, #570, #759. |
| FIX-144 | Call fit_temperature with test_split_id in run_bundle.py and fail example tests on the deprecation | bug, area: evaluation | P2 | S | SYNC-01, SYNC-02 | — | Related #571. |
| FIX-145 | Pass test_split_id in the tests that still omit it | bug, area: evaluation | P2 | M | SYNC-03 | FIX-144 | |
| FIX-146 | List the import-only examples and add the OpenMP guard to example 13 | documentation, area: documentation | P3 | S | SYNC-04 (revised), SYNC-05 | — | #572's policy stays; running these examples in CI is Scope: Out. |
| FIX-147 | Stop skipping tests for packages they do not use | bug, area: packaging | P3 | S | SYNC-06, SYNC-07 | — | Related #571. |
| FIX-148 | Say in the README that GGUF and Ollama exports need a custom runtime | documentation, area: interoperability | P3 | S | SCOPE-04 | — | First check whether README already says this; if it does, set status DROP. Related #255. |
| FIX-149 | Match pruning layer patterns on every registration path and validate sparsity | bug, area: model-transforms | P3 | M | SMELL-57 (steps 2–3) | — | Keep the `layer_pattern` keyword. Related #224, #554. |

## FEAT tickets

Labels for all FEAT tickets: `enhancement` plus the listed area. Several titles end in "(part 1/2/3…)" where a split was needed.

| ID | Title | Area | Pri | Size | Sources | Depends | Writer notes |
|---|---|---|---|---|---|---|---|
| FEAT-059 | Load nnx.decisions and nnx.export_conformance lazily on import nnx | packaging | P3 | M | SCOPE-01 (step 1), SCOPE-03 (step 1) | — | Needs a module-returning branch in `__getattr__`. `tests/test_imports.py:530` must hold, and #255's "import nnx works without the extra". |
| FEAT-060 | Move abstention.decide into nnx.decisions and break the import cycle | decisions | P3 | M | SCOPE-01 (step 2), SMELL-42 | FEAT-059 | |
| FEAT-061 | Split export_conformance.validate_record and share one SHA-256 check | interoperability | P3 | M | SCOPE-03 (steps 2–3) | — | Related #255, #566. |
| FEAT-062 | Drop the unused libcairo2 step from the hosted MPS gate | packaging (+github_actions) | P3 | S | CI-07 | — | Related #445. |
| FEAT-063 | Remove dead code in nn_run, distributed, objectives, peft, benchmark and deepen | training | P3 | S | DEAD-01, DEAD-02, DEAD-06, DEAD-07, DEAD-08, DEAD-09 | — | Leave the `validation_ids` report key alone (#570, #571). |
| FEAT-064 | Remove helpers that only tests use | packaging | P3 | S | DEAD-03, DEAD-11, DEAD-12 | — | Keep #568's indent-6 check. |
| FEAT-065 | Remove the Python 3.8 version shim and its stale comment | packaging | P3 | S | DEAD-04, DOC-35 | — | Related #418. |
| FEAT-066 | Remove torch-version guards below the torch 2.4 floor | numerics | P3 | S | DEAD-05 | — | Related #210. |
| FEAT-067 | Remove the inert noqa: BLE001 markers | packaging | P3 | S | DEAD-14 (delete them) | — | Enabling BLE is Scope: Out. |
| FEAT-068 | Rename the test_pass2_*_series modules after what they test | packaging | P3 | M | CLUT-02 | FIX-109 | Put the rename map in the PR body. |
| FEAT-069 | Shrink the README banner under the 500 KB file limit | documentation | P3 | S | CLUT-03 | — | Name the tool exactly, with a fallback (Pillow quantize). |
| FEAT-070 | Trim audit-history narration from the PyPI keyword comments | packaging | P3 | S | CLUT-05 (revised) | — | Related #573, #575. |
| FEAT-071 | Add a shared atomic-I/O module (part 1 of 3) | persistence | P2 | M | SMELL-01 (part 1) | FIX-114, FIX-115, FIX-116, FIX-117 | Related #552, #443, #260. |
| FEAT-072 | Move the checkpoint, run, history and bundle writers onto nnx._atomic_io (part 2 of 3) | persistence | P2 | M | SMELL-01 (part 2) | FEAT-071 | Keep `bundles._fsync_directory` as a test seam (#552). |
| FEAT-073 | Move the remaining atomic writers onto nnx._atomic_io and drop their local imports (part 3 of 3) | persistence | P2 | M | SMELL-01 (part 3) | FEAT-072 | |
| FEAT-074 | Move the resume machinery out of nn_model.py (part 1 of 5) | training | P3 | L | SMELL-02 (step 1) | FEAT-081, FEAT-079, FEAT-088, FEAT-089, FEAT-090, FEAT-103 | Related #536. |
| FEAT-075 | Move the transform-replay helpers into nnx.transforms (part 2 of 5) | model-transforms | P3 | M | SMELL-02 (step 2) | FEAT-091, FEAT-108 | |
| FEAT-076 | Move the Hub I/O helpers into nnx/nn/_hub.py (part 3 of 5) | persistence | P3 | L | SMELL-02 (step 3) | FIX-110, FIX-112, FIX-113, FEAT-092, FEAT-101, FEAT-102 | |
| FEAT-077 | Move the DDP fit helpers into nnx/nn/_ddp_fit.py (part 4 of 5) | execution | P3 | L | SMELL-02 (step 4), SMELL-15 (distributed imports) | FEAT-074 | |
| FEAT-078 | Move the lazy-module descriptor into nnx.models (part 5 of 5) | model-integration | P3 | M | SMELL-02 (step 5) | FEAT-093, FEAT-094 | |
| FEAT-079 | Extract four helpers from NNModel._train_impl | training | P3 | M | SMELL-03 (`_train_impl`) | FEAT-081 | Related #536, #576, #765. |
| FEAT-080 | Extract the per-run body of observations_from_runs | evaluation | P3 | S | SMELL-03 (`observations_from_runs`) | — | |
| FEAT-081 | Share the remaining epoch-end blocks between NNModel and Trainer | training | P3 | M | SMELL-06 | — | Leftover of #536. Related #541. |
| FEAT-082 | Move _warn_at_user_frame to a standard-library-only module | packaging | P3 | S | SMELL-04 | FIX-113 | |
| FEAT-083 | Share one warn-while-failing helper | training | P3 | S | SMELL-05 | FEAT-082, FEAT-067 | Related #507. |
| FEAT-084 | Replace two raw warnings.warn calls with _warn_at_user_frame | persistence | P3 | S | SMELL-27 | FEAT-082 | Related #522. |
| FEAT-085 | Share the JSON parse wrappers of history and provenance | persistence | P3 | M | SMELL-07 (history and provenance) | FIX-122 | No change in strictness; add flags. Related #551, #762, #521. |
| FEAT-086 | Share one file-digest helper | persistence | P3 | S | SMELL-08 | — | Related #520. |
| FEAT-087 | Share validation and serialization between NNTrainParams and NNTrainerParams | configuration | P3 | M | SMELL-09 | — | Pin `state()` first. Related #515. |
| FEAT-088 | Pin the checkpoint sidecar bytes before refactoring NNCheckpoint.save (part 1 of 3) | persistence | P3 | S | SMELL-10 (part 1) | — | Test only. |
| FEAT-089 | Pass NNCheckpoint.save's training state as one value (part 2 of 3) | persistence | P3 | M | SMELL-10 (part 2) | FEAT-088 | Related #536, #215. |
| FEAT-090 | Derive single vs named optimizers instead of passing trainer= and named= (part 3 of 3) | training | P3 | M | SMELL-10 (part 3) | FEAT-089 | Keep the `_owner` messages. |
| FEAT-091 | Move _unportable_transform into nnx.transforms and name the QAT transform once | model-transforms | P3 | S | SMELL-11 | — | Related #509, #767. |
| FEAT-092 | Share the Hub strict-load sequence between the preflight and the load | persistence | P3 | M | SMELL-12 | FIX-112 | Explain the `call` parameter. Related #441, #509. |
| FEAT-093 | Bundle NNModel's lazy-module tracking into one object | model-integration | P3 | M | SMELL-13 (revised) | — | Related #562, #764. |
| FEAT-094 | Share the dry-run lazy-inference rules in nnx.models | model-integration | P3 | S | SMELL-14 | — | Related #562, #528. |
| FEAT-095 | Hoist nn_model.py's needless function-local imports | packaging | P3 | M | SMELL-15 (all but distributed) | FEAT-082 | |
| FEAT-096 | Share one FrozenMapping class (part 1 of 2) | configuration | P3 | M | SMELL-16 (part a) | — | Keep the non-recursive `__reduce__`. Related #563, #558, #209, #551. |
| FEAT-097 | Share one thaw function for frozen JSON values (part 2 of 2) | configuration | P3 | M | SMELL-16 (parts b and c) | FEAT-096 | |
| FEAT-098 | Read run.yaml and idps.csv through one helper each | persistence | P3 | M | SMELL-17 | FEAT-063, FEAT-099 | Related #522. |
| FEAT-099 | Share the metadata.yaml record reader | persistence | P3 | S | SMELL-18 | FEAT-063 | |
| FEAT-100 | Share the checkpoint-path guard of the training-state readers | persistence | P3 | S | SMELL-19 | — | Related #206. |
| FEAT-101 | Build the state dict once per save_pretrained | persistence | P3 | S | SMELL-20 | FIX-110 | Related #509, #767. |
| FEAT-102 | Name the tokenizer file once and annotate the Hub stub NoReturn | persistence | P3 | S | SMELL-21 | FIX-110 | |
| FEAT-103 | Remove small duplicates in nn_model and nn_run | training | P3 | S | SMELL-22, SMELL-24, SMELL-28 | — | |
| FEAT-104 | Remove small duplicates in probability, nets and evaluation records | numerics | P3 | S | SMELL-26, SMELL-32, SMELL-33 (rename only) | — | Related #532, #545. |
| FEAT-105 | Share one supervised batch reader between nn_model and paradigm steps | training | P3 | M | SMELL-23 | — | Related #564, #561. |
| FEAT-106 | Replace magic-index module state with named fields | configuration | P3 | S | SMELL-25 | — | |
| FEAT-107 | Build to_onnx's and netron_export's export arguments in one place | interoperability | P3 | M | SMELL-29 | FIX-108 | Name the shared placement helper. Related #556. |
| FEAT-108 | Replay transforms on a bare module in the Hub preflight | model-transforms | P3 | S | SMELL-30 | FEAT-091 | |
| FEAT-109 | Make SplitManifest and Standardizer subclass JsonArtifact | data | P3 | M | SMELL-34 (revised) | — | Pin digests first. Related #521. |
| FEAT-110 | Share the knowledge-distillation loss and move KDObjective beside it | training | P3 | M | SMELL-35 (steps 1–3) | FIX-118 | See conflict 14. Related #214, #513. |
| FEAT-111 | Share forward noising between the diffusion step and objective | training | P3 | S | SMELL-36 (`q_sample`) | FIX-120 | Related #257. |
| FEAT-112 | Share one freeze helper for frozen modules | training | P3 | S | SMELL-36 (`freeze`) | — | Related #782. |
| FEAT-113 | Reuse the shared NLL, Brier and macro-F1 in the decision benchmark | decisions | P3 | M | SMELL-37 (steps 1–2) | — | Exact NLL by default (#243). Related #518. |
| FEAT-114 | Validate decision counts with require_count (part 1 of 4) | decisions | P3 | M | SMELL-38 (decisions sites) | — | Related #544, #231. |
| FEAT-115 | Validate generation, diffusion and plan counts with require_count (part 2 of 4) | configuration | P3 | S | SMELL-38 (generation, diffusion, plans) | — | |
| FEAT-116 | Validate viz counts with require_count (part 3 of 4) | documentation (viz → use area: inference) | P3 | S | SMELL-38 (viz sites) | — | Leftover of #544; keep the messages pinned by its tests. |
| FEAT-117 | Share one _count helper across task modules (part 4 of 4) | training | P3 | S | SMELL-38 (ranking, link, tasks, lm) | — | |
| FEAT-118 | Share task-owned metric registration and the extra_metrics refusal | evaluation | P3 | M | SMELL-39 | — | Related #548. |
| FEAT-119 | Move the module-path walkers out of peft/_targets.py | peft | P3 | M | SMELL-40 | — | Related #224, #523, #524, #526. |
| FEAT-120 | Split DecisionJob's _Runner.arun into launch, harvest and drain | decisions | P3 | M | SMELL-41 (part a) | — | |
| FEAT-121 | Split decisions/benchmark.py by responsibility | decisions | P3 | L | SMELL-41 (part b) | DEC-017, FEAT-113, FEAT-114, FEAT-126, FEAT-127, FEAT-134, FEAT-136 | |
| FEAT-122 | Reuse the shared RNG snapshot in lr_finder and viz.summary | numerics | P3 | S | SMELL-44 | — | `_Snapshot` in compilation is Scope: Out. Related #572, #555, #576. |
| FEAT-123 | Map GGUF quantization labels once and read the FFN width from the model | interoperability | P3 | S | SMELL-45 (steps 1–2), DOC-31 (gguf import hoist) | — | The error-type change is Scope: Out. Related #261, #525, #517. |
| FEAT-124 | Share one reshaped-key check between PEFT and finetune loading | peft | P3 | S | SMELL-47 (revised) | — | Related #528. |
| FEAT-125 | Share save and load bodies across the four PEFT adapter families | peft | P3 | M | SMELL-48 | FIX-119 | Related #201, #224. |
| FEAT-126 | Share JSONL record read and write and the key checks | persistence | P3 | M | SMELL-49 | — | Related #521, #558. |
| FEAT-127 | Define the 1e-6 probability tolerance once | numerics | P3 | S | SMELL-50 | — | Related #211, #219, #244. |
| FEAT-128 | Cross-reference duplicate digest forms and value types in docstrings | documentation | P3 | S | SMELL-51, SMELL-59 | — | Related #557, #521. |
| FEAT-129 | Share the legacy reparameterization walk and record per-API error types | model-transforms | P3 | S | SMELL-52 (revised) | — | Related #523, #524, #525, #554. |
| FEAT-130 | Share the NNModel unwrap in nnx.viz and hoist its imports | inference | P3 | S | SMELL-54, DOC-31 (viz bullet) | — | |
| FEAT-131 | Share the post-mix step of Mixup and CutMix | training | P3 | S | SMELL-55 | FIX-120 | Related #513, #542. |
| FEAT-132 | Share DPO's empty-row lookup | training | P3 | S | SMELL-56 | FIX-131 | |
| FEAT-133 | Share freeze and unfreeze's pattern loop | training | P3 | S | SMELL-57 (step 1) | — | |
| FEAT-134 | Share one non-empty-text validator | configuration | P3 | S | SMELL-58 | FEAT-114 | |
| FEAT-135 | Use full_precision for half-precision widening | numerics | P3 | S | SMELL-60 | — | Related #516, #577, #766, #225. |
| FEAT-136 | Hoist needless function-local imports in nnx.decisions | decisions | P3 | S | SMELL-61 (benchmark, jobs) | FEAT-060 | Related #249. |
| FEAT-137 | Hoist needless function-local imports in offline distillation | training | P3 | S | SMELL-61 (offline_distillation) | — | |
| FEAT-138 | Hoist needless function-local imports in nnx.plans | fluent-api | P3 | M | SMELL-61 (plans.py) | — | |
| FEAT-139 | Hoist needless function-local imports in nnx.transforms | model-transforms | P3 | M | SMELL-61 (transforms.py) | FEAT-075 | |
| FEAT-140 | Hoist needless function-local imports in bundles and export_conformance | persistence | P3 | M | SMELL-61 (bundles, export_conformance) | FEAT-061 | |
| FEAT-141 | Add docstrings to the nnx.decisions members api.md lists without one | documentation | P3 | M | DOC-09 (item 4, decisions slice) | DEC-016 | |

## Writer assignment

| Writer | Tickets |
|---|---|
| W1 | DEC-015 … DEC-027 |
| W2 | FIX-104 … FIX-118 |
| W3 | FIX-119 … FIX-149 |
| W4 | FEAT-059 … FEAT-090 |
| W5 | FEAT-091 … FEAT-115 |
| W6 | FEAT-116 … FEAT-141 |
