# Review tickets: develop f486c85..9ad461e

142 tickets for the thekaveh/NNx board, written to the board's own format (Summary, Context, Acceptance criteria, Evidence, Scope, Dependencies) so a small worker agent can implement each one without guessing. They supersede the finding IDs in `REVIEW-develop-2026-11-27.md`; each ticket's `sources:` names the findings it covers.

| | P1 | P2 | P3 | Total |
|---|---:|---:|---:|---:|
| FIX | 9 | 14 | 23 | 46 |
| FEAT | 0 | 3 | 80 | 83 |
| DEC | 0 | 1 | 12 | 13 |
| **All** | 9 | 18 | 115 | 142 |

- **READY:** a worker can start once its `Blocked by` tickets are merged.
- **DECISION:** the owner records an option in a comment first; each option's worker steps are in the ticket.
- **Order:** a work order that respects every dependency, highest priority first (P1, then P2, then P3).
- **Labels:** every ticket carries the label `review: 2026-11-27`, which does not exist on the board yet.

| Order | ID | Title | Pri | Size | Status | Blocked by | Sources | Board history |
|---:|---|---|---|---|---|---|---|---|
| 1 | [FIX-104](FIX-104.md) | Score samples whose input has no stable digest instead of failing evaluate() | P1 | S | READY | — | BUG-01, DOC-03 | #557, #243 |
| 2 | [FIX-105](FIX-105.md) | Version benchmark reports so reports saved by 0.5.0 still compare | P1 | M | READY | FIX-104 | BUG-02, DOC-03 | #243, #762 |
| 10 | [FIX-106](FIX-106.md) | Let compare_reports compare reports of different revisions on request | P2 | S | READY | FIX-105 | BUG-14, DOC-26 | #243 |
| 3 | [FIX-107](FIX-107.md) | Report only state the forward pass never reads in to_onnx, and warn after the file is written | P1 | M | READY | — | BUG-04, BUG-05, BUG-06 | — |
| 4 | [FIX-108](FIX-108.md) | Write to_onnx's external data under the final file name | P1 | M | READY | FIX-107 | BUG-03, DOC-16 | #255, #570 |
| 5 | [FIX-109](FIX-109.md) | Keep reading run directories whose id ends in a dot or a space | P1 | M | READY | — | BUG-07 | #546, #522 |
| 6 | [FIX-110](FIX-110.md) | Remove a stale tokenizer.json only after the new Hub save is in place, and only if NNx wrote it | P1 | S | READY | — | BUG-08 | #260 |
| 11 | [FIX-111](FIX-111.md) | State what a failed save_pretrained leaves behind | P2 | S | READY | FIX-110 | BUG-09 | #260, #509 |
| 12 | [FIX-112](FIX-112.md) | Build a registered model factory once per from_pretrained | P2 | S | READY | — | BUG-17, DOC-02 | #216 |
| 25 | [FIX-113](FIX-113.md) | Attribute the Hub "no manifest" warning to the caller's line | P3 | S | READY | — | BUG-18 | #522 |
| 7 | [FIX-114](FIX-114.md) | Flush committed export files through a writable descriptor | P1 | S | READY | — | BUG-10 | #552, #443 |
| 26 | [FIX-115](FIX-115.md) | Flush standalone artifact files with F_FULLFSYNC on macOS | P3 | S | READY | FIX-114, FIX-116 | BUG-11 | #552, #443 |
| 13 | [FIX-116](FIX-116.md) | Apply the current umask at write time and keep an existing file's mode | P2 | M | READY | FIX-110, FIX-114 | BUG-12, BUG-13 | #260 |
| 27 | [FIX-117](FIX-117.md) | Write the tokenizer file through a unique partial file | P3 | S | READY | FIX-116 | BUG-26 | #260, #522 |
| 14 | [FIX-118](FIX-118.md) | Refuse extra_metrics in task-less alpha=1 distillation and document the loss-only record | P2 | S | READY | — | BUG-15, DOC-06 | #513, #542 |
| 31 | [FIX-119](FIX-119.md) | Warn when save_lora_weights skips a DoRA magnitude | P3 | S | READY | — | BUG-19 | #201, #225 |
| 15 | [FIX-120](FIX-120.md) | Leave classification metrics unset in paradigm steps that do not compute them | P2 | S | READY | — | BUG-20, DOC-31 | #257, #82, #212, #251 |
| 16 | [FIX-121](FIX-121.md) | Write real safetensors from export_to_safetensors without the safetensors package | P2 | M | READY | — | BUG-21, DEAD-10 | #237 |
| 32 | [FIX-122](FIX-122.md) | Parse bundle JSON with the shared strict parser | P3 | S | READY | — | BUG-22, SMELL-07 | #551, #218 |
| 33 | [FIX-123](FIX-123.md) | Validate top_p as a finite real number | P3 | S | READY | — | BUG-23 | #544, #254 |
| 34 | [FIX-124](FIX-124.md) | Validate train_contrastive's n_epochs and batch_size as counts | P3 | S | READY | — | BUG-24 | #542, #544, #231 |
| 35 | [FIX-125](FIX-125.md) | Freeze TeacherRecord's provenance mapping | P3 | S | READY | — | BUG-25 | #558, #244 |
| 36 | [FIX-126](FIX-126.md) | Hash bfloat16 weights in export-conformance digests | P3 | S | READY | — | BUG-27 | #255 |
| 17 | [FIX-127](FIX-127.md) | Register every optional extra in nnx._optional | P2 | S | READY | — | BUG-28 | #249, #220, #234 |
| 18 | [FIX-128](FIX-128.md) | Route quantize, prune, GGUF, Jev and search imports through nnx._optional.require | P2 | M | READY | FIX-127 | BUG-28 | #249, #220, #234 |
| 19 | [FIX-129](FIX-129.md) | Route embeddings, viz, onnxscript and Hub imports through nnx._optional.require | P2 | M | READY | FIX-127 | BUG-28 | #249, #234, #570 |
| 8 | [FIX-130](FIX-130.md) | Keep Dependabot pull requests off the self-hosted runners | P1 | S | READY | — | CI-01 | #572, #420, #757 |
| 9 | [FIX-131](FIX-131.md) | Document DPO's refusal of responses with no valid token | P1 | S | READY | — | DOC-01 | — |
| 20 | [FIX-132](FIX-132.md) | Document GGUF export's refusal of state no tensor carries | P2 | S | READY | — | DOC-04 | #525 |
| 21 | [FIX-133](FIX-133.md) | Document the test_split_id deprecation and the new threshold refusals | P2 | S | READY | — | DOC-05 | #217, #218 |
| 37 | [FIX-134](FIX-134.md) | Document StreamingMetrics' refusal of unnormalized probability rows | P3 | S | READY | — | DOC-11 | #518 |
| 38 | [FIX-135](FIX-135.md) | Bring the Hub guide and docstrings up to date with manifests and refusals | P3 | S | READY | — | DOC-12, DOC-13, DOC-21, DOC-22 | #767, #509 |
| 39 | [FIX-136](FIX-136.md) | Add the missing CHANGELOG entries and fix two train() shorthands | P3 | S | READY | — | DOC-14, DOC-20 | — |
| 40 | [FIX-137](FIX-137.md) | Document export_bundle's resume-point verification and "best" refusal | P3 | S | READY | — | DOC-15 | #537 |
| 41 | [FIX-138](FIX-138.md) | Re-sync the README with the quickstart split | P3 | S | READY | — | DOC-17, DOC-18, DOC-24 | #472, #262 |
| 42 | [FIX-139](FIX-139.md) | List SDK_RANGE and SUPPORTED_QUANTIZATIONS in the API reference | P3 | S | READY | — | DOC-19 | #220, #235 |
| 43 | [FIX-140](FIX-140.md) | Correct stale tooling comments and versions in config and contributor docs | P3 | S | READY | — | DOC-23, DOC-27, DOC-29, DOC-34 | #567, #757 |
| 44 | [FIX-141](FIX-141.md) | Deduplicate release-please's changelog list when nothing is curated | P3 | M | READY | — | DOC-25 | #418, #535 |
| 45 | [FIX-142](FIX-142.md) | Say what the embeddings extra installs and why | P3 | S | READY | — | DOC-28 | — |
| 46 | [FIX-143](FIX-143.md) | Correct stale code comments and docstrings | P3 | S | READY | — | DOC-30, DOC-31, DOC-32, DOC-33 | #472, #570, #759 |
| 22 | [FIX-144](FIX-144.md) | Call fit_temperature with test_split_id in run_bundle.py and fail example tests on the deprecation | P2 | S | READY | — | SYNC-01, SYNC-02 | #571, #217 |
| 23 | [FIX-145](FIX-145.md) | Pass test_split_id in the tests that still omit it | P2 | M | READY | FIX-144 | SYNC-03 | #217, #218 |
| 47 | [FIX-146](FIX-146.md) | List the import-only examples and add the OpenMP guard to example 13 | P3 | S | READY | — | SYNC-04, SYNC-05 | #572, #570 |
| 48 | [FIX-147](FIX-147.md) | Stop skipping tests for packages they do not use | P3 | S | READY | — | SYNC-06, SYNC-07 | #571 |
| 49 | [FIX-148](FIX-148.md) | Say in the README that GGUF and Ollama exports need a custom runtime | P3 | S | READY | — | SCOPE-04 | #255, #525, #517 |
| 50 | [FIX-149](FIX-149.md) | Match pruning layer patterns on every registration path and validate sparsity | P3 | M | READY | — | SMELL-57 | #224, #554 |
| 51 | [FEAT-059](FEAT-059.md) | Load nnx.decisions and nnx.export_conformance lazily on import nnx | P3 | M | READY | — | SCOPE-01, SCOPE-03 | #219, #220, #243, #245, #234, #263, #255, #249, #566 |
| 52 | [FEAT-060](FEAT-060.md) | Move abstention.decide into nnx.decisions and break the import cycle | P3 | M | READY | FEAT-059 | SCOPE-01, SMELL-42 | #219, #263, #245 |
| 53 | [FEAT-061](FEAT-061.md) | Split export_conformance.validate_record and share one SHA-256 check | P3 | M | READY | — | SCOPE-03 | #255, #566 |
| 54 | [FEAT-062](FEAT-062.md) | Drop the unused libcairo2 step from the hosted MPS gate | P3 | S | READY | — | CI-07 | #445, #572 |
| 55 | [FEAT-063](FEAT-063.md) | Remove dead code in nn_run, distributed, objectives, peft, benchmark and deepen | P3 | S | READY | — | DEAD-01, DEAD-02, DEAD-06, DEAD-07, DEAD-08, DEAD-09 | #570, #571 |
| 56 | [FEAT-064](FEAT-064.md) | Remove helpers that only tests use | P3 | S | READY | — | DEAD-03, DEAD-11, DEAD-12 | #568 |
| 57 | [FEAT-065](FEAT-065.md) | Remove the Python 3.8 version shim and its stale comment | P3 | S | READY | — | DEAD-04, DOC-35 | #418 |
| 58 | [FEAT-066](FEAT-066.md) | Remove torch-version guards below the torch 2.4 floor | P3 | S | READY | — | DEAD-05 | #210, #572 |
| 59 | [FEAT-067](FEAT-067.md) | Remove the inert noqa: BLE001 markers | P3 | S | READY | — | DEAD-14 | #507 |
| 60 | [FEAT-068](FEAT-068.md) | Rename the test_pass2_*_series modules after what they test | P3 | M | READY | FIX-109 | CLUT-02 | #206, #211, #755, #757 |
| 61 | [FEAT-069](FEAT-069.md) | Shrink the README banner under the 500 KB file limit | P3 | S | READY | — | CLUT-03 | — |
| 62 | [FEAT-070](FEAT-070.md) | Trim audit-history narration from the PyPI keyword comments | P3 | S | READY | — | CLUT-05 | #573, #575 |
| 28 | [FEAT-071](FEAT-071.md) | Add a shared atomic-I/O module (part 1 of 3) | P2 | M | READY | FIX-114, FIX-115, FIX-116, FIX-117 | SMELL-01 | #552, #443, #260 |
| 29 | [FEAT-072](FEAT-072.md) | Move the checkpoint, run, history and bundle writers onto nnx._atomic_io (part 2 of 3) | P2 | M | READY | FEAT-071 | SMELL-01 | #552, #443, #260 |
| 30 | [FEAT-073](FEAT-073.md) | Move the remaining atomic writers onto nnx._atomic_io and drop their local imports (part 3 of 3) | P2 | M | READY | FEAT-072 | SMELL-01 | #552, #260, #522 |
| 88 | [FEAT-074](FEAT-074.md) | Move the resume machinery out of nn_model.py (part 1 of 5) | P3 | L | READY | FEAT-079, FEAT-081, FEAT-088, FEAT-089, FEAT-090, FEAT-103 | SMELL-02 | #536, #507, #206 |
| 95 | [FEAT-075](FEAT-075.md) | Move the transform-replay helpers into nnx.transforms (part 2 of 5) | P3 | M | READY | FEAT-091, FEAT-108 | SMELL-02 | #509, #767 |
| 96 | [FEAT-076](FEAT-076.md) | Move the Hub I/O helpers into nnx/nn/_hub.py (part 3 of 5) | P3 | L | READY | FIX-110, FIX-112, FIX-113, FEAT-061, FEAT-073, FEAT-074, FEAT-075, FEAT-086, FEAT-092, FEAT-101, FEAT-102 | SMELL-02 | #509, #441, #260, #767 |
| 89 | [FEAT-077](FEAT-077.md) | Move the DDP fit helpers into nnx/nn/_ddp_fit.py (part 4 of 5) | P3 | L | READY | FEAT-074 | SMELL-02, SMELL-15 | #478, #539, #540 |
| 79 | [FEAT-078](FEAT-078.md) | Move the lazy-module descriptor into nnx.models (part 5 of 5) | P3 | M | READY | FEAT-093, FEAT-094 | SMELL-02 | #562, #764, #528 |
| 65 | [FEAT-079](FEAT-079.md) | Extract four helpers from NNModel._train_impl | P3 | M | READY | FEAT-081 | SMELL-03 | #536, #576, #765, #540, #768 |
| 63 | [FEAT-080](FEAT-080.md) | Extract the per-run body of observations_from_runs | P3 | S | READY | — | SMELL-03 | — |
| 64 | [FEAT-081](FEAT-081.md) | Share the remaining epoch-end blocks between NNModel and Trainer | P3 | M | READY | — | SMELL-06 | #536, #541 |
| 66 | [FEAT-082](FEAT-082.md) | Move _warn_at_user_frame to a standard-library-only module | P3 | S | READY | FIX-113 | SMELL-04 | #522 |
| 67 | [FEAT-083](FEAT-083.md) | Share one warn-while-failing helper | P3 | S | READY | FEAT-067, FEAT-082 | SMELL-05 | #507 |
| 68 | [FEAT-084](FEAT-084.md) | Replace two raw warnings.warn calls with _warn_at_user_frame | P3 | S | READY | FIX-107, FEAT-082 | SMELL-27 | #522 |
| 69 | [FEAT-085](FEAT-085.md) | Share the JSON parse wrappers of history and provenance | P3 | M | READY | FIX-122 | SMELL-07 | #551, #762, #521 |
| 70 | [FEAT-086](FEAT-086.md) | Share one file-digest helper | P3 | S | READY | — | SMELL-08 | #520 |
| 71 | [FEAT-087](FEAT-087.md) | Share validation and serialization between NNTrainParams and NNTrainerParams | P3 | M | READY | — | SMELL-09 | #515 |
| 72 | [FEAT-088](FEAT-088.md) | Pin the checkpoint sidecar bytes before refactoring NNCheckpoint.save (part 1 of 3) | P3 | S | READY | — | SMELL-10 | #536, #215, #206, #394 |
| 73 | [FEAT-089](FEAT-089.md) | Pass NNCheckpoint.save's training state as one value (part 2 of 3) | P3 | M | READY | FEAT-088 | SMELL-10 | #536, #215, #502 |
| 74 | [FEAT-090](FEAT-090.md) | Derive single vs named optimizers instead of passing trainer= and named= (part 3 of 3) | P3 | M | READY | FEAT-089 | SMELL-10 | #536, #215, #206, #502 |
| 75 | [FEAT-091](FEAT-091.md) | Move _unportable_transform into nnx.transforms and name the QAT transform once | P3 | S | READY | — | SMELL-11 | #509, #767 |
| 76 | [FEAT-092](FEAT-092.md) | Share the Hub strict-load sequence between the preflight and the load | P3 | M | READY | FIX-112 | SMELL-12 | #441, #509 |
| 77 | [FEAT-093](FEAT-093.md) | Bundle NNModel's lazy-module tracking into one object | P3 | M | READY | — | SMELL-13 | #562, #764 |
| 78 | [FEAT-094](FEAT-094.md) | Share the dry-run lazy-inference rules in nnx.models | P3 | S | READY | — | SMELL-14 | #562, #528 |
| 97 | [FEAT-095](FEAT-095.md) | Hoist nn_model.py's needless function-local imports | P3 | M | READY | FEAT-082, FEAT-091, FEAT-108 | SMELL-15 | — |
| 80 | [FEAT-096](FEAT-096.md) | Share one FrozenMapping class (part 1 of 2) | P3 | M | READY | — | SMELL-16 | #563, #558, #209, #551 |
| 81 | [FEAT-097](FEAT-097.md) | Share one thaw function for frozen JSON values (part 2 of 2) | P3 | M | READY | FEAT-096 | SMELL-16 | #551, #563 |
| 83 | [FEAT-098](FEAT-098.md) | Read run.yaml and idps.csv through one helper each | P3 | M | READY | FEAT-063, FEAT-099 | SMELL-17 | #522 |
| 82 | [FEAT-099](FEAT-099.md) | Share the metadata.yaml record reader | P3 | S | READY | FEAT-063 | SMELL-18 | — |
| 84 | [FEAT-100](FEAT-100.md) | Share the checkpoint-path guard of the training-state readers | P3 | S | READY | — | SMELL-19 | #206 |
| 85 | [FEAT-101](FEAT-101.md) | Build the state dict once per save_pretrained | P3 | S | READY | FIX-110 | SMELL-20 | #509, #767 |
| 86 | [FEAT-102](FEAT-102.md) | Name the tokenizer file once and annotate the Hub stub NoReturn | P3 | S | READY | FIX-110 | SMELL-21 | #260 |
| 87 | [FEAT-103](FEAT-103.md) | Remove small duplicates in nn_model and nn_run | P3 | S | READY | — | SMELL-22, SMELL-24, SMELL-28 | — |
| 90 | [FEAT-104](FEAT-104.md) | Remove small duplicates in probability, nets and evaluation records | P3 | S | READY | — | SMELL-26, SMELL-32, SMELL-33 | #532, #545, #544 |
| 91 | [FEAT-105](FEAT-105.md) | Share one supervised batch reader between nn_model and paradigm steps | P3 | M | READY | — | SMELL-23 | #564, #561 |
| 92 | [FEAT-106](FEAT-106.md) | Replace magic-index module state with named fields | P3 | S | READY | — | SMELL-25 | #562 |
| 93 | [FEAT-107](FEAT-107.md) | Build to_onnx's and netron_export's export arguments in one place | P3 | M | READY | FIX-108 | SMELL-29 | #556 |
| 94 | [FEAT-108](FEAT-108.md) | Replay transforms on a bare module in the Hub preflight | P3 | S | READY | FEAT-091 | SMELL-30 | — |
| 98 | [FEAT-109](FEAT-109.md) | Make SplitManifest and Standardizer subclass JsonArtifact | P3 | M | READY | — | SMELL-34 | #521 |
| 99 | [FEAT-110](FEAT-110.md) | Share the knowledge-distillation loss and move KDObjective beside it | P3 | M | READY | FIX-118, FEAT-105 | SMELL-35 | #214, #513, #542, #544 |
| 100 | [FEAT-111](FEAT-111.md) | Share forward noising between the diffusion step and objective | P3 | S | READY | FIX-120 | SMELL-36 | #257, #561 |
| 101 | [FEAT-112](FEAT-112.md) | Share one freeze helper for frozen modules | P3 | S | READY | — | SMELL-36 | #782, #531 |
| 102 | [FEAT-113](FEAT-113.md) | Reuse the shared NLL, Brier and macro-F1 in the decision benchmark | P3 | M | READY | — | SMELL-37 | #243, #518 |
| 103 | [FEAT-114](FEAT-114.md) | Validate decision counts with require_count (part 1 of 4) | P3 | M | READY | — | SMELL-38 | #544, #231 |
| 104 | [FEAT-115](FEAT-115.md) | Validate generation, diffusion and plan counts with require_count (part 2 of 4) | P3 | S | READY | — | SMELL-38 | #544, #231 |
| 105 | [FEAT-116](FEAT-116.md) | Validate viz counts with require_count (part 3 of 4) | P3 | S | READY | — | SMELL-38 | #544, #231 |
| 106 | [FEAT-117](FEAT-117.md) | Share one _count helper across task modules (part 4 of 4) | P3 | S | READY | — | SMELL-38 | #544, #231 |
| 107 | [FEAT-118](FEAT-118.md) | Share task-owned metric registration and the extra_metrics refusal | P3 | M | READY | — | SMELL-39 | #548 |
| 108 | [FEAT-119](FEAT-119.md) | Move the module-path walkers out of peft/_targets.py | P3 | M | READY | FIX-149 | SMELL-40 | #224, #523, #524, #526 |
| 109 | [FEAT-120](FEAT-120.md) | Split DecisionJob's _Runner.arun into launch, harvest and drain | P3 | M | READY | — | SMELL-41 | #245, #557 |
| 140 | [FEAT-121](FEAT-121.md) | Split decisions/benchmark.py by responsibility | P3 | L | READY | FIX-104, FIX-105, FIX-106, FEAT-060, FEAT-096, FEAT-113, FEAT-114, FEAT-126, FEAT-127, FEAT-128, FEAT-134, FEAT-136, FEAT-141, DEC-016, DEC-017, DEC-025 | SMELL-41 | #243, #558, #219 |
| 110 | [FEAT-122](FEAT-122.md) | Reuse the shared RNG snapshot in lr_finder and viz.summary | P3 | S | READY | — | SMELL-44 | #572, #555, #576 |
| 111 | [FEAT-123](FEAT-123.md) | Map GGUF quantization labels once and read the FFN width from the model | P3 | S | READY | — | SMELL-45, DOC-31 | #261, #525, #517, #472, #570, #575 |
| 112 | [FEAT-124](FEAT-124.md) | Share one reshaped-key check between PEFT and finetune loading | P3 | S | READY | — | SMELL-47 | #528 |
| 113 | [FEAT-125](FEAT-125.md) | Share save and load bodies across the four PEFT adapter families | P3 | M | READY | FIX-119, FEAT-124 | SMELL-48 | #201, #224 |
| 114 | [FEAT-126](FEAT-126.md) | Share JSONL record read and write and the key checks | P3 | M | READY | — | SMELL-49 | #521, #558, #243, #244 |
| 115 | [FEAT-127](FEAT-127.md) | Define the 1e-6 probability tolerance once | P3 | S | READY | — | SMELL-50 | #211, #219, #244 |
| 116 | [FEAT-128](FEAT-128.md) | Cross-reference duplicate digest forms and value types in docstrings | P3 | S | READY | — | SMELL-51, SMELL-59 | #557, #521 |
| 117 | [FEAT-129](FEAT-129.md) | Share the legacy reparameterization walk and record per-API error types | P3 | S | READY | — | SMELL-52 | #523, #524, #525, #554 |
| 118 | [FEAT-130](FEAT-130.md) | Share the NNModel unwrap in nnx.viz and hoist its imports | P3 | S | READY | FIX-129, FEAT-116, FEAT-122 | SMELL-54, DOC-31 | #249, #472, #570 |
| 119 | [FEAT-131](FEAT-131.md) | Share the post-mix step of Mixup and CutMix | P3 | S | READY | FIX-120 | SMELL-55 | #513, #542 |
| 120 | [FEAT-132](FEAT-132.md) | Share DPO's empty-row lookup | P3 | S | READY | FIX-131 | SMELL-56 | — |
| 121 | [FEAT-133](FEAT-133.md) | Share freeze and unfreeze's pattern loop | P3 | S | READY | — | SMELL-57 | #224, #554 |
| 122 | [FEAT-134](FEAT-134.md) | Share one non-empty-text validator | P3 | S | READY | FEAT-114 | SMELL-58 | #558 |
| 123 | [FEAT-135](FEAT-135.md) | Use full_precision for half-precision widening | P3 | S | READY | — | SMELL-60 | #516, #577, #766, #225 |
| 124 | [FEAT-136](FEAT-136.md) | Hoist needless function-local imports in nnx.decisions | P3 | S | READY | FIX-104, FEAT-060, FEAT-113, FEAT-126 | SMELL-61 | #249 |
| 125 | [FEAT-137](FEAT-137.md) | Hoist needless function-local imports in offline distillation | P3 | S | READY | FEAT-118, FEAT-126, FEAT-127, FEAT-134 | SMELL-61 | #249, #244 |
| 126 | [FEAT-138](FEAT-138.md) | Hoist needless function-local imports in nnx.plans | P3 | M | READY | — | SMELL-61 | #249 |
| 127 | [FEAT-139](FEAT-139.md) | Hoist needless function-local imports in nnx.transforms | P3 | M | READY | FEAT-075, FEAT-119 | SMELL-61 | #249, #509, #767 |
| 128 | [FEAT-140](FEAT-140.md) | Hoist needless function-local imports in bundles and export_conformance | P3 | M | READY | FIX-109, FIX-122, FEAT-061, FEAT-072, FEAT-091 | SMELL-61 | #249, #552, #255, #566 |
| 130 | [FEAT-141](FEAT-141.md) | Add docstrings to the nnx.decisions members api.md lists without one | P3 | M | READY | DEC-016 | DOC-09 | #219, #521, #243 |
| 24 | [DEC-015](DEC-015.md) | Decide whether the MPS workflow keeps its ComfyUI and Ollama GPU coordination | P2 | M | DECISION | FIX-130 | SCOPE-02, CI-02, CI-03, CI-04, CI-05, CI-06, DOC-07 | #572, #755 |
| 129 | [DEC-016](DEC-016.md) | Decide whether benchmark records keep a provider's own false probability | P3 | M | DECISION | — | BUG-16, DOC-08, DOC-09, DOC-10 | #219, #243 |
| 131 | [DEC-017](DEC-017.md) | Decide whether nnx.decisions moves to its own distribution | P3 | L | DECISION | FEAT-059, FEAT-060 | SCOPE-01 | #219, #220, #243, #245, #234, #263, #262 |
| 132 | [DEC-018](DEC-018.md) | Decide whether nnx.export_conformance moves out of the runtime package | P3 | S | DECISION | FEAT-059 | SCOPE-03 | #255, #566, #573, #756 |
| 133 | [DEC-019](DEC-019.md) | Decide how export_bundle treats run configs nested deeper than its 64-level bound | P3 | S | DECISION | FIX-122 | BUG-22 | #551, #218 |
| 134 | [DEC-020](DEC-020.md) | Decide whether NNx adds save and load helpers for DoRA adapters | P3 | S | DECISION | FIX-119 | BUG-19 | #201, #225 |
| 135 | [DEC-021](DEC-021.md) | Decide which permission bits NNx gives the files it writes | P3 | S | DECISION | FIX-116 | SMELL-31 | #260, #552 |
| 136 | [DEC-022](DEC-022.md) | Decide whether the .maintenance/ ledgers stay in the repository | P3 | S | DECISION | — | CLUT-01 | — |
| 137 | [DEC-023](DEC-023.md) | Decide whether dev and docs tooling leave the published PyPI extras | P3 | M | DECISION | FIX-140 | CLUT-04 | #220, #225 |
| 138 | [DEC-024](DEC-024.md) | Decide whether LogitsChain is deprecated in favour of OrderedLogitsPipeline | P3 | S | DECISION | — | SMELL-46 | #254, #578 |
| 139 | [DEC-025](DEC-025.md) | Decide which input-modality rule nnx.decisions shares | P3 | S | DECISION | — | SMELL-43 | #557, #763 |
| 141 | [DEC-026](DEC-026.md) | Decide one probability floor for link-task NLL | P3 | S | DECISION | FEAT-113 | SMELL-37 | #243, #518 |
| 142 | [DEC-027](DEC-027.md) | Decide whether train_contrastive becomes a thin wrapper over NNModel.train | P3 | M | DECISION | FIX-124 | SMELL-53 | #542, #544 |
