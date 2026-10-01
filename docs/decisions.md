# 11. Typed decisions

`nnx.decisions` is NNx's **inference branch**: a provider-neutral way to ask a
typed question and get labelled probabilities back, instead of reading
positional logits. A question is one of three primitives; a **provider**
declares what it can answer and returns validated results; the fixed-head
adapter turns a trained NNx classifier into such a provider. Nothing here
trains, exports or executes actions, and importing it starts no provider or
model backend and needs no hosted-SDK extra.

```python
from nnx.decisions import Choice, FixedHeadProvider, Option

question = Choice(
    "Which animal is in the photo?",
    (Option("sp-fox", "A fox"), Option("sp-cat", "A cat"), Option("sp-dog", "A dog")),
)
provider = FixedHeadProvider(model, option_map={"sp-cat": "cat", "sp-dog": "dog", "sp-fox": "fox"})
results = provider.decide(question, photos)     # one ChoiceResult per row
results[0].distribution                         # (("sp-fox", 0.01), ("sp-cat", 0.94), ("sp-dog", 0.05))
results[0].top                                  # "sp-cat"
```

[`examples/decision_fixed_head.py`](../examples/decision_fixed_head.py) runs
this end to end on a deterministic local model.

## 1. Questions

| Primitive | Holds | Answered by |
|---|---|---|
| `Choice(prompt, options)` | 2+ `Option(id, description)` with unique ids | a distribution over the options |
| `Boolean(prompt)` | the prompt | one `p_true` in `[0, 1]` |
| `Score(prompt, levels)` | 2+ ordered `Option(level_id, description)`, lowest first | a distribution over the levels |

Options may be given as `(id, description)` pairs; anything else (a bare
string, say) raises `InvalidDecisionRequest`. `kind` (`"choice"`,
`"boolean"`, `"score"`) is the discriminator, and `state()` /
`question_from_state(...)` round-trip a question as plain data — malformed
state raises `InvalidDecisionRequest`, never a bare `KeyError`.

**Bookkeeping ids vs model-facing text.** An option's `id` is how results are
keyed and never reaches a model; its `description` is what a model reads.
`question.model_view()` returns exactly the model-facing part — the prompt and
the option texts in order, no ids — so a text provider can render a question
without leaking routing keys such as `"team-7f3a"`.

**Digest.** `question.digest()` is a SHA-256 over the kind, the prompt and
every option's id and description **in order**. Reordering the options or
rewording a description changes it, and every result records the digest of
the question it answers, so a cached or batched result can never be matched
to a different question.

## 2. Results and validation

`ChoiceResult` and `ScoreResult` carry a `distribution` — `(id, p)` pairs in
the question's order — that must hold 2+ unique ids, each `p` finite in
`[0, 1]`, summing to 1 within `PROBABILITY_TOLERANCE = 1e-6`. `BooleanResult`
carries one `p_true` (`p_false` is derived). A provider's own output stays in
`raw` (kept out of equality) and never mixes with the normalized fields.

`validate_response(question, response)` is the one validator every provider
uses. For a Choice or Score, `response` is the provider's keyed output — a
mapping or `(id, p)` pairs in any order. It is reordered into the question's
order, and a **missing**, **duplicate**, **unknown** or **unlabeled** (empty or
non-string) id, a probability outside `[0, 1]`, or a distribution that does not
sum to 1 raises `InvalidDecisionResponse`. A malformed distribution is **never
renormalized**: dividing by the sum would hide a provider bug. For a Boolean,
`response` is `p_true` or `{"true": p, "false": 1 - p}`.

**Scores are ordinal.** `ScoreResult.expected_index` is `sum(i * p_i)` over
zero-based levels: a position between levels, useful for ranking, but the
levels' spacing is undefined, so it is not an interval-scale score. A provider
that reports its own number (for example a 1–10 rating) passes it as
`vendor_score`, which is kept separate from the distribution.

## 3. Providers and capabilities

A `DecisionProvider` has two methods: `capabilities()` and
`decide(question, inputs) -> list[result]` (one result per input).
`Capabilities` declares:

- `primitives` — the question kinds it answers;
- `modalities` — the inputs it reads (`"tensor"`, `"text"`, …);
- `dynamic_labels` / `labels` — whether a request may bring any option set,
  or must map onto a fixed label space;
- `max_batch` — the most inputs per call;
- `inference` / `training` / `export` — what the backend can do. A hosted
  provider declares inference only; it inherits no training or export
  capability.

`capabilities().check(question, modality=..., batch_size=..., label_ids=None)`
raises `UnsupportedCapability` for anything undeclared — including, without
dynamic labels, a Choice or Score whose ids (or `label_ids`, the provider's
own mapping of them) are not exactly the declared `labels` — and providers
call it **before any model call or network I/O**.

## 4. The fixed-head adapter

`FixedHeadProvider(model, labels=None, option_map=None, ordinal=False,
max_batch=None)` adapts a trained `NNModel`. The head decides what it can
answer:

| Head | Answers |
|---|---|
| categorical (softmax over `C` classes) | `Choice`; also `Score` with `ordinal=True`, whose levels must follow the head's class order |
| one Bernoulli logit (`BCEWithLogitsLoss`, or a one-output multilabel task) | `Boolean` |
| regression, or multi-output multilabel | nothing — rejected at construction |

- **Label space.** `labels` names the head's columns in order; it defaults to
  the model's `TaskSpec` labels and must equal them when both are given. A
  label count that differs from the head's width is rejected at construction
  when the width is known (a task's count or a built-in net's `output_dim`),
  otherwise with `InvalidDecisionRequest` on the first call. A request's
  option ids must be exactly those labels (in any order), or `option_map`
  must map them one-to-one onto the labels. An unseen or missing label
  raises `UnsupportedCapability` before the `model_calls` counter advances.
- **Columns.** Logits come from `model.predict_proba` (the FEAT-001
  prediction contract); probabilities are recomputed from them in float64
  (so a half-precision head still meets the `1e-6` tolerance), reordered
  into the request's option order and passed through `validate_response`.
  Each row's logits (a copy) become `raw`. An empty batch returns `[]`
  without calling the model.
- **Modes.** Prediction runs in eval mode and restores every submodule's
  training mode, on success and on failure.
- **Failures.** An exception from the model itself becomes `ProviderFailure`,
  with the original error as `__cause__`.

## 5. Benchmarking providers

`nnx.decisions.benchmark` (FEAT-021) scores decision providers on
**identical samples**, offline first: a live collection runs once, and every
report after that is a replay of saved records — no provider call, no
credentials, no network, nothing fitted.

```python
from nnx.decisions.benchmark import Budget, Sample, collect, evaluate, read_records, write_records

samples = [Sample("pet-0", question, row, "cat", family="pets"), ...]
collection = collect(provider, samples, provider_id="fixed-head-v1", budget=Budget(max_calls=20))
write_records("records.jsonl", collection.records)           # once, live
report = evaluate(samples, read_records("records.jsonl"), split="animals-v1")   # any time, offline
print(report.text())
```

- **Samples** carry the join key (`id`), the typed question (its digest is
  the schema identity), the provider input, the true label, the task
  `family` (`heldout=True` families report apart), the grouping unit
  (`group`) and the `perturbation` that produced them. `permute_options`,
  `redescribe` (new label descriptions), `add_distractors`,
  `add_none_of_the_above`, `add_context` (long irrelevant context) and
  `rewrite_input` (multilingual or adversarial state) derive perturbed
  samples that keep their original's grouping unit.
- **Replay format.** One `Record` per provider output, one JSON object per
  line (`nnx.decision-record/1`, strict JSON): sample id, question digest,
  provider, status, the answer (`distribution` or `p_true`) or the
  `reason` there is none, model `revision`, `prompt_identity` (by default a
  digest of the provider's `record()`) and `execution` metadata (batch,
  size, `partial_batch`, seconds, attempt).
- **Coverage statuses.** Records join samples by sample id **and** question
  digest, never by row position. Each sample is `eligible` (one answered
  record), `missing`, `duplicate`, `mismatched` (a record for another
  question digest), `invalid` (an answer that does not fit its question),
  `unsupported` or `failed` (each with its reasons); records for no sample
  are the report's `extra` count. The CSV carries every count, so they add
  up to each slice's samples.
- **Capabilities.** A provider that declares it cannot serve a request —
  `FixedHeadProvider` outside its label space, for instance, through its
  declared `capabilities()` (or its own `check(question, inputs)`) — gives
  `unsupported` records with its own reason, before any call and without
  spending budget. They are counted in the slice's coverage, never in a
  metric's denominator, and never scored as wrong.
- **Budgets.** `collect` needs an explicit provider, provider id and
  `Budget(max_calls, max_samples=None)`. It attempts each batch once
  (retries are the provider's own), stops when the budget is spent (the
  rest is `missing`, and `Collection.stopped` says why) and marks a batch
  the sample budget cut short as `partial_batch`. Each answer is checked
  against its sample's question (digest) and put into the question's
  option order; a provider error, a malformed answer or inputs that cannot
  form one batch give `failed` records rather than ending the collection.
- **Metrics,** per slice (`in_family`, `heldout`, `family:<name>`,
  `perturbation:<name>`): accuracy, macro-F1, NLL (exact by default:
  `+inf` when a true label has probability 0; `epsilon=` floors it), Brier,
  ECE with reliability bins and — given an `nnx.abstention` policy, applied
  as given — selective coverage and risk. Each is a `MetricValue` with its
  denominator, or unavailable with the reason. NLL and Brier are the named
  `nll` / `brier` metrics' terms; a Boolean is the options `("true",
  "false")`.
- **Resources** (`Resources`) say how cost was obtained — warmup, hardware,
  timing boundary, concurrency, batch count, seconds and whether the
  numbers were `measured` (hardware required) or `supplied`; anything not
  declared stays `null`.
- **Intervals.** `bootstrap_interval` resamples grouping units (whole
  groups, never single rows of one) with a recorded seed, over the same
  provider's records `evaluate` scores, and flags a degenerate sample
  (fewer than two units, no variation, or a non-finite bound such as an
  exact NLL of `+inf`). Perturbation helpers give each variant a distinct id
  (the kind plus a digest of the change), or the `id=` you pass.
- **Exports.** `to_json()`, `to_csv()` and `text()` agree on units,
  eligible and failure counts and unavailable states. `compare_reports`
  gives `b - a` per slice and metric only for reports of the same split and
  metric identity (metric set, `epsilon`, `n_bins`, policy).

It is not a leaderboard: no paid remote run is a default, and it never
tunes a threshold or a prompt on test outcomes.
[`examples/decision_benchmark_offline.py`](../examples/decision_benchmark_offline.py)
collects once and replays with sockets disabled.

## 6. Errors

All are `nnx.decisions.DecisionError`s, and their names are stable:

| Error | Raised for |
|---|---|
| `InvalidDecisionRequest` (also a `ValueError`) | malformed questions: fewer than 2 options, duplicate or empty ids, empty text; misconfigured adapters |
| `InvalidDecisionResponse` (also a `ValueError`) | responses that do not fit their question (see §2) |
| `UnsupportedCapability` | undeclared primitives, modalities, batch sizes or label spaces — before any model call |
| `ProviderFailure` (also a `RuntimeError`) | the backend failing on a valid, supported request |

## 7. Consumers

Planned decision features share this digest, the `kind` discriminators and
`validate_response` rather than defining their own: the optional Jev SDK
adapter ([#220](https://github.com/thekaveh/NNx/issues/220)), a local
label-conditioned baseline adapter
([#234](https://github.com/thekaveh/NNx/issues/234)), offline
teacher-distribution datasets
([#244](https://github.com/thekaveh/NNx/issues/244)), applicative batching for
independent decisions ([#245](https://github.com/thekaveh/NNx/issues/245)) and an
optional `Result` at fallible boundaries
([#263](https://github.com/thekaveh/NNx/issues/263)); `nnx.decisions` does not
depend on any of them. The provider benchmark (§5, from
[#243](https://github.com/thekaveh/NNx/issues/243)) has landed.

## 8. What this does not do

- It does not claim every classifier is a universal decision-maker: the
  fixed-head adapter answers only what its head justifies.
- It does not execute actions based on a decision.
- It does not assume that separately asked questions describe independent
  events; each question is answered on its own.
- It does not call hosted models; a hosted provider is a separate adapter
  that declares its own capabilities.
