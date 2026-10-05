# 11. Typed decisions

`nnx.decisions` is NNx's **inference branch**: a provider-neutral way to ask a
typed question and get labelled probabilities back, instead of reading
positional logits. A question is one of three primitives; a **provider**
declares what it can answer and returns validated results; the fixed-head
adapter turns a trained NNx classifier into such a provider, the NLI
adapter scores candidates supplied at inference with a caller-supplied NLI
model, and the Jev adapter asks hosted Jev models through the TypeSafe SDK.
Nothing here trains, exports or executes actions, and importing it starts no
provider or model backend and needs no hosted-SDK extra.

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
non-string) id, a probability outside `[0, 1]` (one too large for a float
included), or a distribution that does not sum to 1 raises
`InvalidDecisionResponse`. A malformed distribution is **never
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
max_batch=None, question=None)` adapts a trained `NNModel`. The head decides
what it can answer:

| Head | Answers |
|---|---|
| categorical (softmax over `C` classes) | `Choice`; also `Score` with `ordinal=True`, whose levels must follow the head's class order |
| one Bernoulli logit (`BCEWithLogitsLoss`, or a one-output multilabel task) | the one `Boolean` it was trained for (`question=`) |
| regression, or multi-output multilabel | nothing — rejected at construction |

- **Label space.** `labels` names the head's columns in order; it defaults to
  the model's `TaskSpec` labels and must equal them when both are given. A
  label count that differs from the head's width is rejected at construction
  when the width is known (a task's count or a built-in net's `output_dim`),
  otherwise with `InvalidDecisionRequest` on the first call. A request's
  option ids must be exactly those labels (in any order), or `option_map`
  must map them one-to-one onto the labels. An unseen or missing label
  raises `UnsupportedCapability` before the `model_calls` counter advances.
- **A Boolean head's question.** A one-logit head is one probability bound to
  no prompt, so `question=Boolean(...)` names the question it was trained for:
  it is required for such a head and refused for a categorical one (which
  answers by its labels). Any other Boolean — another digest, however close
  its wording — raises `UnsupportedCapability` before the `model_calls`
  counter advances, so a decision job refuses it and a benchmark records it
  `unsupported`, never answered with the head's probability.
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
- **Checking without calling.** `check(question, inputs)` runs every check
  `decide` makes before the model call (question, modality, batch size and
  label space, or a Boolean's digest) and calls nothing; a decision job (§7)
  uses it to refuse a request up front.

## 5. The NLI baseline adapter

A fixed head answers only the labels it was trained on. `NLIProvider`
(FEAT-011) is a **local label-conditioned baseline**: it scores a text (the
premise) against candidate descriptions that arrive with each request, each
rendered into a hypothesis, with a natural-language-inference (NLI)
cross-encoder **the caller supplies** — so no candidate needs to exist when
the provider is built, and no output head is resized.

```python
from nnx.decisions import Boolean, Choice, NLIProvider

provider = NLIProvider(
    model, tokenizer,                    # loaded by the caller (a local path, a pinned revision)
    entailment_id="entailment",          # or the class id; names resolve through model.config.label2id
    contradiction_id="contradiction",
    hypothesis_template="This text is about {}.",
    revision="<the revision you loaded>",
)
provider.decide(Choice("Topic?", (("t-sport", "sports"), ("t-econ", "the economy"))), texts)
provider.decide(Boolean("This review is positive."), texts)
```

- **Scoring, explicit.** A `Choice` scores every (text, candidate) pair and
  softmaxes the **entailment logits across the candidates**
  (`choice_scoring="entailment_softmax"`): logits `log(3)` and `0` give
  `(0.75, 0.25)`, keyed by the request's option ids in its order. Only the
  descriptions are scored — the Choice's prompt is not part of any
  hypothesis, so write it into `hypothesis_template` when it matters. A
  `Boolean` scores one pair per text — its prompt rendered by
  `boolean_template` — and softmaxes that pair's **contradiction and
  entailment** logits alone (`boolean_scoring="entailment_vs_contradiction"`):
  `0` and `log(4)` give `p_true = 0.8`; texts are never normalized together.
  `Score` is unsupported and refused before any model call.
- **Validated settings.** The entailment and contradiction ids — given or
  resolved from names — must differ; when the model's config declares a
  class count (`num_labels`), they must fit it and the logits must have
  exactly that many classes. A label name must be one its config knows,
  and an integer id that contradicts the config's own `entailment` /
  `contradiction` names is refused. Templates hold exactly one bare `{}`
  (no conversion or format spec).
  Unknown scoring or truncation policies are refused at construction.
- **The provider contract.** NNx imports no NLI library and downloads
  nothing; importing `nnx.decisions` or constructing the provider calls no
  `from_pretrained`. The tokenizer is called HuggingFace-style
  (`tokenizer(premises, hypotheses, truncation=..., max_length=...,
  padding=True, return_tensors="pt")`, plus unpadded measurement calls with
  `truncation=False, padding=False` whose `attention_mask` sums or
  `input_ids` row lengths give each pair's token count) and the model as
  `model(**encoded)`,
  returning logits or an object with `.logits`. Pairs go in chunks of
  `pair_batch_size` (the last may be shorter) to the model's device; the
  model runs in eval mode under no-grad and every submodule's training flag
  is restored on success and failure. A model error becomes
  `ProviderFailure`.
- **Truncation, reported.** Pairs are measured untruncated first.
  `truncation="only_first"` (default) cuts the premise to `max_length` and
  marks which pairs were cut in `raw["truncated"]`; a hypothesis that leaves
  no room for the premise (measured with an empty premise) is refused
  (`InvalidDecisionRequest`) before any model call. `truncation="error"` refuses an over-long request
  before any model call.
- **Records, never calibrated.** Each result's `raw` holds the pair logits,
  the truncation report and `provider.record()` — the templates, label ids,
  scoring methods, truncation policy and model `revision` — with
  `"calibrated": False`. These are zero-shot scores from a model trained for
  another task: they are not calibrated probabilities, and how well they
  transfer to a decision task stays empirical — measure it on labelled
  records, as [`examples/decision_nli.py`](../examples/decision_nli.py) does
  (accuracy, macro-F1, categorical NLL and Brier, with the split, revision and
  settings recorded).

It does not extend `nnx.embeddings.embed_texts` or the FAISS export, whose
signatures are unchanged: those embed texts with a bi-encoder you trained;
this scores pairs with a cross-encoder you supply.

## 6. The Jev adapter

`JevProvider` (FEAT-010, the `jev` extra: `pip install "thekaveh-nnx[jev]"`)
answers Choice, Boolean and Score questions about text on **Jev models**
through the TypeSafe Python SDK's `system_one` call — NNx adds no transport
stack of its own. The adapter is tested against `typesafe-sdk>=0.7,<0.8`
(`nnx.decisions.jev.SDK_RANGE`); importing `nnx` or `nnx.decisions` never
imports the SDK, and building a provider without an injected client raises an
`ImportError` naming `thekaveh-nnx[jev]` when it is missing.

```python
from nnx.decisions import Boolean, Choice, JevProvider, Score

with JevProvider(model="jev-1.13.0") as provider:          # TYPESAFE_API_KEY from the environment
    topic, urgent, severity = provider.decide_many(
        [Choice("Topic?", (("t-bill", "billing"), ("t-ship", "shipping"))),
         Boolean("Does this need an answer today?"),
         Score("How severe?", (("s0", "minor"), ("s1", "serious"), ("s2", "critical")))],
        texts,                                              # one system_one request per text
    )
topic[0].distribution          # (("t-bill", 0.7), ("t-ship", 0.3)), in the question's order
topic[0].raw                    # {"provider": "jev", "model": "jev-1.13.0", "request_id": ..., "usage": {...},
                                #  "provider_confidence": 0.7}
```

- **Translation.** A Choice becomes a Jev `choice` keyed by its options'
  **descriptions** — the model-facing text; option ids are never sent — a
  Score a `score` whose criteria are its levels' descriptions, lowest first,
  and a Boolean a `noul`; the prompt is the `instructions`. Every question of
  one `decide_many` shares the request for each text (`max_questions` is
  unbounded), so a decision job batches them into it (§7). Answers are
  realigned to each question's own option order and checked by
  `validate_response` (§2); a Choice whose options repeat a description is
  refused before any request. A Score's expected score stays in
  `vendor_score`.
- **Confidence.** A Choice's or Score's `confidence` is Jev's own certainty
  in its selection, recorded as `raw["provider_confidence"]`. It is **not** a
  probability that the answer is correct — use the distribution, and measure
  correctness on labelled records (§8). A Boolean has none.
- **Metadata.** `raw` is plain JSON: `provider`, the `model` the service
  **resolved** (even when an alias was requested), `requested_model` when one
  was set, `request_id` and the reported `usage` token counts. A field the
  service did not report is absent; clients, headers and credentials are
  never recorded, and `record()` (the benchmark identity) holds none either.
- **Retries.** The SDK's `RetryPolicy` owns retries — pass `retry=` when the
  provider builds its client, or configure the client you inject. The adapter
  sends each request once and adds no retry loop, nor does a decision job or
  the benchmark.
- **Failures.** What still fails is typed, keeps the SDK error as
  `__cause__` and — when the service answered — its `request_id` (a timeout
  or a lost connection has none): `JevTimeout` (also a
  `TimeoutError`), `JevAuthenticationError` (401 / 403), `JevRateLimited`
  (429, with `retry_after_ms`), `JevMalformedResponse` (also an
  `InvalidDecisionResponse`: an unparseable body, a missing answer or a
  distribution that does not fit), and `JevError` for the rest — all
  `ProviderFailure`s. Crossing a decision job or the benchmark keeps the type
  and request id (`JobFailed.__cause__`, a record's `reason`).
- **Cost.** Every text is its own request, so one provider call over `n`
  texts sends `n` requests: a benchmark's `Budget.max_calls` and a job's
  `Limits.max_requests` count provider calls, not requests — set `max_batch`
  to bound the requests per call. A failure on one text stops the call;
  answers already received for earlier texts are not returned.
- **Lifecycle.** A `client` / `async_client` you inject stays yours: the
  provider never closes it and never builds a second client beside it —
  with only a sync client injected the provider is sync-only (`adecide` /
  `adecide_many` are `None`, so `DecisionJob.arun` runs it in its worker
  thread, one call at a time); with only an async one, the sync methods are
  refused. With none injected,
  the provider builds its own (lazily, at the first call, from `api_key` /
  `base_url` / `timeout` / `retry` or the SDK's environment variables; a
  missing key fails as a `JevError`): `close()` (`with`) closes a built sync
  client and `aclose()` (`async with`) both — after an error or a
  cancellation too. A built async client belongs to one event loop: a call
  or `aclose()` from another loop drops it with a `ResourceWarning`, so close
  it with `async with` inside each `asyncio.run`.
- **Async.** `adecide` / `adecide_many` use the async client (built, or
  injected); `DecisionJob.arun` awaits them directly.

Ordinary tests need no credentials or network: `tests/test_decision_jev.py`
drives a recording fake of `system_one` that returns real SDK responses, and
[`examples/decision_jev.py`](../examples/decision_jev.py) runs real SDK clients
over an offline mock transport, sync and async. The opt-in live smoke,
`scripts/smoke_jev_live.py`, sends one request to the pinned `jev-1.13.0` (never
an alias) and prints the resolved model, SDK version, request id, token usage
and wall time; it spends real quota, so CI never runs it.

## 7. Decision jobs: batching and chaining

A `DecisionJob` is an **immutable, deferred description** of decision work.
Building one calls no provider, runs no callback and draws no RNG; `run` and
`arun` are the only effect boundaries.

```python
from nnx.decisions import DecisionJob as Job, Follow, Limits

flags = Job.collect({                                   # independent questions, keyed
    "goal": Job.ask(Boolean("Mentions goal"), id="goal"),
    "bank": Job.ask(Boolean("Mentions bank"), id="bank"),
})
routed = Job.ask(topic, id="topic").then(               # a dependent question
    lambda answers: Follow(Job.ask(keeper, id="keeper"), state=sports_texts(answers))
)
result = Job.collect({"flags": flags, "routed": routed}).run(provider, state=texts, limits=Limits(max_questions=8))
result.value["flags"]["goal"]                          # one result per text
result.outcomes["keeper"].kind                         # "answered"
result.calls                                           # provider calls made
```

| Builder | Value |
|---|---|
| `Job.ask(question, id=..., policy=None, model_id=None)` | the provider's results, one per input row; with an abstention `policy` ([concepts §19](concepts.md)), the selective decisions |
| `Job.collect({key: job, ...})` | a mapping of each key to its job's value, in the given key order |
| `job.map(fn)` | `fn(value)`: a pure function, no provider call |
| `job.then(fn)` | `fn(value)` returns `Follow(next_job, state=...)`; the value is the next job's |

- **Batching.** Independent questions over the same state are sent in the
  fewest calls: `ceil(count / cap)` per state object (continuations that
  return the same object share it; equal but distinct objects do not), in
  the order they were collected, keeping their ids and the `collect` key order. `cap` is the
  provider's: a provider with `decide_many(questions, inputs)` declares
  `max_questions` (`None`: any number per call); a provider with only
  `decide`, such as `FixedHeadProvider`, answers one question per call.
  `Limits.max_questions` lowers it.
- **Refused before any call.** Duplicate question ids, a question the
  provider cannot serve (its `check(question, inputs)` when it has one, else
  `capabilities().check(...)`) and a `Limits.max_tokens` cap the provider
  cannot enforce (it needs `count_tokens(questions, inputs)`) raise
  `InvalidJob` with no call made; so does an abstention `policy` whose
  labels or `model_id` do not fit its question (checked by `ask`). A continuation's questions are checked
  when the continuation runs, before they are sent.
- **Dependent questions.** A `then` continuation runs **once**, after its
  prerequisite succeeded, and maps the answer explicitly into the follow-up
  `state`; it never feeds a sibling question. `Limits.max_depth` bounds
  nesting and `Limits.max_requests` the run's calls (`JobLimitExceeded`).
- **Answers are checked.** A call's answers must hold one result list per
  question, one result per input row, each a result of that question (its
  digest): answers in the wrong order, short rows or a flat list are an
  `InvalidDecisionResponse` failure, never filed under the wrong id.
- **Fail-fast.** When a call raises — or a provider hook fails: its
  `check`, `capabilities()` or `count_tokens` raising anything but a
  declared refusal — nothing more is scheduled. `JobFailed`
  carries `outcomes` (every outcome so far, in scheduling order),
  `completed` (the answered ones), `failed` (the questions of that call) and
  `skipped` (known questions never sent); continuations
  waiting on a failed answer never run, and nothing completed is re-run.
  A job refused before any call has no outcomes and nothing skipped. The
  error pickles across a process boundary: a provider error that does not
  survive the round trip is replaced by a `ProviderFailure` naming it.
  The job never retries: retries belong to the provider. A **partial
  result** is only what the error carries: `JobFailed` has no `value`.
- **Async and cancellation.** `await job.arun(provider, state=..., limits=...,
  cancel=None)` runs up to `Limits.max_concurrency` calls at once through
  the provider's `adecide_many` / `adecide`. A provider with only
  synchronous methods is called in a worker thread, never alongside another
  call (its methods need not be thread-safe); a running thread cannot be
  interrupted, so the run waits for it before returning, and a cancellation
  that arrives meanwhile is delivered once it is done. Setting the `cancel`
  event (an `asyncio.Event` of the running loop — one bound to another loop
  is an `InvalidJob`, never a cancellation; set at any point, even if
  cleared again) stops
  scheduling — no continuation runs after it — and returns a `JobResult`
  with `status="cancelled"` and no value. A cancellation the provider
  raises itself is a provider failure, not a cancelled job. Cancelling the task cancels only the
  job's own tasks and re-raises. `Limits.timeout` raises `JobTimeout`; a
  request or depth limit lets the calls already in flight finish, then
  raises `JobLimitExceeded`. A request whose call began is reported with
  `sent=True`, never as rolled back: whatever the provider did with it
  stays done.
  The provider's hooks (`check`, `capabilities()`, `count_tokens`) are
  synchronous and run on the event loop in `arun`: keep them local and
  fast (no network round trip). The timeout is checked between hook
  calls, so a slow hook overruns it by at most one call. `arun` still needs the provider's synchronous
  `decide` (the decision protocol); `adecide` / `adecide_many` are used
  when present. A cancel set in the same tick as the last answer still
  cancels the run: its result has no value.
- **Outcomes.** `result.outcomes[id].kind` is `"answered"` (rows may still
  be `"abstained"` under a policy), `"failed"` (with `error`), `"skipped"`
  (never sent: a failure, a limit or the timeout stopped scheduling) or
  `"cancelled"` (with `sent`); the four never blur.
- **The provider is borrowed.** `run` and `arun` never close it, and it
  stays usable after a failure or a cancellation.
- **Serialisation.** A job of `ask` and `collect` only pickles as plain data
  (`job.state()`). A job holding a runtime function (`map`, `then`) refuses
  to pickle: rebuild it where it runs.

**Limits of value equivalence.** `job.map(lambda x: x)` and `job` have the
same value, and `job.map(f).map(g)` equals `job.map(lambda x: g(f(x)))`,
**when the provider is deterministic and answers a question the same way
whatever it is batched with**. A sampling provider, or one whose answer
depends on the other questions in its call, keeps the job's call structure
but not that equality. Batching is computational, not statistical: answers
asked together are separate marginals, and the job never multiplies them
into a joint probability.

**What it is not.** It is not `LogitsChainBuilder`: that is a mutable
builder of LM-decoding processors that `build()` sorts into a canonical
order, while a job is immutable and describes provider requests. It is not
an `ExperimentPlan` (`nnx.plans`): a plan compiles a training run, while a
job only asks questions of an already-trained or hosted provider.

[`examples/decision_jobs.py`](../examples/decision_jobs.py) runs batching, a
continuation, fail-fast and the async path end to end.

## 8. Benchmarking providers

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
  samples that keep their original's grouping unit. A sample's input is a
  text, a tensor or array, or a tuple of them (several inputs per sample,
  batched part by part).
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
  `FixedHeadProvider` outside its label space (or its `option_map`) or
  asked a Boolean other than its own, for instance, through its own
  `check(question, inputs)` (or, without one, its declared
  `capabilities()`) — gives `unsupported` records with its own
  reason, before any call and without spending budget; a check that itself
  raises gives `failed` records, also without a call. Batches never exceed
  the provider's declared `max_batch`. They are counted in the slice's coverage, never in a
  metric's denominator, and never scored as wrong.
- **Budgets.** `collect` needs an explicit provider, provider id and
  `Budget(max_calls, max_samples=None)`. It attempts each batch once
  (retries are the provider's own), stops when the budget is spent (the
  rest is `missing`, and `Collection.stopped` says why) and marks a batch
  the sample budget cut short as `partial_batch`. Each answer is checked
  against its sample's question (digest) and put into the question's
  option order, one by one: a malformed answer fails its own sample only;
  a provider error or inputs that cannot form one batch fail the batch.
  Neither ends the collection.
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
  (the kind plus a digest of the change — the same in every process for
  text, numeric arrays and tensors, tuples of them and JSON-able values), or
  the `id=` you pass; an input with no stable digest needs `id=`.
- **Exports.** `to_json()`, `to_csv()` and `text()` agree on units,
  eligible, failure and `extra` counts and unavailable states (a non-finite
  value is `"Infinity"` in the JSON and the CSV). `compare_reports` gives
  `b - a` per slice and metric only for reports of the same split, metric
  identity (metric set, `epsilon`, `n_bins`, policy) and sample set (a
  digest of the samples' ids, questions, labels and slicing). `collect`
  refuses duplicate sample ids before any call.

It is not a leaderboard: no paid remote run is a default, and it never
tunes a threshold or a prompt on test outcomes.
[`examples/decision_benchmark_offline.py`](../examples/decision_benchmark_offline.py)
collects once and replays with sockets disabled.

## 9. Offline teacher distributions

A decision provider's answers are distributions over a question's options.
Stored, they can teach a student offline — `nnx.paradigms.offline_distillation`
(FEAT-022) trains on them without calling the provider again:

```python
from nnx.paradigms.offline_distillation import TeacherDataset, TeacherRecord, write_teacher_records

records = [
    TeacherRecord(
        sample_id=sample.id, question_id="pet-kind",
        candidates=tuple(option_id for option_id, _ in result.distribution),
        probabilities=tuple(p for _, p in result.distribution),
        teacher="fixed-head-v1", revision="r42", schema_digest=question.digest(),
        semantics="predictive",
        provenance={"source": "records.jsonl, 2026-09 collection", "training_rights": "declared: internal use"},
        label=sample.label,
    )
    for sample, result in answered
]
write_teacher_records("teacher.jsonl", records)
data = TeacherDataset(records, inputs, candidates=("cat", "dog"), schema_digest=question.digest())
student.train(params=..., objective=data.objective(alpha=0.7))
```

- **What a record states.** The sample and question ids, the ordered
  candidate ids (the question's option ids) and their probabilities, the
  teacher and its revision, the schema digest (the question's `digest()`),
  the probability semantics and the provenance. A record without one of
  them fails validation, and `read_teacher_records` names the `path:line`
  of a malformed line — a number too large for a float included — in its
  `TeacherRecordError`.
- **Provenance is declared, not verified.** `provenance` names the record's
  `source` and the `training_rights` under which it may train a student, as
  the exporter declares them. NNx checks that the declaration is there; it
  cannot check that it is true — whether a provider's terms allow training
  on its outputs is yours to establish.
- **Probabilities, not logits.** Live distillation
  (`kd_train_step_factory`, `kd_objective`) softens a running teacher's
  logits at a temperature and scales the KL by `T²`; a stored distribution
  has no logits to soften, so the offline objective is `KL(teacher ‖
  student)` at temperature 1, plus an optional hard cross-entropy on the
  label — nothing invented, no `T²`.
- **Agreement is not quality.** The reports keep teacher agreement and
  imitation loss (over every record) apart from labelled quality (over the
  labelled records), each with its own denominator: a student that imitates
  its teacher perfectly is wrong wherever the teacher is.

See [Concepts §10.1](concepts.md#101-knowledge-distillation) and
[`examples/offline_teacher_distillation.py`](../examples/offline_teacher_distillation.py).

## 10. Errors

All are `nnx.decisions.DecisionError`s, and their names are stable:

| Error | Raised for |
|---|---|
| `InvalidDecisionRequest` (also a `ValueError`) | malformed questions: fewer than 2 options, duplicate or empty ids, empty text; misconfigured adapters |
| `InvalidDecisionResponse` (also a `ValueError`) | responses that do not fit their question (see §2) |
| `UnsupportedCapability` | undeclared primitives, modalities, batch sizes or label spaces — before any model call |
| `ProviderFailure` (also a `RuntimeError`) | the backend failing on a valid, supported request |
| `JevError` and its `JevTimeout`, `JevAuthenticationError`, `JevRateLimited`, `JevMalformedResponse` | a Jev request failing (§6): `__cause__` is the SDK error, `request_id` the service's |
| `JobError` | the base of the decision-job errors (§7); `outcomes` holds every outcome so far, `completed` the answered ones |
| `InvalidJob` (also a `ValueError`) | a job that cannot run as described — before any call |
| `JobFailed` | a provider call failing inside a job (fail-fast): `failed`, `skipped`, `__cause__` |
| `JobLimitExceeded` | a job's `max_depth` or `max_requests` stopping it before the next call |
| `JobTimeout` (also a `TimeoutError`) | a job's `timeout` elapsing |

## 11. Consumers

Planned decision features share this digest, the `kind` discriminators and
`validate_response` rather than defining their own: an optional `Result` at
fallible boundaries ([#263](https://github.com/thekaveh/NNx/issues/263)). `nnx.decisions` does not
depend on any of them. Decision jobs (§7, from
[#245](https://github.com/thekaveh/NNx/issues/245)) were the first consumer to
land, followed by the provider benchmark (§8, from
[#243](https://github.com/thekaveh/NNx/issues/243)) and offline
teacher-distribution datasets (§9, from
[#244](https://github.com/thekaveh/NNx/issues/244)). The local
label-conditioned baseline ([#234](https://github.com/thekaveh/NNx/issues/234))
is `NLIProvider` (§5), and the Jev SDK adapter
([#220](https://github.com/thekaveh/NNx/issues/220)) is `JevProvider` (§6).

## 12. What this does not do

- It does not claim every classifier is a universal decision-maker: the
  fixed-head adapter answers only what its head justifies.
- It does not execute actions based on a decision.
- It does not assume that separately asked questions describe independent
  events; each question is answered on its own.
- It does not call hosted models except through an adapter you build:
  `JevProvider` (§6) is the only one, needs the `jev` extra and your
  credentials, and never falls back to it silently. It does not fine-tune Jev
  models or ship credentials, and vendor numbers are not benchmarks — measure
  them (§8).
- It does not download or train models: `NLIProvider` uses the NLI model the
  caller supplies, as is, and its scores are not calibrated. Training a
  student on stored answers is `nnx.paradigms.offline_distillation` (§9).
