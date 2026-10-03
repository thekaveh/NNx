# 11. Typed decisions

`nnx.decisions` is NNx's **inference branch**: a provider-neutral way to ask a
typed question and get labelled probabilities back, instead of reading
positional logits. A question is one of three primitives; a **provider**
declares what it can answer and returns validated results; the fixed-head
adapter turns a trained NNx classifier into such a provider, and the NLI
adapter scores candidates supplied at inference with a caller-supplied NLI
model. Nothing here
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
adapter ([#220](https://github.com/thekaveh/NNx/issues/220)), a reproducible
decision-provider benchmark ([#243](https://github.com/thekaveh/NNx/issues/243)),
offline teacher-distribution datasets
([#244](https://github.com/thekaveh/NNx/issues/244)), applicative batching for
independent decisions ([#245](https://github.com/thekaveh/NNx/issues/245)) and an
optional `Result` at fallible boundaries
([#263](https://github.com/thekaveh/NNx/issues/263)). None of them has landed;
`nnx.decisions` does not depend on any of them. The local label-conditioned
baseline ([#234](https://github.com/thekaveh/NNx/issues/234)) is
`NLIProvider` (§5).

## 8. What this does not do

- It does not claim every classifier is a universal decision-maker: the
  fixed-head adapter answers only what its head justifies.
- It does not execute actions based on a decision.
- It does not assume that separately asked questions describe independent
  events; each question is answered on its own.
- It does not call hosted models; a hosted provider is a separate adapter
  that declares its own capabilities.
- It does not download or train models: `NLIProvider` uses the NLI model the
  caller supplies, as is, and its scores are not calibrated.
