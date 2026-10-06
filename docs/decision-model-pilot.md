# 24. Decision-model pilot

*Status: **experimental — no-go**. The mechanics are verified offline; the
empirical study is **blocked**: its resources are not supplied, and its
training and baseline runs are not part of this repository. Nothing on this
page is a quality claim.*

## 1. Question

Can a **local** model that scores `(state, question, candidate description)`
triples — one shared encoder, one shared scalar head — beat NNx's local
baselines on decision tasks, including task families it never saw? It is a
bounded research pilot (FEAT-023), not a new foundation model and not a
reproduction of any hosted service.

## 2. Recipe

The recipe lives in an example,
[`examples/decision_model_pilot.py`](https://github.com/thekaveh/NNx/blob/main/examples/decision_model_pilot.py),
and adds no library surface: the scorer registers through
`nnx.models.register_model_factory` (no new `Nets` member; the fixed-head
`predict` is unchanged) and trains through `NNModel.train`.

- **Scoring.** `score(state, question, candidate) = head([s, q, c, s·c, q·c])`
  over a shared encoder. A new candidate is just another description: the
  head's size never changes, and scores are permutation-equivariant across
  candidates.
- **Choice** softmaxes the candidates' scores (cross-entropy across
  candidates); a **Boolean** scores its one proposition with one logit
  (binary cross-entropy). The pilot's training step trains both on the fit
  families; on the synthetic families the Boolean path's measured effect is
  near the trivial baseline — mechanics wiring, not a result. **Score**
  decisions are refused.
- The trained scorer answers as a `DecisionProvider`, so its predictions go
  through `nnx.decisions.benchmark.collect` into replay records
  ([Typed decisions §8](decisions.md)) carrying the pilot's candidate ids, the
  question (schema) digest, the template id and the model revision.

## 3. The pre-execution manifest

Before anything runs, a `PilotManifest` records: each resource's licence and
permitted use (the empirical manifest's are placeholders until the operator
supplies the resources — its blocked report lists them as unrecorded); the task-family splits (`fit`, `select`, `heldout` — disjoint);
the candidate schemas per family; the seeds; the baselines and the resource
each needs; the hardware; the compute cap (`max_steps`, `max_seconds`); the
stop rule; the model identity (encoder, revision); and the **predeclared
promotion trade-off** (`metric`, `min_gain` over the best baseline,
`max_latency_ratio`). A manifest lacking a split, the budget or the model
identity is rejected.

Held-out families never enter fitting or selection; they are scored once, at
the end. The compute cap is enforced at the update boundary (`max_steps`
committed updates or `max_seconds`, whichever first); a run that reaches it
writes a stop record and keeps every artifact — with no checkpoint if no
epoch completed, marked unproduced.

## 4. Modes

| Mode | Runs | Produces | Verdict |
|---|---|---|---|
| `mechanics` (default, CI) | Offline, CPU, no downloads: synthetic keyword-described task families exercise the whole recipe | manifest, checkpoint (reloaded through the registry), replay records, trial record (split, template, model revision, question digests), stop record when capped, report | always **no-go**: synthetic families measure plumbing, not quality |
| `empirical` | Writes and validates its own manifest, then checks resources: encoder weights, the NLI and GLiClass baselines, the task-family data (local paths named by environment variables; nothing is downloaded) | the manifest and a **blocked** report naming the missing resources — or, with all present, saying the empirical runs happen outside this repository | **blocked** |

## 5. Completion and promotion

Completion is a reproducible report plus a go/no-go. The report links every
produced artifact (manifest, checkpoint, replay records, trial and stop
records) and marks the unproduced ones; a negative result still completes the
pilot. Promotion out of *experimental* needs the manifest's predeclared
trade-off met on the empirical study — quality, calibration, latency and
memory, label paraphrase and permutation, and unseen-family results, negatives
included, all through replay records. Until then the capability tables keep
the pilot experimental ([Framework comparison §3.11](comparison.md)).

## 6. Current result

| Item | Status |
|---|---|
| Mechanics (offline) | Verified by `tests/test_decision_model_pilot.py` and the examples smoke test |
| Empirical study | **Blocked**: encoder weights, NLI and GLiClass baselines and task-family data not supplied; its training and baseline runs are not part of this repository |
| Baselines | Fixed supervised head: needs the empirical data; NLI, GLiClass: not available — disclosed, not run |
| Verdict | **No-go** (experimental) |
