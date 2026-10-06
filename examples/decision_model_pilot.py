"""Label-conditioned decision-model research pilot (FEAT-023).

One bounded pilot of a **local** model that scores ``(state, question,
candidate description)`` triples with a shared encoder and a shared scalar
head — so a new candidate needs no new output unit — and a measured go/no-go
on whether it beats local baselines. It is neither a new foundation model
nor a reproduction of any hosted service, and it stays **experimental**
until its manifest's predeclared trade-off is met.

Two modes:

- ``--mode mechanics`` (default; CPU, offline, no downloads — run in CI):
  synthetic task families with keyword-described candidates exercise the
  whole recipe end to end — the pre-execution manifest, a registered
  candidate scorer (``nnx.models.register_model_factory``; no new ``Nets``
  member, the fixed-head ``predict`` untouched) trained through
  ``NNModel.train`` on the *fit* families only (Choice and Boolean batches),
  selection on the *select* families, held-out families scored only at the
  end, replay records through ``nnx.decisions.benchmark.collect`` carrying
  the pilot's candidate ids and the question (schema) digest, the
  checkpoint reloaded through the registry, the compute cap enforced (a
  stop record when reached), and a report. Its numbers are **mechanics
  evidence only** — never a quality claim — so its verdict is always
  ``no-go``.
- ``--mode empirical``: writes and validates the empirical manifest, then
  checks its resources (the encoder weights, the NLI and GLiClass
  baselines, the task data). Missing resources yield a **blocked** report
  naming them; the empirical training itself is not part of this
  repository, so even with the resources present the report is blocked
  (saying so) — never fabricated results.

Choice decisions softmax the candidates' scores (cross-entropy across
candidates); a Boolean decision scores its single proposition with one logit
(binary cross-entropy) — both are trained in mechanics mode; Score decisions
are refused.

Run:
    python examples/decision_model_pilot.py --output pilot-out           # mechanics
    python examples/decision_model_pilot.py --mode empirical --output out

Requires no optional extras. ``tests/test_decision_model_pilot.py``
exercises the mechanics; ``tests/test_examples_smoke.py`` runs the bounded
``pilot_workflow()`` helper.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import numbers
import os
import tempfile
import time
import zlib
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
from torch import nn

from nnx import Devices, Losses, NNModel, NNModelParams, NNOptimParams, NNTrainParams
from nnx.decisions import Boolean, Capabilities, Choice, UnsupportedCapability, validate_response
from nnx.decisions.benchmark import Budget, Sample, collect, write_records
from nnx.models import ModelSpec, PositionalInputs, register_model_factory
from nnx.nn.callbacks import Callback
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

FACTORY_ID = "examples.decision-pilot.candidate-scorer"
FACTORY_VERSION = 1
TEMPLATE_ID = "state|question|candidate:v1"
VOCAB = 512
MAX_TOKENS = 12
CHOICE, BOOLEAN = 0, 1


# --- the pre-execution manifest -----------------------------------------------------------------


class ManifestError(ValueError):
    """The pre-execution manifest is incomplete or inconsistent."""


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive(value: Any, *, integer: bool) -> bool:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        return False
    return not integer or float(value).is_integer()


@dataclasses.dataclass(frozen=True, kw_only=True)
class PilotManifest:
    """Everything the pilot commits to before it runs.

    ``resources`` maps each resource (model weights, datasets, baselines) to
    its ``licence`` and permitted ``use``; ``splits`` names the task families
    per role (``fit``, ``select``, ``heldout`` — disjoint, non-empty lists);
    ``candidate_schemas`` the candidate schema id per family; ``seeds``
    (ints), ``baselines`` (each naming the declared resource it needs, or
    ``None`` when the pilot cannot run it — then disclosed as unavailable),
    ``hardware``, ``compute_cap`` (``max_steps`` committed updates and
    ``max_seconds``), the ``stop_rule``, the ``model`` identity (``encoder``
    and ``revision``) and the predeclared promotion ``tradeoff`` (``metric``,
    numeric ``min_gain`` over the best baseline)."""

    resources: Mapping[str, Mapping[str, str]]
    splits: Mapping[str, Sequence[str]]
    candidate_schemas: Mapping[str, str]
    seeds: Sequence[int]
    baselines: Mapping[str, Optional[str]]
    hardware: str
    compute_cap: Mapping[str, float]
    stop_rule: str
    model: Mapping[str, str]
    tradeoff: Mapping[str, Any]

    def validate(self) -> None:
        problems: list[str] = []

        def mapping(name: str) -> Mapping[str, Any]:
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                problems.append(f"{name} must be a mapping, got {type(value).__name__}")
                return {}
            return value

        splits, cap, model = mapping("splits"), mapping("compute_cap"), mapping("model")
        resources, schemas, baselines, tradeoff = (
            mapping("resources"),
            mapping("candidate_schemas"),
            mapping("baselines"),
            mapping("tradeoff"),
        )
        roles: list[set[str]] = []
        for role in ("fit", "select", "heldout"):
            families = splits.get(role)
            if not isinstance(families, (list, tuple)) or not families:
                problems.append(f"splits.{role} must be a non-empty list of family names")
                roles.append(set())
                continue
            if not all(_text(family) for family in families) or len(set(families)) != len(families):
                problems.append(f"splits.{role} must hold distinct family names")
            roles.append(set(families))
        if (roles[0] & roles[1]) or (roles[0] & roles[2]) or (roles[1] & roles[2]):
            problems.append("splits must be disjoint: a held-out family can never enter fit or selection")
        for family in sorted(set().union(*roles)):
            if not _text(schemas.get(family)):
                problems.append(f"candidate_schemas misses family {family!r}")
        seeds = self.seeds
        if (
            not isinstance(seeds, (list, tuple))
            or not seeds
            or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        ):
            problems.append("seeds must be a non-empty list of ints")
        if not (_positive(cap.get("max_steps"), integer=True) and _positive(cap.get("max_seconds"), integer=False)):
            problems.append("compute_cap needs a positive integer max_steps and positive max_seconds (the budget)")
        if not (_text(model.get("encoder")) and _text(model.get("revision"))):
            problems.append("model identity needs an encoder and a revision")
        if not _text(self.hardware):
            problems.append("hardware is empty")
        if not _text(self.stop_rule):
            problems.append("stop_rule is empty")
        if not baselines:
            problems.append("baselines is empty: the pilot is measured against declared baselines")
        for baseline, resource in baselines.items():
            if resource is not None and resource not in resources:
                problems.append(f"baseline {baseline!r} needs resource {resource!r}, which has no licence entry")
        for resource, terms in resources.items():
            if not (isinstance(terms, Mapping) and _text(terms.get("licence")) and _text(terms.get("use"))):
                problems.append(f"resource {resource!r} needs a licence and a permitted use")
        if not (_text(tradeoff.get("metric")) and _positive(tradeoff.get("min_gain"), integer=False)):
            problems.append("tradeoff must predeclare a metric and a positive numeric min_gain")
        if problems:
            raise ManifestError("invalid pilot manifest: " + "; ".join(problems))

    def state(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), sort_keys=True, default=list))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.state(), sort_keys=True).encode("utf-8")).hexdigest()


# --- the candidate scorer -----------------------------------------------------------------------


def tokenize(text: str) -> list[int]:
    """Stable hashed bag of words (no vocabulary file, no download)."""
    ids = [zlib.crc32(word.lower().encode("utf-8")) % (VOCAB - 1) + 1 for word in text.split()]
    return (ids + [0] * MAX_TOKENS)[:MAX_TOKENS]


class CandidateScorer(nn.Module):
    """``score(state, question, candidate)``: a shared bag-of-words encoder
    and one scalar head over ``[s, q, c, s*c, q*c]``. The head's size never
    depends on the number of candidates; every candidate is scored by the
    same function, so the scores are permutation-equivariant."""

    def __init__(self, width: int = 32, hidden: int = 32) -> None:
        super().__init__()
        self.embed = nn.EmbeddingBag(VOCAB, width, mode="mean", padding_idx=0)
        self.head = nn.Sequential(nn.Linear(5 * width, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1, ids.size(-1))
        return self.embed(flat).reshape(*ids.shape[:-1], -1)

    def forward(self, state: torch.Tensor, question: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """``state`` ``[B, L]``, ``question`` ``[B, L]``, ``candidates``
        ``[B, C, L]`` → scores ``[B, C]``."""
        s, q, c = self.encode(state), self.encode(question), self.encode(candidates)
        s, q = s.unsqueeze(1).expand_as(c), q.unsqueeze(1).expand_as(c)
        return self.head(torch.cat([s, q, c, s * c, q * c], dim=-1)).squeeze(-1)


def ensure_registered() -> None:
    """Register (or re-register: a module re-run defines a new class) the
    candidate scorer's factory for this process."""
    register_model_factory(FACTORY_ID, FACTORY_VERSION, lambda config: CandidateScorer(**config), replace=True)


def candidate_tensor(descriptions: Sequence[str]) -> torch.Tensor:
    return torch.tensor([tokenize(text) for text in descriptions])


def choice_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """A Choice decision: cross-entropy across the candidates' scores."""
    return nn.functional.cross_entropy(scores, labels)


def boolean_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """A Boolean decision: one logit per proposition, binary cross-entropy."""
    return nn.functional.binary_cross_entropy_with_logits(scores.squeeze(-1), labels.float())


def pilot_train_step(ctx: Any) -> NNEvaluationDataPoint:
    """The pilot's training step: a Choice batch ``(state, question,
    candidates, label index, CHOICE)`` takes cross-entropy across its
    candidates; a Boolean batch ``(state, proposition, proposition, 0/1,
    BOOLEAN)`` takes binary cross-entropy on its one logit."""
    if ctx.grad_clip_norm is not None or ctx.accumulate_grad_batches != 1 or ctx.scaler is not None:
        raise ValueError("pilot_train_step runs plain FP32 updates: no clipping, accumulation or AMP scaler")
    model = ctx.model
    model.net.train()
    state, question, candidates, labels, kind = (tensor.to(model.device) for tensor in ctx.batch)
    ctx.optimizer.zero_grad()
    scores = model.net(state, question, candidates)
    boolean = int(kind[0]) == BOOLEAN
    loss = boolean_loss(scores, labels) if boolean else choice_loss(scores, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite pilot training loss ({float(loss)!r})")
    loss.backward()
    ctx.optimizer.step()
    ctx.report_update()
    predicted = (scores.squeeze(-1) > 0).long() if boolean else scores.argmax(dim=1)
    edp = NNEvaluationDataPoint.of(Y=labels.cpu().numpy(), Y_hat=predicted.detach().cpu().numpy())
    return edp.with_loss(value=float(loss.detach()))


class PilotProvider:
    """A ``DecisionProvider`` over a trained scorer: Choice decisions softmax
    the candidates' scores, a Boolean decision takes ``sigmoid`` of its one
    proposition's logit; Score decisions are refused."""

    def __init__(self, scorer: CandidateScorer, *, revision: str) -> None:
        self.scorer = scorer
        self.revision = revision

    def capabilities(self) -> Capabilities:
        return Capabilities(
            primitives=frozenset({"choice", "boolean"}), modalities=frozenset({"text"}), dynamic_labels=True
        )

    def decide(self, question: Any, inputs: Any) -> list[Any]:
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        self.capabilities().check(question, modality="text", batch_size=len(texts))
        if isinstance(question, Choice):
            descriptions = [option.description for option in question.options]
        elif isinstance(question, Boolean):
            descriptions = [question.prompt]
        else:
            raise UnsupportedCapability(f"the pilot scores choice and boolean decisions, not {question.kind}")
        state = torch.tensor([tokenize(text) for text in texts])
        prompt = torch.tensor([tokenize(question.prompt)] * len(texts))
        cands = candidate_tensor(descriptions).unsqueeze(0).expand(len(texts), -1, -1)
        was_training = self.scorer.training
        self.scorer.eval()
        try:
            with torch.no_grad():
                scores = self.scorer(state, prompt, cands)
        finally:
            self.scorer.train(was_training)
        results = []
        for row in scores:
            if isinstance(question, Choice):
                probabilities = torch.softmax(row.double(), dim=0).tolist()
                total = math.fsum(probabilities)
                response = {option: p / total for option, p in zip(question.option_ids, probabilities, strict=True)}
            else:
                p_true = float(torch.sigmoid(row[0].double()))
                response = {"true": p_true, "false": 1.0 - p_true}
            results.append(validate_response(question, response, provider=f"decision-pilot@{self.revision}"))
        return results


class ComputeCap(Callback):
    """Enforces the manifest's compute cap: stops at the update boundary
    where ``max_steps`` committed updates or ``max_seconds`` are reached."""

    distributed = "all"

    def __init__(self, max_steps: int, max_seconds: float, clock: Any = time.monotonic) -> None:
        self.max_steps, self.max_seconds, self.clock = int(max_steps), float(max_seconds), clock
        self.reason: Optional[str] = None
        self.updates = 0

    def on_train_begin(self, ctx: Any) -> None:
        self.started = self.clock()
        ctx.update_listeners.append(self.on_update)

    def on_update(self, ctx: Any) -> None:
        self.updates = ctx.committed_updates
        if self.updates >= self.max_steps:
            self.reason, ctx.stop_at_update = "max_steps", True
        elif self.clock() - self.started >= self.max_seconds:
            self.reason, ctx.stop_at_update = "max_seconds", True


# --- synthetic task families (mechanics mode) ---------------------------------------------------

FAMILIES: dict[str, dict[str, str]] = {
    # family -> {candidate id: description}; a state mentions its label's keywords
    "topic-sports": {"football": "football match goal team", "tennis": "tennis serve racket court"},
    "topic-weather": {"rain": "rain storm wet cloud", "sun": "sun warm bright clear", "snow": "snow cold ice winter"},
    "topic-food": {"pasta": "pasta noodle sauce tomato", "salad": "salad leaf green fresh"},
    "topic-music": {"guitar": "guitar string chord riff", "drum": "drum beat rhythm kick"},
    "topic-travel": {"train": "train rail station ticket", "plane": "plane flight airport wing"},
}
SPLITS = {
    "fit": ["topic-sports", "topic-weather", "topic-food"],
    "select": ["topic-music"],
    "heldout": ["topic-travel"],
}


def mechanics_manifest() -> PilotManifest:
    return PilotManifest(
        resources={"synthetic-families": {"licence": "generated in-process", "use": "mechanics testing only"}},
        splits=SPLITS,
        candidate_schemas={family: f"{family}:keyword-descriptions:v1" for family in FAMILIES},
        seeds=(0,),
        baselines={"fixed-supervised-head": "synthetic-families", "nli": None, "gliclass": None},
        hardware="cpu",
        compute_cap={"max_steps": 400, "max_seconds": 120.0},
        stop_rule="stop at the first of max_steps committed updates, max_seconds or the planned epochs; keep every "
        "artifact",
        model={"encoder": "hashed-bag-of-words", "revision": "mechanics-v1"},
        tradeoff={"metric": "heldout_accuracy", "min_gain": 0.05, "max_latency_ratio": 2.0},
    )


def empirical_manifest() -> PilotManifest:
    """What the empirical study commits to (the resources it needs are
    local paths named by environment variables — nothing is downloaded)."""
    licence = {"licence": "as published by its owner (to be recorded with the resource)", "use": "research evaluation"}
    return PilotManifest(
        resources={name: dict(licence) for name in EMPIRICAL_RESOURCES},
        splits={"fit": ["known-families"], "select": ["selection-families"], "heldout": ["unseen-families"]},
        candidate_schemas={
            "known-families": "task-schemas:v1",
            "selection-families": "task-schemas:v1",
            "unseen-families": "task-schemas:v1",
        },
        seeds=(0, 1, 2),
        baselines={
            "fixed-supervised-head": "task-family data",
            "embedding": "encoder weights",
            "nli": "NLI baseline",
            "gliclass": "GLiClass baseline",
        },
        hardware="one CUDA GPU (declared by the operator)",
        compute_cap={"max_steps": 20000, "max_seconds": 4 * 3600.0},
        stop_rule="stop at the compute cap; keep every artifact; report negatives",
        model={"encoder": "declared encoder (EMPIRICAL encoder weights)", "revision": "unresolved"},
        tradeoff={"metric": "heldout_accuracy", "min_gain": 0.03, "max_latency_ratio": 2.0},
    )


def question_for(family: str) -> Choice:
    options = tuple((cid, description) for cid, description in FAMILIES[family].items())
    return Choice(f"Which {family.split('-')[1]} does this state describe?", options)


def family_rows(family: str, n: int, generator: torch.Generator) -> list[tuple[str, str]]:
    """``(state text, label id)`` rows: two of the label's keywords plus filler."""
    ids = list(FAMILIES[family])
    filler = ["the", "a", "today", "we", "saw", "some", "there", "was"]
    rows = []
    for _ in range(n):
        label = ids[int(torch.randint(len(ids), (1,), generator=generator))]
        words = FAMILIES[family][label].split()
        picks = [words[int(i)] for i in torch.randperm(len(words), generator=generator)[:2]]
        noise = [filler[int(i)] for i in torch.randint(len(filler), (3,), generator=generator)]
        rows.append((" ".join(noise[:2] + picks + noise[2:]), label))
    return rows


def boolean_question(description: str) -> Boolean:
    return Boolean(f"Does the state mention {description}?")


def batches_for(
    families: Sequence[str], n: int, generator: torch.Generator, *, boolean: bool = False
) -> list[tuple[torch.Tensor, ...]]:
    """A Choice batch per family (a family's candidates share a width) —
    with ``boolean``, also a Boolean batch per family: each state against
    one candidate's proposition, true when it is the state's label."""
    out = []
    for family in families:
        rows = family_rows(family, n, generator)
        question = question_for(family)
        cands = candidate_tensor([option.description for option in question.options])
        state = torch.tensor([tokenize(text) for text, _ in rows])
        prompt = torch.tensor([tokenize(question.prompt)] * n)
        labels = torch.tensor([question.option_ids.index(label) for _, label in rows])
        choice = (state, prompt, cands.unsqueeze(0).expand(n, -1, -1).contiguous(), labels)
        if not boolean:
            out.append(choice)
            continue
        out.append((*choice, torch.full((n,), CHOICE)))
        picked = torch.randint(len(question.options), (n,), generator=generator)
        propositions = [boolean_question(question.options[int(i)].description).prompt for i in picked]
        prop = torch.tensor([tokenize(text) for text in propositions])
        truth = (picked == labels).long()
        out.append((state, prop, prop.unsqueeze(1), truth, torch.full((n,), BOOLEAN)))
    return out


# --- the pilot ------------------------------------------------------------------------------------


EMPIRICAL_RESOURCES = {
    "encoder weights": "NNX_PILOT_ENCODER",
    "NLI baseline": "NNX_PILOT_NLI_MODEL",
    "GLiClass baseline": "NNX_PILOT_GLICLASS_MODEL",
    "task-family data": "NNX_PILOT_DATA",
}


def check_resources(environ: Mapping[str, str]) -> list[str]:
    """The empirical resources that are not available locally (each is a
    path named by an environment variable — nothing is downloaded)."""
    return [
        f"{name} (${variable})"
        for name, variable in EMPIRICAL_RESOURCES.items()
        if not environ.get(variable) or not os.path.exists(environ[variable])
    ]


def write_json(path: str, value: Any) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
    return path


def choice_accuracy(provider: PilotProvider, family: str, rows: Sequence[tuple[str, str]]) -> float:
    question = question_for(family)
    results = provider.decide(question, [text for text, _ in rows])
    hits = sum(
        max(result.distribution, key=lambda item: item[1])[0] == label
        for result, (_, label) in zip(results, rows, strict=True)
    )
    return hits / len(rows)


def boolean_accuracy(provider: PilotProvider, family: str, rows: Sequence[tuple[str, str]]) -> float:
    """Each state against every candidate's proposition: true for its label."""
    hits = total = 0
    for option_id, description in FAMILIES[family].items():
        results = provider.decide(boolean_question(description), [text for text, _ in rows])
        for result, (_, label) in zip(results, rows, strict=True):
            hits += (result.p_true > 0.5) == (label == option_id)
            total += 1
    return hits / total


def run_mechanics(output: str, *, max_steps: Optional[int] = None, clock: Any = time.monotonic) -> dict[str, Any]:
    """The mechanics pilot: every artifact under ``output`` (paths in the
    report are relative to it); returns the report."""
    manifest = mechanics_manifest()
    if max_steps is not None:
        manifest = dataclasses.replace(manifest, compute_cap={**manifest.compute_cap, "max_steps": max_steps})
    manifest.validate()
    os.makedirs(output, exist_ok=True)
    artifacts: dict[str, Optional[str]] = {"manifest": "manifest.json"}
    write_json(os.path.join(output, "manifest.json"), manifest.state())
    ensure_registered()
    torch.manual_seed(manifest.seeds[0])
    generator = torch.Generator().manual_seed(manifest.seeds[0])
    fit = batches_for(manifest.splits["fit"], 24, generator, boolean=True)  # the fit families only
    select = batches_for(manifest.splits["select"], 24, generator)  # selection on the select families only
    spec = ModelSpec(FACTORY_ID, FACTORY_VERSION, {"width": 32, "hidden": 32}, seed=manifest.seeds[0])
    model = NNModel(
        params=NNModelParams(net=spec, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        batch_adapter=PositionalInputs(3),
    )
    planned_epochs = 12
    cap = ComputeCap(int(manifest.compute_cap["max_steps"]), float(manifest.compute_cap["max_seconds"]), clock)
    previous = os.getcwd()
    os.chdir(output)
    try:
        run = model.train(
            params=NNTrainParams(
                n_epochs=planned_epochs,
                train_loader=fit,
                val_loader=select,
                optim=NNOptimParams.builder().adam(max_lr=0.02).build(),
                data_id=f"pilot:{manifest.digest()[:12]}",
                overwrite_existing=True,
            ),
            callbacks=[cap],
            train_step_fn=pilot_train_step,
        )
        best_path = os.path.join("runs", run.id, "checkpoints", "best.pt")
        artifacts["checkpoint"] = best_path if os.path.exists(best_path) else None
        best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    finally:
        os.chdir(previous)
    stop = None
    planned_steps = planned_epochs * len(fit)
    if cap.reason is not None and cap.updates < planned_steps:  # the cap, not the plan, ended the run
        committed_epochs = sorted({idp.epoch_idx for idp in run.idps})
        stop = {
            "reason": f"compute_cap:{cap.reason}",
            "max_steps": manifest.compute_cap["max_steps"],
            "max_seconds": manifest.compute_cap["max_seconds"],
            "planned_steps": planned_steps,
            "taken_steps": cap.updates,
            # what the checkpoints hold: a partial epoch is never committed
            "committed_epochs": len(committed_epochs),
            "committed_steps": len(run.idps),
        }
    artifacts["stop"] = "stop.json" if stop is not None else None
    if stop is not None:
        write_json(os.path.join(output, "stop.json"), stop)
    report: dict[str, Any] = {
        "mode": "mechanics",
        "verdict": "no-go",
        "why": "mechanics evidence only: synthetic families measure plumbing, not quality; the empirical mode "
        "needs the manifest's resources",
        "manifest_digest": manifest.digest(),
        "baselines": _disclosed_baselines(manifest),
        "experimental": True,
    }
    if best is not None:
        reloaded = NNModel.from_checkpoint(best, batch_adapter=PositionalInputs(3))  # rebuilt through the registry
        scorer = reloaded.net
        assert isinstance(scorer, CandidateScorer)
        provider = PilotProvider(scorer, revision=manifest.model["revision"])
        # Held-out families: scored once, at the end — never fit, never selection.
        heldout_rows = {family: family_rows(family, 40, generator) for family in manifest.splits["heldout"]}
        samples = [
            Sample(
                id=f"{family}-{index}",
                question=question_for(family),
                input=text,
                label=label,
                family=family,
                heldout=True,
            )
            for family, rows in heldout_rows.items()
            for index, (text, label) in enumerate(rows)
        ]
        collection = collect(
            provider,
            samples,
            provider_id="decision-pilot",
            budget=Budget(max_calls=100),
            batch_size=16,
            revision=manifest.model["revision"],
            prompt_identity=TEMPLATE_ID,
        )
        write_records(os.path.join(output, "replay.jsonl"), collection.records)
        artifacts["replay"] = "replay.jsonl"
        report["mechanics_heldout_choice_accuracy"] = {
            family: choice_accuracy(provider, family, rows) for family, rows in heldout_rows.items()
        }
        report["mechanics_heldout_boolean_accuracy"] = {
            family: boolean_accuracy(provider, family, rows) for family, rows in heldout_rows.items()
        }
    else:
        artifacts["replay"] = None  # no committed epoch: nothing to score
    trial = {
        "run_id": run.id,
        "splits": manifest.splits,
        "template": TEMPLATE_ID,
        "model": dict(manifest.model),
        "spec": spec.state(),
        "candidate_schemas": {family: manifest.candidate_schemas[family] for family in FAMILIES},
        "question_digests": {family: question_for(family).digest() for family in FAMILIES},
        "stop": stop,
    }
    write_json(os.path.join(output, "trial.json"), trial)
    artifacts["trial"] = "trial.json"
    report["artifacts"] = artifacts
    # A stop record exists only when the compute cap was reached.
    report["unproduced"] = sorted(name for name in ("checkpoint", "replay", "trial") if not artifacts.get(name))
    write_json(os.path.join(output, "report.json"), report)
    return report


def _disclosed_baselines(manifest: PilotManifest) -> dict[str, str]:
    """Each declared baseline's status: none is run in mechanics mode."""
    return {
        baseline: (
            "unavailable: no resource declared (disclosed, not run)"
            if resource is None
            else f"not run in mechanics mode (declared on {resource!r}; measured only in the empirical study)"
        )
        for baseline, resource in manifest.baselines.items()
    }


def run_empirical(output: str, environ: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """The empirical pilot: the manifest is written and validated first,
    then its resources are checked. It reports blocked — naming the missing
    resources, or, with every resource present, that the empirical training
    is not part of this repository. Nothing is trained or measured."""
    manifest = empirical_manifest()
    manifest.validate()
    os.makedirs(output, exist_ok=True)
    write_json(os.path.join(output, "manifest.json"), manifest.state())
    missing = check_resources(os.environ if environ is None else environ)
    report = {
        "mode": "empirical",
        "verdict": "blocked",
        "manifest_digest": manifest.digest(),
        "missing_resources": missing,
        # the manifest commits to roles and budgets; these are recorded only
        # once the operator supplies the resources
        "unrecorded": ["each resource's licence and permitted use", "the encoder revision"],
        "why": (
            "the pilot does not run, and reports nothing, without its declared resources"
            if missing
            else "every resource is present, but the empirical training and baseline runs are not part of this "
            "repository: run them under this manifest and report through replay records"
        ),
        "baselines": _disclosed_baselines(manifest),
        "artifacts": {"manifest": "manifest.json"},
        "unproduced": ["checkpoint", "replay", "trial"],
        "experimental": True,
    }
    write_json(os.path.join(output, "report.json"), report)
    return report


def pilot_workflow() -> dict[str, Any]:
    """Bounded helper for the examples smoke test: mechanics end to end and
    a blocked empirical run."""
    with tempfile.TemporaryDirectory() as output:
        mechanics = run_mechanics(os.path.join(output, "mechanics"), max_steps=48)
        if mechanics["verdict"] != "no-go" or mechanics["unproduced"]:
            raise RuntimeError(f"mechanics pilot incomplete: {mechanics}")
        blocked = run_empirical(os.path.join(output, "empirical"), environ={})
        if blocked["verdict"] != "blocked":
            raise RuntimeError("an empirical pilot without resources must report blocked")
        return {
            "mechanics": mechanics["verdict"],
            "empirical": blocked["verdict"],
            "stop": mechanics["artifacts"].get("stop") is not None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-conditioned decision-model pilot")
    parser.add_argument("--mode", choices=["mechanics", "empirical"], default="mechanics")
    parser.add_argument("--output", default="pilot-out")
    args = parser.parse_args()
    report = run_mechanics(args.output) if args.mode == "mechanics" else run_empirical(args.output)
    keys = ("mode", "verdict", "artifacts", "unproduced", "missing_resources")
    print(json.dumps({key: report[key] for key in keys if key in report}, indent=2))


if __name__ == "__main__":
    main()
