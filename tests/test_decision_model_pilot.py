"""FEAT-023: the label-conditioned decision-model pilot's mechanics."""

from __future__ import annotations

import dataclasses
import json
import os
import runpy
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pilot = runpy.run_path(str(ROOT / "examples" / "decision_model_pilot.py"), run_name="__nnx_pilot_tests__")

from nnx.decisions import Boolean, Choice, Score, UnsupportedCapability  # noqa: E402
from nnx.decisions.benchmark import read_records  # noqa: E402
from nnx.nn.enum.nets import Nets  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")


# ---------------- the manifest ----------------


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda m: dataclasses.replace(m, splits={**m.splits, "heldout": []}), "splits.heldout"),
        (
            lambda m: dataclasses.replace(m, splits={"fit": m.splits["fit"], "select": m.splits["select"]}),
            "splits.heldout",
        ),
        (lambda m: dataclasses.replace(m, compute_cap={}), "compute_cap"),
        (lambda m: dataclasses.replace(m, compute_cap={"max_steps": 0, "max_seconds": 1}), "compute_cap"),
        (lambda m: dataclasses.replace(m, model={"encoder": "x"}), "model identity"),
        (lambda m: dataclasses.replace(m, splits={**m.splits, "select": m.splits["heldout"]}), "disjoint"),
        (lambda m: dataclasses.replace(m, tradeoff={}), "tradeoff"),
        (lambda m: dataclasses.replace(m, resources={"x": {"licence": ""}}), "licence"),
        (lambda m: dataclasses.replace(m, hardware=" "), "hardware"),
        (lambda m: dataclasses.replace(m, seeds=5), "seeds"),
        (lambda m: dataclasses.replace(m, seeds=(True,)), "seeds"),
        (lambda m: dataclasses.replace(m, compute_cap={"max_steps": 0.5, "max_seconds": 1}), "compute_cap"),
        (lambda m: dataclasses.replace(m, compute_cap={"max_steps": 10, "max_seconds": float("inf")}), "compute_cap"),
        (lambda m: dataclasses.replace(m, model={"encoder": " ", "revision": "r"}), "model identity"),
        (lambda m: dataclasses.replace(m, tradeoff={"metric": "", "min_gain": "lots"}), "tradeoff"),
        (lambda m: dataclasses.replace(m, baselines={}), "baselines"),
        (lambda m: dataclasses.replace(m, baselines={"x": "undeclared"}), "undeclared"),
        (lambda m: dataclasses.replace(m, splits={**m.splits, "fit": "topic-sports"}), "splits.fit"),
        (lambda m: dataclasses.replace(m, splits=None), "splits"),
        (lambda m: dataclasses.replace(m, model=None), "model"),
        (lambda m: dataclasses.replace(m, candidate_schemas={}), "candidate_schemas"),
    ],
)
def test_a_manifest_lacking_split_budget_or_model_identity_is_rejected(change, match):
    manifest = pilot["mechanics_manifest"]()
    manifest.validate()
    with pytest.raises(pilot["ManifestError"], match=match):
        change(manifest).validate()


# ---------------- scoring mechanics ----------------


def _scorer():
    torch.manual_seed(0)
    return pilot["CandidateScorer"](width=8, hidden=8)


def test_choice_softmaxes_candidates_and_boolean_takes_one_logit():
    provider = pilot["PilotProvider"](_scorer(), revision="t")
    two = Choice("Which?", (("a", "alpha thing"), ("b", "beta thing")))
    three = Choice("Which?", (("a", "alpha thing"), ("b", "beta thing"), ("c", "gamma thing")))
    for question, width in ((two, 2), (three, 3)):
        (result,) = provider.decide(question, ["some state text"])
        probabilities = torch.tensor([p for _, p in result.distribution])
        assert probabilities.shape == (width,) and float(probabilities.sum()) == pytest.approx(1.0)
    (yes,) = provider.decide(Boolean("Is it alpha?"), ["some state"])
    assert 0.0 < yes.p_true < 1.0
    with pytest.raises(UnsupportedCapability):
        provider.decide(Score("How much?", (("lo", "low"), ("hi", "high"))), ["x"])


def test_a_new_candidate_needs_no_head_change_and_scores_are_permutation_equivariant():
    scorer = _scorer()
    # the head's shape is fixed by width/hidden, never by the number of candidates
    assert [tuple(p.shape) for p in scorer.head.parameters()] == [(8, 40), (8,), (1, 8), (1,)]
    state = torch.tensor([pilot["tokenize"]("we saw rain today")])
    question = torch.tensor([pilot["tokenize"]("Which weather?")])
    two = pilot["candidate_tensor"](["rain storm", "sun warm"]).unsqueeze(0)
    cands = pilot["candidate_tensor"](["rain storm", "sun warm", "snow cold"]).unsqueeze(0)
    assert scorer(state, question, two).shape == (1, 2)
    scores = scorer(state, question, cands)
    assert scores.shape == (1, 3) and torch.allclose(scores[:, :2], scorer(state, question, two))
    permuted = scorer(state, question, cands[:, [2, 0, 1]])
    assert torch.allclose(permuted, scores[:, [2, 0, 1]])


def test_the_step_trains_choice_with_cross_entropy_and_boolean_with_bce(monkeypatch):
    calls = []
    for name in ("choice_loss", "boolean_loss"):
        original = pilot[name]

        def spy(scores, labels, _name=name, _original=original):
            calls.append((_name, tuple(scores.shape)))
            return _original(scores, labels)

        monkeypatch.setitem(pilot["pilot_train_step"].__globals__, name, spy)
    generator = torch.Generator().manual_seed(0)
    batches = pilot["batches_for"](["topic-weather"], 6, generator, boolean=True)
    assert [int(batch[4][0]) for batch in batches] == [pilot["CHOICE"], pilot["BOOLEAN"]]

    class Ctx:
        grad_clip_norm, accumulate_grad_batches, scaler = None, 1, None

        def __init__(self, batch, model, optimizer):
            self.batch, self.model, self.optimizer = batch, model, optimizer

        def report_update(self):
            calls.append(("update", None))

    class Model:
        net = _scorer()
        device = torch.device("cpu")

    optimizer = torch.optim.SGD(Model.net.parameters(), lr=0.1)
    for batch in batches:
        pilot["pilot_train_step"](Ctx(batch, Model, optimizer))
    assert calls == [("choice_loss", (6, 3)), ("update", None), ("boolean_loss", (6, 1)), ("update", None)]


# ---------------- the mechanics run ----------------


@pytest.fixture(scope="module")
def mechanics(tmp_path_factory):
    out = tmp_path_factory.mktemp("pilot")
    previous = os.getcwd()
    os.chdir(out)
    try:
        report = pilot["run_mechanics"](str(out / "run"), max_steps=24)
    finally:
        os.chdir(previous)
    return out / "run", report


def test_the_recipe_registers_through_feat_006_without_a_new_nets_member(mechanics):
    from nnx.models import registered_model_factories

    assert (pilot["FACTORY_ID"], pilot["FACTORY_VERSION"]) in registered_model_factories()
    assert not [net for net in Nets if "pilot" in net.value or "candidate" in net.value]


def test_the_checkpoint_reconstructs_with_candidate_aligned_records_and_identities(mechanics):
    out, report = mechanics
    from nnx.models import PositionalInputs
    from nnx.nn.nn_model import NNModel
    from nnx.nn.params.nn_checkpoint import NNCheckpoint

    checkpoint = NNCheckpoint.from_file(str(out / report["artifacts"]["checkpoint"]))
    rebuilt = NNModel.from_checkpoint(checkpoint, batch_adapter=PositionalInputs(3))
    assert isinstance(rebuilt.net, pilot["CandidateScorer"])
    trial = json.loads((out / "trial.json").read_text(encoding="utf-8"))
    assert trial["template"] == pilot["TEMPLATE_ID"] and trial["model"]["revision"] == "mechanics-v1"
    assert trial["spec"]["id"] == pilot["FACTORY_ID"] and trial["splits"]["heldout"] == ["topic-travel"]
    records = read_records(out / "replay.jsonl")
    question = pilot["question_for"]("topic-travel")
    assert records and {r.question_digest for r in records} == {question.digest()}
    assert {r.prompt_identity for r in records} == {pilot["TEMPLATE_ID"]} and {r.revision for r in records} == {
        "mechanics-v1"
    }
    for record in records:  # candidate-aligned: one probability per candidate id, in the question's order
        assert [option for option, _ in record.distribution] == list(question.option_ids)
    assert trial["question_digests"]["topic-travel"] == question.digest()


def test_heldout_families_never_enter_fit_or_selection(monkeypatch, tmp_path):
    from nnx.nn.nn_model import NNModel

    def tokens(texts):
        return {token for text in texts for token in pilot["tokenize"](text) if token}

    heldout = pilot["SPLITS"]["heldout"]
    held = tokens(d for f in heldout for d in pilot["FAMILIES"][f].values())
    # hashed ids collide: judge only the ids no fit/select text can produce
    others = [f for f in pilot["FAMILIES"] if f not in heldout]
    allowed = tokens(
        [d for f in others for d in pilot["FAMILIES"][f].values()]
        + [pilot["question_for"](f).prompt for f in others]
        + [pilot["boolean_question"](d).prompt for f in others for d in pilot["FAMILIES"][f].values()]
        + ["the a today we saw some there was"]
    )
    held -= allowed
    assert held  # held-out-only ids exist
    seen_tokens: set[int] = set()
    original = NNModel.train

    def spy(self, params, *args, **kwargs):
        for loader in (params.train_loader, params.val_loader):
            for batch in loader:
                for tensor in batch[:3]:
                    seen_tokens.update(int(token) for token in tensor.flatten() if token)
        return original(self, params, *args, **kwargs)

    monkeypatch.setattr(NNModel, "train", spy)
    pilot["run_mechanics"](str(tmp_path / "spy"), max_steps=24)
    assert seen_tokens and not (seen_tokens & held)  # no held-out candidate description reached fit or selection


def test_budget_exhaustion_records_a_stop_and_keeps_the_artifacts(mechanics):
    out, report = mechanics
    stop = json.loads((out / "stop.json").read_text(encoding="utf-8"))
    assert stop["reason"] == "compute_cap:max_steps" and stop["taken_steps"] == 24 < stop["planned_steps"]
    assert stop["committed_steps"] == 24 and stop["committed_epochs"] == 4  # 6 updates per epoch
    assert report["artifacts"]["stop"] == "stop.json" and report["unproduced"] == []
    for name in ("manifest", "checkpoint", "replay", "trial", "stop"):
        assert (out / report["artifacts"][name]).exists(), name  # paths relative to the report


def test_the_cap_is_never_exceeded_and_a_capped_first_epoch_produces_no_checkpoint(tmp_path):
    report = pilot["run_mechanics"](str(tmp_path / "tiny"), max_steps=1)
    stop = json.loads((tmp_path / "tiny" / "stop.json").read_text(encoding="utf-8"))
    assert stop["taken_steps"] == 1  # stopped at the first update boundary, mid-epoch
    assert stop["committed_steps"] == 0 and stop["committed_epochs"] == 0
    assert report["artifacts"]["checkpoint"] is None and report["unproduced"] == ["checkpoint", "replay"]


def test_max_seconds_is_enforced_with_an_injected_clock(tmp_path):
    ticks = iter(range(10_000))
    report = pilot["run_mechanics"](str(tmp_path / "timed"), clock=lambda: float(next(ticks)) * 30.0)
    stop = json.loads((tmp_path / "timed" / "stop.json").read_text(encoding="utf-8"))
    assert stop["reason"] == "compute_cap:max_seconds" and stop["taken_steps"] < stop["max_steps"]
    assert report["verdict"] == "no-go"


def test_the_report_links_artifacts_and_stays_experimental(mechanics):
    _, report = mechanics
    assert report["verdict"] == "no-go" and report["experimental"]
    assert "mechanics evidence only" in report["why"]
    assert set(report["baselines"]) == {"fixed-supervised-head", "nli", "gliclass"}
    assert report["baselines"]["nli"].startswith("unavailable")
    assert report["baselines"]["fixed-supervised-head"].startswith("not run")
    assert not [key for key in report if key.endswith("accuracy") and not key.startswith("mechanics_")]


def test_the_empirical_mode_validates_resources_first_and_reports_blocked(tmp_path):
    report = pilot["run_empirical"](str(tmp_path / "empirical"), environ={})
    assert report["verdict"] == "blocked" and len(report["missing_resources"]) == 4
    assert report["artifacts"] == {"manifest": "manifest.json"} and "checkpoint" in report["unproduced"]
    manifest = json.loads((tmp_path / "empirical" / "manifest.json").read_text(encoding="utf-8"))
    assert report["manifest_digest"] == pilot["empirical_manifest"]().digest() and manifest["baselines"]["nli"]
    assert json.loads((tmp_path / "empirical" / "report.json").read_text(encoding="utf-8"))["verdict"] == "blocked"


def test_the_empirical_mode_with_every_resource_present_still_reports_blocked(tmp_path):
    environ = {variable: str(tmp_path) for variable in pilot["EMPIRICAL_RESOURCES"].values()}
    report = pilot["run_empirical"](str(tmp_path / "present"), environ=environ)
    assert report["verdict"] == "blocked" and report["missing_resources"] == []
    assert "not part of this repository" in report["why"]
    assert report["unrecorded"]  # licences and the revision are disclosed as unrecorded


def test_a_cap_equal_to_the_planned_run_is_not_a_cap_stop(tmp_path):
    report = pilot["run_mechanics"](str(tmp_path / "full"), max_steps=72)  # 12 epochs x 6 updates
    assert report["artifacts"]["stop"] is None and report["unproduced"] == []


def test_the_pilot_step_refuses_settings_it_does_not_implement():
    class Ctx:
        grad_clip_norm, accumulate_grad_batches, scaler = 1.0, 1, None

    with pytest.raises(ValueError, match="plain FP32"):
        pilot["pilot_train_step"](Ctx())
