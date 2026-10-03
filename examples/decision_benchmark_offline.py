"""A reproducible decision-provider benchmark, collected once and replayed
offline (FEAT-021).

``nnx.decisions.benchmark`` scores providers on identical samples. This
example:

  1. **Builds the samples.** A three-species question over feature rows,
     plus perturbations of it (reordered options, new descriptions, a
     distractor) and a held-out "farm" family whose label the provider's
     head was never trained on. Every perturbation keeps its original's
     grouping unit.
  2. **Collects live, once.** ``collect`` needs an explicit provider, a
     provider id and a ``Budget``; it attempts each batch once and records
     what it could not serve as ``unsupported`` with the provider's own
     reason. The records go to a JSONL file.
  3. **Replays offline.** With sockets disabled and no provider at all,
     ``read_records`` and ``evaluate`` join the records to the samples by
     sample id and question digest and report every slice — in-family,
     held-out, each family and each perturbation — with coverage counts,
     accuracy, macro-F1, exact NLL, Brier, ECE and reliability bins. A metric
     that cannot be computed says why, over how many rows.
  4. **Bootstraps an interval** over the grouping units with a recorded
     seed, and exports JSON, CSV and text that agree.

Fully offline, CPU only.

Run:
    python examples/decision_benchmark_offline.py

The bounded ``decision_benchmark_offline_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory.
"""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Iterator
from pathlib import Path

import torch

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams, TaskSpec
from nnx.decisions import Choice, FixedHeadProvider
from nnx.decisions.benchmark import (
    Budget,
    Sample,
    add_distractors,
    bootstrap_interval,
    collect,
    evaluate,
    permute_options,
    read_records,
    redescribe,
    write_records,
)

SPECIES = Choice("Which animal is it?", (("cat", "A cat"), ("dog", "A dog"), ("fox", "A fox")))
FARM = Choice("Which farm animal is it?", (("cat", "A cat"), ("dog", "A dog"), ("cow", "A cow")))


def _head() -> NNModel:
    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=3, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
            task=TaskSpec.categorical(3, labels=("cat", "dog", "fox")),
        ),
    )
    with torch.no_grad():
        model.net.layers[0].weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]))
        model.net.layers[0].bias.zero_()
    return model


@contextlib.contextmanager
def _offline() -> Iterator[None]:
    """Refuse any socket connection while replaying."""

    def refuse(*args, **kwargs):
        raise AssertionError("replay must not touch the network")

    connect, create = socket.socket.connect, socket.create_connection
    socket.socket.connect, socket.create_connection = refuse, refuse  # type: ignore[method-assign,assignment]
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = connect, create  # type: ignore[method-assign]


def decision_benchmark_offline_workflow(workdir: Path | None = None) -> None:
    workdir = Path(workdir) if workdir is not None else Path.cwd()

    # 1. Samples: originals, perturbations (same grouping unit) and a held-out family.
    rows = torch.tensor([[1.5, 0.0], [0.5, 2.0], [-1.0, 1.0], [1.0, 0.2], [0.2, 1.0], [-0.5, 1.5]])
    labels = ["cat", "dog", "fox", "cat", "dog", "fox"]
    originals = [
        Sample(f"pet-{i}", SPECIES, row, label, family="pets")
        for i, (row, label) in enumerate(zip(rows, labels, strict=True))
    ]
    samples = list(originals)
    for sample in originals[:3]:
        samples.append(permute_options(sample, ["fox", "cat", "dog"]))
        samples.append(redescribe(sample, {"cat": "A small feline", "fox": "A wild canine"}))
        samples.append(add_distractors(sample, [("wolf", "A wolf")]))  # outside the head's label space
    samples.append(Sample("farm-0", FARM, torch.tensor([1.0, 0.0]), "cow", family="farm", heldout=True))

    # 2. Live collection, once: explicit provider, provider id and budget.
    provider = FixedHeadProvider(_head())
    collection = collect(
        provider, samples, provider_id="fixed-head-v1", budget=Budget(max_calls=20), revision="weights@demo"
    )
    print(
        f"collected {len(collection.records)} records in {collection.calls} provider calls (complete={collection.complete})"
    )
    path = workdir / "records.jsonl"
    write_records(path, collection.records)
    calls_after_collection = provider.model_calls

    # 3. Replay: no provider, no network, no fitting — the interval and the exports included.
    with _offline():
        records = read_records(path)
        report = evaluate(samples, records, split="animals-demo-v1")
        interval = bootstrap_interval(samples, records, metric="nll", seed=0, resamples=500)
        (workdir / "report.json").write_text(report.to_json())
        (workdir / "report.csv").write_text(report.to_csv())
    assert provider.model_calls == calls_after_collection  # replay never called the provider
    print(report.text(), end="")
    pets, heldout = report.slices["in_family"], report.slices["heldout"]
    assert pets.metrics["accuracy"].value == 1.0 and pets.coverage.unsupported == 3  # the distractor variants
    assert heldout.coverage.unsupported == 1 and heldout.metrics["accuracy"].reason == "no eligible rows"
    assert report.slices["perturbation:permutation"].metrics["accuracy"].value == 1.0

    # 4. An interval over grouping units (the originals and their variants).
    print(
        f"nll {interval.estimate:.4f}, 95% [{interval.low:.4f}, {interval.high:.4f}] over {interval.units} "
        f"{interval.unit}s (seed {interval.seed})"
    )
    assert interval.units == 6 and not interval.degenerate


def main() -> None:
    decision_benchmark_offline_workflow()


if __name__ == "__main__":
    main()
