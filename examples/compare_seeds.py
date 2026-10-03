"""Multi-seed summaries and a paired comparison of two configurations
(FEAT-032).

One seed is one draw. ``nnx.comparison`` reports repeated runs as a
population: count, mean and the sample standard deviation per
configuration, and a paired ``B - A`` comparison over replicates that share
a declared meaning. This example:

  1. **Fixes the split once.** A stratified ``plan_split`` decides
     train / validation / test membership by sample id; every run uses it,
     and its digest is recorded in each run's provenance. The test split is
     never read: no test metric picks a configuration or a checkpoint.
  2. **Runs three seeds over two configurations** (a narrow and a wide
     hidden layer). Every run gets a fresh model, optimizer, callbacks and
     loader generator, and its own run root. One run (``wide``, seed 2) is
     interrupted after its first epoch: a failed attempt.
  3. **Reads the observations back** with ``observations_from_runs`` —
     ``run.yaml``, ``idps.csv`` and the provenance files only; no model or
     checkpoint is loaded and no run is written.
  4. **Reports** each configuration's validation loss at its last committed
     epoch (the failed attempt stays listed and counted, but has no value
     in the statistics), and the paired deltas ``wide - narrow`` per seed,
     never flipped for a metric to minimize. The bootstrap interval is
     labelled what it is: seed variability over these three replicates.
  5. **Saves and reloads** the report as strict JSON.

Fully offline, CPU only.

Run:
    python examples/compare_seeds.py

The bounded ``compare_seeds_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
    set_seed,
)
from nnx.comparison import Bootstrap, ComparisonReport, Metric, observations_from_runs
from nnx.data_splits import plan_split
from nnx.nn.callbacks import Callback
from nnx.provenance import ExperimentManifest, hash_bytes

SEEDS = (0, 1, 2)
CONFIGS = {"narrow": [4], "wide": [16]}  # hidden layer widths
LOSS = Metric("loss", "minimize", unit="nats")


class _Interrupted(Callback):
    """Fails the run after its first committed epoch (a failed attempt)."""

    def on_epoch_end(self, ctx):
        if ctx.epoch == 1:
            raise RuntimeError("simulated interruption")


@contextlib.contextmanager
def _working_dir(path: Path) -> Iterator[Path]:
    path.mkdir(parents=True, exist_ok=True)
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)


def _dataset() -> TensorDataset:
    generator = torch.Generator().manual_seed(1234)
    X = torch.randn(120, 4, generator=generator)
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).long()
    return TensorDataset(X, y)


def compare_seeds_workflow(base: Path | None = None) -> ComparisonReport:
    base = Path(base) if base is not None else Path.cwd() / "compare_seeds"
    data = _dataset()
    ids = [f"row-{i}" for i in range(len(data))]
    labels = [int(label) for label in data.tensors[1]]

    # 1. One fixed split, decided by sample id; its digest goes into every run's provenance.
    split = plan_split(ids, strategy="stratified", labels=labels, proportions=(0.6, 0.2, 0.2), seed=7)
    rows = split.resolve(ids)
    manifest = ExperimentManifest(data={"train": hash_bytes(data.tensors[0].numpy().tobytes())}, splits={"main": split})
    print(f"split {split.digest()[:20]}…: train={len(rows.train)} val={len(rows.validation)} test={len(rows.test)}")

    # 2. Three seeds over two configurations, each run fresh and in its own run root.
    observations = []
    for config, hidden in CONFIGS.items():
        for seed in SEEDS:
            root = base / config / f"seed-{seed}"
            interrupted = config == "wide" and seed == 2
            with _working_dir(root):
                set_seed(seed)  # the seed also drives the fresh model's initialisation
                model = NNModel(
                    net_params=NNParams(
                        input_dim=4, output_dim=2, hidden_dims=hidden, dropout_prob=0.0, activation=Activations.RELU
                    ),
                    params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
                )
                params = NNTrainParams(
                    n_epochs=3,
                    train_loader=DataLoader(
                        Subset(data, list(rows.train)),
                        batch_size=16,
                        shuffle=True,
                        generator=torch.Generator().manual_seed(seed),  # a fresh loader generator per run
                    ),
                    val_loader=DataLoader(Subset(data, list(rows.validation)), batch_size=64),
                    optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
                    seed=seed,
                )
                callbacks = [_Interrupted()] if interrupted else []  # fresh callbacks per run
                try:
                    model.train(params=params, provenance=manifest, callbacks=callbacks)
                except RuntimeError as error:
                    print(f"{config} seed {seed}: {error} (a failed attempt, kept in the report)")
            # 3. Read the observation back without loading a model or touching the run.
            (run_id,) = [name for name in os.listdir(root / "runs") if name != "best" and not name.startswith(".")]
            observations += observations_from_runs([run_id], metric=LOSS, root=str(root), config={run_id: config})

    # 4. Summaries per configuration and the paired comparison wide - narrow.
    narrow = [item for item in observations if item.config == "narrow"]
    wide = [item for item in observations if item.config == "wide"]
    report = ComparisonReport.build(
        observations,
        [(narrow, wide, "same seed: initialisation, loader order and the fixed split")],
        bootstrap=Bootstrap(seed=0, resamples=1000),
    )
    print(report.text(), end="")
    groups = {group.config: group for group in report.summary.groups}
    assert set(groups) == {"narrow", "wide"} and report.summary.differing == ("config",)
    assert groups["narrow"].n == 3 and groups["wide"].n == 2  # the failed attempt is not a value…
    assert groups["wide"].n_attempts == 3 and groups["wide"].n_failed == 1  # …but it is listed and counted
    (comparison,) = report.comparisons
    assert [pair.replicate for pair in comparison.pairs] == ["seed=0", "seed=1", "seed=2"]
    assert comparison.pairs[2].delta is None and comparison.n == 2  # seed 2 has no complete pair
    assert all(item.split == "validation" and item.selection == "last" for item in observations)  # never test

    # 5. Save and reload: the same report, re-derived and checked.
    path = base / "comparison.json"
    report.save(path)
    assert ComparisonReport.load(path).to_json() == report.to_json()
    print(f"saved {path.name}; reloaded identically")
    return report


def main() -> None:
    compare_seeds_workflow()


if __name__ == "__main__":
    main()
