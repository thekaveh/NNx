"""Immutable fluent experiment plans (FEAT-012).

An ``ExperimentPlan`` holds one experiment's configuration: network, model
and training parameters, data, seed and callbacks. Every ``with_*`` call
returns a new plan, and the plan compiles to the ordinary ``NNModel(...)``
+ ``NNModel.train(...)`` loop. This script shows:

  1. **Sibling isolation.** Branching a base plan into a short and a long
     sibling leaves the base and each sibling unchanged.
  2. **Validate, then probe.** ``validate()`` reports every problem of a
     broken plan at once, with no loader read and no model built.
     ``probe(batch)`` runs one no-grad forward pass on a temporary model
     and restores the ambient RNG.
  3. **Imperative parity.** ``fit()`` gives the same weights and history
     as the equally seeded script ``set_seed(7); NNModel(...).train(...)``.
  4. **Distinct attempts.** Fitting the same plan twice gives two run ids
     and two run directories; nothing is overwritten.
  5. **Child resume.** ``resuming(parent_id)`` continues training in a new
     run that records the parent as its lineage, and the parent's
     artifacts are byte-for-byte intact.

Fully offline, CPU only; every run is written under a temporary directory.

Run:
    python examples/experiment_plan.py

The bounded ``experiment_plan_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory, and the
script itself runs end to end there as a subprocess.
"""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    ExperimentPlan,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
    set_seed,
)

DATA = torch.Generator().manual_seed(0)
X_TRAIN, Y_TRAIN = torch.randn(48, 6, generator=DATA), torch.arange(48) % 3
X_VAL, Y_VAL = torch.randn(24, 6, generator=DATA), torch.arange(24) % 3


def train_loader() -> DataLoader:  # a factory: a fresh loader for every fit
    return DataLoader(TensorDataset(X_TRAIN, Y_TRAIN), batch_size=16, shuffle=True)


def val_loader() -> DataLoader:
    return DataLoader(TensorDataset(X_VAL, Y_VAL), batch_size=12)


NET = NNParams(input_dim=6, output_dim=3, hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU)
MODEL = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
TRAIN = NNTrainParams(n_epochs=2, optim=NNOptimParams.builder().adam(max_lr=1e-2).build())


def _digest(directory: Path) -> str:
    sha = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            sha.update(str(path.relative_to(directory)).encode())
            sha.update(path.read_bytes())
    return sha.hexdigest()


def experiment_plan_workflow() -> dict:
    base = (
        ExperimentPlan()
        .with_net(NET)
        .with_model(MODEL)
        .with_train(TRAIN)
        .with_data(train_loader, val=val_loader, identity="toy-3-class")
        .with_seed(7)
    )

    # 1. Branch: siblings share the immutable parameters and leave the base alone.
    short, long = base.with_epochs(1), base.with_epochs(3)
    assert (base.train.n_epochs, short.train.n_epochs, long.train.n_epochs) == (2, 1, 3)

    # 2. Validate (pure, every problem at once), then probe one batch (restored).
    broken = ExperimentPlan().with_model(MODEL).with_seed(-1).with_data(iter([]))
    problems = broken.validate().paths
    assert set(problems) >= {"net", "train", "data.train", "seed"}
    ambient = random.getstate(), torch.get_rng_state()
    probe = base.probe((X_TRAIN[:4], Y_TRAIN[:4]))
    assert random.getstate() == ambient[0] and torch.equal(torch.get_rng_state(), ambient[1])
    assert probe.output_shape == (4, 3) and probe.loss is not None

    # 3. fit() compiles to the imperative loop: same weights, same history.
    fitted = base.fit()
    set_seed(7)
    loaders = train_loader(), val_loader()
    params = NNTrainParams(
        n_epochs=2,
        optim=TRAIN.optim,
        seed=7,
        data_id="toy-3-class",
        train_loader=loaders[0],
        val_loader=loaders[1],
    )
    twin_model = NNModel(net_params=NET, params=MODEL)
    twin_run = twin_model.train(params)
    same_weights = all(
        torch.equal(a, b)
        for a, b in zip(fitted.model.net.state_dict().values(), twin_model.net.state_dict().values(), strict=True)
    )
    same_history = [idp.state() for idp in fitted.run.idps] == [idp.state() for idp in twin_run.idps]
    assert same_weights and same_history

    # 4. Fitting again is a new attempt with its own run; nothing is overwritten.
    again = base.fit()
    assert again.run.id != fitted.run.id and again.attempt_id != fitted.attempt_id

    # 5. A child resume continues in a new run; the parent's artifacts are untouched.
    parent_dir = Path("runs") / fitted.run.id
    before = _digest(parent_dir)
    child = base.with_epochs(4).resuming(fitted.run.id).fit()
    assert child.run.state()["train"]["parent_run_id"] == fitted.run.id
    assert _digest(parent_dir) == before

    summary = {
        "branches": {"base": base.train.n_epochs, "short": short.train.n_epochs, "long": long.train.n_epochs},
        "broken_plan": list(problems),
        "probe": {"output_shape": probe.output_shape, "loss": round(probe.loss, 4)},
        "imperative_parity": {"weights": same_weights, "history": same_history},
        "attempts": [fitted.run.id, again.run.id],
        "child": {"run": child.run.id, "parent": fitted.run.id, "epochs": len({i.epoch_idx for i in child.run.idps})},
        "val": dict(fitted.metrics["val"].values),
    }
    print(f"branches (epochs): {summary['branches']}")
    print(f"a broken plan reports, all at once: {summary['broken_plan']}")
    print(f"probe: output {probe.output_shape}, loss {probe.loss:.4f}; ambient RNG restored")
    print(f"fit() == imperative run: weights {same_weights}, history {same_history}")
    print(f"two fits, two attempts: {fitted.run.id[:8]} / {again.run.id[:8]}")
    print(f"child {child.run.id[:8]} resumes parent {fitted.run.id[:8]}; parent artifacts intact")
    val = fitted.metrics["val"]
    print(f"final-epoch val metrics (available={val.available}): loss {val.values['loss']:.4f}")
    return summary


def main() -> None:
    home = os.getcwd()
    with tempfile.TemporaryDirectory() as root:
        os.chdir(root)  # runs/ lands in the temporary directory
        try:
            experiment_plan_workflow()
        finally:
            os.chdir(home)


if __name__ == "__main__":
    main()
