"""Full-graph training with an empty validation split — ``NNGraphDataset``
optional loaders (FIX-019).

Demonstrates:

  1. A tiny in-memory PyG ``Data`` graph whose ``val_mask`` selects no
     node. ``NNGraphDataset(sampler="full")`` represents that split as an
     *absent* loader: ``val_loader is None``, resolved ``batch_sizes``
     ``(train, 0, test)`` and ``"0"`` in ``state()`` — the same optional
     contract the tabular / preference wrappers use. An empty
     ``train_mask`` is rejected at construction instead.
  2. Passing the absent loader straight into ``NNModel.train``: no
     validation pass runs (an ``eval_step_fn`` spy proves it), every saved
     ``NNIterationDataPoint`` carries ``val_edp=None`` — also after
     reloading the run from disk — LAST is committed, and ``EarlyStopping``
     works on an explicit train-loss monitor (its default validation
     monitor legitimately has no signal).
  3. The nonempty test split stays independently evaluable, scoring only
     its seed rows.

Fully offline: no Planetoid / Cora download and no neighbor sampling
(``sampler="full"`` needs neither pyg-lib nor torch-sparse). Requires
``torch_geometric`` (an NNx core dependency).

Run:
    python examples/graph_optional_splits.py

The bounded ``graph_optional_splits_workflow()`` helper is executed by
``tests/test_examples_smoke.py -k graph_optional_splits`` in a temporary
working directory.
"""

from __future__ import annotations

import math

import torch

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    EarlyStopping,
    Losses,
    Nets,
    NNCheckpoint,
    NNGraphDataset,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    set_seed,
)


class TinySplitGraph:
    """Six-node directed cycle: train nodes {0, 1, 2}, NO validation
    nodes, test nodes {3, 4, 5}. Satisfies the ``(root, transform)``
    constructor and ``dataset[0]`` / ``num_features`` / ``num_classes``
    surface ``NNGraphDataset`` drives — no disk access."""

    num_features = 4
    num_classes = 2

    def __init__(self, root, transform=None):
        from torch_geometric.data import Data

        n = 6
        g = torch.Generator().manual_seed(3)
        train_mask = torch.tensor([True, True, True, False, False, False])
        self._data = Data(
            x=torch.randn(n, self.num_features, generator=g),
            edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long),
            y=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
            train_mask=train_mask,
            val_mask=torch.zeros(n, dtype=torch.bool),  # empty on purpose
            test_mask=~train_mask,
        )

    def __getitem__(self, idx):
        return self._data


def graph_optional_splits_workflow() -> dict:
    """Bounded demonstration (writes ``runs/`` under the current working
    directory; the smoke test runs it in a temporary one): one two-epoch
    full-graph fit with no validation split, then reload and evaluate."""
    set_seed(3)
    dataset = NNGraphDataset(ds_class=TinySplitGraph, sampler="full")
    assert dataset.val_loader is None and dataset.test_loader is not None
    assert dataset.batch_sizes == (3, 0, 3) and dataset.state()["val_batch_size"] == "0"

    model = NNModel(
        net_params=_net_params_for(dataset),
        params=NNModelParams(net=Nets.GRAPH_CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    eval_calls: list[int] = []

    def eval_spy(ctx):
        eval_calls.append(ctx.epoch_idx)
        raise AssertionError("no validation split → the validation step must never run")

    run = model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=dataset.train_loader,
            val_loader=dataset.val_loader,  # None: validation is skipped, not faked
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        ),
        eval_step_fn=eval_spy,
        callbacks=[EarlyStopping(monitor="train_edp.loss", patience=1)],
    )
    assert eval_calls == [], eval_calls
    assert all(idp.val_edp is None for idp in run.idps)
    assert NNCheckpoint.load(run=run.id, type=Checkpoints.LAST) is not None
    reloaded = NNRun.load(run.id)
    assert all(idp.val_edp is None for idp in reloaded.idps)

    test_edp = model.evaluate(loader=dataset.test_loader)
    assert test_edp.loss is not None and math.isfinite(test_edp.loss)
    summary = dict(
        batch_sizes=dataset.batch_sizes,
        val_batch_size_state=dataset.state()["val_batch_size"],
        epochs=len(run.idps),
        val_edp_persisted=[idp.val_edp for idp in reloaded.idps],
        test_accuracy=test_edp.accuracy,
        test_seed_rows=model.net.seed_count(dataset.test_loader[0]),
    )
    print(f"graph optional splits: {summary}")
    return summary


def _net_params_for(dataset: NNGraphDataset) -> NNParams:
    """Net params sized from the dataset (kept separate so the workflow
    reads top-down)."""
    return NNParams(
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        hidden_dims=[8],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )


def main() -> None:
    summary = graph_optional_splits_workflow()
    print("=" * 60)
    print("Empty validation mask → val_loader=None; validation skipped, not faked")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key:>22}: {value}")


if __name__ == "__main__":
    main()
