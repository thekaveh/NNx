"""Leakage-aware link prediction on a static graph (FEAT-027).

Node masks cannot hold out edges; a held-out edge left in the graph the
model passes messages over is a leaked answer. This script splits edges,
not nodes, and lets NNx check every batch against that split:

  1. ``split_links`` turns a synthetic two-community graph into a versioned
     manifest: canonical undirected edges (a reverse is the same edge),
     train / validation / test positives, fixed held-out negatives drawn
     from the graph's complement, and the seed — ``replay`` re-derives it.
  2. ``LinkTask`` builds candidate batches over the **training** topology
     only — validation and test positives never appear in a message graph.
  3. The registered recipe ``link_predictor_spec(...)`` (GCN encoder, dot
     decoder) trains with the task's objective, selects BEST by
     ``MonitorSpec("auroc")`` over exact, materialised validation AUROC,
     and reloads from the run.
  4. Test candidates are scored with stable candidate ids, and an
     **injected leak** — a test positive slipped into the message graph — is
     refused before the model sees it.

Fully offline, CPU only.

Requires the ``graph`` extra (PyTorch Geometric): ``pip install "thekaveh-nnx[graph]"``.

Run:
    python examples/link_prediction_offline.py

The bounded ``link_prediction_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory, and the
script itself runs end to end there as a subprocess.
"""

from __future__ import annotations

import torch

from nnx import Losses, MonitorSpec, NNCheckpoint, NNModel, NNModelParams, NNOptimParams, NNTrainParams
from nnx.link_tasks import LinkTask, LinkTaskError, link_metrics, link_predictor_spec, split_links
from nnx.nn.enum.checkpoints import Checkpoints

NODES = 80


def two_communities(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense inside two communities, sparse across; features hint the community."""
    g = torch.Generator().manual_seed(seed)
    community = torch.arange(NODES) % 2
    edges = [
        (u, v)
        for u in range(NODES)
        for v in range(u + 1, NODES)
        if torch.rand(1, generator=g).item() < (0.15 if community[u] == community[v] else 0.005)
    ]
    x = torch.nn.functional.one_hot(community, 2).float() + 0.3 * torch.randn(NODES, 2, generator=g)
    return torch.tensor(edges).t().contiguous(), x


def link_prediction_workflow(epochs: int = 6) -> dict:
    edge_index, x = two_communities()
    split = split_links(edge_index, NODES, val=0.1, test=0.1, seed=0, negatives=1)
    assert split.replay(edge_index) == split  # the manifest reproduces its membership
    task = LinkTask(split, train_negatives=1)

    model = NNModel(
        params=NNModelParams(net=link_predictor_spec(input_dim=2, hidden_dims=[16]), loss=Losses.MEAN_SQUARED_ERROR)
    )
    run = model.train(
        params=NNTrainParams(
            n_epochs=epochs,
            train_loader=task.loader("train", x, batch_size=64, seed=0),
            val_loader=task.loader("val", x, batch_size=64),
            optim=NNOptimParams.builder().adam(max_lr=1e-2).build(),
            metrics=task.metric_specs(),
            monitor=MonitorSpec("auroc"),
            seed=0,
        ),
        objective=task.objective(),
        eval_step_fn=task.eval_step(),
    )
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert checkpoint is not None
    reloaded = NNModel.from_checkpoint(checkpoint)  # the recipe rebuilds encoder and decoder

    prediction = task.predict(reloaded, task.loader("test", x, batch_size=32))
    metrics = link_metrics(prediction.logits, prediction.targets, from_logits=True)  # exact at any confidence
    ids = split.candidate_ids("test")
    assert prediction.ids.tolist() == [ids[tuple(pair)] for pair in prediction.pairs.tolist()]

    # An injected leak: a test positive in the message graph is refused.
    (batch,) = list(task.loader("test", x, batch_size=10_000))
    u, v = split.test[0]
    batch.edge_index = torch.cat([batch.edge_index, torch.tensor([[u, v], [v, u]])], dim=1)
    try:
        task.predict(reloaded, [batch])
        leak = None
    except LinkTaskError as error:
        leak = str(error)
    assert leak is not None and "leak" in leak
    return {
        "best_val_auroc": checkpoint.idp.val_edp.metrics.get("auroc"),
        "test_auroc": metrics["auroc"].value,
        "test_ap": metrics["ap"].value,
        "test_candidates": int(prediction.ids.shape[0]),
        "leak": leak,
        "edges": {name: len(split.positives(name)) for name in ("train", "val", "test")},
    }


def main() -> None:
    result = link_prediction_workflow(epochs=20)
    print(f"edges per split: {result['edges']}")
    print(f"best validation AUROC: {result['best_val_auroc']:.3f}")
    print(
        f"test AUROC {result['test_auroc']:.3f}, AP {result['test_ap']:.3f} over {result['test_candidates']} candidates"
    )
    print(f"injected leak refused: {result['leak']}")


if __name__ == "__main__":
    main()
