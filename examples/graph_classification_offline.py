"""Graph-level classification with explicit pooling and graph ids (FEAT-026).

NNx's built-in graph nets classify the *nodes* of one graph. This script
classifies *whole graphs* — one prediction row per graph — through the
ordinary ``train`` / ``evaluate`` / ``predict_proba`` / reload paths:

  1. Build a ``GraphCollection``: synthetic graphs of 2–9 nodes, each with a
     stable integer **graph id** and a class (one graph is deliberately
     unlabeled and flagged as such — an unflagged missing target would be
     refused).
  2. **Split by graph id** — the collection never splits itself — into
     train / validation / test collections.
  3. Train the registered recipe ``graph_classifier_spec(...)`` (GCN encoder
     → mean pooling → linear head) with a categorical task whose
     ``ignore_index`` masks the unlabeled graph, so loss and metrics are
     averaged over **labeled graphs**.
  4. Reload the BEST checkpoint — the encoder, pool and head are rebuilt
     from the run's recipe — and predict the test graphs: one row per
     graph, keyed by its graph id, even from a shuffled loader.

Fully offline, CPU only.

Requires the ``graph`` extra (PyTorch Geometric): ``pip install "thekaveh-nnx[graph]"``.

Run:
    python examples/graph_classification_offline.py

The bounded ``graph_classification_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory, and the
script itself runs end to end there as a subprocess.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from nnx import Losses, NNCheckpoint, NNModel, NNModelParams, NNOptimParams, NNTrainParams, TaskSpec
from nnx.graph_tasks import IGNORE, GraphCollection, graph_classifier_spec
from nnx.nn.enum.checkpoints import Checkpoints

CLASSES = 3
FEATURES = 4


def _chain(n: int) -> torch.Tensor:
    src = torch.arange(n - 1)
    return torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])])


def synthetic_collection(n_graphs: int = 60, seed: int = 0) -> GraphCollection:
    """Path graphs whose class shows in a noisy dominant feature column;
    the last graph is unlabeled (flagged)."""
    rng = np.random.default_rng(seed)
    graphs, targets = [], []
    for i in range(n_graphs):
        label = i % CLASSES
        n = int(rng.integers(2, 10))
        x = rng.normal(scale=0.6, size=(n, FEATURES))
        x[:, label] += 1.0
        graphs.append(Data(x=torch.tensor(x, dtype=torch.float32), edge_index=_chain(n)))
        targets.append(None if i == n_graphs - 1 else label)
    ids = [5000 + 7 * i for i in range(n_graphs)]  # stable ids, not positions
    return GraphCollection(graphs, ids, targets=targets, unlabeled=[ids[-1]], num_classes=CLASSES)


def graph_classification_workflow(epochs: int = 4) -> dict:
    collection = synthetic_collection()
    ids = list(collection.ids)
    train, val, test = collection.subset(ids[:36]), collection.subset(ids[36:48]), collection.subset(ids[48:])

    spec = graph_classifier_spec(input_dim=FEATURES, num_classes=CLASSES, hidden_dims=[32], pool="mean", seed=0)
    model = NNModel(
        params=NNModelParams(
            net=spec, loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(CLASSES, ignore_index=IGNORE)
        )
    )
    run = model.train(
        params=NNTrainParams(
            n_epochs=epochs,
            train_loader=train.loader(batch_size=8, shuffle=True, seed=0),
            val_loader=val.loader(batch_size=8),
            optim=NNOptimParams.builder().adam(max_lr=2e-2).build(),
            seed=0,
        )
    )
    records = [idp.val_edp for idp in run.idps if idp.val_edp is not None]
    assert all(record.count == val.labeled == 12 for record in records)  # labeled graphs, not batches or nodes

    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert checkpoint is not None
    reloaded = NNModel.from_checkpoint(checkpoint)  # the recipe rebuilds encoder, pool and head
    ordered = reloaded.predict_proba(test.loader(batch_size=5))
    shuffled = reloaded.predict_proba(test.loader(batch_size=5, shuffle=True, seed=1))
    assert ordered.sample_ids.tolist() == list(test.ids)
    by_id = dict(zip(ordered.sample_ids.tolist(), ordered.decoded.tolist(), strict=True))
    assert all(
        by_id[i] == label for i, label in zip(shuffled.sample_ids.tolist(), shuffled.decoded.tolist(), strict=True)
    )
    labels = dict(zip(test.ids, test.labels, strict=True))
    scored = [i for i in test.ids if labels[i] != IGNORE]
    accuracy = sum(by_id[i] == labels[i] for i in scored) / len(scored)
    evaluated = reloaded.evaluate(test.loader(batch_size=5))
    assert evaluated.count == len(scored) and abs(evaluated.accuracy - accuracy) < 1e-9
    return {
        "best_val_error": checkpoint.idp.val_edp.error,
        "test_accuracy": accuracy,
        "test_graphs": len(test.ids),
        "labeled_test_graphs": len(scored),
        "predictions": by_id,
    }


def main() -> None:
    result = graph_classification_workflow(epochs=12)
    print(f"best validation error: {result['best_val_error']:.3f}")
    print(
        f"test accuracy over {result['labeled_test_graphs']} labeled graphs "
        f"(of {result['test_graphs']}): {result['test_accuracy']:.3f}"
    )
    for graph_id, label in list(result["predictions"].items())[:5]:
        print(f"  graph {graph_id}: class {label}")


if __name__ == "__main__":
    main()
