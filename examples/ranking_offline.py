"""Query-grouped ranking evaluation with stable candidate IDs (FEAT-035).

A retrieval system answers a query in two steps: an index proposes a
**candidate set** (here FAISS, by inner product over document vectors), and
a scorer re-ranks it. The index returns **positions**; a benchmark needs
**stable document IDs**, joined to graded relevance judgements. This script
keeps that mapping explicit and evaluates per complete query:

  1. **Train** a tiny pairwise scorer with ``RankingTask.objective()`` (the
     pairwise logistic loss over unequal-relevance pairs within a query)
     and validate it with ``RankingTask.eval_step()`` (MRR@k, Recall@k,
     NDCG@k per query, ties broken by candidate id, queries without a
     relevant candidate excluded and counted).
  2. **Select** the checkpoint by a named monitor:
     ``MonitorSpec("ndcg_at_5")`` over ``task.metric_specs()``.
  3. **Save and reload** the BEST checkpoint; the task configuration is
     checkpointed component state ``"ranking.task"``.
  4. **ID-aligned evaluation.** FAISS positions are mapped to stable IDs
     through the corpus's own id list, relevance is looked up by ID, and the
     reloaded scorer re-ranks each query's candidate set. Shuffling the
     rows changes nothing; the record says the candidate sets were
     **sampled** (a top-M retrieval), so the numbers measure re-ranking of
     those candidates, not full-corpus retrieval.

NNx does not manage the index: ``nnx.embeddings.export_to_faiss`` writes a
FAISS index file for a text corpus and returns its path, and the caller
keeps the parallel list of document IDs, exactly as done here. Without
``faiss`` installed the same top-M search runs in NumPy.

Fully offline, CPU only.

Run:
    python examples/ranking_offline.py

The bounded ``ranking_offline_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory, and the
script itself runs end to end there as a subprocess.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from nnx import (
    Activations,
    Devices,
    Losses,
    MonitorSpec,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
)
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.ranking import RankingTask

DIM = 8
N_DOCS = 120
N_QUERIES = 72
CANDIDATES = 12  # top-M per query from the index
K = 5


def _corpus(seed: int = 0):
    """Document vectors, stable IDs that are not their positions, query
    vectors and graded judgements ``{query id: {doc id: grade}}`` from a
    hidden bilinear utility (grade 2 for the best document, 1 for the next
    two)."""
    rng = np.random.default_rng(seed)
    docs = rng.standard_normal((N_DOCS, DIM)).astype("float32")
    doc_ids = [f"doc-{n:04d}" for n in rng.permutation(10_000)[:N_DOCS]]  # stable, unrelated to positions
    queries = rng.standard_normal((N_QUERIES, DIM)).astype("float32")
    mix = np.eye(DIM, dtype="float32") + 1.0 * rng.standard_normal((DIM, DIM)).astype("float32")
    utility = queries @ mix @ docs.T
    judgements: dict[int, dict[str, int]] = {}
    for q in range(N_QUERIES):
        order = np.argsort(-utility[q])
        judgements[q] = {doc_ids[order[0]]: 2, doc_ids[order[1]]: 1, doc_ids[order[2]]: 1}
    return docs, doc_ids, queries, judgements


def _top_m(docs: np.ndarray, queries: np.ndarray, m: int) -> np.ndarray:
    """Positions of each query's top-``m`` documents by inner product —
    FAISS ``IndexFlatIP`` when installed, else the same search in NumPy."""
    try:
        import faiss
    except ImportError:
        return np.argsort(-(queries @ docs.T), axis=1, kind="stable")[:, :m]
    index = faiss.IndexFlatIP(docs.shape[1])
    index.add(docs)
    _, positions = index.search(queries, m)
    return positions


def _features(query: np.ndarray, doc: np.ndarray) -> np.ndarray:
    return np.concatenate([query * doc, query, doc])  # what the scorer sees for one (query, document) pair


def _batches(docs, doc_ids, queries, judgements, query_rows, positions, *, queries_per_batch: int = 4):
    """Ranking batches ``(features, query_ids, candidate_ids, relevance)``
    over whole queries; candidate ids are the stable string IDs."""
    batches = []
    for start in range(0, len(query_rows), queries_per_batch):
        feats, qids, cids, grades = [], [], [], []
        for q in query_rows[start : start + queries_per_batch]:
            for position in positions[q]:
                doc_id = doc_ids[position]  # FAISS position -> stable ID
                feats.append(_features(queries[q], docs[position]))
                qids.append(f"q-{q:03d}")
                cids.append(doc_id)
                grades.append(judgements[q].get(doc_id, 0))
        batches.append((torch.tensor(np.stack(feats)), qids, cids, torch.tensor(grades)))
    return batches


def _scorer(seed: int = 0) -> NNModel:
    torch.manual_seed(seed)
    return NNModel(
        net_params=NNParams(
            input_dim=3 * DIM, output_dim=1, hidden_dims=[32], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR),
    )


class _Ctx:
    def __init__(self, model, loader):
        self.model, self.val_loader, self.extra_metrics, self.epoch_idx = model, loader, None, 0


def ranking_offline_workflow(epochs: int = 6) -> dict:
    """Train -> named selection -> save / reload -> ID-aligned evaluation;
    returns the held-out metrics before and after re-ranking."""
    docs, doc_ids, queries, judgements = _corpus()
    positions = _top_m(docs, queries, CANDIDATES)
    rows = list(range(N_QUERIES))
    train_rows, val_rows, test_rows = rows[:48], rows[48:56], rows[56:]
    # Training candidate sets: the retrieved ones plus every judged document,
    # so each training query has unequal-relevance pairs to learn from.
    train_positions = {
        q: list(dict.fromkeys([*[doc_ids.index(d) for d in judgements[q]], *positions[q]]))[:CANDIDATES]
        for q in train_rows
    }
    task = RankingTask(k=(1, K), max_relevance=2, candidate_sets="sampled", group_size=CANDIDATES)
    train = _batches(docs, doc_ids, queries, judgements, train_rows, train_positions)
    val = _batches(docs, doc_ids, queries, judgements, val_rows, positions)

    # 1-2. Train and select BEST by validation NDCG@5.
    run = _scorer().train(
        params=NNTrainParams(
            n_epochs=epochs,
            train_loader=train,
            val_loader=val,
            optim=NNOptimParams.builder().adam(max_lr=1e-2).build(),
            metrics=task.metric_specs(),
            monitor=MonitorSpec(f"ndcg_at_{K}"),
            seed=0,
        ),
        objective=task.objective(),
        eval_step_fn=task.eval_step(),
    )
    selected = [idp for idp in run.idps if idp.selection is not None]
    assert selected and selected[0].selection.monitor.key == f"val.ndcg_at_{K}"
    best_value = max(idp.val_edp.metrics[f"ndcg_at_{K}"] for idp in selected)

    # 3. Reload the BEST checkpoint: weights and the checkpointed task.
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert checkpoint is not None and checkpoint.idp.val_edp.metrics[f"ndcg_at_{K}"] == best_value
    state = NNCheckpoint.load_training_state(run=run.id, type=Checkpoints.BEST)["components"]["ranking.task"]
    assert RankingTask.from_state(state["state"]["task"]) == task
    scorer = NNModel.from_checkpoint(checkpoint)

    # 4. ID-aligned evaluation on held-out queries: the index's candidate
    #    sets, mapped to stable IDs; rows shuffled to show order independence.
    test = _batches(docs, doc_ids, queries, judgements, test_rows, positions, queries_per_batch=len(test_rows))
    features, qids, cids, grades = test[0]
    order = list(range(len(qids)))
    random.Random(0).shuffle(order)
    shuffled = [(features[order], [qids[i] for i in order], [cids[i] for i in order], grades[order])]
    reranked = task.eval_step()(_Ctx(scorer, test))
    assert task.eval_step()(_Ctx(scorer, shuffled)).metrics == reranked.metrics
    # The index's own order (inner product) as the baseline ranking, same IDs.
    index_scores = [float(queries[int(q[2:])] @ docs[doc_ids.index(d)]) for q, d in zip(qids, cids, strict=True)]
    baseline, _, excluded, _ = task.evaluate_rows(zip(qids, cids, index_scores, [int(g) for g in grades], strict=True))
    assert reranked.metrics["exhaustive_candidates"] == 0.0  # sampled candidates, not full-corpus retrieval
    assert reranked.metrics["excluded_queries"] == excluded
    return {
        "best_val_ndcg": best_value,
        "reranked": {name: reranked.metrics[name] for name in task.metric_names()},
        "index_order": baseline,
        "queries": reranked.metrics["queries"],
        "excluded_queries": reranked.metrics["excluded_queries"],
    }


def main() -> None:
    result = ranking_offline_workflow(epochs=12)
    print(f"best validation NDCG@{K}: {result['best_val_ndcg']:.4f}")
    print(
        f"held-out queries scored: {result['queries']:.0f} (excluded, no relevant candidate retrieved: "
        f"{result['excluded_queries']:.0f})"
    )
    for name, value in result["reranked"].items():
        print(f"  {name:>12}: re-ranked {value:.4f}   index order {result['index_order'][name]:.4f}")


if __name__ == "__main__":
    main()
