"""Full-batch (non-sampling) node-classification loader for NNGraphDataset.

Covers four cases from the A1 spec:

(a) Structural correctness of the one-element full-batch train loader:
    - leading ``batch_size`` rows correspond to the split's nodes (via
      ``input_id``), ``edge_index`` indices are all valid (< num_nodes),
      ``y[:batch_size]`` matches the split's labels.
(b) ``seed_count(batch)`` from ``GraphNNBase`` returns ``n_split`` for a
    full-batch batch, so ``_fwd_pass``'s ``[:n_seed]`` slice selects
    exactly the right rows.
(c) SLOW parity — full-batch ``Nets.GRAPH_CONV`` trained on real Cora
    (``Planetoid``) for 5 CPU epochs reaches val-acc ≥ 0.55.  Downloads
    Cora once (~3 MB) to ``~/.cache/nnx-test-data``.  No pyg-lib /
    torch-sparse required: full-batch mode only touches plain PyG ``Data``
    operations.
(d) Default ``sampler="neighbor"`` still constructs without regression.

(a) and (b) use a tiny in-memory synthetic graph — no NeighborLoader /
pyg-lib involved.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from nnx.nn.dataset.nn_graph_dataset import NNGraphDataset  # noqa: E402

# ---------------------------------------------------------------------------
# Shared synthetic-graph stub
# ---------------------------------------------------------------------------


class _TinyFullBatch:
    """5-node cycle graph with train_mask on nodes {0, 2}.

    Minimal surface that satisfies ``NNGraphDataset``'s (root, transform)
    constructor and the ``dataset[0]`` / ``num_features`` / ``num_classes``
    protocol — no disk access, no pyg-lib.
    """

    num_features = 3
    num_classes = 2

    def __init__(self, root, transform=None):
        n = 5
        g = torch.Generator().manual_seed(7)
        x = torch.randn(n, self.num_features, generator=g)
        # 0→1→2→3→4→0  (directed cycle)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        train_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[[0, 2]] = True  # 2 train nodes
        val_mask = torch.zeros(n, dtype=torch.bool)
        val_mask[[1]] = True  # 1 val node
        test_mask = torch.zeros(n, dtype=torch.bool)
        test_mask[[3, 4]] = True  # 2 test nodes
        self._data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

    def __getitem__(self, idx):
        return self._data


# ---------------------------------------------------------------------------
# (a) Structural correctness of the full-batch train batch
# ---------------------------------------------------------------------------


def test_full_batch_train_batch_structure():
    """Leading batch_size rows are the train-split nodes; edge_index valid."""
    ds = NNGraphDataset(ds_class=_TinyFullBatch, sampler="full")

    assert len(ds.train_loader) == 1
    batch = ds.train_loader[0]

    n_nodes = 5
    n_train = 2  # nodes {0, 2}

    # input_id carries the original global indices of the train nodes
    assert batch.input_id is not None
    assert set(batch.input_id.tolist()) == {0, 2}

    # batch_size == number of train nodes
    assert batch.batch_size == n_train

    # edge_index must stay within [0, num_nodes)
    assert int(batch.edge_index.max()) < n_nodes, "edge_index out of range (too large)"
    assert int(batch.edge_index.min()) >= 0, "edge_index out of range (negative)"

    # The permutation puts split nodes first (in nonzero order), so
    # batch.y[:batch_size] must equal the original labels at those indices.
    original_y = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    expected_leading_y = original_y[batch.input_id]
    assert torch.equal(batch.y[:n_train], expected_leading_y), (
        f"y[:batch_size] = {batch.y[:n_train].tolist()}, "
        f"expected {expected_leading_y.tolist()} (labels of nodes {batch.input_id.tolist()})"
    )


def test_full_batch_val_batch_structure():
    """Val split batch: single val node leads; all splits checked."""
    ds = NNGraphDataset(ds_class=_TinyFullBatch, sampler="full")

    val_batch = ds.val_loader[0]
    assert val_batch.batch_size == 1
    assert set(val_batch.input_id.tolist()) == {1}

    test_batch = ds.test_loader[0]
    assert test_batch.batch_size == 2
    assert set(test_batch.input_id.tolist()) == {3, 4}


# ---------------------------------------------------------------------------
# (b) seed_count returns n_split
# ---------------------------------------------------------------------------


def test_full_batch_seed_count():
    """GraphNNBase.seed_count(batch) returns the number of split nodes."""
    from nnx.nn.enum.activations import Activations
    from nnx.nn.net.graph_conv_nn import GraphConvNN
    from nnx.nn.params.nn_params import NNParams

    ds = NNGraphDataset(ds_class=_TinyFullBatch, sampler="full")

    net = GraphConvNN(
        params=NNParams(
            dropout_prob=0.0,
            activation=Activations.RELU,
            input_dim=ds.input_dim,
            output_dim=ds.output_dim,
            hidden_dims=[8],
        )
    )

    assert net.seed_count(ds.train_loader[0]) == 2  # nodes {0, 2}
    assert net.seed_count(ds.val_loader[0]) == 1  # node {1}
    assert net.seed_count(ds.test_loader[0]) == 2  # nodes {3, 4}


# ---------------------------------------------------------------------------
# (c) SLOW parity — full-batch GCN on real Cora reaches val-acc ≥ 0.55
# ---------------------------------------------------------------------------


def test_full_batch_cora_parity(tmp_path, monkeypatch):
    """Full-batch GCN on real Cora trains 5 CPU epochs and reaches val-acc ≥ 0.55.

    Downloads Cora once to ~/.cache/nnx-test-data (~3 MB).  Does NOT
    require pyg-lib or torch-sparse — full-batch mode iterates a plain
    Python list, so no sampling backend is invoked.

    monkeypatch.chdir keeps run artefacts out of the repo root.
    """
    from torch_geometric.datasets import Planetoid

    from nnx.nn.enum.activations import Activations
    from nnx.nn.enum.devices import Devices
    from nnx.nn.enum.losses import Losses
    from nnx.nn.enum.nets import Nets
    from nnx.nn.nn_model import NNModel
    from nnx.nn.params.nn_model_params import NNModelParams
    from nnx.nn.params.nn_params import NNParams
    from nnx.nn.params.nn_train_params import NNTrainParams

    # Named subclass so ds_class.__name__ == "CoraFullBatch" (meaningful name).
    class CoraFullBatch(Planetoid):
        def __init__(self, root, transform=None):
            super().__init__(root=root, name="Cora", transform=transform)

    cora_root = os.path.expanduser("~/.cache/nnx-test-data")

    ds = NNGraphDataset(
        ds_class=CoraFullBatch,
        sampler="full",
        root_dir=cora_root,
    )

    assert ds.name == "CoraFullBatch"
    assert ds.input_dim == 1433  # Cora feature count
    assert ds.output_dim == 7  # Cora class count

    net_params = NNParams(
        dropout_prob=0.5,
        activation=Activations.RELU,
        input_dim=ds.input_dim,
        output_dim=ds.output_dim,
        hidden_dims=[64],
    )

    model = NNModel(
        net_params=net_params,
        params=NNModelParams(
            net=Nets.GRAPH_CONV,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )

    # chdir so NNModel.train() saves run artefacts to tmp_path, not the repo.
    monkeypatch.chdir(tmp_path)

    model.train(
        params=NNTrainParams(
            n_epochs=5,
            train_loader=ds.train_loader,  # type: ignore[arg-type]
            val_loader=ds.val_loader,  # type: ignore[arg-type]
        )
    )

    result = model.evaluate(loader=ds.val_loader)  # type: ignore[arg-type]
    val_acc = result.accuracy

    assert val_acc >= 0.55, f"Full-batch GCN on Cora (5 epochs) reached val-acc={val_acc:.3f}, expected ≥ 0.55."


# ---------------------------------------------------------------------------
# (d) Default sampler="neighbor" still constructs (no regression)
# ---------------------------------------------------------------------------


class _TinyNeighbor:
    """8-node in-memory graph for the neighbor-sampler smoke test."""

    num_features = 5
    num_classes = 3

    def __init__(self, root, transform=None):
        n = 8
        g = torch.Generator().manual_seed(0)
        x = torch.randn(n, self.num_features, generator=g)
        edge_index = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]], dtype=torch.long)
        train_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[:4] = True
        val_mask = torch.zeros(n, dtype=torch.bool)
        val_mask[4:6] = True
        test_mask = torch.zeros(n, dtype=torch.bool)
        test_mask[6:] = True
        self._data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.arange(n) % self.num_classes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

    def __getitem__(self, idx):
        return self._data


def test_neighbor_sampler_default_unregressed():
    """sampler='neighbor' (default) still constructs NeighborLoader correctly."""
    ds = NNGraphDataset(ds_class=_TinyNeighbor, n_neighbors=[2])

    assert ds.sampler == "neighbor"
    assert ds.name == "_TinyNeighbor"
    assert ds.input_dim == 5
    assert ds.output_dim == 3
    assert ds.batch_sizes == (4, 2, 2)


def test_neighbor_sampler_explicit_still_works():
    """Explicitly passing sampler='neighbor' is identical to the default."""
    ds = NNGraphDataset(ds_class=_TinyNeighbor, n_neighbors=[2], sampler="neighbor")
    assert ds.sampler == "neighbor"
    assert ds.batch_sizes == (4, 2, 2)


def test_full_batch_missing_n_neighbors_is_fine():
    """sampler='full' does not require n_neighbors — omitting it is valid."""
    ds = NNGraphDataset(ds_class=_TinyFullBatch, sampler="full")
    assert ds.n_neighbors is None
    # Loaders are plain lists
    assert isinstance(ds.train_loader, list)
    assert isinstance(ds.val_loader, list)
    assert isinstance(ds.test_loader, list)


def test_neighbor_sampler_requires_n_neighbors():
    """sampler='neighbor' without n_neighbors raises a clear ValueError."""
    with pytest.raises(ValueError, match="n_neighbors is required"):
        NNGraphDataset(ds_class=_TinyNeighbor, sampler="neighbor")


def test_full_batch_batch_sizes_resolved():
    """batch_sizes are resolved to per-split mask counts for full mode too."""
    ds = NNGraphDataset(ds_class=_TinyFullBatch, sampler="full")
    # _TinyFullBatch: 2 train, 1 val, 2 test
    assert ds.batch_sizes == (2, 1, 2)


def test_full_batch_state_includes_sampler():
    """state() carries 'sampler' and omits 'seed' (mirrors the neighbor contract)."""
    ds = NNGraphDataset(ds_class=_TinyFullBatch, sampler="full")
    s = ds.state()
    assert s["sampler"] == "full"
    assert "seed" not in s
    assert s["name"] == "_TinyFullBatch"
    assert s["input_dim"] == 3
    assert s["output_dim"] == 2


def test_full_graph_loader_warm_resume(tmp_path, monkeypatch):
    """FIX-012: the full-batch graph loader is a one-element list; a
    save/resume cycle must complete without neighbor sampling
    extensions, resume at source epoch + 1, and keep scoring only the
    leading seed rows."""
    from nnx import Checkpoints, NNCheckpoint, NNModel
    from nnx.nn.enum.activations import Activations
    from nnx.nn.enum.devices import Devices
    from nnx.nn.enum.losses import Losses
    from nnx.nn.enum.nets import Nets
    from nnx.nn.params.nn_model_params import NNModelParams
    from nnx.nn.params.nn_params import NNParams
    from nnx.nn.params.nn_train_params import NNTrainParams

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    ds = NNGraphDataset(ds_class=_TinyFullBatch, sampler="full")

    def _model() -> NNModel:
        return NNModel(
            net_params=NNParams(
                dropout_prob=0.0,
                activation=Activations.RELU,
                input_dim=ds.input_dim,
                output_dim=ds.output_dim,
                hidden_dims=[8],
            ),
            params=NNModelParams(net=Nets.GRAPH_CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        )

    torch.manual_seed(0)
    first = _model().train(params=NNTrainParams(n_epochs=1, train_loader=ds.train_loader, val_loader=ds.val_loader))
    resumed_model = _model()
    resumed = resumed_model.train(
        params=NNTrainParams(
            n_epochs=1, train_loader=ds.train_loader, val_loader=ds.val_loader, resume_from_run_id=first.id
        )
    )
    assert resumed.idps[0].epoch_idx == 1
    state = NNCheckpoint.load_training_state(resumed.id, Checkpoints.LAST)
    assert state is not None and state["completed_epoch"] == 1

    # Seed-only scoring: the val split has exactly one node, so the
    # evaluated accuracy is 0.0 or 1.0 — never a fraction over context rows.
    edp = resumed_model.evaluate(loader=ds.val_loader)
    assert edp.accuracy in (0.0, 1.0)
    assert resumed.idps[-1].val_edp is not None and resumed.idps[-1].val_edp.accuracy in (0.0, 1.0)


# ---------------------------------------------------------------------------
# FIX-019: training without a validation split
# ---------------------------------------------------------------------------


class _NoValFullBatch(_TinyFullBatch):
    """Same 5-node cycle, but the validation mask is empty (val node 1
    joins the test split: train {0, 2}, val {}, test {1, 3, 4})."""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self._data.val_mask = torch.zeros(5, dtype=torch.bool)
        self._data.test_mask = torch.tensor([False, True, False, True, True])


def test_full_graph_train_without_validation(tmp_path, monkeypatch):
    """With an empty val mask the dataset publishes ``val_loader=None``;
    passing it through NNModel.train and Trainer.train invokes no
    evaluation hook, records and persists ``val_edp=None`` (reloaded from
    disk), commits LAST, keeps EarlyStopping usable on a train monitor, and
    leaves the nonempty test split independently evaluable on its seed
    rows only."""
    from nnx import Checkpoints, EarlyStopping, NNCheckpoint, NNModel, NNOptimParams, NNRun, Optims
    from nnx.nn.enum.activations import Activations
    from nnx.nn.enum.devices import Devices
    from nnx.nn.enum.losses import Losses
    from nnx.nn.enum.nets import Nets
    from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
    from nnx.nn.params.nn_model_params import NNModelParams
    from nnx.nn.params.nn_params import NNParams
    from nnx.nn.params.nn_train_params import NNTrainParams
    from nnx.trainer import NNTrainerParams, Trainer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    ds = NNGraphDataset(ds_class=_NoValFullBatch, sampler="full")
    assert ds.val_loader is None and ds.batch_sizes == (2, 0, 3) and ds.state()["val_batch_size"] == "0"
    assert ds.train_loader[0].batch_size == 2 and ds.test_loader[0].batch_size == 3

    model = NNModel(
        net_params=NNParams(dropout_prob=0.0, activation=Activations.RELU, input_dim=3, output_dim=2, hidden_dims=[8]),
        params=NNModelParams(net=Nets.GRAPH_CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    eval_calls: list[int] = []

    def eval_spy(ctx):
        eval_calls.append(ctx.epoch_idx)
        raise AssertionError("validation must not run without a validation split")

    torch.manual_seed(0)
    run = model.train(
        params=NNTrainParams(n_epochs=2, train_loader=ds.train_loader, val_loader=ds.val_loader),
        eval_step_fn=eval_spy,
        callbacks=[EarlyStopping(monitor="train_edp.loss", patience=1)],
    )
    assert eval_calls == []
    assert len(run.idps) == 2 and all(idp.val_edp is None for idp in run.idps)
    assert all(idp.train_edp.loss is not None and torch.isfinite(torch.tensor(idp.train_edp.loss)) for idp in run.idps)
    assert NNCheckpoint.load(run=run.id, type=Checkpoints.LAST) is not None
    reloaded = NNRun.load(run.id)
    assert len(reloaded.idps) == 2 and all(idp.val_edp is None for idp in reloaded.idps)

    # The nonempty test split is still evaluable, on its 3 seed rows only.
    assert model.net.seed_count(ds.test_loader[0]) == 3
    edp = model.evaluate(loader=ds.test_loader)
    assert edp.loss is not None and torch.isfinite(torch.tensor(edp.loss))
    assert edp.accuracy in (0.0, 1 / 3, 2 / 3, 1.0)

    # Trainer route: the absent loader disables validation there too.
    evaluate_calls: list[int] = []
    original_evaluate = model.evaluate

    def evaluate_spy(*args, **kwargs):
        evaluate_calls.append(1)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(model, "evaluate", evaluate_spy)

    def _graph_step(ctx) -> NNEvaluationDataPoint:
        m = ctx.model
        opt = ctx.optimizers["main"]
        m.net.train()
        opt.zero_grad()
        _, Y, logits, Y_hat = m._fwd_pass(ctx.batch)
        loss = m.loss_fn(logits, Y)
        loss.backward()
        opt.step()
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=0.0,
            precision=0.0,
            loss=float(loss.detach()),
            error=float((Y_hat != Y).float().mean()),
        )

    trainer_run = Trainer(model).train(
        params=NNTrainerParams(
            n_epochs=1,
            train_loader=ds.train_loader,
            val_loader=ds.val_loader,
            optims={"main": NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0)},
            data_id="trainer",
        ),
        trainer_step_fn=_graph_step,
    )
    assert evaluate_calls == [] and all(idp.val_edp is None for idp in trainer_run.idps)
