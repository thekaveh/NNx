"""Graph-level classification with explicit pooling and graph identity (FEAT-026).

NNx's graph nets (``Nets.GRAPH_CONV`` / ``GRAPH_SAGE`` / ``GRAPH_ATT``) and
``NNGraphDataset`` classify the **nodes** of one graph. This module
classifies **whole graphs** — one prediction row per graph, keyed by a
stable graph id — through NNx's ordinary training, evaluation, prediction
and reload paths:

- :class:`GraphCollection` — a collection of PyG ``Data`` graphs, each with
  a stable non-negative integer **graph id** and a class label (or
  explicitly flagged unlabeled). It checks every graph once: unique ids, at
  least one node, a feature matrix of the shared width, edges inside the
  graph, a label for every unflagged graph. ``loader()`` batches graphs in
  the order given (or shuffled, with a seed); a batch carries ``graph_id``
  and one target per graph (``IGNORE`` for an unlabeled graph). It never
  splits itself: split by graph id (correlated graphs leak across splits
  whatever their ids).
- :class:`GraphPool` — ``"mean"`` or ``"sum"`` pooling of node rows into
  graph rows, graph-local and invariant to node order.
- :class:`GraphClassifier` — an encoder of node rows (``(x, edge_index) ->
  node embeddings``), a pool and an optional head; its forward returns
  ``(num_graphs, classes)``. ``unpack_batch`` validates each batch before
  any forward pass — a batch without graph ids, duplicate ids, a zero-node
  graph, an edge across graphs, an inconsistent ``ptr`` / ``batch`` vector
  or a target per graph missing — and ``sample_ids`` gives predictions the
  batch's graph ids, so prediction rows keep their identity through
  shuffling, device moves and concatenation.
- :func:`graph_classifier_spec` — a registered recipe
  (``ModelSpec("nnx.graph_classifier", 1, config)``): a GCN / GraphSAGE /
  GAT encoder, the pool and a linear head, rebuilt from the run on reload.

Train with a categorical task so the loss and metrics are averaged over the
**labeled graphs**: ``NNModelParams(net=graph_classifier_spec(...),
loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(n, ignore_index=IGNORE))``.
Records then count labeled graphs, BEST selection compares those
per-graph means, and ``predict_proba`` returns one row per graph with its
graph id as the sample id. Node-level paths refuse collection batches
rather than scoring them as nodes, and ONNX export of a graph classifier is
refused before anything is written (its pooling is not exported).
"""

from __future__ import annotations

import numbers
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
from torch import nn

__all__ = [
    "FACTORY_ID",
    "IGNORE",
    "POOLS",
    "GraphClassifier",
    "GraphCollection",
    "GraphPool",
    "GraphTaskError",
    "check_graph_batch",
    "graph_classifier_spec",
    "is_graph_collection_batch",
]

IGNORE = -100  # the target of an unlabeled graph (TaskSpec.categorical(..., ignore_index=IGNORE))
POOLS = ("mean", "sum")
FACTORY_ID = "nnx.graph_classifier"
ENCODERS = ("graph_conv", "graph_sage", "graph_att")
ONNX_UNSUPPORTED = (
    "a graph classifier's pooling runs over PyG batch vectors, which this exporter does not support; "
    "export its weights with export_state_dict() instead"
)


class GraphTaskError(ValueError):
    """A graph collection, batch or classifier setting NNx rejects."""


def _count(value: Any, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise GraphTaskError(f"{what} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def is_graph_collection_batch(batch: Any) -> bool:
    """Whether ``batch`` is a batch of whole graphs (it carries graph ids)."""
    return getattr(batch, "graph_id", None) is not None


# --- the collection -----------------------------------------------------------------------------


def _label(value: Any, graph_id: int) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.is_floating_point() or value.dtype == torch.bool:
            raise GraphTaskError(f"graph {graph_id}: a target is one integer class, got {value!r}")
        value = int(value.reshape(-1)[0])
    return _count(value, f"graph {graph_id}'s target", minimum=0)


class GraphCollection(torch.utils.data.Dataset):
    """Whole graphs with stable ids and graph-level targets.

    Args:
        graphs: PyG ``Data`` objects with ``x`` (``(nodes, features)``,
            floating) and ``edge_index`` (``(2, edges)``, node indices of the
            same graph); each graph's target is its ``y`` (one integer
            class) unless ``targets`` gives them.
        ids: one stable non-negative integer id per graph (unique).
        targets: optional class per graph, overriding ``y``.
        unlabeled: ids of graphs deliberately without a target (their
            target becomes :data:`IGNORE`); any other graph without one is
            refused.
        num_classes: when given, every target must be below it.
    """

    def __init__(
        self,
        graphs: Sequence[Any],
        ids: Sequence[int],
        *,
        targets: Optional[Sequence[Optional[int]]] = None,
        unlabeled: Sequence[int] = (),
        num_classes: Optional[int] = None,
    ) -> None:
        from torch_geometric.data import Data

        graphs = list(graphs)
        ids = [_count(i, "a graph id", minimum=0) for i in ids]
        if len(ids) != len(graphs):
            raise GraphTaskError(f"{len(graphs)} graphs but {len(ids)} ids")
        if not graphs:
            raise GraphTaskError("a graph collection needs at least one graph")
        repeated = sorted(i for i, n in Counter(ids).items() if n > 1)
        if repeated:
            raise GraphTaskError(f"duplicate graph ids {repeated}")
        flagged = {_count(i, "an unlabeled id", minimum=0) for i in unlabeled}
        unknown = sorted(flagged - set(ids))
        if unknown:
            raise GraphTaskError(f"unlabeled names ids not in the collection: {unknown}")
        if targets is not None and len(targets) != len(graphs):
            raise GraphTaskError(f"{len(graphs)} graphs but {len(targets)} targets")
        if num_classes is not None:
            num_classes = _count(num_classes, "num_classes", minimum=2)
        width: Optional[int] = None
        items: list[Any] = []
        labels: list[int] = []
        for position, (graph, graph_id) in enumerate(zip(graphs, ids, strict=True)):
            x = getattr(graph, "x", None)
            edge_index = getattr(graph, "edge_index", None)
            if not isinstance(x, torch.Tensor) or x.ndim != 2 or not x.is_floating_point():
                raise GraphTaskError(f"graph {graph_id}: x must be a floating (nodes, features) tensor")
            if x.shape[0] == 0:
                raise GraphTaskError(f"graph {graph_id} has no nodes")
            if width is None:
                width = int(x.shape[1])
            elif x.shape[1] != width:
                raise GraphTaskError(f"graph {graph_id} has {x.shape[1]} features; the collection has {width}")
            if edge_index is None:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            if (
                not isinstance(edge_index, torch.Tensor)
                or edge_index.ndim != 2
                or edge_index.shape[0] != 2
                or edge_index.is_floating_point()
            ):
                raise GraphTaskError(f"graph {graph_id}: edge_index must be an integer (2, edges) tensor")
            if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= x.shape[0]):
                raise GraphTaskError(
                    f"graph {graph_id}: an edge points outside the graph's {x.shape[0]} nodes (a cross-graph edge)"
                )
            raw = targets[position] if targets is not None else getattr(graph, "y", None)
            if graph_id in flagged:
                # A target given as an argument or carried as ``y`` alike (an
                # IGNORE ``y``, as a collection's own items carry, is none).
                if raw is not None and torch.as_tensor(raw).reshape(-1).tolist() != [IGNORE]:
                    raise GraphTaskError(f"graph {graph_id} is flagged unlabeled but has a target")
                label = IGNORE
            elif raw is None:
                raise GraphTaskError(f"graph {graph_id} has no target; give it one or flag it with unlabeled=[...]")
            else:
                label = _label(raw, graph_id)
                if num_classes is not None and label >= num_classes:
                    raise GraphTaskError(f"graph {graph_id}: target {label} is not below num_classes={num_classes}")
            items.append(
                Data(
                    x=x,
                    edge_index=edge_index.long(),
                    y=torch.tensor([label], dtype=torch.long),
                    graph_id=torch.tensor([graph_id], dtype=torch.long),
                )
            )
            labels.append(label)
        self._items = items
        self.ids: tuple[int, ...] = tuple(ids)
        self.labels: tuple[int, ...] = tuple(labels)
        self.input_dim: int = int(width or 0)
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Any:
        return self._items[index]

    @property
    def labeled(self) -> int:
        """How many graphs carry a target."""
        return sum(label != IGNORE for label in self.labels)

    def subset(self, ids: Sequence[int]) -> GraphCollection:
        """The graphs with these ids, in the order given."""
        ids = list(ids)  # read more than once: an iterator would be spent
        positions = {graph_id: position for position, graph_id in enumerate(self.ids)}
        missing = sorted({i for i in ids if i not in positions})
        if missing:
            raise GraphTaskError(f"no graphs with ids {missing}")
        chosen = [self._items[positions[i]] for i in ids]
        flagged = {i for i in ids if self.labels[positions[i]] == IGNORE}
        return GraphCollection(
            chosen,
            list(ids),
            targets=[None if i in flagged else self.labels[positions[i]] for i in ids],
            unlabeled=sorted(flagged),
            num_classes=self.num_classes,
        )

    def loader(self, batch_size: int, *, shuffle: bool = False, seed: Optional[int] = None) -> Any:
        """A PyG loader over whole graphs (in collection order unless
        ``shuffle``; a ``seed`` makes the shuffle reproducible and is refused
        without it)."""
        from torch_geometric.loader import DataLoader

        batch_size = _count(batch_size, "batch_size", minimum=1)
        if seed is not None and not shuffle:
            raise GraphTaskError("a seed orders a shuffled loader; pass shuffle=True or no seed")
        generator = None if seed is None else torch.Generator().manual_seed(_count(seed, "seed", minimum=0))
        return DataLoader(self._items, batch_size=batch_size, shuffle=shuffle, generator=generator)


# --- batches ---------------------------------------------------------------------------------------


def check_graph_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(x, edge_index, batch_vector, ptr)`` of a graph-collection batch,
    checked: graph ids (unique), at least one node per graph, ``ptr`` and the
    batch vector consistent, and no edge across graphs. An edge-label (link)
    batch is refused: pooling it would score candidate edges as graphs."""
    if getattr(batch, "edge_label_index", None) is not None:
        raise GraphTaskError(
            "an edge-label (link) batch asks about candidate edges, not whole graphs; use nnx.link_tasks"
        )
    if not is_graph_collection_batch(batch):
        raise GraphTaskError(
            "a graph classifier reads batches of whole graphs with graph ids (GraphCollection.loader()); "
            f"got {type(batch).__name__} without graph_id"
        )
    x, edge_index = getattr(batch, "x", None), getattr(batch, "edge_index", None)
    vector, ptr = getattr(batch, "batch", None), getattr(batch, "ptr", None)
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise GraphTaskError("a graph batch needs a (nodes, features) x")
    n_nodes = int(x.shape[0])
    ids = batch.graph_id
    if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.is_floating_point():
        raise GraphTaskError("graph_id must be one integer per graph")
    n_graphs = int(ids.shape[0])
    if len(set(ids.tolist())) != n_graphs:
        raise GraphTaskError(f"duplicate graph ids in one batch: {sorted(ids.tolist())}")
    if not isinstance(vector, torch.Tensor) or not isinstance(ptr, torch.Tensor):
        raise GraphTaskError("a graph batch needs its batch vector and ptr (collate graphs with a PyG loader)")
    if ptr.ndim != 1 or ptr.shape[0] != n_graphs + 1 or int(ptr[0]) != 0 or int(ptr[-1]) != n_nodes:
        raise GraphTaskError(f"inconsistent ptr {ptr.tolist()} for {n_graphs} graphs and {n_nodes} nodes")
    sizes = ptr[1:] - ptr[:-1]
    if bool((sizes <= 0).any()):
        empty = [int(ids[i]) for i in (sizes <= 0).nonzero().reshape(-1).tolist()]
        raise GraphTaskError(f"graphs {empty} have no nodes (or ptr is not increasing)")
    if vector.shape != (n_nodes,) or not torch.equal(
        vector.cpu(), torch.repeat_interleave(torch.arange(n_graphs), sizes.cpu())
    ):
        raise GraphTaskError("the batch vector does not match ptr: each graph's nodes must be contiguous")
    if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise GraphTaskError("a graph batch needs a (2, edges) edge_index")
    if edge_index.numel():
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= n_nodes:
            raise GraphTaskError("an edge points outside the batch's nodes")
        if bool((vector[edge_index[0]] != vector[edge_index[1]]).any()):
            raise GraphTaskError("an edge joins two different graphs (a cross-graph edge)")
    return x, edge_index, vector, ptr


def _targets(batch: Any, n_graphs: int) -> Optional[torch.Tensor]:
    y = getattr(batch, "y", None)
    if y is None:
        return None
    if not isinstance(y, torch.Tensor) or y.reshape(-1).shape[0] != n_graphs:
        got = tuple(y.shape) if isinstance(y, torch.Tensor) else type(y).__name__
        raise GraphTaskError(
            f"a graph batch needs one target per graph ({n_graphs}); got {got} — flag a graph without a "
            "target as unlabeled (GraphCollection(unlabeled=[...]))"
        )
    return y.reshape(-1)


# --- pooling and the classifier ------------------------------------------------------------------------


class GraphPool(nn.Module):
    """Pools node rows into one row per graph: ``"mean"`` or ``"sum"`` over
    each graph's own nodes (graph-local, invariant to node order)."""

    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        if mode not in POOLS:
            raise GraphTaskError(f"pool must be one of {POOLS}, got {mode!r}")
        self.mode = mode

    def forward(self, x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        pooled = x.new_zeros((num_graphs, *x.shape[1:])).index_add_(0, batch, x)
        if self.mode == "sum":
            return pooled
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).to(x.dtype)
        return pooled / counts.reshape(-1, *([1] * (x.ndim - 1)))

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}"


class GraphClassifier(nn.Module):
    """``encoder`` (node rows from ``(x, edge_index)``) → :class:`GraphPool`
    → optional ``head``: one output row per graph.

    ``unpack_batch`` validates a graph-collection batch (see
    :func:`check_graph_batch`) and returns its graph-level targets;
    ``sample_ids`` returns its graph ids, which ``predict_proba`` reports as
    the rows' sample ids."""

    onnx_export_unsupported = ONNX_UNSUPPORTED

    def __init__(
        self,
        encoder: nn.Module,
        pool: Any = "mean",
        head: Optional[nn.Module] = None,
        *,
        input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise GraphTaskError(f"encoder must be an nn.Module, got {type(encoder).__name__}")
        self.encoder = encoder
        self.pool = pool if isinstance(pool, GraphPool) else GraphPool(pool)
        self.head = head
        self.input_dim = None if input_dim is None else _count(input_dim, "input_dim", minimum=1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: torch.Tensor
    ) -> torch.Tensor:
        parameter = next(self.parameters(), None)
        if parameter is not None and x.is_floating_point() and x.dtype != parameter.dtype:
            x = x.to(parameter.dtype)  # e.g. float64 features from NumPy for a float32 model
        nodes = self.encoder(x, edge_index)
        if not isinstance(nodes, torch.Tensor) or nodes.shape[0] != x.shape[0]:
            raise GraphTaskError("the encoder must return one row per node")
        graphs = self.pool(nodes, batch, int(ptr.shape[0]) - 1)
        return graphs if self.head is None else self.head(graphs)

    def unpack_batch(self, batch: Any) -> tuple[tuple[torch.Tensor, ...], Optional[torch.Tensor]]:
        x, edge_index, vector, ptr = check_graph_batch(batch)
        if self.input_dim is not None and x.shape[1] != self.input_dim:
            raise GraphTaskError(f"the batch has {x.shape[1]} node features; the classifier reads {self.input_dim}")
        return (x, edge_index, vector, ptr), _targets(batch, int(ptr.shape[0]) - 1)

    def sample_ids(self, batch: Any) -> torch.Tensor:
        """The batch's graph ids, one per output row, in batch order."""
        if not is_graph_collection_batch(batch):
            check_graph_batch(batch)  # raises with the reason
        return batch.graph_id


# --- the registered recipe ---------------------------------------------------------------------------------


class _Encoder(nn.Module):
    """PyG convolutions with an activation (and dropout) after every layer —
    or, with ``last_activation=False``, after every layer but the last, so
    the node embeddings are unconstrained (a dot-product edge decoder needs
    negative logits)."""

    def __init__(
        self, kind: str, dims: Sequence[int], activation: str, dropout: float, *, last_activation: bool = True
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv

        from .nn.enum.activations import Activations

        conv = {"graph_conv": GCNConv, "graph_sage": SAGEConv, "graph_att": GATConv}[kind]
        self.layers = nn.ModuleList(conv(a, b) for a, b in zip(dims, dims[1:], strict=False))
        self.activation = Activations(activation)()
        self.dropout = float(dropout)
        self.last_activation = bool(last_activation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        last = len(self.layers) - 1
        for position, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if position < last or self.last_activation:
                x = self.activation(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


def _check_config(config: Mapping[str, Any]) -> dict[str, Any]:
    known = {"encoder", "input_dim", "hidden_dims", "num_classes", "pool", "activation", "dropout"}
    unknown = sorted(set(config) - known)
    if unknown:
        raise GraphTaskError(f"unknown graph classifier settings {unknown}")
    encoder = config.get("encoder", "graph_conv")
    if encoder not in ENCODERS:
        raise GraphTaskError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
    hidden = config.get("hidden_dims")
    if not isinstance(hidden, (list, tuple)) or not hidden:
        raise GraphTaskError(f"hidden_dims must be a non-empty list of layer widths, got {hidden!r}")
    dropout = config.get("dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, numbers.Real) or not 0.0 <= float(dropout) < 1.0:
        raise GraphTaskError(f"dropout must be in [0, 1), got {dropout!r}")
    pool = config.get("pool", "mean")
    if pool not in POOLS:
        raise GraphTaskError(f"pool must be one of {POOLS}, got {pool!r}")
    from .nn.enum.activations import Activations

    try:
        activation = Activations(
            getattr(config.get("activation", "relu"), "value", config.get("activation", "relu"))
        ).value
    except ValueError as error:
        raise GraphTaskError(f"unknown activation {config.get('activation')!r}") from error
    return {
        "encoder": encoder,
        "input_dim": _count(config.get("input_dim"), "input_dim", minimum=1),
        "hidden_dims": [_count(h, "a hidden width", minimum=1) for h in hidden],
        "num_classes": _count(config.get("num_classes"), "num_classes", minimum=2),
        "pool": pool,
        "activation": activation,
        "dropout": float(dropout),
    }


def _build(config: Mapping[str, Any]) -> GraphClassifier:
    settings = _check_config(config)
    dims = [settings["input_dim"], *settings["hidden_dims"]]
    encoder = _Encoder(settings["encoder"], dims, settings["activation"], settings["dropout"])
    return GraphClassifier(
        encoder, settings["pool"], nn.Linear(dims[-1], settings["num_classes"]), input_dim=settings["input_dim"]
    )


def graph_classifier_spec(
    *,
    input_dim: int,
    num_classes: int,
    hidden_dims: Sequence[int] = (32,),
    encoder: str = "graph_conv",
    pool: str = "mean",
    activation: str = "relu",
    dropout: float = 0.0,
    seed: int = 0,
) -> Any:
    """The registered recipe of a :class:`GraphClassifier`: ``encoder``
    convolutions (``"graph_conv"`` GCN, ``"graph_sage"``, ``"graph_att"``
    GAT) over ``hidden_dims``, ``pool`` and a linear head to
    ``num_classes``. Pass it as ``NNModelParams(net=...)``; a reload
    rebuilds the same module from the run."""
    from .models import ModelSpec

    config = {
        "encoder": encoder,
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "num_classes": num_classes,
        "pool": pool,
        "activation": activation,
        "dropout": dropout,
    }
    return ModelSpec(FACTORY_ID, 1, _check_config(config), seed=seed)


def _register() -> None:
    from .models import register_model_factory, registered_model_factories

    if (FACTORY_ID, 1) not in registered_model_factories():
        register_model_factory(FACTORY_ID, 1, _build)


_register()
