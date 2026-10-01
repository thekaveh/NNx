"""Leakage-aware link and edge prediction on a static homogeneous graph (FEAT-027).

Node masks cannot hold out edges: a held-out edge left in the message graph
is a leaked answer. This module splits **edges**, keeps the graph the model
passes messages over apart from the edges it is asked about, and checks
every batch against that split before any forward pass.

- :class:`LinkSplit` — a versioned manifest: the node count, canonical
  positive edges per split (``train`` / ``val`` / ``test``; an undirected
  edge is ``(min, max)``, so a duplicate or a reverse can never sit in two
  splits), fixed ``val`` / ``test`` negatives, directedness, the self-loop
  policy, the seed, and — for edge-label tasks — the categories, each
  edge's category and whether labelled edges' **existence** is visible
  context. :func:`split_links` derives one from an ``edge_index`` with a
  local seeded generator (never the global RNG); :meth:`LinkSplit.replay`
  re-derives it and refuses a mismatch.
- **Topology.** :meth:`LinkSplit.message_edge_index` is the only graph
  messages pass over: the training edges (and their reverses when
  undirected) — for every split, so evaluation never sees held-out
  positives. In edge-label mode with ``label_existence="visible"`` every
  labelled edge's existence is context (their categories never are).
- **Negatives** come from the static complement — never a positive of any
  split, its reverse, a duplicate or a barred self-loop. A request beyond
  the complement's capacity fails before anything is sampled; ``val`` /
  ``test`` negatives are fixed in the manifest; training negatives are
  re-drawn every pass from ``(seed, pass)``.
- :class:`LinkTask` — ``"binary"`` (link existence: one logit per
  candidate, targets 0/1) or ``"edge_label"`` (categories in their declared
  order; no negatives, so a non-edge is never a category). ``loader()``
  builds batches of candidates over the message graph; ``objective()``,
  ``eval_step()`` and ``predict()`` check each batch against the split —
  a message edge outside the training topology (a hidden positive) or a
  candidate outside its split fails before any update. Evaluation
  materialises every candidate (up to ``max_candidates``) for exact
  AUROC / AP; a one-class set reports them unavailable with the reason.
  The manifest is checkpointed component state ``"link.task"``: a resume
  with another split fails before the first resumed update.
- :func:`link_predictor_spec` — a registered recipe
  (``ModelSpec("nnx.link_predictor", 1, config)``): a GCN / GraphSAGE /
  GAT encoder and a ``"dot"`` or ``"mlp"`` edge decoder, rebuilt on reload.

Homogeneous static graphs only: no temporal or heterogeneous graphs, and
unobserved pairs are negatives only as the split declares them.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import operator
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from .components import ComponentSpec
from .objectives import LossTerm, Objective, ObjectiveContext, ObjectiveResult

__all__ = [
    "FACTORY_ID",
    "FORMAT",
    "LinkEval",
    "LinkObjective",
    "LinkPrediction",
    "LinkPredictor",
    "LinkSplit",
    "LinkTask",
    "LinkTaskError",
    "MetricValue",
    "is_link_batch",
    "link_metrics",
    "link_predictor_spec",
    "split_links",
]

FORMAT = "nnx.link-split/1"
FACTORY_ID = "nnx.link_predictor"
KIND = "link"
MODES = ("binary", "edge_label")
SPLITS = ("train", "val", "test")
DENSE_LIMIT = 2_000_000  # candidate pairs enumerated exactly; larger graphs sample by rejection
Edge = tuple[int, int]


class LinkTaskError(ValueError):
    """A split, batch or setting that would leak or cannot be served."""


def _count(value: Any, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise LinkTaskError(f"{what} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def is_link_batch(batch: Any) -> bool:
    """Whether ``batch`` asks about candidate edges (it has ``edge_label_index``)."""
    return getattr(batch, "edge_label_index", None) is not None


# --- edges -------------------------------------------------------------------------------------------


def _canonical(u: int, v: int, directed: bool) -> Edge:
    return (u, v) if directed or u <= v else (v, u)


def _pairs(edge_index: Any, what: str) -> list[Edge]:
    if isinstance(edge_index, torch.Tensor):
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.is_floating_point():
            raise LinkTaskError(f"{what} must be an integer (2, edges) tensor, got {tuple(edge_index.shape)}")
        return [(int(u), int(v)) for u, v in edge_index.t().tolist()]
    pairs = []
    for item in edge_index:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise LinkTaskError(f"{what} holds (u, v) pairs, got {item!r}")
        pairs.append((_count(item[0], "a node index", minimum=0), _count(item[1], "a node index", minimum=0)))
    return pairs


def _integer(value: Any) -> Any:
    """``value`` as a builtin int when it is any integer (numpy, a 0-d
    tensor); anything else unchanged, for the caller to refuse."""
    if isinstance(value, bool):
        return value
    try:
        return operator.index(value)
    except TypeError:
        return value


def _node_pair(pair: Any, what: str) -> Edge:
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise LinkTaskError(f"{what} holds (u, v) pairs, got {pair!r}")
    return (
        _count(_integer(pair[0]), "a node index", minimum=0),
        _count(_integer(pair[1]), "a node index", minimum=0),
    )


def _total_pairs(num_nodes: int, directed: bool, self_loops: bool) -> int:
    if directed:
        return num_nodes * num_nodes if self_loops else num_nodes * (num_nodes - 1)
    return num_nodes * (num_nodes + 1) // 2 if self_loops else num_nodes * (num_nodes - 1) // 2


# --- the manifest ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LinkSplit:
    """A versioned edge split (see the module docstring). Every edge is
    canonical; validation refuses an edge outside the graph, a barred
    self-loop, a positive in two splits (or twice in one), and a negative
    that is a positive or repeats."""

    num_nodes: int
    train: tuple[Edge, ...]
    val: tuple[Edge, ...] = ()
    test: tuple[Edge, ...] = ()
    val_negatives: tuple[Edge, ...] = ()
    test_negatives: tuple[Edge, ...] = ()
    directed: bool = False
    self_loops: str = "bar"
    seed: Optional[int] = None
    categories: Optional[tuple[str, ...]] = None
    edge_labels: tuple[tuple[Edge, int], ...] = ()
    label_existence: str = "hidden"
    version: int = 1

    def __post_init__(self) -> None:
        n = _count(self.num_nodes, "num_nodes", minimum=2)
        object.__setattr__(self, "num_nodes", n)
        if not isinstance(self.directed, bool):
            raise LinkTaskError(f"directed must be a bool, got {self.directed!r}")
        if self.self_loops not in ("bar", "allow"):
            raise LinkTaskError(f"self_loops must be 'bar' or 'allow', got {self.self_loops!r}")
        object.__setattr__(self, "self_loops", str(self.self_loops))
        if self.seed is not None:
            object.__setattr__(self, "seed", _count(self.seed, "seed", minimum=0))
        if self.version != 1 or isinstance(self.version, bool):
            raise LinkTaskError(f"this NNx reads link split version 1, got {self.version!r}")
        if self.label_existence not in ("hidden", "visible"):
            raise LinkTaskError(f"label_existence must be 'hidden' or 'visible', got {self.label_existence!r}")
        object.__setattr__(self, "label_existence", str(self.label_existence))
        owner: dict[Edge, str] = {}
        for name in ("train", "val", "test", "val_negatives", "test_negatives"):
            edges = tuple(self._edge(pair, name) for pair in _pairs(getattr(self, name), name))
            object.__setattr__(self, name, edges)
            for edge in edges:
                if edge in owner:
                    what = "a duplicate (or a reverse)" if not self.directed else "a duplicate"
                    raise LinkTaskError(
                        f"edge {edge} is {what} of an edge already in {owner[edge]!r} (now in {name!r})"
                    )
                owner[edge] = name
        if not self.train:
            raise LinkTaskError("a link split needs at least one training edge")
        if self.categories is not None:
            categories = tuple(str(c) for c in self.categories)
            if len(categories) < 2 or len(set(categories)) != len(categories) or not all(categories):
                raise LinkTaskError(f"categories must be 2+ distinct non-empty names, got {self.categories!r}")
            object.__setattr__(self, "categories", categories)
            if self.val_negatives or self.test_negatives:
                raise LinkTaskError("an edge-label split has no negatives: a non-edge is never a category")
            labels: dict[Edge, int] = {}
            for pair, category in self.edge_labels:
                edge = self._edge(_node_pair(pair, "edge_labels"), "edge_labels")
                if edge in labels:
                    raise LinkTaskError(f"edge {edge} has more than one category in edge_labels")
                labels[edge] = _count(_integer(category), f"edge {edge}'s category", minimum=0)
                if labels[edge] >= len(categories):
                    raise LinkTaskError(f"edge {edge}'s category {labels[edge]} has no name in {categories}")
            positives = set(self.train) | set(self.val) | set(self.test)
            if set(labels) != positives:
                missing = sorted(positives - set(labels))[:5]
                extra = sorted(set(labels) - positives)[:5]
                raise LinkTaskError(f"every split edge needs exactly one category (missing {missing}, unknown {extra})")
            object.__setattr__(self, "edge_labels", tuple(sorted(labels.items())))
        elif self.edge_labels:
            raise LinkTaskError("edge_labels need categories")
        elif self.label_existence != "hidden":
            raise LinkTaskError("label_existence applies to an edge-label split")

    def _edge(self, pair: Edge, what: str) -> Edge:
        u, v = pair
        if not (0 <= u < self.num_nodes and 0 <= v < self.num_nodes):
            raise LinkTaskError(f"{what}: edge {pair} is outside the graph's {self.num_nodes} nodes")
        if u == v and self.self_loops == "bar":
            raise LinkTaskError(f"{what}: self-loop {pair} with self_loops='bar'")
        edge = _canonical(u, v, self.directed)
        if edge != pair:
            raise LinkTaskError(f"{what}: edge {pair} is not canonical; an undirected edge is (min, max): {edge}")
        return edge

    # ---------- identity ----------

    @property
    def mode(self) -> str:
        return "edge_label" if self.categories is not None else "binary"

    def positives(self, name: str) -> tuple[Edge, ...]:
        return getattr(self, self._split(name))

    def negatives(self, name: str) -> tuple[Edge, ...]:
        name = self._split(name)
        return () if name == "train" else getattr(self, f"{name}_negatives")

    @staticmethod
    def _split(name: str) -> str:
        if name not in SPLITS:
            raise LinkTaskError(f"split must be one of {SPLITS}, got {name!r}")
        return name

    def candidate_ids(self, name: str) -> dict[Edge, int]:
        """Stable integer ids of a split's fixed candidates (positives, then
        negatives): one global numbering over the manifest."""
        offset = 0
        for split in SPLITS:
            if split == name:
                break
            offset += len(self.positives(split))
        ids = {edge: offset + i for i, edge in enumerate(self.positives(name))}
        base = sum(len(self.positives(s)) for s in SPLITS)
        for split in ("val", "test"):
            if split == name:
                ids.update({edge: base + i for i, edge in enumerate(self.negatives(split))})
            base += len(self.negatives(split))
        return ids

    def labels(self) -> dict[Edge, int]:
        return dict(self.edge_labels)

    def message_edge_index(self) -> torch.Tensor:
        """The only edges messages pass over, for every split: the training
        edges (with reverses when undirected) — plus every labelled edge's
        existence in an edge-label split with ``label_existence="visible"``."""
        return self._messages().clone()

    def _messages(self) -> torch.Tensor:
        """The message graph, built once per split (callers get clones)."""

        def build() -> torch.Tensor:
            edges = list(self.train)
            if self.label_existence == "visible":
                edges += [*self.val, *self.test]
            pairs = edges + ([] if self.directed else [(v, u) for u, v in edges if u != v])
            return torch.tensor(pairs, dtype=torch.long).t().contiguous().reshape(2, -1)

        return self._cached("messages_tensor", build)

    def _message_set(self) -> frozenset[Edge]:
        return self._cached("_messages", lambda: frozenset((int(u), int(v)) for u, v in self._messages().t().tolist()))

    def _is_message_graph(self, edge_index: torch.Tensor) -> bool:
        """Whether ``edge_index`` holds exactly the message edges (each once,
        in any order) — a vectorised check, so a large graph costs a sort,
        not a Python set per batch."""
        if edge_index.is_floating_point() or edge_index.is_complex():
            return False
        if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= self.num_nodes):
            return False  # out-of-graph ids would alias keys: the exact check names them
        keys = edge_index[0].long() * self.num_nodes + edge_index[1].long()
        allowed = self._cached(f"message_keys:{keys.device}", lambda: self._message_keys().to(keys.device))
        return keys.shape == allowed.shape and bool(torch.equal(torch.sort(keys).values, allowed))

    def _message_keys(self) -> torch.Tensor:
        messages = self._messages()
        return torch.sort(messages[0] * self.num_nodes + messages[1]).values

    def _all_positives(self) -> frozenset[Edge]:
        return self._cached("_positives", lambda: frozenset((*self.train, *self.val, *self.test)))

    def _cached(self, key: str, build: Any) -> Any:
        """A derived set, built once per (immutable) split."""
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            cache[key] = build()
        return cache[key]

    def state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "format": FORMAT,
            "num_nodes": self.num_nodes,
            "directed": self.directed,
            "self_loops": self.self_loops,
            "seed": self.seed,
            "topology": "train_edges" + ("+labelled_existence" if self.label_existence == "visible" else ""),
        }
        for name in ("train", "val", "test", "val_negatives", "test_negatives"):
            state[name] = [list(edge) for edge in getattr(self, name)]
        if self.categories is not None:
            state["categories"] = list(self.categories)
            state["edge_labels"] = [[list(edge), category] for edge, category in self.edge_labels]
            state["label_existence"] = self.label_existence
        return state

    @staticmethod
    def from_state(state: Mapping[str, Any]) -> LinkSplit:
        if not isinstance(state, Mapping) or state.get("format") != FORMAT:
            raise LinkTaskError(f"not a {FORMAT} manifest")
        try:
            return LinkSplit(
                num_nodes=state["num_nodes"],
                train=tuple(tuple(e) for e in state["train"]),
                val=tuple(tuple(e) for e in state["val"]),
                test=tuple(tuple(e) for e in state["test"]),
                val_negatives=tuple(tuple(e) for e in state["val_negatives"]),
                test_negatives=tuple(tuple(e) for e in state["test_negatives"]),
                directed=state["directed"],
                self_loops=state["self_loops"],
                seed=state["seed"],
                categories=None if state.get("categories") is None else tuple(state["categories"]),
                edge_labels=tuple((tuple(e), c) for e, c in state.get("edge_labels", ())),
                label_existence=state.get("label_existence", "hidden"),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, LinkTaskError):
                raise
            raise LinkTaskError(f"malformed {FORMAT} manifest: {error}") from error

    def _negatives_per_positive(self) -> int:
        """The ``negatives`` ratio ``split_links`` was called with, read from
        whichever held-out split has positives (none drew no negatives)."""
        for positives, negatives in ((self.val, self.val_negatives), (self.test, self.test_negatives)):
            if positives:
                return len(negatives) // len(positives)
        return 0

    def digest(self) -> str:
        text = json.dumps(self.state(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def replay(self, edge_index: Any, *, edge_labels: Optional[Sequence[int]] = None) -> LinkSplit:
        """Re-derive this split from the graph with its own seed and policy;
        a mismatch (different graph, settings or code) raises."""
        if self.seed is None:
            raise LinkTaskError("a split given explicitly (no seed) cannot be replayed")
        replayed = split_links(
            edge_index,
            self.num_nodes,
            val=len(self.val),
            test=len(self.test),
            seed=self.seed,
            directed=self.directed,
            self_loops=self.self_loops,
            negatives=self._negatives_per_positive(),
            edge_labels=edge_labels,
            categories=self.categories,
            label_existence=self.label_existence,
        )
        if replayed.digest() != self.digest():
            raise LinkTaskError(f"replay does not reproduce the split: {replayed.digest()} != {self.digest()}")
        return replayed


def _complement_sample(
    count: int, num_nodes: int, excluded: set[Edge], *, directed: bool, self_loops: bool, rng: np.random.Generator
) -> list[Edge]:
    """``count`` distinct canonical non-edges, drawn with ``rng``; the caller
    has checked the complement holds that many."""
    if count == 0:
        return []
    if _total_pairs(num_nodes, directed, self_loops) <= DENSE_LIMIT:
        pool = [
            (u, v)
            for u in range(num_nodes)
            for v in (range(num_nodes) if directed else range(u, num_nodes))
            if (self_loops or u != v) and (u, v) not in excluded
        ]
        chosen = rng.choice(len(pool), size=count, replace=False)
        return [pool[int(i)] for i in chosen]
    out: list[Edge] = []
    seen = set(excluded)
    while len(out) < count:
        u, v = (int(x) for x in rng.integers(0, num_nodes, size=2))
        if u == v and not self_loops:
            continue
        edge = _canonical(u, v, directed)
        if edge not in seen:
            seen.add(edge)
            out.append(edge)
    return out


def split_links(
    edge_index: Any,
    num_nodes: int,
    *,
    val: Any = 0.1,
    test: Any = 0.1,
    seed: int,
    directed: bool = False,
    self_loops: str = "bar",
    negatives: int = 1,
    edge_labels: Optional[Sequence[int]] = None,
    categories: Optional[Sequence[str]] = None,
    label_existence: str = "hidden",
) -> LinkSplit:
    """Split the graph's edges with a local generator seeded by ``seed``.

    Edges are canonicalised and de-duplicated first (an undirected reverse
    is the same edge), so no duplicate or reverse can cross splits.
    ``val`` / ``test`` are fractions of the edges or exact counts.
    ``negatives`` fixed non-edges per ``val`` / ``test`` positive come from
    the complement of every positive (and barred self-loops); a request
    beyond the complement's capacity fails before sampling. With
    ``categories`` (and one category per input edge in ``edge_labels``) the
    split is an edge-label split: no negatives, and a duplicate edge must
    keep its category."""
    n = _count(num_nodes, "num_nodes", minimum=2)
    if self_loops not in ("bar", "allow"):
        raise LinkTaskError(f"self_loops must be 'bar' or 'allow', got {self_loops!r}")
    seed = _count(seed, "seed", minimum=0)
    pairs = _pairs(edge_index, "edge_index")
    labelled = categories is not None
    if labelled and (edge_labels is None or len(edge_labels) != len(pairs)):
        raise LinkTaskError("an edge-label split needs one category per input edge (edge_labels)")
    if not labelled and edge_labels is not None:
        raise LinkTaskError("edge_labels need categories")
    edges: dict[Edge, Optional[int]] = {}
    for position, (u, v) in enumerate(pairs):
        if not (0 <= u < n and 0 <= v < n):
            raise LinkTaskError(f"edge {(u, v)} is outside the graph's {n} nodes")
        if u == v and self_loops == "bar":
            raise LinkTaskError(f"self-loop {(u, v)} with self_loops='bar'")
        edge = _canonical(u, v, directed)
        label = None if edge_labels is None else int(edge_labels[position])
        if edge in edges and edges[edge] != label:
            raise LinkTaskError(f"edge {edge} appears twice with different categories ({edges[edge]} and {label})")
        edges[edge] = label
    unique = sorted(edges)
    total = len(unique)

    def size(value: Any, what: str) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise LinkTaskError(f"{what} must be a fraction or a count, got {value!r}")
        if isinstance(value, numbers.Integral):
            return _count(value, what, minimum=0)
        if not 0.0 <= float(value) < 1.0:
            raise LinkTaskError(f"{what} fraction must be in [0, 1), got {value!r}")
        return int(round(float(value) * total))

    n_val, n_test = size(val, "val"), size(test, "test")
    if n_val + n_test >= total:
        raise LinkTaskError(f"{n_val} val and {n_test} test edges leave no training edge of {total}")
    rng = np.random.default_rng(seed)
    order = [unique[int(i)] for i in rng.permutation(total)]
    val_edges, test_edges, train_edges = order[:n_val], order[n_val : n_val + n_test], order[n_val + n_test :]
    val_negatives: list[Edge] = []
    test_negatives: list[Edge] = []
    if not labelled:
        per = _count(negatives, "negatives", minimum=0)
        needed = per * (n_val + n_test)
        capacity = _total_pairs(n, directed, self_loops == "allow") - total
        if needed > capacity:
            raise LinkTaskError(
                f"{needed} negatives requested but the complement holds only {capacity} non-edges; lower "
                "negatives or the held-out fraction"
            )
        drawn = _complement_sample(needed, n, set(unique), directed=directed, self_loops=self_loops == "allow", rng=rng)
        val_negatives, test_negatives = drawn[: per * n_val], drawn[per * n_val :]
    return LinkSplit(
        num_nodes=n,
        train=tuple(sorted(train_edges)),
        val=tuple(sorted(val_edges)),
        test=tuple(sorted(test_edges)),
        val_negatives=tuple(sorted(val_negatives)),
        test_negatives=tuple(sorted(test_negatives)),
        directed=directed,
        self_loops=self_loops,
        seed=seed,
        categories=None if categories is None else tuple(categories),
        edge_labels=tuple((edge, int(label)) for edge, label in edges.items() if label is not None),
        label_existence=label_existence,
    )


# --- metrics -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricValue:
    """A metric's value, or ``None`` with the reason it is unavailable."""

    value: Optional[float]
    reason: Optional[str] = None


def link_metrics(probabilities: Any, targets: Any) -> dict[str, MetricValue]:
    """Exact metrics over a whole candidate set. Binary (``probabilities``
    of shape ``(K,)``): ``bce``, ``auroc``, ``ap`` (unavailable, with the
    reason, for a one-class set) and ``accuracy`` at 0.5. Categorical
    (``(K, C)``): ``nll`` and ``accuracy``."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets)
    if p.shape[0] != y.shape[0]:
        raise LinkTaskError(f"{p.shape[0]} predictions for {y.shape[0]} targets")
    if y.shape[0] == 0:
        return {"bce" if p.ndim == 1 else "nll": MetricValue(None, "no candidates")}
    if p.ndim == 2:
        chosen = np.clip(p[np.arange(y.shape[0]), y.astype(np.int64)], 1e-300, 1.0)
        return {
            "nll": MetricValue(float(-np.log(chosen).mean())),
            "accuracy": MetricValue(float((p.argmax(axis=1) == y).mean())),
        }
    clipped = np.clip(p, 1e-300, 1.0 - 1e-16)
    bce = float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean())
    out = {"bce": MetricValue(bce), "accuracy": MetricValue(float(((p >= 0.5) == (y == 1)).mean()))}
    classes = set(np.unique(y).tolist())
    if classes != {0, 1}:
        reason = f"only class {sorted(classes)} present: AUROC and AP need both positives and negatives"
        out["auroc"], out["ap"] = MetricValue(None, reason), MetricValue(None, reason)
    else:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auroc"] = MetricValue(float(roc_auc_score(y, p)))
        out["ap"] = MetricValue(float(average_precision_score(y, p)))
    return out


# --- the model ---------------------------------------------------------------------------------------------


class LinkPredictor(nn.Module):
    """``encoder`` (node rows from ``(x, edge_index)``) and an edge decoder:
    ``"dot"`` (one logit per candidate: ``h_u · h_v``) or ``"mlp"`` (a
    linear head on ``[h_u * h_v, |h_u - h_v|]``: one logit, or one per
    category)."""

    onnx_export_unsupported = (
        "a link predictor reads PyG message graphs and candidate indices, which this exporter does not support; "
        "export its weights with export_state_dict() instead"
    )

    def __init__(self, encoder: nn.Module, decoder: str = "dot", *, width: int = 0, outputs: int = 1) -> None:
        super().__init__()
        if decoder not in ("dot", "mlp"):
            raise LinkTaskError(f"decoder must be 'dot' or 'mlp', got {decoder!r}")
        if decoder == "dot" and outputs != 1:
            raise LinkTaskError("a dot decoder gives one logit per candidate; use 'mlp' for categories")
        self.encoder = encoder
        self.decoder = decoder
        self.head = nn.Linear(2 * width, outputs) if decoder == "mlp" else None
        self.outputs = outputs

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, edge_index)
        hu, hv = h[edge_label_index[0]], h[edge_label_index[1]]
        if self.head is None:
            return (hu * hv).sum(dim=-1)
        logits = self.head(torch.cat([hu * hv, (hu - hv).abs()], dim=-1))
        return logits[:, 0] if self.outputs == 1 else logits

    def unpack_batch(self, batch: Any) -> tuple[tuple[torch.Tensor, ...], Optional[torch.Tensor]]:
        """Refused: the default train / evaluate / predict paths cannot check
        a batch against its split, so a leak would pass unseen."""
        raise LinkTaskError(
            "a link predictor trains, evaluates and predicts through its LinkTask, which checks every batch "
            "against the split: train with objective=task.objective() and eval_step_fn=task.eval_step(), "
            "and predict with task.predict(model, loader)"
        )

    def sample_ids(self, batch: Any) -> torch.Tensor:
        """The candidates' ids, one per output row."""
        _check_structure(batch)
        return batch.candidate_id


def _check_structure(batch: Any) -> None:
    if not is_link_batch(batch):
        raise LinkTaskError("a link predictor reads link batches (edge_label_index); build them with LinkTask.loader()")
    k = int(batch.edge_label_index.shape[1]) if batch.edge_label_index.ndim == 2 else -1
    if batch.edge_label_index.ndim != 2 or batch.edge_label_index.shape[0] != 2:
        raise LinkTaskError("edge_label_index must be (2, candidates)")
    ids = getattr(batch, "candidate_id", None)
    if not isinstance(ids, torch.Tensor) or ids.shape != (k,):
        raise LinkTaskError("a link batch needs one candidate_id per candidate")
    label = getattr(batch, "edge_label", None)
    if not isinstance(label, torch.Tensor) or label.shape != (k,):
        raise LinkTaskError(f"a link batch needs one edge_label (target) per candidate: {k} candidates")


def _check_config(config: Mapping[str, Any]) -> dict[str, Any]:
    from .graph_tasks import ENCODERS

    known = {"encoder", "input_dim", "hidden_dims", "decoder", "num_categories", "activation", "dropout"}
    unknown = sorted(set(config) - known)
    if unknown:
        raise LinkTaskError(f"unknown link predictor settings {unknown}")
    encoder = config.get("encoder", "graph_conv")
    if encoder not in ENCODERS:
        raise LinkTaskError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
    hidden = config.get("hidden_dims")
    if not isinstance(hidden, (list, tuple)) or not hidden:
        raise LinkTaskError(f"hidden_dims must be a non-empty list of layer widths, got {hidden!r}")
    decoder = config.get("decoder", "dot")
    categories = config.get("num_categories")
    if categories is not None:
        categories = _count(categories, "num_categories", minimum=2)
        if decoder != "mlp":
            raise LinkTaskError("an edge-label predictor needs decoder='mlp'")
    if decoder not in ("dot", "mlp"):
        raise LinkTaskError(f"decoder must be 'dot' or 'mlp', got {decoder!r}")
    dropout = config.get("dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, numbers.Real) or not 0.0 <= float(dropout) < 1.0:
        raise LinkTaskError(f"dropout must be in [0, 1), got {dropout!r}")
    from .nn.enum.activations import Activations

    activation = getattr(config.get("activation", "relu"), "value", config.get("activation", "relu"))
    if activation not in {a.value for a in Activations}:
        raise LinkTaskError(f"unknown activation {activation!r}")
    return {
        "encoder": encoder,
        "input_dim": _count(config.get("input_dim"), "input_dim", minimum=1),
        "hidden_dims": [_count(h, "a hidden width", minimum=1) for h in hidden],
        "decoder": decoder,
        "num_categories": categories,
        "activation": activation,
        "dropout": float(dropout),
    }


def _build(config: Mapping[str, Any]) -> LinkPredictor:
    from .graph_tasks import _Encoder

    settings = _check_config(config)
    dims = [settings["input_dim"], *settings["hidden_dims"]]
    # No activation after the last layer: a dot decoder over ReLU embeddings
    # could never give a negative logit, so never predict "no link".
    encoder = _Encoder(settings["encoder"], dims, settings["activation"], settings["dropout"], last_activation=False)
    return LinkPredictor(encoder, settings["decoder"], width=dims[-1], outputs=settings["num_categories"] or 1)


def link_predictor_spec(
    *,
    input_dim: int,
    hidden_dims: Sequence[int] = (32,),
    encoder: str = "graph_conv",
    decoder: str = "dot",
    num_categories: Optional[int] = None,
    activation: str = "relu",
    dropout: float = 0.0,
    seed: int = 0,
) -> Any:
    """The registered recipe of a :class:`LinkPredictor` (pass it as
    ``NNModelParams(net=...)``; a reload rebuilds the same module)."""
    from .models import ModelSpec

    config = {
        "encoder": encoder,
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "decoder": decoder,
        "num_categories": num_categories,
        "activation": activation,
        "dropout": dropout,
    }
    return ModelSpec(FACTORY_ID, 1, _check_config(config), seed=seed)


# --- the task ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LinkPrediction:
    """Candidates and their predictions, aligned row for row: ``ids``,
    ``pairs`` ``(K, 2)``, ``logits`` / ``probabilities`` (``(K,)`` binary,
    ``(K, C)`` categorical) and ``targets`` — all of one ``split``."""

    ids: np.ndarray
    pairs: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray
    split: str


class _Loader:
    """Re-iterable link batches over the split's message graph. A training
    pass draws its negatives and order from ``(seed, pass)``, where the pass
    is the epoch the training loop announced (``set_epoch``) — so an
    uninterrupted run and one resumed at that epoch draw the same batches,
    and extra iterations (a callback scoring the training set) change
    nothing. Before any epoch was announced, the pass is the count of
    earlier iterations; after a fit, it stays the last announced epoch."""

    def __init__(self, task: LinkTask, name: str, x: torch.Tensor, batch_size: int, seed: Optional[int]) -> None:
        self.task, self.name, self.x, self.batch_size, self.seed = task, name, x, batch_size, seed
        self.passes = 0
        self.epoch: Optional[int] = None

    def set_epoch(self, epoch: int) -> None:
        """Called by ``NNModel.train`` / ``Trainer.train`` before each epoch."""
        self.epoch = _count(epoch, "epoch", minimum=0)

    def __iter__(self) -> Iterator[Any]:
        current = self.passes if self.epoch is None else self.epoch
        candidates, targets, ids = self.task._candidates(self.name, self.seed, current)
        self.passes += 1
        messages = self.task.split.message_edge_index()
        for start in range(0, len(candidates), self.batch_size):
            yield self.task._batch(
                self.x,
                messages,
                candidates[start : start + self.batch_size],
                targets[start : start + self.batch_size],
                ids[start : start + self.batch_size],
                self.name,
                current if self.name == "train" else None,
                self.task._negatives_seed(self.seed),
                self.epoch is not None,
            )

    def __len__(self) -> int:
        size = len(self.task.split.positives(self.name)) * (
            1 + (int(self.task.train_negatives or 0) if self.name == "train" and self.task.mode == "binary" else 0)
        ) + len(self.task.split.negatives(self.name))
        return math.ceil(size / self.batch_size)


@dataclass(frozen=True)
class LinkTask:
    """Link existence (``mode="binary"``) or edge categories
    (``"edge_label"``) over a :class:`LinkSplit` (see the module docstring).

    Args:
        split: the manifest.
        mode: ``"binary"`` or ``"edge_label"``; by default the split's (an
            edge-label split has categories).
        train_negatives: non-edges sampled per training positive, per pass
            (binary only; by default 1, and 0 in edge-label mode).
        max_candidates: the most candidates evaluation materialises.

    Training negatives are drawn per epoch from ``(seed, epoch)``, so a
    stateful resume continues them as if uninterrupted; the objective
    checkpoints the seed and each epoch's first training pass, and refuses
    a resumed loader with another seed or one that stopped following the
    epoch.
    """

    split: LinkSplit
    mode: Optional[str] = None
    train_negatives: Optional[int] = None
    max_candidates: int = 1_000_000
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.split, LinkSplit):
            raise LinkTaskError(f"split must be a LinkSplit, got {type(self.split).__name__}")
        if self.mode is None:
            object.__setattr__(self, "mode", self.split.mode)
        if self.train_negatives is None:
            object.__setattr__(self, "train_negatives", 1 if self.mode == "binary" else 0)
        if self.mode not in MODES:
            raise LinkTaskError(f"mode must be one of {MODES}, got {self.mode!r}")
        object.__setattr__(self, "mode", str(self.mode))
        if self.mode != self.split.mode:
            raise LinkTaskError(f"a {self.split.mode} split cannot serve a {self.mode} task")
        negatives = _count(self.train_negatives, "train_negatives", minimum=0)
        if self.mode == "edge_label" and negatives:
            raise LinkTaskError("an edge-label task samples no negatives: a non-edge is never a category")
        object.__setattr__(self, "train_negatives", negatives)
        object.__setattr__(self, "max_candidates", _count(self.max_candidates, "max_candidates", minimum=1))
        if self.version != 1 or isinstance(self.version, bool):
            raise LinkTaskError(f"this NNx supports link task version 1, got {self.version!r}")

    def state(self) -> dict[str, Any]:
        return {
            "kind": KIND,
            "version": self.version,
            "mode": self.mode,
            "train_negatives": self.train_negatives,
            "max_candidates": self.max_candidates,
            "split": self.split.digest(),
        }

    # ---------- batches ----------

    def loader(self, name: str, x: torch.Tensor, batch_size: int, *, seed: Optional[int] = None) -> _Loader:
        """Batches of ``name``'s candidates over the message graph. Training
        negatives and order are re-drawn every pass from ``(seed, pass)``
        (``seed``, an integer >= 0, defaults to the split's); a seed on a
        ``val`` / ``test`` loader, whose candidates are fixed, is refused."""
        LinkSplit._split(name)
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] != self.split.num_nodes:
            raise LinkTaskError(f"x must be ({self.split.num_nodes}, features) node features")
        if seed is not None:
            if name != "train":
                raise LinkTaskError(f"a seed draws training negatives; the {name!r} candidates are fixed")
            seed = _count(seed, "seed", minimum=0)
        return _Loader(self, name, x, _count(batch_size, "batch_size", minimum=1), seed)

    def _negatives_seed(self, seed: Optional[int]) -> int:
        base = self.split.seed if seed is None else seed
        return 0 if base is None else int(base)

    def _candidates(self, name: str, seed: Optional[int], passes: int) -> tuple[list[Edge], list[int], list[int]]:
        split = self.split
        positives = list(split.positives(name))
        ids = split._cached(f"ids:{name}", lambda: split.candidate_ids(name))
        base = self._negatives_seed(seed)
        rows: list[tuple[Edge, int, int]]
        if self.mode == "edge_label":
            labels = split._cached("labels", split.labels)
            rows = [(e, labels[e], ids[e]) for e in positives]
        elif name == "train":
            rows = [(e, 1, ids[e]) for e in positives]
        else:
            negatives = list(split.negatives(name))
            candidates = positives + negatives
            return candidates, [1] * len(positives) + [0] * len(negatives), [ids[e] for e in candidates]
        if name != "train":
            return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
        if self.mode == "binary" and self.train_negatives:
            excluded = set(split._all_positives()) | set(split.val_negatives) | set(split.test_negatives)
            needed = int(self.train_negatives) * len(positives)
            capacity = _total_pairs(split.num_nodes, split.directed, split.self_loops == "allow") - len(excluded)
            if needed > capacity:
                raise LinkTaskError(f"{needed} training negatives requested; the complement holds {capacity}")
            rng = np.random.default_rng([base, passes])
            negatives = _complement_sample(
                needed,
                split.num_nodes,
                excluded,
                directed=split.directed,
                self_loops=split.self_loops == "allow",
                rng=rng,
            )
            rows += [(e, 0, -1) for e in negatives]
        # Every training pass is shuffled from (seed, pass): no batch holds one class or one node range.
        order = np.random.default_rng([base, passes, 1]).permutation(len(rows))
        shuffled = [rows[int(i)] for i in order]
        return [r[0] for r in shuffled], [r[1] for r in shuffled], [r[2] for r in shuffled]

    def _batch(self, x, messages, candidates, targets, ids, name, passes=None, seed=None, followed=False) -> Any:
        from torch_geometric.data import Data

        index = torch.tensor(candidates, dtype=torch.long).t().reshape(2, -1)
        dtype = torch.float32 if self.mode == "binary" else torch.long
        batch = Data(
            x=x,
            edge_index=messages,
            edge_label_index=index,
            edge_label=torch.tensor(targets, dtype=dtype),
            candidate_id=torch.tensor(ids, dtype=torch.long),
            link_split=name,
        )
        if passes is not None:
            batch.link_pass = passes  # the training pass (epoch) that drew its negatives and order
            batch.link_seed = seed  # ... from this seed: checkpointed, so a resume cannot change it
            batch.link_followed = bool(followed)  # whether the pass was the epoch the training loop announced
        return batch

    def check_batch(self, batch: Any) -> None:
        """Refuse a batch that would leak or does not fit the split: message
        edges outside the training topology (a held-out positive is named);
        candidates outside the batch's split, with the wrong target or id; a
        barred self-loop; and a training negative that is a held-out
        (``val`` / ``test``) negative."""
        _check_structure(batch)
        split = self.split
        name = getattr(batch, "link_split", None)
        if not isinstance(name, str):
            raise LinkTaskError("a link batch names its split (link_split): build it with LinkTask.loader()")
        LinkSplit._split(name)
        x = getattr(batch, "x", None)
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] != split.num_nodes:
            raise LinkTaskError(f"a link batch carries the graph's {split.num_nodes} node rows")
        edge_index = getattr(batch, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise LinkTaskError("a link batch carries its message graph as a (2, edges) edge_index")
        if not split._is_message_graph(edge_index):
            allowed = split._message_set()
            messages = {(int(u), int(v)) for u, v in edge_index.t().tolist()}
            held_out = set() if split.label_existence == "visible" else {*split.val, *split.test}
            leaked = sorted((u, v) for u, v in messages - allowed if _canonical(u, v, split.directed) in held_out)
            if leaked:
                raise LinkTaskError(f"leak: held-out positive(s) {leaked[:5]} are in the message graph")
            if messages != allowed:
                extra, missing = sorted(messages - allowed)[:5], sorted(allowed - messages)[:5]
                raise LinkTaskError(
                    f"the message graph must be the split's training topology (unexpected {extra}, missing {missing})"
                )
        pairs = [(int(u), int(v)) for u, v in batch.edge_label_index.t().tolist()]
        targets = batch.edge_label.tolist()
        given = batch.candidate_id.tolist()
        ids = split._cached(f"ids:{name}", lambda: split.candidate_ids(name))
        positives = split._cached(f"positives:{name}", lambda: frozenset(split.positives(name)))
        for u, v in pairs:
            if not (0 <= u < split.num_nodes and 0 <= v < split.num_nodes):
                raise LinkTaskError(f"candidate {(u, v)} is outside the graph's {split.num_nodes} nodes")
            if u == v and split.self_loops == "bar":
                raise LinkTaskError(f"candidate {(u, v)} is a self-loop, barred by the split")
        if self.mode == "edge_label":
            labels = split._cached("labels", split.labels)
            for edge, target, given_id in zip(pairs, targets, given, strict=True):
                if edge not in positives:
                    raise LinkTaskError(f"candidate {edge} is not a {name!r} edge")
                if int(target) != labels[edge]:
                    raise LinkTaskError(f"candidate {edge}'s target {target} is not its category {labels[edge]}")
                if given_id != ids[edge]:
                    raise LinkTaskError(f"candidate {edge} carries id {given_id}, not its id {ids[edge]}")
            return
        fixed = split._cached(f"negatives:{name}", lambda: frozenset(split.negatives(name)))
        held_out = split._cached("held_out_negatives", lambda: frozenset((*split.val_negatives, *split.test_negatives)))
        others = split._all_positives()
        for edge, target, given_id in zip(pairs, targets, given, strict=True):
            if target not in (0, 1, 0.0, 1.0):
                raise LinkTaskError(f"binary targets are 0 / 1, got {target}")
            if target == 1 and edge not in positives:
                raise LinkTaskError(f"candidate {edge} is labelled a {name!r} positive but is not one")
            if target == 0:
                canonical = _canonical(*edge, split.directed)
                if canonical in others:
                    raise LinkTaskError(f"candidate {edge} is a positive edge labelled negative")
                if name == "train" and canonical in held_out:
                    raise LinkTaskError(f"training negative {edge} is a held-out (val / test) negative")
                if name != "train" and edge not in fixed:
                    raise LinkTaskError(f"candidate {edge} is not one of {name!r}'s fixed negatives")
            expected = -1 if name == "train" and target == 0 else ids[edge]
            if given_id != expected:
                raise LinkTaskError(f"candidate {edge} carries id {given_id}, not its id {expected}")

    def _logits(self, model: Any, batch: Any) -> torch.Tensor:
        from .nn.nn_model import _to_device

        device = model.device
        out = model.net(
            _to_device(batch.x, device),
            _to_device(batch.edge_index, device),
            _to_device(batch.edge_label_index, device),
        )
        k = int(batch.edge_label_index.shape[1])
        expected = (k,) if self.mode == "binary" else (k, len(self.split.categories or ()))
        if not isinstance(out, torch.Tensor) or tuple(out.shape) != expected:
            got = tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
            raise LinkTaskError(f"the model gave {got} for {k} candidates; a {self.mode} task needs {expected}")
        return out

    # ---------- training, evaluation, prediction ----------

    def objective(self, *, nonfinite: str = "fail") -> LinkObjective:
        return LinkObjective(self, nonfinite=nonfinite)

    def eval_step(self) -> LinkEval:
        return LinkEval(self)

    def metric_specs(self) -> list[Any]:
        """``MetricSpec``\\ s for ``MonitorSpec("auroc")`` / ``("ap")``
        (binary) — a one-class set leaves them unavailable, so it never
        wins BEST."""
        from .monitors import MetricSpec

        _register_metrics()
        if self.mode != "binary":
            return []
        return [MetricSpec("link.auroc", name="auroc"), MetricSpec("link.ap", name="ap")]

    def predict(self, model: Any, batches: Any) -> LinkPrediction:
        """Every candidate's prediction, in batch order, aligned with its id,
        pair and target (eval mode, no gradients, modes restored)."""
        from .utils import _capture_training_modes, _restore_training_modes

        modes = _capture_training_modes(model.net)
        ids, pairs, logits, targets = [], [], [], []
        names: set[str] = set()
        try:
            model.net.eval()
            with torch.no_grad():
                for batch in batches:
                    self.check_batch(batch)
                    names.add(batch.link_split)
                    if len(names) > 1:
                        raise LinkTaskError(f"a prediction covers one split, got {sorted(names)}")
                    logits.append(self._logits(model, batch).detach().float().cpu())
                    ids.append(batch.candidate_id.cpu())
                    pairs.append(batch.edge_label_index.t().cpu())
                    targets.append(batch.edge_label.cpu())
                    if sum(t.shape[0] for t in ids) > self.max_candidates:
                        raise LinkTaskError(f"more than max_candidates={self.max_candidates} candidates")
        finally:
            _restore_training_modes(modes)
        if not logits:
            raise LinkTaskError("no link batches to predict")
        z = torch.cat(logits)
        probabilities = torch.sigmoid(z) if self.mode == "binary" else torch.softmax(z, dim=-1)
        return LinkPrediction(
            ids=torch.cat(ids).numpy().astype(np.int64),
            pairs=torch.cat(pairs).numpy().astype(np.int64),
            logits=z.numpy(),
            probabilities=probabilities.numpy(),
            targets=torch.cat(targets).numpy(),
            split=names.pop(),
        )


def _no_extra_metrics(ctx: Any) -> None:
    if getattr(ctx, "extra_metrics", None):
        raise LinkTaskError("extra_metrics do not apply to link tasks; their records carry the link metrics")


class LinkObjective(Objective):
    """The task's training objective: binary cross-entropy (or categorical
    cross-entropy) summed over a batch's candidates, over the candidate
    count, after :meth:`LinkTask.check_batch`. Checkpointed as component
    ``"link.task"`` with the manifest (topology policy, candidate ids, fixed
    negatives)."""

    def __init__(self, task: LinkTask, *, nonfinite: str = "fail") -> None:
        super().__init__(nonfinite=nonfinite)
        if not isinstance(task, LinkTask):
            raise LinkTaskError(f"a LinkTask is needed, got {type(task).__name__}")
        self.task = task
        self.seed: Optional[int] = None  # the training negatives' seed, checkpointed
        # (epoch, training pass, whether the pass followed the epoch) of each
        # epoch's first batch, checkpointed
        self.last: Optional[tuple[int, int, bool]] = None
        self._expect: Optional[int] = None
        self._expect_last: Optional[tuple[int, int, bool]] = None

    def __call__(self, ctx: ObjectiveContext) -> ObjectiveResult:
        _no_extra_metrics(ctx)
        batch = ctx.batch
        self.task.check_batch(batch)  # before any forward pass: a leak never trains
        if batch.link_split != "train":
            raise LinkTaskError(f"the link objective trains on 'train' batches only, got a {batch.link_split!r} batch")
        seed, drawn = getattr(batch, "link_seed", None), getattr(batch, "link_pass", None)
        followed = bool(getattr(batch, "link_followed", False))
        expect, self._expect = self._expect, None  # checked once, by the first batch after a restore
        last, self._expect_last = self._expect_last, None
        if last is not None and ctx.epoch_idx != last[0] + 1:
            expect = last = None  # left armed by a resume that failed before training: this is another fit
        if seed is not None and expect is not None and seed != expect:
            raise LinkTaskError(
                f"the checkpoint drew training negatives from seed {expect}, the resumed loader from "
                f"seed {seed}: build the training loader with the same seed"
            )
        if drawn is not None and last is not None:
            epoch, previous, was_followed = last
            # An epoch-following loader keeps following, at the same offset;
            # a materialised list (which never followed) repeats its pass.
            consistent = (
                followed and drawn - ctx.epoch_idx == previous - epoch
                if was_followed
                else not followed and drawn == previous
            )
            if not consistent:
                raise LinkTaskError(
                    f"the resumed run draws training pass {drawn} at epoch {ctx.epoch_idx}"
                    f"{'' if followed else ' (not following the epoch)'}, but the checkpoint drew pass {previous} at "
                    f"epoch {epoch}{' following the epoch' if was_followed else ''}: forward set_epoch(epoch) to the "
                    "link loader through any wrapper, or resume with the same materialised batches"
                )
        if seed is not None:
            self.seed = int(seed)
        if drawn is not None and ctx.batch_idx == 0:
            self.last = (int(ctx.epoch_idx), int(drawn), followed)
        model = ctx.model
        model.net.train()
        logits = self.task._logits(model, batch)
        target = batch.edge_label.to(logits.device)
        if self.task.mode == "binary":
            numerator = nn.functional.binary_cross_entropy_with_logits(logits, target.float(), reduction="sum")
        else:
            numerator = nn.functional.cross_entropy(logits, target.long(), reduction="sum")
        term = LossTerm("link_ce", numerator, int(target.shape[0]))
        from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

        record = (
            NNEvaluationDataPoint(
                loss=term.value, kind=KIND, count=int(target.shape[0]), status="ok" if target.shape[0] else "empty"
            )
            if target.shape[0]
            else NNEvaluationDataPoint(kind=KIND, count=0, status="empty")
        )
        return ObjectiveResult((term,), record)

    def component_spec(self) -> ComponentSpec:
        return ComponentSpec("link.task", version=1)

    def component_state(self) -> dict[str, Any]:
        return {
            "task": self.task.state(),
            "manifest": self.task.split.state(),
            "negatives_seed": self.seed,
            "last_pass": None if self.last is None else list(self.last),
        }

    @staticmethod
    def _seed_state(state: Mapping[str, Any]) -> Any:
        seed = state.get("negatives_seed")
        if seed is None or (isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0):
            return seed
        raise ValueError(f"negatives_seed {seed!r}")

    @staticmethod
    def _last_state(state: Mapping[str, Any]) -> Any:
        last = state.get("last_pass")
        if last is None:
            return None
        if (
            isinstance(last, (list, tuple))
            and len(last) == 3
            and all(type(v) is int and v >= 0 for v in last[:2])
            and type(last[2]) is bool
        ):
            return (int(last[0]), int(last[1]), bool(last[2]))
        raise ValueError(f"last_pass {last!r}")

    def check_component_state(self, state: Mapping[str, Any], *, version: int) -> list[str]:
        if not isinstance(state, Mapping):
            return ["the link task state is not a mapping"]
        problems = []
        if state.get("manifest") != self.task.split.state():
            problems.append("the link split (manifest) changed since the checkpoint")
        if state.get("task") != self.task.state():
            problems.append(f"the link task changed since the checkpoint: {state.get('task')} -> {self.task.state()}")
        for read in (self._seed_state, self._last_state):
            try:
                read(state)
            except ValueError as error:
                problems.append(f"the link task state holds a malformed {error}")
        return problems

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None:
        """The manifest is configuration (checked, not restored). The
        training negatives' seed is restored and must match the resumed
        loader's, and the first resumed batch must continue the passes —
        an epoch-following loader keeps following at its offset from the
        epoch, a materialised list repeats its one pass — or the resume is
        refused. (A loader that never followed the epoch looks like a list:
        a source run through a non-forwarding wrapper resumes unchecked.)"""
        # Assigned whatever they hold: a rollback restores None, which disarms.
        self.seed = self._expect = self._seed_state(state)
        self.last = self._expect_last = self._last_state(state)


class LinkEval:
    """The task's ``eval_step_fn``: every validation candidate materialised
    (at most ``max_candidates``), then exact metrics over the whole set —
    never per-batch averages. The record (``kind="link"``) counts the
    candidates; its ``loss`` is the BCE (or NLL)."""

    def __init__(self, task: LinkTask) -> None:
        if not isinstance(task, LinkTask):
            raise LinkTaskError(f"a LinkTask is needed, got {type(task).__name__}")
        self.task = task

    def __call__(self, ctx: Any) -> Any:
        from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

        _no_extra_metrics(ctx)
        prediction = self.task.predict(ctx.model, ctx.val_loader)
        if prediction.split != "val":
            raise LinkTaskError(
                f"the training evaluator reads the 'val' split, got {prediction.split!r} batches: selecting on "
                "'train' or 'test' would leak; score 'test' with task.predict() and link_metrics() after training"
            )
        expected = self.task.split.candidate_ids(prediction.split)
        seen = prediction.ids.tolist()
        if len(seen) != len(expected) or set(seen) != set(expected.values()):
            raise LinkTaskError(
                f"evaluation materialises every {prediction.split!r} candidate exactly once: got {len(seen)} rows "
                f"({len(set(seen))} distinct) for {len(expected)} candidates"
            )
        metrics = link_metrics(prediction.probabilities, prediction.targets)
        k = int(prediction.ids.shape[0])
        values = {name: m.value for name, m in metrics.items() if m.value is not None}
        if self.task.mode == "binary":
            values["positives"] = float((prediction.targets == 1).sum())
            values["negatives"] = float((prediction.targets == 0).sum())
        loss = values.get("bce", values.get("nll"))
        return NNEvaluationDataPoint(loss=loss, kind=KIND, count=k, status="ok", metrics=values)


class _Unavailable:
    def update(self, target: Any, prediction: Any) -> None:
        raise LinkTaskError("link.auroc / link.ap are computed by LinkTask.eval_step() over the whole candidate set")

    def result(self) -> Optional[float]:
        return None


def _no_config(config: Mapping[str, Any]) -> None:
    if config:
        raise ValueError(f"link metrics take no options, got {sorted(config)}")


def _register_metrics() -> None:
    from .monitors import register_metric, registered_metrics

    known = set(registered_metrics())
    for metric in ("link.auroc", "link.ap"):
        if (metric, 1) not in known:
            register_metric(
                metric, 1, lambda config: _Unavailable(), input="probabilities", mode="max", check_config=_no_config
            )


def _register() -> None:
    from .models import register_model_factory, registered_model_factories

    if (FACTORY_ID, 1) not in registered_model_factories():
        register_model_factory(FACTORY_ID, 1, _build)
    _register_metrics()


_register()
