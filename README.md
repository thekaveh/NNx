# NNx

Lightweight PyTorch training/eval/visualization toolkit. Originally extracted from a personal ML lab (`thekaveh/ml`) where it powers training loops, checkpointing, and result visualization across multiple notebook-based ML tasks.

## What's inside

- **`nnx.nn.net`** — PyTorch model classes: `FeedFwdNN`, `GraphConvNN`, `GraphSAGENN`, `GraphAttNN`.
- **`nnx.nn.dataset`** — dataset abstractions for tabular / vision / graph data.
- **`nnx.nn.params`** — dataclass-based config (model params, training params, optimizer params, scheduler, checkpointing, run metadata).
- **`nnx.nn.enum`** — enums for activations, losses, optimizers, devices, network kinds.
- **`nnx.nn.nn_model.NNModel`** — orchestrator: builds the network from params, runs training/eval loops, manages checkpoints, supports early stopping.
- **`nnx.utils.Utils`** — pretty-printing helpers (`print_tree`, `print_table`, `flatten_dict`).
- **`nnx.vis_utils`** — Plotly-based visualization (convergence curves, t-SNE projections, confusion matrices).

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Minimal usage

```python
from nnx.nn.net.feed_fwd_nn import FeedFwdNN
# Construct with appropriate params; see consumer projects for full examples.
```

## Status

Alpha. API is stable for the existing consumer (`thekaveh/ml` notebooks). Tests cover importability and instantiation; integration is exercised through the consumer's notebooks. Bug reports welcome via GitHub issues.

## License

MIT. See [LICENSE](LICENSE).
