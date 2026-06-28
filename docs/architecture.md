# Architecture diagram

NNx is organized around two public entry points (`NNModel` / `Trainer`), a single
extensibility hook (`train_step_fn` / `trainer_step_fn`), and content-addressed
persistence under `runs/<id>/`. The diagram below shows the eight-layer
top-to-bottom flow; see [Concepts §1](concepts.md#1-architecture) for the full
written breakdown.

![NNx architecture](assets/architecture.svg)

!!! tip "Interactive version"
    Open the [full interactive diagram](architecture.html){target="_blank" rel="noopener"}
    in a new tab — it renders the same flow with hover labels on every node plus a
    per-layer explanation card grid.
