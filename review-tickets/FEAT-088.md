---
id: FEAT-088
title: "[FEAT-088] Pin the checkpoint sidecar bytes before refactoring NNCheckpoint.save (part 1 of 3)"
labels: ["enhancement", "area: persistence", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-10"]
depends_on: []
related_issues: [536, 215, 206, 394]
status: READY
---

**Category:** Persistence · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-088: FEAT-089 and FEAT-090 will refactor how `NNModel.train` and `Trainer.train` hand their optimizer and scheduler state to `NNCheckpoint.save`. The training-state sidecar that `save()` writes is an on-disk format that resumes read, so its content and key order must not change. Add a test, and nothing else, that pins the sidecar's keys in write order and every value that is neither a tensor, an RNG stream nor a random id, for one `NNModel.train` run and one two-optimizer `Trainer` run. Part 1 of 3 of the SMELL-10 work.

## Context

**What happens today.** `NNCheckpoint.save` builds the sidecar as one dict, singular keys for `NNModel.train`'s one optimizer first, the Trainer's name-keyed plural keys later (`nn_checkpoint.py` L785–L829):

```python
            training_state = {
                "nnx_training_state_version": _TRAINING_STATE_FORMAT_VERSION,
                "checkpoint_id": generation,
                "optimizer": optimizer_state,
                "optimizer_type": optimizer_type,
                "optimizer_topology": optimizer_topology,
```

```python
                "components": components,
                "optimizers": optimizers_state,
                "optimizer_types": optimizer_types,
                "optimizer_topologies": optimizer_topologies,
```

It is written with `torch.save`, so the key order is part of the bytes.

**Why it matters.** The refactors that follow must not change which keys a sidecar has, their order, or their values; a pin catches any drift before it reaches a user's checkpoints.

**History.** #394 introduced the generation-stamped sidecar and #215 (FEAT-005) the name-keyed component and optimizer keys; #536 shares the save between the two loops; #206 keeps probing absent state from creating a run. The SMELL-10 work is split: this ticket (pin), FEAT-089 (one value for the training state), FEAT-090 (derive single vs named).

**Background for the worker.**
- *Training-state sidecar*: `<tag>.opt.<generation>.pt`, the `torch.save`d dict above; `NNCheckpoint.load_with_training_state(run=..., type=...)` returns `(checkpoint, state)` with `state` being that dict.
- *What can be pinned exactly*: `checkpoint_id` is a fresh `uuid4` per save, and the optimizer, scheduler, RNG, component and precision states hold tensors or platform-dependent numbers; for those the pin records only whether the value is `None` and either its type (ids, RNG streams, scheduler, scaler, precision and pre-transform model state, whose layout depends on the torch version or device) or its top-level keys (optimizer and component states). Every other value (versions, types, topologies, factories, configs, counters, epochs, the compile and distributed records) is pinned exactly.

**Proposal.**

1. Add `tests/test_checkpoint_sidecar_format.py`:
   ```python
   """FEAT-088: the training-state sidecar NNCheckpoint.save writes is an on-disk
   format. Its keys, their order and every value that is not a tensor, an RNG
   stream or a random id are pinned, for NNModel.train's one optimizer and a
   Trainer's named ones, so refactoring how the loops pass that state cannot
   change what a resume reads."""

   from __future__ import annotations

   import pprint

   import pytest
   import torch

   from nnx import (
       Activations, Checkpoints, Devices, Losses, Nets, NNCheckpoint, NNModel, NNModelParams, NNOptimParams,
       NNParamGroupSpec, NNParams, NNSchedulerParams, NNTrainerParams, NNTrainParams, Optims, Trainer,
       supervised_objective,
   )

   # Recorded by type only: random ids, RNG streams and states whose layout depends on the torch version or device.
   _TYPE_ONLY = {"checkpoint_id", "scheduler", "rng", "model", "scaler", "precision"}
   # Recorded by their top-level keys only: they hold tensors.
   _KEYS_ONLY = {"optimizer", "optimizers", "schedulers", "components"}
   _X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
   _Y = torch.tensor([0, 1, 1, 0])
   _LOADER = [(_X[:2], _Y[:2]), (_X[2:], _Y[2:])]


   def _model() -> NNModel:
       torch.manual_seed(0)
       return NNModel(
           net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
           params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
       )


   def _adam(pattern: str) -> NNOptimParams:
       return NNOptimParams(
           name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
           param_groups=[NNParamGroupSpec(name_pattern=pattern, lr=1e-3)],
       )


   def _sidecar_shape(state: dict) -> list:
       shape = []
       for key, value in state.items():
           if value is not None and key in _TYPE_ONLY:
               value = ("type", type(value).__name__)
           elif value is not None and key in _KEYS_ONLY:
               value = ("keys", sorted(value))
           shape.append((key, value))
       return shape


   def _model_sidecar() -> dict:
       run = _model().train(params=NNTrainParams(n_epochs=1, train_loader=_LOADER, save_phase_checkpoints=False))
       _, state = NNCheckpoint.load_with_training_state(run=run.id, type=Checkpoints.LAST)
       return state


   def _trainer_sidecar() -> dict:
       params = NNTrainerParams(
           n_epochs=1,
           train_loader=_LOADER,
           optims={"body": _adam("layers.0.*"), "head": _adam("layers.1.*")},
           schedulers={"head": NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3)},
           save_phase_checkpoints=False,
       )
       run = Trainer(_model()).train(params, objective=supervised_objective())
       _, state = NNCheckpoint.load_with_training_state(run=run.id, type=Checkpoints.LAST)
       return state


   @pytest.mark.parametrize(("build", "pinned"), [("model", "_PINNED_MODEL"), ("trainer", "_PINNED_TRAINER")])
   def test_the_training_state_sidecar_keeps_its_keys_order_and_values(build, pinned, tmp_path, monkeypatch):
       monkeypatch.chdir(tmp_path)
       state = _model_sidecar() if build == "model" else _trainer_sidecar()
       assert _sidecar_shape(state) == globals()[pinned], pprint.pformat(_sidecar_shape(state))
   ```
2. On `develop`, print the two current shapes and paste them as the module-level lists `_PINNED_MODEL = [...]` and `_PINNED_TRAINER = [...]` (pprint output verbatim):
   ```bash
   uv run python - <<'PY'
   import os, pprint, tempfile
   os.environ["NNX_TQDM_DISABLE"] = "1"
   os.chdir(tempfile.mkdtemp())
   from tests.test_checkpoint_sidecar_format import _model_sidecar, _sidecar_shape, _trainer_sidecar
   print("_PINNED_MODEL = ", end=""); pprint.pprint(_sidecar_shape(_model_sidecar()))
   print("_PINNED_TRAINER = ", end=""); pprint.pprint(_sidecar_shape(_trainer_sidecar()))
   PY
   ```
   Check the pasted model list starts with `("nnx_training_state_version", ...)`, `("checkpoint_id", ("type", "str"))`, and that `("optimizers", None)` appears in it and `("optimizer", None)` in the trainer list.
3. Change nothing in `src/`.

No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `tests/test_checkpoint_sidecar_format.py` passes on `develop` for both cases. Check: `uv run pytest tests/test_checkpoint_sidecar_format.py`
- [ ] The pin is not vacuous: changing the key order in `NNCheckpoint.save` (for example swapping the `"optimizer"` and `"optimizer_type"` lines locally) makes it fail; revert afterwards. Check: `uv run pytest tests/test_checkpoint_sidecar_format.py` fails with the swap and passes without it
- [ ] Nothing under `src/` changes. Check: `git diff --stat origin/develop -- src` prints nothing
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_checkpoint.py L684-L713](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L684-L713) | `NNCheckpoint.save`'s 30 keyword parameters |
| [nn_checkpoint.py L783-L829](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_checkpoint.py#L783-L829) | The sidecar dict and its key order |
| [nn_model.py L786-L803](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L786-L803) | `_Optimization.save_state`: singular keys for one optimizer, plural for a Trainer |
| [test_trainer.py L1036-L1051](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_trainer.py#L1036-L1051) | The two-optimizer `Trainer` configuration the pin reuses |
| #394, #215, #536, #206 | The sidecar format, named state, the shared save, absent-state probing |

## Scope

**In:** one new test file.

**Out:** any source change; the one-value training state (FEAT-089); deriving single vs named (FEAT-090).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-089, FEAT-074
**Related:** #536, #215, #206, #394
