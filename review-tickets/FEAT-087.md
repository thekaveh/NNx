---
id: FEAT-087
title: "[FEAT-087] Share validation and serialization between NNTrainParams and NNTrainerParams"
labels: ["enhancement", "area: configuration", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-09"]
depends_on: []
related_issues: [515]
status: READY
---

**Category:** Configuration · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-087: `NNTrainParams` (`nn/params/nn_train_params.py`) and `NNTrainerParams` (`trainer/params.py`) repeat about 60 lines word for word in `__post_init__`, `state()` and `from_state()`, differing only in the owner name in messages. First pin the exact `state()` of a few fixed objects; then move the shared parts into four module-level helpers in `nn_train_params.py` and call them from both classes. Every message, check order, `state()` key and run id stays the same.

## Context

**What happens today.** Both `__post_init__` methods run the same sequence: `n_epochs` (count ≥ 1), `seed` (0…2³²−1), `data_id`, plain-str lineage (#515), bool flags, the checkpoint tag, parent/resume exclusivity, `_validate_resume_mode`. For example `nn_train_params.py` L246–L251 and `trainer/params.py` L218–L223:

```python
        if self.data_id is not None and (not isinstance(self.data_id, str) or not self.data_id.strip()):
            raise ValueError(f"NNTrainParams.data_id must be a non-empty string when provided, got {self.data_id!r}")
        # Identity and lineage strings are stored as plain str: a numpy.str_
        # failed the first run.yaml write after an epoch trained (#515).
        object.__setattr__(self, "data_id", plain_str(self.data_id))
        object.__setattr__(self, "resume_from_run_id", plain_str(self.resume_from_run_id))
```

```python
        if self.data_id is not None and (not isinstance(self.data_id, str) or not self.data_id.strip()):
            raise ValueError(f"NNTrainerParams.data_id must be a non-empty string when provided, got {self.data_id!r}")
        # Plain str identity and lineage, as on NNTrainParams (#515).
        object.__setattr__(self, "data_id", plain_str(self.data_id))
        object.__setattr__(self, "resume_from_run_id", plain_str(self.resume_from_run_id))
```

Both `state()` methods emit `seed`, `data_id`, the lineage pair, `save_phase_checkpoints`, `metrics` and `monitor` only when not default, and both `from_state()` methods rebuild those seven keywords the same way.

**Why it matters.** A rule fixed in one class can be missed in the other. The omit-when-default keys feed the run id (`yaml.safe_dump(state(), sort_keys=True)`), so they must stay exactly the same.

**History.** #515 (FIX-048) normalized NumPy scalars and strings in params so `run.yaml` saves after training; the normalization moves into the shared helper unchanged.

**Background for the worker.**
- *Omit-when-default*: a params field appears in `state()` only when it differs from its default, so a run created before the field existed keeps its run id.
- *Run id*: a digest of the run's `state()`; any change in the keys or values `state()` emits shifts every existing run's id.
- *Owner name*: the class name that messages embed (`NNTrainParams.data_id must …`); tests match these texts.
- *Check order*: when several fields are invalid, the first failing check decides the message, so the helpers keep today's order.

**Proposal.**

1. **Pin first.** On `develop`, before any refactor, add to `tests/test_params_round_trip.py`:
   ```python
   def _shared_params_cases():
       from nnx.monitors import MetricSpec, MonitorSpec
       from nnx.trainer import NNTrainerParams

       optim, scheduler = NNTrainParams(n_epochs=1).optim, NNTrainParams(n_epochs=1).scheduler
       metrics = (MetricSpec("accuracy", name="acc"),)
       monitor = MonitorSpec(metric="acc", split="val", mode="max", on_missing="error")
       return {
           "train_minimal": NNTrainParams(n_epochs=2),
           "train_full": NNTrainParams(
               n_epochs=3, seed=7, data_id="d1", parent_run_id="abc", save_phase_checkpoints=False,
               resume_epochs="planned", metrics=metrics, monitor=monitor,
           ),
           "train_resume": NNTrainParams(n_epochs=1, resume_from_run_id="abc"),
           "trainer_minimal": NNTrainerParams(n_epochs=2, optims={"main": optim}),
           "trainer_full": NNTrainerParams(
               n_epochs=4, optims={"b": optim, "a": optim}, schedulers={"a": scheduler}, seed=0, data_id="d2",
               parent_run_id="abc", save_phase_checkpoints=False, auto_step_schedulers=False,
               metrics=metrics, monitor=monitor,
           ),
       }


   def test_the_shared_params_state_is_pinned_and_round_trips():
       """FEAT-087: refactoring the shared validation and serialization keeps
       every state() (and so every run id) byte for byte."""
       import yaml

       for name, params in _shared_params_cases().items():
           assert params.state() == _PINNED_STATES[name], name
           assert type(params).from_state(params.state()).state() == _PINNED_STATES[name], name
           assert yaml.safe_dump(params.state(), sort_keys=True) == yaml.safe_dump(_PINNED_STATES[name], sort_keys=True)
   ```
   Print the current values with `uv run python -c "from tests.test_params_round_trip import _shared_params_cases as c; import pprint; pprint.pprint({k: v.state() for k, v in c().items()})"` and paste them as the module-level dict `_PINNED_STATES = {...}` (the pprint output verbatim). Commit this test on its own first; it passes on `develop` and must pass unchanged after the refactor.
2. In `src/nnx/nn/params/nn_train_params.py`, add after `_validate_monitoring` (L109–L121, before `class NNTrainParams`):
   ```python
   def _validate_common(params: Any, owner: str, flags: tuple[str, ...]) -> None:
       """The checks NNTrainParams and NNTrainerParams share, in this order:
       ``n_epochs``, ``seed``, ``data_id``, plain-str lineage (#515), the bool
       ``flags``, the checkpoint tag, parent/resume exclusivity and the resume
       mode. Normalized values are written back on the frozen ``params``;
       messages name ``owner``."""
       object.__setattr__(
           params,
           "n_epochs",
           require_count(
               params.n_epochs,
               "n_epochs",
               owner=owner,
               minimum=1,
               domain_message=f"{owner} requires n_epochs >= 1, got {params.n_epochs}",
           ),
       )
       if params.seed is not None:
           object.__setattr__(
               params, "seed", require_count(params.seed, "seed", owner=owner, minimum=0, maximum=2**32 - 1)
           )
       if params.data_id is not None and (not isinstance(params.data_id, str) or not params.data_id.strip()):
           raise ValueError(f"{owner}.data_id must be a non-empty string when provided, got {params.data_id!r}")
       # Identity and lineage strings are stored as plain str: a numpy.str_
       # failed the first run.yaml write after an epoch trained (#515).
       object.__setattr__(params, "data_id", plain_str(params.data_id))
       object.__setattr__(params, "resume_from_run_id", plain_str(params.resume_from_run_id))
       object.__setattr__(params, "parent_run_id", _parent_run_id(params.parent_run_id, owner))
       # Flags are real bools: a truthy string would act as True while
       # run.yaml recorded the string (and shifted the run id).
       for flag in flags:
           if not isinstance(getattr(params, flag), bool):
               raise ValueError(f"{owner}.{flag} must be a bool, got {getattr(params, flag)!r}")
       object.__setattr__(
           params,
           "resume_from_checkpoint",
           _checkpoint_tag(params.resume_from_checkpoint, owner, resuming=params.resume_from_run_id is not None),
       )
       if params.parent_run_id is not None and params.resume_from_run_id is not None:
           raise ValueError("set resume_from_run_id or parent_run_id, not both")
       _validate_resume_mode(params.resume_mode, owner)


   def _common_state(params: Any, d: dict[str, object]) -> None:
       """Add to ``d``, in this order, the omit-when-default keys both params
       classes emit: ``seed``, ``data_id``, the lineage pair and
       ``save_phase_checkpoints``. A key appears only when its value differs
       from the default, so existing run ids never shift."""
       if params.seed is not None:
           d["seed"] = params.seed
       if params.data_id is not None:
           d["data_id"] = params.data_id
       lineage = params.parent_run_id or params.resume_from_run_id
       if lineage is not None:
           d["parent_run_id"] = lineage
           d["parent_checkpoint"] = params.resume_from_checkpoint
       if params.save_phase_checkpoints is not True:
           d["save_phase_checkpoints"] = params.save_phase_checkpoints


   def _monitoring_state(params: Any, d: dict[str, object]) -> None:
       """Add ``metrics`` and ``monitor`` to ``d`` when declared (both classes, last)."""
       if params.metrics:
           d["metrics"] = [spec.state() for spec in params.metrics]
       if params.monitor is not None:
           d["monitor"] = params.monitor.state()


   def _common_from_state_kwargs(state: Mapping[str, Any]) -> dict[str, Any]:
       """The constructor keywords both classes rebuild from ``state()`` alike.
       A saved lineage the constructor now refuses is passed as ``None``; the
       caller restores it with ``_restore_saved_lineage(params, state.get("parent_run_id"))``."""
       lineage = state.get("parent_run_id")
       return dict(
           seed=state.get("seed"),
           data_id=state.get("data_id"),
           parent_run_id=lineage if lineage is None or _is_run_id_text(lineage) else None,
           resume_from_checkpoint=state.get("parent_checkpoint", "last"),
           save_phase_checkpoints=state.get("save_phase_checkpoints", True),
           metrics=tuple(MetricSpec.from_state(m) for m in state.get("metrics") or ()),
           monitor=MonitorSpec.from_state(state["monitor"]) if state.get("monitor") is not None else None,
       )
   ```
3. In `NNTrainParams` (same file):
   - `__post_init__` (L221–L272): keep the opening comment block and replace the code from the `n_epochs` check through `_validate_resume_mode(...)` with `        _validate_common(self, "NNTrainParams", ("save_phase_checkpoints", "overwrite_existing"))`; the three lines after it (`_validate_resume_epochs`, `plain_str(self.resume_epochs)`, `_validate_monitoring`) stay, in that order.
   - `state()` (L280–L307): after building `d` with `n_epochs`, `optim` and `scheduler`, call `_common_state(self, d)`, keep the `resume_epochs` lines, then call `_monitoring_state(self, d)` and `return d` (keep the omit-when-default comment above `_common_state`).
   - `from_state()` (L310–L330): build with `NNTrainParams(n_epochs=state["n_epochs"], optim=optim_params_from_state(state["optim"]), scheduler=NNSchedulerParams.from_state(state["scheduler"]), resume_epochs=state.get("resume_epochs", "additional"), **_common_from_state_kwargs(state))` and `return _restore_saved_lineage(params, state.get("parent_run_id"))`.
4. In `NNTrainerParams` (`src/nnx/trainer/params.py`):
   - change the import list (L28–L35) to `_common_from_state_kwargs, _common_state, _monitoring_state, _restore_saved_lineage, _validate_common, _validate_monitoring`;
   - `__post_init__`: keep everything before the `n_epochs` check (the mapping conversions, L162–L198) and replace the code from the `n_epochs` check through `_validate_resume_mode(...)` with `        _validate_common(self, "NNTrainerParams", ("save_phase_checkpoints", "auto_step_schedulers", "overwrite_existing"))`; `_validate_monitoring` and the `optims` / `schedulers` checks after it stay;
   - `state()` (L260–L288): after `n_epochs`, `optims` and the optional `schedulers`, call `_common_state(self, d)`, keep the `auto_step_schedulers` lines, then `_monitoring_state(self, d)`;
   - `from_state()` (L291–L308): `NNTrainerParams(n_epochs=state["n_epochs"], optims={...as today...}, schedulers={...as today...}, auto_step_schedulers=state.get("auto_step_schedulers", True), **_common_from_state_kwargs(state))`, then `return _restore_saved_lineage(params, state.get("parent_run_id"))`.
5. Delete imports ruff then reports unused in either file.

No behaviour changes, so no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] `test_the_shared_params_state_is_pinned_and_round_trips` passes on `develop` before the refactor and, unchanged, after it. Check: `uv run pytest tests/test_params_round_trip.py::test_the_shared_params_state_is_pinned_and_round_trips`
- [ ] The shared block exists once. Check: `grep -c "must be a non-empty string when provided" src/nnx/nn/params/nn_train_params.py src/nnx/trainer/params.py` prints `1` for `nn_train_params.py` and `0` for `trainer/params.py`
- [ ] Every params, validation and builder test passes with its message unchanged. Check: `uv run pytest tests -k "trainer_params or params_round_trip or params_boundary_validation or integer_config_validation or nn_trainer_params_builder or experiment_plan"`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0
- [ ] `uv run pyright --warnings` passes. Check: the command exits 0

## Evidence

| Location | What it shows |
|---|---|
| [nn_train_params.py L221-L272](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_train_params.py#L221-L272) | `NNTrainParams.__post_init__` |
| [nn_train_params.py L280-L330](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_train_params.py#L280-L330) | `state()` and `from_state()` |
| [nn_train_params.py L21-L120](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_train_params.py#L21-L120) | The helpers both classes already share (`_checkpoint_tag`, `_parent_run_id`, `_validate_monitoring`, …) |
| [params.py L162-L251](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/params.py#L162-L251) | `NNTrainerParams.__post_init__` |
| [params.py L260-L308](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/trainer/params.py#L260-L308) | `NNTrainerParams.state()` and `from_state()` |
| [test_trainer_params.py L300-L304](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_trainer_params.py#L300-L304) | The `MetricSpec` / `MonitorSpec` forms the pin test reuses |
| #515 | Plain-str normalization |

## Scope

**In:** the four helpers, their use in both classes, the pin test.

**Out:** the class-specific checks (`resume_epochs` on `NNTrainParams`; the `optims` / `schedulers` mapping checks on `NNTrainerParams`); any change to what `state()` emits.

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #515
