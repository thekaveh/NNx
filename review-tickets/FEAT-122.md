---
id: FEAT-122
title: "[FEAT-122] Reuse the shared RNG snapshot in lr_finder and viz.summary"
labels: ["enhancement", "area: numerics", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-44"]
depends_on: []
related_issues: [572, 555, 576]
status: READY
---

**Category:** Numerics · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-122: `lr_finder` and `nnx.viz.summary` each snapshot and restore the random streams by hand, while `nnx.seeding` already owns that logic (`_capture_rng_state`, `_restore_rng_state`, `_global_rng_kept`, `_loader_generators`). This ticket makes both call the shared helpers. One small behaviour change follows and gets a CHANGELOG line: a CUDA sweep now restores every CUDA device's stream, not only the swept device's.

## Context

**What happens today.** `lr_finder` (`src/nnx/lr_finder.py`):

```python
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    cpu_rng_state = torch.get_rng_state()
    device_rng_state: Optional[torch.Tensor] = None
    if device.type == "cuda":
        device_rng_state = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        device_rng_state = torch.mps.get_rng_state()
```

and in its `finally` the same streams are set back one by one, followed by the loader generators it found at four attributes. `_run_summary` (`src/nnx/viz/summary.py`) does the same for torch CPU plus the model device around `torchinfo.summary`:

```python
    cpu_rng_state = torch.get_rng_state()
    device_rng_state = None
    if dev.type == "cuda":
        device_rng_state = torch.cuda.get_rng_state(dev)
    elif dev.type == "mps":
        device_rng_state = torch.mps.get_rng_state()
```

**Why it matters.** The stream list (Python, NumPy, torch CPU, CUDA, MPS, the loader's generators) is decided in `nnx.seeding`; a new stream added there would be missed by these copies.

**History.** #572 (FEAT-050) pinned the MPS branches of both copies on CPU (`test_lr_finder_restores_the_mps_generator`, `test_a_summary_by_input_size_restores_an_mps_model_s_generator`). Both tests patch `torch.mps.get_rng_state` / `set_rng_state` / `torch.backends.mps.is_available`, which `nnx.seeding` looks up at call time, so they keep pinning the shared path without edits. #555 made a failed PEFT apply restore the streams through the same helpers. #576 touched `compilation`'s per-step snapshot; `compilation._Snapshot` is a hot path and stays out of scope.

**Background for the worker.**
- `_capture_rng_state(train_loader=None, *, cuda=True)` (`src/nnx/seeding.py` L102): returns the Python, NumPy and torch CPU states, every CUDA device's state (`torch.cuda.get_rng_state_all()`) when `cuda` is true and CUDA is available, and the MPS state when MPS is available. `cuda=False` never touches CUDA, so no CUDA context is created.
- `_restore_rng_state(state, train_loader=None)` (L134): sets them back (all CUDA devices with `set_rng_state_all`). With a loader it also restores the loader's generators, but it identifies several generators by their `initial_seed` and raises on two with the same seed (a resume check). Two default-constructed `torch.Generator()` objects share a seed, so `lr_finder` passes no loader and keeps restoring its generators itself.
- `_loader_generators(train_loader)` (L90): the four `(key, generator)` places a loader keeps a `torch.Generator` (`generator`, `sampler.generator`, `batch_sampler.generator`, `batch_sampler.sampler.generator`) — the same four `lr_finder` reads today.
- `_global_rng_kept()` (L198): a context manager that captures with `cuda=_cuda_in_use()` (CUDA is read only when a context already exists) and restores on exit, also on an exception.
- **What decides `cuda=` in `lr_finder`:** the sweep's device, `cuda=device.type == "cuda"` — the same condition that decides today whether a CUDA stream is read. `device` is the `device=` argument, or the model's first parameter's device. A CPU sweep never creates a CUDA context.
- **The behaviour change:** today a CUDA sweep restores `torch.cuda.get_rng_state(device)`, the swept device only; through `_capture_rng_state` it restores every CUDA device's stream. Also, on a host with MPS available, the MPS stream is now captured and restored whatever the device, and `summary` now also restores the Python and NumPy streams. A stream the call never drew from is set back to the value it already has, so only a sweep that moved another device's stream (a model spread over several GPUs) behaves differently, and that is the fix.

**Proposal.**
1. `src/nnx/lr_finder.py`:
   - Add `from .seeding import _capture_rng_state, _loader_generators, _restore_rng_state` to the module-top imports; delete `import random` (L20) and `import numpy as np` (L25), whose only uses are the lines replaced below.
   - In `lr_finder`, replace L159-L186 (from `python_rng_state = random.getstate()` through the `loader_gen_states = [...]` statement) with:

     ```python
         rng_state = _capture_rng_state(None, cuda=device.type == "cuda")
         # A loader built per the PyTorch reproducibility recipe draws its
         # shuffle permutation and worker base-seed from its OWN generator
         # (DataLoader(generator=...), an explicit sampler's or batch sampler's),
         # not global RNG — a stream the global restore can't cover. The loader
         # is not handed to _restore_rng_state: it matches several generators by
         # seed (a resume check), and two default generators share one.
         # isinstance filter: an exotic sampler may carry e.g. a numpy
         # Generator under the same attribute name — no get_state(), and not
         # a stream this helper can restore; it stays caller-owned.
         loader_gen_states = [
             (g, g.get_state())
             for g in dict.fromkeys(g for _, g in _loader_generators(train_loader) if isinstance(g, torch.Generator))
         ]
     ```

   - In the `finally` block, replace L270-L278 (from `random.setstate(python_rng_state)` through `torch.mps.set_rng_state(device_rng_state)`) with `_restore_rng_state(rng_state, None)`. Keep the loader-generator loop and the persistent-iterator cleanup after it unchanged.
   - In the module docstring (L8-L14), change `RNG stream the sweep consumes (global CPU, the active device's, and any` / `loader/sampler-attached torch ``generator=``)` to `RNG stream the sweep consumes (Python, NumPy and torch CPU, every CUDA` / `device's on a CUDA sweep, MPS, and any loader/sampler-attached torch` / ```generator=``)`, re-wrapping the paragraph.
2. `src/nnx/viz/summary.py`, `_run_summary`: add `from ..seeding import _global_rng_kept` to the module-top imports; delete L177-L182 (the `cpu_rng_state` / `device_rng_state` capture) and replace the `try: return _ti_summary(...) finally: ...` statement (L183-L201) with:

   ```python
       with _global_rng_kept():
           return _ti_summary(
               model,
               input_size=input_size,
               input_data=input_data,
               depth=depth,
               col_names=col_names,
               verbose=0,
               device=dev,
               dtypes=dtypes,
           )
   ```

   Keep the comment block above it (L158-L165).
3. Add to `CHANGELOG.md` under `## [Unreleased]` → `### Changed`:

   ```
   - **`lr_finder` and `nnx.viz.summary` restore the random streams through NNx's shared snapshot.** A CUDA `lr_finder` sweep now restores every CUDA device's stream (it restored only the swept device's), both restore the MPS stream wherever MPS is available, and `summary(input_size=...)` also restores the Python and NumPy streams. Streams the call never drew from come back unchanged, so a seeded pipeline still sees the same draws whether or not the call ran.
   ```

4. In `tests/test_lr_finder.py`, after `test_lr_finder_restores_the_mps_generator`, add:

   ```python
   def test_lr_finder_restores_every_cuda_device_stream(monkeypatch):
       """FEAT-122: a CUDA sweep restores every CUDA device's stream through
       nnx.seeding, not only the swept device's. A CPU pin: the batches stay on
       the CPU and the CUDA stream calls are recorded."""
       restored = []
       monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
       monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: ["streams before the sweep"])
       monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)

       class Stays:
           def __init__(self, tensor):
               self.tensor = tensor

           def to(self, device):
               assert device == torch.device("cuda")
               return self.tensor

       torch.manual_seed(0)
       data = TensorDataset(torch.randn(16, 4), torch.randint(0, 3, (16,)))
       loader = DataLoader(
           data, batch_size=8, collate_fn=lambda rows: tuple(Stays(torch.stack(c)) for c in zip(*rows, strict=True))
       )
       result = lr_finder(
           nn.Linear(4, 3), loader, loss_fn=nn.functional.cross_entropy, num_iter=2, device=torch.device("cuda")
       )
       assert len(result.losses) == 2 and restored == [["streams before the sweep"]]
   ```

   On `develop` it fails: the sweep calls the unpatched `torch.cuda.get_rng_state(device)`, which a CPU-only torch refuses.
5. Leave `test_lr_finder_restores_the_mps_generator` (L147), `test_lr_finder_restores_global_rng_state` (L502), `test_lr_finder_restores_loader_attached_generator_state` (L522) and `tests/test_viz_summary.py` unchanged; they must pass as they are.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need (plotly for `lr_finder`, torchinfo for `summary`).

## Acceptance criteria

- [ ] `tests/test_lr_finder.py::test_lr_finder_restores_every_cuda_device_stream` fails on `develop` before the change. Check: `uv run pytest tests/test_lr_finder.py::test_lr_finder_restores_every_cuda_device_stream`
- [ ] After the change it passes, together with the #572 MPS pins and the other restore tests, unedited. Check: `uv run pytest tests/test_lr_finder.py tests/test_viz_summary.py tests/test_seeding.py tests/test_core_install.py::test_lr_finder_refuses_before_touching_data_model_or_rng`
- [ ] Neither file reads a random stream by hand any more. Check: `grep -n "get_rng_state\|random.getstate\|np.random.get_state" src/nnx/lr_finder.py src/nnx/viz/summary.py` returns 0 lines
- [ ] The CHANGELOG entry is under `[Unreleased]` → `### Changed`. Check: `grep -n "restore the random streams through NNx's shared snapshot" CHANGELOG.md`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [lr_finder.py L159-L186](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/lr_finder.py#L159-L186) | `lr_finder`: the hand-written stream and loader-generator snapshot |
| [lr_finder.py L264-L278](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/lr_finder.py#L264-L278) | the hand-written restore; CUDA restores the swept device only |
| [summary.py L177-L201](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/summary.py#L177-L201) | `_run_summary`: torch CPU plus the model device |
| [seeding.py L90-L131](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/seeding.py#L90-L131) | `_loader_generators`, `_capture_rng_state` |
| [seeding.py L134-L184](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/seeding.py#L134-L184) | `_restore_rng_state`; the seed-matching that `lr_finder` must not use for its loader |
| [seeding.py L190-L206](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/seeding.py#L190-L206) | `_global_rng_state`, `_global_rng_kept` |
| [test_lr_finder.py L147-L176](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_lr_finder.py#L147-L176) | #572's MPS pin of `lr_finder` |
| [test_viz_summary.py L85-L115](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_viz_summary.py#L85-L115) | #572's MPS pin of `summary` |
| #572, #555, #576 | the MPS pins; the PEFT restore through the same helpers; compilation's snapshot |

## Scope

**In:** the snapshot and restore in `lr_finder` and `viz.summary`, the CHANGELOG line, one new test.

**Out:** `compilation._Snapshot` (a per-step hot path, #576). The `NNModel` unwrap and imports of `viz.summary` (FEAT-130).

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-130
**Related:** #572, #555, #576
