---
id: FEAT-107
title: "[FEAT-107] Build to_onnx's and netron_export's export arguments in one place"
labels: ["enhancement", "area: interoperability", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-29"]
depends_on: ["FIX-108"]
related_issues: [556]
status: READY
---

**Category:** Interoperability · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-107: `NNModel.to_onnx` makes three near-identical `torch.onnx.export` calls and `viz.netron_export` two more; both repeat the "does `torch.onnx.export` take `dynamo=`" check and copy the example-input placement (dtype cast before the device move), which has drifted once already (#556). This ticket adds `src/nnx/_onnx.py` with `export_accepts_dynamo`, `export_kwargs` and the shared placement helper `example_inputs`, makes one export call in each function, separates two interleaved comments in `to_onnx`, and says in `netron_export`'s docstring what `launch=True` starts. The exported files do not change.

## Context

**What happens today.** `to_onnx` (`nn_model.py` L4352–L4391) branches three ways; the first and last branches:

```python
                export_accepts_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
                if dynamo:
                    if not export_accepts_dynamo:
                        raise RuntimeError(
                            "to_onnx(dynamo=True) requires torch>=2.5 (the dynamo-based "
                            "ONNX exporter wasn't available before then). Upgrade torch or "
                            "call with dynamo=False."
                        )
```

```python
                else:
                    torch.onnx.export(
                        self.net,
                        example_input,
                        partial,
                        input_names=in_names,
                        output_names=out_names,
                        dynamic_axes=dynamic_axes,
                        opset_version=opset_version,
                    )
```

The middle branch adds `dynamo=False`; the first passes `dynamic_shapes=` and `dynamo=True` instead of `dynamic_axes=`. `netron_export` (`viz/netron.py` L130–L151) repeats the check and the last two branches with `path` as the target. The placement: `to_onnx`'s `_example` (L4302–L4310) and `netron_export`'s `_placed` (L94–L105) both convert an array through `_writable`, cast floating inputs to the net's floating dtype, then move to the device.

**Why it matters.** Five export calls and two placement helpers drift: #556 had to fix the cast-then-move order in one copy only.

**History.** #556 (FIX-088): "On MPS (Apple Metal Performance Shaders), float64 inputs to `netron_export` ... are moved before the dtype cast, and the move raises `TypeError`." Its tests (`tests/test_mps_float64_paths.py` L147, L162) must keep passing on MPS hardware. FIX-107 and FIX-108 change what `to_onnx` writes and checks; this ticket comes after them and only restructures the call.

**Background for the worker.**
- `torch.onnx.export(model, args, f, **kwargs)`: the exporter; torch>=2.5 takes `dynamo=` (False: the TorchScript exporter, which needs `onnx`; True: the `torch.export`-based one, which needs `onnxscript`). torch>=2.9 defaults to `dynamo=True`, so NNx always passes it when accepted.
- `dynamic_axes` (TorchScript exporter: `{name: {dim: "batch"}}`) vs `dynamic_shapes` (dynamo exporter: one `{dim: torch.export.Dim}` per input).
- Tests replace `torch.onnx.export` with fakes (`tests/test_to_onnx_inputs.py` L219, `tests/test_onnx_dynamo.py` L124), with and without a `dynamo` parameter. So the check must read `torch.onnx.export` at call time, and the call must go through the attribute `torch.onnx.export`, never a name imported once.
- `_writable(array)` (`utils.py`): a writable copy of a read-only NumPy array, so `torch.from_numpy` does not warn (`tests/test_external_module_runtime.py` L1702 spies on `torch.from_numpy`).
- After FIX-108, `to_onnx` reserves `staging, staged = _reserve_export_dir(path)` (replacing `partial = _reserve_partial(path)` and its comment), and the third argument of every export call is `staged`. This ticket uses `staged`.

**Proposal.**
1. Create `src/nnx/_onnx.py`:

   ```python
   """``torch.onnx.export`` arguments shared by ``NNModel.to_onnx`` and
   ``nnx.viz.netron_export``. Internal."""

   from __future__ import annotations

   import inspect
   from collections.abc import Sequence
   from typing import Any, Optional

   import numpy as np
   import torch

   from .utils import _writable


   def export_accepts_dynamo() -> bool:
       """Whether this torch's ``torch.onnx.export`` takes ``dynamo=``
       (torch>=2.5). Read from the attribute at each call."""
       return "dynamo" in inspect.signature(torch.onnx.export).parameters


   def export_kwargs(
       *,
       input_names: Sequence[str],
       output_names: Sequence[str],
       opset_version: int,
       dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
       dynamic_shapes: Optional[tuple[dict[int, Any], ...]] = None,
       dynamo: bool = False,
   ) -> dict[str, Any]:
       """The keyword arguments of one ``torch.onnx.export`` call: the dynamo
       exporter takes ``dynamic_shapes``, the TorchScript one ``dynamic_axes``,
       and ``dynamo=`` is passed whenever this torch accepts it, so the
       TorchScript exporter stays the default on torch>=2.9."""
       kwargs: dict[str, Any] = {"input_names": input_names, "output_names": output_names, "opset_version": opset_version}
       if dynamo:
           kwargs["dynamic_shapes"] = dynamic_shapes
       else:
           kwargs["dynamic_axes"] = dynamic_axes
       if export_accepts_dynamo():
           kwargs["dynamo"] = dynamo
       return kwargs


   def example_inputs(example_input: Any, *, net: torch.nn.Module, device: Any) -> tuple[torch.Tensor, ...]:
       """``example_input`` as the tuple of tensors an export traces. A single
       tensor or NumPy array is one input (a 2-D array is never unpacked row by
       row); an array becomes a tensor through a writable copy; a floating input
       takes the net's floating dtype, cast before the move to ``device``, since
       MPS has no float64 (#556)."""
       if isinstance(example_input, (torch.Tensor, np.ndarray)):
           example_input = (example_input,)
       net_dtype = next((param.dtype for param in net.parameters() if param.is_floating_point()), None)

       def placed(value: Any) -> torch.Tensor:
           tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(_writable(np.asarray(value)))
           if net_dtype is not None and tensor.is_floating_point() and tensor.dtype != net_dtype:
               tensor = tensor.to(net_dtype)
           return tensor.to(device)

       return tuple(placed(e) for e in example_input)
   ```

2. In `src/nnx/nn/nn_model.py`, `NNModel.to_onnx`:
   - add `from .._onnx import example_inputs, export_accepts_dynamo, export_kwargs` to the top-level first-party imports;
   - replace the placement (L4292–L4310, from `# Normalize a single Tensor / np.ndarray ...` to `example_input = tuple(_example(e) for e in example_input)`) with `example_input = example_inputs(example_input, net=self.net, device=self.device)`;
   - replace the body of the inner `try:` (L4353–L4391, from `export_accepts_dynamo = ...` to the end of the third `torch.onnx.export(...)` call) with:

     ```python
                 if dynamo and not export_accepts_dynamo():
                     raise RuntimeError(
                         "to_onnx(dynamo=True) requires torch>=2.5 (the dynamo-based "
                         "ONNX exporter wasn't available before then). Upgrade torch or "
                         "call with dynamo=False."
                     )
                 torch.onnx.export(
                     self.net,
                     example_input,
                     staged,
                     **export_kwargs(
                         input_names=in_names,
                         output_names=out_names,
                         opset_version=opset_version,
                         dynamic_axes=dynamic_axes,
                         dynamic_shapes=dynamic_shapes,
                         dynamo=dynamo,
                     ),
                 )
     ```

     The `finally: _restore_training_modes(training_modes)` stays;
   - separate the comments: move the paragraph from `# Snapshot training mode so the train → to_onnx → train-more pattern` to `# exception.` (L4333–L4340) so it sits directly above `training_modes = _capture_training_modes(self.net)` (L4345). The staging paragraph FIX-108 wrote (`# The export is staged in a new hidden directory beside ``path`` ...`) stays directly above `staging, staged = _reserve_export_dir(path)`.
3. In `src/nnx/viz/netron.py`, `netron_export`:
   - change `from ..utils import _capture_training_modes, _restore_training_modes, _writable` (L29) to `from ..utils import _capture_training_modes, _restore_training_modes`, add `from .._onnx import example_inputs, export_kwargs`, and delete `import inspect` (L21);
   - replace L91–L105 (the `# As to_onnx / predict: ...` comment, `net_dtype`, `_placed` and the `if isinstance(...)` / `else` block) with `example_input = example_inputs(example_input, net=net, device=device)`;
   - replace the export block inside its `try:` (L130–L151) with:

     ```python
             torch.onnx.export(
                 net,
                 example_input,
                 path,
                 **export_kwargs(
                     input_names=in_names, output_names=out_names, opset_version=opset_version, dynamic_axes=dynamic_axes
                 ),
             )
     ```

     keeping the three comment lines above it (`# Mirror NNModel.to_onnx — pin to the legacy TorchScript path ...`, L127–L129);
   - in the docstring, replace the `launch:` entry (L52–L56) with:

     ```
             launch: When True, call `netron.start(path)`, which starts a local
                 HTTP server that serves the file and opens it in the default
                 web browser (Netron's viewer). Requires `pip install thekaveh-nnx[viz-interactive]`
                 (or `pip install netron`). Defaults to False so CI / tests
                 can exercise export without spawning a long-lived process.
     ```

4. Regenerate the API reference, which renders `netron_export`'s docstring: `uv run python -m scripts.docs.build_api_reference`.
5. Add `tests/test_onnx_export_args.py` (no torch export needed):

   ```python
   """The shared torch.onnx.export arguments (nnx._onnx)."""

   import numpy as np
   import torch

   from nnx._onnx import example_inputs, export_kwargs


   def test_export_kwargs_follow_the_exporter_signature(monkeypatch):
       def legacy(model, args, f, *, input_names=None, output_names=None, dynamic_axes=None, opset_version=None):
           raise AssertionError("not called")

       monkeypatch.setattr(torch.onnx, "export", legacy)
       axes = {"input_0": {0: "batch"}}
       assert export_kwargs(input_names=["input_0"], output_names=["output"], opset_version=17, dynamic_axes=axes) == {
           "input_names": ["input_0"], "output_names": ["output"], "opset_version": 17, "dynamic_axes": axes
       }

       def modern(model, args, f, *, dynamo=False, **kwargs):
           raise AssertionError("not called")

       monkeypatch.setattr(torch.onnx, "export", modern)
       assert export_kwargs(input_names=["x"], output_names=["y"], opset_version=17, dynamic_axes=None)["dynamo"] is False
       shapes = ({0: "dim"},)
       kwargs = export_kwargs(input_names=["x"], output_names=["y"], opset_version=17, dynamic_shapes=shapes, dynamo=True)
       assert kwargs["dynamo"] is True and kwargs["dynamic_shapes"] is shapes and "dynamic_axes" not in kwargs


   def test_example_inputs_cast_floats_before_the_move_and_keep_a_2d_array_whole():
       net = torch.nn.Linear(4, 2)
       (one,) = example_inputs(np.zeros((3, 4)), net=net, device="cpu")
       assert one.shape == (3, 4) and one.dtype == torch.float32
       tokens, floats = example_inputs((torch.tensor([1, 2]), np.ones(4)), net=net, device="cpu")
       assert tokens.dtype == torch.int64 and floats.dtype == torch.float32
   ```

6. No error text changes (the `RuntimeError` keeps its words), so there is no CHANGELOG entry.

**Worker setup.** Branch from `develop` after FIX-107 and FIX-108 have merged. `uv sync --frozen --all-extras` installs torch and every extra the tests need. The `#556` tests in `tests/test_mps_float64_paths.py` need Apple-silicon MPS; they skip elsewhere.

## Acceptance criteria

- [ ] The new tests pass. Check: `uv run pytest tests/test_onnx_export_args.py`
- [ ] One export call in each function, and the signature check in one place. Check: `grep -c "torch.onnx.export(" src/nnx/nn/nn_model.py src/nnx/viz/netron.py` prints `1` for each file, and `grep -rn "inspect.signature(torch.onnx.export)" src/nnx` returns 1 line (`_onnx.py`)
- [ ] One placement helper. Check: `grep -rn "def _placed\|def _example" src/nnx/nn/nn_model.py src/nnx/viz/netron.py` returns 0 lines
- [ ] The export suites pass, including the fake-exporter tests. Check: `uv run pytest tests/test_to_onnx_inputs.py tests/test_onnx_dynamo.py tests/test_to_onnx_dropped_state.py tests/test_viz_netron.py tests/test_external_module_runtime.py::test_onnx_exports_hand_torch_a_writable_array tests/test_quantize_qat.py::test_qat_converted_model_onnx_exports`
- [ ] #556's MPS tests pass on Apple silicon. Check: `NNX_REQUIRE_MPS=1 uv run pytest tests/test_mps_float64_paths.py -k netron_export` (on an MPS machine)
- [ ] The docs are regenerated. Check: `uv run python -m scripts.docs.build_api_reference --check` and `uv run python -m scripts.docs.build_docs --check`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [nn_model.py L4292-L4310](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4292-L4310) | `to_onnx`'s placement (`_example`) |
| [nn_model.py L4333-L4345](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4333-L4345) | the interleaved training-mode and partial-file comments |
| [nn_model.py L4351-L4393](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L4351-L4393) | the three `torch.onnx.export` branches |
| [netron.py L91-L105](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/netron.py#L91-L105) | `netron_export`'s placement (`_placed`) |
| [netron.py L126-L153](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/netron.py#L126-L153) | its two export branches |
| [netron.py L52-L56](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/viz/netron.py#L52-L56) | the `launch` docstring |
| [test_to_onnx_inputs.py L207-L227](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_to_onnx_inputs.py#L207-L227), [test_onnx_dynamo.py L117-L128](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_onnx_dynamo.py#L117-L128) | fake exporters without and with `dynamo` |
| [test_mps_float64_paths.py L147-L185](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_mps_float64_paths.py#L147-L185) | #556's MPS placement tests |
| #556 | the placement copies drifted once (MPS) |

## Scope

**In:** `nnx/_onnx.py`, the single export call in `to_onnx` and `netron_export`, the comment move, the `launch` docstring, one test file.

**Out:** what `to_onnx` checks and writes (FIX-107, FIX-108). `predict`'s input placement (`nn_model.py` L6275), which serves another path. The dynamo exporter's own options.

## Dependencies

**Blocked by:** FIX-108
**Blocks:** none
**Related:** #556
