---
id: FEAT-123
title: "[FEAT-123] Map GGUF quantization labels once and read the FFN width from the model"
labels: ["enhancement", "area: interoperability", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-45", "DOC-31"]
depends_on: []
related_issues: [261, 525, 517, 472, 570, 575]
status: READY
---

**Category:** Interoperability · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-123: the GGUF writer maps a quantization label twice, through two if-chains (`_quantization_torch_dtype`, `_gguf_file_type_for`); it recomputes the feed-forward width from `ffn_mult` instead of reading it from the network it exports; and it imports `torch`, `map_tensors`, `json` and `warnings` inside functions, behind two comments that give false reasons. This ticket adds one `_QUANTIZATIONS` table, reads `feed_forward_length` from the first block's `ffn.w1`, and hoists those imports.

## Context

**What happens today.** In `src/nnx/interop/gguf/writer.py`:

```python
def _quantization_torch_dtype(quantization: str):
    ...
    import torch  # local import — torch is a hard dep, this is just to avoid top-level cycles in tests

    if quantization == "F32":
        return torch.float32
```

```python
    ft = gguf_mod.LlamaFileType
    if quantization == "F32":
        return int(ft.ALL_F32)
    if quantization == "F16":
        return int(ft.MOSTLY_F16)
```

and in `write_gguf`:

```python
    # Local import — avoids importing torch / numpy at module top so
    # ``import nnx.interop`` is cheap even when the writer isn't used.
    from .tensor_name_map import map_tensors
```

```python
    feed_forward_length = int(2 * params.ffn_mult * params.d_model / 3)
```

**Why it matters.** Two if-chains over the same three labels must be kept in step by hand. The width is the metadata a reader uses to rebuild shapes, so it should come from the tensors being written, not a copy of `SwiGLU.__init__`'s formula. Both import comments are false: `import nnx.interop` runs `nnx/__init__`, which imports torch, and nothing cycles (`tensor_name_map` imports `torch` and `nnx._reparameterization` only).

**History.** #525 (FIX-058) added `map_tensors`'s refusals (prune masks, legacy norms); they still run before anything is read from the network. #261 (FIX-026) pinned `RMSNorm`'s `eps == 1e-6` next to the writer's `add_layer_norm_rms_eps(1e-6)` (`tests/test_generative_nn_model.py` L387), so reading `eps` from the model is left out. #517 touched the same writer's neighbourhood (benchmark compiles) and changes nothing here. DOC-31's GGUF bullet (the two false comments, #472 / #570 / #575 doc sweeps) is done here because the comments sit on the import lines this ticket removes.

**Background for the worker.**
- `GGUFWriter` (the optional `gguf` package, extra `gguf-write`) picks a tensor's on-disk type from its NumPy dtype, so the writer casts each tensor to a torch dtype first; `general.file_type` is a `gguf.LlamaFileType` member's integer code (F32 = 0, F16 = 1, BF16 = 32).
- `SwiGLU` (`src/nnx/nn/net/transformer_layers.py` L140): `w1` / `w3` are `nn.Linear(d_model, hidden)` with `hidden = int(2 * ffn_mult * d_model / 3)`; `TransformerNN.blocks[i].ffn` is one.
- `map_tensors(transformer_nn)` refuses a PEFT wrapper or a factorized projection (anything but a plain `nn.Linear` for `w1`, …) with a `TypeError`. Reading `blocks.0.ffn.w1.out_features` after it is therefore always a plain `Linear` read. `NNTransformerParams` refuses `n_layers <= 0`, so `blocks.0` exists.
- For every network NNx builds, `w1.out_features` equals today's formula, so the metadata only changes for a network whose feed-forward layers were resized by hand — where today's value contradicts the stored tensors.

**Proposal.**
1. Module-top imports of `src/nnx/interop/gguf/writer.py`: add `import json`, `import warnings`, `import torch`, `from torch import nn`, `cast` to the `typing` import, and `from .tensor_name_map import map_tensors`. Leave the `TYPE_CHECKING` block and the `gguf` comment (L38-L41) as they are.
2. Replace `SUPPORTED_QUANTIZATIONS: tuple[str, ...] = ("F32", "F16", "BF16")` and the comment above it (L44-L46) with:

   ```python
   # Quantizations we can emit directly without shelling out, each with the
   # torch dtype its tensors are cast to (GGUFWriter picks the on-disk type from
   # the NumPy dtype) and the gguf.LlamaFileType member recorded as
   # general.file_type. Everything else requires the C++ llama-quantize binary.
   _QUANTIZATIONS: dict[str, tuple[torch.dtype, str]] = {
       "F32": (torch.float32, "ALL_F32"),
       "F16": (torch.float16, "MOSTLY_F16"),
       "BF16": (torch.bfloat16, "MOSTLY_BF16"),
   }
   SUPPORTED_QUANTIZATIONS: tuple[str, ...] = tuple(_QUANTIZATIONS)
   ```

   The public value stays `("F32", "F16", "BF16")`.
3. Delete `_quantization_torch_dtype` (L94-L110) and `_gguf_file_type_for` (L331-L345). Nothing else references them (`git grep -n "_quantization_torch_dtype\|_gguf_file_type_for" src tests` lists only these definitions and the two calls below), so no alias.
4. In `write_gguf`:
   - Delete `import torch` (L216) and the comment plus local import of `map_tensors` (L223-L225).
   - Delete the comment and formula for the width (L238-L241). Directly after `tensors = map_tensors(transformer_nn)` (L247) add:

     ```python
         # The SwiGLU hidden width, read from the network whose tensors are
         # written (``SwiGLU`` builds it as ``int(2 * ffn_mult * d_model / 3)``),
         # so a reader rebuilding shapes from the metadata gets the stored ones.
         feed_forward_length = cast(nn.Linear, transformer_nn.get_submodule("blocks.0.ffn.w1")).out_features
     ```

   - Replace `writer.add_file_type(_gguf_file_type_for(gguf, quantization))` (L274) with `writer.add_file_type(int(getattr(gguf.LlamaFileType, _QUANTIZATIONS[quantization][1])))`.
   - Replace `target_dtype = _quantization_torch_dtype(quantization)` (L301) with `target_dtype = _QUANTIZATIONS[quantization][0]`.
   - Leave the function-local `from nnx.nn.params.nn_run import _commit_partial, ...` (L258) alone (FEAT-073 moves the atomic writers).
5. Delete the function-local `import json` in `_tokenizer_vocab` (L389) and in `_collect_special_tokens` (L434), and `import warnings` in `_collect_special_tokens` (L443); keep the comments around them.
6. `_validate_quantization` keeps its `ImportError` / `ValueError` split and messages (it reads `SUPPORTED_QUANTIZATIONS`, unchanged in value).
7. In `tests/test_generative_nn_model.py` L387, change the comment `# The GGUF writer exports layer_norm_rms_eps=1e-6; the norms must still match it.` to `# The GGUF writer exports layer_norm_rms_eps=1e-6 (a constant); the norms must still match it.` The assertion stays.
8. In `tests/test_interop_gguf_writer.py`, after the BF16 file-type test (L295-L310), add:

   ```python
   def test_the_feed_forward_length_is_read_from_the_network(tmp_path):
       """FEAT-123: the metadata width is the network's own, a hand-resized
       feed-forward layer included (today's formula would give 85)."""
       import gguf
       from torch import nn

       from nnx.interop.gguf import write_gguf

       net = _tiny_transformer()  # d_model 32, ffn_mult 4: SwiGLU width int(2 * 4 * 32 / 3) == 85
       for block in net.blocks:
           block.ffn.w1 = nn.Linear(32, 64, bias=False)
           block.ffn.w3 = nn.Linear(32, 64, bias=False)
           block.ffn.w2 = nn.Linear(64, 32, bias=False)
       out_path = tmp_path / "resized.gguf"
       write_gguf(net, _tiny_tokenizer(tmp_path), out_path, quantization="F32")
       field = gguf.GGUFReader(str(out_path)).get_field("nnx_transformer.feed_forward_length")
       assert field.contents() == 64
   ```

9. Add to `CHANGELOG.md` under `## [Unreleased]` → `### Fixed`:

   ```
   - **GGUF export reads the feed-forward width from the network.** `write_gguf` records `feed_forward_length` from the first block's `ffn.w1` instead of recomputing it from `ffn_mult`, so a network whose feed-forward layers were resized by hand gets metadata that matches its tensors. Networks NNx builds get the same value as before.
   ```

10. Run `uv run ruff check --fix src/nnx/interop/` and `uv run ruff format src/nnx/interop/ tests/`.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need (`gguf`, `tokenizers`).

## Acceptance criteria

- [ ] `test_the_feed_forward_length_is_read_from_the_network` fails on `develop` before the change (85, not 64). Check: `uv run pytest tests/test_interop_gguf_writer.py::test_the_feed_forward_length_is_read_from_the_network`
- [ ] After the change the GGUF and Ollama suites pass, including the label tests without `gguf` and the BF16 file-type test. Check: `uv run pytest tests/test_interop_gguf_writer.py tests/test_examples_smoke.py -k "gguf or ollama or interop"` and `uv run pytest tests/test_interop_gguf_writer.py`
- [ ] One label table: the two mappers are gone. Check: `grep -n "def _quantization_torch_dtype\|def _gguf_file_type_for" src/nnx/interop/gguf/writer.py` returns 0 lines
- [ ] The only function-local imports left in the writer are `gguf` (in `_require_gguf`) and `nn_run` (in `write_gguf`). Check: `grep -nE "^\s{4,}(import |from \S+ import )" src/nnx/interop/gguf/writer.py` returns exactly 4 lines (the two `TYPE_CHECKING` imports, `import gguf` and the `nn_run` import; 10 today)
- [ ] `SUPPORTED_QUANTIZATIONS` is unchanged. Check: `uv run python -c "from nnx.interop.gguf import SUPPORTED_QUANTIZATIONS as s; assert s == ('F32', 'F16', 'BF16')"`
- [ ] `uv run pytest tests/test_generative_nn_model.py tests/test_imports.py` passes. Check: that command
- [ ] The CHANGELOG entry is under `[Unreleased]` → `### Fixed`. Check: `grep -n "GGUF export reads the feed-forward width from the network" CHANGELOG.md`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [writer.py L44-L46](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/interop/gguf/writer.py#L44-L46) | `SUPPORTED_QUANTIZATIONS` |
| [writer.py L94-L110](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/interop/gguf/writer.py#L94-L110) | `_quantization_torch_dtype`: if-chain one, and the false "cycles in tests" comment |
| [writer.py L331-L345](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/interop/gguf/writer.py#L331-L345) | `_gguf_file_type_for`: if-chain two |
| [writer.py L216-L247](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/interop/gguf/writer.py#L216-L247) | `write_gguf`: local imports, the false "import nnx.interop is cheap" comment, the recomputed width |
| [writer.py L389](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/interop/gguf/writer.py#L389), [L434](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/interop/gguf/writer.py#L434), [L443](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/interop/gguf/writer.py#L443) | local `json` / `warnings` imports |
| [transformer_layers.py L140-L156](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/net/transformer_layers.py#L140-L156) | `SwiGLU`: the width formula and `w1` |
| [test_generative_nn_model.py L387-L388](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_generative_nn_model.py#L387-L388) | #261's `eps == 1e-6` pin |
| #525, #261 | `map_tensors`'s refusals; the RMSNorm eps pin |

## Scope

**In:** the label table, the width read, the four hoisted imports and the two false comments.

**Out:** raising `ValueError` instead of `ImportError` for a llama-quantize label (a behaviour change; not ticketed). Reading `eps` from `norm_out` (pinned by #261's test; every NNx `RMSNorm` uses `1e-6`). `_require_gguf` and its install hint (FIX-128). The `nn_run` partial-file import (FEAT-073). The Ollama memo comment and the other DOC-31 comment fixes (FIX-143).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #261, #525, #517, #472, #570, #575
