---
id: FEAT-136
title: "[FEAT-136] Hoist needless function-local imports in nnx.decisions"
labels: ["enhancement", "area: decisions", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-61"]
depends_on: ["FEAT-060", "FEAT-126", "FEAT-113", "FIX-104"]
related_issues: [249]
status: READY
---

**Category:** Decisions · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-136: `decisions/benchmark.py` has 16 function-local imports and `decisions/jobs.py` 10. None guards an optional extra; the only real cycle (`nnx.abstention`) is broken by FEAT-060. This ticket moves every remaining one to module top, so that afterwards neither file imports inside a function.

## Context

**What happens today.** For example, in `benchmark.py`:

```python
def _byte_dtype() -> Any:
    import torch

    return torch.uint8
```

```python
def _selective(rows: Sequence[_Row], policy: Any, model_id: Optional[str]) -> dict[str, MetricValue]:
    ...
    from ..abstention import AbstentionSchemaError, decide
    from .schema import ChoiceResult
```

while `.schema` is already imported at module top; and in `jobs.py` `_modality` and `_one_value` each `import numpy as np` / `import torch` on every call.

**Why it matters.** A function-local import hides a module's dependencies and suggests a cycle or an optional dependency that is not there.

**History.** #249 (FEAT-031) made NNx's optional extras lazy; that rule — keep an import local only for an optional extra or a real cycle — is the one applied here. The `nnx.abstention` imports (six sites) are SMELL-42's cycle, removed by FEAT-060. FEAT-126 removes the `_artifacts` imports of `write_records` / `read_records`; FEAT-113 rewrites the metric helpers that hold the `..calibration` imports; FIX-104 edits `_fingerprint`. This ticket runs after all four.

**Background for the worker.**
- Classes used below: **hoist** (move to module top), **keep-cycle** (the target imports this module back), **keep-optional-extra** (the target is an optional dependency).
- Cycle check done for this ticket: `nnx.calibration`, `nnx._artifacts` and `nnx.decisions.schema` never import `nnx.decisions.benchmark` or `nnx.decisions.jobs`, directly or through their module-top imports. NumPy and torch are core dependencies, and `import nnx` (which every `nnx.decisions` import runs first) already loads both; `providers.py` and `nli.py` already import NumPy at module top.
- No test monkeypatches any of these names inside `benchmark` or `jobs` (`tests/test_result_boundaries.py` L97 patches `schema.question_from_state` for `nnx.result`, which keeps its own local import).

**Proposal.**
1. `src/nnx/decisions/benchmark.py` — every function-local import at `9ad461e`, with its class:

   | Line | Function | Import | Class | What to do |
   |---|---|---|---|---|
   | L212 | `_fingerprint` | `import numpy as np` | hoist | core dependency |
   | L246 | `_byte_dtype` | `import torch` | hoist | add `import torch` at module top; replace `_byte_dtype()` in `_fingerprint` (L229) with `torch.uint8` and delete `_byte_dtype` (L245-L248): it only existed to import torch lazily |
   | L552 | `write_records` | `from .._artifacts import atomic_write` | — | removed by FEAT-126 |
   | L561 | `read_records` | `from .._artifacts import parse_json, read_text` | — | removed by FEAT-126 |
   | L615 | `_batch_input` | `import torch` | hoist | core dependency |
   | L621 | `_batch_input` | `import numpy as np` | hoist | core dependency |
   | L784 | `_modality` | `import numpy as np` | hoist | core dependency |
   | L923 | `_bins` | `import numpy as np` | hoist | core dependency |
   | L925 | `_bins` | `from ..calibration import _bins as calibration_bins` | hoist | no cycle (see above) |
   | L1101 | `_metrics` | `from ..calibration import expected_calibration_error` | hoist | no cycle |
   | L1117 | `_selective` | `from ..abstention import AbstentionSchemaError, decide` | — | FEAT-060 (the abstention cycle) |
   | L1118 | `_selective` | `from .schema import ChoiceResult` | hoist | add `ChoiceResult` to the module-top `from .schema import (...)` block |
   | L1263 | `evaluate` | `from ..abstention import AbstentionPolicy` | — | FEAT-060 |
   | L1386 | `bootstrap_interval` | `import numpy as np` | hoist | core dependency |
   | L1567 | `BenchmarkReport.save` | `from .._artifacts import atomic_write` | hoist | no cycle |
   | L1575 | `BenchmarkReport.load_state` | `from .._artifacts import parse_json, read_text` | hoist | no cycle |

   Module top then gains `import numpy as np`, `import torch`, `from .._artifacts import atomic_write, parse_json, read_text` (merged with FEAT-126's `from .._artifacts import ...` line) and `from ..calibration import _bins as calibration_bins, expected_calibration_error`. Delete each local import line and any blank line it leaves at the start of a function body.
2. `src/nnx/decisions/jobs.py` — every function-local import at `9ad461e`:

   | Line | Function | Import | Class | What to do |
   |---|---|---|---|---|
   | L157 | `_round_trips` | `import pickle` | hoist | standard library |
   | L429 | `DecisionJob.ask` | `from ..abstention import AbstentionPolicy` | — | FEAT-060 |
   | L437 | `DecisionJob.ask` | `from ..abstention import AbstentionSchemaError` | — | FEAT-060 |
   | L532 | `_from_state` | `from .schema import question_from_state` | hoist | add to the module-top `from .schema import (...)` block |
   | L537 | `_from_state` | `from ..abstention import AbstentionPolicy` | — | FEAT-060 |
   | L586 | `_modality` | `import numpy as np` | hoist | core dependency |
   | L587 | `_modality` | `import torch` | hoist | core dependency |
   | L619 | `_one_value` | `import numpy as np` | hoist | core dependency |
   | L620 | `_one_value` | `import torch` | hoist | core dependency |
   | L974 | `_Runner._record` | `from ..abstention import decide` | — | FEAT-060 |

   Module top gains `import pickle`, `import numpy as np`, `import torch`, and `question_from_state` in the `.schema` block.
3. Any other function-local import that FEAT-113, FEAT-126 or FIX-104 added to these two files is hoisted by the same rule. FEAT-060 deletes all six `..abstention` imports; if one of them is still inside a function when you start, stop and report it on this issue instead of guessing.
4. Run `uv run ruff check --fix src/nnx/decisions/` (sorts the import blocks) and `uv run ruff format src/nnx/decisions/`.

No behaviour changes, so no test changes and no CHANGELOG entry.

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] Neither file imports inside a function. Check: `uv run python -c "import ast; print([(p, n.lineno) for p in ('src/nnx/decisions/benchmark.py', 'src/nnx/decisions/jobs.py') for f in ast.walk(ast.parse(open(p).read())) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(f) if isinstance(n, (ast.Import, ast.ImportFrom))])"` prints `[]`
- [ ] `_byte_dtype` is gone. Check: `grep -n "_byte_dtype" src/nnx/decisions/benchmark.py` returns 0 lines
- [ ] Importing `nnx` and `nnx.decisions` still starts no provider and loads no hosted SDK. Check: `uv run pytest tests/test_imports.py`
- [ ] The decision suites pass. Check: `uv run pytest tests -k decision` and `uv run pytest tests/test_result_boundaries.py`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [benchmark.py L205-L248](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L205-L248) | `_fingerprint` and `_byte_dtype`: local NumPy / torch |
| [benchmark.py L606-L625](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L606-L625) | `_batch_input` |
| [benchmark.py L920-L930](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L920-L930) | `_bins`: local NumPy and calibration |
| [benchmark.py L1110-L1118](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1110-L1118) | `_selective`: the abstention cycle and an already-imported module |
| [benchmark.py L1564-L1576](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/benchmark.py#L1564-L1576) | the report's `_artifacts` imports |
| [jobs.py L156-L162](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L156-L162) | `_round_trips`: local `pickle` |
| [jobs.py L531-L537](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L531-L537) | `_from_state`: local `.schema` |
| [jobs.py L580-L620](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jobs.py#L580-L620) | `_modality`, `_one_value`: local NumPy / torch |
| #249 | the "local only for an optional extra" rule |

## Scope

**In:** the function-local imports of `decisions/benchmark.py` and `decisions/jobs.py`.

**Out:** the `nnx.abstention` imports (FEAT-060). `providers.py` and `nli.py`, which keep torch local inside their model-touching methods (not listed by SMELL-61; not ticketed). Splitting `benchmark.py` (FEAT-121). The arun split (FEAT-120).

## Dependencies

**Blocked by:** FEAT-060, FEAT-126, FEAT-113, FIX-104 (each edits lines this ticket moves)
**Blocks:** FEAT-121
**Related:** #249
