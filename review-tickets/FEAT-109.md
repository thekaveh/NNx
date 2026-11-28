---
id: FEAT-109
title: "[FEAT-109] Make SplitManifest and Standardizer subclass JsonArtifact"
labels: ["enhancement", "area: data", "priority: P3", "review: 2026-11-27"]
sources: ["SMELL-34"]
depends_on: []
related_issues: [521]
status: READY
---

**Category:** Data · **Priority:** P3 · **Size:** M · **Evidence:** read at `9ad461e` (develop); not run: torch was unavailable in the review environment

## Summary

FEAT-109: `SplitManifest` and `Standardizer` re-implement `JsonArtifact`'s canonical bytes, equality, hash, JSON text and atomic save instead of inheriting them, as the calibration and abstention artifacts do. This ticket first pins both digests and `SplitManifest`'s non-ASCII JSON text, then makes both classes subclass `JsonArtifact` and deletes the copies. Their own canonical encoding (`provenance.canonical_bytes`), `digest()` and `SplitManifest`'s `ensure_ascii=False` JSON stay, so digests and files do not change. Their readers already share `check_keys`, `parse_json` and `read_text` and are not touched.

## Context

**What happens today.** `JsonArtifact` (`_artifacts.py` L137–L172) provides, from `state()`:

```python
    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.canonical_bytes() == cast(JsonArtifact, other).canonical_bytes()

    def __hash__(self) -> int:
        return hash(self.canonical_bytes())
```

plus `_canonical` / `canonical_bytes`, `to_json` and `save`. `Standardizer` (`preprocessing.py` L439–L471) has its own copy of each:

```python
    @cached_property
    def _canonical(self) -> bytes:
        return canonical_bytes(self.state())

    def canonical_bytes(self) -> bytes:
        return self._canonical

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Standardizer) and self.canonical_bytes() == other.canonical_bytes()
```

and `SplitManifest` (`data_splits.py` L218–L229, L287–L325) the same, with `to_json(..., ensure_ascii=False)`.

**Why it matters.** Five methods are copied twice and have drifted already (`ensure_ascii`). A fix to the shared artifact behaviour (the file mode of `save`, DEC-021) would miss these two.

**History.** #521 (FIX-054) made `SplitManifest.from_state` and `Standardizer.from_state` refuse unknown or missing keys; their readers now use `check_keys`, `parse_json(..., max_depth=MAX_JSON_DEPTH)` and `read_text`, so the reader side needs nothing here. A shared `JsonArtifact.from_file` is not added.

**Background for the worker.**
- `JsonArtifact`: the base class of NNx's frozen JSON artifacts (calibrators, abstention policies): everything is derived from `state()`.
- Digest: `sha256:<hex>` of the canonical bytes; persisted as the identity of a split or a standardizer (datasets and bundles record it), so it must not change.
- `provenance.canonical_bytes(value)` (`provenance.py` L123): canonical UTF-8 JSON after validating the value. Both classes keep it through their own `_canonical` override; `JsonArtifact._canonical` uses `_config._canonical_json` directly.
- Python rule: a class that defines `__eq__` without `__hash__` gets `__hash__ = None` (unhashable). So both are deleted together here, and the inherited pair applies.
- `@dataclass(frozen=True, eq=False)`: `eq=False` stops the dataclass from generating its own `__eq__`, so the inherited one is used. Both classes already have it.
- Equality difference: `JsonArtifact.__eq__` compares only objects of the same type, while the copies use `isinstance`. The two differ only when a user subclasses `SplitManifest` or `Standardizer`.

**Proposal.**
1. First, pin the current values. On `develop`, before any source change, run:

   ```bash
   uv run python -c "from nnx.data_splits import SplitManifest; print(SplitManifest(strategy='explicit', train=('b', 'a'), validation=('c',), test=('é',), parameters={'note': 'ü'}, seed=7).digest())"
   uv run python -c "from nnx.preprocessing import Standardizer; print(Standardizer(mean=(0.5, -1.0), scale=(2.0, 0.25), columns=('a', 'b')).digest())"
   ```

   and add to `tests/test_data_splits.py` (with the first printed value in place of `<printed digest>`):

   ```python
   _PINNED_SPLIT_DIGEST = "<printed digest>"


   def test_the_split_manifest_digest_and_json_text_are_pinned():
       manifest = SplitManifest(
           strategy="explicit", train=("b", "a"), validation=("c",), test=("é",), parameters={"note": "ü"}, seed=7
       )
       assert manifest.digest() == _PINNED_SPLIT_DIGEST
       assert '"é"' in manifest.to_json() and "\\u00e9" not in manifest.to_json()  # ensure_ascii=False
       assert manifest == SplitManifest.from_json(manifest.to_json()) and hash(manifest) == hash(SplitManifest.from_json(manifest.to_json()))
   ```

   and to `tests/test_preprocessing.py` (with the second printed value):

   ```python
   _PINNED_STANDARDIZER_DIGEST = "<printed digest>"


   def test_the_standardizer_digest_is_pinned():
       standardizer = Standardizer(mean=(0.5, -1.0), scale=(2.0, 0.25), columns=("a", "b"))
       assert standardizer.digest() == _PINNED_STANDARDIZER_DIGEST
       again = Standardizer.from_json(standardizer.to_json())
       assert again == standardizer and hash(again) == hash(standardizer)
   ```

   Commit the tests before the source change; they pass on `develop`.
2. In `src/nnx/data_splits.py`:
   - change `from ._artifacts import check_keys, parse_json, read_text` (L58) to `from ._artifacts import JsonArtifact, check_keys, parse_json, read_text`;
   - change `class SplitManifest:` (L167) to `class SplitManifest(JsonArtifact):`;
   - delete `__eq__` (L218–L219), `__hash__` (L228–L229), `canonical_bytes` (L291–L292) and `save` (L321–L325);
   - keep the `_canonical` override (L287–L289), `_identity`, `digest`, `identity`, `to_json` (it keeps `ensure_ascii=False`), `from_json`, `load` and `__repr__`. Add one line to `_canonical`: a docstring `"""``provenance.canonical_bytes`` of the state (not ``JsonArtifact``'s encoding): the split identity is persisted."""`.
3. In `src/nnx/preprocessing.py`:
   - change `from ._artifacts import check_keys, parse_json, read_text` (L41) to `from ._artifacts import JsonArtifact, check_keys, parse_json, read_text`;
   - change `class Standardizer:` (L239) to `class Standardizer(JsonArtifact):`;
   - delete `canonical_bytes` (L443–L444), `__eq__` (L446–L447), `__hash__` (L449–L450), `to_json` (L456–L457; `JsonArtifact.to_json` is the same text) and `save` (L467–L471);
   - keep the `_canonical` override (L439–L441, add the same one-line docstring as in step 2, with "the schema identity is persisted"), `digest` and the readers;
   - delete `import json` (L28): the deleted `to_json` was its only use.
4. Add to `CHANGELOG.md` under `## [Unreleased]` → `### Changed`:

   ```
   - **`SplitManifest` and `Standardizer` share the artifact base class.** They inherit equality, hashing, JSON text and atomic saving from the same base as the calibration and abstention artifacts. Digests and files are unchanged; an instance of a user subclass no longer compares equal to a base instance with the same state.
   ```

**Worker setup.** Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need.

## Acceptance criteria

- [ ] The two pin tests pass on `develop` before the source change and after it, with the same digests. Check: `uv run pytest tests/test_data_splits.py::test_the_split_manifest_digest_and_json_text_are_pinned tests/test_preprocessing.py::test_the_standardizer_digest_is_pinned`
- [ ] Both classes subclass `JsonArtifact`. Check: `grep -n "class SplitManifest(JsonArtifact)" src/nnx/data_splits.py` and `grep -n "class Standardizer(JsonArtifact)" src/nnx/preprocessing.py` each return 1 line
- [ ] The copies are gone. Check: `grep -nE "    def (__eq__|__hash__|canonical_bytes|save)\(" src/nnx/data_splits.py src/nnx/preprocessing.py` returns 0 lines
- [ ] The data and preprocessing suites pass unchanged. Check: `uv run pytest tests/test_data_splits.py tests/test_preprocessing.py tests/test_run_bundles.py tests/test_calibration.py`
- [ ] The CHANGELOG entry is under `[Unreleased]` → `### Changed`. Check: `grep -n "share the artifact base class" CHANGELOG.md`
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands
- [ ] `uv run pyright --warnings` passes. Check: `uv run pyright --warnings`

## Evidence

| Location | What it shows |
|---|---|
| [_artifacts.py L137-L172](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/_artifacts.py#L137-L172) | `JsonArtifact` |
| [data_splits.py L166-L167](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/data_splits.py#L166-L167), [L218-L229](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/data_splits.py#L218-L229), [L287-L335](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/data_splits.py#L287-L335) | `SplitManifest`: its copies, `digest`, `to_json(ensure_ascii=False)`, readers |
| [preprocessing.py L238-L239](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/preprocessing.py#L238-L239), [L439-L480](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/preprocessing.py#L439-L480) | `Standardizer`: its copies and readers |
| [provenance.py L123-L127](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/provenance.py#L123-L127) | `canonical_bytes`: the encoding both keep |
| #521 | the readers' key checks (already shared primitives) |

## Scope

**In:** the pin tests, the two subclasses, the deleted copies, the CHANGELOG line.

**Out:** `JsonArtifact.from_file` and any change to the readers (#521 already shares their primitives). The file mode of `JsonArtifact.save` (DEC-021). Digest formats across artifact types (FEAT-128).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #521, DEC-021
