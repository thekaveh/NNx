# Test Import Boundaries

## 1. Public Contract Tests

Tests that assert user-facing behavior should import through `nnx` or a documented public facade such as `nnx.viz`, `nnx.embeddings`, `nnx.surgery`, `nnx.generation`, `nnx.finetune`, or `nnx.trainer`.

Use this style when the test is meant to protect what users can reach after `import nnx`.

## 2. Implementation Unit Tests

Deep imports are intentional when a test targets a concrete implementation boundary:

- Private helpers such as `nnx._metrics` and `nnx._step_helpers`.
- Dataclass serialization and builders under `nnx.nn.params`.
- Enum behavior under `nnx.nn.enum`.
- Individual network layers and datasets under `nnx.nn.net` or `nnx.nn.dataset`.
- Submodule-only routing helpers whose public effect is tested elsewhere.

These tests may import the exact module they exercise so a public-facade alias does not hide which internal contract is under review.

## 3. Facade Regression Coverage

`tests/test_public_api_exports.py` protects the curated top-level exports in `nnx.__all__`. When a behavior test needs only a public symbol covered there, prefer importing from `nnx`; when it needs the implementation object itself, keep the deep import and make that intent clear in the test or nearby comment.
