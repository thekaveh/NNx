# Contributing to NNx

Thanks for being interested in contributing. NNx is a small library; the goal is to keep it small, tested, and useful for the existing notebook consumers while inviting new ones.

## 1. Getting set up

```bash
git clone https://github.com/thekaveh/NNx.git
cd NNx
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install              # optional but recommended
```

Verify a clean baseline:

```bash
pytest                          # full suite (~15s on CPU)
ruff check src/ tests/ examples/  # lint
ruff format --check src/ tests/ examples/  # format check (matches CI + pre-commit)
mkdocs build --strict           # docs (gates CI)
```

Useful env vars:

- `NNX_TQDM_DISABLE=1` silences the training progress bar. Set this in CI / non-TTY contexts, and in any test that drives `NNModel.train()` or `Trainer.train()` (the test suite's `conftest.py` already does this session-wide). Accepts `1` / `true` / `yes`, case-insensitive.

## 2. Workflow

1. **Open an issue first** for non-trivial changes — saves churn if the design is off. Tiny fixes can go straight to PR.
2. **Branch from `main`.** Name branches descriptively (`fix/...`, `feat/...`, `docs/...`, `refactor/...`).
3. **Write tests.** Every PR that changes behavior should land with a focused test that fails on `main` and passes on the branch. The existing `tests/test_*_series.py` files (organized by audit pass) are good models.
4. **Keep PRs small.** One coherent change per PR is much easier to review than a sweeping mix.

## 3. What we care about

- **Strict back-compat for the existing notebook consumer.** Don't rename, remove, or restructure public APIs without a migration path. Don't change the on-disk `runs/<id>/` format. New fields on params dataclasses must omit themselves from `.state()` when set to their defaults (preserves `run.id` hashes). See the omit-when-default regression tests in `tests/test_params_round_trip.py` (search for `test_nn_*_state_omits_*_when_*`) for the canonical pattern.
- **State / from_state round-trip.** Every params dataclass with a `state()` method must round-trip cleanly through `from_state(state())`. The contract is enforced by `tests/test_params_round_trip.py`.
- **Tests run on CPU and finish fast.** Keep new tests under a few seconds; use small TensorDataset fixtures from `tests/conftest.py`.
- **One-line update to `CHANGELOG.md` under `[Unreleased]`** for any user-visible change.

## 4. Style

- **Ruff** enforces formatting and a curated lint rule set (`E F W B I UP`). Run `ruff check --fix src/ tests/ examples/` and `ruff format src/ tests/ examples/` before pushing. Pre-commit handles both automatically when installed.
- **Type annotations** are encouraged on new code. We type-check with pyright (basic mode) in CI, with `--strict` planned over time.
- **Docstrings** on public functions / classes explain the *why* (constraints, edge cases) — not just the *what*. Multi-paragraph is fine when warranted.
- **Comments** explain non-obvious decisions, hidden constraints, or surprising behavior. Don't narrate code that's already self-documenting.

## 5. Testing

```bash
pytest                          # full suite
pytest tests/test_pass2_n_series.py::test_n7_evaluate_aggregates_across_batches
pytest -k "graph"               # name filter
pytest --cov=nnx --cov-report=term-missing  # with coverage
```

Tests live under `tests/`. The `conftest.py` registers two session-wide hygiene fixtures (NNX_TQDM_DISABLE, env_snapshot cache reset between tests); otherwise it's intentionally minimal. Add shared fixtures there when boilerplate repeats across multiple tests, not preemptively.

## 6. Submitting a PR

- Push to your fork and open a PR against `main`.
- Fill in the PR template (Summary / Test plan).
- Wait for CI to go green (lint + format + tests + mkdocs on 3.10 / 3.11 / 3.12).
- Address review comments by pushing new commits — we squash on merge.

## 7. Things we won't merge

- Changes that break on-disk format compatibility without a versioned reader.
- Public API renames without a deprecation shim and a `__getattr__` alias for at least one minor version.
- Code without tests.
- Dependencies added to `[project.dependencies]` (the core deps list) when they could go under `[project.optional-dependencies]` instead.

## 8. License

By contributing you agree that your contribution will be licensed under the [MIT License](LICENSE).
