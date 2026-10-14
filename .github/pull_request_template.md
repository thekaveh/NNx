## Summary

<!-- 1-3 sentences. What changed and why. Link the issue (`Fixes #N`) if applicable. -->

## Test plan

<!-- A bulleted checklist of how you exercised the change. Include any new tests
     and the command used to run them. -->

- [ ] `uv run pytest -n logical --dist loadgroup` (full suite passes locally)
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/`
- [ ] `uv run pyright --warnings`
- [ ] Added / updated tests covering the change
- [ ] For a user-visible change: an entry in `CHANGELOG.md` under `[Unreleased]`

## Notes for reviewers

<!-- Anything unusual: format change, deprecation, behavior shift, etc. -->
