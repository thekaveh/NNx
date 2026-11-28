---
id: FEAT-070
title: "[FEAT-070] Trim audit-history narration from the PyPI keyword comments"
labels: ["enhancement", "area: packaging", "priority: P3", "review: 2026-11-27"]
sources: ["CLUT-05"]
depends_on: []
related_issues: [573, 575]
status: READY
---

**Category:** Packaging · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop)

## Summary

FEAT-070: four comment blocks inside `pyproject.toml`'s `[project] keywords` narrate how the list grew ("Megamerge subsystems", "rather than thekaveh/ml internals", "flagged by the post-#49 overnight-maintenance audit", two "overnight-maintenance pass" notes). Rewrite those four blocks to say only which subsystems the keywords cover, keeping the convention sentence, every module → keyword comment, every FEAT-0xx and PR reference, and every keyword.

## Context

**What happens today.** `pyproject.toml` L34–L54:

```toml
    # Megamerge subsystems (PRs #29–#37) — surfaced here for PyPI
    # discoverability now that the package ships these as first-class
    # entry points rather than thekaveh/ml internals.
```

```toml
    # Additional subsystems flagged by the post-#49 overnight-maintenance
    # audit as missing from PyPI keywords despite being first-class
    # entry points in nnx.*. The convention: every first-class nnx.*
    # subsystem has a keyword here, or a comment naming the existing
    # keyword that covers it, so PyPI search finds it.
```

```toml
    # Post-#51 overnight-maintenance pass — three more first-class
    # entry points that shipped without matching PyPI keywords:
    # `nnx.lr_finder` (PR #35), `nnx.trainer.Trainer` multi-optim
    # composite (PR #46), and the cross-cutting Builder pattern
    # (`<Class>.builder()` on every params dataclass, PRs #43-#46, #48).
```

```toml
    # Post-#54 overnight-maintenance pass — nnx.viz (six model-internals
    # primitives + Captum attribution) had no matching keyword.
```

**Why it matters.** These comments record internal process and a private repository name in a published manifest; they do not explain why a keyword is there.

**History.** The original finding proposed replacing the whole comment block with one line. That would undo two board decisions, so this ticket is narrowed:
- #573 (FEAT-051) required "The new entries sit under a comment block that names the modules" and that the convention be stated: module → keyword comments and the convention sentence stay.
- #575 (FEAT-053) kept tracking ids in code comments out of scope ("Out: Ids in code comments"): FEAT-0xx and PR references in these comments stay.
Only the narration goes: "Megamerge", "thekaveh/ml", the overnight-maintenance audit and pass wording, and "shipped without matching PyPI keywords" / "had no matching keyword".

**Background for the worker.**
- *Keywords*: `[project] keywords` feeds the `Keywords:` header of the wheel's METADATA, which PyPI shows. Comments inside the TOML array are not published; the keyword strings are.
- `tests/test_pypi_publication.py` (L369–L372) checks that each first-class module's keyword is present; it must keep passing unchanged.

**Proposal.**

1. In `pyproject.toml`, replace L34–L36 (the three "Megamerge" lines) with:
   ```toml
       # First-class subsystems from PRs #29–#37.
   ```
2. Replace L40–L44 (the five "Additional subsystems … so PyPI search finds it." lines) with:
   ```toml
       # The convention: every first-class nnx.* subsystem has a keyword here,
       # or a comment naming the existing keyword that covers it, so PyPI
       # search finds it.
   ```
3. Replace L47–L51 (the five "Post-#51 overnight-maintenance pass" lines) with:
   ```toml
       # nnx.lr_finder (PR #35), the nnx.trainer.Trainer multi-optim composite
       # (PR #46), and the Builder pattern (`<Class>.builder()` on every params
       # dataclass, PRs #43-#46, #48).
   ```
4. Replace L53–L54 (the two "Post-#54 overnight-maintenance pass" lines) with:
   ```toml
       # nnx.viz (six model-internals primitives + Captum attribution).
   ```
5. Leave every keyword string and every other comment (the FEAT-029, FEAT-030, FEAT-033, "Later first-class subsystems" and FEAT-051 blocks) unchanged.

No published metadata changes (comments only), so no CHANGELOG entry.

**Worker setup.** No torch needed (config only); `uv sync --frozen --all-extras` installs what `tests/test_pypi_publication.py` imports.

## Acceptance criteria

- [ ] The narration is gone. Check: `grep -n "Megamerge\|thekaveh/ml\|overnight\|shipped without matching\|had no matching" pyproject.toml` prints nothing
- [ ] The keyword list is identical to `develop`'s. Check: `uv run python -c "import subprocess, tomllib; old = tomllib.loads(subprocess.run(['git', 'show', 'origin/develop:pyproject.toml'], capture_output=True, text=True, check=True).stdout)['project']['keywords']; new = tomllib.load(open('pyproject.toml', 'rb'))['project']['keywords']; assert old == new, (old, new)"`
- [ ] The convention sentence and the FEAT-0xx references remain. Check: `grep -c "The convention: every first-class nnx" pyproject.toml` prints `1` and `grep -c "# FEAT-0" pyproject.toml` prints `4`
- [ ] The publication tests pass. Check: `uv run pytest tests/test_pypi_publication.py`

## Evidence

| Location | What it shows |
|---|---|
| [pyproject.toml L30-L74](https://github.com/thekaveh/NNx/blob/9ad461e/pyproject.toml#L30-L74) | The keyword list and its comments |
| [pyproject.toml L34-L36](https://github.com/thekaveh/NNx/blob/9ad461e/pyproject.toml#L34-L36), [L40-L44](https://github.com/thekaveh/NNx/blob/9ad461e/pyproject.toml#L40-L44), [L47-L51](https://github.com/thekaveh/NNx/blob/9ad461e/pyproject.toml#L47-L51), [L53-L54](https://github.com/thekaveh/NNx/blob/9ad461e/pyproject.toml#L53-L54) | The four narrating blocks |
| [test_pypi_publication.py L369-L372](https://github.com/thekaveh/NNx/blob/9ad461e/tests/test_pypi_publication.py#L369-L372) | The keyword-coverage test |
| #573 | Requires a module-naming comment block and the stated convention |
| #575 | Keeps ids in code comments |

## Scope

**In:** the four comment blocks.

**Out:** adding, removing or reordering keywords; the FEAT-0xx and PR references (#575).

## Dependencies

**Blocked by:** none
**Blocks:** none
**Related:** #573, #575
