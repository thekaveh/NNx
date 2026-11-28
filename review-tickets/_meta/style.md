# Ticket style guide: thekaveh/NNx board, worker = Haiku-class agent

## 1. Output file

Write one file per ticket: `/tmp/claude-0/-home-user-NNx/11f05a29-8365-5138-9342-787fd28b7c18/scratchpad/tickets/out/<ID>.md`. Use exactly this front matter, then the body:

```
---
id: FIX-104
title: "[FIX-104] Score samples whose input has no stable digest instead of failing evaluate()"
labels: ["bug", "area: decisions", "priority: P1", "review: 2026-11-27"]
sources: ["BUG-01", "DOC-03"]
depends_on: []          # new-ticket IDs that must land first (from the plan, plus any you discover)
related_issues: [557, 243]   # existing board issue numbers
status: READY           # READY | DECISION (DEC tickets) | DROP (only if the plan allows it and you verified it; say why in the body)
---
```

Front-matter rules:
- **Title:** use the plan's title verbatim, prefixed `[ID]`.
- **Labels:** in this order:
  1. the type label from the plan (`bug`, `enhancement`, `documentation` or `question`)
  2. any extra labels from the plan (`github_actions`, `dependencies`)
  3. one or two `area: …` labels
  4. `priority: Px`
  5. `review: 2026-11-27`
- **Areas:** use only these: configuration, data, decisions, documentation, evaluation, execution, fluent-api, graph, inference, interoperability, model-integration, model-transforms, numerics, packaging, peft, persistence, training.

## 2. Body: mirror the board's tickets (see #782 and #767)

The first line is the header:
`**Category:** <Area, title case> · **Priority:** Px · **Size:** S|M|L · **Evidence:** read at \`9ad461e\` (develop)<; not run: torch was unavailable in the review environment>`

Then these sections, in this order, with these exact `##` headings:

1. `## Summary`: start with `<ID>:`. In 1–3 sentences, say what is wrong (or what to decide) and what changes.
2. `## Context`, which holds:
   - **What happens today.** Quote the current code (3–15 lines) from 9ad461e, without "..." elisions inside a quote. If you shorten, quote two separate exact fragments instead.
   - **Why it matters**, and where it came from: the commit or PR (e.g. 3ed0133 / #853).
   - **History.** What the related board issues decided, quoting the key line, e.g. "#552 chose…". If an issue made part of the original finding deliberate, say so and narrow the ticket to what is left.
   - `**Background for the worker.**` Define, in one line each, every project term the ticket relies on: partial file, `_full_fsync`, `F_FULLFSYNC`, meta device, Hub manifest, `runs/best`, `extra_metrics`, `TaskSpec`, `require_count`, `_warn_at_user_frame`, release-please marker, and so on. Do not assume the worker has read anything else.
   - `**Proposal.**` A numbered list of exact steps:
     - Name the file, the function, and the code to add or replace. Give final code, or a precise sketch you have checked against the real code.
     - **No options, no "or", no "optionally", no "consider".** Optional ideas go to Scope: Out.
     - For a doc or text change, give the exact old text and the exact new text.
     - For any changed error or warning message, give the new text verbatim, and list every test that asserts the old text (grep `tests/`) and how it changes.
     - For every behaviour change, give the exact CHANGELOG bullet, its `[Unreleased]` subsection (`### Fixed` / `### Changed` / `### Added`), and its text in the house style: `- **Bold summary.** Explanation.`
     - When moving or renaming a private helper, keep the old name as an alias. List the tests that monkeypatch the old location (grep `monkeypatch.setattr` and `patch(` in `tests/`) and say how each is retargeted.
     - Say how to regenerate generated files when touched: `uv run python -m scripts.docs.build_api_reference` for docstrings that appear in `docs/api.md`, and `uv run python -m scripts.docs.build_pypi_readme` for README.
   - `**Worker setup.**` Use one of these lines:
     - "Branch from `develop`. `uv sync --frozen --all-extras` installs torch and every extra the tests need."
     - "No torch needed (docs/config only)."
     - "Needs <platform> to observe; on Linux the test below proves it by monkeypatching …"
3. `## Acceptance criteria`: GitHub checkboxes.
   - Every criterion is observable, and ends with `Check: \`<command>\`` (a pytest node id, a grep that must return N lines, a `--check` build, etc.).
   - Include:
     - the new tests: file, test name, and what each asserts;
     - every existing test that must be rewritten (the enshrining tests) and its new assertion;
     - "`uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass";
     - "`uv run pyright --warnings` passes";
     - `uv run python -m scripts.docs.build_docs --check`, when docs or docstrings change;
     - the CHANGELOG entry, when behaviour changes.
   - For tests that need torch, write the test first: one criterion says "the new test fails on `develop` before the change".
4. `## Evidence`: a table `| Location | What it shows |`.
   - Every code location is a permalink: `[file.py L12-L20](https://github.com/thekaveh/NNx/blob/9ad461e/<path>#L12-L20)`.
   - Verify the line numbers yourself in the worktree `/tmp/claude-0/-home-user-NNx/11f05a29-8365-5138-9342-787fd28b7c18/scratchpad/develop`. Also list related board issues (`#552`) and the source commit or PR.
5. `## Scope`: `**In:** …` and `**Out:** …`. Out-of-scope items name the ticket that covers them, if any (by new ID, e.g. "FEAT-071").
6. `## Dependencies`:
   - `**Blocked by:** <new IDs>`; `**Blocks:** <new IDs>`; `**Related:** #n …`
   - Refer to new tickets by ID only (`FIX-107`). Their issue numbers do not exist yet and are filled in at posting time.

## 3. DEC tickets (copy the #767 pattern)

Sections: Summary, Context, Options, Acceptance criteria, Evidence, Scope, Dependencies.
- **Options:** a table `| Option | What changes | Trade-off |`, with exactly one option marked `(recommended)` and one line saying why.
- **Acceptance criteria:**
  - The first box is "- [ ] The owner records the chosen option in a comment on this issue."
  - Then "If option A: …" boxes. Each must be fully executable by a worker (exact files, steps or texts, tests and Check commands), because a worker will implement the chosen option straight from this ticket.
  - Keep each option's steps in a `**If option X, the worker does:**` numbered list in Context or Options, so the criteria stay short.
- Status is `DECISION`.

## 4. Haiku-readiness checklist (every ticket must pass all of these)

- [ ] One focused PR. If it needs more, tell the coordinator; do not silently grow the ticket.
- [ ] No unresolved choice, conditional ("if X was done"), or "optional" step in Proposal or Acceptance criteria (DEC tickets excepted, through their per-option criteria).
- [ ] Every location is a verified permalink plus a function or class name.
- [ ] Every new name (function, parameter, constant, test) is spelled out.
- [ ] Every behaviour change has its CHANGELOG text.
- [ ] Every Haiku blocking question listed for the source finding (`check/H1-summary.md` and `H2-summary.md`) is answered in the ticket.
- [ ] Every cross-check "Required change" for the source (`check/X1`, `X2`, `X3`) is applied, and the related issues are linked.
- [ ] No contradiction with a recorded board decision. If one exists, the ticket is narrowed to what is left and says so.
- [ ] No secrets, no personal paths, no AI model names. Do not mention the review tooling. Write as the repository owner's tracker.
- [ ] Plain, direct English. No filler.

## 5. Sources to read

- The plan: `/tmp/claude-0/-home-user-NNx/11f05a29-8365-5138-9342-787fd28b7c18/scratchpad/tickets/plan.md` (titles, labels, dependencies, conflict resolutions; binding).
- The findings: `/home/user/NNx/REVIEW-develop-2026-11-27.md` (each `### <ID>` section).
- The check results: `/tmp/claude-0/-home-user-NNx/11f05a29-8365-5138-9342-787fd28b7c18/scratchpad/check/*.md`.
- The board issues with full bodies, for quoting history: `/tmp/claude-0/-home-user-NNx/11f05a29-8365-5138-9342-787fd28b7c18/scratchpad/issues/issues.md` (grep it; it is 2 MB) and `issues.json`.
- The code: worktree `/tmp/claude-0/-home-user-NNx/11f05a29-8365-5138-9342-787fd28b7c18/scratchpad/develop` at 9ad461e (read-only; never modify it).
