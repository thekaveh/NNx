# Three-Surface Documentation Remediation Design

## Goal

Resolve every finding from the 2026-07-31 three-surface documentation audit while preserving NNx's canonical-repository, generated-site, and generated-wiki publication model.

## Canonical opener

`README.md` and `docs/index.md` will begin with the same concise tagline, the same restrained health-badge row, the same product poster, and the same 100-150 word executive summary. Relative asset destinations may differ, but visible copy and badge targets must remain identical. Automated tests will extract these opener elements and reject drift.

The summary will explain NNx's supervised training loop, content-addressed persistence, callbacks, model specializations, and core differentiator: NNx owns routine experiment infrastructure while keeping model and step logic replaceable.

## Numbered documents

The manifest remains the single inventory. Every canonical Markdown source will bake its manifest number into its H1. The plain-text Apache license will receive its numbered H1 only during projection. Validation will reject a source whose first H1 does not start with its manifest number and will verify the projected License H1 separately.

## Visual assets

A new standalone HTML master will define a product-level NNx poster. The existing extractor will continue deriving SVG assets from all HTML masters and will additionally render committed PNG fallbacks with CairoSVG. Repository and wiki Markdown will use PNG assets; the generated site will prefer SVG. The projection builder will select the appropriate format while retaining physical copies of both formats on each generated surface.

## Claims and lifecycle accuracy

The README lifecycle description will state that scheduler stepping and checkpoint persistence occur once per completed epoch. Unsubstantiated comparative superlatives and anecdotal performance numbers will be removed or narrowed to behavior exercised by the test suite. DoRA will be described structurally with a primary-paper citation; framework comparison language will describe NNx's scope without exclusivity claims. KV-cache documentation will cite the repository's repeatable CPU regression threshold and avoid unsupported GPU extrapolation.

## Verification

Tests will cover opener parity and word count, badges and poster placement, manifest-to-H1 agreement, surface-specific SVG/PNG rewriting, and deterministic diagram outputs. Existing projection, wiki, strict MkDocs, lint, type, and repository test suites remain green. CI will install the Cairo runtime before checking committed diagram derivatives.
