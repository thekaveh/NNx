# NNx Opener Remediation Design

## Decision

Use the shared opener pattern demonstrated by Atlas and nativ: a full-width brand banner, an unnumbered centered product H1, centered value copy, grouped shield badges, and then the executive summary. Navigation labels remain numbered, but navigation numbers are not part of product identity.

Two alternatives were rejected. A CSS-only patch would not fix GitHub or PyPI rendering, and retaining the current workflow infographic as the poster would keep architecture detail in the brand slot. The selected approach uses GitHub-compatible HTML for shared structure and reserves technical diagrams for the architecture section.

## Opener Contract

`README.md` and `docs/index.md` use the same semantic sequence:

1. Full-width local banner image.
2. `<h1 align="center">NNx</h1>` with no navigation number.
3. Centered bold tagline and centered supporting value line.
4. Centered project-status badges.
5. Centered core-stack and optional-ecosystem shield badges.
6. The existing 100-150 word executive summary.

The site and wiki continue to derive from `docs/index.md`; the PyPI description continues to derive from `README.md`. Local image paths are rewritten by the existing projection pipeline.

## Navigation Numbering

The manifest gains an explicit `numbered_h1: false` override for brand-facing pages. The default remains `true`, preserving baked H1 numbering everywhere else. `README.md` and `docs/index.md` opt out while their navigation labels remain `15. Repository overview` and `1. Home`.

## Banner

Replace the current workflow infographic with brand-led raster artwork. It contains no baked text, tiny labels, flow arrows, or architecture claims. The image should evoke transparent neural layers, graph structure, repeatable training, and durable checkpoints through abstract but inspectable visual forms. The separate architecture diagram remains the authoritative technical visual.

## Responsive Site Treatment

GitHub-compatible alignment attributes provide the baseline. MkDocs CSS adds constrained banner sizing, centered opener spacing, and wrapping badge rows without relying on viewport-scaled type. Mobile rendering must not overflow or shrink badge labels into unreadability.

## CI And Tests

Tests must fail unless:

- both canonical openers have an unnumbered centered `NNx` H1;
- the banner precedes the H1;
- tagline, support line, badge groups, and executive-summary opening match;
- core stack badges include PyTorch, PyTorch Geometric, NumPy, pandas, scikit-learn, and Plotly;
- optional badges represent TensorBoard, W&B, ONNX, torchao, Hugging Face, safetensors, and FAISS;
- numbered manifest pages retain their numbered H1s unless explicitly exempt;
- site, wiki, and PyPI projections preserve the opener contract;
- security checks run on pushes to both `main` and `develop` as well as pull requests.

Verification includes focused red-green tests, deterministic diagram and projection checks, strict MkDocs, the full documentation test set, visual inspection of banner derivatives, and repository lint/type/test gates appropriate to changed Python code.
