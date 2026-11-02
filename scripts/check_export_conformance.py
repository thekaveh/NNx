"""Run the ONNX export conformance profiles and save their evidence (FEAT-038).

    python scripts/check_export_conformance.py --output conformance.json

Runs every tested profile in ``nnx.export_conformance.PROFILES`` (or the
``--profile`` names given): each exports ``FeedFwdNN`` 4-8-2 with its
exporter into ``--artifacts/<profile>/`` (default: ``<output>-artifacts``
next to the JSON), checks it, verifies the artifact hashes, loads it in
ONNX Runtime on CPU and compares every raw logit with the native model.
The JSON report records each stage's outcome separately, with the source
revision, config, artifact SHA-256s (external data included), exporter
options, opset, dependency and runtime versions, provider, dtype, input
cases and tolerances.

The exit status separates failures — it never passes by skipping:

     0 every profile passed       14 hash-mismatch
    10 missing-dependency         15 load-error
    11 export-error               16 mismatch
    12 state-changed              17 input-contract
    13 checker-error              18 runtime-error

1 is an error and 2 a usage error: an artifact folder that already holds a
profile's files is refused unless ``--force`` clears it, and a profile
folder that is a symlink (dangling or not) is refused even with
``--force``, which never touches a link or its target. Every folder is
checked before any is cleared or written. Nothing is downloaded. Needs
``thekaveh-nnx[onnx-runtime,onnx-dynamo]``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from nnx.export_conformance import PROFILES, exit_code, run_profiles, save_report, summary_lines

    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--output", required=True, type=Path, help="where to write the JSON report")
    parser.add_argument(
        "--artifacts", type=Path, help="directory for the exported files (default: <output stem>-artifacts)"
    )
    parser.add_argument(
        "--profile", action="append", choices=sorted(PROFILES), help="run only this profile (repeatable)"
    )
    parser.add_argument(
        "--source-revision", help="the source revision to record (default: the git checkout NNx is imported from)"
    )
    parser.add_argument("--force", action="store_true", help="clear the profiles' earlier artifacts first")
    args = parser.parse_args(argv)

    artifacts = args.artifacts or args.output.with_name(f"{args.output.stem}-artifacts")
    names = list(dict.fromkeys(args.profile or PROFILES))
    held = []
    for name in names:  # every folder is checked before --force clears any
        folder = Path(artifacts) / name
        if folder.is_symlink():  # never followed or cleared, even with --force
            parser.error(
                f"{folder} is a symlink: a profile writes its artifacts only into its own folder; "
                "remove the link, or pass a new --artifacts folder"
            )
        try:
            filled = folder.exists() and (not folder.is_dir() or any(folder.iterdir()))
        except OSError as error:
            parser.error(f"{folder} cannot be read: {type(error).__name__}: {error}")
        if filled:
            if not args.force:
                parser.error(f"{folder} already holds artifacts: pass --force to clear it, or a new --artifacts folder")
            held.append(folder)
    for folder in held:
        shutil.rmtree(folder) if folder.is_dir() else folder.unlink()
    os.makedirs(artifacts, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run_profiles(artifacts, names, source_revision=args.source_revision)
    save_report(report, args.output)
    for line in summary_lines(report):
        print(line)
    code = exit_code(report)
    print(f"report: {args.output}; artifacts: {artifacts}; exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
