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
folder that is a symlink (dangling or not), or that cannot be read, is
refused even with ``--force``, which never touches a link or its target,
and so is one that cannot be created (``--artifacts`` a dangling symlink
or not writable, say). An ``--output`` that is a directory, that is,
holds or lies inside a profile folder, or whose parent cannot be created, is refused too. Every folder, and the output,
is checked before any folder is cleared or written. A folder ``--force``
cannot clear (one holding a read-only subfolder, say) is refused, and may
be left partly cleared. An ``--output`` folder that exists but cannot be
written in is only found once the profiles ran: that exits 2 too, and keeps
the artifacts. Nothing is downloaded. Needs ``thekaveh-nnx[onnx-runtime,onnx-dynamo]``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from nnx.export_conformance import PROFILES, ConformanceError, exit_code, run_profiles, save_report, summary_lines

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
        try:
            mode = folder.lstat().st_mode  # the folder itself, never a link's target
            filled = not stat.S_ISLNK(mode) and (not stat.S_ISDIR(mode) or any(folder.iterdir()))
        except FileNotFoundError:
            continue  # a new folder
        except OSError as error:  # the folder, or its parent, cannot be read
            parser.error(f"{folder} cannot be read: {type(error).__name__}: {error}")
        if stat.S_ISLNK(mode):  # never followed or cleared, even with --force
            parser.error(
                f"{folder} is a symlink: a profile writes its artifacts only into its own folder; "
                "remove the link, or pass a new --artifacts folder"
            )
        if filled:
            if not args.force:
                parser.error(f"{folder} already holds artifacts: pass --force to clear it, or a new --artifacts folder")
            held.append(folder)
    if os.path.isdir(args.output):  # refused before --force clears anything; any OSError is left to the mkdir below
        parser.error(f"{args.output} is a directory: pass the path of the JSON report to write")
    output = Path(os.path.realpath(args.output))  # realpath never raises, even on a symlink loop
    for name in names:  # a report inside a profile folder would be cleared, or mixed with the artifacts
        folder = Path(os.path.realpath(Path(artifacts) / name))
        if folder in output.parents:
            parser.error(f"{args.output} is inside the profile folder {folder}: write the report outside it")
        if output == folder or output in folder.parents:  # the export would make it a directory
            parser.error(f"{args.output} is the profile folder {folder} or holds it: pass the path of the JSON report")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f"{args.output.parent} cannot be created: {type(error).__name__}: {error}")
    for folder in held:
        try:
            shutil.rmtree(folder) if folder.is_dir() else folder.unlink()
        except OSError as error:  # it may be left partly cleared
            parser.error(f"{folder} cannot be cleared: {type(error).__name__}: {error}")
    try:  # run_profiles creates every profile folder, and the root, before the first profile runs
        report = run_profiles(artifacts, names, source_revision=args.source_revision)
    except ConformanceError as error:  # a folder refused before anything is written
        parser.error(str(error))
    try:
        save_report(report, args.output)
    except OSError as error:  # only known once the profiles ran: os.access is unreliable with ACLs and root
        parser.error(
            f"{args.output} cannot be written: {type(error).__name__}: {error}; "
            f"the artifacts in {artifacts} are kept, so rerun with --force once the report can be written"
        )
    for line in summary_lines(report):
        print(line)
    code = exit_code(report)
    print(f"report: {args.output}; artifacts: {artifacts}; exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
