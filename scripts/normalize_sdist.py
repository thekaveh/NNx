"""Normalize sdist container metadata for byte-reproducible releases."""

from __future__ import annotations

import argparse
import copy
import gzip
import io
import os
import tarfile
import tempfile
from pathlib import Path


def normalize_sdist(path: Path, *, epoch: int) -> None:
    path = Path(path)
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(path, "r:gz") as source:
        for member in sorted(source.getmembers(), key=lambda item: item.name):
            stream = source.extractfile(member) if member.isfile() else None
            members.append((copy.copy(member), stream.read() if stream is not None else None))

    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target:
                    for member, data in members:
                        member.uid = 0
                        member.gid = 0
                        member.uname = ""
                        member.gname = ""
                        member.mtime = epoch
                        member.pax_headers = {}
                        target.addfile(member, io.BytesIO(data) if data is not None else None)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    normalize_sdist(args.path, epoch=args.epoch)


if __name__ == "__main__":
    main()
