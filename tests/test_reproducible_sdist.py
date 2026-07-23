from __future__ import annotations

import hashlib
import io
import tarfile

from scripts.normalize_sdist import normalize_sdist


def _write_sdist(path, *, mtime: int, uid: int, reverse: bool = False) -> None:
    entries = [("pkg/PKG-INFO", b"Name: pkg\n"), ("pkg/src/mod.py", b"VALUE = 1\n")]
    if reverse:
        entries.reverse()
    with tarfile.open(path, "w:gz") as archive:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = mtime
            info.uid = uid
            info.gid = uid
            info.uname = f"user-{uid}"
            info.gname = f"group-{uid}"
            archive.addfile(info, io.BytesIO(content))


def test_normalize_sdist_removes_archive_metadata_variance(tmp_path):
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_sdist(first, mtime=100, uid=501)
    _write_sdist(second, mtime=200, uid=1000, reverse=True)

    normalize_sdist(first, epoch=42)
    normalize_sdist(second, epoch=42)

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == sorted(member.name for member in members)
    assert {(member.mtime, member.uid, member.gid, member.uname, member.gname) for member in members} == {
        (42, 0, 0, "", "")
    }
