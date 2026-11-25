"""Clear identifying archive headers without changing any packaged file bytes."""

from __future__ import annotations

import argparse
import gzip
import os
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def _safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")


def normalize_archive(path: Path) -> None:
    """Atomically rewrite one built wheel or sdist using neutral archive headers."""
    path = Path(path)
    if not (path.suffix == ".whl" or path.name.endswith(".tar.gz")):
        raise ValueError("Expected a wheel or .tar.gz source archive")
    descriptor, temporary = tempfile.mkstemp(prefix=".normalized-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            if path.suffix == ".whl":
                with zipfile.ZipFile(path) as source, zipfile.ZipFile(
                    output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9,
                ) as target:
                    for old in sorted(source.infolist(), key=lambda item: item.filename):
                        _safe_name(old.filename)
                        if (old.external_attr >> 16) & 0o170000 == 0o120000:
                            raise ValueError("Symbolic links are not allowed in a wheel")
                        new = zipfile.ZipInfo(old.filename, date_time=ZIP_EPOCH)
                        new.compress_type = zipfile.ZIP_DEFLATED
                        new.create_system = 3
                        mode = 0o40755 if old.is_dir() else 0o100644
                        new.external_attr = mode << 16
                        target.writestr(new, source.read(old), compresslevel=9)
            else:
                with tarfile.open(path, "r:gz") as source, gzip.GzipFile(
                    filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9,
                ) as compressed, tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT,
                ) as target:
                    for old in sorted(source.getmembers(), key=lambda item: item.name):
                        _safe_name(old.name)
                        if not (old.isfile() or old.isdir()):
                            raise ValueError("Only regular files and directories are allowed")
                        new = tarfile.TarInfo(old.name)
                        new.uid = new.gid = 0
                        new.uname = new.gname = ""
                        new.mtime = 0
                        new.type = tarfile.DIRTYPE if old.isdir() else tarfile.REGTYPE
                        new.mode = 0o755 if old.isdir() else 0o644
                        new.size = old.size if old.isfile() else 0
                        stream = source.extractfile(old) if old.isfile() else None
                        target.addfile(new, stream)
                        if stream is not None:
                            stream.close()
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path, nargs="?", default=Path("dist"))
    args = parser.parse_args()
    archives = sorted(args.dist_dir.glob("*.whl")) + sorted(args.dist_dir.glob("*.tar.gz"))
    if not archives:
        parser.error("No wheel or source archive found")
    for archive in archives:
        normalize_archive(archive)
        print(f"{archive.name}: normalized archive headers")


if __name__ == "__main__":
    main()
