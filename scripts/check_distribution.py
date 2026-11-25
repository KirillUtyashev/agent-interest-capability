"""Reject weights, notebooks, and research outputs in release distributions."""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


FORBIDDEN_SUFFIXES = {
    ".ipynb", ".pt", ".pth", ".pkl", ".npz", ".png", ".jpg", ".jpeg",
    ".pdf", ".csv", ".log", ".pyc",
}
FORBIDDEN_PARTS = {
    "checkpoints", "notebooks", "slurm_experiments", "slurm_logs",
    "heuristic_results", "reward_matrices", ".venv", ".python312",
    "__pycache__", "tikz",
    ".git", ".hg", ".svn", ".ssh", ".vscode", ".idea",
}


ALLOWED_DOMAINS = {
    "docs.astral.sh", "pypi.org", "files.pythonhosted.org",
    "download.pytorch.org", "download-r2.pytorch.org", "www.w3.org",
}
EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PERSONAL_PATH = re.compile(r"/(?:[uh]|home|Users)/[A-Za-z0-9_.-]+")
URL = re.compile(r"https?://[^\s\"'<>)}]+")
CONTACT_METADATA = re.compile(
    r"^(?:Author|Author-email|Maintainer|Maintainer-email|Home-page|Project-URL):\s*\S",
    re.M,
)
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def _check_anonymous_content(files: dict[str, bytes]) -> None:
    for name, payload in files.items():
        text = payload.decode("utf-8")
        if EMAIL.search(text) or PERSONAL_PATH.search(text):
            raise ValueError(f"Contact detail or personal path in {name}")
        for match in URL.finditer(text):
            if urlsplit(match.group()).netloc not in ALLOWED_DOMAINS:
                raise ValueError(f"Unapproved external link in {name}")
        if name.endswith(("METADATA", "PKG-INFO")):
            header = text.split("\n\n", 1)[0]
            if CONTACT_METADATA.search(header):
                raise ValueError(f"Identifying package metadata in {name}")
            if "Name: task-collection\n" not in header + "\n":
                raise ValueError(f"Unexpected package name in {name}")


def check_archive(path: Path, *, anonymous: bool = False) -> int:
    files = {}
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if anonymous:
                if archive.comment:
                    raise ValueError("ZIP archive comment must be empty")
                for member in archive.infolist():
                    if member.date_time != ZIP_EPOCH or member.extra or member.comment:
                        raise ValueError("ZIP member metadata must be normalized")
                    if (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("Symbolic links are not allowed")
                    if not member.is_dir():
                        files[member.filename] = archive.read(member)
    else:
        with tarfile.open(path, "r:gz") as archive:
            names = [member.name for member in archive.getmembers() if member.isfile()]
            if anonymous:
                with path.open("rb") as stream:
                    header = stream.read(10)
                if len(header) != 10 or header[3] != 0 or header[4:8] != bytes(4):
                    raise ValueError("Gzip metadata must be normalized")
                if archive.pax_headers:
                    raise ValueError("Global PAX metadata is not allowed")
                for member in archive.getmembers():
                    if not (member.isfile() or member.isdir()):
                        raise ValueError("Only regular files and directories are allowed")
                    if member.uid or member.gid or member.uname or member.gname or member.mtime:
                        raise ValueError("Tar owner/group/time metadata must be normalized")
                    if set(member.pax_headers) - {"path"}:
                        raise ValueError("Unexpected PAX metadata")
                    if member.isfile():
                        files[member.name] = archive.extractfile(member).read()
    if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
           or "\\" in name for name in names):
        raise ValueError("Unsafe archive member path")
    forbidden = [
        name for name in names
        if PurePosixPath(name).suffix.lower() in FORBIDDEN_SUFFIXES
        or FORBIDDEN_PARTS.intersection(PurePosixPath(name).parts)
    ]
    if forbidden:
        raise ValueError(f"Forbidden artifacts in {path}: {forbidden}")
    for required in (
        "task_collection/train.py", "task_collection/reproduce.py",
        "task_collection/configs/examples/smoke_cpu.yaml",
        "task_collection/configs/examples/centralized_width64_scaling.yaml",
    ):
        if not any(name.endswith(required) for name in names):
            raise ValueError(f"Missing {required} in {path}")
    if anonymous:
        _check_anonymous_content(files)
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path, nargs="?", default=Path("dist"))
    parser.add_argument("--anonymous", action="store_true", help="Check contents and normalized metadata")
    args = parser.parse_args()
    artifacts = sorted(args.dist_dir.glob("*.whl")) + sorted(args.dist_dir.glob("*.tar.gz"))
    if not artifacts:
        parser.error(f"No distribution archives found in {args.dist_dir}")
    for artifact in artifacts:
        count = check_archive(artifact, anonymous=args.anonymous)
        status = "; anonymous metadata/content checks passed" if args.anonymous else ""
        print(f"{artifact.name}: {count} files; no generated artifacts{status}")


if __name__ == "__main__":
    main()
