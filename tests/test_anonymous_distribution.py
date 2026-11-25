"""Regression tests for anonymous source and wheel archive generation."""

import io
import runpy
import tarfile
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
check_archive = runpy.run_path(str(ROOT / "scripts/check_distribution.py"))["check_archive"]
normalize_archive = runpy.run_path(str(ROOT / "scripts/normalize_distribution.py"))["normalize_archive"]

REQUIRED = {
    "task_collection/train.py": b"# training entry point\n",
    "task_collection/reproduce.py": b"# reproduction entry point\n",
    "task_collection/configs/examples/smoke_cpu.yaml": b"env: {}\n",
    "task_collection/configs/examples/centralized_width64_scaling.yaml": b"env: {}\n",
}


def create_archive(tmp_path, kind, extra=None):
    files = dict(REQUIRED)
    files.update(extra or {})
    if kind == "wheel":
        path = tmp_path / "task_collection-0.1.1-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as target:
            target.comment = b"local build comment"
            for name, payload in files.items():
                member = zipfile.ZipInfo(name, (2026, 1, 1, 12, 0, 0))
                member.comment = b"local member comment"
                member.extra = b"\x55\x54\x01\x00\x00"
                target.writestr(member, payload)
    else:
        path = tmp_path / "task_collection-0.1.1.tar.gz"
        with tarfile.open(path, "w:gz") as target:
            for name, payload in files.items():
                member = tarfile.TarInfo("task_collection-0.1.1/" + name)
                member.uid, member.gid = 123, 456
                member.uname, member.gname = "local_user", "local_group"
                member.mtime = 1000
                member.size = len(payload)
                member.pax_headers = {"mtime": "1000.1"}
                target.addfile(member, io.BytesIO(payload))
    return path


def read_payloads(path, kind):
    if kind == "wheel":
        with zipfile.ZipFile(path) as source:
            return {item.filename: source.read(item) for item in source.infolist()}
    with tarfile.open(path, "r:gz") as source:
        return {item.name: source.extractfile(item).read() for item in source if item.isfile()}


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_normalization_clears_metadata_and_preserves_all_payloads(tmp_path, kind):
    path = create_archive(tmp_path, kind)
    before = read_payloads(path, kind)
    with pytest.raises(ValueError, match="metadata|comment"):
        check_archive(path, anonymous=True)
    normalize_archive(path)
    assert read_payloads(path, kind) == before
    assert check_archive(path, anonymous=True) == len(REQUIRED)
    normalized = path.read_bytes()
    normalize_archive(path)
    assert path.read_bytes() == normalized


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("payload, message", [
    (b"Contact: reviewer" + bytes([64]) + b"example.test", "Contact detail"),
    (b"directory: /" + b"home/local_user/code", "personal path"),
    (b"link: https:" + b"//example.test/project", "Unapproved external link"),
])
def test_anonymous_check_rejects_content_leaks(tmp_path, kind, payload, message):
    path = create_archive(tmp_path, kind, {"README.md": payload})
    normalize_archive(path)
    with pytest.raises(ValueError, match=message):
        check_archive(path, anonymous=True)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_package_contact_metadata_is_rejected(tmp_path, kind):
    path = create_archive(tmp_path, kind, {
        "task_collection-0.1.1.dist-info/METADATA":
            b"Name: task-collection\nAuthor: local_author\n\nDescription\n",
    })
    normalize_archive(path)
    with pytest.raises(ValueError, match="Identifying package metadata"):
        check_archive(path, anonymous=True)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_third_party_dependency_links_are_allowed(tmp_path, kind):
    path = create_archive(tmp_path, kind, {
        "README.md": b"https://docs.astral.sh/uv/\nhttps://download.pytorch.org/whl/cpu\n",
    })
    normalize_archive(path)
    assert check_archive(path, anonymous=True) == len(REQUIRED) + 1


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_normalizer_rejects_unsafe_member_names_without_replacing_original(tmp_path, kind):
    path = create_archive(tmp_path, kind, {"../escape.py": b"unsafe\n"})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Unsafe archive member"):
        normalize_archive(path)
    assert path.read_bytes() == before
