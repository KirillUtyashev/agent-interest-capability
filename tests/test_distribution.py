"""Guard the code-only release against accidentally bundled research outputs."""

import io
import runpy
import tarfile
import zipfile
from pathlib import Path

import pytest


check_archive = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "check_distribution.py")
)["check_archive"]

REQUIRED = (
    "task_collection/train.py",
    "task_collection/reproduce.py",
    "task_collection/configs/examples/smoke_cpu.yaml",
    "task_collection/configs/examples/centralized_width64_scaling.yaml",
)


def make_archive(tmp_path, kind, names):
    if kind == "wheel":
        path = tmp_path / "test.whl"
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                archive.writestr(name, "example")
    else:
        path = tmp_path / "test.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name in names:
                payload = b"example"
                member = tarfile.TarInfo("task-collection-0.1.1/" + name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    return path


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_code_only_archive_is_accepted(tmp_path, kind):
    assert check_archive(make_archive(tmp_path, kind, REQUIRED)) == len(REQUIRED)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("artifact", [
    "task_collection/checkpoint.pt",
    "task_collection/analysis.ipynb",
    "task_collection/metrics.csv",
    "task_collection/figure.png",
    "task_collection/run.log",
    "slurm_experiments/run/metadata.yaml",
])
def test_generated_artifacts_are_rejected(tmp_path, kind, artifact):
    with pytest.raises(ValueError, match="Forbidden artifacts"):
        check_archive(make_archive(tmp_path, kind, (*REQUIRED, artifact)))


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_missing_example_is_rejected(tmp_path, kind):
    with pytest.raises(ValueError, match="Missing task_collection/configs/examples"):
        check_archive(make_archive(tmp_path, kind, REQUIRED[:-1]))
