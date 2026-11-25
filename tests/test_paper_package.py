"""Check the value-learning package's import and configuration surface."""

import importlib
from importlib.resources import files
import pkgutil

import task_collection
from task_collection.config import load_config
from task_collection.enums import AlgorithmName


def test_all_runtime_modules_import_without_optional_extensions():
    for module in pkgutil.walk_packages(task_collection.__path__, "task_collection."):
        if ".tests" not in module.name:
            importlib.import_module(module.name)


def test_reference_and_examples_select_value_learning():
    configs = files("task_collection").joinpath("configs")
    paths = [configs.joinpath("reference.yaml")]
    paths.extend(path for path in configs.joinpath("examples").iterdir()
                 if path.name.endswith(".yaml"))
    for path in paths:
        assert load_config(str(path)).train.algorithm.name == AlgorithmName.VALUE
    assert list(AlgorithmName) == [AlgorithmName.VALUE]
