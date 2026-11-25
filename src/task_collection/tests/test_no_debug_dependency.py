"""Regression tests that task_collection stays independent from debug."""

from __future__ import annotations

import ast
from pathlib import Path


def _imports_debug(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "debug" or alias.name.startswith("debug."):
                    return True
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "debug" or node.module.startswith("debug."):
                return True
    return False


def test_task_collection_package_has_no_debug_imports():
    task_collection_root = Path(__file__).resolve().parents[1]

    for path in task_collection_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        assert not _imports_debug(tree), f"Unexpected debug import found in {path}"
