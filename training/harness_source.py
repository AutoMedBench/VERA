"""Select the separately maintained harness checkout before importing its package.

The three branches share a package name, so installing all three into one venv
is not a dependency selection mechanism. No package installation occurs here.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


def runtime_paths(root: Path, environment=None) -> list[str]:
    env = os.environ if environment is None else environment
    root = Path(root).resolve(strict=True)
    selected = env.get("EVA_HARNESS_ROOT")
    if not selected:
        return [str(root), str(root / "src")]
    harness = Path(selected).resolve(strict=True)
    for name in ("runtime.py", "policy_budget.py", "research_memory.py"):
        if not (harness / "src/eva_agent/codex_runtime" / name).is_file():
            raise ValueError("selected harness lacks required v1.2 runtime modules")
    return list(dict.fromkeys((str(root), str(harness / "src"), str(root / "src"))))


def activate_harness(root: Path) -> list[str]:
    paths = runtime_paths(root)
    sys.path[:0] = paths
    if os.environ.get("EVA_HARNESS_ROOT"):
        expected = Path(os.environ["EVA_HARNESS_ROOT"]).resolve() / "src/eva_agent"
        package = importlib.import_module("eva_agent")
        if Path(package.__file__).resolve().parent != expected:
            raise ValueError("another eva_agent checkout was already imported")
    return paths
