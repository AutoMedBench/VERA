"""Select legacy Python code independently from immutable medical authorities.

This module only resolves a source directory. It never imports the legacy
package, changes PYTHONPATH, or changes any authority/data/skill location.
LegacyExecutionBindingResolver remains the explicit package loader.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def legacy_python_root(
    authority_root: str | Path, environment: Mapping[str, str] | None = None
) -> Path:
    """Return the historical sibling source or one explicitly selected tree."""

    environment = os.environ if environment is None else environment
    selected = environment.get("EVA_LEGACY_PYTHON_ROOT")
    if selected is None:
        # Preserve the historical default and its resolver-owned validation.
        return Path(authority_root) / "src"
    if not isinstance(selected, str) or not selected or not Path(selected).is_absolute():
        raise ValueError("legacy Python root must be an explicit absolute source directory")
    try:
        root = Path(selected).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("legacy Python source directory is unavailable") from exc
    package = root / "rlevo_med_research"
    if not root.is_dir() or not package.is_dir() or not (package / "__init__.py").is_file():
        raise ValueError("legacy Python root must contain the rlevo_med_research package")
    return root
