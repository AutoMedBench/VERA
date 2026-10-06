"""Small public-only I/O helpers for host-preflighted prescribed-model jobs.

The caller owns full input/model manifest verification, process isolation, GPU
admission, and audit copies of these helper sources AND emitted artifact bytes.
No scorer directory or reference-data argument is accepted here.
"""
from __future__ import annotations

from pathlib import Path
import re

from blake3 import blake3


def public_input(binding, case, filename):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", case):
        raise ValueError("Invalid public case ID")
    relative = f"inputs/{case}/{filename}"
    rows = [row for row in binding["selected"][case] if row["path"] == relative]
    if len(rows) != 1:
        raise ValueError("Expected exactly one committed public track input")
    workspace = Path(binding["workspace"]).resolve(strict=True)
    path = (workspace / relative).resolve(strict=True)
    if not path.is_relative_to(workspace / "inputs" / case):
        raise ValueError("Public input escapes its bound case")
    return path


def case_output(binding, output_root, case):
    workspace = Path(binding["workspace"]).resolve(strict=True)
    output_root = Path(output_root).resolve()
    if not output_root.is_relative_to(workspace / "outputs"):
        raise ValueError("Analysis output must stay inside bound workspace outputs")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", case):
        raise ValueError("Invalid public case ID")
    publisher = binding.get("_publication")
    path = publisher.case_path(case) if publisher else output_root / case
    path.mkdir(parents=True, exist_ok=False)
    return path


def artifact(binding, path):
    if binding.get("_publication"):
        return binding["_publication"].artifact(path)
    digest = blake3()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return {"path": path.relative_to(Path(binding["workspace"]).resolve()).as_posix(),
            "bytes": path.stat().st_size, "blake3": digest.hexdigest()}
