#!/usr/bin/env python3
"""Build or verify the pinned, external-only legacy EvaMed skill manifest.

The source declares ``license: other`` and private research data, so this tool
does not copy skill bodies into the open-source plugin.  It records exact byte
digests and catalog provenance; the MCP bridge mounts them only after reopening
and verifying the pinned source tree.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PLUGIN_ROOT.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
REPOSITORY_PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"
if (
    REPOSITORY_PYTHON.is_file()
    and Path(sys.prefix).resolve() != (REPOSITORY_ROOT / ".venv").resolve()
):
    os.execv(
        str(REPOSITORY_PYTHON),
        (str(REPOSITORY_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]),
    )
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex  # noqa: E402


REVISION = "79dd2a31f5f"
DEFAULT_SOURCE = (
    REPOSITORY_ROOT.parent
    / "rlevo-med-research"
    / "harness"
    / "source"
    / "rlevo-Med-RL-data"
    / f"rev-{REVISION}"
)
DEFAULT_OUTPUT = PLUGIN_ROOT / "references" / "legacy-skill-manifest.v1.json"


class ManifestError(ValueError):
    pass


def _catalog_rows(source: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for catalog_path in sorted((source / "harnesses").glob("H*/catalog.json")):
        catalog = json.loads(catalog_path.read_bytes())
        if catalog.get("schema") != "batmed-training-aligned-skill-catalog-v1":
            raise ManifestError(f"unexpected skill catalog: {catalog_path}")
        harness = catalog_path.parent.name
        for item in catalog.get("skills", ()):
            relative = Path("harnesses") / harness / str(item["document_path"])
            skill_path = source / relative
            raw = skill_path.read_bytes()
            digest = blake3_bytes(raw)
            skill_id = str(item["skill_id"])
            if skill_path.parent.name != skill_id:
                raise ManifestError(f"skill path identity differs: {relative}")
            stages = tuple(str(stage) for stage in item["allowed_stages"])
            metadata = {
                "skill_id": skill_id,
                "description": str(item["description"]),
                "allowed_stages": stages,
                "content_blake3": digest,
                "bytes": len(raw),
            }
            row = rows.setdefault(
                digest,
                {
                    **metadata,
                    "canonical_source_path": relative.as_posix(),
                    "source_paths": [],
                    "catalog_paths": [],
                    "harness_revisions": [],
                },
            )
            for key in ("skill_id", "description", "allowed_stages", "bytes"):
                if row[key] != metadata[key]:
                    raise ManifestError(
                        f"same content digest has conflicting {key}: {digest}"
                    )
            row["source_paths"].append(relative.as_posix())
            row["catalog_paths"].append(
                catalog_path.relative_to(source).as_posix()
            )
            row["harness_revisions"].append(harness)
    return rows


def build(source: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    readme = source / "README.md"
    if "license: other" not in readme.read_text(encoding="utf-8"):
        raise ManifestError("expected pinned source license marker is absent")
    rows = _catalog_rows(source)
    skills = []
    for digest, row in sorted(rows.items(), key=lambda item: item[1]["skill_id"]):
        row["source_paths"] = sorted(set(row["source_paths"]))
        row["catalog_paths"] = sorted(set(row["catalog_paths"]))
        row["harness_revisions"] = sorted(set(row["harness_revisions"]))
        skills.append(row)
    occurrences = sum(len(row["source_paths"]) for row in skills)
    if occurrences != 174 or len(skills) != 24:
        raise ManifestError(
            f"legacy inventory differs: {occurrences} occurrences / {len(skills)} unique"
        )
    core = {
        "schema": "eva.evamed-legacy-skill-manifest.v1",
        "source_repository": "rlevo-Med-RL-data",
        "source_revision": REVISION,
        "source_root_hint": (
            "rlevo-med-research/harness/source/rlevo-Med-RL-data/"
            f"rev-{REVISION}"
        ),
        "source_readme_blake3": blake3_bytes(readme.read_bytes()),
        "source_license_marker": "other",
        "redistribution": "external-only-license-unresolved",
        "occurrence_count": occurrences,
        "unique_content_count": len(skills),
        "skills": skills,
    }
    return {**core, "manifest_blake3": blake3_hex(core)}


def _bytes(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    built = build(args.source)
    payload = _bytes(built)
    if args.verify:
        if not args.output.is_file() or args.output.read_bytes() != payload:
            raise ManifestError("persisted legacy skill manifest differs")
        print(
            json.dumps(
                {
                    "status": "verified",
                    "occurrences": built["occurrence_count"],
                    "unique": built["unique_content_count"],
                    "manifest_blake3": built["manifest_blake3"],
                },
                sort_keys=True,
            )
        )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
