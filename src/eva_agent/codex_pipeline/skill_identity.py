"""Read-only content identity for an existing canonical skill mount catalog.

The canonical catalog is reconstructed and checked unchanged. Only its runtime
mount paths are excluded from the additional, explicitly versioned identity.
No skill is materialized, edited, loaded into a model, or counted as used here.
"""
from __future__ import annotations

import json
from pathlib import Path
import stat
from typing import Mapping, Sequence

from eva_agent.codex_runtime import CodexSkill
from eva_agent.pipeline.contracts import Stage
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value, is_blake3
from .skills import (ACTOR_SKILL_MATERIALIZATION_SCHEMA, ACTOR_SKILL_PATH_POLICY,
    LEGACY_SKILL_MANIFEST_SCHEMA, LEGACY_SKILL_MANIFEST_BLAKE3, LEGACY_SKILL_SOURCE_REVISION,
    LEGACY_SKILL_UNIQUE_CONTENTS, LEGACY_SKILL_OCCURRENCES, _normalized_absolute,
    _relative_skill_path, _stable_regular_bytes)

SCHEMA = "eva.codex-skill-content-identity.v1"


class SkillContentIdentityError(ValueError):
    pass


def require(value, code):
    if not value:
        raise SkillContentIdentityError(code)


def reopen_skill_content_identity(*, inventory: Sequence[Mapping], mounted_catalog_blake3: str,
                                  legacy_manifest_path: Path, materialization_root: Path) -> dict:
    """Verify all 24 legacy skills plus native bootstrap from existing mounts."""
    require(is_blake3(mounted_catalog_blake3), "mounted_catalog_digest_invalid")
    manifest_path = _normalized_absolute(legacy_manifest_path, label="skill identity manifest")
    raw_manifest = _stable_regular_bytes(manifest_path, label="skill identity manifest")
    manifest = json.loads(raw_manifest)
    manifest_core = {k: v for k, v in manifest.items() if k != "manifest_blake3"}
    require(manifest.get("manifest_blake3") == LEGACY_SKILL_MANIFEST_BLAKE3 == blake3_hex(manifest_core)
            and manifest.get("schema") == LEGACY_SKILL_MANIFEST_SCHEMA
            and manifest.get("source_revision") == LEGACY_SKILL_SOURCE_REVISION
            and manifest.get("occurrence_count") == LEGACY_SKILL_OCCURRENCES
            and manifest.get("unique_content_count") == LEGACY_SKILL_UNIQUE_CONTENTS
            and manifest.get("source_license_marker") == "other"
            and manifest.get("redistribution") == "external-only-license-unresolved",
            "pinned_legacy_manifest_differs")
    rows = canonical_value(inventory)
    expected = manifest["skills"]
    require(len(rows) == len(expected) == LEGACY_SKILL_UNIQUE_CONTENTS, "skill_inventory_count_differs")
    root = _normalized_absolute(materialization_root, label="retained skill materialization")
    require(root.is_dir() and stat.S_IMODE(root.stat().st_mode) == 0o500,
            "materialization_directory_mode_differs")
    entries = sorted(root.iterdir())
    require(len(entries) == LEGACY_SKILL_UNIQUE_CONTENTS + 1, "materialization_inventory_differs")
    native_path = entries[0] / "SKILL.md"
    native_bytes = _stable_regular_bytes(_normalized_absolute(native_path, label="native mounted skill"),
                                         label="native mounted skill")
    native_digest = blake3_bytes(native_bytes)
    native_id = "eva-workflow/stage-rollout"
    payload_rows = [{"ordinal": 0, "skill_id": native_id, "name": "stage-rollout",
        "relative_path": f"000-{native_digest}/SKILL.md", "content_blake3": native_digest, "bytes": len(native_bytes)}]
    require(entries[0].name == f"000-{native_digest}", "native_materialization_path_differs")
    descriptions = []
    previous = ""
    stages = {stage.value: [] for stage in Stage}
    for index, (row, source) in enumerate(zip(rows, expected, strict=True), start=1):
        require(set(row) == {"skill_id", "description", "allowed_stages", "source_path", "content_blake3", "bytes", "path"},
                "actual_inventory_fields_differ")
        for key in ("skill_id", "description", "allowed_stages", "content_blake3", "bytes"):
            require(row[key] == source[key], "skill_body_or_metadata_differs")
        require(row["skill_id"] > previous and row["allowed_stages"] == sorted(set(row["allowed_stages"]))
                and row["allowed_stages"] and all(stage in stages for stage in row["allowed_stages"]),
                "skill_order_or_stage_grants_differ")
        previous = row["skill_id"]
        relative = _relative_skill_path(source["canonical_source_path"])
        source_path = Path(row["source_path"])
        require(source_path.is_absolute() and ".." not in source_path.parts
                and source_path.parts[-len(relative.parts):] == relative.parts, "legacy_source_reference_differs")
        path = root / f"{index:03d}-{row['content_blake3']}" / "SKILL.md"
        require(row["path"] == str(path), "actual_mount_path_differs")
        data = _stable_regular_bytes(_normalized_absolute(path, label="retained legacy skill"), label="retained legacy skill")
        require(len(data) == row["bytes"] and blake3_bytes(data) == row["content_blake3"], "retained_skill_bytes_differ")
        payload_rows.append({"ordinal": index, "skill_id": row["skill_id"], "name": row["skill_id"],
            "relative_path": path.relative_to(root).as_posix(), "content_blake3": row["content_blake3"], "bytes": len(data)})
        descriptions.append({key: row[key] for key in ("skill_id", "description", "allowed_stages")})
        for stage in row["allowed_stages"]: stages[stage].append(index)
    require({path.name for path in entries} == {Path(row["relative_path"]).parts[0] for row in payload_rows},
            "materialization_extra_or_missing_entry")
    for path in entries:
        require(not path.is_symlink() and path.is_dir() and stat.S_IMODE(path.stat().st_mode) == 0o500
                and {p.name for p in path.iterdir()} == {"SKILL.md"}
                and stat.S_IMODE((path / "SKILL.md").stat().st_mode) == 0o400,
                "materialization_topology_or_permissions_differs")
    materialization = {"schema": ACTOR_SKILL_MATERIALIZATION_SCHEMA,
                       "path_policy": ACTOR_SKILL_PATH_POLICY, "skills": payload_rows}
    materialization_digest = blake3_hex(materialization)
    require(root.name == "blake3-" + materialization_digest, "materialization_commitment_differs")
    catalog_entries = [canonical_value(CodexSkill(row["skill_id"], row["name"], str(root / row["relative_path"]),
                                                row["content_blake3"]).catalog_entry()) for row in payload_rows]
    catalog = {"schema": "eva.codex-actor-skill-mount-catalog.v1", "legacy_manifest_blake3": manifest["manifest_blake3"],
        "materialization": {"schema": ACTOR_SKILL_MATERIALIZATION_SCHEMA, "root": str(root),
            "blake3": materialization_digest, "path_policy": ACTOR_SKILL_PATH_POLICY},
        "native_skill": catalog_entries[0],
        "stage_mounts": {stage: [catalog_entries[0], *(catalog_entries[index] for index in stages[stage])]
                         for stage in sorted(stages)}}
    require(blake3_hex(catalog) == mounted_catalog_blake3, "original_mounted_catalog_not_reconstructed")
    stable_catalog = {**catalog,
        "materialization": {k: v for k, v in catalog["materialization"].items() if k != "root"},
        "native_skill": {k: v for k, v in catalog["native_skill"].items() if k != "path"},
        "stage_mounts": {stage: [{k: v for k, v in row.items() if k != "path"} for row in values]
                         for stage, values in catalog["stage_mounts"].items()}}
    content = {"schema": SCHEMA, "mount_path_independent_catalog": stable_catalog,
               "legacy_skill_metadata": descriptions}
    core = {"schema": SCHEMA, "content_identity_blake3": blake3_hex(content), "content": content,
        "mounted_catalog_blake3": mounted_catalog_blake3, "reconstructed_mounted_catalog": catalog,
        "materialization_blake3": materialization_digest, "legacy_manifest_blake3": manifest["manifest_blake3"],
        "legacy_manifest_file_blake3": blake3_bytes(raw_manifest), "all_skill_files_reopened": len(payload_rows),
        "all_skill_bytes_reopened": sum(row["bytes"] for row in payload_rows),
        "excluded_identity_fields": ["materialization.root", "native_skill.path", "stage_mounts.*[].path"],
        "bodies_retained_in_sidecar": False, "catalog_mutated": False, "skill_use_inferred": False}
    return {**core, "document_blake3": blake3_hex(core)}
