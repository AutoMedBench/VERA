#!/usr/bin/env python3
"""Offline byte, stage-grant and real search/load checks for installed skills.

This checks installed capabilities; it does not change any actor's skill mount,
permissions, native candidate, reward or production configuration.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


PLUGIN = Path(__file__).resolve().parents[1]
REPOSITORY = PLUGIN.parents[1]
BUILTINS = {
    "sandbox-construction": "construction",
    "stage-rollout": "rollout-orchestration",
    "summary-failures": "actor-recovery",
    "trajectory-sft": "admitted-trajectory-export",
    "workspace-agent-judge": "independent-evaluation",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def frontmatter(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(bool(lines) and lines[0] == "---", "skill_frontmatter_missing")
    end = lines.index("---", 1)
    values = {}
    for line in lines[1:end]:
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            values[key] = value.strip()
    require(bool(values.get("name")) and bool(values.get("description")),
            "skill_identity_missing")
    return values["name"], values["description"]


def verify(output: Path) -> dict:
    # The only materialization and temporary files belong to the requested output.
    output = output.resolve()
    require(not output.is_relative_to(REPOSITORY), "generated_output_must_be_outside_checkout")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    output.chmod(0o700)
    for directory in ("tmp", "cache", "materialized"):
        (output / directory).mkdir(mode=0o700)
        # Shared filesystems may inherit setgid despite mkdir(mode=0o700).
        (output / directory).chmod(0o700)
    os.environ.update(TMPDIR=str(output / "tmp"), XDG_CACHE_HOME=str(output / "cache"),
                      PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPOSITORY / "src"))

    from eva_agent.codex_pipeline.skills import VerifiedActorSkillCatalog
    from eva_agent.harness.contracts import HarnessContractError
    from eva_agent.harness.skills import SkillCatalog, SkillDocument
    from eva_agent.harness.tools import VALID_STAGES
    from eva_agent.pipeline import Stage
    from eva_agent.pipeline.digests import blake3_bytes, blake3_hex

    provenance = read(PLUGIN / "vendor/provenance.v1.json")
    vendor = (PLUGIN / "vendor" / provenance["root"]).resolve(strict=True)
    require(vendor.is_relative_to(PLUGIN / "vendor"), "vendor_path_escape")
    manifest_path = PLUGIN / "references/legacy-skill-manifest.v1.json"
    manifest = read(manifest_path)
    require(provenance["source_manifest_blake3"] == manifest["manifest_blake3"],
            "legacy_provenance_manifest_differs")
    required = {"README.md"}
    for row in manifest["skills"]:
        required.update(row["source_paths"])
        required.update(row["catalog_paths"])
        for relative in row["source_paths"]:
            path = vendor / relative
            require(path.resolve(strict=True) == path and path.is_file(), "legacy_source_topology")
            payload = path.read_bytes()
            require(len(payload) == row["bytes"] and blake3_bytes(payload) == row["content_blake3"],
                    "legacy_occurrence_bytes_changed")
    require({row["path"] for row in provenance["files"]} == required,
            "legacy_vendoring_inventory_differs")
    for row in provenance["files"]:
        path = vendor / row["path"]
        require(path.resolve(strict=True).is_relative_to(vendor) and not path.is_symlink(),
                "vendor_source_path_escape")
        raw = path.read_bytes()
        require(len(raw) == row["bytes"] and blake3_bytes(raw) == row["blake3"],
                "vendor_provenance_bytes_changed")
    require(len(required) == 183, "legacy_file_count_differs")

    pinned = VerifiedActorSkillCatalog(
        manifest_path=manifest_path.resolve(), legacy_source_root=vendor,
        native_stage_skill_path=(PLUGIN / "skills/stage-rollout/SKILL.md").resolve(),
        runtime_root=(output / "materialized").resolve(),
    )
    inventory = pinned.inventory()
    require(len(inventory) == 24, "legacy_unique_count_differs")
    documents, rows = [], []
    for item in inventory:
        path = Path(item["path"])
        content = path.read_text(encoding="utf-8")
        stages = frozenset(item["allowed_stages"])
        documents.append(SkillDocument(item["skill_id"], item["description"], content, stages))
        rows.append({"skill_id": item["skill_id"], "group": "legacy", "role": "stage-bound-workflow",
                     "allowed_stages": sorted(stages), "file_blake3": blake3_bytes(path.read_bytes())})
    for stage in Stage:
        expected = {row["skill_id"] for row in rows if stage.value in row["allowed_stages"]}
        require({item.skill_id for item in pinned.for_stage(stage)} == expected | {"eva-workflow/stage-rollout"},
                "legacy_stage_grants_changed")

    actual_builtin_names = {p.parent.name for p in (PLUGIN / "skills").glob("*/SKILL.md")}
    require(actual_builtin_names == set(BUILTINS), "builtin_inventory_differs")
    for folder, role in sorted(BUILTINS.items()):
        path = PLUGIN / "skills" / folder / "SKILL.md"
        name, description = frontmatter(path)
        require(name == folder, "builtin_name_differs")
        skill_id = "summary_failures" if folder == "summary-failures" else "eva-workflow/" + folder
        documents.append(SkillDocument(skill_id, description, path.read_text(encoding="utf-8")))
        rows.append({"skill_id": skill_id, "group": "plugin", "role": role,
                     "allowed_stages": sorted(VALID_STAGES), "file_blake3": blake3_bytes(path.read_bytes())})

    projects = read(PLUGIN / "skills-project/provenance.v1.json")
    require(len(projects["skills"]) == 9, "project_skill_count_differs")
    require({p.parent.name for p in (PLUGIN / "skills-project").glob("*/SKILL.md")} ==
            {row["name"] for row in projects["skills"]}, "project_skill_inventory_differs")
    for item in projects["skills"]:
        path = PLUGIN / "skills-project" / item["path"]
        require(path.resolve(strict=True).is_relative_to(PLUGIN / "skills-project") and not path.is_symlink(),
                "project_skill_path_escape")
        raw = path.read_bytes()
        require(len(raw) == item["bytes"] and blake3_bytes(raw) == item["content_blake3"],
                "project_skill_bytes_changed")
        name, description = frontmatter(path)
        require(name == item["name"], "project_skill_name_differs")
        documents.append(SkillDocument(name, description, raw.decode("utf-8")))
        rows.append({"skill_id": name, "group": "project", "role": "medical-research-workflow",
                     "allowed_stages": sorted(VALID_STAGES), "file_blake3": blake3_bytes(raw)})
    require(len(documents) == len({d.skill_id for d in documents}) == 38, "installed_inventory_differs")

    tools = {item.name: item for item in SkillCatalog(documents).tool_definitions()}

    async def exercise():
        searches = loads = denied = 0
        for document in documents:
            for stage in sorted(document.allowed_stages):
                found = await tools["search_skills"].handler({"query": document.skill_id, "stage": stage})
                require(document.skill_id in {row["skill_id"] for row in found["matches"]}, "skill_not_discoverable")
                value = await tools["load_skill"].handler({"skill_id": document.skill_id, "stage": stage})
                require(value["content"] == document.content and value["content_blake3"] == blake3_hex(document.content),
                        "skill_load_content_differs")
                searches += 1
                loads += 1
            for stage in sorted(VALID_STAGES - document.allowed_stages):
                try:
                    await tools["load_skill"].handler({"skill_id": document.skill_id, "stage": stage})
                except HarnessContractError:
                    denied += 1
                else:
                    raise ValueError("forbidden_stage_skill_load_succeeded")
        return {"searches": searches, "loads": loads, "forbidden_stage_loads_rejected": denied}

    checks = asyncio.run(exercise())
    # Unpromoted private experiments are intentionally not part of this export.
    require(not (PLUGIN / "skills-experimental").exists(),
            "experimental_assets_are_not_part_of_this_release")
    experimental = {"skills": []}
    require(not any("artifact-handoff" in document.skill_id for document in documents),
            "experimental_skill_entered_installed_catalog")

    result = {"schema": "eva.installed-skill-readiness.v1", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "legacy_manifest_blake3": pinned.manifest_blake3,
              "legacy_occurrences_verified": 174, "legacy_catalog_files_verified": 8,
              "legacy_unique_skills": 24, "plugin_skills": 5, "project_skills": 9,
              "installed_unique_capabilities": 38, "experimental_drafts_excluded": len(experimental["skills"]),
              "all_body_bytes_match_export_manifest": True, "checks": checks, "skills": rows,
              "api_calls": 0, "gpu_jobs": 0, "production_modified": False,
              "automatic_actor_mount_changes": False, "promotion_claimed": False,
              "scope": "Offline installation/discovery/load readiness. Workflow roles remain distinct; local test availability is not a rollout permission grant or a model-performance result."}
    result["document_blake3"] = blake3_hex(result)
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return {"passed": True, "receipt": str(output / "receipt.json"),
            "document_blake3": result["document_blake3"], "installed_unique_capabilities": 38, **checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New workspace-owned output directory outside the checkout")
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.output), sort_keys=True))
