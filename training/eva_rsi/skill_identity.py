"""Explicit RSI sidecar binding to reopened, unchanged actor skill mounts.

This module neither edits historical evidence nor materializes new skills. The
stable content identity is an additional adapter-level comparison, not a new
canonical catalog, a skill-use claim, or permission to accept different bodies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eva_agent.codex_pipeline.skill_identity import reopen_skill_content_identity, require
from eva_agent.codex_pipeline.skills import _normalized_absolute, _stable_regular_bytes
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex
from training.automedbench_lite.adapter import canonical as actor_canonical

SCHEMA = "eva.rsi-skill-content-binding.v1"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "plugins/evamed-codex/references/legacy-skill-manifest.v1.json"


def build_skill_content_binding(run_root: Path, *, legacy_manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """Independently reopen an actual track attempt and every mounted skill."""
    root = _normalized_absolute(run_root, label="skill source actor run")
    attempt_path = root / "track-rollouts/attempt.json"
    raw = _stable_regular_bytes(attempt_path, label="retained actor attempt")
    attempt = json.loads(raw)
    require(attempt.get("schema") == "eva.automedbench-track-attempt.v1", "skill_source_attempt_schema")
    require(attempt.get("document_blake3") == blake3_bytes(actor_canonical(
        {k: v for k, v in attempt.items() if k != "document_blake3"})),
            "skill_source_attempt_commitment")
    inventory = attempt["skill_inventory"]
    require(isinstance(inventory, list) and len(inventory) == 24, "skill_source_inventory_count")
    mount_roots = {Path(row["path"]).parent.parent for row in inventory}
    require(len(mount_roots) == 1, "skill_source_mount_roots_differ")
    mount_root = mount_roots.pop()
    require(mount_root.parent == root / "track-rollouts/skill-preflight", "skill_source_mount_outside_attempt")
    manifest = _normalized_absolute(legacy_manifest_path, label="skill source manifest")
    proof = reopen_skill_content_identity(inventory=inventory,
        mounted_catalog_blake3=attempt["verified_skill_catalog_blake3"],
        legacy_manifest_path=manifest, materialization_root=mount_root)
    core = {"schema": SCHEMA, "benchmark_run_root": str(root),
        "attempt": {"path": str(attempt_path), "file_blake3": blake3_bytes(raw),
                    "document_blake3": attempt["document_blake3"]},
        "legacy_manifest": {"path": str(manifest), "file_blake3": proof["legacy_manifest_file_blake3"]},
        "mounted_catalog_blake3": proof["mounted_catalog_blake3"],
        "content_identity_blake3": proof["content_identity_blake3"], "verified_content": proof,
        "original_evidence_mutated": False, "skill_use_inferred": False}
    return {**core, "document_blake3": blake3_hex(core)}


def verify_skill_content_binding(sidecar_path: Path, *, expected_run_root: Path,
                                 expected_content_id: str) -> dict:
    """Recompute rather than trusting either sidecar hashes or cached claims."""
    path = _normalized_absolute(sidecar_path, label="skill content binding")
    raw = _stable_regular_bytes(path, label="skill content binding")
    supplied = json.loads(raw)
    require(supplied.get("schema") == SCHEMA, "skill_content_binding_schema")
    actual = build_skill_content_binding(expected_run_root,
        legacy_manifest_path=Path(supplied["legacy_manifest"]["path"]))
    require(supplied == actual, "skill_content_binding_reopen_differs")
    require(actual["content_identity_blake3"] == expected_content_id, "skill_content_identity_differs")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = build_skill_content_binding(args.run_root, legacy_manifest_path=args.legacy_manifest)
    # Explicit caller-selected, exclusive sidecar creation; no actor mutations.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("schema", "mounted_catalog_blake3", "content_identity_blake3")}))


if __name__ == "__main__":
    main()
