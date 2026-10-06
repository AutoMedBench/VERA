"""Synthetic skill bytes only: no external licensed corpus or providers."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from eva_agent.codex_pipeline import skill_identity as identity
from eva_agent.codex_pipeline.skills import _materialize_skill_payloads
from eva_agent.codex_runtime import CodexSkill
from eva_agent.pipeline.contracts import Stage
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex


@pytest.fixture
def skill_mount_factory(tmp_path, monkeypatch):
    payloads = [{"skill_id": "eva-workflow/stage-rollout", "name": "stage-rollout", "payload": b"Synthetic native skill."}]
    source_rows = []
    for number in range(24):
        skill_id = f"fixture-{number:02}"
        payload = f"Synthetic skill {number}. No medical content.".encode()
        payloads.append({"skill_id": skill_id, "name": skill_id, "payload": payload})
        source_rows.append({"skill_id": skill_id, "description": f"Synthetic description {number} — fixture",
            "allowed_stages": ["S1", "S2"] if number % 2 else ["S3"],
            "canonical_source_path": f"public-fixtures/{skill_id}/SKILL.md",
            "content_blake3": blake3_bytes(payload), "bytes": len(payload)})
    core = {"schema": identity.LEGACY_SKILL_MANIFEST_SCHEMA, "source_revision": identity.LEGACY_SKILL_SOURCE_REVISION,
        "occurrence_count": 174, "unique_content_count": 24, "source_license_marker": "other",
        "redistribution": "external-only-license-unresolved", "skills": source_rows}
    manifest = {**core, "manifest_blake3": blake3_hex(core)}
    monkeypatch.setattr(identity, "LEGACY_SKILL_MANIFEST_BLAKE3", manifest["manifest_blake3"])
    manifest_path = tmp_path / "synthetic-manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def make(name, *, native=b"Synthetic native skill."):
        actor_root = tmp_path / name
        runtime = actor_root / "track-rollouts/skill-preflight"
        runtime.mkdir(parents=True, mode=0o700)
        # Shared filesystems may inherit setgid; the catalog requires exact 0700.
        runtime.chmod(0o700)
        values = deepcopy(payloads)
        values[0]["payload"] = native
        for row in values: row["content_blake3"] = blake3_bytes(row["payload"])
        root, digest, paths = _materialize_skill_payloads(runtime, tuple(values))
        inventory = [{k: v for k, v in row.items() if k != "canonical_source_path"} | {
            "source_path": str(tmp_path / "source" / row["canonical_source_path"]), "path": str(paths[row["skill_id"]])}
            for row in source_rows]
        entries = {row["skill_id"]: CodexSkill(row["skill_id"], row["name"], str(paths[row["skill_id"]]),
            row["content_blake3"]).catalog_entry() for row in values}
        native_entry = entries["eva-workflow/stage-rollout"]
        catalog = {"schema": "eva.codex-actor-skill-mount-catalog.v1",
            "legacy_manifest_blake3": manifest["manifest_blake3"],
            "materialization": {"schema": identity.ACTOR_SKILL_MATERIALIZATION_SCHEMA, "root": str(root),
                "blake3": digest, "path_policy": identity.ACTOR_SKILL_PATH_POLICY},
            "native_skill": native_entry,
            "stage_mounts": {stage.value: [native_entry, *(entries[row["skill_id"]] for row in source_rows
                if stage.value in row["allowed_stages"])] for stage in Stage}}
        return {"inventory": inventory, "mounted_catalog_blake3": blake3_hex(catalog),
                "legacy_manifest_path": manifest_path, "materialization_root": root}
    return make


def test_equal_bytes_different_mounts_preserve_original_catalogs(skill_mount_factory):
    first, second = skill_mount_factory("first"), skill_mount_factory("second")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)
              for source in (first, second) for path in source["materialization_root"].rglob("SKILL.md")}
    a, b = (identity.reopen_skill_content_identity(**source) for source in (first, second))
    assert a["mounted_catalog_blake3"] != b["mounted_catalog_blake3"]
    assert a["content_identity_blake3"] == b["content_identity_blake3"]
    assert a["all_skill_files_reopened"] == b["all_skill_files_reopened"] == 25
    assert a["content"] == b["content"]
    assert a["reconstructed_mounted_catalog"]["materialization"]["root"] != b["reconstructed_mounted_catalog"]["materialization"]["root"]
    assert all((path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode) == value for path, value in before.items())
    assert a["catalog_mutated"] is a["skill_use_inferred"] is False


@pytest.mark.parametrize("native", [False, True])
def test_changed_actual_body_is_rejected(skill_mount_factory, native):
    args = skill_mount_factory("modified")
    victim = (sorted(args["materialization_root"].iterdir())[0] / "SKILL.md" if native
              else Path(args["inventory"][0]["path"]))
    victim.chmod(0o600)
    victim.write_bytes(b"Changed actual mounted bytes.")
    victim.chmod(0o400)
    with pytest.raises(identity.SkillContentIdentityError, match="native_materialization_path_differs|retained_skill_bytes_differ"):
        identity.reopen_skill_content_identity(**args)


@pytest.mark.parametrize("field,value", [("description", "Altered description"), ("allowed_stages", ["S5"])])
def test_changed_metadata_not_hidden_by_equal_bodies(skill_mount_factory, field, value):
    args = skill_mount_factory("changed-metadata")
    args["inventory"][0][field] = value
    with pytest.raises(identity.SkillContentIdentityError, match="skill_body_or_metadata_differs"):
        identity.reopen_skill_content_identity(**args)


def test_self_consistent_manifest_rewrite_rejected(skill_mount_factory):
    args = skill_mount_factory("changed-manifest")
    manifest = json.loads(args["legacy_manifest_path"].read_text())
    manifest["skills"][0]["description"] = args["inventory"][0]["description"] = "Changed metadata"
    manifest["manifest_blake3"] = blake3_hex({k: v for k, v in manifest.items() if k != "manifest_blake3"})
    args["legacy_manifest_path"].write_text(json.dumps(manifest))
    with pytest.raises(identity.SkillContentIdentityError, match="pinned_legacy_manifest_differs"):
        identity.reopen_skill_content_identity(**args)


def test_valid_native_body_change_yields_different_content_identity(skill_mount_factory):
    a = identity.reopen_skill_content_identity(**skill_mount_factory("native-one"))
    b = identity.reopen_skill_content_identity(**skill_mount_factory("native-two", native=b"Different native source version."))
    assert a["content_identity_blake3"] != b["content_identity_blake3"]


def test_wrong_mounted_identity_is_not_repaired(skill_mount_factory):
    args = skill_mount_factory("wrong-catalog")
    args["mounted_catalog_blake3"] = "0" * 64
    with pytest.raises(identity.SkillContentIdentityError, match="original_mounted_catalog_not_reconstructed"):
        identity.reopen_skill_content_identity(**args)
