from __future__ import annotations

import json
from pathlib import Path

import pytest

from eva_agent.codex_pipeline import (
    ACTOR_SKILL_PATH_POLICY,
    CodexPipelineError,
    LEGACY_SKILL_MANIFEST_BLAKE3,
    VerifiedActorSkillCatalog,
)
from eva_agent.pipeline import Stage
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, is_blake3


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "evamed-codex"
LEGACY_SOURCE = (
    PLUGIN
    / "vendor"
    / "rlevo-Med-RL-data"
    / "rev-79dd2a31f5f"
)


def _runtime_root(tmp_path: Path) -> Path:
    root = tmp_path / "runtime-skills"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root.resolve()


def _catalog(tmp_path: Path) -> VerifiedActorSkillCatalog:
    runtime_root = tmp_path / "runtime-skills"
    if not runtime_root.exists():
        runtime_root.mkdir(mode=0o700)
        runtime_root.chmod(0o700)
    return VerifiedActorSkillCatalog(
        manifest_path=(
            PLUGIN / "references" / "legacy-skill-manifest.v1.json"
        ).resolve(),
        legacy_source_root=LEGACY_SOURCE.resolve(),
        native_stage_skill_path=(
            PLUGIN / "skills" / "stage-rollout" / "SKILL.md"
        ).resolve(),
        runtime_root=runtime_root.resolve(),
    )


def test_verified_actor_skill_catalog_mounts_exact_stage_scopes(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    expected_external = {
        Stage.S1: 5,
        Stage.S2: 12,
        Stage.S3: 19,
        Stage.S4: 10,
        Stage.S5: 9,
        Stage.E2E: 0,
    }
    assert is_blake3(catalog.manifest_blake3)
    assert catalog.manifest_blake3 == LEGACY_SKILL_MANIFEST_BLAKE3
    assert is_blake3(catalog.catalog_blake3)
    assert is_blake3(catalog.materialization_blake3)
    assert catalog.materialization_path_policy == ACTOR_SKILL_PATH_POLICY
    assert catalog.materialization_root.parent == (tmp_path / "runtime-skills")
    assert catalog.materialization_root.name == (
        f"blake3-{catalog.materialization_blake3}"
    )
    assert catalog.materialization_root.stat().st_mode & 0o777 == 0o500
    assert isinstance(catalog.inventory(), tuple)
    assert len(catalog.inventory()) == 24
    for stage, external_count in expected_external.items():
        skills = catalog.for_stage(stage)
        assert len(skills) == external_count + 1
        assert skills[0].skill_id == "eva-workflow/stage-rollout"
        assert len({skill.skill_id for skill in skills}) == len(skills)
        assert len({skill.name for skill in skills}) == len(skills)
        for skill in skills:
            skill_path = Path(skill.path)
            payload = skill_path.read_bytes()
            assert blake3_bytes(payload) == skill.content_blake3
            assert skill_path.is_relative_to(catalog.materialization_root)
            assert skill_path.stat().st_mode & 0o777 == 0o400
            assert skill_path.parent.stat().st_mode & 0o777 == 0o500

    reopened = _catalog(tmp_path)
    assert reopened.materialization_blake3 == catalog.materialization_blake3
    assert reopened.catalog_blake3 == catalog.catalog_blake3
    assert reopened.for_stage(Stage.S3) == catalog.for_stage(Stage.S3)


def test_verified_actor_skill_catalog_rejects_a_changed_external_body(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy"
    source.mkdir()
    runtime_root = _runtime_root(tmp_path)
    # Symlinking the pinned tree is explicitly disallowed before any body is
    # opened, even if every target byte would otherwise match.
    (source / "README.md").symlink_to(LEGACY_SOURCE / "README.md")
    with pytest.raises(CodexPipelineError, match="README uses a symlink"):
        VerifiedActorSkillCatalog(
            manifest_path=(
                PLUGIN / "references" / "legacy-skill-manifest.v1.json"
            ).resolve(),
            legacy_source_root=source.resolve(),
            native_stage_skill_path=(
                PLUGIN / "skills" / "stage-rollout" / "SKILL.md"
            ).resolve(),
            runtime_root=runtime_root,
        )


def test_verified_actor_skill_catalog_rejects_unknown_stage(tmp_path: Path) -> None:
    with pytest.raises(CodexPipelineError, match="stage differs"):
        _catalog(tmp_path).for_stage("S6")


def test_verified_actor_skill_catalog_rejects_self_consistent_manifest_rewrite(
    tmp_path: Path,
) -> None:
    original_path = PLUGIN / "references" / "legacy-skill-manifest.v1.json"
    document = json.loads(original_path.read_bytes())
    document["skills"][0]["allowed_stages"] = ["E2E", "S1", "S3", "S4", "S5"]
    core = dict(document)
    core.pop("manifest_blake3")
    document["manifest_blake3"] = blake3_hex(core)
    rewritten = tmp_path / "legacy-skill-manifest.v1.json"
    rewritten.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    runtime_root = _runtime_root(tmp_path)

    with pytest.raises(CodexPipelineError, match="manifest commitment differs"):
        VerifiedActorSkillCatalog(
            manifest_path=rewritten.resolve(),
            legacy_source_root=LEGACY_SOURCE.resolve(),
            native_stage_skill_path=(
                PLUGIN / "skills" / "stage-rollout" / "SKILL.md"
            ).resolve(),
            runtime_root=runtime_root,
        )


def test_verified_actor_skill_catalog_rejects_materialized_byte_tamper(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    victim = Path(catalog.for_stage(Stage.S1)[0].path)
    original = victim.read_bytes()
    victim.chmod(0o600)
    victim.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    victim.chmod(0o400)

    with pytest.raises(CodexPipelineError, match="bytes differ"):
        _catalog(tmp_path)


def test_verified_actor_skill_catalog_rejects_materialized_symlink_tamper(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    victim = Path(catalog.for_stage(Stage.S1)[0].path)
    skill_directory = victim.parent
    skill_directory.chmod(0o700)
    victim.unlink()
    victim.symlink_to(PLUGIN / "skills" / "stage-rollout" / "SKILL.md")
    skill_directory.chmod(0o500)

    with pytest.raises(CodexPipelineError, match="cannot be reopened safely"):
        _catalog(tmp_path)
