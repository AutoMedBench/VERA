from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID

import pytest

from eva_agent.pipeline.digests import blake3_hex, is_blake3
from eva_agent.sources import (
    LegacyImportError,
    LegacySupervisorV4Importer,
    RubricAvailability,
)


def _canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _artifact(candidate_id: str, family: str, stage: str, track: str, evidence_path: str, evidence: bytes):
    return {
        "schema": "rlevo.med-research-candidate-construction-input.v3",
        "construction_id": f"construction-{candidate_id}",
        "sandbox": {
            "sandbox_id": candidate_id,
            "focus": stage,
            "track": track,
            "domain": "legacy-domain",
            "target_split": "candidate",
        },
        "task_brief": {
            "title": f"Task for {candidate_id}",
            "objective": "Use tools to produce independently verifiable medical-research evidence.",
            "research_question": "Can the evidence support the bounded research claim?",
            "evidence_need": "Use only the typed host evidence object.",
            "task_ids": [candidate_id],
        },
        "content_scope": "medical_research",
        "runtime": {
            "clock_utc": "2026-09-03T00:00:00Z",
            "execution_limits": {"cpus": 1, "wall_time_seconds": 180},
            "policy_budgets": {"max_turns": 24},
            "s4_artifact": {"checks": [{"path": "/answer", "value": "judge-only"}]},
        },
        "safety": {
            "care_facing": False,
            "source_data_synthetic": True,
            "private_reference": "must not cross the projection",
        },
        "source_lineage": [
            {
                "repository": f"fixture/{family}",
                "revision": "fixture-revision-1",
                "source_id": f"fixture:{candidate_id}",
                "source_split": "train",
            }
        ],
        "evidence_objects": [
            {
                "evidence_id": f"evidence-{candidate_id}",
                "source_relative_path": evidence_path,
                "sha256": _sha(evidence),
                "byte_count": len(evidence),
                "content_kind": "permitted-structured-facts",
            }
        ],
        "rubric_noncritical_families": {"provenance": ["source_bound"]},
        "evidence_source_verification": {"status": "verified"},
        "evidence_source_verification_attestation": {"status": "attested"},
        "private_reference": {"answer": "judge-only answer"},
    }


def _candidate(candidate_id: str, family: str, stage: str, artifact_path: str, payload: bytes):
    return {
        "candidate_id": candidate_id,
        "family": family,
        "focus": stage,
        "source_shard": f"{stage.casefold()}-fixture",
        "priority": 1,
        "harness": {
            "owner": "rlevo-med-research",
            "stage_chain": ["S1", "S2", "S3", "S4", "S5"],
            "contract_sha256": "1" * 64,
            "s1_s5_harness_owned": True,
        },
        "frozen_artifact": {"path": artifact_path, "sha256": _sha(payload)},
        "frozen_assertions": [
            {"path": ["schema"], "equals": "rlevo.med-research-candidate-construction-input.v3"},
            {"path": ["sandbox", "sandbox_id"], "equals": candidate_id},
            {"path": ["sandbox", "focus"], "equals": stage},
        ],
        "proofs": {},
    }


def _fixture(tmp_path: Path, specs=None):
    authority = tmp_path / "legacy"
    supervisor = authority / "runs" / "evamed-campaign-supervisor-v4"
    supervisor.mkdir(parents=True)
    specs = specs or [
        ("rlevo-medres-agentclinic-s1-a", "agentclinic", "S1", "AgentClinic"),
        ("rlevo-medres-automedbench-s2-b", "automedbench", "S2", "Cls"),
        ("rlevo-medres-healthbench-e2e-c", "healthbench-professional", "E2E", "Research"),
        ("rlevo-medres-medxpertqa-s5-d", "medxpertqa", "S5", "MedX"),
    ]
    candidates = []
    for candidate_id, family, stage, track in specs:
        evidence_path = f"sources/{candidate_id}/evidence.json"
        evidence = (f'{{"candidate":"{candidate_id}","public":true}}\n').encode()
        evidence = _write_json(authority / evidence_path, json.loads(evidence))
        artifact_path = f"sources/{candidate_id}/construction-input.json"
        artifact = _artifact(candidate_id, family, stage, track, evidence_path, evidence)
        payload = _write_json(authority / artifact_path, artifact)
        candidates.append(_candidate(candidate_id, family, stage, artifact_path, payload))
    candidates.sort(key=lambda row: row["candidate_id"])
    registry = {
        "schema": "rlevo.med-research-campaign-supervisor-registry.v1",
        "campaign_id": "evamed-6000-campaign-supervisor-v1",
        "created_at_utc": "2026-09-05T19:53:17Z",
        "registry_revision": 4,
        "slot_limits": {"provider_total": 6},
        "candidates": candidates,
    }
    registry["registry_sha256"] = _sha(_canonical(registry))
    _write_json(supervisor / "candidate-registry.v4.json", registry)
    return authority, supervisor, candidates


def _contains_key(value, forbidden: str) -> bool:
    if isinstance(value, dict) or hasattr(value, "items"):
        return any(key == forbidden or _contains_key(child, forbidden) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def test_selected_import_is_lazy_mapped_and_privacy_separated(tmp_path: Path) -> None:
    authority, supervisor, candidates = _fixture(tmp_path)
    importer = LegacySupervisorV4Importer(supervisor, authority_root=authority)
    assert importer.candidate_count == 4
    assert is_blake3(importer.registry_file_blake3)

    # Break an unselected artifact after registry validation. Registry-only selection and
    # selected import must never touch it.
    unselected = candidates[0]
    (authority / unselected["frozen_artifact"]["path"]).write_bytes(b"tampered\n")
    selected_id = "rlevo-medres-automedbench-s2-b"
    reference = next(importer.iter_candidates(candidate_ids=[selected_id]))
    assert reference.candidate_id == selected_id
    imported = next(importer.iter_imports(candidate_ids=[selected_id], limit=1))

    assert imported.episode.domain == "automedbench-classification"
    assert imported.episode.stage.value == "S2"
    UUID(imported.episode.episode_id)
    UUID(imported.manifest.import_id)
    assert imported.manifest.rubric_availability is RubricAvailability.READY
    assert is_blake3(imported.manifest.artifact_blake3)
    assert is_blake3(imported.manifest.import_blake3)
    assert imported.manifest.import_blake3 == blake3_hex(imported.manifest.core_document())
    assert not _contains_key(imported.episode.policy_context, "private_reference")
    assert not _contains_key(imported.episode.policy_context, "s4_artifact")
    assert _contains_key(imported.episode.judge_only_reference, "private_reference")
    assert _contains_key(imported.episode.judge_only_reference, "s4_artifact")
    assert imported.episode.initial_files["TASK.md"].startswith(b"Task for")


def test_healthbench_and_automed_research_resolve_to_compiled_rubrics(tmp_path: Path) -> None:
    authority, supervisor, _ = _fixture(
        tmp_path,
        specs=[
            ("rlevo-medres-automedbench-e2e-a", "automedbench", "E2E", "Research"),
            ("rlevo-medres-healthbench-e2e-b", "healthbench-professional", "E2E", "Research"),
        ],
    )
    importer = LegacySupervisorV4Importer(supervisor, authority_root=authority)
    imported = list(importer.iter_imports())
    assert [row.manifest.rubric_domain for row in imported] == [
        "automedbench-research",
        "healthbench-professional",
    ]
    assert all(
        row.manifest.rubric_availability is RubricAvailability.READY for row in imported
    )
    ready = list(importer.iter_imports(rubric_ready_only=True))
    assert [row.manifest.rubric_domain for row in ready] == [
        "automedbench-research",
        "healthbench-professional",
    ]


def test_selected_artifact_or_evidence_tamper_fails_closed(tmp_path: Path) -> None:
    authority, supervisor, candidates = _fixture(
        tmp_path,
        specs=[("rlevo-medres-medxpertqa-s5-a", "medxpertqa", "S5", "MedX")],
    )
    importer = LegacySupervisorV4Importer(supervisor, authority_root=authority)
    reference = next(importer.iter_candidates())
    artifact_path = authority / candidates[0]["frozen_artifact"]["path"]
    original = artifact_path.read_bytes()
    artifact_path.write_bytes(original + b" ")
    with pytest.raises(LegacyImportError, match="artifact SHA-256"):
        importer.import_candidate(reference)

    # Rebuild to get a valid artifact, then alter only its referenced evidence.
    authority, supervisor, candidates = _fixture(
        tmp_path / "evidence-case",
        specs=[("rlevo-medres-medxpertqa-s5-b", "medxpertqa", "S5", "MedX")],
    )
    importer = LegacySupervisorV4Importer(supervisor, authority_root=authority)
    reference = next(importer.iter_candidates())
    artifact = json.loads((authority / candidates[0]["frozen_artifact"]["path"]).read_text())
    evidence_path = authority / artifact["evidence_objects"][0]["source_relative_path"]
    evidence_path.write_bytes(evidence_path.read_bytes() + b" ")
    with pytest.raises(LegacyImportError, match="evidence verification"):
        importer.import_candidate(reference)


def test_path_escape_and_symlink_are_rejected(tmp_path: Path) -> None:
    authority, supervisor, candidates = _fixture(
        tmp_path,
        specs=[("rlevo-medres-agentclinic-s1-a", "agentclinic", "S1", "AgentClinic")],
    )
    registry_path = supervisor / "candidate-registry.v4.json"
    registry = json.loads(registry_path.read_text())
    registry["candidates"][0]["frozen_artifact"]["path"] = "../outside.json"
    registry.pop("registry_sha256")
    registry["registry_sha256"] = _sha(_canonical(registry))
    _write_json(registry_path, registry)
    with pytest.raises(LegacyImportError, match="escapes|normalized"):
        LegacySupervisorV4Importer(supervisor, authority_root=authority)

    authority, supervisor, candidates = _fixture(
        tmp_path / "link-case",
        specs=[("rlevo-medres-agentclinic-s1-b", "agentclinic", "S1", "AgentClinic")],
    )
    importer = LegacySupervisorV4Importer(supervisor, authority_root=authority)
    reference = next(importer.iter_candidates())
    target = authority / candidates[0]["frozen_artifact"]["path"]
    real = target.with_name("real-construction-input.json")
    target.rename(real)
    target.symlink_to(real.name)
    with pytest.raises(LegacyImportError, match="symlink"):
        importer.import_candidate(reference)


def test_registry_logical_tamper_is_rejected_without_source_mutation(tmp_path: Path) -> None:
    authority, supervisor, _ = _fixture(tmp_path)
    registry_path = supervisor / "candidate-registry.v4.json"
    before = registry_path.read_bytes()
    registry = json.loads(before)
    registry["slot_limits"]["provider_total"] = 999
    _write_json(registry_path, registry)  # deliberately retain the prior logical claim
    tampered = registry_path.read_bytes()
    with pytest.raises(LegacyImportError, match="logical SHA-256"):
        LegacySupervisorV4Importer(supervisor, authority_root=authority)
    assert registry_path.read_bytes() == tampered
