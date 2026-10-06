from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

import eva_agent.training.bulk_rl_factory as factory
from eva_agent.campaign.selection_v2 import (
    CampaignSelectionV2,
    SELECTION_V2_METHOD,
    SELECTION_V2_SCHEMA,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.training import (
    BulkRLSandboxFactoryError,
    build_bulk_rl_sandbox_catalog,
    verify_bulk_rl_sandbox_catalog,
)
from scripts.build_bulk_rl_sandboxes_v1 import run


ROOT = Path(__file__).resolve().parents[1]


def _key_material(tmp_path: Path) -> tuple[Path, Path]:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "signer.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o600)
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust = tmp_path / "trust.json"
    trust.write_text(
        json.dumps(
            {
                "schema": "eva.ed25519-trust-store.v1",
                "status": "active",
                "algorithm": "Ed25519",
                "keys": {"bulk-test": base64.b64encode(public).decode("ascii")},
            }
        ),
        encoding="utf-8",
    )
    return private_path, trust


def _fixture(tmp_path: Path):
    rubrics = load_and_compile_registry(ROOT / "rubrics/source/domain-stage-tables.v1.json")
    rubric_rows = tuple(sorted(rubrics.rubrics, key=lambda row: (row.domain, row.stage)))[:6]
    source_rows = []
    entry_cores = []
    for index, rubric in enumerate(rubric_rows):
        source_id = f"fixture-source-{index}"
        upstream_sha = f"{index + 1:064x}"
        source_rows.append(
            {
                "candidate_id": source_id,
                "family": "fixture",
                "focus": rubric.stage,
                "frozen_artifact": {
                    "path": f"benchmarks/fixture/{source_id}.json",
                    "sha256": upstream_sha,
                },
                "frozen_assertions": [{"path": ["schema"], "equals": "fixture.v1"}],
                "harness": {
                    "contract_sha256": f"{index + 20:064x}",
                    "owner": "fixture",
                    "s1_s5_harness_owned": True,
                    "stage_chain": ["S1", "S2", "S3", "S4", "S5"],
                },
                "priority": index + 1,
                "proofs": {},
                "source_shard": "fixture",
            }
        )
        entry_cores.append(
            {
                "candidate_id": str(uuid5(NAMESPACE_URL, f"eva:bulk-test:{index}")),
                "source_candidate_id": source_id,
                "source_family": "fixture",
                "source_artifact_sha256": upstream_sha,
                "domain": rubric.domain,
                "stage": rubric.stage,
                "split": "train",
                "rubric_id": rubric.rubric_id,
                "rubric_blake3": rubric.digest,
                "source_identity_rank_blake3": blake3_hex(f"source-rank:{index}"),
                "construction_readiness": "frozen",
                "readiness_proof_root_blake3": blake3_hex(f"readiness:{index}"),
                "selection_rank_blake3": blake3_hex(f"selection-rank:{index}"),
                "cell_rank": index + 1,
                "cell_target": 1,
                "selection_tier": "primary" if index < 4 else "reserve",
                "queue_ordinal": index + 1,
            }
        )
    source_document = {
        "schema": "rlevo.med-research-campaign-supervisor-registry.v1",
        "campaign_id": "bulk-test-campaign",
        "registry_revision": 24,
        "registry_sha256": "a" * 64,
        "created_at_utc": "2026-09-08T00:00:00Z",
        "slot_limits": {},
        "candidates": source_rows,
    }
    source_path = tmp_path / "source.json"
    source_payload = canonical_json_bytes(source_document)
    source_path.write_bytes(source_payload)
    source_digest = blake3_bytes(source_payload)
    selection_core = {
        "schema": SELECTION_V2_SCHEMA,
        "selection_id": str(uuid5(NAMESPACE_URL, "eva:bulk-test:selection")),
        "plan_id": "bulk-test-plan",
        "plan_blake3": blake3_hex("plan"),
        "source_registry_blake3": blake3_hex("source-registry"),
        "upstream_source_registry_sha256": "b" * 64,
        "rubric_registry_blake3": rubrics.digest,
        "base_selection_id": str(uuid5(NAMESPACE_URL, "eva:bulk-test:base")),
        "base_selection_blake3": blake3_hex("base"),
        "readiness_authority": {
            "campaign_id": "bulk-test-campaign",
            "registry_revision": 24,
            "registry_logical_upstream_sha256": "a" * 64,
            "registry_file_blake3": source_digest,
        },
        "selection_method": SELECTION_V2_METHOD,
        "selected_count": 4,
        "scheduled_count": 6,
        "reserve_count": 2,
        "entries": entry_cores,
    }
    selection = CampaignSelectionV2.from_document(
        {**selection_core, "selection_blake3": blake3_hex(selection_core)}
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_bytes(canonical_json_bytes(selection.to_document()))
    return selection, selection_path, source_document, source_path, source_digest, rubrics


def test_bulk_factory_materializes_source_starts_and_exact_reward_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(factory, "EXPECTED_SCHEDULED", 6)
    monkeypatch.setattr(factory, "EXPECTED_SANDBOXES", 4)
    selection, _, source, _, source_digest, rubrics = _fixture(tmp_path)
    catalog = build_bulk_rl_sandbox_catalog(
        selection=selection,
        rubrics=rubrics,
        frozen_source_registry=source,
        frozen_source_registry_file_blake3=source_digest,
    )
    assert len(catalog.records) == 4
    assert {row["split"] for row in catalog.records} == {"train"}
    first = catalog.records[0]
    assert first["training_policy"]["status"] == "training_ready"
    assert first["training_policy"]["rollout_required_for_materialization"] is False
    assert first["source_binding"]["authority_relative_path"].startswith("benchmarks/")
    assert first["reward_contract"]["rubric_table"] == rubrics.resolve(
        first["domain"], first["stage"]
    ).to_document()
    assert first["workspace_initial_state"]["file_count"] == 2
    assert first["workspace_initial_state"]["files"][1]["content"]["tool_profile"][
        "parallel_tool_calls_allowed"
    ] is True
    verify_bulk_rl_sandbox_catalog(
        catalog.to_document(),
        selection=selection,
        rubrics=rubrics,
        frozen_source_registry=source,
        frozen_source_registry_file_blake3=source_digest,
    )
    changed = catalog.to_document()
    changed["records"][0]["split"] = "development"
    with pytest.raises(BulkRLSandboxFactoryError, match="derivation differs"):
        verify_bulk_rl_sandbox_catalog(
            changed,
            selection=selection,
            rubrics=rubrics,
            frozen_source_registry=source,
            frozen_source_registry_file_blake3=source_digest,
        )


def test_sharded_cli_build_and_verify_are_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(factory, "EXPECTED_SCHEDULED", 6)
    monkeypatch.setattr(factory, "EXPECTED_SANDBOXES", 4)
    _, selection_path, _, source_path, _, _ = _fixture(tmp_path)
    private, trust = _key_material(tmp_path)
    output = tmp_path / "dataset"
    common = {
        "output": output,
        "selection": selection_path,
        "source_registry": source_path,
        "shard_size": 2,
        "private_key": private,
        "key_id": "bulk-test",
        "trust_store": trust,
    }
    built = run(argparse.Namespace(command="build", **common))
    assert built["sandbox_count"] == 4
    assert built["shard_count"] == 2
    assert (output / "sandboxes-00000.jsonl").read_bytes().endswith(b"\n")
    verified = run(argparse.Namespace(command="verify", **common))
    assert verified["status"] == "verified"
    with pytest.raises(ValueError, match="already exists"):
        run(argparse.Namespace(command="build", **common))

