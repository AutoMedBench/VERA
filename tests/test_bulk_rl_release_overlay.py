from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from eva_agent.admission.receipts import issue_signed_envelope
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes
from eva_agent.training import bulk_rl_release_overlay as release


HEX = "a" * 64
RUBRIC = f"blake3:{'b' * 64}"


def _authorities(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(release, "EXPECTED_SANDBOXES", 6)
    monkeypatch.setattr(release, "EXPECTED_SCHEDULED", 8)
    monkeypatch.setattr(
        release,
        "EXPECTED_SPLITS",
        {"train": 4, "development": 1, "sealed_evaluation": 1},
    )
    monkeypatch.setattr(release, "EXPECTED_STAGE_COUNTS", {stage: 1 for stage in release.STAGES})

    splits = ["train", "train", "train", "train", "development", "sealed_evaluation"]
    base_records = []
    entries = []
    execution_rows = []
    for ordinal in range(1, 9):
        candidate_id = f"candidate-{ordinal}"
        stage = release.STAGES[(ordinal - 1) % len(release.STAGES)]
        split = splits[ordinal - 1] if ordinal <= 6 else "train"
        tier = "primary" if ordinal <= 6 else "reserve"
        common = {
            "candidate_id": candidate_id,
            "queue_ordinal": ordinal,
            "source_family": "automedbench",
            "domain": "classification",
            "stage": stage,
            "rubric_id": "rubric-1",
            "rubric_blake3": RUBRIC,
            "source_candidate_id": f"source-{ordinal}",
        }
        entries.append({**common, "selection_tier": tier, "split": split})
        execution_rows.append(
            {
                **common,
                "selection_tier": tier,
                "split": split,
                "execution_status": "rejected" if ordinal == 2 else "source_only",
                "execution_proof": None,
                "row_blake3": HEX,
            }
        )
        if ordinal <= 6:
            base_records.append(
                {
                    "schema": "eva.bulk-rl-sandbox.v1",
                    "candidate_id": candidate_id,
                    "sandbox_id": f"sandbox-{ordinal}",
                    "record_blake3": HEX,
                    "queue_ordinal": ordinal,
                    "source_family": "automedbench",
                    "domain": "classification",
                    "stage": stage,
                    "reward_contract": {"rubric_id": "rubric-1", "rubric_digest": RUBRIC},
                    "source_binding": {"source_candidate_id": f"source-{ordinal}"},
                }
            )
    base_manifest = {
        "schema": "eva.bulk-rl-sandbox-sharded-dataset.v1",
        "sandbox_count": 6,
        "split_counts": {"train": 6},
        "selection_blake3": HEX,
        "dataset_blake3": HEX,
        "catalog_blake3": HEX,
        "shard_count": 1,
    }
    selection = {
        "schema": "eva.medresearch-campaign-selection.v3",
        "selected_count": 6,
        "scheduled_count": 8,
        "selection_blake3": "c" * 64,
        "base_selection_blake3": HEX,
        "plan_blake3": "d" * 64,
        "entries": entries,
    }
    execution_catalog = {
        "schema": "eva.prospective-execution-binding-catalog.v2",
        "selection_blake3": selection["selection_blake3"],
        "primary_split_counts": release.EXPECTED_SPLITS,
        "primary_count": 6,
        "scheduled_count": 8,
        "catalog_blake3": "e" * 64,
        "rows": execution_rows,
    }
    return base_records, base_manifest, selection, execution_catalog


def _signing_material(tmp_path: Path) -> tuple[Path, Path]:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "release-test.pem"
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
    trust_store = tmp_path / "trust-store.json"
    trust_store.write_text(
        json.dumps(
            {
                "schema": "eva.ed25519-trust-store.v1",
                "status": "active",
                "algorithm": "Ed25519",
                "keys": {"release-test": base64.b64encode(public).decode("ascii")},
            }
        ),
        encoding="utf-8",
    )
    return private_path, trust_store


def _installed_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    base_records, base_manifest, selection, execution = _authorities(monkeypatch)
    private_key, trust_store = _signing_material(tmp_path)
    base_root = tmp_path / "base"
    release_root = tmp_path / "release"
    base_root.mkdir()
    release_root.mkdir()

    base_payload = b"".join(canonical_json_bytes(row) for row in base_records)
    base_shard = base_root / "sandboxes-00000.jsonl"
    base_shard.write_bytes(base_payload)
    base_manifest = {
        **base_manifest,
        "stage_counts": release.EXPECTED_STAGE_COUNTS,
        "controls": {"records_are_canonical_jsonl": True},
        "shards": [
            {
                "ordinal": 0,
                "path": base_shard.name,
                "record_count": len(base_records),
                "byte_count": len(base_payload),
                "file_blake3": blake3_bytes(base_payload),
                "shard_blake3": blake3_hex([row["record_blake3"] for row in base_records]),
                "first_sandbox_id": base_records[0]["sandbox_id"],
                "last_sandbox_id": base_records[-1]["sandbox_id"],
            }
        ],
    }
    base_envelope = issue_signed_envelope(
        base_manifest,
        key_id="release-test",
        private_key_path=private_key.resolve(strict=True),
    )
    base_manifest_path = base_root / "manifest.json"
    base_manifest_path.write_bytes(canonical_json_bytes(base_envelope.to_document()))

    rows = release.derive_overlay_rows(
        base_records,
        base_manifest=base_manifest,
        selection=selection,
        execution_catalog=execution,
    )
    overlay_payload = b"".join(canonical_json_bytes(row) for row in rows)
    overlay_path = release_root / "split-overlay.jsonl"
    overlay_path.write_bytes(overlay_payload)
    roles = [f"cascade-{ordinal}" for ordinal in range(5)]
    evaluator = release.fixed_evaluator_recipe(
        {
            "schema": "rlevo.med-research-model-registry.v1",
            "cascade_order": roles,
            "models": [
                *[
                    {
                        "role_id": role,
                        "api_model_id": f"provider/model-{ordinal}",
                        "provider_family": "provider",
                    }
                    for ordinal, role in enumerate(roles)
                ],
                {
                    "role_id": "architect_opus_5",
                    "api_model_id": "aws/anthropic/bedrock-claude-opus-5",
                    "provider_family": "anthropic",
                },
            ],
        }
    )
    release_manifest = release.release_manifest_core(
        rows,
        overlay_path=overlay_path.name,
        overlay_byte_count=len(overlay_payload),
        overlay_blake3=blake3_bytes(overlay_payload),
        base_manifest=base_manifest,
        base_manifest_envelope_blake3=base_envelope.envelope_blake3,
        selection=selection,
        execution_catalog=execution,
        execution_catalog_envelope_blake3="f" * 64,
        model_registry_file_blake3="9" * 64,
        evaluator_recipe=evaluator,
    )
    release_envelope = issue_signed_envelope(
        release_manifest,
        key_id="release-test",
        private_key_path=private_key.resolve(strict=True),
    )
    release_manifest_path = release_root / "manifest.json"
    release_manifest_path.write_bytes(canonical_json_bytes(release_envelope.to_document()))
    for path in (base_shard, base_manifest_path, overlay_path, release_manifest_path):
        path.chmod(0o444)
    base_root.chmod(0o555)
    release_root.chmod(0o555)
    return base_root, release_root, trust_store


def test_release_overlay_preserves_records_and_adds_exact_splits(monkeypatch):
    base, manifest, selection, execution = _authorities(monkeypatch)
    original_base = deepcopy(base)
    rows = release.derive_overlay_rows(
        base,
        base_manifest=manifest,
        selection=selection,
        execution_catalog=execution,
    )

    assert [row["release_split"] for row in rows] == [
        "train", "train", "train", "train", "development", "sealed_evaluation"
    ]
    assert sum(row["immutable_failure"] for row in rows) == 1
    assert len(release.release_view_records(base, rows, split="development")) == 1
    assert base == original_base
    assert len(release.release_view_records(base, rows)) == 6

    evaluator = {
        "schema": release.CASCADE_SCHEMA,
        "rollouts_per_sandbox": 5,
        "recipe_blake3": HEX,
    }
    core = release.release_manifest_core(
        rows,
        overlay_path="split-overlay.jsonl",
        overlay_byte_count=123,
        overlay_blake3=HEX,
        base_manifest=manifest,
        base_manifest_envelope_blake3=HEX,
        selection=selection,
        execution_catalog=execution,
        execution_catalog_envelope_blake3=HEX,
        model_registry_file_blake3=HEX,
        evaluator_recipe=evaluator,
    )
    assert core["split_counts"] == release.EXPECTED_SPLITS
    assert core["immutable_failure_count"] == 1
    assert core["controls"]["base_shards_rewritten"] is False
    unsigned_core = {key: value for key, value in core.items() if key != "release_blake3"}
    assert core["release_blake3"] == blake3_hex(unsigned_core)


def test_release_overlay_fails_closed_on_identity_or_inventory_change(monkeypatch):
    base, manifest, selection, execution = _authorities(monkeypatch)
    changed = deepcopy(base)
    changed[0]["stage"] = "S2"
    with pytest.raises(release.BulkRLReleaseOverlayError, match="immutable sandbox field"):
        release.derive_overlay_rows(
            changed,
            base_manifest=manifest,
            selection=selection,
            execution_catalog=execution,
        )

    rows = release.derive_overlay_rows(
        base,
        base_manifest=manifest,
        selection=selection,
        execution_catalog=execution,
    )
    with pytest.raises(release.BulkRLReleaseOverlayError, match="inventory"):
        release.release_view_records(base[:-1], rows)


def test_fixed_evaluator_uses_signed_five_model_order():
    order = [f"cascade-{ordinal}" for ordinal in range(5)]
    models = [
        {
            "role_id": role,
            "api_model_id": f"provider/model-{ordinal}",
            "provider_family": "provider",
        }
        for ordinal, role in enumerate(order)
    ]
    models.append(
        {
            "role_id": "architect_opus_5",
            "api_model_id": "aws/anthropic/bedrock-claude-opus-5",
            "provider_family": "anthropic",
        }
    )
    recipe = release.fixed_evaluator_recipe(
        {
            "schema": "rlevo.med-research-model-registry.v1",
            "cascade_order": order,
            "models": models,
        }
    )
    assert recipe["cohort_order"] == order
    assert recipe["rollouts_per_sandbox"] == 5
    assert recipe["agent_judge"]["workspace_inspection_required"] is True


def test_authenticated_loader_returns_split_view_without_rehashing_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_root, release_root, trust_store = _installed_release(tmp_path, monkeypatch)
    original_blake3_bytes = release.blake3_bytes
    hashed_sizes: list[int] = []

    def observed_blake3_bytes(payload: bytes) -> str:
        hashed_sizes.append(len(payload))
        return original_blake3_bytes(payload)

    monkeypatch.setattr(release, "blake3_bytes", observed_blake3_bytes)
    view = release.load_verified_release_view(
        base_root,
        release_root,
        trust_store,
        split="development",
    )

    assert len(view) == 1
    assert view[0]["release_split"] == "development"
    assert view[0]["sandbox"]["candidate_id"] == "candidate-5"
    assert hashed_sizes == [(release_root / "split-overlay.jsonl").stat().st_size]


def test_authenticated_loader_rejects_release_file_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_root, release_root, trust_store = _installed_release(tmp_path, monkeypatch)
    overlay_path = release_root / "split-overlay.jsonl"
    release_root.chmod(0o755)
    overlay_path.chmod(0o644)
    payload = overlay_path.read_bytes()
    overlay_path.write_bytes(payload.replace(b"candidate-1", b"candidate-x", 1))
    overlay_path.chmod(0o444)
    release_root.chmod(0o555)

    with pytest.raises(release.BulkRLReleaseOverlayError, match="file BLAKE3"):
        release.load_verified_release_view(base_root, release_root, trust_store)


def test_authenticated_loader_rejects_writable_base_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_root, release_root, trust_store = _installed_release(tmp_path, monkeypatch)
    (base_root / "sandboxes-00000.jsonl").chmod(0o644)

    with pytest.raises(release.BulkRLReleaseOverlayError, match="base shard 0 must be a read-only"):
        release.load_verified_release_view(base_root, release_root, trust_store)
