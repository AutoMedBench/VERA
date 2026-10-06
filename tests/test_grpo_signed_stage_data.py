"""CPU-only signed membership and explicit coverage; no execution resources."""
import base64
from copy import deepcopy
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from eva_agent.admission.receipts import _sign
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.training.grpo_stage_data import (
    StageDataCoverageError, prepare_stage_data, select_executable_stage_rows,
)


def record(identity, domain="automedbench-classification", stage="S1"):
    row = {"sandbox_id": identity, "candidate_id": "candidate-" + identity,
        "domain": domain, "stage": stage, "split": "train", "source_family": "automedbench",
        "source_binding": {"source_candidate_id": "source-" + identity,
            "upstream_source_artifact_sha256": "a" * 64, "source_registry_blake3": "b" * 64},
        "reward_contract": {"rubric_table": {"stage": stage, "domain": domain,
            "rubric_id": "rubric-" + domain, "rubric_digest": "blake3:" + "c" * 64}},
        "lineage": {"selection_blake3": "d" * 64}}
    row["source_binding"].update({key: row[key] for key in ("candidate_id", "domain", "stage", "source_family")})
    row["record_blake3"] = blake3_hex(row)
    return row


def catalog(tmp_path, rows, statuses=None):
    statuses = statuses or {}
    members = []
    for row in rows:
        member = {key: row[key] for key in ("candidate_id", "domain", "stage", "split", "source_family")}
        member.update(source_candidate_id=row["source_binding"]["source_candidate_id"],
            source_artifact_sha256="a" * 64, source_registry_blake3="b" * 64,
            rubric_id=row["reward_contract"]["rubric_table"]["rubric_id"], rubric_blake3="blake3:" + "c" * 64,
            selection_tier="primary", execution_status=statuses.get(row["sandbox_id"], "executable_legacy"),
            construction_readiness="promoted", readiness_proof_root_blake3="e" * 64,
            execution_proof={"kind": "signed_v24_legacy_promoted", "proof_root_blake3": "e" * 64})
        members.append(member)
    payload = {"schema": "eva.prospective-execution-binding-catalog.v3", "rows": members,
               "selection_blake3": "d" * 64}
    payload["catalog_blake3"] = blake3_hex(payload)
    key = Ed25519PrivateKey.generate()
    envelope = _sign(payload, key_id="stage-test", key=key)
    path, trust = tmp_path / "catalog.json", tmp_path / "trust.json"
    path.write_text(json.dumps(envelope.to_document()))
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    trust.write_text(json.dumps({"schema": "eva.ed25519-trust-store.v1", "status": "active",
        "algorithm": "Ed25519", "keys": {"stage-test": base64.b64encode(public).decode()}}))
    return {"execution_catalog_path": path, "trust_store_path": trust}


def bulk(tmp_path, rows):
    root = tmp_path / "bulk"
    root.mkdir()
    (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "manifest.json").write_text(json.dumps({"payload": {"shards": [{"path": "records.jsonl"}]}}))
    return root


def test_excludes_source_only_and_preserves_exact_original_rows(tmp_path):
    rows = [record("b"), record("a"), record("c", domain="automedbench-segmentation")]
    before = deepcopy(rows)
    selected, coverage = select_executable_stage_rows(rows, stage="S1", limit=128,
        **catalog(tmp_path, rows, {"a": "source_only"}))
    assert selected == [rows[0], rows[2]] and rows == before
    assert coverage["excluded_reason_counts"] == {"execution_status:source_only": 1}
    assert coverage["exclusions"][0]["sandbox_id"] == "a"
    assert coverage["limit_shortfall"] == 126
    assert coverage["catalog_signature_verified"] and not coverage["resolver_bindings_reopened"]
    assert not coverage["actor_execution_verified"]


@pytest.mark.parametrize("missing", ["automedbench-detection", "automedbench-report"])
def test_missing_or_source_only_requested_domain_is_not_silently_dropped(tmp_path, missing):
    rows = [record("a"), record("b", domain="automedbench-detection")]
    with pytest.raises(StageDataCoverageError) as error:
        select_executable_stage_rows(rows, stage="S1", limit=128,
            domains=("automedbench-classification", missing),
            **catalog(tmp_path, rows, {"b": "source_only"}))
    assert error.value.coverage["uncovered_requested_domains"] == [missing]
    assert error.value.coverage["domain_coverage"][missing]["selected_rows"] == 0


def test_limit_cannot_silently_omit_an_explicit_domain(tmp_path):
    rows = [record("a"), record("b", domain="automedbench-segmentation")]
    with pytest.raises(StageDataCoverageError):
        select_executable_stage_rows(rows, stage="S1", limit=1,
            domains=tuple(row["domain"] for row in rows), **catalog(tmp_path, rows))


@pytest.mark.parametrize("field", ["source_candidate_id", "upstream_source_artifact_sha256"])
def test_even_rehashed_source_mutation_cannot_borrow_signed_membership(tmp_path, field):
    rows = [record("a")]
    paths = catalog(tmp_path, rows)
    rows[0]["source_binding"][field] = "changed"
    rows[0]["record_blake3"] = blake3_hex({key: value for key, value in rows[0].items() if key != "record_blake3"})
    with pytest.raises(ValueError, match="binding"):
        select_executable_stage_rows(rows, stage="S1", limit=128, **paths)


def test_tampered_signed_status_fails_before_selection(tmp_path):
    rows = [record("a")]
    paths = catalog(tmp_path, rows, {"a": "source_only"})
    path = paths["execution_catalog_path"]
    value = json.loads(path.read_text())
    value["payload"]["rows"][0]["execution_status"] = "executable_legacy"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        select_executable_stage_rows(rows, stage="S1", limit=128, **paths)


def test_blocked_preparation_retains_coverage_but_emits_no_training_file(tmp_path):
    rows = [record("a")]
    output = tmp_path / "blocked"
    with pytest.raises(StageDataCoverageError):
        prepare_stage_data(bulk(tmp_path, rows), output, stage="S1",
            domains=("automedbench-classification", "automedbench-report"), **catalog(tmp_path, rows))
    receipt = json.loads((output / "preparation-receipt.json").read_text())
    assert receipt["status"] == "blocked" and receipt["sample_count"] == 0
    assert not (output / "grpo.jsonl").exists()
    assert receipt["selection"]["uncovered_requested_domains"] == ["automedbench-report"]


def test_optional_mode_keeps_v1_default_and_emits_explicit_v2_receipt(tmp_path):
    rows = [record("a"), record("b")]
    root = bulk(tmp_path, rows)
    plain = prepare_stage_data(root, tmp_path / "old", stage="S1")
    filtered = prepare_stage_data(root, tmp_path / "new", stage="S1",
        **catalog(tmp_path, rows, {"b": "source_only"}))
    assert plain["schema"] == "eva.grpo-stage-target-data.v1" and plain["sample_count"] == 2
    assert "selection" not in plain and "status" not in plain
    assert filtered["schema"] == "eva.grpo-stage-target-data.v2" and filtered["sample_count"] == 1
    assert filtered["selection"]["selected_sandbox_ids"] == ["a"]


def test_premium_is_not_accepted_by_the_legacy_resolver_lane(tmp_path):
    rows = [record("a")]
    with pytest.raises(StageDataCoverageError) as error:
        select_executable_stage_rows(rows, stage="S1", limit=128,
            **catalog(tmp_path, rows, {"a": "executable_premium"}))
    assert error.value.coverage["excluded_reason_counts"] == {"execution_status:executable_premium": 1}


def test_signed_original_e2e_prepares_without_stage_or_rubric_relabeling(tmp_path):
    rows = [record("whole-task", stage="E2E"), record("stage-five", stage="S5")]
    original = deepcopy(rows)
    root, output = bulk(tmp_path, rows), tmp_path / "e2e-data"
    receipt = prepare_stage_data(root, output, stage="E2E", actor_backend="native_codex_sglang",
        **catalog(tmp_path, rows))
    prepared = [json.loads(line) for line in (output / "grpo.jsonl").read_text().splitlines()]
    assert receipt["sample_count"] == 1 and prepared[0]["metadata"]["stage"] == "E2E"
    assert prepared[0]["metadata"]["sandbox_id"] == "whole-task"
    assert rows == original
