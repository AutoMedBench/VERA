from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from eva_agent.pipeline import (
    DeterministicUUIDFactory,
    build_legacy_teacher_sft_dataset,
    verify_legacy_sft_dataset,
    verify_legacy_teacher_source,
)


MODEL_ID = "aws/anthropic/bedrock-claude-opus-5"


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _write(path: Path, value) -> dict[str, str]:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": path.name, "sha256": hashlib.sha256(payload).hexdigest()}


def _signed(private: Ed25519PrivateKey, key_id: str, payload) -> dict:
    wire = _canonical(payload)
    return {
        "schema": "rlevo.med-research-signed-host-receipt.v1",
        "key_id": key_id,
        "payload": payload,
        "payload_sha256": hashlib.sha256(wire).hexdigest(),
        "signature_base64": base64.b64encode(private.sign(wire)).decode(),
    }


def _fixture(tmp_path: Path, *, include_attestation: bool = True) -> tuple[Path, Path]:
    root = tmp_path / "legacy-runs"
    run = root / "teacher-001"
    run.mkdir(parents=True)
    private = Ed25519PrivateKey.generate()
    key_id = "fixture-key"
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    trust = {
        "schema": "rlevo.med-research-host-trust-store.v1",
        "status": "active",
        "algorithm": "Ed25519",
        "keys": {key_id: base64.b64encode(public).decode()},
    }
    trust_path = tmp_path / "trust.json"
    trust_path.write_bytes(_canonical(trust))
    sandbox_id = "fixture-sandbox"
    episode_id = "fixture-episode"
    rollout_id = "fixture-rollout"
    plan_call = "call-plan"
    submit_call = "call-submit"
    transcript = {
        "schema": "rlevo.med-research-agent-transcript.v1",
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
        "rollout_id": rollout_id,
        "model_id": MODEL_ID,
        "hidden_reasoning_recorded": False,
        "raw_provider_response_recorded": False,
        "messages": [
            {"role": "system", "content": "Use the registered tools."},
            {"role": "user", "content": "Complete this verified research workflow."},
            {
                "role": "assistant",
                "content": "",
                "hidden_reasoning_removed": False,
                "tool_calls": [
                    {
                        "id": plan_call,
                        "type": "function",
                        "function": {
                            "name": "materialize_plan",
                            "arguments": '{"objective":"test"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": plan_call,
                "content": '{"gate_passed":true}',
            },
            {
                "role": "assistant",
                "content": "",
                "hidden_reasoning_removed": False,
                "tool_calls": [
                    {
                        "id": submit_call,
                        "type": "function",
                        "function": {
                            "name": "submit_results",
                            "arguments": '{"submitted":true}',
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "source": "harness",
                "content": '{"submitted":true}',
            },
        ],
    }
    transcript_ref = _write(run / "transcript.json", transcript)
    final_payload = {
        "schema": "rlevo.med-research-host-receipt-payload.v1",
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
        "events": [{"ordinal": 0, "kind": "submission-validated"}],
        "observations": {
            "workspace": {"before_blake3": "0" * 64, "after_blake3": "1" * 64},
            "artifact": {"blake3": "2" * 64},
        },
    }
    final_receipt = _signed(private, key_id, final_payload)
    final_receipt_ref = _write(run / "final-host-receipt.json", final_receipt)
    grade_core = {
        "schema": "rlevo.med-research-deterministic-grade.v1",
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
        "table_id": "fixture-rubric-v1",
        "focus": "E2E",
        "normalized_score": 1.0,
        "execution_gate_passed": True,
        "critical_items_passed": True,
        "score_eligible": True,
        "rubric_verdicts": [{"item_id": "workspace", "passed": True}],
        "host_receipt_payload_sha256": final_receipt["payload_sha256"],
    }
    grade = {**grade_core, "grade_sha256": hashlib.sha256(_canonical(grade_core)).hexdigest()}
    grade_ref = _write(run / "final-grade.json", grade)
    grade_ref.update(
        grade_sha256=grade["grade_sha256"], normalized_score=grade["normalized_score"]
    )
    receipt = {
        "schema": "rlevo.med-research-agent-rollout-receipt.v2",
        "status": "completed",
        "termination": "submitted",
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
        "rollout_id": rollout_id,
        "model_id": MODEL_ID,
        "turn_count": 2,
        "turns": [{"identity_match": True}, {"identity_match": True}],
        "exact_identity_all_turns": True,
        "hidden_reasoning_recorded": False,
        "raw_provider_response_recorded": False,
        "infrastructure_healthy": True,
        "transcript": transcript_ref,
        "final_grade": grade_ref,
        "final_host_receipt": final_receipt_ref,
        "stage_evidence": {},
    }
    claim = dict(receipt)
    attestation_payload = {
        "schema": "rlevo.med-research-host-receipt-payload.v1",
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
        "events": [{"ordinal": 0, "kind": "agent-rollout-attested"}],
        "observations": {
            "rollout_claim_sha256": hashlib.sha256(_canonical(claim)).hexdigest(),
            "transcript_sha256": transcript_ref["sha256"],
            "final_host_receipt_sha256": final_receipt_ref["sha256"],
            "final_grade_sha256": grade_ref["sha256"],
        },
    }
    if include_attestation:
        receipt["host_attestation"] = _write(
            run / "rollout-host-attestation.json",
            _signed(private, key_id, attestation_payload),
        )
    _write(run / "rollout-receipt.json", receipt)
    return root, trust_path


def test_legacy_teacher_build_slices_tools_and_verifies(tmp_path: Path) -> None:
    source_root, trust = _fixture(tmp_path)
    source = verify_legacy_teacher_source(
        source_root / "teacher-001" / "rollout-receipt.json",
        source_root=source_root,
        trust_store_path=trust,
        eligible_model_ids=(MODEL_ID,),
    )
    assert source.grade["normalized_score"] == 1.0

    build = build_legacy_teacher_sft_dataset(
        source_root=source_root,
        output_root=tmp_path / "sft",
        trust_store_path=trust,
        eligible_model_ids=(MODEL_ID,),
        id_factory=DeterministicUUIDFactory("legacy-teacher"),
        shard_size=1,
        verification_workers=4,
    )
    assert (build.source_count, build.slice_count, build.shard_count) == (1, 2, 2)
    report = verify_legacy_sft_dataset(tmp_path / "sft", build.dataset_id)
    assert report["valid"] is True
    manifest = json.loads((build.dataset_root / "manifest.json").read_bytes())
    assert manifest["counts_by_stage"] == {
        "E2E": 0,
        "S1": 1,
        "S2": 0,
        "S3": 0,
        "S4": 0,
        "S5": 1,
    }
    last = json.loads((build.dataset_root / "shards/part-00001.jsonl").read_bytes())
    assert last["target_tool_observations"] == [
        {
            "content": '{"submitted":true}',
            "role": "tool",
            "source": "harness",
            "tool_call_id": "call-submit",
        }
    ]
    assert "hidden_reasoning_removed" not in last["supervised_assistant_decision"]


def test_legacy_teacher_dataset_tamper_fails_closed(tmp_path: Path) -> None:
    source_root, trust = _fixture(tmp_path)
    build = build_legacy_teacher_sft_dataset(
        source_root=source_root,
        output_root=tmp_path / "sft",
        trust_store_path=trust,
        eligible_model_ids=(MODEL_ID,),
        id_factory=DeterministicUUIDFactory("legacy-tamper"),
    )
    shard = build.dataset_root / "shards/part-00000.jsonl"
    shard.chmod(0o644)
    shard.write_bytes(shard.read_bytes() + b"{}\n")
    report = verify_legacy_sft_dataset(tmp_path / "sft", build.dataset_id)
    assert report["valid"] is False
    assert "shard commitment differs" in report["errors"][0]
