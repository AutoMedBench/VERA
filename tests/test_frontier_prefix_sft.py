from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.codex_runtime import (
    CodexEvent,
    CodexRole,
    CodexSandbox,
    CodexTurnReceipt,
    codex_turn_receipt_from_document,
)
from eva_agent.pipeline.digests import blake3_hex, canonical_value
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.training.frontier_prefix_sft import (
    FrontierPrefixSFTError,
    _provider_rollout_receipt_claim,
    _require_s1_gate,
    _route_authority,
    _source_slice,
    build_frontier_prefix_sft_dataset,
    verify_frontier_prefix_sft_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
CANARY = ROOT / "runs/codex-teacher-opus5-s1-guidance-canary.v2"
WAVE7_COMPLETED = (
    ROOT
    / "runs/codex-teacher-opus-s1-wave7.v2/trajectories"
    / "7b7001a0-a397-5c62-86ee-38cea28d443c/opus_5/result.json"
)


def _synthetic_receipt() -> CodexTurnReceipt:
    event_core = {
        "event_id": "00000000-0000-4000-8000-000000000001",
        "sequence": 0,
        "method": "turn/completed",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "payload": {},
        "content_redacted": False,
    }
    event = CodexEvent(**event_core, event_blake3=blake3_hex(event_core))
    core = {
        "schema": "eva.codex-turn-receipt.v1",
        "receipt_id": "00000000-0000-4000-8000-000000000002",
        "runtime_thread_id": "00000000-0000-4000-8000-000000000003",
        "runtime_turn_id": "00000000-0000-4000-8000-000000000004",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "role": CodexRole.STRONG_ACTOR,
        "model": "model-1",
        "provider": "provider-1",
        "sandbox": CodexSandbox.READ_ONLY,
        "thread_resumed": False,
        "visibility": "actor-public",
        "status": "completed",
        "final_response": "done",
        "events": (event,),
        "tool_calls": (),
        "selected_skill_ids": (),
        "selected_skill_catalog_blake3": blake3_hex(()),
        "offered_mcp_tool_names": (),
        "offered_tool_schema_blake3": blake3_hex(()),
        "max_parallelism_observed": 0,
        "parallel_tool_calls_supported": True,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "input_blake3": blake3_hex({"input": 1}),
        "config_keys": (),
        "config_values_recorded": False,
        "input_payload_recorded": False,
        "sdk_version": "test-sdk",
        "server_version": "test-server",
    }
    return CodexTurnReceipt(**core, receipt_blake3=blake3_hex(core))


def _passing_observation() -> dict:
    from eva_agent.pipeline.digests import blake3_hex

    result_core = {
        "call_id": "396b72da-38ab-472c-8f8f-e22a562fea43",
        "error_code": None,
        "frontier": 0,
        "name": "materialize_plan",
        "output": {
            "effect": "plan_materialization",
            "failed_check_ids": [],
            "gate_passed": True,
            "next_stage": "S2",
            "stage": "S1",
        },
        "parallel_group_id": "383b508f-fbc7-4e9f-a184-9bc32e190e3e",
        "result_id": "7703fb40-e6d1-4751-bc00-82f4e2b5c759",
        "status": "completed",
        "workspace_after_blake3": "f" * 64,
        "workspace_before_blake3": "6" * 64,
    }
    result = {**result_core, "receipt_blake3": blake3_hex(result_core)}
    core = {
        "schema": "eva.codex-pipeline-tool-observation.v1",
        "call_id": result["call_id"],
        "name": "materialize_plan",
        "arguments": {"schema": "example"},
        "tool_result": result,
    }
    return {**core, "bridge_receipt_blake3": blake3_hex(core)}


def test_frontier_gate_rejects_unmet_check() -> None:
    observation = _passing_observation()
    _require_s1_gate(observation)
    changed = copy.deepcopy(observation)
    changed["tool_result"]["output"]["gate_passed"] = False
    result_core = {
        key: value
        for key, value in changed["tool_result"].items()
        if key != "receipt_blake3"
    }
    from eva_agent.pipeline.digests import blake3_hex

    changed["tool_result"]["receipt_blake3"] = blake3_hex(result_core)
    bridge_core = {
        key: value for key, value in changed.items() if key != "bridge_receipt_blake3"
    }
    changed["bridge_receipt_blake3"] = blake3_hex(bridge_core)
    with pytest.raises(FrontierPrefixSFTError, match="did not pass"):
        _require_s1_gate(changed)


def test_provider_claim_is_distinct_and_nested_receipt_tamper_rejects() -> None:
    receipt = _synthetic_receipt()
    source = {
        "provider_receipt_blake3": "f" * 64,
        "model_id": receipt.model,
        "provider": receipt.provider,
        "assistant_output": receipt.final_response,
    }
    assert receipt.receipt_blake3 != source["provider_receipt_blake3"]
    assert (
        _provider_rollout_receipt_claim(source, receipt=receipt)
        == source["provider_receipt_blake3"]
    )
    tampered = canonical_value(receipt)
    tampered["receipt_blake3"] = "0" * 64
    with pytest.raises(ValueError, match="receipt BLAKE3 differs"):
        codex_turn_receipt_from_document(tampered)


def _synthetic_route(route_id: str, family: str, model_id: str) -> SimpleNamespace:
    from eva_agent.codex_providers import ROUTE_DEFINITIONS

    definition = ROUTE_DEFINITIONS[route_id]
    return SimpleNamespace(
        route_id=route_id,
        model_id=model_id,
        model_env_name=definition.model_env_name,
        registry_role_id=definition.registry_role_id,
        provider_family=family,
        config=SimpleNamespace(model_id=model_id, provider_id=f"eva_{route_id}"),
    )


def test_v3_native_route_requires_exact_authority() -> None:
    with pytest.raises(FrontierPrefixSFTError, match="authorities are absent"):
        _route_authority("gpt_5_6_sol", routes=None)
    route = _synthetic_route("gpt_5_6_sol", "openai", "openai/gpt-5.6-sol")
    authority, provider, scope = _route_authority(
        "gpt_5_6_sol", routes={"gpt_5_6_sol": route}
    )
    assert authority is route
    assert provider == "eva_gpt_5_6_sol"
    assert scope == "native_codex_direct"


def test_v3_adapted_route_requires_route_batch_evidence_scope() -> None:
    route = _synthetic_route(
        "gemini_3_1_pro", "google", "gcp/google/gemini-3.1-pro-preview"
    )
    _authority, provider, scope = _route_authority(
        "gemini_3_1_pro", routes={"gemini_3_1_pro": route}
    )
    assert provider == "eva_adapter_gemini_3_1_pro"
    assert scope == "adapted_route_batch_unbound"
    route.provider_family = "openai"
    with pytest.raises(FrontierPrefixSFTError, match="definition differs"):
        _route_authority("gemini_3_1_pro", routes={"gemini_3_1_pro": route})


@pytest.mark.skipif(not CANARY.exists(), reason="generated canary is not checked into git")
def test_real_opus_canary_exports_one_verified_s1_prefix(tmp_path: Path) -> None:
    registry = load_and_compile_registry(
        ROOT / "rubrics/source/domain-stage-tables.v1.json"
    )
    build = build_frontier_prefix_sft_dataset(
        source_root=CANARY,
        source_base=ROOT,
        bulk_root=ROOT / "runs/bulk-rl-sandboxes.v1",
        output_root=tmp_path,
        registry=registry,
    )
    report = verify_frontier_prefix_sft_dataset(
        dataset_root=build.dataset_root,
        source_base=ROOT,
        registry=registry,
    )
    assert report["valid"] is True
    assert build.source_count == build.slice_count == 1
    row = json.loads((build.dataset_root / "shards/part-00000.jsonl").read_text())
    assert row["source_terminal_status"] == "later_turn_failed"
    assert row["accepted_frontier_status"] == "completed"
    assert row["artifact_relative_path"] == "work/stage-plan.json"
    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert row["later_stage_content_included"] is False
    assert row["terminal_answer_included"] is False
    assert row["agent_judged"] is False


@pytest.mark.skipif(
    not WAVE7_COMPLETED.exists(), reason="generated wave7 trajectory is not checked into git"
)
def test_completed_trajectory_keeps_provider_and_codex_receipts_distinct() -> None:
    registry = load_and_compile_registry(
        ROOT / "rubrics/source/domain-stage-tables.v1.json"
    )
    source = json.loads(WAVE7_COMPLETED.read_text())
    nested = source["provider_metadata"]["codex_turn_receipt"]["receipt_blake3"]
    assert source["provider_receipt_blake3"] != nested
    row = _source_slice(
        source_root=ROOT / "runs/codex-teacher-opus-s1-wave7.v2",
        source_base=ROOT,
        bulk_root=ROOT / "runs/bulk-rl-sandboxes.v1",
        registry=registry,
        source_document_path=WAVE7_COMPLETED,
    )
    assert row["codex_turn_receipt_blake3"] == nested
    assert (
        row["source_provider_rollout_receipt_blake3"]
        == source["provider_receipt_blake3"]
    )
    assert row["provider_rollout_receipt_recomputed"] is False
    assert row["codex_turn_receipt_reopened"] is True
    assert row["adapter_evidence_scope"] == "route_batch_unbound"
