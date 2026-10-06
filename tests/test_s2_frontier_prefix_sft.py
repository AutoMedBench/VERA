from __future__ import annotations

import base64
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.codex_pipeline.native_policy_v2 import StageToolGuidanceV1
from eva_agent.pipeline import Stage
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from eva_agent.training.s2_frontier_prefix_sft import (
    S2_FRONTIER_DATASET_SCHEMA,
    S2_FRONTIER_SLICE_SCHEMA,
    S2FrontierPrefixSFTError,
    _authoritative_legacy_tool_catalog,
    _effective_s2_contract,
    _require_hydrated_s1,
    _require_s2_control_frontiers,
    _sha256_value,
    _workspace_snapshot_files,
)
from eva_agent.training.s2_skill_frontier_support import (
    canonical_skill_tools, require_skill_prefix, skill_export_fields, verified_offered_catalog,
)


def _skill_delivery():
    return {"schema": "eva.teacher-progressive-skill-surface.v1", "mode": "search-then-load",
            "catalog_blake3": "a" * 64, "discovery_tools": ["search_skills", "load_skill"],
            "initial_skill_mount_count": 0, "visible_skill_count": 1,
            "visible_skill_ids": ["medical-fixture"]}


def _reseal_test_call(call):
    observation = call.output["result"]["structuredContent"]
    result = observation["tool_result"]
    result["receipt_blake3"] = blake3_hex({key: value for key, value in result.items() if key != "receipt_blake3"})
    observation["bridge_receipt_blake3"] = blake3_hex({key: value for key, value in observation.items() if key != "bridge_receipt_blake3"})
    return call


def _skill_prefix_fixture(*, parallel=False):
    content = "# Medical fixture\nUse only the frozen evidence.\n"
    search = _mcp_call(name="search_skills", frontier=1,
        arguments={"query": "medical", "stage": "S2"},
        output={"matches": [{"skill_id": "medical-fixture", "description": "Medical evidence skill"}]})
    load = _mcp_call(name="load_skill", frontier=1 if parallel else 2,
        arguments={"skill_id": "medical-fixture", "stage": "S2"},
        output={"skill_id": "medical-fixture", "content": content, "content_blake3": blake3_hex(content),
                "delivery": "policy-visible-tool-observation"})
    for call in (search, load):
        result = call.output["result"]["structuredContent"]["tool_result"]
        result["workspace_after_blake3"] = result["workspace_before_blake3"]
        if parallel: result["parallel_group_id"] = "same-parallel-skill-group"
        _reseal_test_call(call)
    retrieve = _mcp_call(name="retrieve_frozen_evidence", frontier=0, arguments={"evidence_id": "e"}, output={})
    select = _mcp_call(name="materialize_evidence_selection", frontier=2 if parallel else 3, arguments={}, output={})
    return SimpleNamespace(tool_calls=(retrieve, search, load, select)), content


@pytest.mark.parametrize("parallel", (False, True))
def test_s2_v2_retains_exact_skill_content_and_actual_parallel_frontiers(parallel):
    receipt, content = _skill_prefix_fixture(parallel=parallel)
    observed, controls = require_skill_prefix(receipt, ("retrieve_frozen_evidence", "materialize_evidence_selection"), _skill_delivery())
    assert len(observed) == 4 and len(controls) == 2
    assert observed[-1].tool_result["frontier"] == (2 if parallel else 3)
    messages = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "tool"}]
    receipt.offered_tool_schema_blake3 = blake3_hex(canonical_skill_tools())
    fields = skill_export_fields(observed, _skill_delivery(), canonical_skill_tools(), receipt, messages)
    assert len(messages) == 8
    assert messages[5]["content"]["tool_result"]["output"]["content"] == content
    assert fields["loss_message_indices"] == [6]
    assert fields["loss_bearing_assistant_tool_decisions"] == 1
    assert fields["assistant_tool_decisions"] == 3
    assert fields["skill_frontiers"][-1]["frontier"] == (1 if parallel else 2)


@pytest.mark.parametrize("mutation,match", (
    ("wrong-stage", "stage"), ("changed-content", "content"),
    ("workspace-write", "read-only"), ("ungranted-skill", "visibility"),
    ("renumbered-frontier", "frontier"),
))
def test_s2_v2_rejects_resealed_skill_boundary_tampering(mutation, match):
    receipt, _content = _skill_prefix_fixture()
    call = receipt.tool_calls[2]
    observation = call.output["result"]["structuredContent"]
    if mutation == "wrong-stage":
        call.arguments["stage"] = "S3"
        observation["arguments"]["stage"] = "S3"
    elif mutation == "changed-content":
        observation["tool_result"]["output"]["content"] = "forged"
    elif mutation == "workspace-write":
        observation["tool_result"]["workspace_after_blake3"] = "f" * 64
    elif mutation == "ungranted-skill":
        call.arguments["skill_id"] = "not-visible"
        observation["arguments"]["skill_id"] = "not-visible"
        observation["tool_result"]["output"]["skill_id"] = "not-visible"
    else:
        observation["tool_result"]["frontier"] = 4
    _reseal_test_call(call)
    with pytest.raises(S2FrontierPrefixSFTError, match=match):
        require_skill_prefix(receipt, ("retrieve_frozen_evidence", "materialize_evidence_selection"), _skill_delivery())


def test_s2_v2_canonical_offered_skill_catalog_must_match_codex_commitment():
    source = ({"name": "retrieve_frozen_evidence", "description": "unchanged", "input_schema": {"type": "object"}},)
    catalog = tuple(sorted((*source, *canonical_skill_tools()), key=lambda row: row["name"]))
    receipt = SimpleNamespace(offered_mcp_tool_names=tuple("evamed/" + row["name"] for row in catalog), offered_tool_schema_blake3=blake3_hex(catalog))
    assert verified_offered_catalog(source, receipt, _skill_delivery()) == catalog
    receipt.offered_tool_schema_blake3 = "f" * 64
    with pytest.raises(S2FrontierPrefixSFTError, match="catalog"):
        verified_offered_catalog(source, receipt, _skill_delivery())


def test_s2_v1_still_rejects_interleaved_skills_without_rewriting_indices():
    receipt, _content = _skill_prefix_fixture()
    _s1, s2, guidance = _contracts_and_guidance()
    with pytest.raises(S2FrontierPrefixSFTError, match="sequence differs"):
        _require_s2_control_frontiers(receipt, guidance=guidance, s2_contract=s2)


def _contracts_and_guidance():
    sandbox_id = "rlevo-medres-fixture-s2-0000000000000000"
    episode_id = "fixture-episode-s2"
    s1 = {
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
        "stage_artifacts": {
            "S1": "work/stage-plan.json",
            "S2": "work/evidence-selection.json",
        },
    }
    evidence_id = "fixture-evidence-1"
    s2 = {
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
        "question": "What does the frozen fixture support?",
        "evidence_need": "Use only the frozen fixture.",
        "required_evidence_ids": [evidence_id],
        "evidence_objects": [{"evidence_id": evidence_id}],
        "selection_relative_path": "work/evidence-selection.json",
        "s1_receipt_sha256": "0" * 64,
    }
    plan = {
        "schema": "rlevo.med-research-stage-plan-artifact.v1",
        "contract_sha256": _sha256_value(s1),
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
    }
    frontiers = (
        {
            "frontier_index": 0,
            "stage": "S1",
            "tool_name": "materialize_plan",
            "arguments_kind": "complete-dispatchable-example",
            "arguments": plan,
            "parallel_allowed": False,
            "single_call_required": True,
            "must_observe_gate_passed_before_next": True,
        },
        {
            "frontier_index": 1,
            "stage": "S2",
            "tool_name": "retrieve_frozen_evidence",
            "arguments_kind": "exact-dispatchable-example",
            "arguments": {"evidence_id": evidence_id},
            "parallel_allowed": False,
            "single_call_required": True,
            "must_observe_gate_passed_before_next": True,
        },
        {
            "frontier_index": 2,
            "stage": "S2",
            "tool_name": "materialize_evidence_selection",
            "arguments_kind": "runtime-bound-blueprint",
            "parallel_allowed": False,
            "single_call_required": True,
            "must_observe_gate_passed_before_next": True,
        },
    )
    guidance = StageToolGuidanceV1(
        source_candidate_id=sandbox_id,
        focus=Stage.S2,
        active_episode_id=episode_id,
        source_policy_blake3="1" * 64,
        source_tool_catalog_blake3="2" * 64,
        public_runtime_context_blake3="3" * 64,
        frontiers=frontiers,
    )
    return s1, s2, guidance


def _tool_result(*, name: str, frontier: int, output: dict, before: str, after: str):
    core = {
        "result_id": f"result-{name}-{frontier}",
        "call_id": f"call-{name}-{frontier}",
        "name": name,
        "frontier": frontier,
        "parallel_group_id": f"group-{name}-{frontier}",
        "status": "completed",
        "output": output,
        "error_code": None,
        "workspace_before_blake3": before,
        "workspace_after_blake3": after,
    }
    return {**core, "receipt_blake3": blake3_hex(core)}


def _hydration_source():
    s1, s2, guidance = _contracts_and_guidance()
    receipt_sha256 = "a" * 64
    effective_s2 = copy.deepcopy(s2)
    effective_s2["s1_receipt_sha256"] = receipt_sha256
    result = _tool_result(
        name="materialize_plan",
        frontier=0,
        before="4" * 64,
        after="5" * 64,
        output={
            "stage": "S1",
            "effect": "plan_materialization",
            "gate_passed": True,
            "failed_check_ids": [],
            "next_stage": "S2",
            "receipt_sha256": receipt_sha256,
            "evidence_contract_sha256": _sha256_value(effective_s2),
        },
    )
    hydration = {
        "schema": "eva.codex-teacher-prerequisite-hydration.v1",
        "focus": "S2",
        "source_candidate_id": guidance.source_candidate_id,
        "guidance_blake3": guidance.guidance_blake3,
        "hydrated_frontier_indices": [0],
        "next_actor_frontier_index": 1,
        "artifact_relative_paths": ["work/stage-plan.json"],
        "tool_result": result,
        "workspace_before_blake3": "4" * 64,
        "workspace_after_blake3": "5" * 64,
        "provider_calls": 0,
        "canonical_tool_schemas_changed": False,
        "mcp_wire_schema_changed": False,
    }
    source = {
        "messages": [
            {"role": "system", "content": "fixture"},
            {
                "role": "user",
                "content": {"teacher_prerequisite_hydration": hydration},
            },
        ]
    }
    return source, s1, s2, guidance


def _mcp_call(*, name: str, frontier: int, arguments: dict, output: dict):
    result = _tool_result(
        name=name,
        frontier=frontier,
        before="5" * 64,
        after="5" * 64 if name == "retrieve_frozen_evidence" else "6" * 64,
        output=output,
    )
    observation_core = {
        "schema": "eva.codex-pipeline-tool-observation.v1",
        "call_id": result["call_id"],
        "name": name,
        "arguments": arguments,
        "tool_result": result,
    }
    observation = {
        **observation_core,
        "bridge_receipt_blake3": blake3_hex(observation_core),
    }
    envelope = {
        "durationMs": 1,
        "error": None,
        "result": {"_meta": None, "content": [], "structuredContent": observation},
    }
    return SimpleNamespace(
        fully_qualified_name=f"evamed/{name}",
        status="completed",
        arguments=canonical_value(arguments),
        output=canonical_value(envelope),
        tool_call_id=f"codex-{name}-{frontier}",
        receipt_blake3=f"{frontier + 7:064x}",
    )


def test_hydrated_s1_requires_exact_successful_host_observation() -> None:
    source, s1, s2, guidance = _hydration_source()
    hydration = _require_hydrated_s1(
        source, guidance=guidance, s1_contract=s1, s2_contract=s2
    )
    assert hydration["tool_result"]["output"]["gate_passed"] is True

    tampered = copy.deepcopy(source)
    tampered["messages"][1]["content"]["teacher_prerequisite_hydration"][
        "tool_result"
    ]["output"]["gate_passed"] = False
    with pytest.raises(S2FrontierPrefixSFTError, match="prerequisite S1 binding"):
        _require_hydrated_s1(
            tampered, guidance=guidance, s1_contract=s1, s2_contract=s2
        )


def test_s2_actor_receipt_starts_at_retrieval_local_frontier_zero() -> None:
    source, _s1, s2_template, guidance = _hydration_source()
    hydration = source["messages"][1]["content"]["teacher_prerequisite_hydration"]
    s2 = _effective_s2_contract(hydration, template=s2_template)
    retrieval_sha = "7" * 64
    retrieval = _mcp_call(
        name="retrieve_frozen_evidence",
        frontier=0,
        arguments={"evidence_id": "fixture-evidence-1"},
        output={
            "stage": "S2",
            "effect": "frozen_evidence_retrieval",
            "gate_passed": True,
            "failed_check_ids": [],
            "evidence_id": "fixture-evidence-1",
            "retrieval_receipt_sha256": retrieval_sha,
            "verified_evidence_content_sha256": "8" * 64,
        },
    )
    selection_arguments = {
        "schema": "rlevo.med-research-evidence-selection.v1",
        "contract_sha256": _sha256_value(s2),
        "sandbox_id": s2["sandbox_id"],
        "episode_id": s2["episode_id"],
        "question": s2["question"],
        "evidence_need": s2["evidence_need"],
        "selected_evidence": [
            {
                "evidence_id": "fixture-evidence-1",
                "retrieval_receipt_sha256": retrieval_sha,
            }
        ],
        "care_directive": False,
    }
    selection = _mcp_call(
        name="materialize_evidence_selection",
        frontier=1,
        arguments=selection_arguments,
        output={
            "stage": "S2",
            "effect": "evidence_selection_materialization",
            "gate_passed": True,
            "failed_check_ids": [],
            "next_stage": "S3",
            "receipt_sha256": "9" * 64,
        },
    )
    observed, accepted = _require_s2_control_frontiers(
        SimpleNamespace(tool_calls=(retrieval, selection)),
        guidance=guidance,
        s2_contract=s2,
    )
    assert len(observed) == 2
    assert accepted.arguments["schema"] == selection_arguments["schema"]

    bad_retrieval = _mcp_call(
        name="retrieve_frozen_evidence",
        frontier=1,
        arguments={"evidence_id": "fixture-evidence-1"},
        output=canonical_value(retrieval.output)["result"]["structuredContent"][
            "tool_result"
        ]["output"],
    )
    with pytest.raises(S2FrontierPrefixSFTError, match="gate did not pass exactly"):
        _require_s2_control_frontiers(
            SimpleNamespace(tool_calls=(bad_retrieval, selection)),
            guidance=guidance,
            s2_contract=s2,
        )


def test_workspace_snapshot_commitment_reopens_exact_bytes() -> None:
    payload = b"fixture\n"
    row = {
        "path": "work/stage-plan.json",
        "content": {"$bytes_base64": base64.b64encode(payload).decode("ascii")},
        "byte_count": len(payload),
        "mode": "0600",
        "content_blake3": blake3_bytes(payload),
    }
    core = {"files": [row], "file_count": 1, "byte_count": len(payload)}
    snapshot = {
        "label": "before-rollout",
        **core,
        "tree_blake3": blake3_hex(core),
    }
    tree, files = _workspace_snapshot_files(
        snapshot, expected_label="before-rollout"
    )
    assert tree == snapshot["tree_blake3"]
    assert files == {"work/stage-plan.json": payload}


def test_s2_schemas_are_distinct_from_s1_frontier_schemas() -> None:
    assert "s2-frontier-prefix" in S2_FRONTIER_DATASET_SCHEMA
    assert "s2-frontier-prefix" in S2_FRONTIER_SLICE_SCHEMA


def test_legacy_three_field_tool_catalog_is_strictly_reconstructed() -> None:
    minimal = [
        {
            "name": name,
            "description": f"Exact public description for {name}.",
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        }
        for name in (
            "materialize_plan",
            "retrieve_frozen_evidence",
            "materialize_evidence_selection",
            "execute_code",
            "submit_results",
        )
    ]
    authoritative = tuple(
        sorted(
            (
                {
                    **row,
                    "visibility": "actor_public",
                    "handler_origin": "signed_legacy_evamed",
                    "parallel_safe": False,
                }
                for row in minimal
            ),
            key=lambda row: row["name"],
        )
    )
    reopened = _authoritative_legacy_tool_catalog(
        minimal, expected_blake3=blake3_hex(authoritative)
    )
    assert canonical_value(reopened) == canonical_value(authoritative)

    tampered = copy.deepcopy(minimal)
    tampered[0]["description"] += " changed"
    with pytest.raises(
        S2FrontierPrefixSFTError,
        match="reconstructed tool catalog commitment differs",
    ):
        _authoritative_legacy_tool_catalog(
            tampered, expected_blake3=blake3_hex(authoritative)
        )
