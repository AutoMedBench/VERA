from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from eva_agent.codex_pipeline.native_policy_v2 import (
    NATIVE_POLICY_V2_CONFIG_OVERRIDES,
    NATIVE_POLICY_V2_NATIVE_SURFACE,
    STAGE_TOOL_GUIDANCE_V1_SCHEMA,
    NativePolicyV2Error,
    advance_guided_stage_frontier_v1,
    build_stage_tool_guidance_v1,
    build_native_policy_v2,
    guard_guided_stage_tool_runtime_v1,
    materialize_s2_selection_arguments_v1,
    native_policy_v2_config_overrides,
    start_guided_stage_frontier_v1,
    verify_guided_stage_call_v1,
    verify_native_policy_document_v2,
    verify_guided_stage_frontier_state_v1,
    verify_stage_tool_guidance_document_v1,
    verify_stage_tool_guidance_v1,
)
from eva_agent.codex_runtime import (
    CodexEvent,
    CodexRole,
    CodexSandbox,
    CodexSkill,
    CodexThreadOptions,
    CodexToolCall,
    CodexTurnReceipt,
)
from eva_agent.deployment.native_fastpath_v2 import (
    NATIVE_FASTPATH_V2_ACTOR_INSTRUCTION,
    NATIVE_FASTPATH_V2_JUDGE_INSTRUCTION,
    NativeFastpathV2Error,
    prepare_native_fastpath_v2,
    verify_native_actor_judge_pair_v2,
    verify_native_actor_receipt_v2,
    verify_native_judge_receipt_v2,
    verify_native_rollout_effect_binding_v2,
)
from eva_agent.pipeline import Stage
from eva_agent.pipeline.contracts import ToolCall, ToolResult
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from eva_agent.pipeline.ids import DeterministicUUIDFactory


def _turn_mcp_metadata(tmp_path: Path) -> dict[str, object]:
    private_root = (tmp_path / "turn-mcp-private").resolve()
    private_root.mkdir(exist_ok=True)
    private_root.chmod(0o700)
    proxy_python = Path(sys.executable).resolve()
    source_root = Path(__file__).resolve().parents[1] / "src" / "eva_agent"
    proxy_exec = (source_root / "codex_pipeline" / "turn_mcp_exec.py").resolve()
    proxy_script = (source_root / "codex_pipeline" / "turn_mcp_proxy.py").resolve()
    random_component_bytes = 8
    calculated_socket_path_bytes = (
        len(os.fsencode(str(private_root)))
        + 1
        + len(os.fsencode("eva-turn-mcp-"))
        + random_component_bytes
        + 1
        + len(os.fsencode("broker.sock"))
    )
    probed_socket_path_bytes = min(calculated_socket_path_bytes, 107)
    core = {
        "schema": "eva.turn-mcp-bridge-launch.v1",
        "protocol_version": "2025-06-18",
        "server_version": "0.1.0",
        "proxy_python": str(proxy_python),
        "proxy_python_blake3": blake3_bytes(proxy_python.read_bytes()),
        "proxy_exec_script": str(proxy_exec),
        "proxy_exec_script_blake3": blake3_bytes(proxy_exec.read_bytes()),
        "proxy_script": str(proxy_script),
        "proxy_script_blake3": blake3_bytes(proxy_script.read_bytes()),
        "maximum_parallel_calls": 64,
        "startup_timeout_seconds": 10.0,
        "tool_timeout_seconds": 900,
        "final_child_environment_names": (
            "EVA_TURN_MCP_MAXIMUM",
            "EVA_TURN_MCP_NONCE",
            "EVA_TURN_MCP_SOCKET",
            "LANG",
            "LC_ALL",
            "PATH",
            "TZ",
        ),
        "inherited_parent_environment": False,
        "environment_values_recorded": False,
        "turn_nonce_recorded": False,
        "unix_socket_preflight": {
            "schema": "eva.turn-mcp-unix-socket-preflight.v1",
            "temp_root": str(private_root),
            "temp_root_uid": os.getuid(),
            "temp_root_mode": 0o700,
            "temp_root_owned_by_process": True,
            "temp_root_is_symlink": False,
            "directory_prefix": "eva-turn-mcp-",
            "random_component_bytes": random_component_bytes,
            "socket_filename": "broker.sock",
            "probed_socket_path_bytes": probed_socket_path_bytes,
            "sockaddr_un_path_capacity_bytes": 108,
            "terminator_bytes": 1,
            "name_length_source": "tempfile.mkdtemp_probe_removed",
            "preflight_passed": True,
        },
    }
    return {**core, "launch_blake3": blake3_hex(core)}


def _legacy_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _legacy_runtime_materials():
    sandbox_id = "rlevo-medres-test-e2e-guidance"
    episode_id = "test-e2e-guidance-episode"
    evidence_id = "test-evidence-guidance"
    statement_ids = ["test-guidance-statement-a", "test-guidance-statement-b"]
    s1 = {
        "schema": "rlevo.med-research-stage-plan-contract.v1",
        "contract_id": "test.stage-plan-contract.guidance",
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
        "focus": "E2E",
        "policy_sha256": "a" * 64,
        "task_contract": {
            "task_ids": ["test.guidance.task"],
            "final_artifact_relative_path": "work/result.json",
            "final_artifact_schema_sha256": "b" * 64,
        },
        "host_inputs": [
            {
                "input_id": evidence_id,
                "inspection_tool": "retrieve_frozen_evidence",
            }
        ],
        "stage_artifacts": {
            "S1": "work/stage-plan.json",
            "S2": "work/evidence-selection.json",
            "S3": "work/pilot.json",
            "S4": "work/result.json",
            "S5": "work/submission-receipt.json",
        },
        "budgets": {
            "max_turns": 10,
            "wall_time_seconds": 300,
            "minimum_s5_reserved_turns": 2,
            "max_clean_retries_per_execution_stage": 1,
        },
        "max_plan_bytes": 16384,
    }
    s2 = {
        "schema": "rlevo.med-research-evidence-service-contract.v1",
        "contract_id": "test.evidence-service.guidance",
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
        "focus": "E2E",
        "policy_sha256": "a" * 64,
        "s1_plan_contract_sha256": _legacy_sha256(s1),
        "s1_receipt_sha256": "0" * 64,
        "question": "What result follows from the frozen test evidence only?",
        "evidence_need": "Use the one bound frozen evidence object.",
        "evidence_objects": [
            {
                "evidence_id": evidence_id,
                "relative_path": "evidence/test-evidence-guidance.json",
                "byte_count": 128,
                "sha256": "c" * 64,
                "source_id": "test:guidance",
                "source_revision": "revision-1",
                "license_id": "test-license",
                "redistribution": "permitted",
                "statement_ids": statement_ids,
            }
        ],
        "required_evidence_ids": [evidence_id],
        "required_claim_ids": ["test-guidance-claim"],
        "limits": {
            "max_retrievals": 1,
            "min_selected": 1,
            "max_selected": 1,
            "max_selection_bytes": 16384,
            "max_selection_attempts": 1,
        },
        "selection_relative_path": "work/evidence-selection.json",
    }
    catalog = [
        {
            "name": "materialize_evidence_selection",
            "description": "Materialize the episode-bound S2 evidence selection.",
            "input_schema": {"type": "object", "additionalProperties": True},
            "visibility": "actor_public",
            "handler_origin": "signed_legacy_evamed",
            "parallel_safe": False,
        },
        {
            "name": "materialize_plan",
            "description": "Materialize the episode-bound S1 execution plan.",
            "input_schema": {"type": "object", "additionalProperties": True},
            "visibility": "actor_public",
            "handler_origin": "signed_legacy_evamed",
            "parallel_safe": False,
        },
        {
            "name": "retrieve_frozen_evidence",
            "description": "Retrieve one immutable S2 evidence object by ID.",
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id"],
                "properties": {"evidence_id": {"type": "string"}},
            },
            "visibility": "actor_public",
            "handler_origin": "signed_legacy_evamed",
            "parallel_safe": False,
        },
    ]
    catalog.sort(key=lambda row: row["name"])
    context = {
        "schema": "eva.legacy-candidate-runtime-context.v1",
        "source_candidate_id": sandbox_id,
        "source_family": "test",
        "focus": "E2E",
        "target_split": "train",
        "active_episode_id": episode_id,
        "policy_blake3": "d" * 64,
        "tool_catalog_blake3": blake3_hex(catalog),
        "s1_plan_contract": s1,
        "s2_evidence_contract": s2,
        "execution_stages": {
            "S3": {"stage": "S3"},
            "S4": {"stage": "S4"},
        },
        "s5_terminal_json_schema": {"type": "object"},
    }
    return context, catalog


def _advance_guidance_to_selection(guidance):
    ids = DeterministicUUIDFactory("guided-state-to-selection")
    state = start_guided_stage_frontier_v1(guidance)
    plan = canonical_value(guidance.frontiers[0]["arguments"])
    call_id = ids.new("call")
    proof = verify_guided_stage_call_v1(
        guidance,
        state=state,
        frontier_index=0,
        tool_call_id=call_id,
        tool_name="materialize_plan",
        arguments=plan,
    )
    state = advance_guided_stage_frontier_v1(
        guidance,
        state=state,
        frontier_index=0,
        tool_call_id=call_id,
        tool_name="materialize_plan",
        arguments=plan,
        call_proof=proof,
        tool_result=_tool_result(
            call_id=call_id,
            name="materialize_plan",
            frontier=0,
            identity="plan",
            output={
                "stage": "S1",
                "gate_passed": True,
                "next_stage": "S2",
                "evidence_contract_sha256": "e" * 64,
            },
        ),
    )
    for frontier in guidance.frontiers[1:-1]:
        arguments = canonical_value(frontier["arguments"])
        call_id = ids.new("call")
        proof = verify_guided_stage_call_v1(
            guidance,
            state=state,
            frontier_index=frontier["frontier_index"],
            tool_call_id=call_id,
            tool_name="retrieve_frozen_evidence",
            arguments=arguments,
        )
        state = advance_guided_stage_frontier_v1(
            guidance,
            state=state,
            frontier_index=frontier["frontier_index"],
            tool_call_id=call_id,
            tool_name="retrieve_frozen_evidence",
            arguments=arguments,
            call_proof=proof,
            tool_result=_tool_result(
                call_id=call_id,
                name="retrieve_frozen_evidence",
                frontier=frontier["frontier_index"],
                identity=str(frontier["frontier_index"]),
                output={
                    "stage": "S2",
                    "gate_passed": True,
                    "evidence_id": arguments["evidence_id"],
                    "retrieval_receipt_sha256": "f" * 64,
                },
            ),
        )
    return state


def _tool_result(
    *,
    call_id: str,
    name: str,
    frontier: int,
    identity: str,
    output,
) -> ToolResult:
    ids = DeterministicUUIDFactory(f"guided-tool-result:{identity}")
    core = {
        "result_id": ids.new("result"),
        "call_id": call_id,
        "name": name,
        "frontier": frontier,
        "parallel_group_id": ids.new("group"),
        "status": "completed",
        "output": output,
        "error_code": None,
        "workspace_before_blake3": "a" * 64,
        "workspace_after_blake3": "b" * 64,
    }
    return ToolResult(**core, receipt_blake3=blake3_hex(core))


def _skill(tmp_path: Path) -> CodexSkill:
    path = (tmp_path / "SKILL.md").resolve()
    content = b"# Evidence synthesis\n"
    path.write_bytes(content)
    return CodexSkill(
        skill_id="evidence-synthesis",
        name="evidence-synthesis",
        path=str(path),
        content_blake3=blake3_bytes(content),
    )


def _policy(tmp_path: Path, stage: Stage = Stage.S3):
    return build_native_policy_v2(
        stage=stage,
        sdk_version="0.147.0",
        sdk_protocol="app-server/experimental",
        cli_version="0.147.0",
        cli_executable_blake3="2" * 64,
        skills=(_skill(tmp_path),),
        turn_mcp_metadata=_turn_mcp_metadata(tmp_path),
    )


def _actor_config(root: Path) -> dict[str, object]:
    return {
        "mcp_servers": {"evamed": {"required": True}},
        "features": {"shell_tool": False, "unified_exec": False},
        "sandbox_workspace_write": {
            "network_access": False,
            "writable_roots": [str(root)],
            "exclude_slash_tmp": True,
            "exclude_tmpdir_env_var": True,
        },
    }


def _options(tmp_path: Path):
    actor_root = (tmp_path / "actor").resolve()
    judge_root = (tmp_path / "judge").resolve()
    actor_root.mkdir()
    judge_root.mkdir()
    actor = CodexThreadOptions(
        role=CodexRole.STRONG_ACTOR,
        model="openai/gpt-5.6-sol",
        provider="openai",
        cwd=str(actor_root),
        sandbox=CodexSandbox.WORKSPACE_WRITE,
        config=_actor_config(actor_root),
    )
    judge = CodexThreadOptions(
        role=CodexRole.JUDGE,
        model="anthropic/claude-opus-5",
        provider="anthropic",
        cwd=str(judge_root),
        sandbox=CodexSandbox.READ_ONLY,
        config={"features": {"shell_tool": False, "unified_exec": False}},
    )
    return actor, judge


def _deployment(tmp_path: Path, stage: Stage = Stage.S4):
    actor, judge = _options(tmp_path)
    skill = _skill(tmp_path)
    context = catalog = None
    candidate_id = str(uuid5(NAMESPACE_URL, "native-fastpath-candidate"))
    if stage is Stage.E2E:
        context, catalog = _legacy_runtime_materials()
        candidate_id = context["source_candidate_id"]
    return prepare_native_fastpath_v2(
        candidate_id=candidate_id,
        stage=stage,
        actor_options=actor,
        judge_options=judge,
        actor_skills=(skill,),
        sdk_version="0.147.0",
        sdk_protocol="app-server/experimental",
        cli_version="0.147.0",
        cli_executable_blake3="2" * 64,
        turn_mcp_metadata=_turn_mcp_metadata(tmp_path),
        applied_config_overrides=native_policy_v2_config_overrides(actor.cwd),
        public_runtime_context=context,
        source_tool_catalog=catalog,
    )


def _event(ids, sequence: int, method: str, thread: str, turn: str, payload):
    core = {
        "event_id": ids.new("event"),
        "sequence": sequence,
        "method": method,
        "thread_id": thread,
        "turn_id": turn,
        "payload": payload,
        "content_redacted": False,
    }
    return CodexEvent(**core, event_blake3=blake3_hex(core))


def _receipt(
    deployment,
    *,
    judge: bool = False,
    tool_type: str | None = "fileChange",
    path: str = "analysis/result.md",
    identity: str = "a",
    resumed: bool = False,
):
    ids = DeterministicUUIDFactory(f"native-receipt:{identity}")
    options = deployment.judge_options if judge else deployment.actor_options
    thread = f"thread-{identity}"
    turn = f"turn-{identity}"
    calls = ()
    events = [_event(ids, 0, "turn/started", thread, turn, {"turn": {"id": turn}})]
    if tool_type is not None:
        item = {"id": f"item-{identity}", "type": tool_type, "status": "inProgress"}
        if tool_type == "fileChange":
            arguments = {
                "changes": (
                    {"path": path, "kind": {"type": "add"}, "diff": "+result\n"},
                )
            }
            output = {"status": "completed"}
        else:
            arguments = {"command": "pwd", "commandActions": (), "cwd": options.cwd}
            output = {"aggregatedOutput": options.cwd, "durationMs": 1, "exitCode": 0}
        events.append(_event(ids, 1, "item/started", thread, turn, {"item": item}))
        completed_item = {**item, "status": "completed"}
        events.append(
            _event(ids, 2, "item/completed", thread, turn, {"item": completed_item})
        )
        call_core = {
            "tool_call_id": ids.new("tool-call"),
            "upstream_item_id": item["id"],
            "tool_type": tool_type,
            "name": tool_type,
            "mcp_server": None,
            "mcp_tool": None,
            "fully_qualified_name": None,
            "status": "completed",
            "arguments": arguments,
            "output": output,
            "lifecycle": ("item/started", "item/completed"),
            "first_event_sequence": 1,
        }
        calls = (CodexToolCall(**call_core, receipt_blake3=blake3_hex(call_core)),)
    events.append(
        _event(ids, len(events), "turn/completed", thread, turn, {"turn": {"id": turn}})
    )
    skills = () if judge else deployment.policy.skills
    core = {
        "schema": "eva.codex-turn-receipt.v1",
        "receipt_id": ids.new("receipt"),
        "runtime_thread_id": ids.new("runtime-thread"),
        "runtime_turn_id": ids.new("runtime-turn"),
        "thread_id": thread,
        "turn_id": turn,
        "role": options.role,
        "model": options.model,
        "provider": options.provider,
        "sandbox": options.sandbox,
        "thread_resumed": resumed,
        "visibility": "judge-only" if judge else "actor-public",
        "status": "completed",
        "final_response": "done",
        "events": tuple(events),
        "tool_calls": calls,
        "selected_skill_ids": tuple(skill.skill_id for skill in skills),
        "selected_skill_catalog_blake3": blake3_hex(
            tuple(skill.catalog_entry() for skill in skills)
        ),
        "offered_mcp_tool_names": (),
        "offered_tool_schema_blake3": blake3_hex(()),
        "max_parallelism_observed": 1 if calls else 0,
        "parallel_tool_calls_supported": True,
        "usage": {},
        "input_blake3": blake3_hex({"input": identity}),
        "config_keys": options.config_keys,
        "config_values_recorded": False,
        "input_payload_recorded": False,
        "sdk_version": deployment.policy.sdk_version,
        "server_version": deployment.policy.cli_version,
    }
    return CodexTurnReceipt(**core, receipt_blake3=blake3_hex(core))


def _effect_metadata(receipt: CodexTurnReceipt):
    call = receipt.tool_calls[0]
    change = canonical_value(call.arguments["changes"][0])
    file_state = {
        "content_blake3": "5" * 64,
        "byte_count": 7,
        "mode": "0644",
    }
    effect_core = {
        "tool_call_id": call.tool_call_id,
        "upstream_item_id": call.upstream_item_id,
        "change_index": 0,
        "kind": "add",
        "source_path": change["path"],
        "destination_path": None,
        "diff_blake3": blake3_bytes(change["diff"].encode("utf-8")),
        "path_effects": (
            {"path": change["path"], "before": None, "after": file_state},
        ),
    }
    effect = {**effect_core, "effect_blake3": blake3_hex(effect_core)}
    binding_core = {
        "schema": "eva.codex-native-file-change-effect-binding.v1",
        "workspace_before_tree_blake3": "3" * 64,
        "workspace_after_tree_blake3": "4" * 64,
        "declared_changed_paths": (change["path"],),
        "actual_changed_paths": (change["path"],),
        "effects": (effect,),
    }
    binding = {**binding_core, "binding_blake3": blake3_hex(binding_core)}
    return {
        "schema": "eva.codex-provider-rollout-projection.v2-native-file-change",
        "codex_turn_receipt": canonical_value(receipt),
        "tool_call_groups": (),
        "codex_to_pipeline_call_ids": {},
        "raw_input_recorded": False,
        "semantic_retry_count": 0,
        "native_file_change_effect_binding": binding,
    }


@pytest.mark.parametrize("stage", (Stage.S3, Stage.S4, Stage.E2E))
def test_policy_is_deterministic_and_binds_runtime_catalogs(
    tmp_path: Path, stage: Stage
) -> None:
    first = _policy(tmp_path, stage)
    second = _policy(tmp_path, stage)
    assert first.policy_blake3 == second.policy_blake3
    document = first.to_document()
    assert verify_native_policy_document_v2(document) == first.policy_blake3
    assert document["native_action_types"] == ["fileChange"]
    assert document["native_surface"] == dict(NATIVE_POLICY_V2_NATIVE_SURFACE)
    assert document["thread_lifecycle"]["resume_allowed"] is False
    assert document["judge_boundary"]["mode"] == "read-only"
    assert document["skill_input_binding"]["sdk_input_type"] == "SkillInput"
    assert document["turn_mcp_binding"]["launch_blake3"] == _turn_mcp_metadata(tmp_path)[
        "launch_blake3"
    ]
    assert document["schema_compatibility"] == {
        "evamed_schema_changed": False,
        "mcp_schema_changed": False,
        "rubric_schema_changed": False,
        "sandbox_schema_changed": False,
    }


@pytest.mark.parametrize("stage", (Stage.S1, Stage.S2, Stage.S5))
def test_policy_rejects_non_coding_stages(tmp_path: Path, stage: Stage) -> None:
    with pytest.raises(NativePolicyV2Error, match="only S3, S4, or E2E"):
        _policy(tmp_path, stage)


def test_policy_rejects_unpinned_sdk_and_tampered_turn_mcp(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    with pytest.raises(NativePolicyV2Error, match="pinned Tier-A"):
        build_native_policy_v2(
            stage=Stage.S3,
            sdk_version="0.148.0",
            sdk_protocol="experimental",
            cli_version="0.147.0",
            cli_executable_blake3="2" * 64,
            skills=(skill,),
            turn_mcp_metadata=_turn_mcp_metadata(tmp_path),
        )

    skill_root = (tmp_path / "real-skill-root").resolve()
    skill_root.mkdir()
    skill_file = skill_root / "SKILL.md"
    skill_payload = b"# No-follow skill\n"
    skill_file.write_bytes(skill_payload)
    linked_root = (tmp_path / "linked-skill-root").resolve()
    linked_root.symlink_to(skill_root, target_is_directory=True)
    linked_skill = CodexSkill(
        skill_id="linked-skill",
        name="linked-skill",
        path=str(linked_root / "SKILL.md"),
        content_blake3=blake3_bytes(skill_payload),
    )
    with pytest.raises(NativePolicyV2Error, match="path contains a symlink"):
        build_native_policy_v2(
            stage=Stage.S3,
            sdk_version="0.147.0",
            sdk_protocol="experimental",
            cli_version="0.147.0",
            cli_executable_blake3="2" * 64,
            skills=(linked_skill,),
            turn_mcp_metadata=_turn_mcp_metadata(tmp_path),
        )
    tampered = _turn_mcp_metadata(tmp_path)
    tampered["maximum_parallel_calls"] = 65
    with pytest.raises(NativePolicyV2Error, match="TurnMCP.*BLAKE3"):
        build_native_policy_v2(
            stage=Stage.S3,
            sdk_version="0.147.0",
            sdk_protocol="experimental",
            cli_version="0.147.0",
            cli_executable_blake3="2" * 64,
            skills=(skill,),
            turn_mcp_metadata=tampered,
        )

    fake_skill = CodexSkill(
        skill_id="tampered-skill",
        name="tampered-skill",
        path=_skill(tmp_path).path,
        content_blake3="1" * 64,
    )
    with pytest.raises(NativePolicyV2Error, match="SkillInput content"):
        build_native_policy_v2(
            stage=Stage.S3,
            sdk_version="0.147.0",
            sdk_protocol="experimental",
            cli_version="0.147.0",
            cli_executable_blake3="2" * 64,
            skills=(fake_skill,),
            turn_mcp_metadata=_turn_mcp_metadata(tmp_path),
        )


def test_public_guidance_prevents_empty_first_call_and_serializes_s1_s2() -> None:
    context, catalog = _legacy_runtime_materials()
    first = build_stage_tool_guidance_v1(
        public_runtime_context=context,
        source_tool_catalog=catalog,
    )
    second = build_stage_tool_guidance_v1(
        public_runtime_context=context,
        source_tool_catalog=catalog,
    )
    assert first.to_document() == second.to_document()
    assert first.to_document()["schema"] == STAGE_TOOL_GUIDANCE_V1_SCHEMA
    assert verify_stage_tool_guidance_v1(
        first,
        public_runtime_context=context,
        source_tool_catalog=catalog,
    ) == first.guidance_blake3
    assert verify_stage_tool_guidance_document_v1(
        first.to_document(),
        public_runtime_context=context,
        source_tool_catalog=catalog,
    ) == first.guidance_blake3
    assert "Never call it with {}" in first.prompt_text
    assert "S1 FIRST" in first.prompt_text
    assert "Do not probe tools or schemas" in first.prompt_text
    assert "schema-invalid protocol call is rejected" in first.prompt_text
    assert first.prompt_blake3 == blake3_bytes(first.prompt_text.encode("utf-8"))
    assert all(row["parallel_allowed"] is False for row in first.frontiers)
    assert [row["tool_name"] for row in first.frontiers] == [
        "materialize_plan",
        "retrieve_frozen_evidence",
        "materialize_evidence_selection",
    ]

    plan = canonical_value(first.frontiers[0]["arguments"])
    state = start_guided_stage_frontier_v1(first)
    call_id = str(uuid5(NAMESPACE_URL, "guided-first-call"))
    assert verify_guided_stage_frontier_state_v1(first, state) == state.state_blake3
    assert plan
    assert set(plan) == {
        "schema",
        "contract_sha256",
        "sandbox_id",
        "episode_id",
        "objective",
        "deliverable",
        "host_inputs",
        "pipeline",
        "candidate_method",
        "uncertainties",
        "budgets",
        "stop_rule",
        "recovery",
        "claims_unseen_results",
    }
    proof = verify_guided_stage_call_v1(
        first,
        state=state,
        frontier_index=0,
        tool_call_id=call_id,
        tool_name="materialize_plan",
        arguments=plan,
    )
    assert proof["guidance_blake3"] == first.guidance_blake3
    assert len(proof["proof_blake3"]) == 64

    with pytest.raises(NativePolicyV2Error, match="must not be empty"):
        verify_guided_stage_call_v1(
            first,
            state=state,
            frontier_index=0,
            tool_call_id=call_id,
            tool_name="materialize_plan",
            arguments={},
        )
    with pytest.raises(NativePolicyV2Error, match="frontier"):
        verify_guided_stage_call_v1(
            first,
            state=state,
            frontier_index=0,
            tool_call_id=call_id,
            tool_name="materialize_plan",
            arguments=plan,
            concurrent_call_count=2,
        )
    with pytest.raises(NativePolicyV2Error, match="tool or parallel frontier"):
        verify_guided_stage_call_v1(
            first,
            state=state,
            frontier_index=0,
            tool_call_id=call_id,
            tool_name="retrieve_frozen_evidence",
            arguments={"evidence_id": "wrong-evidence"},
        )

    with pytest.raises(NativePolicyV2Error, match="frontier"):
        verify_guided_stage_call_v1(
            first,
            state=state,
            frontier_index=1,
            tool_call_id=call_id,
            tool_name="retrieve_frozen_evidence",
            arguments={"evidence_id": "test-evidence-guidance"},
        )


def test_s2_guidance_starts_after_host_hydrated_s1_without_schema_changes() -> None:
    context, catalog = _legacy_runtime_materials()
    s1 = dict(context["s1_plan_contract"])
    s1["focus"] = "S2"
    s2 = dict(context["s2_evidence_contract"])
    s2["focus"] = "S2"
    s2["s1_plan_contract_sha256"] = _legacy_sha256(s1)
    context = {
        **context,
        "focus": "S2",
        "s1_plan_contract": s1,
        "s2_evidence_contract": s2,
    }
    guidance = build_stage_tool_guidance_v1(
        public_runtime_context=context,
        source_tool_catalog=catalog,
    )

    assert guidance.focus is Stage.S2
    assert guidance.to_document()["first_frontier_index"] == 1
    assert "HOST_HYDRATED_S1_FRONTIER=" in guidance.prompt_text
    assert "Do not call materialize_plan" in guidance.prompt_text
    assert "Start with each listed retrieve_frozen_evidence call" in guidance.prompt_text
    assert guidance.frontiers[0]["tool_name"] == "materialize_plan"
    assert guidance.frontiers[1]["tool_name"] == "retrieve_frozen_evidence"
    assert verify_stage_tool_guidance_v1(
        guidance,
        public_runtime_context=context,
        source_tool_catalog=catalog,
    ) == guidance.guidance_blake3


def test_s3_guidance_starts_after_host_hydrated_s1_s2_only() -> None:
    context, catalog = _legacy_runtime_materials()
    s1 = dict(context["s1_plan_contract"])
    s1["focus"] = "S3"
    s2 = dict(context["s2_evidence_contract"])
    s2["focus"] = "S3"
    s2["s1_plan_contract_sha256"] = _legacy_sha256(s1)
    context = {
        **context,
        "focus": "S3",
        "s1_plan_contract": s1,
        "s2_evidence_contract": s2,
    }
    guidance = build_stage_tool_guidance_v1(
        public_runtime_context=context,
        source_tool_catalog=catalog,
    )

    assert guidance.focus is Stage.S3
    assert guidance.to_document()["first_frontier_index"] == len(
        guidance.frontiers
    )
    assert "Continue at S3 only" in guidance.prompt_text
    assert "work/stage-plan.json" in guidance.prompt_text
    assert "work/evidence-selection.json" in guidance.prompt_text
    assert "FIRST_FRONTIER=" not in guidance.prompt_text
    assert "S2_RETRIEVAL_FRONTIERS=" not in guidance.prompt_text
    assert "S2_SELECTION_BLUEPRINT=" not in guidance.prompt_text
    assert verify_stage_tool_guidance_v1(
        guidance,
        public_runtime_context=context,
        source_tool_catalog=catalog,
    ) == guidance.guidance_blake3


def test_s3_extension_preserves_exact_s1_s2_guidance_bytes() -> None:
    context, catalog = _legacy_runtime_materials()
    expected = {
        "S1": (
            "0b41e34c65800e695897cb9f0c85a4a8a3d9c06b5b280911ce8aeb7465b7c86b",
            "d4fb2a93ccbbd509a3d97ba97cbf566925f6036042e663ec26e9c4913f145d73",
        ),
        "S2": (
            "e97f877ab25d438819b5d0630becbd8759870280363c7a47e56fcb98b4015c9f",
            "1db07c11dd1c34aaf2fefebc257d3ee4737c0375a5805d23410b3b38fb627e6a",
        ),
    }
    for focus, digests in expected.items():
        s1 = dict(context["s1_plan_contract"])
        s1["focus"] = focus
        s2 = dict(context["s2_evidence_contract"])
        s2["focus"] = focus
        s2["s1_plan_contract_sha256"] = _legacy_sha256(s1)
        guidance = build_stage_tool_guidance_v1(
            public_runtime_context={
                **context,
                "focus": focus,
                "s1_plan_contract": s1,
                "s2_evidence_contract": s2,
            },
            source_tool_catalog=catalog,
        )
        assert (guidance.guidance_blake3, guidance.prompt_blake3) == digests


def test_public_guidance_materializes_runtime_bound_s2_selection() -> None:
    context, catalog = _legacy_runtime_materials()
    guidance = build_stage_tool_guidance_v1(
        public_runtime_context=context,
        source_tool_catalog=catalog,
    )
    state = _advance_guidance_to_selection(guidance)
    selection = materialize_s2_selection_arguments_v1(
        guidance,
        state=state,
        inference_text_by_claim_id={
            "test-guidance-claim": "The frozen evidence supports the bound claim."
        },
        unresolved_gaps=("The frozen evidence does not resolve every uncertainty.",),
        limitations=("Only the declared frozen source was available.",),
    )
    proof = verify_guided_stage_call_v1(
        guidance,
        state=state,
        frontier_index=2,
        tool_call_id=str(uuid5(NAMESPACE_URL, "guided-selection-call")),
        tool_name="materialize_evidence_selection",
        arguments=selection,
    )
    assert proof["frontier_index"] == 2
    assert selection["selected_evidence"][0]["retrieval_receipt_sha256"] == "f" * 64
    assert "$runtime" not in json.dumps(canonical_value(selection), sort_keys=True)
    assert "$author" not in json.dumps(canonical_value(selection), sort_keys=True)

    tampered_catalog = canonical_value(catalog)
    tampered_catalog[0]["description"] += " changed"
    with pytest.raises(NativePolicyV2Error, match="catalog BLAKE3"):
        build_stage_tool_guidance_v1(
            public_runtime_context=context,
            source_tool_catalog=tampered_catalog,
        )


def test_host_owned_runtime_guard_rejects_before_one_shot_effect() -> None:
    context, catalog = _legacy_runtime_materials()
    guidance = build_stage_tool_guidance_v1(
        public_runtime_context=context,
        source_tool_catalog=catalog,
    )

    class RecordingRuntime:
        def __init__(self):
            self.calls = []

        def execute(self, calls):
            self.calls.append(tuple(calls))
            call = calls[0]
            return (
                _tool_result(
                    call_id=call.call_id,
                    name=call.name,
                    frontier=0,
                    identity="host-guard",
                    output={
                        "stage": "S1",
                        "gate_passed": True,
                        "next_stage": "S2",
                        "evidence_contract_sha256": "e" * 64,
                    },
                ),
            )

        def trace(self):
            raise AssertionError("trace is not used in this focused guard test")

    ids = DeterministicUUIDFactory("host-owned-guided-runtime")
    runtime = RecordingRuntime()
    guarded = guard_guided_stage_tool_runtime_v1(runtime, guidance)
    empty = ToolCall(ids.new("call"), "materialize_plan", {})
    with pytest.raises(NativePolicyV2Error, match="must not be empty"):
        guarded.execute((empty,))
    assert runtime.calls == []

    plan = canonical_value(guidance.frontiers[0]["arguments"])
    first = ToolCall(ids.new("call"), "materialize_plan", plan)
    second = ToolCall(ids.new("call"), "materialize_plan", plan)
    with pytest.raises(NativePolicyV2Error, match="one sequential call"):
        guarded.execute((first, second))
    assert runtime.calls == []

    result = guarded.execute((first,))
    assert result[0].status == "completed"
    assert len(runtime.calls) == 1
    assert guarded.frontier_state_document["next_frontier_index"] == 1


@pytest.mark.parametrize(
    ("gate_passed", "runtime_frontier"),
    ((False, 0), (True, 1)),
)
def test_host_owned_runtime_guard_latches_after_failed_effect_validation(
    gate_passed: bool,
    runtime_frontier: int,
) -> None:
    context, catalog = _legacy_runtime_materials()
    guidance = build_stage_tool_guidance_v1(
        public_runtime_context=context,
        source_tool_catalog=catalog,
    )

    class InvalidResultRuntime:
        def __init__(self):
            self.call_count = 0

        def execute(self, calls):
            self.call_count += 1
            call = calls[0]
            return (
                _tool_result(
                    call_id=call.call_id,
                    name=call.name,
                    frontier=runtime_frontier,
                    identity=f"invalid-{gate_passed}-{runtime_frontier}",
                    output={
                        "stage": "S1",
                        "gate_passed": gate_passed,
                        "next_stage": "S2",
                        "evidence_contract_sha256": "e" * 64,
                    },
                ),
            )

        def trace(self):
            raise AssertionError("trace is not used in this guard test")

    ids = DeterministicUUIDFactory(
        f"host-owned-invalid-result-{gate_passed}-{runtime_frontier}"
    )
    runtime = InvalidResultRuntime()
    guarded = guard_guided_stage_tool_runtime_v1(runtime, guidance)
    plan = canonical_value(guidance.frontiers[0]["arguments"])
    first = ToolCall(ids.new("call"), "materialize_plan", plan)
    expected = "did not pass" if not gate_passed else "receipt differs"
    with pytest.raises(NativePolicyV2Error, match=expected):
        guarded.execute((first,))
    assert runtime.call_count == 1

    retry = ToolCall(ids.new("call"), "materialize_plan", plan)
    with pytest.raises(NativePolicyV2Error, match="permanently failed"):
        guarded.execute((retry,))
    assert runtime.call_count == 1


def test_deployment_seals_one_root_offline_actor_and_read_only_judge(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    document = deployment.to_document()
    assert document["actor"]["sandbox"] == "workspace-write"
    assert document["actor"]["native_action_types"] == ["fileChange"]
    assert document["judge"]["sandbox"] == "read-only"
    assert document["judge"]["native_actions_allowed"] is False
    assert NATIVE_FASTPATH_V2_ACTOR_INSTRUCTION in (
        deployment.actor_options.developer_instructions or ""
    )
    assert NATIVE_FASTPATH_V2_JUDGE_INSTRUCTION in (
        deployment.judge_options.developer_instructions or ""
    )
    assert document["launch_binding"]["config_overrides"] == list(
        native_policy_v2_config_overrides(deployment.actor_options.cwd)
    )
    assert "{candidate_root}" in " ".join(NATIVE_POLICY_V2_CONFIG_OVERRIDES)


def test_e2e_deployment_binds_nonempty_sequential_stage_guidance(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path, Stage.E2E)
    guidance = deployment.stage_tool_guidance
    assert guidance is not None
    binding = deployment.to_document()["stage_tool_guidance_binding"]
    assert binding["source_candidate_id"] == deployment.candidate_id
    assert binding["guidance_blake3"] == guidance.guidance_blake3
    assert binding["prompt_blake3"] == guidance.prompt_blake3
    assert binding["host_pre_effect_guard"] == "GuidedStageToolRuntimeV1"
    assert guidance.prompt_text in (
        deployment.actor_options.developer_instructions or ""
    )

    with pytest.raises(NativeFastpathV2Error, match="guidance identity"):
        replace(deployment, candidate_id="different-source-candidate")

    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    actor, judge = _options(mismatch_root)
    context, catalog = _legacy_runtime_materials()
    with pytest.raises(NativeFastpathV2Error, match="guidance identity"):
        prepare_native_fastpath_v2(
            candidate_id="different-source-candidate",
            stage=Stage.E2E,
            actor_options=actor,
            judge_options=judge,
            actor_skills=(),
            sdk_version="0.147.0",
            sdk_protocol="app-server/experimental",
            cli_version="0.147.0",
            cli_executable_blake3="2" * 64,
            turn_mcp_metadata=_turn_mcp_metadata(mismatch_root),
            applied_config_overrides=native_policy_v2_config_overrides(actor.cwd),
            public_runtime_context=context,
            source_tool_catalog=catalog,
        )


def test_deployment_rejects_network_extra_root_and_non_read_only_judge(
    tmp_path: Path,
) -> None:
    actor, judge = _options(tmp_path)
    config = _actor_config(Path(actor.cwd))
    config["sandbox_workspace_write"]["network_access"] = True
    unsafe_actor = replace(actor, config=config)
    with pytest.raises(NativeFastpathV2Error, match="network|offline sandbox"):
        prepare_native_fastpath_v2(
            candidate_id="candidate",
            stage=Stage.S3,
            actor_options=unsafe_actor,
            judge_options=judge,
            actor_skills=(),
            sdk_version="0.147.0",
            sdk_protocol="experimental",
            cli_version="0.147.0",
            cli_executable_blake3="2" * 64,
            turn_mcp_metadata=_turn_mcp_metadata(tmp_path),
            applied_config_overrides=native_policy_v2_config_overrides(actor.cwd),
        )

    config = _actor_config(Path(actor.cwd))
    config["features"]["standalone_web_search"] = True
    web_actor = replace(actor, config=config)
    with pytest.raises(NativeFastpathV2Error, match="web search"):
        prepare_native_fastpath_v2(
            candidate_id="candidate",
            stage=Stage.S3,
            actor_options=web_actor,
            judge_options=judge,
            actor_skills=(),
            sdk_version="0.147.0",
            sdk_protocol="experimental",
            cli_version="0.147.0",
            cli_executable_blake3="2" * 64,
            turn_mcp_metadata=_turn_mcp_metadata(tmp_path),
            applied_config_overrides=native_policy_v2_config_overrides(actor.cwd),
        )

    unsafe_judge = replace(judge, sandbox=CodexSandbox.WORKSPACE_WRITE)
    with pytest.raises(NativeFastpathV2Error, match="read-only"):
        prepare_native_fastpath_v2(
            candidate_id="candidate",
            stage=Stage.S3,
            actor_options=actor,
            judge_options=unsafe_judge,
            actor_skills=(),
            sdk_version="0.147.0",
            sdk_protocol="experimental",
            cli_version="0.147.0",
            cli_executable_blake3="2" * 64,
            turn_mcp_metadata=_turn_mcp_metadata(tmp_path),
            applied_config_overrides=native_policy_v2_config_overrides(actor.cwd),
        )

    write_config_judge = replace(
        judge,
        config={
            "features": {"shell_tool": False, "unified_exec": False},
            "sandbox_workspace_write": {
                "network_access": False,
                "writable_roots": [judge.cwd],
                "exclude_slash_tmp": True,
                "exclude_tmpdir_env_var": True,
            },
        },
    )
    with pytest.raises(NativeFastpathV2Error, match="writable roots|write sandbox"):
        prepare_native_fastpath_v2(
            candidate_id="candidate",
            stage=Stage.S3,
            actor_options=actor,
            judge_options=write_config_judge,
            actor_skills=(),
            sdk_version="0.147.0",
            sdk_protocol="experimental",
            cli_version="0.147.0",
            cli_executable_blake3="2" * 64,
            turn_mcp_metadata=_turn_mcp_metadata(tmp_path),
            applied_config_overrides=native_policy_v2_config_overrides(actor.cwd),
        )


def test_actor_receipt_accepts_only_workspace_relative_file_change(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    verify_native_actor_receipt_v2(_receipt(deployment), deployment)

    command = _receipt(deployment, tool_type="commandExecution", identity="command")
    with pytest.raises(NativeFastpathV2Error, match="forbidden non-fileChange"):
        verify_native_actor_receipt_v2(command, deployment)

    traversal = _receipt(deployment, path="../escape.txt", identity="traversal")
    with pytest.raises(NativeFastpathV2Error, match="workspace-relative"):
        verify_native_actor_receipt_v2(traversal, deployment)


def test_judge_is_native_free_fresh_and_separate(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path)
    actor = _receipt(deployment, identity="actor")
    judge = _receipt(deployment, judge=True, tool_type=None, identity="judge")
    verify_native_judge_receipt_v2(judge, deployment)
    verify_native_actor_judge_pair_v2(actor, judge, deployment)

    native_judge = _receipt(
        deployment, judge=True, tool_type="fileChange", identity="native-judge"
    )
    with pytest.raises(NativeFastpathV2Error, match="judge emitted a native action"):
        verify_native_judge_receipt_v2(native_judge, deployment)

    resumed = _receipt(deployment, identity="resumed", resumed=True)
    with pytest.raises(NativeFastpathV2Error, match="binding differs"):
        verify_native_actor_receipt_v2(resumed, deployment)


def test_projection_v2_effect_binding_is_cross_bound_to_actor_receipt(
    tmp_path: Path,
) -> None:
    deployment = _deployment(tmp_path)
    receipt = _receipt(deployment, identity="effect")
    metadata = _effect_metadata(receipt)
    digest = verify_native_rollout_effect_binding_v2(
        metadata,
        receipt,
        deployment,
        workspace_before_tree_blake3="3" * 64,
        workspace_after_tree_blake3="4" * 64,
    )
    assert digest == metadata["native_file_change_effect_binding"]["binding_blake3"]

    tampered = canonical_value(metadata)
    tampered["native_file_change_effect_binding"]["effects"][0]["path_effects"][0][
        "after"
    ]["byte_count"] = 8
    with pytest.raises(NativeFastpathV2Error, match="binding commitment"):
        verify_native_rollout_effect_binding_v2(
            tampered,
            receipt,
            deployment,
            workspace_before_tree_blake3="3" * 64,
            workspace_after_tree_blake3="4" * 64,
        )

    integer_mode = canonical_value(metadata)
    binding = integer_mode["native_file_change_effect_binding"]
    effect = binding["effects"][0]
    effect["path_effects"][0]["after"]["mode"] = 0o644
    effect_core = {key: value for key, value in effect.items() if key != "effect_blake3"}
    effect["effect_blake3"] = blake3_hex(effect_core)
    binding_core = {
        key: value for key, value in binding.items() if key != "binding_blake3"
    }
    binding["binding_blake3"] = blake3_hex(binding_core)
    with pytest.raises(NativeFastpathV2Error, match="file state differs"):
        verify_native_rollout_effect_binding_v2(
            integer_mode,
            receipt,
            deployment,
            workspace_before_tree_blake3="3" * 64,
            workspace_after_tree_blake3="4" * 64,
        )
