"""Bind real AutoMed Codex turns to the existing exact-rubric workspace Judge.

Phase names are requested scopes, never stage-completion assertions. Clinical
native scores and private references are deliberately not inputs to this bridge.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import json
import os
from pathlib import Path
from uuid import uuid4

from eva_agent.codex_runtime import codex_turn_receipt_from_document
from eva_agent.codex_pipeline.adapter import (
    _agent_judge_terminal_object, _judge_offers, _mcp_structured_output,
    _tool_groups, _validate_tool_inventory,
)
from eva_agent.pipeline import FileSnapshot, TrajectoryEvent, WorkspaceSnapshot
from eva_agent.pipeline.codec import _judgment
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes, canonical_value, is_blake3
from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.training.agent_judge import _validated_judgment
from eva_agent.training.agent_judge_worker import judgment_document
from eva_agent.training.slime_agent_judge import grade_rollout, prepare_workspace_rollout, validate_native_judge_timeout
from training.automedbench_lite.adapter import canonical, read_document, safe_file

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("01-planning", "02-implementation", "03-review")
TRACK_FEEDBACK_RESOURCE_POLICY = {
    "schema": "eva.automedbench-full-mutable-feedback-budget.v1",
    "maximum_files_per_snapshot": 16000, "maximum_bytes_per_snapshot": 8 * 1024**3,
    "maximum_file_bytes": 512 * 1024**2, "maximum_rollout_json_bytes": 24 * 1024**3,
    "source_file_bytes_omitted": False, "source_file_bytes_synthesized": False,
    "full_binary_reads_required": False,
}


class FeedbackBridgeError(ValueError):
    """A fixed, public validation error; never include source/provider text."""


def require(condition, code):
    if not condition:
        raise FeedbackBridgeError(code)


def write_private(path: Path, value) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value))


def read_json(path: Path):
    return json.loads(safe_file(path.parent, path.name, maximum=128 * 1024 * 1024).read_bytes())


def read_feedback_rollout(path: Path, preflight):
    """Only explicitly bound full-track rollouts receive the larger host budget.

    Receipts, grades, and historical unbound case evidence keep the 128MiB
    default. This preserves every retained byte; it does not claim Judge reads.
    """
    expanded = (preflight.get("schema") == "eva.automedbench-track-feedback-preflight.v1"
        and isinstance(preflight.get("exact_rubric_registry"), dict)
        and preflight.get("snapshot_resource_policy") == TRACK_FEEDBACK_RESOURCE_POLICY)
    maximum = TRACK_FEEDBACK_RESOURCE_POLICY["maximum_rollout_json_bytes"] if expanded else 128 * 1024**2
    value = json.loads(safe_file(path.parent, path.name, maximum=maximum).read_bytes())
    if expanded:
        require(value.get("schema") == "eva.automedbench-track-workspace-feedback.v1"
            and value["provider_metadata"].get("snapshot_resource_policy") == TRACK_FEEDBACK_RESOURCE_POLICY
            and value["provider_metadata"].get("exact_rubric_registry") == preflight["exact_rubric_registry"],
            "feedback_large_rollout_scope_changed")
    return value


def snapshot(directory: Path, label: str):
    """Reopen retained bytes, not a concurrently changing live actor workspace."""
    manifest = read_document(directory / "manifest.json")
    rows = manifest["files"]
    require(manifest["schema"] == "eva.automedbench-workspace-snapshot.v1"
            and manifest["retained_bytes_relative"] == "files"
            and manifest["tree_blake3"] == blake3_bytes(canonical_json_bytes(rows)), "snapshot_manifest_binding")
    files = []
    for row in rows:
        path = row["path"]
        require(path == "task.json" or path.startswith(("inputs/", "notes/", "outputs/")),
                "nonpublic_snapshot_path")
        payload = safe_file(directory / "files", path, maximum=128 * 1024 * 1024).read_bytes()
        require(len(payload) == row["bytes"] and blake3_bytes(payload) == row["blake3"],
                "snapshot_payload_commitment")
        require(type(row.get("mode")) is int and 0 <= row["mode"] <= 0o777, "snapshot_original_mode")
        files.append(FileSnapshot(path, payload, len(payload), f'{row["mode"]:04o}', blake3_bytes(payload)))
    require(len(files) <= 256 and sum(row.byte_count for row in files) <= 128 * 1024 * 1024,
            "snapshot_bounds")
    require([row.path for row in files] == sorted({row.path for row in files}), "snapshot_path_inventory")
    require(manifest["file_count"] == len(files) and manifest["bytes"] == sum(row.byte_count for row in files),
            "snapshot_inventory_totals")
    core = {"files": tuple(files), "file_count": len(files), "byte_count": sum(row.byte_count for row in files)}
    value = WorkspaceSnapshot(label=label, **core, tree_blake3=blake3_hex(core))
    JudgeWorkspaceTools._validate_snapshot(value, label)
    return value, manifest


def inventory(snapshot_value):
    return {"files": [{"path": row.path, "bytes": row.byte_count, "blake3": row.content_blake3}
                      for row in snapshot_value.files],
            "file_count": snapshot_value.file_count, "bytes": snapshot_value.byte_count}


def event(role, content, call_ids=(), *, event_id=None):
    core = {"event_id": event_id or str(uuid4()), "role": role, "content": content,
            "tool_call_ids": tuple(call_ids)}
    return TrajectoryEvent(**core, event_blake3=blake3_hex(core))


def visible_response(response, receipt_blake3):
    """Keep all text; bind binary image bytes without making them prose context."""
    value = canonical_value(response)
    for block in value.get("content", []):
        if block.get("type") == "image" and isinstance(block.get("data"), str):
            data = base64.b64decode(block.pop("data"), validate=True)
            block["retained_binary_payload"] = {"bytes": len(data), "blake3": blake3_bytes(data),
                "original_codex_tool_call_receipt_blake3": receipt_blake3,
                "projection": "binary-only-payload-not-copied-to-Judge-text-sidecar",
                "original_receipt_unchanged": True}
    return value


def convert_turn(*, request, receipt_document, host_events, before, after):
    """Verify one actual transport/host join and preserve visible event order."""
    raw_receipt = dict(receipt_document)
    raw_receipt.pop("document_blake3", None)
    receipt = codex_turn_receipt_from_document(raw_receipt)
    logical = request["logical_input"]
    require(receipt.visibility == "actor-public" and logical["judge_only_context"] is None,
            "actor_visibility_boundary")
    require(logical["role"] == receipt.role.value, "logical_actor_role")
    require(receipt.input_blake3 == blake3_hex(logical), "logical_input_commitment")
    require(receipt.offered_tool_schema_blake3 == blake3_hex(logical["offered_tools"]),
            "tool_catalog_commitment")
    require(receipt.selected_skill_catalog_blake3 == blake3_hex(logical["skills"]),
            "skill_catalog_commitment")
    metadata = logical["offered_tool_metadata"]
    require(list(receipt.offered_mcp_tool_names) == [row["fully_qualified_name"] for row in metadata],
            "offered_tool_names")
    require(all(call.fully_qualified_name in receipt.offered_mcp_tool_names for call in receipt.tool_calls),
            "unoffered_actor_tool")
    require(receipt.status == "completed", "actor_turn_not_completed")
    groups = _tool_groups(receipt, verify_event_peak=True)
    group_by_call = {call.tool_call_id: index for index, group in enumerate(groups) for call in group}
    group_ids = {index: str(uuid4()) for index in range(len(groups))}
    host_by_id = {}
    for row in host_events:
        require(row["schema"] == "eva.automedbench-public-tool-event.v1", "host_event_schema")
        require(row["event_blake3"] == blake3_bytes(canonical({k: v for k, v in row.items()
                                                              if k != "event_blake3"})), "host_event_commitment")
        require(row["event_id"] not in host_by_id, "duplicate_host_event")
        host_by_id[row["event_id"]] = row
    joined, results, used = {}, [], set()
    for call in receipt.tool_calls:
        require(call.tool_type == "mcpToolCall" and call.mcp_server == "automed_eval"
                and call.status == "completed", "unsupported_actor_action")
        output = canonical_value(call.output)
        response = output.get("result")
        require(isinstance(response, dict), "missing_mcp_response")
        structured = response.get("structuredContent")
        require(isinstance(structured, dict), "missing_host_event_binding")
        host_id = structured.get("event_id")
        require(host_id in host_by_id and host_id not in used, "host_event_join")
        host = host_by_id[host_id]
        require(host["name"] == call.mcp_tool == structured["name"]
                and host["arguments"] == canonical_value(call.arguments)
                and host["result"] == structured["result"]
                and host["is_error"] == response.get("isError", False), "host_transport_payload_differs")
        require(output.get("error") is None, "mcp_transport_error")
        # The installed app-server omits the MCP success default isError:false.
        # Restore only that exact protocol default for the original wire digest.
        original_response = dict(response)
        if original_response.get("_meta", False) is None:
            original_response.pop("_meta")  # App-server inserts this absent optional field as null.
        if "isError" not in original_response:
            original_response["isError"] = False
        require(host["response_blake3"] == blake3_bytes(canonical(original_response)), "mcp_response_commitment")
        used.add(host_id)
        core = {"result_id": host_id, "call_id": call.tool_call_id, "name": host["name"],
                "frontier": group_by_call[call.tool_call_id],
                "parallel_group_id": group_ids[group_by_call[call.tool_call_id]],
                "status": "failed" if host["is_error"] else "completed",
                "output": {"actual_mcp_response_text_and_binary_commitments": visible_response(response, call.receipt_blake3),
                           "host_event": host},
                "error_code": host["result"].get("error_code") if host["is_error"] else None,
                # These commitments are the actual retained inventory format,
                # not fabricated intermediate byte-bearing WorkspaceSnapshots.
                "workspace_before_blake3": blake3_bytes(canonical(host["workspace_before"])),
                "workspace_after_blake3": blake3_bytes(canonical(host["workspace_after"]))}
        row = {**core, "receipt_blake3": blake3_hex(core)}
        joined[call.upstream_item_id] = (call, row)
        results.append(row)
    actual = [row for row in host_events if row["event_id"] in used]
    state = inventory(before)
    for host in actual:
        require(host["workspace_before"] == state, "host_workspace_chain")
        state = host["workspace_after"]
    require(state == inventory(after), "terminal_workspace_chain")
    messages = [event("system", {"base_instructions": request["base_instructions"],
                                "developer_instructions": request["developer_instructions"]}),
                event("user", logical)]
    started, completed = set(), set()
    for source in receipt.events:
        item = canonical_value(source.payload).get("item", {})
        if source.method == "item/started" and item.get("id") in joined:
            call, _ = joined[item["id"]]
            messages.append(event("assistant", {"name": call.fully_qualified_name,
                "arguments": canonical_value(call.arguments), "codex_tool_call_receipt": call.receipt_blake3},
                (call.tool_call_id,), event_id=source.event_id))
            started.add(call.tool_call_id)
        elif source.method == "item/completed" and item.get("id") in joined:
            call, row = joined[item["id"]]
            require(call.tool_call_id in started, "tool_completion_without_start")
            messages.append(event("tool", row, (call.tool_call_id,), event_id=source.event_id))
            completed.add(call.tool_call_id)
        elif source.method == "item/completed" and item.get("type") == "agentMessage":
            require(isinstance(item.get("text"), str), "visible_assistant_text_absent")
            messages.append(event("assistant", {"text": item["text"], "phase": item.get("phase")},
                                  event_id=source.event_id))
    require(started == completed == {call.tool_call_id for call in receipt.tool_calls}, "actor_event_coverage")
    require(any(row.role == "assistant" for row in messages), "assistant_evidence_absent")
    return receipt, messages, results, actual, len(groups)


def prepare_case(*, run_root: Path, case_id: str, phase: str, stage: str,
                 checkpoint_identity: Path, output_root: Path, registry_path: Path = ROOT / "rubrics/source/domain-stage-tables.v1.json"):
    require(phase in PHASES and stage in ("S1", "S2", "S3", "S4", "S5", "E2E"), "phase_or_stage")
    require(Path(case_id).name == case_id and case_id not in (".", ".."), "case_path_component")
    require(stage != "S1" or phase == PHASES[0], "s1_requires_planning_prefix")
    manifest = read_document(run_root / "run-manifest.json")
    require(manifest["diagnostic_only"] is False and manifest["public_workspaces_only"] is True
            and manifest["private_references_in_actor_workspace"] is False, "run_is_not_public_model_evaluation")
    cases = [row for row in manifest["cases"] if row["case_id"] == case_id]
    require(len(cases) == 1, "case_identity")
    case = cases[0]
    rubric = load_and_compile_registry(registry_path).resolve("automedbench-" + case["track"], stage)
    case_root = run_root / "codex-rollouts" / case_id
    event_path = case_root / "mcp-events.jsonl"
    host_events = [json.loads(line) for line in event_path.read_bytes().splitlines() if line.strip()] if event_path.exists() else []
    binding = read_json(checkpoint_identity)
    attempt = read_document(run_root / "codex-rollouts/attempt.json")
    server = attempt["server_binding"]
    identity_digest = blake3_bytes(checkpoint_identity.read_bytes())
    require(server["identity_file_blake3"] == identity_digest and server["identity"] == binding
            and server["canary"]["checkpoint_identity_blake3"] == identity_digest
            and server["canary"]["exact_final_model_path"] == binding["exact_final_model_path"]
            and server["canary"]["model_info"]["model_path"] == binding["exact_final_model_path"]
            and server["actual_process_argv_rechecked"] is True
            and server["canary"]["status"] == "complete", "actor_checkpoint_binding")
    require(is_blake3(attempt.get("verified_skill_catalog_blake3", "")), "mcp_skill_catalog_identity")
    messages, results, sources, turns, selected_hosts = [], [], [], [], []
    first_before, previous_after, frontier_offset = None, None, 0
    for name in PHASES[:PHASES.index(phase) + 1]:
        turn_root = case_root / "turns" / name
        request = read_document(turn_root / "request.json")
        receipt_raw = read_json(turn_root / "receipt.json")
        before, before_manifest = snapshot(turn_root / "before", "before")
        after, after_manifest = snapshot(turn_root / "after", "after")
        if previous_after is not None:
            require(before.tree_blake3 == previous_after.tree_blake3, "turn_workspace_discontinuity")
        first_before = first_before or before
        receipt, visible, trace, actual, frontier_count = convert_turn(
            request=request, receipt_document=receipt_raw, host_events=host_events, before=before, after=after)
        if turns:
            require(receipt.thread_id == turns[0].thread_id and receipt.model == turns[0].model
                    and receipt.provider == turns[0].provider, "thread_or_model_discontinuity")
        for result in trace:
            result["frontier"] += frontier_offset
            result["receipt_blake3"] = blake3_hex({k: v for k, v in result.items() if k != "receipt_blake3"})
        # Result documents in tool events must match prospective global-frontier
        # offsets; preserve the original host and Codex commitments separately.
        by_call = {row["call_id"]: row for row in trace}
        visible = [event(row.role, by_call[row.tool_call_ids[0]], row.tool_call_ids, event_id=row.event_id)
                   if row.role == "tool" else row for row in visible]
        messages.extend(visible); results.extend(trace); turns.append(receipt); selected_hosts.extend(actual)
        sources.append({"phase_intent": name, "request_document_blake3": request["document_blake3"],
            "codex_receipt_blake3": receipt.receipt_blake3,
            "before_manifest_blake3": before_manifest["document_blake3"],
            "after_manifest_blake3": after_manifest["document_blake3"]})
        previous_after = after
        frontier_offset += frontier_count
    # The actor's own retained checkpoint binding is checked by the caller-facing
    # request, not inferred from an unchanged served model alias.
    require(binding.get("exact_final_model_path") or binding.get("model_path"), "checkpoint_path_absent")
    require([row["event_id"] for row in selected_hosts] == [row["event_id"] for row in host_events[:len(selected_hosts)]],
            "unaccounted_prefix_host_event")
    trace_core = {"results": results, "declared_call_ids": [row["call_id"] for row in results],
                  "joined_call_ids": [row["call_id"] for row in results], "frontier_count": frontier_offset,
                  "max_parallelism_observed": max(row.max_parallelism_observed for row in turns), "retry_count": 0}
    sample_id = str(uuid4())
    rollout = {"schema": "eva.automedbench-codex-workspace-feedback.v1", "task_id": sample_id,
        "sandbox_id": case["workspace_id"], "candidate_id": sample_id,
        "executable_episode_id": case_id, "route_id": "qwen_3_5_9b", "model_id": turns[-1].model,
        "provider": turns[-1].provider, "messages": canonical_value(messages),
        "assistant_output": turns[-1].final_response or "", "provider_receipt_blake3": turns[-1].receipt_blake3,
        "workspace_before": canonical_value(first_before), "workspace_after": canonical_value(previous_after),
        "tool_trace": {**trace_core, "trace_blake3": blake3_hex(trace_core)}, "rubric_table": rubric.to_document(),
        "provider_metadata": {"case_id": case_id, "phase_intent": phase, "evaluated_stage": stage,
            "phase_label_proves_stage_success": False, "checkpoint_identity_blake3": blake3_bytes(checkpoint_identity.read_bytes()),
            "checkpoint_identity": binding, "codex_turn_receipts": canonical_value(turns), "source_turns": sources,
            "selected_skill_ids_by_turn": [list(row.selected_skill_ids) for row in turns],
            "selected_skill_catalog_blake3_by_turn": [row.selected_skill_catalog_blake3 for row in turns],
            "mcp_verified_skill_catalog_blake3": attempt["verified_skill_catalog_blake3"],
            "actual_mcp_skill_loads": [{"host_event_id": row["event_id"], "arguments": row["arguments"],
                "result_blake3": blake3_bytes(canonical(row["result"])), "tool_is_error": row["is_error"]}
                for row in selected_hosts if row["name"] == "load_skill"],
            "skill_attribution_performed": False, "native_clinical_metrics_included": False,
            "private_references_included": False, "hidden_reasoning_included": False,
            "snapshot_mode_semantics": "original-inode-mode-retained-by-actor-snapshot",
            "binary_image_sidecar_projection": "all-text-retained;binary-pixels-committed-in-original-receipt-and-source-snapshots",
            "run_attempt_document_blake3": attempt["document_blake3"],
            "tool_frontier_semantics": "observed-Codex-transport-overlap-not-host-parallel-execution",
            "host_execution_semantics": "actual-serialized-PublicTools-lock-inventory-chain"}}
    output_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    source_path = output_root / "rollout.json"
    write_private(source_path, rollout)
    prepared = prepare_workspace_rollout(rollout, rubric=rubric, source_path=source_path,
        judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra")
    # Reopen persisted bytes through the same typed immutable evidence contract.
    reopened = prepare_workspace_rollout(read_json(source_path), rubric=rubric, source_path=source_path,
        judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra")
    require(prepared.trajectory.source_blake3 == reopened.trajectory.source_blake3, "persisted_rollout_differs")
    report = {"schema": "eva.automedbench-feedback-preflight.v1", "verification_id": str(uuid4()),
        "valid": True, "sample_id": sample_id, "case_id": case_id, "domain": rubric.domain, "stage": stage,
        "phase_intent": phase, "stage_completion_claimed": False, "retained_turn_count": len(turns),
        "actual_host_call_count": len(selected_hosts), "rubric_digest": rubric.digest,
        "source_rollout_blake3": prepared.trajectory.source_blake3,
        "source_workspace_file_count": [first_before.file_count, previous_after.file_count],
        "source_workspace_byte_count": prepared.source_workspace_byte_count,
        "snapshot_bytes_reopened": True, "checkpoint_identity_blake3": blake3_bytes(checkpoint_identity.read_bytes()),
        "provider_calls": 0, "gpu_calls": 0, "judge_calls": 0, "reward_emitted": False,
        "private_references_read": False, "native_clinical_scores_read": False}
    write_private(output_root / "preflight.json", report)
    return rollout, rubric, report


def _bound_feedback_rubric(preflight, rollout, registry_path=None):
    """New track evidence pins its registry; old receipts reopen with v1."""
    binding = preflight.get("exact_rubric_registry")
    if binding is not None:
        require(binding == rollout["provider_metadata"].get("exact_rubric_registry"), "feedback_registry_binding_differs")
        recorded = Path(binding["path"])
        require(registry_path is None or Path(registry_path).resolve() == recorded.resolve(), "feedback_registry_path_differs")
        registry_path = recorded
        payload = safe_file(recorded.parent, recorded.name).read_bytes()
        require(blake3_bytes(payload) == binding["source_file_blake3"], "feedback_registry_source_changed")
    path = Path(registry_path) if registry_path is not None else ROOT / "rubrics/source/domain-stage-tables.v1.json"
    registry = load_and_compile_registry(path)
    require(binding is None or registry.digest == binding["compiled_registry_digest"], "feedback_compiled_registry_changed")
    rubric = registry.resolve(preflight["domain"], preflight["stage"])
    require(rollout["rubric_table"] == canonical_value(rubric.to_document()), "current_exact_rubric_binding")
    return rubric


def judge_once(output_root: Path, *, registry_path=None, native_turn_timeout_seconds: int = 240):
    validate_native_judge_timeout(native_turn_timeout_seconds)
    preflight = read_json(output_root / "preflight.json")
    rollout = read_feedback_rollout(output_root / "rollout.json", preflight)
    require(preflight["valid"] is True and preflight["source_rollout_blake3"] == blake3_hex(rollout),
            "prepared_evidence_commitment")
    rubric = _bound_feedback_rubric(preflight, rollout, registry_path)
    write_private(output_root / "judge-attempt.json", {"attempt_id": str(uuid4()), "backend": "native_astra",
        "semantic_attempt_count": 1, "retry_count": 0, "automatic_fallback": False,
        "native_turn_timeout_seconds": native_turn_timeout_seconds,
        "judge_implementation_blake3": blake3_bytes((ROOT / "src/eva_agent/training/slime_agent_judge.py").read_bytes())})
    asyncio.run(grade_rollout(rollout, rubric=rubric, output_root=output_root / "judge",
                            sample_id=preflight["sample_id"], backend="native_astra",
                            native_turn_timeout_seconds=native_turn_timeout_seconds))
    return verify_feedback(output_root)


def verify_feedback(output_root: Path, *, registry_path=None):
    """Independent persisted assessment/read-byte/score reopening, no providers."""
    preflight = read_json(output_root / "preflight.json")
    rollout = read_feedback_rollout(output_root / "rollout.json", preflight)
    rubric = _bound_feedback_rubric(preflight, rollout, registry_path)
    root = output_root / "judge" / preflight["sample_id"]
    require(read_feedback_rollout(root / "rollout.json", preflight) == rollout, "judge_rollout_copy_differs")
    prepared = prepare_workspace_rollout(rollout, rubric=rubric, source_path=output_root / "rollout.json",
        judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra")
    assessment_raw = read_json(root / "assessment.json")
    require(assessment_raw["assessment_blake3"] == blake3_hex({k: v for k, v in assessment_raw.items()
                                                              if k != "assessment_blake3"}), "assessment_commitment")
    assessment = _judgment(assessment_raw)
    prepared = replace(prepared, task=replace(prepared.task, judge_task_id=assessment.judgment_id))
    snapshots = {"before": prepared.evidence.workspace_before, "after": prepared.evidence.workspace_after}
    reads = 0
    for row in assessment.agent_trace.results:
        if row.name != "workspace_read" or row.status != "completed":
            continue
        args, out = row.arguments, row.output
        snap = snapshots[args["snapshot"]]
        file = next(value for value in snap.files if value.path == args["path"])
        data = out["content"].encode() if out["encoding"] == "utf-8" else base64.b64decode(out["content"], validate=True)
        require(data == file.content[args["offset"]:args["offset"] + args["max_bytes"]]
                and out["content_blake3"] == file.content_blake3 and out["tree_blake3"] == snap.tree_blake3
                and out["returned_bytes"] == len(data) and out["total_bytes"] == file.byte_count,
                "judge_read_bytes_differ")
        reads += 1
    judgment = judgment_document(prepared, assessment, maximum_workspace_tool_frontiers=32,
                                  maximum_provider_turn_count=65)
    require(canonical_value(judgment) == read_json(root / "judgment.json"), "judgment_reopen_differs")
    scores, _ = _validated_judgment(judgment, task=prepared.task, trajectory=prepared.trajectory)
    score = rubric.score(scores, evaluation_id=prepared.task.judge_task_id)
    require(canonical_value(score.to_document()) == read_json(root / "score.json"), "exact_score_differs")
    grade = read_json(root / "grade.json")
    require(grade["reward_bps"] == score.reward_bps and grade["item_scores_bps"] == scores
            and grade["source_rollout_blake3"] == prepared.trajectory.source_blake3
            and grade["judge_backend"] == "native_astra", "grade_binding_differs")
    backend = read_json(root / "judge-backend.json")
    require(backend == grade["judge_provenance"] and backend["provenance_blake3"] == blake3_hex({
        k: v for k, v in backend.items() if k != "provenance_blake3"}), "judge_backend_commitment")
    receipt = codex_turn_receipt_from_document(read_json(root / "judge-codex-receipt.json"))
    require(receipt.status == "completed" and receipt.visibility == "judge-only"
            and receipt.model == backend["requested_model"] == assessment.judge_model_id
            and receipt.provider == backend["provider"], "judge_provider_identity")
    _validate_tool_inventory(receipt, _judge_offers(JudgeWorkspaceTools(prepared.evidence), "evamed-judge"),
                             verify_event_peak=True)
    actual_results = {row.call_id: canonical_value(row) for row in assessment.agent_trace.results}
    observed = set()
    for call in receipt.tool_calls:
        if call.fully_qualified_name not in receipt.offered_mcp_tool_names:
            continue  # Existing validator above admits only its known read-only core discovery calls.
        doc = _mcp_structured_output(call)
        core = {k: v for k, v in doc.items() if k != "bridge_receipt_blake3"}
        require(doc["bridge_receipt_blake3"] == blake3_hex(core) and doc["call_id"] not in observed
                and doc["judge_tool_result"] == actual_results.get(doc["call_id"])
                and doc["name"] == call.mcp_tool and doc["arguments"] == canonical_value(call.arguments),
                "judge_transport_trace_binding")
        observed.add(doc["call_id"])
    require(observed == set(actual_results), "judge_transport_result_coverage")
    terminal = _agent_judge_terminal_object(receipt.final_response or "")
    require(terminal == {"item_scores": canonical_value(assessment.item_scores),
        "hard_gates_passed": assessment.hard_gates_passed, "summary": assessment.summary}, "judge_terminal_binding")
    judge_attempt = read_json(output_root / "judge-attempt.json")
    timeout = judge_attempt.get("native_turn_timeout_seconds", 240)
    validate_native_judge_timeout(timeout)
    require(backend["turn_timeout_seconds"] == timeout, "judge_timeout_attempt_binding_differs")
    judge_profile = {k: backend[k] for k in ("judge_backend", "requested_model", "provider", "maximum_execution_frontiers",
        "maximum_workspace_calls", "maximum_transport_projection_turns", "turn_timeout_seconds")}
    judge_profile["implementation_blake3"] = judge_attempt["judge_implementation_blake3"]
    result = {"schema": "eva.automedbench-verified-stage-feedback.v1", "verification_id": str(uuid4()),
        "valid": True, "case_id": preflight["case_id"], "domain": rubric.domain, "stage": rubric.stage,
        "rubric_digest": rubric.digest, "status": "scored", "score": score.to_document(),
        **({"exact_rubric_registry": preflight["exact_rubric_registry"]} if "exact_rubric_registry" in preflight else {}),
        "source_rollout_blake3": prepared.trajectory.source_blake3,
        "checkpoint_identity_blake3": preflight["checkpoint_identity_blake3"],
        "round_identity": {"model_id": rollout["model_id"], "checkpoint_id": preflight["checkpoint_identity_blake3"],
            "skill_catalog_id": rollout["provider_metadata"]["mcp_verified_skill_catalog_blake3"],
            "judge_id": blake3_hex(judge_profile)},
        "judge_profile": judge_profile, "judge_codex_receipt_blake3": receipt.receipt_blake3,
        "actual_workspace_reads": reads, "judge_backend": "native_astra",
        "judge_provenance": grade["judge_provenance"], "native_clinical_metric": None,
        "skill_attribution_performed": False, "training_authorized": False}
    return result
