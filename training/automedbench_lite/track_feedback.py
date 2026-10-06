"""Actual mutable-track evidence to the existing exact-rubric native Judge.

Input files are an explicitly separate immutable commitment, not silently
invented snapshot bytes. Original host event/transport documents remain intact.
"""
from __future__ import annotations

import json
from pathlib import Path
import stat
from uuid import uuid4

from eva_agent.codex_runtime import CodexToolOffer, codex_turn_receipt_from_document
from eva_agent.codex_pipeline.adapter import (_tool_groups, _mcp_groups,
    codex_core_mcp_resource_operation, _validate_codex_core_resource_call)
from eva_agent.pipeline import FileSnapshot, WorkspaceSnapshot
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes, canonical_value
from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.training.slime_agent_judge import prepare_workspace_rollout, validate_native_judge_timeout
from training.benchmark_feedback.automed_codex import (
    require, event, visible_response, read_json, write_private, judge_once, verify_feedback,
    TRACK_FEEDBACK_RESOURCE_POLICY, read_feedback_rollout,
)
from .adapter import canonical, file_digest, read_document, safe_file
from .track_scoring import completed_model_evidence

ROOT = Path(__file__).resolve().parents[2]
PHASES = {"S1": "01-planning", "S2": "02-setup", "S3": "03-smoke", "S4": "04-full-subset", "S5": "05-review"}


def archived_payload(root, row):
    path = root / row["path"]
    require(not Path(row["path"]).is_absolute() and ".." not in Path(row["path"]).parts,
            "track_archived_path")
    cursor = root
    for part in Path(row["path"]).parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "track_archived_symlink")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_size == row["bytes"]
            and info.st_size <= TRACK_FEEDBACK_RESOURCE_POLICY["maximum_file_bytes"],
            "track_archived_payload_size")
    # Hardlinks here point only to retained content-addressed archival blobs,
    # never the mutable actor inode; each read rechecks actual content bytes.
    payload = path.read_bytes()
    require(blake3_bytes(payload) == row["blake3"], "track_archived_payload_commitment")
    return payload


def track_snapshot(directory, label):
    manifest = read_document(directory / "manifest.json", maximum=16 * 1024**2)
    require(manifest["schema"] == "eva.automedbench-track-snapshot.v1" and manifest["inputs_inlined"] is False,
            "track_snapshot_scope")
    rows = manifest["files"]
    require(len(rows) <= TRACK_FEEDBACK_RESOURCE_POLICY["maximum_files_per_snapshot"]
            and manifest["bytes"] <= TRACK_FEEDBACK_RESOURCE_POLICY["maximum_bytes_per_snapshot"],
            "track_full_mutable_snapshot_exceeds_actor_budget")
    files = []
    for row in rows:
        path = row["path"]
        require(path in {"task.json", "inputs-manifest.json"} or path.startswith(("notes/", "code/", "outputs/", "public-guidance/")),
                "track_nonpublic_snapshot_file")
        payload = archived_payload(directory / "files", row)
        require(type(row["mode"]) is int and 0 <= row["mode"] <= 0o777, "track_snapshot_mode")
        files.append(FileSnapshot(path, payload, len(payload), f'{row["mode"]:04o}', row["blake3"]))
    require([row.path for row in files] == sorted({row.path for row in files}), "track_snapshot_inventory")
    require(len(files) == manifest["file_count"] and sum(row.byte_count for row in files) == manifest["bytes"],
            "track_snapshot_totals")
    binding = next(row for row in files if row.path == "inputs-manifest.json")
    require(binding.content_blake3 == manifest["immutable_inputs_manifest_blake3"], "track_input_manifest_binding")
    core = {"files": tuple(files), "file_count": len(files), "byte_count": sum(row.byte_count for row in files)}
    snapshot = WorkspaceSnapshot(label=label, **core, tree_blake3=blake3_hex(core))
    JudgeWorkspaceTools._validate_snapshot(snapshot, label)
    return snapshot, manifest


def validate_inventory(value, audit, binding):
    require(value["inputs_inlined"] is False and value["immutable_inputs_manifest_blake3"] == binding,
            "track_host_input_scope_changed")
    rows = value["files"]
    require([row["path"] for row in rows] == sorted({row["path"] for row in rows})
            and len(rows) == value["file_count"] and sum(row["bytes"] for row in rows) == value["bytes"],
            "track_host_inventory_totals")
    for row in rows:
        payload = (audit / "workspace-blobs" / row["blake3"]).read_bytes()
        require(len(payload) == row["bytes"] and blake3_bytes(payload) == row["blake3"], "track_host_retained_bytes")


def verify_feedback_recovery_reference(run_root, track, stage, reference):
    """Explicitly reopen current-time evidence; never repair the original run."""
    require(isinstance(reference, dict) and set(reference) == {"ownership_path", "recovery_root"},
            "track_recovery_reference_shape")
    require(stage in PHASES, "track_recovery_stage")
    from . import policy_recovery
    ownership = Path(reference["ownership_path"]).resolve(strict=True)
    root = Path(reference["recovery_root"]).resolve(strict=True)
    proof = policy_recovery.verify_recovery(run_root=Path(run_root), track=track, phase=PHASES[stage],
        ownership_path=ownership, recovery_root=root)
    target = Path(run_root) / "track-rollouts" / track / "turns" / PHASES[stage]
    return {"valid": True, "reference": {"ownership_path": str(ownership), "recovery_root": str(root)},
        "proof": proof, "helper_source_blake3": file_digest(Path(policy_recovery.__file__)),
        "original_archival_failure": read_document(target / "failure.json"),
        "original_policy_terminal": read_document(target / "policy-budget-terminal.json"),
        "workspace_after_semantics": "verified_current_time_recovery_not_original_terminal_snapshot",
        "original_status_unchanged": True, "original_after_manifest_blake3": None,
        "recovery_current_after_manifest_blake3": proof["recovery_after_manifest_document_blake3"],
        "stage_completion_claimed": False}


def recovery_source_fields(recovery):
    """Same explicit source boundary for new Judge and baseline prefix checks."""
    return {"after_manifest_blake3": None,
        "recovery_current_after_manifest_blake3": recovery["recovery_current_after_manifest_blake3"],
        "recovery_document_blake3": recovery["proof"]["document_blake3"],
        "workspace_after_semantics": recovery["workspace_after_semantics"]}


def convert_track_turn(request, raw_receipt, hosts, audit, input_binding, *, terminal_proof=None,
                       recovery_proof=None):
    receipt = codex_turn_receipt_from_document(raw_receipt)
    logical = request["logical_input"]
    budget_terminal = (terminal_proof is not None and terminal_proof.get("valid") is True
        and terminal_proof.get("receipt_blake3") == receipt.receipt_blake3
        and terminal_proof.get("actual_terminal_status") == receipt.status)
    recovered_terminal = (recovery_proof is not None and recovery_proof.get("valid") is True
        and recovery_proof["proof"]["source_commitments"]["receipt_blake3"] == receipt.receipt_blake3
        and recovery_proof["proof"]["actual_terminal_status"] == receipt.status == "interrupted")
    require((receipt.status in {"completed", "failed"} or (receipt.status == "interrupted" and (budget_terminal or recovered_terminal))) and receipt.visibility == "actor-public"
            and logical["judge_only_context"] is None and logical["role"] == receipt.role.value,
            "track_actor_turn_unfinished_or_private")
    require(receipt.input_blake3 == blake3_hex(logical)
            and receipt.offered_tool_schema_blake3 == blake3_hex(logical["offered_tools"])
            and receipt.selected_skill_catalog_blake3 == blake3_hex(logical["skills"]), "track_actor_catalog_commitment")
    offers = tuple(CodexToolOffer(fully_qualified_name="automed_eval/" + row["name"], description=row["description"],
        input_schema=row["inputSchema"], parallel_safe=False,
        read_only=row["name"] in {"automed_read_file", "automed_view_input", "search_skills", "load_skill"},
        allowed_stages=("E2E",)) for row in request["public_tool_catalog"])
    require(receipt.offered_mcp_tool_names == tuple(offer.fully_qualified_name for offer in offers)
            and receipt.offered_tool_schema_blake3 == blake3_hex(tuple(offer.canonical_catalog_entry() for offer in offers)),
            "track_exact_offered_inventory")
    for call in receipt.tool_calls:
        require(call.tool_type == "mcpToolCall", "track_unavailable_native_action")
        if call.fully_qualified_name in receipt.offered_mcp_tool_names:
            # Unlike the teacher-success gate, this generic evaluator must keep
            # actual failed domain calls. The host join below proves the failure.
            require(call.status in {"completed", "failed"} and isinstance(canonical_value(call.arguments), dict),
                    "track_domain_call_unfinished")
        else:
            operation = codex_core_mcp_resource_operation(server=call.mcp_server, tool=call.mcp_tool,
                offered_mcp_tool_names=set(receipt.offered_mcp_tool_names))
            require(operation is not None, "track_unoffered_native_control")
            _validate_codex_core_resource_call(receipt, call, operation=operation)
    groups = _mcp_groups(_tool_groups(receipt, verify_event_peak=True), set(receipt.offered_mcp_tool_names))
    by_call = {call.tool_call_id: index for index, group in enumerate(groups) for call in group}
    group_ids = {index: str(uuid4()) for index in range(len(groups))}
    host_index = {}
    for row in hosts:
        require(row["schema"] == "eva.automedbench-track-tool-event.v1"
                and row["event_blake3"] == blake3_bytes(canonical({k: v for k, v in row.items() if k != "event_blake3"})),
                "track_host_event_commitment")
        require(row["event_id"] not in host_index, "track_duplicate_host_event")
        host_index[row["event_id"]] = row
    joined, results, used = {}, [], set()
    controls = []
    for call in receipt.tool_calls:
        if call.fully_qualified_name not in receipt.offered_mcp_tool_names:
            # Existing strict validator above verifies these native controls.
            # They are NOT projected into the domain-tool trace.
            controls.append(canonical_value(call))
            continue
        require(call.mcp_server == "automed_eval" and call.status in {"completed", "failed"}, "track_mcp_identity")
        output = canonical_value(call.output)
        response = output.get("result")
        require(isinstance(response, dict) and output.get("error") is None, "track_mcp_response")
        structured = response.get("structuredContent", {})
        host_id = structured.get("event_id")
        require(host_id in host_index and host_id not in used, "track_actual_host_join")
        host = host_index[host_id]
        require(host["name"] == call.mcp_tool == structured.get("name")
                and host["arguments"] == canonical_value(call.arguments)
                and host["result"] == structured.get("result")
                and host["is_error"] == response.get("isError", call.status == "failed")
                and (call.status == "failed") == host["is_error"], "track_host_payload_differs")
        original = dict(response)
        if original.get("_meta", False) is None: original.pop("_meta")
        # Codex0.153.4 lifts isError into the typed call.status and omits it from
        # result for both true and false. Restore only that independently joined
        # status value, then require the exact original host wire commitment.
        original.setdefault("isError", call.status == "failed")
        require(host["response_blake3"] == blake3_bytes(canonical(original)), "track_response_wire_commitment")
        validate_inventory(host["workspace_before"], audit, input_binding)
        validate_inventory(host["workspace_after"], audit, input_binding)
        used.add(host_id)
        core = {"result_id": host_id, "call_id": call.tool_call_id, "name": host["name"],
            "frontier": by_call[call.tool_call_id], "parallel_group_id": group_ids[by_call[call.tool_call_id]],
            "status": "failed" if host["is_error"] else "completed",
            "output": {"actual_mcp_response_text_and_binary_commitments": visible_response(response, call.receipt_blake3),
                       "host_event": host}, "error_code": host["result"].get("error_code") if host["is_error"] else None,
            "workspace_before_blake3": blake3_bytes(canonical(host["workspace_before"])),
            "workspace_after_blake3": blake3_bytes(canonical(host["workspace_after"]))}
        result = {**core, "receipt_blake3": blake3_hex(core)}
        results.append(result)
        joined[call.upstream_item_id] = (call, result)
    messages = [event("system", {"base_instructions": request["base_instructions"], "developer_instructions": request["developer_instructions"]}),
                event("user", logical)]
    started, completed = set(), set()
    control_index = {row["upstream_item_id"]: row for row in controls}
    for source in receipt.events:
        item = canonical_value(source.payload).get("item", {})
        identifier = item.get("id")
        if source.method == "item/started" and identifier in joined:
            call, result = joined[identifier]
            messages.append(event("assistant", {"name": call.fully_qualified_name, "arguments": canonical_value(call.arguments),
                "codex_tool_call_receipt": call.receipt_blake3}, (call.tool_call_id,), event_id=source.event_id))
            started.add(call.tool_call_id)
        elif source.method == "item/completed" and identifier in joined:
            call, result = joined[identifier]
            require(call.tool_call_id in started, "track_completion_without_start")
            messages.append(event("tool", result, (call.tool_call_id,), event_id=source.event_id))
            completed.add(call.tool_call_id)
        elif source.method == "item/completed" and identifier in control_index:
            messages.append(event("assistant", {"actual_native_control_not_domain_tool": control_index[identifier]}, event_id=source.event_id))
        elif source.method == "item/completed" and item.get("type") == "agentMessage":
            messages.append(event("assistant", {"text": item["text"], "phase": item.get("phase")}, event_id=source.event_id))
    require(started == completed == set(by_call), "track_actor_event_coverage")
    return receipt, messages, results, [row for row in hosts if row["event_id"] in used], len(groups)


def prepare_track_feedback(*, run_root, checkpoint_identity, output_root, stage, track="classification", registry_path=None,
                           allow_policy_budget_terminal=False, material_view="historical-v1",
                           allow_context_budget_terminal=False, context_terminal_source_binding=None,
                           policy_recovery_reference=None):
    require(stage in PHASES, "track_feedback_stage")
    if policy_recovery_reference is not None:
        from eva_agent.training import slime_agent_judge as selected_judge
        require(callable(getattr(selected_judge, "_current_after_recovery_annotation", None)),
                "track_recovery_requires_annotating_judge_core")
    run_root, output_root, checkpoint_identity = Path(run_root), Path(output_root), Path(checkpoint_identity)
    manifest = read_document(run_root / "track-run-manifest.json")
    selected = [row for row in manifest["tracks"] if row["track"] == track]
    require(len(selected) == 1, "track_feedback_membership")
    spec = selected[0]
    audit = run_root / "track-rollouts" / track
    attempt = read_document(run_root / "track-rollouts/attempt.json")
    server = attempt["server_binding"]
    identity = read_json(checkpoint_identity)
    digest = blake3_bytes(checkpoint_identity.read_bytes())
    require(server["identity_file_blake3"] == digest and server["identity"] == identity
            and server["canary"]["checkpoint_identity_blake3"] == digest
            and server["canary"]["exact_final_model_path"] == identity["exact_final_model_path"]
            and server["canary"]["model_info"]["model_path"] == identity["exact_final_model_path"]
            and server["actual_process_argv_rechecked"] is True, "track_checkpoint_binding")
    registry_path = Path(registry_path) if registry_path is not None else ROOT / "rubrics/source/domain-stage-tables.v1.json"
    registry = load_and_compile_registry(registry_path)
    rubric = registry.resolve("automedbench-" + track, stage)
    registry_binding = {"path": str(registry_path.resolve()), "source_file_blake3": file_digest(safe_file(registry_path.parent, registry_path.name)),
                        "compiled_registry_digest": registry.digest}
    host_path = audit / "mcp-events.jsonl"
    host_bytes = host_path.read_bytes() if host_path.exists() else b""
    if host_bytes and not host_bytes.endswith(b"\n"):
        # Later phases may append. Complete-prefix UUID/result joins below still apply.
        host_bytes = host_bytes.rpartition(b"\n")[0]
    hosts = [json.loads(line) for line in host_bytes.splitlines() if line.strip()]
    messages, results, turns, sources, selected_hosts = [], [], [], [], []
    sdk_bindings, terminal_proofs, context_proofs, recovery_proofs = [], [], [], []
    first_before, last_after, frontier = None, None, 0
    for phase_stage, phase in PHASES.items():
        target = audit / "turns" / phase
        request = read_document(target / "request.json")
        require(request["verified_skill_catalog_blake3"] == attempt["verified_skill_catalog_blake3"], "track_skill_catalog_changed")
        before, before_doc = track_snapshot(target / "before", "before")
        recovery = None
        if policy_recovery_reference is not None and phase_stage == stage:
            recovery = verify_feedback_recovery_reference(run_root, track, stage, policy_recovery_reference)
        after_root = Path(recovery["reference"]["recovery_root"]) / "after" if recovery else target / "after"
        after, after_doc = track_snapshot(after_root, "after")
        if recovery:
            require(recovery["recovery_current_after_manifest_blake3"] == after_doc["document_blake3"],
                    "track_recovery_snapshot_changed")
        require(before_doc["immutable_inputs_manifest_blake3"] == after_doc["immutable_inputs_manifest_blake3"] == spec["input_manifest_file_blake3"],
                "track_input_manifest_changed")
        raw_receipt = read_json(target / "receipt.json")
        terminal_proof = None
        context_proof = None
        if allow_context_budget_terminal and raw_receipt.get("status") == "failed":
            from .context_terminal import verify_context_terminal
            context_proof = verify_context_terminal(target, audit, raw_receipt,
                source_binding=context_terminal_source_binding)
            require(phase_stage == stage, "track_context_terminal_cannot_imply_later_phase")
            require(context_proof["before_manifest_blake3"] == before_doc["document_blake3"]
                and context_proof["after_manifest_blake3"] == after_doc["document_blake3"],
                "track_context_terminal_snapshot_changed")
        if allow_policy_budget_terminal and recovery is None and (target / "policy-budget-terminal.json").exists():
            from .policy_terminal import verify_policy_terminal
            terminal_proof = verify_policy_terminal(target, audit, raw_receipt)
            require(phase_stage == stage, "track_policy_terminal_cannot_imply_later_phase")
            require(terminal_proof["after_snapshot_document_blake3"] == after_doc["document_blake3"],
                    "track_policy_terminal_snapshot_changed")
        if allow_policy_budget_terminal and recovery is None and (audit / "track-budget-terminal.json").exists():
            budget_doc = read_document(audit / "track-budget-terminal.json")
            if budget_doc.get("phase_intent") == target.name:
                from .policy_terminal import verify_between_turn_terminal
                terminal_proof = verify_between_turn_terminal(target, audit, raw_receipt)
                require(phase_stage == stage, "track_policy_terminal_cannot_imply_later_phase")
                require(terminal_proof["after_snapshot_document_blake3"] == after_doc["document_blake3"],
                        "track_policy_terminal_snapshot_changed")
        receipt, visible, trace, actual, count = convert_track_turn(request, raw_receipt, hosts,
            audit, spec["input_manifest_file_blake3"], terminal_proof=terminal_proof, recovery_proof=recovery)
        terminal_proofs.append(terminal_proof)
        context_proofs.append(context_proof)
        recovery_proofs.append(recovery)
        if attempt.get("extra_memory_skill_binding") is not None:
            from .sdk_memory_feedback import selected_sdk_memory_context
            sdk_text, sdk_binding = selected_sdk_memory_context(run_root, attempt, request, receipt,
                source_binding=context_terminal_source_binding)
            visible.insert(2, event("user", sdk_text))  # Actual runtime SDK input, not a path-only claim.
            sdk_bindings.append({"stage": phase_stage, "request_document_blake3": request["document_blake3"], **sdk_binding})
        if turns: require(receipt.thread_id == turns[0].thread_id and receipt.model == turns[0].model and receipt.provider == turns[0].provider,
                          "track_thread_or_model_changed")
        for row in trace:
            row["frontier"] += frontier
            row["receipt_blake3"] = blake3_hex({k: v for k, v in row.items() if k != "receipt_blake3"})
        by_call = {row["call_id"]: row for row in trace}
        visible = [event(row.role, by_call[row.tool_call_ids[0]], row.tool_call_ids, event_id=row.event_id)
                   if row.role == "tool" else row for row in visible]
        messages.extend(visible); results.extend(trace); turns.append(receipt); selected_hosts.extend(actual)
        sources.append({"phase_intent": phase, "stage_intent": phase_stage, "request_document_blake3": request["document_blake3"],
            "codex_receipt_blake3": receipt.receipt_blake3, "before_manifest_blake3": before_doc["document_blake3"],
            **(recovery_source_fields(recovery) if recovery else {"after_manifest_blake3": after_doc["document_blake3"]})})
        first_before = first_before or before
        last_after = after
        frontier += count
        if phase_stage == stage: break
    require([row["event_id"] for row in selected_hosts] == [row["event_id"] for row in hosts[:len(selected_hosts)]],
            "track_unaccounted_prefix_event")
    model_jobs = completed_model_evidence(audit=audit, workspace=run_root / spec["workspace_relative"], track=track,
        input_binding=spec["input_manifest_file_blake3"], hosts=selected_hosts,
        visible_files={row.path: {"blake3": row.content_blake3, "bytes": row.byte_count} for row in last_after.files})
    # Explicit supplemental host evidence becomes readable inside the existing
    # immutable tool-trace sidecar. Original wire response, host event, actor
    # message and source workspace remain untouched; helpers are NOT actor code.
    by_host = {row["host_event_id"]: row for row in model_jobs}
    retained_source_digests = set()
    for result in results:
        job = by_host.get(result["output"]["host_event"]["event_id"])
        if job is None:
            continue
        job_sources = []
        for source in job["sources"]:
            if source["blake3"] in retained_source_digests:
                source = {key: value for key, value in source.items() if key != "text"}
                source = {**source, "identical_source_text_in_earlier_provided_tool_evidence": True}
            elif "text" in source:
                retained_source_digests.add(source["blake3"])
            job_sources.append(source)
        result["output"]["supplemental_host_verified_provided_analysis_tool"] = {**job, "sources": job_sources,
            "not_original_tool_response_content": True, "not_actor_authored_code": True,
            "completion_does_not_prove_rubric_stage_success": True}
        result["receipt_blake3"] = blake3_hex({key: value for key, value in result.items() if key != "receipt_blake3"})
    by_call = {row["call_id"]: row for row in results}
    messages = [event(row.role, by_call[row.tool_call_ids[0]], row.tool_call_ids, event_id=row.event_id)
                if row.role == "tool" else row for row in messages]
    trace_core = {"results": results, "declared_call_ids": [row["call_id"] for row in results],
        "joined_call_ids": [row["call_id"] for row in results], "frontier_count": frontier,
        "max_parallelism_observed": max(row.max_parallelism_observed for row in turns), "retry_count": 0}
    sample_id = str(uuid4())
    rollout = {"schema": "eva.automedbench-track-workspace-feedback.v1", "task_id": sample_id,
        "sandbox_id": Path(spec["workspace_relative"]).name, "candidate_id": sample_id, "executable_episode_id": track,
        "route_id": "qwen_3_5_9b", "model_id": turns[-1].model, "provider": turns[-1].provider,
        "messages": canonical_value(messages), "assistant_output": turns[-1].final_response or "",
        "provider_receipt_blake3": turns[-1].receipt_blake3, "workspace_before": canonical_value(first_before),
        "workspace_after": canonical_value(last_after), "tool_trace": {**trace_core, "trace_blake3": blake3_hex(trace_core)},
        "rubric_table": rubric.to_document(), "provider_metadata": {"track": track, "phase_intent": PHASES[stage],
            "evaluated_stage": stage, "phase_label_proves_stage_success": False, "checkpoint_identity": identity,
            "exact_rubric_registry": registry_binding,
            "snapshot_resource_policy": TRACK_FEEDBACK_RESOURCE_POLICY,
            "checkpoint_identity_blake3": digest, "codex_turn_receipts": canonical_value(turns), "source_turns": sources,
            "mcp_verified_skill_catalog_blake3": attempt["verified_skill_catalog_blake3"],
            "selected_skill_ids_by_turn": [list(row.selected_skill_ids) for row in turns],
            **({"policy_recovery_feedback_policy": "explicit_verified_current_workspace_recovery.v1",
                "verified_policy_recovery_proofs": recovery_proofs,
                "recovered_workspace_is_not_original_terminal_snapshot": True}
               if policy_recovery_reference is not None else {}),
            **({"policy_terminal_feedback_policy": "explicit_verified_owned_budget_terminal.v1",
                "verified_policy_terminal_proofs": terminal_proofs,
                "budget_stop_is_not_stage_completion": True}
               if allow_policy_budget_terminal else {}),
            **({"context_terminal_feedback_policy": "explicit_verified_local_context_budget_terminal.v1",
                "verified_context_terminal_proofs": context_proofs,
                "context_budget_stop_is_not_stage_completion": True}
               if allow_context_budget_terminal else {}),
            **({"selected_sdk_skill_content_bindings": sdk_bindings,
                "extra_sdk_skill_body_context": "Exact actor runtime _skill_text input, independently reopened from its bound immutable mount; not a canonical MCP load"}
               if sdk_bindings else {}),
            "actual_mcp_skill_loads": [{"host_event_id": row["event_id"], "arguments": row["arguments"],
                "result_blake3": blake3_bytes(canonical(row["result"])), "tool_is_error": row["is_error"]}
                for row in selected_hosts if row["name"] == "load_skill"],
            "source_workspace_scope": "Actual mutable notes/code/output and public requirements; immutable inputs separate",
            "immutable_inputs_manifest_blake3": spec["input_manifest_file_blake3"],
            "all_immutable_input_bytes_present_in_judge_snapshot": False,
            "binary_image_observations": "Original receipt preserves actual MCP pixels; Judge text retains exact binary commitments",
            "asynchronous_state_semantics": "Each actual inventory independently byte-verified; background fixed jobs can change workspace between calls, no fabricated equality",
            "provided_analysis_tool_job_ids": [row["job_id"] for row in model_jobs],
            "provided_analysis_tool_authorship": "Host-retained implementation/source and actual outputs are supplemental tool-trace evidence, never Qwen-authored code or prior policy observations",
            "tool_frontier_semantics": "Observed Codex overlap projection, not claimed host parallel execution",
            "native_clinical_metrics_included": False, "private_references_included": False,
            "hidden_reasoning_included": False, "skill_attribution_performed": False,
            "run_attempt_document_blake3": attempt["document_blake3"]}}
    output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    path = output_root / "rollout.json"
    write_private(path, rollout)
    prepared = prepare_workspace_rollout(read_feedback_rollout(path, {
        "schema": "eva.automedbench-track-feedback-preflight.v1", "exact_rubric_registry": registry_binding,
        "snapshot_resource_policy": TRACK_FEEDBACK_RESOURCE_POLICY}), rubric=rubric, source_path=path,
        judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra",
        **({"material_view": material_view} if material_view != "historical-v1" else {}))
    report = {"schema": "eva.automedbench-track-feedback-preflight.v1", "verification_id": str(uuid4()), "valid": True,
        "sample_id": sample_id, "case_id": track, "domain": rubric.domain, "stage": stage, "phase_intent": PHASES[stage],
        "stage_completion_claimed": False, "retained_turn_count": len(turns), "actual_host_call_count": len(selected_hosts),
        "rubric_digest": rubric.digest, "source_rollout_blake3": prepared.trajectory.source_blake3,
        "exact_rubric_registry": registry_binding,
        "snapshot_resource_policy": TRACK_FEEDBACK_RESOURCE_POLICY,
        "checkpoint_identity_blake3": digest, "source_workspace_byte_count": prepared.source_workspace_byte_count,
        "provider_calls": 0, "judge_calls": 0, "private_references_read": False,
        "actual_turn_statuses": [row.status for row in turns],
        **({"judge_material_view": material_view} if material_view != "historical-v1" else {}),
        "judge_eligible": all(context_proof is not None or (
            (row.status == "completed" or (row.status == "interrupted" and (proof is not None or recovery is not None)))
            and not any(source.method == "error" for source in row.events))
            for row, proof, context_proof, recovery in zip(turns, terminal_proofs, context_proofs, recovery_proofs)),
        **({"policy_recovery_feedback_policy": "explicit_verified_current_workspace_recovery.v1",
            "verified_policy_recovery_proofs": recovery_proofs,
            "recovered_workspace_is_not_original_terminal_snapshot": True}
           if policy_recovery_reference is not None else {}),
        **({"policy_terminal_feedback_policy": "explicit_verified_owned_budget_terminal.v1",
            "verified_policy_terminal_proofs": terminal_proofs}
           if allow_policy_budget_terminal else {}),
        **({"context_terminal_feedback_policy": "explicit_verified_local_context_budget_terminal.v1",
            "verified_context_terminal_proofs": context_proofs}
           if allow_context_budget_terminal else {}),
        "failed_transport_or_runtime_is_not_rubric_zero": True}
    write_private(output_root / "preflight.json", report)
    return rollout, rubric, report


def judge_track_once(output_root, *, native_turn_timeout_seconds: int = 240):
    validate_native_judge_timeout(native_turn_timeout_seconds)
    report = read_json(Path(output_root) / "preflight.json")
    require(report.get("judge_eligible") is True, "track_infrastructure_or_incomplete_outcome_not_gradeable")
    return judge_once(Path(output_root), native_turn_timeout_seconds=native_turn_timeout_seconds)


def evaluate_track_feedback(run_root, checkpoint_identity, output_root, stages=("S1", "S2", "S3"), *, track="classification", registry_path=None,
                            native_turn_timeout_seconds: int = 240, allow_policy_budget_terminal=False,
                            material_view="historical-v1", allow_context_budget_terminal=False,
                            context_terminal_source_binding=None, policy_recovery_references=None,
                            postround_judge_verdict_replacements=0):
    """Judge each complete retained prefix once; missing later phases are unknown.

Present corrupt commitments/schema still raise. No missing or failed stage is
translated to a rubric score; at least one independently verified grade is needed.
"""
    validate_native_judge_timeout(native_turn_timeout_seconds)
    from .judge_attempt_isolation import replacement_limit
    judge_replacements = replacement_limit(postround_judge_verdict_replacements)
    require(stages and len(set(stages)) == len(stages) and set(stages) <= set(PHASES), "track_feedback_stage_selection")
    require(track in {"classification", "detection", "segmentation", "synthesis", "vqa", "report", "enhancement"},
            "track_feedback_track_selection")
    recoveries = {} if policy_recovery_references is None else policy_recovery_references
    require(isinstance(recoveries, dict) and set(recoveries) <= set(stages) and len(recoveries) <= 1,
            "track_recovery_stage_references")
    registry_path = Path(registry_path) if registry_path is not None else ROOT / "rubrics/source/domain-stage-tables.v1.json"
    registry = load_and_compile_registry(registry_path)
    available = {(rubric.domain, rubric.stage) for rubric in registry.rubrics}
    run_root, output_root = Path(run_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    roots = []
    prepared = []
    coverage = []
    # Validate all present candidate prefixes before spending any Judge calls.
    for stage in stages:
        if ("automedbench-" + track, stage) not in available:
            coverage.append({"stage": stage, "status": "unknown", "reason": "exact_compiled_track_stage_rubric_absent",
                "domain": "automedbench-" + track, "rubric_score": None})
            continue
        prefix = list(PHASES)[:list(PHASES).index(stage) + 1]
        if any(name in recoveries for name in prefix[:-1]):
            coverage.append({"stage": stage, "status": "unknown", "reason": "recovery_does_not_establish_later_phase",
                "rubric_score": None})
            continue
        missing = [name for name in prefix if not (run_root / "track-rollouts" / track / "turns" / PHASES[name] / "receipt.json").exists()]
        if missing:
            coverage.append({"stage": stage, "status": "unknown", "reason": "required_actual_phase_receipt_absent",
                "missing_prefix_stages": missing, "rubric_score": None})
            continue
        target = output_root / stage
        prepared_root = target / "prepared" if judge_replacements else target
        _, _, report = prepare_track_feedback(run_root=run_root, checkpoint_identity=checkpoint_identity, output_root=prepared_root,
            stage=stage, track=track, registry_path=registry_path,
            **({"policy_recovery_reference": recoveries[stage]} if stage in recoveries else {}),
            **({"allow_policy_budget_terminal": True} if allow_policy_budget_terminal else {}),
            **({"material_view": material_view} if material_view != "historical-v1" else {}),
            **({"allow_context_budget_terminal": True} if allow_context_budget_terminal else {}),
            **({"context_terminal_source_binding": context_terminal_source_binding}
               if context_terminal_source_binding is not None else {}))
        if not report["judge_eligible"]:
            coverage.append({"stage": stage, "status": "unknown", "reason": "actual_turn_infrastructure_or_unfinished_outcome",
                "actual_turn_statuses": report["actual_turn_statuses"], "evidence_root": str(prepared_root), "rubric_score": None})
            continue
        prepared.append((stage, target, prepared_root))
        coverage.append({"stage": stage, "status": "prepared_eligible", "evidence_root": str(prepared_root), "rubric_score": None})
    write_private(output_root / "coverage-preflight.json", {"schema": "eva.automedbench-stage-feedback-coverage.v1",
        "requested_stages": list(stages), "stages": coverage, "missing_is_not_zero": True, "judge_calls": 0})
    for stage, target, prepared_root in prepared:
        if judge_replacements:
            from .judge_attempt_isolation import run_stage_judge_attempts
            accepted, verification, _ = run_stage_judge_attempts(target, prepared_root,
                replacements=judge_replacements, judge=judge_track_once, verify=verify_feedback,
                native_turn_timeout_seconds=native_turn_timeout_seconds)
        else:
            timeout_options = ({"native_turn_timeout_seconds": native_turn_timeout_seconds}
                               if native_turn_timeout_seconds != 240 else {})
            feedback = judge_track_once(target, **timeout_options)
            write_private(target / "feedback.json", feedback)
            verification = verify_feedback(target)
            write_private(target / "verification.json", verification)
            accepted = target
        roots.append(accepted)
        entry = next(row for row in coverage if row["stage"] == stage)
        entry.update(status="independently_verified", rubric_score=verification["score"],
                     source_rollout_blake3=verification["source_rollout_blake3"], evidence_root=str(accepted))
    write_private(output_root / "coverage.json", {"schema": "eva.automedbench-stage-feedback-coverage.v1",
        "requested_stages": list(stages), "stages": coverage, "missing_is_not_zero": True,
        "verified_feedback_roots": [str(root) for root in roots], "verified_stage_count": len(roots),
        **({"postround_judge_verdict_replacements": judge_replacements,
            "automatic_additional_judge_attempts": True, "same_judge_attempt_retried": False,
            "additional_judge_attempts_are_not_actor_rerolls": True}
           if judge_replacements else {})})
    require(bool(roots), "track_no_verified_grade_missing_is_not_zero")
    return tuple(roots)
