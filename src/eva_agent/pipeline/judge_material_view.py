"""Versioned lossless policy-visible views; originals remain immutable audit files.

This changes only generated Judge sidecars, never EvidenceBundle policy events,
tool traces, source workspaces, tools, rubrics, or actor receipt commitments.
"""
from __future__ import annotations

from collections.abc import Mapping
from .digests import blake3_bytes, blake3_hex, canonical_json_bytes, canonical_value

MATERIAL_PREFIX = ".eva-agent-judge/"
REQUIRED_PATHS = tuple(MATERIAL_PREFIX + "actor/" + name for name in (
    "assistant-output.txt", "context.json", "messages.json", "tool-trace.json", "workspace-bindings.json"))
AUDIT_PATHS = tuple(MATERIAL_PREFIX + "audit/" + name for name in (
    "messages-full.json", "tool-trace-full.json", "view-binding.json"))
HISTORICAL_VIEW = "historical-v1"
POLICY_VISIBLE_VIEW = "policy-visible-audit-v2"
_RESULT_KEYS = frozenset(("result_id", "call_id", "name", "frontier", "parallel_group_id", "status",
    "output", "error_code", "workspace_before_blake3", "workspace_after_blake3", "receipt_blake3"))
_VISIBLE = "actual_mcp_response_text_and_binary_commitments"
_SUPPLEMENTAL = "supplemental_host_verified_provided_analysis_tool"


def validate_material_view(view):
    if view not in (HISTORICAL_VIEW, POLICY_VISIBLE_VIEW):
        raise ValueError("unsupported Judge material view")
    return view


def policy_visible_payloads(evidence):
    """Project only the known, independently joined AutoMed result envelope."""
    from eva_agent.codex_pipeline.budget import PROJECTION, WORKSPACE_PROJECTION
    native = evidence.safe_provider_metadata.get("schema") in {
        "eva.codex-provider-rollout-projection.v1", PROJECTION, WORKSPACE_PROJECTION}
    if native:
        from .verify import verify_codex_judge_source
        verify_codex_judge_source(evidence)
    events = canonical_value(evidence.policy_events)
    trace = canonical_value(evidence.tool_trace)
    results = {row["call_id"]: row for row in trace["results"]}
    if len(results) != len(trace["results"]):
        raise ValueError("policy-visible view duplicate tool result")
    visible_events, joined, trace_rows = [], set(), []
    for row in events:
        if row["role"] != "tool":
            if row["role"] not in {"system", "user", "assistant"}:
                raise ValueError("policy-visible view unsupported event role")
            visible_events.append(row)  # Exact complete original event, including all text.
            continue
        value = row["content"]
        if native:
            # Native projection has no AutoMed host inventory to remove. Keep
            # every original observation unchanged; rejected actions remain
            # outside ToolTrace and retain their real Codex identity/status.
            visible_events.append(row)
            continue
        if not isinstance(value, Mapping) or set(value) != _RESULT_KEYS:
            raise ValueError("policy-visible view unsupported tool-result envelope")
        output = value["output"]
        if (not isinstance(output, Mapping) or not {_VISIBLE, "host_event"} <= set(output)
                or set(output) - {_VISIBLE, "host_event", _SUPPLEMENTAL}
                or not isinstance(output[_VISIBLE], Mapping) or not isinstance(output["host_event"], Mapping)):
            raise ValueError("policy-visible view unknown output content cannot be dropped")
        call_id = value["call_id"]
        if row["tool_call_ids"] != [call_id] or call_id in joined or results.get(call_id) != value:
            raise ValueError("policy-visible view event/trace join differs")
        host = output["host_event"]
        if not isinstance(host.get("arguments"), Mapping):
            raise ValueError("policy-visible view original arguments absent")
        joined.add(call_id)
        # Do not mislabel a derived event/result as its original committed schema.
        projected = {key: item for key, item in value.items() if key not in {"output", "receipt_blake3"}}
        projected["output"] = {_VISIBLE: output[_VISIBLE]}
        projected["source_result_receipt_blake3"] = value["receipt_blake3"]
        visible_events.append({"schema": "eva.judge-policy-tool-event-view.v2",
            "event_id": row["event_id"], "role": "tool", "tool_call_ids": row["tool_call_ids"],
            "source_event_blake3": row["event_blake3"], "content": projected,
            "original_event_audit": {"path": AUDIT_PATHS[0], "event_id": row["event_id"]}})
        trace_rows.append({key: item for key, item in projected.items() if key != "output"} | {
            "arguments": host["arguments"], "policy_event_id": row["event_id"],
            "complete_visible_response_path": REQUIRED_PATHS[2], "original_result_audit_path": AUDIT_PATHS[1]})
    if not native and joined != set(results):
        raise ValueError("policy-visible view incomplete tool-event coverage")
    # Retain source trace order even if event delivery order differed.
    indexed = {row["call_id"]: row for row in trace_rows}
    trace_view = {key: value for key, value in trace.items() if key not in {"results", "trace_blake3"}}
    trace_view.update(schema="eva.judge-policy-tool-trace-view.v2", source_trace_blake3=trace["trace_blake3"],
        results=trace["results"] if native else [indexed[row["call_id"]] for row in trace["results"]])
    # Original MCP text (including schema-error feedback) is retained in the
    # typed receipt, separately from its structured pipeline observation. Make
    # all of it required Judge material; never call it post-native truncation.
    native_responses = {}
    if native:
        receipt = canonical_value(evidence.safe_provider_metadata["codex_turn_receipt"])
        native_responses = {"native_original_mcp_responses": [
            {key: call[key] for key in ("tool_call_id", "fully_qualified_name", "arguments",
                                       "status", "output", "receipt_blake3")}
            for call in receipt["tool_calls"] if call["tool_type"] == "mcpToolCall"]}
    payloads = {
        REQUIRED_PATHS[2]: canonical_json_bytes({"schema": "eva.judge-policy-events-view.v2",
            "source_policy_events_blake3": blake3_hex(evidence.policy_events),
            "mcp_response_scope": "retained_original_MCP_response_not_verified_post_native_truncation",
            "native_context_projection_reconstructed": False, "events": visible_events,
            **native_responses}),
        REQUIRED_PATHS[3]: canonical_json_bytes(trace_view),
        AUDIT_PATHS[0]: canonical_json_bytes(evidence.policy_events),
        AUDIT_PATHS[1]: canonical_json_bytes(evidence.tool_trace),
    }
    payloads[AUDIT_PATHS[2]] = canonical_json_bytes({"schema": "eva.judge-policy-material-binding.v2",
        "material_view": POLICY_VISIBLE_VIEW, "original_policy_events_blake3": blake3_hex(evidence.policy_events),
        "original_tool_trace_blake3": evidence.tool_trace.trace_blake3,
        "all_retained_actor_public_content_preserved": True, "canonical_evidence_objects_changed": False,
        "preservation_scope": "all_retained_actor_public_events_and_original_MCP_responses",
        "post_native_projection_equals_original_response_verified": False,
        "removed_from_required_view_only": [] if native else ["host_event", _SUPPLEMENTAL],
        "audit_files_are_not_actor_source_workspace": True,
        "files": [{"path": path, "byte_count": len(body), "content_blake3": blake3_bytes(body)}
                  for path, body in sorted(payloads.items())]})
    return payloads


def material_layout(evidence):
    """Validate exact fixed sidecar layout and every derived view commitment."""
    files = {row.path: row for row in evidence.workspace_after.files if row.path.startswith(MATERIAL_PREFIX)}
    paths = tuple(sorted(files))
    if paths == REQUIRED_PATHS:
        return HISTORICAL_VIEW, REQUIRED_PATHS, ()
    if paths != tuple(sorted((*REQUIRED_PATHS, *AUDIT_PATHS))):
        raise ValueError("Judge evidence material set differs")
    for path, payload in policy_visible_payloads(evidence).items():
        if files[path].content != payload or files[path].content_blake3 != blake3_bytes(payload):
            raise ValueError("Judge policy material projection differs from original evidence")
    return POLICY_VISIBLE_VIEW, REQUIRED_PATHS, AUDIT_PATHS
