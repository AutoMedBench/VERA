"""Prospective policy-stop capture; never manufacture a successful turn/grade."""
import asyncio
import json
import os
from pathlib import Path
import time

from eva_agent.codex_runtime import (
    codex_core_mcp_resource_operation,
    verify_codex_turn_receipt,
)
from eva_agent.codex_runtime.policy_budget import CodexPolicyBudgetExceeded
from eva_agent.pipeline.digests import canonical_json_bytes, canonical_value

from .adapter import EvaluationError, blake3, canonical, write_once
from .job_wait import await_model_jobs


async def await_model_publishers(audit, workspace, *, timeout=30, interval=.1):
    """A child exit record precedes the supervisor's final public sidecar write."""
    jobs = sorted((audit / "model-jobs").glob("*/process.json"))
    deadline = time.monotonic() + timeout
    while jobs:
        pending = []
        for path in jobs:
            process = json.loads(path.read_bytes())
            try:
                # Match exact process birth identity; never signal any process.
                tail = Path(f'/proc/{process["pid"]}/stat').read_text().rsplit(")", 1)[1].split()
                live = tail[19] == process["start_ticks"] and tail[0] != "Z"
            except FileNotFoundError:
                live = False
            if live:
                pending.append(path)
                continue
            job_id = path.parent.name
            authoritative = audit / "model-jobs/authoritative" / job_id / "process-exit.json"
            public = workspace / "outputs/agents_outputs/prescribed-model-jobs" / job_id / "process-exit.json"
            if not public.is_file() or public.read_bytes() != authoritative.read_bytes():
                raise EvaluationError("policy_terminal_model_exit_publication_incomplete")
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise EvaluationError("policy_terminal_model_publisher_still_running")
        jobs = pending
        await asyncio.sleep(interval)


def require_joined_host_results(receipt, audit):
    """A cancelled MCP result is NOT evidence its synchronous host worker ended."""
    path = audit / "mcp-events.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
    hosts = {row["event_id"]: row for row in rows}
    if len(hosts) != len(rows):
        raise EvaluationError("policy_terminal_duplicate_host_event")
    used = set()
    controls = []
    for call in receipt.tool_calls:
        output = canonical_value(call.output)
        result = output.get("result") if isinstance(output, dict) else None
        control = codex_core_mcp_resource_operation(
            server=call.mcp_server,
            tool=call.mcp_tool,
            offered_mcp_tool_names=receipt.offered_mcp_tool_names,
        )
        if control is not None:
            # These are Codex control-plane helpers, not automed_eval host
            # workers. The runtime receipt verifier already checks their
            # allowlist; retain their terminal lifecycle separately.
            if (call.tool_type != "mcpToolCall"
                    or call.status not in {"completed", "failed"}
                    or tuple(call.lifecycle) != ("item/started", "item/completed")):
                raise EvaluationError("policy_terminal_control_call_not_terminal")
            controls.append(control)
            continue
        structured = result.get("structuredContent") if isinstance(result, dict) else None
        if not isinstance(structured, dict):
            raise EvaluationError("policy_terminal_host_worker_quiescence_unproved")
        host_id = structured.get("event_id")
        if (call.tool_type != "mcpToolCall" or call.mcp_server != "automed_eval"
                or call.status not in {"completed", "failed"}
                or host_id not in hosts or host_id in used or output.get("error") is not None
                or tuple(call.lifecycle) != ("item/started", "item/completed")):
            raise EvaluationError("policy_terminal_host_worker_quiescence_unproved")
        host = hosts[host_id]
        wire = dict(result)
        if wire.get("_meta", False) is None:
            wire.pop("_meta")
        wire.setdefault("isError", call.status == "failed")
        if (host["schema"] != "eva.automedbench-track-tool-event.v1"
                or host["event_blake3"] != blake3(canonical({k: v for k, v in host.items() if k != "event_blake3"})).hexdigest()
                or host["name"] != call.mcp_tool or structured.get("name") != call.mcp_tool
                or host["arguments"] != canonical_value(call.arguments)
                or host["result"] != structured.get("result")
                or host["is_error"] != (call.status == "failed")
                or host["response_blake3"] != blake3(canonical(wire)).hexdigest()):
            raise EvaluationError("policy_terminal_host_result_binding_invalid")
        used.add(host_id)
    return {"joined_host_event_ids": sorted(used), "joined_host_call_count": len(used),
            **({"native_control_operations": controls,
                "native_control_call_count": len(controls)} if controls else {})}


async def capture_turn(runtime, handle, value, *, timeout, target, audit, inventory, snapshot,
                       track_deadline=None):
    budget = None
    try:
        receipt = await runtime.run_turn(handle, value, policy_timeout_seconds=timeout)
    except CodexPolicyBudgetExceeded as exc:
        receipt, budget = exc.receipt, exc.outcome
    verify_codex_turn_receipt(receipt)
    descriptor = os.open(target / "receipt.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(receipt))
    if budget is not None:
        terminal = {"schema": "eva.automedbench-policy-budget-terminal.v1", "policy_budget": budget,
            "receipt_blake3": receipt.receipt_blake3, "actual_terminal_status": receipt.status,
            "workspace_quiescence_verified": False, "infrastructure_error": None, "reward": None,
            "phase_intent": target.name, "later_stages_evaluated": False}
        if track_deadline is not None:
            terminal["track_budget"] = track_deadline.observation()
        try:
            if receipt.status not in {"interrupted", "completed"} or any(event.method == "error" for event in receipt.events):
                raise EvaluationError("policy_terminal_has_independent_provider_failure")
            terminal.update(require_joined_host_results(receipt, audit))
            # Exact OS exits, not job launch/receipt success, close the remaining
            # known writers. Never resubmit jobs or cancel another track's work.
            if track_deadline is None:
                await await_model_jobs(audit, "policy-terminal-" + target.name)
                await await_model_publishers(audit, getattr(inventory, "workspace", None))
            else:
                from .track_budget import cleanup_owned_jobs
                cleanup = await cleanup_owned_jobs(audit, inventory.workspace)
                terminal["deadline_cleanup_document_blake3"] = cleanup["document_blake3"]
                # At expiry this is an immediate real-exit check, not another
                # two-hour inference allowance. Live writers remain unavailable.
                await await_model_jobs(audit, "policy-terminal-" + target.name,
                    timeout=track_deadline.remaining(), track_deadline=track_deadline)
                await await_model_publishers(audit, getattr(inventory, "workspace", None),
                    timeout=track_deadline.remaining())
            terminal["host_quiescence_observed_ns"] = time.time_ns()
            after = snapshot(inventory, target / "after")
            terminal.update(workspace_quiescence_verified=True,
                            after_snapshot_document_blake3=after["document_blake3"])
        except Exception as exc:
            terminal["infrastructure_error"] = str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__
            write_once(target / "policy-budget-terminal.json", terminal)
            raise
        if track_deadline is not None:
            terminal["track_budget"] = track_deadline.observation()
        write_once(target / "policy-budget-terminal.json", terminal)
        return receipt, terminal
    snapshot(inventory, target / "after")
    return receipt, None
