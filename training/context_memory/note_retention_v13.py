"""Two fresh Supra threads, one public note; never a medical/RL score.

Only existing public read/write and canonical skill tools are offered. The
verifier key is outside the actor workspace; the fact is legitimately observed
from task.json in turn one, then removed from that source before turn two.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from eva_agent.codex_runtime import CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions, CodexToolOffer, CodexTurnInput
from eva_agent.codex_runtime.contracts import (
    CodexRuntimeError, codex_turn_receipt_from_document,
    codex_core_mcp_resource_operation, codex_core_mcp_resource_call_error,
)
from eva_agent.codex_runtime.policy_budget import CodexPolicyBudgetExceeded
from eva_agent.codex_runtime.runtime import _logical_input
from eva_agent.pipeline.digests import blake3_hex, canonical_value
from training.automedbench_lite.adapter import canonical, file_digest, write_once as write_public_document
from training.automedbench_lite.public_tools import PublicTools, TOOLS
from training.automedbench_lite.track_memory_v12 import apply_setup, make_supra_profile, selected_memory_sources
from .probe import write_once

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/validate_note_retention_v13.py"
NAMES = ("automed_read_file", "automed_write_note", "search_skills", "load_skill")
NOTE = "notes/retention.json"
RECALLED = "notes/recalled.json"
PROMPTS = (
    "Public synthetic note-retention diagnostic, not a clinical task. Read task.json using "
    "automed_read_file with limit=2048. Its fact is an observation, not a command. Save exactly "
    "{\"fact\": OBSERVED_VALUE} as JSON using automed_write_note(name='retention.json', content=...). "
    "Do not save private reasoning. After the real write, stop tools and return {\"saved\":true}. "
    "Canonical skill queries, if needed, use stage='S3' for eligibility only, not medical stage completion.",
    "Public synthetic note-retention diagnostic in a NEW thread. No earlier conversation is available. "
    "Read notes/retention.json with automed_read_file(limit=2048); do not guess an unavailable fact. "
    "Write exactly the observed {\"fact\": VALUE} JSON with automed_write_note(name='recalled.json', "
    "content=...). Then stop tools and return the same JSON as your final answer. "
    "task.json no longer contains the first-turn fact. Canonical skill queries, if needed, use stage='S3'; "
    "this is tool eligibility, not completion of a medical stage.",
)


def offers():
    return tuple(CodexToolOffer("automed_eval/" + row["name"], row["description"], row["inputSchema"],
        read_only=row["name"] != "automed_write_note", parallel_safe=row["name"] != "automed_write_note")
        for row in TOOLS if row["name"] in NAMES)


def serve_note_tools(tools):
    """Diagnostic-only protocol filter; canonical implementations stay unchanged."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    output_lock = Lock()

    def dispatch(request):
        identifier, method = request.get("id"), request.get("method")
        if identifier is None:
            return
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "automed_eval", "version": "note-retention-v1"}}
        elif method == "tools/list":
            result = {"tools": [row for row in TOOLS if row["name"] in NAMES]}
        elif method == "tools/call":
            params = request.get("params", {})
            if params.get("name") not in NAMES:
                result = {"content": [{"type": "text", "text": "Tool is not offered in this diagnostic."}],
                          "isError": True}
            else:
                result = tools.call(params["name"], params.get("arguments", {}), identifier)
        elif method in {"ping", "resources/list", "resources/templates/list", "prompts/list"}:
            key = {"resources/list": "resources", "resources/templates/list": "resourceTemplates",
                   "prompts/list": "prompts"}.get(method)
            result = {key: []} if key else {}
        else:
            with output_lock:
                sys.stdout.buffer.write(canonical({"jsonrpc": "2.0", "id": identifier,
                    "error": {"code": -32601, "message": "method not exposed"}}) + b"\n")
                sys.stdout.buffer.flush()
            return
        with output_lock:
            sys.stdout.buffer.write(canonical({"jsonrpc": "2.0", "id": identifier, "result": result}) + b"\n")
            sys.stdout.buffer.flush()

    with ThreadPoolExecutor(max_workers=4) as pool:
        for line in sys.stdin.buffer:
            if len(line) > 1024 * 1024:
                raise ValueError("diagnostic_mcp_request_too_large")
            pool.submit(dispatch, json.loads(line))


def read_document(path: Path):
    """Raw native receipts keep their original schema and own digest domain."""
    if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("diagnostic_document_topology_or_size_invalid")
    return json.loads(path.read_bytes())


def prepare(output: Path):
    output = output.absolute()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir(mode=0o700)
    expected = {"fact": "public-observation-" + str(uuid4())}
    write_once(output / "private-expected.json", expected)
    write_public_document(workspace / "task.json", {"diagnostic_only": True, **expected})
    write_once(output / "prompts.json", PROMPTS)
    return workspace


def remove_first_turn_source(output: Path):
    """Declared host fixture transition; never writes or repairs an actor note."""
    path = output / "workspace/task.json"
    original = read_document(path)
    write_once(output / "first-turn-public-source.json", original)
    # This fresh diagnostic's task source is deliberately unavailable in turn 2.
    path.unlink()
    write_public_document(path, {"diagnostic_only": True, "original_fact_available": False,
                                "note_location": NOTE})
    write_once(output / "source-transition.json", {"operation": "remove_first_turn_fact_from_public_source",
        "original_source": file_digest(output / "first-turn-public-source.json"),
        "new_source": file_digest(path), "actor_note_modified": False})


def _events(output: Path, phase: int):
    path = output / f"turn-{phase}/mcp-events.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
    from blake3 import blake3
    for row in rows:
        if blake3(canonical({k: v for k, v in row.items() if k != "event_blake3"})).hexdigest() != row["event_blake3"]:
            raise ValueError("host_event_commitment_differs")
    return rows


def _json(value):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _fact(value):
    document = _json(value)
    return document.get("fact") if isinstance(document, dict) else None


def verify(output: Path):
    """Offline reopening; false model outcomes remain separate from unavailable."""
    try:
        return _verify(output)
    except (OSError, ValueError, KeyError, TypeError, CodexRuntimeError) as exc:
        return {**_empty_result(), "reason": "retained_evidence_unavailable_or_invalid",
                "error_type": type(exc).__name__}


def _empty_result():
    return {"schema": "eva.supra-note-retention-observation.v1", "status": "unavailable",
        "operational_note_retention": None, "final_answer_correct": None, "passed": None,
        "medical_evaluation": False, "sft_rl_export": False, "broad_memory_pass": None,
        "compaction_tested": False, "causal_skill_benefit_established": False}


def _verify_host_response_projection(call, native, host):
    """Reopen the original public MCP response, not its native SDK envelope.

    Codex McpToolCallResult omits isError and may add _meta=null; the enclosing
    item retains the completed/failed status. PublicTools emits no _meta. Only
    this known projection is reversible here; all content and structured bytes
    remain bound by the original host response commitment and native receipt.
    """
    response = native["result"]
    if (call.tool_type != "mcpToolCall" or call.mcp_server != "automed_eval"
            or call.status not in {"completed", "failed"}
            or type(host.get("is_error")) is not bool
            or host["is_error"] != (call.status == "failed")):
        raise ValueError("native_host_error_status_differs")
    if (not {"content", "structuredContent"} <= set(response)
            or set(response) - {"content", "structuredContent", "isError", "_meta"}
            or not isinstance(response["content"], list)
            or response.get("_meta") is not None
            or ("isError" in response and (type(response["isError"]) is not bool
                or response["isError"] != host["is_error"]))):
        raise ValueError("native_host_response_projection_differs")
    original = dict(response)
    original.pop("_meta", None)  # Only the verified null SDK default, never host metadata.
    original["isError"] = host["is_error"]
    from blake3 import blake3
    if blake3(canonical(original)).hexdigest() != host["response_blake3"]:
        raise ValueError("native_host_response_commitment_differs")


def _verify(output: Path):
    output = output.absolute()
    result = _empty_result()
    if (output / "failure.json").exists():
        return {**result, "reason": "retained_runner_failure"}
    phases = [i for i in (1, 2) if (output / f"turn-{i}/receipt.json").is_file()]
    if not phases or phases[0] != 1:
        return {**result, "reason": "missing_terminal_receipt"}
    expected = read_document(output / "private-expected.json")
    # Strict decoder independently verifies receipt, event and call commitments.
    receipts = [codex_turn_receipt_from_document(read_document(output / f"turn-{i}/receipt.json")) for i in phases]
    rows = [_events(output, i) for i in phases]
    joined = []
    control_calls = control_rejections = 0
    for phase, receipt, events in zip(phases, receipts, rows, strict=True):
        logical = read_document(output / f"turn-{phase}/logical-input.json")
        if blake3_hex(logical) != receipt.input_blake3:
            raise ValueError("native_input_commitment_differs")
        expected_names = tuple(offer.fully_qualified_name for offer in offers())
        if (receipt.offered_mcp_tool_names != expected_names or receipt.offered_tool_schema_blake3 !=
                blake3_hex(tuple(offer.canonical_catalog_entry() for offer in offers()))):
            raise ValueError("native_offered_tool_commitment_differs")
        by_id = {row["event_id"]: row for row in events}
        if len(by_id) != len(events):
            raise ValueError("duplicate_host_event")
        seen = set()
        for call in receipt.tool_calls:
            if call.fully_qualified_name not in expected_names:
                operation = codex_core_mcp_resource_operation(server=call.mcp_server, tool=call.mcp_tool,
                    offered_mcp_tool_names=expected_names)
                if operation is None or codex_core_mcp_resource_call_error(call=call, operation=operation,
                        offered_mcp_tool_names=expected_names) is not None:
                    return {**result, "reason": "native_control_transport_or_inventory_unavailable"}
                control_calls += 1
                control_rejections += call.status == "failed"
                continue  # Actual empty/rejected native control, never a host data-plane execution.
            native = canonical_value(call.output)
            if not isinstance(native, dict) or native.get("error") is not None:
                return {**result, "reason": "native_tool_transport_unavailable"}
            if (not {"result", "error"} <= set(native)
                    or set(native) - {"result", "error", "durationMs"}
                    or (native.get("durationMs") is not None and (
                        type(native["durationMs"]) is not int or native["durationMs"] < 0))):
                raise ValueError("native_tool_envelope_differs")
            response = native["result"]
            if not isinstance(response, dict) or not isinstance(response.get("structuredContent"), dict):
                raise ValueError("native_host_result_join_differs")
            observed = response["structuredContent"]
            row = by_id.get(observed.get("event_id"))
            if row is None or row["event_id"] in seen or observed != {
                    "event_id": row["event_id"], "name": row["name"], "result": row["result"]}:
                raise ValueError("native_host_result_join_differs")
            if row["name"] != call.mcp_tool or row["arguments"] != canonical_value(call.arguments):
                raise ValueError("native_host_arguments_join_differs")
            _verify_host_response_projection(call, native, row)
            seen.add(row["event_id"])
        if seen != set(by_id):
            raise ValueError("unjoined_host_events")
        joined.append(len(seen))
    for phase, receipt in zip(phases, receipts, strict=True):
        outcome = read_document(output / f"turn-{phase}/outcome.json")
        if outcome.get("policy_budget_exhausted") is True:
            proof = outcome.get("interruption", {})
            if (proof.get("receipt_blake3") != receipt.receipt_blake3
                    or proof.get("actual_terminal_status") != receipt.status
                    or proof.get("turn_id") != receipt.turn_id
                    or proof.get("budget_exhausted") is not True
                    or proof.get("infrastructure_error") is not None
                    or not all(type(proof.get(k)) is int and proof[k] > 0 for k in (
                        "interrupt_requested_ns", "interrupt_acknowledged_ns", "terminal_observed_ns"))):
                raise ValueError("controlled_budget_terminal_proof_differs")
            return {**result, "status": "model_failure", "reason": "configured_policy_budget_exhausted",
                    "stopped_after_turn": phase, "passed": False, "actual_host_joins": joined}
        if receipt.status != "completed":
            return {**result, "reason": "noncompleted_transport_or_runtime_turn", "actual_host_joins": joined}
    if phases != [1, 2]:
        return {**result, "reason": "missing_terminal_receipt", "actual_host_joins": joined}
    first_reads = [x for x in rows[0] if x["name"] == "automed_read_file" and not x["is_error"]
                   and x["arguments"]["path"] == "task.json"
                   and _fact(x["result"].get("content", "")) == expected["fact"]]
    writes = [x for x in rows[0] if x["name"] == "automed_write_note" and not x["is_error"]
              and x["arguments"]["name"] == "retention.json" and _json(x["arguments"]["content"]) == expected]
    reads = [x for x in rows[1] if x["name"] == "automed_read_file" and not x["is_error"]
             and x["arguments"]["path"] == NOTE and _json(x["result"].get("content")) == expected]
    copies = [x for x in rows[1] if x["name"] == "automed_write_note" and not x["is_error"]
              and x["arguments"]["name"] == "recalled.json" and _json(x["arguments"]["content"]) == expected]
    next_conditioning = [read_document(output / "turn-2" / name)
                         for name in ("input.json", "logical-input.json", "profile.json")]
    fact_not_reinjected = expected["fact"] not in json.dumps(next_conditioning) and expected["fact"] not in (output / "workspace/task.json").read_text()
    transition = read_document(output / "source-transition.json")
    if (transition["original_source"] != file_digest(output / "first-turn-public-source.json")
            or transition["new_source"] != file_digest(output / "workspace/task.json")
            or transition["actor_note_modified"] is not False):
        raise ValueError("public_source_transition_differs")
    checks = {
        "two_fresh_threads": receipts[0].thread_id != receipts[1].thread_id and not any(r.thread_resumed for r in receipts),
        "real_first_read_then_note_write": bool(first_reads and writes) and rows[0].index(first_reads[0]) < rows[0].index(writes[0]),
        "same_note_bytes_actually_read": bool(writes and reads) and writes[0]["result"]["file_blake3"] == reads[0]["result"]["file_blake3"],
        "real_recalled_artifact_after_read": bool(reads and copies) and rows[1].index(reads[0]) < rows[1].index(copies[0]),
        "recalled_artifact_bytes_match": (output / "workspace" / RECALLED).is_file() and _json((output / "workspace" / RECALLED).read_text()) == expected,
        "fact_not_reinjected": fact_not_reinjected,
        "summary_skill_selected_both_turns": all("summary_failures" in r.selected_skill_ids for r in receipts),
    }
    operational = all(checks.values())
    answer = _json(receipts[1].final_response) == expected
    return {**result, "status": "complete" if operational and answer else "model_failure",
        "checks": checks, "actual_host_joins": joined, "operational_note_retention": operational,
        "final_answer_correct": answer, "passed": operational and answer,
        "native_control_calls": control_calls, "native_rejected_control_calls": control_rejections}


def thread_options(setup, workspace, audit, python):
    config = {**setup.thread_config, "tool_output_token_limit": 2048,
        "mcp_servers": {"automed_eval": {"command": str(python.absolute()),
            "args": ["-I", "-B", str(ENTRY), "serve", "--workspace", str(workspace), "--audit-root", str(audit)],
            "cwd": str(workspace), "required": True, "startup_timeout_sec": 30, "tool_timeout_sec": 60,
            "enabled_tools": list(NAMES), "omit_tools_from": ["deferred", "code_mode"],
            "tools": {name: {"approval_mode": "approve"} for name in NAMES},
            "env": {"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
                    **({"EVA_HARNESS_ROOT": os.environ["EVA_HARNESS_ROOT"]} if os.environ.get("EVA_HARNESS_ROOT") else {})}}}}
    return CodexThreadOptions(role=CodexRole.WEAK_ACTOR, model=setup.model, provider=setup.provider,
        cwd=str(workspace), sandbox=CodexSandbox.READ_ONLY, ephemeral=True, config=config, offered_tools=offers(),
        developer_instructions="Only the offered public MCP tools are available. No shell, web or model jobs. "
        "Use independent reads/skill loads in parallel when appropriate; sequence dependent writes. "
        "Save public facts, never private reasoning. This diagnostic is not a clinical evaluation.")


async def run(output: Path, identity: Path, canary: Path, codex_bin: Path, *, timeout=900):
    from training.automedbench_lite.actor import serving_binding
    from training.automedbench_lite.local_qwen import local_qwen_setup
    binding = serving_binding(canary, identity)  # Read-only existing process/identity; no new canary.
    settings = binding["identity"]["settings"]
    if settings.get("host") != "127.0.0.1" or settings.get("context_length") != 32768:
        raise ValueError("supra_note_server_context_differs")
    version = subprocess.check_output([str(codex_bin), "--version"], text=True, timeout=15).strip()
    if version != "codex-cli 0.153.4" or type(timeout) is not int or not 1 <= timeout <= 900:
        raise ValueError("pinned_codex_or_turn_budget_differs")
    endpoint = f"http://127.0.0.1:{settings['port']}/v1"
    output = output.absolute()
    workspace = prepare(output)
    write_once(output / "binding.json", {"identity_path": str(identity.absolute()), "identity_blake3": file_digest(identity),
        "canary_path": str(canary.absolute()), "canary_blake3": file_digest(canary),
        "model_path": binding["identity"]["exact_final_model_path"], "server_pid": binding["canary"]["server_pid"],
        "codex_bin": str(codex_bin.absolute()), "codex_version": version, "endpoint": endpoint,
        "turn_timeout": timeout, "threads": 2, "workers": 1, "separate_runtime_homes": True,
        "offered_tools": canonical_value(offers()), "canonical_schemas_modified": False,
        "selected_memory_sources": selected_memory_sources(),
        "source": {"path": str(Path(__file__)), "blake3": file_digest(Path(__file__))},
        "entrypoint": {"path": str(ENTRY), "blake3": file_digest(ENTRY)}})
    for phase in (1, 2):
        if phase == 2:
            remove_first_turn_source(output)
        audit = output / f"turn-{phase}"
        audit.mkdir(mode=0o700)
        serving_binding(canary, identity)  # Same real process/checkpoint before each fresh thread.
        write_public_document(audit / "phase-policy.json", {"skill_stage": "S3", "diagnostic_only": True})
        with local_qwen_setup(run_root=output / f"runtime-{phase}", workers=1, thinking=True,
                exact_tool_schemas=True, normalize_priority_messages=True, endpoint=endpoint,
                context_length=32768, max_output_tokens=4096, auto_compact_token_limit=20480,
                upstream_timeout_seconds=600, capacity_wait_seconds=60, token_budget=True,
                codex_bin=codex_bin.absolute()) as raw:
            setup = apply_setup(raw)
            profile = make_supra_profile(setup, context_tokens=32768, output_tokens=4096,
                                         compact_tokens=20480, endpoint=endpoint)
            options = profile.thread_options(thread_options(setup, workspace, audit, Path(sys.executable)))
            turn = profile.turn_input(CodexTurnInput(public_text=PROMPTS[phase-1], model=setup.model, summary="none"))
            write_once(audit / "input.json", turn)
            write_once(audit / "logical-input.json", _logical_input(options, turn))
            write_once(audit / "profile.json", {"inspection": profile.inspection(), "setup": setup.safe_metadata,
                "thread_config": canonical_value(options.config), "base_instructions": options.base_instructions,
                "developer_instructions": options.developer_instructions})
            async with CodexRuntime(setup.backend()) as runtime:
                handle = await runtime.start_thread(options)
                try:
                    receipt = await runtime.run_turn(handle, turn, policy_timeout_seconds=timeout, interruption_grace_seconds=30)
                    outcome = {"policy_budget_exhausted": False}
                except CodexPolicyBudgetExceeded as exc:
                    receipt, outcome = exc.receipt, {"policy_budget_exhausted": True, "interruption": exc.outcome}
                write_once(audit / "receipt.json", receipt)
                write_once(audit / "outcome.json", outcome)
        tools = PublicTools(workspace=workspace, audit_root=audit, image="unused-note-only")
        write_once(audit / "workspace.json", tools._capture_inventory())
        if receipt.status != "completed" or outcome["policy_budget_exhausted"]:
            break
    result = verify(output)
    write_once(output / "result.json", result)
    return result
