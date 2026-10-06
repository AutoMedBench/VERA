"""CPU fixtures: real public file operations, synthetic native receipts, no model."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from eva_agent.codex_runtime import CodexTurnInput
from eva_agent.codex_runtime.contracts import CodexEvent, CodexToolCall, CodexTurnReceipt
from eva_agent.pipeline.digests import blake3_hex, canonical_value
from training.context_memory import note_retention_v13 as probe


def receipt(calls, logical, *, status="completed", final=None):
    """Commit a labelled synthetic terminal using the unchanged native contract."""
    thread, turn = str(uuid4()), str(uuid4())
    event = dict(event_id=str(uuid4()), sequence=0, method="turn/completed", thread_id=thread,
                 turn_id=turn, payload={"synthetic_fixture": True}, content_redacted=False)
    event = CodexEvent(**event, event_blake3=blake3_hex(event))
    core = dict(schema="eva.codex-turn-receipt.v1", receipt_id=str(uuid4()),
        runtime_thread_id=str(uuid4()), runtime_turn_id=str(uuid4()), thread_id=thread, turn_id=turn,
        role=probe.CodexRole.WEAK_ACTOR, model="synthetic-test", provider="fixture", sandbox=probe.CodexSandbox.READ_ONLY,
        thread_resumed=False, visibility="actor-public", status=status, final_response=final,
        events=(event,), tool_calls=tuple(calls), selected_skill_ids=("summary_failures",),
        selected_skill_catalog_blake3=blake3_hex([]),
        offered_mcp_tool_names=tuple("automed_eval/" + name for name in probe.NAMES),
        offered_tool_schema_blake3=blake3_hex(tuple(offer.canonical_catalog_entry() for offer in probe.offers())),
        max_parallelism_observed=1 if calls else 0,
        parallel_tool_calls_supported=True, usage={}, input_blake3=blake3_hex(logical),
        config_keys=(), config_values_recorded=False, input_payload_recorded=False,
        sdk_version="synthetic", server_version="synthetic")
    return CodexTurnReceipt(**core, receipt_blake3=blake3_hex(core))


def record_call(tools, name, arguments, *, projected=False):
    response = tools.call(name, arguments, request_id=str(uuid4()))
    status = "failed" if response["isError"] else "completed"
    output = {"result": response, "error": None}
    if projected:
        # Use the installed official SDK type, not a fabricated response schema.
        # The fixture sets the native status explicitly; it does not run Codex.
        from openai_codex.generated.v2_all import McpToolCallThreadItem
        item = McpToolCallThreadItem.model_validate({"id": str(uuid4()), "type": "mcpToolCall",
            "server": "automed_eval", "tool": name, "arguments": arguments,
            "status": status, "result": response, "error": None, "durationMs": 1})
        native = item.model_dump(mode="json", by_alias=True)
        output = {key: native[key] for key in ("result", "error", "durationMs")}
        assert set(output["result"]) == {"_meta", "content", "structuredContent"}
        assert output["result"]["_meta"] is None
    core = dict(tool_call_id=str(uuid4()), upstream_item_id=str(uuid4()), tool_type="mcpToolCall",
        name=name, mcp_server="automed_eval", mcp_tool=name, fully_qualified_name="automed_eval/" + name,
        status=status, arguments=arguments, output=output,
        lifecycle=("item/started", "item/completed"), first_event_sequence=0)
    return CodexToolCall(**core, receipt_blake3=blake3_hex(core))


def fixture(output, *, malformed=False, omit_second_read=False, empty_final=False, first_budget=False,
            partial_first_read=False, projected=False, recovered_tool_error=False):
    workspace = probe.prepare(output)
    expected = probe.read_document(output / "private-expected.json")
    encoded = json.dumps(expected) if not malformed else "not-json"
    for phase in (1, 2):
        if phase == 2:
            probe.remove_first_turn_source(output)
        audit = output / f"turn-{phase}"
        tools = probe.PublicTools(workspace=workspace, audit_root=audit, image="not-used")
        logical = {"public_text": probe.PROMPTS[phase - 1], "public_context": None}
        probe.write_once(audit / "logical-input.json", logical)
        probe.write_once(audit / "input.json", logical)
        probe.write_once(audit / "profile.json", {"synthetic_fixture": True})
        calls = []
        if phase == 1 and recovered_tool_error:
            calls.append(record_call(tools, "automed_read_file", {
                "path": "notes/missing.json", "limit": 2048}, projected=projected))
        if phase == 1 or not omit_second_read:
            calls.append(record_call(tools, "automed_read_file", {
                "path": "task.json" if phase == 1 else probe.NOTE,
                "limit": 1 if phase == 1 and partial_first_read else 2048}, projected=projected))
        calls.append(record_call(tools, "automed_write_note", {
            "name": "retention.json" if phase == 1 else "recalled.json", "content": encoded},
            projected=projected))
        actual = receipt(calls, logical, status="interrupted" if first_budget else "completed",
                         final="" if empty_final else json.dumps(expected))
        probe.write_once(audit / "receipt.json", actual)
        outcome = {"policy_budget_exhausted": first_budget}
        if first_budget:
            outcome["interruption"] = {"receipt_blake3": actual.receipt_blake3,
                "actual_terminal_status": actual.status, "turn_id": actual.turn_id,
                "budget_exhausted": True, "infrastructure_error": None,
                "interrupt_requested_ns": 1, "interrupt_acknowledged_ns": 2, "terminal_observed_ns": 3}
        probe.write_once(audit / "outcome.json", outcome)
        if first_budget:
            break
    return expected


def test_exact_schema_private_key_and_public_source_transition(tmp_path):
    workspace = probe.prepare(tmp_path / "run")
    expected = probe.read_document(workspace.parent / "private-expected.json")
    assert (workspace.parent / "private-expected.json").stat().st_mode & 0o777 == 0o600
    assert not (workspace / "private-expected.json").exists()
    assert expected["fact"] not in json.dumps(probe.PROMPTS)
    definitions = {row["name"]: row for row in probe.TOOLS}
    for offer in probe.offers():
        source = definitions[offer.fully_qualified_name.split("/")[1]]
        assert canonical_value(offer.input_schema) == source["inputSchema"]
        assert offer.description == source["description"]
    tools = probe.PublicTools(workspace=workspace, audit_root=workspace.parent / "audit", image="not-used")
    private = tools.call("automed_read_file", {"path": "../private-expected.json"})
    assert private["isError"] is True
    probe.remove_first_turn_source(workspace.parent)
    assert expected["fact"] not in (workspace / "task.json").read_text()
    assert not (workspace / probe.NOTE).exists()  # Host never pre-fills the answer note.


def test_real_note_host_operations_with_strict_synthetic_receipts(tmp_path):
    output = tmp_path / "run"
    fixture(output)
    result = probe.verify(output)
    assert result["passed"] is True and result["actual_host_joins"] == [2, 2]
    assert result["operational_note_retention"] is True and result["final_answer_correct"] is True
    assert result["broad_memory_pass"] is None and result["compaction_tested"] is False


@pytest.mark.parametrize("recovered_tool_error", [False, True])
def test_official_sdk_projection_reopens_original_host_commitments(tmp_path, recovered_tool_error):
    output = tmp_path / "run"
    fixture(output, projected=True, recovered_tool_error=recovered_tool_error)
    retained = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    result = probe.verify(output)
    assert result["passed"] is True
    assert result["actual_host_joins"] == [3 if recovered_tool_error else 2, 2]
    assert all(path.read_bytes() == original for path, original in retained.items())


def test_projected_response_does_not_waive_final_json_contract(tmp_path):
    output = tmp_path / "run"
    fixture(output, projected=True, recovered_tool_error=True, empty_final=True)
    result = probe.verify(output)
    assert result["status"] == "model_failure" and result["passed"] is False
    assert result["operational_note_retention"] is True
    assert result["final_answer_correct"] is False


@pytest.mark.parametrize("tamper", [
    "metadata", "unknown_response_key", "content", "structured_content", "missing_content",
    "missing_result", "missing_error", "error_flag", "null_error_flag", "status", "transport",
    "unknown_envelope_key",
])
def test_rehashed_native_projection_still_requires_exact_host_response(tmp_path, tamper):
    output = tmp_path / "run"
    fixture(output, projected=True, recovered_tool_error=True)
    path = output / "turn-1/receipt.json"
    original = probe.codex_turn_receipt_from_document(probe.read_document(path))
    core = canonical_value(original.tool_calls[0].core())
    native, response = core["output"], core["output"]["result"]
    if tamper == "metadata":
        response["_meta"] = {}  # Even empty host metadata was never emitted by PublicTools.
    elif tamper == "unknown_response_key":
        response["unobserved_metadata"] = None
    elif tamper == "content":
        response["content"][0]["text"] = "changed"
    elif tamper == "structured_content":
        response["structuredContent"]["result"] = {"changed": True}
    elif tamper == "missing_content":
        response.pop("content")
    elif tamper == "missing_result":
        native["result"] = None
    elif tamper == "missing_error":
        native.pop("error")
    elif tamper == "error_flag":
        response["isError"] = False  # Actual host failure cannot become success.
    elif tamper == "null_error_flag":
        response["isError"] = None
    elif tamper == "status":
        core["status"] = "completed"
    elif tamper == "transport":
        native["error"] = {"message": "connection reset"}
    else:
        native["unobserved_metadata"] = None
    changed_call = CodexToolCall(**core, receipt_blake3=blake3_hex(core))
    changed = {**original.core(), "tool_calls": (changed_call, *original.tool_calls[1:])}
    path.write_text(json.dumps(canonical_value(CodexTurnReceipt(**changed, receipt_blake3=blake3_hex(changed)))))
    result = probe.verify(output)
    assert result["status"] == "unavailable" and result["passed"] is None


@pytest.mark.parametrize("options,operational", [
    ({"malformed": True}, False), ({"omit_second_read": True}, False), ({"empty_final": True}, True),
    ({"partial_first_read": True}, False),
])
def test_model_failures_not_unavailable_or_broad_pass(tmp_path, options, operational):
    output = tmp_path / "run"
    fixture(output, **options)
    result = probe.verify(output)
    assert result["status"] == "model_failure" and result["passed"] is False
    assert result["operational_note_retention"] is operational


def test_first_turn_captured_budget_failure_not_missing_second_receipt(tmp_path):
    output = tmp_path / "run"
    fixture(output, first_budget=True)
    result = probe.verify(output)
    assert result["status"] == "model_failure" and result["passed"] is False
    assert result["stopped_after_turn"] == 1 and result["actual_host_joins"] == [2]
    assert not (output / "turn-2").exists()


@pytest.mark.parametrize("tamper", ["native", "host", "input", "transport"])
def test_missing_transport_and_tampered_evidence_are_unavailable(tmp_path, tamper):
    output = tmp_path / "run"
    fixture(output)
    if tamper == "native":
        path = output / "turn-2/receipt.json"
        row = probe.read_document(path)
        row["final_response"] = "changed"
        path.write_text(json.dumps(row))
    elif tamper == "host":
        path = output / "turn-2/mcp-events.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[0]["arguments"]["path"] = "notes/different.json"
        path.write_text("\n".join(json.dumps(row) for row in rows))
    elif tamper == "input":
        (output / "turn-2/logical-input.json").write_text('{}')
    else:
        probe.write_once(output / "failure.json", {"reason": "transport_error"})
    result = probe.verify(output)
    assert result["status"] == "unavailable" and result["passed"] is None


def test_actual_supra_composes_once_without_provider_or_permission_expansion(tmp_path):
    from eva_agent.codex_runtime.research_memory import MEMORY_INSTRUCTIONS
    from eva_agent.codex_runtime.supra import WORKFLOW_INSTRUCTIONS
    setup = SimpleNamespace(model="Qwen/Qwen3.5-9B", provider="eva_local_qwen",
                            thread_config={"features.shell_tool": False, "model_context_window": 32768,
                                           "model_auto_compact_token_limit": 20480})
    profile = probe.make_supra_profile(setup, context_tokens=32768, output_tokens=4096,
                                      compact_tokens=20480, endpoint="http://127.0.0.1:30911/v1")
    options = profile.thread_options(probe.thread_options(setup, tmp_path, tmp_path / "audit", Path(sys.executable)))
    value = profile.turn_input(CodexTurnInput(public_text=probe.PROMPTS[1]))
    assert options.developer_instructions.count(MEMORY_INSTRUCTIONS) == 1
    assert options.developer_instructions.count(WORKFLOW_INSTRUCTIONS) == 1
    assert options.config["features.shell_tool"] is False
    assert options.config["tool_output_token_limit"] == 2048
    assert options.config["model_context_window"] == 32768
    assert options.config["model_auto_compact_token_limit"] == 20480
    assert options.ephemeral is True and len(options.offered_tools) == 4
    assert [s.skill_id for s in value.skills] == ["summary_failures"]
    assert profile.inspection()["qwen_thinking_requested"] is True


def test_cli_requires_execute_and_never_changes_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    command = [sys.executable, str(probe.ENTRY), "run", "--output", str(output),
               "--server-checkpoint-identity", "/missing-identity", "--server-canary", "/missing-canary",
               "--codex-bin", "/missing-codex"]
    for extra in ([], ["--execute"]):
        completed = subprocess.run([*command, *extra], capture_output=True, text=True, timeout=20)
        assert completed.returncode == 2
        assert list(output.iterdir()) == []
    assert probe.verify(tmp_path / "absent")["reason"] == "missing_terminal_receipt"


def test_real_stdio_public_tools_bootstrap_without_provider(tmp_path):
    workspace = probe.prepare(tmp_path / "run")
    audit = workspace.parent / "audit"
    audit.mkdir()
    probe.write_public_document(audit / "phase-policy.json", {"skill_stage": "S3"})
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "automed_read_file", "arguments": {"path": "notes/retention.json", "limit": 2048}}}
    requests = [request, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "automed_execute_python", "arguments": {"code": "raise AssertionError('must never execute')"}}}]
    completed = subprocess.run([sys.executable, "-I", "-B", str(probe.ENTRY), "serve", "--workspace",
        str(workspace), "--audit-root", str(audit)], input="\n".join(json.dumps(row) for row in requests) + "\n",
        capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    results = {row["id"]: row["result"] for row in map(json.loads, completed.stdout.splitlines())}
    assert results[1]["isError"] is True  # Missing model-authored note, not a host-filled answer.
    assert results[2]["tools"] == [row for row in probe.TOOLS if row["name"] in probe.NAMES]
    assert results[3]["isError"] is True
    assert len((audit / "mcp-events.jsonl").read_text().splitlines()) == 1
    assert not (audit / "code-executions").exists()


@pytest.mark.parametrize("kind,expected", [("empty", True), ("rejected", True), ("transport", None)])
def test_existing_core_control_policy_does_not_erase_note_recovery(tmp_path, kind, expected):
    output = tmp_path / "run"
    fixture(output)
    path = output / "turn-2/receipt.json"
    original = probe.codex_turn_receipt_from_document(probe.read_document(path))
    response = {"result": {"content": [{"type": "text", "text": '{"resources":[]}'}],
                           "structuredContent": None, "_meta": None}, "error": None}
    if kind != "empty":
        response = {"result": None, "error": {"message": "Mcp error: -32601: Method not found"
                    if kind == "rejected" else "connection reset"}}
    core = dict(tool_call_id=str(uuid4()), upstream_item_id=str(uuid4()), tool_type="mcpToolCall",
        name="list_mcp_resources", mcp_server="codex", mcp_tool="list_mcp_resources",
        fully_qualified_name="codex/list_mcp_resources", status="completed" if kind == "empty" else "failed",
        arguments={}, output=response, lifecycle=("item/started", "item/completed"), first_event_sequence=0)
    control = CodexToolCall(**core, receipt_blake3=blake3_hex(core))
    changed = {**original.core(), "tool_calls": (control, *original.tool_calls)}
    path.write_text(json.dumps(canonical_value(CodexTurnReceipt(**changed, receipt_blake3=blake3_hex(changed)))))
    result = probe.verify(output)
    assert result["passed"] is expected
    if expected:
        assert result["native_control_calls"] == 1
        assert result["native_rejected_control_calls"] == (kind == "rejected")
