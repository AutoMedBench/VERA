"""Provider-free fixtures; simulated Codex notifications are NOT model results."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4

import pytest

from eva_agent.codex_runtime import CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions, CodexToolOffer, CodexTurnInput
from eva_agent.codex_runtime.runtime import _logical_input
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes, canonical_value
from training.automedbench_lite.actor import snapshot
from training.automedbench_lite.adapter import write_once
from training.automedbench_lite.public_tools import PublicTools, TOOLS
from training.benchmark_feedback import automed_codex as bridge
from test_codex_runtime import _FakeBackend, _turn_event


def fixture(tmp_path, *, omit_false_default=False):
    run = tmp_path / "run"
    run.mkdir()
    case_id, workspace_id = "fixture-case", str(uuid4())
    workspace = run / "actors" / workspace_id
    (workspace / "inputs").mkdir(parents=True)
    (workspace / "outputs").mkdir()
    write_once(workspace / "task.json", {"case_id": case_id, "track": "classification",
        "instruction": "Inspect the public input and explain a plan."})
    write_once(run / "run-manifest.json", {"run_id": str(uuid4()), "diagnostic_only": False,
        "public_workspaces_only": True, "private_references_in_actor_workspace": False,
        "cases": [{"case_id": case_id, "track": "classification", "workspace_id": workspace_id}]})
    audit = run / "codex-rollouts" / case_id
    turn = audit / "turns/01-planning"
    turn.mkdir(parents=True)
    identity = {"exact_final_model_path": "/fixture/final-checkpoint"}
    identity_path = tmp_path / "identity.json"
    identity_path.write_bytes(canonical_json_bytes(identity))
    identity_digest = blake3_bytes(identity_path.read_bytes())
    write_once(run / "codex-rollouts/attempt.json", {"verified_skill_catalog_blake3": "b" * 64,
        "server_binding": {"identity_file_blake3": identity_digest,
        "identity": identity, "canary": {"checkpoint_identity_blake3": identity_digest,
            "exact_final_model_path": identity["exact_final_model_path"], "status": "complete",
            "model_info": {"model_path": identity["exact_final_model_path"]}}, "actual_process_argv_rechecked": True}})
    snapshot(workspace, turn / "before")
    tools = PublicTools(workspace=workspace, audit_root=audit, image="unused-provider-free-fixture")
    response = tools.call("automed_read_file", {"path": "task.json"}, 1)
    if omit_false_default:
        response.pop("isError")
        response["_meta"] = None
    snapshot(workspace, turn / "after")
    offer = CodexToolOffer(fully_qualified_name="automed_eval/automed_read_file", description=TOOLS[0]["description"],
                          input_schema=TOOLS[0]["inputSchema"], parallel_safe=False)
    options = CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model="fixture-Qwen", provider="fixture-local",
        cwd=str(workspace), sandbox=CodexSandbox.READ_ONLY, offered_tools=(offer,),
        base_instructions="Fixture base.", developer_instructions="Fixture developer.")
    value = CodexTurnInput(public_text="Read the public task and describe a plan.")
    write_once(turn / "request.json", {"logical_input": canonical_value(_logical_input(options, value)),
        "base_instructions": options.base_instructions, "developer_instructions": options.developer_instructions})

    def script(thread_id, turn_id):
        item = {"id": "fixture-call", "type": "mcpToolCall", "server": "automed_eval",
                "tool": "automed_read_file", "arguments": {"path": "task.json"}, "status": "inProgress"}
        return (_turn_event("turn/started", thread_id, turn_id, turn={"id": turn_id, "status": "inProgress"}),
            _turn_event("item/completed", thread_id, turn_id,
                item={"id": "comment", "type": "agentMessage", "text": "I will inspect the actual public contract.", "phase": "commentary"}),
            _turn_event("item/started", thread_id, turn_id, item=item),
            _turn_event("item/completed", thread_id, turn_id, item={**item, "status": "completed", "result": response}),
            _turn_event("item/completed", thread_id, turn_id,
                item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": "Plan retained; no completion claim."}),
            _turn_event("turn/completed", thread_id, turn_id, turn={"id": turn_id, "status": "completed", "items": []}))

    async def generate():
        async with CodexRuntime(_FakeBackend(script)) as runtime:
            handle = await runtime.start_thread(options)
            return await runtime.run_turn(handle, value)

    receipt = asyncio.run(generate())
    (turn / "receipt.json").write_bytes(canonical_json_bytes(receipt))
    return {"run_root": run, "case_id": case_id, "phase": "01-planning", "stage": "S1",
            "checkpoint_identity": identity_path, "output_root": tmp_path / "prepared"}


def test_retained_actual_host_calls_reopen_without_providers(tmp_path):
    args = fixture(tmp_path)
    rollout, rubric, report = bridge.prepare_case(**args)
    assert report["valid"] and report["actual_host_call_count"] == 1
    assert report["provider_calls"] == report["gpu_calls"] == report["judge_calls"] == 0
    assert not report["stage_completion_claimed"] and not report["reward_emitted"]
    assert rubric.domain == "automedbench-classification" and rubric.stage == "S1"
    assert [row["role"] for row in rollout["messages"]] == ["system", "user", "assistant", "assistant", "tool", "assistant"]
    assert rollout["messages"][2]["content"]["text"].startswith("I will inspect")
    result = rollout["tool_trace"]["results"][0]
    assert result["output"]["host_event"]["arguments"] == {"path": "task.json"}
    assert rollout["provider_metadata"]["selected_skill_ids_by_turn"] == [[]]
    assert not rollout["provider_metadata"]["skill_attribution_performed"]
    assert (args["output_root"] / "rollout.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("kind", ["snapshot", "request", "host", "checkpoint", "diagnostic"])
def test_tampered_or_wrong_scope_evidence_rejected_before_judge(tmp_path, kind):
    args = fixture(tmp_path)
    run, case = args["run_root"], args["case_id"]
    turn = run / "codex-rollouts" / case / "turns/01-planning"
    if kind == "snapshot":
        path = turn / "after/files/task.json"
        path.chmod(0o600)
        path.write_text("changed")
    elif kind == "request":
        path = turn / "request.json"
        document = json.loads(path.read_bytes())
        document["logical_input"]["public_text"] = "changed"
        document.pop("document_blake3")
        path.unlink()
        write_once(path, document)
    elif kind == "host":
        path = run / "codex-rollouts" / case / "mcp-events.jsonl"
        row = json.loads(path.read_text())
        row["result"]["content"] = "invented text"
        path.write_text(json.dumps(row) + "\n")
    elif kind == "checkpoint":
        args["checkpoint_identity"].write_text('{"exact_final_model_path":"other"}')
    else:
        path = run / "run-manifest.json"
        document = json.loads(path.read_bytes())
        document["diagnostic_only"] = True
        document.pop("document_blake3")
        path.unlink()
        write_once(path, document)
    with pytest.raises((bridge.FeedbackBridgeError, ValueError)):
        bridge.prepare_case(**args)
    assert not (args["output_root"] / "judge-attempt.json").exists()


def test_binary_image_projection_preserves_original_and_all_text():
    original = {"content": [{"type": "text", "text": "actual observation"},
        {"type": "image", "mimeType": "image/png", "data": "eHl6"}], "isError": False}
    frozen = deepcopy(original)
    result = bridge.visible_response(original, "a" * 64)
    assert original == frozen and result["content"][0] == original["content"][0]
    assert result["content"][1]["retained_binary_payload"]["blake3"] == blake3_bytes(b"xyz")
    assert "data" not in result["content"][1]


def test_installed_mcp_omitted_success_default_preserves_original_wire_binding(tmp_path):
    args = fixture(tmp_path, omit_false_default=True)
    _, _, report = bridge.prepare_case(**args)
    assert report["actual_host_call_count"] == 1 and report["valid"]


def test_failure_is_one_attempt_no_zero_reward_or_fallback(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    bridge.prepare_case(**args)
    calls = []
    async def unavailable(*_, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("fixture provider unavailable")
    monkeypatch.setattr(bridge, "grade_rollout", unavailable)
    with pytest.raises(RuntimeError):
        bridge.judge_once(args["output_root"])
    with pytest.raises(FileExistsError):
        bridge.judge_once(args["output_root"])
    assert len(calls) == 1 and calls[0]["backend"] == "native_astra"
    assert not (args["output_root"] / "feedback.json").exists()


def test_s1_cannot_use_later_phase_under_same_label(tmp_path):
    args = fixture(tmp_path)
    args["phase"] = "03-review"
    with pytest.raises(bridge.FeedbackBridgeError, match="s1_requires"):
        bridge.prepare_case(**args)
