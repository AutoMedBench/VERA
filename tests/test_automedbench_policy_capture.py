"""Track consumer of the independently tested harness interruption contract."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.codex_runtime import CodexRuntime, CodexTurnInput
from eva_agent.pipeline.digests import canonical_json_bytes
from training.automedbench_lite import policy_capture
from training.automedbench_lite.adapter import EvaluationError
from test_codex_policy_budget import MockAppServer, backend, options


def capture(server, tmp_path, monkeypatch):
    target = tmp_path / "turn"
    target.mkdir()
    async def wait_jobs(*_):
        assert server.order[-1] == "unregistered"
        server.order.append("jobs-os-exited")
    monkeypatch.setattr(policy_capture, "await_model_jobs", wait_jobs)
    def snapshot(_, path):
        assert server.order[-1] == "jobs-os-exited"
        server.order.append("snapshot")
        path.mkdir()
        return {"document_blake3": "mock-snapshot-binding"}
    async def scenario():
        async with CodexRuntime(backend(server)) as runtime:
            handle = await runtime.start_thread(options(tmp_path))
            return await policy_capture.capture_turn(runtime, handle, CodexTurnInput(public_text="fixture"),
                timeout=.02, target=target, audit=tmp_path, inventory=None, snapshot=snapshot)
    return scenario(), target


def test_final_snapshot_after_observed_terminal_host_results_and_job_exits(tmp_path, monkeypatch):
    server = MockAppServer("interrupt", tmp_path)
    scenario, target = capture(server, tmp_path, monkeypatch)
    receipt, terminal = asyncio.run(scenario)
    assert server.order[-3:] == ["unregistered", "jobs-os-exited", "snapshot"]
    assert (target / "receipt.json").read_bytes() == canonical_json_bytes(receipt)
    assert (target / "receipt.json").stat().st_mode & 0o777 == 0o600
    assert terminal["workspace_quiescence_verified"] and terminal["reward"] is None
    assert terminal["actual_terminal_status"] == "interrupted"
    assert not terminal["later_stages_evaluated"]
    assert terminal["after_snapshot_document_blake3"] == "mock-snapshot-binding"
    assert policy_capture.require_joined_host_results(receipt, tmp_path)["joined_host_call_count"] == 1


def test_native_codex_resource_control_is_not_misclassified_as_host_worker(tmp_path, monkeypatch):
    server = MockAppServer("interrupt", tmp_path)
    scenario, _ = capture(server, tmp_path, monkeypatch)
    receipt, _ = asyncio.run(scenario)
    assert set(policy_capture.require_joined_host_results(receipt, tmp_path)) == {
        "joined_host_event_ids", "joined_host_call_count"}
    control = SimpleNamespace(
        tool_type="mcpToolCall", name="list_mcp_resources",
        mcp_server="codex", mcp_tool="list_mcp_resources",
        status="completed", lifecycle=("item/started", "item/completed"),
        output={"result": {"_meta": None, "content": [], "structuredContent": None}},
    )
    mixed = SimpleNamespace(tool_calls=(*receipt.tool_calls, control),
                            offered_mcp_tool_names=receipt.offered_mcp_tool_names)
    result = policy_capture.require_joined_host_results(mixed, tmp_path)
    assert result["joined_host_call_count"] == 1
    assert result["native_control_call_count"] == 1
    assert result["native_control_operations"] == ["list_mcp_resources"]


def test_null_structured_content_from_data_plane_fails_with_fixed_category(tmp_path):
    call = SimpleNamespace(
        tool_type="mcpToolCall", name="automed_read_file",
        mcp_server="automed_eval", mcp_tool="automed_read_file",
        status="completed", lifecycle=("item/started", "item/completed"),
        output={"result": {"_meta": None, "content": [], "structuredContent": None}},
    )
    receipt = SimpleNamespace(tool_calls=(call,),
                              offered_mcp_tool_names=("automed_eval/automed_read_file",))
    with pytest.raises(EvaluationError, match="host_worker_quiescence_unproved"):
        policy_capture.require_joined_host_results(receipt, tmp_path)


@pytest.mark.parametrize("mode,category", [("cancelled_tool", "quiescence_unproved"),
                                         ("provider_error", "independent_provider_failure")])
def test_closed_provider_turn_without_host_proof_is_not_a_final_workspace(tmp_path, monkeypatch, mode, category):
    server = MockAppServer(mode, tmp_path)
    scenario, target = capture(server, tmp_path, monkeypatch)
    with pytest.raises(EvaluationError, match=category):
        asyncio.run(scenario)
    assert (target / "receipt.json").exists()  # Actual failure evidence retained.
    assert not (target / "after").exists()
    sidecar = json.loads((target / "policy-budget-terminal.json").read_bytes())
    assert not sidecar["workspace_quiescence_verified"] and sidecar["reward"] is None
    assert sidecar["infrastructure_error"]


def test_live_exact_job_supervisor_is_not_quiescent_from_child_exit_alone(tmp_path):
    job = tmp_path / "model-jobs/job-A"
    job.mkdir(parents=True)
    ticks = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
    (job / "process.json").write_text(json.dumps({"pid": os.getpid(), "start_ticks": ticks}))
    with pytest.raises(EvaluationError, match="publisher_still_running"):
        asyncio.run(policy_capture.await_model_publishers(tmp_path, tmp_path, timeout=0))


def test_exited_job_publisher_requires_matching_final_public_exit_bytes(tmp_path):
    job = tmp_path / "model-jobs/job-A"
    job.mkdir(parents=True)
    # Deliberately mismatched birth identity means this is not the owned process.
    (job / "process.json").write_text(json.dumps({"pid": os.getpid(), "start_ticks": "not-this-process"}))
    authoritative = tmp_path / "model-jobs/authoritative/job-A"
    public = tmp_path / "outputs/agents_outputs/prescribed-model-jobs/job-A"
    authoritative.mkdir(parents=True)
    public.mkdir(parents=True)
    (authoritative / "process-exit.json").write_bytes(b"fixture-exit-bytes")
    with pytest.raises(EvaluationError, match="publication_incomplete"):
        asyncio.run(policy_capture.await_model_publishers(tmp_path, tmp_path))
    (public / "process-exit.json").write_bytes(b"fixture-exit-bytes")
    asyncio.run(policy_capture.await_model_publishers(tmp_path, tmp_path))
