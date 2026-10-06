"""Failure-only admission matching the retained round4 unknown-server read."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from eva_agent.codex_pipeline import CodexPipelineError, CodexRolloutAdapter
from eva_agent.codex_pipeline.adapter import _validate_codex_core_resource_call
from eva_agent.codex_runtime.contracts import codex_core_mcp_resource_call_error, codex_core_mcp_resource_operation
from eva_agent.pipeline import (Cohort, DeterministicUUIDFactory, EvidenceBundle, FilesystemSandbox,
    ImmutableArtifactStore, PipelineVerifier, PipelineVerificationError)
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.pipeline.runner import evidence_core
from eva_agent.pipeline.verify import _verify_codex_core_resource_call
from test_codex_pipeline_adapter import _RuntimeFactory, _actor_options, _event, _request


SERVER = "rlevo-medres-stage-workspace"
URI = "file:///workspace/task.json"
MESSAGE = f"resources/read failed: unknown MCP server '{SERVER}'"
OFFERED = ("evamed/inspect_a", "evamed/inspect_b")


def call():
    return SimpleNamespace(name="read_mcp_resource", mcp_server=SERVER, mcp_tool="read_mcp_resource",
        lifecycle=("item/started", "item/completed"), status="failed",
        arguments={"server": SERVER, "uri": URI},
        output={"durationMs": 1, "result": None, "error": {"message": MESSAGE}})


def test_actual_failure_shape_is_shared_by_adapter_and_independent_verifier():
    rejected = call()
    assert codex_core_mcp_resource_operation(server=SERVER, tool="read_mcp_resource",
        offered_mcp_tool_names=OFFERED) == "read_mcp_resource"
    assert codex_core_mcp_resource_call_error(call=rejected, operation="read_mcp_resource",
        offered_mcp_tool_names=OFFERED) is None
    _validate_codex_core_resource_call(SimpleNamespace(offered_mcp_tool_names=OFFERED), rejected,
        operation="read_mcp_resource")
    _verify_codex_core_resource_call(rejected, "read_mcp_resource", set(OFFERED))


@pytest.mark.parametrize("change", [
    {"status": "completed"},
    {"arguments": {"server": "other-server", "uri": URI}},
    {"arguments": {"server": SERVER, "uri": ""}},
    {"arguments": {"server": SERVER, "uri": URI, "extra": True}},
    {"mcp_tool": "execute_code"},
    {"lifecycle": ("item/started",)},
    {"output": {"result": {"contents": []}, "error": {"message": MESSAGE}}},
    {"output": {"error": {"message": MESSAGE}}},
    {"output": {"result": None, "error": {"message": MESSAGE}, "contents": []}},
    {"output": {"result": None, "error": {"message": MESSAGE, "contents": []}}},
    {"output": {"result": None, "error": {"message": "resources/read failed: transport disconnected"}}},
    {"output": {"result": None, "error": {"message": "resources/read failed: unknown MCP server 'other-server'"}}},
    {"output": {"result": None, "error": {"message": "Mcp error: -32601: method not exposed"}}},
    {"output": {"result": None, "error": {"message": "method not found"}}},
    {"output": {"result": None, "error": {"message": f"resources/read failed for `{SERVER}` ({MESSAGE}): transport disconnected"}}},
])
def test_success_content_mismatches_and_unproven_errors_fail_both_consumers(change):
    rejected = call()
    for key, value in change.items():
        setattr(rejected, key, deepcopy(value))
    assert codex_core_mcp_resource_call_error(call=rejected, operation="read_mcp_resource",
        offered_mcp_tool_names=OFFERED) is not None
    with pytest.raises(CodexPipelineError):
        _validate_codex_core_resource_call(SimpleNamespace(offered_mcp_tool_names=OFFERED), rejected,
            operation="read_mcp_resource")
    with pytest.raises(PipelineVerificationError):
        _verify_codex_core_resource_call(rejected, "read_mcp_resource", set(OFFERED))


def test_unknown_server_rejection_does_not_apply_to_configured_servers():
    rejected = call()
    assert codex_core_mcp_resource_call_error(call=rejected, operation="read_mcp_resource",
        offered_mcp_tool_names=(*OFFERED, SERVER + "/actual_tool")) is not None
    rejected.mcp_server = "evamed"
    rejected.arguments["server"] = "evamed"
    rejected.output["error"]["message"] = "Mcp error: -32601: method not exposed"
    assert codex_core_mcp_resource_call_error(call=rejected, operation="read_mcp_resource",
        offered_mcp_tool_names=OFFERED) is None  # Existing configured-server policy unchanged.


def test_failed_turn_at_8192_preserves_rejection_without_any_host_call(tmp_path):
    def script(thread, turn, bridge=None):
        item = {"id": "actual-shaped-read", "type": "mcpToolCall", "server": SERVER,
            "tool": "read_mcp_resource", "arguments": {"server": SERVER, "uri": URI},
            "status": "inProgress", "result": None, "error": None}
        return (
            _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
            _event("item/started", thread, turn, item=item),
            _event("item/completed", thread, turn, item={**item, "status": "failed",
                "error": {"message": MESSAGE}, "durationMs": 1}),
            _event("turn/completed", thread, turn, turn={"id": turn, "status": "failed"}),
        )
    ids = DeterministicUUIDFactory("unknown-resource-read-no-host-effect")
    workspace = FilesystemSandbox(tmp_path / "workspaces", ids.new("workspace"),
        {"seed.txt": b"medical evidence\n"})
    request, tools = _request(Cohort.WEAK, workspace.root)
    before = workspace.snapshot("before")
    budget = {"kind": "total_output_budget", "count": 8192, "limit": 8192, "requests": 4}
    adapter = CodexRolloutAdapter(_RuntimeFactory(script), _actor_options,
        controlled_budget_terminal=lambda: budget)
    rollout = adapter.run(request, tools)
    after = workspace.snapshot("after")
    receipt = rollout.safe_metadata["codex_turn_receipt"]
    assert receipt["status"] == "failed" and len(receipt["tool_calls"]) == 1
    rejected = receipt["tool_calls"][0]
    assert rejected["status"] == "failed" and rejected["arguments"] == {"server": SERVER, "uri": URI}
    assert rejected["output"]["error"]["message"] == MESSAGE
    assert tools.execute_count == 0 and tools.trace().results == ()
    assert tools.trace().retry_count == 0 and before.tree_blake3 == after.tree_blake3
    assert rollout.safe_metadata["controlled_budget_termination"]["budget"] == budget
    partial = EvidenceBundle(bundle_id=ids.new("bundle"), rollout_id=request.rollout_id,
        sandbox_manifest=request.sandbox, model=request.model,
        policy_visible_context=request.policy_visible_context,
        context_blake3=blake3_hex(request.policy_visible_context),
        workspace_before=before, workspace_after=after, tool_trace=tools.trace(),
        policy_events=rollout.policy_events, assistant_output=rollout.assistant_output,
        provider_receipt_blake3=rollout.provider_receipt_blake3,
        safe_provider_metadata=rollout.safe_metadata, bundle_blake3="")
    evidence = replace(partial, bundle_blake3=blake3_hex(evidence_core(partial)))
    PipelineVerifier(ImmutableArtifactStore(tmp_path / "verified-artifacts"))._verify_trajectory(
        SimpleNamespace(evidence=evidence, cohort=Cohort.WEAK))
