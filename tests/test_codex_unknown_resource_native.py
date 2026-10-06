"""Opt-in real CLI rejection/receipt fixture; deterministic local replies only.

Run from the paired MED root with this CORE's src on pytest's pythonpath.
No real model, external provider, GPU, or ambient credentials are used.
"""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from eva_agent.codex_pipeline.adapter import CodexToolExecutionBridge, _validate_codex_core_resource_call
from eva_agent.codex_pipeline.turn_mcp import TurnMCPBridgeFactory
from eva_agent.codex_providers.adapter import _UpstreamOutcome
from eva_agent.codex_runtime import (
    CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions, CodexToolOffer,
    CodexTurnInput, verify_codex_turn_receipt,
)
from eva_agent.codex_runtime.contracts import codex_core_mcp_resource_call_error
from eva_agent.pipeline import FilesystemSandbox, ParallelToolRuntime, RandomUUIDFactory, ToolRegistry
from eva_agent.pipeline.digests import canonical_value
from eva_agent.training.task_memory_tools import TASK_MEMORY_TOOLS, augment_registry
from training.automedbench_lite.local_qwen import local_qwen_setup
from test_codex_provider_routes import _chat_response
from test_completed_invalid_tool_history_native import DEFAULT_BINARY


@pytest.mark.skipif(os.environ.get("EVA_RUN_NATIVE_UNKNOWN_RESOURCE_FIXTURE") != "1",
                    reason="explicit opt-in real local CLI rejection fixture")
def test_native_unknown_server_resource_read_is_retained_without_host_execution(tmp_path):
    binary = Path(os.environ.get("EVA_NATIVE_CODEX_BIN", str(DEFAULT_BINARY)))
    if not binary.is_file():
        pytest.skip("Pinned local Codex 0.153.4 is unavailable")
    workspace = FilesystemSandbox(tmp_path / "workspaces", str(uuid4()), {"task.json": b'{"public":true}'})
    before = workspace.snapshot("before-rejected-read")
    tools = ParallelToolRuntime(workspace=workspace, registry=augment_registry(ToolRegistry(())),
                                id_factory=RandomUUIDFactory())
    bridge = CodexToolExecutionBridge(tools, id_factory=RandomUUIDFactory())
    server = "rlevo-medres-stage-workspace"
    arguments = {"server": server, "uri": "file:///workspace/task.json"}
    expected_error = f"unknown MCP server '{server}'"
    captured = []

    def upstream(binding, body):
        captured.append(deepcopy(body))
        if len(captured) == 1:
            # Select the CLI's actual built-in helper. It is deliberately not
            # offered by our canonical MCP server or implemented by this test.
            builtin = next(row["function"] for row in body.get("tools", [])
                           if row["function"]["name"].endswith("read_mcp_resource"))
            reply = _chat_response(binding.model_id, content=None, tool_calls=[{
                "id": "call-native-unknown-resource", "type": "function",
                "function": {"name": builtin["name"], "arguments": json.dumps(arguments)}}])
        else:
            assert len(captured) == 2, "Unexpected retry or extra model request"
            assert expected_error in json.dumps(body["messages"])
            reply = _chat_response(binding.model_id, content="NATIVE_UNKNOWN_RESOURCE_REJECTION_RETAINED")
        return _UpstreamOutcome(200, json.dumps(reply).encode(), 1)

    async def run(setup, mcp_root):
        if not callable(getattr(setup, "bind_canonical_tools", None)):
            pytest.skip("Run from paired MED root to use its canonical-binding local_qwen_setup")
        offers = tuple(CodexToolOffer("evamed/" + row["name"], row["description"], row["inputSchema"],
            parallel_safe=row["name"] == "automed_read_file", read_only=row["name"] == "automed_read_file")
            for row in TASK_MEMORY_TOOLS)
        setup.bind_canonical_tools(offers)
        options = CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model=setup.model,
            provider=setup.provider, cwd=str(workspace.root), sandbox=CodexSandbox.READ_ONLY,
            ephemeral=False, config=setup.thread_config, offered_tools=offers)
        factory = TurnMCPBridgeFactory(proxy_python=Path(sys.executable).resolve(),
            proxy_script=Path(__file__).resolve().parents[1] / "src/eva_agent/codex_pipeline/turn_mcp_proxy.py",
            temp_root=mcp_root, tool_timeout_seconds=15)
        with factory.open_actor(options, bridge) as bound:
            async with CodexRuntime(setup.backend()) as runtime:
                thread = await runtime.start_thread(bound)
                return await asyncio.wait_for(runtime.run_turn(thread, CodexTurnInput(
                    public_text="Attempt the fixture resource read, observe the rejection, then finish.")), 45)

    with TemporaryDirectory(prefix="eva-resource-mcp-", dir="/tmp") as mcp_root:
        with local_qwen_setup(run_root=tmp_path / "local-runtime", workers=1,
                exact_tool_schemas=True, normalize_priority_messages=True,
                codex_bin=binary, auto_compact_token_limit=16000,
                upstream_transport=upstream) as setup:
            receipt = asyncio.run(run(setup, Path(mcp_root)))
    verify_codex_turn_receipt(receipt)
    assert receipt.status == "completed"
    assert receipt.final_response == "NATIVE_UNKNOWN_RESOURCE_REJECTION_RETAINED"
    assert len(captured) == 2 and len(receipt.tool_calls) == 1
    rejected = receipt.tool_calls[0]
    assert rejected.name == rejected.mcp_tool == "read_mcp_resource"
    assert rejected.mcp_server == server and rejected.status == "failed"
    assert canonical_value(rejected.arguments) == arguments
    assert rejected.output["result"] is None
    assert expected_error in rejected.output["error"]["message"]
    assert codex_core_mcp_resource_call_error(call=rejected, operation="read_mcp_resource",
        offered_mcp_tool_names=receipt.offered_mcp_tool_names) is None
    _validate_codex_core_resource_call(receipt, rejected, operation="read_mcp_resource")
    assert tools.trace().results == () and tools.trace().retry_count == 0
    assert workspace.snapshot("after-rejected-read").tree_blake3 == before.tree_blake3
