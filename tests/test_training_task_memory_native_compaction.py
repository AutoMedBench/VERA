"""Opt-in real CLI/MCP compaction fixture; deterministic replies, not model recall.

No inference server, GPU, external provider, or ambient credentials are used.
The mock's token-usage count forces the real CLI's automatic compaction path.
Run from the paired MED root with this CORE's src on pytest's pythonpath, as in
production: that local_qwen_setup owns exact canonical MCP schema restoration.
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

from eva_agent.codex_pipeline.adapter import CodexToolExecutionBridge
from eva_agent.codex_pipeline.turn_mcp import TurnMCPBridgeFactory
from eva_agent.codex_providers.adapter import _UpstreamOutcome
from eva_agent.codex_runtime import (
    CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions, CodexToolOffer,
    CodexTurnInput, verify_codex_turn_receipt,
)
from eva_agent.pipeline import FilesystemSandbox, ParallelToolRuntime, RandomUUIDFactory, ToolRegistry
from eva_agent.pipeline.digests import canonical_value
from eva_agent.training.task_memory_tools import TASK_MEMORY_TOOLS, augment_registry
from training.automedbench_lite.local_qwen import local_qwen_setup
from test_codex_provider_routes import _chat_response
from test_completed_invalid_tool_history_native import DEFAULT_BINARY


@pytest.mark.skipif(os.environ.get("EVA_RUN_NATIVE_NOTE_FIXTURE") != "1",
                    reason="explicit opt-in real local CLI transport-only fixture")
def test_real_cli_note_write_compact_read_has_native_receipts(tmp_path):
    binary = Path(os.environ.get("EVA_NATIVE_CODEX_BIN", str(DEFAULT_BINARY)))
    if not binary.is_file():
        pytest.skip("Pinned local Codex 0.153.4 is unavailable")
    workspace = FilesystemSandbox(tmp_path / "workspaces", str(uuid4()), {})
    registry = augment_registry(ToolRegistry(()))
    tools = ParallelToolRuntime(workspace=workspace, registry=registry, id_factory=RandomUUIDFactory())
    bridge = CodexToolExecutionBridge(tools, id_factory=RandomUUIDFactory())
    captured = []
    note = "Fixture durable state: validate artifact, then continue S4."

    def upstream(binding, body):
        captured.append(deepcopy(body))
        ordinal = len(captured)
        if ordinal in (1, 3):
            name = "automed_write_note" if ordinal == 1 else "automed_read_file"
            offered = next(row["function"] for row in body.get("tools", [])
                           if row["function"]["name"].endswith(name))
            descriptor = next(row for row in TASK_MEMORY_TOOLS if row["name"] == name)
            assert offered["parameters"] == descriptor["inputSchema"]
            assert offered["description"] == descriptor["description"]
            arguments = ({"name": "task-state.md", "content": note} if ordinal == 1
                         else {"path": "notes/task-state.md"})
            reply = _chat_response(binding.model_id, content=None, tool_calls=[{
                "id": f"call-note-{ordinal}", "type": "function",
                "function": {"name": offered["name"], "arguments": json.dumps(arguments)}}])
            if ordinal == 1:
                reply["usage"] = {"prompt_tokens": 20000, "completion_tokens": 10, "total_tokens": 20010}
        elif ordinal == 2:
            assert not body.get("tools"), "Expected actual tool-free CLI compaction request"
            assert workspace.read_bytes("notes/task-state.md").decode() == note
            reply = _chat_response(binding.model_id,
                content="The task state was saved in notes/task-state.md. Reopen that file to continue.")
        else:
            assert ordinal == 4, "Unexpected extra mock request"
            assert note in json.dumps(body["messages"])
            reply = _chat_response(binding.model_id, content="NATIVE_NOTE_TRANSPORT_COMPLETE")
        return _UpstreamOutcome(200, json.dumps(reply).encode(), 1)

    async def run(setup, mcp_root):
        offers = tuple(CodexToolOffer("evamed/" + row["name"], row["description"], row["inputSchema"],
                        parallel_safe=row["name"] == "automed_read_file",
                        read_only=row["name"] == "automed_read_file") for row in TASK_MEMORY_TOOLS)
        if not callable(getattr(setup, "bind_canonical_tools", None)):
            pytest.skip("Run from paired MED root to use its canonical-binding local_qwen_setup")
        setup.bind_canonical_tools(offers)
        options = CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model=setup.model,
            provider=setup.provider, cwd=str(workspace.root), sandbox=CodexSandbox.READ_ONLY,
            ephemeral=False, config=setup.thread_config, offered_tools=offers)
        proxy = Path(__file__).resolve().parents[1] / "src/eva_agent/codex_pipeline/turn_mcp_proxy.py"
        factory = TurnMCPBridgeFactory(proxy_python=Path(sys.executable).resolve(),
            proxy_script=proxy, temp_root=mcp_root, tool_timeout_seconds=15)
        with factory.open_actor(options, bridge) as bound:
            async with CodexRuntime(setup.backend()) as runtime:
                thread = await runtime.start_thread(bound)
                return await asyncio.wait_for(runtime.run_turn(thread, CodexTurnInput(
                    public_text="Write the fixture task note, then read it after compaction and finish.")), 45)

    with TemporaryDirectory(prefix="eva-note-mcp-", dir="/tmp") as mcp_root:
        with local_qwen_setup(run_root=tmp_path / "local-runtime", workers=1,
                exact_tool_schemas=True, normalize_priority_messages=True,
                codex_bin=binary, auto_compact_token_limit=16000,
                upstream_transport=upstream) as setup:
            receipt = asyncio.run(run(setup, Path(mcp_root)))
    verify_codex_turn_receipt(receipt)
    assert receipt.status == "completed"
    assert receipt.final_response == "NATIVE_NOTE_TRANSPORT_COMPLETE"
    assert len(captured) == 4
    results = tools.trace().results
    assert [row.name for row in results] == ["automed_write_note", "automed_read_file"]
    assert all(row.status == "completed" for row in results)
    assert results[0].workspace_after_blake3 == results[1].workspace_before_blake3
    assert results[1].output["content"] == note
    assert len(receipt.tool_calls) == 2
    events = canonical_value(receipt.events)
    assert any(row.get("payload", {}).get("item", {}).get("type") == "contextCompaction"
               for row in events), "Native receipt must retain an actual CLI compaction event"
