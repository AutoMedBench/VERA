from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from eva_agent.harness import FunctionCall, HarnessContractError, LocalVenvWorkspace, ToolRegistry


def test_local_runtime_reads_and_executes_without_http(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("medical evidence", encoding="utf-8")
    runtime = LocalVenvWorkspace(workspace_root=tmp_path)
    registry = ToolRegistry(runtime.tool_definitions())
    calls = (
        FunctionCall(
            event_id="f6bbffca-47e2-4db4-a34c-386f2231d8ea",
            provider_call_id="read-a",
            name="read_file",
            arguments={"path": "note.txt"},
        ),
        FunctionCall(
            event_id="19f2c93a-d4c9-49c2-979b-972c95b8bf04",
            provider_call_id="list-a",
            name="list_files",
            arguments={"path": "."},
        ),
    )
    group = asyncio.run(
        registry.execute_group(
            calls, stage="S2", max_parallel_tools=8, timeout_seconds=30
        )
    )
    assert group.parallel is True
    assert group.max_parallelism_observed >= 1
    assert group.observations[0].output["content"] == "medical evidence"
    assert group.observations[1].output["files"] == ["note.txt"]


def test_local_runtime_rejects_escape(tmp_path: Path) -> None:
    runtime = LocalVenvWorkspace(workspace_root=tmp_path)
    read_tool = next(
        tool for tool in runtime.tool_definitions() if tool.name == "read_file"
    )
    registry = ToolRegistry((read_tool,))
    call = FunctionCall(
        event_id="a28e6cd9-d33f-45c8-805b-b93c2198e78e",
        provider_call_id="escape",
        name="read_file",
        arguments={"path": "../secret"},
    )
    group = asyncio.run(
        registry.execute_group(
            (call,), stage="S2", max_parallel_tools=1, timeout_seconds=30
        )
    )
    assert group.observations[0].ok is False
    assert group.observations[0].error_type == "HarnessContractError"
