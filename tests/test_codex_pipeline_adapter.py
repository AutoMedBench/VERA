from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import Any, AsyncIterator, Mapping, Sequence

import pytest

import eva_agent.codex_pipeline.adapter as codex_adapter
from eva_agent.codex_pipeline import (
    CodexOpus5AgentJudge,
    CodexPipelineError,
    CodexRolloutAdapter,
    CodexToolExecutionBridge,
    CodexTurnRunnerPort,
    SyncCodexTurnRunner,
)
from eva_agent.codex_runtime import (
    BackendInputItem,
    BackendTurnOptions,
    CodexRole,
    CodexRuntime,
    CodexRuntimeError,
    CodexSandbox,
    CodexSkill,
    CodexThreadOptions,
    CodexToolCall,
    CodexToolOffer,
    CodexTurnInput,
    PersistentCodexRuntimeRunner,
)
from eva_agent.pipeline import (
    BenchmarkEpisode,
    BenchmarkSource,
    Cohort,
    DeterministicUUIDFactory,
    EvidenceBundle,
    FilesystemSandbox,
    ImmutableArtifactStore,
    InfrastructureQuarantineError,
    JudgeRequest,
    ModelTarget,
    OpenAIStyleOpus5Judge,
    ParallelToolRuntime,
    PipelineVerifier,
    PipelineVerificationError,
    RolloutRequest,
    SandboxManifest,
    Stage,
    ToolCall,
    ToolTrace,
    ToolDefinition,
    ToolRegistry,
    TrajectoryEvent,
    VerifiableDataPipeline,
    WeightedRubricRewarder,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from eva_agent.pipeline.contracts import ContractError
from eva_agent.pipeline.runner import evidence_core
from eva_agent.rubrics import load_and_compile_registry


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _Notification:
    method: str
    payload: Mapping[str, Any]


@dataclass
class _Concurrency:
    barrier: Barrier | None = None
    active: int = 0
    maximum: int = 0
    lock: Lock = field(default_factory=Lock)


class _Turn:
    def __init__(self, turn_id: str, notifications: Sequence[_Notification], state: _Concurrency):
        self.id = turn_id
        self._notifications = notifications
        self._state = state

    async def stream(self) -> AsyncIterator[_Notification]:
        with self._state.lock:
            self._state.active += 1
            self._state.maximum = max(self._state.maximum, self._state.active)
        try:
            if self._state.barrier is not None:
                self._state.barrier.wait(timeout=3)
            for notification in self._notifications:
                await asyncio.sleep(0)
                yield notification
        finally:
            with self._state.lock:
                self._state.active -= 1


class _Thread:
    def __init__(self, thread_id: str, backend: "_Backend") -> None:
        self.id = thread_id
        self._backend = backend

    async def turn(
        self, items: Sequence[BackendInputItem], options: BackendTurnOptions
    ) -> _Turn:
        self._backend.turns.append((tuple(items), options))
        turn_id = f"turn-{len(self._backend.turns)}"
        return _Turn(
            turn_id,
            self._backend.script(self.id, turn_id, self._backend.bridge),
            self._backend.state,
        )


class _Backend:
    sdk_version = "fake-codex-sdk"
    server_version = "fake-app-server"

    def __init__(self, script, state: _Concurrency | None = None) -> None:
        self.script = script
        self.state = state or _Concurrency()
        self.starts: list[CodexThreadOptions] = []
        self.turns: list[tuple[tuple[BackendInputItem, ...], BackendTurnOptions]] = []
        self.open_count = 0
        self.close_count = 0
        self.bridge: CodexToolExecutionBridge | None = None

    async def open(self) -> None:
        self.open_count += 1

    async def close(self) -> None:
        self.close_count += 1

    async def start_thread(self, options: CodexThreadOptions) -> _Thread:
        self.starts.append(options)
        return _Thread(f"thread-{len(self.starts)}", self)

    async def resume_thread(self, thread_id: str, options: CodexThreadOptions) -> _Thread:
        raise AssertionError((thread_id, options, "resume is forbidden"))


def _event(method: str, thread: str, turn_id: str, **extra: Any) -> _Notification:
    return _Notification(method, {"threadId": thread, "turnId": turn_id, **extra})


def _mcp_call_result(result: Mapping[str, Any], *, error: Any = None) -> CodexToolCall:
    core = {
        "tool_call_id": "00000000-0000-4000-8000-000000000001",
        "upstream_item_id": "observed-codex-mcp-call",
        "tool_type": "mcpToolCall",
        "name": "materialize_plan",
        "mcp_server": "evamed",
        "mcp_tool": "materialize_plan",
        "fully_qualified_name": "evamed/materialize_plan",
        "status": "completed",
        "arguments": {},
        "output": {"result": result, "error": error, "durationMs": 17},
        "lifecycle": ("item/started", "item/completed"),
        "first_event_sequence": 1,
    }
    return CodexToolCall(**core, receipt_blake3=blake3_hex(core))


def test_mcp_result_accepts_codex_app_server_success_without_is_error() -> None:
    """Match the Codex 0.153.4 McpToolCallResult observed in v14."""

    structured = {
        "schema": "eva.codex-pipeline-tool-observation.v1",
        "call_id": "00000000-0000-4000-8000-000000000002",
        "name": "materialize_plan",
        "arguments": {},
        "tool_result": {"status": "completed"},
        "bridge_receipt_blake3": "0" * 64,
    }
    # The app-server result requires content, preserves structuredContent, and
    # normalizes away the MCP protocol's optional/default-false isError field.
    observed = {
        "content": [{"type": "text", "text": "committed observation"}],
        "structuredContent": structured,
    }
    assert codex_adapter._mcp_result_object(_mcp_call_result(observed)) == structured


@pytest.mark.parametrize("is_error", (True, None, 0, "false"))
def test_mcp_result_still_rejects_non_false_explicit_error_flags(is_error: Any) -> None:
    result = {
        "content": [],
        "structuredContent": {"schema": "committed-observation.v1"},
        "isError": is_error,
    }
    with pytest.raises(CodexPipelineError, match="MCP structured result differs"):
        codex_adapter._mcp_result_object(_mcp_call_result(result))


def test_mcp_result_rejects_app_server_error_even_with_structured_content() -> None:
    result = {
        "content": [],
        "structuredContent": {"schema": "committed-observation.v1"},
    }
    with pytest.raises(CodexPipelineError, match="MCP result envelope differs"):
        codex_adapter._mcp_result_object(
            _mcp_call_result(result, error={"message": "tool failed"})
        )


def _simple_script(
    thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
) -> tuple[_Notification, ...]:
    del bridge
    return (
        _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
        _event(
            "item/completed",
            thread,
            turn,
            item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": "done"},
        ),
        _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
    )


def _failed_script(
    thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
) -> tuple[_Notification, ...]:
    del bridge
    return (
        _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
        _event(
            "item/completed",
            thread,
            turn,
            item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": "failed"},
        ),
        _event("turn/completed", thread, turn, turn={"id": turn, "status": "failed"}),
    )


def _tool_script(
    thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
) -> tuple[_Notification, ...]:
    assert bridge is not None
    observations = bridge.execute_group(
        (
            ("inspect_a", {"path": "seed.txt"}),
            ("inspect_b", {"path": "seed.txt"}),
        )
    )
    one = {
        "id": "tool-1",
        "type": "mcpToolCall",
        "server": "evamed",
        "tool": "inspect_a",
        "arguments": {"path": "seed.txt"},
        "status": "inProgress",
    }
    two = {
        "id": "tool-2",
        "type": "mcpToolCall",
        "server": "evamed",
        "tool": "inspect_b",
        "arguments": {"path": "seed.txt"},
        "status": "inProgress",
    }
    return (
        _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
        _event("item/started", thread, turn, item=one),
        _event("item/started", thread, turn, item=two),
        _event(
            "item/completed",
            thread,
            turn,
            item={
                **two,
                "status": "completed",
                "result": {
                    "content": [],
                    "structuredContent": observations[1].structured_content,
                    "isError": False,
                },
            },
        ),
        _event(
            "item/completed",
            thread,
            turn,
            item={
                **one,
                "status": "completed",
                "result": {
                    "content": [],
                    "structuredContent": observations[0].structured_content,
                    "isError": False,
                },
            },
        ),
        _event(
            "item/completed",
            thread,
            turn,
            item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": "verified"},
        ),
        _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
    )


def _rejected_s1_materialization_script(
    thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
) -> tuple[_Notification, ...]:
    """Retain a provider-submitted invalid S1 plan as semantic evidence."""

    assert bridge is not None
    observation = bridge.execute_group((("materialize_plan", {}),))[0]
    item = {
        "id": "s1-materialize-plan",
        "type": "mcpToolCall",
        "server": "evamed",
        "tool": "materialize_plan",
        "arguments": {},
        "status": "inProgress",
    }
    return (
        _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
        _event("item/started", thread, turn, item=item),
        _event(
            "item/completed",
            thread,
            turn,
            item={
                **item,
                "status": "completed",
                "result": {
                    "content": [],
                    "structuredContent": observation.structured_content,
                    "isError": False,
                },
            },
        ),
        _event(
            "item/completed",
            thread,
            turn,
            item={
                "id": "answer",
                "type": "agentMessage",
                "phase": "final_answer",
                "text": "S1 materialization was rejected by the signed contract.",
            },
        ),
        _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
    )


def _failed_turn_after_rejected_s1_script(
    thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
) -> tuple[_Notification, ...]:
    """Mirror Codex terminating immediately after a deterministic tool gate."""

    assert bridge is not None
    observation = bridge.execute_group((("materialize_plan", {}),))[0]
    item = {
        "id": "s1-materialize-plan-failed-turn",
        "type": "mcpToolCall",
        "server": "evamed",
        "tool": "materialize_plan",
        "arguments": {},
        "status": "inProgress",
    }
    return (
        _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
        _event("item/started", thread, turn, item=item),
        _event(
            "item/completed",
            thread,
            turn,
            item={
                **item,
                "status": "completed",
                "result": {
                    "content": [],
                    "structuredContent": observation.structured_content,
                    "isError": False,
                },
            },
        ),
        _event("turn/completed", thread, turn, turn={"id": turn, "status": "failed"}),
    )


def _core_resource_then_tool_script(
    thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
) -> tuple[_Notification, ...]:
    assert bridge is not None
    observation = bridge.execute_group((("inspect_a", {"path": "seed.txt"}),))[0]
    core = {
        "id": "core-resource",
        "type": "mcpToolCall",
        "server": "evamed",
        "tool": "read_mcp_resource",
        "arguments": {"server": "evamed", "uri": "eva://unavailable"},
        "status": "inProgress",
        "result": None,
        "error": None,
    }
    domain = {
        "id": "domain-tool",
        "type": "mcpToolCall",
        "server": "evamed",
        "tool": "inspect_a",
        "arguments": {"path": "seed.txt"},
        "status": "inProgress",
    }
    return (
        _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
        _event("item/started", thread, turn, item=core),
        _event(
            "item/completed",
            thread,
            turn,
            item={
                **core,
                "status": "failed",
                "result": None,
                "error": {
                    "message": "resources/read failed: Mcp error: -32601: method not found"
                },
                "durationMs": 1,
            },
        ),
        _event("item/started", thread, turn, item=domain),
        _event(
            "item/completed",
            thread,
            turn,
            item={
                **domain,
                "status": "completed",
                "result": {
                    "content": [],
                    "structuredContent": observation.structured_content,
                    "isError": False,
                },
            },
        ),
        _event(
            "item/completed",
            thread,
            turn,
            item={
                "id": "answer",
                "type": "agentMessage",
                "phase": "final_answer",
                "text": "done",
            },
        ),
        _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
    )


def _pipeline_tool_script(
    thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
) -> tuple[_Notification, ...]:
    assert bridge is not None
    reads = bridge.execute_group(
        (("inspect_a", {"path": "seed.txt"}), ("inspect_b", {"path": "seed.txt"}))
    )
    writes = bridge.execute_group((("write_once", {"path": "result.txt"}),))
    first = {
        "id": "read-a", "type": "mcpToolCall", "server": "evamed",
        "tool": "inspect_a", "arguments": {"path": "seed.txt"}, "status": "inProgress",
    }
    second = {
        "id": "read-b", "type": "mcpToolCall", "server": "evamed",
        "tool": "inspect_b", "arguments": {"path": "seed.txt"}, "status": "inProgress",
    }
    write = {
        "id": "write", "type": "mcpToolCall", "server": "evamed",
        "tool": "write_once", "arguments": {"path": "result.txt"}, "status": "inProgress",
    }

    def completed(item, observation):
        return {
            **item,
            "status": "completed",
            "result": {
                "content": [],
                "structuredContent": observation.structured_content,
                "isError": False,
            },
        }

    return (
        _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
        _event("item/started", thread, turn, item=first),
        _event("item/started", thread, turn, item=second),
        _event("item/completed", thread, turn, item=completed(second, reads[1])),
        _event("item/completed", thread, turn, item=completed(first, reads[0])),
        _event("item/started", thread, turn, item=write),
        _event("item/completed", thread, turn, item=completed(write, writes[0])),
        _event(
            "item/completed", thread, turn,
            item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": "done"},
        ),
        _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
    )


def _mixed_capability_transport_script(
    thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
) -> tuple[_Notification, ...]:
    """One visible transport group split by the runtime capability scheduler."""

    assert bridge is not None
    observations = bridge.execute_group(
        (
            ("write_once", {"path": "result.txt"}),
            ("inspect_a", {"path": "seed.txt"}),
        )
    )
    write = {
        "id": "mixed-write",
        "type": "mcpToolCall",
        "server": "evamed",
        "tool": "write_once",
        "arguments": {"path": "result.txt"},
        "status": "inProgress",
    }
    read = {
        "id": "mixed-read",
        "type": "mcpToolCall",
        "server": "evamed",
        "tool": "inspect_a",
        "arguments": {"path": "seed.txt"},
        "status": "inProgress",
    }

    def completed(item, observation):
        return {
            **item,
            "status": "completed",
            "result": {
                "content": [],
                "structuredContent": observation.structured_content,
                "isError": False,
            },
        }

    return (
        _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
        _event("item/started", thread, turn, item=write),
        _event("item/started", thread, turn, item=read),
        _event("item/completed", thread, turn, item=completed(read, observations[1])),
        _event("item/completed", thread, turn, item=completed(write, observations[0])),
        _event(
            "item/completed", thread, turn,
            item={
                "id": "answer",
                "type": "agentMessage",
                "phase": "final_answer",
                "text": "mixed transport complete",
            },
        ),
        _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
    )


def _falsely_split_transport_script(
    thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
) -> tuple[_Notification, ...]:
    """Expose one real execution frontier as two sequential transport groups."""

    assert bridge is not None
    observations = bridge.execute_group(
        (
            ("inspect_a", {"path": "seed.txt"}),
            ("inspect_b", {"path": "seed.txt"}),
        )
    )
    items = tuple(
        {
            "id": f"split-{index}",
            "type": "mcpToolCall",
            "server": "evamed",
            "tool": name,
            "arguments": {"path": "seed.txt"},
            "status": "inProgress",
        }
        for index, name in enumerate(("inspect_a", "inspect_b"))
    )
    events = [_event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"})]
    for item, observation in zip(items, observations, strict=True):
        events.append(_event("item/started", thread, turn, item=item))
        events.append(
            _event(
                "item/completed",
                thread,
                turn,
                item={
                    **item,
                    "status": "completed",
                    "result": {
                        "content": [],
                        "structuredContent": observation.structured_content,
                        "isError": False,
                    },
                },
            )
        )
    events.extend(
        (
            _event(
                "item/completed", thread, turn,
                item={
                    "id": "answer",
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": "split transport complete",
                },
            ),
            _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
        )
    )
    return tuple(events)


def _hybrid_native_script(
    thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
) -> tuple[_Notification, ...]:
    """Two overlapping MCP reads followed by one native fileChange."""

    assert bridge is not None
    reads = bridge.execute_group(
        (("inspect_a", {"path": "seed.txt"}), ("inspect_b", {"path": "seed.txt"}))
    )
    workspace = bridge._tools._workspace  # exact bound test workspace
    workspace.write_bytes("native-result.txt", b"verified native edit\n", create_only=True)
    first = {
        "id": "hybrid-read-a", "type": "mcpToolCall", "server": "evamed",
        "tool": "inspect_a", "arguments": {"path": "seed.txt"}, "status": "inProgress",
    }
    second = {
        "id": "hybrid-read-b", "type": "mcpToolCall", "server": "evamed",
        "tool": "inspect_b", "arguments": {"path": "seed.txt"}, "status": "inProgress",
    }

    def completed(item, observation):
        return {
            **item,
            "status": "completed",
            "result": {
                "content": [],
                "structuredContent": observation.structured_content,
                "isError": False,
            },
        }

    changed = {
        "id": "hybrid-file-change",
        "type": "fileChange",
        "changes": [{
            "path": "native-result.txt",
            "kind": {"type": "add"},
            "diff": "+verified native edit\n",
        }],
        "status": "inProgress",
    }
    return (
        _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
        _event("item/started", thread, turn, item=first),
        _event("item/started", thread, turn, item=second),
        _event("item/completed", thread, turn, item=completed(second, reads[1])),
        _event("item/completed", thread, turn, item=completed(first, reads[0])),
        _event("item/started", thread, turn, item=changed),
        _event("item/completed", thread, turn, item={**changed, "status": "completed"}),
        _event(
            "item/completed", thread, turn,
            item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": "hybrid verified"},
        ),
        _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
    )


def _native_command_script(command: str, action: Mapping[str, Any]):
    def script(
        thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
    ) -> tuple[_Notification, ...]:
        assert bridge is not None
        item = {
            "id": "native-command",
            "type": "commandExecution",
            "command": command,
            "commandActions": [dict(action)],
            "cwd": str(bridge._tools._workspace.root),
            "status": "inProgress",
            "source": "agent",
        }
        return (
            _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
            _event("item/started", thread, turn, item=item),
            _event(
                "item/completed", thread, turn,
                item={
                    **item,
                    "status": "completed",
                    "aggregatedOutput": "unsafe output",
                    "exitCode": 0,
                    "durationMs": 1,
                },
            ),
            _event(
                "item/completed", thread, turn,
                item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": "done"},
            ),
            _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
        )

    return script


def _native_file_change_script(changes: Sequence[Mapping[str, Any]]):
    def script(
        thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
    ) -> tuple[_Notification, ...]:
        assert bridge is not None
        item = {
            "id": "native-file-change",
            "type": "fileChange",
            "changes": [dict(change) for change in changes],
            "status": "inProgress",
        }
        return (
            _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
            _event("item/started", thread, turn, item=item),
            _event("item/completed", thread, turn, item={**item, "status": "completed"}),
            _event(
                "item/completed", thread, turn,
                item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": "done"},
            ),
            _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
        )

    return script


def _judge_native_script(
    thread: str, turn: str, bridge: CodexToolExecutionBridge | None = None
) -> tuple[_Notification, ...]:
    del bridge
    item = {
        "id": "judge-native",
        "type": "commandExecution",
        "command": "cat seed.txt",
        "commandActions": [{
            "type": "read", "command": "cat seed.txt", "name": "cat", "path": "seed.txt",
        }],
        "cwd": "/tmp/eva-codex-judge",
        "status": "inProgress",
        "source": "agent",
    }
    return (
        _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
        _event("item/started", thread, turn, item=item),
        _event(
            "item/completed", thread, turn,
            item={
                **item,
                "status": "completed",
                "aggregatedOutput": "private read",
                "exitCode": 0,
                "durationMs": 1,
            },
        ),
        _event(
            "item/completed", thread, turn,
            item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": "{}"},
        ),
        _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
    )


class _RuntimeFactory:
    def __init__(self, script, state: _Concurrency | None = None) -> None:
        self.script = script
        self.state = state
        self.backends: list[_Backend] = []
        self._lock = Lock()

    def __call__(self, bridge: CodexToolExecutionBridge | None = None) -> CodexRuntime:
        backend = _Backend(self.script, self.state)
        backend.bridge = bridge
        with self._lock:
            self.backends.append(backend)
            ordinal = len(self.backends)
        return CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory(f"codex-pipeline-runtime-{ordinal}")
        )


def _rubric(stage: Stage = Stage.E2E):
    return load_and_compile_registry(
        ROOT / "rubrics/source/domain-stage-tables.v1.json"
    ).resolve("medxpertqa", stage.value)


def _manifest(seed: str = "manifest", *, stage: Stage = Stage.E2E) -> SandboxManifest:
    rubric = _rubric(stage)
    episode = BenchmarkEpisode(
        episode_id="codex-adapter-episode",
        source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
        domain="medxpertqa",
        stage=stage,
        instruction="Use the offered tools and produce evidence.",
        policy_context={"question": "fixture"},
        initial_files={"seed.txt": b"medical evidence\n"},
    )
    return SandboxManifest.create(
        sandbox_id=DeterministicUUIDFactory(seed).new("sandbox"), episode=episode, rubric=rubric
    )


def _empty_trace() -> ToolTrace:
    core = {
        "results": (),
        "declared_call_ids": (),
        "joined_call_ids": (),
        "frontier_count": 0,
        "max_parallelism_observed": 0,
        "retry_count": 0,
    }
    return ToolTrace(**core, trace_blake3=blake3_hex(core))


class _OuterTools:
    def __init__(self, root: Path) -> None:
        self.workspace_root = root
        self._trace = _empty_trace()
        self.execute_count = 0

    def execute(self, calls):
        self.execute_count += 1
        raise AssertionError(calls)

    def trace(self) -> ToolTrace:
        return self._trace


def _public_tools() -> tuple[Mapping[str, Any], ...]:
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    return tuple(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Inspect {name}.",
                "parameters": schema,
                "x-eva-kind": "tool",
                "x-eva-parallel-safe": True,
            },
        }
        for name in ("inspect_a", "inspect_b")
    )


def _request(
    cohort: Cohort,
    root: Path,
    *,
    suffix: str = "",
    stage: Stage = Stage.E2E,
) -> tuple[RolloutRequest, _OuterTools]:
    manifest = _manifest(f"manifest-{cohort.value}-{suffix}", stage=stage)
    request = RolloutRequest(
        rollout_id=DeterministicUUIDFactory(f"rollout-{cohort.value}-{suffix}").new("rollout"),
        sandbox=manifest,
        model=ModelTarget(cohort, f"model-{cohort.value}", "fake-provider"),
        policy_visible_context={"question": "fixture", "stage": stage.value},
        available_tools=_public_tools(),
    )
    return request, _OuterTools(root)


_ROLE = {
    Cohort.WEAK: CodexRole.WEAK_ACTOR,
    Cohort.MIDDLE: CodexRole.MIDDLE_ACTOR,
    Cohort.STRONG: CodexRole.STRONG_ACTOR,
}


def _actor_options(
    request: RolloutRequest,
    cwd: str,
    offers: tuple[CodexToolOffer, ...],
    bridge: CodexToolExecutionBridge,
):
    del bridge
    return CodexThreadOptions(
        role=_ROLE[request.model.cohort],
        model=request.model.model_id,
        provider=request.model.provider,
        cwd=cwd,
        sandbox=CodexSandbox.READ_ONLY,
        offered_tools=offers,
    )


def _native_actor_options(
    request: RolloutRequest,
    cwd: str,
    offers: tuple[CodexToolOffer, ...],
    bridge: CodexToolExecutionBridge,
):
    return replace(
        _actor_options(request, cwd, offers, bridge),
        sandbox=CodexSandbox.WORKSPACE_WRITE,
    )


def test_actor_projects_parallel_codex_evidence_without_outer_reexecution(tmp_path: Path) -> None:
    factory = _RuntimeFactory(_tool_script)
    adapter = CodexRolloutAdapter(
        factory, _actor_options, id_factory=DeterministicUUIDFactory("actor-projection")
    )
    request, _unused = _request(Cohort.STRONG, tmp_path)
    workspace = FilesystemSandbox(
        tmp_path / "actor-workspaces",
        DeterministicUUIDFactory("actor-workspace").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    inspection_barrier = Barrier(2)

    def inspect(bound, arguments):
        inspection_barrier.wait(timeout=2)
        return {"content": bound.read_bytes(arguments["path"]).decode("utf-8")}

    tools = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(
            (
                ToolDefinition(
                    "inspect_a", "Inspect inspect_a.", schema, inspect,
                    parallel_safe=True, read_only=True,
                ),
                ToolDefinition(
                    "inspect_b", "Inspect inspect_b.", schema, inspect,
                    parallel_safe=True, read_only=True,
                ),
            )
        ),
        id_factory=DeterministicUUIDFactory("actor-tools"),
    )
    rollout = adapter.run(request, tools)

    grouped = [event for event in rollout.policy_events if len(event.tool_call_ids) == 2]
    assert len(grouped) == 1
    assert rollout.assistant_output == "verified"
    assert len(tools.trace().results) == 2
    assert tools.trace().max_parallelism_observed == 2
    assert rollout.safe_metadata["semantic_retry_count"] == 0
    assert rollout.safe_metadata["tool_call_groups"] == (grouped[0].tool_call_ids,)
    receipt = rollout.safe_metadata["codex_turn_receipt"]
    assert receipt["schema"] == "eva.codex-turn-receipt.v1"
    assert len(receipt["events"]) == 7
    assert len(receipt["tool_calls"]) == 2
    assert factory.backends[0].open_count == factory.backends[0].close_count == 1
    assert factory.backends[0].starts[0].sandbox is CodexSandbox.READ_ONLY
    assert factory.backends[0].turns[0][1].sandbox is CodexSandbox.READ_ONLY

    with pytest.raises(CodexPipelineError, match="already consumed"):
        adapter.run(request, tools)
    assert len(factory.backends) == 1


def test_actor_accepts_one_transport_frontier_split_by_runtime_capability(
    tmp_path: Path,
) -> None:
    """A mixed Codex batch retains one decision and exact serial ToolTrace rows."""

    request, _unused = _request(
        Cohort.STRONG,
        tmp_path,
        suffix="mixed-capability-transport",
    )
    workspace = FilesystemSandbox(
        tmp_path / "mixed-capability-workspace",
        DeterministicUUIDFactory("mixed-capability-workspace").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def inspect(bound, arguments):
        return {"content": bound.read_bytes(arguments["path"]).decode("utf-8")}

    def write_once(bound, arguments):
        bound.write_bytes(arguments["path"], b"verified\n", create_only=True)
        return {"written": arguments["path"]}

    registry = ToolRegistry(
        (
            ToolDefinition(
                "inspect_a",
                "Inspect one frozen input.",
                schema,
                inspect,
                parallel_safe=True,
                read_only=True,
            ),
            ToolDefinition(
                "write_once",
                "Write one output.",
                schema,
                write_once,
                parallel_safe=False,
                read_only=False,
            ),
        )
    )
    request = replace(request, available_tools=registry.public_schemas())
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=registry,
        id_factory=DeterministicUUIDFactory("mixed-capability-tools"),
    )
    adapter = CodexRolloutAdapter(
        _RuntimeFactory(_mixed_capability_transport_script),
        _actor_options,
        id_factory=DeterministicUUIDFactory("mixed-capability-adapter"),
    )

    rollout = adapter.run(request, runtime)
    trace = runtime.trace()

    assert rollout.assistant_output == "mixed transport complete"
    assert [result.name for result in trace.results] == ["inspect_a", "write_once"]
    assert [result.frontier for result in trace.results] == [0, 1]
    assert trace.frontier_count == 2
    assert trace.max_parallelism_observed == 1
    assert trace.declared_call_ids == tuple(reversed(trace.joined_call_ids))
    assert tuple(map(len, rollout.safe_metadata["tool_call_groups"])) == (2,)
    assert workspace.read_bytes("result.txt") == b"verified\n"


def test_actor_rejects_one_execution_frontier_split_across_transport_groups(
    tmp_path: Path,
) -> None:
    """The relaxed mixed-capability mapping remains fail closed on false splitting."""

    request, _unused = _request(
        Cohort.STRONG,
        tmp_path,
        suffix="falsely-split-transport",
    )
    workspace = FilesystemSandbox(
        tmp_path / "falsely-split-workspace",
        DeterministicUUIDFactory("falsely-split-workspace").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def inspect(bound, arguments):
        return {"content": bound.read_bytes(arguments["path"]).decode("utf-8")}

    registry = ToolRegistry(
        tuple(
            ToolDefinition(
                name,
                f"Inspect {name}.",
                schema,
                inspect,
                parallel_safe=True,
                read_only=True,
            )
            for name in ("inspect_a", "inspect_b")
        )
    )
    request = replace(request, available_tools=registry.public_schemas())
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=registry,
        id_factory=DeterministicUUIDFactory("falsely-split-tools"),
    )
    adapter = CodexRolloutAdapter(
        _RuntimeFactory(_falsely_split_transport_script),
        _actor_options,
        id_factory=DeterministicUUIDFactory("falsely-split-adapter"),
    )

    with pytest.raises(
        CodexPipelineError,
        match="Codex MCP transport frontier differs from ToolTrace",
    ) as captured:
        adapter.run(request, runtime)

    assert captured.value.receipt is not None
    assert runtime.trace().frontier_count == 1
    assert runtime.trace().max_parallelism_observed == 2


def test_s1_rejected_materialization_is_retained_as_semantic_terminal_evidence(
    tmp_path: Path,
) -> None:
    """An invalid signed-plan submission is model evidence, not adapter loss."""

    plan_schema = {
        "type": "object",
        "properties": {
            "episode_id": {"type": "string"},
            "contract_sha256": {"type": "string"},
            "steps": {"type": "array"},
        },
        "required": ["episode_id", "contract_sha256", "steps"],
        "additionalProperties": False,
    }
    failure_detail = {
        "stage": "S1",
        "effect": "plan_materialization",
        "gate_passed": False,
        "error": "submitted_plan_schema_rejected",
        "failure_detail": {
            "classification": "semantic_terminal",
            "submitted_arguments_blake3": blake3_hex({}),
            "required_fields": ("episode_id", "contract_sha256", "steps"),
        },
    }

    def reject_invalid_plan(_workspace, arguments):
        assert canonical_value(arguments) == {}
        return failure_detail

    definition = ToolDefinition(
        "materialize_plan",
        "Materialize one plan under the exact signed S1 contract.",
        plan_schema,
        reject_invalid_plan,
    )
    request, _unused = _request(
        Cohort.STRONG,
        tmp_path,
        suffix="s1-semantic-terminal",
        stage=Stage.S1,
    )
    request = replace(request, available_tools=(definition.public_schema(),))
    workspace = FilesystemSandbox(
        tmp_path / "s1-semantic-terminal",
        DeterministicUUIDFactory("s1-semantic-workspace").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry((definition,)),
        id_factory=DeterministicUUIDFactory("s1-semantic-tools"),
    )
    factory = _RuntimeFactory(_rejected_s1_materialization_script)
    adapter = CodexRolloutAdapter(
        factory,
        _actor_options,
        id_factory=DeterministicUUIDFactory("s1-semantic-adapter"),
    )

    before = workspace.snapshot("s1-semantic-before")
    rollout = adapter.run(request, runtime)
    after = workspace.snapshot("s1-semantic-after")

    result = runtime.trace().results[0]
    assert result.status == "completed"
    assert result.error_code is None
    assert canonical_value(result.output) == canonical_value(failure_detail)
    assert before.tree_blake3 == after.tree_blake3
    assert rollout.safe_metadata["schema"] == "eva.codex-provider-rollout-projection.v1"
    receipt = rollout.safe_metadata["codex_turn_receipt"]
    call = receipt["tool_calls"][0]
    assert call["fully_qualified_name"] == "evamed/materialize_plan"
    assert canonical_value(call["arguments"]) == {}
    assert canonical_value(
        call["output"]["result"]["structuredContent"]["tool_result"]
    ) == canonical_value(result)
    expected_offer = CodexToolOffer(
        fully_qualified_name="evamed/materialize_plan",
        description=definition.description,
        input_schema=plan_schema,
        visibility="public",
        parallel_safe=False,
    )
    assert receipt["offered_tool_schema_blake3"] == blake3_hex(
        (expected_offer.canonical_catalog_entry(),)
    )
    visible_id = rollout.safe_metadata["codex_to_pipeline_call_ids"][
        call["tool_call_id"]
    ]
    observations = tuple(
        event
        for event in rollout.policy_events
        if event.role == "tool" and event.tool_call_ids == (visible_id,)
    )
    assert len(observations) == 1
    assert canonical_value(observations[0].content) == canonical_value(
        {
            "status": "completed",
            "output": failure_detail,
            "error_code": None,
            "receipt_blake3": result.receipt_blake3,
        }
    )

    evidence_ids = DeterministicUUIDFactory("s1-semantic-evidence")
    partial = EvidenceBundle(
        bundle_id=evidence_ids.new("bundle"),
        rollout_id=request.rollout_id,
        sandbox_manifest=request.sandbox,
        model=request.model,
        policy_visible_context=request.policy_visible_context,
        context_blake3=blake3_hex(request.policy_visible_context),
        workspace_before=before,
        workspace_after=after,
        tool_trace=runtime.trace(),
        policy_events=rollout.policy_events,
        assistant_output=rollout.assistant_output,
        provider_receipt_blake3=rollout.provider_receipt_blake3,
        safe_provider_metadata=rollout.safe_metadata,
        bundle_blake3="",
    )
    evidence = replace(partial, bundle_blake3=blake3_hex(evidence_core(partial)))
    PipelineVerifier(
        ImmutableArtifactStore(tmp_path / "s1-semantic-artifacts")
    )._verify_trajectory(SimpleNamespace(evidence=evidence, cohort=Cohort.STRONG))


def test_failed_codex_turn_after_s1_contract_gate_is_semantic_evidence(
    tmp_path: Path,
) -> None:
    schema = {"type": "object", "additionalProperties": False}

    def reject_invalid_plan(_workspace, _arguments):
        raise ContractError("plan contract requires committed fields")

    definition = ToolDefinition(
        "materialize_plan",
        "Materialize an S1 plan.",
        schema,
        reject_invalid_plan,
    )
    request, _unused = _request(
        Cohort.STRONG,
        tmp_path,
        suffix="s1-failed-turn-semantic",
        stage=Stage.S1,
    )
    request = replace(request, available_tools=(definition.public_schema(),))
    workspace = FilesystemSandbox(
        tmp_path / "s1-failed-turn-semantic",
        DeterministicUUIDFactory("s1-failed-turn-workspace").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry((definition,)),
        id_factory=DeterministicUUIDFactory("s1-failed-turn-tools"),
    )
    adapter = CodexRolloutAdapter(
        _RuntimeFactory(_failed_turn_after_rejected_s1_script),
        _actor_options,
        id_factory=DeterministicUUIDFactory("s1-failed-turn-adapter"),
    )

    before = workspace.snapshot("s1-failed-turn-before")
    rollout = adapter.run(request, runtime)
    after = workspace.snapshot("s1-failed-turn-after")
    failed = runtime.trace().results[0]
    assert failed.status == "immutable_failure"
    assert failed.error_code == "ContractError"
    assert rollout.assistant_output == (
        "[trajectory terminated at a signed semantic tool gate]"
    )
    detail = rollout.safe_metadata["semantic_tool_gate_failure"]
    assert detail["classification"] == "signed_tool_gate_failure"
    assert detail["failed_call_ids"] == (failed.call_id,)
    assert detail["error_codes"] == ("ContractError",)
    assert detail["tool_trace_blake3"] == runtime.trace().trace_blake3
    assert detail["synthetic_terminal_marker"] is True
    assert rollout.safe_metadata["semantic_retry_count"] == 0

    evidence_ids = DeterministicUUIDFactory("s1-failed-turn-evidence")
    partial = EvidenceBundle(
        bundle_id=evidence_ids.new("bundle"),
        rollout_id=request.rollout_id,
        sandbox_manifest=request.sandbox,
        model=request.model,
        policy_visible_context=request.policy_visible_context,
        context_blake3=blake3_hex(request.policy_visible_context),
        workspace_before=before,
        workspace_after=after,
        tool_trace=runtime.trace(),
        policy_events=rollout.policy_events,
        assistant_output=rollout.assistant_output,
        provider_receipt_blake3=rollout.provider_receipt_blake3,
        safe_provider_metadata=rollout.safe_metadata,
        bundle_blake3="",
    )
    evidence = replace(partial, bundle_blake3=blake3_hex(evidence_core(partial)))
    PipelineVerifier(
        ImmutableArtifactStore(tmp_path / "s1-failed-turn-artifacts")
    )._verify_trajectory(SimpleNamespace(evidence=evidence, cohort=Cohort.STRONG))


def test_failed_core_resource_read_is_retained_without_domain_capability(
    tmp_path: Path,
) -> None:
    factory = _RuntimeFactory(_core_resource_then_tool_script)
    adapter = CodexRolloutAdapter(
        factory,
        _actor_options,
        id_factory=DeterministicUUIDFactory("core-resource-projection"),
    )
    request, _unused = _request(
        Cohort.WEAK, tmp_path, suffix="core-resource-projection"
    )
    workspace = FilesystemSandbox(
        tmp_path / "core-resource-workspace",
        DeterministicUUIDFactory("core-resource-workspace").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    tools = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(
            (
                ToolDefinition(
                    "inspect_a",
                    "Inspect inspect_a.",
                    schema,
                    lambda bound, arguments: {
                        "content": bound.read_bytes(arguments["path"]).decode("utf-8")
                    },
                    parallel_safe=True,
                    read_only=True,
                ),
                ToolDefinition(
                    "inspect_b",
                    "Inspect inspect_b.",
                    schema,
                    lambda _bound, _arguments: {"content": "unused"},
                    parallel_safe=True,
                    read_only=True,
                ),
            )
        ),
        id_factory=DeterministicUUIDFactory("core-resource-tools"),
    )
    rollout = adapter.run(request, tools)
    # The failed native protocol helper is model behavior, not a successful
    # resource read or an extra canonical tool offered to the policy.
    assert any('read_mcp_resource' in str(event.content) for event in rollout.policy_events)
    assert len(tools.trace().results) == 1


@pytest.mark.parametrize("unsafe_kind", ("successful_read", "nonempty_list"))
def test_core_resource_bypass_attempts_fail_closed(
    tmp_path: Path, unsafe_kind: str
) -> None:
    def unsafe_script(
        thread: str,
        turn: str,
        bridge: CodexToolExecutionBridge | None = None,
    ) -> tuple[_Notification, ...]:
        del bridge
        item = {
            "id": "unsafe-core-resource",
            "type": "mcpToolCall",
            "server": "evamed",
            "tool": "read_mcp_resource",
            "arguments": {"server": "evamed", "uri": "eva://hidden"},
            "status": "completed",
            "error": None,
        }
        if unsafe_kind == "successful_read":
            item["result"] = {
                "content": [{"type": "text", "text": "uncommitted bytes"}],
                "structuredContent": None,
                "_meta": None,
            }
        else:
            item.update(
                {
                    "server": "codex",
                    "tool": "list_mcp_resources",
                    "arguments": {},
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": '{"resources":[{"uri":"eva://hidden"}]}',
                            }
                        ],
                        "structuredContent": None,
                        "_meta": None,
                    },
                }
            )
        started = {**item, "status": "inProgress", "result": None, "error": None}
        return (
            _event(
                "turn/started",
                thread,
                turn,
                turn={"id": turn, "status": "inProgress"},
            ),
            _event("item/started", thread, turn, item=started),
            _event("item/completed", thread, turn, item=item),
            _event(
                "item/completed",
                thread,
                turn,
                item={
                    "id": "answer",
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": "done",
                },
            ),
            _event(
                "turn/completed",
                thread,
                turn,
                turn={"id": turn, "status": "completed"},
            ),
        )

    factory = _RuntimeFactory(unsafe_script)
    adapter = CodexRolloutAdapter(factory, _actor_options)
    request, tools = _request(
        Cohort.WEAK, tmp_path / unsafe_kind, suffix=unsafe_kind
    )
    with pytest.raises(
        (CodexPipelineError, CodexRuntimeError),
        match="unoffered MCP tool|uncommitted resources",
    ):
        adapter.run(request, tools)


def test_actor_projects_parallel_mcp_then_native_file_change(
    tmp_path: Path,
) -> None:
    factory = _RuntimeFactory(_hybrid_native_script)
    adapter = CodexRolloutAdapter(
        factory,
        _native_actor_options,
        id_factory=DeterministicUUIDFactory("actor-hybrid-projection"),
        enable_native_actions=True,
    )
    assert adapter.native_actions_enabled is True
    request, _unused = _request(Cohort.STRONG, tmp_path, suffix="hybrid-native")
    workspace = FilesystemSandbox(
        tmp_path / "hybrid-workspaces",
        DeterministicUUIDFactory("hybrid-workspace").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def inspect(bound, arguments):
        return {"content": bound.read_bytes(arguments["path"]).decode("utf-8")}

    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(
            tuple(
                ToolDefinition(
                    name,
                    f"Inspect {name}.",
                    schema,
                    inspect,
                    parallel_safe=True,
                    read_only=True,
                )
                for name in ("inspect_a", "inspect_b")
            )
        ),
        id_factory=DeterministicUUIDFactory("hybrid-tools"),
    )
    before = workspace.snapshot("before-hybrid")
    rollout = adapter.run(request, runtime)
    after = workspace.snapshot("after-hybrid")

    decisions = tuple(
        event for event in rollout.policy_events if event.role == "assistant" and event.tool_call_ids
    )
    assert tuple(len(event.tool_call_ids) for event in decisions) == (2, 1)
    assert len(runtime.trace().results) == 2
    assert runtime.trace().max_parallelism_observed == 2
    assert rollout.safe_metadata["tool_call_groups"] == (decisions[0].tool_call_ids,)
    assert workspace.read_bytes("native-result.txt") == b"verified native edit\n"
    native_observations = tuple(
        event
        for event in rollout.policy_events
        if event.role == "tool"
        and isinstance(event.content, Mapping)
        and "codex_native_action" in event.content
    )
    assert tuple(
        event.content["codex_native_action"]["tool_type"]
        for event in native_observations
    ) == ("fileChange",)
    assert rollout.safe_metadata["schema"] == (
        "eva.codex-provider-rollout-projection.v2-native-file-change"
    )
    binding = rollout.safe_metadata["native_file_change_effect_binding"]
    assert binding["schema"] == "eva.codex-native-file-change-effect-binding.v1"
    assert binding["declared_changed_paths"] == ("native-result.txt",)
    assert binding["actual_changed_paths"] == ("native-result.txt",)
    assert factory.backends[0].starts[0].sandbox is CodexSandbox.WORKSPACE_WRITE
    assert factory.backends[0].turns[0][1].sandbox is CodexSandbox.WORKSPACE_WRITE

    evidence_ids = DeterministicUUIDFactory("hybrid-evidence")
    partial = EvidenceBundle(
        bundle_id=evidence_ids.new("bundle"),
        rollout_id=request.rollout_id,
        sandbox_manifest=request.sandbox,
        model=request.model,
        policy_visible_context=request.policy_visible_context,
        context_blake3=blake3_hex(request.policy_visible_context),
        workspace_before=before,
        workspace_after=after,
        tool_trace=runtime.trace(),
        policy_events=rollout.policy_events,
        assistant_output=rollout.assistant_output,
        provider_receipt_blake3=rollout.provider_receipt_blake3,
        safe_provider_metadata=rollout.safe_metadata,
        bundle_blake3="",
    )
    evidence = replace(partial, bundle_blake3=blake3_hex(evidence_core(partial)))
    verifier = PipelineVerifier(ImmutableArtifactStore(tmp_path / "hybrid-artifacts"))
    verifier._verify_trajectory(SimpleNamespace(evidence=evidence, cohort=Cohort.STRONG))

    tampered_metadata = canonical_value(evidence.safe_provider_metadata)
    tampered_metadata["codex_turn_receipt"]["tool_calls"][-1]["output"]["status"] = "failed"
    tampered = replace(evidence, safe_provider_metadata=tampered_metadata)
    with pytest.raises(PipelineVerificationError, match="receipt could not be reopened"):
        verifier._verify_trajectory(
            SimpleNamespace(evidence=tampered, cohort=Cohort.STRONG)
        )


def test_actor_native_actions_are_default_off_and_retain_receipt(
    tmp_path: Path,
) -> None:
    factory = _RuntimeFactory(_hybrid_native_script)
    adapter = CodexRolloutAdapter(factory, _actor_options)
    assert adapter.native_actions_enabled is False
    request, _unused = _request(Cohort.STRONG, tmp_path, suffix="native-disabled")
    workspace = FilesystemSandbox(
        tmp_path / "disabled-workspaces",
        DeterministicUUIDFactory("disabled-workspace").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(
            tuple(
                ToolDefinition(
                    name,
                    f"Inspect {name}.",
                    schema,
                    lambda bound, arguments: {
                        "content": bound.read_bytes(arguments["path"]).decode("utf-8")
                    },
                    parallel_safe=True,
                    read_only=True,
                )
                for name in ("inspect_a", "inspect_b")
            )
        ),
        id_factory=DeterministicUUIDFactory("disabled-tools"),
    )
    with pytest.raises(CodexPipelineError, match="forbidden by this turn capability") as captured:
        adapter.run(request, runtime)
    assert captured.value.receipt is not None


def test_actor_default_requires_read_only_before_provider(tmp_path: Path) -> None:
    factory = _RuntimeFactory(_simple_script)
    adapter = CodexRolloutAdapter(factory, _native_actor_options)
    request, tools = _request(
        Cohort.STRONG, tmp_path, suffix="default-workspace-write-blocked"
    )

    with pytest.raises(CodexPipelineError, match="thread options differ"):
        adapter.run(replace(request, available_tools=()), tools)

    assert factory.backends == []


def test_actor_native_actions_require_workspace_write_before_provider(
    tmp_path: Path,
) -> None:
    factory = _RuntimeFactory(_simple_script)
    adapter = CodexRolloutAdapter(
        factory, _actor_options, enable_native_actions=True
    )
    request, tools = _request(
        Cohort.STRONG, tmp_path, suffix="native-read-only-blocked"
    )

    with pytest.raises(CodexPipelineError, match="thread options differ"):
        adapter.run(replace(request, available_tools=()), tools)

    assert factory.backends == []


@pytest.mark.parametrize(
    ("command", "action"),
    (
        (
            "cat /etc/passwd",
            {"type": "read", "command": "cat /etc/passwd", "name": "cat", "path": "/etc/passwd"},
        ),
        (
            "cat ../secret.txt",
            {"type": "read", "command": "cat ../secret.txt", "name": "cat", "path": "../secret.txt"},
        ),
        (
            "cat seed.txt | rg evidence",
            {"type": "read", "command": "cat seed.txt | rg evidence", "name": "cat", "path": "seed.txt"},
        ),
        (
            "python -c pass",
            {"type": "read", "command": "python -c pass", "name": "python", "path": "seed.txt"},
        ),
        (
            "find .",
            {"type": "unknown", "command": "find ."},
        ),
        (
            "cat $SECRET",
            {"type": "read", "command": "cat $SECRET", "name": "cat", "path": "seed.txt"},
        ),
    ),
)
def test_actor_native_command_allowlist_fails_closed_with_receipt(
    tmp_path: Path, command: str, action: Mapping[str, Any]
) -> None:
    factory = _RuntimeFactory(_native_command_script(command, action))
    adapter = CodexRolloutAdapter(
        factory, _native_actor_options, enable_native_actions=True
    )
    request, _unused = _request(
        Cohort.STRONG, tmp_path, suffix=blake3_hex(command)[:8]
    )
    workspace = FilesystemSandbox(
        tmp_path / blake3_hex(command)[:8],
        DeterministicUUIDFactory(command).new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(()),
        id_factory=DeterministicUUIDFactory(f"unsafe:{command}"),
    )
    with pytest.raises(CodexPipelineError) as captured:
        adapter.run(replace(request, available_tools=()), runtime)
    assert captured.value.receipt is not None


def test_actor_native_actions_are_stage_gated(tmp_path: Path) -> None:
    action = {"type": "read", "command": "cat seed.txt", "name": "cat", "path": "seed.txt"}
    factory = _RuntimeFactory(_native_command_script("cat seed.txt", action))
    adapter = CodexRolloutAdapter(
        factory, _native_actor_options, enable_native_actions=True
    )
    request, _unused = _request(
        Cohort.STRONG, tmp_path, suffix="stage-gated", stage=Stage.S1
    )
    workspace = FilesystemSandbox(
        tmp_path / "stage-gated",
        DeterministicUUIDFactory("stage-gated").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(()),
        id_factory=DeterministicUUIDFactory("stage-gated-tools"),
    )
    with pytest.raises(CodexPipelineError, match="unauthorized") as captured:
        adapter.run(replace(request, available_tools=()), runtime)
    assert captured.value.receipt is None
    assert factory.backends == []


@pytest.mark.parametrize("topology", ("symlink", "hardlink"))
def test_actor_native_capability_rejects_link_topology_before_provider(
    tmp_path: Path, topology: str
) -> None:
    command = "cat linked.txt"
    action = {"type": "read", "command": command, "name": "cat", "path": "linked.txt"}
    factory = _RuntimeFactory(_native_command_script(command, action))
    adapter = CodexRolloutAdapter(
        factory, _native_actor_options, enable_native_actions=True
    )
    request, _unused = _request(
        Cohort.STRONG, tmp_path, suffix=f"native-{topology}"
    )
    workspace = FilesystemSandbox(
        tmp_path / topology,
        DeterministicUUIDFactory(f"native-{topology}").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    outside = tmp_path / f"outside-{topology}.txt"
    outside.write_bytes(b"private\n")
    linked = workspace.root / "linked.txt"
    if topology == "symlink":
        linked.symlink_to(outside)
    else:
        linked.hardlink_to(outside)
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(()),
        id_factory=DeterministicUUIDFactory(f"native-{topology}-tools"),
    )

    with pytest.raises(
        CodexPipelineError, match="snapshot failed closed"
    ) as captured:
        adapter.run(replace(request, available_tools=()), runtime)
    assert captured.value.receipt is None
    assert factory.backends == []


@pytest.mark.parametrize(
    "change",
    (
        {"path": "/tmp/escape.txt", "kind": {"type": "add"}, "diff": "+escape\n"},
        {"path": "../escape.txt", "kind": {"type": "add"}, "diff": "+escape\n"},
        {
            "path": "seed.txt",
            "kind": {"type": "update", "move_path": "/tmp/moved.txt"},
            "diff": "+escape\n",
        },
        {
            "path": "seed.txt",
            "kind": {"type": "update", "move_path": "../moved.txt"},
            "diff": "+escape\n",
        },
    ),
)
def test_actor_native_file_change_paths_fail_closed_with_receipt(
    tmp_path: Path, change: Mapping[str, Any]
) -> None:
    factory = _RuntimeFactory(_native_file_change_script((change,)))
    adapter = CodexRolloutAdapter(
        factory, _native_actor_options, enable_native_actions=True
    )
    request, _unused = _request(
        Cohort.STRONG, tmp_path, suffix=blake3_hex(change)[:8]
    )
    workspace = FilesystemSandbox(
        tmp_path / blake3_hex(change)[:8],
        DeterministicUUIDFactory(f"file-change:{blake3_hex(change)}").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(()),
        id_factory=DeterministicUUIDFactory(f"file-change-tools:{blake3_hex(change)}"),
    )

    with pytest.raises(CodexPipelineError, match="workspace-relative") as captured:
        adapter.run(replace(request, available_tools=()), runtime)
    assert captured.value.receipt is not None


def test_read_only_tool_runtime_accepts_a_thirty_two_call_codex_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The widened Codex path is real execution capacity, not metadata only."""

    ids = DeterministicUUIDFactory("wide-read-only-frontier")
    workspace = FilesystemSandbox(
        tmp_path / "wide-workspaces",
        ids.new("workspace"),
        {"seed.txt": b"frozen medical evidence\n"},
    )
    def inspect(bound, arguments):
        return {
            "ordinal": arguments["ordinal"],
            "content": bound.read_bytes("seed.txt").decode("utf-8"),
        }

    snapshot_labels: list[str] = []
    original_snapshot = workspace.snapshot

    def counted_snapshot(label: str):
        snapshot_labels.append(label)
        return original_snapshot(label)

    monkeypatch.setattr(workspace, "snapshot", counted_snapshot)

    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(
            (
                ToolDefinition(
                    "inspect",
                    "Inspect one independent evidence projection.",
                    {
                        "type": "object",
                        "properties": {"ordinal": {"type": "integer"}},
                        "required": ["ordinal"],
                        "additionalProperties": False,
                    },
                    inspect,
                    parallel_safe=True,
                    read_only=True,
                ),
            )
        ),
        id_factory=ids,
        maximum_parallel_calls=64,
    )
    calls = tuple(
        ToolCall(
            call_id=ids.new("call"),
            name="inspect",
            arguments={"ordinal": ordinal},
        )
        for ordinal in range(32)
    )

    results = runtime.execute(calls)
    assert len(results) == 32
    assert all(result.status == "completed" for result in results)
    assert runtime.trace().max_parallelism_observed == 32
    assert len(snapshot_labels) == 2
    assert len({result.workspace_before_blake3 for result in results}) == 1
    assert len({result.workspace_after_blake3 for result in results}) == 1


def test_parallel_read_only_frontier_fails_as_one_group_on_workspace_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = DeterministicUUIDFactory("read-only-mutation-frontier")
    workspace = FilesystemSandbox(
        tmp_path / "mutation-workspaces",
        ids.new("workspace"),
        {"seed.txt": b"frozen medical evidence\n"},
    )
    snapshot_labels: list[str] = []
    original_snapshot = workspace.snapshot

    def counted_snapshot(label: str):
        snapshot_labels.append(label)
        return original_snapshot(label)

    monkeypatch.setattr(workspace, "snapshot", counted_snapshot)

    def mutate(bound, _arguments):
        bound.write_bytes("seed.txt", b"mutated despite read-only declaration\n")
        return {"mutated": True}

    def inspect(bound, _arguments):
        return {"content": bound.read_bytes("seed.txt").decode("utf-8")}

    schema = {"type": "object", "additionalProperties": False}
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=ToolRegistry(
            (
                ToolDefinition(
                    "mutate",
                    "Incorrectly declared read-only mutation.",
                    schema,
                    mutate,
                    parallel_safe=True,
                    read_only=True,
                ),
                ToolDefinition(
                    "inspect",
                    "Inspect frozen evidence.",
                    schema,
                    inspect,
                    parallel_safe=True,
                    read_only=True,
                ),
            )
        ),
        id_factory=ids,
        maximum_parallel_calls=64,
    )
    calls = tuple(
        ToolCall(call_id=ids.new("call"), name=name, arguments={})
        for name in ("mutate", "inspect")
    )

    results = runtime.execute(calls)
    assert len(snapshot_labels) == 2
    assert len(results) == 2
    assert all(result.status == "immutable_failure" for result in results)
    assert all(result.error_code == "read_only_workspace_mutation" for result in results)
    assert all(result.output is None for result in results)
    assert len({result.workspace_before_blake3 for result in results}) == 1
    assert len({result.workspace_after_blake3 for result in results}) == 1
    assert results[0].workspace_before_blake3 != results[0].workspace_after_blake3
    assert runtime.trace().max_parallelism_observed == 2


def test_three_cohorts_overlap_without_adapter_global_throttle(tmp_path: Path) -> None:
    state = _Concurrency(barrier=Barrier(3))
    factory = _RuntimeFactory(_simple_script, state)
    adapter = CodexRolloutAdapter(factory, _actor_options)
    triples = [
        _request(cohort, tmp_path / cohort.value, suffix="parallel")
        for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG)
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(adapter.run, request, tools) for request, tools in triples]
        outputs = [future.result(timeout=5).assistant_output for future in futures]
    assert outputs == ["done", "done", "done"]
    assert state.maximum == 3
    assert sum(len(backend.turns) for backend in factory.backends) == 3


def test_actor_accepts_one_shared_persistent_runner_for_fresh_threads(tmp_path: Path) -> None:
    backend = _Backend(_simple_script)
    service = PersistentCodexRuntimeRunner(
        lambda: CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory("persistent-actor-runtime")
        )
    )
    runner: CodexTurnRunnerPort = service
    adapter = CodexRolloutAdapter(options_factory=_actor_options, runner=runner)
    triples = []
    for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG):
        request, tools = _request(
            cohort, tmp_path / cohort.value, suffix="persistent-runner"
        )
        triples.append((replace(request, available_tools=()), tools))

    with service:
        outputs = tuple(adapter.run(request, tools).assistant_output for request, tools in triples)

    assert outputs == ("done", "done", "done")
    assert backend.open_count == backend.close_count == 1
    assert len(backend.starts) == len(backend.turns) == 3
    assert len({thread_options.role for thread_options in backend.starts}) == 3


def test_actor_mounts_exact_candidate_skills_through_codex_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill_path = tmp_path / "skills" / "medical-evidence" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_bytes = b"# Medical evidence\nUse frozen evidence through the offered tools.\n"
    skill_path.write_bytes(skill_bytes)
    skill = CodexSkill(
        skill_id="medical-evidence",
        name="medical-evidence",
        path=str(skill_path),
        content_blake3=blake3_bytes(skill_bytes),
    )
    factory = _RuntimeFactory(_simple_script)
    adapter = CodexRolloutAdapter(
        factory,
        _actor_options,
        skills_factory=lambda _request: (skill,),
    )
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    checks: list[bool] = []
    stable_read = codex_adapter._stable_actor_skill_bytes

    def counted_read(value, *, receipt=None):
        checks.append(receipt is not None)
        return stable_read(value, receipt=receipt)

    monkeypatch.setattr(codex_adapter, "_stable_actor_skill_bytes", counted_read)
    request, tools = _request(
        Cohort.STRONG, candidate_root, suffix="mounted-skill"
    )
    rollout = adapter.run(replace(request, available_tools=()), tools)

    assert rollout.assistant_output == "done"
    input_items = factory.backends[0].turns[0][0]
    assert not tuple(item for item in input_items if item.kind == "skill")
    embedded = next(
        item.text for item in input_items
        if item.kind == "text" and item.text and "eva.codex-skill-mount-sidecar.v1" in item.text
    )
    skill_document = json.loads(embedded)
    assert skill_document["skills"][0]["content"] == skill_bytes.decode()
    receipt = rollout.safe_metadata["codex_turn_receipt"]
    assert receipt["selected_skill_ids"] == (skill.skill_id,)
    assert receipt["selected_skill_catalog_blake3"] == blake3_hex(
        (skill.catalog_entry(),)
    )
    assert checks == [False, True]


def test_actor_rejects_skill_inside_candidate_workspace_before_provider(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    skill_path = candidate_root / "skill" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    payload = b"# Candidate-local skill\n"
    skill_path.write_bytes(payload)
    skill = CodexSkill(
        skill_id="candidate-local",
        name="candidate-local",
        path=str(skill_path),
        content_blake3=blake3_bytes(payload),
    )
    factory = _RuntimeFactory(_simple_script)
    adapter = CodexRolloutAdapter(
        factory, _actor_options, skills_factory=lambda _request: (skill,)
    )
    request, tools = _request(
        Cohort.STRONG, candidate_root, suffix="candidate-local-skill"
    )

    with pytest.raises(CodexPipelineError, match="inside the candidate workspace"):
        adapter.run(replace(request, available_tools=()), tools)

    assert factory.backends == []


def test_actor_reopens_skill_bytes_before_provider_execution(tmp_path: Path) -> None:
    skill_path = tmp_path / "skills" / "evidence" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    original = b"# Evidence\nUse the frozen source.\n"
    skill_path.write_bytes(original)
    skill = CodexSkill(
        skill_id="evidence",
        name="evidence",
        path=str(skill_path),
        content_blake3=blake3_bytes(original),
    )
    skill_path.write_bytes(b"# Evidence\nChanged after selection.\n")
    factory = _RuntimeFactory(_simple_script)
    adapter = CodexRolloutAdapter(
        factory,
        _actor_options,
        skills_factory=lambda _request: (skill,),
    )
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    request, tools = _request(Cohort.WEAK, candidate_root, suffix="tampered-skill")

    with pytest.raises(CodexPipelineError, match="skill bytes differ"):
        adapter.run(replace(request, available_tools=()), tools)
    assert factory.backends == []


def test_actor_rejects_hard_linked_skill_before_provider_execution(tmp_path: Path) -> None:
    original = tmp_path / "original-SKILL.md"
    original.write_bytes(b"# Evidence\nUse the frozen source.\n")
    skill_path = tmp_path / "linked-SKILL.md"
    skill_path.hardlink_to(original)
    skill = CodexSkill(
        skill_id="evidence",
        name="evidence",
        path=str(skill_path),
        content_blake3=blake3_bytes(skill_path.read_bytes()),
    )
    factory = _RuntimeFactory(_simple_script)
    adapter = CodexRolloutAdapter(
        factory,
        _actor_options,
        skills_factory=lambda _request: (skill,),
    )
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    request, tools = _request(Cohort.WEAK, candidate_root, suffix="hard-linked-skill")

    with pytest.raises(CodexPipelineError, match="skill bytes differ"):
        adapter.run(replace(request, available_tools=()), tools)
    assert factory.backends == []


def test_actor_rejects_same_content_skill_inode_swap_during_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill_path = tmp_path / "SKILL.md"
    payload = b"# Evidence\n" + (b"Use frozen evidence.\n" * 70_000)
    skill_path.write_bytes(payload)
    skill = CodexSkill(
        skill_id="evidence",
        name="evidence",
        path=str(skill_path),
        content_blake3=blake3_bytes(payload),
    )
    original_read = codex_adapter.os.read
    moved = tmp_path / "original-inode"
    swapped = False

    def replacing_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, count)
        if chunk and not swapped:
            swapped = True
            skill_path.rename(moved)
            skill_path.write_bytes(payload)
        return chunk

    monkeypatch.setattr(codex_adapter.os, "read", replacing_read)
    factory = _RuntimeFactory(_simple_script)
    adapter = CodexRolloutAdapter(
        factory,
        _actor_options,
        skills_factory=lambda _request: (skill,),
    )
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    request, tools = _request(Cohort.WEAK, candidate_root, suffix="skill-inode-swap")

    with pytest.raises(CodexPipelineError, match="skill bytes differ"):
        adapter.run(replace(request, available_tools=()), tools)
    assert swapped is True
    assert factory.backends == []


def test_actor_retains_receipt_but_rejects_skill_changed_during_turn(
    tmp_path: Path,
) -> None:
    skill_path = tmp_path / "SKILL.md"
    original = b"# Evidence\nUse the frozen source.\n"
    skill_path.write_bytes(original)
    skill = CodexSkill(
        skill_id="evidence",
        name="evidence",
        path=str(skill_path),
        content_blake3=blake3_bytes(original),
    )

    def changing_script(thread, turn, bridge):
        skill_path.write_bytes(b"# Evidence\nChanged during execution.\n")
        return _simple_script(thread, turn, bridge)

    factory = _RuntimeFactory(changing_script)
    adapter = CodexRolloutAdapter(
        factory,
        _actor_options,
        skills_factory=lambda _request: (skill,),
    )
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    request, tools = _request(
        Cohort.STRONG, candidate_root, suffix="skill-turn-change"
    )

    with pytest.raises(CodexPipelineError, match="skill bytes differ") as captured:
        adapter.run(replace(request, available_tools=()), tools)
    assert captured.value.receipt is not None
    assert len(factory.backends) == 1


def test_sync_facade_rejects_active_event_loop_before_backend_use(tmp_path: Path) -> None:
    factory = _RuntimeFactory(_simple_script)
    runner = SyncCodexTurnRunner(factory)
    options = CodexThreadOptions(
        role=CodexRole.STRONG_ACTOR,
        model="model",
        provider="provider",
        cwd=str(tmp_path),
        sandbox=CodexSandbox.WORKSPACE_WRITE,
    )

    async def misuse() -> None:
        with pytest.raises(CodexPipelineError, match="active event loop"):
            runner.run_once(options, CodexTurnInput(public_text="task"))

    asyncio.run(misuse())
    assert factory.backends == []


@pytest.mark.parametrize("budget", [None, {"kind": "http_error", "count": 1, "limit": 1, "requests": 1},
    {"kind": "max_requests", "count": 3, "limit": 4, "generated_tokens": 64}])
def test_budget_option_does_not_admit_unknown_or_unexhausted_failure(tmp_path: Path, budget) -> None:
    adapter = CodexRolloutAdapter(_RuntimeFactory(_failed_script), _actor_options,
        controlled_budget_terminal=lambda: budget)
    request, tools = _request(Cohort.WEAK, tmp_path / "budget-failed", suffix="budget-failed")
    with pytest.raises(CodexPipelineError):
        adapter.run(request, tools)


def test_actor_fails_closed_on_terminal_status_and_schema_drift(tmp_path: Path) -> None:
    failed_factory = _RuntimeFactory(_failed_script)
    adapter = CodexRolloutAdapter(failed_factory, _actor_options)
    request, tools = _request(Cohort.WEAK, tmp_path / "failed", suffix="failed")
    with pytest.raises(CodexPipelineError, match="status is not completed") as caught:
        adapter.run(request, tools)
    assert caught.value.receipt is not None
    assert caught.value.receipt.status == "failed"
    assert len(failed_factory.backends) == 1

    fresh_request, fresh_tools = _request(
        Cohort.MIDDLE, tmp_path / "schema", suffix="schema-drift"
    )

    def drifted_options(
        value: RolloutRequest,
        cwd: str,
        offers: tuple[CodexToolOffer, ...],
        bridge: CodexToolExecutionBridge,
    ) -> CodexThreadOptions:
        first = offers[0]
        altered = CodexToolOffer(
            fully_qualified_name=first.fully_qualified_name,
            description=first.description + " altered",
            input_schema=first.input_schema,
        )
        return _actor_options(value, cwd, (altered, *offers[1:]), bridge)

    unused_factory = _RuntimeFactory(_simple_script)
    drifted = CodexRolloutAdapter(unused_factory, drifted_options)
    with pytest.raises(CodexPipelineError, match="options differ"):
        drifted.run(fresh_request, fresh_tools)
    assert unused_factory.backends == []


def _trajectory_event(ids: DeterministicUUIDFactory, role: str, content: Any) -> TrajectoryEvent:
    core = {
        "event_id": ids.new("event"),
        "role": role,
        "content": content,
        "tool_call_ids": (),
    }
    return TrajectoryEvent(**core, event_blake3=blake3_hex(core))


def _evidence(tmp_path: Path) -> EvidenceBundle:
    ids = DeterministicUUIDFactory("judge-evidence")
    manifest = _manifest("judge-manifest")
    sandbox = FilesystemSandbox(
        tmp_path / "snapshot", ids.new("workspace"), {"seed.txt": b"medical evidence\n"}
    )
    before = sandbox.snapshot("before")
    after = sandbox.snapshot("after")
    trace = _empty_trace()
    events = (
        _trajectory_event(ids, "system", "system"),
        _trajectory_event(ids, "user", {"question": "fixture"}),
        _trajectory_event(ids, "assistant", "answer"),
    )
    partial = EvidenceBundle(
        bundle_id=ids.new("bundle"),
        rollout_id=ids.new("rollout"),
        sandbox_manifest=manifest,
        model=ModelTarget(Cohort.STRONG, "actor", "fake-provider"),
        policy_visible_context={"question": "fixture"},
        context_blake3=blake3_hex({"question": "fixture"}),
        workspace_before=before,
        workspace_after=after,
        tool_trace=trace,
        policy_events=events,
        assistant_output="answer",
        provider_receipt_blake3=blake3_hex("provider"),
        safe_provider_metadata={"fixture": True},
        bundle_blake3="",
    )
    return EvidenceBundle(
        **{key: getattr(partial, key) for key in partial.__dataclass_fields__ if key != "bundle_blake3"},
        bundle_blake3=blake3_hex(evidence_core(partial)),
    )


def _judge_script(
    evidence: EvidenceBundle,
    rubric,
    *,
    cite: str = "workspace:after:seed.txt",
    path: str = "seed.txt",
):
    selected = next(row for row in evidence.workspace_after.files if row.path == path)
    content = selected.content.decode("utf-8")
    args = {"snapshot": "after", "path": path, "offset": 0, "max_bytes": 65536}
    observation = {
        "status": "completed",
        "output": {
            "snapshot": "after",
            "tree_blake3": evidence.workspace_after.tree_blake3,
            "path": path,
            "content_blake3": selected.content_blake3,
            "total_bytes": selected.byte_count,
            "offset": 0,
            "returned_bytes": selected.byte_count,
            "truncated": False,
            "encoding": "utf-8",
            "content": content,
        },
        "error_code": None,
        "inspected_evidence_refs": [f"workspace:after:{path}"],
        "content_inspection": True,
        "evidence_bundle_blake3": evidence.bundle_blake3,
    }
    rows = []
    hard_passed = True
    for item in rubric.items:
        document = canonical_value(item)
        maximum = max(level["score_bps"] for level in document["partial_credit"]["levels"])
        gate = document.get("hard_gate")
        if isinstance(gate, dict) and maximum < gate["minimum_score_bps"]:
            hard_passed = False
        rows.append(
            {
                "item_id": document["item_id"],
                "score": maximum / 10_000,
                "evidence_refs": [cite],
                "rationale": "Inspected committed workspace content.",
            }
        )
    final = json.dumps(
        {"item_scores": rows, "hard_gates_passed": hard_passed, "summary": "verified"}
    )

    def script(
        thread: str,
        turn: str,
        bridge: CodexToolExecutionBridge | None = None,
    ) -> tuple[_Notification, ...]:
        del bridge
        call = {
            "id": "judge-read",
            "type": "mcpToolCall",
            "server": "evamed-judge",
            "tool": "workspace_read",
            "arguments": args,
            "status": "inProgress",
        }
        completed = {
            **call,
            "status": "completed",
            "result": {
                "content": [{"type": "text", "text": "committed observation"}],
                "structuredContent": observation,
                "isError": False,
            },
        }
        return (
            _event("turn/started", thread, turn, turn={"id": turn, "status": "inProgress"}),
            _event("item/started", thread, turn, item=call),
            _event("item/completed", thread, turn, item=completed),
            _event(
                "item/completed",
                thread,
                turn,
                item={"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": final},
            ),
            _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"}),
        )

    return script


def _judge_options(request: JudgeRequest, offers: tuple[CodexToolOffer, ...]):
    return CodexThreadOptions(
        role=CodexRole.JUDGE,
        model=request.judge_model_id,
        provider="anthropic",
        cwd="/tmp/eva-codex-judge",
        sandbox=CodexSandbox.READ_ONLY,
        offered_tools=offers,
    )


def test_opus5_codex_judge_replays_read_and_maps_exact_compiled_rubric(tmp_path: Path) -> None:
    rubric = _rubric()
    evidence = _evidence(tmp_path)
    factory = _RuntimeFactory(_judge_script(evidence, rubric))
    judge = CodexOpus5AgentJudge(
        factory,
        _judge_options,
        model_id="claude-opus-5",
        id_factory=DeterministicUUIDFactory("codex-judge"),
    )
    request = JudgeRequest(
        judgment_id=DeterministicUUIDFactory("judgment").new("judgment"),
        judge_model_id="claude-opus-5",
        policy_visible_context=evidence.policy_visible_context,
        workspace_evidence=evidence,
        judge_only_reference={"reference_answer": "private fixture"},
    )
    assessment = judge.judge(request, rubric)

    assert assessment.rubric_digest == rubric.digest
    assert [row.item_id for row in assessment.item_scores] == [
        canonical_value(item)["item_id"] for item in rubric.items
    ]
    assert assessment.agent_trace.content_inspection_count == 1
    assert assessment.agent_trace.inspected_evidence_refs == ("workspace:after:seed.txt",)
    assert assessment.agent_trace.provider_turn_count == 2
    assert assessment.agent_trace.retry_count == 0
    assert assessment.agent_trace.policy_events[-1].content.startswith("{")
    assert judge.receipt_for(request.judgment_id).schema == "eva.codex-turn-receipt.v1"


def test_codex_judge_native_action_fails_closed_and_retains_receipt(
    tmp_path: Path,
) -> None:
    rubric = _rubric()
    evidence = _evidence(tmp_path)
    factory = _RuntimeFactory(_judge_native_script)
    judge = CodexOpus5AgentJudge(
        factory,
        _judge_options,
        model_id="claude-opus-5",
        id_factory=DeterministicUUIDFactory("codex-judge-native-rejected"),
    )
    request = JudgeRequest(
        judgment_id=DeterministicUUIDFactory("native-judgment").new("judgment"),
        judge_model_id="claude-opus-5",
        policy_visible_context=evidence.policy_visible_context,
        workspace_evidence=evidence,
        judge_only_reference={"reference_answer": "private fixture"},
    )

    with pytest.raises(
        CodexPipelineError, match="forbidden by this turn capability"
    ) as captured:
        judge.judge(request, rubric)
    assert captured.value.receipt is not None
    assert captured.value.receipt.tool_calls[0].tool_type == "commandExecution"


def test_judge_accepts_one_shared_persistent_runner_for_fresh_threads(tmp_path: Path) -> None:
    rubric = _rubric()
    evidence = _evidence(tmp_path)
    backend = _Backend(_judge_script(evidence, rubric))
    service = PersistentCodexRuntimeRunner(
        lambda: CodexRuntime(
            backend, id_factory=DeterministicUUIDFactory("persistent-judge-runtime")
        )
    )
    runner: CodexTurnRunnerPort = service
    judge = CodexOpus5AgentJudge(
        options_factory=_judge_options,
        model_id="claude-opus-5",
        runner=runner,
        id_factory=DeterministicUUIDFactory("persistent-codex-judge"),
    )

    with service:
        for suffix in ("one", "two"):
            request = JudgeRequest(
                judgment_id=DeterministicUUIDFactory(suffix).new("judgment"),
                judge_model_id="claude-opus-5",
                policy_visible_context=evidence.policy_visible_context,
                workspace_evidence=evidence,
                judge_only_reference={"reference_answer": "private fixture"},
            )
            assert judge.judge(request, rubric).rubric_digest == rubric.digest

    assert backend.open_count == backend.close_count == 1
    assert len(backend.starts) == len(backend.turns) == 2
    assert all(options.role is CodexRole.JUDGE for options in backend.starts)


def test_codex_judge_rejects_uninspected_citation_and_never_retries(tmp_path: Path) -> None:
    rubric = _rubric()
    evidence = _evidence(tmp_path)
    factory = _RuntimeFactory(
        _judge_script(evidence, rubric, cite="workspace:after:not-inspected.txt")
    )
    judge = CodexOpus5AgentJudge(
        factory, _judge_options, model_id="claude-opus-5"
    )
    request = JudgeRequest(
        judgment_id=DeterministicUUIDFactory("bad-judgment").new("judgment"),
        judge_model_id="claude-opus-5",
        policy_visible_context=evidence.policy_visible_context,
        workspace_evidence=evidence,
        judge_only_reference=None,
    )
    with pytest.raises(CodexPipelineError, match="did not inspect"):
        judge.judge(request, rubric)
    assert sum(len(backend.turns) for backend in factory.backends) == 1
    with pytest.raises(CodexPipelineError, match="already consumed"):
        judge.judge(request, rubric)
    assert sum(len(backend.turns) for backend in factory.backends) == 1


class _PipelineJudgeCompletions:
    def __init__(self, rubric) -> None:
        self._rubric = rubric
        self.calls = 0
        self._lock = Lock()

    def create(self, **request):
        with self._lock:
            self.calls += 1
        if request["messages"][-1]["role"] != "tool":
            return {
                "choices": [{"message": {
                    "content": "Inspect the committed source.",
                    "tool_calls": [{
                        "id": "judge-read",
                        "function": {
                            "name": "workspace_read",
                            "arguments": json.dumps({
                                "snapshot": "after",
                                "path": "seed.txt",
                                "offset": 0,
                                "max_bytes": 65536,
                            }),
                        },
                    }],
                }}]
            }
        rows = []
        hard_passed = True
        for raw in self._rubric.items:
            item = canonical_value(raw)
            score_bps = max(level["score_bps"] for level in item["partial_credit"]["levels"])
            gate = item.get("hard_gate")
            if isinstance(gate, dict) and score_bps < gate["minimum_score_bps"]:
                hard_passed = False
            rows.append({
                "item_id": item["item_id"],
                "score": score_bps / 10_000,
                "evidence_refs": ["workspace:after:seed.txt"],
                "rationale": "Read the committed source content.",
            })
        return {"choices": [{"message": {"content": json.dumps({
            "item_scores": rows,
            "hard_gates_passed": hard_passed,
            "summary": "Workspace evidence verified.",
        })}}]}


def test_full_pipeline_verifier_accepts_same_execution_trace_and_one_mutation(
    tmp_path: Path,
) -> None:
    rubric = _rubric()
    ids = DeterministicUUIDFactory("codex-full-pipeline")
    read_barrier = Barrier(6)
    mutation_counts: dict[str, int] = {}
    mutation_lock = Lock()
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def inspect(workspace, arguments):
        read_barrier.wait(timeout=3)
        return {"content": workspace.read_bytes(arguments["path"]).decode("utf-8")}

    def write_once(workspace, arguments):
        workspace.write_bytes(arguments["path"], b"one committed mutation\n", create_only=True)
        with mutation_lock:
            mutation_counts[workspace.sandbox_id] = mutation_counts.get(workspace.sandbox_id, 0) + 1
        return {"path": arguments["path"], "written": True}

    registry = ToolRegistry((
        ToolDefinition(
            "inspect_a", "Inspect source A.", schema, inspect,
            parallel_safe=True, read_only=True,
        ),
        ToolDefinition(
            "inspect_b", "Inspect source B.", schema, inspect,
            parallel_safe=True, read_only=True,
        ),
        ToolDefinition("write_once", "Write one result.", schema, write_once),
    ))
    factory = _RuntimeFactory(_pipeline_tool_script)
    provider = CodexRolloutAdapter(
        factory, _actor_options, id_factory=DeterministicUUIDFactory("codex-full-actor")
    )
    judge_client = _PipelineJudgeCompletions(rubric)
    artifacts = ImmutableArtifactStore(tmp_path / "artifacts")
    pipeline = VerifiableDataPipeline(
        workspace_root=tmp_path / "workspaces",
        artifact_store=artifacts,
        tool_registry=registry,
        rollout_provider=provider,
        judge=OpenAIStyleOpus5Judge(
            SimpleNamespace(chat=SimpleNamespace(completions=judge_client)),
            model_id="claude-opus-5",
            id_factory=ids,
        ),
        rewarder=WeightedRubricRewarder(),
        id_factory=ids,
        maximum_parallel_models=3,
        maximum_parallel_tools=8,
    )
    episode = BenchmarkEpisode(
        episode_id="codex-full-e2e",
        source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
        domain="medxpertqa",
        stage=Stage.E2E,
        instruction="Use the bound tools and leave verifiable workspace evidence.",
        policy_context={"question": "fixture"},
        initial_files={"seed.txt": b"medical evidence\n"},
        judge_only_reference={"reference_answer": "private"},
    )
    targets = tuple(
        ModelTarget(cohort, f"model-{cohort.value}", "fake-provider")
        for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG)
    )

    result = pipeline.run(episode=episode, rubric=rubric, targets=targets)
    report = PipelineVerifier(artifacts).verify_or_raise(result=result, rubric=rubric)

    assert report.valid is True
    assert len(factory.backends) == 3
    assert sum(len(backend.turns) for backend in factory.backends) == 3
    assert len(mutation_counts) == 3
    assert set(mutation_counts.values()) == {1}
    for evaluation in result.evaluations:
        trace = evaluation.evidence.tool_trace
        assert len(trace.results) == 3
        assert trace.retry_count == 0
        assert trace.max_parallelism_observed == 2
        assert any(len(event.tool_call_ids) == 2 for event in evaluation.evidence.policy_events)
        assert evaluation.evidence.workspace_before.tree_blake3 != evaluation.evidence.workspace_after.tree_blake3
        assert sum(row.path == "result.txt" for row in evaluation.evidence.workspace_after.files) == 1


@pytest.mark.parametrize("controlled_budget", [False, True, "empty_final"])
def test_full_pipeline_scores_and_verifies_failed_signed_tool_gate(
    tmp_path: Path, controlled_budget: bool,
) -> None:
    rubric = _rubric(Stage.S1)
    ids = DeterministicUUIDFactory("codex-signed-gate-full-pipeline")
    schema = {"type": "object", "additionalProperties": False}

    def reject_invalid_plan(_workspace, _arguments):
        raise ContractError("plan contract requires committed fields")

    registry = ToolRegistry((
        ToolDefinition(
            "materialize_plan",
            "Materialize an S1 plan.",
            schema,
            reject_invalid_plan,
        ),
    ))
    def script(thread, turn, bridge):
        events = _failed_turn_after_rejected_s1_script(thread, turn, bridge)
        if controlled_budget == "empty_final":
            events = tuple(replace(event, payload={**event.payload,
                "turn": {"id": turn, "status": "completed"}})
                if event.method == "turn/completed" else event for event in events)
        return events
    provider = CodexRolloutAdapter(
        _RuntimeFactory(script),
        _actor_options,
        id_factory=DeterministicUUIDFactory("codex-signed-gate-actor"),
        controlled_budget_terminal=(lambda: {"kind": "total_output_budget", "count": 8192,
            "limit": 8192, "requests": 6}) if controlled_budget is True else None,
        allow_empty_final_response=controlled_budget == "empty_final",
    )
    artifacts = ImmutableArtifactStore(tmp_path / "signed-gate-artifacts")
    pipeline = VerifiableDataPipeline(
        workspace_root=tmp_path / "signed-gate-workspaces",
        artifact_store=artifacts,
        tool_registry=registry,
        rollout_provider=provider,
        judge=OpenAIStyleOpus5Judge(
            SimpleNamespace(
                chat=SimpleNamespace(completions=_PipelineJudgeCompletions(rubric))
            ),
            model_id="claude-opus-5",
            id_factory=ids,
        ),
        rewarder=WeightedRubricRewarder(),
        id_factory=ids,
    )
    episode = BenchmarkEpisode(
        episode_id="codex-signed-gate-s1",
        source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
        domain="medxpertqa",
        stage=Stage.S1,
        instruction="Materialize a valid signed plan.",
        policy_context={"question": "fixture"},
        initial_files={"seed.txt": b"medical evidence\n"},
        judge_only_reference={"reference_answer": "private"},
    )
    targets = tuple(
        ModelTarget(cohort, f"model-{cohort.value}", "fake-provider")
        for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG)
    )

    result = pipeline.run(episode=episode, rubric=rubric, targets=targets)
    report = PipelineVerifier(artifacts).verify_or_raise(result=result, rubric=rubric)

    assert report.valid is True
    assert len(result.evaluations) == 3
    if controlled_budget == "empty_final":
        for evaluation in result.evaluations:
            metadata = evaluation.evidence.safe_provider_metadata
            assert metadata["workspace_terminal"]["sft_eligible"] is False
            assert metadata["codex_turn_receipt"]["status"] == "completed"
            assert "without a final text answer" in evaluation.evidence.assistant_output
        return
    if controlled_budget:
        for evaluation in result.evaluations:
            metadata = evaluation.evidence.safe_provider_metadata
            assert metadata["controlled_budget_termination"]["sft_eligible"] is False
            assert metadata["controlled_budget_termination"]["reward_requires_workspace_agent_judge"] is True
            assert metadata["codex_turn_receipt"]["status"] == "failed"
            assert "not a model answer" in evaluation.evidence.assistant_output
        return
    assert all(
        evaluation.evidence.safe_provider_metadata[
            "semantic_tool_gate_failure"
        ]["classification"] == "signed_tool_gate_failure"
        for evaluation in result.evaluations
    )
    assert all(
        evaluation.evidence.tool_trace.results[0].error_code == "ContractError"
        for evaluation in result.evaluations
    )


def test_pipeline_retains_safe_codex_failure_receipt_before_quarantine(
    tmp_path: Path,
) -> None:
    rubric = _rubric()
    ids = DeterministicUUIDFactory("codex-safe-failure-evidence")
    artifacts_root = tmp_path / "safe-failure-artifacts"
    artifacts = ImmutableArtifactStore(artifacts_root)
    provider = CodexRolloutAdapter(
        _RuntimeFactory(_failed_script),
        _actor_options,
        id_factory=DeterministicUUIDFactory("codex-safe-failure-actor"),
    )
    pipeline = VerifiableDataPipeline(
        workspace_root=tmp_path / "safe-failure-workspaces",
        artifact_store=artifacts,
        tool_registry=ToolRegistry(()),
        rollout_provider=provider,
        judge=SimpleNamespace(judge=lambda *_args, **_kwargs: None),
        rewarder=WeightedRubricRewarder(),
        id_factory=ids,
    )
    episode = BenchmarkEpisode(
        episode_id="codex-safe-failure-e2e",
        source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
        domain="medxpertqa",
        stage=Stage.E2E,
        instruction="Use available evidence.",
        policy_context={"question": "fixture"},
        initial_files={"seed.txt": b"medical evidence\n"},
        judge_only_reference={"reference_answer": "private"},
    )
    targets = tuple(
        ModelTarget(cohort, f"model-{cohort.value}", "fake-provider")
        for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG)
    )

    with pytest.raises(InfrastructureQuarantineError):
        pipeline.run(episode=episode, rubric=rubric, targets=targets)

    retained = sorted(artifacts_root.glob("*/**/provider-failure-evidence.json"))
    assert len(retained) == 3
    for path in retained:
        document = json.loads(path.read_text("utf-8"))
        assert document["schema"] == "eva.provider-failure-evidence.v2"
        assert document["failure_type"] == "CodexPipelineError"
        assert document["failure_category"] == "compatibility_contract"
        assert document["failure_message"] == "Codex turn status is not completed"
        assert document["failure_provenance"] == "eva_agent.codex_pipeline"
        assert document["raw_exception_message_recorded"] is False
        assert document["retry_count"] == 0
        assert document["provider_turn_receipt"]["status"] == "failed"
        assert document["provider_turn_receipt"]["input_payload_recorded"] is False
        core = {
            key: value
            for key, value in document.items()
            if key != "failure_evidence_blake3"
        }
        assert document["failure_evidence_blake3"] == blake3_hex(core)


def test_codex_failure_retention_drops_interpolated_secret_details() -> None:
    error = CodexPipelineError(
        "Codex turn status is not completed: sk-secret-provider-detail"
    )

    assert error.safe_failure_category == "compatibility_contract"
    assert error.safe_failure_message == "Codex turn status is not completed"
    assert error.safe_failure_provenance == "eva_agent.codex_pipeline"
    assert "sk-secret" not in error.safe_failure_message


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        ({}, 900),
        ({"budgets": {"wall_time_seconds": 1200}}, 900),
        ({"s1_plan_contract": {"budgets": {"wall_time_seconds": 37}}}, 37),
        (
            {
                "episode_context": {
                    "execution_binding": {
                        "s1_plan_contract": {
                            "budgets": {"wall_time_seconds": 41}
                        }
                    }
                }
            },
            41,
        ),
        (
            {
                "policy_visible_context": {
                    "episode_context": {
                        "execution_binding": {
                            "s1_plan_contract": {
                                "budgets": {"wall_time_seconds": 43}
                            }
                        }
                    }
                },
                "workspace_evidence": {"wall_time_seconds": 1},
            },
            43,
        ),
        (
            {
                "s1_plan_contract": {"budgets": {"wall_time_seconds": 37}},
                "evidence": {"wall_time_seconds": 1},
            },
            37,
        ),
        (
            {
                "episode_context": {
                    "s1_plan_contract": {"budgets": {"wall_time_seconds": 1}}
                }
            },
            900,
        ),
        ({"evidence": [{"wall_time_seconds": 1}]}, 900),
    ),
)
def test_codex_turn_wall_time_is_policy_bound_and_capped(context, expected) -> None:
    assert codex_adapter._policy_wall_time(context) == expected


def test_full_pipeline_verifier_accepts_hybrid_native_sidecars(
    tmp_path: Path,
) -> None:
    rubric = _rubric()
    ids = DeterministicUUIDFactory("codex-hybrid-full-pipeline")
    read_barrier = Barrier(6)
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def inspect(workspace, arguments):
        read_barrier.wait(timeout=3)
        return {"content": workspace.read_bytes(arguments["path"]).decode("utf-8")}

    registry = ToolRegistry(
        tuple(
            ToolDefinition(
                name,
                f"Inspect {name}.",
                schema,
                inspect,
                parallel_safe=True,
                read_only=True,
            )
            for name in ("inspect_a", "inspect_b")
        )
    )
    provider = CodexRolloutAdapter(
        _RuntimeFactory(_hybrid_native_script),
        _native_actor_options,
        id_factory=DeterministicUUIDFactory("codex-hybrid-full-actor"),
        enable_native_actions=True,
    )
    artifacts = ImmutableArtifactStore(tmp_path / "hybrid-full-artifacts")
    pipeline = VerifiableDataPipeline(
        workspace_root=tmp_path / "hybrid-full-workspaces",
        artifact_store=artifacts,
        tool_registry=registry,
        rollout_provider=provider,
        judge=OpenAIStyleOpus5Judge(
            SimpleNamespace(
                chat=SimpleNamespace(completions=_PipelineJudgeCompletions(rubric))
            ),
            model_id="claude-opus-5",
            id_factory=ids,
        ),
        rewarder=WeightedRubricRewarder(),
        id_factory=ids,
        maximum_parallel_models=3,
        maximum_parallel_tools=8,
    )
    episode = BenchmarkEpisode(
        episode_id="codex-hybrid-full-e2e",
        source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
        domain="medxpertqa",
        stage=Stage.E2E,
        instruction="Use the bound evidence and leave a verified result.",
        policy_context={"question": "fixture"},
        initial_files={"seed.txt": b"medical evidence\n"},
        judge_only_reference={"reference_answer": "private"},
    )
    targets = tuple(
        ModelTarget(cohort, f"model-{cohort.value}", "fake-provider")
        for cohort in (Cohort.WEAK, Cohort.MIDDLE, Cohort.STRONG)
    )

    result = pipeline.run(episode=episode, rubric=rubric, targets=targets)
    report = PipelineVerifier(artifacts).verify_or_raise(result=result, rubric=rubric)

    assert report.valid is True
    for evaluation in result.evaluations:
        assert len(evaluation.evidence.tool_trace.results) == 2
        assert evaluation.evidence.tool_trace.max_parallelism_observed == 2
        assert any(
            isinstance(event.content, Mapping)
            and "codex_native_action" in event.content
            for event in evaluation.evidence.policy_events
        )
        assert {row.path for row in evaluation.evidence.workspace_after.files} == {
            "seed.txt",
            "native-result.txt",
        }
