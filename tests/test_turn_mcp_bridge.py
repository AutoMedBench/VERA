from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
from threading import Barrier, Lock
import time
from typing import Any, AsyncIterator, Mapping, Sequence

import pytest

from eva_agent.codex_pipeline import (
    CodexOpus5AgentJudge,
    CodexRolloutAdapter,
    DEFAULT_ACTOR_SYSTEM_INSTRUCTION,
    TurnMCPBridgeFactory,
    TurnMCPError,
)
from eva_agent.codex_runtime import (
    BackendInputItem,
    BackendTurnOptions,
    CodexRole,
    CodexRuntime,
    CodexSandbox,
    CodexThreadOptions,
    CodexToolOffer,
)
from eva_agent.pipeline import (
    BenchmarkEpisode,
    BenchmarkSource,
    Cohort,
    DeterministicUUIDFactory,
    EvidenceBundle,
    FilesystemSandbox,
    JudgeRequest,
    ModelTarget,
    ParallelToolRuntime,
    RolloutRequest,
    SandboxManifest,
    Stage,
    ToolDefinition,
    ToolRegistry,
    ToolTrace,
    TrajectoryEvent,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from eva_agent.pipeline.runner import evidence_core
from eva_agent.rubrics import load_and_compile_registry


ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / "src/eva_agent/codex_pipeline/turn_mcp_proxy.py"


@dataclass
class _Notification:
    method: str
    payload: Mapping[str, Any]


class _Turn:
    def __init__(self, turn_id: str, events: Sequence[_Notification]) -> None:
        self.id = turn_id
        self._events = tuple(events)

    async def stream(self) -> AsyncIterator[_Notification]:
        for event in self._events:
            await asyncio.sleep(0)
            yield event


def _event(method: str, thread: str, turn_id: str, **extra: Any) -> _Notification:
    return _Notification(method, {"threadId": thread, "turnId": turn_id, **extra})


def _read_json_line(stream: Any) -> Mapping[str, Any]:
    line = stream.readline()
    if not line:
        raise AssertionError("turn MCP proxy ended before a JSON-RPC response")
    value = json.loads(line)
    assert isinstance(value, dict)
    return value


def _exercise_proxy(
    options: CodexThreadOptions,
    groups: tuple[tuple[tuple[str, Mapping[str, Any]], ...], ...],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    config = canonical_value(options.config)
    assert isinstance(config, dict)
    servers = config["mcp_servers"]
    assert isinstance(servers, dict) and len(servers) == 1
    server = next(iter(servers.values()))
    assert isinstance(server, dict)
    names = [offer.name for offer in options.offered_tools]
    assert server["enabled_tools"] == names
    assert server["omit_tools_from"] == ["deferred", "code_mode"]
    assert server["supports_parallel_tool_calls"] is True
    assert server["tools"] == {
        name: {"approval_mode": "approve"} for name in names
    }
    assert "default_tools_approval_mode" not in server
    environment = os.environ.copy()
    environment.update(server["env"])
    environment["EVA_MCP_TEST_AMBIENT_SECRET"] = "must-not-cross-exec-boundary"
    process = subprocess.Popen(
        [server["command"], *server["args"]],
        cwd=server["cwd"],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdin is not None and process.stdout is not None

    def send(value: Mapping[str, Any]) -> None:
        process.stdin.write(json.dumps(value, sort_keys=True) + "\n")
        process.stdin.flush()

    send(
        {
            "jsonrpc": "2.0",
            "id": "initialize",
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )
    initialized = _read_json_line(process.stdout)
    assert initialized["id"] == "initialize"
    process_environment = Path(f"/proc/{process.pid}/environ")
    if process_environment.is_file():
        names = {
            entry.split(b"=", 1)[0]
            for entry in process_environment.read_bytes().split(b"\0")
            if entry
        }
        assert names == {
            b"EVA_TURN_MCP_SOCKET",
            b"EVA_TURN_MCP_NONCE",
            b"EVA_TURN_MCP_MAXIMUM",
            b"LANG",
            b"LC_ALL",
            b"PATH",
            b"TZ",
        }
    send({"jsonrpc": "2.0", "id": "catalog", "method": "tools/list", "params": {}})
    catalog = _read_json_line(process.stdout)
    assert catalog["id"] == "catalog"
    listed = tuple(catalog["result"]["tools"])

    observed_groups: list[tuple[Mapping[str, Any], ...]] = []
    ordinal = 0
    for group in groups:
        expected: dict[str, tuple[str, Mapping[str, Any]]] = {}
        for name, arguments in group:
            request_id = f"call-{ordinal}"
            ordinal += 1
            expected[request_id] = (name, arguments)
            # All requests in a group cross stdin before any result is read,
            # exercising the proxy and broker's actual concurrent frontier.
            process.stdin.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "tools/call",
                        "params": {"name": name, "arguments": arguments},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        process.stdin.flush()
        by_id = {
            response["id"]: response
            for response in (_read_json_line(process.stdout) for _ in group)
        }
        assert set(by_id) == set(expected)
        observed_groups.append(tuple(by_id[key]["result"] for key in expected))

    process.stdin.close()
    return_code = process.wait(timeout=10)
    error = "" if process.stderr is None else process.stderr.read()
    assert return_code == 0, error
    return listed, tuple(observed_groups)


class _MCPThread:
    def __init__(self, thread_id: str, backend: "_MCPBackend", options: CodexThreadOptions) -> None:
        self.id = thread_id
        self._backend = backend
        self._options = options

    async def turn(
        self, items: Sequence[BackendInputItem], options: BackendTurnOptions
    ) -> _Turn:
        del items, options
        listed, outcomes = await asyncio.to_thread(
            _exercise_proxy, self._options, self._backend.groups
        )
        self._backend.catalogs.append(listed)
        turn_id = f"turn-{len(self._backend.catalogs)}"
        events: list[_Notification] = [
            _event(
                "turn/started",
                self.id,
                turn_id,
                turn={"id": turn_id, "status": "inProgress"},
            )
        ]
        ordinal = 0
        for group, results in zip(self._backend.groups, outcomes, strict=True):
            pending = []
            for (name, arguments), result in zip(group, results, strict=True):
                item = {
                    "id": f"tool-{ordinal}",
                    "type": "mcpToolCall",
                    "server": self._backend.server,
                    "tool": name,
                    "arguments": arguments,
                    "status": "inProgress",
                }
                ordinal += 1
                pending.append((item, result))
                events.append(_event("item/started", self.id, turn_id, item=item))
            for item, result in reversed(pending):
                events.append(
                    _event(
                        "item/completed",
                        self.id,
                        turn_id,
                        item={**item, "status": "completed", "result": result},
                    )
                )
        events.extend(
            (
                _event(
                    "item/completed",
                    self.id,
                    turn_id,
                    item={
                        "id": "answer",
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": self._backend.final_text,
                    },
                ),
                _event(
                    "turn/completed",
                    self.id,
                    turn_id,
                    turn={"id": turn_id, "status": "completed"},
                ),
            )
        )
        return _Turn(turn_id, events)


class _MCPBackend:
    sdk_version = "fake-codex-sdk-with-real-mcp"
    server_version = "fake-codex-app-server"

    def __init__(
        self,
        *,
        server: str,
        groups: tuple[tuple[tuple[str, Mapping[str, Any]], ...], ...],
        final_text: str,
    ) -> None:
        self.server = server
        self.groups = groups
        self.final_text = final_text
        self.catalogs: list[tuple[Mapping[str, Any], ...]] = []
        self.starts: list[CodexThreadOptions] = []
        self.open_count = 0
        self.close_count = 0

    async def open(self) -> None:
        self.open_count += 1

    async def close(self) -> None:
        self.close_count += 1

    async def start_thread(self, options: CodexThreadOptions) -> _MCPThread:
        self.starts.append(options)
        return _MCPThread(f"thread-{len(self.starts)}", self, options)

    async def resume_thread(self, thread_id: str, options: CodexThreadOptions) -> _MCPThread:
        raise AssertionError((thread_id, options, "resume is forbidden"))


class _RuntimeFactory:
    def __init__(self, backend: _MCPBackend, seed: str) -> None:
        self.backend = backend
        self.seed = seed

    def __call__(self, _bridge: Any = None) -> CodexRuntime:
        return CodexRuntime(
            self.backend,
            id_factory=DeterministicUUIDFactory(self.seed),
        )


def _rubric():
    return load_and_compile_registry(
        ROOT / "rubrics/source/domain-stage-tables.v1.json"
    ).resolve("medxpertqa", "E2E")


def _manifest(seed: str) -> SandboxManifest:
    episode = BenchmarkEpisode(
        episode_id="turn-mcp-candidate",
        source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
        domain="medxpertqa",
        stage=Stage.E2E,
        instruction="Inspect the seed and write a result with the offered tools.",
        policy_context={"question": "fixture"},
        initial_files={"seed.txt": b"medical evidence\n"},
    )
    return SandboxManifest.create(
        sandbox_id=DeterministicUUIDFactory(seed).new("sandbox"),
        episode=episode,
        rubric=_rubric(),
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


def _trajectory(ids: DeterministicUUIDFactory, role: str, content: Any) -> TrajectoryEvent:
    core = {
        "event_id": ids.new("event"),
        "role": role,
        "content": content,
        "tool_call_ids": (),
    }
    return TrajectoryEvent(**core, event_blake3=blake3_hex(core))


def _evidence(tmp_path: Path) -> EvidenceBundle:
    ids = DeterministicUUIDFactory("turn-mcp-judge-evidence")
    workspace = FilesystemSandbox(
        tmp_path / "judge-snapshot",
        ids.new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    partial = EvidenceBundle(
        bundle_id=ids.new("bundle"),
        rollout_id=ids.new("rollout"),
        sandbox_manifest=_manifest("turn-mcp-judge-manifest"),
        model=ModelTarget(Cohort.STRONG, "actor", "fake-provider"),
        policy_visible_context={"question": "fixture"},
        context_blake3=blake3_hex({"question": "fixture"}),
        workspace_before=workspace.snapshot("before"),
        workspace_after=workspace.snapshot("after"),
        tool_trace=_empty_trace(),
        policy_events=(
            _trajectory(ids, "system", "system"),
            _trajectory(ids, "user", {"question": "fixture"}),
            _trajectory(ids, "assistant", "answer"),
        ),
        assistant_output="answer",
        provider_receipt_blake3=blake3_hex("provider"),
        safe_provider_metadata={"fixture": True},
        bundle_blake3="",
    )
    return EvidenceBundle(
        **{
            key: getattr(partial, key)
            for key in partial.__dataclass_fields__
            if key != "bundle_blake3"
        },
        bundle_blake3=blake3_hex(evidence_core(partial)),
    )


def _transport(tmp_path: Path) -> tuple[TurnMCPBridgeFactory, Path]:
    del tmp_path
    # AF_UNIX path names have a small platform bound, so keep this root short
    # while still asserting complete child lifecycle cleanup below.
    temp_root = Path(tempfile.mkdtemp(prefix="eva-mcp-test-", dir="/tmp"))
    return (
        TurnMCPBridgeFactory(
            proxy_python=Path(sys.executable).resolve(),
            proxy_script=PROXY.resolve(),
            temp_root=temp_root,
            maximum_parallel_tools=16,
        ),
        temp_root,
    )


def test_turn_mcp_launch_metadata_commits_sanitized_exec_boundary(
    tmp_path: Path,
) -> None:
    transport, temp_root = _transport(tmp_path)
    metadata = dict(transport.public_metadata())
    digest = metadata.pop("launch_blake3")
    assert digest == transport.launch_blake3 == blake3_hex(metadata)
    assert metadata["schema"] == "eva.turn-mcp-bridge-launch.v1"
    assert metadata["inherited_parent_environment"] is False
    assert metadata["environment_values_recorded"] is False
    assert metadata["turn_nonce_recorded"] is False
    assert tuple(metadata["final_child_environment_names"]) == (
        "EVA_TURN_MCP_MAXIMUM",
        "EVA_TURN_MCP_NONCE",
        "EVA_TURN_MCP_SOCKET",
        "LANG",
        "LC_ALL",
        "PATH",
        "TZ",
    )
    assert metadata["proxy_exec_script_blake3"] == blake3_bytes(
        PROXY.with_name("turn_mcp_exec.py").read_bytes()
    )
    socket_preflight = metadata["unix_socket_preflight"]
    assert socket_preflight == {
        "schema": "eva.turn-mcp-unix-socket-preflight.v1",
        "temp_root": str(temp_root.resolve()),
        "temp_root_uid": os.getuid(),
        "temp_root_mode": 0o700,
        "temp_root_owned_by_process": True,
        "temp_root_is_symlink": False,
        "directory_prefix": "eva-turn-mcp-",
        "random_component_bytes": 8,
        "socket_filename": "broker.sock",
        "probed_socket_path_bytes": len(
            os.fsencode(
                str(temp_root.resolve() / "eva-turn-mcp-XXXXXXXX" / "broker.sock")
            )
        ),
        "sockaddr_un_path_capacity_bytes": 108,
        "terminator_bytes": 1,
        "name_length_source": "tempfile.mkdtemp_probe_removed",
        "preflight_passed": True,
    }
    assert socket_preflight["probed_socket_path_bytes"] < 108
    temp_root.rmdir()


def test_turn_mcp_preflight_rejects_overlong_unix_socket_path() -> None:
    temp_root = Path(
        tempfile.mkdtemp(prefix=f"eva-mcp-path-bound-{'x' * 60}", dir="/tmp")
    )
    try:
        expected_bytes = len(
            os.fsencode(str(temp_root / "eva-turn-mcp-XXXXXXXX" / "broker.sock"))
        )
        assert expected_bytes >= 108
        with pytest.raises(TurnMCPError, match="Unix socket path"):
            TurnMCPBridgeFactory(
                proxy_python=Path(sys.executable).resolve(),
                proxy_script=PROXY.resolve(),
                temp_root=temp_root,
            )
        assert list(temp_root.iterdir()) == []
    finally:
        temp_root.rmdir()


def test_turn_mcp_preflight_rejects_non_private_root() -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="eva-mcp-mode-test-", dir="/tmp"))
    temp_root.chmod(0o755)
    try:
        with pytest.raises(TurnMCPError, match="uid-owned mode-0700"):
            TurnMCPBridgeFactory(
                proxy_python=Path(sys.executable).resolve(),
                proxy_script=PROXY.resolve(),
                temp_root=temp_root,
            )
    finally:
        temp_root.chmod(0o700)
        temp_root.rmdir()


def test_turn_mcp_preflight_rejects_symlink_root(tmp_path: Path) -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="eva-mcp-link-test-", dir="/tmp"))
    alias = tmp_path / "mcp-root-link"
    alias.symlink_to(temp_root, target_is_directory=True)
    try:
        with pytest.raises(TurnMCPError, match="uid-owned mode-0700"):
            TurnMCPBridgeFactory(
                proxy_python=Path(sys.executable).resolve(),
                proxy_script=PROXY.resolve(),
                temp_root=alias.absolute(),
            )
    finally:
        alias.unlink()
        temp_root.rmdir()


def test_actor_real_mcp_process_executes_exact_candidate_registry_once(
    tmp_path: Path,
) -> None:
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    counts = {"inspect_a": 0, "inspect_b": 0, "write_once": 0}
    lock = Lock()
    read_barrier = Barrier(2)

    def inspect(name: str):
        def handler(workspace: FilesystemSandbox, arguments: Mapping[str, Any]):
            with lock:
                counts[name] += 1
            read_barrier.wait(timeout=3)
            return {"content": workspace.read_bytes(arguments["path"]).decode()}

        return handler

    def write_once(workspace: FilesystemSandbox, arguments: Mapping[str, Any]):
        with lock:
            counts["write_once"] += 1
        workspace.write_bytes(arguments["path"], b"verified\n", create_only=True)
        return {"written": arguments["path"]}

    registry = ToolRegistry(
        (
            ToolDefinition(
                "inspect_a", "Inspect candidate evidence A.", schema,
                inspect("inspect_a"), parallel_safe=True, read_only=True,
            ),
            ToolDefinition(
                "inspect_b", "Inspect candidate evidence B.", schema,
                inspect("inspect_b"), parallel_safe=True, read_only=True,
            ),
            ToolDefinition(
                "write_once", "Write the candidate result once.", schema,
                write_once, parallel_safe=False, read_only=False,
            ),
        )
    )
    workspace = FilesystemSandbox(
        tmp_path / "actor-workspaces",
        DeterministicUUIDFactory("turn-mcp-actor-workspace").new("workspace"),
        {"seed.txt": b"medical evidence\n"},
    )
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=registry,
        id_factory=DeterministicUUIDFactory("turn-mcp-actor-tools"),
    )
    manifest = _manifest("turn-mcp-actor-manifest")
    request = RolloutRequest(
        rollout_id=DeterministicUUIDFactory("turn-mcp-rollout").new("rollout"),
        sandbox=manifest,
        model=ModelTarget(Cohort.STRONG, "strong-model", "fake-provider"),
        policy_visible_context={"question": "fixture"},
        available_tools=registry.public_schemas(),
    )
    backend = _MCPBackend(
        server="evamed",
        groups=(
            (
                ("inspect_a", {"path": "seed.txt"}),
                ("inspect_b", {"path": "seed.txt"}),
            ),
            (("write_once", {"path": "result.txt"}),),
        ),
        final_text="actor-complete",
    )
    transport, temp_root = _transport(tmp_path)

    def options_factory(request, cwd, offers, bridge):
        del bridge
        return CodexThreadOptions(
            role=CodexRole.STRONG_ACTOR,
            model=request.model.model_id,
            provider=request.model.provider,
            cwd=cwd,
            sandbox=CodexSandbox.READ_ONLY,
            offered_tools=offers,
            config={"features": {"multi_tool": True}},
        )

    adapter = CodexRolloutAdapter(
        _RuntimeFactory(backend, "turn-mcp-actor-runtime"),
        options_factory,
        id_factory=DeterministicUUIDFactory("turn-mcp-actor-adapter"),
        turn_mcp_factory=transport,
    )
    rollout = adapter.run(request, runtime)

    assert rollout.assistant_output == "actor-complete"
    assert rollout.policy_events[0].content == DEFAULT_ACTOR_SYSTEM_INSTRUCTION
    assert counts == {"inspect_a": 1, "inspect_b": 1, "write_once": 1}
    assert workspace.read_bytes("result.txt") == b"verified\n"
    trace = runtime.trace()
    assert len(trace.results) == 3
    assert trace.frontier_count == 2
    assert trace.max_parallelism_observed == 2
    assert [row["name"] for row in backend.catalogs[0]] == [
        "inspect_a", "inspect_b", "write_once"
    ]
    assert canonical_value(backend.catalogs[0]) == canonical_value(
        tuple(offer.mcp_transport_entry() for offer in backend.starts[0].offered_tools)
    )
    assert backend.open_count == backend.close_count == 1
    assert list(temp_root.iterdir()) == []
    temp_root.rmdir()


def test_judge_real_mcp_process_reads_immutable_evidence_once_without_replay(
    tmp_path: Path,
) -> None:
    rubric = _rubric()
    evidence = _evidence(tmp_path)
    rows = []
    hard_passed = True
    for raw in rubric.items:
        item = canonical_value(raw)
        score_bps = max(level["score_bps"] for level in item["partial_credit"]["levels"])
        gate = item.get("hard_gate")
        if isinstance(gate, dict) and score_bps < gate["minimum_score_bps"]:
            hard_passed = False
        rows.append(
            {
                "item_id": item["item_id"],
                "score": score_bps / 10_000,
                "evidence_refs": ["workspace:after:seed.txt"],
                "rationale": "Inspected the immutable committed content.",
            }
        )
    backend = _MCPBackend(
        server="evamed-judge",
        groups=(
            (
                (
                    "workspace_read",
                    {"snapshot": "after", "path": "seed.txt", "offset": 0, "max_bytes": 65536},
                ),
                (
                    "workspace_search",
                    {
                        "snapshot": "after",
                        "query": "medical",
                        "path_prefix": None,
                        "case_sensitive": True,
                        "max_matches": 10,
                    },
                ),
            ),
        ),
        final_text=json.dumps(
            {
                "item_scores": rows,
                "hard_gates_passed": hard_passed,
                "summary": "Immutable workspace inspected.",
            },
            sort_keys=True,
        ),
    )
    transport, temp_root = _transport(tmp_path)
    judge_cwd = tmp_path / "judge-cwd"
    judge_cwd.mkdir()

    def options_factory(request: JudgeRequest, offers: tuple[CodexToolOffer, ...]):
        return CodexThreadOptions(
            role=CodexRole.JUDGE,
            model=request.judge_model_id,
            provider="anthropic",
            cwd=str(judge_cwd),
            sandbox=CodexSandbox.READ_ONLY,
            offered_tools=offers,
        )

    judge = CodexOpus5AgentJudge(
        _RuntimeFactory(backend, "turn-mcp-judge-runtime"),
        options_factory,
        model_id="claude-opus-5",
        id_factory=DeterministicUUIDFactory("turn-mcp-judge-adapter"),
        turn_mcp_factory=transport,
        maximum_parallel_tools=8,
    )
    request = JudgeRequest(
        judgment_id=DeterministicUUIDFactory("turn-mcp-judgment").new("judgment"),
        judge_model_id="claude-opus-5",
        policy_visible_context=evidence.policy_visible_context,
        workspace_evidence=evidence,
        judge_only_reference={"reference_answer": "fixture"},
    )
    assessment = judge.judge(request, rubric)

    # Exactly the two live MCP calls exist. A post-turn replay would either
    # add another frontier/results or trip reused identities.
    assert len(assessment.agent_trace.results) == 2
    assert assessment.agent_trace.frontier_count == 1
    assert assessment.agent_trace.max_parallelism_observed == 2
    assert assessment.agent_trace.content_inspection_count == 2
    assert assessment.agent_trace.retry_count == 0
    assert assessment.agent_trace.inspected_evidence_refs == (
        "workspace:after:seed.txt",
    )
    assert [row["name"] for row in backend.catalogs[0]] == [
        "workspace_diff", "workspace_list", "workspace_read", "workspace_search"
    ]
    assert all(offer.visibility == "judge-only" for offer in backend.starts[0].offered_tools)
    assert backend.open_count == backend.close_count == 1
    assert list(temp_root.iterdir()) == []
    temp_root.rmdir()


class _RecordingExecutor:
    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.calls: list[tuple[tuple[str, Mapping[str, Any]], ...]] = []

    def execute_group(self, calls):
        group = tuple(calls)
        self.calls.append(group)
        return tuple({"variant": self.variant, "arguments": arguments} for _, arguments in group)


def test_each_turn_exposes_one_exact_schema_variant_and_rejects_bad_nonce(
    tmp_path: Path,
) -> None:
    transport, temp_root = _transport(tmp_path)
    schemas = (
        {
            "type": "object",
            "properties": {"gene": {"type": "string"}},
            "required": ["gene"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"doi": {"type": "string"}},
            "required": ["doi"],
            "additionalProperties": False,
        },
    )
    catalogs = []
    for ordinal, (schema, arguments) in enumerate(
        zip(schemas, ({"gene": "BRCA1"}, {"doi": "10.1/example"}), strict=True)
    ):
        executor = _RecordingExecutor(str(ordinal))
        options = CodexThreadOptions(
            role=CodexRole.STRONG_ACTOR,
            model="model",
            provider="provider",
            cwd=str(tmp_path),
            sandbox=CodexSandbox.WORKSPACE_WRITE,
            offered_tools=(
                CodexToolOffer(
                    fully_qualified_name="evamed/lookup",
                    description="Candidate-specific lookup.",
                    input_schema=schema,
                ),
            ),
        )
        session = transport.open_actor(options, executor)
        with session as bound:
            mode = stat.S_IMODE(session.broker.socket_path.stat().st_mode)
            directory_mode = stat.S_IMODE(session.broker.directory.stat().st_mode)
            assert mode == 0o600 and directory_mode == 0o700

            # A process that merely discovers the socket cannot call a tool;
            # the per-turn nonce is required and authentication fails before
            # the executor boundary.
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(session.broker.socket_path))
                client.sendall(
                    json.dumps(
                        {
                            "nonce": "wrong",
                            "request": {
                                "jsonrpc": "2.0",
                                "id": "bad-auth",
                                "method": "tools/call",
                                "params": {"name": "lookup", "arguments": arguments},
                            },
                        }
                    ).encode()
                    + b"\n"
                )
                response = json.loads(client.makefile("rb").readline())
            assert response["response"]["error"]["message"] == "TurnMCPError"
            assert executor.calls == []

            listed, outcomes = _exercise_proxy(bound, ((("lookup", arguments),),))
            catalogs.append(listed)
            assert outcomes[0][0]["structuredContent"]["variant"] == str(ordinal)
        assert not session.broker.directory.exists()
        assert executor.calls == [(("lookup", arguments),)]

    assert canonical_value(catalogs[0][0]["inputSchema"]) == canonical_value(schemas[0])
    assert canonical_value(catalogs[1][0]["inputSchema"]) == canonical_value(schemas[1])
    assert catalogs[0] != catalogs[1]
    assert list(temp_root.iterdir()) == []
    temp_root.rmdir()


def test_real_mcp_process_batches_more_than_sixteen_independent_calls(
    tmp_path: Path,
) -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="eva-mcp-wide-test-", dir="/tmp"))
    transport = TurnMCPBridgeFactory(
        proxy_python=Path(sys.executable).resolve(),
        proxy_script=PROXY.resolve(),
        temp_root=temp_root,
        maximum_parallel_tools=64,
    )
    class SlowFirstExecutor(_RecordingExecutor):
        def execute_group(self, values):
            result = super().execute_group(values)
            if len(self.calls) == 1:
                # Keep the first small arrival wave live while the remaining
                # proxy workers connect; the next broker frontier then proves
                # an actual execution group wider than the historical 16 cap.
                time.sleep(0.1)
            return result

    executor = SlowFirstExecutor("wide")
    options = CodexThreadOptions(
        role=CodexRole.STRONG_ACTOR,
        model="model",
        provider="provider",
        cwd=str(tmp_path),
        sandbox=CodexSandbox.WORKSPACE_WRITE,
        offered_tools=(
            CodexToolOffer(
                fully_qualified_name="evamed/inspect",
                description="Inspect one independent item.",
                input_schema={
                    "type": "object",
                    "properties": {"ordinal": {"type": "integer"}},
                    "required": ["ordinal"],
                    "additionalProperties": False,
                },
                parallel_safe=True,
                read_only=True,
            ),
        ),
    )
    calls = tuple(("inspect", {"ordinal": index}) for index in range(32))

    with transport.open_actor(options, executor) as bound:
        _listed, outcomes = _exercise_proxy(bound, (calls,))

    assert len(outcomes[0]) == 32
    executed = [arguments["ordinal"] for group in executor.calls for _, arguments in group]
    assert sorted(executed) == list(range(32))
    assert max(map(len, executor.calls)) > 16
    assert list(temp_root.iterdir()) == []
    temp_root.rmdir()


def test_turn_mcp_cleans_private_topology_when_provider_turn_raises(tmp_path: Path) -> None:
    transport, temp_root = _transport(tmp_path)
    executor = _RecordingExecutor("unused")
    options = CodexThreadOptions(
        role=CodexRole.STRONG_ACTOR,
        model="model",
        provider="provider",
        cwd=str(tmp_path),
        sandbox=CodexSandbox.WORKSPACE_WRITE,
        offered_tools=(
            CodexToolOffer(
                fully_qualified_name="evamed/only",
                description="Only this candidate tool.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            ),
        ),
    )
    session = transport.open_actor(options, executor)
    with pytest.raises(RuntimeError, match="provider failed"):
        with session:
            raise RuntimeError("provider failed")
    assert executor.calls == []
    assert not session.broker.directory.exists()
    assert list(temp_root.iterdir()) == []
    temp_root.rmdir()
