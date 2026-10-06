from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from eva_agent.codex_runtime.backend import (
    BackendInputItem,
    BackendTurnOptions,
    CodexLaunchOptions,
)
from eva_agent.codex_runtime.contracts import (
    CodexRole,
    CodexRuntimeError,
    CodexSandbox,
    CodexThreadOptions,
    CodexTurnInput,
)
from eva_agent.codex_runtime.native_backend_v2 import (
    NativeCodexBackendV2,
    NativeRuntimeVersions,
    NativeThreadStartEvidence,
    REQUIRED_CODEX_CLI_VERSION,
    REQUIRED_CODEX_SDK_VERSION,
    installed_native_runtime_versions,
)
from eva_agent.codex_runtime.runtime import CodexRuntime
from eva_agent.pipeline.digests import canonical_value
from eva_agent.pipeline.ids import DeterministicUUIDFactory


PINNED = NativeRuntimeVersions(
    sdk=REQUIRED_CODEX_SDK_VERSION,
    cli=REQUIRED_CODEX_CLI_VERSION,
)


class _FakeTurn:
    def __init__(
        self,
        events: Sequence[Any],
        *,
        approval_handler,
        approval_during_stream: str | None = None,
    ) -> None:
        self.id = "turn-native-v2"
        self.events = tuple(events)
        self.approval_handler = approval_handler
        self.approval_during_stream = approval_during_stream
        self.approval_responses: list[Mapping[str, Any]] = []
        self.interruptions = 0

    async def interrupt(self) -> None:
        self.interruptions += 1

    async def stream(self):
        if self.approval_during_stream is not None:
            self.approval_responses.append(
                self.approval_handler(
                    self.approval_during_stream,
                    {"threadId": "thread-native-v2", "turnId": self.id},
                )
            )
        for event in self.events:
            yield event


class _FakeSession:
    def __init__(self) -> None:
        self.server_version = "0.147.0 aarch64-unknown-linux-gnu"
        self.approval_handler = None
        self.opened = 0
        self.closed = 0
        self.thread_starts: list[Mapping[str, Any]] = []
        self.turn_starts: list[tuple[str, Sequence[Mapping[str, Any]], Mapping[str, Any]]] = []
        self.events: Sequence[Any] = ()
        self.last_turn: _FakeTurn | None = None
        self.approval_during_open: str | None = None
        self.approval_during_start: str | None = None
        self.approval_during_stream: str | None = None
        self.approval_responses: list[Mapping[str, Any]] = []
        self.additional_evidence_root: str | None = None
        self.provider_calls = 0

    async def open(self) -> None:
        self.opened += 1
        if self.approval_during_open is not None:
            self.approval_responses.append(
                self.approval_handler(self.approval_during_open, None)
            )

    async def close(self) -> None:
        self.closed += 1

    async def start_thread(
        self, params: Mapping[str, Any]
    ) -> NativeThreadStartEvidence:
        self.thread_starts.append(params)
        root = params["cwd"]
        configured = params["config"]["sandbox_workspace_write"]
        # Codex 0.147 canonicalizes cwd as the implicit workspace root and
        # returns only *additional* roots in thread/start evidence.
        roots = []
        if self.additional_evidence_root is not None:
            roots.append(self.additional_evidence_root)
        return NativeThreadStartEvidence(
            thread_id="thread-native-v2",
            cwd=root,
            sandbox_policy={
                "type": "workspaceWrite",
                "networkAccess": configured["network_access"],
                "writableRoots": roots,
                "excludeSlashTmp": configured["exclude_slash_tmp"],
                "excludeTmpdirEnvVar": configured["exclude_tmpdir_env_var"],
            },
            approval_policy=params["approvalPolicy"],
            ephemeral=params["ephemeral"],
            approvals_reviewer="user",
        )

    async def start_turn(
        self,
        thread_id: str,
        input_items: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any],
    ) -> _FakeTurn:
        self.turn_starts.append((thread_id, input_items, params))
        turn = _FakeTurn(
            self.events,
            approval_handler=self.approval_handler,
            approval_during_stream=self.approval_during_stream,
        )
        self.last_turn = turn
        if self.approval_during_start is not None:
            self.approval_responses.append(
                self.approval_handler(
                    self.approval_during_start,
                    {"threadId": thread_id, "turnId": turn.id},
                )
            )
        return turn


def _factory(session: _FakeSession):
    def build(_launch, approval_handler):
        session.approval_handler = approval_handler
        return session

    return build


def _options(root: Path, *, config: Mapping[str, Any] | None = None) -> CodexThreadOptions:
    return CodexThreadOptions(
        role=CodexRole.STRONG_ACTOR,
        model="gpt-5.6-sol",
        provider="openai",
        cwd=root.as_posix(),
        sandbox=CodexSandbox.WORKSPACE_WRITE,
        config={} if config is None else config,
        ephemeral=True,
    )


async def _open_thread(
    tmp_path: Path,
    session: _FakeSession,
    *,
    options: CodexThreadOptions | None = None,
):
    backend = NativeCodexBackendV2(
        session_factory=_factory(session),
        version_resolver=lambda: PINNED,
    )
    await backend.open()
    return backend, await backend.start_thread(options or _options(tmp_path))


def _event(method: str, payload: Mapping[str, Any]) -> Any:
    return SimpleNamespace(method=method, payload=payload)


def test_stage_allowed_workspace_write_uses_completed_native_file_change_with_exact_policy(
    tmp_path: Path,
) -> None:
    change = {
        "id": "file-change-1",
        "type": "fileChange",
        "status": "completed",
        "changes": [
            {
                "path": "report.md",
                "kind": {"type": "update", "movePath": None},
                "diff": "@@ -1 +1 @@\n-old\n+new\n",
            }
        ],
    }
    session = _FakeSession()
    session.events = (
        _event(
            "item/started",
            {"threadId": "thread-native-v2", "turnId": "turn-native-v2", "item": change},
        ),
        _event(
            "item/completed",
            {"threadId": "thread-native-v2", "turnId": "turn-native-v2", "item": change},
        ),
        _event(
            "turn/completed",
            {
                "threadId": "thread-native-v2",
                "turn": {"id": "turn-native-v2", "status": "completed", "items": []},
            },
        ),
    )

    async def scenario():
        backend, thread = await _open_thread(tmp_path, session)
        turn = await thread.turn(
            (BackendInputItem(kind="text", text="edit the candidate"),),
            BackendTurnOptions(
                cwd=tmp_path.as_posix(),
                sandbox=CodexSandbox.WORKSPACE_WRITE,
            ),
        )
        events = [event async for event in turn.stream()]
        await backend.close()
        return events

    events = asyncio.run(scenario())
    assert events == list(session.events)
    assert events[0].payload["item"]["type"] == "fileChange"
    assert events[1].payload["item"]["status"] == "completed"
    assert session.provider_calls == 0
    assert session.opened == session.closed == 1

    thread_params = session.thread_starts[0]
    assert thread_params["approvalPolicy"] == "never"
    assert thread_params["approvalsReviewer"] is None
    assert thread_params["sandbox"] == "workspace-write"
    assert thread_params["ephemeral"] is True
    assert thread_params["config"]["features"]["shell_tool"] is False
    assert thread_params["config"]["features"]["unified_exec"] is False
    assert thread_params["config"]["sandbox_workspace_write"] == {
        "network_access": False,
        "writable_roots": [tmp_path.as_posix()],
        "exclude_slash_tmp": True,
        "exclude_tmpdir_env_var": True,
    }

    _thread_id, input_items, turn_params = session.turn_starts[0]
    assert input_items == [{"type": "text", "text": "edit the candidate"}]
    assert turn_params["approvalPolicy"] == "never"
    assert turn_params["approvalsReviewer"] is None
    assert turn_params["sandboxPolicy"] == {
        "type": "workspaceWrite",
        "networkAccess": False,
        "writableRoots": [tmp_path.as_posix()],
        "excludeSlashTmp": True,
        "excludeTmpdirEnvVar": True,
    }

    # Bind the assertion to the exact 0.147.0 generated app-server models,
    # without starting a process or crossing the provider boundary.
    from openai_codex.generated.v2_all import ThreadStartParams, TurnStartParams

    generated_thread = ThreadStartParams.model_validate(thread_params)
    generated_turn = TurnStartParams.model_validate(turn_params)
    assert generated_thread.model_dump(by_alias=True, mode="json")["approvalPolicy"] == "never"
    assert generated_turn.model_dump(by_alias=True, mode="json")["sandboxPolicy"] == (
        turn_params["sandboxPolicy"]
    )


def test_stage_allowed_file_change_seals_standard_runtime_receipt(tmp_path: Path) -> None:
    change = {
        "id": "file-change-receipt",
        "type": "fileChange",
        "status": "completed",
        "changes": [
            {
                "path": "analysis.py",
                "kind": {"type": "update", "movePath": None},
                "diff": "@@ -1 +1 @@\n-old\n+new\n",
            }
        ],
    }
    session = _FakeSession()
    session.events = (
        _event(
            "item/started",
            {"threadId": "thread-native-v2", "turnId": "turn-native-v2", "item": change},
        ),
        _event(
            "item/completed",
            {"threadId": "thread-native-v2", "turnId": "turn-native-v2", "item": change},
        ),
        _event(
            "turn/completed",
            {
                "threadId": "thread-native-v2",
                "turn": {"id": "turn-native-v2", "status": "completed", "items": []},
            },
        ),
    )

    async def scenario():
        backend = NativeCodexBackendV2(
            session_factory=_factory(session),
            version_resolver=lambda: PINNED,
        )
        async with CodexRuntime(
            backend,
            id_factory=DeterministicUUIDFactory("native-file-change-receipt"),
        ) as runtime:
            handle = await runtime.start_thread(_options(tmp_path))
            return await runtime.run_turn(
                handle,
                CodexTurnInput(public_text="Edit only this candidate workspace."),
            )

    receipt = asyncio.run(scenario())
    assert receipt.status == "completed"
    assert receipt.sandbox is CodexSandbox.WORKSPACE_WRITE
    assert receipt.thread_resumed is False
    assert receipt.sdk_version == REQUIRED_CODEX_SDK_VERSION
    assert len(receipt.tool_calls) == 1
    tool = receipt.tool_calls[0]
    assert tool.tool_type == "fileChange"
    assert tool.name == "fileChange"
    assert tool.status == "completed"
    assert tool.lifecycle == ("item/started", "item/completed")
    assert canonical_value(tool.arguments) == {"changes": change["changes"]}
    assert canonical_value(tool.output) == {"status": "completed"}
    assert session.provider_calls == 0


@pytest.mark.parametrize(
    "method",
    (
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
        "item/futureNativeSurface/requestApproval",
    ),
)
def test_unexpected_approval_is_declined_and_turn_fails_closed(
    tmp_path: Path, method: str
) -> None:
    session = _FakeSession()
    session.approval_during_start = method

    async def scenario():
        backend, thread = await _open_thread(tmp_path, session)
        with pytest.raises(CodexRuntimeError, match="forbidden approval request"):
            await thread.turn(
                (BackendInputItem(kind="text", text="work"),),
                BackendTurnOptions(
                    cwd=tmp_path.as_posix(),
                    sandbox=CodexSandbox.WORKSPACE_WRITE,
                ),
            )
        await backend.close()

    asyncio.run(scenario())
    assert session.approval_responses == [{"decision": "decline"}]
    assert session.last_turn is not None
    assert session.last_turn.interruptions == 1
    assert session.provider_calls == 0


def test_approval_during_stream_is_declined_and_interrupts(tmp_path: Path) -> None:
    session = _FakeSession()
    session.approval_during_stream = "item/fileChange/requestApproval"
    session.events = (
        _event(
            "turn/completed",
            {
                "threadId": "thread-native-v2",
                "turn": {"id": "turn-native-v2", "status": "completed", "items": []},
            },
        ),
    )

    async def scenario():
        backend, thread = await _open_thread(tmp_path, session)
        turn = await thread.turn(
            (BackendInputItem(kind="text", text="work"),),
            BackendTurnOptions(
                cwd=tmp_path.as_posix(),
                sandbox=CodexSandbox.WORKSPACE_WRITE,
            ),
        )
        with pytest.raises(CodexRuntimeError, match="forbidden approval request"):
            _ = [event async for event in turn.stream()]
        await backend.close()

    asyncio.run(scenario())
    assert session.last_turn is not None
    assert session.last_turn.approval_responses == [{"decision": "decline"}]
    assert session.last_turn.interruptions == 1


def test_approval_during_app_server_open_is_declined_and_open_fails_closed() -> None:
    session = _FakeSession()
    session.approval_during_open = "item/commandExecution/requestApproval"

    async def scenario():
        backend = NativeCodexBackendV2(
            session_factory=_factory(session),
            version_resolver=lambda: PINNED,
        )
        with pytest.raises(CodexRuntimeError, match="forbidden approval request"):
            await backend.open()

    asyncio.run(scenario())
    assert session.approval_responses == [{"decision": "decline"}]
    assert session.opened == session.closed == 1
    assert session.provider_calls == 0


@pytest.mark.parametrize(
    "event",
    (
        _event(
            "item/started",
            {
                "threadId": "thread-native-v2",
                "turnId": "turn-native-v2",
                "item": {"id": "cmd-1", "type": "commandExecution"},
            },
        ),
        _event(
            "item/commandExecution/outputDelta",
            {
                "threadId": "thread-native-v2",
                "turnId": "turn-native-v2",
                "itemId": "cmd-1",
                "delta": "forbidden",
            },
        ),
        _event(
            "turn/completed",
            {
                "threadId": "thread-native-v2",
                "turn": {
                    "id": "turn-native-v2",
                    "status": "completed",
                    "items": [{"id": "cmd-1", "type": "commandExecution"}],
                },
            },
        ),
    ),
)
def test_command_execution_event_fails_closed_and_interrupts(
    tmp_path: Path, event: Any
) -> None:
    session = _FakeSession()
    session.events = (event,)

    async def scenario():
        backend, thread = await _open_thread(tmp_path, session)
        turn = await thread.turn(
            (BackendInputItem(kind="text", text="work"),),
            BackendTurnOptions(
                cwd=tmp_path.as_posix(),
                sandbox=CodexSandbox.WORKSPACE_WRITE,
            ),
        )
        with pytest.raises(CodexRuntimeError, match="commandExecution is forbidden"):
            _ = [row async for row in turn.stream()]
        await backend.close()

    asyncio.run(scenario())
    assert session.last_turn is not None
    assert session.last_turn.interruptions == 1


def test_resume_and_different_turn_root_are_rejected(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    session = _FakeSession()

    async def scenario():
        backend, thread = await _open_thread(tmp_path, session)
        with pytest.raises(CodexRuntimeError, match="forbids thread resume"):
            await backend.resume_thread("old-thread", _options(tmp_path))
        with pytest.raises(CodexRuntimeError, match="different candidate root"):
            await thread.turn(
                (BackendInputItem(kind="text", text="work"),),
                BackendTurnOptions(
                    cwd=other.as_posix(),
                    sandbox=CodexSandbox.WORKSPACE_WRITE,
                ),
            )
        await backend.close()

    asyncio.run(scenario())
    assert session.turn_starts == []


def test_judge_and_read_only_threads_are_rejected_before_sdk_thread_start(
    tmp_path: Path,
) -> None:
    session = _FakeSession()

    async def scenario():
        backend = NativeCodexBackendV2(
            session_factory=_factory(session),
            version_resolver=lambda: PINNED,
        )
        await backend.open()
        judge = CodexThreadOptions(
            role=CodexRole.JUDGE,
            model="gpt-5.6-sol",
            provider="openai",
            cwd=tmp_path.as_posix(),
            sandbox=CodexSandbox.WORKSPACE_WRITE,
        )
        with pytest.raises(CodexRuntimeError, match="actor-only"):
            await backend.start_thread(judge)
        read_only = CodexThreadOptions(
            role=CodexRole.STRONG_ACTOR,
            model="gpt-5.6-sol",
            provider="openai",
            cwd=tmp_path.as_posix(),
            sandbox=CodexSandbox.READ_ONLY,
        )
        with pytest.raises(CodexRuntimeError, match="requires workspace-write"):
            await backend.start_thread(read_only)
        await backend.close()

    asyncio.run(scenario())
    assert session.thread_starts == []


def test_additional_writable_root_and_caller_permission_config_are_rejected(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    session = _FakeSession()
    session.additional_evidence_root = outside.as_posix()

    async def evidence_scenario():
        backend = NativeCodexBackendV2(
            session_factory=_factory(session),
            version_resolver=lambda: PINNED,
        )
        await backend.open()
        with pytest.raises(CodexRuntimeError, match="additional writable roots"):
            await backend.start_thread(_options(tmp_path))
        await backend.close()

    asyncio.run(evidence_scenario())

    for config in (
        {"sandbox_workspace_write": {"writable_roots": [tmp_path.as_posix()]}},
        {"permissions": {"file_system": "unrestricted"}},
        {"features": {"shell_tool": True}},
    ):
        isolated = _FakeSession()

        async def config_scenario():
            backend = NativeCodexBackendV2(
                session_factory=_factory(isolated),
                version_resolver=lambda: PINNED,
            )
            await backend.open()
            with pytest.raises(CodexRuntimeError):
                await backend.start_thread(_options(tmp_path, config=config))
            await backend.close()

        asyncio.run(config_scenario())
        assert isolated.thread_starts == []


@pytest.mark.parametrize(
    "versions",
    (
        NativeRuntimeVersions(sdk="0.148.0", cli="0.147.0"),
        NativeRuntimeVersions(sdk="0.147.0", cli="0.148.0"),
    ),
)
def test_version_mismatch_fails_before_session_creation(versions: NativeRuntimeVersions) -> None:
    created = False

    def forbidden_factory(_launch, _handler):
        nonlocal created
        created = True
        raise AssertionError("session factory crossed")

    async def scenario():
        backend = NativeCodexBackendV2(
            session_factory=forbidden_factory,
            version_resolver=lambda: versions,
        )
        with pytest.raises(CodexRuntimeError, match="requires openai-codex"):
            await backend.open()

    asyncio.run(scenario())
    assert created is False


def test_app_server_version_mismatch_fails_closed_and_closes_session() -> None:
    session = _FakeSession()
    session.server_version = "0.148.0"

    async def scenario():
        backend = NativeCodexBackendV2(
            session_factory=_factory(session),
            version_resolver=lambda: PINNED,
        )
        with pytest.raises(CodexRuntimeError, match="app-server version differs"):
            await backend.open()

    asyncio.run(scenario())
    assert session.opened == session.closed == 1
    assert session.thread_starts == []
    assert session.provider_calls == 0


def test_unpinned_binary_and_argv_override_are_rejected_before_session_creation() -> None:
    created = False

    def forbidden_factory(_launch, _handler):
        nonlocal created
        created = True
        raise AssertionError("session factory crossed")

    launches = (
        CodexLaunchOptions(codex_bin="/usr/bin/codex"),
        CodexLaunchOptions(launch_args_override=("codex", "app-server")),
        CodexLaunchOptions(config_overrides=("sandbox_mode=\"full-access\"",)),
        CodexLaunchOptions(experimental_api=False),
    )
    for launch in launches:

        async def scenario():
            backend = NativeCodexBackendV2(
                launch,
                session_factory=forbidden_factory,
                version_resolver=lambda: PINNED,
            )
            with pytest.raises(CodexRuntimeError):
                await backend.open()

        asyncio.run(scenario())
    assert created is False


def test_installed_sdk_and_bundled_cli_metadata_are_exactly_pinned() -> None:
    assert installed_native_runtime_versions() == PINNED
