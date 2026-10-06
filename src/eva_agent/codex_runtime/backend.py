"""Injectable async boundary around the official ``openai-codex`` SDK.

Production uses :class:`OpenAICodexBackend`; tests inject a protocol-compatible
backend and therefore never start app-server or contact a provider.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, AsyncIterator, Callable, Mapping, Protocol, Sequence

from eva_agent.pipeline.contracts import JsonValue, freeze_json
from eva_agent.pipeline.digests import canonical_value

from .contracts import CodexRuntimeError, CodexSandbox, CodexThreadOptions


@dataclass(frozen=True)
class BackendInputItem:
    """SDK-independent input item; only official SDK constructors consume it."""

    kind: str
    text: str | None = None
    name: str | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "text":
            if self.text is None or self.name is not None or self.path is not None:
                raise CodexRuntimeError("text input item fields differ")
        elif self.kind in {"skill", "mention"}:
            if not self.name or not self.path or self.text is not None:
                raise CodexRuntimeError(f"{self.kind} input item fields differ")
        else:
            raise CodexRuntimeError("unsupported Codex input item")


@dataclass(frozen=True)
class BackendTurnOptions:
    cwd: str
    sandbox: CodexSandbox
    model: str | None = None
    effort: str | None = None
    output_schema: Mapping[str, JsonValue] | None = None
    summary: str | None = None
    service_tier: str | None = None

    def __post_init__(self) -> None:
        if self.output_schema is not None:
            value = freeze_json(self.output_schema)
            if not isinstance(value, Mapping):  # pragma: no cover - typing guard
                raise CodexRuntimeError("output schema must be an object")
            object.__setattr__(self, "output_schema", value)


class BackendNotification(Protocol):
    method: str
    payload: Any


class BackendTurnPort(Protocol):
    id: str

    def stream(self) -> AsyncIterator[BackendNotification]: ...


class BackendThreadPort(Protocol):
    id: str

    async def turn(
        self,
        items: Sequence[BackendInputItem],
        options: BackendTurnOptions,
    ) -> BackendTurnPort: ...


class CodexBackendPort(Protocol):
    """Narrow injectable port used by :class:`~eva_agent.codex_runtime.CodexRuntime`."""

    sdk_version: str
    server_version: str

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def start_thread(self, options: CodexThreadOptions) -> BackendThreadPort: ...

    async def resume_thread(
        self, thread_id: str, options: CodexThreadOptions
    ) -> BackendThreadPort: ...


@dataclass(frozen=True, repr=False)
class CodexLaunchOptions:
    """Local app-server launch controls; values never enter turn receipts."""

    codex_bin: str | None = None
    launch_args_override: tuple[str, ...] | None = None
    config_overrides: tuple[str, ...] = ()
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    client_name: str = "evamed_codex"
    client_title: str = "EvaMed Codex Runtime"
    client_version: str = "0.1.0"
    experimental_api: bool = True

    def __post_init__(self) -> None:
        if self.launch_args_override is not None:
            object.__setattr__(self, "launch_args_override", tuple(self.launch_args_override))
        object.__setattr__(self, "config_overrides", tuple(self.config_overrides))
        if self.env is not None:
            if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.env.items()):
                raise CodexRuntimeError("Codex launch env must contain text keys and values")
            object.__setattr__(self, "env", MappingProxyType(dict(self.env)))
        if not self.client_name or not self.client_title or not self.client_version:
            raise CodexRuntimeError("Codex client identity differs")

    def __repr__(self) -> str:
        return (
            "CodexLaunchOptions("
            f"codex_bin={self.codex_bin!r}, launch_args_override="
            f"{None if self.launch_args_override is None else '<redacted>'}, "
            f"config_overrides={len(self.config_overrides)}, cwd={self.cwd!r}, "
            f"env_keys={tuple(sorted(self.env or {}))!r}, client_name={self.client_name!r}, "
            f"client_version={self.client_version!r})"
        )


@dataclass(frozen=True)
class CodexSdkBindings:
    """Constructor table allowing a no-process, no-network test double."""

    async_codex: Callable[..., Any]
    codex_config: Callable[..., Any]
    approval_deny_all: Any
    sandbox_read_only: Any
    sandbox_workspace_write: Any
    text_input: Callable[..., Any]
    skill_input: Callable[..., Any]
    mention_input: Callable[..., Any]
    sdk_version: str


def load_official_sdk_bindings() -> CodexSdkBindings:
    """Lazily import the published SDK so pure contract tests need no install."""

    try:
        from openai_codex import (  # type: ignore[import-not-found]
            ApprovalMode,
            AsyncCodex,
            CodexConfig,
            MentionInput,
            Sandbox,
            SkillInput,
            TextInput,
            __version__,
        )
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise CodexRuntimeError(
            "openai-codex is required for the production Codex backend"
        ) from exc
    return CodexSdkBindings(
        async_codex=AsyncCodex,
        codex_config=CodexConfig,
        approval_deny_all=ApprovalMode.deny_all,
        sandbox_read_only=Sandbox.read_only,
        sandbox_workspace_write=Sandbox.workspace_write,
        text_input=TextInput,
        skill_input=SkillInput,
        mention_input=MentionInput,
        sdk_version=__version__,
    )


def _thaw(value: Any) -> Any:
    """Return ordinary JSON containers without altering canonical semantics."""

    return canonical_value(value)


def _cancellation_safe_stream_client(client: Any) -> Any | None:
    """Return the pinned SDK stream client only when its wake path is exact.

    ``openai-codex==0.147.0`` offloads a blocking ``Queue.get`` for every
    streamed notification with :func:`asyncio.to_thread`.  Cancelling the
    coroutine does not cancel that worker.  Worse, the SDK stream unregisters
    the queue while unwinding, so closing app-server can no longer wake the
    orphaned worker and ``shutdown_default_executor`` waits forever.

    Production validates this narrow, version-pinned capability during open.
    Injected test bindings may omit it and retain their ordinary public stream.
    """

    stream_client = getattr(client, "_client", None)
    sync_client = getattr(stream_client, "_sync", None)
    router = getattr(sync_client, "_router", None)
    lock = getattr(router, "_lock", None)
    turn_notifications = getattr(router, "_turn_notifications", None)
    if not (
        callable(getattr(stream_client, "register_turn_notifications", None))
        and callable(getattr(stream_client, "unregister_turn_notifications", None))
        and callable(getattr(stream_client, "next_turn_notification", None))
        and hasattr(lock, "__enter__")
        and hasattr(lock, "__exit__")
        and isinstance(turn_notifications, dict)
    ):
        return None
    return stream_client


def _wake_cancelled_turn_waiter(stream_client: Any, turn_id: str) -> None:
    """Wake exactly one cancelled SDK queue before it is unregistered."""

    router = stream_client._sync._router
    with router._lock:
        waiter = router._turn_notifications.get(turn_id)
    if waiter is None or not callable(getattr(waiter, "put_nowait", None)):
        raise CodexRuntimeError("Codex cancelled-turn notification queue differs")
    waiter.put_nowait(CodexRuntimeError("Codex turn stream was cancelled"))


def _is_completed_notification(notification: Any, turn_id: str) -> bool:
    if getattr(notification, "method", None) != "turn/completed":
        return False
    payload = getattr(notification, "payload", None)
    turn = getattr(payload, "turn", None)
    return str(getattr(turn, "id", "")) == turn_id


class _OfficialTurn:
    def __init__(self, handle: Any, stream_client: Any | None = None) -> None:
        self._handle = handle
        self.id = str(handle.id)
        self._stream_client = stream_client

    def stream(self) -> AsyncIterator[BackendNotification]:
        if self._stream_client is None:
            return self._handle.stream()
        return self._stream_cancellation_safe()

    async def _stream_cancellation_safe(self) -> AsyncIterator[BackendNotification]:
        """Mirror the pinned SDK stream while making cancellation drainable."""

        client = self._stream_client
        client.register_turn_notifications(self.id)
        try:
            while True:
                try:
                    notification = await client.next_turn_notification(self.id)
                except asyncio.CancelledError:
                    # A cancelled ``to_thread`` Future does not stop the
                    # underlying Queue.get.  Wake it while the exact queue is
                    # still registered; only then may the SDK route be removed.
                    _wake_cancelled_turn_waiter(client, self.id)
                    raise
                yield notification
                if _is_completed_notification(notification, self.id):
                    break
        finally:
            client.unregister_turn_notifications(self.id)


class _OfficialThread:
    def __init__(
        self,
        thread: Any,
        bindings: CodexSdkBindings,
        stream_client: Any | None = None,
    ) -> None:
        self._thread = thread
        self._bindings = bindings
        self._stream_client = stream_client
        self.id = str(thread.id)

    async def turn(
        self,
        items: Sequence[BackendInputItem],
        options: BackendTurnOptions,
    ) -> BackendTurnPort:
        sdk_items: list[Any] = []
        for item in items:
            if item.kind == "text":
                sdk_items.append(self._bindings.text_input(text=item.text))
            elif item.kind == "skill":
                sdk_items.append(self._bindings.skill_input(name=item.name, path=item.path))
            elif item.kind == "mention":
                sdk_items.append(self._bindings.mention_input(name=item.name, path=item.path))
            else:  # pragma: no cover - BackendInputItem already validates
                raise CodexRuntimeError("unsupported Codex input item")
        sandbox = (
            self._bindings.sandbox_read_only
            if options.sandbox is CodexSandbox.READ_ONLY
            else self._bindings.sandbox_workspace_write
        )
        turn = await self._thread.turn(
            sdk_items,
            approval_mode=self._bindings.approval_deny_all,
            cwd=options.cwd,
            effort=options.effort,
            model=options.model,
            output_schema=(
                None if options.output_schema is None else _thaw(options.output_schema)
            ),
            sandbox=sandbox,
            service_tier=options.service_tier,
            summary=options.summary,
        )
        return _OfficialTurn(turn, self._stream_client)


class OpenAICodexBackend:
    """Official Python SDK adapter for local Codex app-server JSON-RPC."""

    def __init__(
        self,
        launch: CodexLaunchOptions | None = None,
        *,
        bindings: CodexSdkBindings | None = None,
    ) -> None:
        self._launch = launch or CodexLaunchOptions()
        self._bindings = bindings
        self._client: Any | None = None
        self._stream_client: Any | None = None
        self.sdk_version = "unopened"
        self.server_version = "unopened"

    async def open(self) -> None:
        if self._client is not None:
            return
        require_safe_cancellation = self._bindings is None
        bindings = self._bindings or load_official_sdk_bindings()
        self._bindings = bindings
        config = bindings.codex_config(
            codex_bin=self._launch.codex_bin,
            launch_args_override=self._launch.launch_args_override,
            config_overrides=self._launch.config_overrides,
            cwd=self._launch.cwd,
            env=None if self._launch.env is None else dict(self._launch.env),
            client_name=self._launch.client_name,
            client_title=self._launch.client_title,
            client_version=self._launch.client_version,
            experimental_api=self._launch.experimental_api,
        )
        client = bindings.async_codex(config=config)
        try:
            await client.__aenter__()
        except BaseException:
            try:
                await client.close()
            finally:
                raise
        stream_client = _cancellation_safe_stream_client(client)
        if require_safe_cancellation and stream_client is None:
            try:
                await client.close()
            finally:
                raise CodexRuntimeError(
                    "official Codex SDK cancellation wake capability differs"
                )
        self._client = client
        self._stream_client = stream_client
        self.sdk_version = bindings.sdk_version
        metadata = getattr(client, "metadata", None)
        server_info = getattr(metadata, "serverInfo", None)
        self.server_version = str(getattr(server_info, "version", None) or "unknown")

    async def close(self) -> None:
        client, self._client = self._client, None
        self._stream_client = None
        if client is not None:
            await client.__aexit__(None, None, None)

    def _require_open(self) -> tuple[Any, CodexSdkBindings]:
        if self._client is None or self._bindings is None:
            raise CodexRuntimeError("Codex backend is not open")
        return self._client, self._bindings

    @staticmethod
    def _sandbox(options: CodexThreadOptions, bindings: CodexSdkBindings) -> Any:
        return (
            bindings.sandbox_read_only
            if options.sandbox is CodexSandbox.READ_ONLY
            else bindings.sandbox_workspace_write
        )

    async def start_thread(self, options: CodexThreadOptions) -> BackendThreadPort:
        client, bindings = self._require_open()
        thread = await client.thread_start(
            approval_mode=bindings.approval_deny_all,
            base_instructions=options.base_instructions,
            config=_thaw(options.config),
            cwd=options.cwd,
            developer_instructions=options.developer_instructions,
            ephemeral=options.ephemeral,
            model=options.model,
            model_provider=options.provider,
            sandbox=self._sandbox(options, bindings),
            service_name=options.service_name,
            service_tier=options.service_tier,
        )
        return _OfficialThread(thread, bindings, self._stream_client)

    async def resume_thread(
        self, thread_id: str, options: CodexThreadOptions
    ) -> BackendThreadPort:
        client, bindings = self._require_open()
        if not thread_id:
            raise CodexRuntimeError("Codex thread id is required for resume")
        thread = await client.thread_resume(
            thread_id,
            approval_mode=bindings.approval_deny_all,
            base_instructions=options.base_instructions,
            config=_thaw(options.config),
            cwd=options.cwd,
            developer_instructions=options.developer_instructions,
            model=options.model,
            model_provider=options.provider,
            sandbox=self._sandbox(options, bindings),
            service_tier=options.service_tier,
        )
        return _OfficialThread(thread, bindings, self._stream_client)


__all__ = [
    "BackendInputItem",
    "BackendNotification",
    "BackendThreadPort",
    "BackendTurnOptions",
    "BackendTurnPort",
    "CodexBackendPort",
    "CodexLaunchOptions",
    "CodexSdkBindings",
    "OpenAICodexBackend",
    "load_official_sdk_bindings",
]
