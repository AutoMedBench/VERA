"""Strict Codex 0.147.0 backend for candidate-local native file changes.

This adapter is deliberately narrower than :mod:`eva_agent.codex_runtime.backend`.
It uses a fresh ephemeral thread, sends an explicit workspace-write policy on
every turn, and treats any approval request or native command execution as a
terminal boundary violation.  Tests inject a transport and therefore do not
start app-server or contact a provider.

The official 0.147.0 Python SDK does not expose its approval handler or an
explicit ``WorkspaceWriteSandboxPolicy`` through the high-level async API.
The production transport consequently uses the pinned SDK's low-level client
at two small, version-gated seams.  A different SDK or bundled CLI version is
rejected before a process is created.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import importlib.metadata
from pathlib import Path
import stat
from threading import Lock
from typing import Any, Protocol

from eva_agent.pipeline.contracts import JsonValue
from eva_agent.pipeline.digests import canonical_value

from .backend import (
    BackendInputItem,
    BackendNotification,
    BackendThreadPort,
    BackendTurnOptions,
    BackendTurnPort,
    CodexLaunchOptions,
)
from .contracts import CodexRuntimeError, CodexSandbox, CodexThreadOptions


REQUIRED_CODEX_SDK_VERSION = "0.147.0"
REQUIRED_CODEX_CLI_VERSION = "0.147.0"

_COMMAND_APPROVAL_METHOD = "item/commandExecution/requestApproval"
_FILE_CHANGE_APPROVAL_METHOD = "item/fileChange/requestApproval"
_APPROVAL_METHODS = frozenset({_COMMAND_APPROVAL_METHOD, _FILE_CHANGE_APPROVAL_METHOD})
_APPROVAL_METHOD_SUFFIX = "/requestApproval"
_FORBIDDEN_CALLER_CONFIG_KEYS = frozenset(
    {
        "approval_policy",
        "approvals_reviewer",
        "default_permissions",
        "permissions",
        "profile",
        "profiles",
        "projects",
        "sandbox_mode",
        "sandbox_workspace_write",
    }
)
_COMMAND_FEATURES = frozenset(
    {
        "exec_permission_approvals",
        "request_permissions_tool",
        "shell_tool",
        "unified_exec",
    }
)


@dataclass(frozen=True)
class NativeRuntimeVersions:
    """Installed distribution versions checked before app-server launch."""

    sdk: str
    cli: str


@dataclass(frozen=True)
class NativeThreadStartEvidence:
    """Security-relevant fields returned by ``thread/start``."""

    thread_id: str
    cwd: str
    sandbox_policy: Mapping[str, JsonValue]
    approval_policy: str
    ephemeral: bool
    approvals_reviewer: str | None = None


class NativeSdkTurnPort(Protocol):
    id: str

    def stream(self) -> AsyncIterator[BackendNotification]: ...

    async def interrupt(self) -> Any: ...


class NativeSdkSessionPort(Protocol):
    """The only pinned-SDK seams used by :class:`NativeCodexBackendV2`."""

    server_version: str

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def start_thread(
        self, params: Mapping[str, JsonValue]
    ) -> NativeThreadStartEvidence: ...

    async def start_turn(
        self,
        thread_id: str,
        input_items: Sequence[Mapping[str, JsonValue]],
        params: Mapping[str, JsonValue],
    ) -> NativeSdkTurnPort: ...


ApprovalHandler = Callable[[str, Mapping[str, JsonValue] | None], Mapping[str, JsonValue]]
NativeSessionFactory = Callable[
    [CodexLaunchOptions, ApprovalHandler], NativeSdkSessionPort
]
VersionResolver = Callable[[], NativeRuntimeVersions]


def installed_native_runtime_versions() -> NativeRuntimeVersions:
    """Read package metadata without importing or starting Codex."""

    try:
        sdk = importlib.metadata.version("openai-codex")
        cli = importlib.metadata.version("openai-codex-cli-bin")
    except importlib.metadata.PackageNotFoundError as exc:
        raise CodexRuntimeError(
            "openai-codex and its bundled CLI are required for native backend v2"
        ) from exc
    return NativeRuntimeVersions(sdk=sdk, cli=cli)


def _require_pinned_versions(versions: NativeRuntimeVersions) -> None:
    if versions.sdk != REQUIRED_CODEX_SDK_VERSION:
        raise CodexRuntimeError(
            "native backend requires openai-codex=="
            f"{REQUIRED_CODEX_SDK_VERSION}; observed {versions.sdk!r}"
        )
    if versions.cli != REQUIRED_CODEX_CLI_VERSION:
        raise CodexRuntimeError(
            "native backend requires openai-codex-cli-bin=="
            f"{REQUIRED_CODEX_CLI_VERSION}; observed {versions.cli!r}"
        )


def _require_server_version(value: str) -> None:
    # App-server metadata may append a build target after the semantic version.
    if not isinstance(value, str) or value.split(maxsplit=1)[0] != REQUIRED_CODEX_CLI_VERSION:
        raise CodexRuntimeError(
            "native backend app-server version differs from pinned Codex CLI 0.147.0"
        )


def _require_strict_launch(launch: CodexLaunchOptions) -> None:
    if launch.codex_bin is not None:
        raise CodexRuntimeError("native backend requires the SDK-bundled Codex CLI")
    if launch.launch_args_override is not None:
        raise CodexRuntimeError("native backend forbids launch argv overrides")
    if launch.config_overrides:
        raise CodexRuntimeError("native backend forbids process-wide config overrides")
    if launch.experimental_api is not True:
        raise CodexRuntimeError("native backend requires the pinned experimental app-server API")


def _candidate_root(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CodexRuntimeError("native candidate root differs")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise CodexRuntimeError("native candidate root must be an absolute normalized path")
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise CodexRuntimeError("native candidate root is unavailable") from None
    if (
        resolved != path
        or stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise CodexRuntimeError("native candidate root uses unsafe topology")
    return value


def workspace_write_policy(candidate_root: str) -> dict[str, JsonValue]:
    """Return the one permitted Codex wire policy for a candidate turn."""

    root = _candidate_root(candidate_root)
    return {
        "type": "workspaceWrite",
        "networkAccess": False,
        "writableRoots": [root],
        "excludeSlashTmp": True,
        "excludeTmpdirEnvVar": True,
    }


def _thread_workspace_config(
    caller_config: Mapping[str, JsonValue], candidate_root: str
) -> dict[str, Any]:
    """Bind thread defaults without accepting caller-controlled permissions."""

    config = canonical_value(caller_config)
    if not isinstance(config, dict):  # pragma: no cover - options already validate
        raise CodexRuntimeError("native thread config must be an object")
    # The policy binder records the exact positive workspace sandbox in the
    # thread options.  Reopen and consume that value here; never forward a
    # caller-controlled permission block to the SDK.
    sealed_workspace = config.pop("sandbox_workspace_write", None)
    if sealed_workspace is not None:
        expected_workspace = {
            "network_access": False,
            "writable_roots": [candidate_root],
            "exclude_slash_tmp": True,
            "exclude_tmpdir_env_var": True,
        }
        if canonical_value(sealed_workspace) != expected_workspace:
            raise CodexRuntimeError("native thread workspace commitment differs")
    forbidden = _FORBIDDEN_CALLER_CONFIG_KEYS.intersection(config)
    if forbidden:
        raise CodexRuntimeError(
            "native thread config attempts to control sandbox or permissions"
        )
    features = config.get("features", {})
    if not isinstance(features, dict):
        raise CodexRuntimeError("native thread feature config must be an object")
    for name in _COMMAND_FEATURES:
        if name in features and features[name] is not False:
            raise CodexRuntimeError("native thread config attempts to enable commands")
        features[name] = False
    config["features"] = features
    config["approval_policy"] = "never"
    config["sandbox_mode"] = "workspace-write"
    config["sandbox_workspace_write"] = {
        "network_access": False,
        "writable_roots": [candidate_root],
        "exclude_slash_tmp": True,
        "exclude_tmpdir_env_var": True,
    }
    return config


def _validate_workspace_policy(value: Any, candidate_root: str) -> None:
    observed = _plain(value)
    expected = workspace_write_policy(candidate_root)
    if observed != expected:
        raise CodexRuntimeError(
            "native workspace-write policy differs or contains additional writable roots"
        )


def _validate_thread_start_policy(value: Any) -> None:
    """Reopen SDK 0.147's canonical thread/start sandbox evidence.

    ``cwd`` is the implicit workspace root at thread/start, so the SDK reports
    an empty *additional* writableRoots list.  turn/start still receives the
    explicit one-root policy and is checked by ``_validate_workspace_policy``.
    """

    observed = _plain(value)
    expected = {
        "type": "workspaceWrite",
        "networkAccess": False,
        "writableRoots": [],
        "excludeSlashTmp": True,
        "excludeTmpdirEnvVar": True,
    }
    if observed != expected:
        raise CodexRuntimeError(
            "native thread/start policy differs or contains additional writable roots"
        )


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _plain(value.value)
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(by_alias=True, exclude_none=False, mode="json"))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CodexRuntimeError("native SDK mapping keys must be text")
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    if hasattr(value, "root"):
        return _plain(value.root)
    raise CodexRuntimeError(f"unsupported native SDK value: {type(value).__name__}")


def _wire_inputs(items: Sequence[BackendInputItem]) -> list[dict[str, JsonValue]]:
    output: list[dict[str, JsonValue]] = []
    for item in items:
        if item.kind == "text":
            output.append({"type": "text", "text": item.text or ""})
        elif item.kind in {"skill", "mention"}:
            output.append(
                {
                    "type": item.kind,
                    "name": item.name or "",
                    "path": item.path or "",
                }
            )
        else:  # pragma: no cover - BackendInputItem validates this boundary
            raise CodexRuntimeError("unsupported native Codex input item")
    return output


def _contains_command_execution(value: Any) -> bool:
    """Recognize command items even when nested in a terminal turn payload."""

    if isinstance(value, Mapping):
        return value.get("type") == "commandExecution" or any(
            _contains_command_execution(child) for child in value.values()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_command_execution(child) for child in value)
    return False


@dataclass(frozen=True)
class _ApprovalViolation:
    method: str
    thread_id: str | None
    turn_id: str | None


class _ApprovalTrap:
    """Decline approvals synchronously and retain only non-sensitive routing."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._violations: list[_ApprovalViolation] = []

    def snapshot(self) -> int:
        with self._lock:
            return len(self._violations)

    def __call__(
        self, method: str, params: Mapping[str, JsonValue] | None
    ) -> Mapping[str, JsonValue]:
        # The two approval methods known to 0.147.0 are listed explicitly, and
        # the suffix guard keeps a newly added approval surface deny-by-default.
        # Other app-server requests retain the SDK's empty-response behavior.
        if method not in _APPROVAL_METHODS and not method.endswith(
            _APPROVAL_METHOD_SUFFIX
        ):
            return {}
        params = params if isinstance(params, Mapping) else {}
        thread_id = params.get("threadId")
        turn_id = params.get("turnId")
        violation = _ApprovalViolation(
            method=method,
            thread_id=thread_id if isinstance(thread_id, str) else None,
            turn_id=turn_id if isinstance(turn_id, str) else None,
        )
        with self._lock:
            self._violations.append(violation)
        # ``decline`` is the explicit non-execution decision in app-server
        # 0.147.0.  Merely setting approvalPolicy=never is insufficient because
        # the published SDK's default low-level approval handler accepts an
        # unexpected request.
        return {"decision": "decline"}

    def require_clean_since(self, marker: int) -> None:
        with self._lock:
            rows = tuple(self._violations[marker:])
        if rows:
            raise CodexRuntimeError(
                f"native turn emitted forbidden approval request: {rows[0].method}"
            )


class _OfficialNativeSdkSession:
    """Small 0.147.0-only bridge to SDK low-level typed JSON-RPC methods."""

    def __init__(self, launch: CodexLaunchOptions, approval_handler: ApprovalHandler) -> None:
        try:
            from openai_codex import AsyncCodex, CodexConfig, __version__
            from openai_codex.api import AsyncTurnHandle
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise CodexRuntimeError("openai-codex is required for native backend v2") from exc
        if __version__ != REQUIRED_CODEX_SDK_VERSION:
            raise CodexRuntimeError("imported openai-codex version differs from package pin")
        config = CodexConfig(
            codex_bin=None,
            launch_args_override=None,
            config_overrides=(),
            cwd=launch.cwd,
            env=None if launch.env is None else dict(launch.env),
            client_name=launch.client_name,
            client_title=launch.client_title,
            client_version=launch.client_version,
            experimental_api=True,
        )
        client = AsyncCodex(config=config)
        try:
            async_client = client._client  # noqa: SLF001 - exact pinned seam
            sync_client = async_client._sync  # noqa: SLF001 - exact pinned seam
            original = sync_client._approval_handler  # noqa: SLF001
        except AttributeError as exc:
            raise CodexRuntimeError("openai-codex 0.147.0 approval seam differs") from exc
        if not callable(original):
            raise CodexRuntimeError("openai-codex approval handler seam differs")
        sync_client._approval_handler = approval_handler  # noqa: SLF001
        if sync_client._approval_handler is not approval_handler:  # noqa: SLF001
            raise CodexRuntimeError("openai-codex approval handler installation failed")
        self._client = client
        self._async_turn_handle = AsyncTurnHandle
        self.server_version = "unopened"

    async def open(self) -> None:
        await self._client.__aenter__()
        metadata = self._client.metadata
        server_info = getattr(metadata, "serverInfo", None)
        self.server_version = str(getattr(server_info, "version", None) or "unknown")

    async def close(self) -> None:
        await self._client.__aexit__(None, None, None)

    async def start_thread(
        self, params: Mapping[str, JsonValue]
    ) -> NativeThreadStartEvidence:
        # Version gating makes these two private fields a fixed compatibility
        # surface instead of an unbounded dependency on future SDK internals.
        response = await self._client._client.thread_start(  # noqa: SLF001
            dict(canonical_value(params))
        )
        document = _plain(response)
        if not isinstance(document, dict):  # pragma: no cover - typed SDK result
            raise CodexRuntimeError("native thread/start response differs")
        thread = document.get("thread")
        if not isinstance(thread, dict):
            raise CodexRuntimeError("native thread/start lacks thread evidence")
        return NativeThreadStartEvidence(
            thread_id=str(thread.get("id") or ""),
            cwd=str(document.get("cwd") or ""),
            sandbox_policy=document.get("sandbox", {}),
            approval_policy=str(document.get("approvalPolicy") or ""),
            ephemeral=thread.get("ephemeral") is True,
            approvals_reviewer=(
                str(document["approvalsReviewer"])
                if document.get("approvalsReviewer") is not None
                else None
            ),
        )

    async def start_turn(
        self,
        thread_id: str,
        input_items: Sequence[Mapping[str, JsonValue]],
        params: Mapping[str, JsonValue],
    ) -> NativeSdkTurnPort:
        wire_items = [dict(canonical_value(item)) for item in input_items]
        response = await self._client._client.turn_start(  # noqa: SLF001
            thread_id,
            wire_items,
            dict(canonical_value(params)),
        )
        turn_id = str(getattr(getattr(response, "turn", None), "id", None) or "")
        if not turn_id:
            raise CodexRuntimeError("native turn/start response lacks a turn id")
        return self._async_turn_handle(self._client, thread_id, turn_id)


def _official_session_factory(
    launch: CodexLaunchOptions, approval_handler: ApprovalHandler
) -> NativeSdkSessionPort:
    return _OfficialNativeSdkSession(launch, approval_handler)


class _NativeTurn:
    def __init__(
        self,
        handle: NativeSdkTurnPort,
        approval_trap: _ApprovalTrap,
        approval_marker: int,
    ) -> None:
        self._handle = handle
        self._approval_trap = approval_trap
        self._approval_marker = approval_marker
        self.id = str(handle.id)

    async def _interrupt(self) -> None:
        try:
            await self._handle.interrupt()
        except Exception:
            # The security result is still failure; interruption is best effort
            # after the server has already crossed an unexpected boundary.
            pass

    async def _guard(self, notification: BackendNotification | None = None) -> None:
        try:
            self._approval_trap.require_clean_since(self._approval_marker)
            if notification is not None:
                method = getattr(notification, "method", None)
                if not isinstance(method, str) or not method:
                    raise CodexRuntimeError("native Codex notification method differs")
                if method.startswith("item/commandExecution"):
                    raise CodexRuntimeError("native commandExecution is forbidden")
                payload = _plain(getattr(notification, "payload", None))
                if _contains_command_execution(payload):
                    raise CodexRuntimeError("native commandExecution is forbidden")
        except CodexRuntimeError:
            await self._interrupt()
            raise

    async def require_clean_start(self) -> None:
        await self._guard()

    async def stream(self) -> AsyncIterator[BackendNotification]:
        async for notification in self._handle.stream():
            await self._guard(notification)
            yield notification
        await self._guard()


class _NativeThread:
    def __init__(
        self,
        *,
        session: NativeSdkSessionPort,
        approval_trap: _ApprovalTrap,
        thread_id: str,
        candidate_root: str,
        model: str,
    ) -> None:
        self._session = session
        self._approval_trap = approval_trap
        self._candidate_root = candidate_root
        self._model = model
        self.id = thread_id

    async def turn(
        self,
        items: Sequence[BackendInputItem],
        options: BackendTurnOptions,
    ) -> BackendTurnPort:
        if options.sandbox is not CodexSandbox.WORKSPACE_WRITE:
            raise CodexRuntimeError("native file-change turn requires workspace-write")
        if options.cwd != self._candidate_root:
            raise CodexRuntimeError("native turn attempted a different candidate root")
        _candidate_root(options.cwd)
        if options.model is not None and options.model != self._model:
            raise CodexRuntimeError("native turn cannot change its pinned model")
        wire_items = _wire_inputs(items)
        policy = workspace_write_policy(self._candidate_root)
        params: dict[str, JsonValue] = {
            "threadId": self.id,
            "input": wire_items,
            "approvalPolicy": "never",
            "approvalsReviewer": None,
            "cwd": self._candidate_root,
            "sandboxPolicy": policy,
        }
        optional = {
            "effort": options.effort,
            "model": options.model,
            "outputSchema": (
                None
                if options.output_schema is None
                else canonical_value(options.output_schema)
            ),
            "serviceTier": options.service_tier,
            "summary": options.summary,
        }
        params.update({key: value for key, value in optional.items() if value is not None})
        _validate_workspace_policy(params["sandboxPolicy"], self._candidate_root)
        marker = self._approval_trap.snapshot()
        handle = await self._session.start_turn(self.id, wire_items, params)
        turn = _NativeTurn(handle, self._approval_trap, marker)
        await turn.require_clean_start()
        return turn


class NativeCodexBackendV2:
    """Pinned, actor-only backend for stage-gated native ``fileChange`` turns.

    The caller must prove an S3/S4/E2E policy before selecting this backend;
    read-only S1/S2 actors and all judges use the ordinary read-only runtime.
    """

    def __init__(
        self,
        launch: CodexLaunchOptions | None = None,
        *,
        session_factory: NativeSessionFactory | None = None,
        version_resolver: VersionResolver | None = None,
    ) -> None:
        self._launch = launch or CodexLaunchOptions()
        self._session_factory = session_factory or _official_session_factory
        self._version_resolver = version_resolver or installed_native_runtime_versions
        self._approval_trap = _ApprovalTrap()
        self._session: NativeSdkSessionPort | None = None
        self.sdk_version = "unopened"
        self.cli_version = "unopened"
        self.server_version = "unopened"

    async def open(self) -> None:
        if self._session is not None:
            return
        _require_strict_launch(self._launch)
        versions = self._version_resolver()
        _require_pinned_versions(versions)
        marker = self._approval_trap.snapshot()
        session = self._session_factory(self._launch, self._approval_trap)
        try:
            await session.open()
            self._approval_trap.require_clean_since(marker)
            _require_server_version(session.server_version)
        except BaseException:
            try:
                await session.close()
            finally:
                raise
        self._session = session
        self.sdk_version = versions.sdk
        self.cli_version = versions.cli
        self.server_version = session.server_version

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()

    def _require_open(self) -> NativeSdkSessionPort:
        if self._session is None:
            raise CodexRuntimeError("native Codex backend is not open")
        return self._session

    async def start_thread(self, options: CodexThreadOptions) -> BackendThreadPort:
        session = self._require_open()
        if not options.role.is_actor:
            raise CodexRuntimeError("native file-change backend is actor-only")
        if options.sandbox is not CodexSandbox.WORKSPACE_WRITE:
            raise CodexRuntimeError("native file-change thread requires workspace-write")
        if options.ephemeral is not True:
            raise CodexRuntimeError("native file-change thread must be fresh and ephemeral")
        candidate_root = _candidate_root(options.cwd)
        config = _thread_workspace_config(options.config, candidate_root)
        policy = workspace_write_policy(candidate_root)
        params: dict[str, JsonValue] = {
            "approvalPolicy": "never",
            "approvalsReviewer": None,
            "config": config,
            "cwd": candidate_root,
            "ephemeral": True,
            "model": options.model,
            "modelProvider": options.provider,
            "sandbox": "workspace-write",
            "serviceName": options.service_name,
        }
        optional = {
            "baseInstructions": options.base_instructions,
            "developerInstructions": options.developer_instructions,
            "serviceTier": options.service_tier,
        }
        params.update({key: value for key, value in optional.items() if value is not None})
        marker = self._approval_trap.snapshot()
        evidence = await session.start_thread(params)
        self._approval_trap.require_clean_since(marker)
        if (
            not evidence.thread_id
            or evidence.cwd != candidate_root
            or evidence.approval_policy != "never"
            or evidence.ephemeral is not True
        ):
            raise CodexRuntimeError("native thread/start security evidence differs")
        _validate_thread_start_policy(evidence.sandbox_policy)
        _validate_workspace_policy(policy, candidate_root)
        return _NativeThread(
            session=session,
            approval_trap=self._approval_trap,
            thread_id=evidence.thread_id,
            candidate_root=candidate_root,
            model=options.model,
        )

    async def resume_thread(
        self, thread_id: str, options: CodexThreadOptions
    ) -> BackendThreadPort:
        del thread_id, options
        self._require_open()
        raise CodexRuntimeError("native backend v2 forbids thread resume")


__all__ = [
    "NativeCodexBackendV2",
    "NativeRuntimeVersions",
    "NativeSdkSessionPort",
    "NativeSdkTurnPort",
    "NativeThreadStartEvidence",
    "REQUIRED_CODEX_CLI_VERSION",
    "REQUIRED_CODEX_SDK_VERSION",
    "installed_native_runtime_versions",
    "workspace_write_policy",
]
