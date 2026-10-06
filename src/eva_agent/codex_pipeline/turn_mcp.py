"""Authenticated candidate-scoped MCP execution for a single Codex turn.

Codex app-server launches MCP tools out of process, whereas EvaMed's
``ParallelToolRuntime`` and immutable judge workspace live in the campaign
process.  This module joins those boundaries without serializing handlers or
duplicating side effects: a private Unix socket forwards calls to the exact
in-memory execution bridge supplied by the pipeline.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile
from threading import Event, Thread
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from jsonschema import Draft202012Validator

from eva_agent.codex_runtime import CodexThreadOptions, CodexToolOffer
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value


TURN_MCP_PROTOCOL_VERSION = "2025-06-18"
TURN_MCP_SERVER_VERSION = "0.1.0"
TURN_MCP_MAXIMUM_PARALLEL_CALLS = 256
TURN_MCP_DIRECTORY_PREFIX = "eva-turn-mcp-"
TURN_MCP_SOCKET_FILENAME = "broker.sock"
TURN_MCP_UNIX_SOCKET_PATH_CAPACITY_BYTES = 108
TURN_MCP_PRIVATE_ROOT_MODE = 0o700
# Concurrent JSON-RPC requests have no explicit frontier identifier. Calls
# that arrive before any result is released are therefore coalesced for one
# short, bounded scheduling window. A sequential model call cannot enter this
# window because it is waiting for the preceding result.
TURN_MCP_FRONTIER_WINDOW_SECONDS = 0.025


class TurnMCPError(ValueError):
    """A per-turn MCP transport failed before or during execution."""


def _verified_private_temp_root(path: Path) -> tuple[Path, os.stat_result]:
    """Reopen one private root without following a final-component symlink."""

    if not path.is_absolute():
        raise TurnMCPError("turn MCP temp root must be absolute")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TurnMCPError("turn MCP temp root is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != path
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != TURN_MCP_PRIVATE_ROOT_MODE
    ):
        raise TurnMCPError(
            "turn MCP temp root must be a real uid-owned mode-0700 directory"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TurnMCPError("turn MCP temp root cannot be reopened safely") from exc
    try:
        reopened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        reopened.st_dev != metadata.st_dev
        or reopened.st_ino != metadata.st_ino
        or reopened.st_uid != metadata.st_uid
        or stat.S_IMODE(reopened.st_mode) != TURN_MCP_PRIVATE_ROOT_MODE
        or not stat.S_ISDIR(reopened.st_mode)
    ):
        raise TurnMCPError("turn MCP temp root changed while reopening")
    return resolved, metadata


def _unix_socket_preflight(temp_root: Path) -> dict[str, Any]:
    """Probe the platform tempfile name length without retaining the name."""

    directory: Path | None = None
    try:
        directory = Path(
            tempfile.mkdtemp(prefix=TURN_MCP_DIRECTORY_PREFIX, dir=str(temp_root))
        )
        if directory.parent != temp_root or not directory.name.startswith(
            TURN_MCP_DIRECTORY_PREFIX
        ):
            raise TurnMCPError("turn MCP tempfile topology differs")
        socket_path = directory / TURN_MCP_SOCKET_FILENAME
        socket_path_bytes = len(os.fsencode(str(socket_path)))
        prefix_bytes = len(os.fsencode(TURN_MCP_DIRECTORY_PREFIX))
        directory_component_bytes = len(os.fsencode(directory.name))
        random_component_bytes = directory_component_bytes - prefix_bytes
        if random_component_bytes < 1:
            raise TurnMCPError("turn MCP tempfile random component differs")
    finally:
        if directory is not None:
            try:
                directory.rmdir()
            except OSError as exc:
                raise TurnMCPError("turn MCP preflight directory cleanup failed") from exc
    passed = socket_path_bytes < TURN_MCP_UNIX_SOCKET_PATH_CAPACITY_BYTES
    document = {
        "schema": "eva.turn-mcp-unix-socket-preflight.v1",
        "temp_root": str(temp_root),
        "temp_root_uid": os.getuid(),
        "temp_root_mode": TURN_MCP_PRIVATE_ROOT_MODE,
        "temp_root_owned_by_process": True,
        "temp_root_is_symlink": False,
        "directory_prefix": TURN_MCP_DIRECTORY_PREFIX,
        "random_component_bytes": random_component_bytes,
        "socket_filename": TURN_MCP_SOCKET_FILENAME,
        "probed_socket_path_bytes": socket_path_bytes,
        "sockaddr_un_path_capacity_bytes": TURN_MCP_UNIX_SOCKET_PATH_CAPACITY_BYTES,
        "terminator_bytes": 1,
        "name_length_source": "tempfile.mkdtemp_probe_removed",
        "preflight_passed": passed,
    }
    if not passed:
        raise TurnMCPError("turn MCP Unix socket path exceeds the platform bound")
    return document


class TurnToolGroupExecutor(Protocol):
    def execute_group(
        self, calls: Sequence[tuple[str, Mapping[str, Any]]]
    ) -> Sequence[Any]: ...


def _structured(value: Any) -> Mapping[str, Any]:
    candidate = getattr(value, "structured_content", value)
    projected = canonical_value(candidate)
    if not isinstance(projected, dict):
        raise TurnMCPError("turn tool bridge output must be a JSON object")
    return projected


@dataclass(slots=True)
class _PendingCall:
    name: str
    arguments: Mapping[str, Any]
    future: asyncio.Future[Mapping[str, Any]]


class _TurnBroker:
    def __init__(
        self,
        *,
        offers: tuple[CodexToolOffer, ...],
        executor: TurnToolGroupExecutor,
        maximum_parallel_tools: int,
        temp_root: Path | None,
    ) -> None:
        if not offers or len({row.fully_qualified_name for row in offers}) != len(offers):
            raise TurnMCPError("turn MCP requires a non-empty unique tool catalog")
        servers = {row.server for row in offers}
        if len(servers) != 1:
            raise TurnMCPError("one turn MCP process may expose exactly one server")
        if not callable(getattr(executor, "execute_group", None)):
            raise TurnMCPError("turn MCP executor is unavailable")
        if (
            type(maximum_parallel_tools) is not int
            or not 1 <= maximum_parallel_tools <= TURN_MCP_MAXIMUM_PARALLEL_CALLS
        ):
            raise TurnMCPError(
                f"turn MCP parallel width must be in [1,{TURN_MCP_MAXIMUM_PARALLEL_CALLS}]"
            )
        if temp_root is not None:
            temp_root, _metadata = _verified_private_temp_root(temp_root)
        self.offers = offers
        self.server_name = next(iter(servers))
        self.executor = executor
        self.maximum = maximum_parallel_tools
        self.nonce = secrets.token_hex(32)
        self.directory = Path(
            tempfile.mkdtemp(
                prefix=TURN_MCP_DIRECTORY_PREFIX,
                dir=None if temp_root is None else str(temp_root),
            )
        )
        os.chmod(self.directory, 0o700)
        self.socket_path = self.directory / TURN_MCP_SOCKET_FILENAME
        # Linux sockaddr_un.sun_path is 108 bytes including its terminator.
        # Fail here with a stable production blocker instead of letting the
        # background loop die with a platform-dependent bind error.
        if (
            len(os.fsencode(str(self.socket_path)))
            >= TURN_MCP_UNIX_SOCKET_PATH_CAPACITY_BYTES
        ):
            self.directory.rmdir()
            raise TurnMCPError("turn MCP Unix socket path exceeds the platform bound")
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._queue: asyncio.Queue[_PendingCall] | None = None
        self._ready = Event()
        self._closed = False
        self._failure: BaseException | None = None
        self.catalog_blake3 = blake3_hex(
            tuple(row.canonical_catalog_entry() for row in offers)
        )

    def start(self) -> None:
        if self._thread is not None or self._closed:
            raise TurnMCPError("turn MCP broker lifecycle was reused")
        self._thread = Thread(target=self._thread_main, name="eva-turn-mcp-broker", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            self.close()
            raise TurnMCPError("turn MCP broker did not become ready")
        if self._failure is not None:
            failure = self._failure
            try:
                self.close()
            except TurnMCPError:
                pass
            raise TurnMCPError("turn MCP broker startup failed") from failure
        try:
            metadata = self.socket_path.lstat()
        except OSError as exc:
            self.close()
            raise TurnMCPError("turn MCP socket is unavailable") from exc
        if not stat.S_ISSOCK(metadata.st_mode):
            self.close()
            raise TurnMCPError("turn MCP endpoint is not a Unix socket")
        os.chmod(self.socket_path, 0o600)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self._failure = exc
            self._ready.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self._queue = asyncio.Queue()
        server = await asyncio.start_unix_server(self._client, path=str(self.socket_path))
        batcher = asyncio.create_task(self._batcher())
        self._ready.set()
        await self._stop.wait()
        server.close()
        await server.wait_closed()
        batcher.cancel()
        try:
            await batcher
        except asyncio.CancelledError:
            pass

    async def _client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        response: Mapping[str, Any] | None
        request_id: Any = None
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=30)
            if not line or len(line) > 16 * 1024 * 1024:
                raise TurnMCPError("turn MCP request size differs")
            envelope = json.loads(line)
            if (
                not isinstance(envelope, dict)
                or set(envelope) != {"nonce", "request"}
                or not secrets.compare_digest(str(envelope["nonce"]), self.nonce)
                or not isinstance(envelope["request"], dict)
            ):
                raise TurnMCPError("turn MCP authentication failed")
            request_id = envelope["request"].get("id")
            response = await self._dispatch(envelope["request"])
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": type(exc).__name__},
            }
        payload = json.dumps(
            canonical_value({"response": response}),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        writer.write(payload)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def _dispatch(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
            raise TurnMCPError("turn MCP JSON-RPC request differs")
        request_id = request.get("id")
        method = request["method"]
        if request_id is None:
            return None
        if method == "initialize":
            params = request.get("params")
            requested = params.get("protocolVersion") if isinstance(params, Mapping) else None
            result: Mapping[str, Any] = {
                "protocolVersion": (
                    requested if isinstance(requested, str) else TURN_MCP_PROTOCOL_VERSION
                ),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": self.server_name,
                    "version": TURN_MCP_SERVER_VERSION,
                },
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [row.mcp_transport_entry() for row in self.offers]}
        elif method == "tools/call":
            params = request.get("params")
            if not isinstance(params, Mapping):
                raise TurnMCPError("turn MCP tool parameters differ")
            name = params.get("name")
            arguments = params.get("arguments", {})
            by_name = {row.name: row for row in self.offers}
            if not isinstance(name, str) or name not in by_name or not isinstance(arguments, Mapping):
                raise TurnMCPError("turn MCP requested an unoffered tool")
            errors = tuple(Draft202012Validator(dict(by_name[name].input_schema)).iter_errors(dict(arguments)))
            if errors:
                raise TurnMCPError("turn MCP arguments violate the exact input schema")
            queue = self._queue
            if queue is None:
                raise TurnMCPError("turn MCP executor is not ready")
            future: asyncio.Future[Mapping[str, Any]] = asyncio.get_running_loop().create_future()
            await queue.put(_PendingCall(name, canonical_value(arguments), future))
            structured = await future
            result = {"content": [], "structuredContent": structured, "isError": False}
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "method not found"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    async def _batcher(self) -> None:
        assert self._queue is not None
        while True:
            first = await self._queue.get()
            pending = [first]
            # Coalesce calls already dispatched by the same model frontier.
            await asyncio.sleep(TURN_MCP_FRONTIER_WINDOW_SECONDS)
            while len(pending) < self.maximum:
                try:
                    pending.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                raw = await asyncio.to_thread(
                    self.executor.execute_group,
                    tuple((row.name, row.arguments) for row in pending),
                )
                results = tuple(raw)
                if len(results) != len(pending):
                    raise TurnMCPError("turn MCP executor result cardinality differs")
                for row, value in zip(pending, results, strict=True):
                    if not row.future.done():
                        row.future.set_result(_structured(value))
            except BaseException as exc:
                for row in pending:
                    if not row.future.done():
                        row.future.set_exception(exc)
            finally:
                for _row in pending:
                    self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if (
            self._loop is not None
            and not self._loop.is_closed()
            and self._stop is not None
        ):
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise TurnMCPError("turn MCP broker did not stop")
        try:
            if self.socket_path.exists() or self.socket_path.is_socket():
                self.socket_path.unlink()
            self.directory.rmdir()
        except OSError as exc:
            raise TurnMCPError("turn MCP temporary topology could not be removed") from exc
        if self._failure is not None:
            raise TurnMCPError("turn MCP broker failed") from self._failure


@dataclass(slots=True)
class TurnMCPTransportSession(AbstractContextManager[CodexThreadOptions]):
    """Context-managed thread options whose local broker exists only in-turn."""

    base_options: CodexThreadOptions
    broker: _TurnBroker
    proxy_python: Path
    proxy_exec_script: Path
    proxy_script: Path
    startup_timeout_seconds: float
    tool_timeout_seconds: int
    _bound: CodexThreadOptions | None = None

    def __enter__(self) -> CodexThreadOptions:
        self.broker.start()
        config = canonical_value(self.base_options.config)
        if not isinstance(config, dict) or "mcp_servers" in config:
            self.broker.close()
            raise TurnMCPError("base Codex options already contain an MCP server")
        names = [row.name for row in self.broker.offers]
        config["mcp_servers"] = {
            self.broker.server_name: {
                "command": str(self.proxy_python),
                "args": [
                    "-I",
                    "-S",
                    "-B",
                    str(self.proxy_exec_script),
                    str(self.proxy_python),
                    str(self.proxy_script),
                ],
                "cwd": self.base_options.cwd,
                "env": {
                    "EVA_TURN_MCP_SOCKET": str(self.broker.socket_path),
                    "EVA_TURN_MCP_NONCE": self.broker.nonce,
                    "EVA_TURN_MCP_MAXIMUM": str(self.broker.maximum),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                "required": True,
                "startup_timeout_sec": self.startup_timeout_seconds,
                "tool_timeout_sec": self.tool_timeout_seconds,
                "enabled_tools": names,
                # Keep the exact schemas as ordinary MCP tools.  In
                # particular they must not be hidden behind Codex tool-search
                # deferral or the code-mode namespace surface.
                "omit_tools_from": ["deferred", "code_mode"],
                # Codex's global approval mode remains deny-all.  Approve only
                # the exact candidate-bound MCP names that were already
                # schema-committed above; native tools and any newly surfaced
                # MCP name therefore remain fail-closed.
                "tools": {
                    name: {"approval_mode": "approve"}
                    for name in names
                },
                # This is an execution scheduling capability, not a claim
                # that every handler is safe.  Eva's ParallelToolRuntime still
                # serializes mutating/non-parallel-safe calls and is the
                # authoritative candidate-mutation boundary.
                "supports_parallel_tool_calls": True,
            }
        }
        self._bound = replace(self.base_options, config=config)
        return self._bound

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.broker.close()
        return False


class TurnMCPBridgeFactory:
    """Build actor and judge sessions without owning either tool runtime."""

    def __init__(
        self,
        *,
        proxy_python: str | Path,
        proxy_script: str | Path,
        proxy_exec_script: str | Path | None = None,
        temp_root: str | Path | None = None,
        maximum_parallel_calls: int = 64,
        maximum_parallel_tools: int | None = None,
        startup_timeout_seconds: float = 10.0,
        tool_timeout_seconds: int = 900,
    ) -> None:
        self.proxy_python = self._file(proxy_python, executable=True, label="proxy Python")
        self.proxy_script = self._file(proxy_script, executable=False, label="proxy script")
        default_exec = self.proxy_script.with_name("turn_mcp_exec.py")
        self.proxy_exec_script = self._file(
            default_exec if proxy_exec_script is None else proxy_exec_script,
            executable=False,
            label="proxy exec boundary",
        )
        raw_temp_root = None if temp_root is None else Path(temp_root)
        if raw_temp_root is None:
            raise TurnMCPError("turn MCP private temp root is required")
        self.temp_root, _root_metadata = _verified_private_temp_root(raw_temp_root)
        unix_socket_preflight = _unix_socket_preflight(self.temp_root)
        if maximum_parallel_tools is not None:
            if maximum_parallel_calls != 64 and maximum_parallel_calls != maximum_parallel_tools:
                raise TurnMCPError("turn MCP parallel width aliases differ")
            maximum_parallel_calls = maximum_parallel_tools
        if (
            type(maximum_parallel_calls) is not int
            or not 1 <= maximum_parallel_calls <= TURN_MCP_MAXIMUM_PARALLEL_CALLS
        ):
            raise TurnMCPError(
                f"turn MCP parallel width must be in [1,{TURN_MCP_MAXIMUM_PARALLEL_CALLS}]"
            )
        if type(startup_timeout_seconds) not in {int, float} or not 0 < startup_timeout_seconds <= 120:
            raise TurnMCPError("turn MCP startup timeout differs")
        if type(tool_timeout_seconds) is not int or not 1 <= tool_timeout_seconds <= 86_400:
            raise TurnMCPError("turn MCP tool timeout differs")
        self.maximum = maximum_parallel_calls
        self.startup_timeout = float(startup_timeout_seconds)
        self.tool_timeout = tool_timeout_seconds
        metadata_core = {
            "schema": "eva.turn-mcp-bridge-launch.v1",
            "protocol_version": TURN_MCP_PROTOCOL_VERSION,
            "server_version": TURN_MCP_SERVER_VERSION,
            "proxy_python": str(self.proxy_python),
            "proxy_python_blake3": blake3_bytes(self.proxy_python.read_bytes()),
            "proxy_exec_script": str(self.proxy_exec_script),
            "proxy_exec_script_blake3": blake3_bytes(
                self.proxy_exec_script.read_bytes()
            ),
            "proxy_script": str(self.proxy_script),
            "proxy_script_blake3": blake3_bytes(self.proxy_script.read_bytes()),
            "maximum_parallel_calls": self.maximum,
            "startup_timeout_seconds": self.startup_timeout,
            "tool_timeout_seconds": self.tool_timeout,
            "final_child_environment_names": (
                "EVA_TURN_MCP_MAXIMUM",
                "EVA_TURN_MCP_NONCE",
                "EVA_TURN_MCP_SOCKET",
                "LANG",
                "LC_ALL",
                "PATH",
                "TZ",
            ),
            "inherited_parent_environment": False,
            "environment_values_recorded": False,
            "turn_nonce_recorded": False,
            "unix_socket_preflight": unix_socket_preflight,
        }
        self.launch_blake3 = blake3_hex(metadata_core)
        self._public_metadata = MappingProxyType(
            {**metadata_core, "launch_blake3": self.launch_blake3}
        )

    @staticmethod
    def _file(value: str | Path, *, executable: bool, label: str) -> Path:
        path = Path(value)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise TurnMCPError(f"{label} must be an absolute real file")
        resolved = path.resolve(strict=True)
        if executable and not os.access(resolved, os.X_OK):
            raise TurnMCPError(f"{label} is not executable")
        return resolved

    def public_metadata(self) -> Mapping[str, Any]:
        """Return the value-free launch commitment used by campaign recipes."""

        return self._public_metadata

    def _open(
        self,
        options: CodexThreadOptions,
        executor: TurnToolGroupExecutor,
    ) -> TurnMCPTransportSession:
        if not isinstance(options, CodexThreadOptions):
            raise TurnMCPError("turn MCP base options differ")
        broker = _TurnBroker(
            offers=options.offered_tools,
            executor=executor,
            maximum_parallel_tools=self.maximum,
            temp_root=self.temp_root,
        )
        return TurnMCPTransportSession(
            base_options=options,
            broker=broker,
            proxy_python=self.proxy_python,
            proxy_exec_script=self.proxy_exec_script,
            proxy_script=self.proxy_script,
            startup_timeout_seconds=self.startup_timeout,
            tool_timeout_seconds=self.tool_timeout,
        )

    def open_actor(
        self,
        options: CodexThreadOptions,
        bridge: TurnToolGroupExecutor,
    ) -> TurnMCPTransportSession:
        return self._open(options, bridge)

    def open_judge(
        self,
        options: CodexThreadOptions,
        bridge: TurnToolGroupExecutor,
    ) -> TurnMCPTransportSession:
        return self._open(options, bridge)


__all__ = [
    "TURN_MCP_DIRECTORY_PREFIX",
    "TURN_MCP_PRIVATE_ROOT_MODE",
    "TURN_MCP_PROTOCOL_VERSION",
    "TURN_MCP_SERVER_VERSION",
    "TURN_MCP_FRONTIER_WINDOW_SECONDS",
    "TURN_MCP_MAXIMUM_PARALLEL_CALLS",
    "TURN_MCP_SOCKET_FILENAME",
    "TURN_MCP_UNIX_SOCKET_PATH_CAPACITY_BYTES",
    "TurnMCPBridgeFactory",
    "TurnMCPError",
    "TurnMCPTransportSession",
    "TurnToolGroupExecutor",
]
