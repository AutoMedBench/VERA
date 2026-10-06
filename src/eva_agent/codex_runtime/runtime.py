"""Shared, privacy-preserving Codex actor/judge runtime.

The outer campaign orchestrator owns candidate concurrency.  This module adds
no worker semaphore, retry loop, or model-specific agent loop: Codex app-server
owns one turn's inner loop, including concurrent tool calls.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import os
from pathlib import Path
import re
import stat
from threading import RLock
from typing import Any, Iterator, Mapping, Sequence

from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
    is_blake3,
)
from eva_agent.pipeline.ids import RandomUUIDFactory

from .backend import (
    BackendInputItem,
    BackendThreadPort,
    BackendTurnOptions,
    CodexBackendPort,
)
from .contracts import (
    CodexEvent,
    CodexRole,
    CodexRuntimeError,
    CodexSandbox,
    CodexThreadHandle,
    CodexThreadOptions,
    CodexToolCall,
    CodexTurnInput,
    CodexTurnReceipt,
    RuntimeIdFactory,
    codex_core_mcp_resource_operation,
)


_PRIVATE_KEYS = frozenset(
    {
        "answerkey",
        "expectedanswer",
        "goldanswer",
        "groundtruth",
        "judgeonlycontext",
        "judgeonlyreference",
        "judgereference",
        "privaterubric",
        "privaterubricreference",
        "referenceanswer",
    }
)
_PRIVATE_TEXT = re.compile(
    r"\b(?:private[ _-]?rubric|ground[ _-]?truth|gold[ _-]?answer|"
    r"answer[ _-]?key|judge[ _-]?only[ _-]?(?:context|reference)|"
    r"reference[ _-]?answer)\b",
    re.IGNORECASE,
)
_TOOL_ITEM_TYPES = frozenset(
    {
        "collabAgentToolCall",
        "commandExecution",
        "dynamicToolCall",
        "fileChange",
        "imageGeneration",
        "imageView",
        "mcpToolCall",
        "webSearch",
    }
)


@dataclass
class _ThreadState:
    handle: CodexThreadHandle
    options: CodexThreadOptions
    backend_thread: BackendThreadPort
    construction_permit: object | None = None


@dataclass
class _ConstructionPermit:
    options_blake3: str
    input_blake3: str
    binding_blake3: str
    bound: bool = False


_CONSTRUCTION_POLICY_SENTINEL = object()
_ACTIVE_CONSTRUCTION_PERMIT: ContextVar[tuple[object, object] | None] = ContextVar(
    "eva_active_construction_input_permit", default=None
)
_CONSTRUCTION_INPUT_POLICY_CORE = {
    "schema": "eva.codex-construction-input-policy.v1",
    "scope": "exact-frozen-construction-request",
    "actor_public_text_private_lexeme_exemption": True,
    "actor_private_projection_exemption": False,
    "judge_only_context_allowed": False,
    "fresh_thread_required": True,
    "workspace_mode": "read-only",
    "offered_tool_count": 0,
    "provider_config_keys_bound": True,
    "provider_config_values_bound": False,
    "one_shot": True,
}
_CONSTRUCTION_INPUT_POLICY_BLAKE3 = blake3_hex(_CONSTRUCTION_INPUT_POLICY_CORE)


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reject_private_projection(value: Any, *, path: str = "actor_input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _normalized_key(str(key)) in _PRIVATE_KEYS:
                raise CodexRuntimeError(f"actor projection contains private field at {path}.{key}")
            _reject_private_projection(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _reject_private_projection(child, path=f"{path}[{index}]")


def _to_plain(value: Any) -> Any:
    """Normalize SDK/Pydantic/dataclass notifications into canonical JSON."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _to_plain(value.value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(by_alias=True, exclude_none=False, mode="json")
        return _to_plain(dumped)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CodexRuntimeError("Codex notification mapping keys must be strings")
        return {key: _to_plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_plain(child) for child in value]
    if hasattr(value, "root"):
        return _to_plain(value.root)
    raise CodexRuntimeError(f"unsupported Codex notification value: {type(value).__name__}")


def _route(payload: Mapping[str, Any], fallback_thread: str, fallback_turn: str) -> tuple[str, str]:
    thread_id = payload.get("threadId", payload.get("thread_id", fallback_thread))
    turn_id = payload.get("turnId", payload.get("turn_id"))
    turn = payload.get("turn")
    if turn_id is None and isinstance(turn, Mapping):
        turn_id = turn.get("id")
    if turn_id is None:
        turn_id = fallback_turn
    if not isinstance(thread_id, str) or not isinstance(turn_id, str):
        raise CodexRuntimeError("Codex notification routing differs")
    if thread_id != fallback_thread or turn_id != fallback_turn:
        raise CodexRuntimeError("Codex notification escaped its thread/turn route")
    return thread_id, turn_id


def _redact_event_value(value: Any) -> tuple[Any, bool]:
    """Remove hidden reasoning and echoed user input from committed events."""

    if isinstance(value, list):
        redacted = False
        output = []
        for child in value:
            projected, changed = _redact_event_value(child)
            output.append(projected)
            redacted = redacted or changed
        return output, redacted
    if not isinstance(value, Mapping):
        return value, False
    item_type = value.get("type")
    if item_type in {"reasoning", "userMessage", "hookPrompt"}:
        projected = {
            key: value[key]
            for key in ("id", "type", "phase", "status")
            if key in value
        }
        projected["contentRedacted"] = True
        return projected, True
    output: dict[str, Any] = {}
    redacted = False
    for key, child in value.items():
        normalized = _normalized_key(key)
        if "reasoning" in normalized or normalized in {
            "encryptedcontent",
            "clientusermessage",
        }:
            output[key] = {"contentRedacted": True}
            redacted = True
            continue
        projected, changed = _redact_event_value(child)
        output[key] = projected
        redacted = redacted or changed
    return output, redacted


def _event_payload(method: str, payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    if "reasoning" in method.casefold():
        route = {
            key: payload[key]
            for key in ("threadId", "thread_id", "turnId", "turn_id", "itemId", "item_id")
            if key in payload
        }
        route["contentRedacted"] = True
        return route, True
    projected, changed = _redact_event_value(payload)
    if not isinstance(projected, Mapping):  # pragma: no cover - payload contract above
        raise CodexRuntimeError("Codex event payload must remain an object")
    return projected, changed


def _item(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = payload.get("item")
    return value if isinstance(value, Mapping) else None


def _item_type(item: Mapping[str, Any]) -> str:
    value = item.get("type")
    return value if isinstance(value, str) else ""


def _is_tool(item: Mapping[str, Any]) -> bool:
    kind = _item_type(item)
    return kind in _TOOL_ITEM_TYPES or kind.endswith("ToolCall")


def _text(value: Any, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _tool_name(item: Mapping[str, Any]) -> str:
    kind = _item_type(item)
    if kind in {"mcpToolCall", "dynamicToolCall"}:
        return _text(item.get("tool"), kind)
    return kind


def _tool_arguments(item: Mapping[str, Any]) -> Any:
    kind = _item_type(item)
    if "arguments" in item:
        return item["arguments"]
    if kind == "commandExecution":
        return {
            key: item[key]
            for key in ("command", "commandActions", "cwd")
            if key in item
        }
    if kind == "fileChange":
        return {"changes": item.get("changes", [])}
    if kind == "webSearch":
        return {key: item[key] for key in ("query", "action") if key in item}
    if kind in {"imageView", "imageGeneration"}:
        return {key: item[key] for key in ("path", "revisedPrompt") if key in item}
    return {}


def _tool_output(item: Mapping[str, Any]) -> Any:
    kind = _item_type(item)
    keys_by_type = {
        "mcpToolCall": ("result", "error", "durationMs"),
        "dynamicToolCall": ("contentItems", "success", "durationMs"),
        "commandExecution": ("aggregatedOutput", "exitCode", "durationMs"),
        "fileChange": ("status",),
        "webSearch": ("results",),
        "imageGeneration": ("result", "savedPath", "status"),
        "imageView": ("path",),
        "collabAgentToolCall": ("agentsStates", "status"),
    }
    return {key: item[key] for key in keys_by_type.get(kind, ()) if key in item}


def _status(item: Mapping[str, Any]) -> str:
    value = item.get("status")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("type"), str):
        return value["type"]
    return "completed"


def _turn_status(payload: Mapping[str, Any]) -> str:
    turn = payload.get("turn")
    if isinstance(turn, Mapping):
        value = turn.get("status")
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for key in ("type", "status"):
                if isinstance(value.get(key), str):
                    return value[key]
    return "completed"


def _catalog_text(options: CodexThreadOptions) -> str | None:
    if not options.offered_tools:
        return None
    sidecar = {
        "schema": "eva.codex-tool-catalog-sidecar.v1",
        "note": "Protocol mapping only; each inputSchema value is canonical and immutable.",
        "tools": tuple(tool.mcp_transport_entry() for tool in options.offered_tools),
        "metadata": tuple(tool.sidecar_metadata_entry() for tool in options.offered_tools),
    }
    return canonical_json_bytes(sidecar).decode("utf-8")


def _verified_skill_body(skill: Any) -> str:
    """Reopen committed skill bytes without an unavailable filesystem MCP."""

    path = Path(skill.path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CodexRuntimeError("selected skill input is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or not 1 <= metadata.st_size <= 4 * 1024 * 1024:
            raise CodexRuntimeError("selected skill input topology differs")
        payload = b""
        while len(payload) < metadata.st_size:
            chunk = os.read(descriptor, metadata.st_size - len(payload))
            if not chunk:
                raise CodexRuntimeError("selected skill input was truncated")
            payload += chunk
        if os.read(descriptor, 1):
            raise CodexRuntimeError("selected skill input size differs")
    finally:
        os.close(descriptor)
    if blake3_bytes(payload) != skill.content_blake3:
        raise CodexRuntimeError("selected skill input digest differs")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CodexRuntimeError("selected skill input is not UTF-8") from exc


def _skill_text(value: CodexTurnInput) -> str | None:
    if not value.skills:
        return None
    sidecar = {
        "schema": "eva.codex-skill-mount-sidecar.v1",
        "note": "Selected skill bodies are embedded; no filesystem or MCP resource read is offered.",
        "skills": tuple(
            {**skill.catalog_entry(), "content": _verified_skill_body(skill)}
            for skill in value.skills
        ),
    }
    return canonical_json_bytes(sidecar).decode("utf-8")


def _input_items(options: CodexThreadOptions, value: CodexTurnInput) -> tuple[BackendInputItem, ...]:
    items = [BackendInputItem(kind="text", text=value.public_text)]
    if value.public_context:
        items.append(
            BackendInputItem(
                kind="text",
                text=(
                    "EVA public context; preserve all canonical keys and values exactly:\n"
                    + canonical_json_bytes(value.public_context).decode("utf-8")
                ),
            )
        )
    catalog = _catalog_text(options)
    if catalog is not None:
        items.append(BackendInputItem(kind="text", text=catalog))
    skill_catalog = _skill_text(value)
    if skill_catalog is not None:
        items.append(BackendInputItem(kind="text", text=skill_catalog))
    items.extend(
        BackendInputItem(kind="mention", name=mention.name, path=mention.path)
        for mention in value.mentions
    )
    if value.judge_only_context is not None:
        items.append(
            BackendInputItem(
                kind="text",
                text=(
                    "EVA judge-only context; never expose this material to an actor:\n"
                    + canonical_json_bytes(value.judge_only_context).decode("utf-8")
                ),
            )
        )
    return tuple(items)


def _logical_input(options: CodexThreadOptions, value: CodexTurnInput) -> Mapping[str, Any]:
    return {
        "role": options.role,
        "public_text": value.public_text,
        "public_context": value.public_context,
        "judge_only_context": value.judge_only_context,
        "offered_tools": tuple(tool.canonical_catalog_entry() for tool in options.offered_tools),
        "offered_tool_metadata": tuple(
            tool.sidecar_metadata_entry() for tool in options.offered_tools
        ),
        "skills": tuple(skill.catalog_entry() for skill in value.skills),
        "mentions": tuple({"name": item.name, "path": item.path} for item in value.mentions),
        "output_schema": value.output_schema,
    }


def _construction_options_blake3(options: CodexThreadOptions) -> str:
    """Commit public options while leaving signed route values gateway-owned."""

    return blake3_hex(
        {
            "role": options.role,
            "model": options.model,
            "provider": options.provider,
            "cwd": options.cwd,
            "sandbox": options.sandbox,
            "config_keys": options.config_keys,
            "offered_tools": tuple(
                {
                    "canonical": tool.canonical_catalog_entry(),
                    "sidecar": tool.sidecar_metadata_entry(),
                }
                for tool in options.offered_tools
            ),
            "base_instructions_blake3": (
                None
                if options.base_instructions is None
                else blake3_hex(options.base_instructions)
            ),
            "developer_instructions_blake3": (
                None
                if options.developer_instructions is None
                else blake3_hex(options.developer_instructions)
            ),
            "ephemeral": options.ephemeral,
            "service_name": options.service_name,
            "service_tier": options.service_tier,
        }
    )


def _construction_input_blake3(
    options: CodexThreadOptions, value: CodexTurnInput
) -> str:
    return blake3_hex(
        {
            "logical_input": _logical_input(options, value),
            "sandbox": value.sandbox,
            "model": value.model,
            "effort": value.effort,
            "summary": value.summary,
            "service_tier": value.service_tier,
        }
    )


class _ExactConstructionInputPolicy:
    """One-shot capability for an exact frozen construction turn.

    The type and mint are deliberately private to the runtime module and are
    not re-exported by :mod:`eva_agent.codex_runtime`.  Production
    construction owns the sole policy instance; ordinary rollout and judge
    compositions construct :class:`CodexRuntime` without it.  The active
    opaque capability is context-local and is consumed by one fresh thread.
    """

    __slots__ = ("_lock", "_permits")

    def __init__(self, sentinel: object) -> None:
        if sentinel is not _CONSTRUCTION_POLICY_SENTINEL:
            raise CodexRuntimeError("construction input policy authority differs")
        self._lock = RLock()
        self._permits: dict[object, _ConstructionPermit] = {}

    @property
    def policy_blake3(self) -> str:
        return _CONSTRUCTION_INPUT_POLICY_BLAKE3

    @contextmanager
    def authorize(
        self,
        *,
        options: CodexThreadOptions,
        turn_input: CodexTurnInput,
        request_blake3: str,
        source_request_blake3: str,
        source_phase_request_blake3: str,
        source_output_schema_blake3: str,
    ) -> Iterator[None]:
        bindings = {
            "request_blake3": request_blake3,
            "source_request_blake3": source_request_blake3,
            "source_phase_request_blake3": source_phase_request_blake3,
            "source_output_schema_blake3": source_output_schema_blake3,
        }
        if any(not is_blake3(value) for value in bindings.values()):
            raise CodexRuntimeError("construction input binding digest differs")
        if not (
            options.role.is_actor
            and options.sandbox is CodexSandbox.READ_ONLY
            and not options.offered_tools
            and options.ephemeral
            and options.service_name == "evamed-codex-premium-construction"
            and turn_input.judge_only_context is None
            and turn_input.sandbox in {None, CodexSandbox.READ_ONLY}
            and turn_input.model in {None, options.model}
            and not turn_input.skills
            and not turn_input.mentions
            and turn_input.output_schema is not None
            and blake3_hex(turn_input.output_schema) == source_output_schema_blake3
        ):
            raise CodexRuntimeError("construction input capability boundary differs")
        _reject_private_projection(turn_input.public_context)
        _reject_private_projection(
            turn_input.output_schema, path="actor_output_schema"
        )
        if _ACTIVE_CONSTRUCTION_PERMIT.get() is not None:
            raise CodexRuntimeError("construction input capability cannot be nested")
        token = object()
        permit = _ConstructionPermit(
            options_blake3=_construction_options_blake3(options),
            input_blake3=_construction_input_blake3(options, turn_input),
            binding_blake3=blake3_hex(bindings),
        )
        with self._lock:
            self._permits[token] = permit
        context_token = _ACTIVE_CONSTRUCTION_PERMIT.set((self, token))
        completed = False
        try:
            yield
            completed = True
        finally:
            _ACTIVE_CONSTRUCTION_PERMIT.reset(context_token)
            with self._lock:
                remaining = self._permits.pop(token, None)
            if completed and remaining is not None:
                raise CodexRuntimeError(
                    "construction input capability was not consumed by its runtime"
                )

    def bind_fresh_thread(self, options: CodexThreadOptions) -> object | None:
        active = _ACTIVE_CONSTRUCTION_PERMIT.get()
        if active is None:
            return None
        authority, token = active
        if authority is not self:
            raise CodexRuntimeError("construction input capability authority differs")
        with self._lock:
            permit = self._permits.get(token)
            if (
                permit is None
                or permit.bound
                or permit.options_blake3 != _construction_options_blake3(options)
            ):
                raise CodexRuntimeError("construction thread binding differs")
            permit.bound = True
        return token

    def reject_active_resume(self) -> None:
        active = _ACTIVE_CONSTRUCTION_PERMIT.get()
        if active is not None and active[0] is self:
            raise CodexRuntimeError("construction capability requires a fresh thread")

    def consume(
        self,
        token: object,
        *,
        options: CodexThreadOptions,
        turn_input: CodexTurnInput,
    ) -> None:
        active = _ACTIVE_CONSTRUCTION_PERMIT.get()
        if active != (self, token):
            raise CodexRuntimeError("construction input capability context differs")
        with self._lock:
            permit = self._permits.get(token)
            if not (
                permit is not None
                and permit.bound
                and permit.options_blake3 == _construction_options_blake3(options)
                and permit.input_blake3
                == _construction_input_blake3(options, turn_input)
            ):
                raise CodexRuntimeError("construction input capability binding differs")
            del self._permits[token]


def _new_exact_construction_input_policy() -> _ExactConstructionInputPolicy:
    """Mint the private policy used only by production construction."""

    return _ExactConstructionInputPolicy(_CONSTRUCTION_POLICY_SENTINEL)


class CodexRuntime:
    """One SDK/app-server compatibility runtime for actor and judge cohorts."""

    def __init__(
        self,
        backend: CodexBackendPort,
        *,
        id_factory: RuntimeIdFactory | None = None,
        _construction_input_policy: _ExactConstructionInputPolicy | None = None,
    ) -> None:
        if _construction_input_policy is not None and not isinstance(
            _construction_input_policy, _ExactConstructionInputPolicy
        ):
            raise CodexRuntimeError("construction input policy differs")
        self._backend = backend
        self._ids = id_factory or RandomUUIDFactory()
        self._construction_input_policy = _construction_input_policy
        self._threads: dict[str, _ThreadState] = {}
        self._open = False

    async def __aenter__(self) -> "CodexRuntime":
        await self.open()
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()

    async def open(self) -> None:
        if not self._open:
            await self._backend.open()
            self._open = True

    async def close(self) -> None:
        if self._open:
            await self._backend.close()
            self._open = False
            self._threads.clear()

    def _require_open(self) -> None:
        if not self._open:
            raise CodexRuntimeError("Codex runtime is not open")

    async def start_thread(self, options: CodexThreadOptions) -> CodexThreadHandle:
        self._require_open()
        permit = (
            None
            if self._construction_input_policy is None
            else self._construction_input_policy.bind_fresh_thread(options)
        )
        backend_thread = await self._backend.start_thread(options)
        handle = CodexThreadHandle(
            runtime_thread_id=self._ids.new("codex-thread"),
            thread_id=str(backend_thread.id),
            role=options.role,
            model=options.model,
            provider=options.provider,
            cwd=options.cwd,
            sandbox=options.sandbox,
            config_keys=options.config_keys,
            resumed=False,
        )
        self._threads[handle.runtime_thread_id] = _ThreadState(
            handle, options, backend_thread, permit
        )
        return handle

    async def resume_thread(
        self, thread_id: str, options: CodexThreadOptions
    ) -> CodexThreadHandle:
        self._require_open()
        if not isinstance(thread_id, str) or not thread_id:
            raise CodexRuntimeError("Codex thread id is required for resume")
        if self._construction_input_policy is not None:
            self._construction_input_policy.reject_active_resume()
        backend_thread = await self._backend.resume_thread(thread_id, options)
        if str(backend_thread.id) != thread_id:
            raise CodexRuntimeError("Codex resumed a different upstream thread")
        handle = CodexThreadHandle(
            runtime_thread_id=self._ids.new("codex-thread-resume"),
            thread_id=thread_id,
            role=options.role,
            model=options.model,
            provider=options.provider,
            cwd=options.cwd,
            sandbox=options.sandbox,
            config_keys=options.config_keys,
            resumed=True,
        )
        self._threads[handle.runtime_thread_id] = _ThreadState(handle, options, backend_thread)
        return handle

    def _state(self, handle: CodexThreadHandle) -> _ThreadState:
        state = self._threads.get(handle.runtime_thread_id)
        if state is None or state.handle != handle:
            raise CodexRuntimeError("unknown or altered Codex thread handle")
        return state

    def _validate_input(self, state: _ThreadState, value: CodexTurnInput) -> None:
        if state.options.role.is_actor:
            if value.judge_only_context is not None:
                raise CodexRuntimeError("actor cannot receive judge-only context")
            private_text_match = _PRIVATE_TEXT.search(value.public_text) is not None
            if private_text_match and state.construction_permit is None:
                raise CodexRuntimeError("actor public text references private judging material")
            _reject_private_projection(value.public_context)
            if value.output_schema is not None:
                _reject_private_projection(value.output_schema, path="actor_output_schema")
            if state.construction_permit is not None:
                if self._construction_input_policy is None:  # pragma: no cover - state invariant
                    raise CodexRuntimeError("construction input policy is unavailable")
                if state.handle.resumed:
                    raise CodexRuntimeError("construction capability requires a fresh thread")
                self._construction_input_policy.consume(
                    state.construction_permit,
                    options=state.options,
                    turn_input=value,
                )
        selected = tuple(skill.skill_id for skill in value.skills)
        if len(selected) != len(set(selected)):
            raise CodexRuntimeError("selected skill IDs must be unique")

    async def run_turn(
        self, handle: CodexThreadHandle, value: CodexTurnInput
    ) -> CodexTurnReceipt:
        """Run one streamed app-server turn and seal a content-verifiable receipt."""

        self._require_open()
        state = self._state(handle)
        self._validate_input(state, value)
        sandbox = value.sandbox or handle.sandbox
        items = _input_items(state.options, value)
        turn = await state.backend_thread.turn(
            items,
            BackendTurnOptions(
                cwd=handle.cwd,
                sandbox=sandbox,
                model=value.model,
                effort=value.effort,
                output_schema=value.output_schema,
                summary=value.summary,
                service_tier=value.service_tier,
            ),
        )
        turn_id = str(turn.id)
        if not turn_id:
            raise CodexRuntimeError("Codex turn id is required")

        events: list[CodexEvent] = []
        tool_states: dict[str, dict[str, Any]] = {}
        usage: Mapping[str, Any] = {}
        final_response: str | None = None
        fallback_response: str | None = None
        status: str | None = None
        active_tools: set[str] = set()
        maximum = 0
        offered_names = {tool.fully_qualified_name for tool in state.options.offered_tools}

        async for notification in turn.stream():
            method = getattr(notification, "method", None)
            if not isinstance(method, str) or not method:
                raise CodexRuntimeError("Codex notification method differs")
            raw = _to_plain(getattr(notification, "payload", None))
            if not isinstance(raw, Mapping):
                raise CodexRuntimeError("Codex notification payload must be an object")
            thread_route, turn_route = _route(raw, handle.thread_id, turn_id)
            projected, redacted = _event_payload(method, raw)
            sequence = len(events)
            event_id = self._ids.new("codex-event")
            event_core = {
                "event_id": event_id,
                "sequence": sequence,
                "method": method,
                "thread_id": thread_route,
                "turn_id": turn_route,
                "payload": projected,
                "content_redacted": redacted,
            }
            events.append(CodexEvent(**event_core, event_blake3=blake3_hex(event_core)))

            current = _item(raw)
            if method in {"item/started", "item/completed"} and current is not None and _is_tool(current):
                upstream_id = current.get("id")
                if not isinstance(upstream_id, str) or not upstream_id:
                    raise CodexRuntimeError("Codex tool item id differs")
                state_row = tool_states.setdefault(
                    upstream_id,
                    {"first": sequence, "lifecycle": [], "item": current},
                )
                state_row["lifecycle"].append(method)
                state_row["item"] = current
                if method == "item/started":
                    active_tools.add(upstream_id)
                    maximum = max(maximum, len(active_tools))
                else:
                    if upstream_id not in active_tools and maximum == 0:
                        maximum = 1
                    active_tools.discard(upstream_id)

            if method == "item/completed" and current is not None and _item_type(current) == "agentMessage":
                candidate = current.get("text")
                if isinstance(candidate, str):
                    fallback_response = candidate
                    if current.get("phase") in {"final_answer", "finalAnswer"}:
                        final_response = candidate
            if method == "thread/tokenUsage/updated":
                candidate_usage = raw.get("tokenUsage", raw.get("token_usage"))
                if isinstance(candidate_usage, Mapping):
                    usage = candidate_usage
            if method == "turn/completed":
                status = _turn_status(raw)

        if status is None or not events or events[-1].method != "turn/completed":
            raise CodexRuntimeError("Codex stream ended without terminal turn/completed")
        if active_tools:
            raise CodexRuntimeError("Codex turn completed with active tool calls")

        tool_calls: list[CodexToolCall] = []
        for upstream_id, row in sorted(tool_states.items(), key=lambda pair: pair[1]["first"]):
            lifecycle = tuple(row["lifecycle"])
            if lifecycle[-1] != "item/completed":
                raise CodexRuntimeError("Codex tool call lacks item/completed")
            current = row["item"]
            kind = _item_type(current)
            server = _text(current.get("server"), "") if kind == "mcpToolCall" else None
            mcp_tool = _text(current.get("tool"), "") if kind == "mcpToolCall" else None
            fqn = f"{server}/{mcp_tool}" if server is not None and mcp_tool is not None else None
            if (
                fqn is not None
                and fqn not in offered_names
                and codex_core_mcp_resource_operation(
                    server=server,
                    tool=mcp_tool,
                    offered_mcp_tool_names=offered_names,
                )
                is None
            ):
                raise CodexRuntimeError(f"Codex invoked unoffered MCP tool: {fqn}")
            call_id = self._ids.new("codex-tool-call")
            core = {
                "tool_call_id": call_id,
                "upstream_item_id": upstream_id,
                "tool_type": kind,
                "name": _tool_name(current),
                "mcp_server": server,
                "mcp_tool": mcp_tool,
                "fully_qualified_name": fqn,
                "status": _status(current),
                "arguments": canonical_value(_tool_arguments(current)),
                "output": canonical_value(_tool_output(current)),
                "lifecycle": lifecycle,
                "first_event_sequence": row["first"],
            }
            tool_calls.append(CodexToolCall(**core, receipt_blake3=blake3_hex(core)))

        logical_input = _logical_input(state.options, value)
        selected_ids = tuple(skill.skill_id for skill in value.skills)
        selected_catalog = tuple(skill.catalog_entry() for skill in value.skills)
        offered_catalog = tuple(
            tool.canonical_catalog_entry() for tool in state.options.offered_tools
        )
        receipt_id = self._ids.new("codex-receipt")
        runtime_turn_id = self._ids.new("codex-turn")
        core = {
            "schema": "eva.codex-turn-receipt.v1",
            "receipt_id": receipt_id,
            "runtime_thread_id": handle.runtime_thread_id,
            "runtime_turn_id": runtime_turn_id,
            "thread_id": handle.thread_id,
            "turn_id": turn_id,
            "role": handle.role,
            "model": value.model or handle.model,
            "provider": handle.provider,
            "sandbox": sandbox,
            "thread_resumed": handle.resumed,
            "visibility": "actor-public" if handle.role.is_actor else "judge-only",
            "status": status,
            "final_response": final_response if final_response is not None else fallback_response,
            "events": tuple(events),
            "tool_calls": tuple(tool_calls),
            "selected_skill_ids": selected_ids,
            "selected_skill_catalog_blake3": blake3_hex(selected_catalog),
            "offered_mcp_tool_names": tuple(
                tool.fully_qualified_name for tool in state.options.offered_tools
            ),
            "offered_tool_schema_blake3": blake3_hex(offered_catalog),
            "max_parallelism_observed": maximum if tool_calls else 0,
            "parallel_tool_calls_supported": True,
            "usage": canonical_value(usage),
            "input_blake3": blake3_hex(logical_input),
            "config_keys": handle.config_keys,
            "config_values_recorded": False,
            "input_payload_recorded": False,
            "sdk_version": self._backend.sdk_version,
            "server_version": self._backend.server_version,
        }
        return CodexTurnReceipt(**core, receipt_blake3=blake3_hex(core))


__all__ = ["CodexRuntime"]
