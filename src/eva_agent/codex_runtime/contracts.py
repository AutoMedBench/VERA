"""Immutable contracts for the shared ``evamed-codex`` runtime.

The Codex app-server owns provider interaction and the inner agent/tool loop.
These contracts only describe the policy boundary and the evidence captured by
EVA-Agent around one app-server turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from eva_agent.pipeline.contracts import JsonValue, freeze_json, uuid_text
from eva_agent.pipeline.digests import blake3_hex, is_blake3


class CodexRuntimeError(ValueError):
    """A Codex runtime boundary failed closed."""


# Codex exposes these MCP protocol helpers alongside configured MCP servers.
# They are app-server control-plane calls, not EvaMed domain tools, and must
# therefore never be added to (or confused with) the canonical tool catalog.
# The runtime retains their receipts so a weak model's attempted use remains a
# scoreable part of the trajectory.  The pipeline separately proves that a
# resource read did not succeed before accepting the receipt.
CODEX_CORE_MCP_RESOURCE_LIST_TOOLS = frozenset(
    {"list_mcp_resources", "list_mcp_resource_templates"}
)
CODEX_CORE_MCP_RESOURCE_TOOLS = CODEX_CORE_MCP_RESOURCE_LIST_TOOLS | {"read_mcp_resource"}


def codex_core_mcp_resource_operation(
    *,
    server: str | None,
    tool: str | None,
    offered_mcp_tool_names: tuple[str, ...] | frozenset[str] | set[str],
) -> str | None:
    """Classify a Codex-owned helper for recording, not granting resource access.

    Codex uses the synthetic ``codex`` server identity for an unscoped list.
    Retaining a scoped helper for an unconfigured server grants no access:
    the shared outcome validator requires its exact unknown-server rejection.
    Arbitrary MCP methods remain unoffered and fail closed. Configured-server
    reads still require the existing observed method-not-found failure.
    """

    if not isinstance(server, str) or not server or not isinstance(tool, str):
        return None
    if tool not in CODEX_CORE_MCP_RESOURCE_TOOLS:
        return None
    # A model can supply an arbitrary server name to Codex's built-in resource
    # helper. Codex dispatches no EVA data-plane tool in that case and
    # returns an unknown-server failure.  Retain that attempted control call so
    # the pipeline can validate and score the model-visible rejection.  The
    # stricter outcome validator still rejects successful/non-empty discovery,
    # resource content, and every non-resource helper for unoffered servers.
    return tool


def _codex_core_contains_method_not_found(value: Any) -> bool:
    if value == -32601:
        return True
    if isinstance(value, str):
        normalized = " ".join(value.casefold().replace("_", " ").split())
        return re.search(
            # EVA's fixed public MCP host spells the same -32601 rejection
            # "method not exposed". Keep the authentic error unchanged.
            r"(?:^|mcp error:\s*)-32601:\s*method not (?:found|exposed)(?:[.;])?\s*$",
            normalized,
        ) is not None
    if isinstance(value, Mapping):
        return any(_codex_core_contains_method_not_found(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(_codex_core_contains_method_not_found(child) for child in value)
    return False


def _codex_core_contains_unknown_server(value: Any, *, server: str) -> bool:
    if isinstance(value, str):
        normalized = " ".join(value.casefold().split())
        expected = server.casefold()
        return (
            f"unknown mcp server '{expected}'" in normalized
            or f'unknown mcp server "{expected}"' in normalized
        )
    if isinstance(value, Mapping):
        return any(
            _codex_core_contains_unknown_server(child, server=server)
            for child in value.values()
        )
    if isinstance(value, (tuple, list)):
        return any(
            _codex_core_contains_unknown_server(child, server=server)
            for child in value
        )
    return False


def _codex_core_unknown_server_read_rejection(output: Mapping[str, Any], *, server: str) -> bool:
    """Exact native rejection only; a phrase hidden in a URI is not proof."""
    error = output.get("error")
    if ("result" not in output or set(output) - {"result", "error", "durationMs"}
            or not isinstance(error, Mapping) or set(error) != {"message"}):
        return False
    return error["message"] == f"resources/read failed: unknown MCP server '{server}'"


def _codex_core_empty_resource_listing(
    value: Any, *, operation: str, server: str | None,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "content", "structuredContent", "_meta",
    }:
        return False
    content = value.get("content")
    if (
        not isinstance(content, (tuple, list))
        or len(content) != 1
        or value.get("structuredContent") is not None
        or value.get("_meta") is not None
    ):
        return False
    row = content[0]
    if not isinstance(row, Mapping) or set(row) != {"type", "text"} or row.get("type") != "text":
        return False
    try:
        document = json.loads(row.get("text", ""))
    except (TypeError, ValueError, RecursionError):
        return False
    key = "resources" if operation == "list_mcp_resources" else "resourceTemplates"
    expected = {key: []} if server is None else {"server": server, key: []}
    return isinstance(document, dict) and document == expected


def codex_core_mcp_resource_call_error(
    *,
    call: Any,
    operation: str,
    offered_mcp_tool_names: tuple[str, ...] | frozenset[str] | set[str],
) -> str | None:
    """Return a safe rejection reason, or ``None`` for a non-bypassing control call.

    This is the single outcome policy used by the adapter and independent
    verifier. The runtime uses :func:`codex_core_mcp_resource_operation` to
    retain only this same bounded family before an outcome is available.
    """

    if (
        getattr(call, "name", None) != operation
        or getattr(call, "lifecycle", None)
        not in {("item/completed",), ("item/started", "item/completed")}
        or getattr(call, "status", None) not in {"completed", "failed"}
        or not isinstance(getattr(call, "arguments", None), Mapping)
        or not isinstance(getattr(call, "output", None), Mapping)
    ):
        return "Codex core MCP resource outcome differs"
    if codex_core_mcp_resource_operation(
        server=getattr(call, "mcp_server", None),
        tool=getattr(call, "mcp_tool", None),
        offered_mcp_tool_names=offered_mcp_tool_names,
    ) != operation:
        return "Codex core MCP resource operation differs"

    arguments = call.arguments
    output = call.output
    if operation == "read_mcp_resource":
        if (
            set(arguments) != {"server", "uri"}
            or arguments["server"] != call.mcp_server
            or not isinstance(arguments["uri"], str)
            or not arguments["uri"]
            or call.status != "failed"
            or output.get("result") is not None
            or not isinstance(output.get("error"), Mapping)
        ):
            return "Codex core resource read returned uncommitted resources or an invalid failure"
        offered_servers = {
            name.split("/", 1)[0]
            for name in offered_mcp_tool_names
            if isinstance(name, str) and name.count("/") == 1
        }
        server_is_unoffered = call.mcp_server != "codex" and call.mcp_server not in offered_servers
        valid_failure = (
            _codex_core_unknown_server_read_rejection(output, server=call.mcp_server)
            if server_is_unoffered else _codex_core_contains_method_not_found(output["error"])
        )
        if not valid_failure:
            return "Codex core resource read returned uncommitted resources or an invalid failure"
        return None

    if operation not in CODEX_CORE_MCP_RESOURCE_LIST_TOOLS:
        return "Codex core MCP resource operation differs"
    if not set(arguments).issubset({"server", "cursor"}) or any(
        not isinstance(value, str) or not value for value in arguments.values()
    ):
        return "Codex core MCP resource-list arguments differ"
    scoped_server = arguments.get("server")
    if scoped_server is not None and scoped_server != call.mcp_server:
        return "Codex core MCP resource-list server differs"
    if call.status == "failed":
        offered_servers = {
            name.split("/", 1)[0]
            for name in offered_mcp_tool_names
            if isinstance(name, str) and name.count("/") == 1
        }
        server_is_unoffered = (
            call.mcp_server != "codex" and call.mcp_server not in offered_servers
        )
        unknown_server_rejection = (
            server_is_unoffered
            and scoped_server == call.mcp_server
            and _codex_core_contains_unknown_server(output.get("error"), server=call.mcp_server)
        )
        if (
            output.get("result") is not None
            or not isinstance(output.get("error"), Mapping)
            or not (
                (not server_is_unoffered and _codex_core_contains_method_not_found(output["error"]))
                or unknown_server_rejection
            )
        ):
            return "Codex core MCP resource-list failure differs"
        return None

    unscoped = call.mcp_server == "codex" and scoped_server is None
    scoped = scoped_server == call.mcp_server and any(
        name.startswith(call.mcp_server + "/") for name in offered_mcp_tool_names
    )
    if (
        not (unscoped or scoped)
        or output.get("error") is not None
        or not _codex_core_empty_resource_listing(
            output.get("result"), operation=operation,
            server=scoped_server if scoped else None,
        )
    ):
        return "Codex core MCP resource discovery returned uncommitted resources"
    return None


class CodexRole(str, Enum):
    """One shared runtime, with visibility fixed for the thread lifetime."""

    WEAK_ACTOR = "weak_actor"
    MIDDLE_ACTOR = "middle_actor"
    STRONG_ACTOR = "strong_actor"
    JUDGE = "judge"

    @property
    def is_actor(self) -> bool:
        return self is not CodexRole.JUDGE


class CodexSandbox(str, Enum):
    """The two permission modes EVA-Agent permits for model turns."""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"


class RuntimeIdFactory(Protocol):
    def new(self, purpose: str) -> str: ...


def _freeze_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, JsonValue]:
    frozen = freeze_json(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - defensive typing guard
        raise CodexRuntimeError(f"{label} must be a JSON object")
    return frozen


def _validate_optional_text(value: str | None, *, label: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise CodexRuntimeError(f"{label} must be non-empty when supplied")


_SECRET_KEY = re.compile(
    r"(?:^|[_-])(api[_-]?key|access[_-]?token|bearer[_-]?token|password|secret|credential)(?:$|[_-])",
    re.IGNORECASE,
)


def _reject_inline_secrets(value: Any, *, path: str = "config") -> None:
    """Require credentials to be referenced by environment name, never inline.

    Codex config commonly uses fields such as ``env_key`` or
    ``bearer_token_env_var``. Those are safe references and remain allowed.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            environment_reference = (
                "env_var" in normalized
                or normalized.endswith("_env")
                or normalized.endswith("_env_name")
                or normalized in {"env", "env_key"}
            )
            if _SECRET_KEY.search(normalized) and not environment_reference:
                raise CodexRuntimeError(
                    f"{path}.{key} may not contain an inline credential; use an environment reference"
                )
            _reject_inline_secrets(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _reject_inline_secrets(child, path=f"{path}[{index}]")


def config_key_paths(value: Mapping[str, Any]) -> tuple[str, ...]:
    """Return leaf key paths without exposing configuration values."""

    leaves: list[str] = []

    def visit(current: Any, prefix: str) -> None:
        if isinstance(current, Mapping):
            if not current:
                leaves.append(prefix)
                return
            for key in sorted(current):
                visit(current[key], f"{prefix}.{key}" if prefix else key)
            return
        if isinstance(current, (tuple, list)):
            if not current:
                leaves.append(prefix)
                return
            for index, child in enumerate(current):
                visit(child, f"{prefix}[{index}]")
            return
        leaves.append(prefix)

    visit(value, "")
    return tuple(path for path in leaves if path)


@dataclass(frozen=True, repr=False)
class CodexThreadOptions:
    """Model/provider and app-server settings for one actor or judge thread.

    ``config`` is passed to the official SDK exactly (after thawing immutable
    containers). Its values are intentionally omitted from ``repr`` and all
    runtime receipts.
    """

    role: CodexRole
    model: str
    provider: str
    cwd: str
    sandbox: CodexSandbox
    config: Mapping[str, JsonValue] = MappingProxyType({})
    offered_tools: tuple["CodexToolOffer", ...] = ()
    base_instructions: str | None = None
    developer_instructions: str | None = None
    ephemeral: bool = True
    service_name: str = "evamed-codex"
    service_tier: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, CodexRole):
            raise CodexRuntimeError("thread role differs")
        if not isinstance(self.sandbox, CodexSandbox):
            raise CodexRuntimeError("thread sandbox must be read-only or workspace-write")
        if not isinstance(self.model, str) or not self.model.strip():
            raise CodexRuntimeError("thread model is required")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise CodexRuntimeError("thread provider is required")
        if not isinstance(self.cwd, str) or not self.cwd or "\x00" in self.cwd:
            raise CodexRuntimeError("thread cwd is required")
        cwd = Path(self.cwd)
        if not cwd.is_absolute() or ".." in cwd.parts:
            raise CodexRuntimeError("thread cwd must be an absolute normalized path")
        if type(self.ephemeral) is not bool:
            raise CodexRuntimeError("thread ephemeral marker differs")
        _validate_optional_text(self.base_instructions, label="base instructions")
        _validate_optional_text(self.developer_instructions, label="developer instructions")
        _validate_optional_text(self.service_name, label="service name")
        _validate_optional_text(self.service_tier, label="service tier")
        config = _freeze_mapping(self.config, label="Codex config")
        _reject_inline_secrets(config)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "offered_tools", tuple(self.offered_tools))
        if self.role.is_actor and any(tool.visibility != "public" for tool in self.offered_tools):
            raise CodexRuntimeError("actor thread cannot mount judge-only MCP tools")
        names = tuple(tool.fully_qualified_name for tool in self.offered_tools)
        if len(names) != len(set(names)):
            raise CodexRuntimeError("offered MCP tool names must be unique")

    @property
    def config_keys(self) -> tuple[str, ...]:
        return config_key_paths(self.config)

    def __repr__(self) -> str:
        return (
            "CodexThreadOptions("
            f"role={self.role.value!r}, model={self.model!r}, provider={self.provider!r}, "
            f"cwd={self.cwd!r}, sandbox={self.sandbox.value!r}, "
            f"config_keys={self.config_keys!r}, offered_tools={len(self.offered_tools)}, "
            "instructions=<redacted>)"
        )


@dataclass(frozen=True)
class CodexSkill:
    skill_id: str
    name: str
    path: str
    content_blake3: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        if (
            not self.skill_id
            or not self.name
            or not self.path
            or "\x00" in self.path
            or not path.is_absolute()
            or ".." in path.parts
            or not is_blake3(self.content_blake3)
        ):
            raise CodexRuntimeError(
                "skill requires an id, name, normalized absolute path, and BLAKE3 digest"
            )

    def catalog_entry(self) -> Mapping[str, str]:
        """Return the exact mount commitment; the skill file is never rewritten."""

        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "path": self.path,
            "content_blake3": self.content_blake3,
        }


@dataclass(frozen=True)
class CodexToolOffer:
    """One concise MCP tool description committed before a turn."""

    fully_qualified_name: str
    description: str
    input_schema: Mapping[str, JsonValue]
    visibility: str = "public"
    parallel_safe: bool | None = None
    read_only: bool | None = None
    allowed_stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.fully_qualified_name
            or self.fully_qualified_name.count("/") != 1
            or any(not part for part in self.fully_qualified_name.split("/"))
            or not self.description.strip()
        ):
            raise CodexRuntimeError("MCP tool offer requires server/tool name and description")
        if self.visibility not in {"public", "judge-only"}:
            raise CodexRuntimeError("MCP tool offer visibility differs")
        if self.parallel_safe is not None and type(self.parallel_safe) is not bool:
            raise CodexRuntimeError("MCP tool parallel-safety metadata differs")
        if self.read_only is not None and type(self.read_only) is not bool:
            raise CodexRuntimeError("MCP tool read-only metadata differs")
        stages = tuple(sorted(self.allowed_stages))
        if len(stages) != len(set(stages)) or any(
            stage not in {"S1", "S2", "S3", "S4", "S5", "E2E"} for stage in stages
        ):
            raise CodexRuntimeError("MCP tool allowed-stage metadata differs")
        schema = _freeze_mapping(self.input_schema, label="MCP input schema")
        if schema.get("type") != "object":
            raise CodexRuntimeError("MCP tool input schema must describe an object")
        object.__setattr__(self, "input_schema", schema)
        object.__setattr__(self, "allowed_stages", stages)

    @property
    def server(self) -> str:
        return self.fully_qualified_name.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.fully_qualified_name.split("/", 1)[1]

    @classmethod
    def from_definition(
        cls,
        *,
        server: str,
        definition: Any,
        visibility: str = "public",
    ) -> "CodexToolOffer":
        """Adapt an existing tool definition without translating its schema.

        Both current EVA tool-definition spellings are accepted.  The source
        ``name``, ``description`` and schema value are copied byte-semantically;
        discovery metadata stays outside that canonical schema.
        """

        name = getattr(definition, "name", None)
        description = getattr(definition, "description", None)
        schema = getattr(definition, "input_schema", None)
        if schema is None:
            schema = getattr(definition, "parameters", None)
        if not isinstance(server, str) or not server or "/" in server:
            raise CodexRuntimeError("MCP server name differs")
        if not isinstance(name, str) or not isinstance(description, str) or not isinstance(
            schema, Mapping
        ):
            raise CodexRuntimeError("source tool definition fields differ")
        return cls(
            fully_qualified_name=f"{server}/{name}",
            description=description,
            input_schema=schema,
            visibility=visibility,
            parallel_safe=getattr(definition, "parallel_safe", None),
            read_only=getattr(definition, "read_only", None),
            allowed_stages=tuple(getattr(definition, "allowed_stages", ())),
        )

    def canonical_catalog_entry(self) -> Mapping[str, Any]:
        """Canonical data-plane fields, preserving their original values."""

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def mcp_transport_entry(self) -> Mapping[str, Any]:
        """Map only the protocol key; ``inputSchema`` retains the exact value."""

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def sidecar_metadata_entry(self) -> Mapping[str, Any]:
        return {
            "fully_qualified_name": self.fully_qualified_name,
            "server": self.server,
            "visibility": self.visibility,
            "input_schema_blake3": blake3_hex(self.input_schema),
            "parallel_safe": self.parallel_safe,
            "read_only": self.read_only,
            "allowed_stages": self.allowed_stages,
        }


@dataclass(frozen=True)
class CodexMention:
    name: str
    path: str

    def __post_init__(self) -> None:
        if not self.name or not self.path:
            raise CodexRuntimeError("mention requires a name and path")


@dataclass(frozen=True, repr=False)
class CodexTurnInput:
    """A structurally separated turn input.

    ``judge_only_context`` is never serialized for an actor role. Actor calls
    that supply it fail before the backend is touched.
    """

    public_text: str
    public_context: Mapping[str, JsonValue] = MappingProxyType({})
    judge_only_context: Mapping[str, JsonValue] | None = None
    skills: tuple[CodexSkill, ...] = ()
    mentions: tuple[CodexMention, ...] = ()
    output_schema: Mapping[str, JsonValue] | None = None
    sandbox: CodexSandbox | None = None
    model: str | None = None
    effort: str | None = None
    summary: str | None = None
    service_tier: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.public_text, str) or not self.public_text.strip():
            raise CodexRuntimeError("turn public text is required")
        public = _freeze_mapping(self.public_context, label="public context")
        private = self.judge_only_context
        if private is not None:
            private = _freeze_mapping(private, label="judge-only context")
        schema = self.output_schema
        if schema is not None:
            schema = _freeze_mapping(schema, label="output schema")
        if self.sandbox is not None and not isinstance(self.sandbox, CodexSandbox):
            raise CodexRuntimeError("turn sandbox differs")
        for label, value in (
            ("turn model", self.model),
            ("turn effort", self.effort),
            ("turn summary", self.summary),
            ("turn service tier", self.service_tier),
        ):
            _validate_optional_text(value, label=label)
        object.__setattr__(self, "public_context", public)
        object.__setattr__(self, "judge_only_context", private)
        object.__setattr__(self, "skills", tuple(self.skills))
        object.__setattr__(self, "mentions", tuple(self.mentions))
        object.__setattr__(self, "output_schema", schema)

    def __repr__(self) -> str:
        return (
            "CodexTurnInput(public_text=<redacted>, "
            f"public_keys={tuple(self.public_context)!r}, "
            f"judge_only={'present' if self.judge_only_context is not None else 'absent'}, "
            f"skills={len(self.skills)}, mentions={len(self.mentions)})"
        )


@dataclass(frozen=True)
class CodexThreadHandle:
    runtime_thread_id: str
    thread_id: str
    role: CodexRole
    model: str
    provider: str
    cwd: str
    sandbox: CodexSandbox
    config_keys: tuple[str, ...]
    resumed: bool

    def __post_init__(self) -> None:
        uuid_text(self.runtime_thread_id, label="Codex runtime_thread_id")
        if not self.thread_id:
            raise CodexRuntimeError("upstream Codex thread id is required")
        if not isinstance(self.role, CodexRole) or not isinstance(self.sandbox, CodexSandbox):
            raise CodexRuntimeError("Codex thread handle role or sandbox differs")
        if type(self.resumed) is not bool:
            raise CodexRuntimeError("Codex thread resume marker differs")
        object.__setattr__(self, "config_keys", tuple(self.config_keys))


@dataclass(frozen=True)
class CodexEvent:
    event_id: str
    sequence: int
    method: str
    thread_id: str
    turn_id: str
    payload: Mapping[str, JsonValue]
    content_redacted: bool
    event_blake3: str

    def __post_init__(self) -> None:
        uuid_text(self.event_id, label="Codex event_id")
        if type(self.sequence) is not int or self.sequence < 0:
            raise CodexRuntimeError("Codex event sequence differs")
        if not self.method or not self.thread_id or not self.turn_id:
            raise CodexRuntimeError("Codex event routing fields are required")
        if type(self.content_redacted) is not bool:
            raise CodexRuntimeError("Codex event redaction marker differs")
        payload = _freeze_mapping(self.payload, label="Codex event payload")
        object.__setattr__(self, "payload", payload)
        if self.event_blake3 != blake3_hex(self.core()):
            raise CodexRuntimeError("Codex event BLAKE3 differs")

    def core(self) -> Mapping[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "method": self.method,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "payload": self.payload,
            "content_redacted": self.content_redacted,
        }


@dataclass(frozen=True)
class CodexToolCall:
    tool_call_id: str
    upstream_item_id: str
    tool_type: str
    name: str
    mcp_server: str | None
    mcp_tool: str | None
    fully_qualified_name: str | None
    status: str
    arguments: JsonValue
    output: JsonValue
    lifecycle: tuple[str, ...]
    first_event_sequence: int
    receipt_blake3: str

    def __post_init__(self) -> None:
        uuid_text(self.tool_call_id, label="Codex tool_call_id")
        if not all((self.upstream_item_id, self.tool_type, self.name, self.status)):
            raise CodexRuntimeError("Codex tool-call identity/status differs")
        mcp_values = (self.mcp_server, self.mcp_tool, self.fully_qualified_name)
        if self.tool_type == "mcpToolCall":
            if any(not isinstance(value, str) or not value for value in mcp_values):
                raise CodexRuntimeError("MCP tool call must retain server/tool identity")
            if self.fully_qualified_name != f"{self.mcp_server}/{self.mcp_tool}":
                raise CodexRuntimeError("MCP tool fully-qualified name differs")
        elif any(value is not None for value in mcp_values):
            raise CodexRuntimeError("non-MCP tool call cannot claim MCP identity")
        if type(self.first_event_sequence) is not int or self.first_event_sequence < 0:
            raise CodexRuntimeError("Codex tool-call event sequence differs")
        arguments = freeze_json(self.arguments)
        output = freeze_json(self.output)
        lifecycle = tuple(self.lifecycle)
        if not lifecycle or any(item not in {"item/started", "item/completed"} for item in lifecycle):
            raise CodexRuntimeError("Codex tool-call lifecycle differs")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "lifecycle", lifecycle)
        if self.receipt_blake3 != blake3_hex(self.core()):
            raise CodexRuntimeError("Codex tool-call BLAKE3 differs")

    def core(self) -> Mapping[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "upstream_item_id": self.upstream_item_id,
            "tool_type": self.tool_type,
            "name": self.name,
            "mcp_server": self.mcp_server,
            "mcp_tool": self.mcp_tool,
            "fully_qualified_name": self.fully_qualified_name,
            "status": self.status,
            "arguments": self.arguments,
            "output": self.output,
            "lifecycle": self.lifecycle,
            "first_event_sequence": self.first_event_sequence,
        }


@dataclass(frozen=True)
class CodexTurnReceipt:
    schema: str
    receipt_id: str
    runtime_thread_id: str
    runtime_turn_id: str
    thread_id: str
    turn_id: str
    role: CodexRole
    model: str
    provider: str
    sandbox: CodexSandbox
    thread_resumed: bool
    visibility: str
    status: str
    final_response: str | None
    events: tuple[CodexEvent, ...]
    tool_calls: tuple[CodexToolCall, ...]
    selected_skill_ids: tuple[str, ...]
    selected_skill_catalog_blake3: str
    offered_mcp_tool_names: tuple[str, ...]
    offered_tool_schema_blake3: str
    max_parallelism_observed: int
    parallel_tool_calls_supported: bool
    usage: Mapping[str, JsonValue]
    input_blake3: str
    config_keys: tuple[str, ...]
    config_values_recorded: bool
    input_payload_recorded: bool
    sdk_version: str
    server_version: str
    receipt_blake3: str

    def __post_init__(self) -> None:
        if self.schema != "eva.codex-turn-receipt.v1":
            raise CodexRuntimeError("Codex receipt schema differs")
        for label, value in (
            ("Codex receipt_id", self.receipt_id),
            ("Codex runtime_thread_id", self.runtime_thread_id),
            ("Codex runtime_turn_id", self.runtime_turn_id),
        ):
            uuid_text(value, label=label)
        if not self.thread_id or not self.turn_id or not self.status:
            raise CodexRuntimeError("Codex upstream turn identity/status differs")
        if not isinstance(self.role, CodexRole) or not isinstance(self.sandbox, CodexSandbox):
            raise CodexRuntimeError("Codex receipt role or sandbox differs")
        expected_visibility = "actor-public" if self.role.is_actor else "judge-only"
        if self.visibility != expected_visibility:
            raise CodexRuntimeError("Codex receipt visibility differs")
        if self.status in {"inProgress", "in_progress"}:
            raise CodexRuntimeError("Codex receipt cannot commit an active turn")
        if type(self.thread_resumed) is not bool:
            raise CodexRuntimeError("Codex receipt resume marker differs")
        if self.config_values_recorded or self.input_payload_recorded:
            raise CodexRuntimeError("Codex receipt must not embed config values or raw input")
        if (
            not is_blake3(self.input_blake3)
            or not is_blake3(self.offered_tool_schema_blake3)
            or not is_blake3(self.selected_skill_catalog_blake3)
        ):
            raise CodexRuntimeError("Codex input or offered-tool commitment differs")
        events = tuple(self.events)
        tools = tuple(self.tool_calls)
        if not events or tuple(event.sequence for event in events) != tuple(range(len(events))):
            raise CodexRuntimeError("Codex receipt event ordering differs")
        if any(
            event.thread_id != self.thread_id or event.turn_id != self.turn_id
            for event in events
        ):
            raise CodexRuntimeError("Codex receipt event routing differs")
        if not any(event.method == "turn/completed" for event in events):
            raise CodexRuntimeError("Codex receipt lacks turn/completed")
        selected = tuple(self.selected_skill_ids)
        offered = tuple(self.offered_mcp_tool_names)
        if len(selected) != len(set(selected)) or len(offered) != len(set(offered)):
            raise CodexRuntimeError("Codex selected skill or offered MCP inventory differs")
        if (
            type(self.max_parallelism_observed) is not int
            or self.max_parallelism_observed < 0
            or self.max_parallelism_observed > max(1, len(tools))
            or type(self.parallel_tool_calls_supported) is not bool
            or not self.parallel_tool_calls_supported
        ):
            raise CodexRuntimeError("Codex same-turn parallelism evidence differs")
        if tools and self.max_parallelism_observed < 1:
            raise CodexRuntimeError("Codex tool calls require observed parallelism >= 1")
        if not tools and self.max_parallelism_observed != 0:
            raise CodexRuntimeError("Codex tool-free turn cannot claim tool parallelism")
        usage = _freeze_mapping(self.usage, label="Codex usage")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "tool_calls", tools)
        object.__setattr__(self, "selected_skill_ids", selected)
        object.__setattr__(self, "offered_mcp_tool_names", offered)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "config_keys", tuple(self.config_keys))
        if self.receipt_blake3 != blake3_hex(self.core()):
            raise CodexRuntimeError("Codex turn receipt BLAKE3 differs")

    def core(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "runtime_thread_id": self.runtime_thread_id,
            "runtime_turn_id": self.runtime_turn_id,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "role": self.role,
            "model": self.model,
            "provider": self.provider,
            "sandbox": self.sandbox,
            "thread_resumed": self.thread_resumed,
            "visibility": self.visibility,
            "status": self.status,
            "final_response": self.final_response,
            "events": self.events,
            "tool_calls": self.tool_calls,
            "selected_skill_ids": self.selected_skill_ids,
            "selected_skill_catalog_blake3": self.selected_skill_catalog_blake3,
            "offered_mcp_tool_names": self.offered_mcp_tool_names,
            "offered_tool_schema_blake3": self.offered_tool_schema_blake3,
            "max_parallelism_observed": self.max_parallelism_observed,
            "parallel_tool_calls_supported": self.parallel_tool_calls_supported,
            "usage": self.usage,
            "input_blake3": self.input_blake3,
            "config_keys": self.config_keys,
            "config_values_recorded": self.config_values_recorded,
            "input_payload_recorded": self.input_payload_recorded,
            "sdk_version": self.sdk_version,
            "server_version": self.server_version,
        }


def verify_codex_turn_receipt(receipt: CodexTurnReceipt) -> None:
    """Reopen every nested BLAKE3 commitment in a turn receipt."""

    if not isinstance(receipt, CodexTurnReceipt):
        raise CodexRuntimeError("value is not a Codex turn receipt")
    for event in receipt.events:
        if event.event_blake3 != blake3_hex(event.core()):
            raise CodexRuntimeError("Codex event BLAKE3 differs")
    for tool_call in receipt.tool_calls:
        if tool_call.receipt_blake3 != blake3_hex(tool_call.core()):
            raise CodexRuntimeError("Codex tool-call BLAKE3 differs")
    if receipt.receipt_blake3 != blake3_hex(receipt.core()):
        raise CodexRuntimeError("Codex turn receipt BLAKE3 differs")


def codex_turn_receipt_from_document(value: Mapping[str, Any]) -> CodexTurnReceipt:
    """Strictly reopen a persisted receipt without weakening its v1 schema.

    This decoder is intentionally colocated with the immutable contract.  It
    accepts the canonical JSON projection produced by ``canonical_value`` and
    rejects missing or additional keys before any nested BLAKE3 is trusted.
    """

    receipt_keys = {
        field
        for field in CodexTurnReceipt.__dataclass_fields__
    }
    event_keys = {field for field in CodexEvent.__dataclass_fields__}
    call_keys = {field for field in CodexToolCall.__dataclass_fields__}
    if not isinstance(value, Mapping) or set(value) != receipt_keys:
        raise CodexRuntimeError("persisted Codex receipt schema differs")
    raw_events = value.get("events")
    raw_calls = value.get("tool_calls")
    if not isinstance(raw_events, (tuple, list)) or not isinstance(
        raw_calls, (tuple, list)
    ):
        raise CodexRuntimeError("persisted Codex receipt collections differ")
    events: list[CodexEvent] = []
    calls: list[CodexToolCall] = []
    try:
        for raw in raw_events:
            if not isinstance(raw, Mapping) or set(raw) != event_keys:
                raise CodexRuntimeError("persisted Codex event schema differs")
            events.append(CodexEvent(**dict(raw)))
        for raw in raw_calls:
            if not isinstance(raw, Mapping) or set(raw) != call_keys:
                raise CodexRuntimeError("persisted Codex tool-call schema differs")
            document = dict(raw)
            lifecycle = document.get("lifecycle")
            if not isinstance(lifecycle, (tuple, list)):
                raise CodexRuntimeError("persisted Codex tool lifecycle differs")
            document["lifecycle"] = tuple(lifecycle)
            calls.append(CodexToolCall(**document))
        document = dict(value)
        document["role"] = CodexRole(document["role"])
        document["sandbox"] = CodexSandbox(document["sandbox"])
        document["events"] = tuple(events)
        document["tool_calls"] = tuple(calls)
        for key in (
            "selected_skill_ids",
            "offered_mcp_tool_names",
            "config_keys",
        ):
            collection = document[key]
            if not isinstance(collection, (tuple, list)):
                raise CodexRuntimeError(f"persisted Codex {key} differs")
            document[key] = tuple(collection)
        receipt = CodexTurnReceipt(**document)
    except CodexRuntimeError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise CodexRuntimeError("persisted Codex receipt value differs") from exc
    verify_codex_turn_receipt(receipt)
    return receipt


__all__ = [
    "CODEX_CORE_MCP_RESOURCE_LIST_TOOLS",
    "CODEX_CORE_MCP_RESOURCE_TOOLS",
    "CodexEvent",
    "CodexMention",
    "CodexRole",
    "CodexRuntimeError",
    "CodexSandbox",
    "CodexSkill",
    "CodexThreadHandle",
    "CodexThreadOptions",
    "CodexToolOffer",
    "CodexToolCall",
    "CodexTurnInput",
    "CodexTurnReceipt",
    "RuntimeIdFactory",
    "codex_core_mcp_resource_call_error",
    "codex_core_mcp_resource_operation",
    "config_key_paths",
    "codex_turn_receipt_from_document",
    "verify_codex_turn_receipt",
]
