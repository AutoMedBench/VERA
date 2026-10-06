"""Bind one immutable EvaMed policy to one Codex/MCP actor deployment.

The deployment preflight talks only to a local stdio MCP process.  It sends
``initialize`` and ``tools/list`` and never starts a Codex thread, invokes an
MCP tool, or contacts a model provider.  A successful result can then be
passed directly to :class:`eva_agent.codex_runtime.CodexRuntime`.

Canonical EvaMed tool schemas are inputs to this module, never outputs of a
translation step.  The only protocol rename is ``input_schema`` to MCP's
``inputSchema`` outer key.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from eva_agent.codex_runtime import (
    CodexRole,
    CodexThreadOptions,
    CodexToolOffer,
)
from eva_agent.mcp_compat import (
    MCPCompatibilityError,
    MCPToolCatalog,
    PolicyCatalogBinding,
    verify_mcp_catalog,
    verify_policy_catalog_binding,
)
from eva_agent.pipeline.contracts import JsonValue, freeze_json
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
    is_blake3,
)


CODEX_MCP_PREFLIGHT_SCHEMA = "eva.codex-mcp-policy-preflight.v1"
_SKILL_TOOL_NAMES = ("search_skills", "load_skill")
_ALLOWED_STAGES = frozenset({"S1", "S2", "S3", "S4", "S5", "E2E"})
_SAFE_INHERITED_ENV = frozenset(
    {"HOME", "LANG", "LC_ALL", "PATH", "PYTHONHOME", "PYTHONPATH", "TMPDIR", "VIRTUAL_ENV"}
)
_RESERVED_SERVER_ENV = frozenset(
    {
        "EVAMED_MCP_ACTOR_MODE",
        "EVAMED_MCP_OFFLINE",
        "EVAMED_MCP_POLICY_BLAKE3",
        "EVAMED_MCP_POLICY_PATH",
    }
)
_SECRET_ENV_KEY = re.compile(
    r"(?:API[_-]?KEY|ACCESS[_-]?KEY|TOKEN|PASSWORD|SECRET|CREDENTIAL)",
    re.IGNORECASE,
)
_PRIVATE_SCHEMA_KEYS = frozenset(
    {
        "answerkey",
        "goldanswer",
        "groundtruth",
        "judgeonlyreference",
        "privatereference",
        "privaterubric",
        "privaterubricitems",
        "referenceanswer",
        "rubricitems",
    }
)


class CodexMCPDeploymentError(ValueError):
    """A policy, MCP process, or Codex deployment boundary failed closed."""


def _frozen_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, JsonValue]:
    try:
        frozen = freeze_json(value)
    except Exception as exc:
        raise CodexMCPDeploymentError(f"{label} is not JSON-shaped") from exc
    if not isinstance(frozen, Mapping):  # pragma: no cover - typing guard
        raise CodexMCPDeploymentError(f"{label} must be an object")
    return frozen


def _clean_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise CodexMCPDeploymentError(f"{label} differs")
    return value


def _normalize_json_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _reject_private_schema_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _normalize_json_key(str(key)) in _PRIVATE_SCHEMA_KEYS:
                raise CodexMCPDeploymentError(
                    f"actor MCP tool exposes private material at {path}.{key}"
                )
            _reject_private_schema_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _reject_private_schema_keys(child, path=f"{path}[{index}]")


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CodexMCPDeploymentError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_object_no_duplicates)
    except CodexMCPDeploymentError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodexMCPDeploymentError(f"{label} is not valid UTF-8 JSON") from exc


def _policy_entries(document: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    candidates: Any = document.get("tools") or document.get("tool_definitions")
    nested = document.get("policy")
    if candidates is None and isinstance(nested, Mapping):
        candidates = nested.get("tools")
    if (
        isinstance(candidates, (str, bytes))
        or not isinstance(candidates, Sequence)
        or not all(isinstance(item, Mapping) for item in candidates)
    ):
        raise CodexMCPDeploymentError("source policy has no canonical tool sequence")
    return candidates


def _canonical_policy_rows(document: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(_policy_entries(document)):
        function = entry.get("function")
        source = function if isinstance(function, Mapping) else entry
        name = source.get("name")
        description = source.get("description")
        schema = source.get(
            "input_schema", source.get("parameters", source.get("inputSchema"))
        )
        if (
            not isinstance(name, str)
            or not isinstance(description, str)
            or not isinstance(schema, Mapping)
        ):
            raise CodexMCPDeploymentError(
                f"source policy tool {index} canonical fields differ"
            )
        rows.append(
            {
                "name": name,
                "description": description,
                "input_schema": canonical_value(schema),
            }
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class StdioMCPServerSpec:
    """One candidate-scoped local MCP server launch and identity."""

    config_name: str
    server_name: str
    command: tuple[str, ...]
    cwd: str
    environment: Mapping[str, str] = MappingProxyType({})
    forwarded_env_vars: tuple[str, ...] = ()
    protocol_version: str = "2025-06-18"
    server_version: str | None = None
    startup_timeout_seconds: float = 10.0
    tool_timeout_seconds: int = 900

    def __post_init__(self) -> None:
        config_name = _clean_text(self.config_name, label="MCP config name")
        if "/" in config_name:
            raise CodexMCPDeploymentError("MCP config name may not contain '/'")
        _clean_text(self.server_name, label="MCP server name")
        _clean_text(self.protocol_version, label="MCP protocol version")
        if self.server_version is not None:
            _clean_text(self.server_version, label="MCP server version")
        command = tuple(self.command)
        if not command or any(not isinstance(part, str) or not part or "\x00" in part for part in command):
            raise CodexMCPDeploymentError("MCP command differs")
        executable = Path(command[0])
        if not executable.is_absolute() or not executable.is_file():
            raise CodexMCPDeploymentError("MCP executable must be an existing absolute file")
        cwd = Path(self.cwd)
        if not cwd.is_absolute() or ".." in cwd.parts or not cwd.is_dir():
            raise CodexMCPDeploymentError("MCP cwd must be an existing normalized absolute directory")
        environment = dict(self.environment)
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or "\x00" in key
            or "\x00" in value
            for key, value in environment.items()
        ):
            raise CodexMCPDeploymentError("MCP environment differs")
        overlap = _RESERVED_SERVER_ENV & set(environment)
        if overlap:
            raise CodexMCPDeploymentError(
                f"MCP environment attempts to override reserved binding key {sorted(overlap)[0]}"
            )
        secret = next((key for key in environment if _SECRET_ENV_KEY.search(key)), None)
        if secret is not None:
            raise CodexMCPDeploymentError(
                f"MCP readiness environment may not carry credential {secret}"
            )
        forwarded = tuple(self.forwarded_env_vars)
        if len(forwarded) != len(set(forwarded)) or any(
            not isinstance(key, str) or not key or "\x00" in key for key in forwarded
        ):
            raise CodexMCPDeploymentError("forwarded MCP environment names differ")
        if not (0 < self.startup_timeout_seconds <= 120):
            raise CodexMCPDeploymentError("MCP startup timeout must be in (0,120]")
        if not (1 <= self.tool_timeout_seconds <= 86_400):
            raise CodexMCPDeploymentError("MCP tool timeout must be in [1,86400]")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "cwd", str(cwd))
        object.__setattr__(self, "environment", MappingProxyType(dict(sorted(environment.items()))))
        object.__setattr__(self, "forwarded_env_vars", forwarded)

    def offline_environment(self, *, policy_path: str, policy_blake3: str) -> dict[str, str]:
        """Return a credential-scrubbed environment for initialize/list only."""

        inherited = {
            key: value
            for key, value in os.environ.items()
            if key in _SAFE_INHERITED_ENV and not _SECRET_ENV_KEY.search(key)
        }
        inherited.update(self.environment)
        inherited.update(
            {
                "EVAMED_MCP_ACTOR_MODE": "1",
                "EVAMED_MCP_OFFLINE": "1",
                "EVAMED_MCP_POLICY_PATH": policy_path,
                "EVAMED_MCP_POLICY_BLAKE3": policy_blake3,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        return inherited

    def codex_server_config(
        self, *, policy_path: str, policy_blake3: str, enabled_tools: Sequence[str]
    ) -> dict[str, Any]:
        environment = {
            **dict(self.environment),
            "EVAMED_MCP_ACTOR_MODE": "1",
            "EVAMED_MCP_OFFLINE": "0",
            "EVAMED_MCP_POLICY_PATH": policy_path,
            "EVAMED_MCP_POLICY_BLAKE3": policy_blake3,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result: dict[str, Any] = {
            "command": self.command[0],
            "args": list(self.command[1:]),
            "cwd": self.cwd,
            "env": environment,
            "required": True,
            "startup_timeout_sec": self.startup_timeout_seconds,
            "tool_timeout_sec": self.tool_timeout_seconds,
            "enabled_tools": list(enabled_tools),
        }
        if self.forwarded_env_vars:
            result["env_vars"] = list(self.forwarded_env_vars)
        return result

    @property
    def launch_blake3(self) -> str:
        return blake3_hex(
            {
                "config_name": self.config_name,
                "server_name": self.server_name,
                "command": self.command,
                "cwd": self.cwd,
                "environment": self.environment,
                "forwarded_env_vars": self.forwarded_env_vars,
                "protocol_version": self.protocol_version,
                "server_version": self.server_version,
                "startup_timeout_seconds": self.startup_timeout_seconds,
                "tool_timeout_seconds": self.tool_timeout_seconds,
            }
        )


@dataclass(frozen=True, slots=True)
class CodexMCPPreflightReceipt:
    candidate_id: str
    policy_id: str
    policy_binding_blake3: str
    source_policy_blake3: str
    policy_schema_catalog_blake3: str
    skill_schema_catalog_blake3: str
    mcp_config_name: str
    mcp_server_name: str
    mcp_server_version: str
    mcp_protocol_version: str
    server_launch_blake3: str
    offered_tools: tuple[Mapping[str, JsonValue], ...]
    sidecar_annotations: tuple[Mapping[str, JsonValue], ...]
    probe_methods: tuple[str, ...]
    provider_calls_made: int
    mcp_tool_calls_made: int
    preflight_blake3: str
    schema: str = CODEX_MCP_PREFLIGHT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "offered_tools",
            tuple(_frozen_mapping(row, label="offered tool") for row in self.offered_tools),
        )
        object.__setattr__(
            self,
            "sidecar_annotations",
            tuple(
                _frozen_mapping(row, label="tool annotation")
                for row in self.sidecar_annotations
            ),
        )
        verify_codex_mcp_preflight_receipt(self)

    def core(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "policy_id": self.policy_id,
            "policy_binding_blake3": self.policy_binding_blake3,
            "source_policy_blake3": self.source_policy_blake3,
            "policy_schema_catalog_blake3": self.policy_schema_catalog_blake3,
            "skill_schema_catalog_blake3": self.skill_schema_catalog_blake3,
            "mcp_config_name": self.mcp_config_name,
            "mcp_server_name": self.mcp_server_name,
            "mcp_server_version": self.mcp_server_version,
            "mcp_protocol_version": self.mcp_protocol_version,
            "server_launch_blake3": self.server_launch_blake3,
            "offered_tools": self.offered_tools,
            "sidecar_annotations": self.sidecar_annotations,
            "probe_methods": self.probe_methods,
            "provider_calls_made": self.provider_calls_made,
            "mcp_tool_calls_made": self.mcp_tool_calls_made,
        }

    def to_document(self) -> dict[str, Any]:
        return {**canonical_value(self.core()), "preflight_blake3": self.preflight_blake3}


def verify_codex_mcp_preflight_receipt(receipt: CodexMCPPreflightReceipt) -> None:
    if receipt.schema != CODEX_MCP_PREFLIGHT_SCHEMA:
        raise CodexMCPDeploymentError("Codex/MCP preflight receipt schema differs")
    for label, value in (
        ("policy binding", receipt.policy_binding_blake3),
        ("source policy", receipt.source_policy_blake3),
        ("policy schema catalog", receipt.policy_schema_catalog_blake3),
        ("skill schema catalog", receipt.skill_schema_catalog_blake3),
        ("server launch", receipt.server_launch_blake3),
        ("preflight", receipt.preflight_blake3),
    ):
        if not is_blake3(value):
            raise CodexMCPDeploymentError(f"{label} BLAKE3 differs")
    if receipt.probe_methods != ("initialize", "tools/list"):
        raise CodexMCPDeploymentError("readiness probe method set differs")
    if receipt.provider_calls_made != 0 or receipt.mcp_tool_calls_made != 0:
        raise CodexMCPDeploymentError("offline preflight may not execute providers or tools")
    names = tuple(row.get("name") for row in receipt.offered_tools)
    fqn_names = tuple(row.get("fully_qualified_name") for row in receipt.sidecar_annotations)
    if len(names) != len(set(names)) or len(names) != len(fqn_names):
        raise CodexMCPDeploymentError("preflight tool identities differ")
    if names[-2:] != _SKILL_TOOL_NAMES:
        raise CodexMCPDeploymentError("preflight lacks unchanged skill loader tools")
    for tool, sidecar in zip(receipt.offered_tools, receipt.sidecar_annotations):
        if set(tool) != {"name", "description", "inputSchema"}:
            raise CodexMCPDeploymentError("preflight canonical MCP tool fields differ")
        if not isinstance(tool["description"], str) or not tool["description"].strip():
            raise CodexMCPDeploymentError("preflight MCP tool description differs")
        expected_fqn = f"{receipt.mcp_config_name}/{tool['name']}"
        if sidecar.get("fully_qualified_name") != expected_fqn:
            raise CodexMCPDeploymentError("preflight fully-qualified tool name differs")
        if sidecar.get("visibility") != "public":
            raise CodexMCPDeploymentError("actor preflight contains a judge-only tool")
        if sidecar.get("input_schema_blake3") != blake3_hex(tool["inputSchema"]):
            raise CodexMCPDeploymentError("preflight sidecar schema BLAKE3 differs")
        if any(
            type(sidecar.get(key)) is not bool
            for key in ("parallel_safe", "read_only", "mutating")
        ):
            raise CodexMCPDeploymentError("preflight tool annotations differ")
    if receipt.preflight_blake3 != blake3_hex(receipt.core()):
        raise CodexMCPDeploymentError("Codex/MCP preflight BLAKE3 differs")


@dataclass(frozen=True, slots=True)
class PreparedCodexMCPDeployment:
    """A provider-neutral Codex thread configuration proven by stdio preflight."""

    binding: PolicyCatalogBinding
    thread_options: CodexThreadOptions
    receipt: CodexMCPPreflightReceipt

    def __post_init__(self) -> None:
        verify_policy_catalog_binding(self.binding)
        verify_codex_mcp_preflight_receipt(self.receipt)
        if self.thread_options.role is CodexRole.JUDGE:
            raise CodexMCPDeploymentError("prepared deployment must be actor-only")
        if self.receipt.candidate_id != self.binding.candidate_id:
            raise CodexMCPDeploymentError("prepared deployment candidate differs")
        if self.receipt.policy_binding_blake3 != self.binding.binding_blake3:
            raise CodexMCPDeploymentError("prepared deployment policy binding differs")
        offered_fqns = tuple(tool.fully_qualified_name for tool in self.thread_options.offered_tools)
        receipt_fqns = tuple(
            str(row["fully_qualified_name"]) for row in self.receipt.sidecar_annotations
        )
        if offered_fqns != receipt_fqns:
            raise CodexMCPDeploymentError("prepared Codex offers differ from MCP preflight")
        offered_transport = tuple(
            canonical_value(tool.mcp_transport_entry())
            for tool in self.thread_options.offered_tools
        )
        if canonical_json_bytes(offered_transport) != canonical_json_bytes(
            self.receipt.offered_tools
        ):
            raise CodexMCPDeploymentError("prepared Codex schemas differ from MCP preflight")
        servers = self.thread_options.config.get("mcp_servers")
        selected = servers.get(self.receipt.mcp_config_name) if isinstance(servers, Mapping) else None
        if not isinstance(selected, Mapping) or tuple(selected.get("enabled_tools", ())) != tuple(
            row["name"] for row in self.receipt.offered_tools
        ):
            raise CodexMCPDeploymentError("prepared Codex enabled tool set differs")


class CodexMCPDeployment:
    """Fail-closed builder for one policy-bound actor thread and MCP process."""

    def __init__(
        self,
        *,
        binding: PolicyCatalogBinding,
        skill_catalog: MCPToolCatalog,
        source_policy_path: str | Path,
        server: StdioMCPServerSpec,
        actor_thread: CodexThreadOptions,
    ) -> None:
        try:
            verify_policy_catalog_binding(binding)
            verify_mcp_catalog(skill_catalog)
        except MCPCompatibilityError as exc:
            raise CodexMCPDeploymentError("catalog binding verification failed") from exc
        if actor_thread.role is CodexRole.JUDGE:
            raise CodexMCPDeploymentError("deployment requires an actor Codex role")
        if actor_thread.offered_tools:
            raise CodexMCPDeploymentError("actor tool offers must come only from MCP preflight")
        if "mcp_servers" in actor_thread.config:
            raise CodexMCPDeploymentError("actor config already contains MCP servers")
        skill_names = tuple(row["name"] for row in skill_catalog.to_document()["tools"])
        if skill_names != _SKILL_TOOL_NAMES:
            raise CodexMCPDeploymentError(
                "skill MCP catalog must be unchanged search_skills/load_skill"
            )
        policy_names = tuple(row["name"] for row in binding.catalog.to_document()["tools"])
        if set(policy_names) & set(skill_names):
            raise CodexMCPDeploymentError("policy tools collide with skill loader tools")
        annotations = {
            row["name"]: row for row in binding.catalog.to_document()["annotations"]
        }
        annotations.update(
            {row["name"]: row for row in skill_catalog.to_document()["annotations"]}
        )
        if any(row["visibility"] != "public" for row in annotations.values()):
            raise CodexMCPDeploymentError("actor cannot bind judge-only MCP tools")
        for row in (*binding.catalog.to_document()["tools"], *skill_catalog.to_document()["tools"]):
            _reject_private_schema_keys(row["inputSchema"], path=f"tool.{row['name']}.inputSchema")

        source = Path(source_policy_path).resolve(strict=True)
        raw = source.read_bytes()
        actual = blake3_bytes(raw)
        if actual != binding.source_policy_blake3:
            raise CodexMCPDeploymentError("source policy does not match bound BLAKE3")
        document = _load_json_bytes(raw, label="source policy")
        if not isinstance(document, Mapping):
            raise CodexMCPDeploymentError("source policy root must be an object")
        declared_policy = document.get("schema")
        if isinstance(declared_policy, str) and declared_policy != binding.policy_id:
            raise CodexMCPDeploymentError("source policy id differs from binding")
        declared_candidate = document.get("candidate_id")
        if isinstance(declared_candidate, str) and declared_candidate != binding.candidate_id:
            raise CodexMCPDeploymentError("source policy candidate differs from binding")
        rows = _canonical_policy_rows(document)
        try:
            verify_policy_catalog_binding(
                binding,
                expected_source_policy_blake3=actual,
                canonical_rows=rows,
            )
        except MCPCompatibilityError as exc:
            raise CodexMCPDeploymentError(
                "source policy canonical tools differ from binding"
            ) from exc

        self._binding = binding
        self._skill_catalog = skill_catalog
        self._source_policy_path = str(source)
        self._source_policy_bytes = raw
        self._server = server
        self._actor_thread = actor_thread
        self._policy_rows = rows

    @property
    def binding(self) -> PolicyCatalogBinding:
        return self._binding

    @property
    def expected_tool_names(self) -> tuple[str, ...]:
        policy = tuple(row["name"] for row in self._binding.catalog.to_document()["tools"])
        return (*policy, *_SKILL_TOOL_NAMES)

    async def preflight(self) -> PreparedCodexMCPDeployment:
        """Initialize/list one local server and return an executable Codex config."""

        current = Path(self._source_policy_path).read_bytes()
        if current != self._source_policy_bytes:
            raise CodexMCPDeploymentError("source policy bytes changed before MCP launch")
        initialization: Mapping[str, Any]
        listed: Mapping[str, Any]
        try:
            initialization, listed = await self._probe_stdio()
        finally:
            after = Path(self._source_policy_path).read_bytes()
            if after != self._source_policy_bytes:
                raise CodexMCPDeploymentError("source policy bytes changed during MCP preflight")

        server_info = initialization.get("serverInfo")
        if not isinstance(server_info, Mapping):
            raise CodexMCPDeploymentError("MCP initialize response lacks serverInfo")
        if server_info.get("name") != self._server.server_name:
            raise CodexMCPDeploymentError("MCP server identity differs")
        version = server_info.get("version")
        if not isinstance(version, str) or not version:
            raise CodexMCPDeploymentError("MCP server version differs")
        if self._server.server_version is not None and version != self._server.server_version:
            raise CodexMCPDeploymentError("MCP server version differs")
        if initialization.get("protocolVersion") != self._server.protocol_version:
            raise CodexMCPDeploymentError("MCP protocol version differs")
        capabilities = initialization.get("capabilities")
        if not isinstance(capabilities, Mapping) or not isinstance(capabilities.get("tools"), Mapping):
            raise CodexMCPDeploymentError("MCP server does not advertise tools")

        offers, sidecars = self._verify_listed_tools(listed)
        config = canonical_value(self._actor_thread.config)
        if not isinstance(config, dict):  # pragma: no cover - contract guarantees mapping
            raise CodexMCPDeploymentError("actor config differs")
        config["mcp_servers"] = {
            self._server.config_name: self._server.codex_server_config(
                policy_path=self._source_policy_path,
                policy_blake3=self._binding.source_policy_blake3,
                enabled_tools=self.expected_tool_names,
            )
        }
        thread = replace(
            self._actor_thread,
            config=config,
            offered_tools=offers,
        )
        receipt_core = {
            "schema": CODEX_MCP_PREFLIGHT_SCHEMA,
            "candidate_id": self._binding.candidate_id,
            "policy_id": self._binding.policy_id,
            "policy_binding_blake3": self._binding.binding_blake3,
            "source_policy_blake3": self._binding.source_policy_blake3,
            "policy_schema_catalog_blake3": self._binding.catalog.schema_catalog_blake3,
            "skill_schema_catalog_blake3": self._skill_catalog.schema_catalog_blake3,
            "mcp_config_name": self._server.config_name,
            "mcp_server_name": self._server.server_name,
            "mcp_server_version": version,
            "mcp_protocol_version": self._server.protocol_version,
            "server_launch_blake3": self._server.launch_blake3,
            "offered_tools": tuple(tool.mcp_transport_entry() for tool in offers),
            "sidecar_annotations": sidecars,
            "probe_methods": ("initialize", "tools/list"),
            "provider_calls_made": 0,
            "mcp_tool_calls_made": 0,
        }
        receipt = CodexMCPPreflightReceipt(
            **receipt_core,
            preflight_blake3=blake3_hex(receipt_core),
        )
        return PreparedCodexMCPDeployment(
            binding=self._binding,
            thread_options=thread,
            receipt=receipt,
        )

    async def _probe_stdio(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        environment = self._server.offline_environment(
            policy_path=self._source_policy_path,
            policy_blake3=self._binding.source_policy_blake3,
        )
        process = await asyncio.create_subprocess_exec(
            *self._server.command,
            cwd=self._server.cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            await process.wait()
            raise CodexMCPDeploymentError("MCP stdio pipes are unavailable")
        try:
            initialized = await self._rpc_request(
                process,
                request_id="eva-preflight-initialize",
                method="initialize",
                params={
                    "protocolVersion": self._server.protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": "eva-agent-preflight", "version": "0.1.0"},
                },
            )
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
            process.stdin.write(canonical_json_bytes(notification))
            await process.stdin.drain()
            listed = await self._rpc_request(
                process,
                request_id="eva-preflight-tools-list",
                method="tools/list",
                params={},
            )
            process.stdin.close()
            await process.stdin.wait_closed()
            try:
                return_code = await asyncio.wait_for(
                    process.wait(), timeout=self._server.startup_timeout_seconds
                )
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()
                raise CodexMCPDeploymentError("MCP server did not exit after readiness probe") from None
            stderr = await process.stderr.read()
            if return_code != 0:
                detail = stderr.decode("utf-8", errors="replace")[-1000:]
                raise CodexMCPDeploymentError(
                    f"MCP readiness process exited {return_code}: {detail}"
                )
            return initialized, listed
        except BaseException:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            raise

    async def _rpc_request(
        self,
        process: asyncio.subprocess.Process,
        *,
        request_id: str,
        method: str,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        assert process.stdin is not None and process.stdout is not None
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        process.stdin.write(canonical_json_bytes(request))
        await process.stdin.drain()
        try:
            line = await asyncio.wait_for(
                process.stdout.readline(), timeout=self._server.startup_timeout_seconds
            )
        except asyncio.TimeoutError:
            raise CodexMCPDeploymentError(f"MCP {method} timed out") from None
        if not line:
            raise CodexMCPDeploymentError(f"MCP closed stdout during {method}")
        response = _load_json_bytes(line, label=f"MCP {method} response")
        if not isinstance(response, Mapping) or response.get("jsonrpc") != "2.0":
            raise CodexMCPDeploymentError(f"MCP {method} response differs")
        if response.get("id") != request_id:
            raise CodexMCPDeploymentError(f"MCP {method} response identity differs")
        if "error" in response:
            raise CodexMCPDeploymentError(f"MCP {method} failed: {response['error']}")
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise CodexMCPDeploymentError(f"MCP {method} result must be an object")
        return result

    def _verify_listed_tools(
        self, listed: Mapping[str, Any]
    ) -> tuple[tuple[CodexToolOffer, ...], tuple[Mapping[str, JsonValue], ...]]:
        candidates = listed.get("tools")
        if (
            isinstance(candidates, (str, bytes))
            or not isinstance(candidates, Sequence)
            or not all(isinstance(row, Mapping) for row in candidates)
        ):
            raise CodexMCPDeploymentError("MCP tools/list result differs")
        by_name: dict[str, Mapping[str, Any]] = {}
        for row in candidates:
            name = row.get("name")
            if not isinstance(name, str) or not name:
                raise CodexMCPDeploymentError("MCP tool name differs")
            if name in by_name:
                raise CodexMCPDeploymentError(f"MCP tools/list repeats {name}")
            by_name[name] = row
        if set(by_name) != set(self.expected_tool_names):
            raise CodexMCPDeploymentError(
                "MCP actor tool set differs from exact policy plus skill loaders"
            )

        policy_document = self._binding.catalog.to_document()
        skill_document = self._skill_catalog.to_document()
        expected_tools = tuple(policy_document["tools"]) + tuple(skill_document["tools"])
        expected_annotations = {
            row["name"]: row
            for row in (*policy_document["annotations"], *skill_document["annotations"])
        }
        expected_commitments = {
            row["name"]: row["input_schema_blake3"]
            for row in (*policy_document["schema_commitments"], *skill_document["schema_commitments"])
        }

        offers: list[CodexToolOffer] = []
        sidecars: list[Mapping[str, JsonValue]] = []
        projected_policy: list[dict[str, Any]] = []
        projected_skills: list[dict[str, Any]] = []
        policy_names = {row["name"] for row in policy_document["tools"]}
        for expected in expected_tools:
            name = expected["name"]
            served = by_name[name]
            projection = {
                "name": served.get("name"),
                "description": served.get("description"),
                "inputSchema": canonical_value(served.get("inputSchema")),
            }
            if canonical_json_bytes(projection) != canonical_json_bytes(expected):
                raise CodexMCPDeploymentError(
                    f"MCP canonical tool projection differs for {name}"
                )
            schema_digest = blake3_hex(projection["inputSchema"])
            if schema_digest != expected_commitments[name]:
                raise CodexMCPDeploymentError(f"MCP input schema BLAKE3 differs for {name}")
            _reject_private_schema_keys(
                projection["inputSchema"], path=f"served.{name}.inputSchema"
            )
            annotation = expected_annotations[name]
            if annotation["visibility"] != "public":
                raise CodexMCPDeploymentError(f"actor MCP tool {name} is judge-only")
            standard = served.get("annotations", {})
            metadata = served.get("_meta", {})
            if not isinstance(standard, Mapping) or not isinstance(metadata, Mapping):
                raise CodexMCPDeploymentError(f"MCP annotations differ for {name}")
            evamed = metadata.get("evamed")
            if not isinstance(evamed, Mapping):
                raise CodexMCPDeploymentError(f"MCP EvaMed sidecar is absent for {name}")
            if evamed.get("plane") != "data":
                raise CodexMCPDeploymentError(f"actor MCP tool {name} is not data-plane")
            if evamed.get("canonicalInputSchemaBlake3") != schema_digest:
                raise CodexMCPDeploymentError(f"MCP sidecar schema BLAKE3 differs for {name}")
            for sidecar_key, expected_key in (
                ("parallelSafe", "parallel_safe"),
                ("readOnly", "read_only"),
                ("mutating", "mutating"),
            ):
                if evamed.get(sidecar_key) is not annotation[expected_key]:
                    raise CodexMCPDeploymentError(
                        f"MCP sidecar {sidecar_key} differs for {name}"
                    )
            visibility = evamed.get("visibility", "public")
            audience = evamed.get("audience", "actor")
            if visibility != "public" or audience in {"judge", "judge-only", "private"}:
                raise CodexMCPDeploymentError(f"MCP tool {name} is not actor-visible")
            if standard.get("readOnlyHint") is not annotation["read_only"]:
                raise CodexMCPDeploymentError(f"MCP readOnlyHint differs for {name}")
            allowed_stages = evamed.get("allowedStages", ())
            if (
                isinstance(allowed_stages, (str, bytes))
                or not isinstance(allowed_stages, Sequence)
                or not set(allowed_stages) <= _ALLOWED_STAGES
            ):
                raise CodexMCPDeploymentError(f"MCP allowed stages differ for {name}")

            offer = CodexToolOffer(
                fully_qualified_name=f"{self._server.config_name}/{name}",
                description=projection["description"],
                input_schema=projection["inputSchema"],
                visibility="public",
                parallel_safe=annotation["parallel_safe"],
                read_only=annotation["read_only"],
                allowed_stages=tuple(allowed_stages),
            )
            offers.append(offer)
            sidecars.append(
                {
                    **offer.sidecar_metadata_entry(),
                    "mutating": annotation["mutating"],
                    "mcp_annotations": canonical_value(standard),
                    "mcp_meta": canonical_value(metadata),
                }
            )
            (projected_policy if name in policy_names else projected_skills).append(projection)

        if blake3_hex(projected_policy) != self._binding.catalog.schema_catalog_blake3:
            raise CodexMCPDeploymentError("per-policy MCP schema catalog BLAKE3 differs")
        if blake3_hex(projected_skills) != self._skill_catalog.schema_catalog_blake3:
            raise CodexMCPDeploymentError("skill MCP schema catalog BLAKE3 differs")
        return tuple(offers), tuple(sidecars)


__all__ = [
    "CODEX_MCP_PREFLIGHT_SCHEMA",
    "CodexMCPDeployment",
    "CodexMCPDeploymentError",
    "CodexMCPPreflightReceipt",
    "PreparedCodexMCPDeployment",
    "StdioMCPServerSpec",
    "verify_codex_mcp_preflight_receipt",
]
