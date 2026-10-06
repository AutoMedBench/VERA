"""Schema-preserving projection of canonical EvaMed tools into MCP catalogs.

This module is deliberately a transport adapter.  It does not decorate, relax,
or otherwise rewrite a canonical input schema.  Operational metadata lives in
an independently committed sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from eva_agent.pipeline.digests import (
    CanonicalValueError,
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
)


CATALOG_SCHEMA = "eva.schema-preserving-mcp-catalog.v1"
_POLICY_ROW_KEYS = frozenset({"name", "description", "input_schema"})
_MCP_TOOL_KEYS = frozenset({"name", "description", "inputSchema"})
_COMMITMENT_KEYS = frozenset({"name", "input_schema_blake3"})
_ANNOTATION_KEYS = frozenset(
    {"name", "visibility", "read_only", "mutating", "parallel_safe"}
)
_CATALOG_KEYS = frozenset(
    {
        "schema",
        "tools",
        "schema_commitments",
        "annotations",
        "schema_catalog_blake3",
        "catalog_blake3",
    }
)


class MCPCompatibilityError(ValueError):
    """A canonical schema or its MCP projection failed a closed boundary."""


def _json_copy(value: Any, *, label: str) -> Any:
    """Return detached canonical JSON data and reject non-JSON values."""

    try:
        copied = canonical_value(value)
        # In particular this rejects NaN/Infinity, which ``canonical_value``
        # intentionally leaves for the canonical encoder to diagnose.
        canonical_json_bytes(copied)
        return copied
    except (CanonicalValueError, TypeError, ValueError, OverflowError) as exc:
        raise MCPCompatibilityError(f"{label} is not canonical JSON") from exc


def _freeze(value: Any) -> Any:
    """Deeply freeze already-normalized JSON without sharing source objects."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _mutable(value: Any) -> Any:
    """Create ordinary dict/list JSON containers for a protocol boundary."""

    if isinstance(value, Mapping):
        return {key: _mutable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_mutable(child) for child in value]
    return value


def _exact_mapping(value: Any, keys: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise MCPCompatibilityError(f"{label} keys differ")
    return value


def _valid_identity(name: Any, description: Any, *, label: str) -> tuple[str, str]:
    if not isinstance(name, str) or not name or not isinstance(description, str) or not description.strip():
        raise MCPCompatibilityError(f"{label} name or description differs")
    return name, description


def _validated_schema(value: Any, *, label: str) -> dict[str, Any]:
    schema = _json_copy(value, label=label)
    if not isinstance(schema, dict):
        raise MCPCompatibilityError(f"{label} must be an object schema")
    if schema.get("type") != "object":
        raise MCPCompatibilityError(f"{label} must describe an object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise MCPCompatibilityError(f"{label} is not a valid JSON Schema") from exc
    return schema


@dataclass(frozen=True, slots=True)
class ToolAnnotations:
    """MCP operational metadata kept outside canonical ``inputSchema``."""

    visibility: str = "public"
    read_only: bool = False
    mutating: bool = True
    parallel_safe: bool = False

    def __post_init__(self) -> None:
        if self.visibility not in {"public", "judge-only"}:
            raise MCPCompatibilityError("tool visibility differs")
        if any(
            type(value) is not bool
            for value in (self.read_only, self.mutating, self.parallel_safe)
        ):
            raise MCPCompatibilityError("tool annotations require boolean flags")
        if self.read_only == self.mutating:
            raise MCPCompatibilityError(
                "tool must be exactly one of read-only or mutating"
            )

    def to_document(self, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "visibility": self.visibility,
            "read_only": self.read_only,
            "mutating": self.mutating,
            "parallel_safe": self.parallel_safe,
        }


def _annotation(value: ToolAnnotations | Mapping[str, Any], *, label: str) -> ToolAnnotations:
    if isinstance(value, ToolAnnotations):
        return value
    row = _exact_mapping(
        value,
        frozenset({"visibility", "read_only", "mutating", "parallel_safe"}),
        label=label,
    )
    return ToolAnnotations(
        visibility=row["visibility"],
        read_only=row["read_only"],
        mutating=row["mutating"],
        parallel_safe=row["parallel_safe"],
    )


def _normalize_annotations(
    names: Sequence[str],
    annotations: Mapping[str, ToolAnnotations | Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    supplied = {} if annotations is None else annotations
    if not isinstance(supplied, Mapping) or any(not isinstance(key, str) for key in supplied):
        raise MCPCompatibilityError("annotations must be keyed by tool name")
    unknown = set(supplied) - set(names)
    if unknown:
        raise MCPCompatibilityError(f"annotations name an unknown tool: {sorted(unknown)[0]}")
    rows: list[dict[str, Any]] = []
    for name in names:
        item = _annotation(supplied[name], label=f"annotations[{name}]") if name in supplied else ToolAnnotations()
        rows.append(item.to_document(name))
    return tuple(rows)


def _project_policy_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise MCPCompatibilityError("canonical tool rows must be a sequence")
    projected: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, candidate in enumerate(rows):
        row = _exact_mapping(candidate, _POLICY_ROW_KEYS, label=f"canonical tool row {index}")
        name, description = _valid_identity(
            row["name"], row["description"], label=f"canonical tool row {index}"
        )
        if name in names:
            raise MCPCompatibilityError(f"duplicate tool name: {name}")
        names.add(name)
        try:
            source_schema_bytes = canonical_json_bytes(row["input_schema"])
            source_schema_blake3 = blake3_hex(row["input_schema"])
        except (CanonicalValueError, TypeError, ValueError, OverflowError) as exc:
            raise MCPCompatibilityError(
                f"canonical tool row {name} input_schema is not canonical JSON"
            ) from exc
        schema = _validated_schema(
            row["input_schema"], label=f"canonical tool row {name} input_schema"
        )
        # This is an explicit invariant rather than relying on ordinary Python
        # equality (where, for example, True == 1).
        if (
            canonical_json_bytes(schema) != source_schema_bytes
            or blake3_hex(schema) != source_schema_blake3
        ):
            raise MCPCompatibilityError(f"canonical schema changed while mapping {name}")
        projected.append(
            {"name": name, "description": description, "inputSchema": schema}
        )
    return tuple(projected)


def policy_rows_to_mcp_tools(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Mechanically rename only ``input_schema`` to MCP ``inputSchema``.

    Returned containers are detached from the source.  Row order is retained.
    """

    return [_mutable(row) for row in _project_policy_rows(rows)]


def _catalog_core(
    *,
    tools: Sequence[Mapping[str, Any]],
    schema_commitments: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    schema_catalog_blake3: str,
) -> dict[str, Any]:
    return {
        "schema": CATALOG_SCHEMA,
        "tools": list(tools),
        "schema_commitments": list(schema_commitments),
        "annotations": list(annotations),
        "schema_catalog_blake3": schema_catalog_blake3,
    }


@dataclass(frozen=True, slots=True)
class MCPToolCatalog:
    """An immutable verified MCP catalog plus its out-of-schema sidecars."""

    tools: tuple[Mapping[str, Any], ...]
    schema_commitments: tuple[Mapping[str, Any], ...]
    annotations: tuple[Mapping[str, Any], ...]
    schema_catalog_blake3: str
    catalog_blake3: str
    schema: str = CATALOG_SCHEMA

    def __post_init__(self) -> None:
        document = {
            "schema": self.schema,
            "tools": list(self.tools),
            "schema_commitments": list(self.schema_commitments),
            "annotations": list(self.annotations),
            "schema_catalog_blake3": self.schema_catalog_blake3,
            "catalog_blake3": self.catalog_blake3,
        }
        _verify_document(document)
        object.__setattr__(self, "tools", tuple(_freeze(_json_copy(row, label="MCP tool")) for row in self.tools))
        object.__setattr__(
            self,
            "schema_commitments",
            tuple(_freeze(_json_copy(row, label="schema commitment")) for row in self.schema_commitments),
        )
        object.__setattr__(
            self,
            "annotations",
            tuple(_freeze(_json_copy(row, label="tool annotation")) for row in self.annotations),
        )

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        *,
        expected_catalog_blake3: str | None = None,
    ) -> "MCPToolCatalog":
        verify_mcp_catalog(
            document, expected_catalog_blake3=expected_catalog_blake3
        )
        return cls(
            schema=document["schema"],
            tools=tuple(document["tools"]),
            schema_commitments=tuple(document["schema_commitments"]),
            annotations=tuple(document["annotations"]),
            schema_catalog_blake3=document["schema_catalog_blake3"],
            catalog_blake3=document["catalog_blake3"],
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tools": _mutable(self.tools),
            "schema_commitments": _mutable(self.schema_commitments),
            "annotations": _mutable(self.annotations),
            "schema_catalog_blake3": self.schema_catalog_blake3,
            "catalog_blake3": self.catalog_blake3,
        }

    def verify(self, *, expected_catalog_blake3: str | None = None) -> None:
        verify_mcp_catalog(
            self, expected_catalog_blake3=expected_catalog_blake3
        )


def build_mcp_catalog(
    rows: Sequence[Mapping[str, Any]],
    *,
    annotations: Mapping[str, ToolAnnotations | Mapping[str, Any]] | None = None,
) -> MCPToolCatalog:
    """Build and commit an MCP catalog from exact canonical policy rows."""

    tools = _project_policy_rows(rows)
    names = tuple(tool["name"] for tool in tools)
    commitments = tuple(
        {
            "name": tool["name"],
            "input_schema_blake3": blake3_hex(tool["inputSchema"]),
        }
        for tool in tools
    )
    annotation_rows = _normalize_annotations(names, annotations)
    schema_catalog_blake3 = blake3_hex(tools)
    core = _catalog_core(
        tools=tools,
        schema_commitments=commitments,
        annotations=annotation_rows,
        schema_catalog_blake3=schema_catalog_blake3,
    )
    return MCPToolCatalog(
        tools=tools,
        schema_commitments=commitments,
        annotations=annotation_rows,
        schema_catalog_blake3=schema_catalog_blake3,
        catalog_blake3=blake3_hex(core),
    )


def _definition_rows(definitions: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    if isinstance(definitions, (str, bytes)) or not isinstance(definitions, Sequence):
        raise MCPCompatibilityError("tool definitions must be a sequence")
    rows: list[dict[str, Any]] = []
    for index, definition in enumerate(definitions):
        try:
            schema = (
                definition.input_schema
                if hasattr(definition, "input_schema")
                else definition.parameters
            )
            rows.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "input_schema": schema,
                }
            )
        except AttributeError as exc:
            raise MCPCompatibilityError(
                f"tool definition {index} lacks its canonical contract"
            ) from exc
    return tuple(rows)


def _definition_annotations(
    definitions: Sequence[Any],
    overrides: Mapping[str, ToolAnnotations | Mapping[str, Any]] | None,
    *,
    force_read_only: bool = False,
) -> dict[str, ToolAnnotations | Mapping[str, Any]]:
    result: dict[str, ToolAnnotations | Mapping[str, Any]] = {}
    for definition in definitions:
        read_only = True if force_read_only else bool(getattr(definition, "read_only", False))
        mutating = False if read_only else bool(getattr(definition, "mutating", True))
        result[definition.name] = ToolAnnotations(
            visibility=str(getattr(definition, "visibility", "public")),
            read_only=read_only,
            mutating=mutating,
            parallel_safe=bool(getattr(definition, "parallel_safe", False)),
        )
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise MCPCompatibilityError("annotations must be keyed by tool name")
        result.update(overrides)
    return result


def tool_definitions_to_mcp_catalog(
    definitions: Sequence[Any],
    *,
    annotations: Mapping[str, ToolAnnotations | Mapping[str, Any]] | None = None,
) -> MCPToolCatalog:
    """Adapt pipeline or harness ToolDefinitions without changing handlers."""

    definitions = tuple(definitions)
    return build_mcp_catalog(
        _definition_rows(definitions),
        annotations=_definition_annotations(definitions, annotations),
    )


def tool_definitions_to_mcp_tools(definitions: Sequence[Any]) -> list[dict[str, Any]]:
    """Return only the MCP list projection for existing ToolDefinitions."""

    return tool_definitions_to_mcp_catalog(definitions).to_document()["tools"]


def skill_catalog_to_mcp_catalog(
    skill_catalog: Any,
    *,
    annotations: Mapping[str, ToolAnnotations | Mapping[str, Any]] | None = None,
) -> MCPToolCatalog:
    """Map SkillCatalog's search/load ToolDefinitions as read-only MCP tools."""

    factory = getattr(skill_catalog, "tool_definitions", None)
    if not callable(factory):
        raise MCPCompatibilityError("skill catalog lacks tool_definitions()")
    definitions = tuple(factory())
    return build_mcp_catalog(
        _definition_rows(definitions),
        annotations=_definition_annotations(
            definitions, annotations, force_read_only=True
        ),
    )


def skill_catalog_to_mcp_tools(skill_catalog: Any) -> list[dict[str, Any]]:
    """Return only the MCP list projection for an existing SkillCatalog."""

    return skill_catalog_to_mcp_catalog(skill_catalog).to_document()["tools"]


def _verify_document(document: Mapping[str, Any]) -> None:
    catalog = _exact_mapping(document, _CATALOG_KEYS, label="MCP catalog")
    if catalog["schema"] != CATALOG_SCHEMA:
        raise MCPCompatibilityError("MCP catalog schema differs")
    tools = catalog["tools"]
    commitments = catalog["schema_commitments"]
    annotations = catalog["annotations"]
    if any(
        isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple))
        for value in (tools, commitments, annotations)
    ):
        raise MCPCompatibilityError("MCP catalog arrays differ")
    if not (len(tools) == len(commitments) == len(annotations)):
        raise MCPCompatibilityError("MCP catalog sidecars are not aligned")

    names: list[str] = []
    for index, candidate in enumerate(tools):
        row = _exact_mapping(candidate, _MCP_TOOL_KEYS, label=f"MCP tool {index}")
        name, _description = _valid_identity(
            row["name"], row["description"], label=f"MCP tool {index}"
        )
        if name in names:
            raise MCPCompatibilityError(f"duplicate tool name: {name}")
        names.append(name)
        schema = _validated_schema(row["inputSchema"], label=f"MCP tool {name} inputSchema")
        if canonical_json_bytes(schema) != canonical_json_bytes(row["inputSchema"]):
            raise MCPCompatibilityError(f"MCP tool {name} schema is not canonical")

    commitment_names: list[str] = []
    for index, candidate in enumerate(commitments):
        row = _exact_mapping(candidate, _COMMITMENT_KEYS, label=f"schema commitment {index}")
        name = row["name"]
        if not isinstance(name, str):
            raise MCPCompatibilityError("schema commitment name differs")
        commitment_names.append(name)
        expected = blake3_hex(tools[index]["inputSchema"])
        if row["input_schema_blake3"] != expected:
            raise MCPCompatibilityError(f"input schema commitment differs for {name}")

    annotation_names: list[str] = []
    for index, candidate in enumerate(annotations):
        row = _exact_mapping(candidate, _ANNOTATION_KEYS, label=f"annotation {index}")
        name = row["name"]
        if not isinstance(name, str):
            raise MCPCompatibilityError("annotation name differs")
        annotation_names.append(name)
        _annotation(
            {
                "visibility": row["visibility"],
                "read_only": row["read_only"],
                "mutating": row["mutating"],
                "parallel_safe": row["parallel_safe"],
            },
            label=f"annotation {name}",
        )
    if names != commitment_names or names != annotation_names:
        raise MCPCompatibilityError("MCP catalog order or sidecar names differ")

    expected_schema_catalog = blake3_hex(tools)
    if catalog["schema_catalog_blake3"] != expected_schema_catalog:
        raise MCPCompatibilityError("MCP schema catalog BLAKE3 differs")
    expected_catalog = blake3_hex(
        _catalog_core(
            tools=tools,
            schema_commitments=commitments,
            annotations=annotations,
            schema_catalog_blake3=expected_schema_catalog,
        )
    )
    if catalog["catalog_blake3"] != expected_catalog:
        raise MCPCompatibilityError("MCP catalog BLAKE3 differs")


def verify_mcp_catalog(
    catalog: MCPToolCatalog | Mapping[str, Any],
    *,
    expected_catalog_blake3: str | None = None,
    canonical_rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Reopen all commitments and optionally pin them to a source catalog."""

    document = catalog.to_document() if isinstance(catalog, MCPToolCatalog) else catalog
    _verify_document(document)
    if expected_catalog_blake3 is not None and document["catalog_blake3"] != expected_catalog_blake3:
        raise MCPCompatibilityError("MCP catalog does not match the expected commitment")
    if canonical_rows is not None:
        expected_tools = _project_policy_rows(canonical_rows)
        if canonical_json_bytes(document["tools"]) != canonical_json_bytes(expected_tools):
            raise MCPCompatibilityError("MCP catalog differs from canonical policy tools")


# Short, discoverable aliases for callers that describe this operation as mapping.
map_policy_tools = policy_rows_to_mcp_tools
map_tool_definitions = tool_definitions_to_mcp_tools
map_skill_catalog = skill_catalog_to_mcp_tools


__all__ = [
    "CATALOG_SCHEMA",
    "MCPCompatibilityError",
    "MCPToolCatalog",
    "ToolAnnotations",
    "build_mcp_catalog",
    "map_policy_tools",
    "map_skill_catalog",
    "map_tool_definitions",
    "policy_rows_to_mcp_tools",
    "skill_catalog_to_mcp_catalog",
    "skill_catalog_to_mcp_tools",
    "tool_definitions_to_mcp_catalog",
    "tool_definitions_to_mcp_tools",
    "verify_mcp_catalog",
]
