"""Restore bound MCP schemas after a recognized native Responses projection.

This is transport compatibility, not argument repair or a replacement catalog:
the original offers remain the only source of names, descriptions and schemas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from eva_agent.codex_runtime import CodexToolOffer
from eva_agent.pipeline.digests import blake3_hex, canonical_value


# The local 0.153.4 MCP capture exercises these erased validation keywords.
# Any different projection still fails the complete-schema comparison below.
_ERASED = frozenset({"$schema", "minLength", "maxLength", "pattern", "minimum", "maximum",
    "format", "maxItems", "uniqueItems"})
_RETAINED = frozenset({"type", "description", "enum", "properties", "required", "items",
    "additionalProperties", "minItems"})

# Full public Responses definitions captured with local Codex 0.153.4, binary
# BLAKE3 6c4d4ccee9b078aba3c9f2f6a2db1cbe735d16e1ade27a38db548856a577fb2d.
# These pins preserve existing native tools; a name alone never authorizes one.
_NATIVE_BUILTIN_DIGESTS = MappingProxyType({
    "list_mcp_resources": "9140e427fd195c5f9998e852e29b96c2eb67538a190d161a4b1cb38b3a45fcba",
    "list_mcp_resource_templates": "04af5399bd552c45cb00ed8e170ce4c97b763e3be953f6b961fa13038ab820ca",
    "read_mcp_resource": "fcd100ff36323a4ec283ce43190a2a96a996bc8f195bb0b078105610fa0218a8",
    "request_user_input": "0092ede01f84071914d0472a178ac9a2cdbf2b8c143bda1449c5f69e1a4c59ea",
})


def _native_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The bounded, observed native lowering; never accept arbitrary deletions."""
    if not isinstance(schema, dict) or set(schema) - (_RETAINED | _ERASED):
        raise ValueError("canonical MCP schema has an unrecognized native projection")
    value = {key: child for key, child in schema.items() if key not in _ERASED}
    for key in ("properties",):
        if key in value:
            value[key] = {name: _native_schema(child) for name, child in value[key].items()}
    for key in ("items", "additionalProperties"):
        if key in value and isinstance(value[key], dict):
            value[key] = _native_schema(value[key])
    schema_type = value.get("type")
    if schema_type is None:
        if set(value) & {"properties", "required", "additionalProperties"}:
            value["type"] = "object"
        elif "items" in value:
            value["type"] = "array"
        elif "enum" in value:
            value["type"] = "string"
    types = value.get("type", [])
    types = [types] if isinstance(types, str) else types
    if "object" in types:
        value.setdefault("properties", {})
    if "array" in types:
        value.setdefault("items", {"type": "string"})
    return value


@dataclass(frozen=True, slots=True)
class CanonicalMCPToolCatalog:
    offers: tuple[CodexToolOffer, ...]
    catalog_blake3: str = field(init=False)
    _by_name: Mapping[str, CodexToolOffer] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.offers, tuple) or any(not isinstance(row, CodexToolOffer) for row in self.offers):
            raise ValueError("canonical MCP catalog requires immutable offers")
        names = {offer.fully_qualified_name: offer for offer in self.offers}
        if len(names) != len(self.offers):
            raise ValueError("canonical MCP catalog has duplicate native identities")
        identity = [{"fully_qualified_name": offer.fully_qualified_name,
            "description": offer.description, "input_schema": offer.input_schema} for offer in self.offers]
        object.__setattr__(self, "catalog_blake3", blake3_hex(identity))
        object.__setattr__(self, "_by_name", MappingProxyType(names))

    def restore_tools(self, responses_tools):
        if responses_tools is not None and not isinstance(responses_tools, list):
            raise ValueError("canonical MCP Responses tools must be an array")
        restored = canonical_value(responses_tools)
        seen = set()
        namespaces = set()
        builtin_count = 0
        restored_count = 0
        servers = {f"mcp__{offer.server}": offer.server for offer in self.offers}
        for namespace in restored or ():
            if not isinstance(namespace, dict):
                raise ValueError("canonical MCP native tool envelope differs")
            name = namespace.get("name")
            if namespace.get("type") == "function":
                if (not isinstance(name, str) or name in seen
                        or blake3_hex(namespace) != _NATIVE_BUILTIN_DIGESTS.get(name)):
                    raise ValueError("canonical MCP native builtin is unknown or changed")
                seen.add(name)
                builtin_count += 1
                continue
            if (set(namespace) != {"type", "name", "description", "tools"}
                    or namespace["type"] != "namespace" or not isinstance(name, str)
                    or name not in servers or name in namespaces
                    or namespace["description"] != f"Tools in the {name} namespace."
                    or not isinstance(namespace["tools"], list)):
                raise ValueError("canonical MCP native namespace differs")
            namespaces.add(name)
            for tool in namespace["tools"]:
                if (not isinstance(tool, dict) or tool.get("type") != "function"
                        or set(tool) != {"type", "name", "description", "parameters", "strict"}
                        or tool["strict"] is not False or not isinstance(tool["name"], str)):
                    raise ValueError("canonical MCP native function envelope differs")
                identity = f"{servers[name]}/{tool['name']}"
                if identity not in self._by_name or identity in seen:
                    raise ValueError("canonical MCP native tool is unknown or duplicated")
                seen.add(identity)
                offer = self._by_name[identity]
                if tool["description"] != offer.description:
                    raise ValueError("canonical MCP native description differs")
                original = canonical_value(offer.input_schema)
                native_digest = blake3_hex(tool["parameters"])
                if native_digest != blake3_hex(original):
                    try:
                        expected = _native_schema(original)
                    except (TypeError, AttributeError) as exc:
                        raise ValueError("canonical MCP native schema projection is unsupported") from exc
                    if native_digest != blake3_hex(expected):
                        raise ValueError("canonical MCP native schema projection differs")
                    restored_count += 1
                tool["parameters"] = original
        return restored, {
            "catalog_blake3": self.catalog_blake3,
            "native_tools_blake3": blake3_hex(responses_tools),
            "restored_tools_blake3": blake3_hex(restored),
            "tool_count": len(seen) - builtin_count, "restored_count": restored_count,
            "native_builtin_count": builtin_count,
            "projection": "codex-0.153.4-mcp-schema-restoration",
        }
