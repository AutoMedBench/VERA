"""Public 0.153.4 capture shapes; no provider, runtime process or GPU required."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from eva_agent.codex_providers.canonical_mcp_schemas import CanonicalMCPToolCatalog
from eva_agent.codex_runtime import CodexToolOffer
from eva_agent.pipeline.digests import blake3_hex, canonical_value


ORIGINAL = {"type": "object", "additionalProperties": False, "required": ["stage", "code"],
    "properties": {"stage": {"enum": ["S3", "S4"]},
        "code": {"type": "string", "minLength": 1, "maxLength": 1024, "pattern": "^x"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        "values": {"type": "array", "items": {"type": "number"},
            "minItems": 1, "maxItems": 3, "uniqueItems": True},
        "payload": {"type": "object", "additionalProperties": True},
        "record": {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object", "properties": {"date": {"type": "string", "format": "date"}}}}}
NATIVE = {"type": "object", "additionalProperties": False, "required": ["stage", "code"],
    "properties": {"stage": {"type": "string", "enum": ["S3", "S4"]},
        "code": {"type": "string"}, "limit": {"type": "integer"},
        "values": {"type": "array", "items": {"type": "number"}, "minItems": 1},
        "payload": {"type": "object", "additionalProperties": True, "properties": {}},
        "record": {"type": "object", "properties": {"date": {"type": "string"}}}}}
OFFER = CodexToolOffer("evamed/execute_code", "Exact public fixture.", ORIGINAL)
READ_RESOURCE = {"type": "function", "name": "read_mcp_resource", "strict": False,
    "description": "Read a specific resource from an MCP server given the server name and resource URI.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["server", "uri"],
        "properties": {"server": {"type": "string", "description": "MCP server name exactly as configured. Must match the 'server' field returned by list_mcp_resources."},
            "uri": {"type": "string", "description": "Resource URI to read. Must be one of the URIs returned by list_mcp_resources."}}}}


def _tools(schema=NATIVE):
    return [deepcopy(READ_RESOURCE), {"type": "namespace", "name": "mcp__evamed",
        "description": "Tools in the mcp__evamed namespace.", "tools": [{"type": "function",
            "name": "execute_code", "description": OFFER.description,
            "parameters": deepcopy(schema), "strict": False}]}]


@pytest.mark.parametrize("schema,restored_count", [(ORIGINAL, 0), (NATIVE, 1)])
def test_known_native_projection_restores_exact_original_without_mutation(schema, restored_count):
    catalog = CanonicalMCPToolCatalog((OFFER,))
    source = _tools(schema)
    before = blake3_hex(source)
    restored, audit = catalog.restore_tools(source)
    assert restored[1]["tools"][0]["parameters"] == ORIGINAL
    assert restored[0] == source[0] == READ_RESOURCE
    assert blake3_hex(source) == before
    assert audit == {"catalog_blake3": catalog.catalog_blake3,
        "native_tools_blake3": before, "restored_tools_blake3": blake3_hex(restored),
        "tool_count": 1, "restored_count": restored_count, "native_builtin_count": 1,
        "projection": "codex-0.153.4-mcp-schema-restoration"}
    restored[1]["tools"][0]["parameters"]["required"].append("invented")
    assert canonical_value(OFFER.input_schema) == ORIGINAL
    assert blake3_hex(source) == before


def test_catalog_is_frozen_copy_isolated_and_binds_server_description_and_schema():
    source = deepcopy(ORIGINAL)
    offer = CodexToolOffer("evamed/execute_code", OFFER.description, source)
    catalog = CanonicalMCPToolCatalog((offer,))
    source["required"].clear()
    assert canonical_value(catalog.offers[0].input_schema) == ORIGINAL
    with pytest.raises(FrozenInstanceError):
        catalog.offers = ()
    with pytest.raises(TypeError):
        catalog.offers[0].input_schema["type"] = "string"
    identities = {CanonicalMCPToolCatalog((row,)).catalog_blake3 for row in (
        offer, replace(offer, fully_qualified_name="other/execute_code"),
        replace(offer, description="Changed."), replace(offer, input_schema=NATIVE))}
    assert len(identities) == 4
    with pytest.raises(ValueError, match="duplicate"):
        CanonicalMCPToolCatalog((offer, offer))
    with pytest.raises(ValueError, match="immutable"):
        CanonicalMCPToolCatalog([offer])


@pytest.mark.parametrize("empty", [None, []])
def test_requests_without_tools_are_unchanged(empty):
    restored, audit = CanonicalMCPToolCatalog((OFFER,)).restore_tools(empty)
    assert restored == empty
    assert audit["tool_count"] == audit["native_builtin_count"] == audit["restored_count"] == 0


@pytest.mark.parametrize("mutation", ["builtin_description", "builtin_schema", "builtin_duplicate",
    "unknown_builtin", "namespace", "namespace_description", "namespace_duplicate", "child_duplicate",
    "unknown_child", "description", "strict", "extra_field", "required", "enum", "minItems",
    "partial_projection", "boolean_instead_of_integer", "unknown_projection"])
def test_changed_or_unknown_native_surface_fails_closed(mutation):
    source = _tools()
    namespace = source[1]
    child = namespace["tools"][0]
    parameters = child["parameters"]
    if mutation == "builtin_description": source[0]["description"] += "changed"
    elif mutation == "builtin_schema": source[0]["parameters"]["required"] = []
    elif mutation == "builtin_duplicate": source.append(deepcopy(source[0]))
    elif mutation == "unknown_builtin": source[0]["name"] = "run_shell"
    elif mutation == "namespace": namespace["name"] = "mcp__unknown"
    elif mutation == "namespace_description": namespace["description"] += "changed"
    elif mutation == "namespace_duplicate": source.append(deepcopy(namespace))
    elif mutation == "child_duplicate": namespace["tools"].append(deepcopy(child))
    elif mutation == "unknown_child": child["name"] = "invented"
    elif mutation == "description": child["description"] += "changed"
    elif mutation == "strict": child["strict"] = True
    elif mutation == "extra_field": child["defer_loading"] = False
    elif mutation == "required": parameters["required"] = ["code"]
    elif mutation == "enum": parameters["properties"]["stage"]["enum"] = ["S3"]
    elif mutation == "minItems": del parameters["properties"]["values"]["minItems"]
    elif mutation == "partial_projection": parameters["properties"]["code"]["minLength"] = 1
    elif mutation == "boolean_instead_of_integer": parameters["properties"]["values"]["minItems"] = True
    else: parameters["properties"]["stage"]["type"] = "integer"
    with pytest.raises(ValueError):
        CanonicalMCPToolCatalog((OFFER,)).restore_tools(source)


def test_unobserved_schema_lowering_is_not_silently_accepted():
    original = {**ORIGINAL, "unobserved_keyword": True}
    catalog = CanonicalMCPToolCatalog((replace(OFFER, input_schema=original),))
    restored, _ = catalog.restore_tools(_tools(original))
    assert restored[1]["tools"][0]["parameters"] == original
    with pytest.raises(ValueError, match="unrecognized"):
        catalog.restore_tools(_tools())
