from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import pytest

from eva_agent.harness import FunctionCall, SkillCatalog, SkillDocument, ToolRegistry
from eva_agent.mcp_compat import (
    MCPCompatibilityError,
    ToolAnnotations,
    bind_policy_catalog,
    build_mcp_catalog,
    build_policy_catalog_inventory,
    policy_rows_to_mcp_tools,
    skill_catalog_to_mcp_catalog,
    verify_mcp_catalog,
    verify_policy_catalog_binding,
    verify_policy_catalog_inventory,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes


def _policy_rows() -> list[dict]:
    return [
        {
            "name": "lookup_trial",
            "description": "Look up a trial without changing the workspace.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "object",
                        "properties": {
                            "phase": {
                                "type": "string",
                                "enum": ["I", "II", "III"],
                            },
                            "status": {"const": "recruiting"},
                        },
                        "required": ["phase", "status"],
                        "additionalProperties": False,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "write_summary",
            "description": "Write the final summary.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string", "minLength": 1}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    ]


def test_policy_projection_is_exact_and_does_not_mutate_source() -> None:
    rows = _policy_rows()
    source_bytes = canonical_json_bytes(rows)
    source_digest = blake3_hex(rows)

    tools = policy_rows_to_mcp_tools(rows)
    catalog = build_mcp_catalog(
        rows,
        annotations={
            "lookup_trial": ToolAnnotations(
                read_only=True, mutating=False, parallel_safe=True
            )
        },
    )

    assert [set(tool) for tool in tools] == [
        {"name", "description", "inputSchema"},
        {"name", "description", "inputSchema"},
    ]
    for source, projected in zip(rows, tools):
        assert canonical_json_bytes(projected["inputSchema"]) == canonical_json_bytes(
            source["input_schema"]
        )
        assert blake3_hex(projected["inputSchema"]) == blake3_hex(
            source["input_schema"]
        )
    assert catalog.to_document()["annotations"][0] == {
        "name": "lookup_trial",
        "visibility": "public",
        "read_only": True,
        "mutating": False,
        "parallel_safe": True,
    }
    assert not any(
        key in tool["inputSchema"]
        for tool in catalog.to_document()["tools"]
        for key in ("visibility", "read_only", "mutating", "parallel_safe")
    )
    assert canonical_json_bytes(rows) == source_bytes
    assert blake3_hex(rows) == source_digest

    # Projected output is detached: transport-side edits cannot reach policy data.
    tools[0]["inputSchema"]["properties"]["query"]["required"].append("new")
    assert canonical_json_bytes(rows) == source_bytes
    verify_mcp_catalog(catalog, canonical_rows=rows)


def test_catalog_fails_closed_on_tamper_reorder_duplicate_and_invalid_schema() -> None:
    rows = _policy_rows()
    catalog = build_mcp_catalog(rows)
    committed = catalog.catalog_blake3

    tampered = catalog.to_document()
    tampered["tools"][0]["inputSchema"]["required"] = []
    with pytest.raises(MCPCompatibilityError, match="commitment|BLAKE3"):
        verify_mcp_catalog(tampered)

    reordered = catalog.to_document()
    reordered["tools"].reverse()
    with pytest.raises(MCPCompatibilityError, match="order|commitment|BLAKE3"):
        verify_mcp_catalog(reordered)

    duplicate = catalog.to_document()
    duplicate["tools"][1]["name"] = duplicate["tools"][0]["name"]
    with pytest.raises(MCPCompatibilityError, match="duplicate"):
        verify_mcp_catalog(duplicate)

    invalid = _policy_rows()
    invalid[0]["input_schema"]["required"] = "query"
    with pytest.raises(MCPCompatibilityError, match="valid JSON Schema"):
        build_mcp_catalog(invalid)

    with pytest.raises(MCPCompatibilityError, match="expected commitment"):
        verify_mcp_catalog(catalog, expected_catalog_blake3="0" * 64)
    verify_mcp_catalog(catalog, expected_catalog_blake3=committed)


def test_source_duplicate_and_reordered_canonical_catalog_are_rejected() -> None:
    duplicate = _policy_rows()
    duplicate[1]["name"] = duplicate[0]["name"]
    with pytest.raises(MCPCompatibilityError, match="duplicate"):
        build_mcp_catalog(duplicate)

    rows = _policy_rows()
    catalog = build_mcp_catalog(rows)
    with pytest.raises(MCPCompatibilityError, match="canonical policy tools"):
        verify_mcp_catalog(catalog, canonical_rows=list(reversed(rows)))


def test_skill_catalog_search_and_load_contracts_are_unchanged() -> None:
    skills = SkillCatalog(
        [
            SkillDocument(
                skill_id="trial-search",
                description="Search clinical trial registries.",
                content="Use registry identifiers and verify trial status.",
                allowed_stages=frozenset({"S2"}),
            )
        ]
    )
    definitions = skills.tool_definitions()
    source_schema_bytes = tuple(
        canonical_json_bytes(definition.parameters) for definition in definitions
    )

    catalog = skill_catalog_to_mcp_catalog(skills)
    document = catalog.to_document()

    assert [tool["name"] for tool in document["tools"]] == [
        "search_skills",
        "load_skill",
    ]
    assert all(row["read_only"] and not row["mutating"] for row in document["annotations"])
    assert tuple(
        canonical_json_bytes(tool["inputSchema"]) for tool in document["tools"]
    ) == source_schema_bytes

    registry = ToolRegistry(definitions)
    search_result = asyncio.run(
        registry.execute_group(
            (
                FunctionCall(
                    event_id="3bcd9f87-fdc7-48ff-9ea1-38ca02abe501",
                    provider_call_id="search-1",
                    name="search_skills",
                    arguments={"query": "trial", "stage": "S2"},
                ),
            ),
            stage="S2",
            max_parallel_tools=1,
            timeout_seconds=2,
        )
    ).observations[0].output
    assert search_result == {
        "matches": [
            {
                "skill_id": "trial-search",
                "description": "Search clinical trial registries.",
            }
        ]
    }

    load_result = asyncio.run(
        registry.execute_group(
            (
                FunctionCall(
                    event_id="f6b6ea17-fc5c-4303-91b2-e9c01984f087",
                    provider_call_id="load-1",
                    name="load_skill",
                    arguments={"skill_id": "trial-search", "stage": "S2"},
                ),
            ),
            stage="S2",
            max_parallel_tools=1,
            timeout_seconds=2,
        )
    ).observations[0].output
    assert load_result == {
        "skill_id": "trial-search",
        "content": "Use registry identifiers and verify trial status.",
        "content_blake3": blake3_hex(
            "Use registry identifiers and verify trial status."
        ),
        "delivery": "policy-visible-tool-observation",
    }
    assert tuple(
        canonical_json_bytes(definition.parameters) for definition in definitions
    ) == source_schema_bytes


def test_serialized_catalog_round_trip_verifies() -> None:
    catalog = build_mcp_catalog(_policy_rows())
    wire_copy = json.loads(json.dumps(catalog.to_document()))
    reopened = type(catalog).from_document(
        wire_copy, expected_catalog_blake3=catalog.catalog_blake3
    )
    assert reopened.to_document() == catalog.to_document()


def test_inventory_keeps_same_name_schema_variants_bound_per_policy() -> None:
    first_rows = _policy_rows()[:1]
    second_rows = deepcopy(first_rows)
    second_rows[0]["input_schema"]["properties"]["query"]["properties"][
        "status"
    ]["const"] = "completed"
    first = bind_policy_catalog(
        policy_id="sandbox-policy.v2",
        candidate_id="candidate-a",
        source_policy_blake3=blake3_bytes(b"literal policy a bytes"),
        rows=first_rows,
    )
    second = bind_policy_catalog(
        policy_id="sandbox-policy.v2",
        candidate_id="candidate-b",
        source_policy_blake3=blake3_bytes(b"literal policy b bytes"),
        rows=second_rows,
    )

    inventory = build_policy_catalog_inventory((second, first))
    reordered_input = build_policy_catalog_inventory((first, second))

    assert inventory.inventory_blake3 == reordered_input.inventory_blake3
    assert [entry["candidate_id"] for entry in inventory.entries] == [
        "candidate-a",
        "candidate-b",
    ]
    assert [variant["name"] for variant in inventory.schema_variants] == [
        "lookup_trial",
        "lookup_trial",
    ]
    assert len(
        {variant["input_schema_blake3"] for variant in inventory.schema_variants}
    ) == 2
    first.verify(
        expected_source_policy_blake3=first.source_policy_blake3,
        canonical_rows=first_rows,
    )
    second.verify(
        expected_source_policy_blake3=second.source_policy_blake3,
        canonical_rows=second_rows,
    )
    reopened = verify_policy_catalog_binding(
        first.to_document(),
        expected_source_policy_blake3=first.source_policy_blake3,
        canonical_rows=first_rows,
    )
    assert reopened.binding_blake3 == first.binding_blake3
    verify_policy_catalog_inventory(
        inventory, expected_inventory_blake3=inventory.inventory_blake3
    )

    with pytest.raises(MCPCompatibilityError, match="duplicate policy/candidate"):
        build_policy_catalog_inventory((first, first))

    wrong_policy = first.to_document()
    wrong_policy["source_policy_blake3"] = "0" * 64
    with pytest.raises(MCPCompatibilityError, match="binding BLAKE3"):
        verify_policy_catalog_binding(wrong_policy)

    tampered = inventory.to_document()
    tampered["schema_variants"][0]["policy_count"] += 1
    with pytest.raises(MCPCompatibilityError, match="variant inventory"):
        verify_policy_catalog_inventory(tampered)
