"""Version-two checks for unchanged, read-only S2 skill frontiers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator
from eva_agent.harness.skills import SkillCatalog
from eva_agent.pipeline.digests import blake3_hex, canonical_value, is_blake3

SKILL_NAMES = frozenset({"search_skills", "load_skill"})


def _error(message: str):
    from .s2_frontier_prefix_sft import S2FrontierPrefixSFTError
    return S2FrontierPrefixSFTError(message)


def canonical_skill_tools() -> tuple[Mapping[str, Any], ...]:
    return tuple(canonical_value({"name": tool.name, "description": tool.description,
                                  "input_schema": tool.parameters})
                 for tool in SkillCatalog(()).tool_definitions())


def require_skill_delivery(delivery: Any) -> Mapping[str, Any]:
    required = {"schema", "mode", "catalog_blake3", "discovery_tools",
                "initial_skill_mount_count", "visible_skill_count", "visible_skill_ids"}
    if not isinstance(delivery, Mapping) or set(delivery) != required:
        raise _error("S2 progressive skill delivery shape differs")
    visible = delivery.get("visible_skill_ids")
    if (delivery.get("schema") != "eva.teacher-progressive-skill-surface.v1"
        or delivery.get("mode") != "search-then-load"
        or not is_blake3(delivery.get("catalog_blake3"))
        or delivery.get("discovery_tools") != ["search_skills", "load_skill"]
        or type(delivery.get("initial_skill_mount_count")) is not int or delivery.get("initial_skill_mount_count") != 0
        or not isinstance(visible, list) or not visible
        or any(not isinstance(value, str) or not value for value in visible)
        or visible != sorted(set(visible)) or delivery.get("visible_skill_count") != len(visible)):
        raise _error("S2 progressive skill delivery binding differs")
    return canonical_value(delivery)


def verified_offered_catalog(source_tools: Sequence[Mapping[str, Any]], receipt: Any,
                             delivery: Any) -> tuple[Mapping[str, Any], ...]:
    require_skill_delivery(delivery)
    if any(row["name"] in SKILL_NAMES for row in source_tools):
        raise _error("S2 canonical source tools collide with skill tools")
    catalog = tuple(sorted((*(canonical_value({key: row[key] for key in
        ("name", "description", "input_schema")}) for row in source_tools),
        *canonical_skill_tools()), key=lambda row: row["name"]))
    if (len({row["name"] for row in catalog}) != len(catalog)
        or tuple(receipt.offered_mcp_tool_names) != tuple("evamed/" + row["name"] for row in catalog)
        or receipt.offered_tool_schema_blake3 != blake3_hex(catalog)):
        raise _error("S2 offered canonical skill/tool catalog differs from Codex receipt")
    return catalog


def require_skill_prefix(receipt: Any, expected_names: Sequence[str], delivery: Any):
    """Observe every original call and validate actual execution groups, not renumbered ones."""
    from .s2_frontier_prefix_sft import _mcp_observation
    visible = set(require_skill_delivery(delivery)["visible_skill_ids"])
    calls = tuple(receipt.tool_calls)
    terminal = next((i for i, call in enumerate(calls)
                     if call.fully_qualified_name == "evamed/materialize_evidence_selection"), None)
    if terminal is None:
        raise _error("S2 skill-aware selection frontier is absent")
    prefix = calls[:terminal + 1]
    names = tuple(call.fully_qualified_name.removeprefix("evamed/") for call in prefix)
    if (not names or names[0] != "retrieve_frozen_evidence"
        or tuple(name for name in names if name not in SKILL_NAMES) != tuple(expected_names)):
        raise _error("S2 skill-aware guided control sequence differs")
    observed = tuple(_mcp_observation(call, expected_name=name) for call, name in zip(prefix, names, strict=True))
    indices = [item.tool_result["frontier"] for item in observed]
    if (any(type(index) is not int or index < 0 for index in indices)
        or indices != sorted(indices) or sorted(set(indices)) != list(range(max(indices) + 1))):
        raise _error("S2 actual execution frontier indices differ")
    group_ids = []
    for index in sorted(set(indices)):
        group = [item for item in observed if item.tool_result["frontier"] == index]
        identities = {item.tool_result["parallel_group_id"] for item in group}
        if len(identities) != 1 or (len(group) > 1 and any(item.call.fully_qualified_name.removeprefix("evamed/") not in SKILL_NAMES for item in group)):
            raise _error("S2 actual execution group binding differs")
        group_ids.extend(identities)
    if len(group_ids) != len(set(group_ids)):
        raise _error("S2 execution group identity was reused")
    definitions = {row["name"]: row for row in canonical_skill_tools()}
    for item, name in zip(observed, names, strict=True):
        if name not in SKILL_NAMES:
            continue
        if (item.arguments.get("stage") != "S2"
            or list(Draft202012Validator(definitions[name]["input_schema"]).iter_errors(canonical_value(item.arguments)))
            or item.tool_result["workspace_before_blake3"] != item.tool_result["workspace_after_blake3"]):
            raise _error("S2 skill call stage, canonical arguments, or read-only boundary differs")
        output = item.output
        if name == "search_skills":
            matches = output.get("matches")
            if set(output) != {"matches"} or not isinstance(matches, list):
                raise _error("S2 skill discovery output differs")
            ids = []
            for match in matches:
                if (not isinstance(match, Mapping) or set(match) != {"skill_id", "description"}
                    or match.get("skill_id") not in visible or not isinstance(match.get("description"), str)
                    or not match["description"]
                    or item.arguments["query"].casefold() not in f"{match['skill_id']} {match['description']}".casefold()):
                    raise _error("S2 skill discovery visibility or query binding differs")
                ids.append(match["skill_id"])
            if ids != sorted(set(ids)):
                raise _error("S2 skill discovery ordering differs")
        elif (set(output) != {"skill_id", "content", "content_blake3", "delivery"}
              or output.get("skill_id") != item.arguments.get("skill_id")
              or output.get("skill_id") not in visible
              or not isinstance(output.get("content"), str) or not output["content"]
              or output.get("content_blake3") != blake3_hex(output["content"])
              or output.get("delivery") != "policy-visible-tool-observation"):
            raise _error("S2 loaded skill content or visibility differs")
    controls = tuple(item for item, name in zip(observed, names, strict=True) if name not in SKILL_NAMES)
    return observed, controls


def skill_export_fields(observed: Sequence[Any], delivery: Any,
                        catalog: Sequence[Mapping[str, Any]], receipt: Any,
                        messages: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    skills = [item for item in observed if item.call.fully_qualified_name.removeprefix("evamed/") in SKILL_NAMES]
    prefix = []
    for item in skills:
        name = item.call.fully_qualified_name.removeprefix("evamed/")
        prefix.extend((
            {"role": "assistant", "content": {"tool_calls": [{"name": name,
                "arguments": item.arguments, "codex_tool_call_receipt_blake3": item.call.receipt_blake3}]}},
            {"role": "tool", "name": name, "content": item.observation},
        ))
    messages[2:2] = prefix
    return {
        "skill_delivery": canonical_value(delivery), "offered_tool_catalog": canonical_value(catalog),
        "offered_tool_catalog_blake3": blake3_hex(catalog),
        "codex_offered_tool_schema_blake3": receipt.offered_tool_schema_blake3,
        "skill_frontiers": [{"name": item.call.fully_qualified_name.removeprefix("evamed/"),
            "frontier": item.tool_result["frontier"], "parallel_group_id": item.tool_result["parallel_group_id"],
            "codex_tool_call_id": item.call.tool_call_id, "codex_tool_call_receipt_blake3": item.call.receipt_blake3,
            "tool_result_receipt_blake3": item.tool_result["receipt_blake3"],
            "bridge_receipt_blake3": item.observation["bridge_receipt_blake3"]} for item in skills],
        "loss_message_indices": [len(messages) - 2], "loss_bearing_assistant_tool_decisions": 1,
        "assistant_tool_decisions": len(skills) + 1, "host_tool_observations": len(skills) + 1,
        "skill_tool_observations_exported": len(skills), "skill_observations_used_as_context_only": True,
    }
