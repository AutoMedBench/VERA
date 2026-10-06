from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import pytest

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "evamed-codex"
SERVER = PLUGIN / "scripts" / "evamed_mcp.py"
ADAPTER_DIR = PLUGIN / "scripts"
LEGACY_SOURCE = (
    ROOT.parent
    / "rlevo-med-research"
    / "harness"
    / "source"
    / "rlevo-Med-RL-data"
    / "rev-79dd2a31f5f"
)

sys.path.insert(0, str(ADAPTER_DIR))
from registry_adapter import (  # noqa: E402
    FORBIDDEN_PUBLIC_KEYS,
    LEGACY_TOOL_NAMES,
    AdapterError,
    build_data_plane,
    load_verified_policy,
    plugin_skill_catalog,
)


def _rpc(messages: list[dict[str, Any]], *, env: Mapping[str, str] | None = None) -> dict[Any, Any]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    completed = subprocess.run(
        [sys.executable, str(SERVER), "--stdio"],
        cwd=ROOT,
        env=merged,
        input="".join(json.dumps(message) + "\n" for message in messages),
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    return {response.get("id"): response for response in responses}


def _call(request_id: int, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": dict(arguments)},
    }


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        if FORBIDDEN_PUBLIC_KEYS & {str(key) for key in value}:
            return True
        return any(_contains_forbidden_key(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def test_plugin_manifest_is_repo_native_and_points_to_real_components() -> None:
    manifest = json.loads((PLUGIN / ".codex-plugin" / "plugin.json").read_bytes())
    assert manifest["name"] == "evamed-codex"
    assert manifest["skills"] == "./skills/"
    assert manifest["mcpServers"] == "./.mcp.json"
    assert not (ROOT / ".agents" / "plugins" / "marketplace.json").exists()
    config = json.loads((PLUGIN / ".mcp.json").read_bytes())["mcpServers"]["evamed"]
    assert config["args"] == ["./scripts/evamed_mcp.py", "--stdio"]
    assert SERVER.is_file()
    assert sorted(path.parent.name for path in (PLUGIN / "skills").glob("*/SKILL.md")) == [
        "sandbox-construction",
        "stage-rollout",
        "trajectory-sft",
        "workspace-agent-judge",
    ]


def test_legacy_skill_manifest_is_exact_174_to_24_external_inventory() -> None:
    manifest_path = PLUGIN / "references" / "legacy-skill-manifest.v1.json"
    manifest = json.loads(manifest_path.read_bytes())
    core = dict(manifest)
    digest = core.pop("manifest_blake3")
    assert digest == blake3_hex(core)
    assert manifest["occurrence_count"] == 174
    assert manifest["unique_content_count"] == 24
    assert manifest["source_revision"] == "79dd2a31f5f"
    assert manifest["source_license_marker"] == "other"
    assert manifest["redistribution"] == "external-only-license-unresolved"
    assert len({row["skill_id"] for row in manifest["skills"]}) == 24
    assert len({row["content_blake3"] for row in manifest["skills"]}) == 24
    assert sum(len(row["source_paths"]) for row in manifest["skills"]) == 174
    for row in manifest["skills"]:
        raw = (LEGACY_SOURCE / row["canonical_source_path"]).read_bytes()
        assert len(raw) == row["bytes"]
        assert blake3_bytes(raw) == row["content_blake3"]
    subprocess.run(
        [sys.executable, str(PLUGIN / "scripts" / "build_legacy_skill_manifest.py"), "--verify"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_actual_skill_catalog_schema_and_handlers_are_preserved() -> None:
    catalog = plugin_skill_catalog(PLUGIN, {})
    canonical = {item.name: item for item in catalog.tool_definitions()}
    adapted = {item.name: item for item in build_data_plane(PLUGIN, {})}
    assert set(adapted) == {"search_skills", "load_skill"}
    for name, definition in canonical.items():
        offered = adapted[name].mcp_definition()
        assert offered["name"] == definition.name
        assert offered["description"] == definition.description
        assert offered["inputSchema"] == definition.parameters
        assert offered["_meta"]["evamed"]["canonicalInputSchemaBlake3"] == blake3_hex(
            definition.parameters
        )
        assert offered["_meta"]["evamed"]["readOnly"] is True

    search = asyncio.run(adapted["search_skills"].invoke({"query": "pilot", "stage": "S3"}))
    assert "pilot-recovery-validation-med" in {
        row["skill_id"] for row in search["matches"]
    }
    hidden = asyncio.run(adapted["search_skills"].invoke({"query": "pilot", "stage": "S2"}))
    assert "pilot-recovery-validation-med" not in {
        row["skill_id"] for row in hidden["matches"]
    }
    with pytest.raises(Exception, match="unavailable for this stage"):
        asyncio.run(
            adapted["load_skill"].invoke(
                {"skill_id": "pilot-recovery-validation-med", "stage": "S2"}
            )
        )
    loaded = asyncio.run(
        adapted["load_skill"].invoke(
            {"skill_id": "pilot-recovery-validation-med", "stage": "S3"}
        )
    )
    manifest = json.loads(
        (PLUGIN / "references" / "legacy-skill-manifest.v1.json").read_bytes()
    )
    row = next(item for item in manifest["skills"] if item["skill_id"] == loaded["skill_id"])
    assert loaded["content"].encode() == (LEGACY_SOURCE / row["canonical_source_path"]).read_bytes()


def test_control_schemas_are_strict_discoverable_and_private_free() -> None:
    responses = _rpc(
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}]
    )
    tools = responses[1]["result"]["tools"]
    assert {"capabilities", "search", "load", "search_skills", "load_skill"} <= {
        tool["name"] for tool in tools
    }
    for tool in tools:
        Draft202012Validator.check_schema(tool["inputSchema"])
        assert not _contains_forbidden_key(tool)
        metadata = tool["_meta"]["evamed"]
        assert {"parallelSafe", "readOnly", "mutating"} <= set(metadata)
        if metadata.get("plane") != "data":
            assert tool["inputSchema"]["additionalProperties"] is False
            assert "Use when" in tool["description"]
            assert "Do not use" in tool["description"]
            Draft202012Validator.check_schema(tool["outputSchema"])


def test_policy_schema_variants_remain_candidate_bound_and_unavailable_without_handler(
    tmp_path: Path,
) -> None:
    variants = []
    for index, maximum in enumerate((64, 4096), start=1):
        schema = {
            "type": "object",
            "properties": {"code": {"type": "string", "maxLength": maximum}},
            "required": ["code"],
            "additionalProperties": False,
        }
        document = {
            "schema": "rlevo.med-research-sandbox-policy.v2",
            "stage": "S3",
            "tools": [
                {
                    "name": "execute_code",
                    "description": f"Pinned execute variant {index}.",
                    "input_schema": schema,
                }
            ],
        }
        path = tmp_path / f"policy-{index}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        rows = load_verified_policy(str(path), blake3_bytes(path.read_bytes()))
        assert len(rows) == 1
        assert rows[0].mcp_definition()["inputSchema"] == schema
        variants.append(rows[0])
    assert variants[0].schema_blake3 != variants[1].schema_blake3
    assert variants[0].description != variants[1].description

    env = {
        "EVAMED_MCP_POLICY_PATH": str(tmp_path / "policy-1.json"),
        "EVAMED_MCP_POLICY_BLAKE3": blake3_bytes((tmp_path / "policy-1.json").read_bytes()),
    }
    responses = _rpc(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            _call(2, "execute_code", {"code": "pass"}),
        ],
        env=env,
    )
    listed = next(tool for tool in responses[1]["result"]["tools"] if tool["name"] == "execute_code")
    assert listed["inputSchema"] == variants[0].input_schema
    assert listed["_meta"]["evamed"]["available"] is False
    assert responses[2]["result"]["isError"] is True
    assert "handler_unavailable" in responses[2]["result"]["structuredContent"]["error"]


FAKE_REGISTRY = '''
import asyncio
import time
from eva_agent.harness import ToolDefinition, ToolRegistry

active = 0
maximum = 0

def schema(name):
    return {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False}

async def invoke(name, arguments):
    global active, maximum
    started = time.monotonic_ns()
    active += 1
    maximum = max(maximum, active)
    await asyncio.sleep(0.12)
    observed = maximum
    active -= 1
    return {"tool": name, "value": arguments["value"], "started_ns": started, "ended_ns": time.monotonic_ns(), "max_active": observed}

def handler(name):
    async def run(arguments):
        return await invoke(name, arguments)
    return run

def build():
    definitions = []
    for name in ("materialize_plan", "retrieve_frozen_evidence", "materialize_evidence_selection", "execute_code", "submit_results", "reopen_s4_artifact"):
        definitions.append(ToolDefinition(name=name, description=f"Canonical {name} fixture.", parameters=schema(name), handler=handler(name), parallel_safe=True))
    definitions.append(ToolDefinition(name="parallel_probe", description="Canonical parallel probe fixture.", parameters=schema("parallel_probe"), handler=handler("parallel_probe"), parallel_safe=True))
    definitions.append(ToolDefinition(name="unsafe_probe", description="Canonical unsafe probe fixture.", parameters=schema("unsafe_probe"), handler=handler("unsafe_probe"), parallel_safe=False))
    return ToolRegistry(definitions)
'''


def _fake_factory(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "fake_registry.py").write_text(FAKE_REGISTRY, encoding="utf-8")
    return {
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ),
        "EVAMED_MCP_REGISTRY_FACTORY": "fake_registry:build",
        "EVAMED_MCP_MAX_PARALLEL": "8",
    }


def test_verified_policy_wires_all_six_legacy_tools_and_forwards_call(tmp_path: Path) -> None:
    env = _fake_factory(tmp_path)
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    policy = {
        "schema": "rlevo.med-research-sandbox-policy.v2",
        "stage": "E2E",
        "tools": [
            {
                "name": name,
                "description": f"Canonical {name} fixture.",
                "input_schema": schema,
            }
            for name in LEGACY_TOOL_NAMES
        ],
    }
    path = tmp_path / "candidate-policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    env.update(
        {
            "EVAMED_MCP_POLICY_PATH": str(path),
            "EVAMED_MCP_POLICY_BLAKE3": blake3_bytes(path.read_bytes()),
            "EVAMED_MCP_ACTOR_MODE": "1",
        }
    )
    responses = _rpc(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            _call(2, "materialize_plan", {"value": 7}),
        ],
        env=env,
    )
    listed = {
        tool["name"]: tool for tool in responses[1]["result"]["tools"]
    }
    assert set(listed) == {*LEGACY_TOOL_NAMES, "search_skills", "load_skill"}
    assert set(LEGACY_TOOL_NAMES) <= set(listed)
    for name in LEGACY_TOOL_NAMES:
        assert listed[name]["inputSchema"] == schema
        assert listed[name]["description"] == f"Canonical {name} fixture."
        assert listed[name]["_meta"]["evamed"]["available"] is True
    result = responses[2]["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["tool"] == "materialize_plan"
    assert result["structuredContent"]["value"] == 7


@pytest.mark.parametrize(
    ("tool_name", "expect_overlap"),
    [("parallel_probe", True), ("unsafe_probe", False)],
)
def test_stdio_tools_call_respects_parallel_safe_metadata(
    tmp_path: Path, tool_name: str, expect_overlap: bool
) -> None:
    responses = _rpc(
        [_call(1, tool_name, {"value": 1}), _call(2, tool_name, {"value": 2})],
        env=_fake_factory(tmp_path),
    )
    first = responses[1]["result"]["structuredContent"]
    second = responses[2]["result"]["structuredContent"]
    overlap = max(first["started_ns"], second["started_ns"]) < min(
        first["ended_ns"], second["ended_ns"]
    )
    assert overlap is expect_overlap
    if expect_overlap:
        assert max(first["max_active"], second["max_active"]) >= 2
    else:
        assert max(first["max_active"], second["max_active"]) == 1


def test_no_work_order_or_skill_claims_unsigned_admission() -> None:
    order = {
        "operation": "stage-rollout",
        "candidate_id": "00000000-0000-4000-8000-000000000001",
        "domain": "medxpertqa",
        "stage": "S3",
        "rubric_blake3": "a" * 64,
        "cohort_rollouts": {"weak": 1, "middle": 1, "strong": 1},
        "parallel_tool_calls": True,
        "retry_count": 0,
        "signed_receipts_required": True,
    }
    result = _rpc([_call(1, "validate_work_order", order)])[1]["result"]
    assert result["isError"] is False
    payload = result["structuredContent"]
    assert payload["rubric_binding_use"] == "judge-and-reward"
    assert payload["provider_calls_made"] == 0
    assert "admitted" not in payload and "admission" not in payload
    sft = (PLUGIN / "skills" / "trajectory-sft" / "SKILL.md").read_text()
    assert "signed admission" in sft.lower()
    assert "supervisor" in sft.lower()


def test_policy_digest_or_schema_drift_fails_closed(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "execute_code",
                        "description": "Pinned execution.",
                        "input_schema": {"type": "object", "additionalProperties": False},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AdapterError, match="does not match policy bytes"):
        load_verified_policy(str(policy), "0" * 64)
