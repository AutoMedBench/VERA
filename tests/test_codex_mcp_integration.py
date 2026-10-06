from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest

from eva_agent.codex_runtime import CodexRole, CodexSandbox, CodexThreadOptions
from eva_agent.deployment import (
    CodexMCPDeployment,
    CodexMCPDeploymentError,
    StdioMCPServerSpec,
    verify_codex_mcp_preflight_receipt,
)
from eva_agent.harness import SkillCatalog, SkillDocument
from eva_agent.mcp_compat import (
    ToolAnnotations,
    bind_policy_catalog,
    skill_catalog_to_mcp_catalog,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SERVER = ROOT / "plugins" / "evamed-codex" / "scripts" / "evamed_mcp.py"


def _schema(maximum: int = 4096) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"code": {"type": "string", "maxLength": maximum}},
        "required": ["code"],
        "additionalProperties": False,
    }


def _policy(
    tmp_path: Path,
    *,
    candidate_id: str = "candidate-a",
    maximum: int = 4096,
    suffix: str = "a",
) -> tuple[Path, list[dict[str, Any]]]:
    rows = [
        {
            "name": "execute_code",
            "description": "Execute the candidate's committed analysis code.",
            "input_schema": _schema(maximum),
        }
    ]
    document = {
        "schema": "rlevo.med-research-sandbox-policy.v2",
        "candidate_id": candidate_id,
        "stage": "S3",
        "tools": [{**rows[0], "parallel_safe": True, "read_only": False}],
    }
    path = tmp_path / f"policy-{suffix}.json"
    path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    return path, rows


def _skills():
    catalog = SkillCatalog(
        [
            SkillDocument(
                skill_id="fixture-skill",
                description="A local fixture skill.",
                content="Use the committed fixture only.",
                allowed_stages=frozenset({"S3"}),
            )
        ]
    )
    return skill_catalog_to_mcp_catalog(catalog)


def _binding(path: Path, rows: list[dict[str, Any]], *, candidate_id: str = "candidate-a", visibility: str = "public"):
    return bind_policy_catalog(
        policy_id="rlevo.med-research-sandbox-policy.v2",
        candidate_id=candidate_id,
        source_policy_blake3=blake3_bytes(path.read_bytes()),
        rows=rows,
        annotations={
            "execute_code": ToolAnnotations(
                visibility=visibility,
                read_only=False,
                mutating=True,
                parallel_safe=True,
            )
        },
    )


def _thread(tmp_path: Path, *, model: str = "gemini-3.1-pro-preview", provider: str = "gemini") -> CodexThreadOptions:
    return CodexThreadOptions(
        role=CodexRole.STRONG_ACTOR,
        model=model,
        provider=provider,
        cwd=str(tmp_path),
        sandbox=CodexSandbox.WORKSPACE_WRITE,
        config={
            "model_providers": {
                provider: {
                    "base_url": "https://provider.invalid/v1",
                    "env_key": "MODEL_PROVIDER_API_KEY",
                }
            }
        },
    )


def _registry_environment(tmp_path: Path, *, maximum: int = 4096) -> dict[str, str]:
    module = tmp_path / "integration_registry.py"
    module.write_text(
        f'''from eva_agent.harness import ToolDefinition, ToolRegistry

SCHEMA = {repr(_schema(maximum))}

async def execute(arguments):
    return {{"ok": True, "code_blake3": "unused-in-readiness"}}

def build():
    return ToolRegistry((ToolDefinition(
        name="execute_code",
        description="Execute the candidate's committed analysis code.",
        parameters=SCHEMA,
        handler=execute,
        parallel_safe=True,
    ),))
''',
        encoding="utf-8",
    )
    return {
        "EVAMED_MCP_REGISTRY_FACTORY": "integration_registry:build",
        "EVAMED_STAGE": "S3",
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(ROOT / "src"))),
    }


def _plugin_spec(tmp_path: Path, *, maximum: int = 4096) -> StdioMCPServerSpec:
    return StdioMCPServerSpec(
        config_name="evamed",
        server_name="evamed-codex",
        server_version="0.1.0",
        command=(str(Path(sys.executable).resolve()), str(PLUGIN_SERVER), "--stdio"),
        cwd=str(ROOT),
        environment=_registry_environment(tmp_path, maximum=maximum),
        startup_timeout_seconds=15,
    )


def _served_tool(tool: Mapping[str, Any], annotation: Mapping[str, Any]) -> dict[str, Any]:
    digest = blake3_hex(tool["inputSchema"])
    return {
        "name": tool["name"],
        "description": tool["description"],
        "inputSchema": deepcopy(tool["inputSchema"]),
        "annotations": {
            "readOnlyHint": annotation["read_only"],
            "destructiveHint": annotation["mutating"],
            "idempotentHint": annotation["read_only"],
            "openWorldHint": False,
        },
        "_meta": {
            "evamed": {
                "plane": "data",
                "available": True,
                "parallelSafe": annotation["parallel_safe"],
                "readOnly": annotation["read_only"],
                "mutating": annotation["mutating"],
                "allowedStages": ["S3"],
                "canonicalInputSchemaBlake3": digest,
                "sourceKind": "fixture",
                "sourceRef": f"fixture:{tool['name']}",
            }
        },
    }


def _fake_tools(binding, skills) -> list[dict[str, Any]]:
    policy = binding.catalog.to_document()
    skill = skills.to_document()
    annotations = {
        row["name"]: row
        for row in (*policy["annotations"], *skill["annotations"])
    }
    return [
        _served_tool(tool, annotations[tool["name"]])
        for tool in (*policy["tools"], *skill["tools"])
    ]


def _fake_server_spec(tmp_path: Path, tools: list[dict[str, Any]]) -> StdioMCPServerSpec:
    script = tmp_path / "fake_mcp.py"
    script.write_text(
        f'''import json
import sys

TOOLS = json.loads({json.dumps(json.dumps(tools))})
for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    if message.get("method") == "initialize":
        result = {{
            "protocolVersion": message["params"]["protocolVersion"],
            "capabilities": {{"tools": {{"listChanged": False}}}},
            "serverInfo": {{"name": "evamed-codex", "version": "0.1.0"}},
        }}
        print(json.dumps({{"jsonrpc": "2.0", "id": message["id"], "result": result}}), flush=True)
    elif message.get("method") == "tools/list":
        print(json.dumps({{"jsonrpc": "2.0", "id": message["id"], "result": {{"tools": TOOLS}}}}), flush=True)
''',
        encoding="utf-8",
    )
    return StdioMCPServerSpec(
        config_name="evamed",
        server_name="evamed-codex",
        server_version="0.1.0",
        command=(str(Path(sys.executable).resolve()), str(script)),
        cwd=str(tmp_path),
    )


def test_plugin_preflight_builds_executable_provider_neutral_codex_config(tmp_path: Path) -> None:
    path, rows = _policy(tmp_path)
    original = path.read_bytes()
    binding = _binding(path, rows)
    deployment = CodexMCPDeployment(
        binding=binding,
        skill_catalog=_skills(),
        source_policy_path=path,
        server=_plugin_spec(tmp_path),
        actor_thread=_thread(tmp_path),
    )

    prepared = asyncio.run(deployment.preflight())
    verify_codex_mcp_preflight_receipt(prepared.receipt)

    assert path.read_bytes() == original
    assert prepared.receipt.provider_calls_made == 0
    assert prepared.receipt.mcp_tool_calls_made == 0
    assert prepared.receipt.probe_methods == ("initialize", "tools/list")
    assert tuple(tool.name for tool in prepared.thread_options.offered_tools) == (
        "execute_code",
        "search_skills",
        "load_skill",
    )
    assert prepared.thread_options.model == "gemini-3.1-pro-preview"
    assert prepared.thread_options.provider == "gemini"
    config = prepared.thread_options.config["mcp_servers"]["evamed"]
    assert config["enabled_tools"] == ("execute_code", "search_skills", "load_skill")
    assert config["required"] is True
    assert config["env"]["EVAMED_MCP_ACTOR_MODE"] == "1"
    assert config["env"]["EVAMED_MCP_POLICY_PATH"] == str(path.resolve())
    assert config["env"]["EVAMED_MCP_POLICY_BLAKE3"] == blake3_bytes(original)
    assert prepared.receipt.preflight_blake3 == blake3_hex(prepared.receipt.core())
    assert all(
        row["fully_qualified_name"].startswith("evamed/")
        and row["visibility"] == "public"
        for row in prepared.receipt.sidecar_annotations
    )


def test_preflight_rejects_server_schema_tamper_and_extra_tool(tmp_path: Path) -> None:
    path, rows = _policy(tmp_path)
    binding = _binding(path, rows)
    skills = _skills()
    tampered = _fake_tools(binding, skills)
    tampered[0]["inputSchema"]["properties"]["code"]["maxLength"] = 1
    tampered[0]["_meta"]["evamed"]["canonicalInputSchemaBlake3"] = blake3_hex(
        tampered[0]["inputSchema"]
    )
    deployment = CodexMCPDeployment(
        binding=binding,
        skill_catalog=skills,
        source_policy_path=path,
        server=_fake_server_spec(tmp_path, tampered),
        actor_thread=_thread(tmp_path),
    )
    with pytest.raises(CodexMCPDeploymentError, match="canonical tool projection"):
        asyncio.run(deployment.preflight())

    extra = _fake_tools(binding, skills)
    extra.append(deepcopy(extra[0]))
    extra[-1]["name"] = "judge_private_workspace"
    deployment = CodexMCPDeployment(
        binding=binding,
        skill_catalog=skills,
        source_policy_path=path,
        server=_fake_server_spec(tmp_path, extra),
        actor_thread=_thread(tmp_path),
    )
    with pytest.raises(CodexMCPDeploymentError, match="exact policy plus skill loaders"):
        asyncio.run(deployment.preflight())


def test_wrong_policy_digest_and_judge_only_tool_fail_before_probe(tmp_path: Path) -> None:
    path, rows = _policy(tmp_path)
    wrong = bind_policy_catalog(
        policy_id="rlevo.med-research-sandbox-policy.v2",
        candidate_id="candidate-a",
        source_policy_blake3="0" * 64,
        rows=rows,
    )
    with pytest.raises(CodexMCPDeploymentError, match="bound BLAKE3"):
        CodexMCPDeployment(
            binding=wrong,
            skill_catalog=_skills(),
            source_policy_path=path,
            server=_plugin_spec(tmp_path),
            actor_thread=_thread(tmp_path),
        )

    judge_only = _binding(path, rows, visibility="judge-only")
    with pytest.raises(CodexMCPDeploymentError, match="judge-only"):
        CodexMCPDeployment(
            binding=judge_only,
            skill_catalog=_skills(),
            source_policy_path=path,
            server=_plugin_spec(tmp_path),
            actor_thread=_thread(tmp_path),
        )


def test_same_name_schema_variants_remain_separate_candidate_deployments(tmp_path: Path) -> None:
    skills = _skills()
    prepared = []
    source_bytes = []
    for suffix, candidate, maximum in (
        ("small", "candidate-small", 64),
        ("large", "candidate-large", 8192),
    ):
        path, rows = _policy(
            tmp_path,
            candidate_id=candidate,
            maximum=maximum,
            suffix=suffix,
        )
        source_bytes.append(path.read_bytes())
        binding = _binding(path, rows, candidate_id=candidate)
        deployment = CodexMCPDeployment(
            binding=binding,
            skill_catalog=skills,
            source_policy_path=path,
            server=_fake_server_spec(tmp_path, _fake_tools(binding, skills)),
            actor_thread=_thread(tmp_path, model="qwen3.5", provider="qwen"),
        )
        prepared.append(asyncio.run(deployment.preflight()))

    first, second = prepared
    first_offer = first.thread_options.offered_tools[0]
    second_offer = second.thread_options.offered_tools[0]
    assert first_offer.name == second_offer.name == "execute_code"
    assert first_offer.input_schema["properties"]["code"]["maxLength"] == 64
    assert second_offer.input_schema["properties"]["code"]["maxLength"] == 8192
    assert blake3_hex(first_offer.input_schema) != blake3_hex(second_offer.input_schema)
    assert first.binding.binding_blake3 != second.binding.binding_blake3
    assert first.receipt.candidate_id == "candidate-small"
    assert second.receipt.candidate_id == "candidate-large"
    assert (tmp_path / "policy-small.json").read_bytes() == source_bytes[0]
    assert (tmp_path / "policy-large.json").read_bytes() == source_bytes[1]


def test_actor_config_cannot_smuggle_an_unpreflighted_mcp_server(tmp_path: Path) -> None:
    path, rows = _policy(tmp_path)
    actor = _thread(tmp_path)
    actor = CodexThreadOptions(
        role=actor.role,
        model=actor.model,
        provider=actor.provider,
        cwd=actor.cwd,
        sandbox=actor.sandbox,
        config={"mcp_servers": {"other": {"url": "https://example.invalid/mcp"}}},
    )
    with pytest.raises(CodexMCPDeploymentError, match="already contains MCP"):
        CodexMCPDeployment(
            binding=_binding(path, rows),
            skill_catalog=_skills(),
            source_policy_path=path,
            server=_plugin_spec(tmp_path),
            actor_thread=actor,
        )


def test_source_policy_bytes_and_nested_schema_are_never_rewritten(tmp_path: Path) -> None:
    path, rows = _policy(tmp_path)
    before = canonical_json_bytes(rows)
    binding = _binding(path, rows)
    skills = _skills()
    deployment = CodexMCPDeployment(
        binding=binding,
        skill_catalog=skills,
        source_policy_path=path,
        server=_fake_server_spec(tmp_path, _fake_tools(binding, skills)),
        actor_thread=_thread(tmp_path),
    )
    asyncio.run(deployment.preflight())
    assert canonical_json_bytes(rows) == before
    parsed = json.loads(path.read_bytes())
    assert canonical_json_bytes(parsed["tools"][0]["input_schema"]) == canonical_json_bytes(
        rows[0]["input_schema"]
    )
