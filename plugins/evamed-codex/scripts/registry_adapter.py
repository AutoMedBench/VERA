#!/usr/bin/env python3
"""Schema-preserving adapters from EvaMed registries and policies to MCP.

Canonical EvaMed input schemas are never enriched here.  Discovery hints and
execution metadata live beside the schema in MCP annotations and ``_meta``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from eva_agent.harness import SkillCatalog, SkillDocument
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value


STAGES = frozenset({"S1", "S2", "S3", "S4", "S5", "E2E"})
LEGACY_TOOL_NAMES = (
    "materialize_plan",
    "retrieve_frozen_evidence",
    "materialize_evidence_selection",
    "execute_code",
    "submit_results",
    "reopen_s4_artifact",
)
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer_key",
        "judge_only_reference",
        "private_rubric",
        "private_rubric_items",
        "rubric_items",
    }
)


class AdapterError(ValueError):
    """A registry, policy, or invocation failed a compatibility check."""


Handler = Callable[[Mapping[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class AdaptedTool:
    """One canonical tool schema plus side-channel execution information."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: Handler | None
    parallel_safe: bool
    read_only: bool
    allowed_stages: frozenset[str]
    source_kind: str
    source_ref: str

    def __post_init__(self) -> None:
        if not self.name or not self.description:
            raise AdapterError("tool name and description are required")
        if not self.allowed_stages or not self.allowed_stages <= STAGES:
            raise AdapterError("tool stages must be a non-empty S1-S5/E2E subset")
        schema = canonical_value(self.input_schema)
        if not isinstance(schema, Mapping):
            raise AdapterError("canonical tool input schema must be an object")
        Draft202012Validator.check_schema(dict(schema))
        _assert_no_private_keys(schema)
        object.__setattr__(self, "input_schema", schema)

    @property
    def available(self) -> bool:
        return self.handler is not None

    @property
    def schema_blake3(self) -> str:
        return blake3_hex(self.input_schema)

    def mcp_definition(self) -> dict[str, Any]:
        """Map only ``input_schema`` -> ``inputSchema``; keep it deeply equal."""

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": canonical_value(self.input_schema),
            "annotations": {
                "title": self.name.replace("_", " ").title(),
                "readOnlyHint": self.read_only,
                "destructiveHint": not self.read_only,
                "idempotentHint": self.read_only,
                "openWorldHint": False,
            },
            "_meta": {
                "evamed": {
                    "plane": "data",
                    "available": self.available,
                    "parallelSafe": self.parallel_safe,
                    "readOnly": self.read_only,
                    "mutating": not self.read_only,
                    "allowedStages": sorted(self.allowed_stages),
                    "canonicalInputSchemaBlake3": self.schema_blake3,
                    "sourceKind": self.source_kind,
                    "sourceRef": self.source_ref,
                }
            },
        }

    async def invoke(self, arguments: Mapping[str, Any]) -> Any:
        if self.handler is None:
            raise AdapterError(
                f"handler_unavailable: {self.name} has a verified schema but no injected handler"
            )
        errors = sorted(
            Draft202012Validator(dict(self.input_schema)).iter_errors(dict(arguments)),
            key=lambda error: tuple(str(part) for part in error.path),
        )
        if errors:
            raise AdapterError(
                f"arguments violate canonical {self.name} schema: {errors[0].message}"
            )
        if inspect.iscoroutinefunction(self.handler):
            result = await self.handler(arguments)
        else:
            result = await asyncio.to_thread(self.handler, arguments)
            if inspect.isawaitable(result):
                result = await result
        return canonical_value(result)


def _assert_no_private_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        overlap = FORBIDDEN_PUBLIC_KEYS & {str(key) for key in value}
        if overlap:
            raise AdapterError(
                f"public contract contains private keys: {', '.join(sorted(overlap))}"
            )
        for child in value.values():
            _assert_no_private_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_private_keys(child)


def _skill_frontmatter(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise AdapterError(f"skill frontmatter is absent: {path}")
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise AdapterError(f"skill frontmatter is not closed: {path}") from None
    values: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    name = values.get("name", "")
    description = values.get("description", "")
    if not name or not description or name != path.parent.name:
        raise AdapterError(f"skill identity differs from its path: {path}")
    return description, text


def _legacy_skill_documents(
    plugin_root: Path, environ: Mapping[str, str]
) -> tuple[SkillDocument, ...]:
    manifest_path = plugin_root / "references" / "legacy-skill-manifest.v1.json"
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, Mapping):
        raise AdapterError("legacy skill manifest root must be an object")
    manifest_core = dict(manifest)
    expected_manifest_digest = manifest_core.pop("manifest_blake3", None)
    if expected_manifest_digest != blake3_hex(manifest_core):
        raise AdapterError("legacy skill manifest BLAKE3 differs")
    source_text = environ.get("EVAMED_LEGACY_SKILL_ROOT")
    if source_text:
        source = Path(source_text)
    else:
        source = (
            plugin_root.parents[2]
            / "rlevo-med-research"
            / "harness"
            / "source"
            / "rlevo-Med-RL-data"
            / f"rev-{manifest['source_revision']}"
        )
    if not source.is_dir():
        raise AdapterError(
            "legacy skill source is unavailable; set EVAMED_LEGACY_SKILL_ROOT"
        )
    readme = source / "README.md"
    if blake3_bytes(readme.read_bytes()) != manifest["source_readme_blake3"]:
        raise AdapterError("legacy source README commitment differs")
    documents = []
    for row in manifest.get("skills", ()):
        if not isinstance(row, Mapping):
            raise AdapterError("legacy skill manifest entry must be an object")
        path = source / str(row["canonical_source_path"])
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or blake3_bytes(raw) != row["content_blake3"]:
            raise AdapterError(f"legacy skill bytes differ: {row['skill_id']}")
        documents.append(
            SkillDocument(
                skill_id=str(row["skill_id"]),
                description=str(row["description"]),
                content=raw.decode("utf-8"),
                allowed_stages=frozenset(str(stage) for stage in row["allowed_stages"]),
            )
        )
    if len(documents) != manifest["unique_content_count"] or len(documents) != 24:
        raise AdapterError("legacy skill manifest cardinality differs")
    return tuple(documents)


def plugin_skill_catalog(
    plugin_root: Path, environ: Mapping[str, str] | None = None
) -> SkillCatalog:
    env = os.environ if environ is None else environ
    documents: list[SkillDocument] = []
    for path in sorted((plugin_root / "skills").glob("*/SKILL.md")):
        description, content = _skill_frontmatter(path)
        documents.append(
            SkillDocument(
                skill_id=path.parent.name,
                description=description,
                content=content,
                allowed_stages=STAGES,
            )
        )
    documents.extend(_legacy_skill_documents(plugin_root, env))
    if not documents:
        raise AdapterError("plugin skill catalog is empty")
    return SkillCatalog(documents)


def _adapt_harness_definition(definition: Any, *, source_ref: str) -> AdaptedTool:
    schema = definition.response_api_schema()
    parameters = canonical_value(schema["parameters"])
    if parameters != canonical_value(definition.parameters):
        raise AdapterError(f"harness schema projection differs for {definition.name}")
    return AdaptedTool(
        name=definition.name,
        description=definition.description,
        input_schema=parameters,
        handler=definition.handler,
        parallel_safe=bool(definition.parallel_safe),
        read_only=bool(
            getattr(definition, "read_only", False)
            or definition.name in {"search_skills", "load_skill"}
        ),
        allowed_stages=frozenset(definition.allowed_stages),
        source_kind="harness-tool-registry",
        source_ref=source_ref,
    )


def _adapt_pipeline_registry(
    registry: Any, *, workspace: Any | None, source_ref: str
) -> tuple[AdaptedTool, ...]:
    rows: list[AdaptedTool] = []
    for public in registry.public_schemas():
        function = public.get("function") if isinstance(public, Mapping) else None
        if not isinstance(function, Mapping):
            raise AdapterError("pipeline public schema lacks function object")
        name = function.get("name")
        if not isinstance(name, str):
            raise AdapterError("pipeline public schema lacks tool name")
        definition = registry.definition(name)
        projected = canonical_value(function.get("parameters"))
        canonical = canonical_value(definition.input_schema)
        if projected != canonical:
            raise AdapterError(f"pipeline schema projection differs for {name}")

        handler: Handler | None = None
        if workspace is not None:
            source_handler = definition.handler

            def invoke(
                arguments: Mapping[str, Any],
                *,
                _handler: Callable[..., Any] = source_handler,
                _workspace: Any = workspace,
            ) -> Any:
                return _handler(_workspace, arguments)

            handler = invoke
        rows.append(
            AdaptedTool(
                name=name,
                description=str(function.get("description", definition.description)),
                input_schema=canonical,
                handler=handler,
                parallel_safe=bool(definition.parallel_safe),
                read_only=bool(definition.read_only),
                allowed_stages=STAGES,
                source_kind="pipeline-tool-registry",
                source_ref=source_ref,
            )
        )
    return tuple(rows)


def _adapt_harness_registry(
    registry: Any, *, stage: str, source_ref: str
) -> tuple[AdaptedTool, ...]:
    if stage not in STAGES:
        raise AdapterError("EVAMED_STAGE must be S1-S5 or E2E")
    definitions = getattr(registry, "_definitions", None)
    if not isinstance(definitions, Mapping):
        raise AdapterError("harness ToolRegistry does not expose verified definitions")
    offered = registry.schemas_for_stage(stage)
    rows: list[AdaptedTool] = []
    for public in offered:
        if not isinstance(public, Mapping) or not isinstance(public.get("name"), str):
            raise AdapterError("harness public schema lacks tool name")
        definition = definitions[public["name"]]
        adapted = _adapt_harness_definition(definition, source_ref=source_ref)
        if canonical_value(public["parameters"]) != adapted.input_schema:
            raise AdapterError(f"harness schema projection differs for {adapted.name}")
        rows.append(adapted)
    return tuple(rows)


def adapt_registry(
    registry: Any,
    *,
    workspace: Any | None = None,
    stage: str = "E2E",
    source_ref: str = "injected:ToolRegistry",
) -> tuple[AdaptedTool, ...]:
    """Adapt either existing EvaMed ToolRegistry shape without schema changes."""

    if hasattr(registry, "schemas_for_stage"):
        return _adapt_harness_registry(registry, stage=stage, source_ref=source_ref)
    if hasattr(registry, "public_schemas") and hasattr(registry, "definition"):
        return _adapt_pipeline_registry(
            registry, workspace=workspace, source_ref=source_ref
        )
    raise AdapterError("factory did not return an EvaMed ToolRegistry")


def _policy_entries(document: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    candidates: Any = document.get("tools") or document.get("tool_definitions")
    if candidates is None and isinstance(document.get("policy"), Mapping):
        candidates = document["policy"].get("tools")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise AdapterError("verified policy has no tool sequence")
    if not all(isinstance(item, Mapping) for item in candidates):
        raise AdapterError("verified policy tool entries must be objects")
    return candidates


def _policy_shape(entry: Mapping[str, Any]) -> tuple[str, str, Mapping[str, Any]]:
    function = entry.get("function")
    source = function if isinstance(function, Mapping) else entry
    name = source.get("name")
    description = source.get("description")
    schema = source.get("input_schema", source.get("parameters", source.get("inputSchema")))
    if not isinstance(name, str) or not isinstance(description, str) or not isinstance(schema, Mapping):
        raise AdapterError("verified policy tool shape is unsupported")
    return name, description, schema


def load_verified_policy(path_text: str, expected_blake3: str) -> tuple[AdaptedTool, ...]:
    path = Path(path_text).resolve(strict=True)
    raw = path.read_bytes()
    actual = blake3_bytes(raw)
    if actual != expected_blake3:
        raise AdapterError("EVAMED_MCP_POLICY_BLAKE3 does not match policy bytes")
    document = json.loads(raw)
    if not isinstance(document, Mapping):
        raise AdapterError("verified policy root must be an object")
    policy = document.get("policy") if isinstance(document.get("policy"), Mapping) else {}
    declared_stage = document.get("stage", document.get("focus", policy.get("stage")))
    allowed_stages = (
        frozenset({str(declared_stage)}) if declared_stage in STAGES else STAGES
    )
    rows = []
    for entry in _policy_entries(document):
        name, description, schema = _policy_shape(entry)
        rows.append(
            AdaptedTool(
                name=name,
                description=description,
                input_schema=schema,
                handler=None,
                parallel_safe=bool(
                    entry.get("parallel_safe", entry.get("x-eva-parallel-safe", False))
                ),
                read_only=bool(entry.get("read_only", False)),
                allowed_stages=allowed_stages,
                source_kind="verified-policy",
                source_ref=f"{path}:{actual}",
            )
        )
    return tuple(rows)


def _factory_product(specification: str) -> Any:
    if specification.count(":") != 1:
        raise AdapterError("EVAMED_MCP_REGISTRY_FACTORY must be module:function")
    module_name, attribute = specification.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise AdapterError("registry factory is not callable")
    product = factory()
    if inspect.isawaitable(product):
        raise AdapterError("registry factory must be synchronous; handlers may be async")
    return product


def _merge_tools(rows: Iterable[AdaptedTool]) -> tuple[AdaptedTool, ...]:
    merged: dict[str, AdaptedTool] = {}
    for row in rows:
        prior = merged.get(row.name)
        if prior is None:
            merged[row.name] = row
            continue
        if (
            prior.description != row.description
            or prior.input_schema != row.input_schema
            or prior.schema_blake3 != row.schema_blake3
        ):
            raise AdapterError(f"canonical schema collision for {row.name}")
        if prior.handler is None and row.handler is not None:
            merged[row.name] = replace(
                row,
                source_ref=f"{prior.source_ref}+{row.source_ref}",
            )
    return tuple(merged[name] for name in sorted(merged))


def build_data_plane(plugin_root: Path, environ: Mapping[str, str] | None = None) -> tuple[AdaptedTool, ...]:
    env = os.environ if environ is None else environ
    skill_catalog = plugin_skill_catalog(plugin_root, env)
    skill_rows: list[AdaptedTool] = [
        _adapt_harness_definition(
            definition, source_ref="eva_agent.harness.SkillCatalog"
        )
        for definition in skill_catalog.tool_definitions()
    ]

    policy_path = env.get("EVAMED_MCP_POLICY_PATH")
    policy_digest = env.get("EVAMED_MCP_POLICY_BLAKE3")
    if bool(policy_path) != bool(policy_digest):
        raise AdapterError(
            "EVAMED_MCP_POLICY_PATH and EVAMED_MCP_POLICY_BLAKE3 are required together"
        )
    policy_rows: tuple[AdaptedTool, ...] = ()
    if policy_path and policy_digest:
        policy_rows = load_verified_policy(policy_path, policy_digest)

    registry_rows: tuple[AdaptedTool, ...] = ()
    specification = env.get("EVAMED_MCP_REGISTRY_FACTORY")
    if specification:
        product = _factory_product(specification)
        workspace = None
        stage = env.get("EVAMED_STAGE", "E2E")
        source_ref = f"factory:{specification}"
        if isinstance(product, Mapping):
            registry = product.get("registry")
            workspace = product.get("workspace")
            stage = str(product.get("stage", stage))
            source_ref = str(product.get("source_ref", source_ref))
        else:
            registry = product
        registry_rows = adapt_registry(
                registry,
                workspace=workspace,
                stage=stage,
                source_ref=source_ref,
            )
    if policy_rows:
        policy_names = {tool.name for tool in policy_rows}
        registry_rows = tuple(tool for tool in registry_rows if tool.name in policy_names)
        return _merge_tools((*skill_rows, *policy_rows, *registry_rows))
    return _merge_tools((*skill_rows, *registry_rows))


def data_plane_digest(tools: Sequence[AdaptedTool]) -> str:
    return blake3_hex(
        tuple(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema_blake3": tool.schema_blake3,
                "available": tool.available,
                "allowed_stages": sorted(tool.allowed_stages),
                "source_ref": tool.source_ref,
            }
            for tool in tools
        )
    )


__all__ = [
    "AdaptedTool",
    "AdapterError",
    "FORBIDDEN_PUBLIC_KEYS",
    "LEGACY_TOOL_NAMES",
    "STAGES",
    "adapt_registry",
    "build_data_plane",
    "data_plane_digest",
    "load_verified_policy",
    "plugin_skill_catalog",
]
