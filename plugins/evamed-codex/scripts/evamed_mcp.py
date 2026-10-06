#!/usr/bin/env python3
"""Concurrent, compatibility-first stdio MCP bridge for EvaMed.

Control-plane helpers are plugin-owned.  Data-plane tools come only from a
verified EvaMed policy or injected ToolRegistry, and their canonical schemas
are exposed without enrichment or rewriting.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, AsyncIterator, Callable, Mapping, Sequence
from uuid import UUID


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PLUGIN_ROOT.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
REPOSITORY_PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"
if (
    REPOSITORY_PYTHON.is_file()
    and Path(sys.prefix).resolve() != (REPOSITORY_ROOT / ".venv").resolve()
):
    os.execv(
        str(REPOSITORY_PYTHON),
        (str(REPOSITORY_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]),
    )
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from eva_agent.orchestration import ReceiptJournal  # noqa: E402
from eva_agent.pipeline.digests import blake3_hex, canonical_value, is_blake3  # noqa: E402

from registry_adapter import (  # noqa: E402
    AdaptedTool,
    AdapterError,
    LEGACY_TOOL_NAMES,
    build_data_plane,
    data_plane_digest,
)


SERVER_NAME = "evamed-codex"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2025-06-18"
STAGES = ("S1", "S2", "S3", "S4", "S5", "E2E")
DOMAINS = (
    "agentclinic",
    "automedbench-classification",
    "automedbench-detection",
    "automedbench-segmentation",
    "medxpertqa",
)
OPERATIONS = (
    "sandbox-construction",
    "stage-rollout",
    "workspace-agent-judge",
    "trajectory-sft",
)

_CAPABILITY_RECORDS: dict[str, dict[str, Any]] = {
    "sandbox-construction": {
        "summary": "Materialize one answer-free benchmark episode with one immutable rubric binding.",
        "when": "Use when a pinned benchmark episode must become a candidate sandbox before rollout.",
        "when_not": "Do not use to run providers, score a trajectory, admit a candidate, or export SFT.",
        "runtime_symbols": (
            "eva_agent.rubrics.load_and_compile_registry",
            "eva_agent.pipeline.BenchmarkEpisode",
            "eva_agent.pipeline.VerifiableDataPipeline",
        ),
        "input_contract": (
            "pinned source episode and revision",
            "domain and S1-S5/E2E stage",
            "one compiled rubric binding by BLAKE3",
            "answer-free policy context and initial files",
        ),
        "output_contract": (
            "UUID candidate and sandbox identities",
            "immutable manifest and file BLAKE3 commitments",
            "local reopening result; no admission decision",
        ),
        "evidence_ref_contract": (
            "context:policy-visible",
            "workspace:before:<relative-path>",
        ),
        "parallel_safe": True,
        "read_only": False,
        "mutating": True,
    },
    "stage-rollout": {
        "summary": "Run one weak, middle, and strong trajectory with bounded parallel tools.",
        "when": "Use when a constructed sandbox needs a stage-wise or E2E ability-separation rollout.",
        "when_not": "Do not use to retry a failed attempt, replace a candidate, or create admission evidence.",
        "runtime_symbols": (
            "eva_agent.pipeline.VerifiableDataPipeline",
            "eva_agent.harness.EvaMedHarness",
            "eva_agent.harness.ToolRegistry.execute_group",
        ),
        "input_contract": (
            "one constructed sandbox and compiled rubric binding",
            "exactly one distinct weak, middle, and strong target",
            "bounded model and tool parallelism",
        ),
        "output_contract": (
            "three policy-visible trajectories or infrastructure quarantine",
            "before/after workspace snapshots and tool-call groups",
            "per-item rewards and ability-separation statistics",
        ),
        "evidence_ref_contract": (
            "trajectory:event:<UUID>",
            "actor-tool:<UUID>",
            "workspace:before:<relative-path>",
            "workspace:after:<relative-path>",
        ),
        "parallel_safe": True,
        "read_only": False,
        "mutating": True,
    },
    "workspace-agent-judge": {
        "summary": "Judge committed context and workspace snapshots with read-only evidence tools.",
        "when": "Use when an Opus 5 judge must score one retained trajectory against its bound rubric.",
        "when_not": "Do not use against live mutable files or as a substitute for signed admission.",
        "runtime_symbols": (
            "eva_agent.pipeline.JudgeWorkspaceTools",
            "eva_agent.pipeline.PipelineVerifier",
            "eva_agent.pipeline.WeightedRubricRewarder",
        ),
        "input_contract": (
            "committed evidence bundle",
            "same compiled rubric binding used for reward",
            "judge-only reference confined to the judge boundary",
        ),
        "output_contract": (
            "atomic rubric item scores with observed evidence references",
            "read-only judge tool trace and BLAKE3 assessment",
            "typed infrastructure failure without semantic score",
        ),
        "evidence_ref_contract": (
            "context:policy-visible",
            "trajectory:event:<UUID>",
            "actor-tool:<UUID>",
            "judge-reference:<committed-id>",
            "workspace:before:<relative-path>",
            "workspace:after:<relative-path>",
        ),
        "parallel_safe": True,
        "read_only": True,
        "mutating": False,
    },
    "trajectory-sft": {
        "summary": "Slice one signed-admitted strong trajectory into atomic lineage-complete examples.",
        "when": "Use when pipeline and external supervisor evidence independently verify an admission.",
        "when_not": "Do not use for unsigned, rejected, quarantined, weak, or middle trajectories.",
        "runtime_symbols": (
            "eva_agent.pipeline.slice_admitted_trajectory",
            "eva_agent.pipeline.AdmissionEvidenceVerifier",
        ),
        "input_contract": (
            "verified eligible pipeline result and strong evaluation",
            "verified signed admission receipt and supervisor transition",
            "policy-visible trajectory events only",
        ),
        "output_contract": (
            "one SFT example per assistant decision",
            "atomic same-turn multi-tool targets",
            "BLAKE3 lineage to source rollout, evaluation, and result",
        ),
        "evidence_ref_contract": (
            "trajectory:event:<UUID>",
            "receipt:admission:<BLAKE3>",
            "receipt:supervisor-transition:<BLAKE3>",
        ),
        "parallel_safe": True,
        "read_only": False,
        "mutating": True,
    },
}


class BridgeError(ValueError):
    """A caller supplied a work order outside EvaMed's public contract."""


def _canonical_uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise BridgeError(f"{label} must be canonical UUID text")
    try:
        parsed = UUID(value)
    except ValueError:
        raise BridgeError(f"{label} must be canonical UUID text") from None
    if str(parsed) != value:
        raise BridgeError(f"{label} must be canonical UUID text")
    return value


def _repository_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise BridgeError(f"{label} must be a repository-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise BridgeError(f"{label} must be a repository-relative path")
    candidate = REPOSITORY_ROOT.joinpath(*pure.parts)
    cursor = REPOSITORY_ROOT
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BridgeError(f"{label} cannot traverse a symlink")
    resolved = candidate.resolve(strict=False)
    if REPOSITORY_ROOT not in (resolved, *resolved.parents):
        raise BridgeError(f"{label} escapes the repository")
    return candidate


def _discovery_arguments(arguments: Mapping[str, Any], *, search: bool) -> tuple[str, str]:
    expected = {"domain", "stage", "query", "max_results"} if search else {
        "domain",
        "stage",
    }
    if set(arguments) != expected:
        raise BridgeError("discovery fields differ from the closed contract")
    domain = arguments["domain"]
    stage = arguments["stage"]
    if domain not in DOMAINS or stage not in STAGES:
        raise BridgeError("domain or stage is outside the installed registry")
    return str(domain), str(stage)


def _summary(name: str, *, stage: str) -> dict[str, Any]:
    record = _CAPABILITY_RECORDS[name]
    return {
        "name": name,
        "summary": record["summary"],
        "stage": stage,
        "parallel_safe": record["parallel_safe"],
        "read_only": record["read_only"],
        "mutating": record["mutating"],
    }


def _capabilities(arguments: Mapping[str, Any]) -> dict[str, Any]:
    domain, stage = _discovery_arguments(arguments, search=False)
    rows = tuple(_summary(name, stage=stage) for name in OPERATIONS)
    core = {
        "schema": "eva.evamed-capabilities.v1",
        "domain": domain,
        "stage": stage,
        "capabilities": rows,
        "provider_execution_exposed": False,
    }
    return {**core, "catalog_blake3": blake3_hex(core)}


def _search(arguments: Mapping[str, Any]) -> dict[str, Any]:
    domain, stage = _discovery_arguments(arguments, search=True)
    query = arguments["query"]
    limit = arguments["max_results"]
    if not isinstance(query, str) or not 1 <= len(query) <= 128:
        raise BridgeError("query length must be in [1,128]")
    if type(limit) is not int or not 1 <= limit <= 4:
        raise BridgeError("max_results must be in [1,4]")
    needle = query.casefold()
    matches = []
    for name in OPERATIONS:
        record = _CAPABILITY_RECORDS[name]
        haystack = " ".join(
            (name, record["summary"], record["when"], record["when_not"])
        ).casefold()
        if needle in haystack:
            matches.append(_summary(name, stage=stage))
    rows = tuple(matches[:limit])
    core = {
        "schema": "eva.evamed-capability-search.v1",
        "domain": domain,
        "stage": stage,
        "query": query,
        "results": rows,
        "truncated": len(matches) > len(rows),
    }
    return {**core, "search_blake3": blake3_hex(core)}


def _load(arguments: Mapping[str, Any]) -> dict[str, Any]:
    if set(arguments) != {"name", "domain", "stage"}:
        raise BridgeError("load fields differ from the closed contract")
    name = arguments["name"]
    domain = arguments["domain"]
    stage = arguments["stage"]
    if name not in _CAPABILITY_RECORDS:
        raise BridgeError("capability is not installed")
    if domain not in DOMAINS or stage not in STAGES:
        raise BridgeError("domain or stage is outside the installed registry")
    record = _CAPABILITY_RECORDS[str(name)]
    core = {
        "schema": "eva.evamed-capability.v1",
        "name": name,
        "domain": domain,
        "stage": stage,
        "summary": record["summary"],
        "when": record["when"],
        "when_not": record["when_not"],
        "runtime_symbols": record["runtime_symbols"],
        "input_contract": record["input_contract"],
        "output_contract": record["output_contract"],
        "evidence_ref_contract": record["evidence_ref_contract"],
        "parallel_safe": record["parallel_safe"],
        "read_only": record["read_only"],
        "mutating": record["mutating"],
    }
    return {**core, "capability_blake3": blake3_hex(core)}


def _validate_work_order(arguments: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "operation",
        "candidate_id",
        "domain",
        "stage",
        "rubric_blake3",
        "cohort_rollouts",
        "parallel_tool_calls",
        "retry_count",
        "signed_receipts_required",
    }
    if set(arguments) != required:
        raise BridgeError("work order fields differ from the closed contract")
    operation = arguments["operation"]
    domain = arguments["domain"]
    stage = arguments["stage"]
    rollouts = arguments["cohort_rollouts"]
    if operation not in OPERATIONS:
        raise BridgeError("operation is not an EvaMed plugin skill")
    if domain not in DOMAINS:
        raise BridgeError("domain is outside the installed registry")
    if stage not in STAGES:
        raise BridgeError("stage must be S1-S5 or E2E")
    if not is_blake3(arguments["rubric_blake3"]):
        raise BridgeError("rubric_blake3 must be a BLAKE3 digest")
    if not isinstance(rollouts, Mapping) or dict(rollouts) != {
        "weak": 1,
        "middle": 1,
        "strong": 1,
    }:
        raise BridgeError("weak, middle, and strong must each run exactly once")
    if arguments["parallel_tool_calls"] is not True:
        raise BridgeError("parallel_tool_calls must remain enabled")
    if arguments["retry_count"] != 0 or type(arguments["retry_count"]) is not int:
        raise BridgeError("retry_count must be exactly zero")
    if arguments["signed_receipts_required"] is not True:
        raise BridgeError("signed receipts must remain required")
    core = {
        "operation": operation,
        "candidate_id": _canonical_uuid(arguments["candidate_id"], "candidate_id"),
        "domain": domain,
        "stage": stage,
        "rubric_blake3": arguments["rubric_blake3"],
        "cohort_rollouts": {"weak": 1, "middle": 1, "strong": 1},
        "parallel_tool_calls": True,
        "retry_count": 0,
        "signed_receipts_required": True,
    }
    return {
        "schema": "eva.evamed-work-order.v1",
        "valid": True,
        **core,
        "rubric_binding_use": "judge-and-reward",
        "work_order_blake3": blake3_hex(core),
        "provider_calls_made": 0,
    }


def _verify_orchestration_run(arguments: Mapping[str, Any]) -> dict[str, Any]:
    if set(arguments) != {"receipt_root", "run_id"}:
        raise BridgeError("orchestration verification fields differ")
    run_id = _canonical_uuid(arguments["run_id"], "run_id")
    root = _repository_path(arguments["receipt_root"], "receipt_root")
    valid = ReceiptJournal.verify_run(root, run_id)
    core = {
        "schema": "eva.evamed-orchestration-verification.v1",
        "receipt_root": root.relative_to(REPOSITORY_ROOT).as_posix(),
        "run_id": run_id,
        "valid": valid,
        "verifier": "eva_agent.orchestration.ReceiptJournal.verify_run",
        "evidence_refs": (f"receipt:orchestration:{run_id}",),
    }
    return {**core, "verification_blake3": blake3_hex(core)}


_UUID_SCHEMA = {
    "type": "string",
    "format": "uuid",
    "minLength": 36,
    "maxLength": 36,
}
_DIGEST_SCHEMA = {
    "type": "string",
    "pattern": "^[0-9a-f]{64}$",
    "minLength": 64,
    "maxLength": 64,
}
_DOMAIN_SCHEMA = {"type": "string", "enum": list(DOMAINS)}
_STAGE_SCHEMA = {"type": "string", "enum": list(STAGES)}
_OPERATION_SCHEMA = {"type": "string", "enum": list(OPERATIONS)}
_EVIDENCE_REF_SCHEMA = {
    "type": "string",
    "minLength": 1,
    "maxLength": 4096,
    "pattern": "^(context:|trajectory:|actor-tool:|judge-reference:|workspace:|receipt:)",
}


def _closed(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


_SUMMARY_SCHEMA = _closed(
    {
        "name": _OPERATION_SCHEMA,
        "summary": {"type": "string", "minLength": 24, "maxLength": 160},
        "stage": _STAGE_SCHEMA,
        "parallel_safe": {"type": "boolean"},
        "read_only": {"type": "boolean"},
        "mutating": {"type": "boolean"},
    }
)


def _tool_definition(
    *,
    name: str,
    description: str,
    input_schema: Mapping[str, Any],
    output_schema: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": dict(input_schema),
        "outputSchema": dict(output_schema),
        "annotations": {
            "title": name.replace("_", " ").title(),
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "_meta": {
            "evamed": {
                "parallelSafe": True,
                "readOnly": True,
                "mutating": False,
            }
        },
    }


_DISCOVERY_INPUT = _closed({"domain": _DOMAIN_SCHEMA, "stage": _STAGE_SCHEMA})
_TOOLS: dict[str, tuple[dict[str, Any], Callable[[Mapping[str, Any]], dict[str, Any]]]] = {
    "capabilities": (
        _tool_definition(
            name="capabilities",
            description=(
                "Use when starting an EvaMed task to list short stage-filtered capability summaries. "
                "Do not use to load full instructions or execute providers."
            ),
            input_schema=_DISCOVERY_INPUT,
            output_schema=_closed(
                {
                    "schema": {"type": "string", "const": "eva.evamed-capabilities.v1"},
                    "domain": _DOMAIN_SCHEMA,
                    "stage": _STAGE_SCHEMA,
                    "capabilities": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": _SUMMARY_SCHEMA,
                    },
                    "provider_execution_exposed": {"type": "boolean", "const": False},
                    "catalog_blake3": _DIGEST_SCHEMA,
                }
            ),
        ),
        _capabilities,
    ),
    "search": (
        _tool_definition(
            name="search",
            description=(
                "Use when capability summaries are insufficient and a task keyword must be matched within one domain-stage. "
                "Do not use to reveal rubric contents or run a workflow."
            ),
            input_schema=_closed(
                {
                    "domain": _DOMAIN_SCHEMA,
                    "stage": _STAGE_SCHEMA,
                    "query": {"type": "string", "minLength": 1, "maxLength": 128},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 4},
                }
            ),
            output_schema=_closed(
                {
                    "schema": {"type": "string", "const": "eva.evamed-capability-search.v1"},
                    "domain": _DOMAIN_SCHEMA,
                    "stage": _STAGE_SCHEMA,
                    "query": {"type": "string", "minLength": 1, "maxLength": 128},
                    "results": {
                        "type": "array",
                        "maxItems": 4,
                        "items": _SUMMARY_SCHEMA,
                    },
                    "truncated": {"type": "boolean"},
                    "search_blake3": _DIGEST_SCHEMA,
                }
            ),
        ),
        _search,
    ),
    "load": (
        _tool_definition(
            name="load",
            description=(
                "Use when one selected EvaMed capability needs its public runtime and evidence contracts. "
                "Do not use to load private rubric items, answers, or judge-only values."
            ),
            input_schema=_closed(
                {"name": _OPERATION_SCHEMA, "domain": _DOMAIN_SCHEMA, "stage": _STAGE_SCHEMA}
            ),
            output_schema=_closed(
                {
                    "schema": {"type": "string", "const": "eva.evamed-capability.v1"},
                    "name": _OPERATION_SCHEMA,
                    "domain": _DOMAIN_SCHEMA,
                    "stage": _STAGE_SCHEMA,
                    "summary": {"type": "string", "minLength": 24, "maxLength": 160},
                    "when": {"type": "string", "minLength": 24, "maxLength": 200},
                    "when_not": {"type": "string", "minLength": 24, "maxLength": 200},
                    "runtime_symbols": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 6,
                        "items": {"type": "string", "minLength": 3, "maxLength": 160},
                    },
                    "input_contract": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 3, "maxLength": 200},
                    },
                    "output_contract": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 3, "maxLength": 200},
                    },
                    "evidence_ref_contract": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": _EVIDENCE_REF_SCHEMA,
                    },
                    "parallel_safe": {"type": "boolean"},
                    "read_only": {"type": "boolean"},
                    "mutating": {"type": "boolean"},
                    "capability_blake3": _DIGEST_SCHEMA,
                }
            ),
        ),
        _load,
    ),
    "validate_work_order": (
        _tool_definition(
            name="validate_work_order",
            description=(
                "Use when a public EvaMed dispatch order must be checked and BLAKE3-committed before execution. "
                "Do not use to execute, retry, score, or admit the candidate."
            ),
            input_schema=_closed(
                {
                    "operation": _OPERATION_SCHEMA,
                    "candidate_id": _UUID_SCHEMA,
                    "domain": _DOMAIN_SCHEMA,
                    "stage": _STAGE_SCHEMA,
                    "rubric_blake3": _DIGEST_SCHEMA,
                    "cohort_rollouts": _closed(
                        {
                            "weak": {"type": "integer", "const": 1},
                            "middle": {"type": "integer", "const": 1},
                            "strong": {"type": "integer", "const": 1},
                        }
                    ),
                    "parallel_tool_calls": {"type": "boolean", "const": True},
                    "retry_count": {"type": "integer", "const": 0},
                    "signed_receipts_required": {"type": "boolean", "const": True},
                }
            ),
            output_schema=_closed(
                {
                    "schema": {"type": "string", "const": "eva.evamed-work-order.v1"},
                    "valid": {"type": "boolean", "const": True},
                    "operation": _OPERATION_SCHEMA,
                    "candidate_id": _UUID_SCHEMA,
                    "domain": _DOMAIN_SCHEMA,
                    "stage": _STAGE_SCHEMA,
                    "rubric_blake3": _DIGEST_SCHEMA,
                    "cohort_rollouts": _closed(
                        {
                            "weak": {"type": "integer", "const": 1},
                            "middle": {"type": "integer", "const": 1},
                            "strong": {"type": "integer", "const": 1},
                        }
                    ),
                    "parallel_tool_calls": {"type": "boolean", "const": True},
                    "retry_count": {"type": "integer", "const": 0},
                    "signed_receipts_required": {"type": "boolean", "const": True},
                    "rubric_binding_use": {"type": "string", "const": "judge-and-reward"},
                    "work_order_blake3": _DIGEST_SCHEMA,
                    "provider_calls_made": {"type": "integer", "const": 0},
                }
            ),
        ),
        _validate_work_order,
    ),
    "verify_receipts": (
        _tool_definition(
            name="verify_receipts",
            description=(
                "Use when a sealed orchestration run below this repository must be independently reopened. "
                "Do not use to verify model semantics, mutate receipts, or grant admission."
            ),
            input_schema=_closed(
                {
                    "receipt_root": {"type": "string", "minLength": 1, "maxLength": 4096},
                    "run_id": _UUID_SCHEMA,
                }
            ),
            output_schema=_closed(
                {
                    "schema": {
                        "type": "string",
                        "const": "eva.evamed-orchestration-verification.v1",
                    },
                    "receipt_root": {"type": "string", "minLength": 1, "maxLength": 4096},
                    "run_id": _UUID_SCHEMA,
                    "valid": {"type": "boolean"},
                    "verifier": {"type": "string", "minLength": 3, "maxLength": 160},
                    "evidence_refs": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": _EVIDENCE_REF_SCHEMA,
                    },
                    "verification_blake3": _DIGEST_SCHEMA,
                }
            ),
        ),
        _verify_orchestration_run,
    ),
}


def _tool_result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    payload = canonical_value(payload)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    structured = payload if isinstance(payload, Mapping) else {"value": payload}
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
        "isError": is_error,
    }


class InvocationGate:
    """Bound total calls while giving unsafe tools an exclusive frontier."""

    def __init__(self, maximum: int) -> None:
        if not 1 <= maximum <= 256:
            raise BridgeError("EVAMED_MCP_MAX_PARALLEL must be in [1,256]")
        self._maximum = maximum
        self._condition = asyncio.Condition()
        self._active = 0
        self._unsafe_active = False
        self._unsafe_waiting = 0

    @asynccontextmanager
    async def enter(self, *, parallel_safe: bool) -> AsyncIterator[None]:
        async with self._condition:
            if parallel_safe:
                await self._condition.wait_for(
                    lambda: not self._unsafe_active
                    and self._unsafe_waiting == 0
                    and self._active < self._maximum
                )
                self._active += 1
            else:
                self._unsafe_waiting += 1
                try:
                    await self._condition.wait_for(
                        lambda: not self._unsafe_active and self._active == 0
                    )
                    self._unsafe_active = True
                    self._active = 1
                finally:
                    self._unsafe_waiting -= 1
        try:
            yield
        finally:
            async with self._condition:
                self._active -= 1
                if not parallel_safe:
                    self._unsafe_active = False
                self._condition.notify_all()


def _error_payload(exc: Exception, *, tool_name: str) -> dict[str, Any]:
    core = {
        "ok": False,
        "tool_name": tool_name,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "retry_performed": False,
    }
    return {**core, "failure_blake3": blake3_hex(core)}


async def _invoke_control(
    handler: Callable[[Mapping[str, Any]], dict[str, Any]],
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    if inspect.iscoroutinefunction(handler):
        return await handler(arguments)
    return await asyncio.to_thread(handler, arguments)


async def _dispatch(
    message: Mapping[str, Any],
    *,
    data_tools: Mapping[str, AdaptedTool],
    gate: InvocationGate,
    actor_mode: bool,
) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return None
    if method == "initialize":
        params = message.get("params")
        requested = params.get("protocolVersion") if isinstance(params, Mapping) else None
        result = {
            "protocolVersion": requested if isinstance(requested, str) else PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "Use native workflow skills for process guidance. Data-plane tools are exact "
                "schema-preserving projections from the verified EvaMed policy/ToolRegistry; "
                "unavailable handlers fail closed and calls are never retried."
            ),
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {
            "tools": ([] if actor_mode else [definition for definition, _ in _TOOLS.values()])
            + [tool.mcp_definition() for tool in data_tools.values()]
        }
    elif method == "tools/call":
        params = message.get("params")
        if not isinstance(params, Mapping):
            raise BridgeError("tools/call params must be an object")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            raise BridgeError("tool name or arguments differ")
        if name in _TOOLS and not actor_mode:
            definition, handler = _TOOLS[name]
            parallel_safe = bool(
                definition.get("_meta", {}).get("evamed", {}).get("parallelSafe")
            )
            try:
                async with gate.enter(parallel_safe=parallel_safe):
                    payload = await _invoke_control(handler, arguments)
                result = _tool_result(payload)
            except (BridgeError, AdapterError, ValueError) as exc:
                result = _tool_result(_error_payload(exc, tool_name=name), is_error=True)
        elif name in data_tools:
            tool = data_tools[name]
            try:
                async with gate.enter(parallel_safe=tool.parallel_safe):
                    payload = await tool.invoke(arguments)
                result = _tool_result(payload)
            except Exception as exc:
                result = _tool_result(_error_payload(exc, tool_name=name), is_error=True)
        else:
            exc = BridgeError("tool is not offered by the selected verified policy")
            result = _tool_result(_error_payload(exc, tool_name=name), is_error=True)
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "method not found"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


async def _serve_async(data_tools: Sequence[AdaptedTool]) -> int:
    by_name = {tool.name: tool for tool in data_tools}
    actor_mode = os.environ.get("EVAMED_MCP_ACTOR_MODE") == "1"
    active_control_names = set() if actor_mode else set(_TOOLS)
    if len(by_name) != len(data_tools) or set(by_name) & active_control_names:
        raise BridgeError("control/data tool names collide")
    maximum = int(os.environ.get("EVAMED_MCP_MAX_PARALLEL", "16"))
    gate = InvocationGate(maximum)
    write_lock = asyncio.Lock()
    tasks: set[asyncio.Task[None]] = set()

    async def write(response: Mapping[str, Any]) -> None:
        line = json.dumps(response, sort_keys=True, separators=(",", ":"))
        async with write_lock:
            await asyncio.to_thread(sys.stdout.write, line + "\n")
            await asyncio.to_thread(sys.stdout.flush)

    async def handle(line: str) -> None:
        message: Any = None
        try:
            message = json.loads(line)
            if not isinstance(message, Mapping) or message.get("jsonrpc") != "2.0":
                raise BridgeError("JSON-RPC request differs")
            response = await _dispatch(
                message,
                data_tools=by_name,
                gate=gate,
                actor_mode=actor_mode,
            )
        except json.JSONDecodeError:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }
        except (BridgeError, AdapterError) as exc:
            response = {
                "jsonrpc": "2.0",
                "id": message.get("id") if isinstance(message, Mapping) else None,
                "error": {"code": -32602, "message": str(exc)},
            }
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": message.get("id") if isinstance(message, Mapping) else None,
                "error": {"code": -32603, "message": type(exc).__name__},
            }
        if response is not None:
            await write(response)

    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            break
        task = asyncio.create_task(handle(line))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    if tasks:
        await asyncio.gather(*tasks)
    return 0


def _serve(data_tools: Sequence[AdaptedTool]) -> int:
    return asyncio.run(_serve_async(data_tools))


def _self_test(data_tools: Sequence[AdaptedTool]) -> int:
    fixture = {
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
    validated = _validate_work_order(fixture)
    names = tuple(tool.name for tool in data_tools)
    if "search_skills" not in names or "load_skill" not in names:
        raise BridgeError("canonical SkillCatalog tools are absent")
    core = {
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "control_tools": tuple(_TOOLS),
        "data_tools": names,
        "legacy_tool_names": LEGACY_TOOL_NAMES,
        "data_plane_blake3": data_plane_digest(data_tools),
        "work_order_blake3": validated["work_order_blake3"],
        "provider_calls_made": 0,
    }
    print(
        json.dumps(
            {
                "schema": "eva.evamed-codex-mcp-smoke.v1",
                "status": "passed",
                **core,
                "smoke_blake3": blake3_hex(core),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EvaMed local MCP contract bridge")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stdio", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if os.environ.get("EVAMED_MCP_ACTOR_MODE") == "1" and not (
        os.environ.get("EVAMED_MCP_POLICY_PATH")
        and os.environ.get("EVAMED_MCP_POLICY_BLAKE3")
    ):
        raise BridgeError("actor mode requires one BLAKE3-bound candidate policy")
    data_tools = build_data_plane(PLUGIN_ROOT)
    return _serve(data_tools) if args.stdio else _self_test(data_tools)


if __name__ == "__main__":
    raise SystemExit(main())
