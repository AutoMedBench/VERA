"""Execution-verified S2 completion frontiers from Codex teacher turns.

This lane is intentionally separate from the immutable S1 frontier-prefix
schemas.  A row contains only the actor-public
``materialize_evidence_selection`` decision and its exact host observation.
The preceding S1 plan and S2 retrieval frontiers are reopened as prerequisites
but their observations (which may carry private reference bytes) are excluded.
Terminal prose, private reasoning, and S3+ work are never exported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import base64
import binascii
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from uuid import UUID

from eva_agent.codex_pipeline.native_policy_v2 import (
    NativePolicyV2Error,
    StageToolGuidanceV1,
    build_stage_tool_guidance_v1,
    verify_stage_tool_guidance_v1,
)
from eva_agent.codex_providers import CodexProviderRoute
from eva_agent.codex_runtime import (
    CodexTurnReceipt,
    codex_turn_receipt_from_document,
    verify_codex_turn_receipt,
)
from eva_agent.pipeline import RandomUUIDFactory
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
    is_blake3,
)
from eva_agent.rubrics import CompiledRubricRegistry
from eva_agent.training.frontier_prefix_sft import (
    FrontierPrefixSFTError,
    _adapter_receipt,
    _canonical_json,
    _load_bulk_records,
    _read_json,
    _relative,
    _route_authority,
    _route_authority_document,
    _safe_relative,
    _signed_adapter_receipt,
    _strict_object,
    _uuid,
    _workspace_tree,
    _write_once,
)
from eva_agent.training.persistent_teacher import _ADAPTED_PROVIDER_FAMILIES


S2_FRONTIER_DATASET_SCHEMA = (
    "eva.execution-verified-s2-frontier-prefix-sft-dataset.v1"
)
S2_FRONTIER_SLICE_SCHEMA = "eva.execution-verified-s2-frontier-prefix-sft-slice.v1"
S2_FRONTIER_VERIFY_SCHEMA = (
    "eva.execution-verified-s2-frontier-prefix-sft-verification.v1"
)
S2_FRONTIER_AUDIT_SCHEMA = "eva.execution-verified-s2-frontier-prefix-audit.v1"


class S2FrontierPrefixSFTError(ValueError):
    """An S2 source, frontier, artifact, or exported commitment differs."""


@dataclass(frozen=True)
class S2FrontierPrefixSFTBuild:
    dataset_id: str
    dataset_root: Path
    source_count: int
    slice_count: int
    manifest_blake3: str


@dataclass(frozen=True)
class _ObservedFrontier:
    call: Any
    observation: Mapping[str, Any]
    arguments: Mapping[str, Any]
    tool_result: Mapping[str, Any]
    output: Mapping[str, Any]


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value).rstrip(b"\n")).hexdigest()


def _authoritative_legacy_tool_catalog(
    source_tools: Any, *, expected_blake3: Any
) -> tuple[Mapping[str, Any], ...]:
    """Reopen the resolver catalog from the intentionally minimal policy rows.

    Legacy source-policy files contain only the immutable public schema fields.
    The authoritative legacy resolver deterministically supplies the three
    execution annotations below for every one of those tools.  Accept no other
    persisted shape, and require the reconstructed catalog to match the digest
    committed by the runtime context before it is used to rebuild guidance.
    """

    if not isinstance(source_tools, list) or not source_tools:
        raise S2FrontierPrefixSFTError("public S2 source tool catalog differs")
    reconstructed: list[Mapping[str, Any]] = []
    for value in source_tools:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"name", "description", "input_schema"}
            or not isinstance(value.get("name"), str)
            or not value.get("name")
            or not isinstance(value.get("description"), str)
            or not value.get("description")
            or not isinstance(value.get("input_schema"), Mapping)
        ):
            raise S2FrontierPrefixSFTError(
                "public S2 legacy source tool catalog differs"
            )
        reconstructed.append(
            canonical_value(
                {
                    **value,
                    "visibility": "actor_public",
                    "handler_origin": "signed_legacy_evamed",
                    "parallel_safe": False,
                }
            )
        )
    catalog = tuple(sorted(reconstructed, key=lambda row: str(row["name"])))
    names = tuple(str(row["name"]) for row in catalog)
    if (
        len(names) != len(set(names))
        or not is_blake3(expected_blake3)
        or blake3_hex(catalog) != expected_blake3
    ):
        raise S2FrontierPrefixSFTError(
            "public S2 reconstructed tool catalog commitment differs"
        )
    return catalog


def _first_passing_s2_adapter_receipt(
    source_root: Path, *, route_id: str, model_id: str
) -> str:
    """Select route-batch evidence without treating signed failed calls as passes.

    A scale batch legitimately retains both passing and terminal adapter
    receipts in the same route directory.  Reopen every signature and identity,
    but only passing first-frontier receipts are eligible evidence for a slice.
    Failed receipts remain preserved and are never counted as positive proof.
    """

    receipt_root = source_root / "adapter-receipts" / route_id
    if receipt_root.is_symlink() or not receipt_root.is_dir():
        raise S2FrontierPrefixSFTError("adapter receipt root differs")
    passing = []
    for path in sorted(receipt_root.glob("*.json")):
        document, payload = _canonical_json(path)
        try:
            receipt = _signed_adapter_receipt(document)
        except FrontierPrefixSFTError as exc:
            raise S2FrontierPrefixSFTError(str(exc)) from exc
        if (
            path.stem != receipt.envelope_blake3
            or blake3_bytes(payload) != blake3_bytes(canonical_json_bytes(document))
            or receipt.payload.get("route_id") != route_id
            or receipt.payload.get("model_id") != model_id
        ):
            raise S2FrontierPrefixSFTError("adapter receipt route or model differs")
        if receipt.payload.get("status") == "passed":
            passing.append(receipt)
    first = [
        receipt
        for receipt in passing
        if receipt.payload["request_shape"].get("function_call_output_count") == 0
        and receipt.payload["response_shape"].get("parallel_function_call_count")
        == 1
        and "function_call"
        in receipt.payload["response_shape"].get("output_item_types", [])
    ]
    if not first:
        raise S2FrontierPrefixSFTError("first passing S2 adapter receipt is absent")
    selected = min(
        first,
        key=lambda receipt: (
            str(receipt.payload.get("created_at_utc")), receipt.envelope_blake3
        ),
    )
    return selected.envelope_blake3


def _source_entries(
    sources: Sequence[Path], *, base: Path
) -> tuple[tuple[Path, Path], ...]:
    entries = tuple(
        sorted(
            (
                (source, path)
                for source in sources
                for path in (
                    *((source / "errors").glob("*.provider-failure.json")),
                    *((source / "trajectories").glob("*/*/result.json")),
                )
            ),
            key=lambda item: item[1].as_posix(),
        )
    )
    for source, path in entries:
        resolved = path.resolve(strict=True)
        if source not in resolved.parents:
            raise S2FrontierPrefixSFTError("teacher source lies outside its root")
        _relative(resolved, base=base, label="teacher source")
    return entries


def _mcp_observation(call: Any, *, expected_name: str) -> _ObservedFrontier:
    if (
        call.fully_qualified_name != f"evamed/{expected_name}"
        or call.status != "completed"
    ):
        raise S2FrontierPrefixSFTError("S2 guided Codex tool call differs")
    envelope = canonical_value(call.output)
    result_envelope = envelope.get("result") if isinstance(envelope, Mapping) else None
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != {"durationMs", "error", "result"}
        or type(envelope.get("durationMs")) is not int
        or envelope["durationMs"] < 0
        or envelope.get("error") is not None
        or not isinstance(result_envelope, Mapping)
        or set(result_envelope) != {"_meta", "content", "structuredContent"}
        or result_envelope.get("_meta") is not None
        or result_envelope.get("content") != []
    ):
        raise S2FrontierPrefixSFTError("S2 Codex MCP result envelope differs")
    observation = result_envelope.get("structuredContent")
    if (
        not isinstance(observation, Mapping)
        or set(observation)
        != {
            "schema",
            "call_id",
            "name",
            "arguments",
            "tool_result",
            "bridge_receipt_blake3",
        }
        or observation.get("schema")
        != "eva.codex-pipeline-tool-observation.v1"
    ):
        raise S2FrontierPrefixSFTError("S2 host tool observation shape differs")
    observation_core = {
        key: value
        for key, value in observation.items()
        if key != "bridge_receipt_blake3"
    }
    result = observation.get("tool_result")
    arguments = observation.get("arguments")
    if (
        observation.get("bridge_receipt_blake3") != blake3_hex(observation_core)
        or observation.get("name") != expected_name
        or not isinstance(arguments, Mapping)
        or canonical_value(arguments) != canonical_value(call.arguments)
        or not isinstance(result, Mapping)
        or set(result)
        != {
            "result_id",
            "call_id",
            "name",
            "frontier",
            "parallel_group_id",
            "status",
            "output",
            "error_code",
            "workspace_before_blake3",
            "workspace_after_blake3",
            "receipt_blake3",
        }
    ):
        raise S2FrontierPrefixSFTError("S2 host tool observation value differs")
    result_core = {key: value for key, value in result.items() if key != "receipt_blake3"}
    output = result.get("output")
    if (
        result.get("receipt_blake3") != blake3_hex(result_core)
        or result.get("call_id") != observation.get("call_id")
        or result.get("name") != expected_name
        or result.get("status") != "completed"
        or result.get("error_code") is not None
        or not isinstance(result.get("parallel_group_id"), str)
        or not result.get("parallel_group_id")
        or not is_blake3(result.get("workspace_before_blake3"))
        or not is_blake3(result.get("workspace_after_blake3"))
        or not isinstance(output, Mapping)
    ):
        raise S2FrontierPrefixSFTError("S2 host ToolResult differs")
    return _ObservedFrontier(
        call=call,
        observation=observation,
        arguments=arguments,
        tool_result=result,
        output=output,
    )


def _require_passed_gate(
    frontier: _ObservedFrontier,
    *,
    stage: str,
    effect: str,
    frontier_index: int,
) -> None:
    output = frontier.output
    if (
        frontier.tool_result.get("frontier") != frontier_index
        or output.get("stage") != stage
        or output.get("effect") != effect
        or output.get("gate_passed") is not True
        or output.get("failed_check_ids") != []
        or output.get("error") not in {None, ""}
    ):
        raise S2FrontierPrefixSFTError("S2 guided host gate did not pass exactly")


def _required_evidence_ids(contract: Mapping[str, Any]) -> tuple[str, ...]:
    required = contract.get("required_evidence_ids")
    objects = contract.get("evidence_objects")
    if (
        not isinstance(required, list)
        or not required
        or len(required) != len(set(required))
        or any(not isinstance(value, str) or not value for value in required)
        or not isinstance(objects, list)
    ):
        raise S2FrontierPrefixSFTError("S2 required evidence contract differs")
    object_ids = [
        item.get("evidence_id") for item in objects if isinstance(item, Mapping)
    ]
    if len(object_ids) != len(objects) or set(object_ids) != set(required):
        raise S2FrontierPrefixSFTError("S2 evidence object inventory differs")
    return tuple(required)


def _workspace_snapshot_files(
    value: Any, *, expected_label: str
) -> tuple[str, Mapping[str, bytes]]:
    """Reopen a byte-bearing teacher workspace snapshot exactly."""

    if (
        not isinstance(value, Mapping)
        or set(value) != {"label", "files", "file_count", "byte_count", "tree_blake3"}
        or value.get("label") != expected_label
        or not isinstance(value.get("files"), list)
        or type(value.get("file_count")) is not int
        or type(value.get("byte_count")) is not int
        or not is_blake3(value.get("tree_blake3"))
    ):
        raise S2FrontierPrefixSFTError("S2 workspace snapshot shape differs")
    files = value["files"]
    payloads: dict[str, bytes] = {}
    paths: list[str] = []
    for row in files:
        content = row.get("content") if isinstance(row, Mapping) else None
        encoded = content.get("$bytes_base64") if isinstance(content, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "content", "byte_count", "mode", "content_blake3"}
            or not isinstance(encoded, str)
            or set(content) != {"$bytes_base64"}
            or not isinstance(row.get("mode"), str)
            or len(str(row.get("mode"))) != 4
        ):
            raise S2FrontierPrefixSFTError("S2 workspace snapshot file differs")
        relative = _safe_relative(row.get("path"), label="snapshot workspace")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise S2FrontierPrefixSFTError(
                "S2 workspace snapshot file encoding differs"
            ) from exc
        if (
            relative in payloads
            or row.get("byte_count") != len(payload)
            or row.get("content_blake3") != blake3_bytes(payload)
        ):
            raise S2FrontierPrefixSFTError("S2 workspace snapshot file differs")
        paths.append(relative)
        payloads[relative] = payload
    core = {
        "files": files,
        "file_count": len(files),
        "byte_count": sum(len(payload) for payload in payloads.values()),
    }
    if (
        paths != sorted(paths)
        or value.get("file_count") != len(files)
        or value.get("byte_count") != core["byte_count"]
        or value.get("tree_blake3") != blake3_hex(core)
    ):
        raise S2FrontierPrefixSFTError("S2 workspace snapshot commitment differs")
    return str(value["tree_blake3"]), payloads


def _require_s2_control_frontiers(
    receipt: CodexTurnReceipt,
    *,
    guidance: StageToolGuidanceV1,
    s2_contract: Mapping[str, Any],
    skill_aware: bool = False,
    skill_delivery: Mapping[str, Any] | None = None,
) -> tuple[tuple[_ObservedFrontier, ...], _ObservedFrontier]:
    """Reopen the actor-owned retrievals -> S2 selection sequence.

    The authoritative guidance retains S1 at absolute frontier zero, but an
    S2 teacher workspace is hydrated by the host before Codex starts.  The
    actor receipt must therefore begin at guidance frontier one while its
    fresh host ToolTrace numbers the first retrieval as local frontier zero.
    """

    frontiers = tuple(guidance.frontiers)
    if guidance.focus.value != "S2" or guidance.core_document().get(
        "first_frontier_index"
    ) != 1:
        raise S2FrontierPrefixSFTError("authoritative S2 actor boundary differs")
    actor_frontiers = frontiers[1:]
    calls = tuple(receipt.tool_calls)
    if len(calls) < len(actor_frontiers):
        raise S2FrontierPrefixSFTError("S2 guided frontier sequence is incomplete")
    expected_names = tuple(str(row.get("tool_name")) for row in actor_frontiers)
    if not skill_aware and tuple(call.fully_qualified_name for call in calls[: len(actor_frontiers)]) != tuple(
        f"evamed/{name}" for name in expected_names
    ):
        raise S2FrontierPrefixSFTError("S2 guided frontier sequence differs")
    required_ids = _required_evidence_ids(s2_contract)
    if (
        frontiers[0].get("tool_name") != "materialize_plan"
        or expected_names[-1] != "materialize_evidence_selection"
        or expected_names[:-1]
        != tuple("retrieve_frozen_evidence" for _ in required_ids)
    ):
        raise S2FrontierPrefixSFTError("authoritative S2 guidance sequence differs")
    if skill_aware:
        from .s2_skill_frontier_support import require_skill_prefix
        all_observed, observed = require_skill_prefix(receipt, expected_names, skill_delivery)
    else:
        observed = tuple(
            _mcp_observation(call, expected_name=name)
            for call, name in zip(
                calls[: len(actor_frontiers)], expected_names, strict=True
            )
        )
        all_observed = observed
    retrieval_receipts: dict[str, str] = {}
    for offset, (evidence_id, item) in enumerate(
        zip(required_ids, observed[:-1], strict=True)
    ):
        _require_passed_gate(
            item,
            stage="S2",
            effect="frozen_evidence_retrieval",
            frontier_index=item.tool_result["frontier"] if skill_aware else offset,
        )
        receipt_sha256 = item.output.get("retrieval_receipt_sha256")
        if (
            canonical_value(item.arguments)
            != canonical_value({"evidence_id": evidence_id})
            or item.output.get("evidence_id") != evidence_id
            or not isinstance(receipt_sha256, str)
            or len(receipt_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in receipt_sha256)
            or not isinstance(item.output.get("verified_evidence_content_sha256"), str)
            or len(str(item.output.get("verified_evidence_content_sha256"))) != 64
        ):
            raise S2FrontierPrefixSFTError("S2 retrieval binding differs")
        retrieval_receipts[evidence_id] = receipt_sha256
    selection = observed[-1]
    _require_passed_gate(
        selection,
        stage="S2",
        effect="evidence_selection_materialization",
        frontier_index=selection.tool_result["frontier"] if skill_aware else len(actor_frontiers) - 1,
    )
    arguments = selection.arguments
    selected = arguments.get("selected_evidence")
    if (
        selection.output.get("next_stage") != "S3"
        or not isinstance(selection.output.get("receipt_sha256"), str)
        or len(str(selection.output.get("receipt_sha256"))) != 64
        or arguments.get("schema") != "rlevo.med-research-evidence-selection.v1"
        or arguments.get("contract_sha256") != _sha256_value(s2_contract)
        or arguments.get("sandbox_id") != s2_contract.get("sandbox_id")
        or arguments.get("episode_id") != s2_contract.get("episode_id")
        or arguments.get("question") != s2_contract.get("question")
        or arguments.get("evidence_need") != s2_contract.get("evidence_need")
        or arguments.get("care_directive") is not False
        or not isinstance(selected, list)
        or [item.get("evidence_id") for item in selected if isinstance(item, Mapping)]
        != list(required_ids)
        or any(
            not isinstance(item, Mapping)
            or item.get("retrieval_receipt_sha256")
            != retrieval_receipts.get(str(item.get("evidence_id")))
            for item in selected
        )
    ):
        raise S2FrontierPrefixSFTError("S2 evidence selection binding differs")
    return all_observed, selection


def _require_hydrated_s1(
    source: Mapping[str, Any],
    *,
    guidance: StageToolGuidanceV1,
    s1_contract: Mapping[str, Any],
    s2_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Verify the exact host-owned S1 observation exposed to the S2 actor."""

    messages = source.get("messages")
    user = messages[1] if isinstance(messages, list) and len(messages) >= 2 else None
    content = user.get("content") if isinstance(user, Mapping) else None
    hydration = (
        content.get("teacher_prerequisite_hydration")
        if isinstance(content, Mapping)
        else None
    )
    required = {
        "schema",
        "focus",
        "source_candidate_id",
        "guidance_blake3",
        "hydrated_frontier_indices",
        "next_actor_frontier_index",
        "artifact_relative_paths",
        "tool_result",
        "workspace_before_blake3",
        "workspace_after_blake3",
        "provider_calls",
        "canonical_tool_schemas_changed",
        "mcp_wire_schema_changed",
    }
    result = hydration.get("tool_result") if isinstance(hydration, Mapping) else None
    if (
        not isinstance(hydration, Mapping)
        or set(hydration) != required
        or hydration.get("schema")
        != "eva.codex-teacher-prerequisite-hydration.v1"
        or hydration.get("focus") != "S2"
        or hydration.get("source_candidate_id") != guidance.source_candidate_id
        or hydration.get("guidance_blake3") != guidance.guidance_blake3
        or hydration.get("hydrated_frontier_indices") != [0]
        or hydration.get("next_actor_frontier_index") != 1
        or hydration.get("provider_calls") != 0
        or hydration.get("canonical_tool_schemas_changed") is not False
        or hydration.get("mcp_wire_schema_changed") is not False
        or not isinstance(result, Mapping)
        or set(result)
        != {
            "result_id",
            "call_id",
            "name",
            "frontier",
            "parallel_group_id",
            "status",
            "output",
            "error_code",
            "workspace_before_blake3",
            "workspace_after_blake3",
            "receipt_blake3",
        }
    ):
        raise S2FrontierPrefixSFTError("S2 prerequisite hydration shape differs")
    result_core = {key: value for key, value in result.items() if key != "receipt_blake3"}
    output = result.get("output")
    first = guidance.frontiers[0]
    artifacts = s1_contract.get("stage_artifacts")
    plan_path = artifacts.get("S1") if isinstance(artifacts, Mapping) else None
    receipt_sha256 = output.get("receipt_sha256") if isinstance(output, Mapping) else None
    effective_s2_contract = canonical_value(s2_contract)
    if not isinstance(effective_s2_contract, dict):
        raise S2FrontierPrefixSFTError("S2 prerequisite contract differs")
    effective_s2_contract["s1_receipt_sha256"] = receipt_sha256
    if (
        result.get("receipt_blake3") != blake3_hex(result_core)
        or result.get("name") != "materialize_plan"
        or result.get("frontier") != 0
        or result.get("status") != "completed"
        or result.get("error_code") is not None
        or not isinstance(result.get("parallel_group_id"), str)
        or not result.get("parallel_group_id")
        or not is_blake3(result.get("workspace_before_blake3"))
        or not is_blake3(result.get("workspace_after_blake3"))
        or result.get("workspace_before_blake3")
        == result.get("workspace_after_blake3")
        or hydration.get("workspace_before_blake3")
        != result.get("workspace_before_blake3")
        or hydration.get("workspace_after_blake3")
        != result.get("workspace_after_blake3")
        or hydration.get("artifact_relative_paths") != [plan_path]
        or first.get("frontier_index") != 0
        or first.get("stage") != "S1"
        or first.get("tool_name") != "materialize_plan"
        or not isinstance(first.get("arguments"), Mapping)
        or not isinstance(output, Mapping)
        or output.get("stage") != "S1"
        or output.get("effect") != "plan_materialization"
        or output.get("gate_passed") is not True
        or output.get("failed_check_ids") != []
        or output.get("error") not in {None, ""}
        or output.get("next_stage") != "S2"
        or not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in receipt_sha256)
        or output.get("evidence_contract_sha256")
        != _sha256_value(effective_s2_contract)
        or first["arguments"].get("schema")
        != "rlevo.med-research-stage-plan-artifact.v1"
        or first["arguments"].get("contract_sha256")
        != _sha256_value(s1_contract)
        or first["arguments"].get("sandbox_id") != s1_contract.get("sandbox_id")
        or first["arguments"].get("episode_id") != s1_contract.get("episode_id")
    ):
        raise S2FrontierPrefixSFTError("S2 prerequisite S1 binding differs")
    return hydration


def _effective_s2_contract(
    hydration: Mapping[str, Any], *, template: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Bind the public S2 template to the exact reopened S1 host receipt."""

    tool_result = hydration.get("tool_result")
    output = tool_result.get("output") if isinstance(tool_result, Mapping) else None
    receipt_sha256 = output.get("receipt_sha256") if isinstance(output, Mapping) else None
    effective = canonical_value(template)
    if not isinstance(effective, dict) or not isinstance(receipt_sha256, str):
        raise S2FrontierPrefixSFTError("hydrated S2 contract differs")
    effective["s1_receipt_sha256"] = receipt_sha256
    if output.get("evidence_contract_sha256") != _sha256_value(effective):
        raise S2FrontierPrefixSFTError("hydrated S2 contract commitment differs")
    return effective


def _verify_completed_projection(
    source: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    receipt: CodexTurnReceipt,
    accepted: Sequence[_ObservedFrontier],
) -> str:
    claimed = source.get("provider_receipt_blake3")
    messages = source.get("messages")
    trace = source.get("tool_trace")
    if (
        not is_blake3(claimed)
        or source.get("model_id") != receipt.model
        or source.get("provider") != receipt.provider
        or source.get("assistant_output") != receipt.final_response
        or metadata.get("schema") != "eva.codex-provider-rollout-projection.v1"
        or not isinstance(messages, list)
        or len(messages) < 3
        or not isinstance(trace, Mapping)
    ):
        raise S2FrontierPrefixSFTError("completed teacher provider projection differs")
    for event in messages:
        if not isinstance(event, Mapping):
            raise S2FrontierPrefixSFTError("completed teacher policy event differs")
        core = {
            "event_id": event.get("event_id"),
            "role": event.get("role"),
            "content": event.get("content"),
            "tool_call_ids": event.get("tool_call_ids"),
        }
        if event.get("event_blake3") != blake3_hex(core):
            raise S2FrontierPrefixSFTError("completed teacher policy event differs")
    terminal = messages[-1]
    if (
        terminal.get("role") != "assistant"
        or terminal.get("tool_call_ids") != []
        or canonical_value(terminal.get("content"))
        != canonical_value(
            {
                "response": receipt.final_response,
                "codex_turn_receipt_blake3": receipt.receipt_blake3,
            }
        )
    ):
        raise S2FrontierPrefixSFTError("completed teacher terminal projection differs")
    trace_core = {key: value for key, value in trace.items() if key != "trace_blake3"}
    bindings = metadata.get("codex_to_pipeline_call_ids")
    results = trace.get("results")
    if (
        trace.get("trace_blake3") != blake3_hex(trace_core)
        or not isinstance(bindings, Mapping)
        or not isinstance(results, list)
    ):
        raise S2FrontierPrefixSFTError("completed teacher tool trace differs")
    for frontier in accepted:
        bound = bindings.get(frontier.call.tool_call_id)
        matches = [
            item
            for item in results
            if isinstance(item, Mapping) and item.get("call_id") == bound
        ]
        if (
            bound != frontier.tool_result.get("call_id")
            or len(matches) != 1
            or canonical_value(matches[0]) != canonical_value(frontier.tool_result)
        ):
            raise S2FrontierPrefixSFTError("completed teacher S2 frontier binding differs")
    return str(claimed)


def _source_slice(
    *,
    source_root: Path,
    source_base: Path,
    bulk_root: Path,
    registry: CompiledRubricRegistry,
    source_document_path: Path,
    bulk_record: tuple[Mapping[str, Any], str],
    route_authorities: Mapping[str, CodexProviderRoute],
    signed_route_level_adapter_receipt_blake3: str | None = None,
    adapter_receipt_cache: dict[tuple[Path, str, str], str] | None = None,
    schema_version: int = 1,
) -> Mapping[str, Any]:
    selected_source = source_document_path.resolve(strict=True)
    source_document, source_bytes = _canonical_json(selected_source)
    source_schema = source_document.get("schema")
    metadata: Mapping[str, Any] = {}
    if source_schema == "eva.codex-teacher-failure-evidence.v1":
        failure_core = {
            key: value
            for key, value in source_document.items()
            if key != "failure_evidence_blake3"
        }
        if (
            source_document.get("failure_evidence_blake3") != blake3_hex(failure_core)
            or source_document.get("raw_exception_message_recorded") is not False
            or source_document.get("raw_provider_material_recorded") is not False
            or source_document.get("retry_count") != 0
            or source_document.get("stage") != "S2"
        ):
            raise S2FrontierPrefixSFTError("teacher S2 failure evidence differs")
        receipt_document = source_document.get("provider_turn_receipt")
        nested_claim = source_document.get("provider_turn_receipt_blake3")
        provider_rollout_claim: str | None = None
        domain_claim = source_document.get("domain")
        source_record_kind = "failed_teacher_turn_with_completed_s2_prefix"
        source_terminal_status = "later_turn_failed"
        source_commitment = source_document["failure_evidence_blake3"]
    elif source_schema == "eva.codex-teacher-full-trajectory.v1":
        metadata_value = source_document.get("provider_metadata")
        if (
            not isinstance(metadata_value, Mapping)
            or metadata_value.get("raw_input_recorded") is not False
            or source_document.get("semantic_retry_count") != 0
        ):
            raise S2FrontierPrefixSFTError("completed S2 teacher trajectory differs")
        metadata = metadata_value
        receipt_document = metadata.get("codex_turn_receipt")
        nested_claim = None
        provider_rollout_claim = None
        domain_claim = None
        source_record_kind = "completed_teacher_turn_s2_prefix"
        source_terminal_status = "completed_full_turn_prefix_only"
        source_commitment = blake3_bytes(source_bytes)
    else:
        raise S2FrontierPrefixSFTError("teacher source document schema differs")
    sandbox_id = _uuid(source_document.get("sandbox_id"), label="sandbox_id")
    candidate_id = _uuid(source_document.get("candidate_id"), label="candidate_id")
    route_id = str(source_document.get("route_id"))
    task_id = str(source_document.get("task_id"))
    if task_id != f"{sandbox_id}--{route_id}":
        raise S2FrontierPrefixSFTError("teacher S2 task identity differs")
    if not isinstance(receipt_document, Mapping):
        raise S2FrontierPrefixSFTError("teacher source lacks a Codex receipt")
    try:
        receipt = codex_turn_receipt_from_document(receipt_document)
        verify_codex_turn_receipt(receipt)
    except Exception as exc:
        raise S2FrontierPrefixSFTError("Codex S2 turn receipt verification failed") from exc
    if (
        (nested_claim is not None and receipt.receipt_blake3 != nested_claim)
        or receipt.status != "completed"
        or receipt.visibility != "actor-public"
        or receipt.selected_skill_ids
        or receipt.sandbox.value != "read-only"
        or not receipt.role.is_actor
    ):
        raise S2FrontierPrefixSFTError("Codex S2 actor receipt boundary differs")
    authority, expected_provider, scope = _route_authority(
        route_id, routes=route_authorities
    )
    if receipt.model != authority.model_id or receipt.provider != expected_provider:
        raise S2FrontierPrefixSFTError("S2 route provider or model binding differs")

    workspace_parent = source_root / "workspaces" / task_id
    workspaces = tuple(path for path in workspace_parent.iterdir() if path.is_dir())
    if len(workspaces) != 1 or workspaces[0].name != sandbox_id:
        raise S2FrontierPrefixSFTError("teacher S2 workspace identity differs")
    workspace = workspaces[0]
    runtime_context, runtime_bytes = _canonical_json(workspace / ".eva/runtime-context.json")
    source_policy, policy_bytes = _canonical_json(workspace / ".eva/source-policy.json")
    if (
        runtime_context.get("focus") != "S2"
        or runtime_context.get("policy_blake3") != blake3_hex(source_policy)
        or source_policy.get("focus") != "S2"
    ):
        raise S2FrontierPrefixSFTError("public S2 runtime context differs")
    source_tools = _authoritative_legacy_tool_catalog(
        source_policy.get("tools"),
        expected_blake3=runtime_context.get("tool_catalog_blake3"),
    )
    skill_catalog = None
    if schema_version == 2:
        from .s2_skill_frontier_support import verified_offered_catalog
        skill_catalog = verified_offered_catalog(source_tools, receipt, source_document.get("skill_delivery"))
    try:
        guidance = build_stage_tool_guidance_v1(
            public_runtime_context=runtime_context,
            source_tool_catalog=source_tools,
        )
        verify_stage_tool_guidance_v1(
            guidance,
            public_runtime_context=runtime_context,
            source_tool_catalog=source_tools,
        )
    except NativePolicyV2Error as exc:
        raise S2FrontierPrefixSFTError("authoritative S2 guidance differs") from exc
    s1_contract = runtime_context.get("s1_plan_contract")
    s2_contract = runtime_context.get("s2_evidence_contract")
    if not isinstance(s1_contract, Mapping) or not isinstance(s2_contract, Mapping):
        raise S2FrontierPrefixSFTError("public S1/S2 contracts differ")
    hydration = _require_hydrated_s1(
        source_document,
        guidance=guidance,
        s1_contract=s1_contract,
        s2_contract=s2_contract,
    )
    effective_s2_contract = _effective_s2_contract(
        hydration, template=s2_contract
    )
    observed, selection = _require_s2_control_frontiers(
        receipt,
        guidance=guidance,
        s2_contract=effective_s2_contract,
        skill_aware=schema_version == 2,
        skill_delivery=source_document.get("skill_delivery"),
    )
    if source_schema == "eva.codex-teacher-full-trajectory.v1":
        provider_rollout_claim = _verify_completed_projection(
            source_document,
            metadata=metadata,
            receipt=receipt,
            accepted=observed,
        )

    record, bulk_binding_blake3 = bulk_record
    if (
        record.get("candidate_id") != candidate_id
        or (domain_claim is not None and record.get("domain") != domain_claim)
        or record.get("stage") != "S2"
    ):
        raise S2FrontierPrefixSFTError("bulk S2 source identity differs")
    domain = str(record.get("domain"))
    reward = record.get("reward_contract")
    rubric = reward.get("rubric_table") if isinstance(reward, Mapping) else None
    compiled = registry.resolve(domain, "S2")
    if (
        not isinstance(rubric, Mapping)
        or canonical_value(rubric) != canonical_value(compiled.to_document())
        or reward.get("rubric_id") != compiled.rubric_id
        or reward.get("rubric_digest") != compiled.digest
        or reward.get("registry_digest") != registry.digest
        or reward.get("benchmark_judging_and_rollout_reward_same_object") is not True
    ):
        raise S2FrontierPrefixSFTError("source S2 reward rubric differs")

    stage_artifacts = s1_contract.get("stage_artifacts")
    if not isinstance(stage_artifacts, Mapping):
        raise S2FrontierPrefixSFTError("declared S2 stage artifacts differ")
    plan_relative = _safe_relative(stage_artifacts.get("S1"), label="S1 artifact")
    selection_relative = _safe_relative(
        stage_artifacts.get("S2"), label="S2 artifact"
    )
    if selection_relative != _safe_relative(
        s2_contract.get("selection_relative_path"), label="S2 selection"
    ):
        raise S2FrontierPrefixSFTError("declared S2 artifact paths differ")
    plan_artifact, plan_bytes = _read_json(
        workspace.joinpath(*PurePosixPath(plan_relative).parts)
    )
    selection_artifact, selection_bytes = _read_json(
        workspace.joinpath(*PurePosixPath(selection_relative).parts)
    )
    if (
        canonical_value(plan_artifact)
        != canonical_value(guidance.frontiers[0].get("arguments"))
        or plan_artifact.get("schema")
        != "rlevo.med-research-stage-plan-artifact.v1"
        or canonical_value(selection_artifact) != canonical_value(selection.arguments)
        or selection_artifact.get("schema")
        != "rlevo.med-research-evidence-selection.v1"
    ):
        raise S2FrontierPrefixSFTError("declared S2 artifacts differ from decisions")

    initial = record.get("workspace_initial_state")
    initial_rows = initial.get("files") if isinstance(initial, Mapping) else None
    if not isinstance(initial_rows, list):
        raise S2FrontierPrefixSFTError("bulk S2 initial workspace differs")
    initial_paths: set[str] = {
        ".eva/runtime-context.json",
        ".eva/source-policy.json",
        "TASK.md",
    }
    for row in initial_rows:
        if not isinstance(row, Mapping):
            raise S2FrontierPrefixSFTError("bulk S2 workspace row differs")
        relative = _safe_relative(row.get("path"), label="bulk workspace")
        payload = canonical_json_bytes(row.get("content"))
        if (
            row.get("content_blake3") != blake3_bytes(payload)
            or row.get("byte_count") != len(payload)
            or workspace.joinpath(*PurePosixPath(relative).parts).read_bytes() != payload
        ):
            raise S2FrontierPrefixSFTError("bulk S2 workspace materialization differs")
        initial_paths.add(relative)
    before_plan = _workspace_tree(workspace, tuple(initial_paths))
    after_plan = _workspace_tree(workspace, tuple(initial_paths | {plan_relative}))
    after_selection = _workspace_tree(
        workspace, tuple(initial_paths | {plan_relative, selection_relative})
    )
    hydration_result = hydration["tool_result"]
    source_before, source_before_files = _workspace_snapshot_files(
        source_document.get("workspace_before"), expected_label="before-rollout"
    )
    source_after, source_after_files = _workspace_snapshot_files(
        source_document.get("workspace_after"), expected_label="after-rollout"
    )
    expected_before_paths = initial_paths | {plan_relative}
    expected_after_paths = expected_before_paths | {selection_relative}
    if (
        hydration_result["workspace_before_blake3"] != before_plan
        or hydration_result["workspace_after_blake3"] != after_plan
        or source_before != after_plan
        or source_after != after_selection
        or set(source_before_files) != expected_before_paths
        or set(source_after_files) != expected_after_paths
        or any(
            source_before_files[path]
            != workspace.joinpath(*PurePosixPath(path).parts).read_bytes()
            for path in expected_before_paths
        )
        or any(
            source_after_files[path]
            != workspace.joinpath(*PurePosixPath(path).parts).read_bytes()
            for path in expected_after_paths
        )
        or any(
            item.tool_result["workspace_before_blake3"] != after_plan
            or item.tool_result["workspace_after_blake3"] != after_plan
            for item in observed[:-1]
        )
        or selection.tool_result["workspace_before_blake3"] != after_plan
        or selection.tool_result["workspace_after_blake3"] != after_selection
    ):
        raise S2FrontierPrefixSFTError("S2 workspace frontier deltas do not reopen")

    adapter_key = (source_root, route_id, receipt.model)
    if scope == "native_codex_direct":
        if signed_route_level_adapter_receipt_blake3 is not None:
            raise S2FrontierPrefixSFTError("native S2 route cannot claim adapter evidence")
        adapter_receipt_blake3 = None
    elif signed_route_level_adapter_receipt_blake3 is not None:
        adapter_receipt_blake3 = _adapter_receipt(
            source_root,
            route_id=route_id,
            model_id=receipt.model,
            selected_blake3=signed_route_level_adapter_receipt_blake3,
        )
    elif adapter_receipt_cache is not None and adapter_key in adapter_receipt_cache:
        adapter_receipt_blake3 = adapter_receipt_cache[adapter_key]
    else:
        adapter_receipt_blake3 = _first_passing_s2_adapter_receipt(
            source_root, route_id=route_id, model_id=receipt.model
        )
        if adapter_receipt_cache is not None:
            adapter_receipt_cache[adapter_key] = adapter_receipt_blake3
    policy_messages = source_policy.get("messages")
    if (
        not isinstance(policy_messages, list)
        or len(policy_messages) != 2
        or [row.get("role") for row in policy_messages if isinstance(row, Mapping)]
        != ["system", "user"]
    ):
        raise S2FrontierPrefixSFTError("actor-public S2 source messages differ")
    exported_messages = [
        canonical_value(policy_messages[0]),
        {
            "role": "user",
            "content": {
                "task": policy_messages[1]["content"],
                "public_runtime_context": runtime_context,
                "rubric_table": rubric,
                "verified_prerequisite_tool_result_receipt_blake3s": [
                    hydration_result["receipt_blake3"],
                    *(item.tool_result["receipt_blake3"] for item in observed[:-1]),
                ],
            },
        },
        {
            "role": "assistant",
            "content": {
                "tool_calls": [
                    {
                        "name": "materialize_evidence_selection",
                        "arguments": selection.arguments,
                        "codex_tool_call_receipt_blake3": selection.call.receipt_blake3,
                    }
                ]
            },
        },
        {
            "role": "tool",
            "name": "materialize_evidence_selection",
            "content": selection.observation,
        },
    ]
    skill_fields = {}
    if schema_version == 2:
        from .s2_skill_frontier_support import skill_export_fields
        skill_fields = skill_export_fields(observed, source_document["skill_delivery"], skill_catalog, receipt, exported_messages)
    return {
        "source_task_id": task_id,
        "sandbox_id": sandbox_id,
        "candidate_id": candidate_id,
        "route_id": route_id,
        "model_id": receipt.model,
        "provider": receipt.provider,
        "provider_evidence_scope": scope,
        "signed_route_level_adapter_receipt_blake3": adapter_receipt_blake3,
        "domain": domain,
        "stage": "S2",
        "source_record_kind": source_record_kind,
        "source_terminal_status": source_terminal_status,
        "accepted_frontier_status": "completed",
        "prefix_boundary": "after_s2_materialize_evidence_selection_observation",
        "source_root": _relative(source_root, base=source_base, label="source root"),
        "source_record_path": _relative(
            selected_source, base=source_base, label="teacher source"
        ),
        "source_record_file_blake3": blake3_bytes(source_bytes),
        "source_record_commitment_blake3": source_commitment,
        "codex_turn_receipt_blake3": receipt.receipt_blake3,
        "source_provider_rollout_receipt_blake3": provider_rollout_claim,
        "provider_rollout_receipt_recomputed": False,
        "codex_turn_receipt_reopened": True,
        "codex_input_blake3": receipt.input_blake3,
        "stage_tool_guidance_blake3": guidance.guidance_blake3,
        "prerequisite_tool_result_receipt_blake3s": [
            hydration_result["receipt_blake3"],
            *(item.tool_result["receipt_blake3"] for item in observed[:-1]),
        ],
        "codex_tool_call_receipt_blake3": selection.call.receipt_blake3,
        "tool_result_id": selection.tool_result["result_id"],
        "tool_result_receipt_blake3": selection.tool_result["receipt_blake3"],
        "bridge_receipt_blake3": selection.observation["bridge_receipt_blake3"],
        "workspace_root": _relative(workspace, base=source_base, label="workspace"),
        "workspace_before_blake3": after_plan,
        "workspace_after_blake3": after_selection,
        "s1_artifact_relative_path": plan_relative,
        "s1_artifact_file_blake3": blake3_bytes(plan_bytes),
        "artifact_relative_path": selection_relative,
        "artifact_file_blake3": blake3_bytes(selection_bytes),
        "artifact_byte_count": len(selection_bytes),
        "runtime_context_file_blake3": blake3_bytes(runtime_bytes),
        "source_policy_file_blake3": blake3_bytes(policy_bytes),
        "bulk_binding_blake3": bulk_binding_blake3,
        "rubric_id": compiled.rubric_id,
        "rubric_digest": compiled.digest,
        "rubric_table": rubric,
        "messages": exported_messages,
        "assistant_tool_decisions": 1,
        "host_tool_observations": 1,
        "prerequisite_tool_observations_exported": 0,
        "later_stage_content_included": False,
        "terminal_answer_included": False,
        "hidden_reasoning_included": False,
        "private_reference_observations_included": False,
        "agent_judged": False,
        "strict_full_trajectory_sft_eligible": False,
        **skill_fields,
    }


def _scan(
    *,
    source_roots: Sequence[Path],
    source_base: Path,
    bulk_root: Path,
    registry: CompiledRubricRegistry,
    route_authorities: Mapping[str, CodexProviderRoute],
    schema_version: int = 1,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], dict[str, int]]:
    base = source_base.resolve(strict=True)
    sources = tuple(path.resolve(strict=True) for path in source_roots)
    if not sources or len(sources) != len(set(sources)):
        raise S2FrontierPrefixSFTError("S2 teacher roots differ")
    entries = _source_entries(sources, base=base)
    if not entries:
        raise S2FrontierPrefixSFTError("S2 teacher roots have no terminal rows")
    inventory: list[dict[str, Any]] = []
    sandbox_ids: list[str] = []
    for source, path in entries:
        document, payload = _canonical_json(path)
        sandbox_id = _uuid(document.get("sandbox_id"), label="sandbox_id")
        sandbox_ids.append(sandbox_id)
        inventory.append(
            {
                "path": _relative(path, base=base, label="teacher source"),
                "file_blake3": blake3_bytes(payload),
                "sandbox_id": sandbox_id,
                "route_id": document.get("route_id"),
                "source_schema": document.get("schema"),
                "source_root": _relative(source, base=base, label="source root"),
            }
        )
    bulk = bulk_root.resolve(strict=True)
    bulk_records = _load_bulk_records(bulk, tuple(sorted(set(sandbox_ids))))
    accepted: list[Mapping[str, Any]] = []
    rejected: Counter[str] = Counter()
    adapter_cache: dict[tuple[Path, str, str], str] = {}
    for (source, path), sandbox_id in zip(entries, sandbox_ids, strict=True):
        try:
            accepted.append(
                _source_slice(
                    source_root=source,
                    source_base=base,
                    bulk_root=bulk,
                    registry=registry,
                    source_document_path=path,
                    bulk_record=bulk_records[sandbox_id],
                    route_authorities=route_authorities,
                    adapter_receipt_cache=adapter_cache,
                    schema_version=schema_version,
                )
            )
        except (S2FrontierPrefixSFTError, FrontierPrefixSFTError) as exc:
            rejected[str(exc)] += 1
    return accepted, inventory, dict(sorted(rejected.items()))


def audit_s2_frontier_prefix_sources(
    *,
    source_roots: Sequence[Path],
    source_base: Path,
    bulk_root: Path,
    registry: CompiledRubricRegistry,
    route_authorities: Mapping[str, CodexProviderRoute],
) -> Mapping[str, Any]:
    accepted, inventory, rejected = _scan(
        source_roots=source_roots,
        source_base=source_base,
        bulk_root=bulk_root,
        registry=registry,
        route_authorities=route_authorities,
    )
    route_counts = Counter(str(row["route_id"]) for row in accepted)
    scope_counts = Counter(str(row["provider_evidence_scope"]) for row in accepted)
    core = {
        "schema": S2_FRONTIER_AUDIT_SCHEMA,
        "provider_calls": 0,
        "scanned_terminal_teacher_rows": len(inventory),
        "accepted_s2_prefixes": len(accepted),
        "rejected_rows": len(inventory) - len(accepted),
        "rejected_by_reason": rejected,
        "accepted_by_route": dict(sorted(route_counts.items())),
        "accepted_by_provider_evidence_scope": dict(sorted(scope_counts.items())),
        "source_inventory_blake3": blake3_hex(inventory),
    }
    return {**core, "audit_blake3": blake3_hex(core)}


def build_s2_frontier_prefix_sft_dataset(
    *,
    source_roots: Sequence[Path],
    source_base: Path,
    bulk_root: Path,
    output_root: Path,
    registry: CompiledRubricRegistry,
    route_authorities: Mapping[str, CodexProviderRoute],
    schema_version: int = 1,
) -> S2FrontierPrefixSFTBuild:
    if schema_version not in {1, 2}:
        raise S2FrontierPrefixSFTError("S2 frontier schema version differs")
    accepted, inventory, rejected = _scan(
        source_roots=source_roots,
        source_base=source_base,
        bulk_root=bulk_root,
        registry=registry,
        route_authorities=route_authorities,
        schema_version=schema_version,
    )
    if not accepted:
        raise S2FrontierPrefixSFTError("teacher roots have no execution-verified S2 prefix")
    dataset_id = str(UUID(RandomUUIDFactory().new("s2-frontier-prefix-sft-dataset")))
    rows: list[dict[str, Any]] = []
    for derived in accepted:
        example_id = str(UUID(RandomUUIDFactory().new("s2-frontier-prefix-sft-example")))
        row_core = {
            "schema": S2_FRONTIER_SLICE_SCHEMA if schema_version == 1 else S2_FRONTIER_SLICE_SCHEMA.removesuffix("v1") + "v2",
            "dataset_id": dataset_id,
            "example_id": example_id,
            "quality_tier": "execution_verified_s2_frontier_prefix",
            "selection_status": "s2_prefix_verified_full_trajectory_not_claimed",
            **derived,
        }
        rows.append({**row_core, "slice_blake3": blake3_hex(row_core)})
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    dataset_root = output_root / dataset_id
    dataset_root.mkdir(mode=0o700)
    shards = dataset_root / "shards"
    shards.mkdir(mode=0o700)
    shard_payload = b"".join(canonical_json_bytes(row) for row in rows)
    _write_once(shards / "part-00000.jsonl", shard_payload)
    route_counts = Counter(str(row["route_id"]) for row in rows)
    scope_counts = Counter(str(row["provider_evidence_scope"]) for row in rows)
    base = source_base.resolve(strict=True)
    bulk = bulk_root.resolve(strict=True)
    sources = tuple(path.resolve(strict=True) for path in source_roots)
    manifest_core = {
        "schema": S2_FRONTIER_DATASET_SCHEMA if schema_version == 1 else S2_FRONTIER_DATASET_SCHEMA.removesuffix("v1") + "v2",
        "dataset_id": dataset_id,
        "quality_tier": "execution_verified_s2_frontier_prefix",
        "selection_status": "s2_prefix_verified_full_trajectory_not_claimed",
        "source_base_label": base.name,
        "source_roots": sorted(
            _relative(path, base=base, label="source root") for path in sources
        ),
        "bulk_root": _relative(bulk, base=base, label="bulk root"),
        "rubric_registry_digest": registry.digest,
        "source_inventory": inventory,
        "route_authorities": _route_authority_document(
            [str(row["route_id"]) for row in inventory], routes=route_authorities
        ),
        "inventory": {
            "scanned_terminal_teacher_rows": len(inventory),
            "accepted_s2_prefixes": len(rows),
            "rejected_rows": len(inventory) - len(rows),
            "rejected_by_reason": rejected,
        },
        "source_count": len(rows),
        "slice_count": len(rows),
        "counts_by_stage": {"S2": len(rows)},
        "counts_by_route": dict(sorted(route_counts.items())),
        "counts_by_provider_evidence_scope": dict(sorted(scope_counts.items())),
        "selection": {
            "require_authoritative_s1_s2_guidance": True,
            "require_exact_sequential_prerequisites": True,
            "require_completed_codex_turn_receipt": True,
            "allow_later_terminal_failure": True,
            "require_completed_s2_selection_frontier": True,
            "require_gate_passed_true": True,
            "require_empty_failed_check_ids": True,
            "require_exact_declared_s2_artifact": True,
            "require_reopened_workspace_delta": True,
            "native_provider_families": ["glm", "openai"],
            "adapted_provider_families": sorted(_ADAPTED_PROVIDER_FAMILIES),
            "native_routes_require_authoritative_route_binding": True,
            "adapted_routes_require_signed_route_level_adapter_receipt": True,
            "provider_rollout_receipt_claim_recorded_when_present": True,
            "provider_rollout_receipt_recomputed": False,
            "codex_turn_receipt_reopened": True,
            "export_prerequisite_tool_observations": False,
            "export_later_stages": False,
            "export_terminal_answer": False,
            "export_hidden_reasoning": False,
            "export_private_reference_observations": False,
        },
        "shards": [
            {
                "path": "shards/part-00000.jsonl",
                "slice_count": len(rows),
                "first_example_id": rows[0]["example_id"],
                "last_example_id": rows[-1]["example_id"],
                "content_blake3": blake3_bytes(shard_payload),
                "byte_count": len(shard_payload),
            }
        ],
        "provider_calls": 0,
        "agent_judge_calls": 0,
        "agent_judged": False,
        "strict_full_trajectory_sft_eligible": False,
        "hidden_reasoning_included": False,
        "private_reference_observations_included": False,
    }
    if schema_version == 2:
        manifest_core["selection"].update({
            "allow_successful_s2_readonly_skill_interleaving": True,
            "require_exact_offered_skill_tool_catalog": True,
            "export_complete_visible_skill_observations": True,
            "loss_scope": "s2_selection_decision_only",
        })
    manifest = {**manifest_core, "manifest_blake3": blake3_hex(manifest_core)}
    _write_once(dataset_root / "manifest.json", canonical_json_bytes(manifest))
    os.chmod(shards, 0o555)
    os.chmod(dataset_root, 0o555)
    return S2FrontierPrefixSFTBuild(
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        source_count=len(rows),
        slice_count=len(rows),
        manifest_blake3=manifest["manifest_blake3"],
    )


def verify_s2_frontier_prefix_sft_dataset(
    *,
    dataset_root: Path,
    source_base: Path,
    registry: CompiledRubricRegistry,
    route_authorities: Mapping[str, CodexProviderRoute],
) -> Mapping[str, Any]:
    """Independently rebuild every accepted S2 frontier from source bytes."""

    try:
        root = dataset_root.resolve(strict=True)
        base = source_base.resolve(strict=True)
        manifest, manifest_bytes = _canonical_json(root / "manifest.json")
        schema_version = 2 if manifest.get("schema") == S2_FRONTIER_DATASET_SCHEMA.removesuffix("v1") + "v2" else 1
        core = {key: value for key, value in manifest.items() if key != "manifest_blake3"}
        dataset_id = _uuid(manifest.get("dataset_id"), label="dataset_id")
        selection = manifest.get("selection")
        if (
            root.name != dataset_id
            or manifest.get("schema") != (S2_FRONTIER_DATASET_SCHEMA if schema_version == 1 else S2_FRONTIER_DATASET_SCHEMA.removesuffix("v1") + "v2")
            or manifest.get("quality_tier")
            != "execution_verified_s2_frontier_prefix"
            or manifest.get("selection_status")
            != "s2_prefix_verified_full_trajectory_not_claimed"
            or manifest.get("manifest_blake3") != blake3_hex(core)
            or manifest.get("rubric_registry_digest") != registry.digest
            or type(manifest.get("source_count")) is not int
            or manifest.get("source_count") < 1
            or manifest.get("slice_count") != manifest.get("source_count")
            or manifest.get("agent_judged") is not False
            or manifest.get("hidden_reasoning_included") is not False
            or manifest.get("private_reference_observations_included") is not False
            or not isinstance(selection, Mapping)
            or selection.get("require_authoritative_s1_s2_guidance") is not True
            or selection.get("require_exact_sequential_prerequisites") is not True
            or selection.get("export_prerequisite_tool_observations") is not False
            or selection.get("export_later_stages") is not False
            or selection.get("export_terminal_answer") is not False
            or selection.get("export_hidden_reasoning") is not False
            or selection.get("export_private_reference_observations") is not False
        ):
            raise S2FrontierPrefixSFTError("S2 frontier manifest identity differs")
        if schema_version == 2 and any(selection.get(key) != value for key, value in {
            "allow_successful_s2_readonly_skill_interleaving": True,
            "require_exact_offered_skill_tool_catalog": True,
            "export_complete_visible_skill_observations": True,
            "loss_scope": "s2_selection_decision_only",
        }.items()):
            raise S2FrontierPrefixSFTError("S2 skill-aware selection contract differs")
        source_roots_value = manifest.get("source_roots")
        if not isinstance(source_roots_value, list) or not source_roots_value:
            raise S2FrontierPrefixSFTError("S2 frontier teacher roots differ")
        source_roots = {
            relative: base.joinpath(*PurePosixPath(relative).parts)
            for relative in (
                _safe_relative(value, label="source root")
                for value in source_roots_value
            )
        }
        if len(source_roots) != len(source_roots_value):
            raise S2FrontierPrefixSFTError("S2 frontier teacher roots are duplicated")
        bulk_root = base.joinpath(
            *PurePosixPath(_safe_relative(manifest.get("bulk_root"), label="bulk root")).parts
        )
        inventory = manifest.get("source_inventory")
        summary = manifest.get("inventory")
        if not isinstance(inventory, list) or not isinstance(summary, Mapping):
            raise S2FrontierPrefixSFTError("S2 frontier source inventory differs")
        inventory_paths: list[Path] = []
        inventory_roots: list[Path] = []
        sandbox_ids: list[str] = []
        for item in inventory:
            if not isinstance(item, Mapping):
                raise S2FrontierPrefixSFTError("S2 source inventory row differs")
            source_relative = _safe_relative(item.get("source_root"), label="source root")
            source_root = source_roots.get(source_relative)
            relative = _safe_relative(item.get("path"), label="teacher source")
            path = base.joinpath(*PurePosixPath(relative).parts)
            _document, payload = _canonical_json(path)
            if (
                source_root is None
                or source_root not in path.parents
                or item.get("file_blake3") != blake3_bytes(payload)
            ):
                raise S2FrontierPrefixSFTError("S2 source inventory changed")
            inventory_paths.append(path)
            inventory_roots.append(source_root)
            sandbox_ids.append(_uuid(item.get("sandbox_id"), label="sandbox_id"))
        expected_authorities = _route_authority_document(
            [str(item.get("route_id")) for item in inventory], routes=route_authorities
        )
        if canonical_value(manifest.get("route_authorities")) != canonical_value(
            expected_authorities
        ):
            raise S2FrontierPrefixSFTError("S2 route authorities differ")
        shards = manifest.get("shards")
        if not isinstance(shards, list) or len(shards) != 1:
            raise S2FrontierPrefixSFTError("S2 shard inventory differs")
        descriptor = shards[0]
        shard_relative = _safe_relative(descriptor.get("path"), label="S2 shard")
        if shard_relative != "shards/part-00000.jsonl":
            raise S2FrontierPrefixSFTError("S2 shard path differs")
        shard_payload = root.joinpath(*PurePosixPath(shard_relative).parts).read_bytes()
        if (
            descriptor.get("byte_count") != len(shard_payload)
            or descriptor.get("content_blake3") != blake3_bytes(shard_payload)
        ):
            raise S2FrontierPrefixSFTError("S2 shard commitment differs")
        lines = shard_payload.splitlines(keepends=True)
        if (
            len(lines) != manifest["slice_count"]
            or descriptor.get("slice_count") != len(lines)
            or any(canonical_json_bytes(json.loads(line)) != line for line in lines)
        ):
            raise S2FrontierPrefixSFTError("S2 shard JSON differs")
        rows = [json.loads(line, object_pairs_hook=_strict_object) for line in lines]
        observed_by_source: dict[str, Mapping[str, Any]] = {}
        example_ids: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise S2FrontierPrefixSFTError("S2 frontier slice differs")
            row_core = {key: value for key, value in row.items() if key != "slice_blake3"}
            example_id = _uuid(row.get("example_id"), label="example_id")
            source_path = str(row.get("source_record_path"))
            if (
                row.get("schema") != (S2_FRONTIER_SLICE_SCHEMA if schema_version == 1 else S2_FRONTIER_SLICE_SCHEMA.removesuffix("v1") + "v2")
                or row.get("dataset_id") != dataset_id
                or row.get("slice_blake3") != blake3_hex(row_core)
                or row.get("stage") != "S2"
                or row.get("provider_evidence_scope")
                not in {"native_codex_direct", "adapted_route_batch_unbound"}
                or row.get("later_stage_content_included") is not False
                or row.get("terminal_answer_included") is not False
                or row.get("hidden_reasoning_included") is not False
                or row.get("private_reference_observations_included") is not False
                or source_path in observed_by_source
            ):
                raise S2FrontierPrefixSFTError("S2 frontier slice commitment differs")
            observed_by_source[source_path] = row
            example_ids.append(example_id)
        if (
            len(example_ids) != len(set(example_ids))
            or descriptor.get("first_example_id") != example_ids[0]
            or descriptor.get("last_example_id") != example_ids[-1]
        ):
            raise S2FrontierPrefixSFTError("S2 example inventory differs")
        bulk_records = _load_bulk_records(
            bulk_root, tuple(sorted(set(sandbox_ids)))
        )
        expected_by_source: dict[str, Mapping[str, Any]] = {}
        rejected: Counter[str] = Counter()
        for source_root, path, sandbox_id in zip(
            inventory_roots, inventory_paths, sandbox_ids, strict=True
        ):
            source_relative = _relative(path, base=base, label="teacher source")
            observed_row = observed_by_source.get(source_relative)
            try:
                expected = _source_slice(
                    source_root=source_root,
                    source_base=base,
                    bulk_root=bulk_root,
                    registry=registry,
                    source_document_path=path,
                    bulk_record=bulk_records[sandbox_id],
                    route_authorities=route_authorities,
                    schema_version=schema_version,
                    signed_route_level_adapter_receipt_blake3=(
                        None
                        if observed_row is None
                        else observed_row.get(
                            "signed_route_level_adapter_receipt_blake3"
                        )
                    ),
                )
                expected_by_source[str(expected["source_record_path"])] = expected
            except (S2FrontierPrefixSFTError, FrontierPrefixSFTError) as exc:
                rejected[str(exc)] += 1
        if (
            len(expected_by_source) != len(rows)
            or summary.get("scanned_terminal_teacher_rows") != len(inventory)
            or summary.get("accepted_s2_prefixes") != len(rows)
            or summary.get("rejected_rows") != len(inventory) - len(rows)
            or summary.get("rejected_by_reason") != dict(sorted(rejected.items()))
        ):
            raise S2FrontierPrefixSFTError("S2 acceptance inventory differs")
        route_counts: Counter[str] = Counter()
        scope_counts: Counter[str] = Counter()
        for source_relative, row in observed_by_source.items():
            expected = expected_by_source.get(source_relative)
            observed = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "schema",
                    "dataset_id",
                    "example_id",
                    "quality_tier",
                    "selection_status",
                    "slice_blake3",
                }
            }
            if expected is None or canonical_value(observed) != canonical_value(expected):
                raise S2FrontierPrefixSFTError("S2 source reconstruction differs")
            route_counts[str(row["route_id"])] += 1
            scope_counts[str(row["provider_evidence_scope"])] += 1
        if (
            set(observed_by_source) != set(expected_by_source)
            or manifest.get("counts_by_stage") != {"S2": len(rows)}
            or manifest.get("counts_by_route") != dict(sorted(route_counts.items()))
            or manifest.get("counts_by_provider_evidence_scope")
            != dict(sorted(scope_counts.items()))
        ):
            raise S2FrontierPrefixSFTError("S2 aggregate counts differ")
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if actual_files != {"manifest.json", "shards/part-00000.jsonl"}:
            raise S2FrontierPrefixSFTError("S2 dataset contains an extra file")
        report_core = {
            "schema": S2_FRONTIER_VERIFY_SCHEMA if schema_version == 1 else S2_FRONTIER_VERIFY_SCHEMA.removesuffix("v1") + "v2",
            "dataset_id": dataset_id,
            "valid": True,
            "source_count": len(rows),
            "slice_count": len(rows),
            "manifest_file_blake3": blake3_bytes(manifest_bytes),
            "manifest_blake3": manifest["manifest_blake3"],
            "checks": [
                "completed_actor_public_codex_receipt",
                "authoritative_s1_s2_guidance",
                "exact_sequential_s1_and_retrieval_prerequisites",
                "completed_s2_selection_gate",
                "route_authority_provider_evidence_scope",
                "signed_adapter_evidence_when_adapted",
                "declared_s2_artifact_exact",
                "workspace_frontier_deltas_reopened",
                "s2_rubric_table_exact",
                "selection_observation_only",
                "no_private_reasoning_final_or_later_stage",
                "manifest_and_shard_commitments",
            ],
        }
        if schema_version == 2:
            report_core["checks"][9] = "complete_visible_skill_context_and_selection_only_loss"
            report_core["checks"].extend(["canonical_offered_skill_catalog_reopened", "actual_readonly_skill_frontiers_reopened"])
        return {**report_core, "verification_blake3": blake3_hex(report_core)}
    except (
        S2FrontierPrefixSFTError,
        FrontierPrefixSFTError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        return {"schema": S2_FRONTIER_VERIFY_SCHEMA, "valid": False, "error": str(exc)}


__all__ = [
    "S2_FRONTIER_AUDIT_SCHEMA",
    "S2_FRONTIER_DATASET_SCHEMA",
    "S2_FRONTIER_SLICE_SCHEMA",
    "S2_FRONTIER_VERIFY_SCHEMA",
    "S2FrontierPrefixSFTBuild",
    "S2FrontierPrefixSFTError",
    "audit_s2_frontier_prefix_sources",
    "build_s2_frontier_prefix_sft_dataset",
    "verify_s2_frontier_prefix_sft_dataset",
]
