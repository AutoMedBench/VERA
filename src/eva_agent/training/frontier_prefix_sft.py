"""Execution-verified stage-frontier prefixes from later-failed teacher turns.

This lane is deliberately distinct from full-trajectory and Agent-Judged SFT.
It retains only an actor-public tool decision and its exact completed host
observation when that frontier produced the declared workspace artifact.  A
later failure of the same Codex turn neither invalidates that prefix nor gets
silently represented as a successful full trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence
from uuid import UUID

from eva_agent.codex_providers import (
    ROUTE_DEFINITIONS,
    CodexProviderRoute,
    SignedAdapterReceipt,
    verify_adapter_receipt,
)
from eva_agent.codex_runtime import (
    CodexTurnReceipt,
    codex_turn_receipt_from_document,
    verify_codex_turn_receipt,
)
from eva_agent.pipeline import FileSnapshot, RandomUUIDFactory
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
    is_blake3,
)
from eva_agent.rubrics import CompiledRubricRegistry
from eva_agent.training.persistent_teacher import _ADAPTED_PROVIDER_FAMILIES


FRONTIER_PREFIX_DATASET_SCHEMA = "eva.execution-verified-frontier-prefix-sft-dataset.v2"
FRONTIER_PREFIX_SLICE_SCHEMA = "eva.execution-verified-frontier-prefix-sft-slice.v2"
FRONTIER_PREFIX_VERIFY_SCHEMA = "eva.execution-verified-frontier-prefix-sft-verification.v2"
FRONTIER_PREFIX_DATASET_SCHEMA_V3 = "eva.execution-verified-frontier-prefix-sft-dataset.v3"
FRONTIER_PREFIX_SLICE_SCHEMA_V3 = "eva.execution-verified-frontier-prefix-sft-slice.v3"
FRONTIER_PREFIX_VERIFY_SCHEMA_V3 = "eva.execution-verified-frontier-prefix-sft-verification.v3"


class FrontierPrefixSFTError(ValueError):
    """A frontier, source commitment, or exported dataset differs."""


@dataclass(frozen=True)
class FrontierPrefixSFTBuild:
    dataset_id: str
    dataset_root: Path
    source_count: int
    slice_count: int
    manifest_blake3: str


def _route_authority(
    route_id: str,
    *,
    routes: Mapping[str, CodexProviderRoute] | None,
) -> tuple[CodexProviderRoute, str, str]:
    if routes is None:
        raise FrontierPrefixSFTError("v3 route authorities are absent")
    route = routes.get(route_id)
    definition = ROUTE_DEFINITIONS.get(route_id)
    if (
        route is None
        or definition is None
        or route.route_id != definition.route_id
        or route.model_env_name != definition.model_env_name
        or route.registry_role_id != definition.registry_role_id
        or route.provider_family != definition.provider_family
        or route.config.model_id != route.model_id
    ):
        raise FrontierPrefixSFTError("v3 authoritative route definition differs")
    adapted = route.provider_family in _ADAPTED_PROVIDER_FAMILIES
    if not adapted and route.provider_family not in {"openai", "glm"}:
        raise FrontierPrefixSFTError("v3 provider execution family differs")
    expected_provider = (
        f"eva_adapter_{route_id}" if adapted else route.config.provider_id
    )
    scope = (
        "adapted_route_batch_unbound" if adapted else "native_codex_direct"
    )
    return route, expected_provider, scope


def _route_authority_document(
    route_ids: Sequence[str],
    *,
    routes: Mapping[str, CodexProviderRoute] | None,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for route_id in sorted(set(route_ids)):
        route, provider, scope = _route_authority(route_id, routes=routes)
        rows.append(
            {
                "route_id": route.route_id,
                "model_id": route.model_id,
                "model_env_name": route.model_env_name,
                "registry_role_id": route.registry_role_id,
                "provider_family": route.provider_family,
                "expected_codex_provider": provider,
                "provider_evidence_scope": scope,
            }
        )
    return rows


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FrontierPrefixSFTError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
        if (
            resolved != path.absolute()
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise FrontierPrefixSFTError("JSON source topology differs")
        payload = path.read_bytes()
        document = json.loads(payload, object_pairs_hook=_strict_object)
    except FrontierPrefixSFTError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise FrontierPrefixSFTError("JSON source differs") from None
    if not isinstance(document, Mapping):
        raise FrontierPrefixSFTError("JSON source is not an object")
    return document, payload


def _canonical_json(path: Path) -> tuple[Mapping[str, Any], bytes]:
    document, payload = _read_json(path)
    if canonical_json_bytes(document) != payload:
        raise FrontierPrefixSFTError("JSON source is not canonical")
    return document, payload


def _uuid(value: Any, *, label: str) -> str:
    try:
        result = str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        raise FrontierPrefixSFTError(f"{label} is not a UUID") from None
    if result != value:
        raise FrontierPrefixSFTError(f"{label} is not canonical")
    return result


def _relative(path: Path, *, base: Path, label: str) -> str:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink():
        raise FrontierPrefixSFTError(f"{label} topology differs")
    try:
        value = resolved.relative_to(base).as_posix()
    except ValueError:
        raise FrontierPrefixSFTError(f"{label} lies outside source base") from None
    if value in {"", "."}:
        raise FrontierPrefixSFTError(f"{label} equals source base")
    return value


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise FrontierPrefixSFTError(f"{label} path differs")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FrontierPrefixSFTError(f"{label} path differs")
    return path.as_posix()


def _load_bulk_records(
    bulk_root: Path, sandbox_ids: Sequence[str]
) -> Mapping[str, tuple[Mapping[str, Any], str]]:
    requested = frozenset(sandbox_ids)
    if not requested or len(requested) != len(tuple(sandbox_ids)):
        raise FrontierPrefixSFTError("bulk sandbox request differs")
    manifest, manifest_bytes = _canonical_json(bulk_root / "manifest.json")
    if manifest.get("schema") != "eva.signed-envelope.v1":
        raise FrontierPrefixSFTError("bulk manifest schema differs")
    if manifest.get("payload_blake3") != blake3_hex(manifest.get("payload")):
        raise FrontierPrefixSFTError("bulk manifest payload commitment differs")
    unsigned = {key: value for key, value in manifest.items() if key != "envelope_blake3"}
    if manifest.get("envelope_blake3") != blake3_hex(unsigned):
        raise FrontierPrefixSFTError("bulk manifest envelope commitment differs")
    payload = manifest.get("payload")
    shards = payload.get("shards") if isinstance(payload, Mapping) else None
    if not isinstance(shards, list):
        raise FrontierPrefixSFTError("bulk manifest shards differ")
    found: dict[str, tuple[Mapping[str, Any], str]] = {}
    for descriptor in shards:
        if not isinstance(descriptor, Mapping):
            raise FrontierPrefixSFTError("bulk shard descriptor differs")
        shard_name = _safe_relative(descriptor.get("path"), label="bulk shard")
        shard_path = bulk_root / shard_name
        raw = shard_path.read_bytes()
        missing = requested - set(found)
        needles = tuple(
            f'"sandbox_id":"{sandbox_id}"'.encode("utf-8") for sandbox_id in missing
        )
        if not any(needle in raw for needle in needles):
            continue
        # Trust the already-verified signed bulk envelope and reopen only the
        # shard that actually supplies a requested record.  This avoids
        # repeatedly hashing the 6,000-record corpus in the SFT hot path.
        if len(raw) != descriptor.get("byte_count") or blake3_bytes(raw) != descriptor.get(
            "file_blake3"
        ):
            raise FrontierPrefixSFTError("bulk shard file commitment differs")
        lines = raw.splitlines(keepends=True)
        if len(lines) != descriptor.get("record_count"):
            raise FrontierPrefixSFTError("bulk shard record count differs")
        for line in lines:
            try:
                row = json.loads(line, object_pairs_hook=_strict_object)
            except (UnicodeError, ValueError):
                raise FrontierPrefixSFTError("bulk record JSON differs") from None
            if not isinstance(row, Mapping):
                raise FrontierPrefixSFTError("bulk record is not an object")
            sandbox_id = row.get("sandbox_id")
            if sandbox_id in requested:
                if sandbox_id in found:
                    raise FrontierPrefixSFTError("bulk sandbox is duplicated")
                core = {key: value for key, value in row.items() if key != "record_blake3"}
                if row.get("record_blake3") != blake3_hex(core):
                    raise FrontierPrefixSFTError("bulk record commitment differs")
                found[str(sandbox_id)] = (
                    row,
                    blake3_bytes(manifest_bytes + shard_name.encode("utf-8")),
                )
        if set(found) == requested:
            break
    if set(found) != requested:
        raise FrontierPrefixSFTError("bulk sandbox is absent")
    return found


def _load_bulk_record(bulk_root: Path, sandbox_id: str) -> tuple[Mapping[str, Any], str]:
    return _load_bulk_records(bulk_root, (sandbox_id,))[sandbox_id]


def _signed_adapter_receipt(document: Mapping[str, Any]) -> SignedAdapterReceipt:
    if set(document) != {
        "schema",
        "signature_domain",
        "payload",
        "payload_blake3",
        "key_id",
        "public_key_base64",
        "public_key_blake3",
        "algorithm",
        "signature_base64",
        "envelope_blake3",
    }:
        raise FrontierPrefixSFTError("adapter receipt fields differ")
    receipt = SignedAdapterReceipt(
        payload=document["payload"],
        payload_blake3=str(document["payload_blake3"]),
        key_id=str(document["key_id"]),
        public_key_base64=str(document["public_key_base64"]),
        public_key_blake3=str(document["public_key_blake3"]),
        algorithm=str(document["algorithm"]),
        signature_base64=str(document["signature_base64"]),
        envelope_blake3=str(document["envelope_blake3"]),
    )
    try:
        verify_adapter_receipt(receipt)
    except Exception as exc:
        raise FrontierPrefixSFTError("adapter receipt verification failed") from exc
    if canonical_value(receipt.to_dict()) != canonical_value(document):
        raise FrontierPrefixSFTError("adapter receipt reconstruction differs")
    return receipt


def _adapter_receipt(
    source_root: Path,
    *,
    route_id: str,
    model_id: str,
    selected_blake3: str | None = None,
) -> str:
    receipt_root = source_root / "adapter-receipts" / route_id
    if receipt_root.is_symlink() or not receipt_root.is_dir():
        raise FrontierPrefixSFTError("adapter receipt root differs")
    if selected_blake3 is not None and not is_blake3(selected_blake3):
        raise FrontierPrefixSFTError("selected adapter receipt digest differs")
    paths = (
        (receipt_root / f"{selected_blake3}.json",)
        if selected_blake3 is not None
        else tuple(sorted(receipt_root.glob("*.json")))
    )
    receipts: list[SignedAdapterReceipt] = []
    for path in paths:
        document, payload = _canonical_json(path)
        receipt = _signed_adapter_receipt(document)
        if (
            path.stem != receipt.envelope_blake3
            or blake3_bytes(payload) != blake3_bytes(canonical_json_bytes(document))
            or receipt.payload.get("route_id") != route_id
            or receipt.payload.get("model_id") != model_id
            or receipt.payload.get("status") != "passed"
        ):
            raise FrontierPrefixSFTError("adapter receipt route or model differs")
        receipts.append(receipt)
    if not receipts:
        raise FrontierPrefixSFTError("adapter receipts are absent")
    first = [
        item
        for item in receipts
        if item.payload["request_shape"].get("function_call_output_count") == 0
        and item.payload["response_shape"].get("parallel_function_call_count") == 1
        and "function_call" in item.payload["response_shape"].get("output_item_types", [])
    ]
    if not first:
        raise FrontierPrefixSFTError("first frontier adapter receipt is absent")
    selected = min(
        first,
        key=lambda item: (
            str(item.payload.get("created_at_utc")), item.envelope_blake3
        ),
    )
    if selected_blake3 is not None and selected.envelope_blake3 != selected_blake3:
        raise FrontierPrefixSFTError("selected adapter receipt differs")
    return selected.envelope_blake3


def _workspace_tree(root: Path, paths: Sequence[str]) -> str:
    files: list[FileSnapshot] = []
    for relative in sorted(paths):
        path = root.joinpath(*PurePosixPath(relative).parts)
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise FrontierPrefixSFTError("workspace file topology differs")
        payload = path.read_bytes()
        files.append(
            FileSnapshot(
                path=relative,
                content=payload,
                byte_count=len(payload),
                mode=f"{stat.S_IMODE(metadata.st_mode):04o}",
                content_blake3=blake3_bytes(payload),
            )
        )
    return blake3_hex(
        {
            "files": tuple(files),
            "file_count": len(files),
            "byte_count": sum(item.byte_count for item in files),
        }
    )


def _require_s1_gate(observation: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if set(observation) != {
        "schema",
        "call_id",
        "name",
        "arguments",
        "tool_result",
        "bridge_receipt_blake3",
    } or observation.get("schema") != "eva.codex-pipeline-tool-observation.v1":
        raise FrontierPrefixSFTError("host tool observation shape differs")
    core = {key: value for key, value in observation.items() if key != "bridge_receipt_blake3"}
    if observation.get("bridge_receipt_blake3") != blake3_hex(core):
        raise FrontierPrefixSFTError("host tool bridge receipt differs")
    result = observation.get("tool_result")
    arguments = observation.get("arguments")
    if not isinstance(result, Mapping) or not isinstance(arguments, Mapping):
        raise FrontierPrefixSFTError("host tool observation value differs")
    if set(result) != {
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
    }:
        raise FrontierPrefixSFTError("host ToolResult shape differs")
    result_core = {key: value for key, value in result.items() if key != "receipt_blake3"}
    output = result.get("output")
    if (
        result.get("receipt_blake3") != blake3_hex(result_core)
        or observation.get("name") != "materialize_plan"
        or result.get("name") != "materialize_plan"
        or result.get("call_id") != observation.get("call_id")
        or result.get("status") != "completed"
        or result.get("error_code") is not None
        or not isinstance(output, Mapping)
        or output.get("stage") != "S1"
        or output.get("effect") != "plan_materialization"
        or output.get("gate_passed") is not True
        or output.get("failed_check_ids") != []
        or output.get("next_stage") != "S2"
        or result.get("workspace_before_blake3")
        == result.get("workspace_after_blake3")
        or not is_blake3(result.get("workspace_before_blake3"))
        or not is_blake3(result.get("workspace_after_blake3"))
    ):
        raise FrontierPrefixSFTError("S1 host gate did not pass exactly")
    return arguments, result


def _provider_rollout_receipt_claim(
    source: Mapping[str, Any], *, receipt: CodexTurnReceipt
) -> str:
    """Retain, but do not claim to recompute, the producer's rollout digest."""

    claimed = source.get("provider_receipt_blake3")
    if (
        not is_blake3(claimed)
        or source.get("model_id") != receipt.model
        or source.get("provider") != receipt.provider
        or source.get("assistant_output") != receipt.final_response
    ):
        raise FrontierPrefixSFTError("completed teacher provider projection differs")
    return str(claimed)


def _verify_completed_teacher_projection(
    source: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    receipt: CodexTurnReceipt,
    first_tool_result: Mapping[str, Any],
) -> str:
    """Reopen the full-trajectory projection without conflating its two receipts.

    ``provider_receipt_blake3`` commits the provider rollout projection.  The
    nested Codex receipt has its own independently verifiable
    ``receipt_blake3``; they are deliberately different commitments.
    """

    provider_rollout_receipt = _provider_rollout_receipt_claim(source, receipt=receipt)
    messages = source.get("messages")
    trace = source.get("tool_trace")
    if (
        metadata.get("schema") != "eva.codex-provider-rollout-projection.v1"
        or not isinstance(messages, list)
        or len(messages) < 3
        or not isinstance(trace, Mapping)
    ):
        raise FrontierPrefixSFTError("completed teacher provider projection differs")
    for event in messages:
        if not isinstance(event, Mapping):
            raise FrontierPrefixSFTError("completed teacher policy event differs")
        core = {
            "event_id": event.get("event_id"),
            "role": event.get("role"),
            "content": event.get("content"),
            "tool_call_ids": event.get("tool_call_ids"),
        }
        if event.get("event_blake3") != blake3_hex(core):
            raise FrontierPrefixSFTError("completed teacher policy event differs")
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
        raise FrontierPrefixSFTError("completed teacher terminal projection differs")
    trace_core = {key: value for key, value in trace.items() if key != "trace_blake3"}
    if trace.get("trace_blake3") != blake3_hex(trace_core):
        raise FrontierPrefixSFTError("completed teacher tool trace differs")
    first_call = receipt.tool_calls[0]
    bindings = metadata.get("codex_to_pipeline_call_ids")
    results = trace.get("results")
    if not isinstance(bindings, Mapping) or not isinstance(results, list):
        raise FrontierPrefixSFTError("completed teacher tool binding differs")
    bound_call_id = bindings.get(first_call.tool_call_id)
    matches = [
        result
        for result in results
        if isinstance(result, Mapping) and result.get("call_id") == bound_call_id
    ]
    if (
        bound_call_id != first_tool_result.get("call_id")
        or len(matches) != 1
        or canonical_value(matches[0]) != canonical_value(first_tool_result)
    ):
        raise FrontierPrefixSFTError("completed teacher first-frontier binding differs")
    return str(provider_rollout_receipt)


def _source_slice(
    *,
    source_root: Path,
    source_base: Path,
    bulk_root: Path,
    registry: CompiledRubricRegistry,
    source_document_path: Path | None = None,
    bulk_record: tuple[Mapping[str, Any], str] | None = None,
    signed_route_level_adapter_receipt_blake3: str | None = None,
    adapter_receipt_cache: dict[tuple[Path, str, str], str] | None = None,
    schema_version: int = 2,
    route_authorities: Mapping[str, CodexProviderRoute] | None = None,
) -> Mapping[str, Any]:
    if schema_version not in {2, 3}:
        raise FrontierPrefixSFTError("frontier-prefix schema version differs")
    if source_document_path is None:
        sources = tuple(sorted((source_root / "errors").glob("*.provider-failure.json")))
        if len(sources) != 1:
            raise FrontierPrefixSFTError("source must contain exactly one teacher failure")
        selected_source = sources[0]
    else:
        selected_source = source_document_path.resolve(strict=True)
        error_parent = (source_root / "errors").resolve(strict=True)
        trajectory_parent = (source_root / "trajectories").absolute()
        if not (
            (
                selected_source.parent == error_parent
                and selected_source.name.endswith(".provider-failure.json")
            )
            or (
                selected_source.name == "result.json"
                and trajectory_parent in selected_source.parents
            )
        ):
            raise FrontierPrefixSFTError("teacher source document path differs")
    source_document, source_bytes = _canonical_json(selected_source)
    source_schema = source_document.get("schema")
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
            or source_document.get("stage") != "S1"
        ):
            raise FrontierPrefixSFTError("teacher failure evidence differs")
        receipt_document = source_document.get("provider_turn_receipt")
        codex_receipt_blake3 = source_document.get("provider_turn_receipt_blake3")
        provider_rollout_receipt_claim_blake3: str | None = None
        domain = source_document.get("domain")
        source_record_kind = "failed_teacher_turn_with_completed_prefix"
        source_terminal_status = "later_turn_failed"
        source_record_commitment = source_document["failure_evidence_blake3"]
    elif source_schema == "eva.codex-teacher-full-trajectory.v1":
        metadata = source_document.get("provider_metadata")
        receipt_document = (
            metadata.get("codex_turn_receipt") if isinstance(metadata, Mapping) else None
        )
        # The top-level provider receipt is a projection commitment, not the
        # nested Codex-turn receipt.  The latter self-authenticates below.
        codex_receipt_blake3 = None
        provider_rollout_receipt_claim_blake3 = None
        domain = None
        source_record_kind = "completed_teacher_turn_prefix"
        source_terminal_status = "completed_full_turn_prefix_only"
        source_record_commitment = blake3_bytes(source_bytes)
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("raw_input_recorded") is not False
            or source_document.get("semantic_retry_count") != 0
        ):
            raise FrontierPrefixSFTError("completed teacher trajectory differs")
    else:
        raise FrontierPrefixSFTError("teacher source document schema differs")
    sandbox_id = _uuid(source_document.get("sandbox_id"), label="sandbox_id")
    candidate_id = _uuid(source_document.get("candidate_id"), label="candidate_id")
    task_id = str(source_document.get("task_id"))
    route_id = str(source_document.get("route_id"))
    if task_id != f"{sandbox_id}--{route_id}":
        raise FrontierPrefixSFTError("teacher task identity differs")
    if not isinstance(receipt_document, Mapping):
        raise FrontierPrefixSFTError("teacher source lacks a Codex receipt")
    try:
        receipt = codex_turn_receipt_from_document(receipt_document)
        verify_codex_turn_receipt(receipt)
    except Exception as exc:
        raise FrontierPrefixSFTError("Codex turn receipt verification failed") from exc
    if (
        (
            codex_receipt_blake3 is not None
            and receipt.receipt_blake3 != codex_receipt_blake3
        )
        or receipt.status != "completed"
        or receipt.visibility != "actor-public"
        or receipt.selected_skill_ids
        or receipt.sandbox.value != "read-only"
        or not receipt.role.is_actor
    ):
        raise FrontierPrefixSFTError("Codex actor receipt boundary differs")
    provider_evidence_scope: str | None = None
    if schema_version == 3:
        authority, expected_provider, provider_evidence_scope = _route_authority(
            route_id, routes=route_authorities
        )
        if receipt.model != authority.model_id or receipt.provider != expected_provider:
            raise FrontierPrefixSFTError("v3 route provider or model binding differs")
    calls = tuple(receipt.tool_calls)
    if not calls or calls[0].fully_qualified_name != "evamed/materialize_plan":
        raise FrontierPrefixSFTError("S1 materialize_plan was not the first tool frontier")
    call = calls[0]
    if call.status != "completed":
        raise FrontierPrefixSFTError("S1 Codex tool call did not complete")
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
        raise FrontierPrefixSFTError("S1 Codex MCP result envelope differs")
    structured = result_envelope.get("structuredContent")
    if not isinstance(structured, Mapping):
        raise FrontierPrefixSFTError("S1 host observation is absent")
    arguments, tool_result = _require_s1_gate(structured)
    if canonical_value(arguments) != canonical_value(call.arguments):
        raise FrontierPrefixSFTError("assistant and host S1 arguments differ")
    if source_schema == "eva.codex-teacher-full-trajectory.v1":
        provider_rollout_receipt_claim_blake3 = _verify_completed_teacher_projection(
            source_document,
            metadata=metadata,
            receipt=receipt,
            first_tool_result=tool_result,
        )

    workspace_parent = source_root / "workspaces" / task_id
    workspaces = tuple(path for path in workspace_parent.iterdir() if path.is_dir())
    if len(workspaces) != 1 or workspaces[0].name != sandbox_id:
        raise FrontierPrefixSFTError("teacher workspace identity differs")
    workspace = workspaces[0]
    runtime_context, runtime_bytes = _canonical_json(workspace / ".eva/runtime-context.json")
    source_policy, policy_bytes = _canonical_json(workspace / ".eva/source-policy.json")
    if (
        runtime_context.get("focus") != "S1"
        or runtime_context.get("policy_blake3") != blake3_hex(source_policy)
        or source_policy.get("focus") != "S1"
    ):
        raise FrontierPrefixSFTError("public runtime context differs")
    plan_contract = runtime_context.get("s1_plan_contract")
    stage_artifacts = plan_contract.get("stage_artifacts") if isinstance(plan_contract, Mapping) else None
    if not isinstance(stage_artifacts, Mapping):
        raise FrontierPrefixSFTError("declared stage artifacts differ")
    artifact_relative = _safe_relative(stage_artifacts.get("S1"), label="S1 artifact")
    artifact, artifact_bytes = _read_json(
        workspace.joinpath(*PurePosixPath(artifact_relative).parts)
    )
    if (
        canonical_value(artifact) != canonical_value(arguments)
        or artifact.get("schema") != "rlevo.med-research-stage-plan-artifact.v1"
    ):
        raise FrontierPrefixSFTError("declared S1 artifact differs from tool decision")

    record, bulk_binding_blake3 = (
        bulk_record
        if bulk_record is not None
        else _load_bulk_record(bulk_root, sandbox_id)
    )
    if (
        record.get("candidate_id") != candidate_id
        or (domain is not None and record.get("domain") != domain)
        or record.get("stage") != "S1"
    ):
        raise FrontierPrefixSFTError("bulk source identity differs")
    reward = record.get("reward_contract")
    rubric = reward.get("rubric_table") if isinstance(reward, Mapping) else None
    domain = record.get("domain")
    compiled = registry.resolve(str(domain), "S1")
    if (
        not isinstance(rubric, Mapping)
        or canonical_value(rubric) != canonical_value(compiled.to_document())
        or reward.get("rubric_id") != compiled.rubric_id
        or reward.get("rubric_digest") != compiled.digest
        or reward.get("registry_digest") != registry.digest
        or reward.get("benchmark_judging_and_rollout_reward_same_object") is not True
    ):
        raise FrontierPrefixSFTError("source reward rubric differs")
    initial = record.get("workspace_initial_state")
    initial_rows = initial.get("files") if isinstance(initial, Mapping) else None
    if not isinstance(initial_rows, list):
        raise FrontierPrefixSFTError("bulk initial workspace differs")
    initial_paths: set[str] = {".eva/runtime-context.json", ".eva/source-policy.json", "TASK.md"}
    for row in initial_rows:
        if not isinstance(row, Mapping):
            raise FrontierPrefixSFTError("bulk initial workspace row differs")
        relative = _safe_relative(row.get("path"), label="bulk workspace")
        payload = canonical_json_bytes(row.get("content"))
        if (
            row.get("content_blake3") != blake3_bytes(payload)
            or row.get("byte_count") != len(payload)
            or workspace.joinpath(*PurePosixPath(relative).parts).read_bytes() != payload
        ):
            raise FrontierPrefixSFTError("bulk workspace materialization differs")
        initial_paths.add(relative)
    # Reopen each declaration even though later-stage outputs are deliberately
    # excluded from the prefix.  Additional host receipts may exist after S1;
    # the exact S1 before/after tree commitments below prove they were not
    # present at the accepted frontier without leaking their bytes into SFT.
    for value in stage_artifacts.values():
        _safe_relative(value, label="declared stage artifact")
    actual_paths = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    if not initial_paths.issubset(actual_paths) or artifact_relative not in actual_paths:
        raise FrontierPrefixSFTError("S1 workspace materialization is incomplete")
    before = _workspace_tree(workspace, tuple(initial_paths))
    after = _workspace_tree(workspace, tuple(initial_paths | {artifact_relative}))
    if (
        before != tool_result["workspace_before_blake3"]
        or after != tool_result["workspace_after_blake3"]
    ):
        raise FrontierPrefixSFTError("S1 workspace delta does not reopen")

    adapter_key = (source_root, route_id, receipt.model)
    adapter_required = schema_version == 2 or provider_evidence_scope == "adapted_route_batch_unbound"
    if not adapter_required:
        if signed_route_level_adapter_receipt_blake3 is not None:
            raise FrontierPrefixSFTError("native route cannot claim adapter evidence")
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
        adapter_receipt_blake3 = _adapter_receipt(
            source_root,
            route_id=route_id,
            model_id=receipt.model,
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
        raise FrontierPrefixSFTError("actor-public source messages differ")
    messages = [
        canonical_value(policy_messages[0]),
        {
            "role": "user",
            "content": {
                "task": policy_messages[1]["content"],
                "public_runtime_context": runtime_context,
                "rubric_table": rubric,
            },
        },
        {
            "role": "assistant",
            "content": {
                "tool_calls": [
                    {
                        "name": "materialize_plan",
                        "arguments": arguments,
                        "codex_tool_call_receipt_blake3": call.receipt_blake3,
                    }
                ]
            },
        },
        {
            "role": "tool",
            "name": "materialize_plan",
            "content": structured,
        },
    ]
    provider_fields = (
        {
            "signed_route_level_adapter_receipt_blake3": adapter_receipt_blake3,
            "adapter_evidence_scope": "route_batch_unbound",
        }
        if schema_version == 2
        else {
            "signed_route_level_adapter_receipt_blake3": adapter_receipt_blake3,
            "provider_evidence_scope": provider_evidence_scope,
        }
    )
    return {
        "source_task_id": task_id,
        "sandbox_id": sandbox_id,
        "candidate_id": candidate_id,
        "route_id": route_id,
        "model_id": receipt.model,
        "provider": receipt.provider,
        "domain": domain,
        "stage": "S1",
        "source_record_kind": source_record_kind,
        "source_terminal_status": source_terminal_status,
        "accepted_frontier_status": "completed",
        "prefix_boundary": "after_s1_materialize_plan_observation",
        "source_root": _relative(source_root, base=source_base, label="source root"),
        "source_record_path": _relative(selected_source, base=source_base, label="teacher source"),
        "source_record_file_blake3": blake3_bytes(source_bytes),
        "source_record_commitment_blake3": source_record_commitment,
        "codex_turn_receipt_blake3": receipt.receipt_blake3,
        "source_provider_rollout_receipt_blake3": provider_rollout_receipt_claim_blake3,
        "provider_rollout_receipt_recomputed": False,
        "codex_turn_receipt_reopened": True,
        "codex_input_blake3": receipt.input_blake3,
        "codex_tool_call_receipt_blake3": call.receipt_blake3,
        "tool_result_id": tool_result["result_id"],
        "tool_result_receipt_blake3": tool_result["receipt_blake3"],
        "bridge_receipt_blake3": structured["bridge_receipt_blake3"],
        **provider_fields,
        "workspace_root": _relative(workspace, base=source_base, label="workspace"),
        "workspace_before_blake3": before,
        "workspace_after_blake3": after,
        "artifact_relative_path": artifact_relative,
        "artifact_file_blake3": blake3_bytes(artifact_bytes),
        "artifact_byte_count": len(artifact_bytes),
        "runtime_context_file_blake3": blake3_bytes(runtime_bytes),
        "source_policy_file_blake3": blake3_bytes(policy_bytes),
        "bulk_binding_blake3": bulk_binding_blake3,
        "rubric_id": compiled.rubric_id,
        "rubric_digest": compiled.digest,
        "rubric_table": rubric,
        "messages": messages,
        "assistant_tool_decisions": 1,
        "host_tool_observations": 1,
        "later_stage_content_included": False,
        "terminal_answer_included": False,
        "hidden_reasoning_included": False,
        "private_reference_included": False,
        "agent_judged": False,
        "strict_full_trajectory_sft_eligible": False,
    }


def _write_once(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short frontier-prefix SFT write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def build_frontier_prefix_sft_dataset(
    *,
    source_root: Path | None = None,
    source_roots: Sequence[Path] = (),
    source_base: Path,
    bulk_root: Path,
    output_root: Path,
    registry: CompiledRubricRegistry,
    schema_version: int = 2,
    route_authorities: Mapping[str, CodexProviderRoute] | None = None,
) -> FrontierPrefixSFTBuild:
    """Build immutable S1 prefixes from every stable failed row in a teacher root."""

    if schema_version not in {2, 3}:
        raise FrontierPrefixSFTError("frontier-prefix schema version differs")
    dataset_schema = (
        FRONTIER_PREFIX_DATASET_SCHEMA
        if schema_version == 2
        else FRONTIER_PREFIX_DATASET_SCHEMA_V3
    )
    slice_schema = (
        FRONTIER_PREFIX_SLICE_SCHEMA
        if schema_version == 2
        else FRONTIER_PREFIX_SLICE_SCHEMA_V3
    )
    base = source_base.resolve(strict=True)
    requested_sources = tuple(source_roots) or (() if source_root is None else (source_root,))
    if not requested_sources:
        raise FrontierPrefixSFTError("teacher roots are empty")
    sources = tuple(path.resolve(strict=True) for path in requested_sources)
    if len(sources) != len(set(sources)):
        raise FrontierPrefixSFTError("teacher roots are duplicated")
    bulk = bulk_root.resolve(strict=True)
    source_entries = tuple(
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
    if not source_entries:
        raise FrontierPrefixSFTError("teacher root has no terminal rows")
    inventory: list[dict[str, Any]] = []
    sandbox_ids: list[str] = []
    for source, source_path in source_entries:
        source_document, source_bytes = _canonical_json(source_path)
        sandbox_id = _uuid(source_document.get("sandbox_id"), label="sandbox_id")
        sandbox_ids.append(sandbox_id)
        inventory.append(
            {
                "path": _relative(source_path, base=base, label="teacher source"),
                "file_blake3": blake3_bytes(source_bytes),
                "sandbox_id": sandbox_id,
                "route_id": source_document.get("route_id"),
                "source_schema": source_document.get("schema"),
                "source_root": _relative(source, base=base, label="source root"),
            }
        )
    bulk_records = _load_bulk_records(bulk, tuple(sorted(set(sandbox_ids))))
    accepted: list[Mapping[str, Any]] = []
    rejected: dict[str, int] = {}
    adapter_receipt_cache: dict[tuple[Path, str, str], str] = {}
    for (source, source_path), sandbox_id in zip(source_entries, sandbox_ids, strict=True):
        try:
            accepted.append(
                _source_slice(
                    source_root=source,
                    source_base=base,
                    bulk_root=bulk,
                    registry=registry,
                    source_document_path=source_path,
                    bulk_record=bulk_records[sandbox_id],
                    adapter_receipt_cache=adapter_receipt_cache,
                    schema_version=schema_version,
                    route_authorities=route_authorities,
                )
            )
        except FrontierPrefixSFTError as exc:
            reason = str(exc)
            rejected[reason] = rejected.get(reason, 0) + 1
    if not accepted:
        raise FrontierPrefixSFTError("teacher root has no execution-verified S1 prefix")
    dataset_id = str(UUID(RandomUUIDFactory().new("frontier-prefix-sft-dataset")))
    rows: list[dict[str, Any]] = []
    for derived in accepted:
        example_id = str(UUID(RandomUUIDFactory().new("frontier-prefix-sft-example")))
        row_core = {
            "schema": slice_schema,
            "dataset_id": dataset_id,
            "example_id": example_id,
            "quality_tier": "execution_verified_frontier_prefix",
            "selection_status": "stage_prefix_verified_full_trajectory_not_claimed",
            **derived,
        }
        rows.append({**row_core, "slice_blake3": blake3_hex(row_core)})
    root_parent = output_root
    root_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    dataset_root = root_parent / dataset_id
    dataset_root.mkdir(mode=0o700)
    shards = dataset_root / "shards"
    shards.mkdir(mode=0o700)
    shard_payload = b"".join(canonical_json_bytes(row) for row in rows)
    _write_once(shards / "part-00000.jsonl", shard_payload)
    route_counts: dict[str, int] = {}
    scope_counts: dict[str, int] = {}
    for row in rows:
        route = str(row["route_id"])
        route_counts[route] = route_counts.get(route, 0) + 1
        if schema_version == 3:
            scope = str(row["provider_evidence_scope"])
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
    v3_manifest = (
        {}
        if schema_version == 2
        else {
            "route_authorities": _route_authority_document(
                [str(item["route_id"]) for item in inventory],
                routes=route_authorities,
            ),
            "counts_by_provider_evidence_scope": dict(sorted(scope_counts.items())),
        }
    )
    manifest_core = {
        "schema": dataset_schema,
        "dataset_id": dataset_id,
        "quality_tier": "execution_verified_frontier_prefix",
        "selection_status": "stage_prefix_verified_full_trajectory_not_claimed",
        "source_base_label": base.name,
        "source_roots": sorted(
            _relative(source, base=base, label="source root") for source in sources
        ),
        "bulk_root": _relative(bulk, base=base, label="bulk root"),
        "rubric_registry_digest": registry.digest,
        "source_error_inventory": inventory,
        "inventory": {
            "scanned_terminal_teacher_rows": len(source_entries),
            "accepted_s1_prefixes": len(rows),
            "rejected_rows": len(source_entries) - len(rows),
            "rejected_by_reason": dict(sorted(rejected.items())),
        },
        "source_count": len(rows),
        "slice_count": len(rows),
        "counts_by_stage": {"S1": len(rows)},
        "counts_by_route": dict(sorted(route_counts.items())),
        "selection": {
            "require_completed_codex_turn_receipt": True,
            "allow_later_terminal_failure": True,
            "require_completed_frontier": True,
            "require_gate_passed_true": True,
            "require_empty_failed_check_ids": True,
            "require_exact_declared_artifact": True,
            "require_reopened_workspace_delta": True,
            **(
                {
                    "require_signed_route_level_adapter_receipt": True,
                    "adapter_evidence_scope": "route_batch_unbound",
                }
                if schema_version == 2
                else {
                    "native_provider_families": ["glm", "openai"],
                    "adapted_provider_families": sorted(
                        _ADAPTED_PROVIDER_FAMILIES
                    ),
                    "native_routes_require_authoritative_route_binding": True,
                    "adapted_routes_require_signed_route_level_adapter_receipt": True,
                }
            ),
            "provider_rollout_receipt_claim_recorded_when_present": True,
            "provider_rollout_receipt_recomputed": False,
            "codex_turn_receipt_reopened": True,
            "export_later_stages": False,
            "export_terminal_answer": False,
            "export_hidden_reasoning": False,
            "export_private_reference": False,
        },
        **v3_manifest,
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
        "private_reference_included": False,
    }
    manifest = {**manifest_core, "manifest_blake3": blake3_hex(manifest_core)}
    _write_once(dataset_root / "manifest.json", canonical_json_bytes(manifest))
    os.chmod(shards, 0o555)
    os.chmod(dataset_root, 0o555)
    return FrontierPrefixSFTBuild(
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        source_count=len(rows),
        slice_count=len(rows),
        manifest_blake3=manifest["manifest_blake3"],
    )


def verify_frontier_prefix_sft_dataset(
    *,
    dataset_root: Path,
    source_base: Path,
    registry: CompiledRubricRegistry,
    route_authorities: Mapping[str, CodexProviderRoute] | None = None,
) -> Mapping[str, Any]:
    """Independently rebuild the accepted prefix and reopen every commitment."""

    try:
        root = dataset_root.resolve(strict=True)
        base = source_base.resolve(strict=True)
        manifest, manifest_bytes = _canonical_json(root / "manifest.json")
        manifest_schema = manifest.get("schema")
        if manifest_schema == FRONTIER_PREFIX_DATASET_SCHEMA:
            schema_version = 2
            slice_schema = FRONTIER_PREFIX_SLICE_SCHEMA
            verify_schema = FRONTIER_PREFIX_VERIFY_SCHEMA
        elif manifest_schema == FRONTIER_PREFIX_DATASET_SCHEMA_V3:
            schema_version = 3
            slice_schema = FRONTIER_PREFIX_SLICE_SCHEMA_V3
            verify_schema = FRONTIER_PREFIX_VERIFY_SCHEMA_V3
        else:
            raise FrontierPrefixSFTError("frontier-prefix dataset schema differs")
        dataset_id = _uuid(manifest.get("dataset_id"), label="dataset_id")
        core = {key: value for key, value in manifest.items() if key != "manifest_blake3"}
        selection = manifest.get("selection")
        if (
            root.name != dataset_id
            or manifest.get("schema") != manifest_schema
            or manifest.get("quality_tier") != "execution_verified_frontier_prefix"
            or manifest.get("selection_status")
            != "stage_prefix_verified_full_trajectory_not_claimed"
            or manifest.get("manifest_blake3") != blake3_hex(core)
            or manifest.get("rubric_registry_digest") != registry.digest
            or type(manifest.get("source_count")) is not int
            or manifest.get("source_count") < 1
            or manifest.get("slice_count") != manifest.get("source_count")
            or manifest.get("agent_judged") is not False
            or manifest.get("strict_full_trajectory_sft_eligible") is not False
            or manifest.get("hidden_reasoning_included") is not False
            or manifest.get("private_reference_included") is not False
            or not isinstance(selection, Mapping)
            or selection.get("provider_rollout_receipt_claim_recorded_when_present")
            is not True
            or selection.get("provider_rollout_receipt_recomputed") is not False
            or selection.get("codex_turn_receipt_reopened") is not True
        ):
            raise FrontierPrefixSFTError("frontier-prefix manifest identity differs")
        if schema_version == 2:
            if selection.get("adapter_evidence_scope") != "route_batch_unbound":
                raise FrontierPrefixSFTError("frontier-prefix v2 provider evidence differs")
        else:
            expected_selection = {
                "native_provider_families": ["glm", "openai"],
                "adapted_provider_families": sorted(_ADAPTED_PROVIDER_FAMILIES),
                "native_routes_require_authoritative_route_binding": True,
                "adapted_routes_require_signed_route_level_adapter_receipt": True,
            }
            if any(selection.get(key) != value for key, value in expected_selection.items()):
                raise FrontierPrefixSFTError("frontier-prefix v3 provider policy differs")
            authority_rows = manifest.get("route_authorities")
            if not isinstance(authority_rows, list) or not authority_rows:
                raise FrontierPrefixSFTError("frontier-prefix v3 route authorities differ")
            expected_authorities = _route_authority_document(
                [str(item.get("route_id")) for item in authority_rows if isinstance(item, Mapping)],
                routes=route_authorities,
            )
            if canonical_value(authority_rows) != canonical_value(expected_authorities):
                raise FrontierPrefixSFTError("frontier-prefix v3 route authorities differ")
        shards = manifest.get("shards")
        if not isinstance(shards, list) or len(shards) != 1:
            raise FrontierPrefixSFTError("frontier-prefix shard inventory differs")
        descriptor = shards[0]
        shard_relative = _safe_relative(descriptor.get("path"), label="SFT shard")
        if shard_relative != "shards/part-00000.jsonl":
            raise FrontierPrefixSFTError("frontier-prefix shard path differs")
        shard_path = root.joinpath(*PurePosixPath(shard_relative).parts)
        shard_payload = shard_path.read_bytes()
        if (
            len(shard_payload) != descriptor.get("byte_count")
            or blake3_bytes(shard_payload) != descriptor.get("content_blake3")
        ):
            raise FrontierPrefixSFTError("frontier-prefix shard commitment differs")
        lines = shard_payload.splitlines(keepends=True)
        if (
            len(lines) != manifest["slice_count"]
            or descriptor.get("slice_count") != len(lines)
            or any(canonical_json_bytes(json.loads(line)) != line for line in lines)
        ):
            raise FrontierPrefixSFTError("frontier-prefix shard JSON differs")
        rows = [json.loads(line, object_pairs_hook=_strict_object) for line in lines]
        if any(not isinstance(row, Mapping) for row in rows):
            raise FrontierPrefixSFTError("frontier-prefix slice differs")
        example_ids: list[str] = []
        for row in rows:
            row_core = {key: value for key, value in row.items() if key != "slice_blake3"}
            example_id = _uuid(row.get("example_id"), label="example_id")
            example_ids.append(example_id)
            if (
                row.get("schema") != slice_schema
                or row.get("dataset_id") != dataset_id
                or row.get("slice_blake3") != blake3_hex(row_core)
                or row.get("quality_tier") != "execution_verified_frontier_prefix"
                or row.get("provider_rollout_receipt_recomputed") is not False
                or row.get("codex_turn_receipt_reopened") is not True
            ):
                raise FrontierPrefixSFTError("frontier-prefix slice commitment differs")
            if schema_version == 2:
                if row.get("adapter_evidence_scope") != "route_batch_unbound":
                    raise FrontierPrefixSFTError("frontier-prefix v2 slice provider evidence differs")
            elif row.get("provider_evidence_scope") not in {
                "native_codex_direct",
                "adapted_route_batch_unbound",
            }:
                raise FrontierPrefixSFTError("frontier-prefix v3 slice provider evidence differs")
        if (
            descriptor.get("first_example_id") != example_ids[0]
            or descriptor.get("last_example_id") != example_ids[-1]
            or len(example_ids) != len(set(example_ids))
        ):
            raise FrontierPrefixSFTError("frontier-prefix example inventory differs")
        source_roots_value = manifest.get("source_roots")
        if not isinstance(source_roots_value, list) or not source_roots_value:
            raise FrontierPrefixSFTError("frontier-prefix teacher roots differ")
        source_roots = {
            relative: base.joinpath(*PurePosixPath(relative).parts)
            for relative in (
                _safe_relative(value, label="source root")
                for value in source_roots_value
            )
        }
        if len(source_roots) != len(source_roots_value):
            raise FrontierPrefixSFTError("frontier-prefix teacher roots are duplicated")
        bulk_root = base.joinpath(
            *PurePosixPath(
                _safe_relative(manifest.get("bulk_root"), label="bulk root")
            ).parts
        )
        source_inventory = manifest.get("source_error_inventory")
        inventory_summary = manifest.get("inventory")
        if not isinstance(source_inventory, list) or not isinstance(
            inventory_summary, Mapping
        ):
            raise FrontierPrefixSFTError("frontier-prefix source inventory differs")
        inventory_paths: list[Path] = []
        inventory_roots: list[Path] = []
        sandbox_ids: list[str] = []
        for item in source_inventory:
            if not isinstance(item, Mapping):
                raise FrontierPrefixSFTError("frontier-prefix source inventory row differs")
            relative = _safe_relative(item.get("path"), label="teacher source")
            path = base.joinpath(*PurePosixPath(relative).parts)
            source_root_relative = _safe_relative(
                item.get("source_root"), label="source root"
            )
            source_root = source_roots.get(source_root_relative)
            if source_root is None or source_root not in path.parents:
                raise FrontierPrefixSFTError("frontier-prefix source root binding differs")
            _document, payload = _canonical_json(path)
            sandbox_id = _uuid(item.get("sandbox_id"), label="sandbox_id")
            if blake3_bytes(payload) != item.get("file_blake3"):
                raise FrontierPrefixSFTError("frontier-prefix failure inventory changed")
            inventory_paths.append(path)
            inventory_roots.append(source_root)
            sandbox_ids.append(sandbox_id)
        if {
            _safe_relative(item.get("source_root"), label="source root")
            for item in source_inventory
        } != set(source_roots):
            raise FrontierPrefixSFTError("frontier-prefix teacher-root inventory differs")
        if schema_version == 3 and canonical_value(manifest.get("route_authorities")) != canonical_value(
            _route_authority_document(
                [str(item.get("route_id")) for item in source_inventory],
                routes=route_authorities,
            )
        ):
            raise FrontierPrefixSFTError("frontier-prefix v3 source route authorities differ")
        bulk_records = _load_bulk_records(
            bulk_root, tuple(sorted(set(sandbox_ids)))
        )
        observed_row_by_source = {
            str(row.get("source_record_path")): row for row in rows
        }
        if len(observed_row_by_source) != len(rows):
            raise FrontierPrefixSFTError("frontier-prefix source was duplicated")
        expected_by_source: dict[str, Mapping[str, Any]] = {}
        rejected: dict[str, int] = {}
        for source_root, source_path, sandbox_id in zip(
            inventory_roots, inventory_paths, sandbox_ids, strict=True
        ):
            try:
                source_relative = _relative(
                    source_path, base=base, label="teacher source"
                )
                observed_row = observed_row_by_source.get(source_relative)
                expected = _source_slice(
                    source_root=source_root,
                    source_base=base,
                    bulk_root=bulk_root,
                    registry=registry,
                    source_document_path=source_path,
                    bulk_record=bulk_records[sandbox_id],
                    signed_route_level_adapter_receipt_blake3=(
                        None
                        if observed_row is None
                        or observed_row.get(
                            "signed_route_level_adapter_receipt_blake3"
                        )
                        is None
                        else str(
                            observed_row.get(
                                "signed_route_level_adapter_receipt_blake3"
                            )
                        )
                    ),
                    schema_version=schema_version,
                    route_authorities=route_authorities,
                )
                expected_by_source[str(expected["source_record_path"])] = expected
            except FrontierPrefixSFTError as exc:
                reason = str(exc)
                rejected[reason] = rejected.get(reason, 0) + 1
        if (
            len(expected_by_source) != manifest["slice_count"]
            or inventory_summary.get("scanned_terminal_teacher_rows")
            != len(source_inventory)
            or inventory_summary.get("accepted_s1_prefixes") != len(rows)
            or inventory_summary.get("rejected_rows")
            != len(source_inventory) - len(rows)
            or inventory_summary.get("rejected_by_reason") != dict(sorted(rejected.items()))
        ):
            raise FrontierPrefixSFTError("frontier-prefix acceptance inventory differs")
        observed_paths: set[str] = set()
        route_counts: dict[str, int] = {}
        scope_counts: dict[str, int] = {}
        for row in rows:
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
            source_relative = str(observed.get("source_record_path"))
            expected = expected_by_source.get(source_relative)
            if expected is None or canonical_value(observed) != canonical_value(expected):
                raise FrontierPrefixSFTError("frontier-prefix source reconstruction differs")
            if source_relative in observed_paths:
                raise FrontierPrefixSFTError("frontier-prefix source was duplicated")
            observed_paths.add(source_relative)
            route = str(row["route_id"])
            route_counts[route] = route_counts.get(route, 0) + 1
            if schema_version == 3:
                scope = str(row["provider_evidence_scope"])
                scope_counts[scope] = scope_counts.get(scope, 0) + 1
        if (
            observed_paths != set(expected_by_source)
            or manifest.get("counts_by_stage") != {"S1": len(rows)}
            or manifest.get("counts_by_route") != dict(sorted(route_counts.items()))
            or (
                schema_version == 3
                and manifest.get("counts_by_provider_evidence_scope")
                != dict(sorted(scope_counts.items()))
            )
        ):
            raise FrontierPrefixSFTError("frontier-prefix aggregate counts differ")
        expected_files = {"manifest.json", "shards/part-00000.jsonl"}
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise FrontierPrefixSFTError("frontier-prefix dataset contains an extra file")
        provider_checks = (
            [
                "signed_route_level_adapter_receipts",
                "provider_rollout_receipt_claim_recorded_not_recomputed",
            ]
            if schema_version == 2
            else [
                "route_authority_provider_evidence_scopes",
                "signed_route_level_adapter_receipts_when_adapted",
                "provider_rollout_receipt_claim_recorded_not_recomputed",
            ]
        )
        report_core = {
            "schema": verify_schema,
            "dataset_id": dataset_id,
            "valid": True,
            "source_count": len(rows),
            "slice_count": len(rows),
            "manifest_file_blake3": blake3_bytes(manifest_bytes),
            "manifest_blake3": manifest["manifest_blake3"],
            "checks": [
                "later_failed_turn_reopened",
                "completed_codex_actor_receipt",
                *provider_checks,
                "s1_first_frontier_exact",
                "host_gate_passed_and_no_failed_checks",
                "declared_artifact_exact",
                "workspace_before_after_reconstructed",
                "rubric_table_exact",
                "observable_prefix_only",
                "manifest_and_shard_commitments",
            ],
        }
        return {**report_core, "verification_blake3": blake3_hex(report_core)}
    except (FrontierPrefixSFTError, OSError, ValueError, TypeError, KeyError) as exc:
        return {
            "schema": locals().get("verify_schema", FRONTIER_PREFIX_VERIFY_SCHEMA_V3),
            "valid": False,
            "error": str(exc),
        }


__all__ = [
    "FRONTIER_PREFIX_DATASET_SCHEMA",
    "FRONTIER_PREFIX_SLICE_SCHEMA",
    "FRONTIER_PREFIX_VERIFY_SCHEMA",
    "FrontierPrefixSFTBuild",
    "FrontierPrefixSFTError",
    "build_frontier_prefix_sft_dataset",
    "verify_frontier_prefix_sft_dataset",
]
