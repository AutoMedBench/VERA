"""Independent reopening verifier for pipeline results and immutable evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
import json
from types import SimpleNamespace
from typing import Any, Mapping

from eva_agent.codex_runtime.contracts import (
    codex_core_mcp_resource_call_error,
    codex_core_mcp_resource_operation,
)

from .adapters import WeightedRubricRewarder
from .artifacts import ImmutableArtifactStore
from .codec import load_pipeline_result
from .contracts import (
    COHORT_ORDER,
    Cohort,
    CompiledRubricTable,
    ContractError,
    ModelEvaluation,
    PipelineResult,
    RewardComputer,
    ToolCall,
    uuid_text,
)
from .digests import blake3_bytes, blake3_hex, canonical_json_bytes, canonical_value, is_blake3
from .ids import DeterministicUUIDFactory
from .judge_workspace import JudgeWorkspaceError, JudgeWorkspaceTools
from .runner import (
    SeparationPolicy,
    evaluation_core,
    evaluation_manifest_document,
    evidence_core,
    partial_cohort_outcome_document,
    pipeline_result_document,
    recommendation_core,
    result_core,
    separation_core,
    telemetry_core,
)


class PipelineVerificationError(ContractError):
    """Independent verification found a mismatch."""


_EARLY_COHORT_CANCELLATION_NAME = "_EarlyCohortCancellation"


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]
    result_blake3: str
    report_blake3: str


def _workspace_core(snapshot: Any) -> dict[str, Any]:
    return {
        "files": snapshot.files,
        "file_count": snapshot.file_count,
        "byte_count": snapshot.byte_count,
    }


def _tool_result_core(result: Any) -> dict[str, Any]:
    return {
        "result_id": result.result_id,
        "call_id": result.call_id,
        "name": result.name,
        "frontier": result.frontier,
        "parallel_group_id": result.parallel_group_id,
        "status": result.status,
        "output": result.output,
        "error_code": result.error_code,
        "workspace_before_blake3": result.workspace_before_blake3,
        "workspace_after_blake3": result.workspace_after_blake3,
    }


def _codex_tool_groups(receipt: Any) -> tuple[tuple[Any, ...], ...]:
    """Independently recover the app-server's observable call frontiers."""

    if not receipt.tool_calls:
        return ()
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    for event in receipt.events:
        if event.method not in {"item/started", "item/completed"}:
            continue
        payload = canonical_value(event.payload)
        item = payload.get("item") if isinstance(payload, dict) else None
        item_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(item_id, str):
            continue
        target = starts if event.method == "item/started" else ends
        if item_id in target:
            raise PipelineVerificationError("Codex tool lifecycle identity was reused")
        target[item_id] = event.sequence
    intervals: list[tuple[int, int, Any]] = []
    seen_calls: set[str] = set()
    seen_items: set[str] = set()
    for call in receipt.tool_calls:
        if call.tool_call_id in seen_calls or call.upstream_item_id in seen_items:
            raise PipelineVerificationError("Codex receipt tool identity was reused")
        seen_calls.add(call.tool_call_id)
        seen_items.add(call.upstream_item_id)
        # Older/pinned app-server receipts may expose an MCP tool only at
        # ``item/completed``.  The runtime contract records the first observed
        # event sequence on the call for precisely that completion-only form.
        # Native actions are still required to carry both lifecycle events by
        # the independent native-action verifier below.
        start = starts.get(call.upstream_item_id, call.first_event_sequence)
        end = ends.get(call.upstream_item_id)
        if end is None or end < start or call.first_event_sequence != start:
            raise PipelineVerificationError("Codex tool lifecycle interval differs")
        intervals.append((start, end, call))
    intervals.sort(key=lambda row: (row[0], row[1], row[2].tool_call_id))
    groups: list[tuple[Any, ...]] = []
    cursor = 0
    while cursor < len(intervals):
        _start, cutoff, first = intervals[cursor]
        calls = [first]
        cursor += 1
        while cursor < len(intervals) and intervals[cursor][0] < cutoff:
            _next_start, end, call = intervals[cursor]
            calls.append(call)
            cutoff = min(cutoff, end)
            cursor += 1
        groups.append(tuple(calls))
    if max(map(len, groups)) != receipt.max_parallelism_observed:
        raise PipelineVerificationError("Codex receipt parallelism differs")
    return tuple(groups)


def _verify_codex_core_resource_call(
    call: Any, operation: str, offered_names: set[str]
) -> None:
    error = codex_core_mcp_resource_call_error(
        call=call,
        operation=operation,
        offered_mcp_tool_names=offered_names,
    )
    if error is not None:
        raise PipelineVerificationError(error)


def _trace_core(trace: Any) -> dict[str, Any]:
    return {
        "results": trace.results,
        "declared_call_ids": trace.declared_call_ids,
        "joined_call_ids": trace.joined_call_ids,
        "frontier_count": trace.frontier_count,
        "max_parallelism_observed": trace.max_parallelism_observed,
        "retry_count": trace.retry_count,
    }


_CODEX_PROJECTION_KEYS = frozenset({
    "schema", "codex_turn_receipt", "tool_call_groups", "codex_to_pipeline_call_ids",
    "raw_input_recorded", "semantic_retry_count", "controlled_budget_termination",
    "workspace_terminal", "semantic_tool_gate_failure", "native_file_change_effect_binding",
    "schema_validation_rejections", "schema_validation_rejection_catalog",
    "legacy_empty_code_rejections",
})


def _codex_schema_rejections(receipt: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen exact public schemas and rejected calls, never invent ToolResults."""
    rows = metadata.get("schema_validation_rejections")
    catalog = metadata.get("schema_validation_rejection_catalog")
    if rows is None and catalog is None:
        return {}
    from jsonschema import Draft202012Validator
    from eva_agent.codex_runtime import CodexToolOffer

    if (not isinstance(rows, (tuple, list)) or not rows
            or not isinstance(catalog, (tuple, list)) or not catalog
            or len(catalog) != len(receipt.offered_mcp_tool_names)
            or blake3_hex(catalog) != receipt.offered_tool_schema_blake3):
        raise PipelineVerificationError("Codex rejection catalogue binding differs")
    offers = {}
    for name, row in zip(receipt.offered_mcp_tool_names, catalog, strict=True):
        if (not isinstance(row, Mapping) or set(row) != {"name", "description", "input_schema"}
                or name.rsplit("/", 1)[-1] != row["name"] or name in offers):
            raise PipelineVerificationError("Codex rejection offer differs")
        try:
            offer = CodexToolOffer(fully_qualified_name=name, description=row["description"],
                                   input_schema=row["input_schema"])
            Draft202012Validator.check_schema(canonical_value(offer.input_schema))
        except Exception as exc:
            raise PipelineVerificationError("Codex rejection schema differs") from exc
        offers[name] = offer
    calls = {call.tool_call_id: call for call in receipt.tool_calls}
    rejected, rejection_ids = {}, set()
    keys = {"schema", "rejection_id", "fully_qualified_name", "arguments_blake3",
        "input_schema_blake3", "offered_catalog_blake3", "validation_issues",
        "tool_trace_blake3_at_rejection", "host_execution_attempted", "host_result_claimed",
        "rejection_blake3"}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
                "codex_tool_call_id", "codex_tool_call_receipt_blake3", "rejection"}:
            raise PipelineVerificationError("Codex rejection metadata differs")
        call = calls.get(row["codex_tool_call_id"])
        proof = canonical_value(row["rejection"])
        if (call is None or call.tool_call_id in rejected or call.tool_type != "mcpToolCall"
                or call.status not in {"completed", "failed"}
                or call.fully_qualified_name not in offers or not isinstance(call.arguments, Mapping)
                or row["codex_tool_call_receipt_blake3"] != call.receipt_blake3
                or not isinstance(proof, dict) or set(proof) != keys):
            raise PipelineVerificationError("Codex rejected-call identity differs")
        uuid_text(proof["rejection_id"], label="schema rejection_id")
        offer = offers[call.fully_qualified_name]
        issues = []
        for error in Draft202012Validator(canonical_value(offer.input_schema)).iter_errors(
                canonical_value(call.arguments)):
            issues.append({"validator": str(error.validator),
                "instance_path": list(error.absolute_path), "schema_path": list(error.absolute_schema_path),
                "required_properties": [value for value in error.validator_value
                    if isinstance(value, str) and value not in error.instance]
                    if error.validator == "required" and isinstance(error.validator_value, list)
                    and isinstance(error.instance, Mapping) else [],
                "expected_types": [value for value in (
                    [error.validator_value] if isinstance(error.validator_value, str)
                    else error.validator_value if isinstance(error.validator_value, list) else [])
                    if isinstance(value, str)] if error.validator == "type" else []})
        issues.sort(key=lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False,
                                                sort_keys=True, separators=(",", ":")))
        output = canonical_value(call.output)
        result = output.get("result") if isinstance(output, dict) else None
        if (not issues or proof["validation_issues"] != issues
                or proof["schema"] != "eva.codex-tool-schema-validation-rejection.v1"
                or proof["rejection_id"] in rejection_ids
                or proof["fully_qualified_name"] != call.fully_qualified_name
                or proof["arguments_blake3"] != blake3_hex(call.arguments)
                or proof["input_schema_blake3"] != blake3_hex(offer.input_schema)
                or proof["offered_catalog_blake3"] != receipt.offered_tool_schema_blake3
                or not is_blake3(proof["tool_trace_blake3_at_rejection"])
                or proof["host_execution_attempted"] is not False
                or proof["host_result_claimed"] is not False
                or proof["rejection_blake3"] != blake3_hex({k: v for k, v in proof.items()
                                                          if k != "rejection_blake3"})
                or not isinstance(result, dict) or output.get("error") is not None
                or not (result.get("isError") is True or (
                    "isError" not in result and call.status == "failed"))
                or result.get("structuredContent") != proof):
            raise PipelineVerificationError("Codex schema rejection proof differs")
        rejected[call.tool_call_id] = proof
        rejection_ids.add(proof["rejection_id"])
    # No unclaimed error-result can masquerade as a successful dispatched call.
    for call in receipt.tool_calls:
        output = canonical_value(call.output)
        result = output.get("result") if isinstance(output, dict) else None
        structured = result.get("structuredContent") if isinstance(result, dict) else None
        if (call.fully_qualified_name in offers and isinstance(result, dict)
                and (result.get("isError") is True or (isinstance(structured, dict)
                    and structured.get("schema") == "eva.codex-tool-schema-validation-rejection.v1"))
                and call.tool_call_id not in rejected):
            raise PipelineVerificationError("Codex rejection inventory is incomplete")
    return rejected


def verify_codex_judge_source(evidence: Any) -> None:
    """Verify native receipt/events before Judge-only reconstruction/materials.

    Teacher archives rebuild manifest/rollout identities for the Judge. Therefore
    this check does not pretend to recompute their original provider-core digest;
    the normal PipelineVerifier still requires that complete binding. Training
    telemetry is separate from the strict adapter projection metadata below.
    """
    metadata = evidence.safe_provider_metadata
    from eva_agent.codex_pipeline.budget import PROJECTION, WORKSPACE_PROJECTION
    if metadata.get("schema") not in {"eva.codex-provider-rollout-projection.v1",
            "eva.codex-provider-rollout-projection.v2-native-file-change", PROJECTION,
            WORKSPACE_PROJECTION}:
        raise PipelineVerificationError("Judge source is not a supported native projection")
    projected = replace(evidence, safe_provider_metadata={
        key: value for key, value in metadata.items() if key in _CODEX_PROJECTION_KEYS})
    verifier = PipelineVerifier(None)
    verifier._verify_trajectory(SimpleNamespace(evidence=projected, cohort=projected.model.cohort),
                               check_provider_binding=False)


def _assessment_core(value: Any) -> dict[str, Any]:
    return {
        "judgment_id": value.judgment_id,
        "judge_model_id": value.judge_model_id,
        "rubric_digest": value.rubric_digest,
        "agent_trace": value.agent_trace,
        "item_scores": value.item_scores,
        "hard_gates_passed": value.hard_gates_passed,
        "summary": value.summary,
    }


def _judge_tool_result_core(value: Any) -> dict[str, Any]:
    return {
        "result_id": value.result_id,
        "call_id": value.call_id,
        "name": value.name,
        "arguments": value.arguments,
        "frontier": value.frontier,
        "parallel_group_id": value.parallel_group_id,
        "status": value.status,
        "output": value.output,
        "error_code": value.error_code,
        "inspected_evidence_refs": value.inspected_evidence_refs,
        "content_inspection": value.content_inspection,
        "evidence_bundle_blake3": value.evidence_bundle_blake3,
    }


def _judge_trace_core(value: Any) -> dict[str, Any]:
    return {
        "policy_events": value.policy_events,
        "results": value.results,
        "declared_call_ids": value.declared_call_ids,
        "joined_call_ids": value.joined_call_ids,
        "inspected_evidence_refs": value.inspected_evidence_refs,
        "content_inspection_count": value.content_inspection_count,
        "frontier_count": value.frontier_count,
        "max_parallelism_observed": value.max_parallelism_observed,
        "provider_turn_count": value.provider_turn_count,
        "retry_count": value.retry_count,
        "evidence_bundle_blake3": value.evidence_bundle_blake3,
    }


def _reward_core(value: Any) -> dict[str, Any]:
    return {
        "reward_id": value.reward_id,
        "rubric_digest": value.rubric_digest,
        "item_scores": value.item_scores,
        "total_reward": value.total_reward,
        "hard_gates_passed": value.hard_gates_passed,
    }


def _manifest_core(manifest: Any) -> dict[str, Any]:
    return {
        "sandbox_id": manifest.sandbox_id,
        "episode_id": manifest.episode_id,
        "source": manifest.source,
        "domain": manifest.domain,
        "stage": manifest.stage,
        "instruction": manifest.instruction,
        "policy_context": manifest.policy_context,
        "initial_files": manifest.initial_files,
        "rubric": manifest.rubric,
        "rubric_table_count": 1,
    }


def _artifact_payload(store: ImmutableArtifactStore, relative_path: str) -> bytes:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise PipelineVerificationError("artifact path is unsafe")
    return store.root.joinpath(*pure.parts).read_bytes()


class PipelineVerifier:
    """Recompute every local commitment without trusting the producer."""

    def __init__(
        self,
        artifact_store: ImmutableArtifactStore,
        *,
        separation_policy: SeparationPolicy | None = None,
        rewarder: RewardComputer | None = None,
    ) -> None:
        self._artifacts = artifact_store
        self._policy = separation_policy or SeparationPolicy()
        self._rewarder = rewarder or WeightedRubricRewarder()

    def _verify_workspace(self, snapshot: Any) -> None:
        if snapshot.file_count != len(snapshot.files):
            raise PipelineVerificationError("workspace file count differs")
        if snapshot.byte_count != sum(row.byte_count for row in snapshot.files):
            raise PipelineVerificationError("workspace byte count differs")
        paths: set[str] = set()
        for row in snapshot.files:
            pure = PurePosixPath(row.path)
            if (
                pure.is_absolute()
                or pure.as_posix() != row.path
                or any(part in {"", ".", ".."} for part in pure.parts)
                or row.path in paths
            ):
                raise PipelineVerificationError("workspace file path differs")
            paths.add(row.path)
            if (
                not isinstance(row.content, bytes)
                or row.byte_count != len(row.content)
                or row.content_blake3 != blake3_bytes(row.content)
            ):
                raise PipelineVerificationError("workspace file content commitment differs")
        if snapshot.tree_blake3 != blake3_hex(_workspace_core(snapshot)):
            raise PipelineVerificationError("workspace tree commitment differs")

    @staticmethod
    def _persisted_native_path(value: Any, *, allow_dot: bool = False) -> str:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise PipelineVerificationError("Codex native path differs")
        pure = PurePosixPath(value)
        if value == "." and allow_dot:
            return value
        if (
            pure.is_absolute()
            or pure.as_posix() != value
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise PipelineVerificationError("Codex native path is not workspace-relative")
        return value

    def _verify_persisted_native_call(
        self, evaluation: ModelEvaluation, call: Any
    ) -> None:
        evidence = evaluation.evidence
        if (
            evidence.sandbox_manifest.stage.value not in {"S3", "S4", "E2E"}
            or call.tool_type != "fileChange"
            or call.name != "fileChange"
            or call.status != "completed"
            or call.lifecycle != ("item/started", "item/completed")
            or any(
                value is not None
                for value in (call.mcp_server, call.mcp_tool, call.fully_qualified_name)
            )
        ):
            raise PipelineVerificationError("Codex native action authorization differs")
        arguments = canonical_value(call.arguments)
        output = canonical_value(call.output)
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"changes"}
            or not isinstance(arguments["changes"], list)
            or not arguments["changes"]
            or output != {"status": "completed"}
        ):
            raise PipelineVerificationError("Codex native file-change shape differs")
        for change in arguments["changes"]:
            if (
                not isinstance(change, dict)
                or set(change) != {"diff", "kind", "path"}
                or not isinstance(change["diff"], str)
                or not change["diff"]
                or not isinstance(change["kind"], dict)
            ):
                raise PipelineVerificationError("Codex native file-change entry differs")
            self._persisted_native_path(change["path"])
            kind = change["kind"]
            kind_type = kind.get("type")
            if kind_type in {"add", "delete"}:
                if set(kind) != {"type"}:
                    raise PipelineVerificationError("Codex native file-change kind differs")
            elif kind_type == "update":
                if set(kind) not in (
                    {"type"},
                    {"type", "move_path"},
                    {"type", "movePath"},
                ):
                    raise PipelineVerificationError("Codex native update shape differs")
                move = kind.get("move_path", kind.get("movePath"))
                if move is not None:
                    self._persisted_native_path(move)
            else:
                raise PipelineVerificationError("Codex native file-change kind differs")

    @staticmethod
    def _persisted_native_file_state(row: Any | None) -> Mapping[str, Any] | None:
        if row is None:
            return None
        return {
            "content_blake3": row.content_blake3,
            "byte_count": row.byte_count,
            "mode": row.mode,
        }

    def _expected_native_file_change_binding(
        self,
        evaluation: ModelEvaluation,
        native_calls: tuple[Any, ...],
    ) -> Mapping[str, Any]:
        evidence = evaluation.evidence
        before = {row.path: row for row in evidence.workspace_before.files}
        after = {row.path: row for row in evidence.workspace_after.files}
        state = self._persisted_native_file_state
        actual_paths = tuple(
            sorted(
                path
                for path in set(before) | set(after)
                if state(before.get(path)) != state(after.get(path))
            )
        )
        declared_paths: set[str] = set()
        effects: list[Mapping[str, Any]] = []
        for call in native_calls:
            arguments = canonical_value(call.arguments)
            if not isinstance(arguments, dict) or not isinstance(
                arguments.get("changes"), list
            ):
                raise PipelineVerificationError("native effect call shape differs")
            for change_index, change in enumerate(arguments["changes"]):
                if not isinstance(change, dict) or not isinstance(change.get("kind"), dict):
                    raise PipelineVerificationError("native effect change shape differs")
                source_path = self._persisted_native_path(change.get("path"))
                kind = change["kind"].get("type")
                raw_destination = change["kind"].get(
                    "move_path", change["kind"].get("movePath")
                )
                destination_path = (
                    None
                    if raw_destination is None
                    else self._persisted_native_path(raw_destination)
                )
                paths = (source_path,) if destination_path is None else (
                    source_path,
                    destination_path,
                )
                if any(path in declared_paths for path in paths):
                    raise PipelineVerificationError(
                        "native effect path was declared more than once"
                    )
                declared_paths.update(paths)
                source_before = state(before.get(source_path))
                source_after = state(after.get(source_path))
                destination_before = (
                    None if destination_path is None else state(before.get(destination_path))
                )
                destination_after = (
                    None if destination_path is None else state(after.get(destination_path))
                )
                if kind == "add":
                    valid = (
                        destination_path is None
                        and source_before is None
                        and source_after is not None
                    )
                elif kind == "delete":
                    valid = (
                        destination_path is None
                        and source_before is not None
                        and source_after is None
                    )
                elif kind == "update" and destination_path is None:
                    valid = (
                        source_before is not None
                        and source_after is not None
                        and source_before != source_after
                    )
                elif kind == "update":
                    valid = (
                        source_before is not None
                        and source_after is None
                        and destination_before is None
                        and destination_after is not None
                    )
                else:
                    valid = False
                if not valid:
                    raise PipelineVerificationError(
                        "native declared effect differs from workspace evidence"
                    )
                try:
                    diff_blake3 = blake3_bytes(change["diff"].encode("utf-8"))
                except (KeyError, AttributeError, UnicodeEncodeError):
                    raise PipelineVerificationError(
                        "native effect diff is not canonical UTF-8"
                    ) from None
                path_effects = tuple(
                    {
                        "path": path,
                        "before": state(before.get(path)),
                        "after": state(after.get(path)),
                    }
                    for path in paths
                )
                effect_core = {
                    "tool_call_id": call.tool_call_id,
                    "upstream_item_id": call.upstream_item_id,
                    "change_index": change_index,
                    "kind": kind,
                    "source_path": source_path,
                    "destination_path": destination_path,
                    "diff_blake3": diff_blake3,
                    "path_effects": path_effects,
                }
                effects.append(
                    {**effect_core, "effect_blake3": blake3_hex(effect_core)}
                )
        declared = tuple(sorted(declared_paths))
        if declared != actual_paths:
            raise PipelineVerificationError(
                "native declared paths and actual workspace effects differ"
            )
        binding_core = {
            "schema": "eva.codex-native-file-change-effect-binding.v1",
            "workspace_before_tree_blake3": evidence.workspace_before.tree_blake3,
            "workspace_after_tree_blake3": evidence.workspace_after.tree_blake3,
            "declared_changed_paths": declared,
            "actual_changed_paths": actual_paths,
            "effects": tuple(effects),
        }
        return {**binding_core, "binding_blake3": blake3_hex(binding_core)}

    def _verify_codex_actor_projection(self, evaluation: ModelEvaluation, *,
                                       check_provider_binding: bool = True) -> tuple[str, ...]:
        """Reopen the full receipt and bind native sidecars to visible events."""

        evidence = evaluation.evidence
        metadata = evidence.safe_provider_metadata
        projection_schema = metadata.get("schema")
        projection_v1 = "eva.codex-provider-rollout-projection.v1"
        projection_v2 = (
            "eva.codex-provider-rollout-projection.v2-native-file-change"
        )
        from eva_agent.codex_pipeline.budget import (PROJECTION, TERMINAL_MARKER, budget_metadata,
            WORKSPACE_PROJECTION, WORKSPACE_TERMINAL_MARKER, WORKSPACE_TERMINAL)
        if projection_schema not in {projection_v1, projection_v2, PROJECTION, WORKSPACE_PROJECTION}:
            return ()
        expected_metadata_keys = {
            "schema",
            "codex_turn_receipt",
            "tool_call_groups",
            "codex_to_pipeline_call_ids",
            "raw_input_recorded",
            "semantic_retry_count",
        }
        semantic_tool_gate = metadata.get("semantic_tool_gate_failure")
        budget_terminal = metadata.get("controlled_budget_termination")
        workspace_terminal = metadata.get("workspace_terminal")
        if projection_schema == WORKSPACE_PROJECTION:
            expected_metadata_keys.add("workspace_terminal")
            if (canonical_value(workspace_terminal) != canonical_value(WORKSPACE_TERMINAL)
                    or semantic_tool_gate is not None or budget_terminal is not None):
                raise PipelineVerificationError("Codex workspace terminal metadata differs")
        if projection_schema == PROJECTION:
            expected_metadata_keys.add("controlled_budget_termination")
            if budget_terminal is None or semantic_tool_gate is not None:
                raise PipelineVerificationError("Codex budget projection lacks its explicit termination")
        if semantic_tool_gate is not None:
            expected_metadata_keys.add("semantic_tool_gate_failure")
        if "schema_validation_rejections" in metadata or "schema_validation_rejection_catalog" in metadata:
            expected_metadata_keys.update({"schema_validation_rejections", "schema_validation_rejection_catalog"})
        if "legacy_empty_code_rejections" in metadata:
            expected_metadata_keys.add("legacy_empty_code_rejections")
        if projection_schema == projection_v2:
            expected_metadata_keys.add("native_file_change_effect_binding")
        if set(metadata) != expected_metadata_keys or metadata[
            "raw_input_recorded"
        ] is not False or metadata[
            "semantic_retry_count"
        ] != 0:
            raise PipelineVerificationError("Codex rollout metadata differs")
        try:
            from eva_agent.codex_runtime.contracts import (
                CodexRole,
                CodexRuntimeError,
                CodexSandbox,
                codex_turn_receipt_from_document,
            )

            receipt = codex_turn_receipt_from_document(metadata["codex_turn_receipt"])
        except (CodexRuntimeError, TypeError) as exc:
            raise PipelineVerificationError("Codex receipt could not be reopened") from exc
        rejections = _codex_schema_rejections(receipt, metadata)
        expected_role = {
            Cohort.WEAK: CodexRole.WEAK_ACTOR,
            Cohort.MIDDLE: CodexRole.MIDDLE_ACTOR,
            Cohort.STRONG: CodexRole.STRONG_ACTOR,
        }[evaluation.cohort]
        final_messages = []
        for event in receipt.events:
            if event.method != "item/completed":
                continue
            payload = canonical_value(event.payload)
            item = payload.get("item") if isinstance(payload, dict) else None
            if (
                isinstance(item, dict)
                and item.get("type") == "agentMessage"
                and item.get("phase") in {"final_answer", "finalAnswer"}
                and isinstance(item.get("text"), str)
            ):
                final_messages.append(item["text"])
        ordinary_terminal = semantic_tool_gate is None and budget_terminal is None and workspace_terminal is None
        if (
            receipt.role is not expected_role
            or receipt.model != evidence.model.model_id
            or receipt.provider != evidence.model.provider
            or receipt.thread_resumed
            or (ordinary_terminal and receipt.status != "completed")
            or (
                ordinary_terminal
                and receipt.final_response != evidence.assistant_output
            )
            or (ordinary_terminal and not final_messages)
            or (
                ordinary_terminal
                and final_messages[-1] != evidence.assistant_output
            )
            or receipt.config_values_recorded
            or receipt.input_payload_recorded
        ):
            raise PipelineVerificationError("Codex actor receipt identity differs")

        groups = _codex_tool_groups(receipt)
        offered_names = set(receipt.offered_mcp_tool_names)
        mcp_calls = tuple(
            call
            for group in groups
            for call in group
            if call.tool_type == "mcpToolCall"
            and call.fully_qualified_name in offered_names
        )
        auxiliary_mcp_calls = tuple(
            call
            for group in groups
            for call in group
            if call.tool_type == "mcpToolCall"
            and call.fully_qualified_name not in offered_names
        )
        native_calls = tuple(
            call for group in groups for call in group if call.tool_type != "mcpToolCall"
        )
        native_capability = (
            isinstance(evidence.policy_events[0].content, str)
            and "This run explicitly permits sequential Codex-native fileChange actions"
            in evidence.policy_events[0].content
        )
        if projection_schema == projection_v2 and not native_capability:
            raise PipelineVerificationError(
                "Codex native v2 projection lacks its explicit capability"
            )
        if projection_schema in {projection_v1, PROJECTION, WORKSPACE_PROJECTION} and (native_capability or native_calls):
            raise PipelineVerificationError(
                "Codex native action lacks a v2 effect binding"
            )
        expected_sandbox = (
            CodexSandbox.WORKSPACE_WRITE
            if native_capability
            else CodexSandbox.READ_ONLY
        )
        if receipt.sandbox is not expected_sandbox:
            raise PipelineVerificationError("Codex actor sandbox capability differs")
        for call in auxiliary_mcp_calls:
            operation = codex_core_mcp_resource_operation(
                server=call.mcp_server,
                tool=call.mcp_tool,
                offered_mcp_tool_names=offered_names,
            )
            if operation is None:
                raise PipelineVerificationError("Codex invoked an unoffered MCP tool")
            _verify_codex_core_resource_call(call, operation, offered_names)
        mapping = metadata["codex_to_pipeline_call_ids"]
        if not isinstance(mapping, Mapping) or set(mapping) != {
            call.tool_call_id for call in mcp_calls if call.tool_call_id not in rejections
        } or any(not isinstance(value, str) for value in mapping.values()):
            raise PipelineVerificationError("Codex/pipeline call mapping differs")
        for value in mapping.values():
            uuid_text(value, label="Codex projected MCP call_id")
        if len(set(mapping.values())) != len(mapping):
            raise PipelineVerificationError("Codex/pipeline call mapping was reused")
        trace_by_id = {row.call_id: row for row in evidence.tool_trace.results}
        if set(trace_by_id) != set(mapping.values()):
            raise PipelineVerificationError("Codex MCP mapping and ToolTrace differ")
        if rejections:
            # Rejected IDs are deliberately absent; every *dispatched* call
            # still needs its exact original bridge receipt, not merely a
            # self-consistent replacement event/ToolResult.
            for call in mcp_calls:
                if call.tool_call_id in rejections:
                    continue
                output = canonical_value(call.output)
                result = output.get("result") if isinstance(output, dict) else None
                observed = result.get("structuredContent", result) if isinstance(result, dict) else None
                expected_result = trace_by_id[mapping[call.tool_call_id]]
                core = {"schema": "eva.codex-pipeline-tool-observation.v1",
                    "call_id": expected_result.call_id, "name": expected_result.name,
                    "arguments": canonical_value(call.arguments), "tool_result": canonical_value(expected_result)}
                expected = {**core, "bridge_receipt_blake3": blake3_hex(core)}
                if (not isinstance(result, dict) or output.get("error") is not None
                        or result.get("isError", False) is not False or observed != expected
                        or expected_result.name != call.mcp_tool):
                    raise PipelineVerificationError("Codex rejection-adjacent host result differs")
        from eva_agent.codex_pipeline.empty_code_rejection import (
            BOOTSTRAP_PATHS, empty_code_rejection_ids, semantic_tool_failure,
        )
        empty_code_ids = ()
        if "legacy_empty_code_rejections" in metadata:
            proof = metadata["legacy_empty_code_rejections"]
            if not isinstance(proof, Mapping) or set(proof) != {"call_ids", "offered_catalog"}:
                raise PipelineVerificationError("Codex legacy empty-code metadata differs")
            empty_code_ids = empty_code_rejection_ids(
                receipt=receipt, trace=evidence.tool_trace, mapping=mapping,
                catalog=proof["offered_catalog"], context=evidence.policy_visible_context,
                bootstrap={row.path: row.content for row in evidence.workspace_before.files
                           if row.path in BOOTSTRAP_PATHS},
            )
            if not empty_code_ids or canonical_value(empty_code_ids) != canonical_value(proof["call_ids"]):
                raise PipelineVerificationError("Codex legacy empty-code binding differs")
        if workspace_terminal is not None:
            if (receipt.status != "completed" or (receipt.final_response or "").strip()
                or any(value.strip() for value in final_messages)
                or evidence.assistant_output != WORKSPACE_TERMINAL_MARKER
                or evidence.tool_trace.retry_count != 0
                or any(row.status != "completed" and not (
                    semantic_tool_failure(row, empty_code_ids)
                ) for row in evidence.tool_trace.results)):
                raise PipelineVerificationError("Codex completed workspace terminal differs")
        if budget_terminal is not None:
            try:
                expected_budget = budget_metadata(budget_terminal["budget"])
            except (ValueError, KeyError, TypeError) as exc:
                raise PipelineVerificationError("Codex budget evidence differs") from exc
            if (receipt.status != "failed"
                or evidence.assistant_output != TERMINAL_MARKER
                or canonical_value(budget_terminal) != canonical_value(expected_budget)
                or evidence.tool_trace.retry_count != 0
                or any(row.status != "completed" and not (
                    semantic_tool_failure(row, empty_code_ids)
                ) for row in evidence.tool_trace.results)):
                raise PipelineVerificationError("Codex controlled-budget projection differs")
        if semantic_tool_gate is not None:
            failed_results = tuple(
                row for row in evidence.tool_trace.results
                if row.status == "immutable_failure"
            )
            expected_semantic_tool_gate = {
                "classification": "signed_tool_gate_failure",
                "codex_turn_status": "failed",
                "failed_call_ids": tuple(row.call_id for row in failed_results),
                "error_codes": tuple(row.error_code for row in failed_results),
                "tool_trace_blake3": evidence.tool_trace.trace_blake3,
                "synthetic_terminal_marker": not (
                    isinstance(receipt.final_response, str)
                    and receipt.final_response.strip()
                ),
            }
            expected_output = (
                receipt.final_response
                if isinstance(receipt.final_response, str)
                and receipt.final_response.strip()
                else "[trajectory terminated at a signed semantic tool gate]"
            )
            if (
                receipt.status != "failed"
                or not mcp_calls
                or native_calls
                or not failed_results
                or any(
                    row.status != "completed"
                    and not (
                        semantic_tool_failure(row, empty_code_ids)
                    )
                    for row in evidence.tool_trace.results
                )
                or evidence.tool_trace.retry_count != 0
                or evidence.assistant_output != expected_output
                or canonical_value(semantic_tool_gate)
                != canonical_value(expected_semantic_tool_gate)
            ):
                raise PipelineVerificationError(
                    "Codex semantic tool-gate projection differs"
                )
        expected_mcp_groups = tuple(
            selected
            for group in groups
            if (
                selected := tuple(
                    mapping[call.tool_call_id]
                    for call in group
                    if call.tool_type == "mcpToolCall"
                    and call.fully_qualified_name in offered_names
                    and call.tool_call_id not in rejections
                )
            )
        )
        if canonical_value(metadata["tool_call_groups"]) != canonical_value(
            expected_mcp_groups
        ):
            raise PipelineVerificationError("Codex MCP group projection differs")
        for group in groups:
            if len(group) > 1 and any(call.tool_type != "mcpToolCall" for call in group):
                raise PipelineVerificationError("Codex native action was parallelized")
        for call in native_calls:
            self._verify_persisted_native_call(evaluation, call)
        if projection_schema == projection_v2:
            expected_binding = self._expected_native_file_change_binding(
                evaluation, native_calls
            )
            if canonical_value(
                metadata["native_file_change_effect_binding"]
            ) != canonical_value(expected_binding):
                raise PipelineVerificationError(
                    "Codex native file-change effect binding differs"
                )

        events = evidence.policy_events
        cursor = 2
        for group in groups:
            if cursor >= len(events):
                raise PipelineVerificationError("Codex projected call group is absent")
            assistant = events[cursor]
            cursor += 1
            visible_ids = tuple(
                mapping[call.tool_call_id]
                if call.tool_type == "mcpToolCall"
                and call.fully_qualified_name in offered_names
                and call.tool_call_id not in rejections
                else call.tool_call_id
                for call in group
            )
            decisions = []
            for call, visible_id in zip(group, visible_ids, strict=True):
                if call.tool_call_id in rejections:
                    decisions.append({
                        "call_id": visible_id, "upstream_item_id": call.upstream_item_id,
                        "schema_validation_rejection": {
                            "fully_qualified_name": call.fully_qualified_name,
                            "arguments": call.arguments, "receipt_blake3": call.receipt_blake3}})
                elif (
                    call.tool_type == "mcpToolCall"
                    and call.fully_qualified_name in offered_names
                ):
                    decisions.append(
                        {
                            "call_id": visible_id,
                            "codex_tool_call_id": call.tool_call_id,
                            "upstream_item_id": call.upstream_item_id,
                            "fully_qualified_name": call.fully_qualified_name,
                            "arguments": call.arguments,
                            "receipt_blake3": call.receipt_blake3,
                        }
                    )
                elif call.tool_type == "mcpToolCall":
                    decisions.append(
                        {
                            "call_id": visible_id,
                            "upstream_item_id": call.upstream_item_id,
                            "codex_core_mcp": {
                                "fully_qualified_name": call.fully_qualified_name,
                                "arguments": call.arguments,
                                "receipt_blake3": call.receipt_blake3,
                            },
                        }
                    )
                else:
                    decisions.append(
                        {
                            "call_id": visible_id,
                            "upstream_item_id": call.upstream_item_id,
                            "tool_type": call.tool_type,
                            "name": call.name,
                            "arguments": call.arguments,
                            "receipt_blake3": call.receipt_blake3,
                        }
                    )
            if assistant.role != "assistant" or assistant.tool_call_ids != visible_ids or canonical_value(
                assistant.content
            ) != canonical_value({"codex_tool_calls": decisions}):
                raise PipelineVerificationError("Codex tool decision projection differs")
            for call, visible_id in zip(group, visible_ids, strict=True):
                if cursor >= len(events):
                    raise PipelineVerificationError("Codex projected observation is absent")
                observation = events[cursor]
                cursor += 1
                if call.tool_call_id in rejections:
                    expected = {"schema_validation_rejection": rejections[call.tool_call_id],
                        "codex_tool_call_status": call.status,
                        "codex_tool_call_receipt_blake3": call.receipt_blake3}
                elif (
                    call.tool_type == "mcpToolCall"
                    and call.fully_qualified_name in offered_names
                ):
                    result = trace_by_id[visible_id]
                    expected = {
                        "status": result.status,
                        "output": result.output,
                        "error_code": result.error_code,
                        "receipt_blake3": result.receipt_blake3,
                    }
                    from .execution_diagnostics import diagnostic_content
                    try:
                        supplemental = diagnostic_content(call.output, call_id=result.call_id)
                    except (ValueError, TypeError) as exc:
                        raise PipelineVerificationError("Codex execution diagnostic differs") from exc
                    if supplemental:
                        expected["supplemental_content"] = supplemental
                elif call.tool_type == "mcpToolCall":
                    expected = {
                        "codex_core_mcp": {
                            "fully_qualified_name": call.fully_qualified_name,
                            "status": call.status,
                            "output": call.output,
                            "receipt_blake3": call.receipt_blake3,
                        }
                    }
                else:
                    expected = {
                        "codex_native_action": {
                            "tool_type": call.tool_type,
                            "name": call.name,
                            "status": call.status,
                            "output": call.output,
                            "receipt_blake3": call.receipt_blake3,
                        }
                    }
                if (
                    observation.role != "tool"
                    or observation.tool_call_ids != (visible_id,)
                    or canonical_value(observation.content) != canonical_value(expected)
                ):
                    raise PipelineVerificationError("Codex tool observation projection differs")
        if cursor != len(events) - 1:
            raise PipelineVerificationError("Codex projection event count differs")
        terminal = events[cursor]
        if terminal.role != "assistant" or terminal.tool_call_ids or canonical_value(
            terminal.content
        ) != canonical_value(
            {
                "response": evidence.assistant_output,
                "codex_turn_receipt_blake3": receipt.receipt_blake3,
            }
        ):
            raise PipelineVerificationError("Codex terminal projection differs")
        if native_calls and not native_capability:
            raise PipelineVerificationError("Codex native capability instruction is absent")
        provider_core = {
            "rollout_id": evidence.rollout_id,
            "model_id": evidence.model.model_id,
            "sandbox_manifest_blake3": evidence.sandbox_manifest.manifest_blake3,
            "codex_turn_receipt_blake3": receipt.receipt_blake3,
            "policy_events": events,
            "tool_trace_blake3": evidence.tool_trace.trace_blake3,
            "assistant_output": evidence.assistant_output,
        }
        if check_provider_binding and evidence.provider_receipt_blake3 != blake3_hex(provider_core):
            raise PipelineVerificationError("Codex provider projection receipt differs")
        return tuple(
            call.tool_call_id for call in (*native_calls, *auxiliary_mcp_calls)
        ) + tuple(rejections)

    def _verify_trajectory(self, evaluation: ModelEvaluation, *, check_provider_binding: bool = True) -> None:
        evidence = evaluation.evidence
        events = evidence.policy_events
        if len(events) < 3 or events[0].role != "system" or events[1].role != "user":
            raise PipelineVerificationError("full policy-visible trajectory framing differs")
        if canonical_value(events[1].content) != canonical_value(evidence.policy_visible_context):
            raise PipelineVerificationError("trajectory user context differs from retained context")
        declared: list[str] = []
        observed: list[str] = []
        for event in events:
            core = {
                "event_id": event.event_id,
                "role": event.role,
                "content": event.content,
                "tool_call_ids": event.tool_call_ids,
            }
            uuid_text(event.event_id, label="trajectory event_id")
            if event.event_blake3 != blake3_hex(core):
                raise PipelineVerificationError("trajectory event commitment differs")
            if event.role == "assistant":
                if set(event.tool_call_ids) & set(declared):
                    raise PipelineVerificationError("trajectory tool-call identity was reused")
                declared.extend(event.tool_call_ids)
            elif event.role == "tool":
                if len(event.tool_call_ids) != 1 or event.tool_call_ids[0] not in declared:
                    raise PipelineVerificationError("tool observation lacks an earlier assistant decision")
                observed.append(event.tool_call_ids[0])
        if sorted(declared) != sorted(observed) or len(observed) != len(set(observed)):
            raise PipelineVerificationError("tool observations do not exactly cover calls")
        native_ids = self._verify_codex_actor_projection(
            evaluation, check_provider_binding=check_provider_binding)
        trace_ids = [row.call_id for row in evidence.tool_trace.results]
        if sorted([*trace_ids, *native_ids]) != sorted(declared):
            raise PipelineVerificationError("tool trace and visible trajectory differ")
        if not evidence.assistant_output.strip():
            raise PipelineVerificationError("terminal assistant output is absent")

    @staticmethod
    def _workspace_reference(reference: str) -> tuple[str, str] | None:
        if not reference.startswith("workspace:"):
            return None
        parts = reference.split(":", 2)
        if len(parts) != 3 or parts[1] not in {"before", "after"}:
            raise PipelineVerificationError("judge workspace evidence reference differs")
        path = parts[2]
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or pure.as_posix() != path
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise PipelineVerificationError("judge workspace evidence path differs")
        return parts[1], path

    def _verify_judge_trace(self, evaluation: ModelEvaluation) -> None:
        evidence = evaluation.evidence
        assessment = evaluation.judgment
        trace = assessment.agent_trace
        if trace.evidence_bundle_blake3 != evidence.bundle_blake3:
            raise PipelineVerificationError("judge trace targets a different evidence bundle")
        if trace.retry_count != 0 or not trace.results:
            raise PipelineVerificationError("judge trace retry/result contract differs")

        available_paths = {
            "before": {row.path for row in evidence.workspace_before.files},
            "after": {row.path for row in evidence.workspace_after.files},
        }
        result_by_call: dict[str, Any] = {}
        for result in trace.results:
            for value, label in (
                (result.result_id, "judge tool result_id"),
                (result.call_id, "judge tool call_id"),
                (result.parallel_group_id, "judge tool parallel_group_id"),
            ):
                uuid_text(value, label=label)
            if result.call_id in result_by_call:
                raise PipelineVerificationError("judge tool call identity was reused")
            result_by_call[result.call_id] = result
            if result.evidence_bundle_blake3 != evidence.bundle_blake3:
                raise PipelineVerificationError("judge tool targets a different evidence bundle")
            if result.receipt_blake3 != blake3_hex(_judge_tool_result_core(result)):
                raise PipelineVerificationError("judge tool receipt differs")
            try:
                replay_runtime = JudgeWorkspaceTools(
                    evidence,
                    id_factory=DeterministicUUIDFactory(
                        f"judge-tool-replay:{result.call_id}"
                    ),
                    maximum_parallel_tools=1,
                )
                replayed = replay_runtime.execute_group(
                    (
                        ToolCall(
                            call_id=result.call_id,
                            name=result.name,
                            arguments=result.arguments,
                        ),
                    )
                )[0]
            except (ContractError, JudgeWorkspaceError) as exc:
                raise PipelineVerificationError("judge tool could not be replayed") from exc
            if canonical_value(
                {
                    "status": result.status,
                    "output": result.output,
                    "error_code": result.error_code,
                    "inspected_evidence_refs": result.inspected_evidence_refs,
                    "content_inspection": result.content_inspection,
                }
            ) != canonical_value(
                {
                    "status": replayed.status,
                    "output": replayed.output,
                    "error_code": replayed.error_code,
                    "inspected_evidence_refs": replayed.inspected_evidence_refs,
                    "content_inspection": replayed.content_inspection,
                }
            ):
                raise PipelineVerificationError("judge tool observation differs from snapshot replay")
            if result.content_inspection != (
                result.status == "completed"
                and result.name in {"workspace_read", "workspace_search"}
                and bool(result.inspected_evidence_refs)
            ):
                raise PipelineVerificationError("judge tool content-inspection claim differs")
            for reference in result.inspected_evidence_refs:
                parsed = self._workspace_reference(reference)
                if parsed is None or parsed[1] not in available_paths[parsed[0]]:
                    raise PipelineVerificationError("judge tool cites an absent workspace file")

        expected_joined = tuple(result.call_id for result in trace.results)
        if trace.joined_call_ids != expected_joined or set(trace.declared_call_ids) != set(expected_joined):
            raise PipelineVerificationError("judge trace call inventory differs")
        expected_inspected = tuple(
            sorted(
                {
                    reference
                    for result in trace.results
                    if result.status == "completed"
                    for reference in result.inspected_evidence_refs
                }
            )
        )
        if trace.inspected_evidence_refs != expected_inspected:
            raise PipelineVerificationError("judge trace inspected-file inventory differs")
        expected_content_count = sum(result.content_inspection for result in trace.results)
        if trace.content_inspection_count != expected_content_count:
            raise PipelineVerificationError("judge trace content-inspection count differs")
        if any(available_paths.values()) and expected_content_count < 1:
            raise PipelineVerificationError("agent judge did not inspect workspace file content")

        frontiers = {result.frontier for result in trace.results}
        if frontiers != set(range(trace.frontier_count)):
            raise PipelineVerificationError("judge tool frontiers are not contiguous")
        group_sizes: dict[str, int] = {}
        for result in trace.results:
            group_sizes[result.parallel_group_id] = group_sizes.get(result.parallel_group_id, 0) + 1
        if trace.max_parallelism_observed != max(group_sizes.values()):
            raise PipelineVerificationError("judge trace parallelism claim differs")

        declared: list[str] = []
        observed: list[str] = []
        observed_events: dict[str, Any] = {}
        assistant_turns = 0
        for index, event in enumerate(trace.policy_events):
            core = {
                "event_id": event.event_id,
                "role": event.role,
                "content": event.content,
                "tool_call_ids": event.tool_call_ids,
            }
            uuid_text(event.event_id, label="judge trajectory event_id")
            if event.event_blake3 != blake3_hex(core):
                raise PipelineVerificationError("judge trajectory event commitment differs")
            if event.role == "assistant":
                assistant_turns += 1
                if set(event.tool_call_ids) & set(declared):
                    raise PipelineVerificationError("judge trajectory reused a tool call")
                declared.extend(event.tool_call_ids)
                if event.tool_call_ids:
                    content = event.content
                    if not isinstance(content, Mapping) or set(content) != {"text", "tool_calls"}:
                        raise PipelineVerificationError("judge tool decision projection differs")
                    calls = content["tool_calls"]
                    if not isinstance(calls, tuple) or len(calls) != len(event.tool_call_ids):
                        raise PipelineVerificationError("judge tool decision call group differs")
                    for call_id, call in zip(event.tool_call_ids, calls, strict=True):
                        result = result_by_call.get(call_id)
                        if (
                            result is None
                            or not isinstance(call, Mapping)
                            or set(call) != {"call_id", "name", "arguments"}
                            or call["call_id"] != call_id
                            or call["name"] != result.name
                            or canonical_value(call["arguments"])
                            != canonical_value(result.arguments)
                        ):
                            raise PipelineVerificationError("judge tool decision arguments differ")
            elif event.role == "tool":
                if len(event.tool_call_ids) != 1 or event.tool_call_ids[0] not in declared:
                    raise PipelineVerificationError("judge tool observation lacks its decision")
                call_id = event.tool_call_ids[0]
                observed.append(call_id)
                observed_events[call_id] = event
                result = result_by_call[call_id]
                expected_visible = {
                    "status": result.status,
                    "output": result.output,
                    "error_code": result.error_code,
                    "receipt_blake3": result.receipt_blake3,
                    "inspected_evidence_refs": result.inspected_evidence_refs,
                    "content_inspection": result.content_inspection,
                }
                if canonical_value(event.content) != canonical_value(expected_visible):
                    raise PipelineVerificationError("judge tool observation event differs")
            if index == len(trace.policy_events) - 1 and (
                event.role != "assistant" or event.tool_call_ids or not isinstance(event.content, str)
            ):
                raise PipelineVerificationError("judge trajectory lacks a terminal scoring decision")
        if tuple(declared) != trace.declared_call_ids or sorted(observed) != sorted(declared):
            raise PipelineVerificationError("judge trajectory and trace calls differ")
        if assistant_turns != trace.provider_turn_count:
            raise PipelineVerificationError("judge provider-turn count differs")
        if len(observed_events) != len(result_by_call):
            raise PipelineVerificationError("judge tool observations are incomplete")
        user_content = trace.policy_events[1].content
        if not isinstance(user_content, Mapping):
            raise PipelineVerificationError("judge request projection differs")
        projected_evidence = user_content.get("workspace_evidence")
        if (
            not isinstance(projected_evidence, Mapping)
            or projected_evidence.get("bundle_blake3") != evidence.bundle_blake3
        ):
            raise PipelineVerificationError("judge request does not bind the evidence bundle")
        if trace.trace_blake3 != blake3_hex(_judge_trace_core(trace)):
            raise PipelineVerificationError("judge agent trace commitment differs")

        actor_event_ids = {event.event_id for event in evidence.policy_events}
        actor_tool_ids = {result.call_id for result in evidence.tool_trace.results}
        for score in assessment.item_scores:
            for reference in score.evidence_refs:
                parsed = self._workspace_reference(reference)
                if parsed is not None:
                    if reference not in trace.inspected_evidence_refs:
                        raise PipelineVerificationError("rubric score cites unread workspace evidence")
                    continue
                if reference == "context:policy-visible":
                    continue
                if reference.startswith("trajectory:event:"):
                    if reference.removeprefix("trajectory:event:") not in actor_event_ids:
                        raise PipelineVerificationError("rubric score cites an absent trajectory event")
                    continue
                if reference.startswith("actor-tool:"):
                    if reference.removeprefix("actor-tool:") not in actor_tool_ids:
                        raise PipelineVerificationError("rubric score cites an absent actor tool call")
                    continue
                if reference.startswith("judge-reference:"):
                    continue
                raise PipelineVerificationError("rubric score evidence reference namespace differs")

    def _verify_evaluation(
        self, evaluation: ModelEvaluation, rubric: CompiledRubricTable, run_id: str
    ) -> None:
        uuid_text(evaluation.evaluation_id, label="evaluation_id")
        evidence = evaluation.evidence
        uuid_text(evidence.bundle_id, label="bundle_id")
        uuid_text(evidence.rollout_id, label="rollout_id")
        if evaluation.cohort is not evaluation.model.cohort or evidence.model != evaluation.model:
            raise PipelineVerificationError("evaluation model/cohort binding differs")
        if evidence.sandbox_manifest.rubric.digest != rubric.digest:
            raise PipelineVerificationError("evidence rubric binding differs")
        if evidence.context_blake3 != blake3_hex(evidence.policy_visible_context):
            raise PipelineVerificationError("policy context commitment differs")
        self._verify_workspace(evidence.workspace_before)
        self._verify_workspace(evidence.workspace_after)
        for row in evidence.tool_trace.results:
            for value, label in (
                (row.result_id, "tool result_id"),
                (row.call_id, "tool call_id"),
                (row.parallel_group_id, "tool parallel_group_id"),
            ):
                uuid_text(value, label=label)
            if row.receipt_blake3 != blake3_hex(_tool_result_core(row)):
                raise PipelineVerificationError("tool result receipt differs")
        trace = evidence.tool_trace
        if trace.retry_count != 0 or trace.joined_call_ids != tuple(
            row.call_id for row in trace.results
        ):
            raise PipelineVerificationError("tool trace retry/join contract differs")
        if trace.trace_blake3 != blake3_hex(_trace_core(trace)):
            raise PipelineVerificationError("tool trace commitment differs")
        self._verify_trajectory(evaluation)
        if evidence.bundle_blake3 != blake3_hex(evidence_core(evidence)):
            raise PipelineVerificationError("evidence bundle commitment differs")
        assessment = evaluation.judgment
        uuid_text(assessment.judgment_id, label="judgment_id")
        if "opus-5" not in assessment.judge_model_id.casefold().replace("_", "-"):
            raise PipelineVerificationError("judgment was not produced by Opus 5")
        if assessment.rubric_digest != rubric.digest:
            raise PipelineVerificationError("judgment rubric differs")
        self._verify_judge_trace(evaluation)
        if assessment.assessment_blake3 != blake3_hex(_assessment_core(assessment)):
            raise PipelineVerificationError("judgment commitment differs")
        reward = evaluation.reward
        uuid_text(reward.reward_id, label="reward_id")
        expected = self._rewarder.compute(
            reward_id=reward.reward_id,
            rubric=rubric,
            assessment=assessment,
            evidence=evidence,
        )
        if canonical_value(expected) != canonical_value(reward):
            raise PipelineVerificationError("reward was not recomputed by the bound rubric")
        if evaluation.evaluation_blake3 != blake3_hex(evaluation_core(evaluation)):
            raise PipelineVerificationError("evaluation commitment differs")
        if not evaluation.manifest_artifact.relative_path.startswith(
            f"{run_id}/{evaluation.cohort.value}/"
        ) or not evaluation.evidence_artifact.relative_path.startswith(
            f"{run_id}/{evaluation.cohort.value}/"
        ):
            raise PipelineVerificationError("evaluation artifact namespace differs")
        if not self._artifacts.verify(evaluation.manifest_artifact) or not self._artifacts.verify(
            evaluation.evidence_artifact
        ):
            raise PipelineVerificationError("immutable evaluation artifact differs")
        if _artifact_payload(self._artifacts, evaluation.evidence_artifact.relative_path) != canonical_json_bytes(evidence):
            raise PipelineVerificationError("evidence artifact payload differs")
        if _artifact_payload(self._artifacts, evaluation.manifest_artifact.relative_path) != canonical_json_bytes(
            evaluation_manifest_document(evaluation)
        ):
            raise PipelineVerificationError("evaluation manifest payload differs")

    def verify_evaluation_or_raise(
        self,
        *,
        evaluation: ModelEvaluation,
        rubric: CompiledRubricTable,
        run_id: str,
    ) -> None:
        """Reopen one evaluation before it may release slower cohorts."""

        uuid_text(run_id, label="pipeline run_id")
        self._verify_evaluation(evaluation, rubric, run_id)

    def _verify_or_raise(
        self, result: PipelineResult, rubric: CompiledRubricTable
    ) -> tuple[str, ...]:
        checks: list[str] = []
        uuid_text(result.run_id, label="pipeline run_id")
        if len(result.sandbox_manifests) != 1:
            raise PipelineVerificationError("pipeline must contain exactly one logical sandbox")
        manifest = result.sandbox_manifests[0]
        uuid_text(manifest.sandbox_id, label="sandbox_id")
        if manifest.manifest_blake3 != blake3_hex(_manifest_core(manifest)):
            raise PipelineVerificationError("sandbox manifest commitment differs")
        if (
            manifest.rubric.rubric_id != rubric.rubric_id
            or manifest.rubric.version != rubric.version
            or manifest.rubric.digest != rubric.digest
            or manifest.rubric.domain != rubric.domain
            or manifest.rubric.stage.value != rubric.stage
        ):
            raise PipelineVerificationError("sandbox does not bind the supplied exact compiled rubric")
        checks.append("one_exact_rubric_bound_sandbox")

        evaluations = result.evaluations
        report_is_v2 = (
            result.separation.admission_policy_schema
            == "eva.trajectory-admission-policy.v2"
        )
        expected_order = tuple(
            cohort
            for cohort in COHORT_ORDER
            if any(row.cohort is cohort for row in evaluations)
        )
        if (
            not evaluations
            or tuple(row.cohort for row in evaluations) != expected_order
            or len(expected_order) != len(evaluations)
            or (not report_is_v2 and expected_order != COHORT_ORDER)
        ):
            raise PipelineVerificationError(
                "evaluations must contain unique cohorts in weak, middle, strong order"
            )
        if len({row.model.model_id for row in evaluations}) != len(evaluations) or len(
            {row.evidence.rollout_id for row in evaluations}
        ) != len(evaluations):
            raise PipelineVerificationError("model or rollout identity is reused")
        for evaluation in evaluations:
            if evaluation.evidence.sandbox_manifest != manifest:
                raise PipelineVerificationError("evaluation sandbox manifest differs")
            self._verify_evaluation(evaluation, rubric, result.run_id)
        checks.extend(("single_rollout_per_completed_cohort", "full_context_workspace_evidence", "opus5_same_rubric_rewards"))

        before = {row.evidence.workspace_before.tree_blake3 for row in evaluations}
        if len(before) != 1:
            raise PipelineVerificationError("cohort workspaces were not byte-identical fresh resets")
        checks.append("fresh_reset_integrity")

        rewards = {row.cohort.value: row.reward.total_reward for row in evaluations}
        raw = {
            row.cohort.value: {score.item_id: score.score for score in row.reward.item_scores}
            for row in evaluations
        }
        def delta(left: str, right: str) -> float | None:
            if left not in rewards or right not in rewards:
                return None
            return rewards[left] - rewards[right]

        strong_minus_weak = delta("strong", "weak")
        strong_minus_middle = delta("strong", "middle")
        middle_minus_weak = delta("middle", "weak")
        by_cohort = {row.cohort: row for row in evaluations}
        complete = set(by_cohort) == set(COHORT_ORDER)
        material = complete and any(
            abs(raw["strong"][item_id] - raw["weak"][item_id])
            >= self._policy.material_item_delta
            for item_id in raw["strong"]
        )
        passed = bool(
            complete
            and rewards["strong"] >= self._policy.minimum_strong_reward
            and strong_minus_weak is not None
            and strong_minus_weak >= self._policy.minimum_strong_minus_weak
            and material
            and (
                not self._policy.require_strong_hard_gates
                or by_cohort[Cohort.STRONG].reward.hard_gates_passed
            )
        )
        expected_separation = {
            "rewards_by_cohort": rewards,
            "raw_item_scores_by_cohort": raw,
            "strong_minus_weak": strong_minus_weak,
            "strong_minus_middle": strong_minus_middle,
            "middle_minus_weak": middle_minus_weak,
            "perfect_monotonic_staircase_required": False,
            "policy_passed": passed,
        }
        if report_is_v2:
            if (
                result.separation.ability_separation_required_for_admission
                != self._policy.ability_separation_required_for_admission
                or result.separation.minimum_valid_judged_trajectories
                != self._policy.minimum_valid_judged_trajectories
            ):
                raise PipelineVerificationError("trajectory admission policy differs")
            failure_types = dict(result.separation.failure_types_by_cohort or {})
            completed = {row.cohort.value for row in evaluations}
            expected_failed = {cohort.value for cohort in COHORT_ORDER} - completed
            if set(failure_types) != expected_failed or any(
                not isinstance(value, str) or not value
                for value in failure_types.values()
            ):
                raise PipelineVerificationError("failed cohort evidence differs")
            qualifying = tuple(row.evaluation_blake3 for row in evaluations)
            selected = max(
                evaluations,
                key=lambda row: (
                    float(row.reward.total_reward),
                    COHORT_ORDER.index(row.cohort),
                ),
            )
            admission_passed = bool(
                len(qualifying) >= self._policy.minimum_valid_judged_trajectories
                and (
                    not self._policy.ability_separation_required_for_admission
                    or passed
                )
            )
            expected_outcomes = {
                cohort.value: (
                    "completed"
                    if cohort.value in completed
                    else (
                        "cancelled_after_valid_evaluation"
                        if failure_types[cohort.value]
                        == _EARLY_COHORT_CANCELLATION_NAME
                        else "infrastructure_failure"
                    )
                )
                for cohort in COHORT_ORDER
            }
            cancelled = {
                cohort
                for cohort, outcome in expected_outcomes.items()
                if outcome == "cancelled_after_valid_evaluation"
            }
            if cancelled and (
                not self._policy.early_continuation_after_first_valid_trajectory
                or not admission_passed
            ):
                raise PipelineVerificationError("early cohort cancellation policy differs")
            expected_separation.update(
                {
                    "admission_policy_schema": self._policy.admission_policy_schema,
                    "ability_separation_required_for_admission": (
                        self._policy.ability_separation_required_for_admission
                    ),
                    "minimum_valid_judged_trajectories": (
                        self._policy.minimum_valid_judged_trajectories
                    ),
                    "valid_judged_trajectory_count": len(qualifying),
                    "qualifying_evaluation_blake3s": qualifying,
                    "selected_evaluation_blake3": selected.evaluation_blake3,
                    "selected_evaluation_cohort": selected.cohort.value,
                    "cohort_outcomes": expected_outcomes,
                    "failure_types_by_cohort": failure_types,
                    "admission_policy_passed": admission_passed,
                }
            )
            if failure_types:
                partial = partial_cohort_outcome_document(
                    run_id=result.run_id,
                    manifest_blake3=manifest.manifest_blake3,
                    evaluations=evaluations,
                    failures=failure_types,
                    early_continuation_enabled=(
                        self._policy.early_continuation_after_first_valid_trajectory
                    ),
                )
                partial_path = (
                    self._artifacts.root
                    / result.run_id
                    / "partial-cohort-failures.json"
                )
                try:
                    partial_payload = partial_path.read_bytes()
                except OSError as exc:
                    raise PipelineVerificationError(
                        "partial cohort outcome evidence is absent"
                    ) from exc
                if partial_payload != canonical_json_bytes(partial):
                    raise PipelineVerificationError(
                        "partial cohort outcome evidence differs"
                    )
                checks.append("partial_cohort_outcomes_reopened")
        if canonical_value(separation_core(result.separation)) != canonical_value(expected_separation):
            raise PipelineVerificationError("ability separation report differs")
        if result.separation.report_blake3 != blake3_hex(separation_core(result.separation)):
            raise PipelineVerificationError("ability separation commitment differs")
        checks.append("ability_separation_recomputed")

        if result.recommendation.admission_authorized or result.recommendation.signed_decision_created:
            raise PipelineVerificationError("pipeline recommendation improperly claims admission authority")
        admission_passed = (
            result.separation.admission_policy_passed
            if report_is_v2
            else passed
        )
        expected_name = "eligible_for_signed_supervisor_admission" if admission_passed else "not_eligible"
        if result.recommendation.recommendation != expected_name:
            raise PipelineVerificationError("admission recommendation differs")
        if result.recommendation.recommendation_blake3 != blake3_hex(
            recommendation_core(result.recommendation)
        ):
            raise PipelineVerificationError("recommendation commitment differs")
        checks.append("unsigned_recommendation_only")

        if len(result.telemetry) != len(evaluations):
            raise PipelineVerificationError("telemetry cohort coverage differs")
        for row, evaluation in zip(result.telemetry, evaluations, strict=True):
            trace = evaluation.evidence.tool_trace
            expected = {
                "cohort": evaluation.cohort,
                "model_id": evaluation.model.model_id,
                "rubric_digest": evaluation.reward.rubric_digest,
                "reward": evaluation.reward.total_reward,
                "item_scores": {score.item_id: score.score for score in evaluation.reward.item_scores},
                "tool_call_count": len(trace.results),
                "tool_failure_count": sum(item.status != "completed" for item in trace.results),
                "parallelism_observed": trace.max_parallelism_observed,
                "workspace_changed": evaluation.evidence.workspace_before.tree_blake3 != evaluation.evidence.workspace_after.tree_blake3,
                "hard_gates_passed": evaluation.reward.hard_gates_passed,
                "outcome": "completed",
            }
            if canonical_value(telemetry_core(row)) != canonical_value(expected):
                raise PipelineVerificationError("telemetry projection differs")
            if row.telemetry_blake3 != blake3_hex(telemetry_core(row)):
                raise PipelineVerificationError("telemetry commitment differs")
        checks.append("outcome_telemetry_recomputed")

        if result.result_blake3 != blake3_hex(result_core(result)):
            raise PipelineVerificationError("pipeline result commitment differs")
        top_path = self._artifacts.root / result.run_id / "pipeline-result.json"
        try:
            payload = top_path.read_bytes()
        except OSError as exc:
            raise PipelineVerificationError("persisted top-level result is absent") from exc
        if payload != canonical_json_bytes(pipeline_result_document(result)):
            raise PipelineVerificationError("persisted top-level result payload differs")
        checks.append("immutable_top_level_result")
        return tuple(checks)

    def verify(
        self, *, result: PipelineResult, rubric: CompiledRubricTable
    ) -> VerificationReport:
        errors: tuple[str, ...] = ()
        checks: tuple[str, ...] = ()
        try:
            checks = self._verify_or_raise(result, rubric)
            valid = True
        except Exception as exc:
            valid = False
            errors = (f"{type(exc).__name__}:{exc}",)
        core = {
            "valid": valid,
            "checks": checks,
            "errors": errors,
            "result_blake3": result.result_blake3,
        }
        return VerificationReport(
            **core,
            report_blake3=blake3_hex(core),
        )

    def verify_or_raise(
        self, *, result: PipelineResult, rubric: CompiledRubricTable
    ) -> VerificationReport:
        report = self.verify(result=result, rubric=rubric)
        if not report.valid:
            raise PipelineVerificationError(report.errors[0])
        return report


def verify_result_document(
    path: str | Path,
    *,
    artifact_store: ImmutableArtifactStore,
    rubric: CompiledRubricTable,
    separation_policy: SeparationPolicy | None = None,
    rewarder: RewardComputer | None = None,
) -> VerificationReport:
    """Load and independently verify one canonical result document/path."""

    result = load_pipeline_result(path)
    return PipelineVerifier(
        artifact_store,
        separation_policy=separation_policy,
        rewarder=rewarder,
    ).verify(result=result, rubric=rubric)


__all__ = [
    "PipelineVerificationError",
    "PipelineVerifier",
    "VerificationReport",
    "verify_result_document",
]
