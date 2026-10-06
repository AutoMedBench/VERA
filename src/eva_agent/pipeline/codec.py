"""Strict codec for reopening immutable pipeline result documents."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import stat
from typing import Any, Mapping

from .contracts import (
    AdmissionRecommendation,
    AbilitySeparationReport,
    ArtifactRef,
    BenchmarkSource,
    Cohort,
    ContractError,
    EvidenceBundle,
    FileSnapshot,
    JudgeAgentTrace,
    JudgeAssessment,
    JudgeToolResult,
    ModelEvaluation,
    ModelTarget,
    OutcomeTelemetry,
    PipelineResult,
    RewardRecord,
    RubricBinding,
    RubricItemScore,
    SandboxManifest,
    Stage,
    ToolResult,
    ToolTrace,
    TrajectoryEvent,
    WorkspaceSnapshot,
    freeze_json,
)
from .digests import canonical_json_bytes


class PipelineCodecError(ContractError):
    """A persisted pipeline document is not the exact declared schema."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineCodecError(f"{label} must be an object")
    return value


def _exact(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    item = _mapping(value, label=label)
    if set(item) != keys:
        raise PipelineCodecError(f"{label} fields differ")
    return item


def _decode_bytes(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"$bytes_base64"}:
            encoded = value["$bytes_base64"]
            if not isinstance(encoded, str):
                raise PipelineCodecError("encoded byte payload differs")
            try:
                return base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError):
                raise PipelineCodecError("encoded byte payload is invalid base64") from None
        return {str(key): _decode_bytes(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_decode_bytes(child) for child in value]
    return value


def _source(value: Any) -> BenchmarkSource:
    item = _exact(value, {"benchmark", "source_file", "source_revision"}, label="source")
    return BenchmarkSource(**item)  # type: ignore[arg-type]


def _binding(value: Any) -> RubricBinding:
    item = _exact(
        value, {"rubric_id", "version", "digest", "domain", "stage"}, label="rubric binding"
    )
    return RubricBinding(
        rubric_id=item["rubric_id"],
        version=item["version"],
        digest=item["digest"],
        domain=item["domain"],
        stage=Stage(item["stage"]),
    )


def _manifest(value: Any) -> SandboxManifest:
    item = _exact(
        value,
        {
            "sandbox_id", "episode_id", "source", "domain", "stage", "instruction",
            "policy_context", "initial_files", "rubric", "manifest_blake3",
        },
        label="sandbox manifest",
    )
    context = freeze_json(item["policy_context"])
    files = item["initial_files"]
    if not isinstance(context, Mapping) or not isinstance(files, Mapping):
        raise PipelineCodecError("sandbox context or initial files differ")
    return SandboxManifest(
        sandbox_id=item["sandbox_id"],
        episode_id=item["episode_id"],
        source=_source(item["source"]),
        domain=item["domain"],
        stage=Stage(item["stage"]),
        instruction=item["instruction"],
        policy_context=context,
        initial_files=files,
        rubric=_binding(item["rubric"]),
        manifest_blake3=item["manifest_blake3"],
    )


def _model(value: Any) -> ModelTarget:
    item = _exact(value, {"cohort", "model_id", "provider"}, label="model target")
    return ModelTarget(Cohort(item["cohort"]), item["model_id"], item["provider"])


def _file(value: Any) -> FileSnapshot:
    item = _exact(
        value, {"path", "content", "byte_count", "mode", "content_blake3"}, label="file snapshot"
    )
    return FileSnapshot(**item)  # type: ignore[arg-type]


def _workspace(value: Any) -> WorkspaceSnapshot:
    item = _exact(
        value, {"label", "files", "file_count", "byte_count", "tree_blake3"}, label="workspace snapshot"
    )
    return WorkspaceSnapshot(
        label=item["label"],
        files=tuple(_file(row) for row in item["files"]),
        file_count=item["file_count"],
        byte_count=item["byte_count"],
        tree_blake3=item["tree_blake3"],
    )


def _tool_result(value: Any) -> ToolResult:
    keys = {
        "result_id", "call_id", "name", "frontier", "parallel_group_id", "status",
        "output", "error_code", "workspace_before_blake3", "workspace_after_blake3",
        "receipt_blake3",
    }
    return ToolResult(**_exact(value, keys, label="tool result"))  # type: ignore[arg-type]


def _tool_trace(value: Any) -> ToolTrace:
    item = _exact(
        value,
        {
            "results", "declared_call_ids", "joined_call_ids", "frontier_count",
            "max_parallelism_observed", "retry_count", "trace_blake3",
        },
        label="tool trace",
    )
    return ToolTrace(
        results=tuple(_tool_result(row) for row in item["results"]),
        declared_call_ids=tuple(item["declared_call_ids"]),
        joined_call_ids=tuple(item["joined_call_ids"]),
        frontier_count=item["frontier_count"],
        max_parallelism_observed=item["max_parallelism_observed"],
        retry_count=item["retry_count"],
        trace_blake3=item["trace_blake3"],
    )


def _event(value: Any) -> TrajectoryEvent:
    item = _exact(
        value, {"event_id", "role", "content", "tool_call_ids", "event_blake3"}, label="trajectory event"
    )
    return TrajectoryEvent(
        event_id=item["event_id"],
        role=item["role"],
        content=item["content"],
        tool_call_ids=tuple(item["tool_call_ids"]),
        event_blake3=item["event_blake3"],
    )


def _evidence(value: Any) -> EvidenceBundle:
    keys = {
        "bundle_id", "rollout_id", "sandbox_manifest", "model", "policy_visible_context",
        "context_blake3", "workspace_before", "workspace_after", "tool_trace",
        "policy_events", "assistant_output", "provider_receipt_blake3",
        "safe_provider_metadata", "bundle_blake3",
    }
    item = _exact(value, keys, label="evidence bundle")
    context = freeze_json(item["policy_visible_context"])
    metadata = freeze_json(item["safe_provider_metadata"])
    if not isinstance(context, Mapping) or not isinstance(metadata, Mapping):
        raise PipelineCodecError("evidence context or safe metadata differs")
    return EvidenceBundle(
        bundle_id=item["bundle_id"],
        rollout_id=item["rollout_id"],
        sandbox_manifest=_manifest(item["sandbox_manifest"]),
        model=_model(item["model"]),
        policy_visible_context=context,
        context_blake3=item["context_blake3"],
        workspace_before=_workspace(item["workspace_before"]),
        workspace_after=_workspace(item["workspace_after"]),
        tool_trace=_tool_trace(item["tool_trace"]),
        policy_events=tuple(_event(row) for row in item["policy_events"]),
        assistant_output=item["assistant_output"],
        provider_receipt_blake3=item["provider_receipt_blake3"],
        safe_provider_metadata=metadata,
        bundle_blake3=item["bundle_blake3"],
    )


def _score(value: Any) -> RubricItemScore:
    item = _exact(value, {"item_id", "score", "evidence_refs", "rationale"}, label="item score")
    return RubricItemScore(
        item_id=item["item_id"], score=item["score"],
        evidence_refs=tuple(item["evidence_refs"]), rationale=item["rationale"],
    )


def _judge_tool_result(value: Any) -> JudgeToolResult:
    item = _exact(
        value,
        {
            "result_id", "call_id", "name", "arguments", "frontier",
            "parallel_group_id", "status", "output", "error_code",
            "inspected_evidence_refs", "content_inspection",
            "evidence_bundle_blake3", "receipt_blake3",
        },
        label="judge tool result",
    )
    return JudgeToolResult(
        **{
            **item,
            "inspected_evidence_refs": tuple(item["inspected_evidence_refs"]),
        }
    )


def _judge_agent_trace(value: Any) -> JudgeAgentTrace:
    item = _exact(
        value,
        {
            "policy_events", "results", "declared_call_ids", "joined_call_ids",
            "inspected_evidence_refs", "content_inspection_count", "frontier_count",
            "max_parallelism_observed", "provider_turn_count", "retry_count",
            "evidence_bundle_blake3", "trace_blake3",
        },
        label="judge agent trace",
    )
    return JudgeAgentTrace(
        **{
            **item,
            "policy_events": tuple(_event(row) for row in item["policy_events"]),
            "results": tuple(_judge_tool_result(row) for row in item["results"]),
            "declared_call_ids": tuple(item["declared_call_ids"]),
            "joined_call_ids": tuple(item["joined_call_ids"]),
            "inspected_evidence_refs": tuple(item["inspected_evidence_refs"]),
        }
    )


def _judgment(value: Any) -> JudgeAssessment:
    item = _exact(
        value,
        {
            "judgment_id", "judge_model_id", "rubric_digest", "agent_trace",
            "item_scores", "hard_gates_passed", "summary", "assessment_blake3",
        },
        label="judge assessment",
    )
    return JudgeAssessment(
        **{
            **item,
            "agent_trace": _judge_agent_trace(item["agent_trace"]),
            "item_scores": tuple(_score(row) for row in item["item_scores"]),
        }
    )


def _reward(value: Any) -> RewardRecord:
    item = _exact(
        value,
        {"reward_id", "rubric_digest", "item_scores", "total_reward", "hard_gates_passed", "reward_blake3"},
        label="reward record",
    )
    return RewardRecord(
        **{**item, "item_scores": tuple(_score(row) for row in item["item_scores"])}
    )


def _artifact(value: Any) -> ArtifactRef:
    return ArtifactRef(**_exact(value, {"relative_path", "byte_count", "mode", "content_blake3"}, label="artifact reference"))  # type: ignore[arg-type]


def _evaluation(value: Any) -> ModelEvaluation:
    item = _exact(
        value,
        {"evaluation_id", "cohort", "model", "evidence", "judgment", "reward", "manifest_artifact", "evidence_artifact", "evaluation_blake3"},
        label="model evaluation",
    )
    return ModelEvaluation(
        evaluation_id=item["evaluation_id"], cohort=Cohort(item["cohort"]),
        model=_model(item["model"]), evidence=_evidence(item["evidence"]),
        judgment=_judgment(item["judgment"]), reward=_reward(item["reward"]),
        manifest_artifact=_artifact(item["manifest_artifact"]),
        evidence_artifact=_artifact(item["evidence_artifact"]),
        evaluation_blake3=item["evaluation_blake3"],
    )


def pipeline_result_from_document(value: Any) -> PipelineResult:
    """Reconstruct one result from its exact canonical JSON projection."""

    value = _decode_bytes(value)
    item = _exact(
        value,
        {"run_id", "sandbox_manifests", "evaluations", "separation", "recommendation", "telemetry", "result_blake3"},
        label="pipeline result",
    )
    separation_v1_keys = {
        "rewards_by_cohort", "raw_item_scores_by_cohort", "strong_minus_weak",
        "strong_minus_middle", "middle_minus_weak",
        "perfect_monotonic_staircase_required", "policy_passed", "report_blake3",
    }
    separation_v2_keys = separation_v1_keys | {
        "admission_policy_schema", "ability_separation_required_for_admission",
        "minimum_valid_judged_trajectories", "valid_judged_trajectory_count",
        "qualifying_evaluation_blake3s", "selected_evaluation_blake3",
        "selected_evaluation_cohort", "cohort_outcomes",
        "failure_types_by_cohort", "admission_policy_passed",
    }
    raw_separation = item["separation"]
    if not isinstance(raw_separation, dict) or frozenset(raw_separation) not in {
        frozenset(separation_v1_keys), frozenset(separation_v2_keys)
    }:
        raise PipelineCodecError("separation report shape differs")
    sep = dict(raw_separation)
    if set(sep) == separation_v2_keys:
        sep["qualifying_evaluation_blake3s"] = tuple(
            sep["qualifying_evaluation_blake3s"]
        )
    rec = _exact(
        item["recommendation"],
        {"recommendation", "reasons", "admission_authorized", "signed_decision_created", "recommendation_blake3"},
        label="admission recommendation",
    )
    telemetry = []
    for raw in item["telemetry"]:
        row = _exact(
            raw,
            {"cohort", "model_id", "rubric_digest", "reward", "item_scores", "tool_call_count", "tool_failure_count", "parallelism_observed", "workspace_changed", "hard_gates_passed", "outcome", "telemetry_blake3"},
            label="outcome telemetry",
        )
        telemetry.append(OutcomeTelemetry(**{**row, "cohort": Cohort(row["cohort"])}))
    return PipelineResult(
        run_id=item["run_id"],
        sandbox_manifests=tuple(_manifest(row) for row in item["sandbox_manifests"]),
        evaluations=tuple(_evaluation(row) for row in item["evaluations"]),
        separation=AbilitySeparationReport(**sep),  # type: ignore[arg-type]
        recommendation=AdmissionRecommendation(
            **{**rec, "reasons": tuple(rec["reasons"])}
        ),
        telemetry=tuple(telemetry),
        result_blake3=item["result_blake3"],
    )


def load_pipeline_result(path: str | Path) -> PipelineResult:
    """Open a no-symlink canonical result file and reconstruct its contracts."""

    source = Path(path)
    try:
        info = source.lstat()
        if source.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > 512 * 1024 * 1024:
            raise PipelineCodecError("pipeline result file topology or size differs")
        payload = source.read_bytes()
        value = json.loads(payload)
    except PipelineCodecError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise PipelineCodecError("pipeline result document cannot be opened") from exc
    if canonical_json_bytes(value) != payload:
        raise PipelineCodecError("pipeline result document is not canonical JSON")
    return pipeline_result_from_document(value)


__all__ = [
    "PipelineCodecError",
    "load_pipeline_result",
    "pipeline_result_from_document",
]
