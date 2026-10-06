"""One-shot Codex/Opus 5 Agent Judge worker over a teacher trajectory.

The asynchronous scheduler owns retries (there are none) and concurrency.  This
module reconstructs the already-validated teacher evidence as immutable pipeline
contracts, invokes one workspace-aware judge, and emits only the public scoring
receipt.  Provider receipts and private model reasoning are never serialized.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from eva_agent.pipeline import (
    BenchmarkEpisode,
    BenchmarkSource,
    Cohort,
    EvidenceBundle,
    FileSnapshot,
    JudgeAssessment,
    JudgeRequest,
    ModelTarget,
    RandomUUIDFactory,
    SandboxManifest,
    Stage,
    ToolResult,
    ToolTrace,
    TrajectoryEvent,
    WorkspaceSnapshot,
)
from eva_agent.codex_pipeline import (
    CODEX_JUDGE_MATERIAL_PREFIX,
    CODEX_JUDGE_REQUIRED_MATERIAL_PATHS,
    codex_judge_evidence_index,
)
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
)
from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
from eva_agent.pipeline.runner import evidence_core
from eva_agent.rubrics import CompiledRubricRegistry

from .agent_judge import (
    AgentJudgeSelectionError,
    AgentJudgeTask,
    REQUIRED_INSPECTION_SECTIONS,
    ValidatedTrajectory,
    validate_full_trajectory,
)


class AgentJudgePort(Protocol):
    """The existing CodexOpus5AgentJudge surface used by this worker."""

    def judge(self, request: JudgeRequest, rubric: Any) -> JudgeAssessment: ...


@dataclass(frozen=True)
class PreparedAgentJudgeTask:
    task: AgentJudgeTask
    trajectory: ValidatedTrajectory
    evidence: EvidenceBundle
    source_workspace_refs: tuple[str, ...]
    source_workspace_byte_count: int
    fast_workspace_judge: bool = False


def _bytes(value: Any, *, label: str) -> bytes:
    if not isinstance(value, Mapping) or set(value) != {"$bytes_base64"}:
        raise AgentJudgeSelectionError(f"{label} bytes differ")
    encoded = value["$bytes_base64"]
    if not isinstance(encoded, str):
        raise AgentJudgeSelectionError(f"{label} bytes differ")
    try:
        return base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError):
        raise AgentJudgeSelectionError(f"{label} bytes differ") from None


def _workspace(value: Any, *, expected_label: str) -> WorkspaceSnapshot:
    if not isinstance(value, Mapping) or not isinstance(value.get("files"), list):
        raise AgentJudgeSelectionError(f"workspace {expected_label} differs")
    rows = []
    for raw in value["files"]:
        if not isinstance(raw, Mapping):
            raise AgentJudgeSelectionError(f"workspace {expected_label} file differs")
        rows.append(
            FileSnapshot(
                path=raw["path"],
                content=_bytes(raw["content"], label=f"workspace {expected_label}"),
                byte_count=raw["byte_count"],
                mode=raw["mode"],
                content_blake3=raw["content_blake3"],
            )
        )
    return WorkspaceSnapshot(
        label=str(value.get("label") or expected_label),
        files=tuple(rows),
        file_count=value["file_count"],
        byte_count=value["byte_count"],
        tree_blake3=value["tree_blake3"],
    )


def _tool_trace(value: Any) -> ToolTrace:
    if not isinstance(value, Mapping) or not isinstance(value.get("results"), list):
        raise AgentJudgeSelectionError("teacher tool trace differs")
    results = []
    for raw in value["results"]:
        if not isinstance(raw, Mapping):
            raise AgentJudgeSelectionError("teacher tool result differs")
        keys = {
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
        if set(raw) != keys:
            raise AgentJudgeSelectionError("teacher tool result shape differs")
        result_core = {key: raw[key] for key in keys if key != "receipt_blake3"}
        if raw["receipt_blake3"] != blake3_hex(result_core):
            raise AgentJudgeSelectionError("teacher tool result commitment differs")
        results.append(
            ToolResult(**result_core, receipt_blake3=raw["receipt_blake3"])
        )
    trace = ToolTrace(
        results=tuple(results),
        declared_call_ids=tuple(value["declared_call_ids"]),
        joined_call_ids=tuple(value["joined_call_ids"]),
        frontier_count=value["frontier_count"],
        max_parallelism_observed=value["max_parallelism_observed"],
        retry_count=value["retry_count"],
        trace_blake3=value["trace_blake3"],
    )
    if (
        trace.retry_count != 0
        or trace.joined_call_ids != tuple(row.call_id for row in trace.results)
        or set(trace.declared_call_ids) != set(trace.joined_call_ids)
        or len(trace.declared_call_ids) != len(set(trace.declared_call_ids))
    ):
        raise AgentJudgeSelectionError("teacher tool trace execution differs")
    return trace


def _policy_events(value: Any) -> tuple[TrajectoryEvent, ...]:
    if not isinstance(value, list):
        raise AgentJudgeSelectionError("teacher trajectory messages differ")
    events = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise AgentJudgeSelectionError("teacher trajectory event differs")
        events.append(
            TrajectoryEvent(
                event_id=raw["event_id"],
                role=raw["role"],
                content=raw["content"],
                tool_call_ids=tuple(raw["tool_call_ids"]),
                event_blake3=raw["event_blake3"],
            )
        )
    return tuple(events)


def _policy_visible_context(events: tuple[TrajectoryEvent, ...]) -> Mapping[str, Any]:
    for event in events:
        if event.role == "user":
            context = canonical_value(event.content)
            if isinstance(context, Mapping):
                return context
            break
    raise AgentJudgeSelectionError("policy-visible context is not an object")


def reconstruct_evidence(
    trajectory: ValidatedTrajectory,
    *,
    trajectory_path: str | Path,
) -> EvidenceBundle:
    """Reconstruct immutable pipeline evidence without touching a live workspace."""

    value = trajectory.document
    before = _workspace(value["workspace_before"], expected_label="before")
    after = _workspace(value["workspace_after"], expected_label="after")
    events = _policy_events(value["messages"])
    context = _policy_visible_context(events)
    initial_files = {row.path: row.content for row in before.files}
    episode = BenchmarkEpisode(
        episode_id=str(value["executable_episode_id"]),
        source=BenchmarkSource(
            benchmark=str(trajectory.rubric.domain),
            source_file=Path(trajectory_path).name,
            source_revision=trajectory.source_blake3,
        ),
        domain=trajectory.rubric.domain,
        stage=Stage(trajectory.rubric.stage),
        instruction=(
            "Judge the complete retained teacher trajectory and immutable "
            "workspace evidence."
        ),
        policy_context=context,
        initial_files=initial_files,
    )
    manifest = SandboxManifest.create(
        sandbox_id=str(value["sandbox_id"]), episode=episode, rubric=trajectory.rubric
    )
    partial = EvidenceBundle(
        bundle_id=RandomUUIDFactory().new("agent-judge-evidence"),
        rollout_id=RandomUUIDFactory().new("teacher-rollout-reopen"),
        sandbox_manifest=manifest,
        model=ModelTarget(Cohort.STRONG, str(value["model_id"]), str(value["provider"])),
        policy_visible_context=context,
        context_blake3=blake3_hex(context),
        workspace_before=before,
        workspace_after=after,
        tool_trace=_tool_trace(value["tool_trace"]),
        policy_events=events,
        assistant_output=str(value["assistant_output"]),
        provider_receipt_blake3=str(value["provider_receipt_blake3"]),
        safe_provider_metadata=(
            value["provider_metadata"]
            if isinstance(value.get("provider_metadata"), Mapping)
            else {}
        ),
        bundle_blake3="",
    )
    return EvidenceBundle(
        **{
            key: getattr(partial, key)
            for key in partial.__dataclass_fields__
            if key != "bundle_blake3"
        },
        bundle_blake3=blake3_hex(evidence_core(partial)),
    )


def _snapshot_index(snapshot: WorkspaceSnapshot) -> Mapping[str, Any]:
    return {
        "label": snapshot.label,
        "file_count": snapshot.file_count,
        "byte_count": snapshot.byte_count,
        "tree_blake3": snapshot.tree_blake3,
        "files": tuple(
            {
                "path": row.path,
                "byte_count": row.byte_count,
                "mode": row.mode,
                "content_blake3": row.content_blake3,
            }
            for row in snapshot.files
        ),
    }


def _materialize_agent_judge_evidence(evidence: EvidenceBundle) -> EvidenceBundle:
    """Add deterministic mode-0400 sidecars without changing tool schemas."""

    if any(
        row.path.startswith(CODEX_JUDGE_MATERIAL_PREFIX)
        for snapshot in (evidence.workspace_before, evidence.workspace_after)
        for row in snapshot.files
    ):
        raise AgentJudgeSelectionError("reserved Agent Judge workspace path is occupied")
    payloads = {
        ".eva-agent-judge/actor/assistant-output.txt": (
            evidence.assistant_output.encode("utf-8")
        ),
        ".eva-agent-judge/actor/context.json": canonical_json_bytes(
            evidence.policy_visible_context
        ),
        ".eva-agent-judge/actor/messages.json": canonical_json_bytes(
            evidence.policy_events
        ),
        ".eva-agent-judge/actor/tool-trace.json": canonical_json_bytes(
            evidence.tool_trace
        ),
        ".eva-agent-judge/actor/workspace-bindings.json": canonical_json_bytes(
            {
                "before": _snapshot_index(evidence.workspace_before),
                "after": _snapshot_index(evidence.workspace_after),
            }
        ),
    }
    if tuple(sorted(payloads)) != CODEX_JUDGE_REQUIRED_MATERIAL_PATHS:
        raise AgentJudgeSelectionError("Agent Judge material paths differ")
    sidecars = tuple(
        FileSnapshot(
            path=path,
            content=payload,
            byte_count=len(payload),
            mode="0400",
            content_blake3=blake3_bytes(payload),
        )
        for path, payload in sorted(payloads.items())
    )
    files = tuple(sorted((*evidence.workspace_after.files, *sidecars), key=lambda row: row.path))
    core = {
        "files": files,
        "file_count": len(files),
        "byte_count": sum(row.byte_count for row in files),
    }
    after = WorkspaceSnapshot(
        label=evidence.workspace_after.label,
        **core,
        tree_blake3=blake3_hex(core),
    )
    partial = replace(evidence, workspace_after=after, bundle_blake3="")
    return replace(partial, bundle_blake3=blake3_hex(evidence_core(partial)))


def prepare_agent_judge_task(
    task: AgentJudgeTask,
    *,
    registry: CompiledRubricRegistry,
    expected_source_blake3: str | None = None,
    fast_workspace_judge: bool = False,
) -> PreparedAgentJudgeTask:
    if type(fast_workspace_judge) is not bool:
        raise AgentJudgeSelectionError("Agent Judge inspection mode differs")
    if task.judge_route_id != "opus_5" or "opus-5" not in task.judge_model_id.casefold():
        raise AgentJudgeSelectionError("Agent Judge must use the exact Opus 5 route")
    trajectory = validate_full_trajectory(
        task.trajectory_path, registry=registry, expected_task=task
    )
    if (
        expected_source_blake3 is not None
        and trajectory.source_blake3 != expected_source_blake3
    ):
        raise AgentJudgeSelectionError("source trajectory BLAKE3 differs")
    source_evidence = reconstruct_evidence(
        trajectory, trajectory_path=task.trajectory_path
    )
    source_workspace_refs = tuple(
        f"workspace:{name}:{row.path}"
        for name, snapshot in (
            ("before", source_evidence.workspace_before),
            ("after", source_evidence.workspace_after),
        )
        for row in snapshot.files
    )
    source_workspace_byte_count = (
        source_evidence.workspace_before.byte_count
        + source_evidence.workspace_after.byte_count
    )
    evidence = _materialize_agent_judge_evidence(source_evidence)
    # Constructor verification opens every byte commitment without executing a
    # provider or exposing a mutable filesystem root.
    JudgeWorkspaceTools(evidence)
    return PreparedAgentJudgeTask(
        task=task,
        trajectory=trajectory,
        evidence=evidence,
        source_workspace_refs=source_workspace_refs,
        source_workspace_byte_count=source_workspace_byte_count,
        fast_workspace_judge=fast_workspace_judge,
    )


def _workspace_read_coverage(assessment: JudgeAssessment, evidence: EvidenceBundle) -> None:
    """Require complete byte coverage for actor evidence and both workspaces."""

    expected = {
        (snapshot_name, row.path): row.byte_count
        for snapshot_name, snapshot in (
            ("before", evidence.workspace_before),
            ("after", evidence.workspace_after),
        )
        for row in snapshot.files
    }
    material_paths = tuple(
        row.path
        for row in evidence.workspace_after.files
        if row.path.startswith(CODEX_JUDGE_MATERIAL_PREFIX)
    )
    if material_paths != CODEX_JUDGE_REQUIRED_MATERIAL_PATHS:
        raise AgentJudgeSelectionError("Agent Judge evidence material set differs")
    intervals: dict[tuple[str, str], list[tuple[int, int]]] = {
        key: [] for key in expected
    }
    for result in assessment.agent_trace.results:
        if result.status != "completed" or result.name != "workspace_read":
            continue
        arguments = result.arguments
        output = result.output
        if not isinstance(output, Mapping):
            continue
        snapshot = arguments.get("snapshot")
        path = arguments.get("path")
        offset = output.get("offset")
        returned = output.get("returned_bytes")
        key = (snapshot, path)
        total = expected.get(key)
        if (
            key in intervals
            and type(offset) is int
            and type(returned) is int
            and output.get("snapshot") == snapshot
            and output.get("path") == path
            and output.get("total_bytes") == total
            and arguments.get("offset") == offset
            and offset >= 0
            and returned >= 0
            and offset + returned <= total
        ):
            intervals[key].append((offset, offset + returned))
    missing: list[str] = []
    for key, total in expected.items():
        cursor = 0
        for start, end in sorted(intervals[key]):
            if start > cursor:
                break
            cursor = max(cursor, end)
        if cursor < total:
            missing.append(f"workspace:{key[0]}:{key[1]}")
    if missing:
        raise AgentJudgeSelectionError(
            "Agent Judge did not completely read actor evidence and both workspace snapshots"
        )


def _fast_workspace_read_coverage(
    assessment: JudgeAssessment,
    evidence: EvidenceBundle,
    *, maximum_workspace_tool_frontiers: int = 4,
    maximum_provider_turn_count: int | None = None,
) -> None:
    """Require full sidecars plus bounded source inspection for the fast path."""

    if type(maximum_workspace_tool_frontiers) is not int or not 1 <= maximum_workspace_tool_frontiers <= 32:
        raise AgentJudgeSelectionError("fast Agent Judge inspection budget differs")
    provider_turn_limit = (maximum_workspace_tool_frontiers + 1
                           if maximum_provider_turn_count is None else maximum_provider_turn_count)
    if type(provider_turn_limit) is not int or not 1 <= provider_turn_limit <= 65:
        raise AgentJudgeSelectionError("fast Agent Judge provider-turn budget differs")

    expected = {
        (snapshot_name, row.path): row.byte_count
        for snapshot_name, snapshot in (
            ("before", evidence.workspace_before),
            ("after", evidence.workspace_after),
        )
        for row in snapshot.files
    }
    intervals: dict[tuple[str, str], list[tuple[int, int]]] = {
        key: [] for key in expected
    }
    source_inspected = {"before": False, "after": False}
    for result in assessment.agent_trace.results:
        if result.status != "completed" or result.name not in {
            "workspace_read",
            "workspace_search",
        }:
            continue
        arguments = result.arguments
        if result.name == "workspace_search":
            snapshot = arguments.get("snapshot")
            if snapshot in source_inspected:
                source_inspected[snapshot] = any(
                    not ref.startswith(
                        f"workspace:{snapshot}:{CODEX_JUDGE_MATERIAL_PREFIX}"
                    )
                    for ref in result.inspected_evidence_refs
                ) or source_inspected[snapshot]
            continue
        output = result.output
        if not isinstance(output, Mapping):
            continue
        snapshot = arguments.get("snapshot")
        path = arguments.get("path")
        offset = output.get("offset")
        returned = output.get("returned_bytes")
        key = (snapshot, path)
        total = expected.get(key)
        if (
            key in intervals
            and type(offset) is int
            and type(returned) is int
            and output.get("snapshot") == snapshot
            and output.get("path") == path
            and output.get("total_bytes") == total
            and arguments.get("offset") == offset
            and offset >= 0
            and returned >= 0
            and offset + returned <= total
        ):
            intervals[key].append((offset, offset + returned))
            if not str(path).startswith(CODEX_JUDGE_MATERIAL_PREFIX):
                source_inspected[str(snapshot)] = True

    for path in CODEX_JUDGE_REQUIRED_MATERIAL_PATHS:
        key = ("after", path)
        total = expected.get(key)
        if total is None:
            raise AgentJudgeSelectionError("Agent Judge evidence material set differs")
        cursor = 0
        for start, end in sorted(intervals[key]):
            if start > cursor:
                break
            cursor = max(cursor, end)
        if cursor < total:
            raise AgentJudgeSelectionError(
                "fast Agent Judge did not completely read every actor evidence sidecar"
            )
    if source_inspected != {"before": True, "after": True}:
        raise AgentJudgeSelectionError(
            "fast Agent Judge did not inspect source workspace before and after"
        )
    trace = assessment.agent_trace
    if (
        len(trace.results) > 64
        or trace.frontier_count > maximum_workspace_tool_frontiers
        or trace.provider_turn_count > provider_turn_limit
    ):
        raise AgentJudgeSelectionError("fast Agent Judge exceeded its inspection budget")


def judgment_document(
    prepared: PreparedAgentJudgeTask,
    assessment: JudgeAssessment,
    *, maximum_workspace_tool_frontiers: int = 4,
    maximum_provider_turn_count: int | None = None,
) -> Mapping[str, Any]:
    task = prepared.task
    if (
        assessment.judgment_id != task.judge_task_id
        or assessment.judge_model_id != task.judge_model_id
        or assessment.rubric_digest != prepared.trajectory.rubric.digest
        or assessment.agent_trace.retry_count != 0
    ):
        raise AgentJudgeSelectionError("Agent Judge assessment binding differs")
    if prepared.fast_workspace_judge:
        _fast_workspace_read_coverage(assessment, prepared.evidence,
                                     maximum_workspace_tool_frontiers=maximum_workspace_tool_frontiers,
                                     maximum_provider_turn_count=maximum_provider_turn_count)
    else:
        _workspace_read_coverage(assessment, prepared.evidence)
    all_inspected = set(assessment.agent_trace.inspected_evidence_refs)
    inspected = [
        ref
        for ref in assessment.agent_trace.inspected_evidence_refs
        if ref in prepared.source_workspace_refs
    ]
    rows = []
    for score in assessment.item_scores:
        refs = list(score.evidence_refs)
        source_workspace_refs = [
            ref
            for ref in refs
            if ref.startswith(("workspace:before:", "workspace:after:"))
        ]
        if not source_workspace_refs:
            raise AgentJudgeSelectionError("every rubric item must cite workspace evidence")
        if not any(
            ref in all_inspected and ref in prepared.source_workspace_refs
            for ref in source_workspace_refs
        ):
            raise AgentJudgeSelectionError(
                "every rubric item must cite inspected workspace evidence"
            )
        public_refs = [
            ref for ref in refs if ref in prepared.trajectory.evidence_refs
        ]
        rows.append(
            {
                "item_id": score.item_id,
                "score_bps": round(float(score.score) * 10_000),
                "evidence_refs": public_refs,
                "rationale": score.rationale,
            }
        )
    results = assessment.agent_trace.results
    return {
        "schema": "eva.codex-agent-judge-result.v1",
        "judge_task_id": task.judge_task_id,
        "source_task_id": task.source_task_id,
        "source_trajectory_blake3": prepared.trajectory.source_blake3,
        "judge_model_id": task.judge_model_id,
        "inspected_sections": list(REQUIRED_INSPECTION_SECTIONS),
        "inspected_evidence_refs": inspected,
        "judge_tool_trace": {
            "workspace_read_count": sum(
                row.status == "completed" and row.name == "workspace_read"
                for row in results
            ),
            "workspace_search_count": sum(
                row.status == "completed" and row.name == "workspace_search"
                for row in results
            ),
            "provider_turn_count": assessment.agent_trace.provider_turn_count,
            "retry_count": assessment.agent_trace.retry_count,
        },
        "item_scores": rows,
        "summary": assessment.summary,
    }


def execute_agent_judge_task(
    prepared: PreparedAgentJudgeTask,
    *,
    judge: AgentJudgePort,
) -> Mapping[str, Any]:
    """Run exactly one semantic judge attempt; the caller must not retry it."""

    task = prepared.task
    required_workspace_refs = sorted(
        ref for ref in prepared.trajectory.evidence_refs if ref.startswith("workspace:")
    )
    request = JudgeRequest(
        judgment_id=task.judge_task_id,
        judge_model_id=task.judge_model_id,
        policy_visible_context=prepared.evidence.policy_visible_context,
        workspace_evidence=prepared.evidence,
        judge_only_reference={
            "selection_policy": (
                "score the exact compiled rubric without revealing private reasoning"
            ),
            "required_sections": REQUIRED_INSPECTION_SECTIONS,
            "required_complete_workspace_reads": required_workspace_refs,
            "source_trajectory_blake3": prepared.trajectory.source_blake3,
            "semantic_attempt_count": 1,
            "retry_count": 0,
            "fast_workspace_judge": prepared.fast_workspace_judge,
        },
    )
    assessment = judge.judge(request, prepared.trajectory.rubric)
    return judgment_document(prepared, assessment)


def provider_free_preflight(prepared: PreparedAgentJudgeTask) -> Mapping[str, Any]:
    """Reopen one task and prove all byte-bearing judge inputs without a provider."""

    evidence = prepared.evidence
    workspace_refs = sorted(
        f"workspace:{name}:{row.path}"
        for name, snapshot in (
            ("before", evidence.workspace_before),
            ("after", evidence.workspace_after),
        )
        for row in snapshot.files
    )
    initial_index = codex_judge_evidence_index(evidence)
    actor_materials = tuple(
        row
        for row in evidence.workspace_after.files
        if row.path.startswith(CODEX_JUDGE_MATERIAL_PREFIX)
    )
    return {
        "schema": "eva.codex-opus5-agent-judge-worker-preflight.v1",
        "provider_calls": 0,
        "judge_attempts": 0,
        "retry_count": 0,
        "judge_task_id": prepared.task.judge_task_id,
        "source_task_id": prepared.task.source_task_id,
        "source_trajectory_blake3": prepared.trajectory.source_blake3,
        "rubric_digest": prepared.trajectory.rubric.digest,
        "rubric_item_count": len(prepared.trajectory.rubric.items),
        "required_sections": list(REQUIRED_INSPECTION_SECTIONS),
        "required_workspace_refs": workspace_refs,
        "workspace_file_count": len(workspace_refs),
        "workspace_byte_count": prepared.source_workspace_byte_count,
        "actor_evidence_byte_count": sum(row.byte_count for row in actor_materials),
        "turn_mcp_required": True,
        "initial_evidence_index_bytes": len(canonical_json_bytes(initial_index)),
        "large_bytes_embedded_in_initial_turn": False,
        "actor_evidence_snapshot": "after",
        "required_actor_evidence_paths": list(
            CODEX_JUDGE_REQUIRED_MATERIAL_PATHS
        ),
        "one_semantic_attempt": True,
        "complete_before_after_read_required": True,
        "private_reasoning_recorded": False,
    }


__all__ = [
    "AgentJudgePort",
    "PreparedAgentJudgeTask",
    "execute_agent_judge_task",
    "judgment_document",
    "prepare_agent_judge_task",
    "provider_free_preflight",
    "reconstruct_evidence",
]
