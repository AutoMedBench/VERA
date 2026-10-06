"""Candidate-only SFT slicing from completed Codex/MCP teacher trajectories.

This module is intentionally separate from the signed-admission and Opus-judged
SFT paths.  A completed provider turn is not enough: the source task must also
have a committed workspace delta, no failed tool result or unmet gate, and an
exact host-accepted terminal action for its rubric stage.  Rows emitted here
remain selection-pending until the workspace-aware Agent Judge promotes them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
from typing import Any

from eva_agent.codex_runtime import (
    CodexRole,
    codex_turn_receipt_from_document,
    verify_codex_turn_receipt,
)
from eva_agent.pipeline import RandomUUIDFactory, RuntimeIdFactory, uuid_text
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
)
from eva_agent.rubrics import CompiledRubricRegistry

from .agent_judge import AgentJudgeTask, validate_full_trajectory
from .agent_judge_worker import reconstruct_evidence


CANDIDATE_SFT_DATASET_SCHEMA = "eva.execution-verified-candidate-sft-dataset.v1"
CANDIDATE_SFT_SLICE_SCHEMA = "eva.execution-verified-candidate-sft-slice.v1"
CANDIDATE_SFT_VERIFY_SCHEMA = "eva.execution-verified-candidate-sft-verification.v1"
MAXIMUM_SHARD_BYTES = 256 * 1024 * 1024
DEFAULT_CANDIDATE_ROUTES = (
    "gpt_6_astra",
    "gpt_6_astra_openai",
    "gpt_6_astra_azure",
    "gpt_5_6_sol",
    "gpt_5_6_terra",
    "gpt_5_6_luna",
    "gemini_3_8_flash",
    "qwen_3_5_122b_a10b",
    "qwen_3_5_397b_a17b",
)
_STAGE_COMPLETION_TOOL = {
    "S1": "materialize_plan",
    "S2": "materialize_evidence_selection",
    "S3": "execute_code",
    "S4": "execute_code",
    "S5": "submit_results",
    "E2E": "submit_results",
}
_BLOCKED_TERMINAL = re.compile(
    r"\b(?:blocked|not completed|not complete|incomplete|cannot complete|"
    r"could not complete|unable to complete|did not successfully|"
    r"no artifact (?:was )?submitted|no submission (?:was )?accepted|"
    r"contract not completed)\b",
    re.IGNORECASE,
)


class ExecutionVerifiedSFTError(ValueError):
    """A source, candidate slice, or dataset commitment differs."""


@dataclass(frozen=True)
class ExecutionVerifiedSFTBuild:
    dataset_id: str
    dataset_root: Path
    scanned_trajectories: int
    source_count: int
    slice_count: int
    shard_count: int
    rejected_count: int
    manifest_blake3: str


@dataclass(frozen=True)
class _SourceTask:
    source_root: Path
    source_root_relative: str
    trajectory_path: Path
    trajectory_relative: str
    task_id: str
    sandbox_id: str
    candidate_id: str
    route_id: str
    stage: str
    domain: str


@dataclass(frozen=True)
class _AcceptedSource:
    task: _SourceTask
    document: Mapping[str, Any]
    source_blake3: str
    rubric_id: str
    rubric_digest: str
    rubric_item_count: int
    codex_turn_receipt_blake3: str
    tool_trace_blake3: str
    workspace_before_blake3: str
    workspace_after_blake3: str
    completion_result_id: str
    completion_receipt_blake3: str
    completion_artifact_path: str
    messages: tuple[Mapping[str, Any], ...]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionVerifiedSFTError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload, object_pairs_hook=_strict_object)
    except ExecutionVerifiedSFTError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise ExecutionVerifiedSFTError("candidate SFT JSON differs") from None
    if not isinstance(value, Mapping):
        raise ExecutionVerifiedSFTError("candidate SFT JSON is not an object")
    return value, payload


def _relative_directory(path: Path, *, base: Path, label: str) -> str:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise ExecutionVerifiedSFTError(f"{label} topology differs")
    try:
        relative = resolved.relative_to(base).as_posix()
    except ValueError:
        raise ExecutionVerifiedSFTError(f"{label} is outside source base") from None
    if relative in {"", "."}:
        raise ExecutionVerifiedSFTError(f"{label} cannot equal source base")
    return relative


def _discover_source_tasks(
    source_roots: Sequence[Path], *, source_base: Path
) -> tuple[_SourceTask, ...]:
    tasks: list[_SourceTask] = []
    seen: set[str] = set()
    for requested_root in source_roots:
        root = requested_root.resolve(strict=True)
        root_relative = _relative_directory(root, base=source_base, label="teacher root")
        checkpoint = root / "checkpoint.sqlite3"
        info = checkpoint.lstat()
        if checkpoint.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise ExecutionVerifiedSFTError("teacher checkpoint topology differs")
        with sqlite3.connect(
            f"file:{checkpoint}?mode=ro", uri=True, timeout=30
        ) as database:
            rows = database.execute(
                """SELECT task_id,sandbox_id,candidate_id,route_id,stage,domain,
                result_path FROM tasks WHERE state='succeeded' ORDER BY rowid"""
            ).fetchall()
        for task_id, sandbox_id, candidate_id, route_id, stage, domain, result in rows:
            if task_id in seen:
                raise ExecutionVerifiedSFTError("teacher task identity is duplicated")
            seen.add(task_id)
            expected = (
                root / "trajectories" / sandbox_id / route_id / "result.json"
            ).resolve(strict=True)
            observed = Path(result)
            if observed.is_absolute():
                resolved = observed.resolve(strict=True)
            else:
                if any(part in {"", ".", ".."} for part in observed.parts):
                    raise ExecutionVerifiedSFTError(
                        "teacher result topology differs"
                    )
                resolved = source_base.joinpath(*observed.parts).resolve(strict=True)
            if (
                resolved != expected
                or expected.is_symlink()
                or not expected.is_file()
                or task_id != f"{sandbox_id}--{route_id}"
            ):
                raise ExecutionVerifiedSFTError("teacher result topology differs")
            try:
                trajectory_relative = expected.relative_to(source_base).as_posix()
            except ValueError:
                raise ExecutionVerifiedSFTError(
                    "teacher trajectory is outside source base"
                ) from None
            tasks.append(
                _SourceTask(
                    source_root=root,
                    source_root_relative=root_relative,
                    trajectory_path=expected,
                    trajectory_relative=trajectory_relative,
                    task_id=str(task_id),
                    sandbox_id=str(sandbox_id),
                    candidate_id=str(candidate_id),
                    route_id=str(route_id),
                    stage=str(stage),
                    domain=str(domain),
                )
            )
    return tuple(tasks)


def _contains_failed_gate(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("gate_passed") is False:
            return True
        error = value.get("error")
        if error is not None and error != "":
            return True
        failed = value.get("failed_check_ids")
        if isinstance(failed, (list, tuple)) and bool(failed):
            return True
        return any(_contains_failed_gate(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_failed_gate(item) for item in value)
    return False


def _completion_output_matches(output: Any, *, stage: str) -> bool:
    expected_stage = "S5" if stage == "E2E" else stage
    error = output.get("error") if isinstance(output, Mapping) else None
    return (
        isinstance(output, Mapping)
        and output.get("gate_passed") is True
        and output.get("stage") == expected_stage
        and (error is None or error == "")
        and not output.get("failed_check_ids")
    )


def _policy_event_tool_joins(
    messages: Sequence[Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    seen_calls: set[str] = set()
    observations: dict[str, Mapping[str, Any]] = {}
    for index, event in enumerate(messages):
        call_ids = event.get("tool_call_ids")
        if not isinstance(call_ids, (list, tuple)):
            raise ExecutionVerifiedSFTError("trajectory tool-call inventory differs")
        if event.get("role") != "assistant" or not call_ids:
            continue
        if any(
            not isinstance(call_id, str)
            or not call_id
            or call_id in seen_calls
            for call_id in call_ids
        ):
            raise ExecutionVerifiedSFTError("trajectory tool-call identity differs")
        seen_calls.update(call_ids)
        joined: dict[str, Mapping[str, Any]] = {}
        for later in messages[index + 1 :]:
            if later.get("role") == "assistant":
                break
            later_ids = later.get("tool_call_ids")
            if (
                later.get("role") == "tool"
                and isinstance(later_ids, (list, tuple))
                and len(later_ids) == 1
                and later_ids[0] in call_ids
            ):
                if later_ids[0] in joined or later_ids[0] in observations:
                    raise ExecutionVerifiedSFTError(
                        "trajectory tool observation is duplicated"
                    )
                joined[later_ids[0]] = later
        if set(joined) != set(call_ids):
            raise ExecutionVerifiedSFTError(
                "trajectory assistant decision omits a tool observation"
            )
        observations.update(joined)
    return observations


def _declared_completion_artifact_path(
    messages: Sequence[Mapping[str, Any]], *, stage: str
) -> str | None:
    expected_stage = "S5" if stage == "E2E" else stage
    for event in messages:
        if event.get("role") != "user" or not isinstance(event.get("content"), Mapping):
            continue
        binding = event["content"].get("execution_binding")
        stages = binding.get("execution_stages") if isinstance(binding, Mapping) else None
        stage_contract = stages.get(expected_stage) if isinstance(stages, Mapping) else None
        value = (
            stage_contract.get("artifact_relative_path")
            if isinstance(stage_contract, Mapping)
            else None
        )
        if not isinstance(value, str) or not value:
            continue
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            return None
        return value
    return None


def _workspace_file_digests(value: Any) -> Mapping[str, str] | None:
    files = value.get("files") if isinstance(value, Mapping) else None
    if not isinstance(files, list):
        return None
    result: dict[str, str] = {}
    for row in files:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("content_blake3"), str)
            or row["path"] in result
        ):
            return None
        result[row["path"]] = row["content_blake3"]
    return result


def _cheap_rejection_reason(
    task: _SourceTask, value: Mapping[str, Any]
) -> str | None:
    """Reject obvious non-completions before decoding very large Codex receipts."""

    if (
        value.get("schema") != "eva.codex-teacher-full-trajectory.v1"
        or value.get("task_id") != task.task_id
        or value.get("sandbox_id") != task.sandbox_id
        or value.get("candidate_id") != task.candidate_id
        or value.get("route_id") != task.route_id
    ):
        return "source_identity_differs"
    if (
        value.get("selection_pending") is not True
        or value.get("score_kind") != "selection_pending"
        or value.get("semantic_retry_count") != 0
        or value.get("judge_calls") != 0
        or value.get("admission_writes") != 0
    ):
        return "teacher_selection_state_differs"
    if _BLOCKED_TERMINAL.search(str(value.get("assistant_output") or "")):
        return "blocked_or_incomplete_terminal_response"
    trace = value.get("tool_trace")
    results = trace.get("results") if isinstance(trace, Mapping) else None
    if not isinstance(results, list) or not results:
        return "task_has_no_host_tool_results"
    if any(
        not isinstance(result, Mapping) or result.get("status") != "completed"
        for result in results
    ):
        return "immutable_or_failed_tool_result"
    if any(
        result.get("error_code") is not None
        or _contains_failed_gate(result.get("output"))
        for result in results
    ):
        return "unmet_tool_gate"
    before = value.get("workspace_before")
    after = value.get("workspace_after")
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or before.get("tree_blake3") == after.get("tree_blake3")
    ):
        return "workspace_unchanged"
    completion_tool = _STAGE_COMPLETION_TOOL.get(task.stage)
    completion = tuple(
        result
        for result in results
        if result.get("name") == completion_tool
        and _completion_output_matches(result.get("output"), stage=task.stage)
    )
    if len(completion) != 1:
        return "task_stage_not_completed"
    messages = value.get("messages")
    if not isinstance(messages, list):
        return "source_messages_differ"
    artifact_path = _declared_completion_artifact_path(messages, stage=task.stage)
    before_files = _workspace_file_digests(before)
    after_files = _workspace_file_digests(after)
    if (
        artifact_path is None
        or before_files is None
        or after_files is None
        or artifact_path not in after_files
        or before_files.get(artifact_path) == after_files.get(artifact_path)
    ):
        return "task_stage_artifact_not_materialized"
    return None


def _inspect_source(
    task: _SourceTask,
    *,
    registry: CompiledRubricRegistry,
    allowed_routes: frozenset[str],
) -> tuple[_AcceptedSource | None, str | None]:
    if task.route_id not in allowed_routes:
        return None, "route_not_selected"
    expected = AgentJudgeTask(
        judge_task_id="00000000-0000-4000-8000-000000000000",
        source_task_id=task.task_id,
        sandbox_id=task.sandbox_id,
        candidate_id=task.candidate_id,
        actor_route_id=task.route_id,
        stage=task.stage,
        domain=task.domain,
        trajectory_path=str(task.trajectory_path),
        judge_route_id="opus_5",
        judge_model_id="opus-5-agent-judge",
    )
    try:
        raw, _payload = _read_json(task.trajectory_path)
        cheap_reason = _cheap_rejection_reason(task, raw)
        if cheap_reason is not None:
            return None, cheap_reason
        trajectory = validate_full_trajectory(
            task.trajectory_path, registry=registry, expected_task=expected
        )
        evidence = reconstruct_evidence(
            trajectory, trajectory_path=task.trajectory_path
        )
        document = trajectory.document
        metadata = document.get("provider_metadata")
        if not isinstance(metadata, Mapping):
            raise ExecutionVerifiedSFTError("provider metadata differs")
        receipt_document = metadata.get("codex_turn_receipt")
        if not isinstance(receipt_document, Mapping):
            raise ExecutionVerifiedSFTError("Codex turn receipt is absent")
        receipt = codex_turn_receipt_from_document(receipt_document)
        verify_codex_turn_receipt(receipt)
        if (
            receipt.status != "completed"
            or receipt.role is not CodexRole.STRONG_ACTOR
            or receipt.thread_resumed
            or receipt.visibility != "actor-public"
            or receipt.model != document.get("model_id")
            or receipt.provider != document.get("provider")
            or receipt.final_response != document.get("assistant_output")
        ):
            return None, "codex_turn_not_execution_complete"
        if any(call.status != "completed" for call in receipt.tool_calls):
            return None, "codex_tool_call_not_completed"
        results = evidence.tool_trace.results
        if not results:
            return None, "task_has_no_host_tool_results"
        if any(result.status != "completed" for result in results):
            return None, "immutable_or_failed_tool_result"
        if any(
            result.error_code is not None or _contains_failed_gate(result.output)
            for result in results
        ):
            return None, "unmet_tool_gate"
        if evidence.workspace_before.tree_blake3 == evidence.workspace_after.tree_blake3:
            return None, "workspace_unchanged"
        completion_tool = _STAGE_COMPLETION_TOOL.get(task.stage)
        completion = tuple(
            result
            for result in results
            if result.name == completion_tool
            and _completion_output_matches(result.output, stage=task.stage)
        )
        if len(completion) != 1:
            return None, "task_stage_not_completed"
        messages = tuple(document["messages"])
        _policy_event_tool_joins(messages)
        artifact_path = _declared_completion_artifact_path(
            messages, stage=task.stage
        )
        before_files = {row.path: row.content_blake3 for row in evidence.workspace_before.files}
        after_files = {row.path: row.content_blake3 for row in evidence.workspace_after.files}
        if (
            artifact_path is None
            or artifact_path not in after_files
            or before_files.get(artifact_path) == after_files.get(artifact_path)
        ):
            return None, "task_stage_artifact_not_materialized"
        return (
            _AcceptedSource(
                task=task,
                document=document,
                source_blake3=trajectory.source_blake3,
                rubric_id=trajectory.rubric.rubric_id,
                rubric_digest=trajectory.rubric.digest,
                rubric_item_count=len(trajectory.rubric.items),
                codex_turn_receipt_blake3=receipt.receipt_blake3,
                tool_trace_blake3=evidence.tool_trace.trace_blake3,
                workspace_before_blake3=evidence.workspace_before.tree_blake3,
                workspace_after_blake3=evidence.workspace_after.tree_blake3,
                completion_result_id=completion[0].result_id,
                completion_receipt_blake3=completion[0].receipt_blake3,
                completion_artifact_path=artifact_path,
                messages=messages,
            ),
            None,
        )
    except Exception as exc:
        return None, f"source_validation_failed:{type(exc).__name__}"


def _target_observations(
    messages: Sequence[Mapping[str, Any]], index: int
) -> tuple[Mapping[str, Any], ...]:
    target = messages[index]
    call_ids = tuple(target.get("tool_call_ids", ()))
    if not call_ids:
        return ()
    by_id: dict[str, Mapping[str, Any]] = {}
    for later in messages[index + 1 :]:
        if later.get("role") == "assistant":
            break
        later_ids = later.get("tool_call_ids")
        if (
            later.get("role") == "tool"
            and isinstance(later_ids, (list, tuple))
            and len(later_ids) == 1
            and later_ids[0] in call_ids
        ):
            by_id[later_ids[0]] = later
    if set(by_id) != set(call_ids):
        raise ExecutionVerifiedSFTError("candidate target tool observations differ")
    return tuple(by_id[call_id] for call_id in call_ids)


def _slice_source(
    source: _AcceptedSource,
    *,
    dataset_id: str,
    id_factory: RuntimeIdFactory,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(source.messages):
        if target.get("role") != "assistant":
            continue
        call_ids = tuple(target.get("tool_call_ids", ()))
        observations = _target_observations(source.messages, index)
        example_id = uuid_text(
            id_factory.new("execution-verified-candidate-sft-slice"),
            label="candidate SFT example_id",
        )
        core = {
            "schema": CANDIDATE_SFT_SLICE_SCHEMA,
            "dataset_id": dataset_id,
            "example_id": example_id,
            "quality_tier": "execution_verified_candidate",
            "selection_status": "pending_workspace_agent_judge",
            "strict_sft_eligible": False,
            "agent_judged": False,
            "source_task_id": source.task.task_id,
            "sandbox_id": source.task.sandbox_id,
            "candidate_id": source.task.candidate_id,
            "route_id": source.task.route_id,
            "model_id": source.document["model_id"],
            "domain": source.task.domain,
            "stage": source.task.stage,
            "decision_ordinal": len(rows),
            "decision_event_id": target["event_id"],
            "prefix": source.messages[:index],
            "supervised_assistant_decision": target,
            "target_tool_observations": observations,
            "atomic_multi_tool_target": len(call_ids) > 1,
            "rubric_binding": {
                "rubric_id": source.rubric_id,
                "rubric_digest": source.rubric_digest,
                "item_count": source.rubric_item_count,
                "domain": source.task.domain,
                "stage": source.task.stage,
            },
            "execution_verification": {
                "codex_turn_status": "completed",
                "codex_turn_receipt_blake3": source.codex_turn_receipt_blake3,
                "tool_trace_blake3": source.tool_trace_blake3,
                "all_tool_results_completed": True,
                "unmet_gate_count": 0,
                "task_stage_completed": True,
                "completion_result_id": source.completion_result_id,
                "completion_receipt_blake3": source.completion_receipt_blake3,
                "completion_artifact_path": source.completion_artifact_path,
                "workspace_before_blake3": source.workspace_before_blake3,
                "workspace_after_blake3": source.workspace_after_blake3,
                "workspace_changed": True,
            },
            "provenance": {
                "source_root": source.task.source_root_relative,
                "source_trajectory_path": source.task.trajectory_relative,
                "source_trajectory_blake3": source.source_blake3,
            },
            "hidden_reasoning_included": False,
            "private_reference_included": False,
        }
        rows.append({**core, "slice_blake3": blake3_hex(core)})
    if not rows:
        raise ExecutionVerifiedSFTError("completed source has no assistant decision")
    return tuple(rows)


def _write_once(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short candidate SFT artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def _accepted_source_row(source: _AcceptedSource, *, slice_count: int) -> dict[str, Any]:
    return {
        "source_task_id": source.task.task_id,
        "sandbox_id": source.task.sandbox_id,
        "candidate_id": source.task.candidate_id,
        "route_id": source.task.route_id,
        "model_id": source.document["model_id"],
        "domain": source.task.domain,
        "stage": source.task.stage,
        "source_root": source.task.source_root_relative,
        "source_trajectory_path": source.task.trajectory_relative,
        "source_trajectory_blake3": source.source_blake3,
        "rubric_id": source.rubric_id,
        "rubric_digest": source.rubric_digest,
        "codex_turn_receipt_blake3": source.codex_turn_receipt_blake3,
        "tool_trace_blake3": source.tool_trace_blake3,
        "workspace_before_blake3": source.workspace_before_blake3,
        "workspace_after_blake3": source.workspace_after_blake3,
        "completion_artifact_path": source.completion_artifact_path,
        "slice_count": slice_count,
    }


def _inspect_all(
    tasks: Sequence[_SourceTask],
    *,
    registry: CompiledRubricRegistry,
    allowed_routes: frozenset[str],
    workers: int,
) -> tuple[
    tuple[_AcceptedSource, ...],
    Counter[str],
    Mapping[str, Mapping[str, int]],
    Mapping[str, Mapping[str, int]],
]:
    def inspect(task: _SourceTask) -> tuple[_AcceptedSource | None, str | None]:
        return _inspect_source(task, registry=registry, allowed_routes=allowed_routes)

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(tasks)))) as pool:
        inspected = tuple(pool.map(inspect, tasks))
    accepted = tuple(item for item, _reason in inspected if item is not None)
    rejected = Counter(
        str(reason) for item, reason in inspected if item is None and reason is not None
    )
    rejected_by_route: dict[str, Counter[str]] = {}
    rejected_by_source_root: dict[str, Counter[str]] = {}
    for task, (item, reason) in zip(tasks, inspected, strict=True):
        if item is not None or reason is None:
            continue
        rejected_by_route.setdefault(task.route_id, Counter())[str(reason)] += 1
        rejected_by_source_root.setdefault(
            task.source_root_relative, Counter()
        )[str(reason)] += 1
    if len(accepted) + sum(rejected.values()) != len(tasks):
        raise ExecutionVerifiedSFTError("candidate SFT inspection totals differ")
    return (
        accepted,
        rejected,
        {
            key: dict(sorted(counts.items()))
            for key, counts in sorted(rejected_by_route.items())
        },
        {
            key: dict(sorted(counts.items()))
            for key, counts in sorted(rejected_by_source_root.items())
        },
    )


def build_execution_verified_candidate_sft_dataset(
    *,
    source_roots: Sequence[Path],
    source_base: Path,
    output_root: Path,
    registry: CompiledRubricRegistry,
    id_factory: RuntimeIdFactory | None = None,
    eligible_routes: Sequence[str] = DEFAULT_CANDIDATE_ROUTES,
    shard_size: int = 512,
    verification_workers: int = 64,
) -> ExecutionVerifiedSFTBuild:
    """Reopen completed teacher evidence and publish only clean candidate slices."""

    if not source_roots:
        raise ExecutionVerifiedSFTError("candidate SFT source roots are empty")
    if type(shard_size) is not int or not 1 <= shard_size <= 100_000:
        raise ExecutionVerifiedSFTError("candidate SFT shard size differs")
    if (
        type(verification_workers) is not int
        or not 1 <= verification_workers <= 256
    ):
        raise ExecutionVerifiedSFTError("candidate SFT verification width differs")
    routes = tuple(sorted(set(eligible_routes)))
    if not routes or len(routes) != len(tuple(eligible_routes)):
        raise ExecutionVerifiedSFTError("candidate SFT route selection differs")
    base = Path(source_base).resolve(strict=True)
    tasks = _discover_source_tasks(source_roots, source_base=base)
    accepted, rejected, rejected_by_route, rejected_by_source_root = _inspect_all(
        tasks,
        registry=registry,
        allowed_routes=frozenset(routes),
        workers=verification_workers,
    )
    ids = id_factory or RandomUUIDFactory()
    dataset_id = uuid_text(
        ids.new("execution-verified-candidate-sft-dataset"),
        label="candidate SFT dataset_id",
    )
    output = Path(output_root)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.is_symlink() or not output.is_dir():
        raise ExecutionVerifiedSFTError("candidate SFT output root differs")
    dataset_root = output / dataset_id
    dataset_root.mkdir(mode=0o700)
    shard_root = dataset_root / "shards"
    shard_root.mkdir(mode=0o700)

    source_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    pending: list[bytes] = []
    pending_ids: list[str] = []
    pending_size = 0
    slice_count = 0

    def flush() -> None:
        nonlocal pending, pending_ids, pending_size
        if not pending:
            return
        ordinal = len(shard_rows)
        name = f"part-{ordinal:05d}.jsonl"
        payload = b"".join(pending)
        _write_once(shard_root / name, payload)
        shard_rows.append(
            {
                "path": f"shards/{name}",
                "slice_count": len(pending),
                "first_example_id": pending_ids[0],
                "last_example_id": pending_ids[-1],
                "content_blake3": blake3_bytes(payload),
                "byte_count": len(payload),
            }
        )
        pending = []
        pending_ids = []
        pending_size = 0

    for source in accepted:
        rows = _slice_source(source, dataset_id=dataset_id, id_factory=ids)
        source_rows.append(_accepted_source_row(source, slice_count=len(rows)))
        for row in rows:
            encoded = canonical_json_bytes(row)
            if len(encoded) > MAXIMUM_SHARD_BYTES:
                raise ExecutionVerifiedSFTError("one candidate SFT row is too large")
            if pending and (
                len(pending) >= shard_size
                or pending_size + len(encoded) > MAXIMUM_SHARD_BYTES
            ):
                flush()
            pending.append(encoded)
            pending_ids.append(row["example_id"])
            pending_size += len(encoded)
            slice_count += 1
            stage_counts[row["stage"]] += 1
            route_counts[row["route_id"]] += 1
            domain_counts[row["domain"]] += 1
    flush()

    roots = tuple(
        sorted(
            {
                _relative_directory(Path(root), base=base, label="teacher root")
                for root in source_roots
            }
        )
    )
    manifest_core = {
        "schema": CANDIDATE_SFT_DATASET_SCHEMA,
        "dataset_id": dataset_id,
        "quality_tier": "execution_verified_candidate",
        "selection_status": "pending_workspace_agent_judge",
        "strict_sft_eligible": False,
        "source_base_label": base.name,
        "source_roots": roots,
        "rubric_registry_digest": registry.digest,
        "selection": {
            "eligible_routes": routes,
            "require_checkpoint_succeeded": True,
            "require_completed_codex_turn": True,
            "require_all_tool_results_completed": True,
            "reject_immutable_failure": True,
            "reject_unmet_gate": True,
            "require_exact_task_stage_completion": True,
            "require_workspace_change": True,
            "require_agent_judge_for_strict_sft": True,
            "hidden_reasoning_allowed": False,
            "private_reference_allowed": False,
        },
        "inventory": {
            "scanned_completed_teacher_trajectories": len(tasks),
            "execution_verified_candidate_trajectories": len(accepted),
            "rejected_trajectories": sum(rejected.values()),
            "rejected_by_reason": dict(sorted(rejected.items())),
            "rejected_by_route": rejected_by_route,
            "rejected_by_source_root": rejected_by_source_root,
            "provider_calls": 0,
            "agent_judge_calls": 0,
        },
        "source_count": len(source_rows),
        "slice_count": slice_count,
        "shard_count": len(shard_rows),
        "counts_by_stage": dict(sorted(stage_counts.items())),
        "counts_by_route": dict(sorted(route_counts.items())),
        "counts_by_domain": dict(sorted(domain_counts.items())),
        "sources": tuple(source_rows),
        "shards": tuple(shard_rows),
        "agent_judged": False,
        "private_reference_included": False,
        "hidden_reasoning_included": False,
    }
    manifest = {**manifest_core, "manifest_blake3": blake3_hex(manifest_core)}
    _write_once(dataset_root / "manifest.json", canonical_json_bytes(manifest))
    for directory in (shard_root, dataset_root):
        os.chmod(directory, 0o555)
    return ExecutionVerifiedSFTBuild(
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        scanned_trajectories=len(tasks),
        source_count=len(accepted),
        slice_count=slice_count,
        shard_count=len(shard_rows),
        rejected_count=sum(rejected.values()),
        manifest_blake3=manifest["manifest_blake3"],
    )


def _safe_shard_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str):
        raise ExecutionVerifiedSFTError("candidate SFT shard path differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] != "shards"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ExecutionVerifiedSFTError("candidate SFT shard path differs")
    return path


def verify_execution_verified_candidate_sft_dataset(
    *,
    dataset_root: Path,
    source_base: Path,
    registry: CompiledRubricRegistry,
    verification_workers: int = 64,
) -> Mapping[str, Any]:
    """Independently reopen sources, execution evidence, slices, and aggregates."""

    checks: list[str] = []
    observed_slices = 0
    dataset_id: str | None = None
    manifest_blake3: str | None = None
    try:
        root = Path(dataset_root).resolve(strict=True)
        base = Path(source_base).resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise ExecutionVerifiedSFTError("candidate SFT dataset topology differs")
        manifest, manifest_bytes = _read_json(root / "manifest.json")
        if canonical_json_bytes(manifest) != manifest_bytes:
            raise ExecutionVerifiedSFTError("candidate SFT manifest is not canonical")
        dataset_id = uuid_text(
            manifest.get("dataset_id"), label="candidate SFT dataset_id"
        )
        if (
            root.name != dataset_id
            or manifest.get("schema") != CANDIDATE_SFT_DATASET_SCHEMA
            or manifest.get("quality_tier") != "execution_verified_candidate"
            or manifest.get("selection_status")
            != "pending_workspace_agent_judge"
            or manifest.get("strict_sft_eligible") is not False
            or manifest.get("agent_judged") is not False
            or manifest.get("private_reference_included") is not False
            or manifest.get("hidden_reasoning_included") is not False
            or manifest.get("rubric_registry_digest") != registry.digest
        ):
            raise ExecutionVerifiedSFTError("candidate SFT manifest identity differs")
        manifest_core = {
            key: value for key, value in manifest.items() if key != "manifest_blake3"
        }
        manifest_blake3 = str(manifest.get("manifest_blake3"))
        if manifest_blake3 != blake3_hex(manifest_core):
            raise ExecutionVerifiedSFTError("candidate SFT manifest commitment differs")
        selection = manifest.get("selection")
        inventory = manifest.get("inventory")
        routes = selection.get("eligible_routes") if isinstance(selection, Mapping) else None
        roots = manifest.get("source_roots")
        if (
            not isinstance(routes, list)
            or not routes
            or routes != sorted(set(routes))
            or not isinstance(roots, list)
            or not roots
            or roots != sorted(set(roots))
            or not isinstance(inventory, Mapping)
        ):
            raise ExecutionVerifiedSFTError("candidate SFT selection differs")
        source_roots = []
        for relative in roots:
            path = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath("/")
            if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                raise ExecutionVerifiedSFTError("candidate SFT source root differs")
            resolved = base.joinpath(*path.parts).resolve(strict=True)
            if not resolved.is_relative_to(base):
                raise ExecutionVerifiedSFTError("candidate SFT source root escapes")
            source_roots.append(resolved)
        tasks = _discover_source_tasks(source_roots, source_base=base)
        accepted, rejected, rejected_by_route, rejected_by_source_root = _inspect_all(
            tasks,
            registry=registry,
            allowed_routes=frozenset(routes),
            workers=verification_workers,
        )
        recomputed_by_path = {
            source.task.trajectory_relative: source for source in accepted
        }
        declared_sources = manifest.get("sources")
        if not isinstance(declared_sources, list):
            raise ExecutionVerifiedSFTError("candidate SFT source inventory differs")
        declared_by_path: dict[str, Mapping[str, Any]] = {}
        for row in declared_sources:
            if not isinstance(row, Mapping):
                raise ExecutionVerifiedSFTError("candidate SFT source row differs")
            path = row.get("source_trajectory_path")
            if not isinstance(path, str) or path in declared_by_path:
                raise ExecutionVerifiedSFTError("candidate SFT source path differs")
            declared_by_path[path] = row
        if set(declared_by_path) != set(recomputed_by_path):
            raise ExecutionVerifiedSFTError("candidate SFT accepted sources differ")
        for path, source in recomputed_by_path.items():
            expected = _accepted_source_row(
                source,
                slice_count=sum(
                    event.get("role") == "assistant" for event in source.messages
                ),
            )
            if canonical_value(declared_by_path[path]) != canonical_value(expected):
                raise ExecutionVerifiedSFTError("candidate SFT source commitment differs")
        expected_rejections = dict(sorted(rejected.items()))
        if (
            inventory.get("scanned_completed_teacher_trajectories") != len(tasks)
            or inventory.get("execution_verified_candidate_trajectories")
            != len(accepted)
            or inventory.get("rejected_trajectories") != sum(rejected.values())
            or inventory.get("rejected_by_reason") != expected_rejections
            or inventory.get("rejected_by_route") != rejected_by_route
            or inventory.get("rejected_by_source_root")
            != rejected_by_source_root
            or inventory.get("provider_calls") != 0
            or inventory.get("agent_judge_calls") != 0
        ):
            raise ExecutionVerifiedSFTError("candidate SFT inventory differs")
        checks.extend(
            (
                "completed_teacher_checkpoint_bindings",
                "codex_turn_receipts",
                "tool_result_and_gate_integrity",
                "workspace_commitments",
                "exact_task_stage_completion",
            )
        )

        expected_files = {"manifest.json"}
        seen_ids: set[str] = set()
        seen_decisions: set[tuple[str, str]] = set()
        stage_counts: Counter[str] = Counter()
        route_counts: Counter[str] = Counter()
        domain_counts: Counter[str] = Counter()
        source_slice_counts: Counter[str] = Counter()
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise ExecutionVerifiedSFTError("candidate SFT shard inventory differs")
        for descriptor in shards:
            if not isinstance(descriptor, Mapping):
                raise ExecutionVerifiedSFTError("candidate SFT shard descriptor differs")
            relative = _safe_shard_path(descriptor.get("path"))
            expected_files.add(relative.as_posix())
            shard_path = root.joinpath(*relative.parts)
            info = shard_path.lstat()
            if (
                shard_path.is_symlink()
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
            ):
                raise ExecutionVerifiedSFTError("candidate SFT shard topology differs")
            payload = shard_path.read_bytes()
            if (
                len(payload) != descriptor.get("byte_count")
                or len(payload) > MAXIMUM_SHARD_BYTES
                or blake3_bytes(payload) != descriptor.get("content_blake3")
            ):
                raise ExecutionVerifiedSFTError("candidate SFT shard commitment differs")
            rows: list[Mapping[str, Any]] = []
            for line in payload.splitlines():
                row = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(row, Mapping):
                    raise ExecutionVerifiedSFTError("candidate SFT slice differs")
                if (
                    row.get("schema") != CANDIDATE_SFT_SLICE_SCHEMA
                    or row.get("dataset_id") != dataset_id
                    or row.get("quality_tier") != "execution_verified_candidate"
                    or row.get("selection_status")
                    != "pending_workspace_agent_judge"
                    or row.get("strict_sft_eligible") is not False
                    or row.get("agent_judged") is not False
                    or row.get("hidden_reasoning_included") is not False
                    or row.get("private_reference_included") is not False
                ):
                    raise ExecutionVerifiedSFTError("candidate SFT slice identity differs")
                example_id = uuid_text(
                    row.get("example_id"), label="candidate SFT example_id"
                )
                if example_id in seen_ids:
                    raise ExecutionVerifiedSFTError(
                        "candidate SFT example identity is duplicated"
                    )
                seen_ids.add(example_id)
                core = {
                    key: value for key, value in row.items() if key != "slice_blake3"
                }
                if row.get("slice_blake3") != blake3_hex(core):
                    raise ExecutionVerifiedSFTError("candidate SFT slice commitment differs")
                provenance = row.get("provenance")
                source_path = (
                    provenance.get("source_trajectory_path")
                    if isinstance(provenance, Mapping)
                    else None
                )
                source = recomputed_by_path.get(str(source_path))
                if source is None:
                    raise ExecutionVerifiedSFTError("candidate SFT source binding differs")
                event_id = row.get("decision_event_id")
                pair = (str(source_path), str(event_id))
                if pair in seen_decisions:
                    raise ExecutionVerifiedSFTError(
                        "candidate SFT decision is duplicated"
                    )
                seen_decisions.add(pair)
                try:
                    index = next(
                        index
                        for index, event in enumerate(source.messages)
                        if event.get("event_id") == event_id
                    )
                except StopIteration:
                    raise ExecutionVerifiedSFTError(
                        "candidate SFT target event is absent"
                    ) from None
                target = source.messages[index]
                observations = _target_observations(source.messages, index)
                if (
                    target.get("role") != "assistant"
                    or row.get("decision_ordinal")
                    != sum(
                        event.get("role") == "assistant"
                        for event in source.messages[:index]
                    )
                    or canonical_value(row.get("prefix"))
                    != canonical_value(source.messages[:index])
                    or canonical_value(row.get("supervised_assistant_decision"))
                    != canonical_value(target)
                    or canonical_value(row.get("target_tool_observations"))
                    != canonical_value(observations)
                    or row.get("atomic_multi_tool_target")
                    is not (len(tuple(target.get("tool_call_ids", ()))) > 1)
                ):
                    raise ExecutionVerifiedSFTError("candidate SFT target projection differs")
                verification = row.get("execution_verification")
                rubric = row.get("rubric_binding")
                if (
                    not isinstance(verification, Mapping)
                    or verification.get("codex_turn_status") != "completed"
                    or verification.get("all_tool_results_completed") is not True
                    or verification.get("unmet_gate_count") != 0
                    or verification.get("task_stage_completed") is not True
                    or verification.get("workspace_changed") is not True
                    or verification.get("codex_turn_receipt_blake3")
                    != source.codex_turn_receipt_blake3
                    or verification.get("tool_trace_blake3")
                    != source.tool_trace_blake3
                    or verification.get("workspace_before_blake3")
                    != source.workspace_before_blake3
                    or verification.get("workspace_after_blake3")
                    != source.workspace_after_blake3
                    or verification.get("completion_artifact_path")
                    != source.completion_artifact_path
                    or not isinstance(rubric, Mapping)
                    or rubric.get("rubric_id") != source.rubric_id
                    or rubric.get("rubric_digest") != source.rubric_digest
                ):
                    raise ExecutionVerifiedSFTError(
                        "candidate SFT execution or rubric binding differs"
                    )
                stage_counts[str(row.get("stage"))] += 1
                route_counts[str(row.get("route_id"))] += 1
                domain_counts[str(row.get("domain"))] += 1
                source_slice_counts[str(source_path)] += 1
                observed_slices += 1
                rows.append(row)
            if (
                len(rows) != descriptor.get("slice_count")
                or not rows
                or rows[0]["example_id"] != descriptor.get("first_example_id")
                or rows[-1]["example_id"] != descriptor.get("last_example_id")
            ):
                raise ExecutionVerifiedSFTError("candidate SFT shard index differs")
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise ExecutionVerifiedSFTError("candidate SFT file inventory differs")
        if any(
            source_slice_counts[path] != row["slice_count"]
            for path, row in declared_by_path.items()
        ):
            raise ExecutionVerifiedSFTError("candidate SFT source slice counts differ")
        if (
            observed_slices != manifest.get("slice_count")
            or len(shards) != manifest.get("shard_count")
            or len(declared_sources) != manifest.get("source_count")
            or dict(sorted(stage_counts.items())) != manifest.get("counts_by_stage")
            or dict(sorted(route_counts.items())) != manifest.get("counts_by_route")
            or dict(sorted(domain_counts.items())) != manifest.get("counts_by_domain")
        ):
            raise ExecutionVerifiedSFTError("candidate SFT aggregate counts differ")
        checks.extend(
            (
                "manifest_and_shard_commitments",
                "atomic_tool_result_joins",
                "exact_rubric_bindings",
                "privacy_and_selection_labels",
                "aggregate_counts",
            )
        )
        core = {
            "schema": CANDIDATE_SFT_VERIFY_SCHEMA,
            "valid": True,
            "dataset_id": dataset_id,
            "source_count": len(accepted),
            "slice_count": observed_slices,
            "shard_count": len(shards),
            "rejected_count": sum(rejected.values()),
            "manifest_blake3": manifest_blake3,
            "checks": tuple(checks),
            "errors": (),
        }
    except Exception as exc:
        core = {
            "schema": CANDIDATE_SFT_VERIFY_SCHEMA,
            "valid": False,
            "dataset_id": dataset_id,
            "source_count": 0,
            "slice_count": observed_slices,
            "shard_count": 0,
            "rejected_count": 0,
            "manifest_blake3": manifest_blake3,
            "checks": tuple(checks),
            "errors": (f"{type(exc).__name__}:{exc}",),
        }
    return canonical_value({**core, "report_blake3": blake3_hex(core)})


__all__ = [
    "CANDIDATE_SFT_DATASET_SCHEMA",
    "CANDIDATE_SFT_SLICE_SCHEMA",
    "CANDIDATE_SFT_VERIFY_SCHEMA",
    "DEFAULT_CANDIDATE_ROUTES",
    "ExecutionVerifiedSFTBuild",
    "ExecutionVerifiedSFTError",
    "build_execution_verified_candidate_sft_dataset",
    "verify_execution_verified_candidate_sft_dataset",
]
