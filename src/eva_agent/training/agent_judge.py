"""Asynchronous, one-shot Agent Judge selection for teacher trajectories.

Teacher generation never imports this module.  A separate process discovers
completed trajectories through the teacher SQLite checkpoint, validates their
full evidence, and schedules exactly one workspace-aware judgment per source
trajectory.  Only an independently recomputed high rubric score can create an
SFT slice.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import subprocess
from typing import Any
from uuid import uuid4

from eva_agent.codex_providers import SignedAdapterReceipt, verify_adapter_receipt
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
)
from eva_agent.rubrics import CompiledRubric, CompiledRubricRegistry


REQUIRED_INSPECTION_SECTIONS = (
    "messages",
    "policy_visible_context",
    "tool_trace",
    "workspace_before",
    "workspace_after",
    "rubric_table",
)
TERMINAL_STATES = frozenset({"succeeded", "failed"})


class AgentJudgeSelectionError(ValueError):
    """A trajectory, judgment, checkpoint, or selection failed closed."""


def _write_new_or_verify_diagnostic(path: Path, payload: bytes) -> None:
    """Commit one immutable diagnostic without replacing prior evidence."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise AgentJudgeSelectionError(
                "Agent Judge diagnostic is unreadable"
            ) from exc
        if path.is_symlink() or existing != payload:
            raise AgentJudgeSelectionError("Agent Judge diagnostic bytes differ")
        return
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def persist_agent_judge_adapter_receipt(
    output_root: str | Path,
    receipt: SignedAdapterReceipt,
    *,
    judge_task_id: str,
    expected_route_id: str,
) -> Path:
    """Persist one task-bound, signed, raw-provider-free adapter receipt."""

    verify_adapter_receipt(receipt)
    for label, value in (
        ("judge task ID", judge_task_id),
        ("judge route ID", expected_route_id),
    ):
        component = PurePosixPath(value)
        if (
            not value
            or component.is_absolute()
            or len(component.parts) != 1
            or component.as_posix() != value
            or value in {".", ".."}
        ):
            raise AgentJudgeSelectionError(f"{label} is not an artifact component")
    receipt_route = receipt.payload.get("route_id")
    if receipt_route not in {None, expected_route_id}:
        raise AgentJudgeSelectionError("Agent Judge adapter receipt route differs")
    route = expected_route_id if receipt_route == expected_route_id else "unbound"
    path = (
        Path(output_root)
        / "adapter-receipts"
        / judge_task_id
        / route
        / f"{receipt.envelope_blake3}.json"
    )
    _write_new_or_verify_diagnostic(path, canonical_json_bytes(receipt.to_dict()))
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise AgentJudgeSelectionError(f"{label} is missing or invalid JSON") from None
    if not isinstance(value, Mapping):
        raise AgentJudgeSelectionError(f"{label} is not an object")
    return value


def _safe_workspace_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise AgentJudgeSelectionError("workspace path differs")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise AgentJudgeSelectionError("workspace path differs")
    return value


def _workspace_refs(value: Any, *, label: str) -> set[str]:
    if not isinstance(value, Mapping) or not isinstance(value.get("files"), list):
        raise AgentJudgeSelectionError(f"{label} workspace snapshot differs")
    rows = value["files"]
    refs: set[str] = set()
    paths: list[str] = []
    byte_count = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise AgentJudgeSelectionError(f"{label} workspace file differs")
        path = _safe_workspace_path(row.get("path"))
        encoded = row.get("content")
        if not isinstance(encoded, Mapping) or set(encoded) != {"$bytes_base64"}:
            raise AgentJudgeSelectionError(f"{label} workspace content differs")
        try:
            payload = base64.b64decode(encoded["$bytes_base64"], validate=True)
        except (TypeError, ValueError):
            raise AgentJudgeSelectionError(f"{label} workspace content differs") from None
        if (
            type(row.get("byte_count")) is not int
            or row["byte_count"] != len(payload)
            or row.get("content_blake3") != blake3_bytes(payload)
            or not isinstance(row.get("mode"), str)
            or len(row["mode"]) != 4
            or any(character not in "01234567" for character in row["mode"])
            or row["mode"].startswith("12")
        ):
            raise AgentJudgeSelectionError(f"{label} workspace commitment differs")
        if path in paths or any(
            path.startswith(existing + "/") or existing.startswith(path + "/")
            for existing in paths
        ):
            raise AgentJudgeSelectionError(f"{label} workspace path is duplicated")
        paths.append(path)
        byte_count += len(payload)
        refs.add(f"workspace:{label}:{path}")
    if paths != sorted(paths):
        raise AgentJudgeSelectionError(f"{label} workspace paths differ")
    snapshot_core = {
        "files": rows,
        "file_count": len(rows),
        "byte_count": byte_count,
    }
    if (
        value.get("file_count") != len(rows)
        or value.get("byte_count") != byte_count
        or value.get("tree_blake3") != blake3_hex(snapshot_core)
    ):
        raise AgentJudgeSelectionError(f"{label} workspace totals differ")
    return refs


@dataclass(frozen=True)
class ValidatedTrajectory:
    document: Mapping[str, Any]
    source_blake3: str
    rubric: CompiledRubric
    evidence_refs: frozenset[str]


def validate_full_trajectory(
    path: str | Path,
    *,
    registry: CompiledRubricRegistry,
    expected_task: "AgentJudgeTask | None" = None,
) -> ValidatedTrajectory:
    """Reopen all judge-visible sections and bind the exact domain/stage rubric."""

    source = Path(path).resolve(strict=True)
    value = _json_object(source, label="teacher trajectory")
    if value.get("schema") != "eva.codex-teacher-full-trajectory.v1":
        raise AgentJudgeSelectionError("teacher trajectory schema differs")
    required = (
        "task_id",
        "sandbox_id",
        "candidate_id",
        "route_id",
        "model_id",
        "messages",
        "tool_trace",
        "workspace_before",
        "workspace_after",
        "rubric_table",
    )
    if any(key not in value for key in required):
        raise AgentJudgeSelectionError("teacher trajectory is incomplete")
    if not isinstance(value["messages"], list) or not value["messages"]:
        raise AgentJudgeSelectionError("teacher trajectory messages differ")
    policy_events: set[str] = set()
    policy_context_seen = False
    for event in value["messages"]:
        if not isinstance(event, Mapping) or event.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise AgentJudgeSelectionError("teacher trajectory event differs")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise AgentJudgeSelectionError("teacher trajectory event identity differs")
        event_core = {
            "event_id": event_id,
            "role": event["role"],
            "content": event.get("content"),
            "tool_call_ids": event.get("tool_call_ids"),
        }
        if event.get("event_blake3") != blake3_hex(event_core):
            raise AgentJudgeSelectionError("teacher trajectory event commitment differs")
        policy_events.add(f"trajectory:event:{event_id}")
        policy_context_seen = policy_context_seen or event.get("role") == "user"
    if not policy_context_seen:
        raise AgentJudgeSelectionError("policy-visible context is absent from messages")
    trace = value["tool_trace"]
    if not isinstance(trace, Mapping) or not isinstance(trace.get("results"), list):
        raise AgentJudgeSelectionError("teacher tool trace differs")
    trace_core = dict(trace)
    recorded_trace_blake3 = trace_core.pop("trace_blake3", None)
    if recorded_trace_blake3 != blake3_hex(trace_core):
        raise AgentJudgeSelectionError("teacher tool trace commitment differs")
    actor_tool_refs = {
        f"actor-tool:{row['call_id']}"
        for row in trace["results"]
        if isinstance(row, Mapping) and isinstance(row.get("call_id"), str)
    }
    rubric_document = value["rubric_table"]
    if not isinstance(rubric_document, Mapping):
        raise AgentJudgeSelectionError("teacher rubric table differs")
    domain = rubric_document.get("domain")
    stage = rubric_document.get("stage")
    if not isinstance(domain, str) or not isinstance(stage, str):
        raise AgentJudgeSelectionError("teacher rubric domain or stage differs")
    rubric = registry.resolve(domain, stage)
    if canonical_value(rubric_document) != canonical_value(rubric.to_document()):
        raise AgentJudgeSelectionError("teacher rubric differs from the exact registry table")
    if not 5 <= len(rubric.items) <= 10:
        raise AgentJudgeSelectionError("teacher rubric item count differs")
    if expected_task is not None and (
        value["task_id"] != expected_task.source_task_id
        or value["sandbox_id"] != expected_task.sandbox_id
        or value["candidate_id"] != expected_task.candidate_id
        or value["route_id"] != expected_task.actor_route_id
        or domain != expected_task.domain
        or stage != expected_task.stage
    ):
        raise AgentJudgeSelectionError("teacher checkpoint and trajectory differ")
    references = {
        "context:policy-visible",
        *policy_events,
        *actor_tool_refs,
        *_workspace_refs(value["workspace_before"], label="before"),
        *_workspace_refs(value["workspace_after"], label="after"),
    }
    return ValidatedTrajectory(
        document=value,
        source_blake3=blake3_hex(value),
        rubric=rubric,
        evidence_refs=frozenset(references),
    )


@dataclass(frozen=True)
class AgentJudgeTask:
    judge_task_id: str
    source_task_id: str
    sandbox_id: str
    candidate_id: str
    actor_route_id: str
    stage: str
    domain: str
    trajectory_path: str
    judge_route_id: str
    judge_model_id: str


class AgentJudgeCheckpoint:
    """Independent SQLite WAL queue with one terminal judge attempt per source."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS judge_tasks (
                  judge_task_id TEXT PRIMARY KEY,
                  source_task_id TEXT NOT NULL UNIQUE,
                  sandbox_id TEXT NOT NULL,
                  candidate_id TEXT NOT NULL,
                  actor_route_id TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  domain TEXT NOT NULL,
                  trajectory_path TEXT NOT NULL,
                  judge_route_id TEXT NOT NULL,
                  judge_model_id TEXT NOT NULL,
                  state TEXT NOT NULL CHECK(state IN ('queued','running','succeeded','failed')),
                  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count BETWEEN 0 AND 1),
                  started_at TEXT,
                  finished_at TEXT,
                  selection_path TEXT,
                  sft_path TEXT,
                  reward_bps INTEGER,
                  error_type TEXT
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def schedule(self, tasks: Iterable[AgentJudgeTask]) -> int:
        inserted = 0
        with self._connect() as db:
            for task in tasks:
                before = db.total_changes
                db.execute(
                    """INSERT OR IGNORE INTO judge_tasks(
                    judge_task_id,source_task_id,sandbox_id,candidate_id,actor_route_id,
                    stage,domain,trajectory_path,judge_route_id,judge_model_id,state
                    ) VALUES(?,?,?,?,?,?,?,?,?,?, 'queued')""",
                    (
                        task.judge_task_id,
                        task.source_task_id,
                        task.sandbox_id,
                        task.candidate_id,
                        task.actor_route_id,
                        task.stage,
                        task.domain,
                        task.trajectory_path,
                        task.judge_route_id,
                        task.judge_model_id,
                    ),
                )
                inserted += db.total_changes - before
        return inserted

    def queued(self, *, limit: int | None = None) -> list[AgentJudgeTask]:
        suffix = " LIMIT ?" if limit is not None else ""
        values: tuple[Any, ...] = (limit,) if limit is not None else ()
        with self._connect() as db:
            rows = db.execute(
                """SELECT judge_task_id,source_task_id,sandbox_id,candidate_id,
                actor_route_id,stage,domain,trajectory_path,judge_route_id,judge_model_id
                FROM judge_tasks WHERE state='queued' ORDER BY rowid""" + suffix,
                values,
            ).fetchall()
        return [AgentJudgeTask(*row) for row in rows]

    def claim(self, task: AgentJudgeTask) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE judge_tasks SET state='running',attempt_count=1,started_at=?
                WHERE judge_task_id=? AND state='queued' AND attempt_count=0""",
                (_now(), task.judge_task_id),
            )
            return cursor.rowcount == 1

    def fail_interrupted(self) -> int:
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE judge_tasks SET state='failed',finished_at=?,
                error_type='InterruptedAttempt' WHERE state='running'""",
                (_now(),),
            )
            return cursor.rowcount

    def finish(
        self,
        task: AgentJudgeTask,
        *,
        succeeded: bool,
        selection_path: Path | None,
        sft_path: Path | None,
        reward_bps: int | None,
        error: str | None,
    ) -> None:
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE judge_tasks SET state=?,finished_at=?,selection_path=?,
                sft_path=?,reward_bps=?,error_type=?
                WHERE judge_task_id=? AND state='running' AND attempt_count=1""",
                (
                    "succeeded" if succeeded else "failed",
                    _now(),
                    str(selection_path) if selection_path else None,
                    str(sft_path) if sft_path else None,
                    reward_bps,
                    error,
                    task.judge_task_id,
                ),
            )
            if cursor.rowcount != 1:
                raise AgentJudgeSelectionError("judge terminal transition differs")

    def counts(self) -> dict[str, int]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT state,COUNT(*) FROM judge_tasks GROUP BY state"
            ).fetchall()
            selected = db.execute(
                "SELECT COUNT(*) FROM judge_tasks WHERE sft_path IS NOT NULL"
            ).fetchone()[0]
        result = {state: 0 for state in ("queued", "running", "succeeded", "failed")}
        result.update({str(state): int(count) for state, count in rows})
        result["selected_for_sft"] = int(selected)
        return result


def completed_teacher_tasks(
    teacher_root: str | Path,
    *,
    judge_route_id: str,
    judge_model_id: str,
    limit: int | None = None,
) -> tuple[AgentJudgeTask, ...]:
    """Discover immutable successful results without locking the teacher writer."""

    root = Path(teacher_root).resolve(strict=True)
    checkpoint = root / "checkpoint.sqlite3"
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise AgentJudgeSelectionError("teacher checkpoint is unavailable")
    with sqlite3.connect(f"file:{checkpoint}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(
            """SELECT task_id,sandbox_id,candidate_id,route_id,stage,domain,result_path
            FROM tasks WHERE state='succeeded' ORDER BY rowid"""
        ).fetchall()
    tasks: list[AgentJudgeTask] = []
    for source_task_id, sandbox_id, candidate_id, actor_route, stage, domain, result in rows:
        expected = (root / "trajectories" / sandbox_id / actor_route / "result.json").resolve(
            strict=True
        )
        path = Path(result)
        if path.is_absolute():
            resolved = path.resolve(strict=True)
        else:
            expected_tail = Path(
                "trajectories", sandbox_id, actor_route, "result.json"
            )
            if any(part in {"", ".", ".."} for part in path.parts):
                raise AgentJudgeSelectionError("teacher result topology differs")
            prefix_size = len(path.parts) - len(expected_tail.parts)
            prefix = path.parts[:prefix_size]
            tail = path.parts[prefix_size:]
            if tail != expected_tail.parts or (
                prefix and tuple(root.parts[-len(prefix) :]) != prefix
            ):
                raise AgentJudgeSelectionError("teacher result topology differs")
            resolved = expected
        if resolved != expected or resolved.is_symlink() or not resolved.is_file():
            raise AgentJudgeSelectionError("teacher result topology differs")
        tasks.append(
            AgentJudgeTask(
                judge_task_id=str(uuid4()),
                source_task_id=source_task_id,
                sandbox_id=sandbox_id,
                candidate_id=candidate_id,
                actor_route_id=actor_route,
                stage=stage,
                domain=domain,
                trajectory_path=str(resolved),
                judge_route_id=judge_route_id,
                judge_model_id=judge_model_id,
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tuple(tasks)


def _validated_judgment(
    value: Mapping[str, Any],
    *,
    task: AgentJudgeTask,
    trajectory: ValidatedTrajectory,
) -> tuple[dict[str, int], Mapping[str, Any]]:
    expected_keys = {
        "schema",
        "judge_task_id",
        "source_task_id",
        "source_trajectory_blake3",
        "judge_model_id",
        "inspected_sections",
        "inspected_evidence_refs",
        "judge_tool_trace",
        "item_scores",
        "summary",
    }
    if set(value) != expected_keys or value.get("schema") != "eva.codex-agent-judge-result.v1":
        raise AgentJudgeSelectionError("judge result shape differs")
    if (
        value["judge_task_id"] != task.judge_task_id
        or value["source_task_id"] != task.source_task_id
        or value["source_trajectory_blake3"] != trajectory.source_blake3
        or value["judge_model_id"] != task.judge_model_id
        or tuple(value["inspected_sections"]) != REQUIRED_INSPECTION_SECTIONS
    ):
        raise AgentJudgeSelectionError("judge result binding differs")
    inspected = value["inspected_evidence_refs"]
    if (
        not isinstance(inspected, list)
        or not inspected
        or len(inspected) != len(set(inspected))
        or any(ref not in trajectory.evidence_refs for ref in inspected)
    ):
        raise AgentJudgeSelectionError("judge inspected evidence references differ")
    trace = value["judge_tool_trace"]
    if not isinstance(trace, Mapping) or set(trace) != {
        "workspace_read_count",
        "workspace_search_count",
        "provider_turn_count",
        "retry_count",
    }:
        raise AgentJudgeSelectionError("judge tool trace differs")
    if (
        type(trace["workspace_read_count"]) is not int
        or type(trace["workspace_search_count"]) is not int
        or trace["workspace_read_count"] + trace["workspace_search_count"] < 1
        or type(trace["provider_turn_count"]) is not int
        or trace["provider_turn_count"] < 1
        or trace["retry_count"] != 0
    ):
        raise AgentJudgeSelectionError("judge did not inspect workspace evidence once")
    rows = value["item_scores"]
    items = trajectory.rubric.items
    if not isinstance(rows, list) or len(rows) != len(items):
        raise AgentJudgeSelectionError("judge item score coverage differs")
    scores: dict[str, int] = {}
    for row, item in zip(rows, items, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "item_id",
            "score_bps",
            "evidence_refs",
            "rationale",
        }:
            raise AgentJudgeSelectionError("judge item score row differs")
        item_id = str(item["item_id"])
        refs = row["evidence_refs"]
        if row["item_id"] != item_id or not isinstance(refs, list) or not refs:
            raise AgentJudgeSelectionError("judge item score identity differs")
        if (
            len(refs) != len(set(refs))
            or any(ref not in trajectory.evidence_refs for ref in refs)
            or not any(ref.startswith("workspace:") for ref in refs)
            or any(ref.startswith("workspace:") and ref not in inspected for ref in refs)
        ):
            raise AgentJudgeSelectionError("judge item cites uninspected workspace evidence")
        score = row["score_bps"]
        allowed = {
            int(level["score_bps"])
            for level in item["partial_credit"]["levels"]
        }
        if type(score) is not int or score not in allowed:
            raise AgentJudgeSelectionError("judge item score is not a compiled level")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise AgentJudgeSelectionError("judge item rationale differs")
        scores[item_id] = score
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise AgentJudgeSelectionError("judge summary differs")
    return scores, value


def _worker_environment(
    task: AgentJudgeTask, *, trajectory_blake3: str, output_root: Path
) -> Mapping[str, str]:
    return {
        **os.environ,
        "EVA_AGENT_JUDGE_TASK_ID": task.judge_task_id,
        "EVA_AGENT_JUDGE_SOURCE_TASK_ID": task.source_task_id,
        "EVA_AGENT_JUDGE_TRAJECTORY_PATH": task.trajectory_path,
        "EVA_AGENT_JUDGE_SOURCE_BLAKE3": trajectory_blake3,
        "EVA_AGENT_JUDGE_ROUTE_ID": task.judge_route_id,
        "EVA_AGENT_JUDGE_MODEL_ID": task.judge_model_id,
        "EVA_AGENT_JUDGE_OUTPUT_ROOT": str(output_root.resolve()),
    }


def run_agent_judge_batch(
    *,
    checkpoint: AgentJudgeCheckpoint,
    output_root: str | Path,
    registry: CompiledRubricRegistry,
    worker_command: Sequence[str],
    workers: int,
    minimum_sft_score_bps: int,
    limit: int | None = None,
) -> Mapping[str, Any]:
    """Judge queued trajectories once without interacting with teacher generation."""

    if (
        not worker_command
        or not 1 <= workers <= 256
        or type(minimum_sft_score_bps) is not int
        or not 0 <= minimum_sft_score_bps <= 10_000
    ):
        raise AgentJudgeSelectionError("judge batch options differ")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = checkpoint.queued(limit=limit)

    def run_one(
        task: AgentJudgeTask,
    ) -> tuple[AgentJudgeTask, bool, Path | None, Path | None, int | None, str | None]:
        if not checkpoint.claim(task):
            return task, False, None, None, None, "ClaimConflict"
        try:
            trajectory = validate_full_trajectory(
                task.trajectory_path, registry=registry, expected_task=task
            )
            result = subprocess.run(
                tuple(worker_command),
                env=_worker_environment(
                    task,
                    trajectory_blake3=trajectory.source_blake3,
                    output_root=root,
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=None,
                check=False,
            )
            if result.returncode != 0:
                errors = root / "errors"
                errors.mkdir(parents=True, exist_ok=True)
                (errors / f"{task.judge_task_id}.stderr.txt").write_text(
                    (result.stderr or result.stdout or "judge exited without output")[-65536:],
                    encoding="utf-8",
                )
                return task, False, None, None, None, "WorkerExit"
            try:
                raw = json.loads(result.stdout)
            except (UnicodeError, ValueError):
                raise AgentJudgeSelectionError("judge result is not JSON") from None
            if not isinstance(raw, Mapping):
                raise AgentJudgeSelectionError("judge result is not an object")
            item_scores, judgment = _validated_judgment(
                raw, task=task, trajectory=trajectory
            )
            evaluation_id = str(uuid4())
            rubric_score = trajectory.rubric.score(
                item_scores, evaluation_id=evaluation_id
            )
            selected = (
                rubric_score.hard_gate_passed
                and rubric_score.reward_bps >= minimum_sft_score_bps
            )
            selection_id = str(uuid4())
            core = {
                "schema": "eva.async-agent-judge-selection.v1",
                "selection_id": selection_id,
                "judge_task_id": task.judge_task_id,
                "source_task_id": task.source_task_id,
                "sandbox_id": task.sandbox_id,
                "candidate_id": task.candidate_id,
                "actor_route_id": task.actor_route_id,
                "judge_route_id": task.judge_route_id,
                "judge_model_id": task.judge_model_id,
                "domain": task.domain,
                "stage": task.stage,
                "source_trajectory_blake3": trajectory.source_blake3,
                "rubric_id": trajectory.rubric.rubric_id,
                "rubric_digest": trajectory.rubric.digest,
                "judgment": judgment,
                "rubric_score": rubric_score.to_document(),
                "minimum_sft_score_bps": minimum_sft_score_bps,
                "selected_for_sft": selected,
                "judge_attempt_count": 1,
                "retry_count": 0,
            }
            selection = {**core, "selection_blake3": blake3_hex(core)}
            selection_root = root / "selections" / task.sandbox_id / task.actor_route_id
            selection_root.mkdir(parents=True, exist_ok=False)
            selection_path = selection_root / "selection.json"
            selection_path.write_bytes(canonical_json_bytes(selection))
            sft_path: Path | None = None
            if selected:
                sft_id = str(uuid4())
                sft_core = {
                    "schema": "eva.agent-judged-sft-trajectory.v1",
                    "sft_id": sft_id,
                    "sandbox_id": task.sandbox_id,
                    "candidate_id": task.candidate_id,
                    "source_task_id": task.source_task_id,
                    "actor_route_id": task.actor_route_id,
                    "actor_model_id": trajectory.document["model_id"],
                    "judge_model_id": task.judge_model_id,
                    "domain": task.domain,
                    "stage": task.stage,
                    "reward_bps": rubric_score.reward_bps,
                    "rubric_digest": trajectory.rubric.digest,
                    "selection_blake3": selection["selection_blake3"],
                    "messages": trajectory.document["messages"],
                    "tool_trace": trajectory.document["tool_trace"],
                    "private_reference_included": False,
                    "hidden_reasoning_included": False,
                }
                sft = {**sft_core, "sft_slice_blake3": blake3_hex(sft_core)}
                sft_root = root / "sft-slices" / task.sandbox_id
                sft_root.mkdir(parents=True, exist_ok=True)
                sft_path = sft_root / f"{task.actor_route_id}.json"
                sft_path.write_bytes(canonical_json_bytes(sft))
            return (
                task,
                True,
                selection_path,
                sft_path,
                rubric_score.reward_bps,
                None,
            )
        except Exception as exc:
            return task, False, None, None, None, type(exc).__name__

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            task, succeeded, selection, sft, reward, error = future.result()
            checkpoint.finish(
                task,
                succeeded=succeeded,
                selection_path=selection,
                sft_path=sft,
                reward_bps=reward,
                error=error,
            )
    return {
        "schema": "eva.async-agent-judge-batch-result.v1",
        "counts": checkpoint.counts(),
        "judge_attempts_per_trajectory": 1,
        "retry_count": 0,
    }


def verify_selection_artifact(
    selection_path: str | Path,
    *,
    trajectory_path: str | Path,
    registry: CompiledRubricRegistry,
) -> bool:
    """Provider-free independent reopening of one selection and its source."""

    selection = _json_object(Path(selection_path), label="selection")
    shadow = dict(selection)
    digest = shadow.pop("selection_blake3", None)
    if digest != blake3_hex(shadow):
        raise AgentJudgeSelectionError("selection BLAKE3 differs")
    task = AgentJudgeTask(
        judge_task_id=str(selection["judge_task_id"]),
        source_task_id=str(selection["source_task_id"]),
        sandbox_id=str(selection["sandbox_id"]),
        candidate_id=str(selection["candidate_id"]),
        actor_route_id=str(selection["actor_route_id"]),
        stage=str(selection["stage"]),
        domain=str(selection["domain"]),
        trajectory_path=str(Path(trajectory_path).resolve(strict=True)),
        judge_route_id=str(selection["judge_route_id"]),
        judge_model_id=str(selection["judge_model_id"]),
    )
    trajectory = validate_full_trajectory(
        trajectory_path, registry=registry, expected_task=task
    )
    if selection.get("source_trajectory_blake3") != trajectory.source_blake3:
        raise AgentJudgeSelectionError("selection source trajectory differs")
    judgment = selection.get("judgment")
    if not isinstance(judgment, Mapping):
        raise AgentJudgeSelectionError("selection judgment differs")
    scores, _ = _validated_judgment(judgment, task=task, trajectory=trajectory)
    score_document = selection.get("rubric_score")
    if not isinstance(score_document, Mapping):
        raise AgentJudgeSelectionError("selection rubric score differs")
    recomputed = trajectory.rubric.score(
        scores, evaluation_id=str(score_document.get("evaluation_id"))
    ).to_document()
    if canonical_value(recomputed) != canonical_value(score_document):
        raise AgentJudgeSelectionError("selection rubric score does not recompute")
    expected_selected = bool(recomputed["hard_gate_passed"]) and int(
        recomputed["reward_bps"]
    ) >= int(selection["minimum_sft_score_bps"])
    if selection.get("selected_for_sft") is not expected_selected:
        raise AgentJudgeSelectionError("selection threshold decision differs")
    return True


__all__ = [
    "AgentJudgeCheckpoint",
    "AgentJudgeSelectionError",
    "AgentJudgeTask",
    "REQUIRED_INSPECTION_SECTIONS",
    "ValidatedTrajectory",
    "completed_teacher_tasks",
    "run_agent_judge_batch",
    "validate_full_trajectory",
    "verify_selection_artifact",
]
