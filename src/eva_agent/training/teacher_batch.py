"""Durable, zero-retry teacher-rollout batch scheduling.

The scheduler deliberately knows nothing about admission or ability cascades.  A
worker receives exactly one (sandbox, model-route) tuple and must return a full
Codex/MCP trajectory receipt.  SQLite is the sole mutable checkpoint so a
launcher started under ``nohup`` survives SSH loss and can be inspected while
it runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from threading import Lock
from typing import Any, Callable

from eva_agent.codex_providers import SignedAdapterReceipt, verify_adapter_receipt
from eva_agent.codex_runtime import CodexTurnReceipt, verify_codex_turn_receipt
from eva_agent.pipeline.digests import (
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
)


TEACHER_ROUTES = (
    "gpt_6_astra",
    "gpt_6_astra_openai",
    "gpt_6_astra_azure",
    "opus_5",
    "opus_4_8",
    "gpt_5_6_sol",
    "gpt_5_6_terra",
    "gpt_5_6_luna",
    "gemini_3_1_pro",
    "gemini_3_5_flash",
    "gemini_3_8_flash",
    "glm_5_1",
    "glm_5_2",
    "glm_5_3",
    "glm_5_3_flash",
    "qwen_3_5_0_8b",
    "qwen_3_5_9b",
    "qwen_3_5_35b_a3b",
    "qwen_3_5_122b_a10b",
    "qwen_3_5_397b_a17b",
    "qwen_3_6_27b",
    "deepseek_v4_flash",
    "deepseek_v4_pro",
)
TERMINAL_STATES = frozenset({"succeeded", "failed"})


class TeacherBatchError(ValueError):
    """The batch plan, worker result, or checkpoint differs."""


def _write_new_or_verify(path: Path, payload: bytes) -> None:
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
            raise TeacherBatchError("teacher diagnostic is unreadable") from exc
        if path.is_symlink() or existing != payload:
            raise TeacherBatchError("teacher diagnostic bytes differ")
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


def persist_teacher_adapter_receipt(
    output_root: str | Path, receipt: SignedAdapterReceipt
) -> Path:
    """Persist one self-verifying, raw-provider-free adapter receipt."""

    verify_adapter_receipt(receipt)
    route_id = receipt.payload.get("route_id")
    route = route_id if route_id in TEACHER_ROUTES else "unbound"
    path = (
        Path(output_root)
        / "adapter-receipts"
        / route
        / f"{receipt.envelope_blake3}.json"
    )
    _write_new_or_verify(path, canonical_json_bytes(receipt.to_dict()))
    return path


def persist_teacher_failure_evidence(
    *, task: "TeacherTask", error: BaseException, output_root: str | Path
) -> Path | None:
    """Retain the safe terminal Codex turn attached to a failed teacher task."""

    receipt = getattr(error, "receipt", None)
    if receipt is not None:
        if not isinstance(receipt, CodexTurnReceipt):
            raise TeacherBatchError("teacher failure turn receipt type differs")
        verify_codex_turn_receipt(receipt)
    category = getattr(error, "safe_failure_category", None)
    message = getattr(error, "safe_failure_message", None)
    provenance = getattr(error, "safe_failure_provenance", None)
    safe_values = (category, message, provenance)
    if receipt is None and not all(isinstance(value, str) and value for value in safe_values):
        return None
    if not all(isinstance(value, str) and value for value in safe_values):
        category, message, provenance = (
            "provider_boundary",
            "Teacher provider boundary failed closed",
            "eva_agent.training.teacher_batch",
        )
    receipt_value = canonical_value(receipt) if receipt is not None else None
    core = {
        "schema": "eva.codex-teacher-failure-evidence.v1",
        "task_id": task.task_id,
        "sandbox_id": task.sandbox_id,
        "candidate_id": task.candidate_id,
        "route_id": task.route_id,
        "stage": task.stage,
        "domain": task.domain,
        "failure_type": type(error).__name__,
        "failure_category": category,
        "failure_message": message,
        "failure_provenance": provenance,
        "failure_message_blake3": blake3_hex(str(error)),
        "provider_turn_receipt": receipt_value,
        "provider_turn_receipt_blake3": (
            receipt.receipt_blake3 if receipt is not None else None
        ),
        "raw_exception_message_recorded": False,
        "raw_provider_material_recorded": False,
        "retry_count": 0,
    }
    document = {**core, "failure_evidence_blake3": blake3_hex(core)}
    path = Path(output_root) / "errors" / f"{task.task_id}.provider-failure.json"
    _write_new_or_verify(path, canonical_json_bytes(document))
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _strict_json(line: str, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(line)
    except (UnicodeError, ValueError):
        raise TeacherBatchError(f"{label} is not JSON") from None
    if not isinstance(value, Mapping):
        raise TeacherBatchError(f"{label} is not an object")
    return value


def iter_bulk_sandboxes(root: str | Path) -> Iterable[Mapping[str, Any]]:
    """Stream the already independently verified bulk-RL records."""

    bulk = Path(root).resolve(strict=True)
    manifest = _strict_json((bulk / "manifest.json").read_text(), label="manifest")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping) or payload.get("sandbox_count") != 6000:
        raise TeacherBatchError("bulk package is not the exact 6,000-record release")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise TeacherBatchError("bulk shard inventory differs")
    seen: set[str] = set()
    count = 0
    for descriptor in shards:
        if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
            raise TeacherBatchError("bulk shard descriptor differs")
        path = bulk / descriptor["path"]
        if path.parent != bulk or path.is_symlink() or not path.is_file():
            raise TeacherBatchError("bulk shard topology differs")
        for raw in path.read_text(encoding="utf-8").splitlines():
            row = _strict_json(raw, label="bulk sandbox")
            sandbox_id = row.get("sandbox_id")
            candidate_id = row.get("candidate_id")
            if not isinstance(sandbox_id, str) or not isinstance(candidate_id, str):
                raise TeacherBatchError("bulk sandbox identity differs")
            if sandbox_id in seen:
                raise TeacherBatchError("bulk sandbox identity is duplicated")
            seen.add(sandbox_id)
            count += 1
            yield row
    if count != 6000:
        raise TeacherBatchError("bulk sandbox count differs")


@dataclass(frozen=True)
class TeacherTask:
    task_id: str
    sandbox_id: str
    candidate_id: str
    route_id: str
    stage: str
    domain: str


class TeacherCheckpoint:
    """Small SQLite WAL checkpoint with atomic one-shot task claims."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # A single launcher can expose hundreds of worker threads.  SQLite's
        # busy timeout coordinates separate processes, but it does not make
        # opening hundreds of same-process write transactions at once useful.
        # Serialize only the tiny checkpoint mutations; provider work remains
        # fully parallel.
        self._write_lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self._connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS tasks (
                  task_id TEXT PRIMARY KEY,
                  sandbox_id TEXT NOT NULL,
                  candidate_id TEXT NOT NULL,
                  route_id TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  domain TEXT NOT NULL,
                  state TEXT NOT NULL CHECK(state IN ('queued','running','succeeded','failed')),
                  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count BETWEEN 0 AND 1),
                  started_at TEXT,
                  finished_at TEXT,
                  result_path TEXT,
                  error_type TEXT,
                  UNIQUE(sandbox_id, route_id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def schedule(self, rows: Sequence[Mapping[str, Any]], routes: Sequence[str]) -> int:
        if not routes or any(route not in TEACHER_ROUTES for route in routes):
            raise TeacherBatchError("teacher route set differs")
        inserted = 0
        with self._write_lock, self._connect() as db:
            for row in rows:
                for route in routes:
                    task_id = f"{row['sandbox_id']}--{route}"
                    before = db.total_changes
                    db.execute(
                        "INSERT OR IGNORE INTO tasks(task_id,sandbox_id,candidate_id,route_id,stage,domain,state) VALUES(?,?,?,?,?,?, 'queued')",
                        (task_id, row["sandbox_id"], row["candidate_id"], route, row["stage"], row["domain"]),
                    )
                    inserted += db.total_changes - before
        return inserted

    def fail_interrupted(self) -> int:
        """Honor zero retry: a process-loss attempt becomes terminal failure."""

        with self._write_lock, self._connect() as db:
            cursor = db.execute(
                "UPDATE tasks SET state='failed',finished_at=?,error_type='InterruptedAttempt' WHERE state='running'",
                (_now(),),
            )
            return cursor.rowcount

    def queued(self, *, limit: int | None = None) -> list[TeacherTask]:
        suffix = " LIMIT ?" if limit is not None else ""
        values: tuple[Any, ...] = (limit,) if limit is not None else ()
        with self._connect() as db:
            records = db.execute(
                "SELECT task_id,sandbox_id,candidate_id,route_id,stage,domain FROM tasks WHERE state='queued' ORDER BY rowid" + suffix,
                values,
            ).fetchall()
        return [TeacherTask(*row) for row in records]

    def claim(self, task: TeacherTask) -> bool:
        with self._write_lock, self._connect() as db:
            cursor = db.execute(
                "UPDATE tasks SET state='running',attempt_count=1,started_at=? WHERE task_id=? AND state='queued' AND attempt_count=0",
                (_now(), task.task_id),
            )
            return cursor.rowcount == 1

    def finish(self, task: TeacherTask, *, succeeded: bool, path: Path | None, error: str | None) -> None:
        with self._write_lock, self._connect() as db:
            cursor = db.execute(
                "UPDATE tasks SET state=?,finished_at=?,result_path=?,error_type=? WHERE task_id=? AND state='running' AND attempt_count=1",
                ("succeeded" if succeeded else "failed", _now(), str(path) if path else None, error, task.task_id),
            )
            if cursor.rowcount != 1:
                raise TeacherBatchError("task terminal transition differs")

    def counts(self) -> dict[str, int]:
        with self._connect() as db:
            rows = db.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state").fetchall()
        counts = {state: 0 for state in ("queued", "running", "succeeded", "failed")}
        counts.update({str(state): int(count) for state, count in rows})
        return counts


def _worker_environment(
    task: TeacherTask, *, bulk_root: Path, output_root: Path
) -> dict[str, str]:
    return {
        **os.environ,
        "EVA_TEACHER_TASK_ID": task.task_id,
        "EVA_TEACHER_SANDBOX_ID": task.sandbox_id,
        "EVA_TEACHER_CANDIDATE_ID": task.candidate_id,
        "EVA_TEACHER_ROUTE_ID": task.route_id,
        "EVA_TEACHER_STAGE": task.stage,
        "EVA_TEACHER_DOMAIN": task.domain,
        "EVA_TEACHER_BULK_ROOT": str(bulk_root.resolve(strict=True)),
        "EVA_TEACHER_OUTPUT_ROOT": str(output_root.resolve()),
    }


TeacherReceiptWorker = Callable[[TeacherTask], Mapping[str, Any]]
_TeacherInvocation = Callable[
    [TeacherTask], tuple[Mapping[str, Any] | None, str | None]
]


def _persist_teacher_receipt(
    *,
    task: TeacherTask,
    receipt: Mapping[str, Any],
    output_root: Path,
    minimum_sft_score: float,
) -> Path:
    required = (
        "score",
        "messages",
        "tool_trace",
        "workspace_before",
        "workspace_after",
    )
    if any(key not in receipt for key in required) or not isinstance(
        receipt["score"], (int, float)
    ):
        raise TeacherBatchError("teacher trajectory receipt is incomplete")
    task_root = output_root / "trajectories" / task.sandbox_id / task.route_id
    task_root.mkdir(parents=True, exist_ok=False)
    receipt_path = task_root / "result.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    if float(receipt["score"]) >= minimum_sft_score:
        slice_root = output_root / "sft-slices" / task.sandbox_id
        slice_root.mkdir(parents=True, exist_ok=True)
        (slice_root / f"{task.route_id}.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema": "eva.codex-teacher-sft-slice.v1",
                    "task_id": task.task_id,
                    "sandbox_id": task.sandbox_id,
                    "candidate_id": task.candidate_id,
                    "route_id": task.route_id,
                    "stage": task.stage,
                    "domain": task.domain,
                    "score": receipt["score"],
                    "messages": receipt["messages"],
                    "tool_trace": receipt["tool_trace"],
                }
            )
        )
    return receipt_path


def _run_task_batch(
    *,
    checkpoint: TeacherCheckpoint,
    output_root: Path,
    invoke: _TeacherInvocation,
    workers: int,
    minimum_sft_score: float,
    limit: int | None,
) -> Mapping[str, Any]:
    if workers < 1 or workers > 256:
        raise TeacherBatchError("bounded teacher width differs")
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = checkpoint.queued(limit=limit)

    def run_one(
        task: TeacherTask,
    ) -> tuple[TeacherTask, bool | None, Path | None, str | None]:
        if not checkpoint.claim(task):
            # Another durable launcher owns this attempt.  Only the owner may
            # publish its terminal transition; this process simply skips it.
            return task, None, None, "ClaimConflict"
        try:
            receipt, error = invoke(task)
            if error is not None:
                return task, False, None, error
            if receipt is None:
                return task, False, None, "IncompleteTrajectoryReceipt"
            path = _persist_teacher_receipt(
                task=task,
                receipt=receipt,
                output_root=output_root,
                minimum_sft_score=minimum_sft_score,
            )
            return task, True, path, None
        except Exception as exc:
            persist_teacher_failure_evidence(
                task=task, error=exc, output_root=output_root
            )
            error = (
                "IncompleteTrajectoryReceipt"
                if isinstance(exc, TeacherBatchError)
                else type(exc).__name__
            )
            return task, False, None, error

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            task, succeeded, path, error = future.result()
            if succeeded is not None:
                checkpoint.finish(task, succeeded=succeeded, path=path, error=error)
    return {
        "schema": "eva.codex-teacher-batch-result.v1",
        "counts": checkpoint.counts(),
        "retry_count": 0,
    }


def run_batch(
    *,
    checkpoint: TeacherCheckpoint,
    output_root: str | Path,
    worker_command: Sequence[str],
    workers: int,
    minimum_sft_score: float,
    bulk_root: str | Path | None = None,
    limit: int | None = None,
) -> Mapping[str, Any]:
    """Run one attempt per task; the worker must emit one JSON receipt to stdout."""

    if not worker_command or workers < 1 or workers > 256:
        raise TeacherBatchError("worker command or bounded width differs")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    bulk = Path(
        bulk_root
        if bulk_root is not None
        else os.environ.get("EVA_TEACHER_BULK_ROOT", "")
    )
    if not str(bulk) or not bulk.resolve(strict=True).is_dir():
        raise TeacherBatchError("bulk root is required for worker binding")
    def invoke(task: TeacherTask) -> tuple[Mapping[str, Any] | None, str | None]:
        result = subprocess.run(
            tuple(worker_command),
            env=_worker_environment(task, bulk_root=bulk, output_root=root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=None,
            check=False,
        )
        if result.returncode != 0:
            error_root = root / "errors"
            error_root.mkdir(parents=True, exist_ok=True)
            diagnostic = (
                result.stderr or result.stdout or "worker exited without output"
            )[-65536:]
            (error_root / f"{task.task_id}.stderr.txt").write_text(
                diagnostic,
                encoding="utf-8",
            )
            return None, "WorkerExit"
        return _strict_json(result.stdout, label="worker receipt"), None

    return _run_task_batch(
        checkpoint=checkpoint,
        output_root=root,
        invoke=invoke,
        workers=workers,
        minimum_sft_score=minimum_sft_score,
        limit=limit,
    )


def run_persistent_batch(
    *,
    checkpoint: TeacherCheckpoint,
    output_root: str | Path,
    worker: TeacherReceiptWorker,
    workers: int,
    minimum_sft_score: float,
    limit: int | None = None,
) -> Mapping[str, Any]:
    """Run tasks in one process against a long-lived, thread-safe worker.

    The durable claim is still committed before the worker is entered.  A
    process loss therefore leaves ``running`` rows that the next invocation
    terminalizes via :meth:`TeacherCheckpoint.fail_interrupted`; it never
    causes a second provider attempt.
    """

    if not callable(worker):
        raise TeacherBatchError("persistent teacher worker is unavailable")

    def invoke(task: TeacherTask) -> tuple[Mapping[str, Any] | None, str | None]:
        return worker(task), None

    return _run_task_batch(
        checkpoint=checkpoint,
        output_root=Path(output_root),
        invoke=invoke,
        workers=workers,
        minimum_sft_score=minimum_sft_score,
        limit=limit,
    )
