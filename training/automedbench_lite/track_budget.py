"""Opt-in whole-track wall clock; queueing before admission is not charged."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import time
from uuid import UUID

from .adapter import EvaluationError, write_once

POLICY = "admitted-track-wallclock-3600-v1"


class TrackBudgetExhausted(EvaluationError):
    pass


class TrackDeadline:
    def __init__(self, audit, *, seconds=3600, clock=None):
        if type(seconds) is not int or seconds != 3600:
            raise EvaluationError("track_timeout_must_be_3600_seconds")
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.deadline = self.started + seconds
        self.document = write_once(audit / "track-budget.json", {
            "schema": "eva.automedbench-track-budget.v1", "policy": POLICY,
            "timeout_seconds": seconds, "admitted_at_ns": time.time_ns(),
            "started_monotonic": self.started, "deadline_monotonic": self.deadline,
            "queue_before_admission_charged": False,
            "job_waits_and_runtime_restart_charged": True,
            "cleanup_is_not_additional_policy_time": True,
        })

    def remaining(self):
        return max(0.0, self.deadline - self.clock())

    def require_remaining(self):
        remaining = self.remaining()
        if remaining <= 0:
            raise TrackBudgetExhausted("track_wallclock_budget_exhausted")
        return remaining

    def observation(self):
        observed = self.clock()
        return {"policy": POLICY, "budget_document_blake3": self.document["document_blake3"],
            "timeout_seconds": self.document["timeout_seconds"],
            "elapsed_seconds": max(0.0, observed - self.started),
            "remaining_seconds": max(0.0, self.deadline - observed),
            "deadline_exhausted": observed >= self.deadline,
            "cleanup_overrun_seconds": max(0.0, observed - self.deadline)}


async def cleanup_owned_jobs(audit, workspace, *, timeout=30, trigger="track_deadline"):
    """Stop only birth/argv-bound supervisors; their existing handler drains children."""
    from .job_wait import await_model_jobs
    from .policy_capture import await_model_publishers

    audit, workspace = Path(audit), Path(workspace)
    started = time.monotonic()
    requests = []
    root = audit / "model-jobs"
    expected_script = Path(__file__).resolve().parents[2] / "training/benchmark_models/job_supervisor.py"
    result = {"schema": "eva.automedbench-track-deadline-cleanup.v1",
        "scope": "exact_owned_model_supervisors_only", "policy_time_extended": False,
        "trigger": trigger,
        "signals": requests, "workspace_quiescent": False, "error_category": None}
    try:
        for path in sorted(root.glob("*/submission.json")):
            job_id = str(UUID(path.parent.name))
            submission = json.loads(path.read_bytes())
            process = json.loads((path.parent / "process.json").read_bytes())
            command = process.get("command", [])
            def argument(flag):
                if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
                    raise EvaluationError("track_cleanup_command_binding_invalid")
                return command[command.index(flag) + 1]
            if (submission.get("job_id") != job_id or process.get("job_id") != job_id
                    or submission.get("workspace") != str(workspace)
                    or len(command) < 3 or command[1:3] != ["-B", str(expected_script)]
                    or argument("--workspace") != str(workspace)
                    or argument("--job-id") != job_id
                    or argument("--audit-root") != str(root / "authoritative")):
                raise EvaluationError("track_cleanup_command_binding_invalid")
            # Already published exits need no signal. The publisher is still
            # checked below before any final workspace claim.
            if (root / "authoritative" / job_id / "process-exit.json").is_file():
                continue
            pid = process["pid"]
            if type(pid) is not int or pid <= 1:
                raise EvaluationError("track_cleanup_process_identity_invalid")
            try:
                descriptor = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                proc = Path(f"/proc/{pid}")
                fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                actual = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
                if fields[19] != str(process["start_ticks"]) or actual != command:
                    raise EvaluationError("track_cleanup_process_identity_changed")
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                requests.append({"job_id": job_id, "pid": pid, "start_ticks": fields[19],
                    "signal": "SIGTERM", "requested_at_ns": time.time_ns()})
            except ProcessLookupError:
                pass
            finally:
                os.close(descriptor)
        await await_model_jobs(audit, "track-deadline-cleanup",
            timeout=max(0, timeout - (time.monotonic() - started)), interval=.1)
        await await_model_publishers(audit, workspace,
            timeout=max(0, timeout - (time.monotonic() - started)))
        result["workspace_quiescent"] = True
    except Exception as exc:
        result["error_category"] = str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__
        raise
    finally:
        result["cleanup_elapsed_seconds"] = time.monotonic() - started
        retained = write_once(audit / "track-deadline-cleanup.json", result)
    return retained
