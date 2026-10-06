"""Wait for already-submitted public jobs; never resubmit or synthesize outputs."""
import asyncio
import json
from pathlib import Path
import time

from .adapter import EvaluationError


async def await_model_jobs(audit, phase, *, timeout=7200, interval=5):
    root = Path(audit) / "model-jobs"
    jobs = sorted(path.parent for path in root.glob("*/submission.json"))
    if not jobs:
        return
    started = time.monotonic()
    status_path = Path(audit) / ("waiting-before-" + phase + ".json")
    while True:
        terminal, pending = [], []
        for job in jobs:
            exit_path = root / "authoritative" / job.name / "process-exit.json"
            if exit_path.exists():
                result = json.loads(exit_path.read_text())
                if (result.get("schema") != "eva.prescribed-model-process-exit.v1"
                        or result.get("job_id") != job.name
                        or result.get("os_process_exit_observed") is not True
                        or type(result.get("returncode")) is not int):
                    raise EvaluationError("model_job_exit_evidence_invalid")
                terminal.append({"job_id": job.name, "returncode": result["returncode"]})
            else:
                process = json.loads((job / "process.json").read_text())
                try:
                    fields = Path(f'/proc/{process["pid"]}/stat').read_text().split()
                    live = fields[21] == process["start_ticks"] and fields[2] != "Z"
                except FileNotFoundError:
                    # Recheck a just-published exit record after process teardown.
                    if exit_path.exists():
                        pending.append(job.name)
                        continue
                    live = False
                if not live:
                    raise EvaluationError("model_job_supervisor_missing_without_exit")
                pending.append(job.name)
        elapsed = time.monotonic() - started
        document = {"schema": "eva.automed-model-job-phase-wait.v1", "before_phase": phase,
            "elapsed_seconds": elapsed, "terminal_jobs": terminal, "pending_jobs": pending,
            "provider_calls": 0, "automatic_resubmissions": 0}
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, sort_keys=True) + "\n")
        temporary.replace(status_path)
        if not pending:
            return  # Real failures remain visible for the SAME agent to inspect.
        if elapsed >= timeout:
            raise EvaluationError("model_jobs_still_running_at_phase_wait_limit")
        await asyncio.sleep(interval)
