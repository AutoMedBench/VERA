"""Cancel only live process trees born from this cell's trusted job launcher."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import signal
import time


async def cleanup_owned_jobs(audit: Path, workspace: Path, *, trigger: str,
                             filename: str = "evamed-job-cleanup.json") -> dict:
    import psutil
    from training.automedbench_lite.adapter import read_document, write_once

    observed_start = time.monotonic()
    observation_grace = None
    tool_config = audit / 'tool-config.json'
    deadline = json.loads(tool_config.read_bytes()).get('track_deadline_monotonic') if tool_config.exists() else None
    if deadline is not None and observed_start >= deadline:
        # Each trusted supervisor signals its compute child at its own deadline.
        # Observe only its cancellation/exit journal, never grant inference time.
        stop_observing = time.monotonic() + 3.0
        while time.monotonic() < stop_observing:
            pending = []
            for path in (audit / 'model-jobs').glob('*/process.json'):
                process = read_document(path)
                try:
                    fields = Path(f"/proc/{process['pid']}/stat").read_text().rsplit(')', 1)[1].split()
                    live = fields[19] == str(process['start_ticks']) and fields[0] != 'Z'
                except FileNotFoundError:
                    live = False
                if live:
                    pending.append(path.parent.name)
            if not pending:
                break
            await asyncio.sleep(min(.05, max(0, stop_observing - time.monotonic())))
        observation_grace = {'limit_seconds': 3.0, 'started_monotonic': observed_start,
            'ended_monotonic': time.monotonic(), 'purpose': 'supervisor-owned-child-cancellation-journal-only',
            'policy_time_extended': False}
    owned = {}
    for job in sorted((audit / "model-jobs").glob("*/submission.json")):
        submission = read_document(job)
        process = read_document(job.parent / "process.json")
        if submission.get("workspace") != str(workspace) or process.get("job_id") != submission.get("job_id"):
            raise ValueError("job_cleanup_binding_invalid")
        pid = process["pid"]
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[19] != str(process["start_ticks"]) or fields[0] == "Z":
                continue
            parent = psutil.Process(pid)
            parent.suspend()  # Freeze the owned launcher before traversing children.
            # Enroot wrappers legitimately exec into different argv. Process
            # birth and actual descendant links are the identity boundary.
            owned[parent.pid] = (parent, parent.create_time())
            for _ in range(64):
                added = False
                for child in parent.children(recursive=True):
                    if child.pid not in owned:
                        child.suspend()
                        owned[child.pid] = (child, child.create_time())
                        added = True
                if not added:
                    break
            else:
                raise ValueError("owned_process_tree_did_not_stabilize")
        except (FileNotFoundError, psutil.NoSuchProcess):
            continue
    requested = []
    for pid, (process, birth) in reversed(tuple(owned.items())):
        try:
            if process.create_time() == birth and process.status() != psutil.STATUS_ZOMBIE:
                process.send_signal(signal.SIGKILL)
                requested.append({"pid": pid, "birth_seconds": birth, "signal": "SIGKILL"})
        except psutil.NoSuchProcess:
            pass
    cleanup_wait_deadline = time.monotonic() + 10
    live = dict(owned)
    while live and time.monotonic() < cleanup_wait_deadline:
        for pid, (process, birth) in tuple(live.items()):
            try:
                if process.create_time() != birth or process.status() == psutil.STATUS_ZOMBIE:
                    live.pop(pid)
            except psutil.NoSuchProcess:
                live.pop(pid)
        if live:
            await asyncio.sleep(.1)
    for pid, (process, birth) in live.items():
        try:
            if process.create_time() == birth:
                process.kill()
                requested.append({"pid": pid, "birth_seconds": birth, "signal": "SIGKILL"})
        except psutil.NoSuchProcess:
            pass
    if live:
        _, remaining = psutil.wait_procs([process for process, _ in live.values()], timeout=10)
        if any(process.is_running() and process.status() != psutil.STATUS_ZOMBIE for process in remaining):
            raise ValueError("owned_model_process_did_not_exit")
    return write_once(audit / filename, {"schema": "eva.evamed-owned-job-cleanup.v1",
        "trigger": trigger, "scope": "birth-bound-frozen-owned-job-process-trees", "signals": requested,
        "workspace_quiescent": True, "actor_policy_time_extended": False,
        "deadline_monotonic": deadline, "observation_grace": observation_grace,
        "fallback_killed_processes": bool(requested), "quiescent_observed_monotonic": time.monotonic()})
