"""Wait for the fixed prescribed-model child and retain its actual OS exit.

This is not a general command runner. It accepts the same worker arguments,
requires the host's workspace/audit-root/job-id, preserves the argument vector
and environment, and can signal only its own live Popen child. The existing
worker owns model/GPU admission. No model package is imported here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import UUID

from blake3 import blake3

WORKER = Path(__file__).resolve().with_name("run_prescribed_model.py")


def _targets(arguments):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--job-id", type=UUID, required=True)
    args, _ = parser.parse_known_args(arguments)
    # The child historically permits option abbreviations. Do not allow an
    # unparsed abbreviated binding or duplicate flag to select another target.
    for option in ("--workspace", "--audit-root", "--job-id"):
        keys = [item.split("=", 1)[0] for item in arguments if item.startswith("--")]
        if keys.count(option) != 1 or any(key != option and option.startswith(key) for key in keys):
            raise ValueError("Host job bindings must be explicit and unique")
    workspace = args.workspace.resolve(strict=True)
    audit_root = args.audit_root.resolve()
    if audit_root.is_relative_to(workspace):
        raise ValueError("Authoritative job audit must be outside actor workspace")
    job_id = str(args.job_id)
    audit = audit_root / job_id
    public = workspace / "outputs/agents_outputs/prescribed-model-jobs" / job_id
    if public.resolve() != public or audit.resolve() != audit:
        raise ValueError("Job output path traverses an unexpected symlink")
    if audit.exists() or public.exists():
        raise FileExistsError("This job attempt already has artifacts; never rerun it")
    return job_id, audit, public


def _write_once(path, document):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.resolve() != path.parent:
        raise ValueError("Job output path changed while child ran")
    payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    # Readers see either the complete atomic receipt or no receipt.
    temporary = path.with_suffix(".json.incomplete")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    # A hard link publishes without replacing any existing attempt evidence.
    path.hardlink_to(temporary)
    temporary.unlink()


def supervise(arguments):
    arguments = list(arguments)
    job_id, audit, public = _targets(arguments)
    worker_source = WORKER.read_bytes()
    supervisor_source = Path(__file__).read_bytes()
    started = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    child = None
    received = []
    pending = []

    def forward(signum, _frame):
        received.append(signum)
        if child is not None:
            # Popen polls its owned child before sending; no os.kill/killpg or
            # external PID input, and no signal to unrelated descendants/jobs.
            child.send_signal(signum)
        else:
            pending.append(signum)

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    for signum in previous:
        signal.signal(signum, forward)
    try:
        try:
            child = subprocess.Popen([sys.executable, "-B", str(WORKER), *arguments],
                stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        except OSError as error:
            failure = {"schema": "eva.prescribed-model-supervisor-error.v1", "job_id": job_id,
                       "error_type": type(error).__name__, "os_process_exit_observed": False}
            _write_once(audit / "supervisor-error.json", failure)
            _write_once(public / "supervisor-error.json", failure)
            raise
        # Covers termination received during Popen before the child was bound.
        for signum in pending:
            child.send_signal(signum)
        returncode = child.wait()
        receipt_path = audit / "receipt.json"
        result = {"schema": "eva.prescribed-model-process-exit.v1", "job_id": job_id,
                  "returncode": returncode, "os_process_exit_observed": True, "child_pid": child.pid,
                  "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": time.monotonic() - started_clock,
                  "forwarded_signals": received,
                  "worker_source_blake3_at_launch": blake3(worker_source).hexdigest(),
                  "supervisor_source_blake3": blake3(supervisor_source).hexdigest(),
                  "worker_receipt_blake3": blake3(receipt_path.read_bytes()).hexdigest() if receipt_path.is_file() else None}
        _write_once(audit / "process-exit.json", result)
        _write_once(public / "process-exit.json", result)
        return returncode
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None):
    return supervise(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    status = main()
    raise SystemExit(status if status >= 0 else 128 - status)
