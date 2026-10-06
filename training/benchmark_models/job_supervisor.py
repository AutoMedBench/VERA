"""Wait for the fixed prescribed-model child and retain its actual OS exit.

This is not a general command runner. It accepts the same worker arguments,
requires the host's workspace/audit-root/job-id, preserves the argument vector
and environment. Deadline cancellation signals only its birth-bound child's
private session and observed members. The worker owns model/GPU admission.
No model package is imported here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import UUID
from uuid import uuid4

from blake3 import blake3

if __package__:
    from ._publication import atomic_new, DeadlineGuard, PolicyDeadlineExceeded, PublicationBoundaryViolation
else:
    from _publication import atomic_new, DeadlineGuard, PolicyDeadlineExceeded, PublicationBoundaryViolation

WORKER = Path(__file__).resolve().with_name("run_prescribed_model.py")


def namespace_identity():
    proc_pid = int(Path('/proc/self/stat').read_text().split(' ', 1)[0])
    return {'pid_namespace': os.readlink('/proc/self/ns/pid'),
            'mount_namespace': os.readlink('/proc/self/ns/mnt'),
            'proc_self_pid': proc_pid, 'runtime_getpid': os.getpid(),
            'proc_pid_numbers_match_runtime': proc_pid == os.getpid(),
            'proc_mountinfo': [line for line in Path('/proc/self/mountinfo').read_text().splitlines()
                               if len(line.split()) > 4 and line.split()[4] == '/proc'],
            'node': os.uname().nodename,
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(fields[19]), 'parent_pid': int(fields[1]),
                'process_group_id': int(fields[2]), 'session_id': int(fields[3]), 'state': fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return {'pid': pid, 'start_ticks': None, 'parent_pid': None,
                'process_group_id': None, 'session_id': None, 'state': None}


def session_members(session_id):
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            identity = process_identity(int(path.name))
        except (PermissionError, ProcessLookupError):
            continue
        if identity['session_id'] == session_id:
            result.append(identity)
    return sorted(result, key=lambda row: row['pid'])


def stop_owned_session(child, expected_identity):
    """Freeze a birth-bound private child session, then kill its observed members."""
    current = process_identity(child.pid)
    if (child.poll() is not None or expected_identity is None or
            current['start_ticks'] != expected_identity['start_ticks'] or
            current['start_ticks'] is None or current['session_id'] != child.pid or
            current['process_group_id'] != child.pid):
        raise RuntimeError('deadline_owned_child_session_identity_unverified')
    signals = [{'signal': 'SIGSTOP', 'target_process_group': child.pid,
                'monotonic': time.monotonic()}]
    os.killpg(child.pid, signal.SIGSTOP)
    # The worker launched a new session. Include all observed groups in that
    # session, so a forked helper is retained even after its direct parent dies.
    observed = {}
    for _ in range(8):
        members = session_members(child.pid)
        fresh = [row for row in members if (row['pid'], row['start_ticks']) not in observed]
        for row in fresh:
            observed[(row['pid'], row['start_ticks'])] = row
            live = process_identity(row['pid'])
            if live['start_ticks'] == row['start_ticks'] and live['state'] not in {'Z', None}:
                try:
                    os.kill(row['pid'], signal.SIGSTOP)
                    signals.append({'signal': 'SIGSTOP', 'target_identity': row,
                                    'monotonic': time.monotonic()})
                except ProcessLookupError:
                    pass
        if not fresh:
            break
    else:
        raise RuntimeError('deadline_owned_session_did_not_stabilize')
    # Kill the original group while its birth-bound, frozen leader is retained.
    os.killpg(child.pid, signal.SIGKILL)
    signals.append({'signal': 'SIGKILL', 'target_process_group': child.pid,
                    'monotonic': time.monotonic()})
    # A helper may have made another group within the same owned session.
    for row in observed.values():
        live = process_identity(row['pid'])
        if live['start_ticks'] == row['start_ticks'] and live['state'] not in {'Z', None}:
            try:
                os.kill(row['pid'], signal.SIGKILL)
                signals.append({'signal': 'SIGKILL', 'target_identity': row,
                                'monotonic': time.monotonic()})
            except ProcessLookupError:
                pass
    return list(observed.values()), signals


def _targets(arguments):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--job-id", type=UUID, required=True)
    parser.add_argument("--track-deadline-monotonic", type=float)
    args, _ = parser.parse_known_args(arguments)
    # The child historically permits option abbreviations. Do not allow an
    # unparsed abbreviated binding or duplicate flag to select another target.
    for option in ("--workspace", "--audit-root", "--job-id"):
        keys = [item.split("=", 1)[0] for item in arguments if item.startswith("--")]
        if keys.count(option) != 1 or any(key != option and option.startswith(key) for key in keys):
            raise ValueError("Host job bindings must be explicit and unique")
    deadline_keys = [item.split('=', 1)[0] for item in arguments if item.startswith('--')]
    if (deadline_keys.count('--track-deadline-monotonic') > 1 or any(
            key not in {'--track', '--track-deadline-monotonic'} and '--track-deadline-monotonic'.startswith(key)
            for key in deadline_keys)):
        raise ValueError('Trusted deadline binding must be explicit and unique')
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
    DeadlineGuard(audit, args.track_deadline_monotonic)
    return job_id, audit, public, args.track_deadline_monotonic


def _write_once(path, document, *, private_staging=None, deadline_guard=None):
    if deadline_guard is not None:
        deadline_guard.check('supervisor_public_metadata')
    if deadline_guard is None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.resolve() != path.parent:
        raise ValueError("Job output path changed while child ran")
    payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    # Readers see either the complete atomic receipt or no receipt.
    staging = private_staging or path.parent
    staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = staging / (str(uuid4()) + ".incomplete")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    if deadline_guard is None:
        atomic_new(temporary, path)
    else:
        def publish():
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            atomic_new(temporary, path)
        deadline_guard.publish('supervisor_public_metadata', path, publish)


def supervise(arguments):
    arguments = list(arguments)
    job_id, audit, public, track_deadline = _targets(arguments)
    guard = DeadlineGuard(audit, track_deadline, component='supervisor')
    worker_source = WORKER.read_bytes()
    supervisor_source = Path(__file__).read_bytes()
    started = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    child = None
    received = []
    pending = []
    policy_cancelled = False
    termination = []
    dispatched = None
    supervisor_identity = process_identity(os.getpid())
    namespaces = namespace_identity()
    child_identity = None
    exit_observed = None
    stopped_members = []
    death_observations = []
    session_quiescent = None
    remaining_members = []

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
            guard.check('worker_dispatch')
        except PolicyDeadlineExceeded:
            cancelled = {'schema': 'eva.prescribed-model-policy-cancellation.v1', 'job_id': job_id,
                'status': 'cancelled', 'policy_cancellation': True,
                'cancellation_reason': 'track_wall_clock_budget_exhausted',
                'worker_dispatched': False, 'os_process_exit_observed': False,
                'deadline_publication_evidence': guard.evidence()}
            _write_once(audit / 'policy-cancellation.json', cancelled)
            return 0
        try:
            dispatched = time.monotonic()
            child = subprocess.Popen([sys.executable, "-B", str(WORKER), *arguments],
                stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
            child_identity = process_identity(child.pid)
        except OSError as error:
            failure = {"schema": "eva.prescribed-model-supervisor-error.v1", "job_id": job_id,
                       "error_type": type(error).__name__, "os_process_exit_observed": False}
            _write_once(audit / "supervisor-error.json", failure)
            try:
                _write_once(public / "supervisor-error.json", failure,
                    private_staging=audit / "supervisor-publication", deadline_guard=guard)
            except PolicyDeadlineExceeded:
                pass
            raise
        # Covers termination received during Popen before the child was bound.
        for signum in pending:
            child.send_signal(signum)
        try:
            returncode = child.wait() if track_deadline is None else child.wait(timeout=max(0, track_deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            policy_cancelled = True
            guard.record('policy_cancellation', operation='supervisor_deadline', policy_cancellation=True,
                         cancellation_reason='track_wall_clock_budget_exhausted')
            # Observation grace grants no additional model computation. Stop
            # the owned child immediately, then spend at most three seconds
            # observing its actual OS exit for the host-only receipt.
            stopped_members, termination = stop_owned_session(child, child_identity)
            try:
                returncode = child.wait(timeout=max(0, track_deadline+3-time.monotonic()))
                while True:
                    remaining = [row for row in session_members(child.pid) if row['state'] != 'Z']
                    if not remaining:
                        break
                    if time.monotonic() >= track_deadline + 3:
                        raise subprocess.TimeoutExpired('owned model session exit observation', 3)
                    time.sleep(.01)
                death_observations = [{**row, 'current_identity': process_identity(row['pid']),
                                       'observed_monotonic': time.monotonic()}
                                      for row in stopped_members]
                session_quiescent = True
            except subprocess.TimeoutExpired:
                _write_once(audit / 'deadline-exit-unobserved.json', {
                    'schema': 'eva.prescribed-model-deadline-exit-unobserved.v1', 'job_id': job_id,
                    'supervisor_identity': supervisor_identity, 'child_identity': child_identity,
                    'deadline_termination': termination, 'os_process_exit_observed': False,
                    'observed_session_members': stopped_members, 'owned_session_quiescent': False,
                    'observation_only_grace_seconds': 3, 'host_only': True,
                    'deadline_publication_evidence': guard.evidence()})
                raise
        exit_observed = time.monotonic()
        if namespaces['proc_pid_numbers_match_runtime']:
            remaining_members = [row for row in session_members(child.pid) if row['state'] != 'Z']
            session_quiescent = not remaining_members
        else:
            # A caller may expose host /proc inside another PID namespace.
            # Those numeric identities cannot prove child-session quiescence.
            session_quiescent = False
        receipt_path = audit / "receipt.json"
        if receipt_path.is_file():
            worker_receipt = json.loads(receipt_path.read_bytes())
            policy_cancelled |= worker_receipt.get('policy_cancellation') is True
        result = {"schema": "eva.prescribed-model-process-exit.v1", "job_id": job_id,
                  "returncode": returncode, "os_process_exit_observed": True, "child_pid": child.pid,
                  "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": time.monotonic() - started_clock,
                  "dispatch_monotonic": dispatched, "track_deadline_monotonic": track_deadline,
                  'supervisor_identity': supervisor_identity, 'child_identity': child_identity,
                  'namespace_identity': namespaces, 'remaining_session_members': remaining_members,
                  'exit_observed_monotonic': exit_observed, 'observation_only_grace_seconds': 3,
                  'post_deadline_computation_authorized': False,
                  'observed_session_members': stopped_members, 'member_death_observations': death_observations,
                  'owned_session_quiescent': session_quiescent,
                  "policy_cancellation": policy_cancelled,
                  "deadline_termination": termination, "deadline_publication_evidence": guard.evidence(),
                  "forwarded_signals": received,
                  "worker_source_blake3_at_launch": blake3(worker_source).hexdigest(),
                  "supervisor_source_blake3": blake3(supervisor_source).hexdigest(),
                  "worker_receipt_blake3": blake3(receipt_path.read_bytes()).hexdigest() if receipt_path.is_file() else None}
        _write_once(audit / "process-exit.json", result)
        publication_failure = False
        try:
            _write_once(public / "process-exit.json", result,
                        private_staging=audit / "supervisor-publication", deadline_guard=guard)
        except PolicyDeadlineExceeded:
            pass
        except PublicationBoundaryViolation:
            publication_failure = True
            raise
        finally:
            _write_once(audit / 'supervision-final.json', {'schema': 'eva.prescribed-model-supervision-final.v1',
                'job_id': job_id, 'policy_cancellation': policy_cancelled,
                'supervisor_identity': supervisor_identity, 'child_identity': child_identity,
                'namespace_identity': namespaces, 'remaining_session_members': remaining_members,
                'deadline_termination': termination, 'exit_observed_monotonic': exit_observed,
                'finalized_monotonic': time.monotonic(), 'track_deadline_monotonic': track_deadline,
                'observation_only_grace_seconds': 3, 'post_deadline_computation_authorized': False,
                'observed_session_members': stopped_members, 'member_death_observations': death_observations,
                'owned_session_quiescent': session_quiescent,
                'deadline_publication_evidence': guard.evidence(), 'returncode': returncode,
                'publication_boundary_violation': publication_failure, 'host_only': True})
        return returncode
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None):
    return supervise(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    status = main()
    raise SystemExit(status if status >= 0 else 128 - status)
