"""Private Python edits become public only through a measured host publication."""
from __future__ import annotations

import os
import json
import math
import errno
from pathlib import Path
import shutil
import time

from training.automedbench_lite.adapter import EvaluationError, file_digest, read_document, write_once
from training.benchmark_models._publication import DeadlineGuard, PolicyDeadlineExceeded
from .benchmark_inventory import PublishingInventory


READ_ONLY = ('inputs', 'task.json', 'inputs-manifest.json', 'public-guidance',
             'outputs/agents_outputs/prescribed-model-jobs')


def readonly(path: str) -> bool:
    return any(path == root or path.startswith(root + '/') for root in READ_ONLY)


def prepare_private(workspace: Path, audit: Path, before: dict) -> Path:
    """Copy verified mutable blobs; immutable inputs/model outputs remain ro mounts."""
    private = audit / 'private-workspace'
    private.mkdir(mode=0o700)
    for directory in ('inputs', 'notes', 'code', 'outputs/agents_outputs', 'public-guidance'):
        (private / directory).mkdir(parents=True, exist_ok=True, mode=0o700)
    for row in before['files']:
        relative = row['path']
        if readonly(relative) and relative not in ('task.json', 'inputs-manifest.json'):
            continue
        target = private / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Inventory has already rejected links/nonregular files and bound bytes.
        shutil.copyfile(workspace / relative, target)
        target.chmod(row['mode'])
        if file_digest(target) != row['blake3']:
            raise ValueError('cpu_staging_source_changed')
    for relative in READ_ONLY:
        source = workspace / relative
        if source.is_dir():
            (private / relative).mkdir(parents=True, exist_ok=True, mode=0o700)
    return private


def publish_private(workspace: Path, audit: Path, before: dict, deadline: float | None,
                    *, execution_succeeded: bool = True) -> dict:
    guard = DeadlineGuard(audit, deadline)
    admitted, denied, changes = False, False, []
    private = audit / 'private-workspace'
    try:
        if not execution_succeeded:
            guard.record('cpu_private_edits_discarded', reason='execution_failed_or_timed_out')
            return write_once(audit / 'cpu-publication.json', {
                'schema': 'eva.benchmark-cpu-private-publication.v1',
                'private_workspace': str(private), 'actor_direct_writable_mount': False,
                'public_workspace': str(workspace),
                'track_deadline_monotonic': deadline, 'completed_publication': False,
                'publication_denied_at_deadline': deadline is not None and time.monotonic() >= deadline,
                'execution_succeeded': False, 'changes': [], 'guard_evidence': guard.evidence(),
                'finished_monotonic': time.monotonic(), 'unpublished_private_edits_not_scored': True})
        guard.check('cpu_publication_preparation')
        try:
            observed = PublishingInventory(private, audit / 'private-inventory').capture()
        except (EvaluationError, OSError) as exc:
            rejection = 'track_workspace_symlink' if isinstance(exc, OSError) and exc.errno == errno.ELOOP else str(exc)
            if rejection not in {'track_workspace_symlink', 'track_mutable_file_invalid', 'track_mutable_workspace_limit'}:
                raise
            guard.record('cpu_private_edits_discarded', reason='invalid_policy_authored_artifacts')
            return write_once(audit / 'cpu-publication.json', {
                'schema': 'eva.benchmark-cpu-private-publication.v1',
                'private_workspace': str(private), 'public_workspace': str(workspace),
                'actor_direct_writable_mount': False, 'track_deadline_monotonic': deadline,
                'completed_publication': False, 'publication_denied_at_deadline': False,
                'execution_succeeded': True, 'policy_artifact_rejection': rejection,
                'changes': [], 'guard_evidence': guard.evidence(), 'finished_monotonic': time.monotonic(),
                'unpublished_private_edits_not_scored': True})
        previous = {row['path']: row for row in before['files'] if not readonly(row['path'])}
        current = {row['path']: row for row in observed['files'] if not readonly(row['path'])}
        for relative in sorted(previous.keys() | current.keys()):
            old, new = previous.get(relative), current.get(relative)
            if old is not None and new is not None and (old['blake3'], old['mode']) == (new['blake3'], new['mode']):
                continue
            target, source = workspace / relative, private / relative
            if old is not None and (not target.is_file() or target.is_symlink() or file_digest(target) != old['blake3']):
                raise ValueError('cpu_publication_concurrent_actor_mutation')
            if old is None and (target.exists() or target.is_symlink()):
                raise ValueError('cpu_publication_destination_changed')
            # Parent-directory creation is itself a public mutation, measured
            # separately. Empty directories are never scoring artifacts.
            missing = []
            parent = target.parent
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for directory in reversed(missing):
                guard.publish('cpu_create_directory', directory, lambda path=directory: path.mkdir(mode=0o700))
            operation = 'cpu_delete_file' if new is None else 'cpu_publish_file'
            guard.publish(operation, target, (lambda path=target: path.unlink()) if new is None
                          else (lambda src=source, dst=target: os.replace(src, dst)))
            changes.append({'path': relative, 'operation': operation,
                            'previous_blake3': old['blake3'] if old else None,
                            'published_blake3': new['blake3'] if new else None})
        admitted = True
    except PolicyDeadlineExceeded:
        denied = True
    return write_once(audit / 'cpu-publication.json', {
        'schema': 'eva.benchmark-cpu-private-publication.v1',
        'private_workspace': str(private), 'actor_direct_writable_mount': False,
        'public_workspace': str(workspace),
        'track_deadline_monotonic': deadline, 'completed_publication': admitted,
        'publication_denied_at_deadline': denied, 'changes': changes,
        'execution_succeeded': True,
        'guard_evidence': guard.evidence(), 'finished_monotonic': time.monotonic(),
        'unpublished_private_edits_not_scored': True})


def verify_cpu_publication(audit: Path, deadline: float, execution: dict, *, workspace: Path | None = None) -> dict:
    publication = read_document(audit / 'cpu-publication.json')
    if (publication['document_blake3'] != execution.get('cpu_publication_document_blake3')
            or publication.get('track_deadline_monotonic') != deadline
            or publication.get('actor_direct_writable_mount') is not False
            or publication.get('unpublished_private_edits_not_scored') is not True
            or execution.get('isolation', {}).get('actor_direct_writable_mount') is not False):
        raise ValueError('cpu_publication_binding_invalid')
    public_workspace = Path(publication['public_workspace'])
    if (publication['private_workspace'] != str(audit / 'private-workspace')
            or workspace is not None and public_workspace != workspace
            or public_workspace.resolve() != public_workspace):
        raise ValueError('cpu_publication_workspace_binding_invalid')
    guard = publication['guard_evidence']
    path = audit / 'deadline-publication-events.jsonl'
    if guard['track_deadline_monotonic'] != deadline or guard['path'] != str(path):
        raise ValueError('cpu_publication_guard_binding_invalid')
    if guard['blake3'] is None:
        if path.exists() or publication['changes']:
            raise ValueError('cpu_publication_events_missing')
        events = []
    else:
        if file_digest(path) != guard['blake3'] or path.stat().st_size > 16 * 1024**2:
            raise ValueError('cpu_publication_events_changed')
        events = [json.loads(line) for line in path.read_bytes().splitlines()]
    intentions = {}
    for row in events:
        if row.get('track_deadline_monotonic') != deadline or row.get('host_only') is not True:
            raise ValueError('cpu_publication_event_identity_changed')
        if row['event'] == 'public_publication':
            start, end = row['publication_started_monotonic'], row['publication_completed_monotonic']
            if (not all(type(value) in (int, float) and math.isfinite(value) for value in (start, end))
                    or not 0 <= start <= end < deadline
                    or row.get('complete_interval_before_deadline') is not True):
                raise ValueError('cpu_publication_crossed_track_deadline')
        if row['event'] in {'public_publication_intent', 'public_publication', 'public_publication_cancelled'}:
            intention = intentions.setdefault(row['publication_id'], [])
            intention.append(row)
    for rows in intentions.values():
        if (len(rows) != 2 or rows[0]['event'] != 'public_publication_intent'
                or rows[1]['event'] not in {'public_publication', 'public_publication_cancelled'}
                or rows[0]['operation'] != rows[1]['operation']
                or rows[0]['destination'] != rows[1]['destination']
                or rows[0]['observed_monotonic'] > rows[1]['observed_monotonic']
                or (rows[1]['event'] == 'public_publication_cancelled' and rows[1].get('action_started') is not False)):
            raise ValueError('cpu_publication_intent_unclosed_or_changed')
    succeeded = execution['exit_code'] == 0 and not execution['timed_out'] and not execution['stream_limit_exceeded']
    if publication['execution_succeeded'] != succeeded:
        raise ValueError('cpu_publication_execution_status_changed')
    if not succeeded and (publication['changes'] or any(row['event'] == 'public_publication' for row in events)):
        raise ValueError('failed_cpu_execution_published_edits')
    if publication.get('policy_artifact_rejection') and (publication['changes'] or any(row['event'] == 'public_publication' for row in events)):
        raise ValueError('rejected_cpu_artifacts_published')
    actual = [row for row in events if row['event'] == 'public_publication' and row['operation'] != 'cpu_create_directory']
    if len(actual) != len(publication['changes']):
        raise ValueError('cpu_publication_changes_incomplete')
    for event, change in zip(actual, publication['changes']):
        relative = Path(change['path'])
        if (relative.is_absolute() or '..' in relative.parts or readonly(relative.as_posix())
                or event['destination'] != str(public_workspace / relative)
                or event['operation'] != change['operation']
                or change['operation'] not in {'cpu_publish_file', 'cpu_delete_file'}):
            raise ValueError('cpu_publication_change_binding_invalid')
    return {'publication_document_blake3': publication['document_blake3'], 'guard_events_blake3': guard['blake3'],
            'publications': len(actual), 'private_edits_discarded': not succeeded or publication['publication_denied_at_deadline'],
            'publication_boundary_verified': True}
