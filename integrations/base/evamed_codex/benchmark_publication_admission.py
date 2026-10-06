"""Reopen prescribed-model publication evidence; absence is never a policy score.

This helper performs no process control and signs nothing. Its caller must bind
the returned byte hashes into the fresh host finalization signature and verify
the owned descendant-tree cleanup separately. An interrupted publisher without
complete write-ahead publication evidence remains infrastructure-unknown.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
from uuid import UUID

from training.automedbench_lite.adapter import blake3, canonical, file_digest, read_document

ROOT = Path(__file__).resolve().parents[3]
SOURCE_NAMES = ('_publication.py', 'run_prescribed_model.py', 'job_supervisor.py')
# Reviewed, versioned implementations only. A caller cannot bless arbitrary code
# by supplying its own digest. Old archives remain independently reopenable.
REVIEWED_SOURCE_PROFILES = ({
    '_publication.py': 'e19a9431f8b77c086d4c861a86e8b63ea5ebee0bd10c8acb8c712a9c50650ef9',
    'run_prescribed_model.py': '882c20c179e2ec72523f36c9c906c6d8b4f76e7f4467615e0c3f5f0d01a4851b',
    'job_supervisor.py': 'ae6bd4929533db208a59b2e32a9f003d2c1f689df866d107604a21897cb55512',
}, {
    '_publication.py': '48ec9658d332dd735a4b10f76ef26dc4747eba13edd6020000b6623c1f3241d5',
    'run_prescribed_model.py': '882c20c179e2ec72523f36c9c906c6d8b4f76e7f4467615e0c3f5f0d01a4851b',
    'job_supervisor.py': '4cb561f893f51f16d09bb07355ac47935641847b37146953dde604bb8aec4f19',
}, {
    '_publication.py': '48ec9658d332dd735a4b10f76ef26dc4747eba13edd6020000b6623c1f3241d5',
    'run_prescribed_model.py': '882c20c179e2ec72523f36c9c906c6d8b4f76e7f4467615e0c3f5f0d01a4851b',
    'job_supervisor.py': '41bf7804ff786ea1a6429d2355a228c830794f64bc438f48f3eb947da7e41095',
})
SUBMIT_TOOLS = {'automed_submit_model_job', 'automed_submit_extended_model_job',
                'automed_submit_generative_model_job'}


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _uuid(value):
    _require(isinstance(value, str) and str(UUID(value)) == value, 'publication_job_uuid_invalid')
    return value


def _file(path: Path, maximum=32 * 1024**2):
    path = Path(path)
    _require(path.absolute() == path.resolve(), 'publication_evidence_symlink')
    info = path.stat()
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= maximum,
             'publication_evidence_topology_or_size_invalid')
    return path


def _json(path):
    value = json.loads(_file(path).read_bytes())
    _require(isinstance(value, dict), 'publication_evidence_object_required')
    return value


def _rows(path):
    data = _file(path, 256 * 1024**2).read_bytes()
    _require(not data or data.endswith(b'\n'), 'publication_journal_truncated')
    rows = [json.loads(line) for line in data.splitlines()]
    _require(all(isinstance(row, dict) for row in rows), 'publication_journal_row_invalid')
    return rows


def _dead(process):
    _require(type(process.get('pid')) is int and process['pid'] > 0
             and str(process.get('start_ticks', '')).isdigit(), 'publication_process_identity_invalid')
    try:
        fields = Path(f"/proc/{process['pid']}/stat").read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return True
    return fields[19] != str(process['start_ticks']) or fields[0] == 'Z'


def verify_publication_journal(path: Path, deadline: float, *, output: Path,
                               component: str, evidence: dict | None = None,
                               require_intents: bool = True) -> dict:
    """Validate every interval and exact intent terminal, including denials.

    Legacy completed journals can be reopened with require_intents=False only
    when a source-bound final worker/supervisor receipt proves orderly closure.
    This mode never admits a forcibly killed, potentially unlogged publisher.
    """
    _require(_time(deadline) and component in {'worker', 'supervisor'}, 'publication_clock_invalid')
    path, output = Path(path).absolute(), Path(output).absolute()
    digest = file_digest(_file(path)) if path.exists() else None
    if evidence is not None:
        _require(evidence == {'path': str(path), 'blake3': digest,
                             'track_deadline_monotonic': deadline}, 'publication_journal_commitment_changed')
    rows = _rows(path) if digest is not None else []
    identifiers, intents, terminals = set(), {}, set()
    publications, admissions, cancellations = [], [], []
    previous = 0
    for row in rows:
        _require(row.get('schema') == 'eva.prescribed-model-deadline-event.v1'
                 and row.get('host_only') is True and row.get('track_deadline_monotonic') == deadline,
                 'publication_event_identity_invalid')
        event_id = _uuid(row.get('event_id'))
        _require(event_id not in identifiers, 'publication_event_duplicate')
        identifiers.add(event_id)
        observed = row.get('observed_monotonic')
        _require(_time(observed) and previous <= observed, 'publication_event_time_invalid')
        previous = observed
        event = row.get('event')
        if event == 'work_admission':
            admitted = row.get('admitted_monotonic')
            _require(_time(admitted) and admitted <= observed and admitted < deadline
                     and not cancellations, 'model_work_admitted_after_deadline')
            admissions.append(row)
        elif event == 'policy_cancellation':
            _require(observed >= deadline and row.get('policy_cancellation') is True
                     and row.get('cancellation_reason') == 'track_wall_clock_budget_exhausted',
                     'model_policy_cancellation_unproved')
            cancellations.append(row)
        elif event in {'public_publication_intent', 'public_publication', 'public_publication_cancelled'}:
            destination = Path(row.get('destination', ''))
            _require(destination.is_absolute() and destination == destination.resolve()
                     and destination.is_relative_to(output), 'publication_destination_outside_job')
            operation = row.get('operation')
            allowed = ({'public_job_directory', 'publish_case', 'publish_auxiliary', 'publish_metadata'}
                       if component == 'worker' else {'supervisor_public_metadata'})
            _require(operation in allowed, 'publication_operation_invalid')
            if operation == 'public_job_directory':
                _require(destination == output, 'public_job_directory_binding_invalid')
            publication_id = row.get('publication_id')
            if event == 'public_publication_intent':
                _uuid(publication_id)
                _require(publication_id not in intents, 'publication_intent_duplicate')
                intents[publication_id] = row
                continue
            if require_intents or publication_id is not None:
                _require(publication_id in intents and publication_id not in terminals,
                         'publication_terminal_without_unique_intent')
                intent = intents[publication_id]
                _require(intent['operation'] == operation and intent['destination'] == str(destination),
                         'publication_intent_binding_changed')
                terminals.add(publication_id)
            else:
                intent = None
            if event == 'public_publication_cancelled':
                _require(intent is not None and observed >= deadline
                         and row.get('action_started') is False and row.get('policy_cancellation') is True
                         and any(cancel['operation'] == operation and cancel['observed_monotonic'] >=
                                 intent['observed_monotonic'] for cancel in cancellations),
                         'publication_intent_cancellation_unproved')
                continue
            start, end = row.get('publication_started_monotonic'), row.get('publication_completed_monotonic')
            _require(_time(start) and _time(end) and 0 <= start <= end < deadline
                     and end <= observed and row.get('complete_interval_before_deadline') is True,
                     'publication_interval_crossed_track_deadline')
            if intent is not None:
                _require(intent['observed_monotonic'] <= start, 'publication_action_preceded_intent')
            publications.append(row)
        else:
            raise ValueError('publication_event_unrecognized')
    _require(set(intents) == terminals, 'publication_intent_without_proved_terminal')
    return {'path': str(path), 'blake3': digest, 'event_count': len(rows),
            'publication_count': len(publications), 'publications': publications,
            'work_admissions': admissions, 'policy_cancellations': cancellations,
            'all_publication_intervals_before_deadline': True,
            'write_ahead_protocol_required': require_intents}


def _binding(arguments, option):
    matches = [index for index, value in enumerate(arguments) if value == option]
    _require(len(matches) == 1 and matches[0] + 1 < len(arguments), 'model_command_binding_missing_or_duplicate')
    return arguments[matches[0] + 1]


def _journal_prefix(path, evidence, deadline):
    """A killed worker may retain an earlier receipt, never a final receipt."""
    _require(evidence.get('path') == str(path) and evidence.get('track_deadline_monotonic') == deadline,
             'publication_worker_prefix_identity_invalid')
    expected = evidence.get('blake3')
    if expected is None:
        return
    digest = blake3()
    matched = expected == digest.hexdigest()
    for line in _file(path).read_bytes().splitlines(keepends=True):
        digest.update(line)
        matched |= digest.hexdigest() == expected
    _require(matched, 'publication_worker_receipt_journal_prefix_changed')


def verify_session_exit(exited, final, deadline):
    """Authenticate source-recorded session death, not cross-namespace PID guesses."""
    keys = ('supervisor_identity', 'child_identity', 'namespace_identity',
            'remaining_session_members', 'observed_session_members', 'member_death_observations',
            'owned_session_quiescent', 'deadline_termination', 'exit_observed_monotonic')
    _require(all(key in exited and key in final and exited[key] == final[key] for key in keys),
             'publication_session_final_binding_invalid')
    namespace = exited['namespace_identity']
    _require(namespace.get('proc_pid_numbers_match_runtime') is True
             and namespace.get('proc_self_pid') == namespace.get('runtime_getpid') == exited['supervisor_identity']['pid']
             and isinstance(namespace.get('pid_namespace'), str) and namespace['pid_namespace'].startswith('pid:[')
             and isinstance(namespace.get('mount_namespace'), str) and namespace['mount_namespace'].startswith('mnt:[')
             and namespace.get('proc_mountinfo') and namespace.get('node') and namespace.get('boot_id')
             and exited['remaining_session_members'] == [] and exited['owned_session_quiescent'] is True,
             'publication_owned_session_quiescence_unproved')
    child = exited['child_identity']
    _require(type(child.get('pid')) is int and child['pid'] > 0
             and type(child.get('start_ticks')) is int and child['start_ticks'] > 0
             and child.get('session_id') == child.get('process_group_id') == child['pid']
             and child['pid'] == exited['child_pid'], 'publication_child_session_identity_invalid')
    # Additional liveness check only when these are actually the same numbers.
    if (namespace['pid_namespace'] == os.readlink('/proc/self/ns/pid')
            and namespace['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        _require(_dead(child) and _dead(exited['supervisor_identity']), 'publication_owned_process_still_live')
    observed_exit, finalized = exited['exit_observed_monotonic'], final.get('finalized_monotonic')
    _require(_time(observed_exit) and _time(finalized)
             and exited['dispatch_monotonic'] <= observed_exit <= finalized,
             'publication_supervisor_final_time_invalid')
    termination, members, deaths = (exited[key] for key in
        ('deadline_termination', 'observed_session_members', 'member_death_observations'))
    _require(all(isinstance(value, list) for value in (termination, members, deaths)),
             'publication_session_inventory_shape_invalid')
    if termination:
        _require(exited['policy_cancellation'] is True and observed_exit <= deadline + 3
                 and termination[0].get('signal') == 'SIGSTOP'
                 and termination[0].get('target_process_group') == child['pid']
                 and _time(termination[0].get('monotonic'))
                 and deadline <= termination[0]['monotonic'] <= deadline + 1,
                 'publication_session_deadline_freeze_unproved')
        previous, group_killed = deadline, False
        inventory = {(row['pid'], row['start_ticks']): row for row in members}
        _require(len(inventory) == len(members) and (child['pid'], child['start_ticks']) in inventory,
                 'publication_session_member_inventory_invalid')
        _require(all(row.get('session_id') == child['pid'] for row in members),
                 'publication_foreign_session_member')
        for signal_row in termination:
            clock = signal_row.get('monotonic')
            _require(_time(clock) and previous <= clock <= observed_exit
                     and signal_row.get('signal') in {'SIGSTOP', 'SIGKILL'},
                     'publication_session_signal_time_or_type_invalid')
            previous = clock
            if 'target_process_group' in signal_row:
                _require(signal_row['target_process_group'] == child['pid'], 'publication_foreign_group_signalled')
                group_killed |= signal_row['signal'] == 'SIGKILL'
            else:
                identity = signal_row.get('target_identity', {})
                _require(inventory.get((identity.get('pid'), identity.get('start_ticks'))) == identity,
                         'publication_unregistered_member_signalled')
        _require(group_killed, 'publication_owned_group_not_killed')
        observed = {}
        for row in deaths:
            identity = {key: value for key, value in row.items() if key not in {'current_identity', 'observed_monotonic'}}
            key = (identity.get('pid'), identity.get('start_ticks'))
            _require(key in inventory and key not in observed and inventory[key] == identity,
                     'publication_member_death_inventory_invalid')
            current, clock = row.get('current_identity', {}), row.get('observed_monotonic')
            _require(current.get('pid') == identity['pid'] and _time(clock)
                     and previous <= clock <= observed_exit
                     and (current.get('state') in {None, 'Z'} or current.get('start_ticks') != identity['start_ticks']),
                     'publication_member_death_unproved')
            observed[key] = row
        _require(set(observed) == set(inventory), 'publication_member_death_inventory_incomplete')
    else:
        _require(members == [] and deaths == [], 'publication_unexplained_session_death_evidence')
    return {'namespace_identity': namespace, 'observed_member_count': len(members),
            'owned_session_quiescent': True, 'exit_observed_monotonic': observed_exit}


def verify_publication_jobs(audit: Path, workspace: Path, deadline: float, *,
                            expected_job_ids=None, source_paths=None, cleanup=None) -> dict:
    """Reopen the exact submitted job set and source-bound final publications.

    Unknown/missing final evidence raises ValueError. A zero-job track is valid.
    Live process checks are an additional current observation, never a substitute
    for the caller's signed cleanup and historical OS-exit evidence.
    """
    audit, workspace = Path(audit).absolute(), Path(workspace).absolute()
    _require(audit == audit.resolve() and workspace == workspace.resolve()
             and not audit.is_relative_to(workspace) and _time(deadline), 'publication_root_or_clock_invalid')
    job_root = audit / 'model-jobs'
    registrations = {path.parent.name: path for path in job_root.glob('*/submission.json')}
    _require(len(registrations) <= 80, 'publication_job_budget_exceeded')
    directories = {path.name for path in job_root.iterdir() if path.is_dir() and path.name != 'authoritative'} if job_root.exists() else set()
    _require(directories == set(registrations), 'publication_job_registration_incomplete')
    hosts = _rows(audit / 'mcp-events.jsonl')
    submitted = {}
    for host in hosts:
        _require(host.get('event_blake3') == blake3(canonical({key: value for key, value in host.items()
                                                            if key != 'event_blake3'})).hexdigest(),
                 'publication_host_event_digest_invalid')
        if host.get('name') in SUBMIT_TOOLS and host.get('is_error') is False:
            job_id = _uuid(host['result'].get('job_id'))
            _require(job_id not in submitted, 'publication_job_submitted_twice')
            submitted[job_id] = host
    _require(set(registrations) == set(submitted), 'publication_registered_host_job_set_differs')
    if expected_job_ids is not None:
        expected = list(expected_job_ids)
        _require(len(expected) == len(set(expected)) and set(expected) == set(registrations),
                 'publication_expected_job_set_differs')
    authoritative = {path.name for path in (job_root / 'authoritative').iterdir()} if (job_root / 'authoritative').exists() else set()
    public_root = workspace / 'outputs/agents_outputs/prescribed-model-jobs'
    public = {path.name for path in public_root.iterdir()} if public_root.exists() else set()
    _require(authoritative <= set(registrations) and public <= set(registrations), 'publication_orphan_job_artifacts')
    sources = source_paths or {name: ROOT / 'EVA-Agent/training/benchmark_models' / name for name in SOURCE_NAMES}
    _require(set(sources) == set(SOURCE_NAMES), 'publication_source_inventory_incomplete')
    source_digests = {name: file_digest(_file(Path(path))) for name, path in sources.items()}
    _require(source_digests in REVIEWED_SOURCE_PROFILES, 'publication_source_profile_unreviewed')
    require_intents = source_digests != REVIEWED_SOURCE_PROFILES[0]
    evidence_files = {}
    task = _json(workspace / 'task.json') if registrations else None
    def retain(path):
        path = _file(path)
        evidence_files[path.relative_to(audit).as_posix()] = file_digest(path)
        return _json(path)
    jobs = []
    for job_id, submission_path in sorted(registrations.items()):
        _uuid(job_id)
        submission, process = read_document(_file(submission_path)), read_document(_file(submission_path.with_name('process.json')))
        retain(submission_path); retain(submission_path.with_name('process.json'))
        host = submitted[job_id]
        _require(submission['job_id'] == process['job_id'] == job_id and submission['workspace'] == str(workspace)
                 and submission['case_ids'] == host['result']['case_ids'] == host['arguments']['case_ids']
                 and submission['track'] == task['track'] and _dead(process),
                 'publication_submission_or_live_process_binding_invalid')
        command = process['command']
        target = job_root / 'authoritative' / job_id
        output = public_root / job_id
        _require(isinstance(command, list) and all(isinstance(value, str) for value in command)
                 and _binding(command, '--job-id') == job_id and _binding(command, '--workspace') == str(workspace)
                 and _binding(command, '--audit-root') == str(target.parent)
                 and _binding(command, '--track') == submission['track'], 'publication_command_binding_invalid')
        worker_file, exit_file, final_file = target / 'receipt.json', target / 'process-exit.json', target / 'supervision-final.json'
        cancel_file = target / 'policy-cancellation.json'
        supervisor_log = target / 'supervisor-deadline-publication-events.jsonl'
        if cancel_file.exists():
            cancel = retain(cancel_file)
            _require(cancel.get('schema') == 'eva.prescribed-model-policy-cancellation.v1'
                     and cancel.get('job_id') == job_id and cancel.get('worker_dispatched') is False
                     and cancel.get('policy_cancellation') is True and cancel.get('os_process_exit_observed') is False
                     and cancel.get('cancellation_reason') == 'track_wall_clock_budget_exhausted'
                     and not worker_file.exists() and not exit_file.exists() and not output.exists(),
                     'publication_pre_dispatch_cancellation_invalid')
            journal = verify_publication_journal(supervisor_log, deadline, output=output, component='supervisor',
                evidence=cancel['deadline_publication_evidence'], require_intents=require_intents)
            _require(journal['policy_cancellations'] and not journal['publications'], 'publication_cancelled_job_wrote_public_bytes')
            if supervisor_log.exists():
                evidence_files[supervisor_log.relative_to(audit).as_posix()] = file_digest(supervisor_log)
            jobs.append({'job_id': job_id, 'disposition': 'cancelled_before_worker_dispatch', 'supervisor': journal})
            continue
        _require(exit_file.is_file() and final_file.is_file(),
                 'publication_job_final_evidence_unknown')
        worker = retain(worker_file) if worker_file.exists() else None
        exited, final = retain(exit_file), retain(final_file)
        worker_killed = (require_intents and exited.get('returncode') == -9
                         and exited.get('policy_cancellation') is True
                         and bool(exited.get('deadline_termination')))
        _require(exited.get('schema') == 'eva.prescribed-model-process-exit.v1'
                 and final.get('schema') == 'eva.prescribed-model-supervision-final.v1'
                 and exited.get('job_id') == final.get('job_id') == job_id and final.get('host_only') is True
                 and exited.get('os_process_exit_observed') is True
                 and type(exited.get('returncode')) is int and final.get('returncode') == exited['returncode']
                 and exited.get('worker_receipt_blake3') == (file_digest(worker_file) if worker else None)
                 and exited.get('worker_source_blake3_at_launch') == source_digests['run_prescribed_model.py']
                 and exited.get('supervisor_source_blake3') == source_digests['job_supervisor.py']
                 and exited.get('track_deadline_monotonic') == deadline
                 and all(row.get('publication_boundary_violation', False) is False for row in (worker or {}, final)),
                 'publication_job_final_binding_invalid')
        if worker is None:
            _require(worker_killed and not output.exists(), 'publication_missing_worker_receipt_unknown')
            worker = {}
        else:
            _require(worker.get('schema') == 'eva.prescribed-public-model-job.v1'
                     and worker.get('job_id') == job_id and worker.get('track') == submission['track']
                     and (worker.get('status') in {'complete', 'cancelled', 'failed'} or worker_killed),
                     'publication_worker_receipt_identity_or_finality_invalid')
        _require(_time(exited.get('dispatch_monotonic')) and exited['dispatch_monotonic'] < deadline,
                 'model_worker_dispatched_after_deadline')
        session_proof = verify_session_exit(exited, final, deadline)
        if require_intents:
            _require(exited.get('post_deadline_computation_authorized') is False
                     and final.get('post_deadline_computation_authorized') is False
                     and exited.get('observation_only_grace_seconds') == final.get('observation_only_grace_seconds') == 3
                     and final.get('track_deadline_monotonic') == deadline,
                     'publication_supervisor_process_identity_invalid')
            observed_exit = exited.get('exit_observed_monotonic')
            termination = exited.get('deadline_termination')
            _require(isinstance(termination, list) and termination == final.get('deadline_termination'),
                     'publication_supervisor_termination_changed')
            if not termination and observed_exit >= deadline:
                _require(worker.get('policy_cancellation') is True and observed_exit <= deadline + 3,
                         'publication_late_worker_exit_without_cancellation')
        if worker.get('helper_source_blake3') is not None:
            _require(worker['helper_source_blake3'] == source_digests['run_prescribed_model.py']
                     and worker.get('track_deadline_monotonic') == deadline,
                     'publication_executed_worker_source_changed')
            helper = target / 'executed-helper.py'
            _require(file_digest(_file(helper)) == source_digests['run_prescribed_model.py'], 'publication_worker_archive_changed')
            evidence_files[helper.relative_to(audit).as_posix()] = file_digest(helper)
            modules = worker.get('executed_module_sources', [])
            _require(len({row['path'] for row in modules}) == len(modules), 'publication_module_inventory_duplicate')
            module_map = {row['path']: row['blake3'] for row in modules}
            _require(module_map.get('executed-modules/_publication.py') == source_digests['_publication.py'],
                     'publication_executed_guard_source_changed')
            for relative, digest in module_map.items():
                module = target / relative
                _require(module.is_relative_to(target) and '..' not in Path(relative).parts
                         and not Path(relative).is_absolute() and file_digest(_file(module)) == digest,
                         'publication_executed_module_changed')
                evidence_files[module.relative_to(audit).as_posix()] = digest
        worker_log = target / 'deadline-publication-events.jsonl'
        if worker_killed and worker:
            _journal_prefix(worker_log, worker['deadline_publication_evidence'], deadline)
        worker_journal = verify_publication_journal(worker_log, deadline, output=output, component='worker',
            evidence=None if worker_killed else worker['deadline_publication_evidence'], require_intents=require_intents)
        supervisor_journal = verify_publication_journal(supervisor_log, deadline, output=output, component='supervisor',
            evidence=final['deadline_publication_evidence'], require_intents=require_intents)
        for log in (worker_log, supervisor_log):
            if log.exists(): evidence_files[log.relative_to(audit).as_posix()] = file_digest(log)
        if worker.get('status') == 'cancelled':
            _require(worker.get('success') is False and worker.get('policy_cancellation') is True
                     and worker.get('cancellation_reason') == 'track_wall_clock_budget_exhausted'
                     and worker_journal['policy_cancellations'], 'worker_cancellation_classification_unproved')
        _require(final.get('policy_cancellation') == exited.get('policy_cancellation'),
                 'supervisor_cancellation_classification_changed')
        if exited.get('policy_cancellation') is True:
            _require(worker_journal['policy_cancellations'] or supervisor_journal['policy_cancellations'],
                     'supervisor_policy_cancellation_unproved')
        events = worker_journal['publications'] + supervisor_journal['publications']
        destinations = {Path(row['destination']) for row in events}
        files = list(output.rglob('*')) if output.exists() else []
        case_roots = {Path(row['destination']) for row in events if row['operation'] == 'publish_case'}
        for path in files:
            _require(path.resolve() == path and not path.is_symlink(), 'public_job_output_symlink')
            if path.is_file():
                _require(path in destinations or any(path.is_relative_to(root) for root in case_roots),
                         'public_model_output_without_publication_evidence')
        jobs.append({'job_id': job_id,
                     'disposition': 'supervisor_cancelled_worker' if worker_killed else worker['status'],
                     'retained_worker_receipt_status': worker.get('status'),
                     'worker_receipt_is_final': not worker_killed, 'returncode': exited['returncode'],
                     'worker': worker_journal, 'supervisor': supervisor_journal,
                     'session_exit': session_proof,
                     'registered_launcher_currently_dead': True, 'actual_worker_exit_observed': True})
    if registrations:
        _require(isinstance(cleanup, dict) and cleanup.get('workspace_quiescent') is True
                 and cleanup.get('actor_policy_time_extended') is False, 'publication_owned_tree_cleanup_required')
    return {'schema': 'eva.benchmark-prescribed-publication-admission.v1', 'job_ids': sorted(registrations),
            'jobs': jobs, 'source_blake3': source_digests, 'evidence_file_blake3': evidence_files,
            'mcp_events_file_blake3': file_digest(audit / 'mcp-events.jsonl'),
            'all_registered_publications_before_deadline': True,
            'owned_descendant_tree_cleanup_requires_separate_verification': True,
            'additional_inference_budget_granted': False, 'model_observed_results_claimed': False}
