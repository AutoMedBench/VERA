"""Read-only verification of an existing owned policy-budget terminal proof.

This never interrupts, waits, snapshots, judges, changes a receipt or fabricates
a terminal outcome. Missing RPC/host/job evidence remains inadmissible.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from eva_agent.codex_runtime import codex_turn_receipt_from_document
from .adapter import EvaluationError, file_digest, read_document
from .policy_capture import require_joined_host_results


def require(value, code):
    if not value: raise EvaluationError(code)


def _owned_process_live(process):
    try:
        tail=Path(f'/proc/{process["pid"]}/stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:
        return False
    return tail[0]!='Z' and tail[19]==str(process['start_ticks'])


def verify_track_deadline_binding(audit, observation):
    """Explicit new profile only; historical per-turn budgets are untouched."""
    from .track_budget import POLICY
    audit = Path(audit)
    config = read_document(audit / 'track-budget.json')
    attempt = read_document(audit.parent / 'attempt.json')
    require(attempt.get('track_budget_policy') == POLICY and attempt.get('track_timeout_seconds') == 3600
        and attempt.get('phase_timeout_policy') == 'remaining_track_budget'
        and config.get('schema') == 'eva.automedbench-track-budget.v1'
        and config.get('policy') == observation.get('policy') == POLICY
        and config.get('timeout_seconds') == observation.get('timeout_seconds') == 3600
        and config.get('document_blake3') == observation.get('budget_document_blake3')
        and config.get('queue_before_admission_charged') is False
        and config.get('job_waits_and_runtime_restart_charged') is True,
        'track_deadline_configuration_binding_invalid')
    start, end = config.get('started_monotonic'), config.get('deadline_monotonic')
    elapsed = observation.get('elapsed_seconds')
    require(all(type(v) in (int, float) and math.isfinite(v) for v in (start, end, elapsed))
        and start >= 0 and end - start == 3600 and elapsed >= 3600
        and observation.get('remaining_seconds') == 0 and observation.get('deadline_exhausted') is True
        and math.isclose(observation.get('cleanup_overrun_seconds', -1), elapsed - 3600, abs_tol=1e-6),
        'track_deadline_exhaustion_unproved')
    return {'policy': POLICY, 'budget_document_blake3': config['document_blake3'],
        'timeout_seconds': 3600, 'elapsed_seconds': elapsed,
        'cleanup_overrun_seconds': observation['cleanup_overrun_seconds']}


def _verify_model_jobs(turn_root, audit, after, *, snapshot_root=None):
    rows={row['path']:row for row in after.get('files',[])}
    jobs=[]
    job_root=audit/'model-jobs'
    for source in sorted(job_root.glob('*/submission.json')):
        job_id=source.parent.name
        submission=read_document(source)
        process=read_document(source.parent/'process.json')
        require(submission.get('job_id')==job_id==process.get('job_id')
            and not _owned_process_live(process),'policy_terminal_model_publisher_not_drained')
        exit_path=job_root/'authoritative'/job_id/'process-exit.json'
        require(exit_path.is_file() and not exit_path.is_symlink(),'policy_terminal_model_exit_absent')
        exit_doc=json.loads(exit_path.read_bytes())
        require(exit_doc.get('schema')=='eva.prescribed-model-process-exit.v1'
            and exit_doc.get('job_id')==job_id and exit_doc.get('os_process_exit_observed') is True
            and type(exit_doc.get('returncode')) is int,'policy_terminal_model_exit_invalid')
        relative=f'outputs/agents_outputs/prescribed-model-jobs/{job_id}/process-exit.json'
        archived=(snapshot_root if snapshot_root is not None else turn_root/'after')/'files'/relative
        row=rows.get(relative,{})
        require(archived.is_file() and not archived.is_symlink()
            and archived.read_bytes()==exit_path.read_bytes()
            and row.get('blake3')==file_digest(exit_path)
            and row.get('bytes')==exit_path.stat().st_size,'policy_terminal_model_exit_not_in_final_snapshot')
        jobs.append({'job_id':job_id,'returncode':exit_doc['returncode'],'exit_file_blake3':file_digest(exit_path)})
    return jobs


def verify_deadline_cleanup(audit, terminal, jobs):
    """Only an owned, acknowledged deadline cancellation excuses a nonzero job exit."""
    cleanup=read_document(audit/'track-deadline-cleanup.json')
    require(cleanup['document_blake3']==terminal.get('deadline_cleanup_document_blake3')
        and cleanup.get('schema')=='eva.automedbench-track-deadline-cleanup.v1'
        and cleanup.get('scope')=='exact_owned_model_supervisors_only'
        and cleanup.get('trigger')=='track_deadline'
        and cleanup.get('workspace_quiescent') is True and cleanup.get('error_category') is None
        and cleanup.get('policy_time_extended') is False,'track_deadline_cleanup_binding_invalid')
    cancelled=[];seen=set();job_map={row['job_id']:row for row in jobs}
    for signal in cleanup.get('signals',[]):
        job=signal.get('job_id');root=audit/'model-jobs'
        require(job in job_map and job not in seen and signal.get('signal')=='SIGTERM',
            'track_deadline_cancelled_job_binding_invalid')
        seen.add(job)
        process=read_document(root/job/'process.json')
        require(process.get('pid')==signal.get('pid')
            and str(process.get('start_ticks'))==signal.get('start_ticks'),
            'track_deadline_cancelled_job_binding_invalid')
        outcome=json.loads((root/'authoritative'/job/'process-exit.json').read_bytes())
        if job_map[job]['returncode']!=0:
            require(outcome.get('forwarded_signals')==[15]
                and outcome.get('returncode') in {-15,143},'track_deadline_cancelled_job_exit_unproved')
            cancelled.append(job)
    return cancelled


def verify_policy_terminal(turn_root, audit, raw_receipt):
    """Return checked binding/count metadata for this exact retained turn only."""
    turn_root,audit=Path(turn_root),Path(audit)
    require(turn_root.parent==audit/'turns' and not turn_root.is_symlink(), 'policy_terminal_turn_scope')
    receipt=codex_turn_receipt_from_document(raw_receipt)
    terminal=read_document(turn_root/'policy-budget-terminal.json')
    budget=terminal.get('policy_budget',{})
    require(terminal.get('schema')=='eva.automedbench-policy-budget-terminal.v1'
        and terminal.get('phase_intent')==turn_root.name
        and terminal.get('receipt_blake3')==receipt.receipt_blake3
        and terminal.get('actual_terminal_status')==receipt.status
        and receipt.status in {'interrupted','completed'}
        and terminal.get('workspace_quiescence_verified') is True
        and terminal.get('infrastructure_error') is None and terminal.get('reward') is None
        and terminal.get('later_stages_evaluated') is False,
        'policy_terminal_binding_or_quiescence_invalid')
    require(budget.get('schema')=='eva.codex-policy-turn-budget.v1'
        and budget.get('budget_exhausted') is True and budget.get('turn_id')==receipt.turn_id
        and budget.get('receipt_blake3')==receipt.receipt_blake3
        and budget.get('actual_terminal_status')==receipt.status
        and budget.get('infrastructure_error') is None and budget.get('reward') is None,
        'policy_terminal_owned_budget_binding_invalid')
    requested,acknowledged,observed,quiescent=(budget.get('interrupt_requested_ns'),
        budget.get('interrupt_acknowledged_ns'),budget.get('terminal_observed_ns'),terminal.get('host_quiescence_observed_ns'))
    require(all(type(value) is int and value>0 for value in (requested,acknowledged,observed,quiescent))
        and requested<=min(acknowledged,observed) and max(acknowledged,observed)<=quiescent,
        'policy_terminal_interrupt_drain_order_invalid')
    require(all(type(budget.get(key)) in (int,float) and not isinstance(budget[key],bool)
        and budget[key]>0 for key in ('timeout_seconds','drain_seconds')),
        'policy_terminal_configured_budget_missing')
    require(not any(event.method=='error' for event in receipt.events)
        and receipt.events[-1].method=='turn/completed','policy_terminal_independent_provider_error')
    joined=require_joined_host_results(receipt,audit)
    require(all(terminal.get(key)==value for key,value in joined.items()), 'policy_terminal_host_join_changed')
    after=read_document(turn_root/'after/manifest.json',maximum=16*1024**2)
    require(after['document_blake3']==terminal.get('after_snapshot_document_blake3'),
        'policy_terminal_final_snapshot_binding_invalid')
    jobs=_verify_model_jobs(turn_root,audit,after)
    deadline = None
    cancelled = []
    if 'track_budget' in terminal:
        deadline = verify_track_deadline_binding(audit, terminal['track_budget'])
        turn_budget = read_document(turn_root/'track-budget.json')
        timeout = turn_budget.get('timeout_seconds')
        require(turn_budget.get('schema') == 'eva.automedbench-track-turn-budget.v1'
            and turn_budget.get('budget_document_blake3') == deadline['budget_document_blake3']
            and type(timeout) in (int,float) and 0 < timeout <= 3600
            and timeout == budget['timeout_seconds']
            and math.isclose(turn_budget.get('elapsed_before_turn_seconds', -1) + timeout, 3600, abs_tol=1e-6),
            'track_deadline_turn_budget_differs')
        cancelled=verify_deadline_cleanup(audit,terminal,jobs)
    return {'schema':'eva.verified-policy-budget-terminal.v1','valid':True,
        'terminal_document_blake3':terminal['document_blake3'],'receipt_blake3':receipt.receipt_blake3,
        'actual_terminal_status':receipt.status,'after_snapshot_document_blake3':after['document_blake3'],
        'timeout_seconds':budget['timeout_seconds'],'drain_seconds':budget['drain_seconds'],
        'workspace_quiescence_verified':True,**joined,'model_job_exits':jobs,
        **({'track_budget':deadline,'policy_cancelled_job_ids':cancelled} if deadline is not None else {}),
        'original_receipt_mutated':False,'provider_calls':0,'reward':None}


def verify_between_turn_terminal(turn_root, audit, raw_receipt):
    """A real completed prefix followed by deadline expiry, not a fabricated interrupt."""
    turn_root,audit=Path(turn_root),Path(audit)
    require(turn_root.parent==audit/'turns' and not turn_root.is_symlink(), 'policy_terminal_turn_scope')
    receipt=codex_turn_receipt_from_document(raw_receipt)
    terminal=read_document(audit/'track-budget-terminal.json')
    deadline=verify_track_deadline_binding(audit,terminal.get('track_budget',{}))
    require(terminal.get('schema')=='eva.automedbench-between-turn-budget-terminal.v1'
        and terminal.get('phase_intent')==turn_root.name and terminal.get('receipt_blake3')==receipt.receipt_blake3
        and terminal.get('actual_terminal_status')==receipt.status=='completed'
        and terminal.get('workspace_quiescence_verified') is True and terminal.get('infrastructure_error') is None
        and terminal.get('reward') is None and terminal.get('later_stages_evaluated') is False
        and not any(event.method=='error' for event in receipt.events), 'track_between_turn_terminal_invalid')
    joined=require_joined_host_results(receipt,audit)
    require(all(terminal.get(key)==value for key,value in joined.items()),'policy_terminal_host_join_changed')
    after=read_document(turn_root/'after/manifest.json',maximum=16*1024**2)
    require(after['document_blake3']==terminal.get('after_snapshot_document_blake3'),
        'policy_terminal_final_snapshot_binding_invalid')
    snapshot_options={};cleanup_binding={}
    if 'cleanup_after_snapshot_relative' in terminal:
        root=audit/'track-deadline-after'
        require(terminal['cleanup_after_snapshot_relative']=='track-deadline-after'
            and terminal.get('original_phase_after_preserved') is True
            and terminal.get('late_outputs_excluded_from_prior_stage_A') is True
            and root.is_dir() and not root.is_symlink(),'track_cleanup_snapshot_scope_invalid')
        cleanup_after=read_document(root/'manifest.json',maximum=16*1024**2)
        require(cleanup_after['document_blake3']==terminal.get('cleanup_after_snapshot_document_blake3')
            and cleanup_after.get('immutable_inputs_manifest_blake3')==after.get('immutable_inputs_manifest_blake3'),
            'track_cleanup_snapshot_binding_invalid')
        snapshot_options={'snapshot_root':root}
        cleanup_binding={'model_job_snapshot_root':str(root),
            'cleanup_after_snapshot_document_blake3':cleanup_after['document_blake3'],
            'original_phase_after_preserved':True,'late_outputs_excluded_from_prior_stage_A':True}
    else:
        cleanup_after=after
    jobs=_verify_model_jobs(turn_root,audit,cleanup_after,**snapshot_options)
    cancelled=verify_deadline_cleanup(audit,terminal,jobs)
    return {'schema':'eva.verified-between-turn-track-terminal.v1','valid':True,
        'terminal_document_blake3':terminal['document_blake3'],'receipt_blake3':receipt.receipt_blake3,
        'actual_terminal_status':receipt.status,'after_snapshot_document_blake3':after['document_blake3'],
        'timeout_seconds':3600,'track_budget':deadline,'workspace_quiescence_verified':True,
        **joined,'model_job_exits':jobs,'policy_cancelled_job_ids':cancelled,
        **cleanup_binding,
        'original_receipt_mutated':False,'provider_calls':0,'reward':None}
