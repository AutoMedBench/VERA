"""Opt-in actual-attempt gate; low model performance is not an infra failure."""
from __future__ import annotations
import json
from pathlib import Path
import re

from eva_agent.codex_runtime import codex_turn_receipt_from_document
from eva_agent.pipeline.digests import blake3_hex
from training.automedbench_lite.adapter import read_document, file_digest
from training.automedbench_lite.track_adapter import BY_TRACK
from training.automedbench_lite.track_feedback import (PHASES, verify_feedback_recovery_reference,
    recovery_source_fields)
from training.automedbench_lite.policy_terminal import verify_policy_terminal, _owned_process_live
from .evidence import commitment, read, require

MODE='terminal-attempts-v1'


def require_successful_model_jobs(audit, final_turn, *, policy_cancelled_job_ids=(), model_job_snapshot_root=None):
    """A completed agent turn alone does not establish detached helper health."""
    root=audit/'model-jobs'
    snapshot_root=Path(model_job_snapshot_root) if model_job_snapshot_root is not None else final_turn/'after'
    if model_job_snapshot_root is not None:
        require(snapshot_root==audit/'track-deadline-after' and not snapshot_root.is_symlink(),
            'terminal_attempt_cleanup_snapshot_scope_invalid')
    after=read_document(snapshot_root/'manifest.json',maximum=16*1024**2)
    rows={row['path']:row for row in after.get('files',[])}
    for path in sorted(root.glob('*/submission.json')):
        job=path.parent.name
        submission=read_document(path);process=read_document(path.parent/'process.json')
        require(submission.get('job_id')==job==process.get('job_id') and not _owned_process_live(process),
            'terminal_attempt_model_publisher_not_drained')
        source=root/'authoritative'/job/'process-exit.json';value=read(source)
        require(value.get('schema')=='eva.prescribed-model-process-exit.v1' and value.get('job_id')==job
            and value.get('os_process_exit_observed') is True and type(value.get('returncode')) is int
            and (value['returncode']==0 or job in policy_cancelled_job_ids
                and value['returncode'] in {-15,143} and value.get('forwarded_signals')==[15]),
            'terminal_attempt_task_runtime_failed')
        relative=f'outputs/agents_outputs/prescribed-model-jobs/{job}/process-exit.json'
        archived=snapshot_root/'files'/relative
        require(not source.is_symlink() and not archived.is_symlink() and archived.read_bytes()==source.read_bytes()
            and rows.get(relative,{}).get('blake3')==file_digest(source),
            'terminal_attempt_model_exit_snapshot_differs')


def verify_terminal_track_source(document, *, selected_tracks, expected_attempt_tracks):
    """Verify real terminal evidence for an explicit subset of one actor run.

    This is an adapter-level composition seam.  It never relabels the actor's
    own ``attempt.json``: ``expected_attempt_tracks`` must still equal that
    document, while ``selected_tracks`` says which disjoint rows a caller is
    consuming.  The historical seven-track gate remains the wrapper below.
    """
    selected_tracks=tuple(selected_tracks);expected_attempt_tracks=tuple(expected_attempt_tracks)
    require(bool(selected_tracks) and len(selected_tracks)==len(set(selected_tracks))
        and set(selected_tracks)<=set(BY_TRACK)
        and bool(expected_attempt_tracks) and len(expected_attempt_tracks)==len(set(expected_attempt_tracks))
        and set(expected_attempt_tracks)<=set(BY_TRACK)
        and set(selected_tracks)<=set(expected_attempt_tracks),
        'terminal_attempt_selected_tracks_invalid')
    allow_context=document.get('allow_context_budget_terminal',False)
    require(type(allow_context) is bool,'terminal_attempt_context_mode_requires_boolean')
    recoveries=document.get('policy_recovery_references',{})
    require(isinstance(recoveries,dict) and set(recoveries)<=set(selected_tracks)
        and all(isinstance(value,dict) and len(value)==1 and set(value)<=set(PHASES)
                for value in recoveries.values()), 'terminal_attempt_recovery_references_invalid')
    run=Path(document['benchmark_run_root']).resolve(strict=True)
    identity_path=Path(document['checkpoint_identity'])
    identity=read(identity_path); identity_digest=commitment(identity_path)['blake3']
    attempt=read_document(run/'track-rollouts/attempt.json')
    summary=read_document(run/'track-rollouts/summary.json')
    require(attempt.get('schema')=='eva.automedbench-track-attempt.v1'
        and set(attempt.get('tracks',[]))==set(expected_attempt_tracks)
        and len(attempt['tracks'])==len(expected_attempt_tracks)
        and attempt.get('planned_coding_rollouts')==len(expected_attempt_tracks)
        and attempt.get('one_attempt_per_track') is True,
        'terminal_attempt_gate_requires_all_seven_attempts')
    bound=attempt.get('server_binding',{})
    require(bound.get('identity')==identity and bound.get('identity_file_blake3')==identity_digest
        and bound.get('canary',{}).get('checkpoint_identity_blake3')==identity_digest,
        'terminal_attempt_checkpoint_binding_differs')
    rows=summary.get('tracks',[])
    require(len(rows)==len(expected_attempt_tracks)
        and {row.get('track') for row in rows}==set(expected_attempt_tracks),
        'terminal_attempt_summary_coverage_differs')
    if 'actor_runtime_binding' in document:
        reference=document['actor_runtime_binding']
        require(commitment(reference['path'])==reference,'terminal_attempt_runtime_binding_changed')
        binary=read(reference['path'])
    else:
        binary=read_document(run/'baseline-launch.json')
    require(binary.get('codex_version')=='codex-cli 0.153.4'
        and isinstance(binary.get('codex_binary_blake3'),str)
        and re.fullmatch('[0-9a-f]{64}',binary['codex_binary_blake3']) is not None,
        'terminal_attempt_codex_build_unbound')
    tracks={}; consumed=set()
    for track in selected_tracks:
        audit=run/'track-rollouts'/track
        row=read_document(audit/'rollout.json')
        require(row==next(item for item in rows if item['track']==track)
            and row.get('source_checkpoint')==identity['exact_final_model_path'],
            'terminal_attempt_track_checkpoint_or_summary_differs')
        archival_count=row.get('actual_turn_count')
        require(type(archival_count) is int and 0<=archival_count<=5
            and len(row.get('turn_receipt_blake3s',[]))==archival_count,
            'terminal_attempt_real_turn_missing')
        recovery=None
        if track in recoveries:
            recovered_stage,reference=next(iter(recoveries[track].items()))
            require(archival_count<5 and recovered_stage==list(PHASES)[archival_count],
                'terminal_attempt_recovery_must_be_one_unregistered_final_turn')
            recovery=verify_feedback_recovery_reference(run,track,recovered_stage,reference)
        count=archival_count+(1 if recovery else 0)
        require(count>=1,'terminal_attempt_real_turn_missing')
        stages=list(PHASES)[:count]
        failures=row.get('errors',[])
        is_budget=failures==[{'phase':PHASES[stages[-1]],
                'error':'policy_turn_budget_exhausted','category':'policy_budget',
                'infrastructure_error':None,'reward':None}]
        is_track_budget=failures==[{'phase':PHASES[stages[-1]],
                'error':'track_budget_exhausted','category':'policy_budget',
                'infrastructure_error':None,'reward':None}]
        is_context=allow_context and failures==[{'phase':PHASES[stages[-1]],'error':'turn_not_completed'}]
        is_recovery=recovery is not None and failures==[{key:value for key,value
            in recovery['original_archival_failure'].items() if key!='document_blake3'}]
        if failures:
            require(is_budget or is_track_budget or is_context or is_recovery,'terminal_attempt_runner_or_task_infrastructure_error')
        else:
            require(count==5 and row.get('completed_requested_turns') is True,
                'terminal_attempt_unexplained_workflow_stop')
        sources=[];terminal=None
        for position,stage in enumerate(stages):
            turn=audit/'turns'/PHASES[stage]
            raw=read(turn/'receipt.json');receipt=codex_turn_receipt_from_document(raw)
            require(receipt.runtime_turn_id not in consumed,'terminal_attempt_receipt_reused_across_tracks')
            consumed.add(receipt.runtime_turn_id)
            request=read_document(turn/'request.json')
            expected_receipt=(recovery['proof']['source_commitments']['receipt_blake3']
                if is_recovery and position==count-1 else row['turn_receipt_blake3s'][position])
            require(receipt.receipt_blake3==expected_receipt
                and receipt.thread_id==row.get('thread_id') and receipt.visibility=='actor-public'
                and receipt.model==row.get('requested_model')=='Qwen/Qwen3.5-9B'
                and re.match(r'^0\.153\.4(?:\s|$)',receipt.server_version) is not None
                and receipt.input_blake3==blake3_hex(request['logical_input'])
                and (is_context and position==count-1 or not any(event.method=='error' for event in receipt.events)),
                'terminal_attempt_provider_or_source_binding_invalid')
            before=read_document(turn/'before/manifest.json',maximum=16*1024**2)
            recovered_final=is_recovery and position==count-1
            after_root=Path(recovery['reference']['recovery_root'])/'after' if recovered_final else turn/'after'
            after=read_document(after_root/'manifest.json',maximum=16*1024**2)
            if recovered_final:
                require(receipt.status=='interrupted'
                    and recovery['proof']['source_commitments']['before_manifest_document_blake3']==before['document_blake3']
                    and recovery['recovery_current_after_manifest_blake3']==after['document_blake3'],
                    'terminal_attempt_recovery_snapshot_or_status_differs')
            elif is_context and position==count-1:
                from training.automedbench_lite.context_terminal import verify_context_terminal
                source_binding=read(document['actor_runtime_binding']['path']) if 'actor_runtime_binding' in document else None
                terminal=verify_context_terminal(turn,audit,raw,source_binding=source_binding)
            elif is_budget and position==count-1:
                terminal=verify_policy_terminal(turn,audit,raw)
                require((terminal['timeout_seconds']==900 or terminal.get('track_budget',{}).get('timeout_seconds')==3600)
                    and all(job['returncode']==0 or job['job_id'] in terminal.get('policy_cancelled_job_ids',())
                        for job in terminal['model_job_exits']),
                    'terminal_attempt_task_runtime_failed_or_budget_differs')
            elif is_track_budget and position==count-1:
                from training.automedbench_lite.policy_terminal import verify_between_turn_terminal
                terminal=verify_between_turn_terminal(turn,audit,raw)
                require(all(job['returncode']==0 or job['job_id'] in terminal.get('policy_cancelled_job_ids',())
                    for job in terminal['model_job_exits']),
                    'terminal_attempt_task_runtime_failed_or_budget_differs')
            else:
                require(receipt.status=='completed' and not (turn/'policy-budget-terminal.json').exists(),
                    'terminal_attempt_unproved_interruption')
            sources.append({'stage_intent':stage,'phase_intent':PHASES[stage],
                'codex_receipt_blake3':receipt.receipt_blake3,'request_document_blake3':request['document_blake3'],
                'before_manifest_blake3':before['document_blake3'],
                **(recovery_source_fields(recovery) if recovered_final else {'after_manifest_blake3':after['document_blake3']})})
        require(not any((audit/'turns'/PHASES[stage]/'receipt.json').exists() for stage in list(PHASES)[count:]),
            'terminal_attempt_unaccounted_later_turn')
        if not is_recovery:
            require_successful_model_jobs(audit,audit/'turns'/PHASES[stages[-1]],
                **({'policy_cancelled_job_ids':terminal['policy_cancelled_job_ids']}
                   if terminal is not None and 'policy_cancelled_job_ids' in terminal else {}),
                **({'model_job_snapshot_root':terminal['model_job_snapshot_root']}
                   if terminal is not None and 'model_job_snapshot_root' in terminal else {}))
        else:
            require(recovery['proof']['async_model_jobs_present'] is False,
                'terminal_attempt_recovery_cannot_skip_model_job_proof')
        tracks[track]={'attempted_stages':stages,'unreachable_stages':list(PHASES)[count:],
            'sources':sources,'policy_terminal':terminal,'unreachable_scores':None,
            **({'policy_recovery':recovery,'archival_summary_turn_count':archival_count,
                'verified_retained_turn_count':count,'recovered_unregistered_turn_count':1,
                'original_archival_errors':failures,'original_summary_unchanged':True}
                if recovery else {})}
    return {'schema':'eva.verified-terminal-track-source.v1','valid':True,
        'run_attempt_document_blake3':attempt['document_blake3'],'checkpoint_identity_blake3':identity_digest,
        'selected_tracks':list(selected_tracks),'source_attempt_tracks':list(expected_attempt_tracks),
        'tracks':tracks,'provider_calls':0,'model_score_threshold':None,'unreachable_is_not_zero':True}


def verify_terminal_attempts(document):
    """All seven same-checkpoint attempts must have real terminal source evidence."""
    require(document.get('baseline_gate_mode')==MODE and document.get('evaluation_mode')=='full_single_pass',
        'terminal_attempt_gate_requires_explicit_full_scope')
    proof=verify_terminal_track_source(document,selected_tracks=tuple(BY_TRACK),
        expected_attempt_tracks=tuple(BY_TRACK))
    # Preserve the established public proof byte-semantics: the subset helper's
    # two explicit scope fields are intentionally not added to this old schema.
    return {'schema':'eva.verified-seven-track-terminal-attempts.v1','valid':proof['valid'],
        'run_attempt_document_blake3':proof['run_attempt_document_blake3'],
        'checkpoint_identity_blake3':proof['checkpoint_identity_blake3'],'tracks':proof['tracks'],
        'provider_calls':proof['provider_calls'],'model_score_threshold':proof['model_score_threshold'],
        'unreachable_is_not_zero':proof['unreachable_is_not_zero']}


def require_judged_attempts(document, proof):
    """Each attempted stage needs a real reopened Judge on this exact prefix."""
    from training.benchmark_feedback.automed_codex import verify_feedback, read_feedback_rollout
    expected={(track,stage) for track,row in proof['tracks'].items() for stage in row['attempted_stages']}
    family=None
    if 'judge_comparison_family' in document:
        from .judge_identity import verify_judge_comparison_family,reopen_family_feedback
        require(document.get('judge_identity_mode')=='declared-annotation-family-v1',
            'terminal_attempt_judge_family_opt_in_required')
        family=verify_judge_comparison_family(document['judge_comparison_family'])
    actual=set()
    for root in document.get('feedback_roots',[]):
        root=Path(root)
        grade=(reopen_family_feedback(root,family,include_skill_source_binding=True) if family
               else verify_feedback(root,include_skill_source_binding=True))
        key=(grade['case_id'],grade['stage'])
        require(grade.get('valid') is True and grade.get('status')=='scored' and key in expected and key not in actual
            and grade['round_identity']['checkpoint_id']==proof['checkpoint_identity_blake3']
            and grade['skill_source_binding']['run_attempt_document_blake3']==proof['run_attempt_document_blake3'],
            'terminal_attempt_real_judge_coverage_or_source_invalid')
        preflight=read(root/'preflight.json')
        rollout=read_feedback_rollout(root/'rollout.json',preflight)
        sources=proof['tracks'][key[0]]['sources']
        prefix=sources[:next(i for i,row in enumerate(sources) if row['stage_intent']==key[1])+1]
        retained=rollout['provider_metadata']['source_turns']
        require(len(retained)==len(prefix) and all(all(row.get(k)==v for k,v in expected_row.items())
            for row,expected_row in zip(retained,prefix,strict=True)), 'terminal_attempt_judge_prefix_differs')
        actual.add(key)
    require(actual==expected,'terminal_attempt_real_judge_missing')
    return proof
