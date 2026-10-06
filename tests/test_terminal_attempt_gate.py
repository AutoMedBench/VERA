"""All-seven synthetic host proofs; scores and model events are fixtures only."""
import asyncio
from dataclasses import replace
import json
import time

import pytest

from eva_agent.codex_runtime import CodexRuntime,CodexTurnInput
from eva_agent.codex_runtime.runtime import _logical_input
from eva_agent.codex_runtime.policy_budget import CodexPolicyBudgetExceeded
from eva_agent.pipeline.digests import canonical_json_bytes,canonical_value
from training.automedbench_lite.adapter import write_once
from training.automedbench_lite.policy_capture import require_joined_host_results
from training.automedbench_lite.track_adapter import BY_TRACK
from training.eva_rsi.production import require_complete_baseline
from training.eva_rsi.terminal_attempts import verify_terminal_attempts,require_judged_attempts
from training.eva_rsi.evidence import commitment
from test_codex_policy_budget import MockAppServer,backend,options


def replace_doc(path,value):
    path.unlink();return write_once(path,{k:v for k,v in value.items() if k!='document_blake3'})


@pytest.fixture
def seven(tmp_path):
    run=tmp_path/'run';root=run/'track-rollouts';root.mkdir(parents=True)
    identity=tmp_path/'identity.json';identity.write_text(json.dumps({'exact_final_model_path':'/fixture/trained-hf'}))
    digest=commitment(identity)['blake3']
    attempt=write_once(root/'attempt.json',{'schema':'eva.automedbench-track-attempt.v1',
        'tracks':list(BY_TRACK),'planned_coding_rollouts':7,'one_attempt_per_track':True,
        'server_binding':{'identity':json.loads(identity.read_text()),'identity_file_blake3':digest,
            'canary':{'checkpoint_identity_blake3':digest}}})
    write_once(run/'baseline-launch.json',{'codex_version':'codex-cli 0.153.4','codex_binary_blake3':'b'*64})
    rows=[]
    for track in BY_TRACK:
        audit=root/track;turn=audit/'turns/01-planning';turn.mkdir(parents=True)
        server=MockAppServer('interrupt',audit)
        server.metadata.serverInfo.version='0.153.4 (synthetic CPU fixture)'
        opt=replace(options(tmp_path),model='Qwen/Qwen3.5-9B')
        value=CodexTurnInput(public_text='Synthetic fixture for '+track)
        async def capture():
            async with CodexRuntime(backend(server)) as runtime:
                handle=await runtime.start_thread(opt)
                return await runtime.run_turn(handle,value,policy_timeout_seconds=.02,interruption_grace_seconds=.08)
        with pytest.raises(CodexPolicyBudgetExceeded) as caught:asyncio.run(capture())
        receipt=caught.value.receipt
        (turn/'receipt.json').write_bytes(canonical_json_bytes(receipt))
        write_once(turn/'request.json',{'logical_input':canonical_value(_logical_input(opt,value))})
        for label in ('before','after'):
            directory=turn/label;directory.mkdir()
            doc=write_once(directory/'manifest.json',{'schema':'synthetic-snapshot-fixture','files':[]})
        # Deliberately simulated configured900s sidecar: test does not wait900s
        # or claim a real provider run. Actual SDK owned interrupt is exercised.
        budget={**caught.value.outcome,'timeout_seconds':900}
        write_once(turn/'policy-budget-terminal.json',{'schema':'eva.automedbench-policy-budget-terminal.v1',
            'phase_intent':turn.name,'policy_budget':budget,'receipt_blake3':receipt.receipt_blake3,
            'actual_terminal_status':receipt.status,'workspace_quiescence_verified':True,
            'host_quiescence_observed_ns':time.time_ns(),'after_snapshot_document_blake3':doc['document_blake3'],
            'infrastructure_error':None,'reward':None,'later_stages_evaluated':False,
            **require_joined_host_results(receipt,audit)})
        row=write_once(audit/'rollout.json',{'track':track,'source_checkpoint':'/fixture/trained-hf',
            'actual_turn_count':1,'turn_receipt_blake3s':[receipt.receipt_blake3],'thread_id':receipt.thread_id,
            'requested_model':'Qwen/Qwen3.5-9B','completed_requested_turns':False,
            'errors':[{'phase':'01-planning','error':'policy_turn_budget_exhausted','category':'policy_budget',
                'infrastructure_error':None,'reward':None}]})
        rows.append(row)
    write_once(root/'summary.json',{'tracks':rows})
    return {'benchmark_run_root':str(run),'checkpoint_identity':str(identity),'evaluation_mode':'full_single_pass',
        'baseline_gate_mode':'terminal-attempts-v1'},attempt


def test_seven_proof_backed_budget_stops_are_attempts_not_workflow_success(seven):
    document,_=seven
    proof=require_complete_baseline(document)
    assert proof['valid'] and len(proof['tracks'])==7
    assert all(row['attempted_stages']==['S1'] and row['unreachable_stages']==['S2','S3','S4','S5']
        and row['unreachable_scores'] is None for row in proof['tracks'].values())
    with pytest.raises(ValueError,match='complete_seven_track'):
        require_complete_baseline({k:v for k,v in document.items() if k!='baseline_gate_mode'})


@pytest.mark.parametrize('change',['local429','zero_attempt','missing_track','wrong_checkpoint','missing_drain','unexplained_stop'])
def test_real_failure_categories_not_rewritten_policy_budget(seven,change):
    from pathlib import Path
    document,_=seven;root=Path(document['benchmark_run_root'])/'track-rollouts'
    summary=json.loads((root/'summary.json').read_text());row=summary['tracks'][0];track=row['track']
    if change=='local429':row['errors']=[{'phase':'01-planning','error':'generation_transport_error','http_status':429}]
    elif change=='zero_attempt':row.update(actual_turn_count=0,turn_receipt_blake3s=[])
    elif change=='missing_track':summary['tracks'].pop()
    elif change=='wrong_checkpoint':row['source_checkpoint']='/other-model'
    elif change=='missing_drain':(root/track/'turns/01-planning/policy-budget-terminal.json').unlink()
    else:row['errors']=[]
    if change!='missing_track':summary['tracks'][0]=replace_doc(root/track/'rollout.json',row)
    replace_doc(root/'summary.json',summary)
    with pytest.raises((ValueError,FileNotFoundError)):verify_terminal_attempts(document)


def test_every_attempted_stage_needs_real_reopened_matching_judge(seven,monkeypatch,tmp_path):
    document,attempt=seven;proof=verify_terminal_attempts(document)
    roots=[]
    for track in BY_TRACK:
        root=tmp_path/'feedback'/track/'S1';root.mkdir(parents=True);roots.append(str(root))
        (root/'preflight.json').write_text('{}')
        (root/'rollout.json').write_text(json.dumps({'provider_metadata':{'source_turns':proof['tracks'][track]['sources']}}))
    def verified(root,**kwargs):
        return {'valid':True,'status':'scored','case_id':root.parent.name,'stage':root.name,
            'score':{'fixture_only_zero':True},'round_identity':{'checkpoint_id':proof['checkpoint_identity_blake3']},
            'skill_source_binding':{'run_attempt_document_blake3':attempt['document_blake3']}}
    monkeypatch.setattr('training.benchmark_feedback.automed_codex.verify_feedback',verified)
    monkeypatch.setattr('training.benchmark_feedback.automed_codex.read_feedback_rollout',lambda path,report:json.loads(path.read_text()))
    require_judged_attempts({**document,'feedback_roots':roots},proof)  # No positive-score floor.
    with pytest.raises(ValueError,match='real_judge_missing'):
        require_judged_attempts({**document,'feedback_roots':roots[:-1]},proof)
    from pathlib import Path
    path=Path(roots[0])/'rollout.json'
    value=json.loads(path.read_text());value['provider_metadata']['source_turns'][0]['codex_receipt_blake3']='0'*64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='judge_prefix_differs'):
        require_judged_attempts({**document,'feedback_roots':roots},proof)


def test_context_terminal_is_separate_explicit_verifier_boundary(seven,monkeypatch):
    from pathlib import Path
    document,_=seven;root=Path(document['benchmark_run_root'])/'track-rollouts'
    summary=json.loads((root/'summary.json').read_text());row=summary['tracks'][0]
    row['errors']=[{'phase':'01-planning','error':'turn_not_completed'}]
    summary['tracks'][0]=replace_doc(root/row['track']/'rollout.json',row)
    replace_doc(root/'summary.json',summary)
    calls=[]
    def boundary(turn,audit,raw,*,source_binding=None):
        # Verifier boundary fixture only; actual failed SDK/signature evidence is
        # independently exercised by test_context_terminal_verifier.
        calls.append(turn)
        return {'schema':'synthetic-context-proof','valid':True,'reward':None}
    monkeypatch.setattr('training.automedbench_lite.context_terminal.verify_context_terminal',boundary)
    with pytest.raises(ValueError,match='infrastructure_error'):verify_terminal_attempts(document)
    assert calls==[]
    proof=verify_terminal_attempts({**document,'allow_context_budget_terminal':True})
    assert len(calls)==1 and proof['tracks'][row['track']]['policy_terminal']['reward'] is None
