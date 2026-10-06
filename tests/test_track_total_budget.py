"""Prospective scheduling/ownership fixtures only: no model, provider or GPU."""
import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from training.automedbench_lite import track_actor as actor
from training.automedbench_lite import track_budget as budget
from training.automedbench_lite.adapter import read_document, write_once
from training.automedbench_lite.job_wait import await_model_jobs
from training.eva_rsi.production_eval import actor_profile, actor_arguments, evaluation_scope
from test_policy_terminal_verifier import proof
from test_terminal_attempt_gate import seven, replace_doc


def fake_runtime(monkeypatch):
    records = []; processes = []
    class Runtime:
        def __init__(self, process): self.process = process
        async def __aenter__(self): records.append(('open', self.process.pid)); return self
        async def __aexit__(self, *_): self.process.closed = True; records.append(('close', self.process.pid))
        async def start_thread(self, options): return SimpleNamespace(thread_id=str(options.cwd))
        async def resume_thread(self, thread_id, options):
            records.append(('resume', thread_id)); return SimpleNamespace(thread_id=thread_id)
    def backend():
        process = SimpleNamespace(pid=1000+len(processes), closed=False)
        process.poll = lambda: 0 if process.closed else None
        processes.append(process); return process
    monkeypatch.setattr(actor, 'CodexRuntime', Runtime)
    monkeypatch.setattr(actor, 'app_process', lambda backend: backend)
    return backend, records, processes


def test_three_slots_hold_entire_track_restart_and_do_not_charge_initial_queue(tmp_path, monkeypatch):
    clock = [0.0]; original = budget.TrackDeadline
    monkeypatch.setattr(actor, 'TrackDeadline', lambda audit, seconds: original(audit, seconds=seconds, clock=lambda: clock[0]))
    backend, records, processes = fake_runtime(monkeypatch)
    states=[]; seen=[]; active=set(); peak=[0]
    for number in range(4):
        audit=tmp_path/str(number);audit.mkdir()
        states.append({'audit':audit,'options':SimpleNamespace(cwd=str(audit)), 'receipts':[], 'errors':[]})
    async def turn(runtime,state,index):
        seen.append((state['audit'].name,index,state['deadline'].remaining()))
        clock[0]+=1
        state['receipts'].append(SimpleNamespace())
        await asyncio.sleep(0)
    async def scenario():
        slots=asyncio.Semaphore(3)
        initialization_lock=asyncio.Lock()
        async def admitted(state):
            async with slots:
                active.add(state['audit'].name);peak[0]=max(peak[0],len(active))
                state['admitted_clock']=clock[0]
                await actor.run_whole_track(SimpleNamespace(backend=backend),state,turn,seconds=3600,
                    initialization_lock=initialization_lock)
                active.remove(state['audit'].name)
        await asyncio.gather(*(admitted(s) for s in states))
    asyncio.run(scenario())
    assert peak==[3] and all(p.closed for p in processes) and len(processes)==8
    assert states[3]['admitted_clock']>0
    for state in states:
        assert [i for name,i,_ in seen if name==state['audit'].name]==list(range(5))
        assert state['deadline'].started==state['admitted_clock']
        assert state['actual_restart'] and len(state['receipts'])==5 and not state['errors']
        first,second=state['app_server_pids']
        assert records.index(('close',first))<records.index(('open',second))


def test_actual_run_tracks_passes_decreasing_remaining_budget_not_900_each_phase(tmp_path, monkeypatch):
    clock=[10.0]; original=budget.TrackDeadline
    monkeypatch.setattr(actor,'TrackDeadline',lambda audit,seconds:original(audit,seconds=seconds,clock=lambda:clock[0]))
    backend,records,processes=fake_runtime(monkeypatch)
    workspace=tmp_path/'workspace';(workspace/'notes').mkdir(parents=True);(workspace/'outputs/agents_outputs').mkdir(parents=True)
    write_once(tmp_path/'track-run-manifest.json',{'run_id':'fixture','tracks':[{'track':'classification',
        'workspace_relative':'workspace','task_file_blake3':'fixture','input_manifest_file_blake3':'fixture'}]})
    setup=SimpleNamespace(backend=backend,model='Qwen/Qwen3.5-9B',provider='eva_local_qwen',thread_config={},safe_metadata={})
    @contextmanager
    def local_setup(**kwargs):
        assert kwargs['max_output_tokens']==4096 and kwargs['upstream_timeout_seconds']==600
        yield setup
    monkeypatch.setattr(actor,'local_qwen_setup',local_setup)
    monkeypatch.setattr(actor,'resolve_image',lambda _: 'fixture')
    monkeypatch.setattr(actor,'serving_binding',lambda *_:{'canary':{'exact_final_model_path':'/fixture/model'}})
    monkeypatch.setattr(actor,'file_digest',lambda _: 'fixture')
    monkeypatch.setattr(actor,'VerifiedEvaluationSkills',lambda _:SimpleNamespace(catalog_blake3='fixture',inventory=[]))
    monkeypatch.setattr(actor,'MutableInventory',lambda w,a:SimpleNamespace(workspace=w))
    monkeypatch.setattr(actor,'snapshot',lambda *_:None)
    monkeypatch.setattr(actor,'phase_prompt',lambda skills,index,track,**kwargs:(*actor.PHASES[index],{}))
    received=[];waits=[]
    async def wait_jobs(audit,phase,**kwargs):
        waits.append(kwargs); assert kwargs['track_deadline'].remaining()==kwargs['timeout']
    async def capture(runtime,handle,value,**kwargs):
        received.append(kwargs['timeout']);assert kwargs['track_deadline'].remaining()==kwargs['timeout']
        clock[0]+=100
        return SimpleNamespace(status='completed',tool_calls=(),receipt_blake3=str(len(received)),
            thread_id=handle.thread_id,thread_resumed=len(received)>3),None
    monkeypatch.setattr(actor,'capture_turn',capture);monkeypatch.setattr(actor,'await_model_jobs',wait_jobs)
    args=SimpleNamespace(run_root=tmp_path,tracks=['all'],workers=3,image='fixture',server_canary=None,
        server_identity=None,runtime_manifest=tmp_path/'models.json',codex_bin=tmp_path/'codex',
        public_python=tmp_path/'python',turn_timeout=900,track_timeout=3600)
    asyncio.run(actor.run_tracks(args))
    assert received==[3600,3500,3400,3300,3200] and len(waits)==2
    assert all(p.closed for p in processes)
    outcome=read_document(tmp_path/'track-rollouts/classification/track-budget-outcome.json')
    assert outcome['elapsed_seconds']==500 and not outcome['deadline_exhausted']
    assert read_document(tmp_path/'track-rollouts/attempt.json')['legacy_turn_timeout_not_applied']==900


def test_turn_error_closes_owned_runtime_and_never_runs_missing_suffix(tmp_path,monkeypatch):
    backend,_,processes=fake_runtime(monkeypatch)
    state={'audit':tmp_path,'workspace':tmp_path/'workspace','options':SimpleNamespace(cwd=str(tmp_path)),'receipts':[],'errors':[]}
    called=[]
    async def turn(runtime,state,index):
        called.append(index);state['errors'].append({'error':'retained_fixture_transport_failure'})
    asyncio.run(actor.run_whole_track(SimpleNamespace(backend=backend),state,turn,seconds=3600,
        initialization_lock=asyncio.Lock()))
    assert called==[0] and len(processes)==1 and processes[0].closed


def test_pending_model_job_wait_uses_same_clock_without_a_second_7200s_allowance(tmp_path,monkeypatch):
    clock=[0.0];deadline=budget.TrackDeadline(tmp_path,clock=lambda:clock[0]);clock[0]=3599
    job=tmp_path/'model-jobs'/str(uuid4());job.mkdir(parents=True)
    (job/'submission.json').write_text('{}')
    import os
    tick=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    (job/'process.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':tick}))
    sleeps=[]
    async def sleep(seconds):sleeps.append(seconds);clock[0]+=seconds
    monkeypatch.setattr('training.automedbench_lite.job_wait.asyncio.sleep',sleep)
    with pytest.raises(budget.TrackBudgetExhausted):
        asyncio.run(await_model_jobs(tmp_path,'03-smoke',timeout=7200,track_deadline=deadline))
    assert sleeps==[1] and deadline.observation()['deadline_exhausted']


def test_setting_to_actual_actor_cli_and_historical_profile_unchanged(tmp_path):
    scope=evaluation_scope({'evaluation_mode':'full_single_pass'})
    old=actor_profile({},scope);assert 'track_timeout' not in old['args']
    settings={'evaluation_track_timeout_seconds':3600,'public_model_runtime':'/fixture/models',
        'training_python':'/fixture/python','cpu_image':'fixture','codex_bin':'/fixture/codex'}
    profile=actor_profile(settings,scope)
    args,argv=actor_arguments(tmp_path,tmp_path,settings,scope,profile,
        identity_path=tmp_path/'identity',canary_path=tmp_path/'canary')
    assert args.track_timeout==3600 and args.turn_timeout==900
    assert argv[argv.index('--track-timeout')+1]=='3600'
    assert {k:v for k,v in profile['args'].items() if k!='track_timeout'}==old['args']
    for invalid in (True,900,3601):
        with pytest.raises(ValueError):actor_profile({'evaluation_track_timeout_seconds':invalid},scope)


def bind_deadline(audit, turn, terminal):
    clock=[0.0];deadline=budget.TrackDeadline(audit,clock=lambda:clock[0]);clock[0]=3601
    timeout=terminal['policy_budget']['timeout_seconds']
    write_once(turn/'track-budget.json',{'schema':'eva.automedbench-track-turn-budget.v1',
        'budget_document_blake3':deadline.document['document_blake3'],
        'timeout_seconds':timeout,'elapsed_before_turn_seconds':3600-timeout})
    cleanup=write_once(audit/'track-deadline-cleanup.json',{
        'schema':'eva.automedbench-track-deadline-cleanup.v1','scope':'exact_owned_model_supervisors_only',
        'trigger':'track_deadline','workspace_quiescent':True,'error_category':None,'signals':[],
        'policy_time_extended':False})
    changed={**terminal,'track_budget':deadline.observation(),
        'deadline_cleanup_document_blake3':cleanup['document_blake3']}
    replace_doc(turn/'policy-budget-terminal.json',changed)


def test_real_sdk_interrupt_accepts_only_explicit_remaining_track_budget(proof):
    from training.automedbench_lite.policy_terminal import verify_policy_terminal
    turn,audit,raw,terminal=proof
    write_once(audit.parent/'attempt.json',{'track_budget_policy':budget.POLICY,
        'track_timeout_seconds':3600,'phase_timeout_policy':'remaining_track_budget'})
    bind_deadline(audit,turn,terminal)
    value=verify_policy_terminal(turn,audit,raw)
    assert value['track_budget']['timeout_seconds']==3600 and value['timeout_seconds']==terminal['policy_budget']['timeout_seconds']
    assert value['track_budget']['cleanup_overrun_seconds']==1 and value['reward'] is None
    doc=read_document(turn/'track-budget.json');doc['timeout_seconds']+=1
    replace_doc(turn/'track-budget.json',doc)
    with pytest.raises(ValueError,match='turn_budget_differs'):verify_policy_terminal(turn,audit,raw)


def test_actual_seven_prefix_gate_reopens_optin_deadline_without_relabelling_suffix(seven):
    from training.eva_rsi.terminal_attempts import verify_terminal_attempts
    document,attempt=seven
    root=Path(document['benchmark_run_root'])/'track-rollouts'
    replace_doc(root/'attempt.json',{**attempt,'track_budget_policy':budget.POLICY,
        'track_timeout_seconds':3600,'phase_timeout_policy':'remaining_track_budget'})
    for audit in (p for p in root.iterdir() if p.is_dir()):
        turn=audit/'turns/01-planning';terminal=read_document(turn/'policy-budget-terminal.json')
        terminal['policy_budget']['timeout_seconds']=123.5
        bind_deadline(audit,turn,terminal)
    verified=verify_terminal_attempts(document)
    assert len(verified['tracks'])==7
    assert all(row['attempted_stages']==['S1'] and row['unreachable_scores'] is None
        and row['policy_terminal']['timeout_seconds']==123.5 for row in verified['tracks'].values())


@pytest.mark.parametrize('bad', [None, 'workspace', 'birth', 'missing_exit'])
def test_cleanup_signals_only_bound_supervisor_and_requires_exit_publication(tmp_path,monkeypatch,bad):
    import os,sys
    from training.automedbench_lite import policy_capture,job_wait
    workspace=tmp_path/'workspace';workspace.mkdir()
    audit=tmp_path/'audit';audit.mkdir();job_id=str(uuid4());job=audit/'model-jobs'/job_id;job.mkdir(parents=True)
    pid=os.getpid();tick=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    script=Path(budget.__file__).resolve().parents[2]/'training/benchmark_models/job_supervisor.py'
    command=[sys.executable,'-B',str(script),'--workspace',str(workspace),'--job-id',job_id,
        '--audit-root',str(audit/'model-jobs/authoritative')]
    (job/'submission.json').write_text(json.dumps({'job_id':job_id,'workspace':str(workspace)}))
    recorded=list(command)
    if bad=='workspace':recorded[recorded.index('--workspace')+1]=str(tmp_path/'other')
    (job/'process.json').write_text(json.dumps({'pid':pid,'start_ticks':'wrong' if bad=='birth' else tick,
        'command':recorded,'job_id':job_id}))
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:('\0'.join(command)+'\0').encode()
        if str(p)==f'/proc/{pid}/cmdline' else original(p))
    opened=[];signals=[];order=[]
    monkeypatch.setattr(budget.os,'pidfd_open',lambda value:opened.append(value) or 9876)
    close=budget.os.close
    monkeypatch.setattr(budget.os,'close',lambda fd:None if fd==9876 else close(fd))
    monkeypatch.setattr(budget.signal,'pidfd_send_signal',lambda fd,sig:signals.append((fd,sig)))
    async def wait_jobs(*_,**kwargs):
        order.append('wait_exit');assert len(signals)==1 and 0<=kwargs['timeout']<=30
        if bad=='missing_exit':raise ValueError('fixture missing exit')
    async def publishers(*_,**kwargs):order.append('publish');assert order==['wait_exit','publish']
    monkeypatch.setattr(job_wait,'await_model_jobs',wait_jobs)
    monkeypatch.setattr(policy_capture,'await_model_publishers',publishers)
    if bad:
        with pytest.raises(ValueError):asyncio.run(budget.cleanup_owned_jobs(audit,workspace))
    else:
        result=asyncio.run(budget.cleanup_owned_jobs(audit,workspace));assert result['workspace_quiescent']
    doc=read_document(audit/'track-deadline-cleanup.json')
    assert doc['workspace_quiescent']==(bad is None)
    if bad in ('workspace','birth'):assert signals==[]
    else:assert signals==[(9876,budget.signal.SIGTERM)]
    assert not doc['policy_time_extended']


@pytest.mark.parametrize('changed',[False,True])
def test_deadline_between_turns_separates_late_job_snapshot_from_original_stage(tmp_path,changed):
    from blake3 import blake3
    from test_codex_policy_budget import MockAppServer,run
    from eva_agent.pipeline.digests import canonical_json_bytes
    from training.automedbench_lite.policy_terminal import verify_between_turn_terminal
    from training.eva_rsi.terminal_attempts import require_successful_model_jobs
    audit=tmp_path/'classification';turn=audit/'turns/01-planning';turn.mkdir(parents=True)
    receipt=asyncio.run(run(MockAppServer('normal',audit),tmp_path))
    raw=canonical_json_bytes(receipt);(turn/'receipt.json').write_bytes(raw)
    (turn/'after').mkdir()
    observed={'files':[],'observation_started_ns':1,'observation_completed_ns':2}
    original=write_once(turn/'after/manifest.json',{'schema':'synthetic-snapshot-fixture',**observed})
    original_bytes=(turn/'after/manifest.json').read_bytes()
    current={**observed,'observation_started_ns':3,'observation_completed_ns':4}
    workspace=tmp_path/'workspace';workspace.mkdir();blobs=audit/'workspace-blobs';blobs.mkdir()
    if changed:
        job=str(uuid4());job_root=audit/'model-jobs'/job;job_root.mkdir(parents=True)
        write_once(job_root/'submission.json',{'job_id':job,'workspace':str(workspace)})
        script=Path(budget.__file__).resolve().parents[2]/'training/benchmark_models/job_supervisor.py'
        command=['/fixture/python','-B',str(script),'--workspace',str(workspace),'--job-id',job,
            '--audit-root',str(audit/'model-jobs/authoritative')]
        write_once(job_root/'process.json',{'job_id':job,'pid':2147483000,'start_ticks':'fixture','command':command})
        relative=f'outputs/agents_outputs/prescribed-model-jobs/{job}/process-exit.json'
        data=json.dumps({'schema':'eva.prescribed-model-process-exit.v1','job_id':job,
            'returncode':0,'os_process_exit_observed':True}).encode()
        authoritative=audit/'model-jobs/authoritative'/job/'process-exit.json'
        authoritative.parent.mkdir(parents=True);authoritative.write_bytes(data)
        public=workspace/relative;public.parent.mkdir(parents=True);public.write_bytes(data)
        digest=blake3(data).hexdigest();(blobs/digest).write_bytes(data)
        current['files']=[{'path':relative,'bytes':len(data),'blake3':digest,'mode':0o600}]
    clock=[0.0];deadline=budget.TrackDeadline(audit,clock=lambda:clock[0]);clock[0]=3600
    write_once(tmp_path/'attempt.json',{'track_budget_policy':budget.POLICY,'track_timeout_seconds':3600,
        'phase_timeout_policy':'remaining_track_budget'})
    state={'deadline':deadline,'audit':audit,'workspace':workspace,'receipts':[receipt],
        'errors':[],'inventory':SimpleNamespace(capture=lambda:current,blobs=blobs)}
    asyncio.run(actor.settle_between_turn_deadline(state))
    assert (turn/'receipt.json').read_bytes()==raw and not (turn/'policy-budget-terminal.json').exists()
    proof=verify_between_turn_terminal(turn,audit,json.loads(raw))
    assert proof['valid'] and proof['actual_terminal_status']=='completed' and proof['reward'] is None
    assert state['errors'][0]['error']=='track_budget_exhausted'
    assert (turn/'after/manifest.json').read_bytes()==original_bytes
    assert proof['after_snapshot_document_blake3']==original['document_blake3']
    if changed:
        assert proof['original_phase_after_preserved'] and proof['late_outputs_excluded_from_prior_stage_A']
        require_successful_model_jobs(audit,turn,model_job_snapshot_root=proof['model_job_snapshot_root'])
        with pytest.raises((ValueError,FileNotFoundError)):require_successful_model_jobs(audit,turn)
        cleanup_root=Path(proof['model_job_snapshot_root'])
        (cleanup_root/'files'/relative).write_bytes(b'changed fixture bytes')
        with pytest.raises(ValueError,match='exit_not_in_final_snapshot'):
            verify_between_turn_terminal(turn,audit,json.loads(raw))
    else:
        assert 'model_job_snapshot_root' not in proof and not (audit/'track-deadline-after').exists()


def test_owned_interrupt_capture_plumbs_deadline_cleanup_and_keeps_actual_receipt(tmp_path,monkeypatch):
    from test_codex_policy_budget import MockAppServer,backend,options
    from eva_agent.codex_runtime import CodexRuntime,CodexTurnInput
    from eva_agent.pipeline.digests import canonical_json_bytes
    from training.automedbench_lite import policy_capture
    from training.automedbench_lite.policy_terminal import verify_policy_terminal
    audit=tmp_path/'classification';turn=audit/'turns/01-planning';turn.mkdir(parents=True)
    clock=[0.0];deadline=budget.TrackDeadline(audit,clock=lambda:clock[0]);clock[0]=3599.98
    timeout=deadline.remaining()
    write_once(tmp_path/'attempt.json',{'track_budget_policy':budget.POLICY,'track_timeout_seconds':3600,
        'phase_timeout_policy':'remaining_track_budget'})
    write_once(turn/'track-budget.json',{'schema':'eva.automedbench-track-turn-budget.v1',
        'budget_document_blake3':deadline.document['document_blake3'],'timeout_seconds':timeout,
        'elapsed_before_turn_seconds':3600-timeout})
    server=MockAppServer('interrupt',audit);order=[]
    async def cleanup(actual_audit,workspace):
        assert actual_audit==audit and server.order[-1]=='unregistered'
        clock[0]=3600.1;order.append('cleanup')
        return write_once(audit/'track-deadline-cleanup.json',{
            'schema':'eva.automedbench-track-deadline-cleanup.v1','scope':'exact_owned_model_supervisors_only',
            'trigger':'track_deadline','workspace_quiescent':True,'error_category':None,'signals':[],
            'policy_time_extended':False})
    monkeypatch.setattr(budget,'cleanup_owned_jobs',cleanup)
    def snapshot(inventory,path):
        assert order==['cleanup'];order.append('snapshot');path.mkdir()
        return write_once(path/'manifest.json',{'schema':'synthetic-snapshot-fixture','files':[]})
    async def scenario():
        async with CodexRuntime(backend(server)) as runtime:
            handle=await runtime.start_thread(options(tmp_path))
            return await policy_capture.capture_turn(runtime,handle,CodexTurnInput(public_text='fixture'),
                timeout=timeout,target=turn,audit=audit,inventory=SimpleNamespace(workspace=tmp_path/'workspace'),
                snapshot=snapshot,track_deadline=deadline)
    receipt,terminal=asyncio.run(scenario())
    assert order==['cleanup','snapshot'] and terminal['workspace_quiescence_verified']
    assert (turn/'receipt.json').read_bytes()==canonical_json_bytes(receipt)
    proof=verify_policy_terminal(turn,audit,json.loads(canonical_json_bytes(receipt)))
    assert proof['valid'] and proof['track_budget']['cleanup_overrun_seconds']==pytest.approx(.1)


def test_startup_timeout_still_closes_partially_entered_owned_runtime(tmp_path,monkeypatch):
    calls=[]
    class Starting:
        def __init__(self,_):pass
        async def __aenter__(self):
            calls.append('enter');await asyncio.sleep(10)
        async def __aexit__(self,*_):calls.append('close')
    monkeypatch.setattr(actor,'CodexRuntime',Starting)
    async def scenario():
        deadline = SimpleNamespace(require_remaining=lambda:.005,
            document={'document_blake3':'0'*64})
        async with actor.deadline_runtime(None,deadline,asyncio.Lock(),audit=tmp_path,
                runtime_sequence=1,phase_intent='01-planning'):
            pytest.fail('startup did not time out')
    with pytest.raises(asyncio.TimeoutError):asyncio.run(scenario())
    assert calls==['enter','close']


@pytest.mark.parametrize('returncode,forwarded,valid',[(-15,[15],True),(143,[15],True),(1,[15],False),(-15,[],False)])
def test_only_bound_owned_cancellation_is_policy_stop_not_unrelated_runtime_failure(tmp_path,returncode,forwarded,valid):
    from training.automedbench_lite.policy_terminal import verify_deadline_cleanup
    job=str(uuid4());directory=tmp_path/'model-jobs'/job;directory.mkdir(parents=True)
    write_once(directory/'process.json',{'pid':123,'start_ticks':'456'})
    target=tmp_path/'model-jobs/authoritative'/job;target.mkdir(parents=True)
    (target/'process-exit.json').write_text(json.dumps({'returncode':returncode,'forwarded_signals':forwarded}))
    cleanup=write_once(tmp_path/'track-deadline-cleanup.json',{
        'schema':'eva.automedbench-track-deadline-cleanup.v1','scope':'exact_owned_model_supervisors_only',
        'trigger':'track_deadline','workspace_quiescent':True,'error_category':None,'policy_time_extended':False,
        'signals':[{'job_id':job,'pid':123,'start_ticks':'456','signal':'SIGTERM'}]})
    terminal={'deadline_cleanup_document_blake3':cleanup['document_blake3']}
    if valid:
        assert verify_deadline_cleanup(tmp_path,terminal,[{'job_id':job,'returncode':returncode}])==[job]
    else:
        with pytest.raises(ValueError,match='exit_unproved'):
            verify_deadline_cleanup(tmp_path,terminal,[{'job_id':job,'returncode':returncode}])
