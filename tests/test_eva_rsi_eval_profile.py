"""CPU-only post-round wiring; simulated boundaries are not model/medical results."""
import json
from pathlib import Path
import sys

import pytest

from training.eva_rsi import production_eval as module
from training.eva_rsi.evidence import commitment
from training.automedbench_lite import track_actor
from test_eva_rsi_production_eval import integration


def test_new_full_round_requests_actual_supra_and_matched_limits():
    scope=module.evaluation_scope({'evaluation_mode':'full_single_pass'})
    profile=module.actor_profile({},scope)
    assert profile['supra_profile'] is True and profile['memory_profile'] is False
    assert profile['thinking'] is True
    assert profile['args']=={'workers':3,'context_length':32768,'auto_compact_token_limit':20480,
        'output_tokens':4096,'adapter_capacity_wait_seconds':60,'turn_timeout':900}
    old=module.actor_profile({},module.evaluation_scope({}))
    assert old['name']=='legacy' and not old['supra_profile'] and not old['memory_profile']


@pytest.mark.parametrize('setting,value',[('evaluation_context_length',24576),('evaluation_compact_tokens',12288),
    ('evaluation_output_tokens',8192),('evaluation_workers',5),('evaluation_workers',True)])
def test_unsupported_full_profile_fails_before_any_execution(setting,value):
    with pytest.raises(ValueError):
        module.actor_profile({setting:value},module.evaluation_scope({'evaluation_mode':'full_single_pass'}))


def test_equivalent_cli_matches_exact_inprocess_args(integration):
    x=integration; scope=module.evaluation_scope({'evaluation_mode':'full_single_pass'})
    profile=module.actor_profile({},scope); attempt=Path(x['context']['attempt_root'])
    args,argv=module.actor_arguments(x['run'],attempt,x['settings'],scope,profile,
        identity_path=attempt/'serving/checkpoint-identity.json',canary_path=attempt/'serving/image-canary.json')
    assert args.tracks==scope['tracks'] and len(args.tracks)==7
    assert argv.count('--supra-profile')==1 and '--memory-profile-v12' not in argv
    for name in profile['args']:
        flag='--'+name.replace('_','-')
        assert argv[argv.index(flag)+1]==str(getattr(args,name))
    assert args.endpoint=='http://127.0.0.1:30911/v1'
    assert args.codex_bin==Path(x['settings']['codex_bin'])


def test_tool_output_limit_is_prospective_explicit_and_cli_bound(integration):
    x=integration;scope=module.evaluation_scope({'evaluation_mode':'full_single_pass'})
    old=module.actor_profile({},scope)
    assert old['tool_output_token_limit'] is None and 'tool_output_token_limit' not in old['args']
    profile=module.actor_profile({'evaluation_tool_output_token_limit':2048},scope)
    args,argv=module.actor_arguments(x['run'],Path(x['context']['attempt_root']),x['settings'],scope,profile,
        identity_path=Path('/fixture/identity.json'),canary_path=Path('/fixture/canary.json'))
    assert args.tool_output_token_limit==2048
    assert argv[argv.index('--tool-output-token-limit')+1]=='2048'
    for invalid in (True,0,4096,'2048'):
        with pytest.raises(ValueError,match='tool_output_limit'):
            module.actor_profile({'evaluation_tool_output_token_limit':invalid},scope)


def test_old_actor_profile_missing_is_rejected_before_public_prepare_or_gpu(integration):
    x=integration;x['settings']['evaluation_mode']='full_single_pass'
    with pytest.raises(ValueError,match='profile_not_installed'):
        module.evaluate_round(x['context'],x['settings'])
    assert x['events']==[]


def test_full_round_forwards_profile_context_python_and_writes_content_index(integration,monkeypatch):
    x=integration;x['settings']['evaluation_mode']='full_single_pass'
    x['settings']['native_evaluator_python']=sys.executable
    captures={}
    async def actor(args,*,memory_profile=False,supra_profile=False):
        captures.update(args=args,memory=memory_profile,supra=supra_profile)
        x['events'].append('actor_fixture')
    monkeypatch.setattr(track_actor,'run_tracks',actor)
    monkeypatch.setattr(module,'actor_runtime_sources',lambda:[{'path':'/fixture/actual-import.py','blake3':'a'*64}])
    monkeypatch.setattr(module,'codex_binary_binding',lambda binary:{'codex_version':'codex-cli 0.153.4',
        'codex_binary':str(binary),'codex_binary_blake3':'b'*64})
    native = module.score_native_round
    def native_scores(*args, **kwargs):
        assert not x['state']['serving']
        x['events'].append('native_scores_after_stop')
        return native(*args, **kwargs)
    monkeypatch.setattr(module, 'score_native_round', native_scores)
    def gate(document):
        assert x['events'][-1] == 'native_scores_after_stop'
        x['events'].append('rubric_gate')
    monkeypatch.setattr('training.eva_rsi.production.require_complete_baseline',gate)
    def judge(run,identity,output,*,stages,registry_path,native_turn_timeout_seconds,material_view):
        assert not x['state']['serving']
        assert x['events'][-1] == 'rubric_gate'
        assert material_view == 'policy-visible-audit-v2'
        captures['judge_timeout']=native_turn_timeout_seconds
        captures['stages']=stages
        return tuple(output/track/stage for track in module.evaluation_scope(x['settings'])['tracks'] for stage in stages)
    monkeypatch.setattr(module,'judge_full_round',judge)
    monkeypatch.setattr('training.eva_rsi.skill_identity.build_skill_content_binding',
        lambda run:{'content_identity_blake3':x['context']['skill_catalog_id'],'fixture_only':True})
    monkeypatch.setattr(module,'verify_evaluation',lambda path,context:json.loads(path.read_text()))
    index=module.evaluate_round(x['context'],x['settings'])
    assert captures['supra'] is True and captures['memory'] is False
    assert captures['args'].adapter_capacity_wait_seconds==60
    assert captures['args'].auto_compact_token_limit==20480
    assert captures['judge_timeout']==600 and len(captures['stages'])==5
    assert len(index['feedback_roots'])==35
    assert index['schema']=='eva.rsi-evaluation-index.v2'
    assert index['judge_material_view']=='policy-visible-audit-v2'
    assert index['skill_catalog_identity_mode']=='verified-content-v1'
    assert index['skill_content_identity']==commitment(Path(x['context']['attempt_root'])/'skill-content-binding.json')
    native_summary = Path(x['context']['attempt_root'])/'native-task-scores/summary.json'
    assert index['native_task_scores'] == commitment(native_summary)
    assert str(native_summary) in index['diagnostic_artifacts']
    assert json.loads(native_summary.read_text())['unavailable_tracks'] == 7
    assert len(index['feedback_roots']) == 35  # Native unavailable rows never become stage-zero grades.
    assert x['captures']['server']['python_executable']==Path(x['settings']['training_python'])
    binding=json.loads((Path(x['context']['attempt_root'])/'evaluation-actor-binding.json').read_text())
    assert binding['model_path']==x['context']['model_path']
    assert binding['memory_behavior_verified_by_configuration'] is False
    assert binding['actual_imported_sources'][0]['path']=='/fixture/actual-import.py'
    assert binding['codex_version']=='codex-cli 0.153.4'


def test_codex_actual_version_bound_before_gpu(tmp_path,monkeypatch):
    binary=tmp_path/'codex';binary.write_bytes(b'CPU fixture, never executed');binary.chmod(0o700)
    monkeypatch.setattr(module.subprocess,'check_output',lambda argv,**kwargs:'codex-cli 0.153.4\n')
    binding=module.codex_binary_binding(binary)
    assert binding['codex_binary_blake3']==commitment(binary)['blake3']
    monkeypatch.setattr(module.subprocess,'check_output',lambda argv,**kwargs:'codex-cli wrong\n')
    with pytest.raises(ValueError,match='version_differs'):
        module.codex_binary_binding(binary)


def test_content_change_is_not_hidden_by_new_mount_path(tmp_path,monkeypatch):
    monkeypatch.setattr('training.eva_rsi.skill_identity.build_skill_content_binding',
        lambda run:{'content_identity_blake3':'a'*64})
    with pytest.raises(ValueError,match='skill_content_differs'):
        module.evaluation_skill_index(tmp_path/'run',tmp_path,{'skill_catalog_id':'b'*64},{},
            module.evaluation_scope({'evaluation_mode':'full_single_pass'}))
    assert not (tmp_path/'skill-content-binding.json').exists()
