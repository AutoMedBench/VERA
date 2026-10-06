"""Synthetic real-SDK interrupt fixture; no provider/GPU or clinical score."""
import asyncio
from copy import deepcopy
import json
import time

import pytest

from eva_agent.codex_runtime.policy_budget import CodexPolicyBudgetExceeded
from eva_agent.pipeline.digests import canonical_json_bytes
from training.automedbench_lite.adapter import write_once
from training.automedbench_lite.policy_capture import require_joined_host_results
from training.automedbench_lite.policy_terminal import verify_policy_terminal
from test_codex_policy_budget import MockAppServer,run


@pytest.fixture
def proof(tmp_path):
    audit=tmp_path/'classification';audit.mkdir()
    turn=audit/'turns/01-planning';turn.mkdir(parents=True)
    server=MockAppServer('interrupt',audit)
    with pytest.raises(CodexPolicyBudgetExceeded) as captured:
        asyncio.run(run(server,tmp_path))
    receipt=captured.value.receipt;raw=json.loads(canonical_json_bytes(receipt))
    after=turn/'after';after.mkdir()
    manifest=write_once(after/'manifest.json',{'schema':'synthetic-snapshot-fixture','files':[]})
    terminal={'schema':'eva.automedbench-policy-budget-terminal.v1','phase_intent':turn.name,
        'policy_budget':captured.value.outcome,'receipt_blake3':receipt.receipt_blake3,
        'actual_terminal_status':receipt.status,'workspace_quiescence_verified':True,
        'host_quiescence_observed_ns':time.time_ns(),'after_snapshot_document_blake3':manifest['document_blake3'],
        'infrastructure_error':None,'reward':None,'later_stages_evaluated':False,
        **require_joined_host_results(receipt,audit)}
    write_once(turn/'policy-budget-terminal.json',terminal)
    return turn,audit,raw,terminal


def test_actual_sdk_interrupt_terminal_reopened_without_new_actions(proof):
    turn,audit,raw,_=proof
    before={str(p):p.read_bytes() for p in audit.rglob('*') if p.is_file()}
    value=verify_policy_terminal(turn,audit,raw)
    assert value['valid'] and value['joined_host_call_count']==1
    assert value['model_job_exits']==[] and value['reward'] is None and value['provider_calls']==0
    assert before=={str(p):p.read_bytes() for p in audit.rglob('*') if p.is_file()}


@pytest.mark.parametrize('field',["interrupt_acknowledged_ns","terminal_observed_ns","turn_id","receipt_blake3"])
def test_missing_or_wrong_owned_interrupt_binding_rejected(proof,field):
    turn,audit,raw,original=proof
    value=deepcopy(original)
    value['policy_budget'][field]=None if field.endswith('_ns') else 'different'
    path=turn/'policy-budget-terminal.json';path.unlink();write_once(path,value)
    with pytest.raises(ValueError):verify_policy_terminal(turn,audit,raw)


@pytest.mark.parametrize('field,value',[('workspace_quiescence_verified',False),('infrastructure_error','local429'),
    ('joined_host_call_count',2),('after_snapshot_document_blake3','0'*64)])
def test_task_transport_or_quiescence_failure_never_becomes_budget_reward(proof,field,value):
    turn,audit,raw,original=proof
    changed={**original,field:value}
    path=turn/'policy-budget-terminal.json';path.unlink();write_once(path,changed)
    with pytest.raises(ValueError):verify_policy_terminal(turn,audit,raw)


def test_changed_host_event_rejected(proof):
    turn,audit,raw,_=proof
    (audit/'mcp-events.jsonl').write_text('')
    with pytest.raises(ValueError,match='quiescence_unproved'):
        verify_policy_terminal(turn,audit,raw)
