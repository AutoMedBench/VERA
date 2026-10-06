"""Offline opt-in eligibility using actual immutable workspace-byte reads."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from eva_agent.codex_pipeline.adapter import CodexJudgeToolExecutionBridge, _rubric_items
from eva_agent.pipeline import JudgeRequest, RandomUUIDFactory
from eva_agent.pipeline.digests import blake3_hex, canonical_value
from eva_agent.pipeline.judge_material_view import POLICY_VISIBLE_VIEW, material_layout
from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
from eva_agent.training.agent_judge import AgentJudgeSelectionError
from eva_agent.training.agent_judge_worker import verdict_format_correction_eligibility
from eva_agent.training.slime_agent_judge import _backend_provenance, validate_verdict_format_policy
from test_opus_complete_actor_read_plan import _prepared


def _case(tmp_path, *, omit_tail=False, failed_tool=False, omit_before=False):
    prepared = _prepared(tmp_path, message_bytes=131071)
    request = JudgeRequest(
        judgment_id=prepared.task.judge_task_id, judge_model_id=prepared.task.judge_model_id,
        policy_visible_context=prepared.evidence.policy_visible_context,
        workspace_evidence=prepared.evidence, judge_only_reference={},
    )
    tools = JudgeWorkspaceTools(prepared.evidence, id_factory=RandomUUIDFactory(),
                                maximum_parallel_tools=64)
    bridge = CodexJudgeToolExecutionBridge(tools, id_factory=RandomUUIDFactory(),
        maximum_total_calls=64, maximum_execution_groups=32)
    _, paths, _ = material_layout(prepared.evidence)
    files = {row.path: row for row in prepared.evidence.workspace_after.files}
    calls = []
    for path in paths:
        size = files[path].byte_count
        offsets = list(range(0, max(1, size), 65536))
        if omit_tail and path.endswith('/messages.json'):
            offsets.pop()
        calls.extend(('workspace_read', {'snapshot': 'after', 'path': path,
            'offset': offset, 'max_bytes': min(65536, max(1, size-offset))}) for offset in offsets)
    calls.extend(('workspace_read', {'snapshot': snapshot, 'path': 'input.txt',
        'offset': 0, 'max_bytes': 65536}) for snapshot in ('before', 'after')
                 if not (omit_before and snapshot == 'before'))
    if failed_tool:
        calls.append(('workspace_read', {'snapshot': 'after', 'path': 'missing.txt',
                                         'offset': 0, 'max_bytes': 10}))
    bridge.execute_group(calls)
    items = _rubric_items(prepared.trajectory.rubric)
    gates = not any(isinstance(item.get('hard_gate'), dict)
        and item['hard_gate']['minimum_score_bps'] > 0 for item in items)
    verdict = {'item_scores': [{'item_id': item['item_id'], 'score': 0,
        'evidence_refs': ['workspace:before:input.txt'], 'rationale': 'Observed source bytes.',
        'score_note': 'extraneous metadata'} for item in items],
        'hard_gates_passed': gates, 'summary': 'Offline exact rubric verdict.'}
    return prepared, request, bridge, verdict


def _check(case):
    prepared, request, bridge, verdict = case
    return verdict_format_correction_eligibility(json.dumps(verdict), prepared=prepared,
        request=request, rubric=prepared.trajectory.rubric, judge_bridge=bridge)


@pytest.mark.parametrize('extra', ['rationale_note', 'score_note', 'tool_calls', 'evidence_refs_extra', '{}'])
def test_only_whitelisted_extra_fields_after_complete_real_reads(tmp_path, extra):
    prepared, request, bridge, verdict = case = _case(tmp_path)
    for row in verdict['item_scores']:
        row[extra] = row.pop('score_note')
    original = json.dumps(verdict)
    before = bridge.snapshot_for_verdict_correction()
    result = _check(case)
    assert result is not None
    assert result['score_emitted'] is False
    assert result['complete_actor_sidecars_verified'] is True
    assert result['all_item_source_citations_verified'] is True
    assert result['workspace_result_receipts'] == [row.receipt_blake3 for row in before.results]
    assert bridge.snapshot_for_verdict_correction() == before
    assert json.dumps(verdict) == original
    projection = {**verdict, 'item_scores': [{k: v for k, v in row.items() if k != extra}
                                           for row in verdict['item_scores']]}
    assert result['required_values_blake3'] == blake3_hex(projection)
    message_reads = [r for r in before.results if r.arguments.get('path', '').endswith('/messages.json')]
    assert len(message_reads) > 1
    assert sum(r.output['returned_bytes'] for r in message_reads) > 131071


@pytest.mark.parametrize('failure', ['valid_zero', 'unknown_extra', 'missing_required', 'id',
    'order', 'level', 'hard_gate', 'unread_ref', 'duplicate_ref', 'no_source_ref',
    'top_extra', 'bool_score', 'empty_rationale', 'wrong_evidence_binding', 'inflight'])
def test_semantic_evidence_or_non_extra_errors_never_eligible(tmp_path, failure):
    prepared, request, bridge, verdict = _case(tmp_path)
    row = verdict['item_scores'][0]
    if failure == 'valid_zero':
        for row in verdict['item_scores']:
            row.pop('score_note')
    elif failure == 'unknown_extra': row['not_allowed'] = ''
    elif failure == 'missing_required': row.pop('rationale')
    elif failure == 'id': row['item_id'] = 'wrong-id'
    elif failure == 'order': verdict['item_scores'].reverse()
    elif failure == 'level': row['score'] = .37
    elif failure == 'hard_gate': verdict['hard_gates_passed'] = not verdict['hard_gates_passed']
    elif failure == 'unread_ref': row['evidence_refs'].append('workspace:before:missing.txt')
    elif failure == 'duplicate_ref': row['evidence_refs'] *= 2
    elif failure == 'no_source_ref': row['evidence_refs'] = ['workspace:after:.eva-agent-judge/actor/messages.json']
    elif failure == 'top_extra': verdict['tool_calls'] = []
    elif failure == 'bool_score': row['score'] = False
    elif failure == 'empty_rationale': row['rationale'] = ''
    elif failure == 'wrong_evidence_binding': request = replace(request, judgment_id=RandomUUIDFactory().new('other'))
    elif failure == 'inflight': bridge._calls_started += 1
    assert _check((prepared, request, bridge, verdict)) is None


@pytest.mark.parametrize('kwargs', [{'omit_tail': True}, {'failed_tool': True}])
def test_missing_tail_or_failed_tool_cannot_get_extra_model_call(tmp_path, kwargs):
    assert _check(_case(tmp_path, **kwargs)) is None


@pytest.mark.parametrize('backend,view,timeout', [('native_astra', POLICY_VISIBLE_VIEW, 600),
    ('opus_5', 'historical-v1', 600), ('opus_5', POLICY_VISIBLE_VIEW, 240),
    ('opus_5', POLICY_VISIBLE_VIEW, 601), ('opus_5', POLICY_VISIBLE_VIEW, 600.0)])
def test_selector_requires_exact_new_profile(backend, view, timeout):
    with pytest.raises(AgentJudgeSelectionError):
        validate_verdict_format_policy('extra-row-fields-once-v1', backend=backend,
                                      material_view=view, timeout_seconds=timeout)


def test_explicit_provenance_and_legacy_shape(tmp_path):
    prepared, *_ = _case(tmp_path)
    old = _backend_provenance(prepared, 'opus_5', native_turn_timeout_seconds=600)
    assert 'verdict_format_policy' not in old
    new = _backend_provenance(prepared, 'opus_5', native_turn_timeout_seconds=600,
                             verdict_format_policy='extra-row-fields-once-v1')
    assert new['verdict_format_policy'] == 'extra-row-fields-once-v1'
    assert new['maximum_verdict_format_corrections'] == 1
    assert new['format_correction_total_deadline_seconds'] == 600
    assert {k: v for k, v in old.items() if k != 'provenance_blake3'} == {
        k: v for k, v in new.items() if k != 'provenance_blake3' and k in old}


@pytest.mark.parametrize('unread_before', [False, True])
def test_real_gateway_uses_live_read_eligibility_and_shared_total_deadline(tmp_path, unread_before):
    from types import SimpleNamespace
    from eva_agent.codex_pipeline.adapter import _judge_output_schema
    from eva_agent.codex_providers import adapter
    from eva_agent.codex_providers.judge_verdict_correction import (
        JudgeVerdictCorrectionPolicy, verify_judge_verdict_correction,
    )
    from eva_agent.training.slime_agent_judge import (
        PolicyVisibleOpusWorkspaceJudge, _verdict_format_source_binding,
    )
    from test_codex_provider_routes import _routes, _post_json, _chat_response
    from test_judge_verdict_correction_gateway import FakeUpstream
    from test_opus_complete_actor_read_plan import _Capture

    prepared, request, bridge, verdict = _case(tmp_path, omit_before=unread_before)
    records = []
    policy = JudgeVerdictCorrectionPolicy(judgment_id=request.judgment_id,
        source_binding=_verdict_format_source_binding(prepared, 600), sink=records.append)
    judge = PolicyVisibleOpusWorkspaceJudge(options_factory=lambda *_: None,
        model_id=request.judge_model_id, runner=_Capture(),
        turn_mcp_factory=SimpleNamespace(open_judge=lambda *_: None),
        fast_workspace_judge=True, compact_evidence_index=True,
        maximum_workspace_tool_frontiers=32, turn_timeout_seconds=600,
        verdict_correction_policy=policy, verdict_correction_prepared=prepared)
    deadline = judge._bind_verdict_format_correction(request, prepared.trajectory.rubric, None, bridge)
    assert 0 < policy.remaining_seconds() <= 600
    assert deadline == policy._deadline
    route = _routes(tmp_path)['opus_5']
    original = json.dumps(verdict)
    corrected = json.dumps({**verdict, 'item_scores': [
        {k: v for k, v in row.items() if k != 'score_note'} for row in verdict['item_scores']]})
    fake = FakeUpstream([_chat_response(route.model_id, content=original),
                         _chat_response(route.model_id, content=corrected)])
    body = {'model': route.model_id, 'input': [{'role': 'user', 'content': 'Fixture only.'}],
        'store': False, 'stream': False, 'text': {'format': {'type': 'json_schema',
        'name': 'codex_output_schema', 'strict': True,
        'schema': canonical_value(_judge_output_schema(_rubric_items(prepared.trajectory.rubric)))}}}
    with adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=fake,
                                        limits=adapter.AdapterLimits(upstream_timeout_seconds=240),
                                        judge_verdict_correction=policy) as gateway:
        adapted = gateway.adapted_routes()[route.route_id]
        _, _, (endpoint, token) = adapted.config.for_subprocess()
        status, raw, _ = _post_json(endpoint+'/responses', token, body)
    assert status == 200
    returned = json.loads(raw)['output'][0]['content'][0]['text']
    assert all(0 < t <= 240 for t in fake.timeouts)
    if unread_before:
        assert returned == original and len(fake.calls) == 1
        assert policy.outcome == 'ineligible' and records == []
    else:
        assert returned == corrected and len(fake.calls) == 2
        assert policy.outcome == 'corrected' and len(records) == 2
        proof = verify_judge_verdict_correction(records[-1])
        assert proof['accepted'] is True
        assert proof['eligibility_proof']['workspace_result_receipts'] == [
            row.receipt_blake3 for row in bridge.snapshot_for_verdict_correction().results]
        assert proof['original']['final_text'] == original
        assert proof['corrected']['final_text'] == corrected


def test_live_policy_rejects_wrong_prepared_binding_before_dispatch(tmp_path):
    from types import SimpleNamespace
    from eva_agent.codex_providers.judge_verdict_correction import JudgeVerdictCorrectionPolicy
    from eva_agent.training.slime_agent_judge import PolicyVisibleOpusWorkspaceJudge
    from test_opus_complete_actor_read_plan import _Capture
    prepared, request, bridge, _ = _case(tmp_path)
    policy = JudgeVerdictCorrectionPolicy(judgment_id=request.judgment_id,
        source_binding={'wrong': 'source'}, sink=lambda _: pytest.fail('no provider or signed correction'))
    judge = PolicyVisibleOpusWorkspaceJudge(options_factory=lambda *_: None,
        model_id=request.judge_model_id, runner=_Capture(),
        turn_mcp_factory=SimpleNamespace(open_judge=lambda *_: None),
        fast_workspace_judge=True, compact_evidence_index=True, turn_timeout_seconds=600,
        verdict_correction_policy=policy, verdict_correction_prepared=prepared)
    with pytest.raises(AgentJudgeSelectionError, match='prepared source binding'):
        judge._bind_verdict_format_correction(request, prepared.trajectory.rubric, None, bridge)
