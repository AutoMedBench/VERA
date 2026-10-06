"""Provider-free checks of the approved fixed two-call boundary."""
import json
from pathlib import Path
import threading

import pytest

from training.eva_rsi import report_rejudge as m


def fixture(tmp_path, monkeypatch):
    packet_root = tmp_path / 'packet'
    packet_root.mkdir()
    sources = {}
    for name in ('actor_attempt', 'run_manifest', 'checkpoint_identity', 'actor_runtime_binding', 'registry'):
        path = tmp_path / (name + '.json')
        path.write_text('{}')
        sources[name] = m.commit(path)
    stages = {}
    for stage in m.STAGES:
        target = packet_root / stage
        target.mkdir()
        for name in ('rollout.json', 'preflight.json', 'old-isolation.json'):
            (target / name).write_text('{}')
        stages[stage] = {'root': str(target), 'prepared': {name: m.commit(target / name)
            for name in ('rollout.json', 'preflight.json')},
            'original_unavailable': {'isolation': m.commit(target / 'old-isolation.json'), 'failures': []}}
    binding = {'frozen': 'same'}
    monkeypatch.setattr(m, 'execution_binding', lambda: binding.copy())
    m.write_once(packet_root / 'packet.json', {'policy': m.POLICY, 'stages': stages,
        'source': sources, 'execution_binding': binding,
        'permitted_pairs': [['report', stage] for stage in m.STAGES],
        'retry_count': 0, 'semantic_attempts_per_stage': 1, 'native_turn_timeout_seconds': 600})
    return packet_root, binding


def test_exact_two_parallel_calls_independent_reopen_and_consumed_claim(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path, monkeypatch)
    barrier = threading.Barrier(2)
    calls, verifies = [], []
    def judge(path, **kwargs):
        calls.append((path.name, kwargs))
        barrier.wait(timeout=3)
        return {'stage': path.name}
    def verify(path):
        verifies.append(path.name)
        return {'valid': True, 'status': 'scored', 'score': {'reward_bps': 0}}
    monkeypatch.setattr(m, 'judge_once', judge)
    monkeypatch.setattr(m, 'verify_feedback', verify)
    result = m.execute_packet(root, execute=True)
    assert result['all_verified'] is True
    assert sorted(verifies) == ['S4', 'S5']
    assert sorted(x[0] for x in calls) == ['S4', 'S5']
    assert all(x[1]['native_turn_timeout_seconds'] == 600 for x in calls)
    with pytest.raises(FileExistsError):
        m.execute_packet(root, execute=True)
    assert len(calls) == 2


def test_unavailable_is_not_zero_and_sibling_finishes(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path, monkeypatch)
    calls = []
    def judge(path, **kwargs):
        calls.append(path.name)
        if path.name == 'S4':
            raise RuntimeError('private content must not appear')
        return {}
    monkeypatch.setattr(m, 'judge_once', judge)
    monkeypatch.setattr(m, 'verify_feedback', lambda _: {'valid': True, 'status': 'scored', 'score': {'reward_bps': 0}})
    result = m.execute_packet(root, execute=True)
    assert result['all_verified'] is False and sorted(calls) == ['S4', 'S5']
    failure = m.read_document(root / 'S4/one-shot-result.json')
    assert failure['score'] is None and failure['status'] == 'unavailable'
    assert 'private content' not in json.dumps(failure)
    assert m.read_document(root / 'S5/one-shot-result.json')['status'] == 'verified'


def test_noexecute_or_source_change_never_claims_or_calls(tmp_path, monkeypatch):
    root, binding = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(m, 'judge_once', lambda *a, **k: pytest.fail('provider boundary reached'))
    with pytest.raises(ValueError, match='explicit_execute'):
        m.execute_packet(root)
    binding['frozen'] = 'changed'
    with pytest.raises(ValueError, match='execution_source_changed'):
        m.execute_packet(root, execute=True)
    assert not (root / 'execute-claim.json').exists()


def test_actual_execution_binding_imports_selected_worker():
    binding = m.execution_binding()
    assert 'eva_agent.training.agent_judge_worker' in binding['sources']
    assert len(binding['core_revision']) == 40


def composition_fixture(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path, monkeypatch)
    packet = m.read_document(root / 'packet.json')
    packet['benchmark_run_root'] = str(tmp_path / 'source-run')
    packet['execution_binding'] = {'core_revision': 'actual-new-core', 'sources': {}}
    packet_path = root / 'packet.json'
    packet_path.unlink()  # Synthetic fixture construction only.
    m.write_once(packet_path, {k: v for k, v in packet.items() if k != 'document_blake3'})
    m.write_once(root / 'execution-result.json', {'all_verified': True})
    old_root = tmp_path / 'old-feedback'
    old_root.mkdir()
    (old_root / 'preflight.json').write_text(json.dumps({'case_id': 'report', 'stage': 'S3'}))
    original = {'schema': 'eva.rsi-evaluation-index.v1', 'feedback_roots': [str(old_root)],
        'benchmark_run_root': packet['benchmark_run_root'],
        'checkpoint_identity': packet['source']['checkpoint_identity']['path'],
        'actor_runtime_binding': packet['source']['actor_runtime_binding'],
        'postround_judge_verdict_replacements': 2,
        'postround_judge_attempt_isolation': [{'path': '/old/unchanged', 'blake3': 'original'}],
        'diagnostic_only': True}
    original_root = tmp_path / 'original'
    original_root.mkdir()
    m.write_once(original_root / 'index.json', original)
    results = [{'root': str(root / stage), 'stage': stage, 'track': 'report',
                'judge_receipt_blake3': stage} for stage in m.STAGES]
    monkeypatch.setattr(m, '_verified_execution', lambda path: (packet, results.copy()))
    return original_root / 'index.json', root, original, results


def test_composition_fixed_append_preserves_old_policy_and_sources(tmp_path, monkeypatch):
    original_path, packet_root, original, results = composition_fixture(tmp_path, monkeypatch)
    old_bytes = original_path.read_bytes()
    output = tmp_path / 'new-index'
    index = m.compose_index(original_index=original_path, packet_root=packet_root, output_root=output)
    proof = m.verify_rejudge_index_sidecar(index)
    assert proof['standalone_roots'] == [row['root'] for row in results]
    assert index['feedback_roots'] == original['feedback_roots'] + proof['standalone_roots']
    assert index['postround_judge_attempt_isolation'] == original['postround_judge_attempt_isolation']
    assert original_path.read_bytes() == old_bytes
    index['postround_judge_verdict_replacements'] = 0
    with pytest.raises(ValueError, match='original_index_fields_changed'):
        m.verify_rejudge_index_sidecar(index)


def test_composition_rejects_valid_original_target_no_score_selection(tmp_path, monkeypatch):
    original_path, packet_root, original, _ = composition_fixture(tmp_path, monkeypatch)
    (Path(original['feedback_roots'][0]) / 'preflight.json').write_text(
        json.dumps({'case_id': 'report', 'stage': 'S4'}))
    with pytest.raises(ValueError, match='cannot_replace_valid'):
        m.compose_index(original_index=original_path, packet_root=packet_root, output_root=tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_composition_refuses_unverified_new_stage(tmp_path, monkeypatch):
    original_path, packet_root, _, _ = composition_fixture(tmp_path, monkeypatch)
    def unavailable(_):
        raise ValueError('approved_rejudge_execution_not_two_verified_one_shots')
    monkeypatch.setattr(m, '_verified_execution', unavailable)
    with pytest.raises(ValueError, match='not_two_verified'):
        m.compose_index(original_index=original_path, packet_root=packet_root, output_root=tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_actual_execution_reopen_allows_fresh_verification_uuid_only(tmp_path, monkeypatch):
    from uuid import uuid4
    root, binding = fixture(tmp_path, monkeypatch)
    source = m.commit(tmp_path / 'registry.json')
    binding.update(sources={'core-module': source}, producer=source,
                   feedback_adapter=source, judge_entrypoint=source)
    packet_path = root / 'packet.json'
    packet = m.read_document(packet_path)
    packet['execution_binding'] = binding.copy()
    packet_path.unlink()  # Synthetic fixture construction only.
    m.write_once(packet_path, {k: v for k, v in packet.items() if k != 'document_blake3'})
    def judge(path, **kwargs):
        m.write_private(path / 'judge-attempt.json', {'semantic_attempt_count': 1, 'retry_count': 0,
            'automatic_fallback': False, 'native_turn_timeout_seconds': 600})
        return {}
    def verify(path):
        return {'valid': True, 'status': 'scored', 'verification_id': str(uuid4()),
            'case_id': 'report', 'stage': path.name, 'score': {'reward_bps': 0},
            'judge_codex_receipt_blake3': path.name, 'round_identity': {'same': True}}
    monkeypatch.setattr(m, 'judge_once', judge)
    monkeypatch.setattr(m, 'verify_feedback', verify)
    m.execute_packet(root, execute=True)
    _, results = m._verified_execution(root)
    assert [r['stage'] for r in results] == ['S4', 'S5']
    def changed_score(path):
        return {**verify(path), 'score': {'reward_bps': 9999}}
    monkeypatch.setattr(m, 'verify_feedback', changed_score)
    with pytest.raises(ValueError, match='canonical_verification_differs'):
        m._verified_execution(root)
