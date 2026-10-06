"""Approved Report rows are separate, source-bound, metadata-only display."""
import json
from pathlib import Path

import pytest

from test_retained_eval_dashboard import observer_fixture
from test_live_training_dashboard import dashboard, write


POLICY = 'user-approved-report-s4s5-one-shot-v1'


def document(path, value):
    value = {key: item for key, item in value.items() if key != 'document_blake3'}
    value['document_blake3'] = dashboard._native_document_digest(value)
    write(path, value)
    return value


def reference(path):
    return dashboard._bounded_metadata_binding(path)[1]


def read(path):
    return json.loads(path.read_text())


def reseal_execution(packet):
    """Recommit synthetic outer metadata so tests reach the intended inner check."""
    claim = read(packet / 'execute-claim.json')
    claim['packet'] = reference(packet / 'packet.json')
    claim = document(packet / 'execute-claim.json', claim)
    for stage in ('S4', 'S5'):
        document(packet / stage / 'one-shot-claim.json', {
            'schema': 'eva.approved-report-stage-claim.v1', 'stage': stage,
            'packet_claim_blake3': claim['document_blake3'], 'retry_count': 0})
    document(packet / 'execution-result.json', {
        'schema': 'eva.approved-report-rejudge-execution.v1', 'policy': POLICY,
        'packet': reference(packet / 'packet.json'),
        'claim': reference(packet / 'execute-claim.json'),
        'stage_results': [reference(packet / stage / 'one-shot-result.json') for stage in ('S4', 'S5')],
        'judge_attempts_made': 2, 'retry_count': 0, 'actor_calls': 0, 'all_verified': True})


def fixture(tmp_path):
    loop, attempt, config, observer, progress = observer_fixture(tmp_path)
    run = Path(progress['benchmark_run_root'])
    packet = tmp_path / 'approved-report'
    write(run / 'track-run-manifest.json', {'fixture': True})
    write(tmp_path / 'registry.json', {'fixture': True})
    sources = {
        'actor_attempt': reference(run / 'track-rollouts/attempt.json'),
        'run_manifest': reference(run / 'track-run-manifest.json'),
        'checkpoint_identity': progress['checkpoint_identity'],
        'actor_runtime_binding': progress['actor_runtime_binding'],
        'registry': reference(tmp_path / 'registry.json')}
    stages = {}
    for stage in ('S4', 'S5'):
        target = packet / stage
        old = observer / 'stages/report' / stage / 'judge-attempt-isolation.json'
        document(old, {'status': 'unavailable', 'track': 'report', 'stage': stage,
                      'replacement_limit_exhausted': True, 'attempt_results': []})
        rubric = 'fixture-rubric-' + stage
        write(target / 'rollout.json', {'private_fixture': 'dashboard must never open'})
        write(target / 'receipt.json', {'private_fixture': 'dashboard must never open'})
        write(target / 'preflight.json', {'valid': True, 'case_id': 'report', 'stage': stage,
              'rubric_digest': rubric, 'source_rollout_blake3': 'a' * 64})
        stages[stage] = {'root': str(target), 'rubric_digest': rubric,
            'prepared': {name: reference(target / name) for name in ('preflight.json', 'rollout.json')},
            'source_turns': [{'receipt': reference(target / 'receipt.json')}],
            'original_unavailable': {'isolation': reference(old), 'failures': []}}
        score = {'schema': 'eva.rubric-score.v1', 'rubric_digest': rubric, 'reward_bps': 0}
        write(target / 'verification.json', {
            'schema': 'eva.automedbench-verified-stage-feedback.v1', 'valid': True,
            'status': 'scored', 'case_id': 'report', 'stage': stage, 'score': score,
            'rubric_digest': rubric, 'source_rollout_blake3': 'a' * 64,
            'checkpoint_identity_blake3': sources['checkpoint_identity']['blake3']})
        document(target / 'one-shot-result.json', {
            'schema': 'eva.approved-report-stage-result.v1', 'track': 'report', 'stage': stage,
            'root': str(target), 'status': 'verified', 'semantic_attempt_count': 1,
            'retry_count': 0, 'score': score, 'verification': reference(target / 'verification.json')})
        next(row for row in progress['stages'] if (row['track'], row['stage']) == ('report', stage))[
            'status'] = 'unavailable'
    document(packet / 'packet.json', {
        'schema': 'eva.approved-report-rejudge-packet.v1', 'policy': POLICY,
        'benchmark_run_root': str(run), 'source': sources, 'stages': stages,
        'permitted_pairs': [['report', 'S4'], ['report', 'S5']],
        'semantic_attempts_per_stage': 1, 'retry_count': 0, 'maximum_parallel_judges': 2,
        'native_turn_timeout_seconds': 600, 'actor_rerun': False,
        'canonical_rubric_changed': False, 'judge_calls': 0, 'provider_free_preparation': True})
    document(packet / 'execute-claim.json', {
        'schema': 'eva.approved-report-rejudge-claim.v1',
        'semantic_attempts_per_stage': 1, 'retry_count': 0})
    reseal_execution(packet)
    write(observer / 'progress.json', progress)
    document(observer / 'evaluation-index.json', {
        'schema': 'eva.rsi-evaluation-index.v1', 'benchmark_run_root': str(run),
        'actor_runtime_binding': sources['actor_runtime_binding'],
        'checkpoint_identity': sources['checkpoint_identity']['path'],
        'feedback_roots': [str(observer / 'stages/segmentation/S1')]})
    value = read(config)
    value['approved_report_rejudgments'] = [str(packet)]
    write(config, value)
    return loop, attempt, config, observer, packet


def test_verified_zero_rows_are_separate_and_never_open_private_payloads(tmp_path, monkeypatch):
    loop, _, _, observer, packet = fixture(tmp_path)
    old_table = dashboard.retained_eval_report_lines(loop, observer)
    before = {path: path.read_bytes() for path in observer.rglob('*.json')}
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.name not in {'receipt.json', 'rollout.json', 'trajectory.json',
                                 'assessment.json', 'grade.json', 'preflight.json'}
        assert not any(part in {'inputs', 'workspace', 'tokens', 'policy-context'} for part in path.parts)
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, 'open', metadata_only)
        text = '\n'.join(dashboard.approved_report_rejudgment_lines(loop, packet, retained_reports=[observer]))
        assert dashboard.retained_eval_report_lines(loop, observer) == old_table
    assert 'Report S4 0.00/100; S5 0.00/100; 2/2 verified records' in text
    assert 'Original index retains 1 grade roots + 2 approved supplemental rows' in text
    assert 'shown separately, not added twice' in text and 'no actor rerun' in text
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize('mismatch', ['loop', 'actor', 'checkpoint'])
def test_original_loop_actor_and_checkpoint_must_match(tmp_path, mismatch):
    loop, _, _, _, packet = fixture(tmp_path)
    if mismatch == 'loop':
        state = read(loop / 'state.json'); state['loop_id'] = 'another-loop'
        write(loop / 'state.json', state)
    elif mismatch == 'actor':
        value = read(packet / 'packet.json')
        actor_path = Path(value['source']['actor_attempt']['path'])
        actor = read(actor_path); actor['server_binding']['identity_file_blake3'] = 'b' * 64
        write(actor_path, actor)
        value['source']['actor_attempt'] = reference(actor_path)
        document(packet / 'packet.json', value)
        reseal_execution(packet)
    else:
        path = packet / 'S4/verification.json'
        proof = read(path); proof['checkpoint_identity_blake3'] = 'b' * 64
        write(path, proof)
        result = read(packet / 'S4/one-shot-result.json'); result['verification'] = reference(path)
        document(packet / 'S4/one-shot-result.json', result)
        reseal_execution(packet)
    text = '\n'.join(dashboard.approved_report_rejudgment_lines(loop, packet))
    assert 'pending or unavailable; not zero' in text and '0.00/100' not in text


@pytest.mark.parametrize('missing', ['verification_ref', 'execution', 'failed_stage'])
def test_incomplete_or_failed_execution_never_falls_back_to_numeric_zero(tmp_path, missing):
    loop, _, _, _, packet = fixture(tmp_path)
    if missing == 'execution':
        (packet / 'execution-result.json').unlink()  # Only the temporary fixture.
    else:
        path = packet / 'S4/one-shot-result.json'
        result = read(path)
        if missing == 'verification_ref':
            result['verification']['blake3'] = '0' * 64
        else:
            result['status'] = 'unavailable'  # Deliberately retain the stale numeric score.
        document(path, result)
        reseal_execution(packet)
    text = '\n'.join(dashboard.approved_report_rejudgment_lines(loop, packet))
    assert 'pending or unavailable; not zero' in text
    assert '0.00/100' not in text and '2/2 verified' not in text


def test_live_config_deduplicates_rows_and_preserves_original_table_and_controller_counts(tmp_path, monkeypatch):
    loop, _, config, observer, packet = fixture(tmp_path)
    progress = read(loop / 'state.json')
    monkeypatch.setattr(dashboard, 'rsi_current_status', lambda root: progress)
    monkeypatch.setattr(dashboard, 'publication_counts', lambda root: (0, 0, None))
    monkeypatch.setattr(dashboard, 'training_lines', lambda *args, **kwargs: [])
    monkeypatch.setattr(dashboard, 'memory_observation_footer', lambda *args: [])
    before = (loop / 'state.json').read_bytes()
    old_table = '\n'.join(dashboard.retained_eval_report_lines(loop, observer))
    value = read(config); value['approved_report_rejudgments'] = [str(packet), str(packet)]
    write(config, value)
    text = dashboard.render(config)
    assert text.count('Approved Report rejudgments —') == 1
    assert text.count('Report S4 0.00/100; S5 0.00/100') == 1
    assert old_table in text and '50/500' in text
    assert (loop / 'state.json').read_bytes() == before
    value.pop('approved_report_rejudgments'); write(config, value)
    assert 'Approved Report rejudgments' not in dashboard.render(config)
    value['approved_report_rejudgments'] = ['relative/packet']; write(config, value)
    with pytest.raises(ValueError):
        dashboard.live_configuration(config)
