"""External C/D supplement progress is metadata-only, never controller credit."""
import json
from pathlib import Path

import pytest

from test_live_training_dashboard import current_eval_fixture, dashboard, write


def native(path, value):
    value = {**value, 'document_blake3': dashboard._native_document_digest(value)}
    write(path, value)
    return value


def fixture(tmp_path, *, running=False):
    loop, original, progress, config, _, _ = current_eval_fixture(tmp_path)
    packet = tmp_path / 'packet'
    attempt = packet / 'attempt'
    context = {**json.loads((original / 'context.json').read_text()), 'attempt_root': str(attempt)}
    settings = {'allow_classification_detection_supplement': True, 'supplemental_source_attempt': str(original)}
    write(packet / 'context.json', context)
    write(packet / 'settings.eval.json', settings)
    refs = [dashboard._bounded_metadata_binding(packet / name)[1] for name in ('context.json', 'settings.eval.json')]
    write(packet / 'watcher-launch.json', {'schema': 'eva.rsi-supplement-watcher-launch.v1',
        'inputs': refs, 'automatic_actor_retries': False, 'controller_state_modified': False,
        'initial_preflight': {'original_source_attempt': str(original), 'output_root': str(attempt),
            'tracks_requested': ['classification', 'detection']}})
    write(packet / 'watcher-progress.json', {'schema': 'eva.rsi-supplement-watcher-progress.v1',
        'attempt_root': str(attempt), 'status': 'running_two_fresh_tracks' if running else 'waiting_for_original_cleanup'})
    value = json.loads(config.read_text()); value['supplemental_eval_packets'] = [str(packet)]; write(config, value)
    run = attempt / 'benchmark/new-run'
    if running:
        identity_path = attempt / 'serving/checkpoint-identity.json'
        write(identity_path, {'exact_final_model_path': context['model_path'], 'fresh_identity': True})
        identity_ref = dashboard._bounded_metadata_binding(identity_path)[1]
        write(attempt / 'evaluation-actor-binding.json', {'schema': 'eva.rsi-evaluation-actor-binding.v1',
            'model_path': context['model_path'], 'checkpoint_identity': identity_ref,
            'codex_version': 'codex-cli 0.153.4', 'equivalent_actor_cli': ['run', '--run-root', str(run)]})
        write(run / 'track-rollouts/attempt.json', {'tracks': ['classification', 'detection'],
            'planned_coding_rollouts': 2, 'full_seven_track_evaluation': False,
            'server_binding': {'identity_file_blake3': identity_ref['blake3']}})
        for track in ('classification', 'detection'):
            audit = run / 'track-rollouts' / track
            budget = native(audit / 'track-budget.json', {'schema': 'eva.automedbench-track-budget.v1',
                'policy': 'admitted-track-wallclock-3600-v1', 'timeout_seconds': 3600, 'admitted_at_ns': 1000000000})
            if track == 'classification':
                write(audit / 'turns/01-planning/receipt.json', {'private_fixture': 'must not read'})
            else:
                native(audit / 'track-budget-outcome.json', {'schema': 'eva.automedbench-track-budget-outcome.v1',
                    'budget_document_blake3': budget['document_blake3'], 'timeout_seconds': 3600,
                    'elapsed_seconds': 0.7, 'actual_turn_count': 0})
        target = attempt / 'workspace-feedback/classification/S1'
        grade = target / 'feedback/S1/attempts/new-judge'
        verification = grade / 'verification.json'
        write(verification, {'valid': True, 'status': 'scored', 'case_id': 'classification', 'stage': 'S1',
            'checkpoint_identity_blake3': identity_ref['blake3'], 'score': {'reward_bps': 0}})
        native(target / 'result.json', {'status': 'verified', 'track': 'classification', 'stage': 'S1',
            'feedback_root': str(grade), 'score': 0, 'verification': dashboard._bounded_metadata_binding(verification)[1]})
    return loop, original, progress, config, packet, attempt, run


def test_waiting_supplement_remains_visible_when_original_loop_blocks(tmp_path):
    loop, _, _, _, packet, _, _ = fixture(tmp_path)
    state = json.loads((loop / 'state.json').read_text())
    state.update(status='blocked', active_attempt=None)
    write(loop / 'state.json', state)
    text = '\n'.join(dashboard.supplemental_eval_packet_lines(loop, packet))
    assert 'waiting for original cleanup' in text and 'separate from controller updates' in text
    assert 'classification: waiting/not started; A pending (not zero)' in text
    assert 'detection: waiting/not started; A pending (not zero)' in text
    assert '0.00' not in text


def test_running_terminal_elapsed_and_verified_zero_without_payload_reads(tmp_path, monkeypatch):
    loop, _, _, _, packet, _, _ = fixture(tmp_path, running=True)
    monkeypatch.setattr(dashboard.time, 'time_ns', lambda: 121000000000)
    original_open = Path.open
    def metadata_only(path, *args, **kwargs):
        assert path.name not in {'receipt.json', 'rollout.json', 'grade.json', 'assessment.json', 'trajectory.json'}
        assert not any(part in {'inputs', 'workspace', 'tokens'} for part in path.parts)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', metadata_only)
    text = '\n'.join(dashboard.supplemental_eval_packet_lines(loop, packet))
    assert 'classification: running; phase receipts 1/5; elapsed 120.0/3600s (live wall clock)' in text
    assert 'A graded 1/5: S1 0.00; S2 —' in text
    assert 'detection: unavailable before first recorded turn; phase receipts 0/5; elapsed 0.7/3600s' in text
    assert 'A graded 0/5: S1 —' in text and 'no Judge rerun or checkpoint credit' in text


@pytest.mark.parametrize('mismatch', ['round', 'context', 'checkpoint', 'score'])
def test_stale_or_tampered_metadata_does_not_show_grade(tmp_path, mismatch):
    loop, _, _, _, packet, attempt, _ = fixture(tmp_path, running=True)
    if mismatch == 'round':
        path = loop / 'state.json'; value = json.loads(path.read_text()); value['round'] = 2
    elif mismatch == 'context':
        path = packet / 'context.json'; value = json.loads(path.read_text()); value['model_path'] = 'wrong'
    elif mismatch == 'checkpoint':
        path = attempt / 'serving/checkpoint-identity.json'; value = {'exact_final_model_path': 'wrong'}
    else:
        path = attempt / 'workspace-feedback/classification/S1/result.json'
        value = json.loads(path.read_text()); value['verification']['blake3'] = '0' * 64
    write(path, value)
    text = '\n'.join(dashboard.supplemental_eval_packet_lines(loop, packet))
    assert 'S1 0.00' not in text and 'A graded 1/5' not in text


def test_live_render_adds_supplement_without_changing_loop_counts(tmp_path, monkeypatch):
    loop, _, progress, config, _, _, _ = fixture(tmp_path)
    monkeypatch.setattr(dashboard, 'rsi_current_status', lambda root: progress)
    monkeypatch.setattr(dashboard, 'publication_counts', lambda root: (0, 0, None))
    monkeypatch.setattr(dashboard, 'training_lines', lambda *args, **kwargs: [])
    monkeypatch.setattr(dashboard, 'memory_observation_footer', lambda *args: [])
    before = (loop / 'state.json').read_bytes()
    text = dashboard.render(config)
    assert '50/500' in text and 'C/D supplemental evaluation, round 1' in text
    assert 'waiting for original cleanup' in text
    assert (loop / 'state.json').read_bytes() == before
    value = json.loads(config.read_text()); value.pop('supplemental_eval_packets'); write(config, value)
    assert 'C/D supplemental evaluation' not in dashboard.render(config)


def test_supplement_packet_config_requires_bounded_absolute_list(tmp_path):
    _, _, _, config, _, _, _ = fixture(tmp_path)
    value = json.loads(config.read_text())
    value['supplemental_eval_packets'] = ['relative/path']; write(config, value)
    with pytest.raises(ValueError):
        dashboard.live_configuration(config)
    value['supplemental_eval_packets'] = ['/tmp/fixture'] * 9; write(config, value)
    with pytest.raises(ValueError):
        dashboard.live_configuration(config)
