"""Supplementary scores stay bound to the active run, not training admission."""
import json
from pathlib import Path

import pytest

from test_live_training_dashboard import current_eval_fixture, dashboard, write


def observer_fixture(tmp_path):
    loop, attempt, progress, config, run, identity_ref = current_eval_fixture(tmp_path)
    output = tmp_path / 'observer'
    _, binding_ref = dashboard._bounded_metadata_binding(attempt / 'evaluation-actor-binding.json')
    verification = output / 'stages/segmentation/S1/verification.json'
    write(verification, {'valid': True, 'status': 'scored', 'case_id': 'segmentation', 'stage': 'S1',
        'checkpoint_identity_blake3': identity_ref['blake3'], 'score': {'reward_bps': 0}})
    _, proof_ref = dashboard._bounded_metadata_binding(verification)
    rows = [{'track': track, 'stage': stage, 'status': 'pending'}
            for track in dashboard.TRACKS for stage in dashboard.STAGES]
    next(row for row in rows if (row['track'], row['stage']) == ('segmentation', 'S1')).update(
        status='verified', score_0_100=0, verification=proof_ref)
    next(row for row in rows if (row['track'], row['stage']) == ('detection', 'S1')).update(status='unavailable')
    value = {'schema': 'eva.retained-eval-report-progress.v1', 'source_attempt_root': str(attempt),
        'benchmark_run_root': str(run), 'actor_runtime_binding': binding_ref,
        'checkpoint_identity': identity_ref, 'diagnostic_only': True, 'training_admission': False,
        'stages': rows}
    write(output / 'progress.json', value)
    config_value = json.loads(config.read_text())
    config_value['retained_eval_reports'] = [str(output)]
    write(config, config_value)
    return loop, attempt, config, output, value


def test_current_supplementary_zero_is_numeric_but_missing_is_not(tmp_path, monkeypatch):
    loop, _, config, output, _ = observer_fixture(tmp_path)
    original = Path.open
    def small_metadata_only(path, *args, **kwargs):
        assert path.name in {'state.json', 'progress.json', 'evaluation-actor-binding.json',
                             'checkpoint-identity.json', 'verification.json', 'live.json'}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', small_metadata_only)
    text = '\n'.join(dashboard.retained_eval_report_lines(loop, output))
    assert 'Verified 1/35; running 0; pending 33; unavailable 1 (not zero)' in text
    assert 'NOT training admission' in text and '0.00' in text and '—' in text
    assert dashboard.live_configuration(config)['retained_eval_reports'] == [str(output)]


@pytest.mark.parametrize('tamper', ['source', 'checkpoint', 'numeric_unavailable', 'score', 'proof', 'duplicate', 'admission'])
def test_supplementary_display_rejects_unbound_or_unproved_values(tmp_path, tamper):
    loop, _, _, output, value = observer_fixture(tmp_path)
    verified = next(row for row in value['stages'] if row['status'] == 'verified')
    if tamper == 'source': value['source_attempt_root'] += '-other'
    elif tamper == 'checkpoint': value['checkpoint_identity']['blake3'] = '0' * 64
    elif tamper == 'numeric_unavailable': value['stages'][0]['score_0_100'] = 0
    elif tamper == 'score': verified['score_0_100'] = 100
    elif tamper == 'proof': verified['verification']['blake3'] = '0' * 64
    elif tamper == 'duplicate': value['stages'][-1] = value['stages'][0]
    else: value['training_admission'] = True
    write(output / 'progress.json', value)
    text = '\n'.join(dashboard.retained_eval_report_lines(loop, output))
    assert 'pending/unavailable' in text and 'Verified 1/35' not in text


def test_other_round_does_not_reuse_supplementary_score(tmp_path):
    loop, _, _, output, _ = observer_fixture(tmp_path)
    value = json.loads((loop / 'state.json').read_text())
    value['phase'] = 'train'
    write(loop / 'state.json', value)
    assert 'pending/unavailable' in '\n'.join(dashboard.retained_eval_report_lines(loop, output))
