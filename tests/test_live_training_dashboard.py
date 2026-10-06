"""Explicit sibling roots and bounded metadata; no source trajectories opened."""
import json
from pathlib import Path

import pytest

from test_training_validation_dashboard import dashboard
from test_baseline_score_dashboard import verification, write


def native_stage(root, bps=8333):
    target = verification(root, 'segmentation', 'S1', bps)
    for name in ('verification.json', 'feedback.json', 'preflight.json'):
        path = target / name
        value = json.loads(path.read_text())
        value['checkpoint_identity_blake3'] = None
        value['actor_identity_blake3'] = 'c' * 64
        if name != 'preflight.json':
            value['schema'] = 'eva.automedbench-native-verified-stage-feedback.v1'
        write(path, value)
    return target


def test_native_identity_is_explicit_and_null_checkpoint_is_not_zero_verified(tmp_path):
    stage = native_stage(tmp_path)
    assert dashboard._verified_process_score(stage, 'segmentation', 'S1') is None
    assert dashboard._verified_process_score(stage, 'segmentation', 'S1',
        native_actor=True, expected_identity='c' * 64) == 8333
    assert dashboard._verified_process_score(stage, 'segmentation', 'S1',
        native_actor=True, expected_identity='d' * 64) is None
    value = json.loads((stage / 'preflight.json').read_text())
    value['actor_identity_blake3'] = 'e' * 64
    write(stage / 'preflight.json', value)
    assert dashboard._verified_process_score(stage, 'segmentation', 'S1', native_actor=True) is None


def configured_evaluation(tmp_path, *, native):
    actor, feedback = tmp_path / 'actor', tmp_path / 'judge'
    if native:
        native_stage(feedback)
        binding = {'provider': 'eva_native_astra', 'requested_model': 'gpt-6-astra'}
        attempt = {'tracks': ['segmentation'], 'provider_binding': binding}
        launch = {'source_run': str(actor), 'actor_identity_blake3': 'c' * 64,
                  'checkpoint_identity': None, 'actor_identity': {'provider_binding': binding,
                                                                 'checkpoint_identity': None}}
    else:
        verification(feedback, 'classification', 'S1', 0)
        attempt = {'tracks': ['classification'], 'server_binding': {'identity_file_blake3': 'b' * 64}}
        launch = {'run_root': str(actor)}
    write(actor / 'track-rollouts/attempt.json', attempt)
    write(feedback / 'supervisor/launch.json', launch)
    return {'label': 'Native v2' if native else 'Qwen v4', 'actor_root': str(actor),
            'feedback_root': str(feedback), 'actor_kind': 'native_astra' if native else 'local_qwen'}


@pytest.mark.parametrize('native', [False, True])
def test_explicit_roots_bind_grade_to_the_actual_actor_not_an_older_attempt(tmp_path, native):
    row = configured_evaluation(tmp_path, native=native)
    text = '\n'.join(dashboard.live_evaluation_lines(row))
    assert '1/35 stages (NOT clinical scores)' in text
    assert ('83.33/100' if native else '0.00/100') in text
    launch_path = Path(row['feedback_root']) / 'supervisor/launch.json'
    launch = json.loads(launch_path.read_text())
    launch['source_run' if native else 'run_root'] = str(tmp_path / 'prior-attempt')
    write(launch_path, launch)
    text = '\n'.join(dashboard.live_evaluation_lines(row))
    assert 'binding not ready or differs (not zero)' in text
    assert '/35' not in text


def hbp(root, *, instant=False, earned=0):
    check = {'actor_status': 'completed', 'actor_receipt_blake3': 'a' * 64,
             'judge_receipt_blake3': 'b' * 64, 'earned_points': earned,
             'positive_possible_points': 15, 'successful_judge_read_paths': ['notes/final-answer.md'],
             'final_response_bytes': 50 if instant else 0, 'actual_actor_calls': 5}
    reopen = {'schema': 'eva.healthbench-supra-instant-independent-reopen.v1' if instant else
              'eva.native-healthbench-independent-reopen.v1', 'valid': True}
    reopen.update(check if instant else {'cohorts': {'weak': check}})
    write(root / 'independent-reopen.json', reopen)
    write(root / 'result.json', {'actors': {'weak': {'status': 'completed', 'receipt_blake3': 'a' * 64}},
          'judges': {'weak': {'status': 'completed', 'receipt_blake3': 'b' * 64,
                             'earned_points': earned, 'positive_possible_points': 15}}})


@pytest.mark.parametrize('instant', [False, True])
def test_real_zero_hbp_points_require_matching_independent_actor_and_judge_receipts(tmp_path, instant):
    hbp(tmp_path, instant=instant)
    text = '\n'.join(dashboard.healthbench_pilot_lines(tmp_path, 'HBP fixture'))
    assert 'points 0/15' in text and 'not a clinical leaderboard' in text
    result = json.loads((tmp_path / 'result.json').read_text())
    result['judges']['weak']['receipt_blake3'] = 'changed'
    write(tmp_path / 'result.json', result)
    text = '\n'.join(dashboard.healthbench_pilot_lines(tmp_path, 'HBP fixture'))
    assert 'unavailable (not zero)' in text and 'points 0/15' not in text


def test_live_config_reopens_only_bounded_metadata_and_preserves_published_vs_new_updates(tmp_path, monkeypatch):
    row = configured_evaluation(tmp_path / 'native', native=True)
    pilot = tmp_path / 'hbp'; hbp(pilot, instant=True)
    data = tmp_path / 'data'; loop = tmp_path / 'loop'
    write(loop / 'stop-request.json', {'held': True})
    write(data / 'runs/eva-hf-release-astra-20260910.v3/publish-receipt.json', {
        'remote_readback_verified': True, 'all_repositories_private': True, 'repositories': [
        {'repo_id': 'operator/EVA-Med-SFT-data', 'local_row_count': 21553},
        {'repo_id': 'operator/EVA-Med-RL-data', 'local_row_count': 6000}]})
    config = tmp_path / 'live.json'
    write(config, {'schema': 'eva.live-dashboard-config.v1', 'data_root': str(data), 'rsi_root': str(loop),
        'evaluations': [row], 'healthbench': [{'label': 'Instant HBP', 'run_root': str(pilot)}]})
    original = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.name in {'live.json', 'attempt.json', 'summary.json', 'baseline-process.json',
            'baseline-process-exit.json', 'baseline-controlled-stop.json', 'launch.json',
            'verification.json', 'feedback.json', 'preflight.json', 'result.json', 'independent-reopen.json',
            'publish-receipt.json', 'progress.json', 'score.json', 'failure.json'}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', metadata_only)
    text = dashboard.render(config)
    assert 'RL 6000; SFT 21553' in text and '0/500; HELD' in text
    assert '1/35 stages' in text and 'points 0/15' in text
    assert 'Historical' not in text


def active_rsi_fixture(tmp_path):
    loop = tmp_path / 'loop'
    attempt_id = 'f55bcec2-6721-4127-9b80-3c72b2e68d38'
    attempt = loop / 'attempts' / attempt_id
    progress = {'loop_id': 'current-loop', 'active_attempt': attempt_id, 'round': 1,
        'phase': 'train', 'status': 'running', 'checkpoint_backed_updates': 2,
        'checkpoint_backed_learning_updates': 1, 'completed_rounds': 0}
    write(loop / 'state.json', {**progress, 'attempts': [
        {'attempt_id': attempt_id, 'root': str(attempt), 'phase': 'train'}]})
    write(loop / 'progress.json', progress)
    write(attempt / 'training/run-receipt.json', {'status': 'running', 'argv': [
        'train.py', '--num-rollout', '48', '--global-batch-size', '4']})
    for index, summary in enumerate(({'groups': [{'zero_variance_group': True}]},
            {'zero_variance_group': True}, {'groups': [{'zero_variance_group': False}]})):
        write(attempt / 'training/rollouts' / f'{index:06d}' / 'summary.json', summary)
    write(attempt / 'training/rollouts/000048/summary.json', {'zero_variance_group': True})
    write(loop / 'attempts/old/training/rollouts/000000/summary.json', {'zero_variance_group': True})
    config = tmp_path / 'live.json'
    write(config, {'schema': 'eva.live-dashboard-config.v1', 'data_root': str(tmp_path / 'data'),
                  'rsi_root': str(loop)})
    return loop, attempt, progress, config


def current_eval_fixture(tmp_path):
    loop, attempt, progress, config = active_rsi_fixture(tmp_path)
    progress.update(phase='evaluation', checkpoint_backed_updates=50, checkpoint_backed_learning_updates=28)
    write(loop / 'state.json', {**progress, 'attempts': [
        {'attempt_id': progress['active_attempt'], 'root': str(attempt), 'phase': 'evaluation'}]})
    model = str(tmp_path / 'saved-hf')
    write(attempt / 'context.json', {'attempt_root': str(attempt), 'loop_id': progress['loop_id'],
        'round': 1, 'phase': 'evaluation', 'model_path': model})
    identity_path = attempt / 'serving/checkpoint-identity.json'
    write(identity_path, {'exact_final_model_path': model})
    _, identity_ref = dashboard._bounded_metadata_binding(identity_path)
    run = attempt / 'benchmark/34fda993-4669-4083-b26a-16ee36a3af5b'
    write(attempt / 'evaluation-actor-binding.json', {'schema': 'eva.rsi-evaluation-actor-binding.v1',
        'model_path': model, 'checkpoint_identity': identity_ref, 'codex_version': 'codex-cli 0.153.4',
        'equivalent_actor_cli': ['track_entry.py', 'run', '--run-root', str(run)]})
    write(run / 'track-rollouts/attempt.json', {'tracks': list(dashboard.TRACKS),
        'server_binding': {'identity_file_blake3': identity_ref['blake3']}})
    write(run / 'track-rollouts/segmentation/turns/01-planning/receipt.json', {'private': 'must not read'})
    outcome = {'schema': 'eva.automedbench-track-budget-outcome.v1', 'actual_turn_count': 0}
    outcome['document_blake3'] = dashboard._native_document_digest(outcome)
    write(run / 'track-rollouts/classification/track-budget-outcome.json', outcome)
    return loop, attempt, progress, config, run, identity_ref


def test_default_live_current_evaluation_precedes_history_without_payload_reads(tmp_path, monkeypatch):
    loop, attempt, progress, config, run, _ = current_eval_fixture(tmp_path)
    value = json.loads(config.read_text())
    value['evaluations'] = [configured_evaluation(tmp_path / 'historical', native=False)]
    write(config, value)
    monkeypatch.setattr(dashboard, 'rsi_current_status', lambda root: progress)
    monkeypatch.setattr(dashboard, 'publication_counts', lambda root: (0, 0, None))
    monkeypatch.setattr(dashboard, 'training_lines', lambda *args, **kwargs: [])
    monkeypatch.setattr(dashboard, 'memory_observation_footer', lambda *args: [])
    original = Path.open
    def metadata_only(path, *args, **kwargs):
        assert path.name not in {'receipt.json', 'trajectory.json', 'grade.json', 'assessment.json', 'rollout.json'}
        assert not any(part in {'inputs', 'workspace'} for part in path.parts)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', metadata_only)
    text = dashboard.render(config)
    assert text.index('Current RSI round 1 seven-track evaluation') < text.index('Live Qwen v4')
    assert str(run) in text and '50/500' in text and '1/5' in text
    assert 'unavailable before first recorded turn (not a task/rubric zero)' in text
    assert 'Current evaluation scores pending/unavailable; no historical grades substituted.' in text
    assert 'context failure' not in text  # No invented startup cause.


@pytest.mark.parametrize('mismatch', ['state', 'checkpoint', 'run_root'])
def test_current_evaluation_rejects_stale_attempt_or_checkpoint_or_outside_run(tmp_path, mismatch):
    loop, attempt, progress, _, run, _ = current_eval_fixture(tmp_path)
    if mismatch == 'state':
        value = json.loads((loop / 'state.json').read_text()); value['active_attempt'] = 'different'
        write(loop / 'state.json', value)
    elif mismatch == 'checkpoint':
        write(attempt / 'serving/checkpoint-identity.json', {'exact_final_model_path': 'wrong'})
    else:
        value = json.loads((attempt / 'evaluation-actor-binding.json').read_text())
        value['equivalent_actor_cli'][-1] = str(tmp_path / 'older-run')
        write(attempt / 'evaluation-actor-binding.json', value)
    text = '\n'.join(dashboard._active_rsi_evaluation_lines(loop, progress))
    assert 'source binding pending/unavailable' in text and 'phase receipts' not in text


def test_current_report_requires_exact_current_index_runtime_and_checkpoint(tmp_path):
    loop, attempt, progress, _, run, identity_ref = current_eval_fixture(tmp_path)
    _, binding_ref = dashboard._bounded_metadata_binding(attempt / 'evaluation-actor-binding.json')
    write(attempt / 'evaluation-index.json', {'schema': 'eva.rsi-evaluation-index.v2',
        'benchmark_run_root': str(run), 'actor_runtime_binding': binding_ref,
        'checkpoint_identity': identity_ref['path']})
    _, index_ref = dashboard._bounded_metadata_binding(attempt / 'evaluation-index.json')
    report = {'schema': 'eva.automedbench-seven-track-report.v1', 'benchmark_run_root': str(run),
        'evaluation_index': index_ref, 'checkpoint_identity': identity_ref,
        'tracks': [{'track': track, 'A': 0 if track == 'classification' else None, 'T': None,
                    'A_stage_coverage': 1 if track == 'classification' else 0} for track in dashboard.TRACKS]}
    write(attempt / 'seven-track-report.json', report)
    text = '\n'.join(dashboard._active_rsi_evaluation_lines(loop, progress))
    assert 'classification: A 0.00/100 (1/5 provisional); T N/A' in text
    assert 'Current source-bound report' in text and 'not a Judge-issued score' in text
    report['evaluation_index']['blake3'] = 'wrong-old-index'
    write(attempt / 'seven-track-report.json', report)
    text = '\n'.join(dashboard._active_rsi_evaluation_lines(loop, progress))
    assert 'scores pending/unavailable' in text and 'A 0.00/100' not in text


def test_live_render_shows_only_bound_active_summary_aggregate(tmp_path, monkeypatch):
    loop, attempt, progress, config = active_rsi_fixture(tmp_path)
    monkeypatch.setattr(dashboard, 'rsi_current_status', lambda root: progress)
    monkeypatch.setattr(dashboard, 'publication_counts', lambda root: (0, 0, None))
    monkeypatch.setattr(dashboard, 'training_lines', lambda *args, **kwargs: [])
    monkeypatch.setattr(dashboard, 'memory_observation_footer', lambda *args: [])
    original = Path.open

    def summary_metadata_only(path, *args, **kwargs):
        assert path.name in {'live.json', 'state.json', 'run-receipt.json', 'summary.json'}
        assert 'old' not in path.parts and '000048' not in path.parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', summary_metadata_only)
    text = dashboard.render(config)
    assert 'Active training reported zero-variance prompt groups: 2/3' in text
    assert text.count('zero-variance prompt groups:') == 1
    assert 'summary metadata only, not independent reward verification' in text
    assert 'group 000000:' not in text and 'grade.json files' not in text
    assert '2/500' in text


def test_active_variance_never_uses_stale_progress_or_another_attempt(tmp_path, monkeypatch):
    loop, attempt, progress, _ = active_rsi_fixture(tmp_path)
    monkeypatch.setattr(dashboard, 'rsi_current_status', lambda root: None)
    assert 'zero-variance' not in '\n'.join(dashboard.rsi_lines(loop))
    monkeypatch.setattr(dashboard, 'rsi_current_status', lambda root: progress)
    state = json.loads((loop / 'state.json').read_text())
    state['active_attempt'] = '00000000-0000-4000-8000-000000000000'
    write(loop / 'state.json', state)
    text = '\n'.join(dashboard.rsi_lines(loop))
    assert 'summary unavailable; not inferred as zero' in text and 'groups: 2/3' not in text
    state['active_attempt'] = progress['active_attempt']
    write(loop / 'state.json', state)
    (attempt / 'training/run-receipt.json').write_text('{')
    text = '\n'.join(dashboard.rsi_lines(loop))
    assert 'summary unavailable; not inferred as zero' in text and 'groups: 2/3' not in text


def test_malformed_inflight_config_or_hbp_never_falls_back_to_prior_run(tmp_path):
    config = tmp_path / 'live.json'; config.write_text('{')
    with pytest.raises(ValueError, match='config unavailable'):
        dashboard.render(config)
    hbp(tmp_path, instant=True)
    (tmp_path / 'result.json').write_text('{')
    assert 'points 0/15' not in '\n'.join(dashboard.healthbench_pilot_lines(tmp_path, 'HBP'))
    write(config, {'schema': 'eva.live-dashboard-config.v1', 'data_root': 'relative', 'rsi_root': '/loop'})
    with pytest.raises(ValueError, match='absolute roots'):
        dashboard.live_configuration(config)


def test_primary_live_config_is_default_but_explicit_inventory_has_precedence(tmp_path, monkeypatch):
    default = tmp_path / '.codex/live-dashboard.json'
    explicit = tmp_path / 'operator-selected.json'
    monkeypatch.setattr(dashboard, 'ROOT', tmp_path)
    assert dashboard.selected_live_config(None) is None
    write(default, {'schema': 'eva.live-dashboard-config.v1'})
    assert dashboard.selected_live_config(None) == default
    assert dashboard.selected_live_config(explicit) == explicit


def test_original_task_score_is_bound_and_separate_from_stage_scores(tmp_path):
    root = tmp_path / 'native-scores' / 'classification'
    score = {'schema': 'eva.automedbench-native-whole-track-score.v1', 'run_id': tmp_path.name,
        'track': 'classification', 'native_result': {'track': 'classification', 'status': 'scored',
        'task_score_0_1': 0.8687, 'native_facts': {'output_format_valid': True}}}
    write(root / 'score.json', score)
    text = '\n'.join(dashboard.native_task_score_lines(tmp_path))
    assert '1/7 numeric outcomes' in text and 'balanced_accuracy=0.8687' in text
    assert 'NOT stage grades' in text and 'vqa: --' in text
    score['run_id'] = 'old-run'
    write(root / 'score.json', score)
    assert '0/7 numeric outcomes' in '\n'.join(dashboard.native_task_score_lines(tmp_path))


def test_invalid_output_zero_is_not_confused_with_scorer_failure(tmp_path):
    write(tmp_path / 'native-scores/detection/score.json', {
        'schema': 'eva.automedbench-native-whole-track-score.v1', 'run_id': tmp_path.name,
        'track': 'detection', 'native_result': {'track': 'detection', 'status': 'scored',
        'task_score_0_1': 0, 'native_facts': {'output_format_valid': False}}})
    write(tmp_path / 'native-scores/enhancement/failure.json', {
        'status': 'failed', 'track': 'enhancement', 'error_code': 'native_track_worker_failed', 'reward': None})
    text = '\n'.join(dashboard.native_task_score_lines(tmp_path))
    assert 'detection: task_score=0.0000; invalid submission format; raw metric unavailable' in text
    assert 'enhancement: -- (native_track_worker_failed; not zero)' in text


def test_explicit_separate_stage_evidence_is_counted_without_reading_trajectory(tmp_path):
    row = configured_evaluation(tmp_path, native=False)
    extra = verification(tmp_path / 'separate', 'classification', 'S2', 1667)
    row['additional_stage_feedback'] = [{'track': 'classification', 'stage': 'S2',
        'root': str(extra), 'source_rollout_blake3': 'a' * 64}]
    text = '\n'.join(dashboard.live_evaluation_lines(row))
    assert '2/35 stages' in text and 'S1 0.00/100; S2 16.67/100' in text
    row['additional_stage_feedback'][0]['source_rollout_blake3'] = 'wrong' * 16
    assert '1/35 stages' in '\n'.join(dashboard.live_evaluation_lines(row))


def test_separate_stage_does_not_choose_best_of_two_attempts(tmp_path):
    row = configured_evaluation(tmp_path, native=False)
    extra = verification(tmp_path / 'separate', 'classification', 'S1', 10000)
    row['additional_stage_feedback'] = [{'track': 'classification', 'stage': 'S1',
        'root': str(extra), 'source_rollout_blake3': 'a' * 64}]
    text = '\n'.join(dashboard.live_evaluation_lines(row))
    assert '0/35 stages' in text and 'classification: S1 --' in text
    assert '100.00/100' not in text


def test_live_config_rejects_duplicate_separate_stage_references(tmp_path):
    row = configured_evaluation(tmp_path, native=False)
    extra = {'track': 'classification', 'stage': 'S2', 'root': str(tmp_path / 'separate'),
             'source_rollout_blake3': 'a' * 64}
    row['additional_stage_feedback'] = [extra, extra]
    path = tmp_path / 'live.json'
    write(path, {'schema': 'eva.live-dashboard-config.v1', 'data_root': str(tmp_path),
                 'rsi_root': str(tmp_path / 'loop'), 'evaluations': [row]})
    with pytest.raises(ValueError, match='duplicate additional'):
        dashboard.live_configuration(path)


def supplementary_fixture(tmp_path):
    from training.automedbench_lite.adapter import write_once
    actor, score_root = tmp_path / 'actor', tmp_path / 'supplement-vqa'
    (actor / 'track-rollouts/vqa').mkdir(parents=True)
    score_root.mkdir()
    rollout = write_once(actor / 'track-rollouts/vqa/rollout.json', {
        'run_id': actor.name, 'track': 'vqa', 'completed_requested_turns': True, 'errors': [],
        'fixture_note': '真实格式'})
    score = {'schema': 'eva.automedbench-native-whole-track-score.v1', 'run_id': actor.name,
        'track': 'vqa', 'actor_document_blake3': rollout['document_blake3'],
        'native_result': {'track': 'vqa', 'status': 'scored', 'task_score_0_1': .0015,
            'native_facts': {'valid_outputs': 15, 'expected_outputs': 2005, 'submission_format_valid': False}}}
    score = write_once(score_root / 'score.json', score)
    return actor, {'track': 'vqa', 'root': str(score_root), 'document_blake3': score['document_blake3']}


def test_supplementary_native_score_preserves_original_failure_and_reports_partial_coverage(tmp_path):
    actor, reference = supplementary_fixture(tmp_path)
    failure = actor / 'native-scores/vqa/failure.json'
    write(failure, {'error_code': 'file_topology_or_size_invalid'})
    prior = failure.read_bytes()
    text = '\n'.join(dashboard.native_task_score_lines(actor) +
                     dashboard.supplementary_native_score_lines(actor, [reference]))
    assert 'vqa: -- (file_topology_or_size_invalid; not zero)' in text
    assert 'vqa: task_score=0.0015; valid outputs 15/2005' in text
    assert 'SAME retained actor' in text and failure.read_bytes() == prior


@pytest.mark.parametrize('target', ['score', 'actor', 'reference'])
def test_supplementary_score_changed_binding_is_unavailable_not_zero(tmp_path, target):
    actor, reference = supplementary_fixture(tmp_path)
    if target == 'reference':
        reference['document_blake3'] = 'a' * 64
    else:
        path = (Path(reference['root']) / 'score.json' if target == 'score'
                else actor / 'track-rollouts/vqa/rollout.json')
        value = json.loads(path.read_text())
        value['run_id'] = 'different-actor'
        value['document_blake3'] = dashboard._native_document_digest(value)
        write(path, value)
    text = '\n'.join(dashboard.supplementary_native_score_lines(actor, [reference]))
    assert 'binding unavailable; not zero' in text and 'task_score=' not in text


def test_live_config_rejects_duplicate_supplementary_scores(tmp_path):
    row = configured_evaluation(tmp_path / 'native', native=True)
    _, reference = supplementary_fixture(tmp_path)
    row['supplementary_native_scores'] = [reference, reference]
    path = tmp_path / 'live.json'
    write(path, {'schema': 'eva.live-dashboard-config.v1', 'data_root': str(tmp_path),
                 'rsi_root': str(tmp_path / 'loop'), 'evaluations': [row]})
    with pytest.raises(ValueError, match='duplicate supplementary'):
        dashboard.live_configuration(path)
