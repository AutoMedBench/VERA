"""Bounded public metadata only; never open policy traces or model data."""
import json
from pathlib import Path
from uuid import uuid4

from test_training_validation_dashboard import dashboard


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def verification(root, track, stage, bps):
    target = root / track / stage
    common = {'case_id': track, 'domain': 'automedbench-' + track, 'stage': stage,
        'source_rollout_blake3': 'a' * 64, 'checkpoint_identity_blake3': 'b' * 64, 'rubric_digest': 'fixture-rubric'}
    verified = {**common, 'schema': 'eva.automedbench-verified-stage-feedback.v1', 'valid': True,
        'status': 'scored', 'score': {'schema': 'eva.rubric-score.v1', 'reward_bps': bps, 'rubric_digest': 'fixture-rubric'},
        'actual_workspace_reads': 10, 'native_clinical_metric': None}
    write(target / 'verification.json', verified)
    write(target / 'feedback.json', verified)
    write(target / 'preflight.json', {**common, 'valid': True, 'judge_eligible': True})
    return target


def test_verified_zero_and_missing_are_distinct_and_no_private_files_read(tmp_path, monkeypatch):
    verification(tmp_path, 'detection', 'S1', 6667)
    verification(tmp_path, 'report', 'S1', 0)
    verification(tmp_path, 'report', 'S2', 0)
    write(tmp_path / 'report/S3/grade.json', {'reward_bps': 10000})
    write(tmp_path / 'report/S3/feedback.json', {'valid': True, 'score': {'reward_bps': 10000}})
    write(tmp_path / 'prior-attempt/report/S1/verification.json', {'reward_bps': 10000})
    original = Path.open
    def metadata_only(path, *args, **kwargs):
        assert path.name in {'verification.json', 'feedback.json', 'preflight.json'}
        assert 'prior-attempt' not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', metadata_only)
    text = '\n'.join(dashboard.process_feedback_lines(tmp_path))
    assert '3/35 stages (NOT clinical scores)' in text
    assert 'detection: S1 66.67/100' in text
    assert 'report: S1 0.00/100; S2 0.00/100; S3 --' in text
    assert 'classification: S1 --' in text and '100.00/100' not in text
    assert 'does not rerun the Judge or evidence verifier' in text


def test_partial_or_disagreeing_verification_does_not_display_score(tmp_path):
    target = verification(tmp_path, 'report', 'S1', 0)
    (target / 'verification.json').write_text('{')
    assert dashboard._verified_process_score(target, 'report', 'S1') is None
    target = verification(tmp_path, 'report', 'S1', 0)
    value = json.loads((target / 'feedback.json').read_text())
    value['source_rollout_blake3'] = 'c' * 64
    write(target / 'feedback.json', value)
    assert dashboard._verified_process_score(target, 'report', 'S1') is None
    target = verification(tmp_path, 'report', 'S1', True)
    assert dashboard._verified_process_score(target, 'report', 'S1') is None


def test_new_baseline_score_label_is_separate_from_prior_evidence(tmp_path):
    verification(tmp_path / 'v1', 'detection', 'S1', 6667)
    verification(tmp_path / 'v2', 'detection', 'S1', 0)
    lines = '\n'.join(dashboard.process_feedback_lines(tmp_path / 'v2', label='Fresh baseline v2'))
    assert 'Fresh baseline v2 independently verified' in lines
    assert 'detection: S1 0.00/100' in lines and '66.67/100' not in lines


def test_terminal_zero_of_seven_is_workflow_count_not_clinical_zero(tmp_path):
    write(tmp_path / 'track-rollouts/attempt.json', {'tracks': list(dashboard.TRACKS)})
    write(tmp_path / 'track-rollouts/summary.json', {'tracks': [
        {'track': track, 'completed_requested_turns': False, 'errors': [{'phase': 'S2'}]} for track in dashboard.TRACKS]})
    text = '\n'.join(dashboard.track_eval_lines(tmp_path))
    assert 'Baseline terminal: complete workflows 0/7; failed/incomplete 7/7' in text
    assert 'Not a clinical score' in text
    write(tmp_path / 'stop-request.json', {'reason': 'Actual evaluation first'})
    assert '0/500; HELD' in '\n'.join(dashboard.rsi_lines(tmp_path))


def test_future_baseline_never_borrows_previous_attempt_counts(tmp_path):
    parent = tmp_path / 'v2'
    assert dashboard.fresh_baseline_lines(parent) == ['Fresh seven-track baseline v2: not started.']
    parent.mkdir()
    assert 'awaiting run metadata' in '\n'.join(dashboard.fresh_baseline_lines(parent))
    attempt = parent / str(uuid4())
    write(attempt / 'track-run-manifest.json', {'fixture': True})
    write(attempt / 'track-rollouts/attempt.json', {'tracks': ['report']})
    (attempt / 'track-rollouts/summary.json').write_text('{')
    text = '\n'.join(dashboard.fresh_baseline_lines(parent))
    assert attempt.name in text and '0/5' in text and 'terminal:' not in text
    write(parent / str(uuid4()) / 'track-run-manifest.json', {'fixture': True})
    assert 'multiple attempt roots; no combined counts' in '\n'.join(dashboard.fresh_baseline_lines(parent))


def test_malformed_track_shapes_are_unavailable_not_a_watch_exception(tmp_path):
    write(tmp_path / 'track-rollouts/attempt.json', {'tracks': [{'unexpected': True}]})
    assert dashboard.track_eval_lines(tmp_path) == ['Track metadata unavailable.']
    write(tmp_path / 'track-rollouts/attempt.json', {'tracks': ['report']})
    write(tmp_path / 'track-rollouts/summary.json', {'tracks': None})
    assert 'terminal:' not in '\n'.join(dashboard.track_eval_lines(tmp_path))


def controlled_stop(root, *, exit_pid=41):
    write(root / 'track-rollouts/attempt.json', {'tracks': ['vqa', 'segmentation']})
    write(root / 'baseline-process.json', {'actor_pid': 41})
    write(root / 'baseline-controlled-stop.json', {
        'schema': 'eva.automedbench-baseline-controlled-stop.v1', 'actor_pid': 41,
        'signal': 'SIGINT', 'actual_actor_returncode': -2,
        'reason': 'Proven context guard rejection; preserve partial evidence.'})
    write(root / 'baseline-process-exit.json', {
        'actor_pid': exit_pid, 'returncode': -2, 'os_process_exit_observed': True})


def test_controlled_stop_uses_matched_actual_exit_not_unknown_or_clinical_zero(tmp_path, monkeypatch):
    controlled_stop(tmp_path)
    write(tmp_path / 'track-rollouts/vqa/turns/01-planning/receipt.json', {'private': 'never read'})
    original = Path.open
    def metadata_only(path, *args, **kwargs):
        assert path.name in {'attempt.json', 'summary.json', 'baseline-process.json',
                             'baseline-process-exit.json', 'baseline-controlled-stop.json'}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', metadata_only)
    text = '\n'.join(dashboard.track_eval_lines(tmp_path))
    assert 'STOPPED — controlled runner/context failure; actual OS exit -2' in text
    assert 'status=STOPPED (partial workflow)' in text
    assert 'not a clinical zero' in text and '1/5' in text
    assert 'not terminal or unknown' not in text and 'private' not in text


def test_unmatched_or_unobserved_exit_does_not_claim_stopped(tmp_path):
    controlled_stop(tmp_path, exit_pid=42)
    assert 'STOPPED' not in '\n'.join(dashboard.track_eval_lines(tmp_path))
    write(tmp_path / 'baseline-process-exit.json', {'actor_pid': 41, 'returncode': -2,
                                                   'os_process_exit_observed': False})
    assert 'STOPPED' not in '\n'.join(dashboard.track_eval_lines(tmp_path))
    (tmp_path / 'baseline-process-exit.json').write_text('{')
    assert 'STOPPED' not in '\n'.join(dashboard.track_eval_lines(tmp_path))


def test_process_exit_without_stop_record_is_not_inferred_running(tmp_path):
    write(tmp_path / 'track-rollouts/attempt.json', {'tracks': ['report']})
    write(tmp_path / 'baseline-process.json', {'actor_pid': 41})
    write(tmp_path / 'baseline-process-exit.json', {'actor_pid': 41, 'returncode': 0,
                                                   'os_process_exit_observed': True})
    text = '\n'.join(dashboard.track_eval_lines(tmp_path))
    assert 'Baseline process exited: OS exit 0' in text
    assert 'status=exited; workflow result unavailable' in text
    assert 'STOPPED' not in text and 'evaluation success or a clinical score' in text


def test_highest_version_preferred_with_historical_twelve_scores_separate(tmp_path):
    prefix = 'automedbench-lite-seven-track-baseline-sft-20260910.v'
    old = tmp_path / (prefix + '2') / str(uuid4())
    write(old / 'track-run-manifest.json', {'fixture': True})
    controlled_stop(old)
    feedback = tmp_path / 'automedbench-lite-seven-track-baseline-feedback-20260910.v2'
    for index, track in enumerate(dashboard.TRACKS):
        verification(feedback, track, 'S1', 6667)
        if index < 5:
            verification(feedback, track, 'S2', 0)
    new = tmp_path / (prefix + '3')
    new.mkdir()
    text = '\n'.join(dashboard.baseline_campaign_lines(tmp_path))
    assert text.startswith('Fresh seven-track baseline v3: awaiting run metadata.')
    assert 'Fresh baseline v3 independently verified process-rubric records: 0/35' in text
    assert 'Historical seven-track baseline v2' in text and old.name in text
    assert 'Historical baseline v2 independently verified process-rubric records: 12/35' in text
    current, historical = text.split('Historical seven-track baseline v2', 1)
    assert '66.67/100' not in current and 'S1 --; S2 --' in current
    assert 'S1 66.67/100; S2 0.00/100; S3 --' in historical
    assert 'STOPPED' in historical


def test_version_discovery_ignores_symlinks_and_noncanonical_versions(tmp_path):
    prefix = 'automedbench-lite-seven-track-baseline-sft-20260910.v'
    for suffix in ('03', 'latest', '1000'):
        (tmp_path / (prefix + suffix)).mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    (tmp_path / (prefix + '4')).symlink_to(outside, target_is_directory=True)
    text = '\n'.join(dashboard.baseline_campaign_lines(tmp_path))
    assert text.startswith('Fresh seven-track baseline v2: not started.')
    assert 'baseline v3' not in text and 'baseline v4' not in text
