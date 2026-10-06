from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from uuid import uuid4


SPEC = importlib.util.spec_from_file_location(
    'training_validation_dashboard', Path(__file__).resolve().parents[1] / 'scripts/show_training_validation_v1.py')
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


def run_root(tmp_path, name='grpo-smoke-v2', *, status='running', groups=2, batch=4):
    root = tmp_path / name
    root.mkdir()
    (root / 'run-receipt.json').write_text(json.dumps({
        'status': status, 'argv': ['train.py', '--num-rollout', str(groups),
                                  '--global-batch-size', str(batch), '--num-steps-per-rollout', '1']}))
    return root


def event(rollout, norm, changed, **overrides):
    return {'event': 'optimizer_step', 'rollout_id': rollout, 'step_id': 0,
            'successful_update': True, 'gradient_norm': norm, 'sampled_changed_values': changed,
            **overrides}


def test_zero_gradient_completed_execution_is_not_no_work(tmp_path):
    root = run_root(tmp_path, status='completed_inconclusive_zero_variance')
    rows = [event(index, 0.0, 0, finite_gradient=True, finite_nonzero_gradient=False,
                  learning_update_observed=False) for index in range(2)]
    (root / 'optimizer-steps.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n')
    text = '\n'.join(dashboard.training_lines(tmp_path, 'grpo'))
    assert 'optimizer executed (finite) [████████████████████] 2/2' in text
    assert 'Nonzero observed learning updates [░░░░░░░░░░░░░░░░░░░░] 0/2' in text
    assert 'Learning signal inconclusive' in text and 'Zero-gradient executed steps: 2' in text


def test_two_groups_four_samples_count_only_current_run_and_never_open_content(tmp_path, monkeypatch):
    earlier = run_root(tmp_path, 'grpo-smoke-v1')
    current = run_root(tmp_path)
    os.utime(earlier / 'run-receipt.json', ns=(1, 1))
    os.utime(current / 'run-receipt.json', ns=(2, 2))
    for owner, group_ids in ((earlier, (0, 1)), (current, (0, 1, 2))):
        for group_id in group_ids:
            group = owner / 'rollouts' / f'{group_id:06d}'
            for index in range(4):
                sample = group / str(uuid4())
                sample.mkdir(parents=True)
                (sample / 'trajectory.json').write_text('private actor content must not be opened')
                (sample / 'training-tokens.json').write_text('private token contents')
                if owner == earlier or group_id != 1 or index < 2:
                    judge = sample / 'judge' / sample.name
                    judge.mkdir(parents=True)
                    (judge / 'grade.json').write_text('{')  # Presence, never falsely validated.
            (group / 'summary.json').write_text('{')  # Concurrent partial metadata write.
    original_open = Path.open

    def bounded_metadata_only(path, *args, **kwargs):
        assert path.name not in {'trajectory.json', 'training-tokens.json', 'grade.json'}
        assert earlier not in path.parents
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', bounded_metadata_only)
    text = '\n'.join(dashboard.training_lines(tmp_path, 'grpo'))
    assert 'trajectories 8/8; grade.json files 6/8' in text
    assert 'group 000000: trajectories 4/4; grades 4/4' in text
    assert 'group 000001: trajectories 4/4; grades 2/4' in text
    assert '000002' not in text and 'grpo-smoke-v1' not in text
    assert 'not independent judgment verification' in text


def test_zero_variance_counts_native_and_legacy_without_duplicate_flags(tmp_path, monkeypatch):
    root = run_root(tmp_path)
    summaries = [
        {'zero_variance_group': True, 'groups': [
            {'zero_variance_group': True}, {'zero_variance_group': False}]},
        {'zero_variance_group': True},
    ]
    for index, summary in enumerate(summaries):
        group = root / 'rollouts' / f'{index:06d}'
        group.mkdir(parents=True)
        (group / 'summary.json').write_text(json.dumps(summary))
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.name in {'run-receipt.json', 'summary.json', 'optimizer-steps.jsonl',
                             'latest_checkpointed_iteration.txt'}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', metadata_only)
    text = '\n'.join(dashboard.training_lines(tmp_path, 'grpo'))
    assert 'Reported zero-variance prompt groups: 2/3' in text
    assert 'reported zero variance (1/2 prompt groups)' in text
    assert 'reported zero variance (1/1 prompt groups)' in text
    assert 'summary metadata only, not independent reward verification' in text
    assert 'grade.json files 0/8' in text  # Summary flags never become grades.


def test_zero_variance_metadata_does_not_infer_missing_or_override_nested_false():
    assert dashboard._reported_zero_variance_groups({
        'zero_variance_group': True, 'groups': [{'zero_variance_group': False}]}) == (0, 1)
    assert dashboard._reported_zero_variance_groups({'zero_variance_group': False}) == (0, 1)
    for summary in ({}, {'zero_variance_group': 1}, {'zero_variance_group': 'true'},
                    {'zero_variance_group': True, 'groups': []},
                    {'zero_variance_group': True, 'groups': None},
                    {'groups': [None, {}, {'zero_variance_group': 1}]}):
        assert dashboard._reported_zero_variance_groups(summary) == (0, 0)


def _isolated_draw(root, step, status, trajectories=4, grades=3):
    draw = root / 'rollout-group-attempts' / f'{step:06d}' / str(uuid4())
    group = draw / 'rollouts' / f'{step:06d}'
    group.mkdir(parents=True)
    for index in range(trajectories):
        sample = group / str(uuid4())
        sample.mkdir()
        (sample / 'trajectory.json').write_text('private fixture - do not read')
        if index < grades:
            judge = sample / 'judge' / sample.name
            judge.mkdir(parents=True)
            (judge / 'grade.json').write_text('private fixture - do not read')
    if status is not None:
        (draw / 'result.json').write_text(json.dumps({'status': status}))
    return group


def test_isolation_accepts_only_uuid_root_presence_and_counts_unavailable_separately(tmp_path, monkeypatch):
    root = run_root(tmp_path, groups=1)
    _isolated_draw(root, 0, 'unavailable', grades=3)
    accepted = _isolated_draw(root, 0, 'accepted', grades=4)
    canonical = root / 'rollouts/000000'
    canonical.mkdir(parents=True)
    binding = {'policy': 'completed-judge-verdict-fresh-group-v1', 'status': 'accepted',
               'optimizer_rollout_id': 0, 'accepted_group_root': str(accepted)}
    (canonical / 'isolation.json').write_text(json.dumps(binding))
    (canonical / 'summary.json').write_text(json.dumps({'groups': [{'zero_variance_group': False}]}))
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.name not in {'trajectory.json', 'grade.json', 'training-tokens.json', 'failure.json'}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', metadata_only)
    text = '\n'.join(dashboard.training_lines(tmp_path, 'grpo'))
    assert 'trajectories 4/4; grade.json files 4/4' in text
    assert 'Unavailable isolated draws: 1; excluded' in text
    assert 'accepted UUID draw; unavailable draws 1 (excluded)' in text
    assert 'optimizer executed (finite) [░░░░░░░░░░░░░░░░░░░░] 0/1' in text
    assert 'trajectories 8/' not in text and 'files 7/' not in text


def test_isolation_pending_draw_reports_visible_artifacts_without_admission(tmp_path, monkeypatch):
    root = run_root(tmp_path, groups=1)
    _isolated_draw(root, 0, 'unavailable')
    pending = _isolated_draw(root, 0, None, trajectories=3, grades=1)
    (pending.parent.parent / 'result.json').write_text('{')  # Concurrent partial result.
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.name not in {'trajectory.json', 'grade.json', 'training-tokens.json', 'failure.json'}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', metadata_only)
    text = '\n'.join(dashboard.training_lines(tmp_path, 'grpo'))
    assert 'trajectories 0/4; grade.json files 0/4' in text
    assert 'accepted pending; unavailable draws 1 (excluded)' in text
    assert 'unaccepted draw artifacts: trajectories 3, grade files 1 (NOT accepted scores or updates)' in text
    assert 'Unavailable isolated draws: 1' in text


def test_isolation_terminal_unavailable_and_unsafe_accepted_pointer_never_count_as_accepted(tmp_path):
    root = run_root(tmp_path, groups=1)
    _isolated_draw(root, 0, 'unavailable')
    canonical = root / 'rollouts/000000'
    canonical.mkdir(parents=True)
    (canonical / 'isolation.json').write_text(json.dumps({'status': 'unavailable'}))
    text = '\n'.join(dashboard.training_lines(tmp_path, 'grpo'))
    assert 'accepted unavailable; unavailable draws 1 (excluded)' in text
    assert 'trajectories 0/4; grade.json files 0/4' in text
    (canonical / 'isolation.json').write_text(json.dumps({
        'status': 'accepted', 'accepted_group_root': '/unrelated/private', 'optimizer_rollout_id': 0}))
    text = '\n'.join(dashboard.training_lines(tmp_path, 'grpo'))
    assert 'accepted metadata unavailable (UUID binding differs)' in text
    assert 'trajectories 0/4; grade.json files 0/4' in text


def test_latest_inflight_receipt_does_not_fall_back_and_event_rows_are_bounded(tmp_path):
    earlier = run_root(tmp_path, 'grpo-smoke-v1')
    current = run_root(tmp_path)
    (current / 'run-receipt.json').write_text('{')
    os.utime(earlier / 'run-receipt.json', ns=(1, 1))
    os.utime(current / 'run-receipt.json', ns=(2, 2))
    rows = [event(0, 1.0, 2), event(0, 1.0, 2), event(1, float('nan'), 1),
            event(2, 2.0, 3, successful_update=False), ['invalid metadata']]
    (current / 'optimizer-steps.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n{"event":')
    text = '\n'.join(dashboard.training_lines(tmp_path, 'grpo'))
    assert '1/?' in text and '[grpo-smoke-v2]' in text and 'metadata unavailable' in text
    assert 'artifact targets unavailable' in text and 'grpo-smoke-v1' not in text
    assert dashboard.read_small_json(current / 'run-receipt.json') is None
    (current / 'run-receipt.json').write_bytes(b' ' * (dashboard.MAX_SMALL_JSON_BYTES + 1))
    assert dashboard.read_small_json(current / 'run-receipt.json') is None


def test_plan_and_finite_learning_counts_use_actual_numeric_evidence():
    plan = dashboard.training_plan({'argv': ['--num-rollout=4', '--start-rollout-id', '2',
        '--global-batch-size', '3', '--num-steps-per-rollout', '2']})
    assert list(plan['groups']) == [2, 3]
    assert plan['optimizer_steps'] == 4 and plan['samples_per_group'] == 6
    assert dashboard.training_plan({'argv': ['--num-rollout', '2', '--global-batch-size', 'no']}) is None
    assert dashboard.training_plan({'argv': ['--num-rollout', '2', '--num-rollout', '3',
                                            '--global-batch-size', '4']}) is None
    assert dashboard.optimizer_counts([
        event(0, 0.0, 0), event(1, 1.0, 2), event(2, 1.0, 0),
        event(3, None, 4), event(4, True, 4), event(5, float('inf'), 5),
        event(6, -1.0, 6), event(7, 1.0, 7, finite_gradient=False),
    ]) == (3, 1, 1)


def test_epoch_progress_supersedes_smoke_without_recounting_its_steps(tmp_path):
    smoke = run_root(tmp_path, 'sft-smoke-v6', status='complete', groups=2, batch=2)
    epoch = run_root(tmp_path, 'qwen35-9b-sft-epoch-20260910.v1', groups=171, batch=8)
    path = epoch / 'run-receipt.json'
    receipt = json.loads(path.read_text())
    receipt['epoch_plan'] = {'dataset_rows': 1368, 'dataset_tokens': 13103738}
    path.write_text(json.dumps(receipt))
    os.utime(smoke / 'run-receipt.json', ns=(1, 1))
    os.utime(path, ns=(2, 2))
    (smoke / 'optimizer-steps.jsonl').write_text(json.dumps(event(0, 1.0, 1)))
    rows = [event(i, 1.0, 2, created_at=f'2026-09-10T07:0{i}:00Z') for i in range(3)]
    (epoch / 'optimizer-steps.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    checkpoints = epoch / 'checkpoints'
    checkpoints.mkdir()
    (checkpoints / 'latest_checkpointed_iteration.txt').write_text('1\n')
    text = '\n'.join(dashboard.training_lines(tmp_path, 'sft', extra_receipts=(path,)))
    assert '3/171' in text and '1368 unique examples' in text
    assert 'Recent median 60.0s/update' in text and '2.80h' in text
    assert 'sft-smoke-v6' not in text and '4/171' not in text
    assert 'saved update 2/171' in text and 'bytes are not verified' in text


def test_epoch_eta_is_not_reported_for_failed_run(tmp_path):
    epoch = run_root(tmp_path, 'sft-smoke-v7', status='failed', groups=171, batch=8)
    path = epoch / 'run-receipt.json'
    receipt = json.loads(path.read_text())
    receipt['epoch_plan'] = {'dataset_rows': 1368}
    path.write_text(json.dumps(receipt))
    rows = [event(i, 1.0, 2, created_at=f'2026-09-10T07:0{i}:00Z') for i in range(3)]
    (epoch / 'optimizer-steps.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    assert 'estimated remaining' not in '\n'.join(dashboard.training_lines(tmp_path, 'sft'))


def test_memory_artifact_presence_is_not_behavioral_success(tmp_path):
    for index in range(1, 5):
        (tmp_path / f'turn-{index}.json').write_text('private content not opened')
    text = '\n'.join(dashboard.memory_lines(tmp_path, 'fixture'))
    assert '4/4 result pending' in text and 'checks passed' not in text
    (tmp_path / 'result.json').write_text(json.dumps({'passed': True, 'checks': {'context': False}}))
    assert 'checks failed' in '\n'.join(dashboard.memory_lines(tmp_path, 'fixture'))
    (tmp_path / 'result.json').write_text(json.dumps({'passed': True, 'checks': {'context': True}}))
    assert 'checks passed' in '\n'.join(dashboard.memory_lines(tmp_path, 'fixture'))


def test_medical_progress_does_not_read_trajectories_or_invent_zero_score(tmp_path):
    (tmp_path / 'run-manifest.json').write_text(json.dumps({'diagnostic_only': False,
        'cases': [{'case_id': 'case1'}]}))
    phase = tmp_path / 'codex-rollouts/case1/turns/01-planning'
    phase.mkdir(parents=True)
    (phase / 'receipt.json').write_text('private body must not be parsed')
    text = '\n'.join(dashboard.medical_eval_lines(tmp_path))
    assert '1/3' in text and 'pending (not zero)' in text
    assert 'not full seven-track' in text and 'not rubric success' in text


def test_fixed_diagnostic_predictions_are_not_medical_model_progress(tmp_path):
    (tmp_path / 'run-manifest.json').write_text(json.dumps({'diagnostic_only': True,
        'cases': [{'case_id': 'case1'}]}))
    assert 'no non-diagnostic' in '\n'.join(dashboard.medical_eval_lines(tmp_path))


def test_new_campaign_does_not_count_historical_smoke(tmp_path):
    assert '0/500' in '\n'.join(dashboard.rsi_lines(tmp_path))
    (tmp_path / 'progress.json').write_text(json.dumps({
        'checkpoint_backed_updates': 50, 'checkpoint_backed_learning_updates': 42,
        'completed_rounds': 1, 'status': 'running', 'phase': 'evaluation', 'memory_context_pass': None}))
    text = '\n'.join(dashboard.rsi_lines(tmp_path))
    assert '50/500' in text and 'rounds 1/10' in text and '42' in text and 'None' in text


def test_live_learning_observation_does_not_change_durable_bar(tmp_path, monkeypatch):
    stale = {'checkpoint_backed_updates': 1, 'checkpoint_backed_learning_updates': 0,
             'completed_rounds': 0, 'status': 'running', 'phase': 'train', 'memory_context_pass': None,
             'active_training_observations_not_yet_checkpoint_credited': None}
    (tmp_path / 'progress.json').write_text(json.dumps(stale))
    calls = []

    def current_status(root):
        calls.append(root)
        return {**stale, 'active_training_observations_not_yet_checkpoint_credited': {
            'executions': 2, 'learning_updates': 1, 'duplicate_events': 0}}

    monkeypatch.setattr(dashboard, 'rsi_current_status', current_status)
    before = (tmp_path / 'progress.json').read_bytes()
    text = '\n'.join(dashboard.rsi_lines(tmp_path))
    assert calls == [tmp_path]
    assert '1/500' in text and '3/500' not in text
    assert 'Nonzero checkpoint-backed learning updates 0' in text
    assert 'optimizer executions 2; nonzero learning updates 1' in text
    assert 'observed, NOT checkpoint credited' in text
    assert (tmp_path / 'progress.json').read_bytes() == before


def test_live_observations_none_or_partial_keep_historical_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'rsi_current_status', lambda root: None)
    base = {'checkpoint_backed_updates': 1, 'checkpoint_backed_learning_updates': 0,
            'completed_rounds': 0, 'status': 'running', 'phase': 'train'}
    for observed in (None, {'status': 'partial_or_invalid_event_stream_not_credited'}):
        (tmp_path / 'progress.json').write_text(json.dumps({**base,
            'active_training_observations_not_yet_checkpoint_credited': observed}))
        text = '\n'.join(dashboard.rsi_lines(tmp_path))
        assert '1/500' in text and 'memory/context pass remains None' in text
        assert ('Active training observations unavailable' in text) == isinstance(observed, dict)
        assert 'optimizer executions' not in text


def test_current_status_is_one_existing_read_only_controller_call(tmp_path, monkeypatch):
    import training.eva_rsi.controller as controller_module
    (tmp_path / 'state.json').write_text('{}')
    before = (tmp_path / 'state.json').read_bytes()
    calls = []

    class ReadOnlyController:
        def __init__(self, root):
            calls.append(('init', root))

        def status(self):
            calls.append(('status',))
            return {'existing_status': True}

    monkeypatch.setattr(controller_module, 'Controller', ReadOnlyController)
    assert dashboard.rsi_current_status(tmp_path) == {'existing_status': True}
    assert calls == [('init', tmp_path), ('status',)]
    assert (tmp_path / 'state.json').read_bytes() == before


def test_pending_checkpoint_observations_are_distinct_and_not_credited(tmp_path, monkeypatch):
    progress = {'checkpoint_backed_updates': 1, 'checkpoint_backed_learning_updates': 0,
        'completed_rounds': 0, 'status': 'blocked', 'phase': 'durable_checkpoint',
        'active_training_observations_not_yet_checkpoint_credited': None,
        'pending_training_observations_not_yet_checkpoint_credited': {'executions': 1, 'learning_updates': 1}}
    monkeypatch.setattr(dashboard, 'rsi_current_status', lambda root: progress)
    text = '\n'.join(dashboard.rsi_lines(tmp_path))
    assert '1/500' in text and '2/500' not in text and 'Active training:' not in text
    assert 'Pending checkpoint training: optimizer executions 1; nonzero learning updates 1' in text
    assert 'observed, NOT checkpoint credited' in text
    assert 'Nonzero checkpoint-backed learning updates 0' in text


def test_track_receipts_are_presence_only(tmp_path):
    root = tmp_path / 'track-rollouts'
    root.mkdir()
    (root / 'attempt.json').write_text(json.dumps({'tracks': ['classification']}))
    phase = root / 'classification/turns/01-planning'
    phase.mkdir(parents=True)
    (phase / 'receipt.json').write_text('private content must not be read')
    text = '\n'.join(dashboard.track_eval_lines(tmp_path))
    assert '1/5' in text and 'not a seven-track score' in text
    (root / 'summary.json').write_text(json.dumps({'tracks': [{'track': 'classification',
        'errors': [{'phase': '01-planning', 'error': 'turn_not_completed'}], 'completed_requested_turns': False}]}))
    text = '\n'.join(dashboard.track_eval_lines(tmp_path))
    assert 'failed/incomplete' in text and 'reported_errors=1' in text


def budget_terminal_fixture(tmp_path):
    root = tmp_path / 'track-rollouts/segmentation'
    root.mkdir(parents=True)
    (root.parent / 'attempt.json').write_text(json.dumps({'tracks': ['segmentation']}))
    for phase in dashboard.PHASES[:4]:
        path = root / 'turns' / phase / 'receipt.json'
        path.parent.mkdir(parents=True)
        path.write_text('PRIVATE receipt content must not be opened')

    def committed(path, value):
        value['document_blake3'] = dashboard._native_document_digest(value)
        path.write_text(json.dumps(value))
        return value

    admission = committed(root / 'track-budget.json', {
        'schema': 'eva.automedbench-track-budget.v1',
        'policy': 'admitted-track-wallclock-3600-v1', 'timeout_seconds': 3600})
    outcome = committed(root / 'track-budget-outcome.json', {
        'schema': 'eva.automedbench-track-budget-outcome.v1', 'actual_turn_count': 4,
        'policy': admission['policy'], 'budget_document_blake3': admission['document_blake3'],
        'timeout_seconds': 3600, 'elapsed_seconds': 3600.190, 'remaining_seconds': 0,
        'deadline_exhausted': True})
    cleanup = committed(root / 'track-deadline-cleanup.json', {
        'schema': 'eva.automedbench-track-deadline-cleanup.v1', 'trigger': 'track_deadline',
        'workspace_quiescent': True, 'error_category': None})
    committed(root / 'turns/04-full-subset/policy-budget-terminal.json', {
        'schema': 'eva.automedbench-policy-budget-terminal.v1', 'phase_intent': '04-full-subset',
        'actual_terminal_status': 'interrupted', 'workspace_quiescence_verified': True,
        'infrastructure_error': None, 'track_budget': dict(outcome),
        'deadline_cleanup_document_blake3': cleanup['document_blake3']})
    return root, committed


def test_track_budget_terminal_uses_only_small_committed_metadata(tmp_path, monkeypatch):
    budget_terminal_fixture(tmp_path)
    original = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.name not in {'receipt.json', 'trajectory.json', 'grade.json', 'rollout.json'}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', metadata_only)
    text = '\n'.join(dashboard.track_eval_lines(tmp_path, budget_outcomes=True))
    assert '4/5' in text
    assert 'budget terminal (S4 interrupted; elapsed 3600.190s / 3600s' in text
    assert 'committed metadata, not a rubric pass' in text
    assert 'not terminal or unknown' not in text
    legacy = '\n'.join(dashboard.track_eval_lines(tmp_path))
    assert 'not terminal or unknown' in legacy and 'budget terminal' not in legacy


def test_track_budget_terminal_changed_or_incomplete_metadata_is_not_terminal(tmp_path):
    root, committed = budget_terminal_fixture(tmp_path)
    for relative, key, value in [
        ('track-budget-outcome.json', 'elapsed_seconds', 3610),
        ('track-budget.json', 'timeout_seconds', 999),
        ('turns/04-full-subset/policy-budget-terminal.json', 'phase_intent', '05-review'),
        ('track-deadline-cleanup.json', 'workspace_quiescent', False),
    ]:
        path = root / relative
        original = path.read_bytes()
        document = json.loads(original)
        document[key] = value
        path.write_text(json.dumps(document))
        assert 'budget terminal' not in '\n'.join(dashboard.track_eval_lines(tmp_path, budget_outcomes=True))
        path.write_bytes(original)
    outcome_path = root / 'track-budget-outcome.json'
    outcome = json.loads(outcome_path.read_text())
    for change in [{'actual_turn_count': 3}, {'elapsed_seconds': 3599}, {'deadline_exhausted': False}]:
        committed(outcome_path, {**outcome, **change})
        assert 'budget terminal' not in '\n'.join(dashboard.track_eval_lines(tmp_path, budget_outcomes=True))


def test_zero_turn_budget_still_unavailable_not_a_deadline_terminal(tmp_path):
    root = tmp_path / 'track-rollouts/classification'
    root.mkdir(parents=True)
    (root.parent / 'attempt.json').write_text(json.dumps({'tracks': ['classification']}))
    outcome = {'schema': 'eva.automedbench-track-budget-outcome.v1', 'actual_turn_count': 0}
    outcome['document_blake3'] = dashboard._native_document_digest(outcome)
    (root / 'track-budget-outcome.json').write_text(json.dumps(outcome))
    text = '\n'.join(dashboard.track_eval_lines(tmp_path, budget_outcomes=True))
    assert 'unavailable before first recorded turn (not a task/rubric zero)' in text
    assert 'budget terminal' not in text


def test_native_capture_progress_does_not_read_private_tokens(tmp_path):
    tokens = tmp_path / 'sample-id/private-provider-tokens'
    tokens.mkdir(parents=True)
    for name in ('request-one.json', 'request-two.json', 'segment-one.json'):
        (tokens / name).write_text('PRIVATE invalid JSON deliberately not read')
    text = '\n'.join(dashboard.native_preflight_lines(tmp_path))
    assert 'provider requests=2; retained segments=1' in text
    assert 'PRIVATE' not in text and 'not whole-trajectory/Judge success' in text
