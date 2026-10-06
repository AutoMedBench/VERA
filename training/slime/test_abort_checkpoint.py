"""CPU fixture executes the actual selected Slime train loop with fake RPCs.

No Slime/Ray/Torch import, GPU model, provider, or real checkpoint is involved.
Only top-level imports are replaced; the actual train function is unchanged.
"""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from train_abort_checkpoint import AbortCheckpointError, FirstUpdateCheckpointError, train_with_abort_checkpoint


class RolloutFailure(RuntimeError):
    pass


class CheckpointFailure(RuntimeError):
    pass


class Ref:
    def __init__(self, function):
        self.function = function
        self.finished = False

    def get(self):
        if not self.finished:
            try:
                self.value, self.error = self.function(), None
            except Exception as error:
                self.error = error
            self.finished = True
        if self.error:
            raise self.error
        return self.value


def ray_get(value):
    return [ray_get(item) for item in value] if isinstance(value, list) else value.get()


def fixture(tmp_path, monkeypatch, *, fail_rollout=1, fail_save=False,
            corrupt_optimizer=False, fail_train=False, save_interval=25, per_epoch=None):
    events, generated, trained, saved, lifecycle = [], [], [], [], []
    optimizer = tmp_path / 'optimizer-steps.jsonl'
    monkeypatch.setenv('EVA_OPTIMIZER_RECEIPT', str(optimizer))
    for name in ('BAT_OPD_RECOVER_BOUNDARY_EVAL', 'BAT_OPD_EXIT_AFTER_DURABLE_SAVE'):
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace(
        use_critic=False, release_train=False, async_save=False, num_steps_per_rollout=1,
        start_rollout_id=0, num_rollout=3, offload_rollout=True, offload_train=True,
        check_weight_update_equal=False, eval_interval=None, skip_eval_before_train=True,
        save_interval=save_interval, rollout_global_dataset=True,
        save=str(tmp_path / 'checkpoints'), save_hf=str(tmp_path / 'hf/iter_{rollout_id:07d}'))

    def remote(function):
        return SimpleNamespace(remote=lambda *args, **kwargs: Ref(lambda: function(*args, **kwargs)))

    def generate(index):
        generated.append(index)
        if index == fail_rollout:
            raise RolloutFailure('fixture rollout failed')
        return ('unchanged-opaque-rollout-data-reference', index)

    manager = SimpleNamespace(generate=remote(generate),
        onload_weights=remote(lambda: None), onload_kv=remote(lambda: None),
        offload=remote(lambda: lifecycle.append('offload_rollout')),
        recover_hard_released_engines=remote(lambda: False),
        save=remote(lambda index: lifecycle.append(('dataset_cursor_save', index))),
        dispose=remote(lambda: lifecycle.append('dispose')))

    def train(index, data):
        assert data == ('unchanged-opaque-rollout-data-reference', index)
        trained.append(index)
        if fail_train is True or (type(fail_train) is int and index == fail_train):
            raise RuntimeError('fixture partly executed train; must not save')
        events.append({'event': 'optimizer_step', 'rollout_id': index, 'step_id': 0,
            'successful_update': not corrupt_optimizer, 'gradient_norm': 1.8697,
            'sampled_changed_values': 2045, 'trainable_parameters': 8_953_803_264})
        optimizer.write_text(''.join(json.dumps(row) + '\n' for row in events))
        lifecycle.append(('train_ack', index))

    def save(index, force_sync=False):
        saved.append((index, force_sync))
        lifecycle.append(('save', index))
        if fail_save:
            raise CheckpointFailure('fixture checkpoint error')

    actor = SimpleNamespace(async_train=lambda index, data: [Ref(lambda: train(index, data))],
                            save_model=save, update_weights=lambda: lifecycle.append('update_weights'))

    def create_models(args, pgs, supplied_manager):
        assert supplied_manager is manager  # No wrapper serialized into model workers.
        lifecycle.append('create_actor')
        return actor, None

    source = Path(os.environ.get('EVA_TEST_SLIME_TRAIN',
        str(Path(__file__).resolve().parents[3] / 'slime-upstream/train.py')))
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = {'__name__': 'eva_cpu_upstream_train_fixture', 'os': os,
        'ray': SimpleNamespace(get=ray_get), 'configure_logger': lambda: None,
        'init_tracking': lambda args: None, 'finish_tracking': lambda args: None,
        'create_placement_groups': lambda args: {'rollout': 'fixture-placement'},
        'create_rollout_manager': lambda args, pg: (manager, per_epoch),
        'create_training_models': create_models,
        'should_run_periodic_action': lambda index, interval, *rest:
            interval is not None and ((index + 1) % interval == 0
                                     or (rest and rest[0] is not None and (index + 1) % rest[0] == 0)
                                     or (len(rest) == 2 and index == rest[1] - 1))}
    exec(compile(tree, str(source), 'exec'), namespace)
    return args, namespace, generated, trained, saved, lifecycle


def test_next_rollout_failure_saves_only_acknowledged_finite_prefix_then_reraises(tmp_path, monkeypatch):
    args, namespace, generated, trained, saved, lifecycle = fixture(tmp_path, monkeypatch)
    original_factory = namespace['create_training_models']
    with pytest.raises(RolloutFailure):
        train_with_abort_checkpoint(args, namespace)
    assert generated == [0, 1] and trained == [0] and saved == [(0, True)]
    assert lifecycle.index(('train_ack', 0)) < lifecycle.index(('save', 0))
    assert not any(isinstance(row, tuple) and row[0] == 'dataset_cursor_save' for row in lifecycle)
    receipt = json.loads((tmp_path / 'abort-checkpoint.json').read_text())
    assert receipt['status'] == 'save_returned_requires_independent_verification'
    assert receipt['saved_rollout_id'] == 0 and receipt['training_status'] == 'failed'
    assert receipt['optimizer_prefix']['optimizer_step_executions'] == 1
    assert receipt['optimizer_prefix']['observed_learning_updates'] == 1
    assert receipt['hf_export_requested'] and not receipt['dataset_cursor_saved_on_abort']
    assert namespace['create_training_models'] is original_factory


def test_failure_before_first_train_saves_nothing(tmp_path, monkeypatch):
    args, namespace, generated, trained, saved, _ = fixture(tmp_path, monkeypatch, fail_rollout=0)
    with pytest.raises(RolloutFailure):
        train_with_abort_checkpoint(args, namespace)
    assert generated == [0] and trained == saved == []
    assert json.loads((tmp_path / 'abort-checkpoint.json').read_text())['status'] == 'not_saved_no_completed_train'


@pytest.mark.parametrize('mode', ['save_failure', 'invalid_optimizer'])
def test_checkpoint_or_observer_failure_is_surfaced_not_saved(tmp_path, monkeypatch, mode):
    args, namespace, _, _, saved, _ = fixture(tmp_path, monkeypatch,
        fail_save=mode == 'save_failure', corrupt_optimizer=mode == 'invalid_optimizer')
    with pytest.raises(AbortCheckpointError) as caught:
        train_with_abort_checkpoint(args, namespace)
    assert isinstance(caught.value.__cause__, CheckpointFailure if mode == 'save_failure' else ValueError)
    assert saved == ([(0, True)] if mode == 'save_failure' else [])
    receipt = json.loads((tmp_path / 'abort-checkpoint.json').read_text())
    assert receipt['status'] == 'save_failed' and receipt['original_error_type'] == 'RolloutFailure'
    assert 'saved_rollout_id' not in receipt


def test_failed_training_is_not_recovered_as_prior_weights(tmp_path, monkeypatch):
    args, namespace, _, trained, saved, _ = fixture(tmp_path, monkeypatch, fail_train=True)
    with pytest.raises(RuntimeError, match='partly executed train'):
        train_with_abort_checkpoint(args, namespace)
    assert trained == [0] and saved == []
    assert not (tmp_path / 'abort-checkpoint.json').exists()


def test_already_saved_prefix_is_not_saved_twice(tmp_path, monkeypatch):
    args, namespace, _, _, saved, _ = fixture(tmp_path, monkeypatch, save_interval=1)
    with pytest.raises(RolloutFailure):
        train_with_abort_checkpoint(args, namespace)
    assert saved == [(0, False)]
    assert json.loads((tmp_path / 'abort-checkpoint.json').read_text())['status'] == 'not_saved_already_checkpointed'


def test_successful_loop_keeps_interval_saving_and_exactly_one_train_per_group(tmp_path, monkeypatch):
    args, namespace, generated, trained, saved, _ = fixture(tmp_path, monkeypatch, fail_rollout=None)
    train_with_abort_checkpoint(args, namespace)
    assert generated == trained == [0, 1, 2] and saved == [(2, True)]
    assert not (tmp_path / 'abort-checkpoint.json').exists()


def test_first_extra_save_continues_same_optimizer_all_48_updates_with_existing_cadence(tmp_path, monkeypatch):
    args, namespace, generated, trained, saved, lifecycle = fixture(tmp_path, monkeypatch, fail_rollout=None)
    args.num_rollout = 48
    train_with_abort_checkpoint(args, namespace, checkpoint_first_update=True)
    assert generated == trained == list(range(48))
    assert saved == [(0, True), (24, False), (47, True)]
    assert lifecycle.count('create_actor') == 1 and lifecycle.count('update_weights') == 49
    assert lifecycle.index(('train_ack', 0)) < lifecycle.index(('dataset_cursor_save', 0)) < lifecycle.index(('save', 0))
    assert lifecycle.index(('save', 0)) < lifecycle.index(('train_ack', 1))
    receipt = json.loads((tmp_path / 'first-update-checkpoint.json').read_text())
    assert receipt['status'] == 'save_returned_requires_independent_verification'
    assert receipt['saved_rollout_id'] == 0 and receipt['planned_rollouts'] == 48
    assert receipt['optimizer_prefix']['optimizer_step_executions'] == 1
    assert receipt['dataset_cursor_saved_before_model'] is True
    assert receipt['optimizer_restarted'] is False and receipt['additional_optimizer_updates'] == 0
    assert receipt['remaining_plan_unchanged'] is True
    assert not (tmp_path / 'abort-checkpoint.json').exists()


def test_first_save_survives_later_failed_train_without_saving_partial_state(tmp_path, monkeypatch):
    args, namespace, generated, trained, saved, lifecycle = fixture(
        tmp_path, monkeypatch, fail_rollout=None, fail_train=3)
    args.num_rollout = 4
    with pytest.raises(RuntimeError, match='partly executed train'):
        train_with_abort_checkpoint(args, namespace, checkpoint_first_update=True)
    assert generated == trained == [0, 1, 2, 3] and saved == [(0, True)]
    assert [json.loads(line)['rollout_id'] for line in (tmp_path / 'optimizer-steps.jsonl').read_text().splitlines()] == [0, 1, 2]
    receipt = json.loads((tmp_path / 'first-update-checkpoint.json').read_text())
    assert receipt['saved_rollout_id'] == 0
    assert receipt['optimizer_prefix']['optimizer_step_executions'] == 1
    assert lifecycle.count('create_actor') == 1
    assert not (tmp_path / 'abort-checkpoint.json').exists()


@pytest.mark.parametrize('mode', ['periodic', 'final', 'epoch'])
def test_first_save_delegates_to_existing_boundaries_without_duplicate_save(tmp_path, monkeypatch, mode):
    args, namespace, _, _, saved, _ = fixture(tmp_path, monkeypatch,
        fail_rollout=None if mode == 'final' else 1,
        save_interval=1 if mode == 'periodic' else 25, per_epoch=1 if mode == 'epoch' else None)
    if mode == 'final':
        args.num_rollout = 1
        train_with_abort_checkpoint(args, namespace, checkpoint_first_update=True)
    else:
        with pytest.raises(RolloutFailure):
            train_with_abort_checkpoint(args, namespace, checkpoint_first_update=True)
    assert saved == [(0, mode == 'final')]
    receipt = json.loads((tmp_path / 'first-update-checkpoint.json').read_text())
    assert receipt['status'] == 'delegated_to_normal_checkpoint_boundary'
    assert 'saved_rollout_id' not in receipt  # Normal save has its own evidence.


@pytest.mark.parametrize('mode', ['train', 'rollout'])
def test_enabled_first_checkpoint_does_not_save_before_actual_first_ack(tmp_path, monkeypatch, mode):
    args, namespace, _, _, saved, _ = fixture(tmp_path, monkeypatch,
        fail_train=mode == 'train', fail_rollout=0 if mode == 'rollout' else None)
    with pytest.raises(RuntimeError):
        train_with_abort_checkpoint(args, namespace, checkpoint_first_update=True)
    assert saved == [] and not (tmp_path / 'first-update-checkpoint.json').exists()


@pytest.mark.parametrize('mode', ['save_failure', 'invalid_optimizer'])
def test_extra_save_failure_is_surfaced_and_stops_before_next_group(tmp_path, monkeypatch, mode):
    args, namespace, generated, trained, saved, _ = fixture(tmp_path, monkeypatch,
        fail_save=mode == 'save_failure', corrupt_optimizer=mode == 'invalid_optimizer')
    with pytest.raises(FirstUpdateCheckpointError) as caught:
        train_with_abort_checkpoint(args, namespace, checkpoint_first_update=True)
    assert generated == trained == [0]
    assert isinstance(caught.value.__cause__, CheckpointFailure if mode == 'save_failure' else ValueError)
    assert saved == ([(0, True)] if mode == 'save_failure' else [])
    receipt = json.loads((tmp_path / 'first-update-checkpoint.json').read_text())
    assert receipt['status'] == 'save_failed' and 'saved_rollout_id' not in receipt
    assert not (tmp_path / 'abort-checkpoint.json').exists()


def test_next_rollout_failure_does_not_repeat_successful_first_extra_save(tmp_path, monkeypatch):
    args, namespace, _, _, saved, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(RolloutFailure):
        train_with_abort_checkpoint(args, namespace, checkpoint_first_update=True)
    assert saved == [(0, True)]
    receipt = json.loads((tmp_path / 'abort-checkpoint.json').read_text())
    assert receipt['status'] == 'not_saved_already_checkpointed'
    assert receipt['last_normal_checkpoint_rollout_id'] is None


def test_wrapper_only_flag_never_reaches_upstream_parser(monkeypatch):
    import train_abort_checkpoint as wrapper
    from unittest.mock import Mock

    argv = ['driver.py', '--checkpoint-first-update', '--num-rollout', '48']
    monkeypatch.setattr(wrapper.sys, 'argv', argv[:])
    parsed = SimpleNamespace(original='unchanged Slime arguments')
    def parse_args():
        assert wrapper.sys.argv == ['driver.py', '--num-rollout', '48']
        return parsed
    namespace = {'parse_args': parse_args}
    monkeypatch.setattr(wrapper.runpy, 'run_path', lambda *a, **k: namespace)
    train = Mock()
    monkeypatch.setattr(wrapper, 'train_with_abort_checkpoint', train)
    wrapper.main()
    train.assert_called_once_with(parsed, namespace, checkpoint_first_update=True)
    assert wrapper.sys.argv == argv
