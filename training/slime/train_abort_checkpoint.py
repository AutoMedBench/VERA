"""Opt-in driver hook: checkpoint a completed prefix if the NEXT rollout fails.

Executes the selected, unchanged Slime train.py. Only driver-side handles are
wrapped; worker actors receive their original Ray handles. No loss, sampled
tokens, optimizer calls, successful-rollout ordering, or dataset values change.
This is deliberately not recovery from a failed/partly executed training step.
An optional first-update save runs after its actual acknowledgement, then the
same optimizer continues through the unchanged remaining loop and save cadence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
from uuid import uuid4


class AbortCheckpointError(RuntimeError):
    """The original rollout failed and its unsaved prefix could not be saved."""


class FirstUpdateCheckpointError(RuntimeError):
    """The acknowledged first update could not be independently observed/saved."""


def completed_optimizer_prefix(path: Path, last_rollout: int) -> dict:
    """Reopen actual finite optimizer observations; never infer them from logs."""
    from blake3 import blake3

    payload = path.read_bytes()
    updates = [json.loads(line) for line in payload.splitlines() if line.strip()]
    updates = [row for row in updates if row.get('event') == 'optimizer_step']
    if [(row.get('rollout_id'), row.get('step_id')) for row in updates] != [
            (index, 0) for index in range(last_rollout + 1)]:
        raise ValueError('Abort checkpoint requires the exact completed optimizer prefix')
    for row in updates:
        norm = row.get('gradient_norm')
        changes = row.get('sampled_changed_values')
        if (row.get('successful_update') is not True
                or type(norm) not in (int, float) or not math.isfinite(norm) or norm < 0
                or row.get('trainable_parameters') != 8_953_803_264
                or type(changes) is not int or changes < 0):
            raise ValueError('Abort checkpoint optimizer observation is not a valid finite update')
    return {'path': str(path), 'blake3': blake3(payload).hexdigest(),
            'optimizer_step_executions': len(updates),
            'observed_learning_updates': sum(
                row['gradient_norm'] > 0 and row['sampled_changed_values'] > 0 for row in updates)}


class _Delegate:
    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)


class _AbortState:
    def __init__(self, args, ray, *, checkpoint_first_update=False, periodic_save=None):
        if (args.use_critic or args.release_train or args.async_save
                or args.num_steps_per_rollout != 1
                or args.start_rollout_id not in (None, 0)):
            raise ValueError('Abort checkpoint supports synchronous actor-only one-step fresh warm starts')
        if type(checkpoint_first_update) is not bool:
            raise ValueError('First-update checkpoint selection must be boolean')
        self.args, self.ray = args, ray
        self.checkpoint_first_update = checkpoint_first_update
        self.periodic_save = periodic_save
        self.num_rollout_per_epoch = None
        self.actor = self.manager = None
        self.last_completed = self.last_saved = None
        self.last_normal_saved = None

    def on_train_ack(self, rollout_id):
        if not self.checkpoint_first_update or rollout_id != 0:
            return
        path = Path(self.args.save).parent / 'first-update-checkpoint.json'
        normal_save = self.periodic_save(
            rollout_id, self.args.save_interval, self.num_rollout_per_epoch, self.args.num_rollout)
        record = {
            'schema': 'eva.slime-first-update-checkpoint.v1', 'receipt_id': str(uuid4()),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'trigger': 'first_actual_train_acknowledged',
            'last_acknowledged_train_rollout_id': self.last_completed,
            'planned_rollouts': self.args.num_rollout, 'normal_save_interval': self.args.save_interval,
            'normal_checkpoint_boundary': normal_save, 'force_sync': not normal_save,
            'hf_export_requested': self.args.save_hf is not None,
            'dataset_cursor_saved_before_model': False, 'optimizer_restarted': False,
            'additional_optimizer_updates': 0, 'remaining_plan_unchanged': True,
            'independent_checkpoint_verification_required': True,
            'resume_semantics': 'model_only_warm_start_if_restarted; optimizer/RNG/dataset cursor reset',
        }
        with path.open('x') as stream:
            json.dump({**record, 'status': 'checking'}, stream, indent=2)
        try:
            if normal_save:
                # The unchanged Slime loop will save immediately after this
                # acknowledgement. Do not claim it already succeeded or save twice.
                record['status'] = 'delegated_to_normal_checkpoint_boundary'
            else:
                record['optimizer_prefix'] = completed_optimizer_prefix(
                    Path(os.environ['EVA_OPTIMIZER_RECEIPT']), rollout_id)
                record['status'] = 'saving'
                path.write_text(json.dumps(record, indent=2) + '\n')
                # Match the normal save's ordering. Rollout memory was already
                # offloaded before train; save_model owns actor wake/sleep and HF.
                if self.args.rollout_global_dataset:
                    self.ray.get(self.manager.save.remote(rollout_id))
                    record['dataset_cursor_saved_before_model'] = True
                self.actor.save_model(rollout_id, force_sync=True)
                self.last_saved = rollout_id
                record['saved_rollout_id'] = rollout_id
                record['status'] = 'save_returned_requires_independent_verification'
        except Exception as save_error:
            record['status'] = 'save_failed'
            record['checkpoint_error_type'] = type(save_error).__name__
            raise FirstUpdateCheckpointError('Acknowledged first update could not be saved') from save_error
        finally:
            record['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            path.write_text(json.dumps(record, indent=2) + '\n')

    def on_rollout_failure(self, rollout_id, error):
        # This callback runs ONLY after generate's Ray get has failed, while
        # the previous train get has already acknowledged every actor worker.
        path = Path(self.args.save).parent / 'abort-checkpoint.json'
        record = {
            'schema': 'eva.slime-abort-checkpoint.v1', 'receipt_id': str(uuid4()),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'trigger': 'next_rollout_generation_failed', 'failed_rollout_id': rollout_id,
            'original_error_type': type(error).__name__, 'training_status': 'failed',
            'last_acknowledged_train_rollout_id': self.last_completed,
            'last_normal_checkpoint_rollout_id': self.last_normal_saved,
            'force_sync': True, 'hf_export_requested': self.args.save_hf is not None,
            'dataset_cursor_saved_on_abort': False, 'exact_data_resume_supported': False,
            'resume_semantics': 'model_only_warm_start; optimizer/RNG/dataset cursor reset',
            'independent_checkpoint_verification_required': True,
        }
        with path.open('x') as stream:
            json.dump({**record, 'status': 'checking'}, stream, indent=2)
        try:
            if self.last_completed is None:
                record['status'] = 'not_saved_no_completed_train'
            elif self.last_completed == self.last_saved:
                record['status'] = 'not_saved_already_checkpointed'
            else:
                if rollout_id != self.last_completed + 1:
                    raise ValueError('Failed rollout is not immediately after the acknowledged train')
                record['optimizer_prefix'] = completed_optimizer_prefix(
                    Path(os.environ['EVA_OPTIMIZER_RECEIPT']), self.last_completed)
                record['status'] = 'saving'
                path.write_text(json.dumps(record, indent=2) + '\n')
                # Mirror the ordinary pre-save memory transition. This is not
                # rollout disposal or a new generation request.
                if self.args.offload_rollout:
                    self.ray.get(self.manager.offload.remote())
                # ActorGroup waits all worker saves; force_sync also drains DCP
                # async work. MegatronActor.save_model includes the HF exporter.
                self.actor.save_model(self.last_completed, force_sync=True)
                record['saved_rollout_id'] = self.last_completed
                record['status'] = 'save_returned_requires_independent_verification'
        except Exception as save_error:
            record['status'] = 'save_failed'
            record['checkpoint_error_type'] = type(save_error).__name__
            raise AbortCheckpointError('Rollout failed; saving its completed optimizer prefix also failed') from save_error
        finally:
            record['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            path.write_text(json.dumps(record, indent=2) + '\n')


class _Generate(_Delegate):
    def __init__(self, native, state):
        super().__init__(native)
        self.state = state

    def remote(self, rollout_id):
        try:
            ref = self.native.remote(rollout_id)
            self.state.ray.get(ref)
        except Exception as error:
            self.state.on_rollout_failure(rollout_id, error)
            raise
        # The original train loop retains its own get on the identical ref.
        # Ray caches the completed result; no generation RPC is repeated.
        return ref


class _Manager(_Delegate):
    def __init__(self, native, state):
        super().__init__(native)
        self.generate = _Generate(native.generate, state)


class _Actor(_Delegate):
    def __init__(self, native, state):
        super().__init__(native)
        self.state = state

    def async_train(self, rollout_id, *args, **kwargs):
        refs = self.native.async_train(rollout_id, *args, **kwargs)
        self.state.ray.get(refs)
        # Mark only after ALL actual training workers return successfully.
        self.state.last_completed = rollout_id
        self.state.on_train_ack(rollout_id)
        return refs

    def save_model(self, rollout_id, force_sync=False):
        result = self.native.save_model(rollout_id, force_sync=force_sync)
        self.state.last_saved = rollout_id
        self.state.last_normal_saved = rollout_id
        return result


def train_with_abort_checkpoint(args, namespace, *, checkpoint_first_update=False):
    """Run the real selected Slime loop with isolated driver-side bindings."""
    state = _AbortState(args, namespace['ray'], checkpoint_first_update=checkpoint_first_update,
                        periodic_save=namespace['should_run_periodic_action'])
    train = namespace['train']
    bindings = train.__globals__
    create_manager = bindings['create_rollout_manager']
    create_models = bindings['create_training_models']

    def manager_factory(*values, **keywords):
        native, per_epoch = create_manager(*values, **keywords)
        state.manager = native
        state.num_rollout_per_epoch = per_epoch
        return _Manager(native, state), per_epoch

    def model_factory(args, pgs, manager):
        # Never serialize these driver wrappers into a Ray model worker.
        actor, critic = create_models(args, pgs, manager.native)
        if critic is not None or args.start_rollout_id != 0:
            raise ValueError('Abort checkpoint requires actor-only rollout index zero')
        state.actor = actor
        return _Actor(actor, state), critic

    bindings['create_rollout_manager'] = manager_factory
    bindings['create_training_models'] = model_factory
    try:
        return train(args)
    finally:
        bindings['create_rollout_manager'] = create_manager
        bindings['create_training_models'] = create_models


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--checkpoint-first-update', action='store_true')
    wrapper_args, upstream_args = parser.parse_known_args()
    source = Path(__file__).resolve().parents[3] / 'slime-upstream' / 'train.py'
    namespace = runpy.run_path(str(source), run_name='eva_slime_selected_train')
    original_argv = sys.argv[:]
    try:
        sys.argv[1:] = upstream_args
        args = namespace['parse_args']()
    finally:
        sys.argv[:] = original_argv
    train_with_abort_checkpoint(args, namespace, checkpoint_first_update=wrapper_args.checkpoint_first_update)


if __name__ == '__main__':
    main()
