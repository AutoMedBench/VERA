"""Provider/GPU-free regression checks for the bounded validation command."""

from argparse import Namespace
from pathlib import Path
import unittest
import json
import tempfile
import contextlib
import io
import os
from unittest.mock import patch

from run_full_parameter import (RAY_ENVIRONMENT_KEYS, WORKSPACE, build_command,
                                cuda_runtime_environment, summarize_optimizer_events,
                                plan_sft_epoch, summarize_epoch_consumption)
from run_full_parameter import main


class LaunchCommandTests(unittest.TestCase):
    def command(self, stage, **extra):
        return build_command(Namespace(
            stage=stage, model=WORKSPACE / 'Qwen3.5-9B',
            load=Path('/tmp/verified-sft-checkpoints') if stage == 'grpo' else None,
            data=Path('/tmp/real-eva-data.jsonl'), output=Path('/tmp/new-eva-run'),
            steps=2, batch_size=4 if stage == 'grpo' else 2,
            max_tokens=24576 if stage == 'grpo' else 4096,
            samples_per_prompt=4, max_response_tokens=4096,
            generation_function='eva_agent.training.slime_rollout.generate_grpo_rollout',
            **extra,
        ))

    def test_decode_graphs_are_explicit_bounded_and_prefill_stays_disabled(self):
        command = self.command('grpo', decode_full_graphs=True)
        config = json.loads(command[command.index('--sglang-cuda-graph-config') + 1])
        self.assertEqual(config, {'decode': {'backend': 'full', 'bs': [1, 2, 4], 'max_bs': 4},
                                  'prefill': {'backend': 'disabled'}})
        default = self.command('grpo')
        config = json.loads(default[default.index('--sglang-cuda-graph-config') + 1])
        self.assertEqual(config['decode']['backend'], 'disabled')

    def test_first_update_checkpoint_only_goes_to_explicit_wrapper(self):
        original = self.command('grpo', save_on_rollout_error=True)
        selected = self.command('grpo', save_on_rollout_error=True, checkpoint_first_update=True)
        self.assertEqual(Path(selected[0]).name, 'train_abort_checkpoint.py')
        self.assertEqual(selected[1], '--checkpoint-first-update')
        self.assertEqual(selected[2:], original[1:])
        with self.assertRaisesRegex(ValueError, 'requires'):
            self.command('grpo', checkpoint_first_update=True)
        self.assertIn('EVA_SLIME_LOSS_MEMORY', RAY_ENVIRONMENT_KEYS)

    def test_abort_checkpoint_is_explicit_entrypoint_only_loss_argv_unchanged(self):
        default = self.command('grpo')
        enabled = self.command('grpo', save_on_rollout_error=True)
        self.assertEqual(Path(default[0]).name, 'train.py')
        self.assertEqual(Path(enabled[0]).name, 'train_abort_checkpoint.py')
        self.assertEqual(enabled[1:], default[1:])
        with self.assertRaisesRegex(ValueError, 'GRPO-only'):
            self.command('sft', save_on_rollout_error=True)

    def test_grpo_uses_sampled_logprobs_and_real_rollout(self):
        command = self.command('grpo')
        self.assertIn('--use-rollout-logprobs', command)
        self.assertEqual(command[command.index('--rollout-function-path') + 1],
                         'eva_agent.training.slime_rollout.generate_grpo_rollout')
        self.assertEqual(command[command.index('--n-samples-per-prompt') + 1], '4')
        self.assertEqual(command[command.index('--num-rollout') + 1], '2')
        self.assertIn('--finetune', command)
        self.assertIn('--no-load-optim', command)
        self.assertIn('--offload-optimizer-states', command)

    def test_sft_keeps_exact_masked_supervision(self):
        command = self.command('sft')
        self.assertNotIn('--use-rollout-logprobs', command)
        self.assertEqual(command[command.index('--loss-type') + 1], 'sft_loss')
        self.assertIn('--calculate-per-token-loss', command)
        self.assertEqual(command[command.index('--only-train-params-name-list') + 1],
                         r'^language_model\.(?!mtp\.)')

    def test_grpo_single_step_remainder_is_a_valid_cpu_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data.jsonl'
            data.write_text(json.dumps({'metadata': {'judge_backend': 'native_astra'}}) + '\n')
            argv = ['run_full_parameter.py', '--stage', 'grpo', '--data', str(data),
                    '--output', str(Path(directory) / 'new'), '--steps', '1',
                    '--batch-size', '4', '--samples-per-prompt', '4',
                    '--judge-backend', 'native_astra', '--checkpoint-model-only', '--dry-run',
                    '--reward-post-process-function', 'eva_agent.training.codex_segment_rewards.normalize_segment_rewards']
            output = io.StringIO()
            with patch('sys.argv', argv), contextlib.redirect_stdout(output):
                main()
            command = json.loads(output.getvalue())['argv']
            self.assertEqual(command[command.index('--num-rollout') + 1], '1')
            self.assertIn('--no-save-optim', command)
            self.assertEqual(command[command.index('--custom-reward-post-process-path') + 1],
                             'eva_agent.training.codex_segment_rewards.normalize_segment_rewards')

    def test_zero_grpo_groups_are_inconclusive_not_updates(self):
        event = dict(event='optimizer_step', successful_update=True, gradient_norm=0.0,
                     sampled_changed_values=0)
        result = summarize_optimizer_events([event, event], 'grpo', 2)
        self.assertEqual(result['status'], 'inconclusive')
        self.assertEqual(result['optimizer_step_executions'], 2)
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertEqual(result['zero_gradient_steps'], 2)
        with self.assertRaises(RuntimeError):
            summarize_optimizer_events([event, event], 'sft', 2)

    def test_invalid_gradients_and_skipped_updates_fail(self):
        for norm, success in [(float('nan'), True), (float('inf'), True), (0.0, False)]:
            event = dict(event='optimizer_step', successful_update=success, gradient_norm=norm,
                         sampled_changed_values=0)
            with self.assertRaises(RuntimeError):
                summarize_optimizer_events([event, event], 'grpo', 2)

    def test_cuda_runtime_path_precedes_and_preserves_existing_libraries(self):
        expected = str(WORKSPACE / '.venv/lib/python3.12/site-packages/nvidia/cu13/lib')
        self.assertEqual(cuda_runtime_environment({})['LD_LIBRARY_PATH'], expected)
        self.assertEqual(cuda_runtime_environment({'LD_LIBRARY_PATH': '/example/vendor'})['LD_LIBRARY_PATH'],
                         expected + ':/example/vendor')

    def test_cuda_runtime_path_is_propagated_to_ray_children(self):
        self.assertIn('PATH', RAY_ENVIRONMENT_KEYS)
        self.assertIn('LD_LIBRARY_PATH', RAY_ENVIRONMENT_KEYS)
        self.assertIn('CUDA_HOME', RAY_ENVIRONMENT_KEYS)
        self.assertIn('SLIME_DESTROY_WORLD_PROCESS_GROUP', RAY_ENVIRONMENT_KEYS)
        self.assertIn('EVA_GRPO_ENABLE_THINKING', RAY_ENVIRONMENT_KEYS)
        self.assertIn('EVA_GRPO_MAX_TOOL_FRONTIERS', RAY_ENVIRONMENT_KEYS)
        self.assertIn('EVA_GRPO_CODEX_BIN', RAY_ENVIRONMENT_KEYS)
        self.assertIn('EVA_GRPO_CONTEXT_PROFILE', RAY_ENVIRONMENT_KEYS)
        self.assertIn('EVA_SLIME_JUDGE_CONCURRENCY', RAY_ENVIRONMENT_KEYS)

    def test_explicit_judge_limit_is_propagated_and_rejected_before_data_or_gpu(self):
        with patch.dict(os.environ, {'EVA_SLIME_JUDGE_CONCURRENCY': '4'}):
            child = {key: os.environ[key] for key in RAY_ENVIRONMENT_KEYS if key in os.environ}
            self.assertEqual(child['EVA_SLIME_JUDGE_CONCURRENCY'], '4')
        from eva_agent.training.agent_judge import AgentJudgeSelectionError
        argv = ['run_full_parameter.py', '--stage', 'grpo', '--data', '/missing-data',
                '--output', '/unused-output', '--steps', '1', '--judge-backend', 'native_astra', '--dry-run']
        with patch('sys.argv', argv), patch.dict(os.environ, {'EVA_SLIME_JUDGE_CONCURRENCY': '5'}):
            with self.assertRaises(AgentJudgeSelectionError):
                main()

    def test_jit_executable_path_preserves_existing_path(self):
        expected = str(WORKSPACE / '.venv/bin')
        self.assertEqual(cuda_runtime_environment({})['PATH'], expected)
        self.assertEqual(cuda_runtime_environment({'PATH': '/usr/bin'})['PATH'], expected + ':/usr/bin')

    def test_native_codex_dry_run_requires_actor_and_reward_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data.jsonl'
            row = {'metadata': {'judge_backend': 'native_astra', 'actor_backend': 'native_codex_sglang'}}
            data.write_text(json.dumps(row) + '\n')
            argv = ['run_full_parameter.py', '--stage', 'grpo', '--data', str(data),
                    '--output', str(root / 'new'), '--steps', '50', '--batch-size', '4',
                    '--samples-per-prompt', '4', '--judge-backend', 'native_astra',
                    '--max-tokens', '24576', '--max-response-tokens', '8192',
                    '--generation-function', 'eva_agent.training.codex_slime_rollout.generate_grpo_rollout',
                    '--checkpoint-model-only', '--export-hf', '--save-interval', '25', '--dry-run']
            with patch('sys.argv', argv), patch.dict('os.environ', {'EVA_GRPO_CODEX_BIN': '/bin/true'}):
                with self.assertRaisesRegex(ValueError, 'normalization'):
                    main()
            argv += ['--reward-post-process-function', 'eva_agent.training.codex_segment_rewards.normalize_segment_rewards']
            output = io.StringIO()
            with patch('sys.argv', argv), patch.dict('os.environ', {'EVA_GRPO_CODEX_BIN': '/bin/true'}), contextlib.redirect_stdout(output):
                main()
            command = json.loads(output.getvalue())['argv']
            self.assertEqual(command[command.index('--num-rollout') + 1], '50')
            self.assertEqual(command[command.index('--save-interval') + 1], '25')
            profile_env = {'EVA_GRPO_CODEX_BIN': '/bin/true',
                           'EVA_GRPO_CONTEXT_PROFILE': 'evamed-grpo-native-24576-v1'}
            with patch('sys.argv', argv), patch.dict('os.environ', profile_env), contextlib.redirect_stdout(io.StringIO()):
                main()
            early_profile_env = {
                'EVA_GRPO_CODEX_BIN': '/bin/true',
                'EVA_GRPO_CONTEXT_PROFILE': 'evamed-grpo-native-24576-earlycompact-v2',
            }
            with patch('sys.argv', argv), patch.dict('os.environ', early_profile_env), contextlib.redirect_stdout(io.StringIO()):
                main()
            invalid = list(argv)
            invalid[invalid.index('--max-response-tokens') + 1] = '4096'
            with patch('sys.argv', invalid), patch.dict('os.environ', profile_env):
                with self.assertRaisesRegex(ValueError, '8192 trajectory'):
                    main()
            row['metadata']['actor_backend'] = 'sglang_evamed_tools'
            data.write_text(json.dumps(row) + '\n')
            with patch('sys.argv', argv), patch.dict('os.environ', {'EVA_GRPO_CODEX_BIN': '/bin/true'}):
                with self.assertRaisesRegex(ValueError, 'native Codex actor'):
                    main()

    def epoch_args(self, data, count=16, batch=8, cap=500):
        rows = [{'metadata': {'row_id': str(i), 'tokens': [1, 2, 3],
                             'loss_mask': [0, 1, 1], 'response_length': 2}}
                for i in range(count)]
        data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return Namespace(one_epoch=True, stage='sft', load=None, data=data,
                         batch_size=batch, max_updates=cap, max_tokens=24576)

    def test_epoch_exact_pass_and_update_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.epoch_args(Path(directory) / 'data.jsonl', count=1368)
            plan = plan_sft_epoch(args)
            self.assertEqual(plan['optimizer_updates_planned'], 171)
            self.assertEqual(plan['samples_planned'], 1368)
            self.assertTrue(plan['complete_epoch_planned'])
            args.max_updates = 100
            plan = plan_sft_epoch(args)
            self.assertEqual(plan['samples_planned'], 800)
            self.assertFalse(plan['complete_epoch_planned'])

    def test_epoch_rejects_remainder_duplicates_and_warmstart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            args = self.epoch_args(path, count=17)
            with self.assertRaises(ValueError):
                plan_sft_epoch(args)
            args = self.epoch_args(path)
            with path.open('a') as stream:
                stream.write(path.read_text().splitlines()[0] + '\n')
            with self.assertRaises(ValueError):
                plan_sft_epoch(args)
            args = self.epoch_args(path)
            args.load = Path('/tmp/prior-checkpoint')
            with self.assertRaises(ValueError):
                plan_sft_epoch(args)

    def test_checkpoint_final_only_and_model_only_command(self):
        args = Namespace(stage='sft', model=WORKSPACE / 'Qwen3.5-9B', load=None,
                         data=Path('/tmp/input'), output=Path('/tmp/output'), steps=171,
                         batch_size=8, max_tokens=24576, one_epoch=True, save_interval=0,
                         checkpoint_model_only=True, export_hf=True)
        command = build_command(args)
        self.assertEqual(command[command.index('--save-interval') + 1], '171')
        self.assertIn('--no-save-optim', command)
        self.assertIn('--save-hf', command)
        self.assertEqual(command[command.index('--rollout-function-path') + 1], 'epoch_data.generate_sft_rollout')
        self.assertNotIn('--finetune', command)

    def test_consumption_requires_distinct_consecutive_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'consumption.jsonl'
            plan = {'optimizer_updates_planned': 2, 'samples_planned': 4, 'complete_epoch_planned': True}
            records = [dict(training_iteration=i, epoch_id=0, row_ids=[str(i*2), str(i*2+1)],
                            tokens=6, supervised_tokens=4) for i in range(2)]
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            self.assertEqual(summarize_epoch_consumption(path, plan)['unique_samples_consumed'], 4)
            records[1]['row_ids'][0] = '0'
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            with self.assertRaises(RuntimeError):
                summarize_epoch_consumption(path, plan)


if __name__ == '__main__':
    unittest.main()
