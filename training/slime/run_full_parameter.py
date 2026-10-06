"""Run bounded EVA SFT/GRPO validation with Slime and Megatron on one GB300.

Creates a dedicated local Ray instance. Shared source checkouts and installed
packages are read only. Generated data, optimizer receipts, and checkpoints live
under the requested output directory.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import uuid


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
SLIME = WORKSPACE / 'slime-upstream'
MEGATRON = WORKSPACE / 'Megatron-LM'
CODEX_SEGMENT_REWARD_FUNCTION = 'eva_agent.training.codex_segment_rewards.normalize_segment_rewards'
CODEX_GENERATION_FUNCTION = 'eva_agent.training.codex_slime_rollout.generate_grpo_rollout'
NATIVE_CONTEXT_PROFILE_TOKENS = {
    'evamed-grpo-native-24576-v1': 24576,
    'evamed-grpo-native-24576-earlycompact-v2': 24576,
    'evamed-grpo-native-32768-lossless-headroom-v3': 32768,
}
RAY_ENVIRONMENT_KEYS = (
    'PYTHONPATH', 'PATH', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'CUDA_DEVICE_MAX_CONNECTIONS', 'NCCL_NVLS_ENABLE',
    'SGLANG_ENABLE_JIT_DEEPGEMM', 'EVA_OPTIMIZER_RECEIPT', 'TOKENIZERS_PARALLELISM',
    'SLIME_DESTROY_WORLD_PROCESS_GROUP', 'PYTHONUNBUFFERED', 'no_proxy',
    'EVA_SLIME_JUDGE_BACKEND', 'EVA_SLIME_NATIVE_AUTH_PATH', 'EVA_SLIME_JUDGE_CONCURRENCY',
    'EVA_GRPO_ENABLE_THINKING', 'EVA_GRPO_MAX_TOOL_FRONTIERS', 'EVA_GRPO_CODEX_BIN', 'EVA_HARNESS_ROOT',
    'EVA_GRPO_CONTEXT_PROFILE',
    'EVA_SLIME_LOSS_MEMORY',
    'EVA_GRPO_JUDGE_GROUP_REPLACEMENTS',
    'EVA_MEDRESEARCH_DATA_ROOT',
    'EVA_RSI_SKILL_SELECTION', 'EVA_RSI_SKILL_SELECTION_BLAKE3', 'EVA_RSI_SKILL_CATALOG_ID',
    'EVA_SFT_EPOCH_ROWS', 'EVA_SFT_EPOCH_STEPS', 'EVA_SFT_DATA_RECEIPT',
)


def validate_native_context_profile(max_tokens, max_response_tokens, profile):
    expected_context = NATIVE_CONTEXT_PROFILE_TOKENS.get(profile, 24576)
    if not 8192 <= max_tokens <= expected_context or max_response_tokens < 4096:
        raise ValueError('Native Codex context/output budgets differ from reviewed bounds')
    if profile is not None and (
        profile not in NATIVE_CONTEXT_PROFILE_TOKENS
        or max_tokens != NATIVE_CONTEXT_PROFILE_TOKENS[profile]
        or max_response_tokens != 8192
    ):
        raise ValueError(
            'Measured native GRPO profile requires matched context and 8192 trajectory budget'
        )


def cuda_runtime_environment(inherited=None):
    inherited = os.environ if inherited is None else inherited
    directory = WORKSPACE / '.venv/lib/python3.12/site-packages/nvidia/cu13'
    return {
        'CUDA_HOME': str(directory),
        'PATH': str(WORKSPACE / '.venv/bin') +
        (os.pathsep + inherited['PATH'] if inherited.get('PATH') else ''),
        'LD_LIBRARY_PATH': str(directory / 'lib') +
        (os.pathsep + inherited['LD_LIBRARY_PATH'] if inherited.get('LD_LIBRARY_PATH') else ''),
    }


def capture_eva_source_provenance(logical_paths, *, repo=REPO):
    """Keep historical receipt keys, hash the package actually selected at bootstrap."""
    import eva_agent
    from blake3 import blake3

    repo = Path(repo).resolve()
    package = Path(eva_agent.__file__).resolve().parent
    hashes, actual_paths = {}, {}
    for path in logical_paths:
        relative = Path(path).relative_to(repo)
        key = relative.as_posix()
        if relative.parts[:2] == ('src', 'eva_agent'):
            actual = package.joinpath(*relative.parts[2:]).resolve(strict=True)
        else:
            actual = Path(path).resolve(strict=True)
        hashes[key] = blake3(actual.read_bytes()).hexdigest()
        actual_paths[key] = str(actual)
    return hashes, actual_paths


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['sft', 'grpo'])
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=WORKSPACE / 'Qwen3.5-9B')
    parser.add_argument('--load', type=Path)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--one-epoch', action='store_true', help='SFT only: stop after one unique data pass or max-updates')
    parser.add_argument('--max-updates', type=int, default=500)
    parser.add_argument('--save-interval', type=int, default=1, help='0 saves only at the final update')
    parser.add_argument('--checkpoint-model-only', action='store_true', help='Omit optimizer state; not exact training-resume checkpoints')
    parser.add_argument('--export-hf', action='store_true', help='Also export HF weights at each checkpoint boundary')
    parser.add_argument('--save-on-rollout-error', action='store_true',
                        help='GRPO only: save a completed unsaved optimizer prefix before propagating the next rollout failure')
    parser.add_argument('--checkpoint-first-update', action='store_true',
                        help='With the abort-save wrapper, also save the first acknowledged update')
    parser.add_argument('--judge-group-replacements', type=int, default=0,
                        help='Opt-in native GRPO: collect up to 3 fresh four-sample groups after completed Judge verdict failures')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--generation-function', default='eva_agent.training.slime_rollout.generate_grpo_rollout')
    parser.add_argument('--reward-post-process-function', choices=[CODEX_SEGMENT_REWARD_FUNCTION],
                        help='Explicit original-trajectory normalization for segmented native Codex rollouts')
    parser.add_argument('--samples-per-prompt', type=int, default=2)
    parser.add_argument('--max-response-tokens', type=int, default=1024)
    parser.add_argument('--judge-backend', choices=['opus_5', 'native_astra'])
    parser.add_argument('--decode-full-graphs', action='store_true',
                        help='Prospective GRPO trial: full decode graphs for batch 1/2/4, prefill disabled')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def plan_sft_epoch(args):
    """Audit the exact pre-tokenized rows before any GPU/process allocation."""
    if not getattr(args, 'one_epoch', False):
        return None
    if args.stage != 'sft' or args.load is not None:
        raise ValueError('One-epoch mode requires clean-base SFT without --load')
    if args.max_updates < 1 or args.batch_size < 1:
        raise ValueError('Epoch update cap and batch size must be positive')
    seen, token_count, supervised_count, max_length = set(), 0, 0, 0
    with args.data.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            meta = json.loads(line)['metadata']
            row_id, tokens, mask = meta['row_id'], meta['tokens'], meta['loss_mask']
            if not isinstance(row_id, str) or not row_id or row_id in seen:
                raise ValueError('Epoch input requires distinct nonempty row IDs')
            if len(tokens) != len(mask) or not any(mask) or any(x not in (0, 1) for x in mask):
                raise ValueError('Epoch input token/loss alignment is invalid')
            if len(tokens) > args.max_tokens or meta['response_length'] != len(tokens) - mask.index(1):
                raise ValueError('Epoch input exceeds token ceiling or has an invalid response boundary')
            seen.add(row_id)
            token_count += len(tokens)
            supervised_count += sum(mask)
            max_length = max(max_length, len(tokens))
    full_batches, remainder = divmod(len(seen), args.batch_size)
    if not seen or full_batches < 1:
        raise ValueError('Epoch input contains no complete batch')
    if remainder and full_batches < args.max_updates:
        raise ValueError('Choose a batch-size divisor: one epoch must not drop or repeat remainder rows')
    steps = min(full_batches, args.max_updates)
    return {
        'mode': 'one_epoch_or_update_cap', 'dataset_rows': len(seen),
        'global_batch_size': args.batch_size, 'optimizer_updates_planned': steps,
        'update_cap': args.max_updates, 'samples_planned': steps * args.batch_size,
        'complete_epoch_planned': steps * args.batch_size == len(seen),
        'repeated_samples_allowed': False, 'truncated_rows': 0,
        'dataset_tokens': token_count, 'dataset_supervised_tokens': supervised_count,
        'max_observed_tokens': max_length,
    }


def summarize_epoch_consumption(path, plan):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    row_ids = [row_id for record in records for row_id in record['row_ids']]
    if len(records) != plan['optimizer_updates_planned'] or len(row_ids) != plan['samples_planned']:
        raise RuntimeError('Epoch consumption count differs from the bounded schedule')
    if len(set(row_ids)) != len(row_ids) or any(record['epoch_id'] != 0 for record in records):
        raise RuntimeError('Epoch data repeated or wrapped')
    if [record['training_iteration'] for record in records] != list(range(len(records))):
        raise RuntimeError('Epoch consumption iterations are not consecutive')
    return {'unique_samples_consumed': len(row_ids), 'repeated_samples': 0,
            'epochs_wrapped': 0, 'complete_epoch_consumed': plan['complete_epoch_planned'],
            'tokens_consumed': sum(record['tokens'] for record in records),
            'supervised_tokens_consumed': sum(record['supervised_tokens'] for record in records)}


def build_command(args):
    from judge_group_isolation import GENERATION_FUNCTION, replacement_limit
    replacements = replacement_limit(getattr(args, 'judge_group_replacements', 0))
    abort_save = getattr(args, 'save_on_rollout_error', False)
    checkpoint_first = getattr(args, 'checkpoint_first_update', False)
    if checkpoint_first and not abort_save:
        raise ValueError('Checkpoint-first-update requires --save-on-rollout-error')
    if abort_save and args.stage != 'grpo':
        raise ValueError('Save-on-rollout-error is an explicit GRPO-only policy')
    if replacements and (args.stage != 'grpo' or not abort_save
                         or args.generation_function != CODEX_GENERATION_FUNCTION
                         or args.samples_per_prompt != 4 or args.batch_size != 4):
        raise ValueError('Judge group isolation requires native Codex GRPO, group4 and --save-on-rollout-error')
    save_interval = getattr(args, 'save_interval', 1)
    if save_interval < 0:
        raise ValueError('Save interval must be nonnegative')
    save_interval = save_interval or args.steps
    model_args = [
        '--spec', 'slime_plugins.models.qwen3_5', 'get_qwen3_5_spec',
        '--disable-bias-linear', '--qk-layernorm', '--group-query-attention',
        '--num-attention-heads', '16', '--num-query-groups', '4', '--kv-channels', '256',
        '--num-layers', '32', '--hidden-size', '4096', '--ffn-hidden-size', '12288',
        '--use-gated-attention', '--normalization', 'RMSNorm', '--apply-layernorm-1p',
        '--position-embedding-type', 'rope', '--norm-epsilon', '1e-6',
        '--rotary-percent', '0.25', '--swiglu', '--untie-embeddings-and-output-weights',
        '--vocab-size', '248320', '--rotary-base', '10000000', '--attention-output-gate',
    ]
    common = [
        '--train-backend', 'megatron', '--megatron-to-hf-mode', 'bridge', '--bf16',
        '--actor-num-nodes', '1', '--actor-num-gpus-per-node', '1', '--num-gpus-per-node', '1',
        '--tensor-model-parallel-size', '1', '--pipeline-model-parallel-size', '1',
        '--context-parallel-size', '1', '--expert-model-parallel-size', '1',
        '--expert-tensor-parallel-size', '1',
        '--hf-checkpoint', str(args.model.resolve()), '--load', str((args.load or args.model).resolve()),
        '--save', str(args.output.resolve() / 'checkpoints'), '--save-interval', str(save_interval),
        '--prompt-data', str(args.data.resolve()), '--input-key', 'prompt', '--metadata-key', 'metadata',
        # Qwen ships a vision processor even for these text-only samples. An
        # empty media map gives its loader a conversation wrapper; our hooks
        # consume the original exact token metadata and build their own prompt.
        '--multimodal-keys', '{}',
        '--num-rollout', str(args.steps), '--global-batch-size', str(args.batch_size),
        '--micro-batch-size', '1', '--num-steps-per-rollout', '1',
        '--recompute-granularity', 'full', '--recompute-method', 'uniform', '--recompute-num-layers', '1',
        '--use-dynamic-batch-size', '--max-tokens-per-gpu', str(args.max_tokens),
        '--log-probs-max-tokens-per-gpu', str(args.max_tokens), '--log-probs-chunk-size', '256',
        '--seq-length', str(args.max_tokens), '--attention-dropout', '0', '--hidden-dropout', '0',
        '--accumulate-allreduce-grads-in-fp32', '--attention-softmax-in-fp32', '--attention-backend', 'flash',
        '--optimizer', 'adam', '--offload-optimizer-states', '--lr', '2e-6', '--lr-decay-style', 'constant',
        '--weight-decay', '0', '--adam-beta1', '0.9', '--adam-beta2', '0.95', '--adam-eps', '1e-8',
        '--clip-grad', '1', '--seed', '260910', '--log-interval', '1',
        '--only-train-params-name-list', r'^language_model\.(?!mtp\.)',
        '--custom-megatron-before-train-step-hook-path', 'runtime_hooks.before_train_step',
    ]
    if args.stage == 'sft':
        stage = [
            '--rollout-function-path', 'epoch_data.generate_sft_rollout' if getattr(args, 'one_epoch', False)
            else 'eva_agent.training.slime_data.generate_sft_rollout',
            '--rollout-batch-size', str(args.batch_size), '--n-samples-per-prompt', '1',
            '--loss-type', 'sft_loss', '--loss-mask-type', 'qwen3_5', '--calculate-per-token-loss',
            '--disable-compute-advantages-and-returns', '--debug-train-only',
        ]
        if getattr(args, 'one_epoch', False):
            stage += ['--rollout-shuffle', '--rollout-seed', '260910']
    else:
        if args.batch_size % args.samples_per_prompt:
            raise ValueError('GRPO batch must be divisible by samples per prompt')
        stage = [
            '--rollout-function-path', GENERATION_FUNCTION if replacements else args.generation_function,
            '--rollout-batch-size', str(args.batch_size // args.samples_per_prompt),
            '--n-samples-per-prompt', str(args.samples_per_prompt), '--rollout-temperature', '0.8',
            '--rollout-max-response-len', str(args.max_response_tokens),
            '--use-rollout-logprobs',
            '--advantage-estimator', 'grpo', '--kl-coef', '0', '--kl-loss-coef', '0',
            '--entropy-coef', '0', '--eps-clip', '0.2',
            '--colocate', '--rollout-num-gpus-per-engine', '1', '--sglang-mem-fraction-static', '0.25',
            '--sglang-context-length', str(args.max_tokens), '--sglang-max-running-requests', '4',
            '--sglang-chunked-prefill-size', '2048',
            '--sglang-model-loader-extra-config', '{"enable_multithread_load":false}',
            '--sglang-tool-call-parser', 'qwen3_coder', '--sglang-reasoning-parser', 'qwen3',
            '--sglang-cuda-graph-config', json.dumps({
                'decode': {'backend': 'full', 'bs': [1, 2, 4], 'max_bs': 4}
                if getattr(args, 'decode_full_graphs', False) else {'backend': 'disabled'},
                'prefill': {'backend': 'disabled'}}, separators=(',', ':')),
        ]
    if args.load is not None:
        # Every explicit load is a documented model-only warm start. This
        # includes a failed-attempt checkpoint as well as SFT -> GRPO: reset
        # optimizer, RNG, rollout counter, and data source to their initial state.
        stage += ['--finetune', '--no-load-optim', '--no-load-rng', '--start-rollout-id', '0']
    if getattr(args, 'checkpoint_model_only', False):
        stage += ['--no-save-optim']
    if getattr(args, 'export_hf', False):
        stage += ['--save-hf', str(args.output.resolve() / 'hf' / 'iter_{rollout_id:07d}')]
    if args.stage == 'grpo' and getattr(args, 'reward_post_process_function', None):
        stage += ['--custom-reward-post-process-path', args.reward_post_process_function]
    entrypoint = REPO / 'training/slime/train_abort_checkpoint.py' if abort_save else SLIME / 'train.py'
    return [str(entrypoint), *(['--checkpoint-first-update'] if checkpoint_first else []), *model_args, *common, *stage]


def summarize_optimizer_events(events, stage, requested_steps):
    """Separate executed optimizer steps from evidence of a learning update."""
    updates = [event for event in events if event['event'] == 'optimizer_step']
    if len(updates) != requested_steps:
        raise RuntimeError('Requested optimizer steps were not all observed')
    for event in updates:
        norm = event.get('gradient_norm')
        if not event['successful_update'] or norm is None or not math.isfinite(norm) or norm < 0:
            raise RuntimeError('Optimizer skipped a step or had invalid gradients')
    nonzero = [event for event in updates if event['gradient_norm'] > 0]
    learning = [event for event in nonzero if event['sampled_changed_values'] > 0]
    if stage == 'sft' and len(learning) != requested_steps:
        raise RuntimeError('SFT requires nonzero gradients and observed model-weight changes')
    complete = len(learning) == requested_steps
    return {
        'optimizer_step_executions': len(updates),
        'optimizer_updates': len(learning),
        'zero_gradient_steps': len(updates) - len(nonzero),
        'all_updates_have_finite_nonzero_gradient': len(nonzero) == requested_steps,
        'model_weight_changes_observed': bool(learning),
        'learning_signal_outcome': 'validated' if complete else 'inconclusive_no_learning_signal'
        if not learning else 'inconclusive_insufficient_learning_updates',
        'status': 'complete' if complete else 'inconclusive',
    }


def main():
    args = arguments()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError('Steps and batch size must be positive')
    if args.stage == 'grpo':
        if args.judge_backend is None:
            raise ValueError('GRPO is held until an explicit verified --judge-backend is selected')
        if args.steps < 1:
            raise ValueError('GRPO requires at least one planned optimizer step')
        if 'EVA_SLIME_JUDGE_CONCURRENCY' in os.environ:
            # Validate an explicit launch selection before checkpoint/GPU work.
            # Unset historical launches do not require the new harness helper.
            from eva_agent.training.slime_agent_judge import resolve_judge_concurrency
            resolve_judge_concurrency()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    epoch_plan = plan_sft_epoch(args)
    if epoch_plan is not None:
        args.steps = epoch_plan['optimizer_updates_planned']
    if args.stage == 'grpo':
        rows = 0
        native_codex = args.generation_function == CODEX_GENERATION_FUNCTION
        if native_codex:
            if args.reward_post_process_function != CODEX_SEGMENT_REWARD_FUNCTION:
                raise ValueError('Native Codex segments require original-trajectory reward normalization')
            if os.environ.get('EVA_GRPO_ENABLE_THINKING', '1') != '1':
                raise ValueError('Native Codex GRPO requires thinking enabled')
            binary = os.environ.get('EVA_GRPO_CODEX_BIN')
            if not binary or not Path(binary).is_file() or not os.access(binary, os.X_OK):
                raise ValueError('Native Codex requires explicit EVA_GRPO_CODEX_BIN before GPU allocation')
            profile = os.environ.get('EVA_GRPO_CONTEXT_PROFILE')
            validate_native_context_profile(
                args.max_tokens,
                args.max_response_tokens,
                profile,
            )
        with args.data.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('metadata', {}).get('judge_backend') != args.judge_backend:
                    raise ValueError('GRPO input provenance differs from the explicit judge backend')
                if native_codex and row.get('metadata', {}).get('actor_backend') != 'native_codex_sglang':
                    raise ValueError('GRPO input provenance differs from the native Codex actor')
                rows += 1
        if rows == 0:
            raise ValueError('GRPO input contains no real task')
    command = build_command(args)
    if args.stage == 'grpo' and '--use-rollout-logprobs' not in command:
        raise RuntimeError('GRPO must use the actual sampled policy log probabilities')
    if args.dry_run:
        print(json.dumps({'argv': command, 'epoch_plan': epoch_plan}, indent=2))
        return
    checkpoint_verification = None
    if args.load is not None:
        from checkpoint_preflight import verify_checkpoint
        checkpoint_verification = verify_checkpoint(args.load, args.model)
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be new or empty; use a new version for another attempt')
    gpu_inventory = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,driver_version',
        '--format=csv,noheader',
    ], text=True).strip().splitlines()
    if len(gpu_inventory) != 1:
        raise RuntimeError('This validation launcher requires exactly one visible physical GPU')
    existing_compute = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader',
    ], text=True).strip()
    if existing_compute:
        raise RuntimeError('GPU compute processes are active; release the owned serving endpoint before training')
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO))
    from training.harness_source import activate_harness
    paths = activate_harness(REPO) + [str(REPO / 'training/slime'), str(SLIME), str(MEGATRON)]
    for path in reversed(paths):
        sys.path.insert(0, path)
    os.environ.update({
        'PYTHONPATH': os.pathsep.join(paths),
        'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_NVLS_ENABLE': '0',
        # The reloadable Python singleton PG segfaults in Torch 2.13's
        # synchronous coalesced reduce-scatter wait. Keep native WORLD/groups
        # for this one-GPU run; model/optimizer offload remains enabled for GRPO.
        'SLIME_DESTROY_WORLD_PROCESS_GROUP': '0',
        # SGLang memory-saver preloads a CUDA-linked library before Python can
        # import Torch. CUDA_HOME does not participate in ELF loader lookup.
        **cuda_runtime_environment(),
        'SGLANG_ENABLE_JIT_DEEPGEMM': '0',
        'EVA_OPTIMIZER_RECEIPT': str(args.output / 'optimizer-steps.jsonl'),
        'RAY_USAGE_STATS_ENABLED': '0',
    })
    # Proxy endpoints and credentials are inherited but never copied into receipts.
    no_proxy = os.environ.get('no_proxy', '')
    os.environ['no_proxy'] = 'localhost,127.0.0.1' + (',' + no_proxy if no_proxy else '')
    if args.stage == 'grpo':
        os.environ['EVA_SLIME_JUDGE_BACKEND'] = args.judge_backend
        os.environ['EVA_GRPO_JUDGE_GROUP_REPLACEMENTS'] = str(args.judge_group_replacements)
    if epoch_plan is not None:
        os.environ.update({
            'EVA_SFT_EPOCH_ROWS': str(epoch_plan['dataset_rows']),
            'EVA_SFT_EPOCH_STEPS': str(args.steps),
            'EVA_SFT_DATA_RECEIPT': str(args.output / 'data-consumption.jsonl'),
        })
    packages = {name: importlib.metadata.version(name) for name in (
        'torch', 'transformers', 'slime', 'sglang', 'megatron-core', 'megatron-bridge',
        'transformer-engine', 'ray', 'flash-linear-attention', 'causal-conv1d',
        'numpy', 'flash-attn-4',
    )}
    lock = json.loads((REPO / 'training/slime/runtime-lock.json').read_text())
    if packages != lock['packages']:
        raise RuntimeError('Installed package versions differ from the reviewed runtime lock')
    from blake3 import blake3
    eva_source_paths = [
        Path(__file__).resolve(), REPO / 'training/slime/runtime_hooks.py',
        REPO / 'training/slime/loss_memory.py',
        REPO / 'src/eva_agent/training/slime_data.py',
    ]
    if args.save_on_rollout_error:
        eva_source_paths.append(REPO / 'training/slime/train_abort_checkpoint.py')
    if args.judge_group_replacements:
        eva_source_paths.append(REPO / 'training/slime/judge_group_isolation.py')
    if args.load is not None:
        eva_source_paths.append(REPO / 'training/slime/checkpoint_preflight.py')
    if epoch_plan is not None:
        eva_source_paths.append(REPO / 'training/slime/epoch_data.py')
        if args.load is None:
            eva_source_paths.append(REPO / 'training/slime/checkpoint_preflight.py')
    if args.stage == 'grpo':
        eva_source_paths.extend([
            REPO / 'src/eva_agent/training/slime_rollout.py',
            REPO / 'src/eva_agent/training/slime_agent_judge.py',
        ])
        if args.generation_function == CODEX_GENERATION_FUNCTION:
            eva_source_paths.extend([
                REPO / 'src/eva_agent/training/codex_slime_rollout.py',
                REPO / 'src/eva_agent/training/codex_sglang_transport.py',
                REPO / 'src/eva_agent/training/qwen_tool_types.py',
                REPO / 'src/eva_agent/training/teacher_focus.py',
                REPO / 'src/eva_agent/training/stage_execution_guidance.py',
                REPO / 'src/eva_agent/training/teacher_worker.py',
                REPO / 'src/eva_agent/codex_pipeline/adapter.py',
                REPO / 'src/eva_agent/codex_pipeline/budget.py',
                REPO / 'training/automedbench_lite/local_qwen.py',
                REPO / 'training/eva_rsi/skill_selection.py',
            ])
    if args.reward_post_process_function:
        eva_source_paths.append(REPO / 'src/eva_agent/training/codex_segment_rewards.py')
    eva_source_blake3, actual_eva_source_paths = capture_eva_source_provenance(eva_source_paths)
    source_commits = {}
    source_diff_blake3 = {}
    for source in (SLIME, MEGATRON):
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != lock['source_commits'][source.name]:
            raise RuntimeError('Source revision differs from the reviewed runtime lock')
        source_commits[str(source)] = commit
        diff = subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--binary'])
        source_diff_blake3[str(source)] = blake3(diff).hexdigest()
    receipt = {
        'receipt_id': str(uuid.uuid4()), 'stage': args.stage,
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'status': 'running', 'argv': command, 'packages': packages,
        'judge_backend': args.judge_backend if args.stage == 'grpo' else None,
        'source_commits': source_commits, 'source_diff_blake3': source_diff_blake3,
        'eva_source_blake3': eva_source_blake3,
        'eva_source_paths': actual_eva_source_paths,
        'slime_train_entrypoint': {'path': str(SLIME / 'train.py'),
                                   'blake3': blake3((SLIME / 'train.py').read_bytes()).hexdigest()},
        'training_data_blake3': blake3(args.data.read_bytes()).hexdigest(),
        'gpu_inventory_csv': gpu_inventory,
        'initial_gpu_compute_processes': 0,
        'full_parameter_scope': 'all 8,953,803,264 language parameters; auxiliary vision/MTP frozen for text data',
        'checkpoint_verification': checkpoint_verification,
        'epoch_plan': epoch_plan,
        'checkpoint_policy': {
            'checkpoint_first_update': args.checkpoint_first_update,
            'interval': args.save_interval, 'final_save_required': True,
            'optimizer_state_saved': not args.checkpoint_model_only,
            # The wrapper always uses model-only --finetune for explicit loads,
            # even when an upstream checkpoint happens to contain full state.
            'exact_optimizer_resume_supported': False,
            'hf_export_at_checkpoint_boundaries': args.export_hf,
            'save_on_rollout_error': args.save_on_rollout_error,
            'abort_save_scope': 'prior_acknowledged_finite_optimizer_prefix_only'
            if args.save_on_rollout_error else None,
        },
        'checkpoint_transition': {
            'mode': 'model_only_finetune' if args.load is not None else 'official_hf_initialization',
            'optimizer_state_resumed': False, 'rng_state_resumed': False,
            'starting_rollout_id': 0, 'starting_dataset_offset': 0,
            'repeated_training_examples_possible': args.load is not None,
        },
    }
    path = args.output / 'run-receipt.json'
    if args.judge_group_replacements:
        from judge_group_isolation import POLICY
        receipt['judge_group_isolation'] = {
            'policy': POLICY, 'replacement_limit': args.judge_group_replacements,
            'accepted_evidence_lookup': 'rollouts/{optimizer_rollout_id:06d}/isolation.json',
            'rejected_groups_are_optimizer_updates': False,
            'verdict_retries': 0, 'same_live_optimizer': True,
        }
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    import ray
    ray_tmp = tempfile.mkdtemp(prefix=f'eva-{args.stage}-ray-', dir='/tmp')
    try:
        ray.init(address='local', num_gpus=1, num_cpus=8, include_dashboard=False, _node_ip_address='127.0.0.1',
                 _temp_dir=ray_tmp, object_store_memory=2 * 1024**3,
                 runtime_env={'env_vars': {key: os.environ[key] for key in RAY_ENVIRONMENT_KEYS
                                           if key in os.environ}})
        sys.argv = command
        runpy.run_path(command[0], run_name='__main__')
        events = [json.loads(line) for line in (args.output / 'optimizer-steps.jsonl').read_text().splitlines()]
        optimizer_validation = summarize_optimizer_events(events, args.stage, args.steps)
        checkpoint_root = args.output / 'checkpoints'
        tracker = checkpoint_root / 'latest_checkpointed_iteration.txt'
        if not tracker.is_file() or not tracker.read_text().strip().isdigit():
            raise RuntimeError('No completed Megatron checkpoint tracker exists')
        if int(tracker.read_text().strip()) != args.steps - 1:
            raise RuntimeError('Completed checkpoint does not contain the final requested optimizer update')
        checkpoint_files = [p for p in checkpoint_root.rglob('*') if p.is_file()]
        if not any(p.suffix == '.distcp' for p in checkpoint_files):
            raise RuntimeError('No distributed model checkpoint shard was saved')
        receipt['validation'] = {
            **optimizer_validation,
            'latest_checkpoint_iteration': int(tracker.read_text().strip()),
            'checkpoint_file_count': len(checkpoint_files),
            'checkpoint_bytes': sum(p.stat().st_size for p in checkpoint_files),
        }
        if epoch_plan is not None:
            receipt['validation']['data_consumption'] = summarize_epoch_consumption(
                args.output / 'data-consumption.jsonl', epoch_plan)
        # GRPO durability needs the same fail-closed LM key/shape/storage audit
        # as SFT, irrespective of a successful optimizer return or DCP tracker.
        from checkpoint_preflight import verify_checkpoint
        receipt['validation']['final_language_checkpoint_verification'] = verify_checkpoint(
            checkpoint_root, args.model)
        if args.export_hf:
            hf_directory = args.output / 'hf' / f'iter_{args.steps - 1:07d}'
            if (not (hf_directory / 'config.json').is_file()
                    or not (hf_directory / 'tokenizer_config.json').is_file()
                    or not list(hf_directory.glob('*.safetensors'))):
                raise RuntimeError('Final native HF export is incomplete')
            receipt['validation']['final_hf_export'] = str(hf_directory)
        receipt['status'] = optimizer_validation['status']
    except BaseException as exc:
        receipt['status'] = 'failed'
        receipt['error_type'] = type(exc).__name__
        raise
    finally:
        ray.shutdown()
        if args.save_on_rollout_error:
            abort_path = args.output / 'abort-checkpoint.json'
            if abort_path.is_file():
                receipt['abort_checkpoint_receipt'] = {
                    'path': str(abort_path), 'blake3': blake3(abort_path.read_bytes()).hexdigest()}
        receipt['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        receipt['ray_temp_dir'] = ray_tmp
        path.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
