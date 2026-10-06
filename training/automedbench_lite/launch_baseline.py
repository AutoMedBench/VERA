"""Detached baseline driver: bind identities, run once, retain actual OS exit."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.harness_source import activate_harness, runtime_paths
activate_harness(ROOT)

from training.automedbench_lite.actor import serving_binding
from training.automedbench_lite.adapter import file_digest, read_document, write_once
from training.automedbench_lite.docker_runtime import resolve_image
from eva_agent.codex_providers import adapter as provider_adapter_source
from eva_agent.codex_pipeline import adapter as pipeline_adapter_source
from eva_agent.codex_runtime import runtime as runtime_source
from eva_agent.codex_runtime import policy_budget as policy_budget_source
from training.benchmark_models.run_prescribed_model import (
    SUPPORTED_TRACKS, TRACK_HELPERS, MODEL_DEPENDENCIES, HF_MODELS, GIT_SOURCES, segmentation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'runtime-manifest', 'public-python', 'codex-bin', 'server-identity', 'server-canary'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--image', default='eva-automedbench-cpu:20260910-v1')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--context-length', type=int, default=32768)
    parser.add_argument('--auto-compact-token-limit', type=int, default=12288)
    parser.add_argument('--output-tokens', type=int, default=4096)
    parser.add_argument('--tool-output-token-limit', type=int, default=None,
                        help='Opt-in native Codex context-view limit; original MCP outputs remain retained.')
    parser.add_argument('--adapter-capacity-wait-seconds', type=float, default=0.0)
    parser.add_argument('--turn-timeout', type=int, default=900)
    parser.add_argument('--endpoint', default='http://127.0.0.1:30910/v1')
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument('--memory-profile-v12', action='store_true')
    profiles.add_argument('--supra-profile', action='store_true')
    args = parser.parse_args()
    if args.tool_output_token_limit is not None and not 256 <= args.tool_output_token_limit <= 4096:
        parser.error('tool-output-token-limit must be within 256..4096')
    run = args.run_root.resolve(strict=True)
    if (run / 'track-rollouts').exists() or (run / 'baseline-launch.json').exists():
        raise ValueError('Baseline workspace or launch identity already used')
    planned = read_document(run / 'track-run-manifest.json')
    if {row['track'] for row in planned['tracks']} != set(SUPPORTED_TRACKS):
        raise ValueError('Fresh baseline must contain all seven tracks')
    # Acquisition manifests use their original plain JSON format; their full
    # file digest is bound below, not a fabricated document_blake3 envelope.
    manifest = json.loads(args.runtime_manifest.read_bytes())
    if manifest['status'] != 'complete':
        raise ValueError('Public model acquisition is incomplete')
    models = {}
    for name in set(SUPPORTED_TRACKS) | {key for values in MODEL_DEPENDENCIES.values() for key in values}:
        model = manifest['models'][name]
        expected = HF_MODELS[name][:2] if name in HF_MODELS else GIT_SOURCES[name] if name in GIT_SOURCES else (
            segmentation.REPOSITORY, segmentation.REVISION)
        if (model['repository'], model['revision']) != tuple(expected):
            raise ValueError('Public model identity differs from frozen runtime')
        path = Path(model['path']).resolve(strict=True)
        for row in model['files']:
            if (path / row['path']).stat().st_size != row['bytes']:
                raise ValueError('Retained model file missing or size changed')
        models[name] = {'repository': model['repository'], 'revision': model['revision'], 'path': str(path),
                        'files': len(model['files']), 'bytes': sum(row['bytes'] for row in model['files'])}
    for directory in ('llava-transformers436', 'chexagent-transformers440'):
        if not (Path(manifest['cache_root']) / 'runtimes' / directory / 'transformers/__init__.py').is_file():
            raise ValueError('Frozen generative runtime overlay missing')
    binary = args.codex_bin.resolve(strict=True)
    version = subprocess.check_output([str(binary), '--version'], text=True, timeout=15).strip()
    if version != 'codex-cli 0.153.4':
        raise ValueError('Exact Codex version required')
    binary_digest = file_digest(binary)
    image = resolve_image(args.image)
    binding = serving_binding(args.server_canary, args.server_identity)
    if args.context_length > binding['identity']['settings']['context_length']:
        raise ValueError('Requested context exceeds actual bound serving capacity')
    selected_memory = None
    if args.memory_profile_v12 or args.supra_profile:
        from training.automedbench_lite.track_memory_v12 import selected_memory_sources, make_supra_profile
        selected_memory = selected_memory_sources()
        if args.supra_profile:
            from types import SimpleNamespace
            make_supra_profile(SimpleNamespace(model='Qwen/Qwen3.5-9B', provider='eva_local_qwen'),
                context_tokens=args.context_length, output_tokens=args.output_tokens,
                compact_tokens=args.auto_compact_token_limit, endpoint=args.endpoint)
    command = [str(args.public_python.absolute()), str(ROOT / 'training/automedbench_lite/track_entry.py'),
        'run', '--run-root', str(run), '--runtime-manifest', str(args.runtime_manifest.resolve()),
        '--public-python', str(args.public_python.absolute()), '--codex-bin', str(binary),
        '--server-identity', str(args.server_identity.resolve()), '--server-canary', str(args.server_canary.resolve()),
        '--image', image, '--tracks', 'all', '--turn-timeout', str(args.turn_timeout),
        '--workers', str(args.workers), '--context-length', str(args.context_length),
        '--auto-compact-token-limit', str(args.auto_compact_token_limit),
        '--output-tokens', str(args.output_tokens), '--adapter-capacity-wait-seconds', str(args.adapter_capacity_wait_seconds),
        '--endpoint', args.endpoint]
    if args.supra_profile:
        command.append('--supra-profile')
    elif args.memory_profile_v12:
        command.append('--memory-profile-v12')
    if args.tool_output_token_limit is not None:
        command += ['--tool-output-token-limit', str(args.tool_output_token_limit)]
    sources = [Path(__file__), ROOT / 'training/automedbench_lite/track_actor.py',
        ROOT / 'training/automedbench_lite/track_tools.py', ROOT / 'training/automedbench_lite/job_wait.py',
        ROOT / 'training/automedbench_lite/skill_discovery.py', ROOT / 'training/automedbench_lite/local_qwen.py',
        Path(provider_adapter_source.__file__), Path(pipeline_adapter_source.__file__),
        Path(runtime_source.__file__), Path(policy_budget_source.__file__),
        ROOT / 'training/automedbench_lite/policy_capture.py', ROOT / 'training/harness_source.py',
        ROOT / 'training/benchmark_models/job_supervisor.py', ROOT / 'training/benchmark_models/run_prescribed_model.py']
    sources += [ROOT / 'training/benchmark_models' / filename for names in TRACK_HELPERS.values() for filename in names]
    sources += [ROOT / 'training/automedbench_lite/track_entry.py']
    if selected_memory:
        sources += [ROOT / 'training/automedbench_lite/track_memory_v12.py']
        sources += [Path(row['path']) for row in selected_memory.values()]
    write_once(run / 'baseline-launch.json', {'schema': 'eva.automedbench-seven-track-baseline-launch.v1',
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'driver_pid': os.getpid(),
        'command': command, 'codex_binary': str(binary), 'codex_version': version,
        'codex_binary_blake3': binary_digest, 'server_binding': binding, 'docker_image': image,
        'runtime_manifest_blake3': file_digest(args.runtime_manifest), 'public_models': models,
        'runtime_source_blake3': {(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT)
            else 'selected-harness:' + str(path)): file_digest(path) for path in set(sources)},
        'selected_runtime_pythonpath': runtime_paths(ROOT),
        **({'selected_memory_sources': selected_memory} if selected_memory else {}),
        'memory_profile_v12': args.memory_profile_v12, 'supra_profile': args.supra_profile,
        'context_length': args.context_length, 'auto_compact_token_limit': args.auto_compact_token_limit,
        'output_tokens': args.output_tokens, 'adapter_capacity_wait_seconds': args.adapter_capacity_wait_seconds,
        'native_tool_output_token_limit': args.tool_output_token_limit,
        'original_mcp_responses_retained': True,
        'thinking_requested': True,
        'prelaunch_scope': 'pinned identity, all-file presence/size and overlay/source checks; per-job workers verify full content',
        'all_model_payloads_rehashed_at_launch': False, 'gpu_runtime_success_claimed': False,
        'max_parallel_tracks': args.workers, 'analysis_gpu_jobs_serialized': True, 'training_authorized': False,
        'standalone_canaries': 0, 'automatic_retry': False})
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=os.pathsep.join(runtime_paths(ROOT)))
    process = subprocess.Popen(command, cwd=ROOT, env=environment)
    write_once(run / 'baseline-process.json', {'driver_pid': os.getpid(), 'actor_pid': process.pid,
                                              'command': command})
    print(json.dumps({'status': 'actor_started', 'run_root': str(run), 'actor_pid': process.pid}), flush=True)
    code = process.wait()
    write_once(run / 'baseline-process-exit.json', {'actor_pid': process.pid, 'returncode': code,
        'os_process_exit_observed': True, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'exit_zero_is_not_evaluation_success': True})
    raise SystemExit(code)


if __name__ == '__main__':
    main()
