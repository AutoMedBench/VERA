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
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from training.automedbench_lite.actor import serving_binding
from training.automedbench_lite.adapter import file_digest, read_document, write_once
from training.automedbench_lite.docker_runtime import resolve_image
from training.benchmark_models.run_prescribed_model import (
    SUPPORTED_TRACKS, TRACK_HELPERS, MODEL_DEPENDENCIES, HF_MODELS, GIT_SOURCES, segmentation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'runtime-manifest', 'public-python', 'codex-bin', 'server-identity', 'server-canary'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--image', default='eva-automedbench-cpu:20260910-v1')
    args = parser.parse_args()
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
    command = [str(args.public_python.absolute()), str(ROOT / 'training/automedbench_lite/track_entry.py'),
        'run', '--run-root', str(run), '--runtime-manifest', str(args.runtime_manifest.resolve()),
        '--public-python', str(args.public_python.absolute()), '--codex-bin', str(binary),
        '--server-identity', str(args.server_identity.resolve()), '--server-canary', str(args.server_canary.resolve()),
        '--image', image, '--tracks', 'all', '--turn-timeout', '900']
    sources = [Path(__file__), ROOT / 'training/automedbench_lite/track_actor.py',
        ROOT / 'training/automedbench_lite/track_tools.py', ROOT / 'training/automedbench_lite/job_wait.py',
        ROOT / 'training/automedbench_lite/skill_discovery.py', ROOT / 'training/automedbench_lite/local_qwen.py',
        ROOT / 'src/eva_agent/codex_providers/adapter.py', ROOT / 'src/eva_agent/codex_pipeline/adapter.py',
        ROOT / 'training/benchmark_models/job_supervisor.py', ROOT / 'training/benchmark_models/run_prescribed_model.py']
    sources += [ROOT / 'training/benchmark_models' / filename for names in TRACK_HELPERS.values() for filename in names]
    write_once(run / 'baseline-launch.json', {'schema': 'eva.automedbench-seven-track-baseline-launch.v1',
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'driver_pid': os.getpid(),
        'command': command, 'codex_binary': str(binary), 'codex_version': version,
        'codex_binary_blake3': binary_digest, 'server_binding': binding, 'docker_image': image,
        'runtime_manifest_blake3': file_digest(args.runtime_manifest), 'public_models': models,
        'runtime_source_blake3': {str(path.relative_to(ROOT)): file_digest(path) for path in set(sources)},
        'prelaunch_scope': 'pinned identity, all-file presence/size and overlay/source checks; per-job workers verify full content',
        'all_model_payloads_rehashed_at_launch': False, 'gpu_runtime_success_claimed': False,
        'max_parallel_tracks': 4, 'analysis_gpu_jobs_serialized': True, 'training_authorized': False,
        'standalone_canaries': 0, 'automatic_retry': False})
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'))
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
