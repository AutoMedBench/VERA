"""Prospective attempt-bound optimizer rank coverage for monitors and harness.

The caller supplies independently admitted source-manifest hashes and their
trainer world sizes. This module does not admit arbitrary launches or infer
completion from however many rank receipts happen to exist.
"""
import hashlib
import json
import math
from pathlib import Path

# These pins describe expected rank coverage, not GPU qualification. Admission
# of a new trainer layout is still the launcher's separate responsibility.
HISTORICAL_SOURCES = dict.fromkeys((
    '9f1423ed97098d1f0b43cc61d729f27750fa28bd80a53db51e649c9872d3e69e',
    '09cec22d2122dc97455154d0c31a7a475c9f6a28ecfedf6c7d443d4c3ccf918e',
    '0fe20cb2e63c421c7434bd4dcff258fbfb644584f417607acb76fecbe0bfcad7',
    '0847ca4b7efcc71a942be33b5f095a01fef08b8feab88b06c8e86094ec17c6f9',
    'd244f029acc7f1d99fcd90e61ec92e8826a6a59b9a4c70367bf3d02499d2d30d',
    '2c30c792364489efc96247a4f6bff6b0a38190209d446f0c7b2b0678ee9fd179',
), 64)

# Prior96-trainer feedback remains valid after a version-only restart.
HISTORICAL_SOURCES['7a24a1142c1576f9567d41d71a9ba5a7d3e9502125c5f655a29d8f61eabda4c2'] = 96
HISTORICAL_SOURCES['3a6c5a7ecd39402d02ff31b3f14895030079b975cb8d89496f40151f7c82fadb'] = 96


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(root, run, learning, admitted_sources):
    root, run, learning = (Path(p).resolve(strict=True) for p in (root, run, learning))
    if not run.is_relative_to(root) or learning.name != 'learning' or learning.parent.parent != run/'attempts':
        raise ValueError('optimizer evidence must belong to a run attempt')
    attempt = learning.parent
    sp = run/'submissions'/f'{attempt.name}.json'
    lp = attempt/'launch.json'
    submission, launch = read(sp), read(lp)
    if submission['attempt'] != attempt.name:
        raise ValueError('optimizer attempt identity mismatch')
    batch_script = Path(submission['argv'][-1]).resolve(strict=True)
    if batch_script.name != 'all.sbatch' or not batch_script.is_relative_to(root):
        raise ValueError('unrecognized optimizer launch source')
    manifest = batch_script.parent/'cluster-source.json'
    manifest_sha = sha(manifest)
    n = admitted_sources.get(manifest_sha)
    if type(n) is not int or n not in (64, 96):
        raise ValueError('optimizer source layout is not independently admitted')
    source = read(manifest)
    if root/source['source'] != batch_script.parent:
        raise ValueError('optimizer source manifest location mismatch')
    config_path = Path(submission['pipeline_config']).resolve(strict=True)
    if config_path.parent != run/'pipeline-configs' or sha(config_path) != submission['pipeline_config_sha256']:
        raise ValueError('optimizer pipeline configuration identity mismatch')
    config = read(config_path)
    layout = launch['layout']
    if layout != submission['layout'] or layout != config['layout']:
        raise ValueError('optimizer launch and admitted layout disagree')
    expected = {'actor_gpus': n, 'actor_nodes': n//8, 'gpus_per_node': 8,
        'tensor_parallel': 4, 'pipeline_parallel': 2, 'context_parallel': 1,
        'data_parallel': n//8, 'global_batch_size': 64, 'nodes': 12,
        'total_gpus': 96, 'rollout_gpus': 96, 'rollout_engines': 48,
        'rollout_engine_tensor_parallel': 2, 'overlapping_gpus': n, 'colocate': True}
    if any(layout.get(k) != v for k,v in expected.items()):
        raise ValueError('optimizer topology is inconsistent with source contract')
    argv = launch['argv']
    flags = {'--actor-num-nodes': n//8, '--actor-num-gpus-per-node': 8,
        '--tensor-model-parallel-size': 4, '--pipeline-model-parallel-size': 2,
        '--context-parallel-size': 1, '--global-batch-size': 64}
    for flag, expected_value in flags.items():
        if argv.count(flag) != 1 or argv[argv.index(flag)+1] != str(expected_value):
            raise ValueError(f'optimizer argument/layout mismatch: {flag}')
    return {'attempt': attempt.name, 'expected_optimizer_ranks': n,
        'source_manifest_sha256': manifest_sha,
        'evidence_sha256': {str(p): sha(p) for p in (sp, lp, manifest, config_path)}}


def receipt_paths(learning, update, topology):
    if type(update) is not int or update < 1:
        raise ValueError('one-based optimizer update required')
    learning = Path(learning)
    if learning.parent.name != topology['attempt']:
        raise ValueError('optimizer topology belongs to another attempt')
    n = topology['expected_optimizer_ranks']
    paths = [learning/f'update-{update:04d}-rank-{rank}.json' for rank in range(n)]
    extras = set(learning.glob(f'update-{update:04d}-rank-*.json')) - set(paths)
    if extras:
        raise ValueError('optimizer receipts exceed the admitted rank set')
    return paths


def resolve_attempt(root, run, learning):
    """Resolve historical pins or this sealed release's fixed 96-rank contract.

    Computing this release's manifest digest avoids a self-referential source
    hash. Its inventory must cover this module and the batch script and match
    every listed file. This does not authorize submitting the draft release.
    """
    root = Path(root).resolve(strict=True)
    own = Path(__file__).resolve().parent
    sources = dict(HISTORICAL_SOURCES)
    manifest = own/'cluster-source.json'
    if manifest.is_file():
        value = read(manifest)
        inventory = value['source_hashes']
        required = [own/'optimizer_topology.py', own/'all.sbatch']
        if (root/value['source']).resolve() != own or any(str(p.relative_to(root)) not in inventory for p in required):
            raise ValueError('optimizer source inventory is incomplete')
        for name, expected in inventory.items():
            path = (root/name).resolve(strict=True)
            if not path.is_relative_to(root) or sha(path) != expected:
                raise ValueError('optimizer source inventory changed')
        sources[sha(manifest)] = 96
    return resolve(root, run, learning, sources)


def valid_receipt(value, rank, update):
    norm = value.get('gradient_norm')
    return (value.get('rank') == rank and value.get('optimizer_update') == update
            and value.get('update_successful') is True and type(norm) in (int, float)
            and math.isfinite(norm) and norm > 0
            and value.get('global_sampled_master_shards_changed', 0) > 0)


def validate_feedback_index(root, run, index):
    """Check coverage against the producing attempt, including old 64-rank indices.

    The caller also verifies the index commitment and all evidence file hashes.
    Old indices are read without rewriting their frozen evidence.
    """
    update = index['optimizer_update']
    files = {Path(p).resolve(strict=True) for p in index['evidence_files']}
    rank_zero = [p for p in files if p.name == f'update-{update:04d}-rank-0.json']
    if len(rank_zero) != 1:
        raise ValueError('feedback requires one producing optimizer attempt')
    learning = rank_zero[0].parent
    topology = resolve_attempt(root, run, learning)
    ranks = receipt_paths(learning, update, topology)
    n = topology['expected_optimizer_ranks']
    if index['all_optimizer_ranks_verified'] != n or not set(ranks).issubset(files):
        raise ValueError('feedback optimizer rank coverage mismatch')
    if any(not valid_receipt(read(p), rank, update) for rank, p in enumerate(ranks)):
        raise ValueError('feedback optimizer receipt invalid')
    bound = index.get('optimizer_topology')
    if bound is None:
        if n != 64:
            raise ValueError('96-rank feedback requires explicit topology binding')
    elif bound != topology or not {Path(p) for p in topology['evidence_sha256']}.issubset(files):
        raise ValueError('feedback optimizer topology binding changed')
    return topology
