"""Opt-in fresh-group collection after a completed Judge rejects its own verdict.

This wraps the existing native generation hook, not the Slime train loop. No
sample or reward is repaired; no failed verdict is rerun. The native hook's
ThreadPoolExecutor drains every sibling before raising. A replacement consumes
the next data-source group while the same actor/Adam and optimizer index remain.
"""
from __future__ import annotations

from copy import copy
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

POLICY = 'completed-judge-verdict-fresh-group-v1'
ENVIRONMENT_KEY = 'EVA_GRPO_JUDGE_GROUP_REPLACEMENTS'
GENERATION_FUNCTION = 'judge_group_isolation.generate_grpo_rollout'
MAX_REPLACEMENTS = 3
# These failures occur after native receipt, workspace replay and trace checks.
# Actor, tool integrity, quota, timeout and provider failures are NOT included.
VERDICT_ERRORS = frozenset({
    'judge terminal response is not JSON',
    'judge terminal response shape differs',
    'judge score coverage differs',
    'judge score row identity differs',
    'judge evidence references differ',
    'judge cites workspace evidence it did not inspect',
    'judge score is not finite',
    'judge score is not a compiled rubric level',
    'judge rationale differs',
    'judge scores cite no inspected workspace reference',
    'judge hard-gate or summary claim differs',
})


def replacement_limit(value):
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    if type(value) is not int or not 0 <= value <= MAX_REPLACEMENTS:
        raise ValueError('Judge group replacements must be an integer from 0 to 3')
    return value


def _read(path):
    return json.loads(Path(path).read_text())


def _commit(path):
    from blake3 import blake3
    path = Path(path)
    return {'path': str(path), 'blake3': blake3(path.read_bytes()).hexdigest()}


def _write(path, document):
    path = Path(path)
    with open(path, 'x', opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write('\n')


class _CaptureBuffer:
    def __init__(self, source, seen):
        self.source, self.seen = source, seen
        self.identities = []

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_samples(self, count):
        if self.identities or count != 1:
            raise ValueError('Isolation requires exactly one fresh four-trajectory prompt group')
        groups = self.source.get_samples(count)
        if len(groups) != 1 or len(groups[0]) != 4:
            raise ValueError('Isolation requires one complete four-trajectory group')
        identities = [(sample.group_index, sample.index) for sample in groups[0]]
        if (len(set(identities)) != 4 or any(identity in self.seen for identity in identities)
                or any(type(value) is not int or value < 0 for row in identities for value in row)):
            raise ValueError('Isolation requires fresh original sample identities')
        self.identities = identities
        self.seen.update(identities)
        return groups  # Original objects, tokens, metadata and cursor are unchanged.


def inspect_unavailable_group(group_root, error):
    """Classify only retained completed-native-Judge terminal-contract failures."""
    from eva_agent.codex_pipeline import CodexPipelineError
    from eva_agent.codex_runtime import codex_turn_receipt_from_document

    if not isinstance(error, CodexPipelineError) or str(error) not in VERDICT_ERRORS:
        raise ValueError('not_a_completed_judge_verdict_failure')
    samples = sorted(path for path in Path(group_root).iterdir() if path.is_dir())
    if len(samples) != 4:
        raise ValueError('incomplete_group_evidence')
    failures, completed = [], []
    for sample in samples:
        if sample.is_symlink() or str(UUID(sample.name)) != sample.name:
            raise ValueError('group_sample_identity_differs')
        failure_path = sample / 'failure.json'
        judge = sample / 'judge' / sample.name
        if failure_path.exists():
            outer = _read(failure_path)
            failure = _read(judge / 'failure.json')
            if (outer.get('phase') != 'workspace_agent_judge'
                    or outer.get('reward_emitted') is not False
                    or failure.get('error_type') != 'CodexPipelineError'
                    or failure.get('pipeline_error') not in VERDICT_ERRORS
                    or failure.get('reward_emitted') is not False
                    or failure.get('retry_count') != 0
                    or (judge / 'grade.json').exists()):
                raise ValueError('group_has_nonverdict_failure')
            receipt_path = judge / 'judge-codex-failure-receipt.json'
            receipt = codex_turn_receipt_from_document(_read(receipt_path))
            if (receipt.status != 'completed' or receipt.role.value != 'judge'
                    or failure.get('codex_receipt_blake3') != receipt.receipt_blake3):
                raise ValueError('judge_not_completed_or_receipt_unbound')
            failures.append({'sample_id': sample.name, 'category': failure['pipeline_error'],
                             'failure': _commit(failure_path), 'judge_failure': _commit(judge / 'failure.json'),
                             'judge_receipt': _commit(receipt_path)})
        else:
            summary = _read(sample / 'summary.json')
            if (summary.get('status') != 'complete' or summary.get('workspace_agent_judged') is not True
                    or not (judge / 'grade.json').is_file()):
                raise ValueError('sibling_not_terminal')
            completed.append(_commit(sample / 'summary.json'))
    if not failures:
        raise ValueError('no_failed_judge_in_group')
    return {'failures': failures, 'completed_sibling_summaries': completed,
            'all_four_trajectories_terminal': True}


def collect_group(args, rollout_id, data_buffer, native_generate, *, limit):
    """Return the unchanged nested complete group; never call the optimizer."""
    limit = replacement_limit(limit)
    if not limit:
        return native_generate(args, rollout_id, data_buffer, evaluation=False)
    if (args.rollout_batch_size != 1 or args.n_samples_per_prompt != 4
            or args.num_steps_per_rollout != 1):
        raise ValueError('Isolation supports one four-trajectory group per optimizer step')
    training_root = Path(args.save).resolve().parent
    canonical = training_root / 'rollouts' / f'{rollout_id:06d}'
    canonical.mkdir(parents=True, exist_ok=False, mode=0o700)
    records, seen = [], set()
    for draw in range(limit + 1):
        draw_id = str(uuid4())
        draw_root = training_root / 'rollout-group-attempts' / f'{rollout_id:06d}' / draw_id
        draw_root.mkdir(parents=True, exist_ok=False, mode=0o700)
        candidate_args = copy(args)
        candidate_args.save = str(draw_root / 'checkpoints')
        group_root = draw_root / 'rollouts' / f'{rollout_id:06d}'
        buffer = _CaptureBuffer(data_buffer, seen)
        record = {'schema': 'eva.judge-group-draw.v1', 'policy': POLICY,
                  'draw_id': draw_id, 'draw_index': draw, 'optimizer_rollout_id': rollout_id,
                  'group_root': str(group_root), 'replacement_limit': limit,
                  'optimizer_calls_by_hook': 0, 'verdict_retries': 0,
                  'actor_or_optimizer_restarted': False}
        _write(draw_root / 'request.json', record)
        try:
            result = native_generate(candidate_args, rollout_id, buffer, evaluation=False)
        except Exception as error:
            record.update(status='unavailable', emitted_training_samples=0, emitted_group_reward=False,
                          error_type=type(error).__name__, original_sample_identities=buffer.identities)
            try:
                record.update(inspect_unavailable_group(group_root, error), replacement_eligible=True)
            except (ValueError, OSError, KeyError, TypeError):
                record['replacement_eligible'] = False
            _write(draw_root / 'result.json', record)
            records.append(_commit(draw_root / 'result.json'))
            if not record['replacement_eligible'] or draw == limit:
                _write(canonical / 'isolation.json', {
                    'schema': 'eva.judge-group-isolation.v1', 'policy': POLICY, 'status': 'unavailable',
                    'optimizer_rollout_id': rollout_id, 'replacement_limit': limit,
                    'draws': records, 'emitted_training_samples': 0, 'emitted_group_reward': False,
                    'replacement_limit_exhausted': draw == limit})
                raise  # Existing driver abort-save receives the original failure.
            continue
        if len(result) != 1 or len(result[0]) != 4 or any(not segment for segment in result[0]):
            raise ValueError('Native hook returned an incomplete training group')
        summary_path = group_root / 'summary.json'
        summary = _read(summary_path)
        record.update(status='accepted', original_sample_identities=buffer.identities,
                      emitted_training_trajectories=4, source_summary=_commit(summary_path))
        _write(draw_root / 'result.json', record)
        records.append(_commit(draw_root / 'result.json'))
        binding = {'schema': 'eva.judge-group-isolation.v1', 'policy': POLICY, 'status': 'accepted',
                   'optimizer_rollout_id': rollout_id, 'replacement_limit': limit, 'draws': records,
                   'accepted_group_root': str(group_root), 'accepted_source_summary': _commit(summary_path),
                   'unavailable_groups': draw, 'emitted_training_trajectories': 4}
        _write(canonical / 'isolation.json', binding)
        _write(canonical / 'summary.json', {**summary, 'group_isolation': binding})
        return result  # No mutation/copy of any accepted Sample or segment.


def generate_grpo_rollout(args, rollout_id, data_buffer, evaluation=False):
    from eva_agent.training.codex_slime_rollout import generate_grpo_rollout as native
    if evaluation:
        return native(args, rollout_id, data_buffer, evaluation=True)
    return collect_group(args, rollout_id, data_buffer, native,
                         limit=replacement_limit(os.environ.get(ENVIRONMENT_KEY, '0')))


def resolve_accepted_group(training_root, rollout_id):
    """Resolve committed UUID evidence for a real optimizer index; no rejected rewards."""
    root = Path(training_root).resolve()
    path = root / 'rollouts' / f'{rollout_id:06d}' / 'isolation.json'
    binding = _read(path)
    if (binding.get('policy') != POLICY or binding.get('status') != 'accepted'
            or binding.get('optimizer_rollout_id') != rollout_id
            or binding.get('emitted_training_trajectories') != 4):
        raise ValueError('Optimizer group has no accepted isolation binding')
    group = Path(binding['accepted_group_root'])
    parent = root / 'rollout-group-attempts' / f'{rollout_id:06d}'
    relative = group.relative_to(parent)
    if (len(relative.parts) != 3 or str(UUID(relative.parts[0])) != relative.parts[0]
            or relative.parts[1:] != ('rollouts', f'{rollout_id:06d}') or group.is_symlink()):
        raise ValueError('Accepted group UUID path differs')
    for source in [*binding['draws'], binding['accepted_source_summary']]:
        if _commit(source['path']) != source:
            raise ValueError('Group isolation source commitment differs')
    if binding['accepted_source_summary']['path'] != str(group / 'summary.json'):
        raise ValueError('Accepted summary path differs')
    draws = [_read(source['path']) for source in binding['draws']]
    if (len(draws) != binding['unavailable_groups'] + 1 or draws[-1].get('status') != 'accepted'
            or draws[-1].get('group_root') != str(group)
            or any(row.get('status') != 'unavailable' or row.get('emitted_training_samples') != 0
                   or row.get('emitted_group_reward') is not False for row in draws[:-1])):
        raise ValueError('Accepted/unavailable draw accounting differs')
    summary = _read(path.parent / 'summary.json')
    if summary != {**_read(group / 'summary.json'), 'group_isolation': binding}:
        raise ValueError('Accepted canonical summary projection differs')
    return {'optimizer_rollout_id': rollout_id, 'group_root': str(group),
            'unavailable_groups': binding['unavailable_groups'], 'isolation': _commit(path),
            'source_summary': binding['accepted_source_summary']}
