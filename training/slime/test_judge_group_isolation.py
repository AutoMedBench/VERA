"""Provider/GPU-free group isolation fixtures; execute the real native hook body."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from threading import Barrier, Event, Lock
from types import SimpleNamespace
from uuid import uuid4

import pytest

import judge_group_isolation as isolation
from eva_agent.codex_pipeline import CodexPipelineError
from eva_agent.codex_runtime import contracts
from eva_agent.pipeline.digests import blake3_hex
from run_full_parameter import CODEX_GENERATION_FUNCTION, build_command


ERROR = 'judge cites workspace evidence it did not inspect'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def receipt():
    event = dict(event_id=str(uuid4()), sequence=0, method='turn/completed',
                 thread_id='fixture-thread', turn_id='fixture-turn', payload={}, content_redacted=True)
    event['event_blake3'] = blake3_hex(event)
    core = dict(schema='eva.codex-turn-receipt.v1', receipt_id=str(uuid4()),
                runtime_thread_id=str(uuid4()), runtime_turn_id=str(uuid4()),
                thread_id='fixture-thread', turn_id='fixture-turn', role='judge',
                model='gpt-6-astra', provider='eva_native_astra', sandbox='read-only',
                thread_resumed=False, visibility='judge-only', status='completed', final_response='{}',
                events=[event], tool_calls=[], selected_skill_ids=[], selected_skill_catalog_blake3='a' * 64,
                offered_mcp_tool_names=[], offered_tool_schema_blake3='b' * 64,
                max_parallelism_observed=0, parallel_tool_calls_supported=True, usage={},
                input_blake3='c' * 64, config_keys=[], config_values_recorded=False,
                input_payload_recorded=False, sdk_version='fixture', server_version='fixture')
    core['receipt_blake3'] = blake3_hex(core)
    contracts.codex_turn_receipt_from_document(core)
    return core


class Data:
    def __init__(self):
        self.draws = []

    def get_samples(self, count):
        assert count == 1
        draw = len(self.draws)
        group = [SimpleNamespace(index=draw * 4 + i, group_index=draw,
                                 metadata={'bulk_root': '/public-fixture', 'sandbox_id': 'fixture'})
                 for i in range(4)]
        self.draws.append(group)
        return [group]


def native_fixture(monkeypatch, failures, *, delayed=None):
    """Unchanged installed-selected native generation function, fixture trajectories only."""
    import eva_agent.training.codex_slime_rollout as native_module
    source = Path(native_module.__file__)
    node = next(n for n in ast.parse(source.read_text()).body
                if getattr(n, 'name', None) == 'generate_grpo_rollout')
    captures, returned = [], []
    barriers, barrier_lock = {}, Lock()

    def trajectory(args, original, *, sample_root, sample_id, training_iteration, **unused):
        draw = original.index // 4
        captures.append((draw, original.index, training_iteration, str(sample_root), args))
        sample_root.mkdir(parents=True, exist_ok=False)
        write(sample_root / 'trajectory.json', {'fixture': True, 'original_index': original.index})
        with barrier_lock:
            barrier = barriers.setdefault(draw, Barrier(4))
        barrier.wait(timeout=5)
        judge = sample_root / 'judge' / sample_id
        bad = original.index % 4 in failures.get(draw, {})
        if delayed and original.index == 3:
            delayed[0].set()
            assert delayed[1].wait(5)
        if bad:
            category = failures[draw][original.index % 4]
            raw = receipt()
            write(sample_root / 'failure.json', {'phase': 'workspace_agent_judge', 'reward_emitted': False})
            write(judge / 'failure.json', {'error_type': 'CodexPipelineError', 'pipeline_error': category,
                  'reward_emitted': False, 'retry_count': 0, 'codex_receipt_blake3': raw['receipt_blake3']})
            write(judge / 'judge-codex-failure-receipt.json', raw)
            raise CodexPipelineError(category)
        # Nonzero and zero grades are both accepted, never filtered for quality.
        score = (original.index % 3) / 2
        write(judge / 'grade.json', {'reward': score})
        write(sample_root / 'summary.json', {'status': 'complete', 'workspace_agent_judged': True,
              'training_iteration': training_iteration, 'reward': score})
        segment = SimpleNamespace(tokens=[10, 20, 30], response_length=1, loss_mask=[1],
                                  rollout_log_probs=[-0.73], reward=score,
                                  rollout_id=original.index, group_index=original.group_index,
                                  metadata={'training_iteration': training_iteration,
                                            'trajectory_path': str(sample_root / 'trajectory.json')})
        returned.append(segment)
        return [segment]

    monkeypatch.setitem(sys.modules, 'slime.rollout.sglang_rollout',
                        SimpleNamespace(GenerateState=lambda args: SimpleNamespace(tokenizer=None, sampling_params={})))
    monkeypatch.setitem(sys.modules, 'eva_agent.training.teacher_worker',
                        SimpleNamespace(CampaignV2TeacherContextPool=lambda: None, load_bulk_record=lambda *a: {}))
    ns = dict(REWARD_HOOK=native_module.REWARD_HOOK, native_settings=lambda args: {},
              _CONTEXT_LOCK=Lock(), _CONTEXTS=SimpleNamespace(load=lambda record: (None, None)),
              Path=Path, uuid4=uuid4, run_native_trajectory=trajectory,
              ThreadPoolExecutor=ThreadPoolExecutor, _write_private_json=write)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    return ns['generate_grpo_rollout'], captures, returned


def args(tmp_path):
    return SimpleNamespace(save=str(tmp_path / 'training/checkpoints'), rollout_batch_size=1,
                           n_samples_per_prompt=4, num_steps_per_rollout=1,
                           custom_reward_post_process_path='eva_agent.training.codex_segment_rewards.normalize_segment_rewards')


def test_invalid_four_then_fresh_four_drains_and_returns_exact_original_samples(monkeypatch, tmp_path):
    entered, release = Event(), Event()
    native, captures, returned = native_fixture(monkeypatch, {0: {0: ERROR}}, delayed=(entered, release))
    data, options = Data(), args(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(isolation.collect_group, options, 7, data, native, limit=1)
        assert entered.wait(5)
        assert len(data.draws) == 1 and not future.done()  # No orphan sibling while replacing.
        release.set()
        result = future.result(timeout=5)
    assert len(data.draws) == 2 and len(result) == 1 and len(result[0]) == 4
    assert [s.rollout_id for t in result[0] for s in t] == [4, 5, 6, 7]
    assert all(any(s is original for original in returned) for t in result[0] for s in t)
    assert all((s.tokens, s.response_length, s.loss_mask, s.rollout_log_probs) ==
               ([10, 20, 30], 1, [1], [-0.73]) for t in result[0] for s in t)
    assert all(row[2] == 7 for row in captures) and len({row[3] for row in captures}) == 8
    assert all({k: v for k, v in vars(row[4]).items() if k != 'save'} ==
               {k: v for k, v in vars(options).items() if k != 'save'} for row in captures)
    binding = isolation.resolve_accepted_group(tmp_path / 'training', 7)
    assert binding['unavailable_groups'] == 1
    records = [json.loads(p.read_text()) for p in (tmp_path / 'training/rollout-group-attempts/000007').glob('*/result.json')]
    failed = next(r for r in records if r['status'] == 'unavailable')
    assert failed['emitted_training_samples'] == 0 and failed['emitted_group_reward'] is False
    assert len(failed['failures']) == 1 and len(failed['completed_sibling_summaries']) == 3
    assert all(r['optimizer_calls_by_hook'] == r['verdict_retries'] == 0 for r in records)
    assert len(list((tmp_path / 'training').rglob('trajectory.json'))) == 8
    # Two real optimizer frontiers would remain 7 then 8, not 7 then 9.
    next_result = isolation.collect_group(options, 8, data, native, limit=1)
    assert len(data.draws) == 3 and len(next_result[0]) == 4
    assert isolation.resolve_accepted_group(tmp_path / 'training', 8)['unavailable_groups'] == 0


def test_bounded_exhaustion_raises_original_to_abort_save_without_partial_group(monkeypatch, tmp_path):
    native, captures, _ = native_fixture(monkeypatch, {0: {0: ERROR}, 1: {2: ERROR}})
    data = Data()
    with pytest.raises(CodexPipelineError, match=ERROR):
        isolation.collect_group(args(tmp_path), 0, data, native, limit=1)
    assert len(data.draws) == 2 and len(captures) == 8
    manifest = json.loads((tmp_path / 'training/rollouts/000000/isolation.json').read_text())
    assert manifest['status'] == 'unavailable' and manifest['replacement_limit_exhausted']
    assert manifest['emitted_training_samples'] == 0 and manifest['emitted_group_reward'] is False
    assert not (tmp_path / 'training/rollouts/000000/summary.json').exists()
    with pytest.raises(ValueError, match='no accepted'):
        isolation.resolve_accepted_group(tmp_path / 'training', 0)


def test_transport_or_unknown_sibling_failure_is_not_replaced(monkeypatch, tmp_path):
    native, _, _ = native_fixture(monkeypatch, {0: {0: ERROR, 3: 'judge MCP observation differs from immutable local replay'}})
    data = Data()
    with pytest.raises(CodexPipelineError):
        isolation.collect_group(args(tmp_path), 0, data, native, limit=2)
    assert len(data.draws) == 1


def test_default_calls_legacy_hook_once_with_identical_arguments_and_paths(monkeypatch, tmp_path):
    native, _, _ = native_fixture(monkeypatch, {0: {0: ERROR}})
    options, data = args(tmp_path), Data()
    with pytest.raises(CodexPipelineError):
        isolation.collect_group(options, 0, data, native, limit=0)
    assert len(data.draws) == 1
    assert len(list((tmp_path / 'training/rollouts/000000').glob('*/trajectory.json'))) == 4
    assert not (tmp_path / 'training/rollout-group-attempts').exists()


def test_changed_receipt_is_not_isolated(monkeypatch, tmp_path):
    native, _, _ = native_fixture(monkeypatch, {0: {0: ERROR}})
    options, data = args(tmp_path), Data()
    with pytest.raises(CodexPipelineError) as caught:
        native(options, 0, data)
    group = tmp_path / 'training/rollouts/000000'
    path = next(group.glob('*/judge/*/judge-codex-failure-receipt.json'))
    raw = json.loads(path.read_text()); raw['final_response'] = 'changed'; write(path, raw)
    with pytest.raises(ValueError):
        isolation.inspect_unavailable_group(group, caught.value)


def test_launch_selector_is_opt_in_and_requires_native_group4_abort_save(tmp_path):
    common = dict(stage='grpo', model=tmp_path, load=tmp_path, data=tmp_path, output=tmp_path,
                  steps=19, batch_size=4, samples_per_prompt=4, max_tokens=32768,
                  max_response_tokens=8192, generation_function=CODEX_GENERATION_FUNCTION,
                  save_on_rollout_error=True)
    old = build_command(SimpleNamespace(**common))
    selected = build_command(SimpleNamespace(**common, judge_group_replacements=1))
    offset = old.index('--rollout-function-path') + 1
    assert old[offset] == CODEX_GENERATION_FUNCTION and selected[offset] == isolation.GENERATION_FUNCTION
    assert old[:offset] == selected[:offset] and old[offset + 1:] == selected[offset + 1:]
    for change in ({'save_on_rollout_error': False}, {'batch_size': 8}, {'samples_per_prompt': 2},
                   {'generation_function': 'unrelated.generate'}, {'stage': 'sft'}):
        with pytest.raises(ValueError):
            build_command(SimpleNamespace(**{**common, **change}, judge_group_replacements=1))
    with pytest.raises(ValueError):
        build_command(SimpleNamespace(**common, judge_group_replacements=4))


def test_real_training_and_checkpoint_verifiers_follow_only_accepted_uuid_groups(monkeypatch, tmp_path):
    from training.eva_rsi.evidence import verify_training, verify_checkpoint_evidence, commitment
    from training.slime import checkpoint_preflight

    native, _, _ = native_fixture(monkeypatch, {0: {0: ERROR}})
    options, data = args(tmp_path), Data()
    for step in range(2):
        isolation.collect_group(options, step, data, native, limit=1)
    root = tmp_path / 'training'
    data_path = tmp_path / 'data.jsonl'
    write(data_path, {'metadata': {'stage': 'S3', 'judge_backend': 'native_astra'}})
    run = {'stage': 'grpo', 'judge_backend': 'native_astra', 'status': 'synthetic_fixture',
           'argv': ['--use-rollout-logprobs', '--num-steps-per-rollout', '1', '--num-rollout', '2',
                    '--hf-checkpoint', '/original-qwen', '--load', '/saved/checkpoints',
                    '--save', str(root / 'checkpoints'), '--prompt-data', str(data_path),
                    '--rollout-function-path', isolation.GENERATION_FUNCTION],
           'training_data_blake3': commitment(data_path)['blake3'],
           'judge_group_isolation': {'policy': isolation.POLICY, 'replacement_limit': 1}}
    write(root / 'run-receipt.json', run)
    events = [{'event': 'optimizer_step', 'rollout_id': step, 'step_id': 0,
               'trainable_parameters': 8953803264, 'gradient_norm': float(step),
               'sampled_changed_values': step, 'successful_update': True} for step in range(2)]
    (root / 'optimizer-steps.jsonl').write_text('\n'.join(json.dumps(row) for row in events))
    context = {'remaining_updates': 2, 'architecture_model_path': '/original-qwen',
               'checkpoint_root': '/saved/checkpoints', 's_target': 'S3'}
    training = verify_training(root, context)
    assert training['executions'] == 2 and training['learning_updates'] == 1
    assert [row['optimizer_rollout_id'] for row in training['accepted_rollout_groups']] == [0, 1]
    assert [row['unavailable_groups'] for row in training['accepted_rollout_groups']] == [1, 0]
    monkeypatch.setattr(checkpoint_preflight, 'verify_checkpoint', lambda *args: {'selection': {'iteration': 0}})
    hf = root / 'hf/iter_0000000'
    write(hf / 'config.json', {}); write(hf / 'tokenizer_config.json', {})
    (hf / 'fixture.safetensors').write_bytes(b'synthetic-header-only')
    checkpoint = verify_checkpoint_evidence(training, context)
    assert checkpoint['durable_updates'] == 1 and checkpoint['durable_learning_updates'] == 0
    assert checkpoint['accepted_rollout_groups'] == training['accepted_rollout_groups'][:1]
    # The real consumer detects changed accepted evidence; rejected draws are
    # never used as alternative optimizer indices or credited checkpoints.
    path = Path(training['accepted_rollout_groups'][0]['source_summary']['path'])
    path.write_text('{}')
    with pytest.raises(ValueError, match='commitment differs'):
        verify_training(root, context)
