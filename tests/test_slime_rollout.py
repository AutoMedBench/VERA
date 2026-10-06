"""Provider-free on-policy token/mask contract, including a real host artifact."""
from enum import Enum
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from eva_agent.pipeline import Stage
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.training.slime_rollout import SGLangEvaActor, observable_model_text, preserve_sampled_tokens, stage_actor_instruction


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return 'prompt'

    def encode(self, text, **kwargs):
        return list(text.encode())

    def convert_tokens_to_ids(self, text):
        return 999


class Sample:
    class Status(Enum):
        COMPLETED = 'completed'
        TRUNCATED = 'truncated'

    metadata = None

    def append_response_tokens(self, args=None, *, tokens, log_probs=None, trainable=True,
                               text=None, **kwargs):
        self.tokens.extend(tokens)
        self.response_length += len(tokens)
        self.loss_mask.extend([int(trainable)] * len(tokens))
        self.rollout_log_probs = (self.rollout_log_probs or []) + (
            log_probs if trainable else [0.0] * len(tokens))
        self.response += text or ''


@pytest.fixture
def actor_environment(monkeypatch, tmp_path):
    responses, requests = [], []
    parsed = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs['trust_env'] is False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, *, json):
            requests.append(json)
            value = responses.pop(0)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: value)

    class Parser:
        def __init__(self, **kwargs):
            assert kwargs['tool_call_parser'] == 'qwen3_coder'

        def parse_non_stream(self, text):
            return parsed.pop(0)

    modules = {
        'httpx': {'Client': Client},
        'sglang.srt.entrypoints.openai.protocol': {
            'Tool': SimpleNamespace(model_validate=lambda value: value)},
        'sglang.srt.function_call.function_call_parser': {'FunctionCallParser': Parser},
        'slime.utils.types': {'Sample': Sample},
    }
    for name, members in modules.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)

    tool = {'type': 'function', 'function': {'name': 'materialize_plan', 'description': 'Write plan',
        'parameters': {'type': 'object', 'properties': {'plan': {'type': 'string'}},
                       'required': ['plan'], 'additionalProperties': False}}}
    request = SimpleNamespace(sandbox=SimpleNamespace(stage=SimpleNamespace(value='S1'),
                                                     instruction='Author a plan'),
                              policy_visible_context={'stage': 'S1'}, available_tools=[tool])
    context = SimpleNamespace(stage_tool_guidance=SimpleNamespace(
        focus=Stage.S1, prompt_text='Original public S1 guidance', frontiers=[{
        'tool_name': 'materialize_plan', 'arguments': {'plan': 'Authored plan'}}]))
    sample = Sample()
    args = SimpleNamespace(seq_length=4096, rollout_max_response_len=128,
                           sglang_router_ip='127.0.0.1', sglang_router_port=1,
                           eva_enable_thinking=False)
    actor = SGLangEvaActor(args=args, sample=sample, tokenizer=Tokenizer(), context=context,
                          sampling_params={'temperature': 0.7})
    artifact = tmp_path / 'stage-plan.json'

    def execute(calls):
        assert len(calls) == 1
        call = calls[0]
        artifact.write_text(json.dumps(dict(call.arguments)))
        return [SimpleResult(call.call_id)]

    return SimpleNamespace(actor=actor, request=request, runtime=SimpleNamespace(execute=execute),
                           artifact=artifact, sample=sample, responses=responses,
                           requests=requests, parsed=parsed)


class SimpleResult(dict):
    def __init__(self, call_id):
        super().__init__(call_id=call_id, status='completed', output={'gate_passed': True})
        self.call_id = call_id


def output(text, token_ids, logprobs):
    return {'text': text, 'meta_info': {'finish_reason': {'type': 'stop'},
            'output_token_logprobs': [[p, t, None] for p, t in zip(logprobs, token_ids)]}}


def test_actual_sampled_ids_and_logprobs_survive_tool_observation(actor_environment):
    env = actor_environment
    env.responses.extend([output('tool action', [701, 999], [-0.3, -0.1]),
                          output('Plan created.', [702, 999], [-0.2, -0.1])])
    env.parsed.extend([('', [SimpleNamespace(name='materialize_plan', parameters='{"plan":"Authored plan"}')]),
                       ('Plan created.', [])])
    rollout = env.actor.run(env.request, env.runtime)
    assert env.artifact.read_text() == '{"plan": "Authored plan"}'
    sample = env.sample
    response_ids = sample.tokens[-sample.response_length:]
    trained_ids = [t for t, m in zip(response_ids, sample.loss_mask) if m]
    trained_probs = [p for p, m in zip(sample.rollout_log_probs, sample.loss_mask) if m]
    assert trained_ids == [701, 999, 702, 999]  # not Tokenizer.encode(output['text'])
    assert trained_probs == [-0.3, -0.1, -0.2, -0.1]
    assert sum(sample.loss_mask) == 4 and len(sample.loss_mask) == sample.response_length
    assert all(p == 0 for p, m in zip(sample.rollout_log_probs, sample.loss_mask) if not m)
    assert env.requests[1]['input_ids'] == sample.tokens[:-2]
    assert rollout.safe_metadata['native_codex_actor'] is False
    assert rollout.policy_events[1].content['policy_visible_context']['stage'] == 'S1'
    assert rollout.assistant_output == 'Plan created.'
    for event in rollout.policy_events:
        assert event.event_blake3 == blake3_hex({key: getattr(event, key) for key in
                                                ('event_id', 'role', 'content', 'tool_call_ids')})


def test_malformed_model_arguments_are_outcome_not_host_execution(actor_environment):
    env = actor_environment
    env.responses.append(output('bad action', [701], [-0.3]))
    env.parsed.append(('', [SimpleNamespace(name='materialize_plan', parameters='{"wrong":1}')]))
    rollout = env.actor.run(env.request, env.runtime)
    assert not env.artifact.exists()
    assert rollout.assistant_output == 'bad action'
    assert env.sample.loss_mask == [1]


def test_missing_sampled_logprobs_is_infrastructure_error(actor_environment):
    env = actor_environment
    env.responses.append(output('action', [701], [None]))
    with pytest.raises(ValueError, match='actual sampled-token'):
        env.actor.run(env.request, env.runtime)
    assert not env.artifact.exists()


def test_unexpected_reasoning_is_not_public_even_when_truncated():
    assert observable_model_text('<think>private unfinished reasoning') == ''
    assert observable_model_text('<think>private reasoning</think>Public result') == 'Public result'
    assert observable_model_text('Public text') == 'Public text'
    assert observable_model_text('Unclosed private span', thinking_prefix_open=True) == ''
    assert observable_model_text('Private span</think>Public', thinking_prefix_open=True) == 'Public'


@pytest.mark.parametrize('stage', [Stage.S1, Stage.S2, Stage.S3])
def test_stage_guidance_and_skill_scope_are_preserved(stage):
    context = SimpleNamespace(stage_tool_guidance=SimpleNamespace(focus=stage, prompt_text='UNCHANGED GUIDANCE'))
    instruction = stage_actor_instruction(context, stage.value)
    assert 'UNCHANGED GUIDANCE' in instruction
    assert f"stage='{stage.value}'" in instruction


def test_unsupported_stage_is_not_silently_trained_as_s1():
    with pytest.raises(ValueError, match='S1-S3'):
        stage_actor_instruction(SimpleNamespace(), 'S4')


def test_actual_thinking_is_kept_in_rl_tokens_not_public_events(actor_environment):
    env = actor_environment
    env.actor.args.eva_enable_thinking = True
    template_args = []
    env.actor.tokenizer.apply_chat_template = lambda messages, **kw: template_args.append(kw) or 'prompt<think>\n'
    env.responses.append(output('PRIVATE SPAN</think>Plan pending.', [701, 702, 999], [-0.3, -0.2, -0.1]))
    env.parsed.append(('Plan pending.', []))
    rollout = env.actor.run(env.request, env.runtime)
    assert template_args[0]['enable_thinking'] is True
    assert rollout.safe_metadata['thinking_enabled'] is True
    assert env.sample.tokens[-3:] == [701, 702, 999]
    assert all('PRIVATE SPAN' not in str(event.content) for event in rollout.policy_events)


def test_sampled_policy_evidence_survives_later_judge_error(tmp_path):
    sample = SimpleNamespace(tokens=[10, 11, 701, 98, 99, 702], response_length=4,
                             loss_mask=[1, 0, 0, 1], rollout_log_probs=[-0.3, 0.0, 0.0, -0.2])
    path = tmp_path / 'sampled-policy-tokens.json'
    preserve_sampled_tokens(sample, path)
    with pytest.raises(RuntimeError, match='judge failure'):
        raise RuntimeError('judge failure')
    receipt = json.loads(path.read_text())
    assert receipt['tokens'] == sample.tokens
    assert receipt['sampled_token_count'] == 2
    assert receipt['reward_emitted'] is False
    assert receipt['sft_export_eligible'] is False
    assert 'reward' not in receipt
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        preserve_sampled_tokens(sample, path)


@pytest.mark.parametrize('mutation', [
    {'loss_mask': [1]}, {'rollout_log_probs': [-0.3]},
    {'rollout_log_probs': [-0.3, 0.5]}, {'rollout_log_probs': [float('nan'), 0]},
    {'response_length': 4}, {'loss_mask': [0, 0]},
])
def test_invalid_sampled_policy_evidence_fails_before_write(tmp_path, mutation):
    values = dict(tokens=[10, 701, 98], response_length=2,
                  loss_mask=[1, 0], rollout_log_probs=[-0.3, 0.0])
    values.update(mutation)
    path = tmp_path / 'sampled-policy-tokens.json'
    with pytest.raises(ValueError, match='alignment'):
        preserve_sampled_tokens(SimpleNamespace(**values), path)
    assert not path.exists()
