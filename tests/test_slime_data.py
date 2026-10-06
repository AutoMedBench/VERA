from copy import deepcopy
from types import SimpleNamespace

import pytest

from eva_agent.training.slime_data import render_sft_row, generate_sft_rollout
from eva_agent.training.slime_rollout import tool_observation_tail


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return ''.join('<|im_start|>' + m['role'] + '\n' +
                       ('<think>\n\n</think>\n\n' if m['role'] == 'assistant' else '') +
                       m['content'] + '<|im_end|>\n' for m in messages)

    def __call__(self, text, **kwargs):
        return {'input_ids': list(map(ord, text)), 'offset_mapping': [(i, i+1) for i in range(len(text))]}


def row():
    return {'row_id': 'observable-decision', 'tools': [],
            'messages': [{'role': 'user', 'content': 'original task'},
                         {'role': 'assistant', 'content': 'old decision'},
                         {'role': 'tool', 'content': 'host evidence'},
                         {'role': 'assistant', 'content': 'new decision'}],
            'loss_contract': {'supervise_exactly_one_assistant_message': True, 'assistant_message_index': 3},
            'metadata': {'stage': 'S1', 'quality_tier': 'strict_full_trajectory'}}


def test_only_final_assistant_is_supervised_and_source_is_unchanged():
    source = row()
    before = deepcopy(source)
    sample = render_sft_row(source, CharacterTokenizer())
    assert source == before
    metadata = sample['metadata']
    target = ''.join(chr(t) for t, m in zip(metadata['tokens'], metadata['loss_mask']) if m)
    assert target == 'new decision<|im_end|>\n'
    assert metadata['response_length'] == len(target)
    assert 'host evidence' not in target and '<think>' not in target


def test_sft_hook_preserves_tokens_and_masks():
    rendered = render_sft_row(row(), CharacterTokenizer())
    sample = SimpleNamespace(metadata=rendered['metadata'], index=101)
    data = SimpleNamespace(get_samples=lambda count: [[sample]])
    result = generate_sft_rollout(SimpleNamespace(rollout_batch_size=1), 8, data)
    assert result == [sample]
    assert sample.rollout_id == 101
    assert sample.metadata['training_iteration'] == 8
    assert sample.tokens == rendered['metadata']['tokens']
    assert sample.loss_mask == [1] * sample.response_length
    sample.metadata['loss_mask'] = [0] * len(sample.tokens)
    with pytest.raises(ValueError, match='alignment'):
        generate_sft_rollout(SimpleNamespace(rollout_batch_size=1), 9, data)


def test_independent_sft_trajectories_do_not_collapse_to_one_rollout():
    rendered = render_sft_row(row(), CharacterTokenizer())
    samples = [SimpleNamespace(metadata=deepcopy(rendered['metadata']), index=index) for index in (10, 11)]
    data = SimpleNamespace(get_samples=lambda count: [[sample] for sample in samples])
    result = generate_sft_rollout(SimpleNamespace(rollout_batch_size=2), 7, data)
    assert [sample.rollout_id for sample in result] == [10, 11]
    assert [sample.metadata['training_iteration'] for sample in result] == [7, 7]


def test_tool_tail_matches_qwen_boundaries_without_duplicate_eos():
    tail = tool_observation_tail([{'path': 'work/stage-plan.json'}, {'gate_passed': True}], has_end_token=True)
    assert tail.startswith('\n<|im_start|>user\n<tool_response>')
    assert tail.count('<tool_response>') == 2
    assert tail.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    assert tool_observation_tail([], has_end_token=False).startswith('<|im_end|>\n')
