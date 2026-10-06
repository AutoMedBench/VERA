"""On-policy SGLang rollouts over the existing EvaMed tool/workspace runtime.

Codex interoperability is checked separately by the native harness canary. This
Slime actor is explicitly labelled SGLang (never a fabricated Codex transcript).
Sampled token IDs/log probabilities are kept exactly; host observations have
zero loss. The same workspace Agent Judge and rubric calculate the RL reward.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from eva_agent.pipeline import Cohort, ModelTarget, ToolCall
from eva_agent.pipeline.contracts import ProviderRollout, TrajectoryEvent
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes, canonical_value
from eva_agent.training.teacher_worker import CampaignV2TeacherContextPool, execute_single_rollout, load_bulk_record


_CONTEXTS = None
_CONTEXT_LOCK = Lock()


def _event(role: str, content: Any, call_ids=()) -> TrajectoryEvent:
    core = {'event_id': str(uuid4()), 'role': role, 'content': content,
            'tool_call_ids': tuple(call_ids)}
    return TrajectoryEvent(**core, event_blake3=blake3_hex(core))


def tool_observation_tail(observations: list[dict], *, has_end_token: bool,
                          thinking: bool = False) -> str:
    """Official Qwen3.5 tool-result wrapper appended after sampled assistant IDs."""
    tail = ('\n' if has_end_token else '<|im_end|>\n') + '<|im_start|>user'
    for observation in observations:
        tail += '\n<tool_response>\n' + json.dumps(observation, ensure_ascii=False, separators=(',', ':')) + '\n</tool_response>'
    prefix = '<think>\n' if thinking else '<think>\n\n</think>\n\n'
    return tail + '<|im_end|>\n<|im_start|>assistant\n' + prefix


def observable_model_text(text: str, *, thinking_prefix_open: bool = False) -> str:
    """Keep public output, including when the prompt already opened thinking."""
    if thinking_prefix_open and '</think>' not in text:
        return ''
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[-1]
    # An unfinished private block has no public answer following it. Its token
    # IDs remain private on-policy training evidence, not an SFT/judge message.
    if '<think>' in text:
        text = text.split('<think>', 1)[0]
    return text


def stage_actor_instruction(context, stage: str) -> str:
    """Use the existing focus guidance; do not synthesize later-stage bindings."""
    if stage not in {'S1', 'S2', 'S3'}:
        raise ValueError('GRPO guidance currently supports S1-S3; later stages are not enabled')
    guidance = context.stage_tool_guidance
    if guidance is None:
        raise ValueError('GRPO requires public host-derived stage tool guidance')
    from eva_agent.training.teacher_focus import focus_only_teacher_instructions
    return (
        'You are a medical research coding agent. Use the available tools and '
        'stage-permitted skills to complete the task. Preserve uncertainty and '
        'host-supplied identifiers. Independent read-only calls may run together. '
        'Search/load relevant skills using only advertised identifiers; never '
        'invent a tool or a skill. Keep between-tool commentary brief.\n\n'
        + focus_only_teacher_instructions(context)
    )


def _write_private_json(path: Path, document: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps(document, allow_nan=False) + '\n')


def preserve_sampled_tokens(sample, path: Path) -> None:
    """Retain actual policy samples before an external judge can fail.

    This private artifact is not a reward, SFT export, or accepted GRPO batch.
    It contains original token IDs, including any unprojected model reasoning.
    """
    response_length = sample.response_length
    masks, probabilities = sample.loss_mask, sample.rollout_log_probs
    if (not isinstance(response_length, int) or response_length <= 0
            or response_length > len(sample.tokens)
            or len(masks) != response_length or probabilities is None
            or len(probabilities) != response_length
            or any(mask not in (0, 1) for mask in masks)
            or not any(masks)
            or any(not math.isfinite(value) for value in probabilities)
            or any(value != 0 for value, mask in zip(probabilities, masks) if not mask)):
        raise ValueError('sampled token/logprob/loss-mask alignment differs')
    document = {
        'schema': 'eva.private-sampled-policy-tokens.v1',
        'tokens': sample.tokens, 'response_length': response_length,
        'loss_mask': masks, 'rollout_log_probs': probabilities,
        'sampled_token_count': sum(masks),
        'reward_emitted': False, 'judge_status': 'not_yet_completed',
        'sft_export_eligible': False,
    }
    _write_private_json(path, document)


class SGLangEvaActor:
    def __init__(self, *, args, sample, tokenizer, context, sampling_params):
        self.args, self.sample, self.tokenizer = args, sample, tokenizer
        self.context, self.sampling_params = context, sampling_params

    def run(self, request, runtime) -> ProviderRollout:
        import httpx
        import jsonschema
        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from slime.utils.types import Sample

        system = stage_actor_instruction(self.context, request.sandbox.stage.value)
        thinking = getattr(self.args, 'eva_enable_thinking',
                           os.environ.get('EVA_GRPO_ENABLE_THINKING', '1') == '1')
        max_frontiers = int(os.environ.get('EVA_GRPO_MAX_TOOL_FRONTIERS', '12'))
        if type(thinking) is not bool or not 1 <= max_frontiers <= 64:
            raise ValueError('GRPO thinking or tool-frontier bounds differ')
        user_context = {'instruction': request.sandbox.instruction,
                        'policy_visible_context': canonical_value(request.policy_visible_context)}
        user = json.dumps(user_context, ensure_ascii=False)
        tools = canonical_value(request.available_tools)
        chat = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        prompt = self.tokenizer.apply_chat_template(chat, tools=tools, tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=thinking)
        sample = self.sample
        sample.prompt = prompt
        sample.tokens = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        sample.response = ''
        sample.response_length, sample.loss_mask = 0, []
        sample.rollout_log_probs = None
        initial_length = len(sample.tokens)
        if initial_length + 128 >= self.args.seq_length:
            raise ValueError(f'GRPO prompt {initial_length} exceeds configured context; never truncated')
        events = [_event('system', system), _event('user', user_context)]
        generated, final, requests = 0, '', []
        parser = FunctionCallParser(tools=[Tool.model_validate(tool) for tool in tools], tool_call_parser='qwen3_coder')
        schemas = {tool['function']['name']: tool['function']['parameters'] for tool in tools}
        url = f'http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate'
        with httpx.Client(timeout=180, trust_env=False) as client:
            for step in range(max_frontiers):
                remaining = min(self.args.rollout_max_response_len - generated,
                                self.args.seq_length - len(sample.tokens))
                if remaining < 1:
                    sample.status = Sample.Status.TRUNCATED
                    break
                params = dict(self.sampling_params, max_new_tokens=remaining,
                              skip_special_tokens=False)
                response = client.post(url, json={'input_ids': list(sample.tokens),
                                       'sampling_params': params, 'return_logprob': True})
                response.raise_for_status()
                output = response.json()
                meta = output['meta_info']
                log_rows = meta.get('output_token_logprobs')
                if not log_rows or any(row[0] is None for row in log_rows):
                    raise ValueError('SGLang did not return actual sampled-token log probabilities')
                tokens, logprobs = [int(r[1]) for r in log_rows], [float(r[0]) for r in log_rows]
                text = output['text']
                generated += len(tokens)
                sample.append_response_tokens(self.args, tokens=tokens, log_probs=logprobs,
                    trainable=True, meta_info=meta, text=text, update_terminal_info=False)
                requests.append({'step': step, 'prompt_tokens': len(sample.tokens)-len(tokens),
                                 'generated_tokens': len(tokens), 'finish_reason': meta.get('finish_reason')})
                # Actual sampled IDs remain private RL evidence. A thinking
                # prefix in the prompt must close before any public projection.
                visible = observable_model_text(text, thinking_prefix_open=thinking)
                try:
                    normal, calls = parser.parse_non_stream(visible.replace('<|im_end|>', '').strip())
                except (ValueError, TypeError, KeyError):
                    # Invalid model syntax is retained as an observed outcome,
                    # not confused with a failed infrastructure request.
                    final = visible
                    events.append(_event('assistant', final))
                    sample.status = Sample.Status.COMPLETED
                    break
                if not calls:
                    final = normal
                    events.append(_event('assistant', final))
                    finish = meta.get('finish_reason', {})
                    sample.status = Sample.Status.TRUNCATED if finish.get('type') == 'length' else Sample.Status.COMPLETED
                    break
                host_calls = []
                assistant_calls = []
                try:
                    for call in calls:
                        arguments = json.loads(call.parameters)
                        if call.name not in schemas:
                            raise ValueError('model called a tool outside the unchanged catalog')
                        jsonschema.validate(arguments, schemas[call.name])
                        call_id = str(uuid4())
                        host_calls.append(ToolCall(call_id=call_id, name=call.name, arguments=arguments))
                        assistant_calls.append({'id': call_id, 'type': 'function',
                                                'function': {'name': call.name, 'arguments': arguments}})
                except (ValueError, TypeError, jsonschema.ValidationError):
                    final = visible
                    events.append(_event('assistant', final))
                    sample.status = Sample.Status.COMPLETED
                    break
                events.append(_event('assistant', {'text': normal, 'tool_calls': assistant_calls},
                                     [call.call_id for call in host_calls]))
                results = runtime.execute(host_calls)
                observations = []
                for result in results:
                    observation = canonical_value(result)
                    observations.append(observation)
                    events.append(_event('tool', observation, (result.call_id,)))
                if step == max_frontiers - 1:
                    sample.status = Sample.Status.TRUNCATED
                    break
                tail = tool_observation_tail(observations,
                    has_end_token=tokens[-1] == self.tokenizer.convert_tokens_to_ids('<|im_end|>'),
                    thinking=thinking)
                env_tokens = self.tokenizer.encode(tail, add_special_tokens=False)
                if len(sample.tokens) + len(env_tokens) >= self.args.seq_length:
                    sample.status = Sample.Status.TRUNCATED
                    break
                sample.append_response_tokens(self.args, tokens=env_tokens, trainable=False)
        if not generated:
            raise ValueError('no on-policy action was generated')
        sample.metadata = dict(sample.metadata or {}, actor_backend='sglang_eva_tools',
                               thinking_enabled=thinking, request_receipts=requests,
                               original_prompt_tokens=initial_length,
                               generated_tokens=generated, canonical_tool_schemas_changed=False)
        return ProviderRollout(assistant_output=final, policy_events=tuple(events),
            provider_receipt_blake3=blake3_hex({'requests': requests, 'events': events}),
            safe_metadata={'actor_backend': 'sglang_eva_tools', 'native_codex_actor': False,
                           'thinking_enabled': thinking, 'maximum_tool_frontiers': max_frontiers,
                           'context_compaction_enabled': False,
                           'sampled_token_logprobs_retained': True})


def generate_grpo_rollout(args, rollout_id: int, data_buffer, evaluation=False):
    """Slime hook: concurrent genuine model rollouts, then independent Agent Judge."""
    if evaluation:
        raise ValueError('use separate held-out evaluation for clinical capability claims')
    from slime.rollout.sglang_rollout import GenerateState
    from eva_agent.training.slime_agent_judge import grade_rollout, resolve_judge_backend
    global _CONTEXTS
    with _CONTEXT_LOCK:
        if _CONTEXTS is None:
            _CONTEXTS = CampaignV2TeacherContextPool()
    state = GenerateState(args)
    groups = data_buffer.get_samples(args.rollout_batch_size)
    samples = [sample for group in groups for sample in group]
    output = Path(args.save).parent / 'rollouts' / f'{rollout_id:06d}'
    output.mkdir(parents=True, exist_ok=False)

    def one(index_sample):
        index, sample = index_sample
        meta = sample.metadata
        if meta.get('judge_backend') != resolve_judge_backend():
            raise ValueError('GRPO data and explicitly selected workspace Judge backend differ')
        record = load_bulk_record(Path(meta['bulk_root']), meta['sandbox_id'])
        context, _ = _CONTEXTS.load(record)
        sample_id = str(uuid4())
        sample_root = output / sample_id
        sample_root.mkdir()
        actor = SGLangEvaActor(args=args, sample=sample, tokenizer=state.tokenizer,
                              context=context, sampling_params=state.sampling_params)
        rollout = execute_single_rollout(record=record, context=context, provider=actor,
            target=ModelTarget(Cohort.STRONG, 'Qwen/Qwen3.5-9B', 'sglang-local'),
            workspace_root=sample_root / 'workspace', task_id=sample_id, route_id='qwen_3_5_9b')
        rollout['schema'] = 'eva.sglang-teacher-full-trajectory.v1'
        (sample_root / 'trajectory.json').write_bytes(canonical_json_bytes(rollout))
        preserve_sampled_tokens(sample, sample_root / 'sampled-policy-tokens.json')
        verdict = asyncio.run(grade_rollout(rollout, rubric=context.rubric,
                                            output_root=sample_root / 'judge', sample_id=sample_id))
        sample.reward = verdict['reward']
        # Sibling GRPO trajectories share group_index, never rollout_id: that
        # latter field groups compact segments of a single trajectory in Slime.
        sample.rollout_id = sample.index
        sample.metadata['training_iteration'] = rollout_id
        sample.metadata.update({'judge_result': verdict, 'trajectory_path': str(sample_root / 'trajectory.json')})
        # Private training receipt only: these are the actual sampled IDs, not a
        # re-tokenized reconstruction of a Chat/Codex completion.
        _write_private_json(sample_root / 'training-tokens.json', {
            'tokens': sample.tokens, 'response_length': sample.response_length,
            'loss_mask': sample.loss_mask, 'rollout_log_probs': sample.rollout_log_probs,
            'reward': sample.reward,
        })
        return sample

    with ThreadPoolExecutor(max_workers=min(4, len(samples))) as executor:
        result = list(executor.map(one, enumerate(samples)))
    rewards = [sample.reward for sample in result]
    (output / 'summary.json').write_text(json.dumps({'samples': len(result), 'rewards': rewards,
        'reward_spread': max(rewards)-min(rewards), 'rubric_agent_judged': True,
        'zero_variance_group': max(rewards) == min(rewards)}) + '\n')
    return result
