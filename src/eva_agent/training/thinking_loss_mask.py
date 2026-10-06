"""Replay the actual bound thinking processor to identify forced tokens.

The processor runs against a write-recording logits object, not a vocabulary
tensor. This preserves its exact state machine without GPU work or a second
implementation of its token-counting rules.
"""
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex


def require_unmodified_policy_sampling(params):
    """Bind current PPO to raw-model probabilities on non-forced tokens.

    This SGLang revision returns probabilities before top-p/k/min-p
    renormalization. Transforming the behavior distribution would require a
    separately verified trainer replay implementation.
    """
    fixed = {'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0,
        'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0,
        'min_new_tokens': 0, 'n': 1, 'ignore_eos': False}
    structural = {'max_new_tokens', 'stop', 'stop_token_ids', 'stop_regex',
        'skip_special_tokens', 'spaces_between_special_tokens', 'no_stop_trim', 'sampling_seed'}
    if set(params) - set(fixed) - structural:
        raise ValueError('portable_policy_sampling_transform_unverified')
    if any(params.get(name, expected) != expected for name, expected in fixed.items()):
        raise ValueError('portable_policy_requires_temperature1_full_support_no_penalties')


class _LogitWrites:
    def __init__(self):
        self.cleared = False
        self.forced = None

    def __setitem__(self, key, value):
        if key == (0, slice(None)) and value == -float('inf') and not self.cleared:
            self.cleared = True
        elif (isinstance(key, tuple) and len(key) == 2 and key[0] == 0 and type(key[1]) is int and
                value == 0.0 and self.cleared and self.forced is None):
            self.forced = key[1]
        else:
            raise ValueError('unsupported_thinking_processor_logit_operation')


class QwenThinkingLossMask:
    def __init__(self, *, source_sha256, serialized_processor, wire_budget):
        from sglang.srt.sampling.custom_logit_processor import Qwen35ThinkingBudgetLogitProcessor
        self.processor_class = Qwen35ThinkingBudgetLogitProcessor
        source = Path(inspect.getfile(self.processor_class)).read_bytes()
        if (hashlib.sha256(source).hexdigest() != source_sha256 or
                self.processor_class.to_str() != serialized_processor or
                type(wire_budget) is not int or wire_budget < 0):
            raise ValueError('thinking_mask_processor_binding_changed')
        self.binding = {'schema': 'eva.bound-thinking-processor-loss-mask.v1',
            'processor_class': self.processor_class.__module__ + '.' + self.processor_class.__name__,
            'processor_source_blake3': blake3_bytes(source),
            'processor_source_sha256': source_sha256,
            'serialized_processor': serialized_processor, 'wire_budget': wire_budget,
            'forced_tokens_are_conditioning_only': True, 'sampled_output_ids_modified': False}

    def __call__(self, prompt, output):
        if not prompt or not output:
            raise ValueError('thinking_mask_requires_exact_nonempty_tokens')
        request = SimpleNamespace(origin_input_ids=prompt, output_ids=[])
        params = [{'thinking_budget': self.binding['wire_budget'], '__req__': request}]
        processor = self.processor_class()
        mask, positions = [], []
        for index, token in enumerate(output):
            writes = _LogitWrites()
            if processor(writes, params) is not writes or (writes.cleared and writes.forced is None):
                raise ValueError('thinking_processor_return_or_force_changed')
            if writes.forced is not None:
                if token != writes.forced:
                    raise ValueError('sampled_token_violates_bound_thinking_processor')
                positions.append({'position': index, 'token_id': token})
            mask.append(int(writes.forced is None))
            request.output_ids.append(token)
        return mask, {**deepcopy(self.binding), 'forced_positions': positions,
            'exact_prompt_and_output_blake3': blake3_hex({'prompt': prompt, 'output': output}),
            'loss_mask_blake3': blake3_hex(mask)}


def validate_thinking_mask(segment):
    binding = segment['loss_mask_provenance']
    masker = QwenThinkingLossMask(source_sha256=binding['processor_source_sha256'],
        serialized_processor=binding['serialized_processor'], wire_budget=binding['wire_budget'])
    mask, expected = masker(segment['prompt_token_ids'], segment['output_token_ids'])
    if binding != expected or segment['loss_mask'] != mask:
        raise ValueError('thinking_policy_loss_mask_changed')
    return mask
