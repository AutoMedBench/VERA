"""Exercise real Slime Sample token appends and EVA's pre-judge receipt on CPU."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from run_full_parameter import REPO, SLIME

sys.path[:0] = [str(REPO / 'src'), str(SLIME)]
import torch
from slime.utils.types import Sample
from eva_agent.training.slime_rollout import preserve_sampled_tokens

assert not torch.cuda.is_initialized()
args = SimpleNamespace()
sample = Sample(tokens=[10, 11], response_length=0, loss_mask=[], rollout_log_probs=None)
sample.append_response_tokens(args, tokens=[701, 999], log_probs=[-0.3, -0.1],
                              trainable=True, meta_info={'finish_reason': {'type': 'stop'}},
                              text='synthetic action', update_terminal_info=False)
assert sample.status == Sample.Status.PENDING
sample.append_response_tokens(args, tokens=[97, 98, 99], trainable=False)
sample.append_response_tokens(args, tokens=[702, 999], log_probs=[-0.2, -0.1],
                              trainable=True, meta_info={'finish_reason': {'type': 'stop'}},
                              text='synthetic result', update_terminal_info=False)
sample.status = Sample.Status.COMPLETED
assert sample.tokens == [10, 11, 701, 999, 97, 98, 99, 702, 999]
assert sample.response_length == 7 and sample.effective_response_length == 4
assert sample.loss_mask == [1, 1, 0, 0, 0, 1, 1]
assert sample.rollout_log_probs == [-0.3, -0.1, 0.0, 0.0, 0.0, -0.2, -0.1]
with tempfile.TemporaryDirectory(prefix='eva-native-sample-preflight-') as temporary:
    path = Path(temporary) / 'private-sampled-policy-tokens.json'
    preserve_sampled_tokens(sample, path)
    receipt = json.loads(path.read_text())
    assert receipt['tokens'] == sample.tokens
    assert receipt['loss_mask'] == sample.loss_mask
    assert receipt['rollout_log_probs'] == sample.rollout_log_probs
    assert receipt['sampled_token_count'] == sample.effective_response_length
    assert receipt['reward_emitted'] is False and 'reward' not in receipt
    assert path.stat().st_mode & 0o777 == 0o600
assert not torch.cuda.is_initialized()
print('EVA_NATIVE_SAMPLE_PREFLIGHT ' + json.dumps({
    'status': 'complete', 'native_sample_class': Sample.__module__ + '.' + Sample.__name__,
    'prompt_tokens': 2, 'response_length_including_environment': 7,
    'sampled_loss_tokens': 4, 'environment_tokens_masked_with_zero_logprob': 3,
    'exact_ids_and_probabilities_preserved': True, 'private_receipt_permissions': '0600',
    'provider_calls': 0, 'gpu_initialized': False, 'synthetic_infrastructure_only': True,
}), flush=True)
