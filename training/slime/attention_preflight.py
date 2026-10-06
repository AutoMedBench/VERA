"""Small real-shape Qwen attention forward/backward probe before model loading."""

import argparse
import json
import os
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--backend', choices=['fused', 'flash', 'unfused', 'auto'], default='fused')
parser.add_argument('--layout', choices=['thd', 'sbhd'], default='thd')
parser.add_argument('--lengths', default='1530,1762,36')
args = parser.parse_args()
os.environ['NVTE_DEBUG'] = '1'
os.environ['NVTE_DEBUG_LEVEL'] = '2'
for name, backend in [('NVTE_FUSED_ATTN', 'fused'), ('NVTE_FLASH_ATTN', 'flash'), ('NVTE_UNFUSED_ATTN', 'unfused')]:
    os.environ[name] = str(int(args.backend in (backend, 'auto')))

import torch
from transformer_engine.pytorch import DotProductAttention

torch.manual_seed(260910)
lengths = [int(value) for value in args.lengths.split(',')]
assert lengths and min(lengths) > 0
shape = (sum(lengths),) if args.layout == 'thd' else (max(lengths), 1)
query = torch.randn(*shape, 16, 256, dtype=torch.bfloat16, device='cuda', requires_grad=True)
key = torch.randn(*shape, 4, 256, dtype=torch.bfloat16, device='cuda', requires_grad=True)
value = torch.randn(*shape, 4, 256, dtype=torch.bfloat16, device='cuda', requires_grad=True)
attention = DotProductAttention(16, 256, num_gqa_groups=4, qkv_format=args.layout,
                               attention_dropout=0,
                               attn_mask_type='padding_causal' if args.layout == 'thd' else 'causal')
kwargs = {}
if args.layout == 'thd':
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    boundaries = torch.tensor(offsets, device='cuda', dtype=torch.int32)
    kwargs = dict(cu_seqlens_q=boundaries, cu_seqlens_kv=boundaries,
                  max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths))
started = time.monotonic()
output = attention(query, key, value, **kwargs)
loss = output.float().square().mean()
loss.backward()
torch.cuda.synchronize()
assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
           for tensor in (query, key, value))
print('EVA_ATTENTION_PREFLIGHT ' + json.dumps({
    'status': 'complete', 'backend': args.backend, 'layout': args.layout,
    'query_heads': 16, 'kv_heads': 4, 'head_dim': 256, 'dtype': 'bfloat16',
    'lengths': lengths if args.layout == 'thd' else [max(lengths)],
    'loss': loss.item(), 'finite_nonzero_qkv_gradients': True,
    'seconds': time.monotonic() - started, 'peak_bytes': torch.cuda.max_memory_allocated(),
}), flush=True)
