"""Probe Slime singleton collective completion in an isolated, tiny process.

These tiny synthetic parameters test infrastructure only, never EVA training
acceptance. Use --reloadable to diagnose the version-specific wrapper path.
"""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reloadable', action='store_true')
parser.add_argument('--async-op', action='store_true')
args = parser.parse_args()
workspace = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace / 'slime-upstream'))

import torch
import torch.distributed as dist
from transformer_engine.pytorch.optimizers import FusedAdam
from slime.utils.reloadable_process_group import monkey_patch_torch_dist, register_default_process_group

directory = Path(tempfile.mkdtemp(prefix='eva-collective-preflight-'))
dist.init_process_group('nccl', init_method=(directory / 'rendezvous').as_uri(),
                        rank=0, world_size=1, timeout=timedelta(seconds=30))
# Megatron imports and caches these functions before Slime installs its wrappers.
# Test that exact call path, not only the subsequently patched dist attributes.
cached_coalescing_manager = dist._coalescing_manager
cached_reduce_scatter = dist.reduce_scatter_tensor
monkey_patch_torch_dist()
if args.reloadable:
    register_default_process_group(timedelta(seconds=30))
group = dist.new_group([0], backend='nccl')
parameter = torch.nn.Parameter(torch.ones(1024, device='cuda'))
optimizer = FusedAdam([parameter], lr=2e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=0)
try:
    for index in range(2):
        optimizer.zero_grad()
        parameter.square().mean().backward()
        # The one-rank distributed optimizer reduces its gradient buffer in place.
        output = parameter.grad.view_as(parameter.grad)
        with cached_coalescing_manager(group=group, async_ops=args.async_op) as manager:
            cached_reduce_scatter(output, parameter.grad, group=group, async_op=args.async_op)
        if args.async_op:
            manager.wait()
        parameter.grad = output
        before = parameter.detach().clone()
        optimizer.step()
        torch.cuda.synchronize()
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        assert not torch.equal(before, parameter)
        print('EVA_COLLECTIVE_PREFLIGHT ' + json.dumps({
            'status': 'complete', 'step': index, 'process_group_type': type(group).__name__,
            'reloadable': args.reloadable, 'finite_nonzero_gradient': True,
            'async_op': args.async_op,
            'optimizer': type(optimizer).__name__,
            'weight_changed': True, 'synthetic_infrastructure_only': True,
        }), flush=True)
finally:
    dist.destroy_process_group()
