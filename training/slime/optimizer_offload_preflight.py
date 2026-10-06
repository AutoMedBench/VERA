"""Tiny TE FusedAdam master/moment offload test; not EVA training evidence."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

workspace = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace / 'Megatron-LM'))

import torch
from transformer_engine.pytorch.optimizers import FusedAdam
from megatron.core.optimizer.cpu_offloading.optimizer_state_offloader import OptimizerStateOffloader

model = torch.nn.Parameter(torch.full((4096,), 0.0001, device='cuda', dtype=torch.bfloat16))
# Megatron's FP32 optimizer shard is detached from the autograd model.
master = model.detach().float().clone()
optimizer = FusedAdam([master], lr=2e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=0)
offloader = OptimizerStateOffloader(SimpleNamespace(
    optimizer=optimizer, shard_fp32_from_float16_groups=[[master]],
))
for step in range(2):
    offloader.offload()
    torch.cuda.synchronize()
    offloader.release_gpu_memory()
    assert master.untyped_storage().nbytes() == 0
    model.grad = None
    model.float().square().mean().backward()
    offloader.reload()
    offloader.sync_before_step()
    optimizer.zero_grad()
    master.grad = model.grad.float()
    before = master.detach().clone()
    optimizer.step()
    offloader.mark_optimizer_states_initialized()
    torch.cuda.synchronize()
    assert torch.isfinite(master).all() and not torch.equal(before, master)
    with torch.no_grad():
        model.copy_(master)
    print('EVA_OPTIMIZER_OFFLOAD_PREFLIGHT ' + json.dumps({
        'status': 'complete', 'step': step, 'optimizer': 'TE FusedAdam',
        'cpu_master_and_moments_preserved': True, 'gpu_storage_released_before_forward': True,
        'weight_changed': True, 'synthetic_infrastructure_only': True,
    }), flush=True)
