"""No-model full-epoch rehearsal using installed Slime's actual loader.

Native runtime imports can initialize a CUDA context; no model, optimizer,
forward/backward, or provider generation is performed by this rehearsal.
"""

import argparse
from argparse import Namespace
import json
import os
from pathlib import Path
import sys

from run_full_parameter import (MEGATRON, REPO, SLIME, WORKSPACE, build_command,
                                cuda_runtime_environment, plan_sft_epoch, summarize_epoch_consumption)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', required=True, type=Path)
parser.add_argument('--receipt-dir', required=True, type=Path)
options = parser.parse_args()
options.receipt_dir.mkdir(parents=True, exist_ok=False)
for source in (REPO / 'src', SLIME, MEGATRON):
    sys.path.insert(0, str(source))
os.environ.update(cuda_runtime_environment())
os.environ['SGLANG_ENABLE_JIT_DEEPGEMM'] = '0'

from epoch_data import generate_sft_rollout
from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.arguments import parse_args
from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.misc import should_run_periodic_action
import torch

request = Namespace(stage='sft', model=WORKSPACE / 'Qwen3.5-9B', load=None,
                    data=options.data, output=Path('/tmp/eva-epoch-preflight-no-model-output'),
                    steps=2, batch_size=8, max_tokens=24576, one_epoch=True, max_updates=500,
                    save_interval=50, checkpoint_model_only=True, export_hf=True)
plan = plan_sft_epoch(request)
request.steps = plan['optimizer_updates_planned']
consumption = options.receipt_dir / 'data-consumption.jsonl'
os.environ.update(EVA_SFT_EPOCH_ROWS=str(plan['dataset_rows']),
                  EVA_SFT_EPOCH_STEPS=str(request.steps), EVA_SFT_DATA_RECEIPT=str(consumption))
sys.argv = build_command(request)
args = parse_args()
assert args.no_save_optim and args.num_rollout == request.steps and args.rollout_shuffle
source = RolloutDataSourceWithBuffer(args)
microbatch_counts = []
for iteration in range(request.steps):
    samples = generate_sft_rollout(args, iteration, source)
    schedule = build_dp_schedule(
        args, {'dp_size': 1, 'cp_size': 1, 'vpp_size': 1, 'microbatch_group_size_per_vp_stage': 1},
        [len(sample.tokens) for sample in samples], global_batch_size=8,
        rollout_indices=[sample.rollout_id for sample in samples])
    assert schedule[3] == [8] and sorted(schedule[0][0]) == list(range(8))
    microbatch_counts.extend(schedule[2])
try:
    generate_sft_rollout(args, request.steps, source)
except ValueError:
    pass
else:
    raise AssertionError('Epoch wrapper permitted an additional wrapped batch')
saves = [iteration + 1 for iteration in range(request.steps)
         if should_run_periodic_action(iteration, 50, None, request.steps)]
assert saves[-1] == request.steps
assert torch.cuda.memory_allocated() == 0
report = {'status': 'complete', 'epoch_plan': plan,
          'consumption': summarize_epoch_consumption(consumption, plan),
          'save_after_updates': saves, 'optimizer_state_saved': False,
          'microbatch_count_min': min(microbatch_counts), 'microbatch_count_max': max(microbatch_counts),
          'ray_started': False, 'cuda_context_initialized': torch.cuda.is_initialized(),
          'gpu_tensor_bytes_live': torch.cuda.memory_allocated(),
          'full_model_instantiated': False, 'optimizer_updates': 0, 'provider_calls': 0,
          'extra_wrapped_batch_rejected': True}
(options.receipt_dir / 'preflight-receipt.json').write_text(json.dumps(report, indent=2) + '\n')
print('EVA_SFT_EPOCH_PREFLIGHT ' + json.dumps(report), flush=True)
