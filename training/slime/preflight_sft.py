"""Exercise the native Slime text loader, EVA hook, and DP schedule without Ray/GPU."""

from argparse import Namespace
import json
import os
from pathlib import Path
import sys

from run_full_parameter import MEGATRON, REPO, SLIME, WORKSPACE, build_command

for source in (REPO / 'src', SLIME, MEGATRON):
    sys.path.insert(0, str(source))
os.environ.setdefault('CUDA_HOME', str(WORKSPACE / '.venv/lib/python3.12/site-packages/nvidia/cu13'))
os.environ['SGLANG_ENABLE_JIT_DEEPGEMM'] = '0'

from eva_agent.training.slime_data import generate_sft_rollout
from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.arguments import parse_args
from slime.utils.dp_schedule import build_dp_schedule

request = Namespace(
    stage='sft', model=WORKSPACE / 'Qwen3.5-9B', load=None,
    data=REPO / 'runs/qwen35-9b-eva-data-20260910.v1/sft.jsonl',
    output=Path('/tmp/eva-preflight-no-output'), steps=2, batch_size=2, max_tokens=4096,
)
sys.argv = build_command(request)
args = parse_args()
source = RolloutDataSourceWithBuffer(args)
samples = generate_sft_rollout(args, 0, source)
ids = [sample.rollout_id if sample.rollout_id is not None else sample.index for sample in samples]
assert len(samples) == len(set(ids)) == 2
lengths = [len(sample.tokens) for sample in samples]
schedule = build_dp_schedule(
    args, {'dp_size': 1, 'cp_size': 1, 'vpp_size': 1, 'microbatch_group_size_per_vp_stage': 1},
    lengths, global_batch_size=2, rollout_indices=ids,
)
assert schedule[3] == [2]
assert sorted(schedule[0][0]) == [0, 1]
print('EVA_SFT_PREFLIGHT ' + json.dumps({
    'status': 'complete', 'samples': len(samples), 'distinct_rollout_ids': ids,
    'token_lengths': lengths, 'supervised_tokens': [sum(sample.loss_mask) for sample in samples],
    'scheduled_global_batch_sizes': schedule[3], 'microbatches': schedule[2],
    'gpu_or_ray_started': False,
}), flush=True)
