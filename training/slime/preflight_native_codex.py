"""One explicitly authorized native Codex exact-token trajectory; never train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from eva_agent.training.codex_slime_rollout import native_settings, run_native_trajectory, samples_from_segments
from eva_agent.training.slime_rollout import _write_private_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--server-identity', required=True, type=Path)
    parser.add_argument('--server-canary', required=True, type=Path)
    parser.add_argument('--port', type=int, default=30910)
    cli = parser.parse_args()
    cli.output = cli.output.resolve()
    from training.automedbench_lite.actor import serving_binding
    from eva_agent.training.teacher_worker import CampaignV2TeacherContextPool, load_bulk_record
    from slime.utils.types import Sample
    from transformers import AutoTokenizer
    from blake3 import blake3

    binding = serving_binding(cli.server_canary, cli.server_identity)
    args = SimpleNamespace(seq_length=24576, rollout_max_response_len=8192,
                           sglang_router_ip='127.0.0.1', sglang_router_port=cli.port)
    settings = native_settings(args)
    sampling_params = {'temperature': 0.8, 'top_p': 1.0, 'top_k': -1, 'max_new_tokens': 4096,
                       'skip_special_tokens': False, 'no_stop_trim': True,
                       'spaces_between_special_tokens': False}
    cli.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    row = json.loads(next(line for line in cli.data.read_text().splitlines() if line.strip()))
    original = Sample(index=0, group_index=0, prompt=row['prompt'], metadata=row['metadata'])
    # Execute the installed append-response implementation before any provider
    # call; this is an API/alignment probe, not a training or model forward.
    probe = {'segment_index': 0, 'request_id': 'installed-sample-api-probe',
        'prompt_token_ids': [1, 2], 'output_token_ids': [3], 'output_token_logprobs': [-0.5],
        'loss_mask': [1], 'meta_info': {}, 'requested_model': 'probe', 'returned_model': None}
    checked = samples_from_segments(args, original, [probe], max_requests=32,
        training_iteration=0, sample_id='api-probe', trajectory_path=cli.output / 'not-a-trajectory')
    assert checked[0].tokens == [1, 2, 3] and checked[0].response_length == 1
    _write_private_json(cli.output / 'preflight.json', {
        'schema': 'eva.native-codex-single-trajectory-preflight.v1',
        'status': 'prepared', 'provider_trajectories_authorized': 1, 'training_updates': 0,
        'installed_sample_append_probe_passed': True, 'server_binding': binding,
        'settings': {**settings, 'codex_bin': str(settings['codex_bin'])},
        'data': str(cli.data.resolve()), 'selected_row': 0,
        'metadata': original.metadata, 'tokenizer_model_path': str(cli.model.resolve()),
        'sampling_params': sampling_params,
        'source_blake3': {str(path): blake3(path.read_bytes()).hexdigest() for path in (
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[2] / 'src/eva_agent/training/codex_slime_rollout.py',
            Path(__file__).resolve().parents[2] / 'src/eva_agent/training/codex_sglang_transport.py',
            Path(__file__).resolve().parents[2] / 'src/eva_agent/training/qwen_tool_types.py')}})
    print(json.dumps({'status': 'loading_actual_context', 'output': str(cli.output)}), flush=True)
    record = load_bulk_record(Path(original.metadata['bulk_root']), original.metadata['sandbox_id'])
    context, _ = CampaignV2TeacherContextPool().load(record)
    tokenizer = AutoTokenizer.from_pretrained(cli.model, local_files_only=True, trust_remote_code=True)
    sample_id = str(uuid4())
    print(json.dumps({'status': 'native_actor_start', 'sample_id': sample_id}), flush=True)
    try:
        samples = run_native_trajectory(args, original, tokenizer=tokenizer,
            sampling_params=sampling_params,
            record=record, context=context, sample_root=cli.output / sample_id,
            sample_id=sample_id, training_iteration=0, settings=settings)
        summary = {'status': 'complete', 'sample_id': sample_id, 'training_updates': 0,
            'native_codex_actor': True, 'segments': len(samples),
            'sampled_tokens': sum(s.response_length for s in samples), 'reward': samples[0].reward}
        _write_private_json(cli.output / 'receipt.json', summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        summary = {'status': 'failed', 'sample_id': sample_id, 'training_updates': 0,
            'error_type': type(error).__name__, 'automatic_retry': False, 'reward_emitted': False}
        _write_private_json(cli.output / 'receipt.json', summary)
        print(json.dumps(summary), flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
