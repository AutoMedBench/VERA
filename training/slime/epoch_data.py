"""Strict single-pass wrapper around EVA's unchanged masked SFT hook."""

import json
import os
from pathlib import Path


def generate_sft_rollout(args, rollout_id, data_buffer, evaluation=False):
    from eva_agent.training.slime_data import generate_sft_rollout as generate

    expected_rows = int(os.environ['EVA_SFT_EPOCH_ROWS'])
    expected_steps = int(os.environ['EVA_SFT_EPOCH_STEPS'])
    expected_offset = rollout_id * args.rollout_batch_size
    if (evaluation or not 0 <= rollout_id < expected_steps
            or len(data_buffer) != expected_rows
            or data_buffer.epoch_id != 0
            or data_buffer.sample_offset != expected_offset
            or expected_offset + args.rollout_batch_size > expected_rows
            or getattr(data_buffer, 'buffer', [])):
        raise ValueError('Single-pass SFT refuses dataset drift, replay, wrap, or buffered replacement')
    samples = generate(args, rollout_id, data_buffer, evaluation=False)
    row_ids = [sample.metadata['row_id'] for sample in samples]
    seen = getattr(data_buffer, '_eva_consumed_row_ids', set())
    if (len(samples) != args.rollout_batch_size or len(set(row_ids)) != len(row_ids)
            or seen.intersection(row_ids) or data_buffer.epoch_id != 0
            or data_buffer.sample_offset != expected_offset + len(samples)):
        raise ValueError('Single-pass SFT received repeated or incomplete samples')
    seen.update(row_ids)
    data_buffer._eva_consumed_row_ids = seen
    receipt = {
        'training_iteration': rollout_id, 'epoch_id': data_buffer.epoch_id,
        'sample_offset_before': expected_offset, 'sample_offset_after': data_buffer.sample_offset,
        'row_ids': row_ids, 'tokens': sum(len(sample.tokens) for sample in samples),
        'supervised_tokens': sum(sum(sample.loss_mask) for sample in samples),
    }
    # This records consumed examples, not a claim that their optimizer update completed.
    path = Path(os.environ['EVA_SFT_DATA_RECEIPT'])
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, 'a') as stream:
        stream.write(json.dumps(receipt, sort_keys=True) + '\n')
    return samples
