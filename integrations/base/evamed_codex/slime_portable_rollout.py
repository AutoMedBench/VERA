"""Real Codex → Responses → SGLang sampled tokens → native Slime segments."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
from uuid import uuid4

from .portable_rollout import ROOT, settings, run_case


def capture_trajectory(args, original, *, tokenizer, sampling_params, output, rollout_id=0,
                       purpose='slime-on-policy-training'):
    from eva_agent.training.codex_sglang_transport import CodexSGLangTransport
    from eva_agent.training.codex_slime_rollout import samples_from_segments
    from eva_agent.pipeline.digests import canonical_value
    from .benchmark_provider import setup_provider
    from evamed_portable.integrity import strict_json, write_json
    from evamed_portable.bundle import Bundle
    from eva_agent.training.thinking_loss_mask import require_unmodified_policy_sampling

    if args.seq_length != 262144:
        raise ValueError('portable_actor_requires_262144_context')
    require_unmodified_policy_sampling(sampling_params)
    metadata = original.metadata or {}
    if not {'sandbox_id', 'domain', 'stage', 'bundle_blake3', 'source_record_blake3'} <= set(metadata):
        raise ValueError('portable_source_identity_missing')
    config = settings()
    source_bundle = Bundle(config['bundle'])
    _, source, _, _ = source_bundle.case(metadata['sandbox_id'])
    if (source_bundle.manifest['document_blake3'] != metadata['bundle_blake3'] or
            any(source[name] != metadata[name] for name in ['domain', 'stage']) or
            source['record_blake3'] != metadata['source_record_blake3']):
        raise ValueError('portable_sampling_index_identity_changed')
    # The Slime router can live on another allocated node. The generic think
    # profile does not claim the loopback-only instant-serving profile.
    config.update(route='slime', model='Qwen3.8-27B',
        endpoint=f'http://{args.sglang_router_ip}:{args.sglang_router_port}/v1',
        max_output_tokens=32768, context_length=262144,
        training_public_warning_policy='eva.medresearch-authenticated-training-public-warning.v2')
    processor = strict_json((ROOT / 'evamed-codex/receipts/qwen35-thinking-processor.json').read_bytes())
    config['thinking_logit_processor'] = processor['custom_logit_processor']
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    captured = []

    @contextmanager
    def exact_provider(bound_config, provider_root, unused_token):
        # Source-specific runtime budgets are resolved by run_case before this call.
        from .portable_qwen_types import fresh_argument_projector
        from eva_agent.training.thinking_loss_mask import QwenThinkingLossMask
        masker = QwenThinkingLossMask(source_sha256=processor['source_sha256'],
            serialized_processor=bound_config['thinking_logit_processor'],
            wire_budget=bound_config['thinking_budget_tokens'] - 1)
        transport = CodexSGLangTransport(tokenizer=tokenizer,
            generate_url=bound_config['endpoint'].removesuffix('/v1') + '/generate', sampling_params=sampling_params,
            max_context_tokens=bound_config['context_length'], max_output_tokens=bound_config['max_output_tokens'],
            total_output_budget=args.rollout_max_response_len, max_requests=bound_config['max_turns'],
            evidence_root=output / 'private-provider-tokens', timeout_seconds=min(600, bound_config['max_seconds']),
            argument_projector=fresh_argument_projector(bound_config),
            custom_logit_processor=bound_config['thinking_logit_processor'],
            custom_params={'thinking_budget': bound_config['thinking_budget_tokens'] - 1},
            policy_loss_masker=masker)
        captured.append(transport)
        with setup_provider(bound_config, provider_root, 'local-nonsecret', upstream_transport=transport) as provider:
            yield provider

    case = {'sandbox_id': metadata['sandbox_id'], 'domain': metadata['domain'], 'stage': metadata['stage']}
    result = asyncio.run(run_case(config, case, output, purpose, provider_factory=exact_provider))
    if (result['bundle_blake3'] != metadata['bundle_blake3'] or len(captured) != 1 or
            not result['diagnostic_infrastructure_passed'] or result['errors']):
        raise ValueError('portable_Codex_actor_not_admitted; earlier_segments_are_audit_only')
    transport = captured[0]
    trajectory = output / 'trajectories' / result['run_id']
    owned_terminal = None
    if transport.safe_metadata['failed']:
        if (transport.budget_termination or {}).get('kind') != 'total_output_budget':
            raise ValueError('sampled_provider_transport_failed')
        from .slime_token_admission import create_owned_output_receipt
        owned_terminal, _ = create_owned_output_receipt(trajectory, output, args.rollout_max_response_len)
    segments = samples_from_segments(args, original, transport.segments, max_requests=24,
        training_iteration=rollout_id, sample_id=result['run_id'], trajectory_path=trajectory / 'trajectory.json')
    for index, sample in enumerate(segments):
        write_json(output / f'private-slime-sample-{index:04d}.json', canonical_value({
            'tokens': sample.tokens, 'response_length': sample.response_length, 'loss_mask': sample.loss_mask,
            'rollout_log_probs': sample.rollout_log_probs, 'metadata': sample.metadata,
            'reward': None, 'training_admitted': False}))
        (output / f'private-slime-sample-{index:04d}.json').chmod(0o600)
    write_json(output / 'capture.json', {'schema': 'eva.portable-codex-slime-token-capture.v1',
        'actual_codex_actor': True, 'actual_source_bound_host_tools': True,
        'source_bundle_blake3': result['bundle_blake3'], 'actor_trajectory': str(trajectory),
        'model': config['model'], 'segments': len(segments), 'transport': transport.safe_metadata,
        'original_output_ids_and_logprobs_preserved': True,
        'sampled_outputs_retokenized': False, 'reward_pending': True,
        'actual_codex_terminal_status': result['codex_turn_status'],
        'controlled_output_budget_terminal': owned_terminal is not None,
        'sampled_tokens': sum(s.response_length for s in segments)})
    return segments, trajectory


def generate(args, rollout_id, data_buffer, evaluation=False):
    """Slime's rollout hook; a failed judge pair withholds the entire batch."""
    from slime.rollout.sglang_rollout import GenerateState
    from .portable_judge import judge_trajectory

    if evaluation:
        raise ValueError('held_out_evaluation_uses_separate_runner')
    if args.custom_reward_post_process_path != 'eva_agent.training.codex_segment_rewards.normalize_segment_rewards':
        raise ValueError('normalization_once_per_original_trajectory_required')
    state = GenerateState(args)
    groups = data_buffer.get_samples(args.rollout_batch_size)
    if len(groups) != args.rollout_batch_size or any(len(group) != args.n_samples_per_prompt for group in groups):
        raise ValueError('original_GRPO_group_cardinality_differs')
    output = Path(args.save).parent / 'rollouts' / str(uuid4())
    output.mkdir(parents=True, exist_ok=False)
    concurrency = int(os.environ.get('EVAMED_ROLLOUT_CONCURRENCY', '16'))
    if not 1 <= concurrency <= 64:
        raise ValueError('rollout_concurrency_outside_explicit_1_to_64_range')

    def one(original):
        from evamed_portable.integrity import digest, write_json
        samples, actor = capture_trajectory(args, original, tokenizer=state.tokenizer,
            sampling_params=state.sampling_params, output=output / str(uuid4()), rollout_id=rollout_id)
        pair = judge_trajectory(actor, actor.parent / (actor.name + '-judge'))
        if not pair['reward_admitted']:
            raise ValueError('independent_Opus_reward_verification_failed; batch_not_admitted')
        for sample in samples:
            sample.reward = pair['reward_bps'] / 10000
            sample.metadata.update(judge_pair_path=str(actor.parent / (actor.name + '-judge/pair.json')),
                                   reward_shared_across_original_trajectory=True)
        write_json(actor.parent / (actor.name + '-training-admission.json'), {
            'schema': 'eva.portable-judged-training-segments.v1', 'source_trajectory': str(actor),
            'pair_document_blake3': pair['document_blake3'], 'reward': pair['reward_bps'] / 10000,
            'segments': [{'sample_index': sample.index, 'tokens_and_mask_blake3': digest({
                'tokens': sample.tokens, 'response_length': sample.response_length,
                'loss_mask': sample.loss_mask, 'rollout_log_probs': sample.rollout_log_probs}),
                'reward': sample.reward} for sample in samples],
            'source_outputs_retokenized': False, 'all_segments_share_original_trajectory_reward': True})
        return samples

    originals = [sample for group in groups for sample in group]
    with ThreadPoolExecutor(max_workers=min(concurrency, len(originals))) as pool:
        trajectories = list(pool.map(one, originals))
    width = args.n_samples_per_prompt
    return [trajectories[i:i + width] for i in range(0, len(trajectories), width)]
