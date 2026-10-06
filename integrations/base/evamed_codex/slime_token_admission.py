"""Admit an owned output cap using retained native SGLang evidence only.

This is a post-execution verification receipt. It never changes an original
Codex status or supplies a reward. Private sampled tokens remain local.
"""
from pathlib import Path
import math
from uuid import UUID

from .portable_rollout import ROOT, BUNDLE
from evamed_portable.bundle import Bundle
from evamed_portable.integrity import (Signer, contained_file, digest, file_digest,
    strict_json, verify_receipt, write_json)
from eva_agent.pipeline.digests import blake3_hex

SCHEMA = 'eva.portable-slime-owned-output-terminal.v1'
NAME = 'slime-output-budget-terminal.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _document(path):
    value = strict_json(path.read_bytes())
    require(value['document_blake3'] == digest({k: v for k, v in value.items() if k != 'document_blake3'}),
            'slime_capture_document_changed')
    return value


def verify_owned_output_terminal(actor, capture_root, expected_limit=32768):
    return _verify_native_capture(actor, capture_root, expected_limit, owned_terminal=True)


def verify_completed_capture(actor, capture_root, expected_limit=32768):
    """The same exact-token checks for an ordinary completed Codex turn."""
    return _verify_native_capture(actor, capture_root, expected_limit, owned_terminal=False)


def _verify_native_capture(actor, capture_root, expected_limit, *, owned_terminal):
    actor, capture_root = Path(actor).resolve(strict=True), Path(capture_root).resolve(strict=True)
    require(actor.is_relative_to(ROOT) and capture_root.is_relative_to(ROOT) and
            actor.parent.parent == capture_root and type(expected_limit) is int and expected_limit == 32768,
            'slime_capture_scope_or_fixed_budget_changed')
    trajectory = _document(contained_file(actor, 'trajectory.json'))
    bundle = Bundle(BUNDLE)
    require(trajectory['route'] == 'slime' and trajectory['model'] == 'Qwen3.8-27B' and
            trajectory['bundle_blake3'] == bundle.manifest['document_blake3'] and
            not trajectory['errors'] and
            trajectory['codex_turn_status'] == ('failed' if owned_terminal else 'completed') and
            (not owned_terminal or trajectory['control_stop_reason'] == 'Codex_terminal_or_incomplete_turn') and
            trajectory['diagnostic_infrastructure_passed'] is True,
            'not_an_owned_slime_terminal_candidate')
    public = bundle.manifest['fresh_execution_authority']['public_key_base64']
    episode_raw = strict_json(contained_file(actor, 'episode-receipt.json').read_bytes())
    episode = verify_receipt(episode_raw, public)
    require(digest(episode_raw) == trajectory['episode_receipt_blake3'] and
            episode['turns'] == trajectory['codex_turns'] and episode['run_id'] == trajectory['run_id'],
            'slime_original_episode_binding_changed')
    private = capture_root / 'private-provider-tokens'
    files = {}

    def read(path):
        path = contained_file(capture_root, str(path.relative_to(capture_root)))
        files[str(path.relative_to(capture_root))] = file_digest(path)
        return strict_json(path.read_bytes())

    diagnostics = list(private.glob('safe-failure-diagnostic-*.json'))
    failures = list(private.glob('failure-*.json'))
    require(len(diagnostics) == len(failures) == int(owned_terminal) and not list(private.glob('http-failure-*.json')),
            'slime_has_non_owned_or_multiple_failures')
    if owned_terminal:
        diagnostic, failure = read(diagnostics[0]), read(failures[0])
        require(diagnostics[0].name.removeprefix('safe-failure-diagnostic-') == failures[0].name.removeprefix('failure-'),
                'slime_failure_identity_mismatch')
        require(diagnostic['category'] == 'owned_output_budget' and
                diagnostic['total_output_budget'] == expected_limit == diagnostic['generated_tokens'] and
                diagnostic['max_context_tokens'] == 262144 and diagnostic['max_output_tokens'] == 32768 and
                type(diagnostic['prompt_tokens']) is int and 0 < diagnostic['prompt_tokens'] < 262144 and
                diagnostic['automatic_retry'] is False and failure['automatic_retry'] is False and
                failure['error_type'] == 'ResponsesAdapterError', 'slime_terminal_not_exact_output_cap')
    segments = sorted((read(p) for p in private.glob('segment-*.json')), key=lambda row: row['segment_index'])
    require(0 < len(segments) <= 24 - int(owned_terminal), 'slime_terminal_segment_count_changed')
    if owned_terminal:
        require(diagnostic['requests'] == diagnostic['segments'] == len(segments) and
                failure['requests'] == failure['segments'] == len(segments), 'slime_terminal_segment_count_changed')
    requests = list(private.glob('request-*.json'))
    require(len(requests) == len(segments), 'slime_unfinished_upstream_request_present')
    provider_paths = list((actor / 'provider/requests').glob('*.json'))
    providers = sorted((read(p) for p in provider_paths), key=lambda row: row['request_number'])
    require(len(providers) == len(segments) + int(owned_terminal) == trajectory['actual_model_requests'] and
            [r['request_number'] for r in providers] == list(range(1, len(providers) + 1)),
            'slime_meter_request_sequence_changed')
    from training.automedbench_lite.adapter import read_document
    for path in provider_paths:
        read_document(path)
    for provider in providers:
        require(provider['model'] == trajectory['model'] and provider['automatic_retry'] is False and
                provider['mode'] == 'think' and provider['manual_thinking_budget_requested'] == 24576 and
                provider['manual_thinking_processor_supplied'] is True,
                'slime_provider_meter_changed')
    if owned_terminal:
        require(providers[-1].get('error_type') == 'ResponsesAdapterError' and
                'http_status' not in providers[-1] and 'usage' not in providers[-1],
                'slime_denied_request_reached_upstream')
    generated, identities, counts = 0, set(), []
    processor = strict_json((ROOT / 'evamed-codex/receipts/qwen35-thinking-processor.json').read_bytes())
    from eva_agent.training.thinking_loss_mask import QwenThinkingLossMask, require_unmodified_policy_sampling
    masker = QwenThinkingLossMask(source_sha256=processor['source_sha256'],
        serialized_processor=processor['custom_logit_processor'], wire_budget=24575)
    mask_binding_path = private / 'policy-loss-mask-binding.json'
    if mask_binding_path.exists():
        require(read(mask_binding_path) == masker.binding, 'slime_policy_loss_mask_binding_changed')
    for index, segment in enumerate(segments):
        identifier = segment['request_id']; UUID(identifier)
        require(identifier not in identities and segment['segment_index'] == index, 'slime_segment_identity_changed')
        identities.add(identifier)
        request = read(private / ('request-' + identifier + '.json'))
        policy_params = dict(request['sampling_params'])
        nested_processor_params = policy_params.pop('custom_params', None)
        require_unmodified_policy_sampling(policy_params)
        projection = read(private / ('argument-projection-' + identifier + '.json'))
        prompt, tokens, probs = segment['prompt_token_ids'], segment['output_token_ids'], segment['output_token_logprobs']
        meta = segment['meta_info']
        mask, provenance = masker(prompt, tokens)
        if 'loss_mask_provenance' in segment:
            require(mask_binding_path.exists() and segment['loss_mask_provenance'] == provenance and
                    nested_processor_params == {'thinking_budget': 24575} and 'custom_params' not in request,
                    'slime_policy_loss_mask_provenance_changed')
        else:
            # Historical captures are immutable. They remain admissible only
            # when exact processor replay proves no forced tokens occurred.
            require(not mask_binding_path.exists() and all(mask) and nested_processor_params is None and
                    request.get('custom_params') == {'thinking_budget': 24575},
                    'legacy_slime_capture_contains_unmasked_forced_tokens')
        require(prompt and tokens and all(type(t) is int and t >= 0 for t in prompt + tokens) and
                len(prompt) + len(tokens) <= 262144 and len(tokens) == len(probs) and
                all(type(p) in (int, float) and math.isfinite(p) and p <= 0 for p in probs) and
                segment['loss_mask'] == mask and all(type(bit) is int for bit in segment['loss_mask']) and
                all(prob == 0.0 for prob, bit in zip(probs, mask) if bit == 0) and
                segment['prompt_is_conditioning_only'] is True,
                'slime_token_logprob_alignment_changed')
        require(meta['prompt_tokens'] == len(prompt) and meta['completion_tokens'] == len(tokens) and
                [r[1] for r in meta['output_token_logprobs']] == tokens and
                [r[0] for r in meta['output_token_logprobs']] == probs and
                meta['finish_reason'] == segment['finish_reason'] and
                segment['finish_reason']['type'] in {'length', 'stop'}, 'slime_native_metadata_changed')
        require(request['request_id'] == identifier == projection['request_id'] and
                request['input_ids'] == prompt and request['chat_request_blake3'] == segment['chat_request_blake3'] and
                request['tools_blake3'] == segment['tools_blake3'] and
                request['requested_model'] == segment['requested_model'] == trajectory['model'] and
                request['sampling_params'] == segment['sampling_params'] and
                request['return_logprob'] is True and request['logprob_start_len'] == -1 and
                request['return_text_in_logprobs'] is False and request['stream'] is False and
                request['custom_logit_processor'] == processor['custom_logit_processor'] and
                request['sampling_params']['max_new_tokens'] == min(32768, expected_limit - generated) and
                projection['raw_output_blake3'] == blake3_hex(segment['output_text']),
                'slime_request_or_projection_binding_changed')
        require(providers[index].get('http_status') == 200 and 'error_type' not in providers[index] and
                providers[index]['usage'] == {'prompt_tokens': len(prompt), 'completion_tokens': len(tokens),
                                             'total_tokens': len(prompt) + len(tokens)},
                'slime_provider_token_counts_changed')
        generated += len(tokens)
        counts.append({'segment_index': index, 'request_id': identifier, 'prompt_tokens': len(prompt),
                       'sampled_tokens': len(tokens), 'finish_type': segment['finish_reason']['type']})
    require(generated <= expected_limit, 'slime_generated_output_budget_exceeded')
    if owned_terminal:
        require(generated == expected_limit and segments[-1]['finish_reason']['type'] == 'length',
                'slime_last_generation_did_not_exhaust_owned_cap')
    read(private / 'host-argument-type-binding.json')
    payload = {'schema': SCHEMA, 'run_id': trajectory['run_id'], 'case_id': trajectory['case_id'],
        'bundle_blake3': trajectory['bundle_blake3'], 'source_record_blake3': trajectory['source_record_blake3'],
        'original_trajectory_blake3': trajectory['document_blake3'],
        'original_episode_blake3': trajectory['episode_receipt_blake3'],
        'original_last_turn_blake3': trajectory['codex_turn_receipt_blake3'],
        'actual_terminal_status': 'failed', 'kind': 'total_output_budget', 'limit': expected_limit,
        'generated_tokens': generated, 'actual_sglang_requests': len(segments),
        'denied_provider_request_number': len(providers), 'denied_request_sent_to_sglang': False,
        'capture_root': str(capture_root), 'source_files': files, 'segments': counts,
        'post_execution_verification': True, 'original_evidence_modified': False,
        'raw_policy_tokens_or_reasoning_in_this_receipt': False,
        'output_tokens_retokenized': False, 'reward_requires_independent_Opus_pair': True}
    if not owned_terminal:
        payload.update(schema='eva.portable-slime-completed-token-capture.v1',
            actual_terminal_status='completed', kind='native_completed',
            denied_provider_request_number=None)
    return payload, segments


def create_owned_output_receipt(actor, capture_root, expected_limit=32768):
    actor = Path(actor).resolve(strict=True)
    payload, segments = verify_owned_output_terminal(actor, capture_root, expected_limit)
    host = strict_json(contained_file(actor, 'host-tool-config.json').read_bytes())
    signed = Signer(host['key']).sign(payload)
    write_json(actor / NAME, signed, exclusive=True)
    return payload, segments


def verify_owned_output_receipt(actor):
    actor = Path(actor).resolve(strict=True)
    signed = strict_json(contained_file(actor, NAME).read_bytes())
    public = Bundle(BUNDLE).manifest['fresh_execution_authority']['public_key_base64']
    payload = verify_receipt(signed, public)
    require(payload['schema'] == SCHEMA, 'slime_owned_output_receipt_schema_changed')
    expected, _ = verify_owned_output_terminal(actor, payload['capture_root'], payload['limit'])
    require(payload == expected, 'slime_owned_output_evidence_changed')
    return payload


def verify_thinking_loss_readiness():
    """Reopen real short-cap probes and the current transport/Slime code."""
    from types import SimpleNamespace
    from eva_agent.training.thinking_loss_mask import validate_thinking_mask
    from eva_agent.training.codex_slime_rollout import samples_from_segments
    from slime.utils.types import Sample

    receipt = strict_json((ROOT / 'evamed-codex/receipts/thinking-loss-mask-live.json').read_bytes())
    require(receipt['schema'] == 'eva.live-sglang-thinking-loss-mask-verification.v1' and
            receipt['passed'] is True and receipt['old_evidence_modified'] is False and
            receipt['optimizer_step_performed'] is False, 'thinking_mask_live_receipt_invalid')
    for name in ['EVA-Harness/src/eva_agent/training/thinking_loss_mask.py',
                 'EVA-Harness/src/eva_agent/training/codex_sglang_transport.py',
                 'EVA-Harness/src/eva_agent/training/codex_slime_rollout.py']:
        require(file_digest(contained_file(ROOT, name)) == receipt['sources'][name],
                'thinking_loss_runtime_changed_after_live_verification')
    rows = receipt['live_local_SGLang_native_Slime_checks']
    require([row['wire_budget'] for row in rows] == [0, 7], 'thinking_loss_live_probe_coverage_changed')
    for row in rows:
        files = {name: contained_file(ROOT, name) for name in row['private_source_files']}
        require(all(file_digest(path) == row['private_source_files'][name] for name, path in files.items()),
                'thinking_loss_live_provider_evidence_changed')
        requests = [path for path in files.values() if path.name.startswith('request-')]
        segments = [path for path in files.values() if path.name.startswith('segment-')]
        require(len(requests) == len(segments) == 1, 'thinking_loss_live_request_count_changed')
        request, segment = (strict_json(paths[0].read_bytes()) for paths in (requests, segments))
        mask = validate_thinking_mask(segment)
        require(request['input_ids'] == segment['prompt_token_ids'] and
                request['request_id'] == segment['request_id'] and
                request['sampling_params'] == segment['sampling_params'] and
                request['sampling_params']['custom_params'] == {'thinking_budget': row['wire_budget']} and
                'custom_params' not in request and
                request['custom_logit_processor'] == segment['loss_mask_provenance']['serialized_processor'] and
                segment['loss_mask_provenance']['wire_budget'] == row['wire_budget'] and
                row['loss_mask'] == mask and len(mask) - sum(mask) == row['forced_tokens'] == (1 if row['wire_budget'] == 0 else 2) and
                all(prob == 0.0 for bit, prob in zip(mask, segment['output_token_logprobs']) if not bit),
                'thinking_loss_live_forcing_changed')
        sample = samples_from_segments(SimpleNamespace(seq_length=262144), Sample(index=0, group_index=0),
            [segment], max_requests=24, training_iteration=0, sample_id='readiness-reopen',
            trajectory_path=segments[0].parent)[0]
        require(sample.tokens == segment['prompt_token_ids'] + segment['output_token_ids'] and
                sample.rollout_log_probs == segment['output_token_logprobs'] and sample.loss_mask == mask,
                'thinking_loss_native_slime_sample_changed')
    return {'live_probe_count': len(rows), 'forced_thinking_masks_live_verified': True}


def verify_live_training_receipt(path, *, expected_bundle_blake3):
    """Reopen source-bound on-policy tokens and both real Opus assessments."""
    from training.automedbench_lite.adapter import read_document
    from .benchmark_judge import verify_assessment
    from eva_agent.rubrics.models import CompiledRubric

    thinking = verify_thinking_loss_readiness()

    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(ROOT), 'live_training_receipt_outside_workspace')
    receipt = read_document(path)
    bundle = Bundle(BUNDLE)
    signed = verify_receipt(receipt['verification_signature'],
        bundle.manifest['fresh_execution_authority']['public_key_base64'])
    require(signed == {k: v for k, v in receipt.items() if k not in {'document_blake3', 'verification_signature'}},
            'live_training_verification_signature_changed')
    require(receipt['schema'] == 'eva.live-codex-slime-portable-verification.v1' and receipt['passed'] is True and
            receipt['capture_passed'] is True and receipt['model'] == 'Qwen3.8-27B' and
            receipt['source_bundle_blake3'] == expected_bundle_blake3 == bundle.manifest['document_blake3'] and
            receipt['actual_codex_actor'] is True and receipt['real_sandbox_execution'] is True and
            receipt['output_token_ids_and_logprobs_preserved'] is True and
            receipt['optimizer_updated_with_this_trajectory'] is False,
            'live_training_readiness_identity_changed')
    actor = Path(receipt['trajectory_path']).resolve(strict=True)
    capture = actor.parent.parent
    trajectory = _document(contained_file(actor, 'trajectory.json'))
    owned = trajectory['codex_turn_status'] != 'completed'
    if owned:
        proof = verify_owned_output_receipt(actor)
        _, segments = verify_owned_output_terminal(actor, capture)
    else:
        proof, segments = verify_completed_capture(actor, capture)
    require(proof['generated_tokens'] == receipt['generated_tokens'] and len(segments) == receipt['segments'],
            'live_training_token_counts_changed')
    for index, segment in enumerate(segments):
        sample = strict_json(contained_file(capture, f'private-slime-sample-{index:04d}.json').read_bytes())
        require(sample['tokens'] == segment['prompt_token_ids'] + segment['output_token_ids'] and
                sample['response_length'] == len(segment['output_token_ids']) and
                sample['loss_mask'] == segment['loss_mask'] and
                sample['rollout_log_probs'] == segment['output_token_logprobs'] and
                sample['metadata']['prompt_tokens_are_conditioning_only'] is True and
                sample['metadata']['sampled_outputs_retokenized'] is False,
                'live_native_slime_sample_changed')
    pair_root = Path(receipt['pair_path']).resolve(strict=True).parent
    require(pair_root.is_relative_to(capture), 'live_judge_pair_outside_capture')
    pair = read_document(pair_root / 'pair.json')
    require(pair == receipt['independent_opus_pair'] and pair['reward_admitted'] is True and
            pair['both_assessments_verified'] is True and pair['exact_item_agreement'] is True,
            'live_judge_pair_not_admitted')
    source = strict_json(contained_file(pair_root, 'source.json').read_bytes())
    _, original, _, _ = bundle.case(proof['case_id'])
    require(source['rubric_table'] == original['reward_contract']['rubric_table'] and
            source['provider_metadata']['actual_actor_trajectory_blake3'] == proof['original_trajectory_blake3'] and
            source['provider_metadata']['owned_output_budget_proof_file_blake3'] == (file_digest(actor / NAME) if owned else None) and
            source['provider_metadata']['actual_terminal_status'] == trajectory['codex_turn_status'] and
            source['provider_metadata']['private_token_material_included'] is False,
            'live_assessment_source_changed')
    rubric = CompiledRubric.from_document(source['rubric_table'])
    results = {role: verify_assessment(pair_root / 'source.json', rubric, pair_root / role)
               for role in ['judge', 'reward-verifier']}
    require(results == pair['results'] and results['judge']['assessment_id'] != results['reward-verifier']['assessment_id'] and
            results['judge']['item_scores_bps'] == results['reward-verifier']['item_scores_bps'] and
            results['judge']['reward_bps'] == pair['reward_bps'], 'live_Opus_assessments_changed')
    return {'model': receipt['model'], 'source_bundle_blake3': expected_bundle_blake3,
        'trajectory_path': str(actor), 'segments': len(segments), 'generated_tokens': proof['generated_tokens'],
        'verified_pair_blake3': pair['document_blake3'], 'actual_terminal_status': trajectory['codex_turn_status'],
        'controlled_owned_output_cap': owned, 'reward_bps': pair['reward_bps'],
        **thinking,
        'long_context_optimizer_step_proven': False}
