"""Project authenticated partial turns without inventing native receipts.

Only completed, model-observed calls enter the tool-result trace. Pending
calls and their unobserved host outcomes remain explicitly separate evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex
from training.automedbench_lite.adapter import canonical, read_document
from training.automedbench_lite.track_feedback import validate_inventory
from training.benchmark_feedback.automed_codex import event, visible_response


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def project_observations(request, capture, seal, audit, input_binding):
    """Pure projection of evidence already reopened by the signed verifier."""
    summary = capture['summary']
    logical = request['logical_input']
    _require(summary['partial_deadline_terminal_admissible'] is True and
             logical['judge_only_context'] is None and logical['role'] == capture['start']['role'],
             'partial_judge_visibility_or_terminal_invalid')
    hosts = {row['event_id']: row for row in
             (json.loads(line) for line in (audit / 'mcp-events.jsonl').read_bytes().splitlines())}
    joins = {row['upstream_item_id']: row for row in seal['host_joins']['completed_calls']}
    completed = {row['upstream_item_id']: row for row in summary['complete_calls']
                 if row['upstream_item_id'] in joins}
    pending = {row['upstream_item_id']: row for row in summary['pending_calls']}
    _require(set(completed) == set(joins) and not set(completed) & set(pending),
             'partial_judge_observation_partition_invalid')
    ids = {identifier: str(uuid4()) for identifier in completed}
    groups, group_ids, active, peak = {}, {}, set(), 0
    notifications = capture['actual_notifications']
    native_ids = {row['upstream_item_id'] for row in summary['complete_calls'] + summary['pending_calls']}
    native_active, native_peak = set(), 0
    completed_messages = {row['notification']['payload']['item']['id'] for row in notifications
        if row['notification']['method'] == 'item/completed'
        and row['notification']['payload'].get('item', {}).get('type') == 'agentMessage'}
    for observed in notifications:
        native = observed['notification']
        item = native['payload'].get('item', {})
        identifier = item.get('id')
        if identifier in native_ids:
            if native['method'] == 'item/started':
                native_active.add(identifier)
                native_peak = max(native_peak, len(native_active))
            elif native['method'] == 'item/completed':
                native_peak = max(native_peak, len(native_active | {identifier}))
                native_active.discard(identifier)
        if identifier not in completed:
            continue
        if native['method'] == 'item/started':
            if not active:
                group_ids[len(group_ids)] = str(uuid4())
            groups[identifier] = len(group_ids) - 1
            active.add(identifier)
            peak = max(peak, len(active))
        elif native['method'] == 'item/completed':
            _require(identifier in active, 'partial_judge_completed_call_has_no_start')
            active.remove(identifier)
    _require(not active and set(groups) == set(completed), 'partial_judge_complete_lifecycle_missing')
    results = {}
    for identifier, call in completed.items():
        host = hosts[joins[identifier]['host_event_id']]
        validate_inventory(host['workspace_before'], audit, input_binding)
        validate_inventory(host['workspace_after'], audit, input_binding)
        response = call['actual_native_output']['result']
        core = {'result_id': host['event_id'], 'call_id': ids[identifier], 'name': host['name'],
            'frontier': groups[identifier], 'parallel_group_id': group_ids[groups[identifier]],
            'status': 'failed' if host['is_error'] else 'completed',
            'output': {'actual_mcp_response_text_and_binary_commitments':
                       visible_response(response, call['native_item_blake3']), 'host_event': host},
            'error_code': host['result'].get('error_code') if host['is_error'] else None,
            'workspace_before_blake3': blake3_bytes(canonical(host['workspace_before'])),
            'workspace_after_blake3': blake3_bytes(canonical(host['workspace_after']))}
        results[identifier] = {**core, 'receipt_blake3': blake3_hex(core)}
    messages = [event('system', {'base_instructions': request['base_instructions'],
                                'developer_instructions': request['developer_instructions']}),
                event('user', logical)]
    seen_completed, seen_pending, ordered_results = set(), set(), []
    for observed in notifications:
        native = observed['notification']
        method, item = native['method'], native['payload'].get('item', {})
        identifier = item.get('id')
        evidence = {'native_event_sequence': observed['sequence'],
                    'native_capture_file_blake3': capture['capture_file_blake3']}
        if method == 'item/started' and identifier in completed:
            call = completed[identifier]
            messages.append(event('assistant', {**evidence, 'name': call['fully_qualified_name'],
                'arguments': call['arguments'], 'upstream_item_id': identifier}, (ids[identifier],)))
        elif method == 'item/completed' and identifier in completed:
            messages.append(event('tool', results[identifier], (ids[identifier],)))
            ordered_results.append(results[identifier]); seen_completed.add(identifier)
        elif method == 'item/started' and identifier in pending:
            call = pending[identifier]
            messages.append(event('assistant', {**evidence, 'name': call['fully_qualified_name'],
                'arguments': call['arguments'], 'upstream_item_id': identifier,
                'result_observation': 'unobserved_at_actual_interrupted_terminal'}))
            seen_pending.add(identifier)
        elif method == 'item/completed' and item.get('type') == 'agentMessage':
            _require(isinstance(item.get('text'), str), 'partial_judge_public_message_invalid')
            messages.append(event('assistant', {**evidence, 'text': item['text'], 'phase': item.get('phase')}))
        elif method == 'item/agentMessage/delta':
            payload = native['payload']
            message_id = payload.get('itemId', payload.get('item_id'))
            _require(isinstance(message_id, str) and message_id and isinstance(payload.get('delta'), str)
                     and native.get('content_redacted', False) is False, 'partial_judge_public_delta_invalid')
            if message_id not in completed_messages:
                messages.append(event('assistant', {**evidence, 'upstream_item_id': message_id,
                    'incomplete_public_text_delta': payload['delta'],
                    'completed_public_message_observed': False, 'final_answer_claimed': False}))
        elif method == 'item/completed' and item.get('type') == 'mcpToolCall':
            # Validated native resource controls are public history, not domain results.
            _require(any(call['upstream_item_id'] == identifier for call in summary['complete_calls']),
                     'partial_judge_unknown_native_control')
            messages.append(event('assistant', {**evidence, 'actual_native_control_not_domain_tool': item}))
    _require(seen_completed == set(completed) and seen_pending == set(pending),
             'partial_judge_visible_event_coverage_invalid')
    unobserved = []
    for joined in seal['host_joins']['pending_calls']:
        host = hosts[joined['host_event_id']]
        validate_inventory(host['workspace_before'], audit, input_binding)
        validate_inventory(host['workspace_after'], audit, input_binding)
        unobserved.append({'native_call': pending[joined['upstream_item_id']],
            'host_join': joined, 'host_event': host, 'model_observed_result': False,
            'included_in_completed_tool_trace': False, 'native_completion_synthesized': False})
    return {'messages': messages, 'results': ordered_results, 'frontiers': len(group_ids),
        'max_parallelism_observed': peak, 'assistant_output': '',
        'model': capture['start']['model'], 'provider': capture['start']['provider'],
        'source_receipt_blake3': capture['capture_file_blake3'], 'host_count': len(hosts),
        'metadata': {'evidence_type': 'authenticated_partial_native_turn',
            'projection_source_blake3': blake3_bytes(Path(__file__).read_bytes()),
            'ordinary_CodexTurnReceipt_claimed': False, 'actual_terminal': summary['actual_terminal'],
            'completed_requested_turns': False, 'completed_tool_trace_scope': 'model-observed_completed_calls_only',
            'tool_trace_parallelism_scope': 'completed_model_observed_call_lifecycles_only',
            'observed_native_tool_overlap_peak_including_pending': native_peak,
            'native_overlap_metric_is_physical_compute_concurrency': False,
            'projection_call_ids': ids, 'unobserved_pending_calls': unobserved,
            'last_public_message_is_final_answer': False,
            'native_capture_file_blake3': capture['capture_file_blake3'],
            'host_seal_blake3': blake3_hex(seal), 'private_reasoning_included': False}}


def reopen_and_project(run: Path, track: str, actor: dict, request: dict, input_binding: str):
    from .benchmark_partial_admission import require_partial_scoring_admission, verify_partial_host

    require_partial_scoring_admission(run, track, actor)
    audit = run / 'track-rollouts' / track
    manifest = read_document(run / 'track-run-manifest.json')
    spec = next(row for row in manifest['tracks'] if row['track'] == track)
    proof = read_document(audit / 'policy-partial-terminal.json')
    verified = verify_partial_host(audit, prelaunch=read_document(audit / 'native-capture-prelaunch.json'),
        offered_names=tuple('automed_eval/' + row['name'] for row in request['public_tool_catalog']),
        cleanup=read_document(audit / 'evamed-job-cleanup.json'),
        after_snapshot=read_document(audit / 'turns/01-e2e/after/manifest.json', maximum=16 * 1024**2),
        final_snapshot=read_document(audit / 'final/manifest.json', maximum=16 * 1024**2),
        workspace=run / spec['workspace_relative'], expected_file_blake3=proof['host_seal_file_blake3'])
    result = project_observations(request, verified['capture'], verified['host_seal'], audit, input_binding)
    _require(actor['requested_model'] == result['model'], 'partial_judge_model_identity_changed')
    result['metadata']['signed_host_seal_file_blake3'] = proof['host_seal_file_blake3']
    return result
