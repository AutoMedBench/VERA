"""Supplemental read-only item-type compaction observation for fixture receipts.

Never rewrite the original probe result or treat unrelated later work as exact
current-stage completion. Caller supplies independently verified host events and
public-input totals; original receipt/UUID identities are retained verbatim.
"""
from eva_agent.codex_runtime import verify_codex_turn_receipt
from eva_agent.pipeline.digests import canonical_value


def has_completed_compaction(receipts):
    """Codex compaction is an item type, not necessarily a method name."""
    return any(event.method == "item/completed"
        and canonical_value(event.payload).get("item", {}).get("type") == "contextCompaction"
        for receipt in receipts for event in receipt.events)


def observe_compaction(receipt, host_rows, expected_totals, *, expected_stage_input):
    verify_codex_turn_receipt(receipt)
    host = {row['response']['event_id']: row for row in host_rows}
    initial_errors = {row['response']['event_id'] for row in host_rows
                      if row['phase'] == 1 and row['response']['status'] == 'failed'}
    started, completed, calls = {}, [], []
    for event in receipt.events:
        item = canonical_value(event.payload).get('item', {})
        if item.get('type') == 'contextCompaction':
            if event.method == 'item/started':
                started[item['id']] = event
            elif event.method == 'item/completed':
                begin = started.get(item['id'])
                if begin is None or begin.sequence >= event.sequence:
                    raise ValueError('compaction_completion_without_prior_start')
                completed.append({'item_id': item['id'], 'started_event_id': begin.event_id,
                    'started_sequence': begin.sequence, 'completed_event_id': event.event_id,
                    'completed_sequence': event.sequence})
        elif event.method == 'item/completed' and item.get('type') == 'mcpToolCall':
            response = (item.get('result') or {}).get('structuredContent', {})
            event_id = response.get('event_id')
            if event_id not in host:
                raise ValueError('compaction_observation_host_event_missing')
            row = host[event_id]
            if row['response'] != response:
                raise ValueError('compaction_observation_host_response_differs')
            calls.append((event, row))
    observations = []
    for compact in completed:
        reads = [(event, row) for event, row in calls if event.sequence > compact['completed_sequence']
            and row['name'] == 'read_public_file' and row['arguments']['path'] == 'notes/failures.md'
            and row['response']['status'] == 'completed'
            and any(error_id in row['response']['result']['text'] for error_id in initial_errors)]
        reused = None
        for read_event, read in reads:
            candidates = [(event, row) for event, row in calls if event.sequence > read_event.sequence
                and row['name'] == 'analyze_table' and row['response']['status'] == 'completed'
                and row['response']['result']['total'] == expected_totals[row['arguments']['name']]]
            if candidates:
                event, analysis = candidates[0]
                reused = {'failure_note_read_sequence': read_event.sequence,
                    'failure_note_read_receipt_event_id': read_event.event_id,
                    'failure_note_host_event_id': read['response']['event_id'],
                    'failure_note_content_blake3': read['response']['result']['content_blake3'],
                    'analysis_sequence': event.sequence, 'analysis_receipt_event_id': event.event_id,
                    'analysis_host_event_id': analysis['response']['event_id'],
                    'analysis_input': analysis['arguments']['name'], 'actual_output_total_verified': True,
                    'matches_requested_stage_input': analysis['arguments']['name'] == expected_stage_input}
                break
        observations.append({**compact, 'subsequent_failure_memory_use': reused})
    return {'schema': 'eva.failure-memory-compaction-observation.v1',
        'source_receipt_blake3': receipt.receipt_blake3,
        'actual_compaction_observed': bool(completed), 'observations': observations,
        'post_compaction_note_read_then_correct_analysis_observed': any(
            row['subsequent_failure_memory_use'] is not None for row in observations),
        'original_result_mutated': False, 'new_provider_calls': 0, 'new_judgments': 0,
        'scope': 'actual item-type events and ordered public tool evidence; not a causal skill-benefit claim'}
