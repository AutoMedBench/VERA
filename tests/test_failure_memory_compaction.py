from types import SimpleNamespace as NS

import pytest
from training.context_memory import compaction_observation as module


def fixture(monkeypatch, *, read_sequence=3, analysis_sequence=4):
    monkeypatch.setattr(module, 'verify_codex_turn_receipt', lambda _: None)
    host = [{'phase': 1, 'response': {'event_id': 'actual-error', 'status': 'failed'}}]
    events = [NS(sequence=i, event_id=f'event-{i}', method=f'item/{kind}',
                 payload={'item': {'type': 'contextCompaction', 'id': 'same-item'}})
              for i, kind in [(1, 'started'), (2, 'completed')]]
    for seq, name, args, result in [(read_sequence, 'read_public_file', {'path': 'notes/failures.md'},
            {'text': 'Recorded actual-error. Inspect file delimiter.', 'content_blake3': 'fixture-digest'}),
            (analysis_sequence, 'analyze_table', {'name': 'later.csv'}, {'total': 10})]:
        response = {'event_id': f'host-{seq}', 'status': 'completed', 'result': result}
        host.append({'phase': 2, 'name': name, 'arguments': args, 'response': response})
        events.append(NS(sequence=seq, event_id=f'event-{seq}', method='item/completed',
                         payload={'item': {'type': 'mcpToolCall', 'result': {'structuredContent': response}}}))
    return NS(events=sorted(events, key=lambda event: event.sequence), receipt_blake3='synthetic'), host


def test_item_type_detected_and_scope_drift_not_relabelled(monkeypatch):
    receipt, host = fixture(monkeypatch)
    result = module.observe_compaction(receipt, host, {'later.csv': 10}, expected_stage_input='current.csv')
    assert result['actual_compaction_observed']
    assert result['post_compaction_note_read_then_correct_analysis_observed']
    assert result['observations'][0]['subsequent_failure_memory_use']['matches_requested_stage_input'] is False


def test_pre_compaction_use_is_not_post_compaction_memory(monkeypatch):
    receipt, host = fixture(monkeypatch, read_sequence=-2, analysis_sequence=-1)
    result = module.observe_compaction(receipt, host, {'later.csv': 10}, expected_stage_input='later.csv')
    assert result['actual_compaction_observed']
    assert not result['post_compaction_note_read_then_correct_analysis_observed']


def test_unpaired_completion_not_accepted(monkeypatch):
    receipt, host = fixture(monkeypatch)
    receipt.events = receipt.events[1:]
    with pytest.raises(ValueError, match='without_prior_start'):
        module.observe_compaction(receipt, host, {'later.csv': 10}, expected_stage_input='later.csv')


def test_new_probe_flag_detects_completed_item_not_method_or_start():
    started = NS(method='item/started', payload={'item': {'type': 'contextCompaction'}})
    completed = NS(method='item/completed', payload={'item': {'type': 'contextCompaction'}})
    assert not module.has_completed_compaction([NS(events=[started])])
    assert module.has_completed_compaction([NS(events=[started, completed])])
