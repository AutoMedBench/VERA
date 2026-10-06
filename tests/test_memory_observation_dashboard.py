"""Bounded observational footer; no blanket memory/clinical success inference."""
import json
from pathlib import Path

from blake3 import blake3

from test_training_validation_dashboard import dashboard


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def history(root):
    result = {'schema': 'eva.failure-memory-probe-result.v1', 'passed': False,
              'actual_tool_calls': 44, 'checkpoint_identity_blake3': 'a' * 64}
    write(root / 'result.json', result)
    write(root / 'request-plan.json', {'schema': 'eva.failure-memory-probe-plan.v1',
        'checkpoint_identity_blake3': 'a' * 64, 'skill': {'profile': 'evamed-codex-v1.2-research'}})
    write(root / 'independent-verification.json', {
        'schema': 'eva.failure-memory-independent-verification.v1', 'valid': True,
        'all_snapshot_bytes_verified': True, 'actual_receipts_reopened': 3,
        'actual_host_tool_pairs_reopened': 44, 'operational_memory_checks_passed': True,
        'full_output_contract_passed': False,
        'source_result_blake3': blake3((root / 'result.json').read_bytes()).hexdigest()})


def evaluation(root):
    write(root / 'track-rollouts/attempt.json', {'server_binding': {'identity_file_blake3': 'b' * 64}})
    write(root / 'track-rollouts/extra-memory-skill/profile.json', {
        'schema': 'eva.automedbench-extra-memory-skill-binding.v2', 'profile': 'evamed-codex-v1.3-supra'})
    return {'label': 'Qwen v5', 'actor_kind': 'local_qwen', 'actor_root': str(root)}


def receipt(path, *, completed=1, unfinished=0):
    events = []
    for index in range(completed + unfinished):
        for method in ('item/started', 'item/completed') if index < completed else ('item/started',):
            event = {'sequence': len(events), 'method': method,
                     'payload': {'item': {'type': 'contextCompaction', 'id': str(index)}}}
            event['event_blake3'] = dashboard._content_digest(event, 'event_blake3')
            events.append(event)
    value = {'schema': 'eva.codex-turn-receipt.v1', 'events': events, 'tool_calls': [],
             'selected_skill_ids': ['summary_failures'], 'final_response': 'PRIVATE_NOT_FOR_DISPLAY'}
    value['receipt_blake3'] = dashboard._content_digest(value, 'receipt_blake3')
    write(path, value)


def test_prior_verification_is_bound_and_does_not_promote_failed_output_contract(tmp_path):
    history(tmp_path)
    text = '\n'.join(dashboard.historical_memory_observation_lines(tmp_path))
    assert 'operational VERIFIED; full output contract FAILED; 44 joined' in text
    assert 'profile=evamed-codex-v1.2-research; checkpoint=aaaaaaaaaaaa' in text
    result = json.loads((tmp_path / 'result.json').read_text())
    result['actual_tool_calls'] = 45
    write(tmp_path / 'result.json', result)
    text = '\n'.join(dashboard.historical_memory_observation_lines(tmp_path))
    assert 'unavailable/mismatched' in text and 'operational VERIFIED' not in text


def test_dynamic_completed_events_are_separate_from_unfinished_and_semantic_quality(tmp_path):
    data, actor = tmp_path / 'data', tmp_path / 'actor'
    history(data / 'runs/qwen35-failure-memory-v12-20260910.v1')
    row = evaluation(actor)
    receipt(actor / 'track-rollouts/classification/turns/01-planning/receipt.json', completed=36, unfinished=1)
    text = '\n'.join(dashboard.memory_observation_footer(data, [row]))
    assert 'completed compactions=36; unfinished=1; checked receipts=1' in text
    assert 'summary_failures SDK selections=1/1' in text
    assert 'semantic retention: NOT MEASURED; broad memory_context_pass unchanged' in text
    assert 'checkpoint=aaaaaaaaaaaa' in text and 'checkpoint=bbbbbbbbbbbb' in text
    assert 'v1.2-research' in text and 'v1.3-supra' in text and 'PRIVATE_NOT_FOR_DISPLAY' not in text


def test_receipt_cache_reopens_changes_and_rejects_nested_commitment_tampering(tmp_path, monkeypatch):
    row = evaluation(tmp_path)
    path = tmp_path / 'track-rollouts/classification/turns/01-planning/receipt.json'
    receipt(path)
    dashboard._memory_receipt_observation.cache_clear()
    original, opened = Path.open, []
    def observe(target, *args, **kwargs):
        if target.name == 'receipt.json': opened.append(target)
        assert target.name not in {'training-tokens.json', 'trajectory.json', 'grade.json'}
        return original(target, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', observe)
    assert 'completed compactions=1' in '\n'.join(dashboard.evaluation_memory_observation_lines(row))
    dashboard.evaluation_memory_observation_lines(row)
    assert opened == [path]
    value = json.loads(path.read_text())
    value['events'][0]['payload']['item']['id'] = 'tampered'
    value['receipt_blake3'] = dashboard._content_digest(value, 'receipt_blake3')
    write(path, value)
    text = '\n'.join(dashboard.evaluation_memory_observation_lines(row))
    assert 'checked receipts=0; unavailable/over-limit=1' in text


def test_oversized_or_partial_receipt_is_unavailable_not_verified_zero(tmp_path, monkeypatch):
    row = evaluation(tmp_path)
    path = tmp_path / 'track-rollouts/classification/turns/01-planning/receipt.json'
    receipt(path)
    monkeypatch.setattr(dashboard, 'MAX_MEMORY_RUN_BYTES', 1)
    text = '\n'.join(dashboard.evaluation_memory_observation_lines(row))
    assert 'checked receipts=0; unavailable/over-limit=1' in text
    assert 'selection is not demonstrated skill use' in text
    monkeypatch.setattr(dashboard, 'MAX_MEMORY_RUN_BYTES', 8 * 1024 * 1024)
    path.write_text('{')
    assert 'checked receipts=0; unavailable/over-limit=1' in '\n'.join(dashboard.evaluation_memory_observation_lines(row))
    assert dashboard.evaluation_memory_observation_lines({**row, 'actor_kind': 'native_astra'}) == []
