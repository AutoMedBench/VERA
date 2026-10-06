"""Project actual portable Codex evidence into the unchanged source rubric."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

from .portable_rollout import BUNDLE, ROOT, CONTINUATION_POLICY, joined_calls
from evamed_portable.bundle import Bundle
from evamed_portable.integrity import (byte_digest, contained_file, digest, file_digest,
    relative_path, strict_json, verify_receipt)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_value
from training.automedbench_lite.adapter import write_once


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _document(path):
    document = strict_json(path.read_bytes())
    _require(document['document_blake3'] == digest({k: v for k, v in document.items() if k != 'document_blake3'}),
             'portable_document_commitment_changed')
    return document


def _snapshot(path):
    from eva_agent.pipeline import FileSnapshot, WorkspaceSnapshot
    document = _document(path)
    _require(document['visibility'] == 'actor-public' and document['private_evaluator_data_included'] is False,
             'snapshot_not_actor_public')
    files = []
    for row in document['files']:
        relative_path(row['path'])
        data = row['text'].encode() if 'text' in row else None
        if 'retained_blob_relative_path' in row:
            base = path.parent.parent if path.parent.name == 'snapshots' else path.parent
            name = row['retained_blob_relative_path']
            _require(name == 'public-file-blobs/' + row['blake3'], 'public_blob_path_changed')
            blob = contained_file(base, name)
            _require(blob.stat().st_size == row['bytes'], 'public_blob_size_changed')
            reopened = blob.read_bytes()
            _require(data is None or data == reopened, 'public_inline_blob_mismatch')
            data = reopened
        _require(data is not None, 'public_file_bytes_missing')
        _require(len(data) == row['bytes'] and blake3_bytes(data) == row['blake3'], 'public_snapshot_bytes_changed')
        files.append(FileSnapshot(row['path'], data, len(data), '0600', row['blake3']))
    files.sort(key=lambda row: row.path)
    _require(len(files) == len({row.path for row in files}), 'duplicate_public_file')
    core = {'files': tuple(files), 'file_count': len(files), 'byte_count': sum(row.byte_count for row in files)}
    return WorkspaceSnapshot(label=path.stem, **core, tree_blake3=blake3_hex(core)), document


_IDENTITY = ('run_id', 'case_id', 'bundle_blake3', 'source_record_blake3', 'source_rubric_digest')


def _signed(path, public_key, expected_digest, schema, trajectory, identity=_IDENTITY):
    signed = strict_json(path.read_bytes())
    payload = verify_receipt(signed, public_key)
    _require(digest(signed) == expected_digest and payload['schema'] == schema and all(
        payload[key] == trajectory[key] for key in identity), 'signed_actor_evidence_changed')
    return payload


def _verify_launch(actor, ordinal, row, trajectory, definitions, construction, public_key):
    from eva_agent.codex_runtime import (CodexToolOffer, codex_turn_receipt_from_document,
        verify_codex_turn_receipt)
    _require(row['relative_directory'] == f'turns/{ordinal:02d}', 'native_turn_directory_changed')
    turn = actor / row['relative_directory']
    receipt = codex_turn_receipt_from_document(strict_json(contained_file(turn, 'codex-receipt.json').read_bytes()))
    verify_codex_turn_receipt(receipt)
    _require(receipt.receipt_blake3 == row['receipt_blake3'] and receipt.model == trajectory['model'] and
        receipt.visibility == 'actor-public' and receipt.status == row['status'] and
        receipt.thread_id == row['native_thread_id'] and receipt.sandbox.value == 'read-only',
        'actor_turn_receipt_changed')
    request = strict_json(contained_file(turn, 'logical-request.json').read_bytes())
    logical = request['logical_input']
    launch = _signed(contained_file(turn, 'launch.json'), public_key, row['launch_receipt_blake3'],
        'eva.medresearch-authenticated-Codex-launch.v1', trajectory)
    _require(request['private_evaluator_data_included'] is False and logical['judge_only_context'] is None and
        logical['public_context'] == {} and logical['mentions'] == [] and logical['output_schema'] is None and
        logical['role'] == 'strong_actor' and receipt.role.value == logical['role'] and
        receipt.input_blake3 == blake3_hex(logical) and launch['logical_input_blake3'] == digest(logical),
        'actual_actor_context_changed')
    _require(launch['native_turn'] == ordinal and launch['signed_before_Codex_run_turn'] is True and
        launch['signed_before_Codex_start_thread'] is (ordinal == 1) and
        launch['instruction_hash_encoding'] == 'UTF-8 bytes; no added newline' and
        launch['base_instructions_blake3'] == byte_digest((request['base_instructions'] or '').encode()) and
        launch['developer_instructions_blake3'] == byte_digest((request['developer_instructions'] or '').encode()) and
        launch['public_tool_catalog_blake3'] == digest(request['public_tool_catalog']) and
        launch['codex_binary_blake3'] == trajectory['codex_binary_blake3'] and
        launch['continuation_policy'] == CONTINUATION_POLICY and launch['host_conditioning_loss_weight'] == 0 and
        row['host_conditioning_loss_weight'] == 0 and launch['model'] == trajectory['model'] and
        launch['route'] == trajectory['route'], 'authenticated_launch_changed')
    catalog = [{'name': value['name'], 'description': value['description'], 'inputSchema': value['parameters']}
               for value in definitions.values()]
    offers = [CodexToolOffer(fully_qualified_name='eva_medresearch/' + value['name'],
        description=value['description'], input_schema=value['parameters'], parallel_safe=False,
        read_only=value['name'] == 'retrieve_frozen_evidence', allowed_stages=('S1', 'S2', 'S3', 'S4', 'S5', 'E2E'))
        for value in definitions.values()]
    _require(request['public_tool_catalog'] == catalog and
        logical['offered_tools'] == canonical_value([offer.canonical_catalog_entry() for offer in offers]) and
        logical['offered_tool_metadata'] == canonical_value([offer.sidecar_metadata_entry() for offer in offers]) and
        tuple(receipt.offered_mcp_tool_names) == tuple(offer.fully_qualified_name for offer in offers) and
        receipt.offered_tool_schema_blake3 == blake3_hex(logical['offered_tools']), 'canonical_tool_offer_changed')
    skills = logical['skills']
    _require(launch['selected_skills'] == skills and
        receipt.selected_skill_catalog_blake3 == blake3_hex(skills) and
        tuple(receipt.selected_skill_ids) == tuple(skill['skill_id'] for skill in skills), 'selected_skill_binding_changed')
    for skill in skills:
        path = Path(skill['path'])
        _require(path.is_relative_to(ROOT) and file_digest(contained_file(ROOT, str(path.relative_to(ROOT)))) ==
            skill['content_blake3'], 'selected_skill_source_changed')
    before, before_document = _snapshot(turn / 'before.json')
    after, after_document = _snapshot(turn / 'after.json')
    _require(before_document['document_blake3'] == row['before_snapshot_blake3'] == request['public_snapshot_blake3'] and
        after_document['document_blake3'] == row['after_snapshot_blake3'] and
        before_document['run_id'] == after_document['run_id'] == trajectory['run_id'], 'native_turn_snapshots_changed')
    continuation = request['continuation_injection']
    _require(launch['continuation_injection_blake3'] == (digest(continuation) if continuation else None),
        'host_continuation_commitment_changed')
    if ordinal == 1:
        task_files = {item.path: strict_json(item.content) for item in before.files if item.path in {'task.json', 'semantics.json'}}
        _require(continuation is None and strict_json(logical['public_text']) == {
            **before_document['state'], 'task': task_files['task.json'], 'semantics': task_files['semantics.json']},
            'initial_public_context_changed')
        from eva_agent.codex_runtime.runtime import _PRIVATE_TEXT
        if _PRIVATE_TEXT.search(logical['public_text']) is not None:
            from .portable_public_projection import verify_authorization, NAME, SCHEMA as QUALIFICATION_PUBLIC_POLICY
            from .portable_training_public_projection import SCHEMA as TRAINING_PUBLIC_POLICY
            authorization_schema = launch.get('public_projection_authorization_schema', QUALIFICATION_PUBLIC_POLICY)
            _require(authorization_schema in (QUALIFICATION_PUBLIC_POLICY, TRAINING_PUBLIC_POLICY),
                'unknown_public_projection_authorization_policy')
            if authorization_schema == TRAINING_PUBLIC_POLICY:
                from .portable_training_public_projection import verify_authorization, NAME, verify_consumption, CONSUMPTION_NAME
            authorization = strict_json(contained_file(turn, NAME).read_bytes())
            _require(digest(authorization) == launch.get('public_projection_authorization_blake3'),
                'public_projection_launch_authorization_missing')
            authorization_payload = verify_authorization(authorization, public_key=public_key, logical=logical,
                base=request['base_instructions'], developer=request['developer_instructions'],
                catalog=request['public_tool_catalog'], construction=construction,
                **{key: trajectory[key] for key in _IDENTITY})
            if authorization_schema == TRAINING_PUBLIC_POLICY:
                typed = authorization_payload['typed_options']
                _require(launch['public_projection_runtime_options_commitment'] == authorization_payload['runtime_options_commitment'] and
                    launch['public_projection_runtime_input_commitment'] == authorization_payload['runtime_input_commitment'] and
                    typed['model'] == receipt.model and typed['provider'] == receipt.provider and
                    typed['config_keys'] == list(receipt.config_keys) and receipt.thread_resumed is False,
                    'training_public_actual_Codex_options_changed')
                verify_consumption(strict_json(contained_file(turn, CONSUMPTION_NAME).read_bytes()), authorization,
                    public_key=public_key, native_thread_id=receipt.thread_id, runtime_thread_id=receipt.runtime_thread_id)
    else:
        _require(isinstance(continuation, dict) and set(continuation) == {
            'schema', 'run_id', 'native_turn', 'host_state', 'instruction', 'remaining_model_requests',
            'remaining_tool_attempts', 'conditioning_loss_weight', 'new_task_evidence_or_answers_supplied'} and
            continuation['schema'] == 'eva.medresearch-host-continuation-conditioning.v1' and
            continuation['run_id'] == trajectory['run_id'] and continuation['native_turn'] == ordinal and
            continuation['conditioning_loss_weight'] == 0 and continuation['new_task_evidence_or_answers_supplied'] is False and
            continuation['host_state'] == {key: before_document['state'][key] for key in
                ('next_stage', 'completed_stages', 'retrieved_evidence_ids', 'submission_attempted')} and
            continuation['instruction'] == 'Continue this same source case at the host next_stage with the existing tools and evidence. The formal host submission is incomplete. Do not restart, reset, or change the task.' and
            strict_json(logical['public_text']) == continuation,
            'host_continuation_not_exact_public_conditioning')
        _require(type(continuation['remaining_model_requests']) is int and 0 < continuation['remaining_model_requests'] <= 24 and
            continuation['remaining_tool_attempts'] == max(0, 24 - before_document['state']['tool_calls']),
            'host_continuation_budget_changed')
    return receipt, request, launch, before_document, after_document


def _project_turn(receipt, request, joins, hosts, frontier_offset, *, first):
    from eva_agent.codex_pipeline.adapter import (_tool_groups, _mcp_groups,
        codex_core_mcp_resource_operation, _validate_codex_core_resource_call)
    from training.benchmark_feedback.automed_codex import event
    groups = _mcp_groups(_tool_groups(receipt, verify_event_peak=True), set(receipt.offered_mcp_tool_names))
    frontiers = {call.tool_call_id: index + frontier_offset for index, group in enumerate(groups) for call in group}
    group_ids = {index + frontier_offset: str(uuid4()) for index in range(len(groups))}
    bindings = {row['codex_tool_call_id']: hosts[row['host_event_id']] for row in joins if 'host_event_id' in row}
    results, by_item, controls = [], {}, {}
    for call in receipt.tool_calls:
        if call.fully_qualified_name not in receipt.offered_mcp_tool_names:
            operation = codex_core_mcp_resource_operation(server=call.mcp_server, tool=call.mcp_tool,
                offered_mcp_tool_names=set(receipt.offered_mcp_tool_names))
            _require(operation is not None, 'unoffered_native_control')
            _validate_codex_core_resource_call(receipt, call, operation=operation)
            controls[call.upstream_item_id] = canonical_value(call)
            continue
        row = bindings[call.tool_call_id]
        failed = row['response']['isError']
        _require(call.status == ('failed' if failed else 'completed'), 'tool_failure_status_changed')
        core = {'result_id': row['event_id'], 'call_id': call.tool_call_id, 'name': row['name'],
            'frontier': frontiers[call.tool_call_id], 'parallel_group_id': group_ids[frontiers[call.tool_call_id]],
            'status': call.status, 'output': row['public_result'],
            'error_code': row['public_result'].get('error') if failed else None,
            'workspace_before_blake3': row['workspace_before_blake3'],
            'workspace_after_blake3': row['workspace_after_blake3']}
        result = {**core, 'receipt_blake3': blake3_hex(core)}
        results.append(result)
        _require(call.upstream_item_id not in by_item, 'duplicate_native_tool_item')
        by_item[call.upstream_item_id] = (call, result)
    messages = []
    if first:
        messages.append(event('system', {'base_instructions': request['base_instructions'],
            'developer_instructions': request['developer_instructions']}))
    user = event('user', request['logical_input'])
    messages.append(user)
    started, finished, visible_controls, visible_assistant = set(), set(), set(), set()
    for source_event in receipt.events:
        item = canonical_value(source_event.payload).get('item', {})
        item_id = item.get('id')
        if source_event.method == 'item/started' and item_id in by_item:
            call, result = by_item[item_id]
            _require(call.tool_call_id not in started, 'tool_started_twice')
            messages.append(event('assistant', {'name': call.fully_qualified_name,
                'arguments': canonical_value(call.arguments)}, (call.tool_call_id,), event_id=source_event.event_id))
            started.add(call.tool_call_id)
        elif source_event.method == 'item/completed' and item_id in by_item:
            call, result = by_item[item_id]
            _require(call.tool_call_id in started and call.tool_call_id not in finished, 'tool_completion_order_changed')
            messages.append(event('tool', result, (call.tool_call_id,), event_id=source_event.event_id))
            finished.add(call.tool_call_id)
        elif source_event.method == 'item/completed' and item_id in controls:
            _require(item_id not in visible_controls, 'native_control_completed_twice')
            visible_controls.add(item_id)
            messages.append(event('assistant', {'actual_native_control': controls[item_id]}, event_id=source_event.event_id))
        elif source_event.method == 'item/completed' and item.get('type') == 'agentMessage':
            _require(item_id not in visible_assistant, 'assistant_message_completed_twice')
            visible_assistant.add(item_id)
            messages.append(event('assistant', {'text': item['text'], 'phase': item.get('phase')}, event_id=source_event.event_id))
    _require(started == finished == set(bindings) and visible_controls == set(controls),
        'actor_visible_event_coverage_incomplete')
    return messages, results, len(groups), user.event_id


def prepare_trajectory(actor: Path, output: Path):
    from eva_agent.rubrics.registry import CompiledRubricRegistry
    from eva_agent.rubrics.models import CompiledRubric
    from eva_agent.training.slime_agent_judge import prepare_workspace_rollout

    actor = actor.resolve(strict=True)
    output = output.resolve()
    host_config = strict_json(contained_file(actor, 'host-tool-config.json').read_bytes())
    runtime_root = Path(host_config['run_root']).resolve(strict=True)
    _require(actor.is_relative_to(ROOT) and runtime_root.is_relative_to(ROOT) and
        output.is_relative_to(ROOT) and not output.is_relative_to(runtime_root) and not output.is_relative_to(actor),
        'judge_output_outside_private_workspace')
    trajectory = _document(actor / 'trajectory.json')
    bundle = Bundle(BUNDLE)
    public_key = bundle.manifest['fresh_execution_authority']['public_key_base64']
    recovery, episode_schema, join_turn = None, 'eva.medresearch-authenticated-Codex-episode.v1', joined_calls
    if trajectory['schema'] == 'eva.medresearch-recovered-Codex-trajectory.v1':
        from .portable_recovery import verify_recovery_view, joined_with_empty_templates, EPISODE_SCHEMA
        recovery = verify_recovery_view(actor, trajectory, public_key)
        episode_schema, join_turn = EPISODE_SCHEMA, joined_with_empty_templates
    _require((not trajectory['errors'] or recovery is not None) and trajectory['diagnostic_infrastructure_passed'],
        'actor_infrastructure_not_admitted')
    _require(bundle.manifest['document_blake3'] == trajectory['bundle_blake3'], 'actor_bundle_changed')
    _, source, construction, definitions = bundle.case(trajectory['case_id'])
    _require(source['record_blake3'] == trajectory['source_record_blake3'], 'source_record_changed')
    table = source['reward_contract']['rubric_table']
    CompiledRubricRegistry._verify_rubric_document(table)
    rubric = CompiledRubric.from_document(table)
    _require(rubric.digest == trajectory['source_rubric_digest'] and rubric.stage == trajectory['focus_stage'] and
        rubric.domain == trajectory['domain'], 'source_rubric_changed')
    episode = _signed(actor / 'episode-receipt.json', public_key, trajectory['episode_receipt_blake3'],
        episode_schema, trajectory)
    _require(episode['turns'] == trajectory['codex_turns'] and
        episode['all_joined_host_results'] == trajectory['joined_host_results'] and
        episode['private_reasoning_retained'] is False and episode['host_conditioning_loss_weight'] == 0 and all(
            episode[key] == trajectory[key] for key in ('before_snapshot_blake3', 'after_snapshot_blake3',
                'host_final_receipt_blake3', 'continuation_policy', 'control_stop_reason', 'model', 'route',
                'actual_model_requests', 'actual_tool_attempts')), 'authenticated_episode_changed')
    _require(trajectory['continuation_policy'] == CONTINUATION_POLICY and
        1 <= len(episode['turns']) == trajectory['actual_native_turn_count'] <= CONTINUATION_POLICY['max_native_turns'] and
        trajectory['actual_model_requests'] <= construction['runtime']['policy_budgets']['max_turns'],
        'native_episode_policy_changed')
    codex = ROOT / 'tools/codex-0.153.4/package/vendor/x86_64-unknown-linux-musl/bin/codex'
    _require(file_digest(codex) == trajectory['codex_binary_blake3'], 'actual_pinned_Codex_binary_changed')
    before, before_document = _snapshot(actor / 'before.json')
    after, after_document = _snapshot(actor / 'after.json')
    _require(before_document['document_blake3'] == trajectory['before_snapshot_blake3'] and
        after_document['document_blake3'] == trajectory['after_snapshot_blake3'] and
        before_document['run_id'] == after_document['run_id'] == trajectory['run_id'], 'actor_snapshots_changed')
    host = _signed(actor / 'host-final-disposition.json', public_key, trajectory['host_final_receipt_blake3'],
        'eva.medresearch-fresh-final-disposition.v1', trajectory)
    _require(host['all_stage_gates_passed'] == trajectory['all_stage_gates_passed'] and
        host['completed_stages'] == after_document['state']['completed_stages'], 'host_disposition_state_changed')
    event_path = actor / 'mcp-events.jsonl'
    events = [strict_json(line) for line in event_path.read_bytes().splitlines()] if event_path.exists() else []
    _require(len(events) == trajectory['actual_tool_attempts'], 'actual_host_event_count_changed')
    hosts, snapshot_documents = {}, {}
    previous, signed_previous, signed_count = before_document['document_blake3'], None, 0
    for ordinal, row in enumerate(events, 1):
        _require(row['event_blake3'] == digest({k: v for k, v in row.items() if k != 'event_blake3'}) and
            row['response_blake3'] == digest(row['response']) and row['attempt_number'] == ordinal and
            row['private_reasoning_retained'] is False, 'actual_tool_event_changed')
        _require(row['event_id'] not in hosts, 'duplicate_actual_host_event')
        for label in ['before', 'after']:
            _, snapshot = _snapshot(actor / 'snapshots' / (row['event_id'] + '-' + label + '.json'))
            _require(snapshot['document_blake3'] == row['workspace_' + label + '_blake3'] and
                snapshot['run_id'] == trajectory['run_id'], 'tool_snapshot_changed')
            snapshot_documents[snapshot['document_blake3']] = snapshot
        _require(previous == row['workspace_before_blake3'], 'workspace_transition_chain_changed')
        previous = row['workspace_after_blake3']
        result = row['public_result']
        if 'receipt_id' in result:
            signed_event = strict_json(contained_file(runtime_root / 'host', 'events/' + result['receipt_id'] + '.json').read_bytes())
            effect = verify_receipt(signed_event, public_key)
            signed_count += 1
            _require(digest(signed_event) == result['receipt_blake3'] and
                effect['event_id'] == result['receipt_id'] and effect['ordinal'] == signed_count and
                effect['previous_event_blake3'] == signed_previous and effect['tool'] == row['name'] and
                effect['arguments_blake3'] == digest(row['arguments']) and
                effect['public_result'] == {k: v for k, v in result.items() if k not in {'receipt_id', 'receipt_blake3'}} and
                all(effect[key] == trajectory[key] for key in ('run_id', 'case_id', 'source_record_blake3', 'source_rubric_digest')),
                'signed_host_effect_binding_changed')
            signed_previous = result['receipt_blake3']
        else:
            _require(result.get('gate_passed') is False and 'error' in result, 'unsigned_successful_host_effect')
        hosts[row['event_id']] = row
    _require(previous == after_document['document_blake3'] and host['event_count'] == signed_count and
        host['last_event_blake3'] == signed_previous, 'final_workspace_or_effect_chain_changed')

    messages, results, joins, receipts, source_turns = [], [], [], [], []
    event_offset, frontier_count, previous = 0, 0, before_document['document_blake3']
    stable_context, native_thread, runtime_thread, provider = None, None, None, None
    for ordinal, row in enumerate(episode['turns'], 1):
        receipt, request, launch, turn_before, turn_after = _verify_launch(actor, ordinal, row, trajectory, definitions, construction, public_key)
        _require(row['host_event_offset'] == event_offset and type(row['host_event_end']) is int and
            event_offset <= row['host_event_end'] <= len(events), 'native_turn_host_offsets_not_contiguous')
        joined = join_turn(receipt, actor, (event_offset, row['host_event_end']))
        _require(joined == row['joined_host_results'] and turn_before['document_blake3'] == previous,
            'native_turn_evidence_chain_changed')
        end_hash = events[row['host_event_end'] - 1]['workspace_after_blake3'] if row['host_event_end'] > event_offset else previous
        _require(end_hash == turn_after['document_blake3'], 'native_turn_after_not_actual_host_boundary')
        context = {key: request[key] for key in ('base_instructions', 'developer_instructions', 'public_tool_catalog')}
        context.update(skills=request['logical_input']['skills'], argument_type_binding_blake3=launch['argument_type_binding_blake3'])
        if ordinal == 1:
            stable_context, native_thread, runtime_thread, provider = context, receipt.thread_id, receipt.runtime_thread_id, receipt.provider
        _require(context == stable_context and receipt.thread_id == native_thread and receipt.runtime_thread_id == runtime_thread and
            receipt.provider == provider and receipt.thread_resumed is False, 'native_thread_or_actor_context_reset')
        _require(ordinal == len(episode['turns']) or receipt.status == 'completed', 'continued_after_incomplete_native_turn')
        turn_messages, turn_results, count, user_event_id = _project_turn(receipt, request, joined, hosts, frontier_count, first=ordinal == 1)
        messages.extend(turn_messages); results.extend(turn_results); joins.extend(joined); receipts.append(receipt)
        source_turns.append({**row, 'native_turn': ordinal, 'runtime_thread_id': receipt.runtime_thread_id,
            'runtime_turn_id': receipt.runtime_turn_id, 'native_turn_id': receipt.turn_id,
            'logical_input_blake3': launch['logical_input_blake3'],
            'base_instructions_blake3': launch['base_instructions_blake3'],
            'developer_instructions_blake3': launch['developer_instructions_blake3'],
            'host_conditioning_user_event_id': user_event_id, 'host_conditioning_loss_weight': 0,
            'continuation_injection': request['continuation_injection']})
        frontier_count += count; event_offset = row['host_event_end']; previous = turn_after['document_blake3']
    _require(event_offset == len(events) and previous == after_document['document_blake3'] and
        joins == trajectory['joined_host_results'] and len(results) == len(events) and
        len({row['call_id'] for row in results}) == len(results) and len({r.turn_id for r in receipts}) == len(receipts),
        'aggregate_actual_event_coverage_changed')
    last = receipts[-1]
    _require(last.receipt_blake3 == trajectory['codex_turn_receipt_blake3'] and last.status == trajectory['codex_turn_status'] and
        episode['turns'][-1]['launch_receipt_blake3'] == trajectory['launch_receipt_blake3'], 'last_native_turn_alias_changed')
    terminal = None
    if trajectory['authenticated_budget_terminal_blake3'] is not None:
        terminal = _signed(actor / 'host-budget-terminal.json', public_key, trajectory['authenticated_budget_terminal_blake3'],
            'eva.medresearch-codex-budget-terminal.v1', trajectory, identity=('run_id', 'case_id', 'bundle_blake3'))
        _require(terminal['codex_turn_receipt_blake3'] == last.receipt_blake3 and terminal['clinical_or_ability_score'] is None,
            'terminal_receipt_binding_changed')
        outcome = terminal['outcome']
        if trajectory['policy_budget_terminal']:
            _require(outcome['schema'] == 'eva.codex-policy-turn-budget.v1' and
                outcome['receipt_blake3'] == last.receipt_blake3 and outcome['actual_terminal_status'] == last.status and
                outcome['turn_id'] == last.turn_id and outcome['budget_exhausted'] is True and
                outcome['infrastructure_error'] is None and outcome['terminal_observed_ns'] is not None,
                'native_policy_terminal_changed')
        else:
            _require(outcome['control_stop_reason'] == trajectory['control_stop_reason'] and
                outcome['native_turn_count'] == len(receipts) and outcome['actual_model_requests'] == trajectory['actual_model_requests'],
                'host_control_terminal_changed')
    owned_output_proof = None
    if last.status != 'completed' and not trajectory['policy_budget_terminal']:
        proof_path = actor / 'slime-output-budget-terminal.json'
        if proof_path.is_file():
            from .slime_token_admission import verify_owned_output_receipt
            # The verifier reopens exact tokens/logprobs and original receipt joins;
            # only the proof commitment enters Judge context, never private token material.
            verify_owned_output_receipt(actor)
            owned_output_proof = file_digest(proof_path)
    _require(last.status == 'completed' or (trajectory['policy_budget_terminal'] and terminal is not None) or
        owned_output_proof is not None, 'unfinished_actor_without_verified_policy_budget')
    _require(trajectory['control_stop_reason'] == 'host_submission_boundary' or terminal is not None or recovery is not None,
        'host_control_stop_without_authenticated_terminal')
    _require(recovery is None or (last.status == 'completed' and after_document['state']['submission_attempted'] is True),
        'recovered_capture_not_actual_formal_terminal')
    trace = {'results': results, 'declared_call_ids': [r['call_id'] for r in results],
        'joined_call_ids': [r['call_id'] for r in results], 'frontier_count': frontier_count,
        'max_parallelism_observed': max(r.max_parallelism_observed for r in receipts), 'retry_count': 0}
    identifier = str(uuid4())
    document = {'schema': 'eva.portable-codex-source-rubric-assessment.v1', 'task_id': identifier,
        'sandbox_id': trajectory['case_id'], 'candidate_id': identifier, 'executable_episode_id': trajectory['run_id'],
        'route_id': trajectory['route'], 'model_id': last.model, 'provider': last.provider,
        'messages': canonical_value(messages), 'assistant_output': last.final_response or '',
        'provider_receipt_blake3': trajectory['episode_receipt_blake3'],
        'tool_trace': {**trace, 'trace_blake3': blake3_hex(trace)},
        'workspace_before': canonical_value(before), 'workspace_after': canonical_value(after),
        'rubric_table': table, 'provider_metadata': {'actual_actor_trajectory_blake3': trajectory['document_blake3'],
            'bundle_blake3': trajectory['bundle_blake3'], 'focus_stage': trajectory['focus_stage'],
            'observation_boundary': 'whole_actual_fresh_S1_through_S5_attempt',
            'source_contract_unchanged': True, 'private_reasoning_included': False,
            'host_gate_is_not_rubric_score': True, 'host_final_receipt_blake3': trajectory['host_final_receipt_blake3'],
            'provider_receipt_semantics': 'host-signed aggregate binding every original native Codex receipt',
            'authenticated_episode_blake3': trajectory['episode_receipt_blake3'],
            'native_codex_turns': source_turns, 'actual_native_turn_count': len(receipts),
            'native_thread_id': native_thread, 'same_native_thread_verified': True,
            'all_visible_messages_and_host_results_joined': True, 'host_continuation_conditioning_loss_weight': 0,
            'continuation_policy': CONTINUATION_POLICY, 'actual_terminal_status': last.status,
            'control_stop_reason': trajectory['control_stop_reason'],
            'authenticated_budget_terminal_blake3': trajectory['authenticated_budget_terminal_blake3'],
            'authenticated_terminal_outcome': terminal['outcome'] if terminal else None,
            'owned_output_budget_proof_file_blake3': owned_output_proof,
            'private_token_material_included': False,
            'posthoc_capture_recovery': {key: recovery[key] for key in (
                'schema', 'admission_blake3', 'original_trajectory_blake3', 'original_episode_blake3',
                'original_capture_errors', 'snapshot_substitutions', 'actor_replayed',
                'receipt_admission_scope')} if recovery else None}}
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    source_path = output / 'source.json'
    source_path.write_text(json.dumps(document, separators=(',', ':')))
    source_path.chmod(0o600)
    prepared = prepare_workspace_rollout(document, rubric=rubric, source_path=source_path,
        judge_model_id='aws/anthropic/bedrock-claude-opus-5', judge_route_id='opus_5')
    write_once(output / 'preflight.json', {'schema': 'eva.portable-codex-judge-preflight.v1',
        'source_blake3': prepared.trajectory.source_blake3, 'rubric_digest': rubric.digest,
        'stage': rubric.stage, 'domain': rubric.domain, 'actual_host_calls': len(events),
        'actual_native_turns': len(receipts), 'authenticated_episode_blake3': trajectory['episode_receipt_blake3'],
        'exact_source_rubric_unchanged': True, 'providers_called': 0})
    return rubric


_JUDGE_ADMISSION_LOCK = threading.Lock()
_JUDGE_ADMISSION = None
_JUDGE_LIMIT = None


def _judge_admission():
    """One limit shared by every portable judge pair in this training process."""
    global _JUDGE_ADMISSION, _JUDGE_LIMIT
    value = os.environ.get('EVAMED_JUDGE_CONCURRENCY', '8')
    _require(value.isdecimal() and 1 <= int(value) <= 32, 'invalid_EVAMED_JUDGE_CONCURRENCY')
    with _JUDGE_ADMISSION_LOCK:
        if _JUDGE_ADMISSION is None:
            _JUDGE_LIMIT = int(value)
            _JUDGE_ADMISSION = threading.BoundedSemaphore(_JUDGE_LIMIT)
        _require(_JUDGE_LIMIT == int(value), 'judge_concurrency_changed_during_process')
        return _JUDGE_ADMISSION, _JUDGE_LIMIT


def _bounded_grade(grade_once, source, rubric, output, route, codex):
    admission, limit = _judge_admission()
    queued = time.monotonic()
    with admission:
        started = time.monotonic()
        try:
            return grade_once(source, rubric, output, route, codex)
        finally:
            write_once(output.parent / (output.name + '-admission.json'), {
                'schema': 'eva.portable-judge-concurrency-admission.v1',
                'scope': 'all portable judge calls in this training process', 'process_id': os.getpid(),
                'concurrency_limit': limit, 'queue_wait_seconds': started - queued,
                'judge_runtime_seconds': time.monotonic() - started,
                'queue_wait_included_in_judge_runtime': False})


def judge_trajectory(actor, output):
    from .benchmark_judge import grade_once
    _judge_admission()
    rubric = prepare_trajectory(Path(actor), Path(output))
    spec = importlib.util.spec_from_file_location('evamed_judge_credentials', ROOT / 'evamed-codex/harness-audit/real_codex_inference_hub_bridge.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    route = helper._route('opus_5', 'aws/anthropic/bedrock-claude-opus-5', 'anthropic', helper._secret_keys()[0])
    codex = ROOT / 'tools/codex-0.153.4/package/vendor/x86_64-unknown-linux-musl/bin/codex'
    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(_bounded_grade, grade_once, Path(output) / 'source.json', rubric, Path(output) / name, route, codex)
                   for name in ['judge', 'reward-verifier']}
        for name, future in futures.items():
            try: results[name] = future.result()
            except Exception as exc: results[name] = {'verified': False, 'error_type': type(exc).__name__}
    valid = all(row['verified'] for row in results.values())
    agree = valid and results['judge']['item_scores_bps'] == results['reward-verifier']['item_scores_bps']
    if valid:
        _require(results['judge']['assessment_id'] != results['reward-verifier']['assessment_id'], 'judge_not_independent')
    return write_once(Path(output) / 'pair.json', {'schema': 'eva.portable-opus-independent-reward.v1',
        'pair_id': str(uuid4()), 'domain': rubric.domain, 'stage': rubric.stage, 'rubric_digest': rubric.digest,
        'results': results, 'both_assessments_verified': valid, 'exact_item_agreement': agree,
        'reward_admitted': agree, 'reward_bps': results['judge']['reward_bps'] if agree else None,
        'attribution_reward_weight': 0, 'stage_target_evidence_requires_separate_typed_admission': True})
