"""Narrow S3 pilot slices with portable, verifier-only host execution evidence.

This is not an S2 tier, an Agent Judge result, or a clinical-performance claim.
Only completed native turns with one S3 execution and optional successful S3
skill calls are supported. Host-only contracts/checks never enter messages.
Verification needs the trusted public key store and original signed bulk data,
but not a live provider, execution container, teacher workspace, or host session.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator
from eva_agent.admission.receipts import SignedEnvelope, verify_signed_envelope
from eva_agent.codex_pipeline.native_policy_v2 import (
    advance_guided_stage_frontier_v1, build_stage_tool_guidance_v1,
    start_guided_stage_frontier_v1, verify_guided_stage_call_v1,
)
from eva_agent.codex_runtime import codex_turn_receipt_from_document
from eva_agent.pipeline import ToolResult
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes, canonical_value
from eva_agent.sources.legacy_execution import _load_legacy_modules
from .eva_hf_release import _credential_bytes_rule
from .frontier_prefix_sft import _load_bulk_records, _safe_relative, _strict_object, _uuid
from .native_astra_teacher import NATIVE_ASTRA_MODEL, NATIVE_ASTRA_PROVIDER, NATIVE_ASTRA_ROUTE
from .s2_frontier_prefix_sft import (
    _authoritative_legacy_tool_catalog, _mcp_observation,
    _verify_completed_projection, _workspace_snapshot_files,
)
from .s2_skill_frontier_support import SKILL_NAMES, verified_offered_catalog

DATASET_SCHEMA = 'eva.execution-verified-s3-pilot-sft-dataset.v1'
SLICE_SCHEMA = 'eva.execution-verified-s3-pilot-sft-slice.v1'
VERIFY_SCHEMA = 'eva.execution-verified-s3-pilot-sft-verification.v1'


class S3PilotSFTError(ValueError):
    """Static, safe rejection reason for unsupported or mismatched evidence."""


def _require(condition: Any, reason: str) -> None:
    if not condition:
        raise S3PilotSFTError(reason)


def _json(payload: bytes) -> Any:
    return json.loads(payload, object_pairs_hook=_strict_object)


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b''.join(canonical_json_bytes(row) for row in rows)


def _safe(name: str) -> str:
    return _safe_relative(name, label='S3 evidence')


def _read(path: Path) -> bytes:
    _require(path.is_file() and not path.is_symlink(), 'evidence_file_boundary')
    data = path.read_bytes()
    _require(path.name != 'auth.json' and _credential_bytes_rule(data) is None, 'credential_pattern')
    return data


def _skills(observed: Sequence[Any], catalog: Sequence[Mapping[str, Any]], delivery: Mapping[str, Any]) -> None:
    definitions = {row['name']: row for row in catalog}
    visible = set(delivery['visible_skill_ids'])
    for item in observed:
        name, args, output = item.tool_result['name'], item.arguments, item.output
        _require(args.get('stage') == 'S3', 'actor_tool_stage')
        _require(not list(Draft202012Validator(definitions[name]['input_schema']).iter_errors(canonical_value(args))), 'canonical_tool_arguments')
        if name not in SKILL_NAMES:
            continue
        _require(item.tool_result['workspace_before_blake3'] == item.tool_result['workspace_after_blake3'], 'skill_workspace_mutation')
        if name == 'search_skills':
            _require(set(output) == {'matches'} and isinstance(output['matches'], list), 'skill_search_shape')
            ids = []
            for match in output['matches']:
                _require(set(match) == {'skill_id', 'description'} and match['skill_id'] in visible
                    and isinstance(match['description'], str) and bool(match['description'])
                    and args['query'].casefold() in f"{match['skill_id']} {match['description']}".casefold(), 'skill_search_visibility')
                ids.append(match['skill_id'])
            _require(ids == sorted(set(ids)), 'skill_search_inventory')
        else:
            _require(set(output) == {'skill_id', 'content', 'content_blake3', 'delivery'}
                and output['skill_id'] == args['skill_id'] and output['skill_id'] in visible
                and isinstance(output['content'], str) and bool(output['content'])
                and output['content_blake3'] == blake3_hex(output['content'])
                and output['delivery'] == 'policy-visible-tool-observation', 'skill_load_binding')


def _hydration(source: Mapping[str, Any], runtime: Mapping[str, Any], tools: Any,
               actor_files: Mapping[str, bytes], bundle: Mapping[str, bytes], modules: Any, trust: Any) -> str:
    hydration = source['messages'][1]['content']['teacher_prerequisite_hydration']
    guidance = build_stage_tool_guidance_v1(public_runtime_context=runtime, source_tool_catalog=tools)
    rows = hydration['tool_results']
    _require(hydration['focus'] == 'S3' and hydration['schema'] == 'eva.codex-teacher-prerequisite-hydration.v1'
        and hydration['source_candidate_id'] == guidance.source_candidate_id
        and hydration['guidance_blake3'] == guidance.guidance_blake3
        and hydration['provider_calls'] == 0 and hydration['canonical_tool_schemas_changed'] is False
        and hydration['mcp_wire_schema_changed'] is False and len(rows) == len(guidance.frontiers)
        and hydration['hydrated_frontier_indices'] == list(range(len(rows)))
        and hydration['next_actor_frontier_index'] == len(rows), 'S3_hydration_binding')
    envelopes = {_json(value)['payload_sha256']: _json(value) for name, value in bundle.items()
                 if name.startswith('prerequisites/') and name.endswith('/host-receipt.json')}
    by_digest = {}
    for envelope in envelopes.values():
        modules.host_receipts.verify_host_receipt(envelope, trust)
        by_digest[modules.canonical.sha256_value(envelope)] = envelope
    state = start_guided_stage_frontier_v1(guidance)
    cursor = hydration['workspace_before_blake3']
    artifacts = runtime['s1_plan_contract']['stage_artifacts']
    _require(hydration['artifact_relative_paths'] == [artifacts['S1'], artifacts['S2']], 'hydration_artifact_paths')
    for index, (frontier, row) in enumerate(zip(guidance.frontiers, rows, strict=True)):
        name = frontier['tool_name']
        args = _json(actor_files[_safe(artifacts['S2'])]) if name == 'materialize_evidence_selection' else frontier['arguments']
        if name == 'materialize_plan':
            _require(_json(actor_files[_safe(artifacts['S1'])]) == canonical_value(args), 'hydrated_plan_bytes')
        proof = verify_guided_stage_call_v1(guidance, state=state, frontier_index=index,
            tool_call_id=row['call_id'], tool_name=name, arguments=args)
        state = advance_guided_stage_frontier_v1(guidance, state=state, frontier_index=index,
            tool_call_id=row['call_id'], tool_name=name, arguments=args, call_proof=proof, tool_result=ToolResult(**row))
        _require(row['workspace_before_blake3'] == cursor and row['output']['gate_passed'] is True
            and row['output']['failed_check_ids'] == [], 'hydration_host_gate')
        cursor = row['workspace_after_blake3']
        envelope = by_digest[row['output']['receipt_sha256']]
        expected = modules.agent_rollout._stage_tool_result(envelope, trust)
        _require(all(row['output'].get(k) == v for k, v in expected.items()), 'hydration_signed_observation')
    _require(hydration['tool_result_receipt_blake3s'] == list(state.tool_result_receipt_blake3s)
        and hydration['frontier_state_blake3'] == state.state_blake3
        and cursor == hydration['workspace_after_blake3'] == source['workspace_before']['tree_blake3'], 'hydration_final_state')
    return rows[-1]['output']['receipt_sha256']


def _execution(bundle: Mapping[str, bytes], execution: Any, runtime: Mapping[str, Any],
               actor_files: Mapping[str, bytes], prerequisite: str, modules: Any, trust: Any) -> Mapping[str, Any]:
    contract, envelope = _json(bundle['host/execution-contract.json']), _json(bundle['host/host-receipt.json'])
    payload = modules.host_receipts.verify_host_receipt(envelope, trust)
    modules.validation.validate_execution_contract(contract)
    obs = payload['observations']
    _require(contract['stage'] == 'S3' and contract['focus'] == 'S3'
        and contract['episode_id'] == runtime['active_episode_id'] == payload['episode_id']
        and contract['prerequisite_receipt_sha256'] == prerequisite
        and obs['prerequisites']['s2_verified'] is True, 'execution_prerequisite_binding')
    declared = runtime['execution_stages']['S3']
    _require(declared['artifact_relative_path'] == contract['artifact']['relative_path']
        and declared['artifact_json_schema'] == contract['artifact']['json_schema']
        and declared['limits'] == contract['limits'], 'public_execution_contract')
    code = bundle['host/submission.py']
    _require(code == execution.arguments['code'].encode()
        and obs['contract']['submission_sha256'] == modules.canonical.sha256_bytes(code)
        and obs['contract']['contract_sha256'] == modules.canonical.sha256_value(contract), 'actual_code_contract_binding')
    for entry in contract['inputs']:
        content = bundle['inputs/' + _safe(entry['relative_path'])]
        _require(len(content) == entry['byte_count'] and modules.canonical.sha256_bytes(content) == entry['sha256'], 'execution_input_binding')
    expected = {**modules.agent_rollout._tool_result_from_execution(envelope), 'attempt': 1,
        'host_receipt_sha256': modules.canonical.sha256_bytes(bundle['host/host-receipt.json'])}
    _require(canonical_value(execution.output) == expected, 'actual_signed_tool_output')
    _require(expected['gate_passed'] is True and expected['failed_check_ids'] == [], 'S3_host_gate_not_passed')
    path = _safe(contract['artifact']['relative_path'])
    names = {name.removeprefix('host/workspace/') for name in bundle if name.startswith('host/workspace/')}
    _require(names == {path}, 'unexpected_execution_files')
    content = bundle['host/workspace/' + path]
    evaluated = modules.execution.evaluate_json_artifact_content(contract['artifact'], content)
    artifact = obs['artifact']
    _require(evaluated['all_checks_passed'] and evaluated['critical_checks_passed']
        and all(evaluated[k] == artifact[k] for k in ('byte_count', 'sha256', 'parser_valid', 'schema_valid', 'checks', 'critical_checks_passed'))
        and artifact['newly_created'] and artifact['host_reopened'] and artifact['reopen_sha256_match']
        and artifact['no_unexpected_files'] and obs['workspace']['changed']
        and obs['workspace']['file_count'] == 1 and obs['workspace']['total_bytes'] == len(content)
        and obs['workspace']['within_limits'] and obs['isolation']['verified'], 'independent_host_artifact_check')
    return {'artifact_relative_path': path, 'artifact_file_blake3': blake3_bytes(content),
        'artifact_byte_count': len(content), 'submitted_code_blake3': blake3_bytes(code),
        'host_receipt_blake3': blake3_hex(envelope), 'execution_contract_blake3': blake3_hex(contract),
        'passed_artifact_checks': len(evaluated['checks']), 'host_gate_passed': True,
        'execution_workspace_changed': True, 'no_unexpected_execution_files': True}


def _slice(bundle: Mapping[str, bytes], *, bulk: Any, registry: Any, modules: Any, trust: Any) -> Mapping[str, Any]:
    source, native = _json(bundle['source.json']), _json(bundle['native-route.json'])
    record, bulk_binding = bulk
    _require(_json(bundle['bulk.json']) == record and record['stage'] == 'S3' and record['split'] == 'train'
        and source['candidate_id'] == record['candidate_id'] and source['sandbox_id'] == record['sandbox_id']
        and source['schema'] == 'eva.codex-teacher-full-trajectory.v1'
        and source['route_id'] == NATIVE_ASTRA_ROUTE
        and source['task_id'] == f"{source['sandbox_id']}--{NATIVE_ASTRA_ROUTE}", 'original_S3_identity')
    _require(native['document_blake3'] == blake3_hex({k: v for k, v in native.items() if k != 'document_blake3'})
        and native['trajectory_blake3'] == blake3_hex(source) and native['returned_model'] is None
        and native['task_id'] == source['task_id'] and native['requested_model'] == NATIVE_ASTRA_MODEL
        and native['provider'] == NATIVE_ASTRA_PROVIDER and native['semantic_retry_count'] == 0
        and native['instruction_scope'] == 'focus-only-S3' and native['skill_stage_binding'] == 'S3', 'native_route_provenance')
    metadata = source['provider_metadata']
    receipt = codex_turn_receipt_from_document(metadata['codex_turn_receipt'])
    _require(receipt.status == 'completed' and receipt.sandbox.value == 'read-only' and receipt.role.is_actor
        and receipt.visibility == 'actor-public' and not receipt.selected_skill_ids
        and receipt.model == NATIVE_ASTRA_MODEL and receipt.provider == NATIVE_ASTRA_PROVIDER
        and not any('reasoning' in event.method.lower() for event in receipt.events)
        and source['semantic_retry_count'] == 0 and source['judge_calls'] == 0, 'native_actor_boundary')
    names = [call.fully_qualified_name.removeprefix('evamed/') for call in receipt.tool_calls]
    _require(names and names[-1] == 'execute_code' and names.count('execute_code') == 1
        and all(name in SKILL_NAMES for name in names[:-1]), 'single_S3_execution_prefix')
    observed = [_mcp_observation(call, expected_name=name) for call, name in zip(receipt.tool_calls, names, strict=True)]
    _verify_completed_projection(source, metadata=metadata, receipt=receipt, accepted=observed)
    before, actor_files = _workspace_snapshot_files(source['workspace_before'], expected_label='before-rollout')
    after, after_files = _workspace_snapshot_files(source['workspace_after'], expected_label='after-rollout')
    _require(before == after and actor_files == after_files
        and all(item.tool_result['workspace_before_blake3'] == before == item.tool_result['workspace_after_blake3'] for item in observed), 'actor_workspace_boundary')
    runtime, policy = _json(actor_files['.eva/runtime-context.json']), _json(actor_files['.eva/source-policy.json'])
    context = source['messages'][1]['content']
    _require(runtime['focus'] == policy['focus'] == 'S3' and runtime['policy_blake3'] == blake3_hex(policy)
        and context['execution_binding'] == runtime, 'public_runtime_context')
    compiled = registry.resolve(record['domain'], 'S3')
    rubric = canonical_value(compiled.to_document())
    _require(source['rubric_table'] == record['reward_contract']['rubric_table'] == rubric
        and context['public_reward_contract']['rubric_table'] == rubric
        and record['reward_contract']['registry_digest'] == registry.digest, 'shared_S3_rubric')
    tools = _authoritative_legacy_tool_catalog(policy['tools'], expected_blake3=runtime['tool_catalog_blake3'])
    catalog = verified_offered_catalog(tools, receipt, source['skill_delivery'])
    _skills(observed, catalog, source['skill_delivery'])
    prerequisite = _hydration(source, runtime, tools, actor_files, bundle, modules, trust)
    evidence = _execution(bundle, observed[-1], runtime, actor_files, prerequisite, modules, trust)
    groups = []
    for item in observed:
        key = (item.tool_result['frontier'], item.tool_result['parallel_group_id'])
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        groups[-1][1].append(item)
    _require([key[0] for key, _ in groups] == list(range(len(groups)))
        and len({key[1] for key, _ in groups}) == len(groups)
        and len(groups[-1][1]) == 1, 'actual_execution_groups')
    messages = [{'role': row['role'], 'content': row['content']} for row in source['messages'][:2]]
    for _key, group in groups:
        messages.append({'role': 'assistant', 'content': {'tool_calls': [
            {'name': item.tool_result['name'], 'arguments': item.arguments,
             'codex_tool_call_receipt_blake3': item.call.receipt_blake3} for item in group]}})
        messages.extend({'role': 'tool', 'name': item.tool_result['name'], 'content': item.observation} for item in group)
    return canonical_value({'schema': SLICE_SCHEMA, 'tier': 'execution_verified_s3_pilot',
        'sandbox_id': source['sandbox_id'], 'candidate_id': source['candidate_id'], 'source_task_id': source['task_id'],
        'route_id': source['route_id'], 'model_id': receipt.model, 'provider': receipt.provider, 'returned_model': None,
        'stage': 'S3', 'domain': record['domain'], 'rubric_id': compiled.rubric_id, 'rubric_digest': compiled.digest,
        'rubric_table': rubric, 'bulk_binding_blake3': bulk_binding, 'source_record_file_blake3': blake3_bytes(bundle['source.json']),
        'codex_turn_receipt_blake3': receipt.receipt_blake3, 'tool_result_receipt_blake3': observed[-1].tool_result['receipt_blake3'],
        'actor_workspace_before_blake3': before, 'actor_workspace_after_blake3': after, 'actor_workspace_changed': False,
        'messages': messages, 'loss_message_indices': [len(messages) - 2], 'loss_bearing_assistant_tool_decisions': 1,
        'assistant_tool_decisions': len(observed), 'host_tool_observations': len(observed),
        'offered_tool_catalog': catalog, 'offered_tool_catalog_blake3': blake3_hex(catalog),
        'codex_offered_tool_schema_blake3': receipt.offered_tool_schema_blake3, 'skill_delivery': source['skill_delivery'],
        'skill_frontiers': [{'name': item.tool_result['name'], 'frontier': item.tool_result['frontier'],
                            'parallel_group_id': item.tool_result['parallel_group_id']} for item in observed[:-1]],
        'skill_observations_used_as_context_only': True, 'target_host_observation_used_as_context': False,
        'host_evidence_visibility': 'verifier-only', 'hidden_reasoning_included': False,
        'private_reference_observations_in_training_messages': False, 'terminal_answer_included': False,
        'agent_judged': False, 'strict_full_trajectory_sft_eligible': False, **evidence})


def _outcome(bundle: Mapping[str, bytes], **kwargs: Any) -> tuple[Any, str | None]:
    try:
        return _slice(bundle, **kwargs), None
    except S3PilotSFTError as error:
        return None, str(error)
    except Exception as error:
        return None, f'malformed_or_unsupported_evidence:{type(error).__name__}'


def _public_input_root(authority_root: Path, source_id: str, modules: Any) -> Path:
    registry = _json(_read(authority_root / 'runs/evamed-campaign-supervisor-v24-attempt1/candidate-registry.v24.json'))
    candidates = [row for row in registry['candidates'] if row['candidate_id'] == source_id]
    _require(len(candidates) == 1, 'construction_candidate_locator')
    reference = candidates[0]['proofs']['promoted']['subject']
    subject_path = authority_root / _safe(reference['path'])
    _require(subject_path.resolve(strict=True).is_relative_to(authority_root.resolve()), 'construction_subject_boundary')
    subject_bytes = _read(subject_path)
    _require(modules.canonical.sha256_bytes(subject_bytes) == reference['sha256'], 'construction_subject_commitment')
    subject = _json(subject_bytes)
    references = [row['artifact'] for row in subject['source_evidence']
                  if row['role'] in {'construction_manifest', 'candidate_construction_manifest'}]
    _require(len(references) == 1, 'construction_input_locator')
    manifest_path = authority_root / _safe(references[0]['path'])
    _require(manifest_path.resolve(strict=True).is_relative_to(authority_root.resolve())
        and modules.canonical.sha256_bytes(_read(manifest_path)) == references[0]['sha256'], 'construction_locator_commitment')
    inputs = manifest_path.parent / 'construction-private/04-solver-input'
    _require(inputs.resolve(strict=True).is_relative_to(authority_root.resolve()), 'public_input_root_boundary')
    return inputs


def _collect(source_path: Path, source_root: Path, record: Mapping[str, Any], runtime_root: Path,
             authority_root: Path, modules: Any) -> dict[str, bytes]:
    source_bytes = _read(source_path)
    source = _json(source_bytes)
    workspace = source_root / 'workspaces' / source['task_id'] / source['sandbox_id']
    _require(workspace.resolve(strict=True).is_relative_to(source_root.resolve()), 'source_workspace_path')
    identity = blake3_bytes(str(workspace.resolve()).encode())[:32]
    hosts = tuple(runtime_root.glob(f'*/workspace-{identity}'))
    _require(len(hosts) == 1 and not hosts[0].is_symlink(), 'unique_host_session')
    host = hosts[0]
    bundle = {'source.json': source_bytes, 'bulk.json': canonical_json_bytes(record),
        'native-route.json': _read(source_root / 'native-route-receipts' / (source['task_id'] + '.json'))}
    # Preserve every actual attempt: the validator rejects anything but one S3 attempt.
    attempts = tuple((host / 'executions').glob('*'))
    _require(len(attempts) == 1 and attempts[0].name == 's3-attempt-1', 'single_host_execution_attempt')
    for base, prefix in [(host / 'stages', 'prerequisites'), (attempts[0], 'host')]:
        for path in base.rglob('*'):
            if path.is_file():
                name = path.relative_to(base).as_posix()
                if prefix == 'host' and name == 'submission/solution.py':
                    name = 'submission.py'
                bundle[prefix + '/' + _safe(name)] = _read(path)
    # /inputs is a separate actor-permitted mount, not part of the Codex workspace.
    # The index is only a locator: each byte is bound to the signed execution contract below.
    source_id = source['messages'][1]['content']['execution_binding']['source_candidate_id']
    inputs = _public_input_root(authority_root, source_id, modules)
    contract = _json(bundle['host/execution-contract.json'])
    for entry in contract['inputs']:
        relative = _safe(entry['relative_path'])
        path = inputs / relative
        _require(path.resolve(strict=True).is_relative_to(inputs.resolve()), 'public_input_mount_boundary')
        data = _read(path)
        _require(len(data) == entry['byte_count'] and modules.canonical.sha256_bytes(data) == entry['sha256'], 'public_input_mount_bytes')
        bundle['inputs/' + relative] = data
    # Live actor files are checked before copying; subsequent verification uses the retained snapshots.
    _, files = _workspace_snapshot_files(source['workspace_after'], expected_label='after-rollout')
    _require({p.relative_to(workspace).as_posix() for p in workspace.rglob('*') if p.is_file()} == set(files)
        and all(_read(workspace / name) == data for name, data in files.items()), 'live_actor_workspace_bytes')
    return bundle


def _resources(bulk_root: Path, ids: Sequence[str], trust_store: Path, legacy_python_root: Path) -> tuple[Any, Any, Any]:
    envelope = SignedEnvelope.from_document(_json(_read(bulk_root / 'manifest.json')))
    verify_signed_envelope(envelope, trust_store_path=trust_store)
    return _load_bulk_records(bulk_root, ids), _load_legacy_modules(legacy_python_root), _json(_read(trust_store))['keys']


def build_s3_pilot_dataset(*, source_roots: Sequence[Path], output_root: Path, bulk_root: Path,
                         registry: Any, trust_store: Path, legacy_python_root: Path, runtime_root: Path) -> Mapping[str, Any]:
    _require(not output_root.exists(), 'output_already_exists')
    entries = [(root.resolve(), p) for root in source_roots for p in sorted(root.glob('trajectories/*/*/result.json'))]
    ids = [_json(_read(path))['sandbox_id'] for _, path in entries]
    _require(bool(ids) and len(ids) == len(set(ids)), 'empty_or_duplicate_source_pairs')
    bulk, modules, trust = _resources(bulk_root, ids, trust_store, legacy_python_root)
    output_root.mkdir(parents=True, mode=0o700)
    rows, inventory = [], []
    for (source_root, source_path), sandbox_id in zip(entries, ids, strict=True):
        bundle = _collect(source_path, source_root, bulk[sandbox_id][0], runtime_root, legacy_python_root.resolve().parent, modules)
        row, reason = _outcome(bundle, bulk=bulk[sandbox_id], registry=registry, modules=modules, trust=trust)
        prefix = 'evidence/' + str(uuid4())
        files = []
        for name, data in sorted(bundle.items()):
            target = output_root / prefix / _safe(name)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with target.open('xb') as stream:
                stream.write(data)
            files.append({'path': name, 'byte_count': len(data), 'file_blake3': blake3_bytes(data)})
        entry = {'sandbox_id': sandbox_id, 'evidence_root': prefix, 'files': files, 'rejection_reason': reason}
        if row is not None:
            row = {**row, 'slice_id': str(uuid4()), 'evidence_root': prefix}
            row['slice_blake3'] = blake3_hex(row)
            rows.append(row)
            entry['slice_blake3'] = row['slice_blake3']
        inventory.append(entry)
    payload = _jsonl(rows)
    with (output_root / 'slices.jsonl').open('xb') as stream:
        stream.write(payload)
    manifest = {'schema': DATASET_SCHEMA, 'dataset_id': str(uuid4()), 'source_count': len(entries),
        'slice_count': len(rows), 'rejected_count': len(entries) - len(rows), 'evidence': inventory,
        'slice_file': {'path': 'slices.jsonl', 'file_blake3': blake3_bytes(payload), 'byte_count': len(payload)},
        'registry_digest': registry.digest, 'trust_store_blake3': blake3_bytes(_read(trust_store)),
        'legacy_runtime_source_blake3': modules.source_blake3, 'host_evidence_visibility': 'verifier-only',
        'verification_requires_original_bulk': True, 'live_host_workspace_required': False, 'agent_judged': False}
    manifest['manifest_blake3'] = blake3_hex(manifest)
    with (output_root / 'manifest.json').open('xb') as stream:
        stream.write(canonical_json_bytes(manifest))
    return {k: manifest[k] for k in ('dataset_id', 'source_count', 'slice_count', 'rejected_count', 'manifest_blake3')}


def verify_s3_pilot_dataset(*, dataset_root: Path, bulk_root: Path, registry: Any,
                          trust_store: Path, legacy_python_root: Path) -> Mapping[str, Any]:
    manifest = _json(_read(dataset_root / 'manifest.json'))
    _require(manifest['schema'] == DATASET_SCHEMA and manifest['manifest_blake3'] == blake3_hex(
        {k: v for k, v in manifest.items() if k != 'manifest_blake3'}), 'manifest_commitment')
    _uuid(manifest['dataset_id'], label='S3 dataset_id')
    entries = manifest['evidence']
    ids = [entry['sandbox_id'] for entry in entries]
    bulk, modules, trust = _resources(bulk_root, ids, trust_store, legacy_python_root)
    _require(manifest['registry_digest'] == registry.digest and manifest['trust_store_blake3'] == blake3_bytes(_read(trust_store)), 'trusted_verifier_context')
    descriptor = manifest['slice_file']
    raw = _read(dataset_root / _safe(descriptor['path']))
    _require(len(raw) == descriptor['byte_count'] and blake3_bytes(raw) == descriptor['file_blake3'], 'slice_file_commitment')
    rows = [_json(line) for line in raw.splitlines()]
    expected_files = {'manifest.json', descriptor['path']}
    accepted, rejected = 0, Counter()
    for entry in entries:
        prefix = _safe(entry['evidence_root'])
        bundle = {}
        for item in entry['files']:
            name = _safe(item['path'])
            path = dataset_root / prefix / name
            _require(path.resolve(strict=True).is_relative_to(dataset_root.resolve()), 'portable_evidence_boundary')
            data = _read(path)
            _require(name not in bundle and len(data) == item['byte_count'] and blake3_bytes(data) == item['file_blake3'], 'portable_evidence_commitment')
            bundle[name] = data
            expected_files.add(prefix + '/' + name)
        row, reason = _outcome(bundle, bulk=bulk[entry['sandbox_id']], registry=registry, modules=modules, trust=trust)
        _require(reason == entry['rejection_reason'], 'rejection_recomputation')
        if row is None:
            rejected[str(reason)] += 1
            continue
        actual = rows[accepted]
        _uuid(actual['slice_id'], label='S3 slice_id')
        expected = {**row, 'slice_id': actual['slice_id'], 'evidence_root': prefix}
        expected['slice_blake3'] = blake3_hex(expected)
        _require(actual == expected and entry['slice_blake3'] == expected['slice_blake3'], 'slice_recomputation')
        accepted += 1
    _require({p.relative_to(dataset_root).as_posix() for p in dataset_root.rglob('*') if p.is_file()} == expected_files, 'dataset_file_inventory')
    _require(accepted == len(rows) == manifest['slice_count'] and len(entries) == manifest['source_count']
        and sum(rejected.values()) == manifest['rejected_count'], 'dataset_counts')
    return {'schema': VERIFY_SCHEMA, 'valid': True, 'source_count': len(entries), 'slice_count': accepted,
        'rejections': dict(rejected), 'manifest_blake3': manifest['manifest_blake3'], 'agent_judged': False,
        'current_legacy_source_matches_build': manifest['legacy_runtime_source_blake3'] == modules.source_blake3,
        'provider_calls': 0, 'authored_code_reexecuted': False, 'live_host_workspace_required': False}
