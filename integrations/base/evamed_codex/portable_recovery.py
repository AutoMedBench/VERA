"""Post hoc verification of retained empty native resource-template discovery.

No actor execution, API calls, source mutations, or historical signature claims.
The new admission states which originally missing per-turn snapshot is a copy
of the already retained and originally signed final workspace observation.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from .portable_rollout import BUNDLE, ROOT, joined_calls
from evamed_portable.bundle import Bundle
from evamed_portable.integrity import (Signer, contained_file, digest, file_digest, strict_json,
    timestamp, verify_receipt, write_json)
from eva_agent.pipeline.digests import canonical_value

SCHEMA = 'eva.medresearch-empty-native-template-recovery.v1'
EPISODE_SCHEMA = 'eva.medresearch-recovered-Codex-episode.v1'
TRAJECTORY_SCHEMA = 'eva.medresearch-recovered-Codex-trajectory.v1'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def document(path):
    value = strict_json(path.read_bytes())
    require(value['document_blake3'] == digest({k:v for k,v in value.items() if k != 'document_blake3'}),
        'recovery_source_document_changed')
    return value


def joined_with_empty_templates(receipt, audit, event_slice=None):
    """Original full-response joins plus exactly one additional empty control."""
    templates, retained = {}, []
    for call in receipt.tool_calls:
        if call.mcp_server != 'codex' or call.mcp_tool != 'list_mcp_resource_templates':
            retained.append(call)
            continue
        output = canonical_value(call.output)
        native = output.get('result') or {}
        text = native.get('content', [])
        require(call.tool_type == 'mcpToolCall' and call.status == 'completed' and
            canonical_value(call.arguments) == {} and output.get('error') is None and
            not (set(native) - {'_meta', 'content', 'structuredContent', 'isError'}) and
            native.get('_meta') is None and native.get('structuredContent') is None and
            native.get('isError', False) is False and len(text) == 1 and set(text[0]) == {'type', 'text'} and
            text[0]['type'] == 'text' and strict_json(text[0]['text']) == {'resourceTemplates': []},
            'recovery_native_template_control_not_exact_empty')
        require(call.tool_call_id not in templates, 'duplicate_native_template_call')
        templates[call.tool_call_id] = {'kind':'native_empty_resource_template_discovery',
            'codex_tool_call_id':call.tool_call_id,'codex_tool_receipt_blake3':call.receipt_blake3,
            'actual_host_effect':False}
    joined = {row['codex_tool_call_id']:row for row in joined_calls(SimpleNamespace(tool_calls=retained), audit, event_slice)}
    require(not set(joined).intersection(templates), 'duplicate_recovery_call_identity')
    joined.update(templates)
    return [joined[call.tool_call_id] for call in receipt.tool_calls]


def verify_recovery_view(view, trajectory, public_key):
    signed = strict_json(contained_file(view, 'recovery-admission.json').read_bytes())
    admission = verify_receipt(signed, public_key)
    require(admission['schema'] == SCHEMA and admission['recovered_trajectory_blake3'] == trajectory['document_blake3'] and
        admission['recovered_episode_blake3'] == trajectory['episode_receipt_blake3'] and
        trajectory['schema'] == TRAJECTORY_SCHEMA and admission['actor_replayed'] is False and
        admission['original_source_files_modified'] is False and admission['historical_or_prelaunch_signature_fabricated'] is False,
        'recovery_admission_changed')
    original = ROOT / admission['original_actor_relative_path']
    original = original.resolve(strict=True)
    require(original.is_relative_to(ROOT) and original != view and
        admission['original_trajectory_blake3'] == document(contained_file(original, 'trajectory.json'))['document_blake3'],
        'recovery_original_source_changed')
    paths = set()
    for row in admission['source_files']:
        require(row['view_relative_path'] not in paths, 'duplicate_recovery_view_source')
        paths.add(row['view_relative_path'])
        source = contained_file(original, row['original_relative_path'])
        copied = contained_file(view, row['view_relative_path'])
        require(source.stat().st_size == copied.stat().st_size == row['bytes'] and
            file_digest(source) == file_digest(copied) == row['blake3'], 'recovery_original_bytes_changed')
    original_trajectory = document(contained_file(view, 'original/trajectory.json'))
    original_signed = strict_json(contained_file(view, 'original/episode-receipt.json').read_bytes())
    original_episode = verify_receipt(original_signed, public_key)
    require(digest(original_signed) == original_trajectory['episode_receipt_blake3'] == admission['original_episode_blake3'] and
        original_trajectory['errors'] == trajectory['errors'] == admission['original_capture_errors'] and
        original_episode['schema'] == 'eva.medresearch-authenticated-Codex-episode.v1' and all(
            original_episode[key] == trajectory[key] for key in ('run_id','case_id','bundle_blake3','source_record_blake3',
                'source_rubric_digest','before_snapshot_blake3','after_snapshot_blake3','host_final_receipt_blake3','model','route',
                'actual_model_requests','actual_tool_attempts')), 'recovery_original_authority_changed')
    require(admission['formal_submission_boundary_observed'] is True and
        trajectory['control_stop_reason'] == 'original_capture_rejection_after_completed_native_turn',
        'recovery_not_formal_terminal_observation')
    return {**admission, 'admission_blake3':digest(signed)}


def recover_empty_templates(actor, output, key):
    from eva_agent.codex_runtime import codex_turn_receipt_from_document, verify_codex_turn_receipt
    from .portable_judge import prepare_trajectory
    actor, output = Path(actor).resolve(strict=True), Path(output).resolve()
    require(actor.is_relative_to(ROOT) and output.is_relative_to(ROOT) and not output.is_relative_to(actor),
        'recovery_paths_outside_workspace')
    # Frozen cohort completion is mandatory: never alter an active cohort's admission.
    require((actor.parent.parent / 'cohort.json').is_file(), 'frozen_cohort_not_finished')
    original = document(contained_file(actor, 'trajectory.json'))
    require(original['schema'] == 'eva.medresearch-real-codex-trajectory.v1' and original['errors'] and
        all(row['type'] == 'ValueError' and row['frames'][-1]['function'] == 'joined_calls' and
            row['frames'][-1]['file'] == 'portable_rollout.py' for row in original['errors']) and
        original['codex_turn_status'] == 'completed' and original['policy_budget_terminal'] is False,
        'original_failure_not_bounded_native_join_rejection')
    bundle = Bundle(BUNDLE); public_key = bundle.manifest['fresh_execution_authority']['public_key_base64']
    signer = Signer(Path(key).resolve(strict=True)); require(signer.public == public_key, 'recovery_signer_not_bundle_authority')
    old_signed = strict_json(contained_file(actor, 'episode-receipt.json').read_bytes())
    old_episode = verify_receipt(old_signed, public_key)
    require(digest(old_signed) == original['episode_receipt_blake3'], 'original_episode_commitment_changed')
    final = document(contained_file(actor, 'after.json'))
    require(final['document_blake3'] == old_episode['after_snapshot_blake3'] and
        final['state']['submission_attempted'] is True, 'capture_rejection_before_formal_submission_not_recoverable')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    view = output / 'view'; view.mkdir(mode=0o700)
    source_files, sources = [], set()
    def copy(source_name, target_name=None):
        target_name = target_name or source_name
        require(target_name not in sources, 'duplicate_recovery_copy')
        sources.add(target_name)
        src = contained_file(actor, source_name); dst = view / target_name
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        raw = src.read_bytes(); dst.write_bytes(raw); dst.chmod(0o600)
        source_files.append({'original_relative_path':source_name,'view_relative_path':target_name,
            'bytes':len(raw),'blake3':file_digest(src)})
    for name in ('host-tool-config.json','before.json','after.json','host-final-disposition.json','mcp-events.jsonl'):
        copy(name)
    copy('trajectory.json', 'original/trajectory.json'); copy('episode-receipt.json', 'original/episode-receipt.json')
    for directory in ('snapshots', 'public-file-blobs', 'turns'):
        for path in sorted((actor / directory).rglob('*')):
            if path.is_file(): copy(str(path.relative_to(actor)))
    events = [strict_json(line) for line in contained_file(actor,'mcp-events.jsonl').read_bytes().splitlines()]
    turn_dirs = sorted((actor/'turns').iterdir())
    require(turn_dirs and len(turn_dirs) <= 6, 'recovery_turn_inventory_invalid')
    turns, joins, substitutions, offset, template_count = [], [], [], 0, 0
    for ordinal, directory in enumerate(turn_dirs, 1):
        require(directory.name == f'{ordinal:02d}', 'recovery_native_turn_gap')
        relative = str(directory.relative_to(actor))
        receipt = codex_turn_receipt_from_document(strict_json(contained_file(directory, 'codex-receipt.json').read_bytes()))
        verify_codex_turn_receipt(receipt)
        require(receipt.status == 'completed', 'recovery_native_turn_not_completed')
        launch_signed = strict_json(contained_file(directory, 'launch.json').read_bytes())
        launch = verify_receipt(launch_signed, public_key)
        require(launch['native_turn'] == ordinal and launch['run_id'] == original['run_id'], 'recovery_launch_identity_changed')
        end = offset + sum(call.mcp_server == 'eva_medresearch' for call in receipt.tool_calls)
        joined = joined_with_empty_templates(receipt, actor, (offset, end))
        template_count += sum(row.get('kind') == 'native_empty_resource_template_discovery' for row in joined)
        before = document(contained_file(directory, 'before.json'))
        if (directory/'after.json').is_file():
            after = document(contained_file(directory, 'after.json'))
        else:
            require(ordinal == len(turn_dirs), 'nonfinal_native_turn_after_missing')
            copy('after.json', relative + '/after.json'); after = final
            substitutions.append({'view_relative_path':relative+'/after.json', 'source_relative_path':'after.json',
                'document_blake3':final['document_blake3'],
                'meaning':'Copy of the actual retained final observation originally bound by the signed episode; original per-turn after file was absent.'})
            for path in sorted((actor/'public-file-blobs').glob('*')):
                target = relative + '/public-file-blobs/' + path.name
                if target not in sources:copy(str(path.relative_to(actor)), target)
        turns.append({'relative_directory':relative,'receipt_blake3':receipt.receipt_blake3,
            'launch_receipt_blake3':digest(launch_signed),'joined_host_results':joined,
            'host_event_offset':offset,'host_event_end':end,'before_snapshot_blake3':before['document_blake3'],
            'after_snapshot_blake3':after['document_blake3'],'native_thread_id':receipt.thread_id,
            'status':receipt.status,'host_conditioning_loss_weight':0})
        joins.extend(joined); offset=end
    require(template_count > 0 and offset == len(events) and turns[:len(old_episode['turns'])] == old_episode['turns'] and
        turns[-1]['receipt_blake3'] == original['codex_turn_receipt_blake3'] and
        turns[-1]['launch_receipt_blake3'] == original['launch_receipt_blake3'], 'recovery_original_turn_binding_changed')
    recovered_episode = {**old_episode, 'schema':EPISODE_SCHEMA, 'turns':turns,'all_joined_host_results':joins,
        'control_stop_reason':'original_capture_rejection_after_completed_native_turn',
        'original_episode_blake3':original['episode_receipt_blake3'], 'original_capture_errors':original['errors'],
        'new_admission_time':timestamp(),'original_signatures_replaced':False,'actor_replayed':False}
    recovered_signed = signer.sign(recovered_episode); write_json(view/'episode-receipt.json', recovered_signed)
    recovered = {**original,'schema':TRAJECTORY_SCHEMA,'codex_turns':turns,'joined_host_results':joins,
        'actual_native_turn_count':len(turns),'episode_receipt_blake3':digest(recovered_signed),
        'control_stop_reason':recovered_episode['control_stop_reason'],'diagnostic_infrastructure_passed':True,
        'original_capture_rejection_preserved':True,'original_trajectory_blake3':original['document_blake3']}
    recovered['document_blake3'] = digest({k:v for k,v in recovered.items() if k != 'document_blake3'})
    write_json(view/'trajectory.json', recovered)
    admission = {'schema':SCHEMA,'recovery_id':str(uuid4()),'at':timestamp(),
        'original_actor_relative_path':str(actor.relative_to(ROOT)), 'original_trajectory_blake3':original['document_blake3'],
        'original_episode_blake3':original['episode_receipt_blake3'], 'original_capture_errors':original['errors'],
        'recovered_trajectory_blake3':recovered['document_blake3'],'recovered_episode_blake3':digest(recovered_signed),
        'source_files':source_files,'snapshot_substitutions':substitutions,
        'exact_empty_template_controls':template_count,'formal_submission_boundary_observed':True,
        'source_code_blake3':file_digest(__file__),'actor_replayed':False,'original_source_files_modified':False,
        'historical_or_prelaunch_signature_fabricated':False,
        'receipt_admission_scope':'New post hoc admission of retained native receipts; no claim that the original rejected aggregate included the missing turn.'}
    write_json(view/'recovery-admission.json', signer.sign(admission))
    # Admission is usable only if the ordinary all-turn verifier independently accepts every source/effect binding.
    prepare_trajectory(view, output/'verified-projection')
    write_json(output/'recovery-complete.json', signer.sign({'schema':'eva.medresearch-native-discovery-recovery-complete.v1',
        'recovered_view':str(view.relative_to(ROOT)), 'recovery_admission_blake3':digest(strict_json((view/'recovery-admission.json').read_bytes())),
        'actual_source_rubric_projection_verified':True,'providers_called':0,'actor_replayed':False}))
    return view
