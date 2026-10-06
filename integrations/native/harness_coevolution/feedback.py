"""Export only authenticated, actor-public training feedback for harness search."""
import json
import os
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'EVA-Harness').is_dir() and (p/'evamed-codex').is_dir())
for relative in ('EVA-Harness/src', 'EVA-Harness', 'evamed-codex/execution-bundles/EVA-medresearch-v2f-harder-e2e-20260920/runtime'):
    path = str(ROOT/relative)
    if path not in sys.path:
        sys.path.append(path)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
# Preserve the short, workspace-backed container alias for checkpoint sockets.
if Path('/evamed-tmp').is_dir():
    if not os.path.samefile('/evamed-tmp', ROOT/'tmp'):
        raise ValueError('workspace_temp_mount_mismatch')
    os.environ['TMPDIR'] = '/evamed-tmp'
else:
    os.environ['TMPDIR'] = str(ROOT/'tmp')
from evamed_portable.integrity import digest, file_digest, strict_json, verify_receipt, write_json

BUNDLE = ROOT/'evamed-codex/execution-bundles/EVA-medresearch-v2f-harder-e2e-20260920'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(ROOT), 'feedback_outside_workspace')
    return strict_json(path.read_bytes())


def committed(path):
    value = read(path)
    require(value['document_blake3'] == digest({k:v for k,v in value.items() if k != 'document_blake3'}),
            'feedback_document_changed')
    return value


def require_training_source(record):
    split = record.get('split')
    if record.get('schema') == 'eva.fresh-synthetic-training-candidate-record.v1':
        require(record.get('domain') == 'synthetic-annotation-migration' and record.get('stage') == 'E2E',
                'synthetic_feedback_record_scope_changed')
        split = record['provenance']['split']
    require(split == 'train', 'held_out_source_forbidden')
    from task_splits import require_training_case
    require_training_case(record['sandbox_id'])


def actor_public(actor, manifest, *, probe_evidence=None, probe_context=None):
    actor = Path(actor).resolve(strict=True)
    tr = committed(actor/'trajectory.json')
    require(tr['focus_stage'] == 'E2E', 'only_actual_training_E2E_feedback')
    probe = None
    if probe_evidence is not None:
        from .feedback_probe import PURPOSE, verify_probe
        require(tr['purpose'] == PURPOSE, 'feedback_probe_purpose_changed')
        require(probe_context is not None, 'feedback_probe_boundary_required')
        probe = verify_probe(probe_evidence, expected_actor=actor, expected_context=probe_context)
    else:
        require(tr['purpose'] == 'slime-on-policy-training', 'only_actual_training_E2E_feedback')
    require(tr['bundle_blake3'] == manifest['document_blake3'], 'feedback_bundle_changed')
    descriptor = next(c for c in manifest['cases'] if c['case_id'] == tr['case_id'])
    record_path = BUNDLE/descriptor['record_path']
    require(file_digest(record_path) == descriptor['record_file_blake3'], 'feedback_source_file_changed')
    record = read(record_path)
    require_training_source(record)
    require(record['record_blake3'] == tr['source_record_blake3'], 'feedback_source_identity_changed')
    public_key = manifest['fresh_execution_authority']['public_key_base64']
    signed = read(actor/'episode-receipt.json')
    episode = verify_receipt(signed, public_key)
    require(digest(signed) == tr['episode_receipt_blake3'] and
            all(episode[k] == tr[k] for k in ('run_id', 'case_id', 'bundle_blake3', 'source_record_blake3')),
            'feedback_signed_episode_changed')
    before, after = committed(actor/'before.json'), committed(actor/'after.json')
    for name, snapshot in [('before', before), ('after', after)]:
        require(snapshot['visibility'] == 'actor-public' and snapshot['private_evaluator_data_included'] is False
                and snapshot['document_blake3'] == episode[name+'_snapshot_blake3'],
                'feedback_snapshot_not_authenticated_public')
    turn = actor/'turns/01'
    launch_envelope = read(turn/'launch.json')
    launch = verify_receipt(launch_envelope, public_key)
    require(digest(launch_envelope) == episode['turns'][0]['launch_receipt_blake3'], 'feedback_launch_changed')
    request = read(turn/'logical-request.json')
    logical = request['logical_input']
    require(request['private_evaluator_data_included'] is False and logical['judge_only_context'] is None
            and digest(logical) == launch['logical_input_blake3']
            and digest(request['public_tool_catalog']) == launch['public_tool_catalog_blake3'],
            'feedback_context_not_authenticated_public')
    from evamed_portable.integrity import byte_digest
    require(byte_digest(request['developer_instructions'].encode()) == launch['developer_instructions_blake3'],
            'feedback_instructions_changed')
    allowed = {x['host_event_id']:x['host_event_blake3'] for x in episode['all_joined_host_results'] if 'host_event_id' in x}
    events = [strict_json(line) for line in (actor/'mcp-events.jsonl').read_bytes().splitlines()]
    require(len(events) == episode['actual_tool_attempts'] == len(allowed), 'feedback_event_coverage_changed')
    public_events = []
    for event in events:
        require(event['event_blake3'] == allowed.get(event['event_id']) ==
                digest({k:v for k,v in event.items() if k != 'event_blake3'}), 'feedback_event_changed')
        public_events.append({k:event[k] for k in ('event_id', 'name', 'arguments', 'public_result')})
    skills = []
    require(logical['skills'] == launch['selected_skills'], 'feedback_skill_binding_changed')
    for skill in logical['skills']:
        path = Path(skill['path']).resolve(strict=True)
        require(path.is_relative_to(ROOT) and file_digest(path) == skill['content_blake3'], 'feedback_skill_changed')
        skills.append({'skill_id':skill['skill_id'], 'content':path.read_text(), 'content_blake3':skill['content_blake3']})
    # No judge source, assessment, reward, hidden construction, or evaluator report
    # is copied. These are the exact task text and tool outputs the actor saw.
    profile = episode['execution_profile']
    require(profile['reasoning_effort'] == 'medium' and profile['output_tokens'] == 16384 and
            profile['enable_thinking'] is True, 'feedback_native_execution_profile_changed')
    return {'domain':tr['domain'], 'case_id':tr['case_id'], 'run_id':tr['run_id'],
            'feedback_origin':'native_missing_family_probe' if probe else 'saved_production_trajectory',
            'execution_profile':profile,
            'initial_public_text':logical['public_text'], 'tool_events':public_events,
            'final_public_state':after['state'], 'baseline_developer_instructions':request['developer_instructions'],
            'public_tool_catalog':request['public_tool_catalog'], 'selected_skills':skills}, {
            'actor':str(actor), 'source_split':'train', 'episode_receipt_blake3':digest(signed),
            'record_file_blake3':descriptor['record_file_blake3'], 'actual_actor_model':episode['model'],
            'probe_observation':str(probe_evidence) if probe else None,
            'probe_observation_blake3':probe['document_blake3'] if probe else None,
            'probe_optimizer_input':False if probe else None}


def export(actors, output, *, probe_evidence=None, probe_context=None):
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT), 'feedback_output_outside_workspace')
    manifest = committed(BUNDLE/'bundle.json')
    probe_evidence = probe_evidence or {}
    require(set(probe_evidence) <= {str(Path(a).resolve(strict=True)) for a in actors},
            'unused_feedback_probe_evidence')
    rows, lineage = [], []
    for actor in actors:
        public, provenance = actor_public(actor, manifest,
            probe_evidence=probe_evidence.get(str(Path(actor).resolve(strict=True))), probe_context=probe_context)
        rows.append(public); lineage.append(provenance)
    require(len({r['run_id'] for r in rows}) == len(rows), 'duplicate_feedback_episode')
    packet = {'schema':'eva.harness-training-public-feedback.v1', 'visibility':'actor-public-training-only',
              'scope':'Candidate generation only; saved training episodes and explicitly identified native training-case probes are not a paired evaluation.',
              'evaluator_assessments_included':False, 'episodes':rows}
    packet['document_blake3'] = digest(packet)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    from .trial_runner import journal
    journal(output/'public-feedback.json',packet)
    journal(output/'private-lineage.json',{'public_feedback_blake3':packet['document_blake3'],'episodes':lineage})
    return packet
