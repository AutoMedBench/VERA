"""Carry a closed, unpromoted parent round into a separate task-version state."""
from pathlib import Path

from .feedback import ROOT,BUNDLE,committed,require
from .trial_runner import journal
from evamed_portable.integrity import digest,file_digest

NAMESPACE='task-v2f'
PARENT_NAMESPACE='task-v2e2'
ANCESTOR_NAMESPACE='task-v2d'
ANCESTOR_BUNDLE='04486fa23fedf7db989255dfd4e6c76f96fc74a5580eb9abcda327412738ee16'
PARENT_BUNDLE='9c6a305afd3dd30bf7c9f35a33f406b3f71a78243d6c0735f3ef8717f8daa264'
CHILD_BUNDLE='8660156f46c17422a26cf7aaadc4a3e6eefa1eb9f1e4ebd372e14798e2031502'
PARENT_CONFIG=ROOT/'evamed-codex/training/direct-grpo-12nodes-v39-public-feedback-recovery/harness-driver-config.json'


def seal(value):return {**value,'document_blake3':digest(value)}


def reference_plan(parent):
    """Exact additive cohort derivation: keep fourteen groups and add two."""
    from copy import deepcopy
    import hashlib
    from common import INDEX
    from task_splits import NEW_PLAN_SHA256, training_ids
    from .trial_plan import SEED
    manifest=committed(BUNDLE/'bundle.json')
    require(manifest['document_blake3']==CHILD_BUNDLE and len(parent['groups'])==14,
            'harder_reference_parent_or_bundle_changed')
    index={row['label']:row for row in map(__import__('json').loads,INDEX.read_text().splitlines())}
    result=deepcopy(parent)
    result.pop('document_blake3')
    for group in result['groups']:
        group['metadata']=index[group['case_id']]['metadata']
    eligible=training_ids()
    cases=[c for c in manifest['cases'] if c['domain']=='synthetic-annotation-migration'
           and c['stage']=='E2E' and c['case_id'] in eligible]
    cases.sort(key=lambda c:hashlib.sha256(f'{SEED}:{c["case_id"]}'.encode()).hexdigest())
    for case in cases[:2]:
        ordinal=len(result['groups'])
        result['groups'].append({'group_index':ordinal,'case_id':case['case_id'],'domain':case['domain'],
            'fixture_family':None,'metadata':index[case['case_id']]['metadata'],
            'evidence_blake3':[e['blake3'] for e in case['evidence']],
            'requested_sampling_seeds':[SEED+ordinal*100+k for k in range(8)]})
    require(len(result['groups'])==16,'harder_reference_cases_missing')
    result.update(bundle_blake3=CHILD_BUNDLE,parent_task_version_plan_blake3=parent['document_blake3'],
        harder_split_plan_sha256=NEW_PLAN_SHA256,original_trajectories_per_policy=256,
        task_version_transition={'policy':'eva.additive-harder-e2e.v1','parent_bundle_blake3':PARENT_BUNDLE,
            'bundle_blake3':CHILD_BUNDLE,'historical_observations_reused':False},
        scope='Sixteen-group training-development cohort: fourteen preserved original groups and two fresh harder groups; no clinical or held-out evaluation claim.')
    return seal(result)


def validate(config):
    transition=config['task_version_transition']
    require(transition=={'schema':'eva.closed-round-task-version-transition.v1',
        'state_subdirectory':NAMESPACE,'prior_config_path':str(PARENT_CONFIG),
        'prior_config_blake3':committed(PARENT_CONFIG)['document_blake3'],
        'prior_bundle_blake3':PARENT_BUNDLE,'bundle_blake3':CHILD_BUNDLE},
        'task_transition_configuration_changed')
    require(committed(BUNDLE/'bundle.json')['document_blake3']==CHILD_BUNDLE,
            'task_transition_bundle_changed')
    old=committed(committed(PARENT_CONFIG)['reference_plan_path'])
    new=committed(config['reference_plan_path'])
    require(old['bundle_blake3']==PARENT_BUNDLE and new['bundle_blake3']==CHILD_BUNDLE and
            new.get('parent_task_version_plan_blake3')==old['document_blake3'],
            'task_transition_reference_plan_changed')
    require(new==reference_plan(old),'task_transition_changed_comparison_design')
    return transition


def latest(path,pattern):
    files=sorted(path.glob(pattern));require(files,'task_transition_parent_history_missing')
    return files[-1]


def ancestor_evidence(legacy_root):
    """Authenticate the retained v2d ancestor using the existing v2e2 transfer.

    A directory name alone is insufficient. Check the transfer receipt, exact
    ancestor endpoint paths, frozen endpoint positions and every committed input.
    This function is read-only and never imports old comparison observations.
    """
    pointer=legacy_root/PARENT_NAMESPACE/'transition.json'
    previous=committed(pointer)
    require(previous['schema']=='eva.harness-task-version-state-transfer.v1' and
            previous['bundle_blake3']==PARENT_BUNDLE and
            previous['parent_bundle_blake3']==ANCESTOR_BUNDLE and
            previous['old_comparison_reinterpreted'] is False and
            previous['new_comparison_executed'] is False,
            'task_transition_ancestor_lineage_changed')
    ancestor=legacy_root/ANCESTOR_NAMESPACE
    intent=latest(ancestor,'round-*/intent.json')
    choice=latest(ancestor/'choices','rollout-*.json')
    require(str(intent)==previous['parent_intent_path'] and
            str(choice)==previous['parent_choice_path'],
            'task_transition_ancestor_advanced')
    files={str(pointer):file_digest(pointer)}
    required={str(ancestor/'config.json'),str(intent),str(intent.parent/'closed.json'),str(choice)}
    require(required<=set(previous['evidence_files']),
            'task_transition_ancestor_evidence_incomplete')
    for path,expected in previous['evidence_files'].items():
        source=Path(path).resolve(strict=True)
        require(source.is_relative_to(ROOT) and file_digest(source)==expected,
                'task_transition_ancestor_evidence_changed')
        files[str(source)]=expected
    return files


def initialize(legacy_root,rollout_id,identity,config):
    """Called by the synchronous rollout owner, never by an external watcher.

    The imported anchor is explicitly closed parent history, not a comparison
    under the new bundle. Writes are additive immutable journals and resumable.
    """
    legacy_root=Path(legacy_root).resolve()
    require(legacy_root.is_relative_to(ROOT) and legacy_root.name=='driver',
            'task_transition_root_outside_driver')
    require(not [p for p in legacy_root.glob('task-v2*') if p.name not in (NAMESPACE,PARENT_NAMESPACE,ANCESTOR_NAMESPACE,'task-v2c')],
            'task_transition_alternate_child_already_initialized')
    transition=validate(config)
    ancestors=ancestor_evidence(legacy_root)
    root=legacy_root/NAMESPACE
    pointer=root/'transition.json'
    parent_root=legacy_root/PARENT_NAMESPACE
    parent_intent_path=latest(parent_root,'round-*/intent.json')
    parent_choice_path=latest(parent_root/'choices','rollout-*.json')
    if pointer.exists():
        saved=committed(pointer)
        require(saved['config_blake3']==config['document_blake3'] and
                saved['bundle_blake3']==CHILD_BUNDLE and rollout_id>=saved['first_rollout_id'],
                'task_transition_resume_identity_changed')
        require(str(parent_intent_path)==saved['parent_intent_path'] and
                str(parent_choice_path)==saved['parent_choice_path'],'parent_driver_advanced_after_transition')
        for path,expected in saved['evidence_files'].items():
            source=Path(path).resolve(strict=True)
            require(source.is_relative_to(ROOT) and file_digest(source)==expected,
                    'task_transition_parent_evidence_changed')
        if rollout_id==saved['first_rollout_id']:
            require(identity==saved['checkpoint_identity'],'task_transition_checkpoint_changed')
    else:
        binding=committed(parent_root/'config.json')
        require(binding['config_path']==transition['prior_config_path'] and
                binding['config_blake3']==transition['prior_config_blake3'],
                'task_transition_parent_config_changed')
        parent_intent=committed(parent_intent_path)
        closed_path=parent_intent_path.parent/'closed.json'
        require(closed_path.is_file(),'task_transition_requires_closed_parent_round')
        closed=committed(closed_path);previous=committed(parent_choice_path)
        require(previous['rollout_id']<=rollout_id and
                previous['checkpoint_identity']['completed_optimizer_updates']==previous['rollout_id'] and
                identity['completed_optimizer_updates']==rollout_id,
                'task_transition_policy_boundary_changed')
        require(parent_intent['config_blake3']==previous['config_blake3']==binding['config_blake3'] and
                parent_intent['start_rollout_id']<=closed['closed_rollout_id']<=rollout_id,
                'task_transition_parent_round_binding_changed')
        require(closed['candidate_promoted'] is False and
                previous['choice']['candidate_path'] is previous['choice']['candidate_blake3'] is None,
                'task_transition_requires_unpromoted_baseline')
        from .cycle import validate_spec
        require(validate_spec(config['seed_spec_path'])['baseline_candidate_path'] is None,
                'task_transition_new_baseline_changed')
        files={str(p):file_digest(p) for p in
               (parent_root/'config.json',parent_intent_path,closed_path,parent_choice_path,PARENT_CONFIG)}
        files.update(ancestors)
        for path,expected in previous['evidence_files'].items():
            source=Path(path).resolve(strict=True)
            require(source.is_relative_to(ROOT) and file_digest(source)==expected,
                    'task_transition_parent_choice_evidence_changed')
            files[str(source)]=expected
        saved=seal({'schema':'eva.harness-task-version-state-transfer.v1',
            'first_rollout_id':rollout_id,'checkpoint_identity':identity,
            'config_blake3':config['document_blake3'],'bundle_blake3':CHILD_BUNDLE,
            'parent_bundle_blake3':PARENT_BUNDLE,'parent_intent_path':str(parent_intent_path),
            'parent_choice_path':str(parent_choice_path),'parent_round_intent':parent_intent,
            'parent_round_closed':closed,'parent_choice_blake3':previous['document_blake3'],
            'baseline_candidate':None,'evidence_files':files,
            'next_generation_due_rollout_id':parent_intent['start_rollout_id']+config['interval_optimizer_updates'],
            'old_comparison_reinterpreted':False,'new_comparison_executed':False,
            'sampler_modified':False,'optimizer_update_performed':False})
        journal(pointer,saved)
    prior=saved['parent_round_intent'];closed=saved['parent_round_closed']
    journal(root/'round-0000/intent.json',seal({'round_index':0,
        'start_rollout_id':prior['start_rollout_id'],'start_checkpoint_identity':prior['start_checkpoint_identity'],
        'baseline_candidate':None,'config_blake3':config['document_blake3'],'seed_cycle':False,
        'inherited_closed_parent_round':True,'task_transition_blake3':saved['document_blake3']}))
    journal(root/'round-0000/closed.json',seal({'reason':'inherited_closed_parent_task_round',
        'candidate_promoted':False,'closed_rollout_id':closed['closed_rollout_id'],
        'task_transition_blake3':saved['document_blake3'],'comparison_executed_for_current_bundle':False}))
    return root
