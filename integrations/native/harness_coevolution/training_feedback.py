"""Index only saved training work, then select public feedback without scores."""
from pathlib import Path
from optimizer_topology import resolve_attempt, receipt_paths, valid_receipt, validate_feedback_index

from .feedback import ROOT, BUNDLE, committed, read, require, actor_public
from evamed_portable.integrity import digest, file_digest
from .trial_plan import DOMAINS, fixture_family
from .trial_runner import checkpoint_identity, journal
from .public_feedback_selector import Candidate, public_category, select as select_public_candidates


def register_previous_update(args, rollout_id):
    """Run immediately after a complete saved update, before the next rollout."""
    if rollout_id == 0:return None
    run = Path(args.save).resolve().parent
    output = run/'harness-coevolution/training-feedback-index'/f'update-{rollout_id:04d}.json'
    identity = checkpoint_identity(args,rollout_id)
    if output.exists():
        saved = committed(output)
        require(saved['checkpoint_identity'] == identity, 'feedback_saved_checkpoint_changed')
        for path, expected in saved['evidence_files'].items():
            require(file_digest(path) == expected, 'feedback_saved_update_evidence_changed')
        validate_feedback_index(ROOT, run, saved)
        return saved
    cutoff = (Path(args.save)/f'iter_{rollout_id-1:07d}'/'.metadata').stat().st_mtime_ns
    audits = list(run.glob(f'attempts/*/learning/rollout-{rollout_id-1:04d}.json'))
    audits += list(run.glob(f'learning/rollout-{rollout_id-1:04d}.json'))
    candidates = []
    for path in audits:
        audit = read(path)
        if 'production_batch_manifest' not in audit:continue # Historical v24/v25 did not record this binding.
        manifest_path = Path(audit['production_batch_manifest']).resolve(strict=True)
        require(manifest_path.is_relative_to(run/'rollouts') and
                file_digest(manifest_path) == audit['production_batch_manifest_blake3'], 'feedback_batch_index_changed')
        batch = committed(manifest_path)
        wanted = {int(k) for g in audit['groups'] for k in g['trajectory_segment_counts']}
        require(batch['rollout_id'] == rollout_id-1 == audit['rollout_id'] and
                batch['purpose'] == 'slime-on-policy-training' and batch['original_count'] == 64 and
                len(batch['actors']) == len(wanted) == 64 and
                {a['original_index'] for a in batch['actors']} == wanted, 'feedback_original_trajectory_coverage_changed')
        topology = resolve_attempt(ROOT, run, path.parent)
        ranks = receipt_paths(path.parent, rollout_id, topology)
        if not all(p.is_file() for p in ranks):continue
        latest = max(p.stat().st_mtime_ns for p in ranks)
        if latest > cutoff:continue
        success = True
        for rank, p in enumerate(ranks):
            value = read(p)
            success &= (valid_receipt(value, rank, rollout_id) and
                        value.get('production_batch_manifest_blake3') == audit['production_batch_manifest_blake3'])
        if success:candidates.append((latest,batch,[path,manifest_path,*ranks,*map(Path,topology['evidence_sha256'])],topology))
    if not candidates:return None
    _, batch, files, topology = max(candidates,key=lambda row:row[0])
    rows = []
    for actor in batch['actors']:
        path = Path(actor['actor']).resolve(strict=True)
        require(path.is_relative_to(run/'rollouts'), 'feedback_actor_outside_training_run')
        tr = committed(path/'trajectory.json')
        require(tr['case_id'] == actor['case_id'] and tr['domain'] == actor['domain'] and
                tr['focus_stage'] == actor['stage'] and tr['purpose'] == 'slime-on-policy-training',
                'feedback_actor_identity_changed')
        rows.append({**actor,'optimizer_update':rollout_id,'trajectory_blake3':tr['document_blake3'],'bundle_blake3':tr['bundle_blake3']})
        files.append(path/'trajectory.json')
    value = {'schema':'eva.saved-training-feedback-index.v1','optimizer_update':rollout_id,
             'checkpoint_identity':identity,'all_optimizer_ranks_verified':topology['expected_optimizer_ranks'],
             'optimizer_topology':topology,'actors':rows,
             'selection_uses_reward_values':False,'evidence_files':{str(p):file_digest(p) for p in files}}
    value['document_blake3'] = digest(value);journal(output,value)
    return value


def reserved_sources(plan, manifest):
    cases = {c['case_id']:c for c in manifest['cases']}
    ids = {g['case_id'] for g in plan['groups']}
    common = set(plan['shared_public_methods_blake3'])
    blobs = {e['blake3'] for case_id in ids for e in cases[case_id]['evidence']} - common
    families = {fixture_family(cases[i]) for i in ids if cases[i]['domain'] == 'healthbench-professional'}
    return ids,blobs,families,common


def select_feedback(run, through_update, reference_plan, *, window=20):
    """Authenticate the full native window, then rank only actor-public signals."""
    select_public_candidates([], through_update, window=window)
    run = Path(run).resolve();require(run.is_relative_to(ROOT), 'feedback_run_outside_workspace')
    manifest = committed(BUNDLE/'bundle.json')
    plan = committed(reference_plan)
    require(plan['bundle_blake3'] == manifest['document_blake3'], 'feedback_comparison_bundle_changed')
    excluded_ids,excluded_blobs,excluded_families,common = reserved_sources(plan,manifest)
    cases = {c['case_id']:c for c in manifest['cases']}
    candidates, rows_by_actor, sources, version_skips = [], {}, [], []
    for update in range(through_update,max(0,through_update-window),-1):
        path = run/'harness-coevolution/training-feedback-index'/f'update-{update:04d}.json'
        if not path.exists():continue
        index = committed(path)
        require(index['optimizer_update'] == update,
                'feedback_update_index_contract_changed')
        for source, expected in index['evidence_files'].items():
            require(file_digest(source) == expected, 'feedback_update_source_changed')
        validate_feedback_index(ROOT, run, index)
        sources.append({'path':str(path),'document_blake3':index['document_blake3']})
        for row in sorted(index['actors'],key=lambda item:item['original_index']):
            if row.get('bundle_blake3') != manifest['document_blake3']:
                version_skips.append({'source_index':str(path),'original_index':row['original_index'],
                    'case_id':row['case_id'],'bundle_blake3':row.get('bundle_blake3'),
                    'reason':'missing_task_version' if 'bundle_blake3' not in row else 'different_task_version'})
                continue
            domain, case_id = row['domain'],row['case_id']
            from task_splits import require_training_case
            require_training_case(case_id)
            if row['stage'] != 'E2E' or case_id in excluded_ids:continue
            case = cases[case_id]
            require(case['domain'] == domain and case['stage'] == 'E2E', 'feedback_case_identity_changed')
            blobs = {e['blake3'] for e in case['evidence']} - common
            if blobs & excluded_blobs or fixture_family(case) in excluded_families:continue
            actor = Path(row['actor']).resolve(strict=True)
            require(actor.is_relative_to(run/'rollouts'), 'feedback_selected_actor_outside_run')
            tr = committed(actor/'trajectory.json')
            require(tr['document_blake3'] == row['trajectory_blake3'] and tr['case_id'] == case_id and
                    tr['bundle_blake3'] == row['bundle_blake3'] == manifest['document_blake3'] and
                    tr['domain'] == domain and tr['focus_stage'] == 'E2E' and tr['purpose'] == 'slime-on-policy-training',
                    'feedback_selected_actor_changed')
            public, _ = actor_public(actor, manifest)
            require(public['feedback_origin'] == 'saved_production_trajectory' and
                    public['domain'] == domain and public['case_id'] == case_id,
                    'feedback_public_identity_changed')
            candidates.append(Candidate(domain, update, row['original_index'], case_id,
                                        str(actor), public_category(public)))
            rows_by_actor[str(actor)] = row
    selected = {candidate.domain:rows_by_actor[candidate.actor]
                for candidate in select_public_candidates(candidates, through_update, window=window)}
    return {'schema':'eva.harness-feedback-selection.v1','through_optimizer_update':through_update,
            'window_updates':window,'selection_uses_reward_values':False,
            'actors':[selected[d] for d in DOMAINS if d in selected],
            'missing_domains':[d for d in DOMAINS if d not in selected],
            'fixed_development_plan_blake3':plan['document_blake3'],'source_indices':sources,
            'excluded_task_version_rows':version_skips,
            'development_cases_and_case_specific_evidence_excluded':True,
            'healthbench_development_fixture_families_excluded':sorted(excluded_families)}
