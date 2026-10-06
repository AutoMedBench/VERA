"""Score-blind matched development cohort and immutable comparison contract."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from .feedback import ROOT, BUNDLE, committed, read, require, require_training_source
from evamed_portable.integrity import digest, file_digest, write_json

DOMAINS = ('agentclinic', 'automedbench-classification', 'automedbench-detection',
           'automedbench-research', 'automedbench-segmentation', 'healthbench-professional', 'medxpertqa',
           'synthetic-annotation-migration')
POLICY = 'eva.e2e-process50-completion50.v1'
SAMPLING_DESIGN = 'eva.native-stochastic-case-blocks.v1'
SEED = 20260917


def validate_training_plan(plan):
    from common import INDEX, sha
    from derived_task_admission import verify_evidence_bindings, ASSETS
    from evamed_portable.integrity import strict_json
    from task_splits import PLAN_SHA256, NEW_PLAN_SHA256, require_training_case
    manifest = committed(BUNDLE/'bundle.json')
    require(plan['bundle_blake3'] == manifest['document_blake3'] and
            plan.get('reserved_split_plan_sha256') == PLAN_SHA256 and
            plan.get('harder_split_plan_sha256') == NEW_PLAN_SHA256,
            'trial_task_version_or_reserved_split_changed')
    verify_evidence_bindings()
    raw = INDEX.read_bytes()
    require(sha(raw) == read(ASSETS/'asset-receipt.json')['index_sha256'], 'trial_index_bytes_changed')
    rows = [strict_json(line) for line in raw.splitlines()]
    index = {r['label']:r for r in rows}
    require(len(rows) == len(index) == 6080, 'trial_index_inventory_changed')
    require(Counter(group['domain'] for group in plan['groups']) == {domain:2 for domain in DOMAINS}
            and len({g['case_id'] for g in plan['groups']}) == 2*len(DOMAINS),
            'trial_family_coverage_changed')
    for group in plan['groups']:
        require_training_case(group['case_id'])
        require(group['metadata'] == index[group['case_id']]['metadata'] and
                group['metadata']['reserved_split'] == 'train', 'trial_effective_metadata_changed')


def fixture_family(case):
    """Group the released HealthBench synthetic variants by their E2E template."""
    if case['domain'] != 'healthbench-professional':
        return None
    names = [e['evidence_id'] for e in case['evidence'] if e['content_kind'] == 'synthetic']
    require(len(names) == 1, 'healthbench_fixture_identity_ambiguous')
    match = re.fullmatch(r'hbp\.fixture\.(e2e-\d+)-v\d+\.v1', names[0])
    require(match is not None, 'healthbench_fixture_family_unknown')
    return match[1]


def select_cases(manifest, generation_case_ids, count_per_domain=2):
    require(count_per_domain >= 2, 'trial_requires_multiple_cases_per_domain')
    from task_splits import training_ids
    eligible = training_ids()
    cases = [c for c in manifest['cases'] if c['stage'] == 'E2E' and c['case_id'] in eligible]
    generation = [c for c in cases if c['case_id'] in generation_case_ids]
    require(len(generation) == len(generation_case_ids), 'generation_case_not_in_training_bundle')
    frequency = Counter(e['blake3'] for c in cases for e in c['evidence'])
    # A shared public methods document is a dependence to disclose, not a reason
    # to drop HealthBench. Its role is identified by the exact released name.
    common_methods = {e['blake3'] for c in cases for e in c['evidence']
                      if e['evidence_id'] == 'hbp.professional.methods.v1'}
    used_blobs = {e['blake3'] for c in generation for e in c['evidence']} - common_methods
    used_families = {fixture_family(c) for c in generation if c['domain'] == 'healthbench-professional'}
    selected, seen_evidence = [], set(used_blobs)
    for domain in DOMAINS:
        candidates = [c for c in cases if c['domain'] == domain and c['case_id'] not in generation_case_ids]
        candidates.sort(key=lambda c:hashlib.sha256(f'{SEED}:{c["case_id"]}'.encode()).hexdigest())
        chosen = []
        for case in candidates:
            blobs = {e['blake3'] for e in case['evidence']} - common_methods
            family = fixture_family(case)
            if not blobs or blobs & seen_evidence or (family is not None and family in used_families):
                continue
            chosen.append(case); seen_evidence.update(blobs)
            if family is not None: used_families.add(family)
            if len(chosen) == count_per_domain:break
        require(len(chosen) == count_per_domain, 'insufficient_disjoint_development_cases:'+domain)
        selected.extend(chosen)
    return selected, sorted(common_methods), {h:n for h,n in frequency.items() if n > 1}


def prepare(candidate_path, feedback_path, output):
    candidate, feedback = committed(candidate_path), committed(feedback_path)
    require(candidate['feedback_blake3'] == feedback['document_blake3'], 'trial_candidate_feedback_changed')
    manifest = committed(BUNDLE/'bundle.json')
    selected, common, shared = select_cases(manifest, {r['case_id'] for r in feedback['episodes']})
    from common import INDEX
    index = {row['metadata']['sandbox_id']:row for row in map(json.loads, INDEX.read_text().splitlines())}
    rows = []
    for ordinal, case in enumerate(selected):
        record_path = BUNDLE/case['record_path']
        require(file_digest(record_path) == case['record_file_blake3'], 'trial_record_file_changed')
        record = read(record_path); require_training_source(record)
        row = index[case['case_id']]
        require(row['metadata']['source_record_blake3'] == record['record_blake3'] and
                row['metadata']['bundle_blake3'] == manifest['document_blake3'], 'trial_index_binding_changed')
        rows.append({'group_index':ordinal, 'case_id':case['case_id'], 'domain':case['domain'],
                     'fixture_family':fixture_family(case), 'metadata':row['metadata'],
                     'evidence_blake3':[e['blake3'] for e in case['evidence']],
                     'requested_sampling_seeds':[SEED+ordinal*100+k for k in range(8)]})
    plan = {'schema':'eva.harness-matched-trial-plan.v1', 'status':'prepared_not_executed',
        'candidate_path':str(Path(candidate_path).resolve()), 'candidate_blake3':candidate['document_blake3'],
        'generation_feedback_blake3':feedback['document_blake3'], 'bundle_blake3':manifest['document_blake3'],
        'groups':rows, 'groups_per_domain':2, 'samples_per_group':8, 'arms':['baseline','candidate'],
        'original_trajectories_per_policy':len(rows)*8*2, 'reward_policy':POLICY,
        'reasoning_effort':'medium', 'trajectory_output_budget':16384, 'max_model_requests':24,
        'max_tool_attempts':24, 'wall_seconds_per_actor':900, 'selection_uses_scores':False,
        'shared_public_methods_blake3':common, 'shared_evidence_counts':shared,
        'analysis_unit':'Paired domain means across case groups; K=8 samples are dependent within a case; shared source/template dependence disclosed.',
        'sampling_design':SAMPLING_DESIGN, 'requested_seeds_used_by_sampler':False,
        'seed_caveat':'The unchanged production backend ignores request seeds when deterministic inference is disabled. Seeds identify requests only; these are stochastic rollouts on matched cases, without paired random streams or bitwise reproducibility.',
        'admission_rule':{'all_required_assessments_valid':True, 'primary':'strictly positive lower 95% domain-bootstrap completion difference',
                          'completion_definition':'Full host workflow completion with no hacking veto and passed source hard gates; raw host completion is also reported separately.',
                          'composite_mean_difference_nonnegative':True, 'no_domain_completion_regression':True,
                          'hacking_veto_count_must_not_increase':True, 'bootstrap_seed':SEED, 'bootstrap_resamples':10000},
        'reserved_split_plan_sha256': __import__('task_splits').PLAN_SHA256,
        'harder_split_plan_sha256': __import__('task_splits').NEW_PLAN_SHA256,
        'checkpoint_binding':'Bind the actual loaded complete checkpoint before either arm; no optimizer update while the comparison is running.',
        'optimizer_input':False, 'held_out_evaluation':False, 'scope':'Sixteen-group development trial across eight families; not full-corpus or clinical validation.'}
    plan['document_blake3'] = digest(plan)
    output = Path(output).resolve(); require(output.is_relative_to(ROOT), 'trial_plan_outside_workspace')
    write_json(output, plan, exclusive=True)
    return plan


def renew(reference_plan, candidate_path, feedback_path, output):
    """Reuse a fixed development cohort, always excluded from teacher feedback."""
    from copy import deepcopy
    from .training_feedback import reserved_sources
    from .trial_runner import journal
    reference, candidate, feedback = committed(reference_plan), committed(candidate_path), committed(feedback_path)
    require(reference['reward_policy'] == POLICY and reference['sampling_design'] == SAMPLING_DESIGN and
            reference['requested_seeds_used_by_sampler'] is False and candidate['feedback_blake3'] == feedback['document_blake3'],
            'renewed_trial_candidate_or_policy_changed')
    manifest = committed(BUNDLE/'bundle.json')
    require(reference['bundle_blake3'] == manifest['document_blake3'], 'renewed_trial_bundle_changed')
    cases = {c['case_id']:c for c in manifest['cases']}
    ids,blobs,families,common = reserved_sources(reference,manifest)
    for row in feedback['episodes']:
        case = cases[row['case_id']]
        require(case['case_id'] not in ids and not ({e['blake3'] for e in case['evidence']}-common) & blobs and
                fixture_family(case) not in families, 'renewed_trial_feedback_overlaps_development')
        record_path = BUNDLE/case['record_path']
        require(file_digest(record_path) == case['record_file_blake3'], 'renewed_trial_source_changed')
        require_training_source(read(record_path))
    plan = deepcopy(reference);plan.pop('document_blake3')
    plan.update(candidate_path=str(Path(candidate_path).resolve()),candidate_blake3=candidate['document_blake3'],
                generation_feedback_blake3=feedback['document_blake3'],status='prepared_not_executed',
                reference_development_plan_blake3=reference['document_blake3'],
                adaptive_development_cohort=True,
                held_out_evaluation=False,
                scope='Fixed sixteen-group training-development cohort reused across cycles; adaptive selection may overfit it. Separate held-out release evaluation is required.')
    plan['document_blake3'] = digest(plan)
    journal(Path(output).resolve(),plan)
    return plan
