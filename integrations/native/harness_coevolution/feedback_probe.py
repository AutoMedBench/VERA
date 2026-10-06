"""Bounded native feedback for rare training families, never optimizer inputs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from .feedback import ROOT, BUNDLE, committed, read
from .trial_plan import DOMAINS, fixture_family, validate_training_plan
from .trial_runner import (archive_evidence, capture_versions, checkpoint_identity,
                           journal, recover_capture)
from .training_feedback import reserved_sources
from evamed_portable.integrity import digest, timestamp, write_json

PURPOSE = 'harness-training-feedback-probe'


class FeedbackProbeIntegrityError(ValueError):
    """Reject the harness round with an explicit integrity diagnostic."""


class FeedbackProbeUnavailable(ValueError):
    """Bounded capture work did not produce an admissible feedback actor."""


def require(condition, reason):
    if not condition:
        raise FeedbackProbeIntegrityError(reason)


def seal(value):
    return {**value, 'document_blake3':digest(value)}


def select_groups(reference_plan, missing_domains, rollout_id):
    """Select one case per absent family without reading rewards or test tasks."""
    from common import INDEX, sha
    from derived_task_admission import ASSETS
    from task_splits import training_ids
    from evamed_portable.integrity import strict_json
    require(type(rollout_id) is int and rollout_id >= 0 and missing_domains and
            len(set(missing_domains)) == len(missing_domains) and
            set(missing_domains) <= set(DOMAINS), 'feedback_probe_families_invalid')
    reference = committed(reference_plan)
    validate_training_plan(reference)
    manifest = committed(BUNDLE/'bundle.json')
    ids, blobs, families, common = reserved_sources(reference, manifest)
    train = training_ids()
    raw = INDEX.read_bytes()
    require(sha(raw) == read(ASSETS/'asset-receipt.json')['index_sha256'], 'feedback_probe_index_changed')
    rows = [strict_json(line) for line in raw.splitlines()]
    index = {r['label']:r for r in rows}
    require(len(rows) == len(index) == 6080, 'feedback_probe_index_inventory_changed')
    selected = []
    for domain in DOMAINS:
        if domain not in missing_domains:
            continue
        candidates = [c for c in manifest['cases'] if c['domain'] == domain and c['stage'] == 'E2E'
                      and c['case_id'] in train and c['case_id'] not in ids
                      and not (({e['blake3'] for e in c['evidence']} - common) & blobs)
                      and fixture_family(c) not in families]
        require(candidates, 'feedback_probe_no_disjoint_training_case:'+domain)
        case = min(candidates, key=lambda c:hashlib.sha256(
            f'feedback:{rollout_id}:{domain}:{c["case_id"]}'.encode()).hexdigest())
        metadata = index[case['case_id']]['metadata']
        require(metadata['reserved_split'] == 'train' and metadata['stage'] == 'E2E' and
                metadata['domain'] == domain and metadata['bundle_blake3'] == manifest['document_blake3'],
                'feedback_probe_metadata_changed')
        selected.append({'case_id':case['case_id'], 'domain':domain,
                         'group_index':len(selected), 'metadata':metadata,
                         'requested_sampling_seed':20260918+rollout_id*10+len(selected)})
    return selected


def validate_plan(path):
    plan = committed(path)
    require(plan['schema'] == 'eva.native-harness-feedback-probe-plan.v1' and
            plan['purpose'] == PURPOSE and plan['optimizer_input'] is False and plan['reward_weight'] == 0 and
            plan['cases_per_missing_family'] == 1 and plan['max_capture_attempts_per_case'] == 3 and
            plan['requested_seeds_used_by_sampler'] is False and
            plan['interrupted_attempts_consume_slots'] is True and
            plan['max_parallel_captures'] == len(plan['groups']) and
            1 <= len(plan['groups']) <= len(DOMAINS), 'feedback_probe_plan_scope_changed')
    require(committed(plan['reference_plan_path'])['document_blake3'] == plan['reference_plan_blake3'] and
            plan['groups'] == select_groups(plan['reference_plan_path'], plan['missing_domains'], plan['rollout_id']),
            'feedback_probe_selected_cases_changed')
    candidate = plan['baseline_candidate_path']
    require((committed(candidate)['document_blake3'] if candidate else None) == plan['baseline_candidate_blake3'],
            'feedback_probe_baseline_changed')
    return plan


def verify_boundary(path, observation, plan, expected_context):
    """An export must name the expected boundary and the exact planned cell."""
    require(isinstance(expected_context, dict) and
            Path(observation['plan_path']).resolve(strict=True) ==
                Path(expected_context['plan_path']).resolve(strict=True) and
            plan['rollout_id'] == expected_context['rollout_id'] and
            plan['checkpoint_identity'] == expected_context['checkpoint_identity'],
            'feedback_probe_boundary_changed')
    expected_cell = Path(expected_context['plan_path']).resolve(strict=True).parent/observation['domain']/'observation.json'
    require(Path(path).resolve(strict=True) == expected_cell,
            'feedback_probe_observation_outside_planned_cell')


def verify_probe(path, *, expected_actor=None, expected_context):
    """Reopen exact native tokens and retained artifacts before public export."""
    try:
        return _verify_probe(path, expected_actor=expected_actor, expected_context=expected_context)
    except FeedbackProbeIntegrityError:
        raise
    except (ValueError, KeyError, OSError, TypeError):
        raise FeedbackProbeIntegrityError('feedback_probe_retained_verification_failed') from None


def verify_admitted_capture(actor, capture):
    """An archived observation must already have every required native receipt."""
    from evamed_codex.slime_token_admission import (NAME, verify_completed_capture,
                                                   verify_owned_output_receipt)
    trajectory = committed(actor/'trajectory.json')
    if trajectory['codex_turn_status'] == 'completed':
        verify_completed_capture(actor, capture, 16384)
    else:
        require((actor/NAME).is_file(), 'feedback_probe_archived_terminal_receipt_missing')
        proof = verify_owned_output_receipt(actor)
        require(Path(proof['capture_root']).resolve(strict=True) == capture.resolve(strict=True) and
                proof['limit'] == 16384, 'feedback_probe_archived_terminal_scope_changed')


def _verify_probe(path, *, expected_actor, expected_context):
    observation = committed(path)
    require(observation['schema'] == 'eva.native-harness-feedback-probe-observation.v1' and
            observation['purpose'] == PURPOSE and observation['optimizer_input'] is False and
            observation['reward_weight'] == 0 and observation['capture_admitted'] is True,
            'feedback_probe_observation_scope_changed')
    plan = validate_plan(observation['plan_path'])
    require(plan['document_blake3'] == observation['plan_blake3'], 'feedback_probe_plan_changed')
    verify_boundary(path, observation, plan, expected_context)
    actor = Path(observation['actor']).resolve(strict=True)
    root = Path(path).resolve(strict=True).parent
    capture = Path(observation['capture_root']).resolve(strict=True)
    require(root.is_relative_to(Path(plan['checkpoint_identity']['checkpoint']).parent/'harness-coevolution/driver')
            and capture.is_relative_to(root) and actor.is_relative_to(capture), 'feedback_probe_scope_escaped')
    if expected_actor is not None:
        require(actor == Path(expected_actor).resolve(strict=True), 'feedback_probe_actor_changed')
    trajectory = committed(actor/'trajectory.json')
    group = next((g for g in plan['groups'] if g['case_id'] == observation['case_id']), None)
    require(group is not None and observation['domain'] == group['domain'] and
            trajectory['case_id'] == group['case_id'] and
            trajectory['domain'] == group['domain'] and trajectory['focus_stage'] == 'E2E' and
            trajectory['bundle_blake3'] == group['metadata']['bundle_blake3'] and
            trajectory['source_record_blake3'] == group['metadata']['source_record_blake3'] and
            trajectory['purpose'] == PURPOSE and
            trajectory['harness_binding']['candidate_blake3'] == plan['baseline_candidate_blake3'],
            'feedback_probe_actor_binding_changed')
    intent = read(root/(capture.name+'-intent.json'))
    require(intent['allocation_id'] == observation['allocation_id'] and
            intent['plan_blake3'] == plan['document_blake3'], 'feedback_probe_capture_intent_changed')
    verify_admitted_capture(actor, capture)
    require(capture_versions(capture, group['requested_sampling_seed']) == observation['serving_weight_version'],
            'feedback_probe_serving_version_changed')
    files, links = archive_evidence(root, {capture}, actor, trajectory['codex_binary_blake3'])
    require(files == observation['evidence_files'] and links == observation['evidence_symlinks'],
            'feedback_probe_retained_evidence_changed')
    return observation


def capture_one(args, rollout_id, state, plan_path, group, output, baseline):
    """Retain admitted task failures; retry only infrastructure/capture failures."""
    try:
        return _capture_one(args, rollout_id, state, plan_path, group, output, baseline)
    except (FeedbackProbeIntegrityError, FeedbackProbeUnavailable):
        raise
    except (ValueError, KeyError, OSError, TypeError):
        raise FeedbackProbeIntegrityError('feedback_probe_state_verification_failed') from None


def _capture_one(args, rollout_id, state, plan_path, group, output, baseline):
    from evamed_codex.slime_portable_rollout import capture_trajectory
    from slime.utils.types import Sample
    from capture_fallback import RETRYABLE
    plan = validate_plan(plan_path)
    require(group in plan['groups'] and plan['rollout_id'] == rollout_id,
            'feedback_probe_group_not_in_plan')
    final = output/'observation.json'
    if final.exists():
        cached = verify_probe(final, expected_context={'plan_path':str(plan_path), 'rollout_id':rollout_id,
                                                       'checkpoint_identity':plan['checkpoint_identity']})
        require(cached['plan_path'] == str(plan_path) and cached['case_id'] == group['case_id'],
                'feedback_probe_cached_cell_changed')
        return cached
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    original = Sample(index=group['group_index'], group_index=group['group_index'], prompt='',
                      metadata=deepcopy(group['metadata']), label=group['case_id'])
    params = {**deepcopy(state.sampling_params), 'sampling_seed':group['requested_sampling_seed']}
    actor = capture = None
    for number in range(1, 4):
        actor = None
        target = output/f'capture-{number:02d}'
        outcome = output/f'capture-{number:02d}-outcome.json'
        intent = output/f'capture-{number:02d}-intent.json'
        recorded = read(outcome) if outcome.exists() else None
        if recorded is not None:
            require(type(recorded.get('admitted')) is bool and type(recorded.get('retryable')) is bool and
                    not (recorded['admitted'] and recorded['retryable']),
                    'feedback_probe_outcome_shape_changed')
        if recorded is not None and recorded['admitted'] is False:
            if not recorded['retryable']:
                raise FeedbackProbeUnavailable('feedback_probe_prior_nonretryable_capture_failure')
            continue
        if target.exists():
            try:
                recovered = recover_capture(target)
            except (ValueError, KeyError, OSError, TypeError):
                raise FeedbackProbeIntegrityError('feedback_probe_recovery_verification_failed') from None
            if recorded is not None and recorded['admitted'] is True:
                require(recovered is not None, 'feedback_probe_admitted_capture_changed')
            if recovered is not None:
                actor, capture = recovered, target
                break
            if not outcome.exists():
                journal(outcome, {'admitted':False, 'retryable':True,
                                 'reason':'interrupted_or_unadmitted_capture', 'retained':True})
            continue
        if intent.exists():
            journal(outcome, {'admitted':False, 'retryable':True,
                             'reason':'allocation_interrupted_before_capture', 'retained':True})
            continue
        journal(intent, {'allocation_id':os.environ.get('SLURM_JOB_ID'),
                         'created_at_utc':timestamp(), 'plan_blake3':plan['document_blake3']})
        try:
            samples, actor = capture_trajectory(args, original, tokenizer=state.tokenizer,
                sampling_params=params, output=target, rollout_id=rollout_id,
                purpose=PURPOSE, harness_candidate=baseline)
            del samples  # Probe outputs never enter a production batch or optimizer.
        except Exception as exc:
            retryable = isinstance(exc, ValueError) and str(exc).split(';', 1)[0] in RETRYABLE
            journal(outcome, {'admitted':False, 'retryable':retryable, 'error_type':type(exc).__name__})
            if not retryable:
                raise FeedbackProbeUnavailable('feedback_probe_capture_unavailable') from None
            continue
        journal(outcome, {'admitted':True, 'retryable':False, 'actor':str(actor)})
        capture = target
        break
    if actor is None or capture is None:
        raise FeedbackProbeUnavailable('feedback_probe_capture_recovery_exhausted')
    require(Path(actor).resolve().parent.parent == Path(capture).resolve(),
            'feedback_probe_actor_capture_pair_changed')
    trajectory = committed(actor/'trajectory.json')
    require(trajectory['purpose'] == PURPOSE and trajectory['case_id'] == group['case_id'] and
            trajectory['harness_binding']['candidate_blake3'] == plan['baseline_candidate_blake3'],
            'feedback_probe_executed_actor_changed')
    version = capture_versions(capture, group['requested_sampling_seed'])
    files, links = archive_evidence(output, {capture}, actor, trajectory['codex_binary_blake3'])
    intent = read(output/(capture.name+'-intent.json'))
    value = seal({'schema':'eva.native-harness-feedback-probe-observation.v1', 'purpose':PURPOSE,
        'actor':str(actor), 'capture_root':str(capture), 'case_id':group['case_id'], 'domain':group['domain'],
        'plan_path':str(plan_path), 'plan_blake3':plan['document_blake3'],
        'capture_admitted':True, 'optimizer_input':False, 'reward_weight':0,
        'serving_weight_version':version, 'allocation_id':intent['allocation_id'],
        'evidence_files':files, 'evidence_symlinks':links})
    journal(final, value)
    return value


def complete_feedback(args, rollout_id, selection, reference_plan, output, baseline, *, expected_identity=None):
    """Called only by the synchronous driver, before normal production sampling."""
    require(getattr(args, 'evamed_in_synchronous_rollout', False) is True,
            'feedback_probe_requires_synchronous_rollout_owner')
    require(selection['through_optimizer_update'] == rollout_id and selection['missing_domains'],
            'feedback_probe_requires_current_missing_families')
    require(args.rollout_max_response_len == 16384 and args.seq_length == 262144,
            'feedback_probe_native_budget_changed')
    from .trial_runner import verify_sampling_mode
    verify_sampling_mode(args)
    run = Path(args.save).resolve().parent
    output = Path(output).resolve()
    require(output.is_relative_to(run/'harness-coevolution/driver') and output.is_relative_to(ROOT),
            'feedback_probe_output_outside_driver')
    identity = checkpoint_identity(args, rollout_id)
    if expected_identity is not None:
        require(identity == expected_identity, 'feedback_probe_requested_checkpoint_changed')
    from slime.rollout.sglang_rollout import GenerateState
    state = GenerateState(args)
    plan = seal({'schema':'eva.native-harness-feedback-probe-plan.v1', 'purpose':PURPOSE,
        'rollout_id':rollout_id, 'checkpoint_identity':identity,
        'reference_plan_path':str(Path(reference_plan).resolve()),
        'reference_plan_blake3':committed(reference_plan)['document_blake3'],
        'missing_domains':selection['missing_domains'],
        'groups':select_groups(reference_plan, selection['missing_domains'], rollout_id),
        'baseline_candidate_path':str(baseline) if baseline else None,
        'baseline_candidate_blake3':committed(baseline)['document_blake3'] if baseline else None,
        'sampling_params':deepcopy(state.sampling_params), 'requested_seeds_used_by_sampler':False,
        'cases_per_missing_family':1, 'max_capture_attempts_per_case':3,
        'max_parallel_captures':len(selection['missing_domains']),
        'interrupted_attempts_consume_slots':True,
        'exhaustion_behavior':'Withhold generation for this round; continue ordinary training with the prior harness.',
        'optimizer_input':False, 'reward_weight':0, 'saved_feedback_selection_blake3':selection['document_blake3']})
    plan_path = output/'plan.json'
    if plan_path.exists():
        previous = committed(plan_path)
        require(previous['rollout_id'] == rollout_id and previous['checkpoint_identity'] == identity,
                'feedback_probe_resume_boundary_changed')
        require(previous == plan, 'feedback_probe_resume_plan_changed')
    else:
        journal(plan_path, plan)
    observations, failures, integrity_failures = [], [], []
    with ThreadPoolExecutor(max_workers=len(plan['groups'])) as pool:
        pending = {pool.submit(capture_one, args, rollout_id, state, plan_path, group,
                              output/group['domain'], baseline):group for group in plan['groups']}
        for future in as_completed(pending):
            group = pending[future]
            try:
                observations.append(future.result())
            except FeedbackProbeIntegrityError as exc:
                problem = {'domain':group['domain'], 'reason_code':str(exc), 'round_rejected':True,
                           'candidate_promoted':False, 'optimizer_input':False}
                journal(output/('integrity-failure-'+group['domain']+'.json'), problem)
                integrity_failures.append(problem)
            except Exception as exc:
                failures.append({'domain':group['domain'], 'error_type':type(exc).__name__})
            write_json(output/'status.json', {'purpose':PURPOSE, 'completed_originals':len(observations),
                'planned_originals':len(plan['groups']), 'infrastructure_failures':len(failures),
                'integrity_failures':len(integrity_failures),
                'optimizer_input':False, 'reward_weight':0, 'checked_at_utc':timestamp()})
    if integrity_failures:
        raise FeedbackProbeIntegrityError('feedback_probe_integrity_rejected')
    require(checkpoint_identity(args, rollout_id) == identity, 'feedback_probe_policy_changed')
    observations.sort(key=lambda row:row['domain'])
    failures.sort(key=lambda row:row['domain'])
    versions = {}
    for row in observations:
        require(row['allocation_id'], 'feedback_probe_allocation_identity_missing')
        versions.setdefault(row['allocation_id'], set()).add(row['serving_weight_version'])
    require(all(len(v) == 1 for v in versions.values()), 'feedback_probe_serving_policy_changed_between_cases')
    return observations, failures
