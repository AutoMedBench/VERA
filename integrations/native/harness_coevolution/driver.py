"""Periodic train-feedback → teacher proposal → independent review → native cycle."""
from pathlib import Path
import os

from .feedback import ROOT, committed, read, require, export
from evamed_portable.integrity import digest, file_digest, timestamp, write_json
from .cycle import prepare_spec, validate_spec, before_rollout as run_cycle
from .trial_runner import checkpoint_identity, journal
from .trial_plan import DOMAINS, renew
from .training_feedback import register_previous_update, select_feedback
from .generate import generate
from .review import review


def seal(value):return {**value,'document_blake3':digest(value)}


def prepare_config(seed_spec, output):
    seed_spec = Path(seed_spec).resolve(strict=True);spec = validate_spec(seed_spec)
    config = seal({'schema':'eva.harness-periodic-driver.v1','seed_spec_path':str(seed_spec),
        'seed_spec_blake3':spec['document_blake3'],'reference_plan_path':spec['plan_path'],
        'reference_plan_blake3':spec['plan_blake3'],'interval_optimizer_updates':20,'feedback_window_updates':20,
        'feedback_families_required':len(DOMAINS),'max_generation_attempts':2,'max_review_attempts_per_candidate':2,
        'teacher_model':'openai/openai/gpt-6-astra','static_reviewer_model':'aws/anthropic/bedrock-claude-opus-5',
        'reward_weight_of_generation_and_static_review':0,'activation':'prospective_release_only'})
    journal(Path(output).resolve(),config);return config


def validate_config(path):
    value = committed(path)
    require(value['schema'] == 'eva.harness-periodic-driver.v1' and value['interval_optimizer_updates'] == 20 and
            value['feedback_window_updates'] == 20 and value['feedback_families_required'] == len(DOMAINS) and
            value['max_generation_attempts'] == value['max_review_attempts_per_candidate'] == 2,
            'harness_driver_contract_changed')
    spec = validate_spec(value['seed_spec_path'])
    require(spec['document_blake3'] == value['seed_spec_blake3'] and
            committed(value['reference_plan_path'])['document_blake3'] == value['reference_plan_blake3'],
            'harness_driver_seed_or_cohort_changed')
    if 'task_version_transition' in value:
        from .task_transition import validate
        validate(value)
    return value


def stage(root, rollout_id, name, **extra):
    value = {'schema':'eva.harness-driver-status.v1','optimizer_step':rollout_id+1,
        'job_id':os.environ.get('SLURM_JOB_ID'),'attempt':os.environ.get('EVAMED_RUN_ATTEMPT'),
        'stage':name,'training_mean_reward':None,'graded_training_originals':0,'planned_training_originals':64,
        'scope':'Harness work before production sampling; this is not an optimizer update.',
        'checked_at_utc':timestamp(),**extra}
    write_json(root/'status.json',value)
    if root.name == 'task-v2f':
        write_json(root.parent/'status.json',{**value,'driver_state_root':str(root)})


def generate_round(run, rollout_id, root, config, baseline, status_root, *, args=None):
    selection_path = root/'feedback-selection.json'
    if selection_path.exists():selection = committed(selection_path)
    else:
        stage(status_root,rollout_id,'harness_feedback_selection')
        selection = seal(select_feedback(run,rollout_id,config['reference_plan_path'],window=config['feedback_window_updates']))
        journal(selection_path,selection)
    actors = list(selection['actors'])
    probe_evidence = {}
    probe_context = None
    if selection['missing_domains']:
        if args is None:
            return None, 'insufficient_saved_feedback_all_eight_families_required'
        from .feedback_probe import complete_feedback
        stage(status_root,rollout_id,'harness_feedback_selection',
              activity='native_missing_family_feedback',
              feedback_probe_status_path=str(root/'feedback-probes/status.json'))
        boundary_identity = checkpoint_identity(args,rollout_id)
        probes, failures = complete_feedback(args,rollout_id,selection,config['reference_plan_path'],
                                            root/'feedback-probes',baseline,expected_identity=boundary_identity)
        journal(root/'feedback-completion.json',seal({'saved_selection_blake3':selection['document_blake3'],
            'probe_observations':[{k:p[k] for k in ('actor','case_id','domain','document_blake3')} for p in probes],
            'failures':failures,'optimizer_inputs_added':0,'reward_weight':0}))
        if failures or {p['domain'] for p in probes} != set(selection['missing_domains']):
            return None, 'missing_family_native_feedback_unavailable'
        for probe in probes:
            actors.append({'actor':probe['actor'],'case_id':probe['case_id'],'domain':probe['domain']})
            probe_evidence[probe['actor']] = root/'feedback-probes'/probe['domain']/'observation.json'
        probe_context = {'plan_path':str(root/'feedback-probes/plan.json'), 'rollout_id':rollout_id,
                         'checkpoint_identity':boundary_identity}
    require(len(actors) == config['feedback_families_required'] == len(DOMAINS) and
            {row['domain'] for row in actors} == set(DOMAINS),
            'driver_feedback_all_eight_families_required')
    packet = export([row['actor'] for row in actors],root/'feedback',
                    probe_evidence=probe_evidence,probe_context=probe_context)
    require({r['case_id'] for r in packet['episodes']} == {r['case_id'] for r in actors},
            'driver_export_changed_selected_cases')
    feedback_path = root/'feedback/public-feedback.json'
    revision_path = None
    for number in range(1,config['max_generation_attempts']+1):
        generation = root/f'generation-{number:02d}'
        stage(status_root,rollout_id,'harness_candidate_generation',generation_attempt=number)
        if (generation/'outcome.json').exists():outcome = read(generation/'outcome.json')
        else:
            try:
                candidate = generate(feedback_path,generation,revision_path)
                outcome = {'valid':True,'candidate_blake3':candidate['document_blake3']}
            except Exception as exc:
                outcome = {'valid':False,'error_type':type(exc).__name__,'promoted':False}
            journal(generation/'outcome.json',outcome)
        if not outcome['valid']:
            if (generation/'proposal.json').exists():
                revision_path = generation/'revision-input.json'
                journal(revision_path,{'proposal':read(generation/'proposal.json'),
                    'local_validation':read(generation/'validation.json') if (generation/'validation.json').exists() else
                    {'valid':False,'error_type':outcome['error_type']}})
            continue
        candidate_path = generation/'candidate/candidate.json'
        require(committed(candidate_path)['document_blake3'] == outcome['candidate_blake3'], 'driver_candidate_changed')
        decision = None
        for review_number in range(1,config['max_review_attempts_per_candidate']+1):
            review_root = generation/f'review-{review_number:02d}'
            stage(status_root,rollout_id,'harness_independent_static_review',generation_attempt=number,review_attempt=review_number)
            if (review_root/'outcome.json').exists():review_outcome = read(review_root/'outcome.json')
            else:
                try:
                    decision = review(feedback_path,candidate_path,review_root)
                    review_outcome = {'valid':True,'decision':decision['decision'],'review_blake3':digest(decision)}
                except Exception as exc:review_outcome = {'valid':False,'error_type':type(exc).__name__}
                journal(review_root/'outcome.json',review_outcome)
            if review_outcome['valid']:
                decision = read(review_root/'review.json')
                require(digest(decision) == review_outcome['review_blake3'], 'driver_static_review_changed')
                break # Valid reject/revise/accept is never retried for a preferred verdict.
        else:return None, 'static_review_unavailable_candidate_not_changed_to_seek_acceptance'
        if decision['decision'] == 'reject':return None, 'independent_static_review_rejected_candidate'
        if decision['decision'] == 'revise':
            revision_path = generation/'revision-input.json'
            journal(revision_path,{'proposal':read(generation/'proposal.json'),'independent_static_review':decision})
            continue
        require(decision['decision'] == 'accept_for_matched_trial', 'driver_unknown_static_review_decision')
        plan_path = root/'matched-trial-plan.json'
        renew(config['reference_plan_path'],candidate_path,feedback_path,plan_path)
        spec_path = root/'cycle-spec.json'
        prepare_spec(plan_path,review_root,spec_path,baseline_candidate=baseline)
        return str(spec_path), 'candidate_prepared_for_native_matched_cycle'
    return None, 'bounded_generation_or_revision_attempts_exhausted'


def before_rollout(args, rollout_id, config_path):
    require(getattr(args,'evamed_in_synchronous_rollout',False) is True, 'driver_requires_synchronous_rollout_owner')
    config = validate_config(config_path)
    run = Path(args.save).resolve().parent
    root = run/'harness-coevolution/driver';require(root.is_relative_to(ROOT), 'driver_output_outside_workspace')
    identity = checkpoint_identity(args,rollout_id)
    if 'task_version_transition' in config:
        from .task_transition import initialize
        root = initialize(root,rollout_id,identity,config)
    config_binding = seal({'config_path':str(Path(config_path).resolve()),'config_blake3':config['document_blake3']})
    journal(root/'config.json',config_binding)
    register_previous_update(args,rollout_id)
    saved_path = root/'choices'/f'rollout-{rollout_id:04d}.json'
    if saved_path.exists():
        saved = committed(saved_path)
        require(saved['checkpoint_identity'] == identity and saved['config_blake3'] == config['document_blake3'],
                'driver_cached_policy_or_config_changed')
        for path, expected in saved['evidence_files'].items():require(file_digest(path) == expected,'driver_cached_evidence_changed')
        return saved['choice']
    previous_paths = sorted((root/'choices').glob('rollout-*.json'))
    previous = committed(previous_paths[-1]) if previous_paths else None
    if previous:require(previous['rollout_id']+1 == rollout_id, 'driver_optimizer_boundary_was_skipped')
    baseline = previous['choice']['candidate_path'] if previous else validate_spec(config['seed_spec_path'])['baseline_candidate_path']
    active_spec = previous['active_spec_path'] if previous else None
    rounds = sorted(root.glob('round-*/intent.json'))
    current = rounds[-1].parent if rounds else None
    if current is None or ((current/'closed.json').exists() and
            rollout_id >= committed(current/'intent.json')['start_rollout_id']+config['interval_optimizer_updates']):
        current = root/f'round-{len(rounds):04d}'
        journal(current/'intent.json',seal({'round_index':len(rounds),'start_rollout_id':rollout_id,
            'start_checkpoint_identity':identity,'baseline_candidate':baseline,'config_blake3':config['document_blake3'],
            'seed_cycle':not rounds}))
    intent = committed(current/'intent.json')
    # A cutoff can land after closing this round but before publishing its
    # boundary choice. Recover its generated spec, not the preceding round's.
    if (current/'closed.json').exists() and (current/'spec-pointer.json').exists():
        pointer = committed(current/'spec-pointer.json')
        if pointer['spec_path'] is not None:
            active_spec = pointer['spec_path']
    choice = None
    if not (current/'closed.json').exists():
        start = intent['start_rollout_id']
        require(rollout_id in (start,start+1), 'driver_open_round_policy_changed')
        pointer_path = current/'spec-pointer.json'
        if pointer_path.exists():pointer = committed(pointer_path)
        else:
            require(rollout_id == start and identity == intent['start_checkpoint_identity'], 'driver_generation_policy_changed')
            if intent['seed_cycle']:
                spec_path, reason = config['seed_spec_path'],'pre_generated_independently_reviewed_seed'
            else:
                try:spec_path,reason = generate_round(run,rollout_id,current,config,intent['baseline_candidate'],root,args=args)
                except Exception as exc:
                    from .feedback_probe import FeedbackProbeIntegrityError
                    integrity = isinstance(exc,FeedbackProbeIntegrityError)
                    spec_path,reason = None,('harness_feedback_integrity_rejected' if integrity else 'harness_preparation_failed')
                    journal(current/'preparation-failure.json',{'error_type':type(exc).__name__,
                        'reason_code':str(exc) if integrity else None,'candidate_promoted':False})
            pointer = seal({'spec_path':spec_path,'reason':reason})
            journal(pointer_path,pointer)
        if pointer['spec_path'] is None:
            journal(current/'closed.json',seal({'reason':pointer['reason'],'candidate_promoted':False,
                                               'closed_rollout_id':rollout_id}))
        else:
            active_spec = pointer['spec_path']
            spec = validate_spec(active_spec)
            trial_root = run/'harness-coevolution'/('cycle-'+spec['document_blake3'])
            stage(root,rollout_id,'harness_native_matched_comparison',
                  comparison_status_path=str(trial_root/('old-policy' if rollout_id == start else 'new-policy')/'status.json'))
            choice = run_cycle(args,rollout_id,active_spec)
            if choice['phase'] not in {'provisional_candidate_one_update','baseline_one_update'}:
                journal(current/'closed.json',seal({'reason':choice['phase'],'candidate_promoted':choice['phase']=='admitted_candidate',
                                                   'closed_rollout_id':rollout_id,'choice_blake3':choice['document_blake3']}))
    if active_spec is not None and choice is None:
        choice = run_cycle(args,rollout_id,active_spec)
    elif choice is None:
        choice = seal({'schema':'eva.harness-training-choice.v1','candidate_path':baseline,
            'candidate_blake3':committed(baseline)['document_blake3'] if baseline else None,'rollout_id':rollout_id,
            'optimizer_step':rollout_id+1,'checkpoint_identity':identity,'phase':'baseline_no_admitted_candidate',
            'applies_to_stage':'E2E','other_stages_use_baseline':True,'production_originals':64,
            'trial_originals_returned_to_optimizer':0,'context_blake3':config['document_blake3']})
    paths = [root/'config.json',current/'intent.json']
    paths += [p for p in (current/'closed.json',current/'spec-pointer.json') if p.exists()]
    decision = seal({'schema':'eva.harness-driver-boundary.v1','rollout_id':rollout_id,'checkpoint_identity':identity,
        'config_blake3':config['document_blake3'],'active_spec_path':active_spec,'choice':choice,
        'evidence_files':{str(p):file_digest(p) for p in paths}})
    journal(saved_path,decision)
    stage(root,rollout_id,'production_rollout_pending',candidate_blake3=choice['candidate_blake3'],
          scope='Harness boundary finished; the normal 64-original production rollout follows.')
    return choice
