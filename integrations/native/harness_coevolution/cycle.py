"""One resumable harness search cycle at synchronous optimizer boundaries.

The first boundary compares two harnesses on one saved policy. At most one
ordinary optimizer update uses the provisionally admitted harness. The next
boundary completes the policy-by-harness comparison and commits a selection or
harness rollback. Model weights are not rolled back by this module.
"""
import json
import os
import re
from pathlib import Path

from .feedback import ROOT, committed, read, require
from evamed_portable.integrity import digest, file_digest
from .trial_plan import DOMAINS, POLICY, SAMPLING_DESIGN
from .trial_runner import checkpoint_identity, execute, journal


def seal(value):
    return {**value, 'document_blake3':digest(value)}


def static_review(plan, folder):
    """Bind the real independent review, including its original API response."""
    folder = Path(folder).resolve(strict=True)
    receipt, review = read(folder/'receipt.json'), read(folder/'review.json')
    request, response = read(folder/'request.json'), read(folder/'response.json')
    require(receipt['candidate_blake3'] == plan['candidate_blake3'] and
            receipt['http_status'] == 200 and receipt['reward_weight'] == 0 and
            receipt['requested_model'] == receipt['actual_model'] == response['model'] ==
            'aws/anthropic/bedrock-claude-opus-5', 'cycle_static_review_identity_changed')
    require(digest(request) == receipt['request_blake3'] and digest(response) == receipt['response_blake3'],
            'cycle_static_review_payload_changed')
    calls = response['choices'][0]['message']['tool_calls']
    require(response['choices'][0]['finish_reason'] == 'tool_calls' and len(calls) == 1 and
            calls[0]['function']['name'] == 'review_harness' and
            json.loads(calls[0]['function']['arguments']) == review and
            review['decision'] == 'accept_for_matched_trial', 'cycle_static_review_not_accepted')
    supplied = json.loads(request['messages'][1]['content'])
    require(supplied['candidate'] == committed(plan['candidate_path']) and
            supplied['public_feedback']['document_blake3'] == plan['generation_feedback_blake3'],
            'cycle_static_review_candidate_changed')
    from .review import validate_citations
    validate_citations(supplied['public_feedback'],review)
    return {str(folder/name):file_digest(folder/name)
            for name in ('receipt.json','review.json','request.json','response.json','host-reconciliation.json')}


def prepare_spec(plan_path, review_folder, output, baseline_candidate=None):
    plan = committed(plan_path)
    require(plan['reward_policy'] == POLICY, 'cycle_reward_policy_changed')
    files = static_review(plan,review_folder)
    files[str(Path(plan_path).resolve())] = file_digest(plan_path)
    candidate = committed(plan['candidate_path'])
    files[plan['candidate_path']] = file_digest(plan['candidate_path'])
    files[candidate['skill']['path']] = file_digest(candidate['skill']['path'])
    if baseline_candidate is not None:
        baseline_candidate = str(Path(baseline_candidate).resolve(strict=True))
        baseline = committed(baseline_candidate)
        files[baseline_candidate] = file_digest(baseline_candidate)
        files[baseline['skill']['path']] = file_digest(baseline['skill']['path'])
    spec = seal({'schema':'eva.harness-cycle-spec.v1', 'plan_path':str(Path(plan_path).resolve()),
        'plan_blake3':plan['document_blake3'],'candidate_path':plan['candidate_path'],
        'candidate_blake3':candidate['document_blake3'],'baseline_candidate_path':baseline_candidate,
        'reward_policy':POLICY, 'evidence_files':files, 'activation':'explicit_future_release_only',
        'four_cell_rule':'Provisional admission requires the preregistered old-policy comparison. Final admission also requires the same rule on the new policy and no candidate-arm domain completion, macro reward, or hacking regression across the update.',
        'optimizer_updates_between_comparisons':1,'production_originals_per_update':64,
        'comparison_originals_are_optimizer_inputs':False})
    journal(Path(output).resolve(),spec)
    return spec


def validate_spec(path):
    spec = committed(path)
    require(spec['schema'] == 'eva.harness-cycle-spec.v1' and spec['reward_policy'] == POLICY and
            spec['optimizer_updates_between_comparisons'] == 1 and spec['production_originals_per_update'] == 64,
            'cycle_spec_contract_changed')
    for name, expected in spec['evidence_files'].items():
        source = Path(name).resolve(strict=True)
        require(source.is_relative_to(ROOT) and file_digest(source) == expected, 'cycle_spec_evidence_changed')
    plan = committed(spec['plan_path'])
    from .trial_plan import validate_training_plan
    validate_training_plan(plan)
    require(plan['document_blake3'] == spec['plan_blake3'] and plan['candidate_path'] == spec['candidate_path'] and
            plan['candidate_blake3'] == spec['candidate_blake3'], 'cycle_spec_plan_changed')
    return spec


def four_cells(before, after):
    """Report policy and harness effects separately; enforce conservative retention."""
    require(before['reward_policy'] == after['reward_policy'] == POLICY, 'cycle_comparison_policy_changed')
    require(before['sampling_design'] == after['sampling_design'] == SAMPLING_DESIGN and
            before['requested_seeds_used_by_sampler'] is after['requested_seeds_used_by_sampler'] is False,
            'cycle_sampling_design_changed')
    require(after['checkpoint_identity']['completed_optimizer_updates'] ==
            before['checkpoint_identity']['completed_optimizer_updates']+1, 'cycle_requires_one_optimizer_update')
    result = {'schema':'eva.harness-four-cell-decision.v1','reward_policy':POLICY,
              'comparison_complete':before['comparison_complete'] and after['comparison_complete'],
              'candidate_promoted':False,'model_weights_rolled_back':False,
              'old_policy_trial_blake3':before['document_blake3'],
              'new_policy_trial_blake3':after['document_blake3'], 'optimizer_input':False}
    if not result['comparison_complete']:
        return seal({**result,'reason':'incomplete_evidence_harness_reverts_to_baseline',
                     'policy_effect_by_harness':None,'interaction_completion_difference':None})
    effect = {}
    for arm in ('baseline','candidate'):
        cd = [after['domains'][d]['arms'][arm]['completion_rate']-
              before['domains'][d]['arms'][arm]['completion_rate'] for d in DOMAINS]
        rd = [after['domains'][d]['arms'][arm]['mean_reward']-
              before['domains'][d]['arms'][arm]['mean_reward'] for d in DOMAINS]
        effect[arm] = {'completion_difference':sum(cd)/len(DOMAINS),
                       'minimum_domain_completion_difference':min(cd),
                       'reward_difference':sum(rd)/len(DOMAINS),
                       'hacking_veto_count_difference':after['hacking_veto_counts'][arm]-before['hacking_veto_counts'][arm]}
    candidate = effect['candidate']
    keep = (before['candidate_admitted_for_next_training_interval'] is True and
            after['candidate_admitted_for_next_training_interval'] is True and
            candidate['minimum_domain_completion_difference'] >= 0 and candidate['reward_difference'] >= 0 and
            candidate['hacking_veto_count_difference'] <= 0)
    return seal({**result,'candidate_promoted':keep,'policy_effect_by_harness':effect,
        'interaction_completion_difference':effect['candidate']['completion_difference']-effect['baseline']['completion_difference'],
        'reason':'four_cell_criteria_passed' if keep else 'four_cell_criteria_not_met_harness_reverts_to_baseline',
        'scope':'A one-update development comparison with dependent case samples; no claim of clinical or held-out improvement.'})


def successful_update_proof(args, rollout_id, choice):
    """Choose the latest complete optimizer receipt set preceding this checkpoint."""
    from optimizer_topology import resolve_attempt, receipt_paths, valid_receipt
    run = Path(args.save).resolve().parent
    cutoff = (Path(args.save)/f'iter_{rollout_id:07d}'/'.metadata').stat().st_mtime_ns
    paths = list(run.glob(f'attempts/*/learning/harness-rollout-{rollout_id:04d}.json'))
    paths += list(run.glob(f'learning/harness-rollout-{rollout_id:04d}.json'))
    successful = []
    for path in paths:
        proof = committed(path)
        require(proof['harness_choice_blake3'] == choice['document_blake3'] and
                proof['original_trajectory_count'] == 64 and proof['rollout_id'] == rollout_id,
                'cycle_update_used_different_harness')
        topology = resolve_attempt(ROOT, run, path.parent)
        ranks = receipt_paths(path.parent, rollout_id+1, topology)
        if not all(p.is_file() for p in ranks):continue
        receipt_time = max(p.stat().st_mtime_ns for p in ranks)
        if receipt_time > cutoff:continue
        valid = True
        for rank, receipt_path in enumerate(ranks):
            receipt = read(receipt_path)
            valid &= (valid_receipt(receipt, rank, rollout_id+1) and
                      receipt.get('harness_batch_receipt_blake3') == proof['document_blake3'])
        if valid:successful.append((receipt_time,path,ranks,topology))
    require(successful, 'cycle_requires_all_rank_update_through_selected_harness')
    _, path, ranks, topology = max(successful,key=lambda value:value[0])
    batch = committed(path)
    return seal({'schema':'eva.harness-complete-update-binding.v1','rollout_id':rollout_id,
                 'harness_choice_blake3':choice['document_blake3'],
                 'optimizer_ranks_verified':topology['expected_optimizer_ranks'], 'optimizer_topology':topology,
                 'e2e_originals':batch['e2e_originals'],'candidate_originals':batch['candidate_originals'],
                 'evidence_files':{str(p):file_digest(p) for p in [path,*ranks,*map(Path,topology['evidence_sha256'])]}})


def before_rollout(args, rollout_id, spec_path):
    """Return one immutable E2E harness choice before reading training prompts."""
    require(getattr(args,'evamed_in_synchronous_rollout',False) is True, 'cycle_requires_synchronous_rollout_owner')
    require(args.rollout_batch_size == args.n_samples_per_prompt == 8 and args.global_batch_size == 64,
            'cycle_requires_unchanged_8_by_8_production_batch')
    spec = validate_spec(spec_path)
    root = Path(args.save).resolve().parent/'harness-coevolution'/('cycle-'+spec['document_blake3'])
    require(root.is_relative_to(ROOT), 'cycle_output_outside_workspace')
    identity = checkpoint_identity(args,rollout_id)
    context_path = root/'context.json'
    if context_path.exists():context = committed(context_path)
    else:
        context = seal({'spec_path':str(Path(spec_path).resolve()),'spec_blake3':spec['document_blake3'],
                        'first_rollout_id':rollout_id,'old_checkpoint_identity':identity})
        journal(context_path,context)
    require(context['spec_blake3'] == spec['document_blake3'], 'cycle_context_spec_changed')
    first = context['first_rollout_id']
    require(rollout_id >= first, 'cycle_cannot_rewind_saved_policy')
    choice_path = root/f'rollout-choice-{rollout_id:04d}.json'
    if choice_path.exists():
        choice = committed(choice_path)
        require(choice['checkpoint_identity'] == identity and choice['context_blake3'] == context['document_blake3'],
                'cycle_cached_choice_policy_changed')
        for path, expected in choice['evidence_files'].items():
            require(file_digest(path) == expected, 'cycle_cached_choice_evidence_changed')
        return choice
    baseline = spec['baseline_candidate_path']
    selected, phase = baseline, 'baseline'
    dependencies = [context_path]
    final_path = root/'final.json'
    if rollout_id == first:
        require(identity == context['old_checkpoint_identity'], 'cycle_initial_policy_changed')
        if (root/'old-policy-result.json').exists():before = committed(root/'old-policy-result.json')
        else:
            before = execute(args,rollout_id,spec['plan_path'],root/'old-policy',baseline_candidate=baseline)
            journal(root/'old-policy-result.json',before)
        dependencies.append(root/'old-policy-result.json')
        if before['comparison_complete']:
            if before['candidate_admitted_for_next_training_interval']:
                selected = spec['candidate_path']; phase = 'provisional_candidate_one_update'
            else:phase = 'baseline_one_update'
        else:
            # Bounded missing-role recovery has run; leave this comparison
            # withheld and continue ordinary training with the previous harness.
            final = seal({'candidate_promoted':False,'comparison_complete':False,
                          'reason':'old_policy_evidence_withheld_cycle_closed','old_policy_trial_blake3':before['document_blake3'],
                          'model_weights_rolled_back':False})
            journal(final_path,final); dependencies.append(final_path)
            phase = 'baseline_incomplete_comparison'
    else:
        before = committed(root/'old-policy-result.json')
        dependencies.append(root/'old-policy-result.json')
        if not final_path.exists():
            require(rollout_id == first+1, 'cycle_post_update_comparison_was_skipped')
            prior = committed(root/f'rollout-choice-{first:04d}.json')
            require(prior['checkpoint_identity'] == context['old_checkpoint_identity'], 'cycle_prior_choice_changed')
            proof = successful_update_proof(args,first,prior)
            journal(root/'successful-update.json',proof)
            if (root/'new-policy-result.json').exists():after = committed(root/'new-policy-result.json')
            else:
                after = execute(args,rollout_id,spec['plan_path'],root/'new-policy',baseline_candidate=baseline)
                journal(root/'new-policy-result.json',after)
            final = four_cells(before,after)
            final.pop('document_blake3')
            final.update(production_update_blake3=proof['document_blake3'],
                         candidate_originals_in_update=proof['candidate_originals'])
            if final['candidate_promoted'] and proof['candidate_originals'] == 0:
                final.update(candidate_promoted=False,reason='no_e2e_original_trained_through_candidate')
            journal(final_path,seal(final))
        final = committed(final_path)
        dependencies.append(final_path)
        if (root/'new-policy-result.json').exists():dependencies.append(root/'new-policy-result.json')
        if (root/'successful-update.json').exists():dependencies.append(root/'successful-update.json')
        selected = spec['candidate_path'] if final['candidate_promoted'] else baseline
        phase = 'admitted_candidate' if final['candidate_promoted'] else 'baseline_after_comparison'
    selected_digest = committed(selected)['document_blake3'] if selected else None
    choice = seal({'schema':'eva.harness-training-choice.v1','context_blake3':context['document_blake3'],
        'checkpoint_identity':identity,'rollout_id':rollout_id,'optimizer_step':rollout_id+1,'phase':phase,
        'candidate_path':selected,'candidate_blake3':selected_digest,'applies_to_stage':'E2E',
        'other_stages_use_baseline':True,'production_originals':64,'trial_originals_returned_to_optimizer':0,
        'evidence_files':{str(p):file_digest(p) for p in dependencies}})
    journal(choice_path,choice)
    return choice


def record_training_batch(args, rollout_id, choice, samples, audit):
    """Bind one normal 64-original rollout to the selected E2E harness."""
    originals = {}
    for sample in samples:
        metadata = sample.metadata
        require(metadata['harness_choice_blake3'] == choice['document_blake3'], 'training_harness_choice_changed')
        is_e2e = metadata['harness_candidate_applies']
        actual = metadata['harness_candidate_blake3']
        require(actual == (choice['candidate_blake3'] if is_e2e else None), 'training_harness_candidate_changed')
        row = (is_e2e,actual)
        if sample.rollout_id in originals:require(originals[sample.rollout_id] == row, 'training_harness_changed_between_segments')
        originals[sample.rollout_id] = row
    require(len(originals) == 64, 'training_harness_original_count_changed')
    value = seal({'schema':'eva.harness-admitted-training-batch.v1','rollout_id':rollout_id,
        'optimizer_step':rollout_id+1,'harness_choice_blake3':choice['document_blake3'],
        'original_trajectory_count':len(originals),'e2e_originals':sum(r[0] for r in originals.values()),
        'candidate_originals':sum(r[1] is not None for r in originals.values()),
        'admitted_rollout_audit':audit,'optimizer_update_complete':False,
        'scope':'Admitted rollout receipt; a subsequent complete checkpoint is separately required.'})
    attempt = Path(os.environ.get('EVAMED_ATTEMPT_ROOT',str(Path(args.save).resolve().parent)))
    journal(attempt/'learning'/f'harness-rollout-{rollout_id:04d}.json',value)
    return value
