"""Resumable matched trials inside the synchronous Slime rollout boundary.

Only this future integration calls the runner; it must never contact the engines
from an external concurrent monitoring process. It does not consume data_buffer
or return samples to the optimizer.
"""
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
from uuid import uuid4

from .feedback import ROOT, committed, read, require
from evamed_portable.integrity import digest, file_digest, timestamp, write_json
from .trial_analysis import analyze
from .trial_plan import POLICY, SAMPLING_DESIGN

CODEX_BINARY = ROOT/'tools/codex-0.153.4/package/vendor/x86_64-unknown-linux-musl/bin/codex'
CODEX_ALIASES = {'apply_patch','applypatch','codex-linux-sandbox','codex-execve-wrapper'}


def owned_symlink(path, actor, binary_digest):
    """Only pinned Codex executable aliases are allowed in its private arg0 dir."""
    arg0 = actor/'provider/codex-isolation/codex/tmp/arg0'
    require(path.is_symlink() and path.parent.parent == arg0 and
            path.parent.name.startswith('codex-arg0') and path.name in CODEX_ALIASES and
            os.readlink(path) == str(CODEX_BINARY) and not CODEX_BINARY.is_symlink() and
            path.resolve(strict=True) == CODEX_BINARY, 'trial_evidence_symlink_forbidden')
    return {'target':os.readlink(path),'target_blake3':binary_digest}


def archive_evidence(root, subtrees, actor, expected_binary_digest):
    files, links, binary_digest = {}, {}, None
    for subtree in set(subtrees):
        for path in subtree.rglob('*'):
            relative = str(path.relative_to(root))
            if path.is_symlink():
                if binary_digest is None:
                    binary_digest = file_digest(CODEX_BINARY)
                    require(binary_digest == expected_binary_digest, 'trial_pinned_binary_changed')
                links[relative] = owned_symlink(path,actor,binary_digest)
            elif path.is_file():files[relative] = file_digest(path)
    return files, links


def journal(path, value):
    """Publish immutable records atomically, including across allocation cutoffs."""
    from common import write
    return write(path, value)


def verify_cached_observation(root, group, arm, sample_index, candidate_path):
    saved = committed(root/'observation.json')
    expected_candidate = committed(candidate_path)['document_blake3'] if candidate_path else None
    require(saved['sampling_design'] == SAMPLING_DESIGN and saved['requested_seeds_used_by_sampler'] is False,
            'cached_trial_sampling_design_changed')
    require((saved['case_id'],saved['arm'],saved['sample_index'],saved['requested_sampling_seed'],saved['harness_candidate_blake3']) ==
            (group['case_id'],arm,sample_index,group['requested_sampling_seeds'][sample_index],expected_candidate),
            'cached_trial_cell_changed')
    require(saved['reward_admitted'] is True, 'cached_trial_observation_is_not_admitted')
    from evamed_portable.integrity import contained_file, relative_path
    for relative, expected in saved['evidence_files'].items():
        require(file_digest(contained_file(root,relative)) == expected, 'completed_trial_evidence_changed')
    if saved.get('evidence_symlinks'):
        actor = Path(saved['actor'])
        require(actor.resolve(strict=True).is_relative_to(root.resolve()), 'trial_actor_outside_cell')
        binary_digest = file_digest(CODEX_BINARY)
        require(binary_digest == committed(actor/'trajectory.json')['codex_binary_blake3'], 'trial_pinned_binary_changed')
        for relative, expected in saved['evidence_symlinks'].items():
            parts = relative_path(relative).parts
            parent = root
            for part in parts[:-1]:
                parent /= part
                require(not parent.is_symlink(), 'trial_evidence_symlink_forbidden')
            require(owned_symlink(root.joinpath(*parts),actor,binary_digest) == expected,
                    'completed_trial_symlink_changed')
    return saved


def finalized_pair(previous):
    """Unavailable roles can recover; admitted rewards must remain unique."""
    from training.automedbench_lite.adapter import read_document
    complete = []
    for folder in previous:
        if (folder/'pair.json').is_file():
            pair = read_document(folder/'pair.json',maximum=64*1024**2)
            if pair.get('reward_admitted') is True:
                complete.append((folder,pair))
    require(len(complete) <= 1, 'trial_has_multiple_final_rewards')
    return complete[0] if complete else None


def checkpoint_identity(args, rollout_id):
    checkpoint = Path(args.save).resolve()
    require(checkpoint.is_relative_to(ROOT), 'trial_checkpoint_outside_workspace')
    iteration = int((checkpoint/'latest_checkpointed_iteration.txt').read_text().strip())
    require(iteration+1 == rollout_id, 'trial_requires_latest_complete_policy_checkpoint')
    source = ROOT/'evamed-codex/training/direct-grpo-maintenance-v1/checkpoints.py'
    spec = importlib.util.spec_from_file_location('trial_checkpoint_inspection', source)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    inspection = helper.inspect(checkpoint, iteration)
    folder = checkpoint/f'iter_{iteration:07d}'
    files = [folder/'.metadata',folder/'metadata.json',folder/'common.pt',
             checkpoint/'rollout'/f'global_dataset_state_dict_{iteration}.pt']
    identity = {'checkpoint':str(checkpoint), 'iteration':iteration, 'completed_optimizer_updates':rollout_id,
                'metadata_commitments':{str(p):file_digest(p) for p in files},
                'storage_bytes':inspection['storage_bytes'], 'storage_files':inspection['storage_files'],
                'sampler_offset':inspection['sampler_offset']}
    identity['document_blake3'] = digest(identity)
    return identity


def capture_versions(capture_root, seed):
    """Check retained request identity and weights; request seeds are not RNG proof."""
    segments = [read(p) for p in (Path(capture_root)/'private-provider-tokens').glob('segment-*.json')]
    require(segments, 'trial_has_no_exact_sampled_tokens')
    versions = set()
    for segment in segments:
        require(segment['sampling_params'].get('sampling_seed') == seed, 'trial_sampling_seed_changed')
        version = segment['meta_info'].get('weight_version')
        require(isinstance(version,(str,int)) and str(version), 'trial_weight_version_missing')
        versions.add(str(version))
    require(len(versions) == 1, 'trial_policy_changed_during_actor')
    return next(iter(versions))


def recover_capture(target):
    """A finalized actor is verified and judged, never rerolled for a score."""
    from evamed_codex.slime_token_admission import (NAME, verify_completed_capture,
        verify_owned_output_receipt, create_owned_output_receipt)
    actors = list((target/'trajectories').glob('*/trajectory.json'))
    if not actors:return None
    require(len(actors) == 1, 'trial_capture_has_multiple_actors')
    actor = actors[0].parent
    trajectory = committed(actors[0])
    if trajectory['errors'] or not trajectory['diagnostic_infrastructure_passed']:
        return None
    if (actor/NAME).exists():
        verify_owned_output_receipt(actor)
    else:
        # These are the same owned-cap proofs used by capture_trajectory. All
        # retained tokens/diagnostics are verified before signing a new receipt.
        diagnostics = list((target/'private-provider-tokens').glob('safe-failure-diagnostic-*.json'))
        owned = any(read(p).get('category') == 'owned_output_budget' for p in diagnostics)
        request_cap = trajectory['codex_turn_status'] == 'failed' and trajectory['model_budget_exhausted'] == 'model_requests'
        if owned or request_cap:
            create_owned_output_receipt(actor,target,16384,request_terminal=request_cap)
        else:
            verify_completed_capture(actor,target,16384)
    return actor


def actor_once(args, group, arm, sample_index, root, state, rollout_id, candidate_path, *, grading_executor=None):
    result_path = root/'observation.json'
    seed = group['requested_sampling_seeds'][sample_index]
    expected_candidate = committed(candidate_path)['document_blake3'] if candidate_path else None
    if result_path.exists():
        return verify_cached_observation(root,group,arm,sample_index,candidate_path)
    from evamed_codex.slime_portable_rollout import capture_trajectory
    from evamed_codex.portable_judge import judge_trajectory
    from slime.utils.types import Sample
    from capture_fallback import RETRYABLE
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    original = Sample(index=group['group_index']*8+sample_index,group_index=group['group_index'],
                      prompt='',metadata=deepcopy(group['metadata']),label=group['case_id'])
    params = {**deepcopy(state.sampling_params),'sampling_seed':seed}
    actor = capture_root = None
    for number in range(1,4):
        target = root/f'capture-{number:02d}'
        outcome = root/f'capture-{number:02d}-outcome.json'
        if outcome.exists() and read(outcome).get('admitted') is False:
            require(read(outcome)['retryable'], 'trial_nonretryable_capture_failure')
            continue
        if target.exists():
            actor = recover_capture(target)
            if outcome.exists() and read(outcome).get('admitted') is True:
                require(actor is not None, 'previously_admitted_capture_integrity_failure')
            if actor is not None:
                capture_root = target; break
            if not outcome.exists():
                journal(outcome,{'admitted':False,'retryable':True,'reason':'interrupted_or_unadmitted_capture',
                                 'retained':True})
            continue
        if (root/f'capture-{number:02d}-intent.json').exists():
            journal(outcome,{'admitted':False,'retryable':True,'reason':'allocation_interrupted_before_capture',
                             'retained':True})
            continue
        journal(root/f'capture-{number:02d}-intent.json',{'allocation_id':os.environ.get('SLURM_JOB_ID'),
            'requested_seed':seed,'requested_seeds_used_by_sampler':False,
            'harness_candidate':str(candidate_path) if candidate_path else None,
            'created_at_utc':timestamp()})
        try:
            samples,actor = capture_trajectory(args,original,tokenizer=state.tokenizer,sampling_params=params,
                output=target,rollout_id=rollout_id,purpose='harness-development-matched-trial',
                harness_candidate=candidate_path)
            del samples
        except Exception as exc:
            retryable = isinstance(exc,ValueError) and str(exc).split(';',1)[0] in RETRYABLE
            journal(outcome,{'admitted':False,'retryable':retryable,'error_type':type(exc).__name__})
            if not retryable:raise
            continue
        journal(outcome,{'admitted':True,'retryable':False,'actor':str(actor)})
        capture_root = target; break
    require(actor is not None and capture_root is not None, 'trial_capture_recovery_exhausted')
    # Authenticate capture identity before handing its immutable evidence to
    # the grading pool. Only assessment/recovery and archival are deferred.
    version = capture_versions(capture_root,seed)
    intent = read(root/(capture_root.name+'-intent.json'))
    trajectory = committed(actor/'trajectory.json')
    require(trajectory['harness_binding']['candidate_blake3'] == expected_candidate, 'trial_executed_harness_changed')

    def finish():
        previous = sorted(root.glob('judgment-*'),key=lambda p:p.stat().st_mtime_ns)
        complete = finalized_pair(previous)
        if complete is not None:
            judge_root, pair = complete
            if pair['reward_admitted']:
                from .reward_resume import verified_prior_roles
                from e2e_reward import combine_e2e, completion_for_source
                verified = verified_prior_roles(judge_root/'source.json',previous)
                require(set(verified) == {'judge','reward-verifier'} and all(
                    verified[role]['assessment_id'] == pair['results'][role]['assessment_id'] for role in verified),
                    'trial_final_reward_assessments_changed')
                check = combine_e2e(verified['judge'],verified['reward-verifier'],
                                   completion_for_source(judge_root/'source.json',judge_root/'completion-evidence.json'))
                require(all(check[k] == pair[k] for k in ('reward_bps','process_score_bps','completion_score_bps',
                                                         'hacking_veto_applied','process_hard_gate_passed')),
                        'trial_final_reward_components_changed')
        else:
            judge_root = root/('judgment-'+str(uuid4()))
            pair = judge_trajectory(actor,judge_root,prior_outputs=previous)
        require(pair['policy'] == POLICY, 'trial_reward_policy_changed')
        fields = ('reward_admitted','reward_bps','process_score_bps','completion_score_bps',
                  'process_hard_gate_passed','hacking_veto_applied')
        observation = {'case_id':group['case_id'],'domain':group['domain'],'arm':arm,'sample_index':sample_index,
            'requested_sampling_seed':seed,'sampling_design':SAMPLING_DESIGN,'requested_seeds_used_by_sampler':False,
            'allocation_id':intent['allocation_id'],'serving_weight_version':version,
            'harness_candidate_blake3':expected_candidate,'reward_policy':POLICY,
            'actor':str(actor),'pair':str(judge_root/'pair.json'),'optimizer_input':False,
            **{k:pair.get(k) for k in fields}}
        files, links = archive_evidence(root,{capture_root,*previous,judge_root},actor,trajectory['codex_binary_blake3'])
        observation.update(evidence_files=files,evidence_symlinks=links)
        observation['document_blake3'] = digest(observation)
        # A withheld pair is retained as an attempt, never as a final zero or a
        # completed cell. A later invocation reuses valid roles and retries missing
        # roles on this same actor; no second actor is sampled for a better reward.
        journal(result_path if observation['reward_admitted'] else
                root/('withheld-observation-'+str(uuid4())+'.json'), observation)
        return observation

    # Release the capture worker while independent assessment and archival run.
    # Direct callers retain the original synchronous behavior.
    return grading_executor.submit(finish) if grading_executor is not None else finish()


def trial_work(plan):
    return [(g,arm,k) for g in plan['groups'] for k in range(8)
            for arm in (plan['arms'] if (g['group_index']+k)%2 == 0 else list(reversed(plan['arms'])))]


def progress_arms(observations, plan):
    result = {}
    for arm in ('baseline','candidate'):
        rows = [r for r in observations if r['arm'] == arm and r.get('reward_admitted') is True]
        result[arm] = {'graded_originals':len(rows),'planned_originals':len(plan['groups'])*8,
                       'mean_reward':sum(r['reward_bps'] for r in rows)/(10000*len(rows)) if rows else None,
                       'hacking_vetoes':sum(r['hacking_veto_applied'] for r in rows)}
    return result


def summarize(plan, observations, identity, binding, errors):
    versions = {}
    for row in observations:
        allocation = row['allocation_id']
        require(allocation, 'trial_allocation_identity_missing')
        versions.setdefault(allocation,set()).add(row['serving_weight_version'])
    require(all(len(v) == 1 for v in versions.values()), 'serving_policy_changed_between_trial_arms')
    result = analyze(plan,observations)
    result.update(checkpoint_identity=identity, binding_blake3=binding['document_blake3'], errors=errors,
                  serving_versions_by_allocation={k:next(iter(v)) for k,v in versions.items()})
    result['document_blake3'] = digest(result)
    return result


def cached_decision(plan, output, identity, binding, baseline_candidate):
    """Recheck retained evidence without sampling or invoking a provider."""
    path = output/'decision.json'
    if not path.exists():return None
    saved = committed(path)
    require(saved['comparison_complete'] is True and saved['binding_blake3'] == binding['document_blake3'],
            'completed_trial_binding_changed')
    rows = [verify_cached_observation(output/f'group-{g["group_index"]:02d}'/arm/f'sample-{k:02d}',
            g,arm,k,Path(plan['candidate_path']) if arm == 'candidate' else baseline_candidate)
            for g,arm,k in trial_work(plan)]
    require(summarize(plan,rows,identity,binding,[]) == saved, 'completed_trial_decision_changed')
    return saved


def verify_sampling_mode(args):
    require(getattr(args,'sglang_enable_deterministic_inference',False) is False and
            getattr(args,'sglang_rl_on_policy_target',None) is None and
            getattr(args,'sglang_config',None) is None and not getattr(args,'rollout_external',False),
            'trial_requires_unchanged_native_stochastic_sampling')
    return {'sampling_design':SAMPLING_DESIGN,'requested_seeds_used_by_sampler':False}


def execute(args, rollout_id, plan_path, output, *, baseline_candidate=None, concurrency=64):
    """Called only before optimizer work, from direct_rl_rollout.generate."""
    from slime.rollout.sglang_rollout import GenerateState
    plan = committed(plan_path)
    from .trial_plan import validate_training_plan, DOMAINS
    validate_training_plan(plan)
    sampling_mode = verify_sampling_mode(args)
    require(plan['sampling_design'] == SAMPLING_DESIGN and plan['requested_seeds_used_by_sampler'] is False,
            'trial_sampling_design_changed')
    require(plan['reward_policy'] == POLICY and plan['arms'] == ['baseline','candidate'] and
            len(plan['groups']) == 2*len(DOMAINS) and plan['groups_per_domain'] == 2 and
            plan['trajectory_output_budget'] == 16384 and plan['reasoning_effort'] == 'medium' and
            committed(plan['candidate_path'])['document_blake3'] == plan['candidate_blake3'],
            'trial_plan_contract_changed')
    require(getattr(args,'evamed_in_synchronous_rollout',False) is True,
            'trial_requires_synchronous_rollout_owner')
    require(plan['samples_per_group'] == 8 and args.rollout_max_response_len == 16384
            and args.seq_length == 262144 and 1 <= concurrency <= 64, 'trial_runtime_contract_changed')
    identity = checkpoint_identity(args,rollout_id)
    output = Path(output).resolve(); require(output.is_relative_to(ROOT), 'trial_output_outside_workspace')
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    state = GenerateState(args)
    binding = {'plan_blake3':plan['document_blake3'],'checkpoint_identity':identity,**sampling_mode,
               'sampling_params':deepcopy(state.sampling_params),
               'baseline_candidate_blake3':committed(baseline_candidate)['document_blake3'] if baseline_candidate else None}
    binding['document_blake3'] = digest(binding)
    if (output/'binding.json').exists():require(committed(output/'binding.json') == binding, 'trial_resume_checkpoint_or_plan_changed')
    else:journal(output/'binding.json',binding)
    cached = cached_decision(plan,output,identity,binding,baseline_candidate)
    if cached is not None:return cached
    work = trial_work(plan)
    observations, errors = [], []
    # Grade completed captures separately so API latency cannot occupy capture
    # workers. Both pools are bounded, and the existing shared judge semaphore
    # continues to bound individual provider calls across both reward roles.
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix='trial-grade') as graders, \
            ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix='trial-capture') as capturers:
        pending = {capturers.submit(actor_once,args,g,arm,k,output/f'group-{g["group_index"]:02d}'/arm/f'sample-{k:02d}',
                              state,rollout_id,Path(plan['candidate_path']) if arm == 'candidate' else baseline_candidate,
                              grading_executor=graders):(g,arm,k,'capture')
                   for g,arm,k in work}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                group,arm,k,phase = pending.pop(future)
                try:
                    result = future.result()
                    if isinstance(result, Future):
                        require(phase == 'capture', 'trial_nested_grading_future')
                        pending[result] = (group,arm,k,'grading')
                    else:
                        observations.append(result)
                except Exception as exc:
                    errors.append({'case_id':group['case_id'],'arm':arm,'sample_index':k,'error_type':type(exc).__name__})
                write_json(output/'status.json',{'stage':'matched_harness_trial','completed_originals':len(observations),
                    'admitted_originals':sum(row['reward_admitted'] for row in observations),
                    'arms':progress_arms(observations, plan), 'capture_workers':concurrency,'grading_workers':concurrency,
                    'total_pipeline_workers':2*concurrency,
                    'grading_cells_pending':sum(v[3] == 'grading' for v in pending.values()),
                    'planned_originals':len(work),'infrastructure_failures':len(errors),'optimizer_input':False,'checked_at_utc':timestamp()})
    require(checkpoint_identity(args,rollout_id) == identity, 'policy_checkpoint_changed_during_trial')
    result = summarize(plan,observations,identity,binding,errors)
    journal(output/'decision.json' if result['comparison_complete'] else
            output/('withheld-decision-'+str(uuid4())+'.json'),result)
    return result
