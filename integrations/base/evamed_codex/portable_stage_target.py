"""Fresh typed stage admission from actual paired Opus/source-rubric evidence.

Each admitted stage requires all seven frozen domains. Missing judgments are
unknown, never zeros. No reward threshold, causal skill claim, E2E substitution,
or host-gate/fixture score enters the selection.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

from .portable_rollout import BUNDLE, ROOT
from .portable_judge import prepare_trajectory
from .benchmark_judge import verify_assessment
from evamed_portable.bundle import Bundle
from evamed_portable.integrity import (Signer, contained_file, digest, file_digest,
    strict_json, timestamp, verify_receipt, write_json)
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.rubrics.models import CompiledRubric
from eva_agent.rubrics.registry import CompiledRubricRegistry
from eva_agent.training.coevolution import (EvaluationStatus, RoundIdentity,
    VerifiedStageEvaluation, plan_stage_target)
from training.automedbench_lite.adapter import read_document as read_adapter_document

SCHEMA = 'eva.portable-paired-source-stage-target.v1'
POLICY = 'eva.portable-seven-domain-paired-stage-admission.v1'
DEFAULT_RECEIPT = ROOT/'evamed-codex/receipts/portable-stage-target.json'
INDEX_RECEIPT = ROOT/'evamed-codex/receipts/portable-rl-index.json'
OPUS = 'aws/anthropic/bedrock-claude-opus-5'
STAGES = ('S1','S2','S3','S4','S5')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def document(path):
    value = strict_json(path.read_bytes())
    require(value['document_blake3'] == digest({k:v for k,v in value.items() if k!='document_blake3'}),
        'stage_target_source_document_changed')
    return value


def local_path(value):
    path = (ROOT/value).resolve(strict=True)
    require(path.is_relative_to(ROOT), 'stage_target_source_outside_workspace')
    return path


def _comparable_projection(value):
    """Normalize only fresh projection UUIDs, retaining native event identities."""
    result = deepcopy(value)
    UUID(result['task_id']); require(result['task_id']==result['candidate_id'], 'projection_candidate_identity_changed')
    result['task_id']=result['candidate_id']='host-projection'
    def tool_result(row):
        require(row['receipt_blake3']==blake3_hex({k:v for k,v in row.items() if k!='receipt_blake3'}),
            'projection_tool_result_changed')
        UUID(row['parallel_group_id']);row['parallel_group_id']=f'frontier-{row["frontier"]}'
        row.pop('receipt_blake3')
    host_ids={}
    for ordinal, message in enumerate(result['messages']):
        core={key:message[key] for key in ('event_id','role','content','tool_call_ids')}
        require(message['event_blake3']==blake3_hex(core), 'projection_message_commitment_changed')
        if message['role'] in ('system','user'):
            UUID(message['event_id']);host_ids[message['event_id']]=f'host-conditioning-{ordinal}'
            message['event_id']=host_ids[message['event_id']]
        elif message['role']=='tool':
            tool_result(message['content'])
        message.pop('event_blake3')
    trace=result['tool_trace']
    require(trace['trace_blake3']==blake3_hex({k:v for k,v in trace.items() if k!='trace_blake3'}),
        'projection_tool_trace_changed')
    trace.pop('trace_blake3')
    for row in trace['results']:
        tool_result(row)
    for row in result['provider_metadata']['native_codex_turns']:
        row['host_conditioning_user_event_id']=host_ids[row['host_conditioning_user_event_id']]
    return result


def _verify_actual_projection(actor, source_path):
    source = strict_json(source_path.read_bytes())
    with TemporaryDirectory(prefix='portable-stage-verify-',dir=ROOT/'tmp') as temporary:
        output=Path(temporary)/'projection'
        rubric=prepare_trajectory(actor,output)
        rebuilt=strict_json((output/'source.json').read_bytes())
        require(_comparable_projection(source)==_comparable_projection(rebuilt),
            'judged_source_not_exact_actual_actor_projection')
    return rubric,source


def _verify_adapter_audit(output, assessment_id):
    from eva_agent.codex_providers.adapter import SignedAdapterReceipt, verify_adapter_receipt
    files=sorted((output/'adapter-receipts').rglob('*.json'))
    require(files, 'actual_Opus_provider_receipts_missing')
    identities,keys,refs=set(),set(),[]
    for path in files:
        raw=strict_json(path.read_bytes())
        receipt=SignedAdapterReceipt(**{key:raw[key] for key in SignedAdapterReceipt.__dataclass_fields__})
        require(receipt.to_dict()==raw, 'Opus_adapter_envelope_fields_changed')
        verify_adapter_receipt(receipt);payload=receipt.payload
        require(path.parent.parent.name==assessment_id and payload['model_id']==OPUS and payload['route_id']=='opus_5' and
            payload['provider_family']=='anthropic' and payload['request_id'] not in identities and
            payload['request_max_retries']==payload['stream_max_retries']==payload['upstream_request_max_retries']==0,
            'Opus_adapter_route_or_attempt_changed')
        identities.add(payload['request_id']);keys.add(receipt.public_key_blake3)
        refs.append({'relative_path':str(path.relative_to(ROOT)),'file_blake3':file_digest(path),
            'envelope_blake3':receipt.envelope_blake3,'status':payload['status']})
    return keys,refs


def _verify_admitted_pair(pair_root, view, rubric, pair, expected_skills, public):
    checked_rubric,source=_verify_actual_projection(view,contained_file(pair_root,'source.json'))
    require(checked_rubric.digest==rubric.digest and source['provider_metadata']['native_codex_turns'] and
        verify_receipt(strict_json(contained_file(view,'turns/01/launch.json').read_bytes()),public)['selected_skills']==expected_skills,
        'paired_actual_round_context_changed')
    verified={role:verify_assessment(pair_root/'source.json',rubric,pair_root/role) for role in ('judge','reward-verifier')}
    require(verified==pair['results'] and all(value['verified'] for value in verified.values()) and
        pair['both_assessments_verified'] is True and pair['exact_item_agreement'] is True and
        verified['judge']['assessment_id']!=verified['reward-verifier']['assessment_id'] and
        verified['judge']['item_scores_bps']==verified['reward-verifier']['item_scores_bps'] and
        verified['judge']['reward_bps']==verified['reward-verifier']['reward_bps']==pair['reward_bps'],
        'paired_independent_reward_not_verified')
    keys,proof=[],[]
    for role in ('judge','reward-verifier'):
        signer_keys,refs=_verify_adapter_audit(pair_root/role,verified[role]['assessment_id']);keys.append(signer_keys);proof.extend(refs)
    require(keys[0].isdisjoint(keys[1]), 'paired_Opus_gateway_authorities_not_independent')
    return rubric.score(verified['judge']['item_scores_bps'],evaluation_id=verified['judge']['assessment_id']),proof


def _index_binding(bundle):
    receipt = strict_json(INDEX_RECEIPT.read_bytes())
    path = local_path(receipt['index_path'])
    require(receipt['schema']=='eva.portable-rl-index-receipt.v1' and receipt['bundle_blake3']==bundle.manifest['document_blake3'] and
        receipt['source_revision']==bundle.manifest['source_revision'] and receipt['rows']==6000 and
        receipt['metadata_only'] is True and receipt['contains_rewards'] is False and
        file_digest(path)==receipt['index_blake3'] and path.stat().st_size==receipt['index_bytes'],
        'portable_index_binding_changed')
    seen=set()
    for line in path.read_bytes().splitlines():
        row=strict_json(line);metadata=row['metadata'];identity=metadata['sandbox_id'];descriptor=bundle.cases[identity]
        require(identity not in seen and row['label']==identity and metadata['bundle_blake3']==bundle.manifest['document_blake3'] and
            metadata['case_descriptor_blake3']==digest(descriptor) and metadata['metadata_only'] is True and
            metadata['precomputed_rewards'] is False and metadata['fresh_on_policy_rollout_required'] is True and
            metadata['stage_target_admission_claimed'] is False and metadata['source_rubric_digest']==descriptor['rubric_digest'] and
            metadata['source_record_file_blake3']==descriptor['record_file_blake3'] and
            metadata['source_construction_blake3']==descriptor['construction_blake3'] and
            metadata['source_tool_catalog_blake3']==descriptor['tool_catalog_blake3'] and
            metadata['domain']==descriptor['domain'] and metadata['stage']==descriptor['stage'], 'portable_index_row_changed')
        seen.add(identity)
    require(seen==set(bundle.cases), 'portable_index_case_inventory_changed')
    return {'relative_path':str(path.relative_to(ROOT)),'file_blake3':file_digest(path),'rows':len(seen),
        'metadata_receipt_relative_path':str(INDEX_RECEIPT.relative_to(ROOT)),'metadata_receipt_file_blake3':file_digest(INDEX_RECEIPT)}


def select_fully_covered_stages(entries, domains):
    require(len(domains)==7 and len(set(domains))==7, 'stage_selection_requires_seven_unique_domains')
    stages,candidates={},[]
    for order,stage in enumerate(STAGES):
        stage_rows=[row for row in entries if row['stage']==stage]
        scored=[row for row in stage_rows if row['status']=='scored']
        eligible=len(scored)==len(domains) and {row['domain'] for row in scored}==set(domains)
        mean=Fraction(sum(row['reward_bps'] for row in scored),len(domains)) if eligible else None
        stages[stage]={'admitted':eligible,'required_domains':domains,'verified_domains':sorted(row['domain'] for row in scored),
            'missing_domains':[row['domain'] for row in stage_rows if row['status']!='scored'],
            'missing_reasons':{row['domain']:row['missing_reason'] for row in stage_rows if row['status']!='scored'},
            'mean_reward_bps':float(mean) if mean is not None else None,
            'mean_reward_bps_exact':{'numerator':mean.numerator,'denominator':mean.denominator} if mean is not None else None,
            'normalized_weighted_source_rubric_mean':float(mean/10000) if mean is not None else None}
        if mean is not None:candidates.append((mean,order,stage))
    target=min(candidates)[2] if candidates else None
    return stages,target


def compute_stage_target(judge_cohort):
    judge_cohort=Path(judge_cohort).resolve(strict=True)
    require(judge_cohort.is_relative_to(ROOT), 'judge_cohort_outside_workspace')
    bundle=Bundle(BUNDLE);public=bundle.manifest['fresh_execution_authority']['public_key_base64']
    selection_signed=strict_json(contained_file(judge_cohort,'selection.json').read_bytes())
    selection=verify_receipt(selection_signed,public)
    complete_signed=strict_json(contained_file(judge_cohort,'cohort.json').read_bytes())
    complete=verify_receipt(complete_signed,public)
    actor_cohort=local_path(selection['actor_cohort_relative_path'])
    frozen=document(contained_file(actor_cohort,'selection.json'))
    require(complete['actor_selection_blake3']==selection['actor_selection_blake3']==frozen['document_blake3'] and
        complete['case_count']==42 and complete['automatic_retries'] is False and complete['actor_replayed'] is False and
        selection['automatic_retries'] is False and selection['actor_replay'] is False and
        selection['independent_assessments_per_case']==2 and selection['judge_model']==OPUS and
        selection['cases']==frozen['cases'] and frozen['attempts_per_case']==1 and frozen['selection_uses_scores'] is False,
        'paired_frozen_cohort_identity_changed')
    domains=sorted({case['domain'] for case in frozen['cases']})
    require(len(domains)==7 and len(frozen['cases'])==42 and
        {(case['domain'],case['stage']) for case in frozen['cases']}=={(domain,stage) for domain in domains for stage in (*STAGES,'E2E')},
        'frozen_domain_stage_coverage_changed')
    statuses={row['case_id']:row for row in complete['statuses'] if row['purpose']=='frozen_cohort'}
    require(len(statuses)==42 and set(statuses)=={case['sandbox_id'] for case in frozen['cases']},
        'paired_judge_case_inventory_changed')
    # Public API model/version reference is exact; provider checkpoint bytes are not disclosed.
    first_actor=actor_cohort/'trajectories'/statuses[frozen['cases'][0]['sandbox_id']]['run_id']
    first_launch=strict_json(contained_file(first_actor,'turns/01/launch.json').read_bytes())
    launch=verify_receipt(first_launch,public)
    skill_identity=digest(launch['selected_skills'])
    identity=RoundIdentity(model_id=frozen['model'],checkpoint_id='hosted-api-weights-undisclosed:'+frozen['model'],
        skill_catalog_id=skill_identity,judge_id=OPUS+';paired-original-source;verifier-code:'+selection['benchmark_judge_source_blake3'])
    typed,entries,proofs=[],[],[]
    for case in frozen['cases']:
        status=statuses[case['sandbox_id']]
        status_path=contained_file(judge_cohort,'case-status/'+status['run_id']+'.json')
        signed_status=strict_json(status_path.read_bytes())
        require(verify_receipt(signed_status,public)==status, 'paired_case_status_signature_changed')
        actor=actor_cohort/'trajectories'/status['run_id']
        actual=document(contained_file(actor,'trajectory.json'))
        require(actual['case_id']==case['sandbox_id'] and actual['document_blake3']==status['original_trajectory_blake3'] and
            actual['model']==frozen['model'] and actual['route']==frozen['route'] and
            actual['source_record_blake3']==case['record_blake3'] and actual['source_rubric_digest']==case['rubric_digest'],
            'paired_case_source_binding_changed')
        _,row,_,_=bundle.case(case['sandbox_id']);table=row['reward_contract']['rubric_table']
        CompiledRubricRegistry._verify_rubric_document(table);rubric=CompiledRubric.from_document(table)
        require(rubric.digest==case['rubric_digest'] and rubric.stage==case['stage'] and rubric.domain==case['domain'],
            'paired_source_rubric_changed')
        assessment=None;reason=None;pair_digest=None;proof=[]
        if status['status']=='pair_completed':
            pair_root=judge_cohort/'assessments'/status['run_id'];pair=read_adapter_document(contained_file(pair_root,'pair.json'))
            pair_digest=pair['document_blake3']
            require(pair_digest==status['pair_document_blake3'] and pair['rubric_digest']==rubric.digest and
                pair['stage']==rubric.stage and pair['domain']==rubric.domain and
                pair['reward_admitted']==status['reward_admitted'], 'paired_reward_summary_changed')
            if pair['reward_admitted']:
                view=local_path(status['recovery_view_relative_path']) if status['recovery_view_relative_path'] else actor
                assessment,proof=_verify_admitted_pair(pair_root,view,rubric,pair,launch['selected_skills'],public)
            else:
                require(pair['reward_bps'] is None and status['reward_bps'] is None,
                    'unadmitted_pair_carries_reward')
                reason='independent_assessments_disagree' if pair['both_assessments_verified'] else 'independent_assessment_invalid_or_unavailable'
        else:
            require(status['reward_admitted'] is False and status['reward_bps'] is None, 'infrastructure_failure_carries_reward')
            reason=status.get('error_code') or status.get('error_type') or 'actor_or_judge_infrastructure_not_admitted'
        evaluation_status=EvaluationStatus.SCORED if assessment else EvaluationStatus.INFRASTRUCTURE_FAILURE
        evaluation=VerifiedStageEvaluation(case_id=case['sandbox_id'],identity=identity,rubric=rubric,status=evaluation_status,
            upstream_verification_id=pair_digest or digest(signed_status),score=assessment)
        evaluation.validate();typed.append(evaluation)
        entries.append({'case_id':case['sandbox_id'],'domain':case['domain'],'stage':case['stage'],
            'actor_run_id':status['run_id'],'original_actor_trajectory_blake3':actual['document_blake3'],
            'status':evaluation_status.value,'missing_reason':reason,'source_rubric_digest':rubric.digest,
            'pair_document_blake3':pair_digest,'reward_bps':assessment.reward_bps if assessment else None,
            'normalized_source_rubric_reward':assessment.reward_bps/10000 if assessment else None,
            'score_document':assessment.to_document() if assessment else None,
            'case_status_file_blake3':file_digest(status_path),'provider_receipts':proof})
    proposal=plan_stage_target(typed)
    stages,target=select_fully_covered_stages(entries,domains)
    return {'policy':POLICY,'bundle_blake3':bundle.manifest['document_blake3'],
        'source_dataset':bundle.manifest['source_dataset'],'source_revision':bundle.manifest['source_revision'],
        'index_binding':_index_binding(bundle),'judge_cohort_relative_path':str(judge_cohort.relative_to(ROOT)),
        'judge_selection_file_blake3':file_digest(judge_cohort/'selection.json'),
        'judge_cohort_file_blake3':file_digest(judge_cohort/'cohort.json'),
        'actor_selection_blake3':frozen['document_blake3'], 'evaluation_model_id':frozen['model'],
        'evaluation_route':frozen['route'],'evaluated_checkpoint_bytes_disclosed':False,
        'training_checkpoint_equivalence_claimed':False,'selected_skill_catalog_blake3':skill_identity,
        'judge_model_id':OPUS,'admitted_stage':target,'rl_launch_admitted':target is not None,
        'selection_rule':'lowest fully seven-domain covered paired source-rubric mean; ties S1,S2,S3,S4,S5',
        'performance_threshold':None,'stages':stages,'case_evidence':entries,
        'e2e':{'used_for_stage_selection':False,'cases':[row for row in entries if row['stage']=='E2E']},
        'typed_descriptive_proposal':proposal,'diagnostic_pair_used_for_selection':False,
        'unknown_is_zero':False,'host_stage_gates_used_as_scores':False,'fixture_scores_used':False,
        'limitations':['Source rubrics and domain constructs differ; normalized weighted rewards are a descriptive training heuristic.',
            'Hosted API checkpoint bytes are undisclosed; no equivalence to a local training checkpoint is asserted.',
            'A higher numbered stage is not assumed harder; missing stage coverage remains unadmitted.',
            'No causal skill diagnosis, performance threshold, or skill modification is authorized by this receipt.']}


def compute_stage_target_with_supplement(judge_cohort, supplemental_judge_cohort):
    """Add first-inference evidence while keeping original exclusions visible."""
    base=compute_stage_target(judge_cohort)
    supplemental=Path(supplemental_judge_cohort).resolve(strict=True)
    require(supplemental.is_relative_to(ROOT), 'supplement_judge_outside_workspace')
    bundle=Bundle(BUNDLE);public=bundle.manifest['fresh_execution_authority']['public_key_base64']
    selected_raw=strict_json(contained_file(supplemental,'selection.json').read_bytes())
    selected=verify_receipt(selected_raw,public)
    complete=verify_receipt(strict_json(contained_file(supplemental,'cohort.json').read_bytes()),public)
    require(selected['schema']=='eva.portable-supplement-source-judge-selection.v1' and
        complete['schema']=='eva.portable-first-inference-supplement-judge-cohort.v1' and
        selected['original_selection_blake3']==complete['original_selection_blake3']==base['actor_selection_blake3'] and
        selected['automatic_retries'] is False and selected['actor_replay'] is False and
        complete['automatic_retries'] is False and complete['actor_replayed'] is False and
        selected['independent_assessments_per_case']==2 and selected['judge_model']==OPUS,
        'supplement_judge_lineage_changed')
    original_judge_selection=verify_receipt(strict_json(contained_file(Path(judge_cohort),'selection.json').read_bytes()),public)
    require(selected['benchmark_judge_source_blake3']==original_judge_selection['benchmark_judge_source_blake3'],
        'supplement_judge_comparison_profile_changed')
    actor_cohort=local_path(selected['actor_supplement_relative_path'])
    actor_selected_raw=strict_json(contained_file(actor_cohort,'selection.json').read_bytes())
    actor_selected=verify_receipt(actor_selected_raw,public)
    actor_complete=verify_receipt(strict_json(contained_file(actor_cohort,'cohort.json').read_bytes()),public)
    require(digest(actor_selected_raw)==selected['actor_selection_receipt_blake3']==complete['actor_selection_receipt_blake3']==actor_complete['selection_receipt_blake3'] and
        actor_selected['original_selection_blake3']==base['actor_selection_blake3'] and actor_selected['cases']==selected['cases'] and
        actor_selected['model']==base['evaluation_model_id'] and actor_selected['route']==base['evaluation_route'] and
        actor_selected['model_outcome_resampling'] is False and actor_selected['original_cohort_modified'] is False,
        'supplement_actor_selection_changed')
    expected={row['case_id']:row for row in base['case_evidence'] if row['domain']=='medxpertqa'}
    require(len(expected)==6 and {row['sandbox_id'] for row in selected['cases']}==set(expected) and
        selected['lineage']==actor_selected['lineage'], 'supplement_source_case_selection_changed')
    statuses={row['case_id']:row for row in complete['statuses']}
    require(len(statuses)==complete['case_count'] and set(statuses)<=set(expected), 'duplicate_or_foreign_supplement_judgment')
    native_rows={row['case_id']:row for row in actor_complete['trajectories']}
    require(len(native_rows)==actor_complete['case_count'] and set(native_rows)<=set(expected), 'duplicate_supplement_native_case')
    supplemental_entries=[]
    for case in selected['cases']:
        case_id=case['sandbox_id'];old=expected[case_id]
        lineage=next(row for row in selected['lineage'] if row['case_id']==case_id)
        require(lineage['original_trajectory_blake3']==old['original_actor_trajectory_blake3'] and
            lineage['original_model_requests']==0 and old['status']!='scored', 'supplement_would_resample_existing_score')
        prior=local_path(lineage['original_actor_relative_path']);prior_t=document(contained_file(prior,'trajectory.json'))
        prior_episode_raw=strict_json(contained_file(prior,'episode-receipt.json').read_bytes())
        prior_episode=verify_receipt(prior_episode_raw,public)
        require(prior_t['document_blake3']==lineage['original_trajectory_blake3'] and
            digest(prior_episode_raw)==lineage['original_episode_blake3'] and prior_t['codex_turn_status'] is None and
            prior_t['actual_model_requests']==prior_episode['actual_model_requests']==0 and not prior_episode['turns'],
            'supplement_original_exclusion_was_actual_inference')
        _,source,construction,_=bundle.case(case_id);rubric=CompiledRubric.from_document(source['reward_contract']['rubric_table'])
        require(case['stage']==source['stage']==rubric.stage==old['stage'] and
            case['domain']==source['domain']==rubric.domain==old['domain']=='medxpertqa' and
            case['record_blake3']==source['record_blake3'] and case['rubric_digest']==rubric.digest,
            'supplement_selected_stage_or_source_identity_changed')
        status=statuses.get(case_id);assessment=None;proof=[];pair_digest=None;reason='supplement_first_inference_not_completed'
        native=None;actual=None;case_status_hash=None
        if status is not None:
            status_path=contained_file(supplemental,'case-status/'+status['run_id']+'.json')
            require(verify_receipt(strict_json(status_path.read_bytes()),public)==status and status['purpose']=='supplemental_first_inference',
                'supplement_case_status_changed')
            case_status_hash=file_digest(status_path);native=actor_cohort/'trajectories'/status['run_id']
            actual=document(contained_file(native,'trajectory.json'))
            require(actual['document_blake3']==status['original_trajectory_blake3']==native_rows[case_id]['document_blake3'] and
                actual['run_id']==native_rows[case_id]['run_id'] and actual['case_id']==case_id and
                actual['source_record_blake3']==source['record_blake3'] and actual['source_rubric_digest']==rubric.digest and
                actual['model']==base['evaluation_model_id'] and actual['route']==base['evaluation_route'],
                'supplement_actual_actor_binding_changed')
            from .portable_public_projection import verify_authorization,NAME
            request=strict_json(contained_file(native,'turns/01/logical-request.json').read_bytes())
            authorization=strict_json(contained_file(native,'turns/01/'+NAME).read_bytes())
            verify_authorization(authorization,public_key=public,logical=request['logical_input'],base=request['base_instructions'],
                developer=request['developer_instructions'],catalog=request['public_tool_catalog'],construction=construction,
                **{key:actual[key] for key in ('bundle_blake3','source_record_blake3','source_rubric_digest','run_id','case_id')})
            require(digest(request['logical_input']['skills'])==base['selected_skill_catalog_blake3'], 'supplement_skill_catalog_changed')
            if status['status']=='pair_completed':
                pair_root=supplemental/'assessments'/status['run_id'];pair=read_adapter_document(contained_file(pair_root,'pair.json'))
                pair_digest=pair['document_blake3']
                require(pair_digest==status['pair_document_blake3'] and pair['rubric_digest']==rubric.digest and
                    pair['stage']==case['stage'] and pair['domain']=='medxpertqa' and pair['reward_admitted']==status['reward_admitted'],
                    'supplement_pair_summary_changed')
                if pair['reward_admitted']:
                    view=local_path(status['recovery_view_relative_path']) if status['recovery_view_relative_path'] else native
                    assessment,proof=_verify_admitted_pair(pair_root,view,rubric,pair,request['logical_input']['skills'],public)
                    reason=None
                else:
                    require(pair['reward_bps'] is None and status['reward_bps'] is None,'unadmitted_supplement_has_reward')
                    reason='independent_assessments_disagree' if pair['both_assessments_verified'] else 'independent_assessment_invalid_or_unavailable'
            else:
                require(status['reward_admitted'] is False and status['reward_bps'] is None,'failed_supplement_has_reward')
                reason=status.get('error_code') or status.get('error_type') or 'supplement_infrastructure_failure'
        supplemental_entries.append({'case_id':case_id,'domain':'medxpertqa','stage':case['stage'],
            'actor_run_id':actual['run_id'] if actual else None,
            'original_actor_trajectory_blake3':actual['document_blake3'] if actual else None,
            'status':'scored' if assessment else 'infrastructure_failure','missing_reason':reason,
            'source_rubric_digest':rubric.digest,'pair_document_blake3':pair_digest,
            'reward_bps':assessment.reward_bps if assessment else None,
            'normalized_source_rubric_reward':assessment.reward_bps/10000 if assessment else None,
            'score_document':assessment.to_document() if assessment else None,
            'case_status_file_blake3':case_status_hash,'provider_receipts':proof,
            'coverage_source':'separately_declared_first_inference_after_original_zero_request_exclusion',
            'original_preflight_exclusion':old,'original_zero_request_lineage':lineage})
    added={row['case_id']:row for row in supplemental_entries}
    coverage=[added.get(row['case_id'],row) for row in base['case_evidence']]
    identity=RoundIdentity(**base['typed_descriptive_proposal']['round_identity']);typed=[]
    for row in coverage:
        _,source,_,_=bundle.case(row['case_id']);rubric=CompiledRubric.from_document(source['reward_contract']['rubric_table'])
        score_document=row['score_document']
        score=rubric.score(score_document['item_scores_bps'],evaluation_id=score_document['evaluation_id']) if score_document else None
        typed.append(VerifiedStageEvaluation(row['case_id'],identity,rubric,
            EvaluationStatus.SCORED if score else EvaluationStatus.INFRASTRUCTURE_FAILURE,
            row['pair_document_blake3'] or row['case_status_file_blake3'] or 'unmeasured-first-inference',score))
    stages,target=select_fully_covered_stages(coverage,base['stages']['S1']['required_domains'])
    return {**base,'supplemental_judge_cohort_relative_path':str(supplemental.relative_to(ROOT)),
        'supplemental_judge_selection_file_blake3':file_digest(supplemental/'selection.json'),
        'supplemental_judge_cohort_file_blake3':file_digest(supplemental/'cohort.json'),
        'supplemental_actor_selection_file_blake3':file_digest(actor_cohort/'selection.json'),
        'supplemental_actor_cohort_file_blake3':file_digest(actor_cohort/'cohort.json'),
        'coverage_audit_version':'eva.portable-zero-inference-supplement-coverage.v1',
        'original_frozen_case_evidence':base['case_evidence'],'supplemental_case_evidence':supplemental_entries,
        'case_evidence':coverage,'original_frozen_stages':base['stages'],'stages':stages,
        'admitted_stage':target,'rl_launch_admitted':target is not None,'typed_descriptive_proposal':plan_stage_target(typed),
        'e2e':{'used_for_stage_selection':False,'cases':[row for row in coverage if row['stage']=='E2E']},
        'original_frozen_cohort_modified':False,'valid_or_failed_model_outcomes_resampled':False}


def create_stage_target_receipt(judge_cohort, output=DEFAULT_RECEIPT, *, key, supplemental_judge_cohort=None):
    analysis=(compute_stage_target_with_supplement(judge_cohort,supplemental_judge_cohort)
        if supplemental_judge_cohort is not None else compute_stage_target(judge_cohort))
    schema='eva.portable-paired-source-stage-target.v2' if supplemental_judge_cohort is not None else SCHEMA
    payload={'schema':schema,'admission_id':str(uuid4()),'created_at':timestamp(),**analysis}
    write_json(Path(output),Signer(Path(key)).sign(payload),exclusive=True)
    return payload


def verify_stage_target_receipt(path, *, expected_bundle_blake3, expected_index_blake3,
        expected_evaluation_model_id='nvidia/qwen/qwen3.6-27b'):
    """Reopen paired source evidence; return bindings only for an admitted stage.

    Raises ValueError if absent, invalid, stale, incompletely covered, or bound to
    another source/model/index. This performs no network calls and emits no reward.
    """
    bundle=Bundle(BUNDLE)
    require(bundle.manifest['document_blake3']==expected_bundle_blake3, 'launch_bundle_binding_changed')
    payload=verify_receipt(strict_json(Path(path).read_bytes()),bundle.manifest['fresh_execution_authority']['public_key_base64'])
    require(payload['schema'] in (SCHEMA,'eva.portable-paired-source-stage-target.v2') and payload['policy']==POLICY, 'stage_target_admission_schema_changed')
    if payload['schema']=='eva.portable-paired-source-stage-target.v2':
        reopened=compute_stage_target_with_supplement(local_path(payload['judge_cohort_relative_path']),local_path(payload['supplemental_judge_cohort_relative_path']))
    else:
        reopened=compute_stage_target(local_path(payload['judge_cohort_relative_path']))
    require(reopened=={k:v for k,v in payload.items() if k not in ('schema','admission_id','created_at')},
        'stage_target_actual_evidence_changed')
    require(payload['bundle_blake3']==expected_bundle_blake3 and
        payload['index_binding']['file_blake3']==expected_index_blake3 and
        payload['evaluation_model_id']==expected_evaluation_model_id and payload['rl_launch_admitted'] is True and
        payload['admitted_stage'] in STAGES and payload['stages'][payload['admitted_stage']]['admitted'] is True,
        'stage_target_not_admitted_for_this_launch')
    return payload
