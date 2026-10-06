"""Authenticated admission of one exact released public warning.

The ordinary Codex guard remains unchanged. This integration admits only an
initial sealed MedXpertQA public projection whose sole lexical match is the
source instruction to avoid requesting a hidden answer key. Private structured
fields, judge context, altered source text, or later-turn reuse still fail.
The actual Codex wire input is never edited or re-encoded by this policy.
"""
from copy import deepcopy
from pathlib import Path

from .portable_rollout import ROOT
from evamed_portable.integrity import (byte_digest, canonical, digest, file_digest,
    strict_json, verify_receipt, write_json)
from eva_agent.codex_runtime import CodexRuntime, CodexRuntimeError, CodexSandbox
from eva_agent.codex_runtime import runtime as core
from eva_agent.pipeline.digests import canonical_value

SCHEMA='eva.medresearch-authenticated-public-warning.v1'
WARNING='without requesting a hidden answer key'
NAME='public-projection-authorization.json'


def require(value,reason):
    if not value:raise CodexRuntimeError(reason)


def _warning_projection(public, construction):
    core._reject_private_projection(public)
    require(public['task']['domain']=='medxpertqa' and public['next_stage']=='S1' and
        public['completed_stages']==[] and public['retrieved_evidence_ids']==[] and
        public['submission_attempted'] is False and public['tool_calls']==0 and
        public['task']['task_brief']==construction['task_brief'], 'public_warning_source_or_initial_state_changed')
    brief=public['task']['task_brief'];objective=brief['objective']
    require(objective.count(WARNING)==1, 'public_warning_source_phrase_changed')
    expected=deepcopy(public);expected['task']['task_brief']['objective']=objective.replace(WARNING,'without requesting forbidden evaluator material',1)
    require(core._PRIVATE_TEXT.search(canonical(expected).decode()) is None and
        [m.group() for m in core._PRIVATE_TEXT.finditer(canonical(public).decode())]==['answer key'],
        'public_warning_contains_other_private_reference')
    return {'path':['task','task_brief','objective'],'source_field_blake3':byte_digest(objective.encode()),
        'matched_text':'answer key','public_warning':WARNING,'source_text_altered':False}


def equivalent_source_logical(logical):
    """Only the two explicit fresh execution UUID fields are alpha-renamed."""
    value=deepcopy(logical);public=strict_json(value['public_text'])
    run=public['run_id'];require(public['task']['run_id']==run,'public_warning_run_identity_changed')
    public['run_id']=public['task']['run_id']='FRESH-RUN-ID'
    value['public_text']=canonical(public).decode()
    return value


def verify_authorization(authorization, *, public_key, logical, base, developer, catalog, construction,
        bundle_blake3, source_record_blake3, source_rubric_digest, run_id, case_id):
    payload=verify_receipt(authorization,public_key)
    public=strict_json(logical['public_text']);scope=_warning_projection(public,construction)
    require(payload['schema']==SCHEMA and payload['run_id']==run_id and payload['case_id']==case_id and
        public['run_id']==run_id and payload['bundle_blake3']==bundle_blake3 and
        payload['source_record_blake3']==source_record_blake3 and payload['source_rubric_digest']==source_rubric_digest and
        payload['logical_input_blake3']==digest(logical) and payload['public_text_bytes_blake3']==byte_digest(logical['public_text'].encode()) and
        payload['base_instructions_blake3']==byte_digest((base or '').encode()) and
        payload['developer_instructions_blake3']==byte_digest((developer or '').encode()) and
        payload['public_tool_catalog_blake3']==digest(catalog) and payload['selected_skills']==logical['skills'] and
        payload['warning_scope']==scope and payload['guard_changes']=='none; separate source-authenticated initial public warning permit' and
        payload['wire_input_changed'] is False and payload['private_field_checks_retained'] is True and
        payload['source_projection_equivalence_blake3']==digest(equivalent_source_logical(logical)),
        'authenticated_public_warning_binding_changed')
    original=(ROOT/payload['original_actor_relative_path']).resolve(strict=True)
    require(original.is_relative_to(ROOT),'public_warning_original_path_changed')
    previous=strict_json((original/'trajectory.json').read_bytes())
    original_episode_raw=strict_json((original/'episode-receipt.json').read_bytes())
    original_episode=verify_receipt(original_episode_raw,public_key)
    original_launch_raw=strict_json((original/'turns/01/launch.json').read_bytes())
    original_launch=verify_receipt(original_launch_raw,public_key)
    original_request=strict_json((original/'turns/01/logical-request.json').read_bytes())
    require(previous['document_blake3']==digest({k:v for k,v in previous.items() if k!='document_blake3'}) and
        previous['document_blake3']==payload['original_trajectory_blake3'] and
        digest(original_episode_raw)==previous['episode_receipt_blake3']==payload['original_episode_blake3'] and
        digest(original_launch_raw)==payload['original_launch_blake3'] and
        original_launch['logical_input_blake3']==digest(original_request['logical_input']) and
        original_launch['base_instructions_blake3']==byte_digest((original_request['base_instructions'] or '').encode()) and
        original_launch['developer_instructions_blake3']==byte_digest((original_request['developer_instructions'] or '').encode()) and
        original_launch['public_tool_catalog_blake3']==digest(original_request['public_tool_catalog']) and
        previous['actual_model_requests']==original_episode['actual_model_requests']==payload['original_model_requests']==0 and
        previous['actual_tool_attempts']==original_episode['actual_tool_attempts']==0 and previous['codex_turn_status'] is None and
        not original_episode['turns'] and previous['case_id']==case_id and previous['source_record_blake3']==source_record_blake3 and
        original_request['base_instructions']==base and original_request['developer_instructions']==developer and
        original_request['public_tool_catalog']==catalog and
        equivalent_source_logical(original_request['logical_input'])==equivalent_source_logical(logical) and
        not list((original/'provider/requests').glob('*.json')) and payload['first_actual_inference_attempt'] is True and
        payload['core_runtime_source_blake3']==file_digest(core.__file__) and payload['public_policy_source_blake3']==file_digest(__file__),
        'public_warning_original_zero_inference_lineage_changed')
    return payload


class PublicWarningAuthority:
    def __init__(self,runtime,audit,original_actor):
        self.runtime,self.audit=runtime,Path(audit)
        self.original=Path(original_actor).resolve(strict=True)
        require(self.original.is_relative_to(ROOT),'public_warning_lineage_outside_workspace')
        self.permit=None;self.consumed=False

    def authorize(self,options,value,turn_dir):
        require(self.permit is None and not self.consumed,'public_warning_permit_reused')
        logical=canonical_value(core._logical_input(options,value));public=strict_json(value.public_text)
        require(public==self.runtime.public_snapshot() and options.role.is_actor and
            options.sandbox is CodexSandbox.READ_ONLY and options.ephemeral and value.judge_only_context is None and
            value.public_context=={} and value.output_schema is None and not value.mentions,
            'public_warning_actor_boundary_changed')
        scope=_warning_projection(public,self.runtime.construction)
        original=strict_json((self.original/'trajectory.json').read_bytes())
        old_episode_raw=strict_json((self.original/'episode-receipt.json').read_bytes())
        old_episode=verify_receipt(old_episode_raw,self.runtime.signer.public)
        old_launch_raw=strict_json((self.original/'turns/01/launch.json').read_bytes())
        old_launch=verify_receipt(old_launch_raw,self.runtime.signer.public)
        old_request=strict_json((self.original/'turns/01/logical-request.json').read_bytes())
        require(original['document_blake3']==digest({k:v for k,v in original.items() if k!='document_blake3'}) and
            digest(old_episode_raw)==original['episode_receipt_blake3'] and
            original['actual_model_requests']==old_episode['actual_model_requests']==0 and
            original['actual_tool_attempts']==old_episode['actual_tool_attempts']==0 and original['codex_turn_status'] is None and
            not old_episode['turns'] and original['case_id']==self.runtime.row['sandbox_id'] and
            original['source_record_blake3']==self.runtime.row['record_blake3'] and
            old_launch['logical_input_blake3']==digest(old_request['logical_input']) and
            old_request['base_instructions']==options.base_instructions and old_request['developer_instructions']==options.developer_instructions and
            old_request['public_tool_catalog']==self.runtime_catalog() and
            equivalent_source_logical(old_request['logical_input'])==equivalent_source_logical(logical),
            'public_warning_not_equivalent_zero_inference_source')
        require(not list((self.original/'provider/requests').glob('*.json')),
            'public_warning_original_provider_request_exists')
        payload={'schema':SCHEMA,'run_id':public['run_id'],'case_id':self.runtime.row['sandbox_id'],
            'bundle_blake3':self.runtime.bundle.manifest['document_blake3'],'source_record_blake3':self.runtime.row['record_blake3'],
            'source_rubric_digest':self.runtime.row['reward_contract']['rubric_digest'],
            'logical_input_blake3':digest(logical),'public_text_bytes_blake3':byte_digest(value.public_text.encode()),
            'runtime_input_commitment':core._construction_input_blake3(options,value),
            'base_instructions_blake3':byte_digest((options.base_instructions or '').encode()),
            'developer_instructions_blake3':byte_digest((options.developer_instructions or '').encode()),
            'public_tool_catalog_blake3':digest(self.runtime_catalog()),'selected_skills':logical['skills'],'warning_scope':scope,
            'source_projection_equivalence_blake3':digest(equivalent_source_logical(logical)),
            'original_actor_relative_path':str(self.original.relative_to(ROOT)),
            'original_trajectory_blake3':original['document_blake3'],'original_episode_blake3':digest(old_episode_raw),
            'original_launch_blake3':digest(old_launch_raw),'original_model_requests':0,'first_actual_inference_attempt':True,
            'guard_changes':'none; separate source-authenticated initial public warning permit',
            'wire_input_changed':False,'private_field_checks_retained':True,'permit_reusable':False,
            'core_runtime_source_blake3':file_digest(core.__file__),'public_policy_source_blake3':file_digest(__file__)}
        receipt=self.runtime.signer.sign(payload);write_json(Path(turn_dir)/NAME,receipt,exclusive=True)
        self.permit=(core._construction_input_blake3(options,value),core._construction_options_blake3(options),receipt)
        return receipt

    def runtime_catalog(self):
        return [{'name':x['name'],'description':x['description'],'inputSchema':x['parameters']} for x in self.runtime.definitions.values()]

    def consume(self,state,value):
        require(self.permit is not None and not self.consumed and not state.handle.resumed and
            state.construction_permit is None,'public_warning_missing_fresh_permit')
        logical=canonical_value(core._logical_input(state.options,value))
        require(core._construction_input_blake3(state.options,value)==self.permit[0] and core._construction_options_blake3(state.options)==self.permit[1],
            'public_warning_actual_input_changed')
        verify_authorization(self.permit[2],public_key=self.runtime.signer.public,logical=logical,
            base=state.options.base_instructions,developer=state.options.developer_instructions,catalog=self.runtime_catalog(),
            construction=self.runtime.construction,bundle_blake3=self.runtime.bundle.manifest['document_blake3'],
            source_record_blake3=self.runtime.row['record_blake3'],source_rubric_digest=self.runtime.row['reward_contract']['rubric_digest'],
            run_id=self.runtime.public_snapshot()['run_id'],case_id=self.runtime.row['sandbox_id'])
        self.consumed=True


class AuthenticatedPublicCodexRuntime(CodexRuntime):
    def __init__(self,backend,*,public_warning_authority):
        super().__init__(backend);self.public_warning_authority=public_warning_authority

    def _validate_input(self,state,value):
        if not state.options.role.is_actor or core._PRIVATE_TEXT.search(value.public_text) is None:
            return super()._validate_input(state,value)
        if value.judge_only_context is not None:
            raise CodexRuntimeError('actor cannot receive judge-only context')
        require(self._construction_input_policy is None and state.construction_permit is None,
            'public_warning_cannot_use_construction_capability')
        core._reject_private_projection(value.public_context)
        if value.output_schema is not None:core._reject_private_projection(value.output_schema,path='actor_output_schema')
        selected=tuple(skill.skill_id for skill in value.skills)
        require(len(selected)==len(set(selected)),'selected skill IDs must be unique')
        self.public_warning_authority.consume(state,value)
        # All checks from the ordinary actor validator remain, with only the
        # signed exact-source lexical match admitted. run_turn receives value unchanged.
