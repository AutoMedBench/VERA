"""Opt-in source-authenticated MedXpertQA public input for future training.

This is a fresh one-thread permit, with no original-exclusion or first-inference
claim. It preserves the released warning, canonical tools and original rubric.
The sealed executor, ordinary Codex guard and qualification policy stay intact.
"""
from pathlib import Path
from uuid import uuid4

from .portable_rollout import ROOT, BUNDLE
from .portable_public_projection import _warning_projection, require
from .portable_public_projection import AuthenticatedPublicCodexRuntime
from evamed_portable.bundle import Bundle
from evamed_portable.runtime import FRESH_SEMANTICS
from evamed_portable.integrity import (byte_digest, digest, file_digest,
    strict_json, verify_receipt, write_json, timestamp)
from eva_agent.codex_runtime import CodexRole, CodexSandbox, CodexToolOffer
from eva_agent.codex_runtime import runtime as core
from eva_agent.pipeline.digests import canonical_value, blake3_hex

SCHEMA = 'eva.medresearch-authenticated-training-public-warning.v2'
NAME = 'training-public-projection-authorization.json'
CONSUMPTION_NAME = 'training-public-projection-consumption.json'


def sealed_initial_public(bundle, case_id, run_id):
    """Pure reconstruction of the sealed executor's initial public projection."""
    descriptor, row, construction, definitions = bundle.case(case_id)
    task = {
        'schema': 'eva.medresearch-public-task.v1', 'run_id': run_id,
        'domain': row['domain'], 'focus_stage': row['stage'],
        'completion_boundary': 'fresh_S1_through_S5',
        'source_stage_completion_boundary': next(item['content']['completion_boundary']
            for item in row['workspace_initial_state']['files'] if item['path'] == 'input/task-contract.json'),
        'task_brief': construction['task_brief'], 'source_sandbox': construction['sandbox'],
        'template_episode_id': construction['runtime']['template_episode_id'],
        'source_policy_budgets': construction['runtime']['policy_budgets'],
        'source_execution_limits': construction['runtime']['execution_limits'],
        'evidence_inventory': [{key: item[key] for key in
            ('evidence_id', 'bytes', 'blake3', 'upstream_sha256', 'content_kind')} for item in descriptor['evidence']],
        'artifact_contracts': {stage: {key: value for key, value in construction['runtime'][name].items()
            if key in {'json_schema', 'relative_path', 'min_bytes', 'max_bytes'}}
            for stage, name in (('S3', 's3_artifact'), ('S4', 's4_artifact'), ('S5', 'terminal'))},
        'primary_tool_definitions': list(definitions.values()),
        'public_evidence_path_pattern': 'input/evidence/<evidence_id>.json',
        'judge_only_contracts_visible': False,
    }
    return {'run_id': run_id, 'next_stage': 'S1', 'completed_stages': [], 'tool_calls': 0,
        'retrieved_evidence_ids': [], 'submission_attempted': False, 'task': task,
        'semantics': FRESH_SEMANTICS}


def _typed_options(options):
    # This exactly mirrors the core typed options commitment. Config values stay
    # gateway-owned; retaining only config_keys prevents credential duplication.
    return canonical_value({
        **{key: getattr(options, key) for key in ('role', 'model', 'provider', 'cwd',
            'sandbox', 'config_keys', 'ephemeral', 'service_name', 'service_tier')},
        'offered_tools': [{'canonical': tool.canonical_catalog_entry(),
            'sidecar': tool.sidecar_metadata_entry()} for tool in options.offered_tools],
        'base_instructions_blake3': None if options.base_instructions is None else blake3_hex(options.base_instructions),
        'developer_instructions_blake3': None if options.developer_instructions is None else blake3_hex(options.developer_instructions),
    })


def _typed_turn(value):
    return canonical_value({key: getattr(value, key) for key in
        ('sandbox', 'model', 'effort', 'summary', 'service_tier')})


def verify_authorization(authorization, *, public_key, logical, base, developer, catalog,
        construction, bundle_blake3, source_record_blake3, source_rubric_digest, run_id, case_id):
    payload = verify_receipt(authorization, public_key)
    bundle = Bundle(BUNDLE)
    descriptor, row, frozen_construction, definitions = bundle.case(case_id)
    require(bundle.manifest['document_blake3'] == bundle_blake3 and
        bundle.manifest['fresh_execution_authority']['public_key_base64'] == public_key and
        row['record_blake3'] == source_record_blake3 and
        row['reward_contract']['rubric_digest'] == source_rubric_digest and construction == frozen_construction,
        'training_public_source_or_original_rubric_changed')
    public = strict_json(logical['public_text'])
    require(public == sealed_initial_public(bundle, case_id, run_id), 'training_public_projection_not_exact_sealed_source')
    scope = _warning_projection(public, construction)
    offers = [CodexToolOffer(fully_qualified_name='eva_medresearch/' + value['name'],
        description=value['description'], input_schema=value['parameters'], parallel_safe=False,
        read_only=value['name'] == 'retrieve_frozen_evidence', allowed_stages=('S1','S2','S3','S4','S5','E2E'))
        for value in definitions.values()]
    expected_catalog = [{'name': value['name'], 'description': value['description'],
        'inputSchema': value['parameters']} for value in definitions.values()]
    options, turn = payload['typed_options'], payload['typed_turn_overrides']
    require(payload['schema'] == SCHEMA and payload['run_id'] == run_id and payload['case_id'] == case_id and
        payload['bundle_blake3'] == bundle_blake3 and payload['source_record_blake3'] == source_record_blake3 and
        payload['source_rubric_digest'] == source_rubric_digest and payload['source_rubric_modified'] is False and
        payload['logical_input_blake3'] == digest(logical) and payload['public_text_bytes_blake3'] == byte_digest(logical['public_text'].encode()) and
        payload['base_instructions_blake3'] == byte_digest((base or '').encode()) and
        payload['developer_instructions_blake3'] == byte_digest((developer or '').encode()) and
        payload['public_tool_catalog_blake3'] == digest(catalog) and catalog == expected_catalog and
        logical['offered_tools'] == canonical_value([offer.canonical_catalog_entry() for offer in offers]) and
        logical['offered_tool_metadata'] == canonical_value([offer.sidecar_metadata_entry() for offer in offers]) and
        logical['role'] == 'strong_actor' and logical['judge_only_context'] is None and logical['public_context'] == {} and
        logical['output_schema'] is None and logical['mentions'] == [] and payload['selected_skills'] == logical['skills'] and
        payload['warning_scope'] == scope and payload['wire_input_changed'] is False and
        payload['private_field_checks_retained'] is True and payload['permit_reusable'] is False and
        payload['original_exclusion_required'] is False and payload['first_inference_claimed'] is False and
        payload['rl_stage_admission_granted'] is False,
        'training_public_authorization_binding_changed')
    require(set(options) == {'role','model','provider','cwd','sandbox','config_keys','ephemeral',
        'service_name','service_tier','offered_tools','base_instructions_blake3','developer_instructions_blake3'} and
        set(turn) == {'sandbox','model','effort','summary','service_tier'} and
        options['role'] == logical['role'] and options['sandbox'] == 'read-only' and options['ephemeral'] is True and
        turn['sandbox'] in (None, 'read-only') and turn['model'] in (None, options['model']) and
        options['offered_tools'] == [{'canonical': a, 'sidecar': b} for a, b in
            zip(logical['offered_tools'], logical['offered_tool_metadata'])] and
        options['base_instructions_blake3'] == (None if base is None else blake3_hex(base)) and
        options['developer_instructions_blake3'] == (None if developer is None else blake3_hex(developer)) and
        payload['runtime_options_commitment'] == blake3_hex(options) and
        payload['runtime_input_commitment'] == blake3_hex({'logical_input': logical, **turn}) and
        payload['core_runtime_source_blake3'] == file_digest(core.__file__) and
        payload['public_policy_source_blake3'] == file_digest(__file__),
        'training_public_typed_runtime_binding_changed')
    return payload


def verify_consumption(consumption, authorization, *, public_key, native_thread_id, runtime_thread_id):
    permit = verify_receipt(authorization, public_key)
    used = verify_receipt(consumption, public_key)
    require(used['schema'] == 'eva.medresearch-training-public-permit-consumption.v2' and
        used['authorization_blake3'] == digest(authorization) and used['permit_id'] == permit['permit_id'] and
        used['run_id'] == permit['run_id'] and used['case_id'] == permit['case_id'] and
        used['native_thread_id'] == native_thread_id and used['runtime_thread_id'] == runtime_thread_id and
        used['runtime_options_commitment'] == permit['runtime_options_commitment'] and
        used['runtime_input_commitment'] == permit['runtime_input_commitment'] and
        used['fresh_unresumed_thread'] is True and used['consumed_before_provider_dispatch'] is True,
        'training_public_permit_actual_thread_changed')
    return used


class TrainingPublicWarningAuthority:
    schema = SCHEMA

    def __init__(self, runtime, audit):
        self.runtime, self.audit = runtime, Path(audit)
        self.permit = self.turn_dir = None
        self.consumed = False

    def runtime_catalog(self):
        return [{'name': value['name'], 'description': value['description'], 'inputSchema': value['parameters']}
            for value in self.runtime.definitions.values()]

    def authorize(self, options, value, turn_dir):
        require(self.permit is None and not self.consumed, 'training_public_permit_reused')
        public = strict_json(value.public_text)
        require(public == self.runtime.public_snapshot() and options.role is CodexRole.STRONG_ACTOR and
            options.sandbox is CodexSandbox.READ_ONLY and options.ephemeral and
            value.sandbox in (None, CodexSandbox.READ_ONLY) and value.model in (None, options.model) and
            value.judge_only_context is None and value.public_context == {} and value.output_schema is None and not value.mentions,
            'training_public_actor_boundary_changed')
        logical = canonical_value(core._logical_input(options, value))
        options_document, turn_document = _typed_options(options), _typed_turn(value)
        require(blake3_hex(options_document) == core._construction_options_blake3(options) and
            blake3_hex({'logical_input': logical, **turn_document}) == core._construction_input_blake3(options, value),
            'typed_Codex_commitment_definition_changed')
        payload = {'schema': SCHEMA, 'permit_id': str(uuid4()), 'run_id': public['run_id'],
            'case_id': self.runtime.row['sandbox_id'], 'bundle_blake3': self.runtime.bundle.manifest['document_blake3'],
            'source_record_blake3': self.runtime.row['record_blake3'],
            'source_rubric_digest': self.runtime.row['reward_contract']['rubric_digest'], 'source_rubric_modified': False,
            'logical_input_blake3': digest(logical), 'public_text_bytes_blake3': byte_digest(value.public_text.encode()),
            'base_instructions_blake3': byte_digest((options.base_instructions or '').encode()),
            'developer_instructions_blake3': byte_digest((options.developer_instructions or '').encode()),
            'public_tool_catalog_blake3': digest(self.runtime_catalog()), 'selected_skills': logical['skills'],
            'typed_options': options_document, 'typed_turn_overrides': turn_document,
            'runtime_options_commitment': core._construction_options_blake3(options),
            'runtime_input_commitment': core._construction_input_blake3(options, value),
            'warning_scope': _warning_projection(public, self.runtime.construction),
            'wire_input_changed': False, 'private_field_checks_retained': True, 'permit_reusable': False,
            'original_exclusion_required': False, 'first_inference_claimed': False, 'rl_stage_admission_granted': False,
            'core_runtime_source_blake3': file_digest(core.__file__), 'public_policy_source_blake3': file_digest(__file__)}
        receipt = self.runtime.signer.sign(payload)
        verify_authorization(receipt, public_key=self.runtime.signer.public, logical=logical,
            base=options.base_instructions, developer=options.developer_instructions,
            catalog=self.runtime_catalog(), construction=self.runtime.construction,
            **{key: payload[key] for key in ('bundle_blake3','source_record_blake3','source_rubric_digest','run_id','case_id')})
        self.turn_dir = Path(turn_dir)
        write_json(self.turn_dir / NAME, receipt, exclusive=True)
        self.permit = receipt
        return receipt

    def consume(self, state, value):
        require(self.permit is not None and not self.consumed and not state.handle.resumed and
            state.construction_permit is None, 'training_public_missing_fresh_permit')
        payload = self.permit['payload']
        require(core._construction_options_blake3(state.options) == payload['runtime_options_commitment'] and
            core._construction_input_blake3(state.options, value) == payload['runtime_input_commitment'] and
            strict_json(value.public_text) == self.runtime.public_snapshot(), 'training_public_actual_input_changed')
        verify_authorization(self.permit, public_key=self.runtime.signer.public,
            logical=canonical_value(core._logical_input(state.options, value)),
            base=state.options.base_instructions, developer=state.options.developer_instructions,
            catalog=self.runtime_catalog(), construction=self.runtime.construction,
            **{key: payload[key] for key in ('bundle_blake3','source_record_blake3','source_rubric_digest','run_id','case_id')})
        write_json(self.turn_dir / CONSUMPTION_NAME, self.runtime.signer.sign({
            'schema': 'eva.medresearch-training-public-permit-consumption.v2', 'consumed_at': timestamp(),
            'authorization_blake3': digest(self.permit), 'permit_id': payload['permit_id'],
            'run_id': payload['run_id'], 'case_id': payload['case_id'],
            'native_thread_id': state.handle.thread_id, 'runtime_thread_id': state.handle.runtime_thread_id,
            'runtime_options_commitment': payload['runtime_options_commitment'],
            'runtime_input_commitment': payload['runtime_input_commitment'],
            'fresh_unresumed_thread': True, 'consumed_before_provider_dispatch': True}), exclusive=True)
        self.consumed = True


class AuthenticatedTrainingPublicCodexRuntime(AuthenticatedPublicCodexRuntime):
    """The unchanged actor guard delegates only its exact-warning permit check."""

    def _validate_input(self, state, value):
        authority = self.public_warning_authority
        if not authority.consumed:
            require(authority.permit is not None and
                core._construction_options_blake3(state.options) == authority.permit['payload']['runtime_options_commitment'] and
                core._construction_input_blake3(state.options, value) == authority.permit['payload']['runtime_input_commitment'],
                'training_public_actual_initial_input_changed')
        return super()._validate_input(state, value)
