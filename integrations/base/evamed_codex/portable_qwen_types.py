"""Explicit Qwen XML type decoding bound to the new portable host handlers.

Canonical tool offers are unchanged. Only supplied values of declared fresh
handler fields are decoded. No defaults, field insertion, shape repair, or
historical handler equivalence is claimed. Raw visible tool arguments and each
projection are retained; hidden provider reasoning is never persisted here.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

from eva_agent.training.qwen_tool_types import HostArgumentProjector
from evamed_portable.integrity import digest, file_digest, write_json
from evamed_portable.runtime import Runtime, FRESH_SEMANTICS
from .benchmark_provider import setup_provider

VERSION='eva.medresearch-fresh-qwen-host-types.v1'
FRESH_FIELD_TYPES={
    'materialize_plan':{'type':'object','properties':{'objective':{'type':'string'},
        'steps':{'type':'array','items':{'type':'object','properties':{'stage':{'type':'string'},'action':{'type':'string'}}}}}},
    'materialize_evidence_selection':{'type':'object','properties':{'evidence_ids':{'type':'array','items':{'type':'string'}},
        'rationale':{'type':'string'}}},
}


class FreshArgumentProjector(HostArgumentProjector):
    def _name(self, wire_name):
        for name in self.public_tools:
            if wire_name in {name,'mcp__eva_medresearch__'+name,'eva_medresearch__'+name,'eva_medresearch/'+name}:
                return name
        return None


def fresh_argument_projector(config_or_runtime):
    if isinstance(config_or_runtime,Runtime): runtime=config_or_runtime
    else:
        config=config_or_runtime
        runtime=Runtime(bundle=config['bundle'],run_root=config['portable_runtime_run_root'],signer=config['key'],
                        bwrap=config['bwrap'],runtime_root=config['runtime_root'])
    source=Path(__import__('evamed_portable.runtime',fromlist=['Runtime']).__file__).resolve()
    expected=next(x for x in runtime.bundle.manifest['runtime_code'] if x['path']=='runtime/evamed_portable/runtime.py')
    if file_digest(source)!=expected['blake3']:
        raise ValueError('fresh_type_binding_actual_handler_source_differs')
    public={name:{'name':name,'description':runtime.definitions[name]['description'],
                  'input_schema':deepcopy(runtime.definitions[name]['parameters'])} for name in FRESH_FIELD_TYPES}
    projector=FreshArgumentProjector(public_tools=public,host_schemas=FRESH_FIELD_TYPES,provenance={
        'schema':VERSION,'binding_authority':'fresh versioned portable effect handlers; not historical validators',
        'bundle_blake3':runtime.bundle.manifest['document_blake3'],'fresh_semantics_blake3':digest(FRESH_SEMANTICS),
        'actual_fresh_handler_source_blake3':file_digest(source),'fresh_type_binding_source_blake3':file_digest(__file__),
        'handler_symbols':{'materialize_plan':'Runtime._materialize_plan','materialize_evidence_selection':'Runtime._materialize_evidence_selection'},
        'source_canonical_schemas_changed':False,'historical_schema_loader_used':False,
        'provider_boundary':'after native XML parsing, before Codex tool invocation',
        'known_array_fields':['materialize_plan.steps','materialize_evidence_selection.evidence_ids']})
    return projector


class FreshTypedChatTransport:
    def __init__(self,upstream,projector,audit,signer):
        self.upstream,self.projector,self.audit,self.signer=upstream,projector,Path(audit),signer
        self.audit.mkdir(parents=True,exist_ok=True,mode=0o700)
        write_json(self.audit/'type-binding.json',signer.sign(projector.binding))

    def __call__(self,binding,body):
        from eva_agent.codex_providers.adapter import _UpstreamOutcome
        from evamed_portable.integrity import byte_digest
        tools=body.get('tools') or []
        offered=self.projector.validate_tools(tools)
        outcome=self.upstream(binding,body)
        if outcome.status!=200:return outcome
        response=json.loads(outcome.body);projections=[];visible_before=[];visible_after=[]
        for choice in response.get('choices') or []:
            message=choice.get('message') or {}
            visible_before.append({k:deepcopy(message[k]) for k in ('role','content','tool_calls') if k in message})
            choice['message'],audit=self.projector.project_message(message,tools)
            projections.append(audit)
            visible_after.append({k:deepcopy(choice['message'][k]) for k in ('role','content','tool_calls') if k in choice['message']})
        projected=json.dumps(response,ensure_ascii=False,allow_nan=False).encode()
        receipt=self.signer.sign({'schema':'eva.medresearch-visible-Qwen-argument-projection.v1',
            'projection_id':str(uuid4()),'binding_blake3':self.projector.binding_blake3,
            'request_blake3':digest(body),'raw_response_blake3':byte_digest(outcome.body),
            'projected_response_blake3':byte_digest(projected),'offered_host_bound_tools':offered,
            'visible_messages_before':visible_before,'visible_messages_after':visible_after,'projections':projections,
            'raw_visible_tool_arguments_retained':True,'raw_response_bytes_persisted':False,
            'private_reasoning_retained':False,'required_or_missing_fields_repaired':False,'public_tool_schema_changed':False})
        write_json(self.audit/(receipt['payload']['projection_id']+'.json'),receipt,exclusive=True)
        return _UpstreamOutcome(outcome.status,projected,outcome.latency_ms)


@contextmanager
def portable_provider(config,run_root,token):
    from evamed_portable.integrity import Signer
    projector=fresh_argument_projector(config)
    with setup_provider(config,run_root,token,upstream_wrapper=lambda upstream:FreshTypedChatTransport(
            upstream,projector,run_root/'argument-projections',Signer(config['key']))) as result:
        yield result
