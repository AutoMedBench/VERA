"""Retain actual native deadline evidence without relaxing Codex receipts.

This backend observer does not change notifications, interruption ordering,
model inputs, or canonical tools. A partial capture is a separate evidence type:
an unfinished native call never acquires a fabricated result or completion.
"""
from __future__ import annotations

import base64
import asyncio
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from blake3 import blake3
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from eva_agent.codex_runtime import runtime as core
from eva_agent.codex_runtime import backend as backend_source
from eva_agent.pipeline.digests import canonical_value, blake3_hex

SCHEMA = 'eva.benchmark-observed-native-turn-capture.v1'
SIGNATURE_SCHEMA = 'eva.benchmark-native-capture-signature.v1'
BINDING_KEYS = {'run_id','track','source_revision','launch_file_blake3','codex_binary_blake3',
    'logical_input_blake3','tool_catalog_blake3','deadline_monotonic','max_seconds'}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def _digest(value):
    return blake3(_canonical(value)).hexdigest()


def _file_digest(path):
    return blake3(Path(path).read_bytes()).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key,value in items:
            _require(key not in result,'duplicate_native_capture_json_key')
            result[key] = value
        return result
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_native_capture_json')))


def _write_once(path, value):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with path.open('xb') as stream:
        stream.write(_canonical(value))


class CaptureSigner:
    """Fresh in-memory authority; callers freeze its public key before launch."""
    def __init__(self):
        self._key = Ed25519PrivateKey.generate()
        self.public_key_base64 = base64.b64encode(self._key.public_key().public_bytes(
            serialization.Encoding.Raw,serialization.PublicFormat.Raw)).decode()
        self.public = self.public_key_base64

    def sign(self,payload):
        return {'schema':SIGNATURE_SCHEMA,'public_key_base64':self.public_key_base64,
            'payload':payload,'payload_blake3':_digest(payload),
            'signature_base64':base64.b64encode(self._key.sign(_canonical(payload))).decode(),
            'historical_authority_claimed':False}


def _verify(envelope, public_key):
    _require(set(envelope)=={'schema','public_key_base64','payload','payload_blake3',
        'signature_base64','historical_authority_claimed'} and envelope['schema']==SIGNATURE_SCHEMA and
        envelope['public_key_base64']==public_key and envelope['historical_authority_claimed'] is False and
        envelope['payload_blake3']==_digest(envelope['payload']),'native_capture_signature_binding_changed')
    Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key,validate=True)).verify(
        base64.b64decode(envelope['signature_base64'],validate=True),_canonical(envelope['payload']))
    return envelope['payload']


def verify_signed_payload(envelope, public_key):
    return _verify(envelope, public_key)


def write_signed_payload(path, payload, *, signer):
    """Write the exact signed envelope, without adapter document wrapping."""
    envelope = signer.sign(payload)
    _write_once(Path(path), envelope)
    return envelope


def _binding(value):
    _require(set(value)==BINDING_KEYS,'native_capture_binding_fields_changed')
    _require(all(isinstance(value[key],str) and value[key] for key in BINDING_KEYS-{'deadline_monotonic','max_seconds'}),
        'native_capture_binding_type_changed')
    for key in ('launch_file_blake3','codex_binary_blake3','logical_input_blake3','tool_catalog_blake3'):
        _require(len(value[key])==64 and all(c in '0123456789abcdef' for c in value[key]),'native_capture_binding_digest_changed')
    _require(all(type(value[key]) in (int,float) and math.isfinite(value[key]) and value[key]>0
        for key in ('deadline_monotonic','max_seconds')),'native_capture_deadline_not_finite')
    return deepcopy(value)


def _summarize(rows, binding, offered_names):
    tools,terminals,interrupts,errors = {},[],[],[]
    for row in rows:
        kind = row['kind']
        if kind=='notification':
            event = row['notification']; method = event['method']; payload = event['payload']
            item = core._item(payload)
            if method in ('item/started','item/completed') and item is not None and core._is_tool(item):
                identifier = item.get('id')
                if not isinstance(identifier,str) or not identifier:
                    errors.append('missing_native_tool_item_id');continue
                state = tools.setdefault(identifier,{'first_sequence':row['sequence'],'lifecycle':[],'item':None})
                state['lifecycle'].append(method);state['item']=item
                if state['lifecycle'] not in (['item/started'],['item/completed'],['item/started','item/completed']):
                    errors.append('duplicate_or_reordered_native_tool_lifecycle')
            if method=='turn/completed':
                terminals.append({'status':core._turn_status(payload),'sequence':row['sequence'],
                    'observed_monotonic':row['observed_monotonic'],'notification_blake3':_digest(event)})
        elif kind.startswith('interrupt_'):
            interrupts.append(row)
        elif kind=='stream_error':
            errors.append('native_stream_error:'+row['error_type'])
    completed,pending = [],[]
    for identifier,state in sorted(tools.items(),key=lambda pair:pair[1]['first_sequence']):
        item=state['item']; tool_type=core._item_type(item)
        row={'upstream_item_id':identifier,'tool_type':tool_type,'first_sequence':state['first_sequence'],
            'mcp_server':item.get('server') if tool_type=='mcpToolCall' else None,
            'mcp_tool':item.get('tool') if tool_type=='mcpToolCall' else None,
            'name':core._tool_name(item),'arguments':canonical_value(core._tool_arguments(item)),
            'lifecycle':state['lifecycle'],'native_status':core._status(item),
            'native_item_blake3':_digest(item),'model_observed_result':state['lifecycle'][-1]=='item/completed'}
        row['fully_qualified_name']=(row['mcp_server']+'/'+row['mcp_tool']
            if isinstance(row['mcp_server'],str) and isinstance(row['mcp_tool'],str) else None)
        if tool_type!='mcpToolCall' or (row['fully_qualified_name'] not in offered_names and
            core.codex_core_mcp_resource_operation(server=row['mcp_server'],tool=row['mcp_tool'],offered_mcp_tool_names=set(offered_names)) is None):
            errors.append('unoffered_native_tool')
        if state['lifecycle'][-1]=='item/completed':
            row['actual_native_output']=canonical_value(core._tool_output(item));completed.append(row)
        else:
            row.update(actual_native_output=None,completion_fabricated=False);pending.append(row)
    requests=[row for row in interrupts if row['kind']=='interrupt_requested']
    acks=[row for row in interrupts if row['kind']=='interrupt_acknowledged']
    failures=[row for row in interrupts if row['kind']=='interrupt_failed']
    terminal=terminals[0] if len(terminals)==1 else None
    notifications=[row for row in rows if row['kind']=='notification']
    terminal_last=bool(terminal and notifications and notifications[-1]['notification']['method']=='turn/completed')
    budget_proven=bool(len(requests)==len(acks)==1 and not failures and
        requests[0]['observed_monotonic']>=binding['deadline_monotonic'] and
        acks[0]['observed_monotonic']>=requests[0]['observed_monotonic'] and terminal and
        terminal['observed_monotonic']>=requests[0]['observed_monotonic'] and terminal['status']=='interrupted')
    return {'actual_native_turn_observed':bool(notifications),'actual_terminal':terminal,
        'terminal_is_last_native_notification':terminal_last,'complete_calls':completed,'pending_calls':pending,
        'actual_notification_count':len(notifications),'protocol_errors':errors,'interrupt_requested_count':len(requests),
        'interrupt_acknowledged_count':len(acks),'interrupt_failure_count':len(failures),
        'local_consumer_cleanup_after_terminal_count':sum(row['kind']=='consumer_cleanup_after_terminal' for row in rows),
        'deadline_interruption_proven':budget_proven,
        'partial_deadline_terminal_admissible':bool(budget_proven and terminal_last and pending and not errors),
        'ordinary_CodexTurnReceipt_claimed':False,'model_observed_pending_host_results':False,
        'host_quiescence_proven':False,'workspace_finality_proven':False,'reward_admitted':False}


class DeadlineCaptureBackend:
    def __init__(self,backend,*,audit_root,signer,binding):
        self.backend,self.audit_root,self.signer = backend,Path(audit_root),signer
        self.binding = _binding(binding)
        self.receipt_paths = []
        self.audit_root.mkdir(parents=True,exist_ok=True,mode=0o700)

    @property
    def sdk_version(self):return self.backend.sdk_version
    @property
    def server_version(self):return self.backend.server_version
    @property
    def receipt_path(self):return self.receipt_paths[-1] if self.receipt_paths else None
    async def open(self):return await self.backend.open()
    async def close(self):return await self.backend.close()

    async def start_thread(self,options):
        _require(options.role.is_actor,'native_deadline_capture_requires_actor')
        thread = await self.backend.start_thread(options)
        return _CapturedThread(self,thread,options,False)

    async def resume_thread(self,thread_id,options):
        _require(options.role.is_actor,'native_deadline_capture_requires_actor')
        thread = await self.backend.resume_thread(thread_id,options)
        return _CapturedThread(self,thread,options,True)


class _CapturedThread:
    def __init__(self,owner,thread,options,resumed):
        self.owner,self.thread,self.options = owner,thread,options
        self.resumed = resumed
        self.id = thread.id

    async def turn(self,items,options):
        actual = await self.thread.turn(items,options)
        return _CapturedTurn(self.owner,actual,self.id,self.options,items,options,self.resumed)


class _CapturedTurn:
    def __init__(self,owner,turn,thread_id,thread_options,items,turn_options,resumed):
        self.owner,self.turn,self.id,self.thread_id = owner,turn,str(turn.id),str(thread_id)
        self.root = owner.audit_root/str(uuid4())
        self.rows,self.previous,self.started,self.finished = [],None,False,False
        self.stream_running,self.stream_closed,self.interrupts_in_flight = False,False,0
        self.start = {'schema':'eva.benchmark-native-capture-start.v1','binding':owner.binding,
            'capture_id':self.root.name,'native_thread_id':self.thread_id,'native_turn_id':self.id,
            'thread_resumed':resumed,'role':thread_options.role.value,
            'model':turn_options.model or thread_options.model,'provider':thread_options.provider,
            'offered_mcp_tool_names':[tool.fully_qualified_name for tool in thread_options.offered_tools],
            'actual_turn_start_rpc_returned':True,'sdk_version':owner.sdk_version,'server_version':owner.server_version,
            'typed_thread_options_blake3':core._construction_options_blake3(thread_options),
            'actual_backend_input_items_blake3':blake3_hex(canonical_value(items)),
            'actual_backend_turn_options':canonical_value(turn_options),
            'core_runtime_source_blake3':_file_digest(core.__file__),
            'core_backend_source_blake3':_file_digest(backend_source.__file__),
            'capture_source_blake3':_file_digest(__file__),
            'created_monotonic':time.monotonic(),'created_at_ns':time.time_ns(),
            'private_reasoning_retained':False}

    def _ensure_started(self):
        if not self.started:
            self.root.mkdir(parents=True,mode=0o700)
            _write_once(self.root/'start.json',self.owner.signer.sign(self.start))
            self.started=True

    def _record(self,kind,**fields):
        _require(not self.finished,'native_capture_already_sealed')
        self._ensure_started()
        row={'sequence':len(self.rows),'kind':kind,'native_thread_id':self.thread_id,'native_turn_id':self.id,
            'previous_record_blake3':self.previous,'observed_monotonic':time.monotonic(),
            'observed_at_ns':time.time_ns(),**fields}
        envelope=self.owner.signer.sign(row)
        with (self.root/'events.jsonl').open('ab') as stream:stream.write(_canonical(envelope))
        self.rows.append(row);self.previous=_digest(envelope)

    def _record_stream_failure(self, exc):
        last=next((row['notification'] for row in reversed(self.rows) if row['kind']=='notification'),None)
        task=asyncio.current_task();cancelling=task.cancelling() if task is not None else 0
        local_cleanup=isinstance(exc,GeneratorExit) or (isinstance(exc,asyncio.CancelledError) and cancelling>0)
        kind=('consumer_cleanup_after_terminal' if local_cleanup and last and last['method']=='turn/completed'
            else 'stream_error')
        self._record(kind,error_type=type(exc).__name__,local_task_cancellation_requests=cancelling)

    async def interrupt(self):
        self.interrupts_in_flight += 1
        self._record('interrupt_requested')
        try:
            result=await self.turn.interrupt()
        except BaseException as exc:
            self._record('interrupt_failed',error_type=type(exc).__name__)
            raise
        else:
            self._record('interrupt_acknowledged')
            return result
        finally:
            self.interrupts_in_flight -= 1
            self._seal_if_ready()

    def _seal_if_ready(self):
        if self.finished or not self.stream_closed or self.interrupts_in_flight:
            return
        self._record('capture_sealed')
        _write_once(self.root/'capture.json',self.owner.signer.sign({
            'schema':SCHEMA,'binding':self.owner.binding,'capture_id':self.root.name,
            'native_thread_id':self.thread_id,'native_turn_id':self.id,
            'start_file_blake3':_file_digest(self.root/'start.json'),
            'events_file_blake3':_file_digest(self.root/'events.jsonl'),'event_records':len(self.rows),
            'last_record_blake3':self.previous,'summary':_summarize(self.rows,self.owner.binding,self.start['offered_mcp_tool_names']),
            'core_receipt_guard_modified':False,'native_notifications_modified':False,
            'native_notifications_synthesized':False,'private_reasoning_retained':False}))
        self.finished=True
        self.owner.receipt_paths.append(self.root/'capture.json')

    async def stream(self):
        _require(not self.stream_running and not self.finished,'native_capture_stream_reused')
        self.stream_running=True
        source=self.turn.stream()
        try:
            async for notification in source:
                method=getattr(notification,'method',None)
                raw=core._to_plain(getattr(notification,'payload',None))
                _require(isinstance(method,str) and method and isinstance(raw,dict),'native_capture_notification_malformed')
                routed_thread,routed_turn=core._route(raw,self.thread_id,self.id)
                projected,redacted=core._event_payload(method,raw)
                self._record('notification',notification={'method':method,'thread_id':routed_thread,
                    'turn_id':routed_turn,'payload':canonical_value(projected),'content_redacted':redacted})
                yield notification
        except BaseException as exc:
            self._record_stream_failure(exc)
            raise
        finally:
            close=getattr(source,'aclose',None)
            if callable(close):
                try:
                    await close()
                except BaseException as exc:
                    self._record_stream_failure(exc)
            self._ensure_started()
            self._record('stream_closed')
            self.stream_closed=True
            self._seal_if_ready()


def verify_capture(path,*,public_key,expected_binding):
    """Reopen actual signed bytes; host cleanup/admission is a separate verifier."""
    path=Path(path);root=path.parent;binding=_binding(expected_binding)
    _require(path.name=='capture.json' and not any(p.is_symlink() for p in (path,root,root/'start.json',root/'events.jsonl')),
        'native_capture_paths_changed')
    capture=_verify(_json(path.read_bytes()),public_key)
    start=_verify(_json((root/'start.json').read_bytes()),public_key)
    _require(capture['schema']==SCHEMA and start['schema']=='eva.benchmark-native-capture-start.v1' and
        capture['binding']==start['binding']==binding and capture['capture_id']==start['capture_id']==root.name and
        capture['native_thread_id']==start['native_thread_id'] and capture['native_turn_id']==start['native_turn_id'] and
        start['actual_turn_start_rpc_returned'] is True and start['private_reasoning_retained'] is False and
        capture['start_file_blake3']==_file_digest(root/'start.json') and
        capture['events_file_blake3']==_file_digest(root/'events.jsonl') and
        capture['core_receipt_guard_modified'] is False and capture['native_notifications_modified'] is False and
        capture['native_notifications_synthesized'] is False and capture['private_reasoning_retained'] is False,
        'native_capture_source_or_manifest_changed')
    rows=[];previous=None;last_time=start['created_monotonic']
    for line in (root/'events.jsonl').read_bytes().splitlines():
        envelope=_json(line);row=_verify(envelope,public_key)
        _require(row['sequence']==len(rows) and row['previous_record_blake3']==previous and
            row['native_thread_id']==capture['native_thread_id'] and row['native_turn_id']==capture['native_turn_id'] and
            type(row['observed_monotonic']) in (int,float) and row['observed_monotonic']>=last_time,
            'native_capture_event_chain_changed')
        _require(row['kind'] in ('notification','interrupt_requested','interrupt_acknowledged','interrupt_failed','stream_error','stream_closed','capture_sealed','consumer_cleanup_after_terminal'),
            'native_capture_unknown_record_kind')
        if row['kind']=='consumer_cleanup_after_terminal':
            previous_native=next((entry['notification'] for entry in reversed(rows) if entry['kind']=='notification'),None)
            _require(previous_native is not None and previous_native['method']=='turn/completed' and
                (row['error_type']=='GeneratorExit' or (row['error_type']=='CancelledError' and
                    type(row['local_task_cancellation_requests']) is int and row['local_task_cancellation_requests']>0)),
                'native_capture_cleanup_without_observed_terminal')
        if row['kind']=='notification':
            event=row['notification'];thread,turn=core._route(event['payload'],capture['native_thread_id'],capture['native_turn_id'])
            _require(event['thread_id']==thread and event['turn_id']==turn,'native_capture_notification_routing_changed')
        rows.append(row);previous=_digest(envelope);last_time=row['observed_monotonic']
    _require(len(rows)==capture['event_records'] and previous==capture['last_record_blake3'] and rows and
        rows[-1]['kind']=='capture_sealed' and sum(row['kind']=='stream_closed' for row in rows)==1 and
        sum(row['kind']=='capture_sealed' for row in rows)==1 and
        capture['summary']==_summarize(rows,binding,start['offered_mcp_tool_names']),'native_capture_summary_changed')
    return {**capture,'start':start,'capture_file_blake3':_file_digest(path),'public_key_base64':public_key,
        'actual_notifications':[row for row in rows if row['kind']=='notification'],
        'interrupt_records':[row for row in rows if row['kind'].startswith('interrupt_')]}
