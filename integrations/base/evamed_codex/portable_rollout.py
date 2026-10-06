"""Real pinned Codex pre-rollouts against the fresh portable five-tool runtime."""
from __future__ import annotations
import asyncio
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / 'evamed-codex/execution-bundles/EVA-medresearch-v1'
CONTINUATION_POLICY={'max_native_turns':6,'max_consecutive_no_progress_turns':2,'same_native_thread':True,
    'shared_model_request_and_wall_budgets':True,'host_continuation_conditioning_loss_weight':0,
    'task_resampling_or_reset':False,'backbone_or_mode_change':False}
sys.path.insert(0, str(BUNDLE / 'runtime'))
from evamed_portable.bundle import Bundle
from evamed_portable.integrity import Signer, byte_digest, canonical, digest, file_digest, strict_json, verify_receipt, write_json
from evamed_portable.runtime import Runtime
from .benchmark_provider import setup_provider
from .portable_qwen_types import portable_provider, fresh_argument_projector


def settings():
    return {'route': 'api', 'model': 'nvidia/qwen/qwen3.6-27b', 'endpoint': 'https://service.example.invalid/v1',
        'mode': 'think', 'max_turns': 24, 'max_seconds': 900, 'max_output_tokens': 32768,
        'context_length': 262144, 'compact_at_tokens': 94208, 'thinking_budget_tokens': 24576,
        'thinking_logit_processor': None,
        'argument_type_projection':'eva.medresearch-fresh-qwen-host-types.v1',
        'codex_bin': str(ROOT / 'tools/codex-0.153.4/package/vendor/x86_64-unknown-linux-musl/bin/codex'),
        'mcp_python': str(ROOT / '.venvs/assets/bin/python'),
        'bwrap': str(ROOT / 'tools/enroot/usr/bin/bwrap'),
        'runtime_root': str(ROOT / '.enroot/data/evamed-slime-v0.3.2'),
        'bundle': str(BUNDLE), 'key': str(ROOT / 'evamed-codex/private/portable-execution/host-key.pem')}


def api_key():
    spec = importlib.util.spec_from_file_location('evamed_api_key_source', ROOT / 'evamed-codex/scripts/api-conformance.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module._secret_keys()[0]


def canonical_tools(runtime):
    return [{'name': x['name'], 'description': x['description'], 'inputSchema': x['parameters']}
            for x in runtime.definitions.values()]


def public_snapshot(runtime, output):
    state = runtime.public_snapshot()
    files = []
    for name in ['task.json', 'semantics.json']:
        raw = (runtime.root / 'public' / name).read_bytes()
        files.append({'path': name, 'bytes': len(raw), 'blake3': file_digest(runtime.root / 'public' / name),
                      'text': raw.decode()})
    # Reopened accepted artifacts are actor-authored public outputs; no check values/rubrics are copied.
    host_state = runtime._load_state()
    for stage in sorted(host_state['completed']):
        value = runtime._reopen(host_state, stage)
        raw = canonical(value)
        files.append({'path': 'accepted/' + stage + '.json', 'bytes': len(raw), 'blake3': digest(value), 'text': raw.decode()})
    # Preserve actual actor source/output files, including rejected attempts.
    # Read-only evidence is retained in exact retrieval tool results instead of duplicated here.
    audit_root=Path(output).parent
    if audit_root.name=='snapshots':audit_root=audit_root.parent
    inline_bytes=sum(item['bytes'] for item in files)
    for attempt in sorted((runtime.root/'executions').glob('*')) if (runtime.root/'executions').is_dir() else []:
        candidates=[attempt/'source.py',*sorted((attempt/'work').rglob('*'))]
        for path in candidates:
            if not path.is_file() or path.is_symlink():continue
            size=path.stat().st_size;hash_=file_digest(path)
            blob=audit_root/'public-file-blobs'/hash_
            if not blob.exists():blob.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,blob)
            item={'path':'executions/'+str(path.relative_to(runtime.root/'executions')),'bytes':size,'blake3':hash_,
                  'retained_blob_relative_path':'public-file-blobs/'+hash_}
            if size<=262144 and inline_bytes+size<=1024*1024:
                try:item['text']=path.read_text();inline_bytes+=size
                except UnicodeError:pass
            files.append(item)
    snapshot = {'schema': 'eva.medresearch-actor-workspace-snapshot.v1', 'run_id': state['run_id'],
        'visibility': 'actor-public', 'state': {k: v for k, v in state.items() if k not in {'task', 'semantics'}},
        'files': files, 'private_evaluator_data_included': False,
        'scope':'public task/semantics, accepted actor artifacts, all actual execution source/work files; retrieved evidence appears in tool results',
        'large_or_binary_public_files_retained_as_blobs':True}
    snapshot['document_blake3'] = digest(snapshot)
    write_json(output, snapshot)
    return snapshot


def serve(configuration):
    runtime = Runtime(bundle=configuration['bundle'], run_root=configuration['run_root'], signer=configuration['key'],
                      bwrap=configuration['bwrap'], runtime_root=configuration['runtime_root'])
    event_root = Path(configuration['audit_root'])
    event_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    events_path = event_root / 'mcp-events.jsonl'
    count = len(events_path.read_bytes().splitlines()) if events_path.exists() else 0
    for line in iter(lambda: sys.stdin.buffer.readline(262145), b''):
        if len(line) > 262144: raise ValueError('bounded_mcp_request_exceeded')
        request = strict_json(line)
        identifier, method = request.get('id'), request.get('method')
        if identifier is None: continue
        if method == 'initialize':
            result = {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'eva_medresearch', 'version': '1.0.0'}}
        elif method == 'tools/list': result = {'tools': canonical_tools(runtime)}
        elif method == 'tools/call':
            parameters = request.get('params', {}); name = parameters.get('name'); arguments = parameters.get('arguments', {})
            event_id = str(uuid4()); before = public_snapshot(runtime, event_root / 'snapshots' / (event_id + '-before.json'))
            count += 1
            try:
                if count > runtime.construction['runtime']['policy_budgets']['max_turns']:
                    raise ValueError('source_total_tool_attempt_budget_exhausted')
                output = runtime.call(name, arguments)
            except Exception as exc:
                output = {'gate_passed': False, 'error': str(exc) if type(exc) is ValueError else type(exc).__name__}
            after = public_snapshot(runtime, event_root / 'snapshots' / (event_id + '-after.json'))
            result = {'content': [{'type': 'text', 'text': canonical(output).decode()}],
                      'structuredContent': output, 'isError': output.get('gate_passed') is False}
            event = {'schema': 'eva.medresearch-codex-host-tool-event.v1', 'event_id': event_id,
                'request_id': identifier, 'name': name, 'arguments': arguments, 'public_result': output,
                'response': result, 'response_blake3': digest(result),
                'workspace_before_blake3': before['document_blake3'], 'workspace_after_blake3': after['document_blake3'],
                'attempt_number': count, 'private_reasoning_retained': False}
            event['event_blake3'] = digest(event)
            with events_path.open('ab') as stream: stream.write(canonical(event))
        elif method == 'ping': result = {}
        elif method in {'resources/list', 'resources/templates/list', 'prompts/list'}:
            result = {{'resources/list': 'resources', 'resources/templates/list': 'resourceTemplates', 'prompts/list': 'prompts'}[method]: []}
        else:
            sys.stdout.buffer.write(canonical({'jsonrpc': '2.0', 'id': identifier, 'error': {'code': -32601, 'message': 'method unavailable'}}));sys.stdout.buffer.flush();continue
        sys.stdout.buffer.write(canonical({'jsonrpc': '2.0', 'id': identifier, 'result': result}));sys.stdout.buffer.flush()


def joined_calls(receipt, audit, event_slice=None):
    from eva_agent.pipeline.digests import canonical_value
    events_path = audit / 'mcp-events.jsonl'
    events = [strict_json(line) for line in events_path.read_bytes().splitlines()] if events_path.exists() else []
    if event_slice is not None:events=events[slice(*event_slice)]
    unused = list(events); joined = []
    for call in receipt.tool_calls:
        if call.tool_type != 'mcpToolCall':
            raise ValueError('unexpected_native_tool_execution')
        if call.mcp_server == 'codex' and call.mcp_tool == 'list_mcp_resources':
            output=canonical_value(call.output)
            native=output.get('result') or {};text=native.get('content',[])
            if (output.get('error') is not None or call.status!='completed' or canonical_value(call.arguments)!={} or
                set(native)-{'_meta','content','structuredContent','isError'} or native.get('_meta') is not None or native.get('structuredContent') is not None or native.get('isError',False) is not False or
                len(text)!=1 or set(text[0])!={'type','text'} or text[0].get('type')!='text' or strict_json(text[0].get('text',''))!={'resources':[]}):
                raise ValueError('unexpected_native_resource_discovery_payload')
            joined.append({'kind':'native_empty_resource_discovery','codex_tool_call_id':call.tool_call_id,
                'codex_tool_receipt_blake3':call.receipt_blake3,'actual_host_effect':False})
            continue
        if call.mcp_server != 'eva_medresearch': raise ValueError('unexpected_mcp_server')
        arguments = canonical_value(call.arguments); output = canonical_value(call.output)
        if output.get('error') is not None or call.status not in {'completed','failed'} or not isinstance(output.get('result'),dict):
            raise ValueError('Codex_actual_MCP_response_wrapper_invalid')
        normalized=deepcopy(output['result'])
        if '_meta' in normalized:
            if normalized.pop('_meta') is not None:raise ValueError('unexpected_Codex_MCP_result_metadata')
        if 'isError' not in normalized:normalized['isError']=call.status=='failed'
        if normalized.get('isError') is not (call.status=='failed'):
            raise ValueError('Codex_MCP_status_response_mismatch')
        match = next((x for x in unused if x['name'] == call.mcp_tool and x['arguments'] == arguments and x['response']==normalized), None)
        if match is None: raise ValueError('Codex_tool_call_missing_host_event')
        if digest(normalized)!=match['response_blake3'] or digest({k:v for k,v in match.items() if k!='event_blake3'})!=match['event_blake3']:
            raise ValueError('actual_host_event_or_response_commitment_mismatch')
        unused.remove(match)
        joined.append({'codex_tool_call_id': call.tool_call_id, 'codex_tool_receipt_blake3': call.receipt_blake3,
                       'host_event_id': match['event_id'], 'host_event_blake3': match['event_blake3']})
    if unused: raise ValueError('unmatched_actual_host_events')
    return joined


async def run_case(configuration, case, cohort, purpose, provider_factory=portable_provider):
    from eva_agent.codex_runtime import (CodexRuntime, CodexThreadOptions, CodexTurnInput, CodexToolOffer,
        CodexRole, CodexSandbox, verify_codex_turn_receipt)
    from eva_agent.codex_runtime.runtime import _logical_input
    from eva_agent.codex_runtime.policy_budget import CodexPolicyBudgetExceeded
    from eva_agent.codex_runtime.supra import SupraProfile, SupraMode
    from eva_agent.codex_runtime.research_memory import ResearchContextPolicy
    from eva_agent.pipeline.digests import canonical_value
    config = dict(configuration)
    runtime = Runtime.create(bundle=config['bundle'], case_id=case['sandbox_id'], runs_root=cohort / 'runtime-runs',
                             signer=config['key'], bwrap=config['bwrap'], runtime_root=config['runtime_root'])
    limits = runtime.construction['runtime']['policy_budgets']
    config.update(max_turns=limits['max_turns'], max_seconds=limits['wall_time_seconds'], max_output_tokens=limits['max_output_tokens'])
    config.update(portable_runtime_run_root=str(runtime.root),portable_case_id=case['sandbox_id'])
    audit = cohort / 'trajectories' / runtime.public_snapshot()['run_id']; audit.mkdir(parents=True, mode=0o700)
    tool_config = {k: config[k] for k in ['bundle', 'key', 'bwrap', 'runtime_root']}
    tool_config.update(run_root=str(runtime.root), audit_root=str(audit))
    tool_config_path = audit / 'host-tool-config.json'; write_json(tool_config_path, tool_config);tool_config_path.chmod(0o600)
    before = public_snapshot(runtime, audit / 'before.json')
    token = api_key() if config['route'] == 'api' else 'local-nonsecret'
    receipt = terminal = launch_receipt = None; errors = []; joins = []; turns=[];provider_count = 0; exhausted = None; started = time.monotonic()
    control_stop=None;no_progress=0
    config_public = {k: v for k, v in config.items() if k not in {'key'}}
    write_json(audit / 'configuration.json', config_public)
    try:
        with provider_factory(config, audit / 'provider', token) as (backend, provider, thread, meter):
            tools = canonical_tools(runtime)
            offers = tuple(CodexToolOffer(fully_qualified_name='eva_medresearch/' + t['name'], description=t['description'],
                input_schema=t['inputSchema'], parallel_safe=False, read_only=t['name']=='retrieve_frozen_evidence',
                allowed_stages=('S1','S2','S3','S4','S5','E2E')) for t in tools)
            names = [t['name'] for t in tools]
            thread['mcp_servers'] = {'eva_medresearch': {'command': config['mcp_python'],
                'args': ['-I', '-B', str(ROOT / 'evamed-codex/scripts/portable-codex-rollout.py'), 'serve', '--config', str(tool_config_path)],
                'cwd': str(runtime.root / 'public'), 'required': True, 'startup_timeout_sec': 60, 'tool_timeout_sec': 200,
                'enabled_tools': names, 'omit_tools_from': ['deferred', 'code_mode'],
                'tools': {n: {'approval_mode': 'approve'} for n in names}}}
            developer = ('Complete this fresh EVA-medresearch S1-through-S5 execution using only the five offered canonical tools. '
                'The explicitly recorded Qwen XML transport binding decodes supplied steps and evidence_ids JSON array text using the fresh host field types, without changing tool schemas, adding fields, or repairing values. '
                'Preserve the released focus stage and source evidence boundary. The new execution semantics below explicitly govern stage prerequisites. '
                'Historical domain tool names in the source brief do not introduce available tool aliases. No private evaluator data is supplied. '
                'Use materialize_plan with objective and five ordered steps [{stage:S1,action:...},...,{stage:S5,action:...}]. '
                'Then retrieve every declared evidence_id and materialize_evidence_selection with exact evidence_ids and rationale. '
                'Use execute_code at S3 then S4 to write each exact output path; S4 receives read-only input/S3-handoff.json. '
                'Finally submit_results once with the exact terminal schema. Print needed evidence-derived summaries or artifact content from your own code. '
                'The tool budget is 24 total attempts, including invalid attempts, with 2 turns reserved for S5; overall wall time is 900 seconds. '
                'Each Python invocation has 180 seconds, 1 CPU, 1 GiB address space, a strict single-process/thread profile, and no network/GPU. '
                'Stage workspaces are fresh; only public inputs and prior accepted handoffs are available. Model requests including compaction are capped at 24. '
                'The host can continue the same native Codex thread up to 6 native turns when submission remains incomplete, stopping after 2 consecutive turns without stage/evidence progress. All original budgets remain shared. '
                'Keep uncertainty explicit and never claim success from prose when the host gate failed.')
            options = CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model=config['model'], provider=provider,
                cwd=str(runtime.root / 'public'), sandbox=CodexSandbox.READ_ONLY, ephemeral=True, config=thread,
                offered_tools=offers, base_instructions='You are a coding agent completing an evidence-bound medical research workflow.',
                developer_instructions=developer)
            value = CodexTurnInput(public_text=canonical(runtime.public_snapshot()).decode(), model=config['model'], effort='xhigh')
            profile = SupraProfile(model=config['model'], provider=provider, mode=SupraMode.THINK, protocol='qwen_template',
                local_qwen_endpoint=config['endpoint'] if config['route']=='local' else None,
                context=ResearchContextPolicy(context_tokens=config['context_length'], output_tokens=config['max_output_tokens'],
                    reserve_tokens=8192, compact_at_tokens=config['compact_at_tokens']))
            options, value = profile.thread_options(options), profile.turn_input(value)
            public_authority = None
            native_runtime = CodexRuntime(backend)
            if config.get('public_warning_original_actor'):
                from .portable_public_projection import PublicWarningAuthority, AuthenticatedPublicCodexRuntime
                public_authority = PublicWarningAuthority(runtime, audit, config['public_warning_original_actor'])
                native_runtime = AuthenticatedPublicCodexRuntime(backend, public_warning_authority=public_authority)
            if config.get('training_public_warning_policy') is not None:
                from .portable_training_public_projection import (SCHEMA as TRAINING_PUBLIC_POLICY,
                    TrainingPublicWarningAuthority, AuthenticatedTrainingPublicCodexRuntime)
                if config['training_public_warning_policy'] != TRAINING_PUBLIC_POLICY or public_authority is not None:
                    raise ValueError('training_public_policy_must_be_exact_and_separate_from_qualification')
                if runtime.row['domain'] == 'medxpertqa':
                    public_authority = TrainingPublicWarningAuthority(runtime, audit)
                    native_runtime = AuthenticatedTrainingPublicCodexRuntime(backend, public_warning_authority=public_authority)
            async with native_runtime as codex:
                handle=None
                for ordinal in range(1,CONTINUATION_POLICY['max_native_turns']+1):
                    remaining=config['max_seconds']-(time.monotonic()-started)
                    if remaining<=0 or meter.requests>=config['max_turns']:
                        control_stop='shared_wall_or_model_request_cap';break
                    current=runtime.public_snapshot()
                    progress_before={k:current[k] for k in ('next_stage','completed_stages','retrieved_evidence_ids','submission_attempted')}
                    if current['submission_attempted'] or current['next_stage']=='complete':
                        control_stop='host_submission_boundary';break
                    continuation=None
                    if ordinal>1:
                        continuation={'schema':'eva.medresearch-host-continuation-conditioning.v1','run_id':current['run_id'],
                            'native_turn':ordinal,'host_state':progress_before,
                            'instruction':'Continue this same source case at the host next_stage with the existing tools and evidence. The formal host submission is incomplete. Do not restart, reset, or change the task.',
                            'remaining_model_requests':config['max_turns']-meter.requests,
                            'remaining_tool_attempts':max(0,limits['max_turns']-current['tool_calls']),
                            'conditioning_loss_weight':0,'new_task_evidence_or_answers_supplied':False}
                        value=replace(value,public_text=canonical(continuation).decode())
                    turn_dir=audit/'turns'/f'{ordinal:02d}';turn_dir.mkdir(parents=True,mode=0o700)
                    turn_before=public_snapshot(runtime,turn_dir/'before.json')
                    logical=canonical_value(_logical_input(options,value))
                    public_authorization = (public_authority.authorize(options, value, turn_dir)
                        if public_authority is not None and ordinal == 1 else None)
                    write_json(turn_dir/'logical-request.json',{'schema':'eva.medresearch-codex-logical-request.v1',
                        'logical_input':logical,'base_instructions':options.base_instructions,
                        'developer_instructions':options.developer_instructions,'public_tool_catalog':tools,
                        'public_snapshot_blake3':turn_before['document_blake3'],'continuation_injection':continuation,
                        'private_evaluator_data_included':False})
                    launch_receipt=runtime.signer.sign({'schema':'eva.medresearch-authenticated-Codex-launch.v1',
                        'run_id':before['run_id'],'case_id':case['sandbox_id'],'bundle_blake3':runtime.bundle.manifest['document_blake3'],
                        'source_record_blake3':runtime.row['record_blake3'],'source_rubric_digest':runtime.row['reward_contract']['rubric_digest'],
                        'logical_input_blake3':digest(logical),'base_instructions_blake3':byte_digest((options.base_instructions or '').encode()),
                        'developer_instructions_blake3':byte_digest((options.developer_instructions or '').encode()),
                        'instruction_hash_encoding':'UTF-8 bytes; no added newline','native_turn':ordinal,
                        'public_tool_catalog_blake3':digest(tools),'selected_skills':canonical_value(value.skills),
                        'codex_binary_blake3':file_digest(config['codex_bin']),
                        'argument_type_binding_blake3':fresh_argument_projector(runtime).binding_blake3,
                        'signed_before_Codex_start_thread':handle is None,'signed_before_Codex_run_turn':True,
                        'continuation_policy':CONTINUATION_POLICY,'continuation_injection_blake3':digest(continuation) if continuation else None,
                        'host_conditioning_loss_weight':0,'model':config['model'],'route':config['route']})
                    if public_authorization is not None:
                        launch_receipt = runtime.signer.sign({**launch_receipt['payload'],
                            'public_projection_authorization_blake3': digest(public_authorization)})
                        if getattr(public_authority, 'schema', None) is not None:
                            launch_receipt = runtime.signer.sign({**launch_receipt['payload'],
                                'public_projection_authorization_schema': public_authority.schema,
                                'public_projection_runtime_options_commitment': public_authorization['payload']['runtime_options_commitment'],
                                'public_projection_runtime_input_commitment': public_authorization['payload']['runtime_input_commitment']})
                    write_json(turn_dir/'launch.json',launch_receipt,exclusive=True)
                    event_path=audit/'mcp-events.jsonl'
                    offset=len(event_path.read_bytes().splitlines()) if event_path.exists() else 0
                    if handle is None:handle=await codex.start_thread(options)
                    try:receipt=await codex.run_turn(handle,value,policy_timeout_seconds=max(1,remaining))
                    except CodexPolicyBudgetExceeded as exc:receipt,terminal=exc.receipt,exc.outcome
                    verify_codex_turn_receipt(receipt)
                    write_json(turn_dir/'codex-receipt.json',canonical_value(receipt))
                    end=len(event_path.read_bytes().splitlines()) if event_path.exists() else 0
                    joined=joined_calls(receipt,audit,(offset,end));joins.extend(joined)
                    turn_after=public_snapshot(runtime,turn_dir/'after.json')
                    turns.append({'relative_directory':str(turn_dir.relative_to(audit)),
                        'receipt_blake3':receipt.receipt_blake3,'launch_receipt_blake3':digest(launch_receipt),
                        'joined_host_results':joined,'host_event_offset':offset,'host_event_end':end,
                        'before_snapshot_blake3':turn_before['document_blake3'],'after_snapshot_blake3':turn_after['document_blake3'],
                        'native_thread_id':receipt.thread_id,'status':receipt.status,'host_conditioning_loss_weight':0})
                    current_after=runtime.public_snapshot()
                    progress_after={k:current_after[k] for k in progress_before}
                    no_progress=no_progress+1 if progress_after==progress_before else 0
                    if terminal is not None or receipt.status!='completed':control_stop='Codex_terminal_or_incomplete_turn';break
                    if progress_after['submission_attempted'] or progress_after['next_stage']=='complete':control_stop='host_submission_boundary';break
                    if no_progress>=CONTINUATION_POLICY['max_consecutive_no_progress_turns']:control_stop='no_progress_cap';break
                else:control_stop='native_turn_cap'
            provider_count,exhausted=meter.requests,meter.exhausted
    except Exception as exc:
        errors.append({'type':type(exc).__name__,'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]})
        if 'meter' in locals():provider_count,exhausted=meter.requests,meter.exhausted
    after=public_snapshot(runtime,audit/'after.json')
    host_final=runtime.finalize();write_json(audit/'host-final-disposition.json',host_final)
    terminal_receipt = None
    if terminal is not None or control_stop not in {None,'host_submission_boundary'}:
        terminal_receipt = runtime.signer.sign({'schema': 'eva.medresearch-codex-budget-terminal.v1',
            'run_id':after['run_id'], 'case_id':case['sandbox_id'], 'bundle_blake3':runtime.bundle.manifest['document_blake3'],
            'codex_turn_receipt_blake3':receipt.receipt_blake3 if receipt else None,
            'outcome':canonical_value(terminal) if terminal is not None else {'control_stop_reason':control_stop,
                'native_turn_count':len(turns),'consecutive_no_progress_turns':no_progress,
                'actual_model_requests':provider_count,'shared_max_model_requests':config['max_turns'],
                'wall_seconds':time.monotonic()-started,'shared_wall_seconds':config['max_seconds']},
            'clinical_or_ability_score':None})
        write_json(audit/'host-budget-terminal.json',terminal_receipt)
    event_path=audit/'mcp-events.jsonl';events=[strict_json(l) for l in event_path.read_bytes().splitlines()] if event_path.exists() else []
    aggregate=runtime.signer.sign({'schema':'eva.medresearch-authenticated-Codex-episode.v1','run_id':after['run_id'],
        'case_id':case['sandbox_id'],'bundle_blake3':runtime.bundle.manifest['document_blake3'],
        'source_record_blake3':runtime.row['record_blake3'],'source_rubric_digest':runtime.row['reward_contract']['rubric_digest'],
        'turns':turns,'before_snapshot_blake3':before['document_blake3'],'after_snapshot_blake3':after['document_blake3'],
        'host_final_receipt_blake3':digest(host_final),'continuation_policy':CONTINUATION_POLICY,
        'control_stop_reason':control_stop,'model':config['model'],'route':config['route'],
        'actual_model_requests':provider_count,'actual_tool_attempts':len(events),'all_joined_host_results':joins,
        'private_reasoning_retained':False,'host_conditioning_loss_weight':0})
    write_json(audit/'episode-receipt.json',aggregate)
    result={'schema':'eva.medresearch-real-codex-trajectory.v1','purpose':purpose,'run_id':after['run_id'],
        'case_id':case['sandbox_id'],'domain':case['domain'],'focus_stage':case['stage'],
        'bundle_blake3':runtime.bundle.manifest['document_blake3'],'source_record_blake3':runtime.row['record_blake3'],
        'source_rubric_digest':runtime.row['reward_contract']['rubric_digest'],
        'model':config['model'],'route':config['route'],'codex_binary_blake3':file_digest(config['codex_bin']),
        'codex_turn_receipt_blake3':receipt.receipt_blake3 if receipt else None,'codex_turn_status':receipt.status if receipt else None,
        'launch_receipt_blake3':digest(launch_receipt) if launch_receipt else None,
        'single_receipt_alias_scope':'last completed native turn only','codex_turns':turns,
        'actual_native_turn_count':len(turns),'episode_receipt_blake3':digest(aggregate),
        'continuation_policy':CONTINUATION_POLICY,'control_stop_reason':control_stop,
        'before_snapshot_blake3':before['document_blake3'],'after_snapshot_blake3':after['document_blake3'],
        'host_final_receipt_blake3':digest(host_final),'all_stage_gates_passed':host_final['payload']['all_stage_gates_passed'],
        'actual_model_requests':provider_count,'actual_tool_attempts':len(events),'joined_host_results':joins,
        'tool_schema_modified':False,'model_budget_exhausted':exhausted,'policy_budget_terminal':terminal is not None,
        'argument_type_projection':config.get('argument_type_projection'),
        'authenticated_budget_terminal_blake3':digest(terminal_receipt) if terminal_receipt else None,
        'wall_seconds':time.monotonic()-started,'errors':errors,'private_reasoning_retained':False,
        'native_codex_ephemeral_thread':True,'clinical_or_ability_score':None,'s_target_evidence':False,
        'diagnostic_infrastructure_passed':receipt is not None and any('host_event_id' in j for j in joins) and not errors}
    result['document_blake3']=digest(result);write_json(audit/'trajectory.json',result)
    print(json.dumps({'trajectory':str(audit/'trajectory.json'),'case_id':case['sandbox_id'],'codex_status':result['codex_turn_status'],
                      'stage_gates':result['all_stage_gates_passed'],'tool_attempts':len(events),'errors':errors}),flush=True)
    return result


async def run(output, purpose, diagnostic_receipt=None):
    config=settings();version=subprocess.check_output([config['codex_bin'],'--version'],text=True,timeout=10).strip()
    if version!='codex-cli 0.153.4':raise ValueError('pinned_Codex_binary_required')
    sample_path=ROOT/'evamed-codex/receipts/rl-sandbox-inventory/sample-proposal.json'
    sample=strict_json(sample_path.read_bytes());selected=[r for r in sample['records'] if r['stratum_position']==1]
    if purpose=='diagnostic':selected=selected[:1]
    elif purpose=='frozen-42':
        if diagnostic_receipt is None:raise ValueError('real_diagnostic_admission_required')
        admitted=strict_json(Path(diagnostic_receipt).read_bytes())
        if (not admitted.get('diagnostic_infrastructure_passed') or admitted['model']!=config['model'] or admitted['route']!='api'
            or admitted.get('continuation_policy')!=CONTINUATION_POLICY or not admitted.get('codex_turns')):
            raise ValueError('API_diagnostic_infrastructure_not_admitted')
    else:raise ValueError('unknown_run_purpose')
    cohort=Path(output).resolve()/str(uuid4());cohort.mkdir(parents=True,mode=0o700)
    selection={'schema':'eva.medresearch-frozen-codex-selection.v1','purpose':purpose,'sample_proposal_blake3':file_digest(sample_path),
        'cases':selected,'model':config['model'],'route':config['route'],'selection_uses_scores':False,'attempts_per_case':1,
        'automatic_retries':False,'local_substitution_permitted':False,'continuation_policy':CONTINUATION_POLICY,
        'driver_source_blake3':file_digest(__file__),
        'qwen_type_binding_source_blake3':file_digest(ROOT/'evamed-codex/src/evamed_codex/portable_qwen_types.py'),
        'provider_source_blake3':file_digest(ROOT/'evamed-codex/src/evamed_codex/benchmark_provider.py')}
    selection['document_blake3']=digest(selection);write_json(cohort/'selection.json',selection)
    concurrency=8 if purpose=='frozen-42' else 1
    semaphore=asyncio.Semaphore(concurrency)
    async def bounded(case):
        async with semaphore:return await run_case(config,case,cohort,purpose)
    results=await asyncio.gather(*(bounded(case) for case in selected))
    write_json(cohort/'cohort.json',{'schema':'eva.medresearch-codex-cohort.v1','purpose':purpose,
        'selection_blake3':selection['document_blake3'],'concurrency_limit':concurrency,'case_count':len(results),
        'trajectory_document_blake3s':[r['document_blake3'] for r in results],
        'completed_stage_chains':sum(r['all_stage_gates_passed'] for r in results),
        'model_rollouts':len(results),'clinical_or_ability_scores':None,'s_target_evidence':False})
    return cohort
