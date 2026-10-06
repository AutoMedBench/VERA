"""One actual task6 local-Qwen instant run; native verifier remains host-only."""
from __future__ import annotations
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import uuid4

from eva_agent.codex_pipeline.adapter import CodexToolExecutionBridge, _mcp_groups, _project_events, _validate_tool_inventory
from eva_agent.codex_pipeline.skills import VerifiedActorSkillCatalog
from eva_agent.codex_pipeline.turn_mcp import TurnMCPBridgeFactory
from eva_agent.codex_runtime import (CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions,
    CodexToolOffer, CodexTurnInput, OpenAICodexBackend, verify_codex_turn_receipt)
from eva_agent.codex_runtime.policy_budget import CodexPolicyBudgetExceeded
from eva_agent.codex_runtime.research_memory import ResearchContextPolicy, memory_launch_options
from eva_agent.codex_runtime.supra import SupraProfile, SupraMode, SupraChatTransport
from eva_agent.codex_runtime.runtime import _logical_input
from eva_agent.codex_providers.adapter import AdapterLimits, ChatCompletionsTransport
from eva_agent.pipeline.contracts import Stage
from eva_agent.pipeline.digests import blake3_bytes, canonical_json_bytes, canonical_value
from eva_agent.pipeline.ids import RandomUUIDFactory
from eva_agent.pipeline.tools import ParallelToolRuntime, ToolDefinition, ToolRegistry
from eva_agent.pipeline.workspace import FilesystemSandbox
from eva_agent.training.progressive_skills import ProgressiveTeacherSkillSurface
from training.automedbench_lite.actor import serving_binding
from training.automedbench_lite.local_qwen import local_qwen_setup
from training.automedbench_lite.token_budget import TokenBudgetChatTransport
from training.healthagent_terminal_v1 import (TOOL_NAME, TOOL_DESCRIPTION, TOOL_SCHEMA,
    execute_terminal, public_binding, resolve_image, write)

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/localhome/local-operator/operator_GB300-2/EVA-Agent')
BASE = ('You are a coding research agent executing one public HealthAgentBench clinical-trial-matching task. '
    'This is benchmark research, not patient care. Follow its exact native instruction and submission format. '
    'Use only the offered healthagent_eval MCP tools. The terminal runs in an isolated CPU Docker container; '
    'native Codex shell, network, private labels, evaluators and other model endpoints are not available. '
    'Do not invent tool results, citations or clinical scores.')
DEVELOPER = ('This is one continuous E2E task, not fabricated S1-S5 phases. Public data is under /workspace/data; '
    'the original instruction is /workspace/instruction.md; pool IDs are /workspace/trial_ncts.txt. '
    'Use the actual terminal for file inspection, Python analysis and the exact final submission. '
    'Only /workspace/submission and /workspace/notes persist across terminal calls. '
    'The task has a 3600-second overall actor budget; each terminal command is bounded to180 seconds. '
    'Search/load suitable medical research skills at stage E2E when relevant; skills may mention broader tools '
    'that are not offered here. Keep concise task-state notes and useful authored scripts in /workspace/notes. '
    'Read back and validate your actual /workspace/submission/eligible_trials.txt before finishing. '
    'A successful command or skill load does not establish clinical correctness. No evaluator is actor-accessible.')

def instant_profile():
    return SupraProfile(model='Qwen/Qwen3.5-9B', provider='eva_adapter_qwen_3_5_9b',
        mode=SupraMode.INSTANT, context=ResearchContextPolicy(), protocol='qwen_template',
        local_qwen_endpoint='http://127.0.0.1:30910/v1')


def instant_transport(output, http_transport=None):
    """The actual false control precedes exact tokenization and generation."""
    profile = instant_profile()
    base = http_transport or ChatCompletionsTransport(AdapterLimits(max_concurrency=1, upstream_timeout_seconds=600))
    token = TokenBudgetChatTransport(base, endpoint=profile.local_qwen_endpoint,
        context_length=32768, audit_root=output / 'token-budget')
    def actual(binding, body):
        result = token(binding, body)
        observed = {'schema':'eva.local-qwen-instant-response.v1', 'http_status':result.status,
                    'raw_output_recorded':False, 'private_reasoning_recorded':False,
                    'reasoning_field_nonempty':None, 'visible_content_characters':None}
        try:
            payload = json.loads(result.body)
            messages = [x.get('message') or {} for x in payload.get('choices', [])]
            observed['reasoning_field_nonempty'] = any(bool(x.get('reasoning_content')) for x in messages)
            observed['visible_content_characters'] = sum(len(x.get('content') or '') for x in messages if isinstance(x.get('content'), str))
        except (ValueError, AttributeError, TypeError):
            observed['metadata_unreadable'] = True
        write(output / 'mode-observations' / (str(uuid4()) + '.json'), observed)
        return result
    return SupraChatTransport(actual, profile, observer=lambda row:
        write(output / 'mode-observations' / (str(uuid4()) + '.json'), row))


class CapturedTools:
    def __init__(self, *, output, bundle, image):
        self.output,self.bundle,self.image = output,bundle,image
        instruction = (bundle / 'public/instruction.md').read_bytes()
        self.workspace = FilesystemSandbox(output / 'workspaces',str(uuid4()),{'instruction.md':instruction})
        for name in ('submission','notes'): (self.workspace.root/name).mkdir(mode=0o700)
        def handler(_workspace, arguments):
            return execute_terminal(public=bundle/'public/workspace',workspace=self.workspace.root,image=image,
                audit_root=output/'terminal-audit',**canonical_value(arguments))
        registry = ToolRegistry([ToolDefinition(TOOL_NAME,TOOL_DESCRIPTION,TOOL_SCHEMA,handler)])
        plugin = ROOT / 'plugins/evamed-codex'
        (output/'skill-mounts').mkdir(mode=0o700)
        self.skill_catalog = VerifiedActorSkillCatalog(
            manifest_path=plugin/'references/legacy-skill-manifest.v1.json',
            legacy_source_root=MAIN.parent/'rlevo-med-research/harness/source/rlevo-Med-RL-data/rev-79dd2a31f5f',
            native_stage_skill_path=plugin/'skills/stage-rollout/SKILL.md',runtime_root=output/'skill-mounts')
        self.registry = ProgressiveTeacherSkillSurface(self.skill_catalog).augment_registry(registry,Stage.E2E)
        self.ids = RandomUUIDFactory()
        self.runtime = ParallelToolRuntime(workspace=self.workspace,registry=self.registry,
            id_factory=self.ids,maximum_parallel_calls=1)
        self.calls,self.after,self.lock = [],{},RLock()
        self.initial = self.workspace.snapshot('initial-mutable-only')
        write(output/'workspace-before.json',self.initial)
        self.offers = tuple(CodexToolOffer('healthagent_eval/'+d.name,d.description,d.input_schema,
            read_only=d.name!=TOOL_NAME,parallel_safe=False) for d in self.registry.definitions())

    def execute(self,calls):
        values=[]
        with self.lock:
            for call in calls:
                self.calls.append(call)
                result=self.runtime.execute((call,))[0]
                values.append(result)
                after=self.workspace.snapshot('after-'+call.call_id)
                self.after[call.call_id]=after
                write(self.output/'host-tools'/(call.call_id+'.json'),{'call':call,'result':result,'workspace_after':after})
                print(json.dumps({'event':'actual_tool','name':call.name,'status':result.status,'call_id':call.call_id}),flush=True)
        return tuple(values)

    def trace(self): return self.runtime.trace()


def run_actor(output,bundle,args,image):
    output.mkdir(parents=True,mode=0o700)
    tools=CapturedTools(output=output,bundle=bundle,image=image)
    outcome={'status':'not_started','requested_model':'Qwen/Qwen3.5-9B','returned_model':None,
        'process_stage':'E2E','stage_completion_claimed':False,'native_score':None}
    with local_qwen_setup(run_root=output/'local-runtime',workers=1,thinking=False,
            exact_tool_schemas=True,normalize_priority_messages=True,max_output_tokens=4096,
            codex_bin=args.codex_bin,auto_compact_token_limit=12288,upstream_timeout_seconds=600,
            token_budget=False,upstream_transport=instant_transport(output/'instant-runtime')) as setup:
        base=CodexThreadOptions(role=CodexRole.WEAK_ACTOR,model=setup.model,provider=setup.provider,
            cwd=str(tools.workspace.root),sandbox=CodexSandbox.READ_ONLY,ephemeral=False,config=setup.thread_config,
            offered_tools=tools.offers,base_instructions=BASE,developer_instructions=DEVELOPER)
        options=instant_profile().thread_options(base)
        turn=instant_profile().turn_input(CodexTurnInput(public_text=(bundle/'public/instruction.md').read_text(),
            public_context={'task_id':'clinical_trial_matching_task_6','stage':'E2E','native_submission_path':'/workspace/submission/eligible_trials.txt'},
            model=setup.model,summary='none'))
        logical=canonical_value(_logical_input(options,turn))
        write(output/'request.json',{'logical_input':logical,'base_instructions':options.base_instructions,
            'developer_instructions':options.developer_instructions,'offered_tools':tools.offers,
            'config':options.config,'provider':setup.safe_metadata,'native_instruction_unchanged':True,
            'native_harbor_agent_used':False,'benchmark_terminal_binding':'eva.healthagent-terminal-execution.v1',
            'network_adaptation':'native allows internet; this user-authorized public-prepared run has network none'})
        write(output/'supra-profile.json',instant_profile().inspection())
        bridge=CodexToolExecutionBridge(tools,id_factory=tools.ids)
        with TemporaryDirectory(prefix='eva-healthagent-mcp-',dir='/tmp') as temporary:
            factory=TurnMCPBridgeFactory(proxy_python=Path(sys.executable).resolve(),
                proxy_script=Path(inspect.getfile(TurnMCPBridgeFactory)).with_name('turn_mcp_proxy.py'),temp_root=Path(temporary),maximum_parallel_calls=1)
            with factory.open_actor(options,bridge) as bound:
                async def lifecycle():
                    async with CodexRuntime(OpenAICodexBackend(memory_launch_options(setup.launch))) as runtime:
                        handle=await runtime.start_thread(bound)
                        write(output/'thread-start.json',{'thread_id':handle.thread_id,'started_utc':datetime.now(timezone.utc).isoformat()})
                        try:
                            return await runtime.run_turn(handle,turn,policy_timeout_seconds=3600,interruption_grace_seconds=60),None
                        except CodexPolicyBudgetExceeded as error:
                            return error.receipt,error.outcome
                try:
                    receipt,budget=asyncio.run(lifecycle())
                    write(output/'receipt.json',receipt)
                    verify_codex_turn_receipt(receipt)
                    outcome.update(status=receipt.status,receipt_blake3=receipt.receipt_blake3,
                        final_response_present=bool(receipt.final_response),policy_budget=budget)
                    # Projection failure must not erase a real terminal result or submission.
                    try:
                        groups=_validate_tool_inventory(receipt,tools.offers)
                        mapping,trace=bridge.bind_receipt(receipt,_mcp_groups(groups,set(receipt.offered_mcp_tool_names)))
                        events=_project_events(tools.ids,system=(options.base_instructions or '')+'\n'+(options.developer_instructions or ''),
                            user=logical,receipt=receipt,groups=groups,actor_results=mapping,retain_full_receipt=False)
                        write(output/'policy-events.json',events)
                        write(output/'tool-trace.json',trace)
                        outcome['visible_evidence_projection_valid']=True
                    except Exception as error:
                        outcome.update(visible_evidence_projection_valid=False,projection_error_type=type(error).__name__)
                except Exception as error:
                    if getattr(error,'receipt',None) is not None and not (output/'receipt.json').exists():
                        write(output/'receipt.json',error.receipt)
                    outcome.update(status='failed',error_type=type(error).__name__,
                        error_category=getattr(error,'safe_failure_category',None),budget_outcome=getattr(error,'outcome',None))
    write(output/'workspace-after.json',tools.workspace.snapshot('terminal-mutable-only'))
    if not (output/'tool-trace.json').exists(): write(output/'tool-trace.json',tools.trace())
    outcome.update(actual_tool_calls=len(tools.calls),workspace=str(tools.workspace.root),
        submission_present=(tools.workspace.root/'submission/eligible_trials.txt').is_file())
    write(output/'outcome.json',outcome)
    print(json.dumps({'event':'actor_terminal',**outcome}),flush=True)
    return outcome,tools.workspace.root


def run(args):
    if args.output.exists(): raise ValueError('fresh_attempt_root_required')
    binding=public_binding(args.bundle)
    image=resolve_image(args.image)
    server=serving_binding(args.server_canary,args.server_identity)
    args.output.mkdir(parents=True,mode=0o700)
    write(args.output/'public-input-binding.json',binding)
    write(args.output/'server-binding.json',server)
    version=subprocess.run([str(args.codex_bin),'--version'],capture_output=True,check=True,timeout=15).stdout.decode().strip()
    if version!='codex-cli 0.153.4': raise ValueError('pinned_codex_version_differs')
    sources=[]
    for source in (Path(__file__),ROOT/'training/healthagent_terminal_v1.py',Path(inspect.getfile(SupraProfile)),
            Path(inspect.getfile(ResearchContextPolicy)),Path(inspect.getfile(local_qwen_setup)),
            ROOT/'training/automedbench_lite/token_budget.py',args.native_scorer):
        payload=source.read_bytes()
        write(args.output/'sources'/source.name,payload)
        sources.append({'path':str(source),'bytes':len(payload),'blake3':blake3_bytes(payload)})
    write(args.output/'attempt.json',{'schema':'eva.healthagent-native-instant-attempt.v1','attempt_id':str(uuid4()),
        'start_utc':datetime.now(timezone.utc).isoformat(),'actor_pid':os.getpid(),'planned_actor_attempts':1,
        'task_id':'clinical_trial_matching_task_6','mode':'instant','enable_thinking':False,'image':image,
        'codex_version':version,'codex_binary_blake3':blake3_bytes(args.codex_bin.read_bytes()),'sources':sources,
        'native_budget_seconds':3600,'cpu':2,'memory_mb':2048,'gpu':0,'network':'none','stage':'E2E',
        'native_harbor_executed':False,'max_terminal_call_seconds':180,'same_harbor_runtime_claimed':False})
    outcome,workspace=run_actor(args.output/'actor',args.bundle,args,image)
    final={'actor':outcome,'native_score_status':'not_run_no_submission','native_reward':None}
    if outcome['submission_present']:
        spec=importlib.util.spec_from_file_location('retained_native_score_wrapper',args.native_scorer)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        try:
            score=module.score(args.bundle,workspace/'submission/eligible_trials.txt',args.output/'native-score')
            final.update(native_score_status='complete',native_reward=score['native_reward'])
        except Exception as error:
            final.update(native_score_status='failed',native_score_error_type=type(error).__name__)
    write(args.output/'result.json',final)
    print(json.dumps({'event':'attempt_terminal','actor_status':outcome['status'],
        'native_score_status':final['native_score_status'],'native_reward':final['native_reward']}),flush=True)
    return 0 if outcome['status']=='completed' and final['native_score_status']=='complete' else 1
