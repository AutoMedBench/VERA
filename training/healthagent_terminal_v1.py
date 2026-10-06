"""Separate benchmark terminal binding; never a canonical EvaMed replacement.

Every actor command executes in an inspected, network-free CPU Docker container.
Only public task inputs and two mutable directories are mounted. The host's
Docker control plane is trusted harness code, not exposed to the actor.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from uuid import uuid4

from eva_agent.pipeline.digests import blake3_bytes, canonical_json_bytes
from training.automedbench_lite.docker_runtime import bounded_start, resolve_image

TOOL_NAME = 'healthagent_terminal_v1'
TOOL_DESCRIPTION = (
    'Run a Bash command (including Python3 scripts) in an isolated CPU2/RAM2GB/GPU0 Linux container. '
    'This is a separate HealthAgentBench evaluation terminal, not a canonical medical tool. '
    'Working directory /workspace. Public /workspace/data, /workspace/topic_id.txt and '
    '/workspace/trial_ncts.txt are read-only; /workspace/submission and /workspace/notes persist and are writable. '
    'No network, credentials, gold labels, evaluator, additional model or Docker socket is available. '
    'Processes do not persist between calls. Use /tmp for temporary files. Stdout/stderr each return at most '
    '16384 characters with an explicit truncation flag; redirect larger results into notes/submission and inspect bounded pieces. '
    'The native submission is /workspace/submission/eligible_trials.txt. Timeout1–180 seconds per call.'
)
TOOL_SCHEMA = {'type':'object','properties':{
    'command':{'type':'string','minLength':1,'maxLength':65536},
    'timeout_seconds':{'type':'integer','minimum':1,'maximum':180,'default':120}},
    'required':['command'],'additionalProperties':False}


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    body = value if isinstance(value, bytes) else canonical_json_bytes(value)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(body)
    return blake3_bytes(body)


def public_binding(bundle: Path):
    bundle = bundle.resolve(strict=True)
    preparation = json.loads((bundle / 'preparation.json').read_bytes())
    if preparation.get('status') != 'complete' or preparation.get('source_revision') != 'bcbb8085fd549469e2dc7455f4bfd68a1b98895a':
        raise ValueError('complete_pinned_task6_required')
    public = bundle / 'public/workspace'
    manifest_path = bundle / 'public/input-manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    seen = set()
    for row in manifest['files']:
        relative = Path(row['path'])
        source = public / relative
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen or source.is_symlink() or not source.resolve().is_relative_to(public):
            raise ValueError('public_input_topology_differs')
        payload = source.read_bytes()
        if len(payload) != row['bytes'] or blake3_bytes(payload) != row['blake3']:
            raise ValueError('public_input_commitment_differs')
        seen.add(str(relative))
    actual = {str(path.relative_to(public)) for path in public.rglob('*') if path.is_file()}
    if actual != seen or any(path.is_symlink() for path in public.rglob('*')):
        raise ValueError('public_input_inventory_differs')
    return {'source_revision':preparation['source_revision'], 'manifest_blake3':blake3_bytes(manifest_path.read_bytes()),
        'files':manifest['files'], 'file_count':len(seen), 'total_bytes':sum(row['bytes'] for row in manifest['files']),
        'immutable_public_root':str(public), 'private_references_mounted':False}


def mounts(public: Path, workspace: Path, script: Path):
    return [(public / 'data', '/workspace/data', False),
        (public / 'topic_id.txt', '/workspace/topic_id.txt', False),
        (public / 'trial_ncts.txt', '/workspace/trial_ncts.txt', False),
        (workspace / 'instruction.md', '/workspace/instruction.md', False),
        (workspace / 'submission', '/workspace/submission', True),
        (workspace / 'notes', '/workspace/notes', True),
        (script, '/eva-command.sh', False)]


def create_command(*, image, name, public, workspace, script):
    if re.fullmatch(r'sha256:[a-f0-9]{64}', image) is None:
        raise ValueError('immutable_image_required')
    command = ['docker','create','--pull','never','--name',name,'--network','none','--read-only',
        '--cap-drop','ALL','--security-opt','no-new-privileges:true','--pids-limit','64',
        '--memory','2048m','--memory-swap','2048m','--cpus','2','--ulimit','nofile=64:64',
        '--ulimit','fsize=134217728:134217728','--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=256m',
        '--user',f'{os.getuid()}:{os.getgid()}','--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1',
        '--env','OPENBLAS_NUM_THREADS=2','--env','OMP_NUM_THREADS=2']
    for source, destination, writable in mounts(public, workspace, script):
        command += ['--mount',f'type=bind,source={source},target={destination}' + ('' if writable else ',readonly')]
    return command + ['--workdir','/workspace','--entrypoint','bash',image,'/eva-command.sh']


def verify_container(document, *, image, public, workspace, script):
    host, config = document['HostConfig'], document['Config']
    actual = {(row['Source'],row['Destination'],row['RW']) for row in document['Mounts'] if row['Type']=='bind'}
    expected = {(str(source),target,rw) for source,target,rw in mounts(public,workspace,script)}
    if (document['Image'] != image or host['NetworkMode'] != 'none' or not host['ReadonlyRootfs']
        or host['Privileged'] or set(host['CapDrop'] or []) != {'ALL'}
        or not any('no-new-privileges' in item for item in host['SecurityOpt'] or [])
        or host.get('DeviceRequests') or host.get('Devices') or host.get('PidMode') == 'host'
        or host['Memory'] != 2147483648 or host['MemorySwap'] != 2147483648
        or host['NanoCpus'] != 2000000000 or host['PidsLimit'] != 64
        or config['User'] != f'{os.getuid()}:{os.getgid()}' or actual != expected
        or config['Entrypoint'] != ['bash'] or config['WorkingDir'] != '/workspace'):
        raise ValueError('native_path_docker_isolation_differs')
    return {'verified_before_start':True,'image':image,'network':'none','cpu':2,'memory_bytes':2147483648,
        'gpu':0,'readonly_root':True,'gold_or_verifier_or_socket_mounted':False,
        'native_paths_preserved':True,'mutable_paths':['/workspace/submission','/workspace/notes']}


def execute_terminal(*, public, workspace, image, audit_root, command, timeout_seconds=120):
    if not isinstance(command,str) or not command.strip() or len(command.encode())>65536 or type(timeout_seconds) is not int or not 1<=timeout_seconds<=180:
        raise ValueError('terminal_request_invalid')
    public,workspace,audit_root = (Path(path).resolve() for path in (public,workspace,audit_root))
    if audit_root.is_relative_to(workspace) or workspace.is_relative_to(audit_root):
        raise ValueError('terminal_audit_must_be_external')
    for relative in ('submission','notes'):
        target = workspace / relative
        if target.is_symlink() or not target.is_dir(): raise ValueError('mutable_mount_topology_differs')
    execution_id = str(uuid4())
    audit = audit_root / execution_id
    audit.mkdir(parents=True,mode=0o700)
    script = audit / 'actor-command.sh'
    write(script,command.encode())
    name = 'eva-healthagent-' + execution_id
    try:
        result = subprocess.run(create_command(image=image,name=name,public=public,workspace=workspace,script=script),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
        if result.returncode: raise ValueError('terminal_container_create_failed')
        inspect = subprocess.run(['docker','inspect',name],capture_output=True,timeout=15,check=True)
        isolation = verify_container(json.loads(inspect.stdout)[0],image=image,public=public,workspace=workspace,script=script)
        write(audit / 'isolation.json',isolation)
        outcome = bounded_start(name,timeout_seconds)
        inspect = subprocess.run(['docker','inspect',name],capture_output=True,timeout=15,check=True)
        state = json.loads(inspect.stdout)[0]['State']
        receipt = {'schema':'eva.healthagent-terminal-execution.v1','execution_id':execution_id,
            'author':'actual_actor_command','command_blake3':blake3_bytes(script.read_bytes()),'isolation':isolation,
            'exit_code':state['ExitCode'],'oom_killed':state['OOMKilled'],'started_at':state['StartedAt'],
            'finished_at':state['FinishedAt'],**outcome}
        commitment = write(audit / 'execution.json',receipt)
        public_result = {key:receipt[key] for key in ('execution_id','exit_code','oom_killed','timed_out','stream_limit_exceeded')}
        for stream in ('stdout','stderr'):
            public_result[stream] = receipt[stream][:16384]
            public_result[stream+'_truncated'] = len(receipt[stream])>16384
        return {**public_result,'execution_receipt_blake3':commitment}
    finally:
        subprocess.run(['docker','rm','--force',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
