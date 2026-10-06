import json
import os
from pathlib import Path
import pytest

from training.healthagent_terminal_v1 import create_command,verify_container,execute_terminal,TOOL_SCHEMA

IMAGE='sha256:'+'a'*64


def test_exact_native_public_mounts_and_no_network(tmp_path):
    command=create_command(image=IMAGE,name='eva-healthagent-test',public=tmp_path/'public',
        workspace=tmp_path/'work',script=tmp_path/'audit/command.sh')
    assert command[command.index('--network')+1]=='none'
    assert '--gpus' not in command
    binds=[command[i+1] for i,x in enumerate(command) if x=='--mount']
    assert len(binds)==7
    assert any('target=/workspace/data,readonly' in x for x in binds)
    assert any('target=/workspace/submission' in x and not x.endswith(',readonly') for x in binds)
    assert not any('/var/run/docker.sock' in x or 'gold' in x or 'verifier' in x for x in binds)
    assert command[-2:]==[IMAGE,'/eva-command.sh']


def test_schema_is_separate_and_bounded():
    assert TOOL_SCHEMA['required']==['command']
    assert TOOL_SCHEMA['additionalProperties'] is False
    assert TOOL_SCHEMA['properties']['timeout_seconds']['maximum']==180


@pytest.mark.parametrize('command,timeout',[('',1),(' ',1),('x'*65537,1),('true',0),('true',181),('true',True)])
def test_invalid_command_fails_before_docker(tmp_path,command,timeout):
    with pytest.raises(ValueError,match='terminal_request_invalid'):
        execute_terminal(public=tmp_path/'public',workspace=tmp_path/'work',image=IMAGE,
            audit_root=tmp_path/'audit',command=command,timeout_seconds=timeout)


def test_unsafe_image_rejected(tmp_path):
    with pytest.raises(ValueError,match='immutable_image'):
        create_command(image='latest',name='test',public=tmp_path,workspace=tmp_path,script=tmp_path/'a')


def test_instant_exact_false_with_cpu_transport(tmp_path):
    from training.healthagent_instant_v1 import instant_profile
    from eva_agent.codex_runtime.supra import SupraChatTransport
    from types import SimpleNamespace
    seen=[]
    def transport(binding,body):
        seen.append(body)
        return SimpleNamespace(status=200,body=b'{"choices":[{"message":{"content":"fixture"}}]}')
    profile=instant_profile()
    SupraChatTransport(transport,profile)(SimpleNamespace(model_id=profile.model),{'model':profile.model})
    assert seen[0]['chat_template_kwargs']['enable_thinking'] is False


def test_sdk_handle_and_tool_runtime_imports():
    from eva_agent.codex_runtime.contracts import CodexThreadHandle
    from training.healthagent_instant_v1 import CapturedTools
    assert 'thread_id' in CodexThreadHandle.__dataclass_fields__
    assert callable(CapturedTools.execute)
