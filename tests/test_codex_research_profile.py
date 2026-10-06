from dataclasses import fields
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.codex_runtime import CodexLaunchOptions
from eva_agent.codex_runtime import research_profile as profile


def test_research_features_retain_medical_coding_memory_and_no_security_overrides():
    document = profile.inspect_profile()
    value = document['features']
    for name in ('plugins', 'skill_search', 'shell_tool', 'unified_exec', 'view_image', 'multi_agent', 'memories'):
        assert value[name] is True
    for name in ('apps', 'image_generation', 'personality', 'remote_plugin', 'recommended_plugins',
                 'in_app_chat', 'in_app_dictation', 'in_app_updates', 'realtime_conversation',
                 'code_mode', 'code_mode_host', 'context_management'):
        assert value[name] is False
    assert 'apply_patch_freeform' not in value and 'tool_search_always_defer_mcp_tools' not in value
    assert 'web_search' not in value
    assert document['provider_calls'] == 0
    assert document['speed_gain_measured'] is document['memory_behavior_verified'] is False


def test_sdk_composition_preserves_every_other_field_and_caller_restrictions():
    base = CodexLaunchOptions(codex_bin='/pinned/codex', cwd='/research/project',
        env={'PRIVATE_FIXTURE_KEY': 'not-output', 'PATH': '/custom/bin'},
        launch_args_override=('app-server', '--strict-config'),
        config_overrides=('model="caller-model"', 'model_provider="caller-provider"',
            'approval_policy="on-request"', 'sandbox_mode="read-only"',
            'features.shell_tool=false', 'features.view_image=false', 'features.multi_agent=false'))
    result = profile.research_launch_options(base)
    assert result.config_overrides[-len(base.config_overrides):] == base.config_overrides
    for field in fields(base):
        if field.name != 'config_overrides':
            assert getattr(result, field.name) == getattr(base, field.name)
    effective = dict(override.split('=', 1) for override in result.config_overrides)
    assert effective['features.shell_tool'] == effective['features.view_image'] == 'false'
    assert effective['features.multi_agent'] == 'false'
    assert effective['model'] == '"caller-model"'


@pytest.mark.parametrize('content', [
    '[features]\napps=false\nmodel="bad"',
    '[features]\napps=false\n[model_providers]\nx="bad"',
    '[features]\napps="false"', '[features]\n',
])
def test_profile_rejects_unrelated_or_nonboolean_settings(tmp_path, content):
    path = tmp_path / 'profile.toml'; path.write_text(content)
    with pytest.raises(profile.ResearchProfileError):
        profile.load_profile(path)


def feature_output(features, *, removed=None):
    return '\n'.join(f'{name} {"removed" if name == removed else "stable"} {str(value).lower()}'
                     for name, value in features.items())


def test_check_has_no_credentials_no_thread_or_turn_and_removes_private_check_home(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'PRIVATE_FIXTURE_NOT_IN_CHILD')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'PRIVATE_FIXTURE_NOT_IN_CHILD')
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert 'OPENAI_API_KEY' not in kwargs['env'] and 'ANTHROPIC_API_KEY' not in kwargs['env']
        assert kwargs['cwd'] == Path(kwargs['env']['CODEX_HOME'])
        return SimpleNamespace(stdout=profile.PINNED_CODEX_VERSION if argv[-1] == '--version'
                               else feature_output(profile.load_profile()))
    monkeypatch.setattr(profile.subprocess, 'run', run)
    initialize = []
    monkeypatch.setattr(profile, '_strict_initialize', lambda *args, **kw: initialize.append(kw))
    result = profile.check_profile('/pinned/codex')
    assert len(calls) == 2 and len(initialize) == 1
    assert result['valid'] is result['strict_config_initialize_passed'] is True
    assert result['thread_requests'] == result['turn_requests'] == 0
    assert result['auth_material_copied'] is result['global_config_writes'] is False
    assert not initialize[0]['cwd'].exists()


@pytest.mark.parametrize('failure', ['wrong_version', 'removed', 'missing', 'wrong_value'])
def test_actual_feature_or_version_mismatch_fails_before_initialize(monkeypatch, failure):
    def run(argv, **kwargs):
        if argv[-1] == '--version':
            return SimpleNamespace(stdout='codex-cli 0.147.0' if failure == 'wrong_version'
                                   else profile.PINNED_CODEX_VERSION)
        features = profile.load_profile()
        if failure == 'missing': features.pop('apps')
        if failure == 'wrong_value': features['apps'] = True
        return SimpleNamespace(stdout=feature_output(features, removed='apps' if failure == 'removed' else None))
    monkeypatch.setattr(profile.subprocess, 'run', run)
    monkeypatch.setattr(profile, '_strict_initialize', lambda *a, **kw: pytest.fail('must fail before startup'))
    with pytest.raises(profile.ResearchProfileError): profile.check_profile('/pinned/codex')


def test_strict_check_sends_only_initialize_and_closes_owned_child(tmp_path, monkeypatch):
    reader, writer = os.pipe()
    os.write(writer, b'{"id":1,"result":{"userAgent":"fixture"}}\n'); os.close(writer)
    requests = []
    class Input(io.BytesIO):
        def close(self):
            requests.extend(json.loads(line) for line in self.getvalue().splitlines())
            super().close()
    process = SimpleNamespace(stdin=Input(), stdout=os.fdopen(reader, 'rb'),
                              wait=lambda **kw: 0)
    def popen(argv, **kwargs):
        assert argv[-4:] == ['app-server', '--strict-config', '--listen', 'stdio://']
        return process
    monkeypatch.setattr(profile.subprocess, 'Popen', popen)
    profile._strict_initialize('/pinned/codex', profile_path=profile.DEFAULT_PROFILE, env={}, cwd=tmp_path)
    assert [row['method'] for row in requests] == ['initialize']
    assert process.stdout.closed and process.stdin.closed


def test_run_is_explicit_passes_caller_arguments_last_and_preserves_environment(monkeypatch):
    monkeypatch.setattr(profile, 'resolve_binary', lambda value: '/pinned/codex')
    monkeypatch.setattr(profile, 'check_profile', lambda *a, **kw: {'valid': True})
    observed = []
    monkeypatch.setattr(profile.subprocess, 'call', lambda argv: observed.append(argv) or 7)
    assert profile.main(['--run', '--', 'exec', '--model', 'caller', '-c', 'features.shell_tool=false', 'task']) == 7
    assert observed[0][-6:] == ['exec', '--model', 'caller', '-c', 'features.shell_tool=false', 'task']
    assert observed[0].index('features.shell_tool=true') < observed[0].index('features.shell_tool=false')


def test_inspect_never_resolves_binary_or_launches_process(monkeypatch, capsys):
    monkeypatch.setattr(profile, 'resolve_binary', lambda *args: pytest.fail('inspect is file-only'))
    assert profile.main(['--inspect']) == 0
    assert json.loads(capsys.readouterr().out)['provider_calls'] == 0
