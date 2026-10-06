from dataclasses import fields, replace

import pytest

from eva_agent.codex_runtime import CodexLaunchOptions
from eva_agent.codex_runtime.contracts import CodexRole, CodexSandbox, CodexThreadOptions, CodexTurnInput
from eva_agent.codex_runtime import research_memory as memory
from eva_agent.codex_runtime.runtime import _verified_skill_body


def actor(**kwargs):
    return CodexThreadOptions(role=CodexRole.WEAK_ACTOR, model='qwen3.5-9b-trained',
        provider='local-qwen', cwd='/public/task', sandbox=CodexSandbox.WORKSPACE_WRITE, **kwargs)


def test_exact_skill_bytes_mounted_and_idempotent_without_tool_schema_change():
    original = CodexTurnInput(public_text='Continue the medical task', public_context={'stage': 'S2'})
    enhanced = memory.memory_turn_input(original)
    assert enhanced.skills[0].skill_id == 'summary_failures'
    assert _verified_skill_body(enhanced.skills[0]) == memory.SKILL_PATH.read_text()
    assert memory.memory_turn_input(enhanced) is enhanced
    for field in fields(original):
        if field.name != 'skills':
            assert getattr(enhanced, field.name) == getattr(original, field.name)


def test_context_defaults_keep_model_tools_and_explicit_user_configuration():
    original = actor(config={'model_auto_compact_token_limit': 10240, 'features': {'shell_tool': False}},
                     developer_instructions='Use only public medical tools.', ephemeral=False)
    enhanced = memory.memory_thread_options(original)
    assert enhanced.config['model_auto_compact_token_limit'] == 10240
    assert enhanced.config['features']['shell_tool'] is False
    assert enhanced.developer_instructions.startswith(original.developer_instructions)
    for field in fields(original):
        if field.name not in ('config', 'developer_instructions', 'service_name'):
            assert getattr(enhanced, field.name) == getattr(original, field.name)
    assert memory.memory_thread_options(actor()).config['model_auto_compact_token_limit'] == 12288


@pytest.mark.parametrize('budget', [
    {'context_tokens': 8192}, {'compact_at_tokens': 20000}, {'output_tokens': 0},
    {'reserve_tokens': True},
])
def test_invalid_context_capacity_does_not_falsely_raise_model_limit(budget):
    with pytest.raises(ValueError):
        memory.ResearchContextPolicy(**budget)


def test_actor_memory_is_not_injected_into_private_judge():
    with pytest.raises(ValueError, match='judge'):
        memory.memory_thread_options(replace(actor(), role=CodexRole.JUDGE))


def test_conflicting_skill_cannot_replace_existing_recorded_bytes(tmp_path):
    other = tmp_path / 'different.md'
    other.write_text('A different, previously selected skill.')
    original = memory.memory_turn_input(CodexTurnInput(public_text='Work'), skill_path=other)
    with pytest.raises(ValueError, match='conflicts'):
        memory.memory_turn_input(original)


def test_launch_preserves_provider_and_security_and_inspection_is_not_behavioral_pass():
    original = CodexLaunchOptions(config_overrides=('features.memories=false', 'sandbox_mode="read-only"'),
                                  env={'PRIVATE_FIXTURE': 'not-printed'})
    enhanced = memory.memory_launch_options(original)
    assert enhanced.env == original.env
    assert enhanced.config_overrides[-2:] == original.config_overrides
    result = memory.inspect_memory_profile()
    assert result['provider_calls'] == 0
    assert not result['behavioral_memory_verified']
    assert not result['canonical_tool_schema_changes']
