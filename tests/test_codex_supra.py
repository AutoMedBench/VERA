from dataclasses import fields, replace
from types import SimpleNamespace as NS
import json

import pytest

from eva_agent.codex_runtime.contracts import CodexRole, CodexSandbox, CodexThreadOptions, CodexTurnInput
from eva_agent.codex_runtime.research_memory import ResearchContextPolicy
from eva_agent.codex_runtime.supra import SupraChatTransport, SupraMode, SupraProfile


def profile(**kwargs):
    return SupraProfile(**{"model": "Qwen/Qwen3.5-9B", "provider": "eva_local_qwen",
        "mode": SupraMode.INSTANT, "context": ResearchContextPolicy(),
        "protocol": "qwen_template", "local_qwen_endpoint": "http://127.0.0.1:30910/v1", **kwargs})


def test_native_think_preserves_tools_and_explicit_security():
    p = profile(model="gpt-6-astra", provider="native", protocol="codex_effort",
        local_qwen_endpoint=None, mode=SupraMode.THINK,
        context=ResearchContextPolicy(context_tokens=131072, compact_at_tokens=49152),
        measured_compacted_input_tokens=13400)
    base = CodexThreadOptions(role=CodexRole.STRONG_ACTOR, model=p.model, provider=p.provider,
        cwd="/public/task", sandbox=CodexSandbox.READ_ONLY, ephemeral=False,
        config={"features": {"shell_tool": False}})
    enhanced = p.thread_options(base)
    assert enhanced.offered_tools == base.offered_tools
    assert enhanced.config["features"]["shell_tool"] is False
    assert enhanced.config["model_auto_compact_token_limit"] == 49152
    assert enhanced.sandbox == base.sandbox and not enhanced.ephemeral
    turn = p.turn_input(CodexTurnInput(public_text="Do this stage", public_context={"stage": "S2"}))
    assert turn.effort == "xhigh" and turn.skills[-1].skill_id == "summary_failures"
    assert turn.public_context["stage"] == "S2"


@pytest.mark.parametrize("mode, expected", [(SupraMode.INSTANT, False), (SupraMode.THINK, True)])
def test_qwen_mode_reaches_actual_transport_without_mutating_tools(mode, expected):
    p = profile(mode=mode)
    tool = {"type": "function", "function": {"name": "research_read",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}
    original = {"model": p.model, "tools": [tool], "messages": [{"role": "user", "content": "private fixture"}],
        "chat_template_kwargs": {"preserve_fixture": True, "enable_thinking": not expected}}
    sent, observations = [], []
    response = NS(status=200, body=json.dumps({"choices": [{"message": {"content": "visible", "reasoning_content": "never log"}}],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 3}}}).encode())
    transport = SupraChatTransport(lambda binding, body: (sent.append(body), response)[1], p, observer=observations.append)
    assert transport(NS(model_id=p.model), original) is response
    assert sent[0]["chat_template_kwargs"] == {"preserve_fixture": True, "enable_thinking": expected}
    assert sent[0]["tools"] == original["tools"] and sent[0]["messages"] == original["messages"]
    assert original["chat_template_kwargs"]["enable_thinking"] is not expected
    assert "private fixture" not in json.dumps(observations) and "never log" not in json.dumps(observations)
    assert observations[-1]["visible_content_characters"] == 7
    assert observations[-1]["thinking_execution_verified"] is False


def test_cloud_instant_is_not_silently_relabelled_low():
    with pytest.raises(ValueError, match="reserved for local Qwen"):
        profile(local_qwen_endpoint=None)
    with pytest.raises(ValueError, match="local Qwen"):
        profile(local_qwen_endpoint="https://provider.example/v1")


def test_measured_compaction_floor_rejects_both_default_and_caller_churn():
    with pytest.raises(ValueError, match="retained-context floor"):
        profile(measured_compacted_input_tokens=13300)
    p = profile(measured_compacted_input_tokens=5000)
    base = CodexThreadOptions(role=CodexRole.WEAK_ACTOR, model=p.model, provider=p.provider,
        cwd="/public/task", sandbox=CodexSandbox.READ_ONLY, config={"model_auto_compact_token_limit": 6000})
    with pytest.raises(ValueError, match="recreates observed churn"):
        p.thread_options(base)


def test_explicit_chat_reasoning_control_and_no_retry():
    p = profile(mode=SupraMode.THINK, local_qwen_endpoint=None, protocol="chat_reasoning_effort")
    sent = []
    response = NS(status=429, body=b"{}")
    wrapper = SupraChatTransport(lambda binding, body: (sent.append(body), response)[1], p)
    assert wrapper(NS(model_id=p.model), {"model": p.model}) is response
    assert len(sent) == 1 and sent[0]["reasoning_effort"] == "xhigh"
    assert not p.inspection()["provider_support_verified"]


def test_cannot_use_profile_for_different_route_or_judge():
    p = profile()
    base = CodexThreadOptions(role=CodexRole.JUDGE, model=p.model, provider=p.provider,
        cwd="/private/judge", sandbox=CodexSandbox.READ_ONLY)
    with pytest.raises(ValueError, match="judge"):
        p.thread_options(base)
    with pytest.raises(ValueError, match="model differs"):
        p.turn_input(CodexTurnInput(public_text="Task", model="another-model"))
    with pytest.raises(ValueError, match="binding differs"):
        SupraChatTransport(lambda *_: None, p)(NS(model_id="other"), {"model": p.model})


def test_measured_local_context_with_explicit_exact_request_guard():
    policy = ResearchContextPolicy(context_tokens=32768, compact_at_tokens=20480,
        output_tokens=4096, reserve_tokens=2048, compaction_headroom_mode="exact_request_guard")
    p = profile(mode=SupraMode.THINK, context=policy, measured_compacted_input_tokens=13100)
    base = CodexThreadOptions(role=CodexRole.WEAK_ACTOR, model=p.model, provider=p.provider,
        cwd="/public/task", sandbox=CodexSandbox.READ_ONLY)
    enhanced = p.thread_options(base)
    assert enhanced.config["model_auto_compact_token_limit"] == 20480
    assert p.inspection()["context_tokens"] == 32768
    assert p.inspection()["compaction_headroom_mode"] == "exact_request_guard"
    assert p.turn_input(CodexTurnInput(public_text="Task")).skills[-1].skill_id == "summary_failures"
    with pytest.raises(ValueError, match="headroom"):
        ResearchContextPolicy(compact_at_tokens=28000, compaction_headroom_mode="exact_request_guard")
    with pytest.raises(ValueError, match="headroom mode"):
        ResearchContextPolicy(compaction_headroom_mode="unbounded")
