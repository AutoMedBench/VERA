"""Adapter setup only: no inference request, server launch, or GPU use."""
import json
import pytest

from eva_agent.codex_providers.adapter import _UpstreamOutcome
from training.automedbench_lite.local_qwen import local_qwen_setup, ThinkingChatTransport


def test_local_gateway_can_enter_with_existing_credential_contract(tmp_path):
    with local_qwen_setup(run_root=tmp_path, workers=1) as setup:
        assert setup.safe_metadata["provider_calls_on_setup"] == 0
        assert setup.safe_metadata["inference_server_launched"] is False
        assert setup.safe_metadata["thinking_explicitly_enabled"] is False
        assert setup.model == "Qwen/Qwen3.5-9B"
        provider = setup.thread_config["model_providers"][setup.provider]
        assert provider["env_key"] == "OPENAI_API_KEY"
        assert provider["request_max_retries"] == 0
        assert not tuple(tmp_path.glob("**/adapter-receipt.json"))


def test_local_gateway_does_not_accept_remote_endpoint(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        with local_qwen_setup(run_root=tmp_path, endpoint="https://example.invalid/v1"):
            pytest.fail("remote endpoint accepted")


def test_explicit_thinking_setup_is_provider_free(tmp_path):
    with local_qwen_setup(run_root=tmp_path, workers=1, thinking=True) as setup:
        assert setup.safe_metadata["thinking_explicitly_enabled"] is True
        assert setup.safe_metadata["thinking_observed_by_this_setup"] is False
        assert not (tmp_path / "thinking-controls").exists()


def test_injected_training_transport_is_provider_free_on_setup(tmp_path):
    calls = []

    def transport(binding, body):
        calls.append(body)
        raise AssertionError("setup must not generate tokens")

    with local_qwen_setup(run_root=tmp_path, workers=1, thinking=True,
                          upstream_transport=transport) as setup:
        assert setup.safe_metadata["provider_calls_on_setup"] == 0
        assert not calls
        assert not (tmp_path / "thinking-controls").exists()


def test_thinking_forwarded_without_recording_private_output(tmp_path):
    captured = []
    raw = {"choices": [{"message": {"reasoning_content": "PRIVATE SENTINEL", "content": "answer"}}],
           "usage": {"completion_tokens_details": {"reasoning_tokens": 31}}}
    result = _UpstreamOutcome(200, json.dumps(raw).encode(), 10)
    def transport(binding, body):
        captured.append(body)
        return result
    original = {"messages": [{"role": "user", "content": "PROMPT SENTINEL"}], "max_tokens": 8192,
                "chat_template_kwargs": {"enable_thinking": False, "other": "preserved"}}
    wrapper = ThinkingChatTransport(transport, tmp_path)
    assert wrapper(None, original) is result
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": True, "other": "preserved"}
    assert original["chat_template_kwargs"]["enable_thinking"] is False
    outcome = json.loads(next(tmp_path.glob("*/outcome.json")).read_text())
    assert outcome["reasoning_tokens"] == 31
    assert outcome["reasoning_field_nonempty"] is True
    for path in tmp_path.glob("*/*.json"):
        assert "SENTINEL" not in path.read_text()


def test_thinking_transport_failure_not_retried(tmp_path):
    calls = []
    def transport(binding, body):
        calls.append(body)
        raise TimeoutError("PRIVATE SENTINEL")
    with pytest.raises(TimeoutError):
        ThinkingChatTransport(transport, tmp_path)(None, {})
    assert len(calls) == 1
    outcome = next(tmp_path.glob("*/outcome.json")).read_text()
    assert "SENTINEL" not in outcome
    assert json.loads(outcome)["status"] == "transport_error"


def test_explicit_compaction_is_requested_not_a_behavioral_pass(tmp_path):
    with local_qwen_setup(run_root=tmp_path, workers=1, auto_compact_token_limit=20000) as setup:
        assert setup.thread_config["model_auto_compact_token_limit"] == 20000
        assert setup.safe_metadata["compaction_observed_by_this_setup"] is False


def test_compaction_reserves_output_space(tmp_path):
    with pytest.raises(ValueError, match="compaction"):
        with local_qwen_setup(run_root=tmp_path, auto_compact_token_limit=32000):
            pytest.fail("no output headroom")


def test_exact_chat_token_budget_setup_is_provider_free(tmp_path):
    with local_qwen_setup(run_root=tmp_path, workers=1, thinking=True,
                          token_budget=True) as setup:
        assert setup.safe_metadata["text_token_budget_enabled"] is True
        assert setup.safe_metadata["multimodal_token_count_verified_by_this_setup"] is False
        assert list((tmp_path / "token-budget").iterdir()) == []
        assert not (tmp_path / "thinking-controls").exists()


def test_chat_tokenizer_cannot_wrap_exact_native_training_transport(tmp_path):
    with pytest.raises(ValueError, match="custom training transport"):
        with local_qwen_setup(run_root=tmp_path, token_budget=True,
                              upstream_transport=lambda *_: None):
            pytest.fail("exact-token transport received redundant Chat tokenizer")
