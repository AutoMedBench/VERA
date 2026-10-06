"""Prospective deadline wiring only; no provider, model, or GPU calls."""
from types import SimpleNamespace

import pytest

from eva_agent.codex_providers.adapter import AdapterLimits, ChatCompletionsTransport
from training.automedbench_lite import local_qwen


@pytest.mark.parametrize("token_budget", [False, True])
def test_explicit_long_deadline_reaches_chat_transport_without_generation(tmp_path, monkeypatch, token_budget):
    configured = []
    def construct(limits):
        configured.append(limits)
        def no_generation(*_):
            pytest.fail("setup attempted provider generation")
        return no_generation
    monkeypatch.setattr(local_qwen, "ChatCompletionsTransport", construct)
    with local_qwen.local_qwen_setup(run_root=tmp_path, workers=1, thinking=True,
            token_budget=token_budget, upstream_timeout_seconds=600) as setup:
        assert len(configured) == 1 and configured[0].upstream_timeout_seconds == 600
        assert setup.safe_metadata["upstream_timeout_seconds"] == 600
        assert setup.safe_metadata["upstream_timeout_scope"] == "local-chat-http-transport"
        assert setup.safe_metadata["provider_calls_on_setup"] == 0
    assert AdapterLimits().upstream_timeout_seconds == 300


def test_local_default_remains_300_and_custom_transport_does_not_claim_deadline(tmp_path):
    with local_qwen.local_qwen_setup(run_root=tmp_path / 'default', workers=1) as setup:
        assert setup.safe_metadata['upstream_timeout_seconds'] == 300
    with local_qwen.local_qwen_setup(run_root=tmp_path / 'custom', workers=1,
            upstream_transport=lambda *_: None, upstream_timeout_seconds=600) as setup:
        assert setup.safe_metadata['upstream_timeout_seconds'] is None
        assert setup.safe_metadata['upstream_timeout_scope'] == 'custom-transport-owned'


@pytest.mark.parametrize('value', [0, 901, True, float('nan'), '600'])
def test_invalid_deadline_rejected_before_setup(tmp_path, value):
    with pytest.raises(ValueError, match='timeout'):
        with local_qwen.local_qwen_setup(run_root=tmp_path, upstream_timeout_seconds=value):
            pytest.fail('invalid timeout accepted')
    assert not list(tmp_path.iterdir())


def test_actual_http_transport_passes_deadline_to_opener_once():
    transport = ChatCompletionsTransport(AdapterLimits(upstream_timeout_seconds=600))
    calls = []
    def open_request(request, *, timeout):
        calls.append(timeout)
        raise TimeoutError('synthetic socket deadline; no network call')
    transport._opener = SimpleNamespace(open=open_request)
    binding = SimpleNamespace(reveal_upstream_boundary=lambda: ('http://127.0.0.1:30910/v1', 'local-nonsecret'))
    with pytest.raises(TimeoutError):
        transport(binding, {'model': 'fixture', 'messages': []})
    assert calls == [600]
