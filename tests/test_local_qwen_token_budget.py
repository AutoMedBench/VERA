"""CPU mocked requests; never start a server or generate a real completion."""
from copy import deepcopy
import json
import stat

import httpx
import pytest

from eva_agent.codex_providers.adapter import ResponsesAdapterError, _UpstreamOutcome
from training.automedbench_lite.local_qwen import ThinkingChatTransport
from training.automedbench_lite.token_budget import TokenBudgetChatTransport, http_error_category


@pytest.fixture
def body():
    return {"model": "Qwen/Qwen3.5-9B", "messages": [{"role": "user", "content": "PRIVATE INPUT SENTINEL"}],
            "tools": [{"type": "function", "function": {"name": "mcp__fixture__read",
            "description": "canonical", "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}},
            "required": ["ids"], "additionalProperties": False}}}],
            "parallel_tool_calls": True, "tool_choice": "auto", "max_tokens": 8192,
            "temperature": 0.8, "top_p": 1.0, "chat_template_kwargs": {"enable_thinking": True}}


def build(tmp_path, *, count=3990, tokenize_status=200, tokenize_error=None, upstream_status=200,
          upstream_error=None, malformed=None):
    seen = {"tokenize": [], "generation": []}
    def tokenize(request):
        assert str(request.url) == 'http://127.0.0.1:30910/v1/tokenize'
        assert 'authorization' not in request.headers
        seen['tokenize'].append(json.loads(request.content))
        data = {"count": count, "tokens": [7] * count, "max_model_len": 262144}
        if tokenize_error:
            data = {"message": tokenize_error, "object": "error", "code": tokenize_status}
        if malformed:
            data.update(malformed)
        return httpx.Response(tokenize_status, json=data)
    def upstream(binding, request):
        seen['generation'].append(deepcopy(request))
        result = {"choices": [{"message": {"content": "visible fixture", "reasoning_content": "PRIVATE COT SENTINEL"}}]}
        if upstream_error:
            result = {"error": {"message": upstream_error}}
        return _UpstreamOutcome(upstream_status, json.dumps(result).encode(), 1)
    wrapped = TokenBudgetChatTransport(upstream, endpoint='http://127.0.0.1:30910/v1',
        context_length=4096, audit_root=tmp_path/'budget', margin=64,
        tokenize_http_transport=httpx.MockTransport(tokenize))
    return wrapped, seen


def receipt(tmp_path):
    return json.loads(next((tmp_path/'budget').glob('*/outcome.json')).read_bytes())


def test_counts_exact_render_fields_and_clamps_only_output(tmp_path, body):
    wrapper, seen = build(tmp_path)
    original = deepcopy(body)
    outcome = wrapper(None, body)
    assert outcome.status == 200 and len(seen['tokenize']) == len(seen['generation']) == 1
    counted = seen['tokenize'][0]
    assert counted['messages'] == body['messages'] and counted['tools'] == body['tools']
    assert counted['chat_template_kwargs'] == {'enable_thinking': True}
    assert counted['parallel_tool_calls'] is True and 'max_tokens' not in counted
    expected = {**body, 'max_tokens': 42}
    assert seen['generation'][0] == expected and body == original
    result = receipt(tmp_path)
    assert result['prompt_tokens'] == 3990 and result['effective_max_tokens'] == 42
    assert result['token_count_exact'] and result['output_budget_clamped'] and not result['input_truncated']
    for path in (tmp_path/'budget').glob('*/outcome.json'):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        text = path.read_text()
        assert 'SENTINEL' not in text and '"tokens"' not in text


def test_thinking_outer_wrapper_forwards_identical_controls_to_tokenize_and_generation(tmp_path, body):
    wrapper, seen = build(tmp_path, count=100)
    thinking = ThinkingChatTransport(wrapper, tmp_path/'thinking')
    body['chat_template_kwargs'] = {'enable_thinking': False}
    thinking(None, body)
    assert seen['tokenize'][0]['chat_template_kwargs'] == seen['generation'][0]['chat_template_kwargs'] == {'enable_thinking': True}
    assert body['chat_template_kwargs'] == {'enable_thinking': False}


def test_tool_free_compaction_transcript_is_counted_without_live_tools(tmp_path, body):
    wrapper, seen = build(tmp_path, count=4020)
    body.pop('tools'); body.pop('tool_choice')
    body['messages'] = [{'role': 'user', 'content': '[EVA_CODEX_TOOL_HISTORY_V1] actual retained fixture'},
                        {'role': 'assistant', 'content': 'actual prior result'},
                        {'role': 'user', 'content': 'Summarize this history.'}]
    wrapper(None, body)
    assert 'tools' not in seen['tokenize'][0] and seen['generation'][0]['max_tokens'] == 12
    assert seen['generation'][0]['messages'] == body['messages']


def test_exhausted_input_never_generates_or_truncates(tmp_path, body):
    wrapper, seen = build(tmp_path, count=4032)
    with pytest.raises(ResponsesAdapterError):wrapper(None, body)
    assert len(seen['tokenize']) == 1 and not seen['generation']
    result = receipt(tmp_path)
    assert result['failure_category'] == 'text_prompt_exhausts_context'
    assert result['generation_attempts'] == 0 and not result['input_truncated']


def test_multimodal_passthrough_is_explicitly_unverified_and_unchanged(tmp_path, body):
    wrapper, seen = build(tmp_path)
    body['messages'][0]['content'] = [{'type': 'text', 'text': 'public image'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,PRIVATEIMAGE'}}]
    original = deepcopy(body)
    wrapper(None, body)
    assert not seen['tokenize'] and seen['generation'] == [original] and body == original
    result = receipt(tmp_path)
    assert result['count_source'] == 'multimodal_count_unverified'
    assert result['prompt_tokens'] is None and not result['token_count_exact']
    assert result['multimodal_forwarded_unchanged'] and not result['output_budget_clamped']


def test_tokenize_http400_category_retained_without_raw_message(tmp_path, body):
    message = "Requested token count exceeds the model's maximum context length. PRIVATE SENTINEL"
    wrapper, seen = build(tmp_path, tokenize_status=400, tokenize_error=message)
    with pytest.raises(ResponsesAdapterError) as error:wrapper(None, body)
    assert 'PRIVATE' not in str(error.value) and not seen['generation']
    assert 'context_length_exceeded' in str(error.value)
    result = receipt(tmp_path)
    assert result['http_error_category'] == 'context_length_exceeded'
    assert result['failure_category'] == 'context_length_exceeded'
    assert result['tokenization_attempts'] == 1 and 'SENTINEL' not in json.dumps(result)


def test_actual_generation400_is_not_retried_and_has_fixed_category(tmp_path, body):
    message = "System message must be at the beginning. PRIVATE SENTINEL"
    wrapper, seen = build(tmp_path, upstream_status=400, upstream_error=message)
    outcome = wrapper(None, body)
    assert outcome.status == 400 and len(seen['generation']) == 1
    result = receipt(tmp_path)
    assert result['http_error_category'] == 'chat_template_invalid' and result['status'] == 'http_rejected'
    assert not result['automatic_retry'] and 'SENTINEL' not in json.dumps(result)


@pytest.mark.parametrize('malformed', [{'count': True}, {'count': 2}, {'tokens': [[1]]}, {'tokens': [True]}])
def test_malformed_tokenizer_evidence_never_generates(tmp_path, body, malformed):
    wrapper, seen = build(tmp_path, malformed=malformed)
    with pytest.raises(ResponsesAdapterError):wrapper(None, body)
    assert not seen['generation']


@pytest.mark.parametrize('message,category', [
    ('Tools cannot be empty if tool choice is set to required.', 'missing_tools_for_choice'),
    ("Tool 1 function has invalid 'parameters' schema: PRIVATE", 'invalid_tool_schema'),
    ('Assistant tool call function.arguments must be valid JSON.', 'invalid_tool_history_arguments'),
    ('max_completion_tokens is too large: 99999.', 'completion_budget_exceeds_context'),
    ('PRIVATE unrecognized reason', 'unclassified_http_400'),
])
def test_error_classifier_emits_only_allowlisted_categories(message, category):
    assert http_error_category(400, json.dumps({'message': message}).encode()) == category
    assert http_error_category(500, b'PRIVATE') == 'http_rejection_other'


def test_both_output_aliases_are_bounded_without_changing_inputs(tmp_path, body):
    wrapper, seen = build(tmp_path)
    body['max_completion_tokens'] = 100
    wrapper(None, body)
    assert seen['generation'][0]['max_tokens'] == seen['generation'][0]['max_completion_tokens'] == 42


def test_loopback_only_and_no_credentials_in_tokenize_route(tmp_path):
    with pytest.raises(ResponsesAdapterError, match='loopback'):
        TokenBudgetChatTransport(None, endpoint='https://service.example.invalid/redacted',
                                 context_length=32768, audit_root=tmp_path/'audit')
