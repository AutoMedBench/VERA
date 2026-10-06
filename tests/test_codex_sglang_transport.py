"""CPU-only sampled-token fixtures: no serving process, model load, or provider."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import stat

import httpx
import pytest

from eva_agent.codex_providers.adapter import ResponsesAdapterError
from eva_agent.training import codex_sglang_transport as module


class Tokenizer:
    def __init__(self):
        self.rendered = []
        self.encoded = []

    def apply_chat_template(self, messages, **kwargs):
        self.rendered.append((deepcopy(messages), deepcopy(kwargs)))
        return json.dumps({"messages": messages, **kwargs})

    def encode(self, prompt, *, add_special_tokens):
        assert add_special_tokens is False
        self.encoded.append(prompt)
        # Deliberately cannot re-encode output into sampled IDs [501, 502].
        return [11, 12, 13, 20 + len(self.encoded)]


@pytest.fixture
def body():
    return {"model": "Qwen/Qwen3.5-9B", "messages": [{"role": "user", "content": "public input"}],
            "tools": [{"type": "function", "function": {"name": "functions__read_file",
                       "description": "exact original", "parameters": {"type": "object",
                       "properties": {"paths": {"type": "array", "items": {"type": "string"},
                       "uniqueItems": True}}, "required": ["paths"], "additionalProperties": False}}}],
            "max_tokens": 12, "chat_template_kwargs": {"enable_thinking": True}}


def setup_transport(tmp_path, monkeypatch, *, output_change=None, status=200, **kwargs):
    seen = []
    tokenizer = Tokenizer()

    def endpoint(request):
        assert request.url.path == "/generate"
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        seen.append(payload)
        output = {"text": "PRIVATE REASONING</think>visible answer", "output_ids": [501, 502],
                  "meta_info": {"prompt_tokens": len(payload["input_ids"]), "completion_tokens": 2,
                  "output_token_logprobs": [[-0.25, 501, None], [-1.5, 502, None]],
                  "finish_reason": {"type": "stop", "matched": 502}, "weight_version": "observed-v1"}}
        if output_change:
            output_change(output)
        return httpx.Response(status, json=output)

    monkeypatch.setattr(module, "_project_completion", lambda text, tools, thinking, tokenizer:
                        {"role": "assistant", "content": "visible answer", "reasoning_content": "PRIVATE REASONING"})
    options = dict(tokenizer=tokenizer, generate_url="http://127.0.0.1:9999/generate",
                   sampling_params={"temperature": 0.8, "top_p": 1.0}, max_context_tokens=100,
                   max_output_tokens=12, total_output_budget=24, max_requests=4,
                   evidence_root=tmp_path / "private", http_transport=httpx.MockTransport(endpoint))
    options.update(kwargs)
    return module.CodexSGLangTransport(**options), tokenizer, seen


def test_exact_tokens_private_evidence_and_no_binding_credentials(tmp_path, monkeypatch, body):
    transport, tokenizer, seen = setup_transport(tmp_path, monkeypatch)
    class Binding:
        def reveal_upstream_boundary(self):
            pytest.fail("credentials must never be read")
    original = deepcopy(body)
    outcome = transport(Binding(), body)
    assert outcome.status == 200 and json.loads(outcome.body)["model"] == body["model"]
    segment = transport.segments[0]
    assert segment["prompt_token_ids"] == seen[0]["input_ids"] == [11, 12, 13, 21]
    assert segment["output_token_ids"] == [501, 502]
    assert segment["output_token_logprobs"] == [-0.25, -1.5]
    assert segment["loss_mask"] == [1, 1] and segment["prompt_is_conditioning_only"]
    assert segment["returned_model"] is None
    assert segment["meta_info"]["weight_version"] == "observed-v1"
    assert len(tokenizer.encoded) == 1 and body == original
    assert tokenizer.rendered[0][1]["tools"] == body["tools"]  # uniqueItems retained
    assert seen[0]["return_logprob"] and seen[0]["logprob_start_len"] == -1
    assert seen[0]["sampling_params"]["skip_special_tokens"] is False
    assert stat.S_IMODE((tmp_path / "private").stat().st_mode) == 0o700
    for path in (tmp_path / "private").iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        json.loads(path.read_text())
    assert "PRIVATE" not in json.dumps(transport.safe_metadata)
    segment["output_token_ids"].append(999)
    assert transport.segments[0]["output_token_ids"] == [501, 502]


def test_owned_budget_termination_excludes_context_and_http_failures(tmp_path, monkeypatch, body):
    transport, _, _ = setup_transport(tmp_path, monkeypatch, max_requests=1)
    transport(None, body)
    with pytest.raises(ResponsesAdapterError):
        transport(None, body)
    assert transport.budget_termination == {"kind": "max_requests", "count": 1, "limit": 1, "generated_tokens": 2}
    for name, kwargs in [("total", {"total_output_budget": 2}), ("context", {"max_context_tokens": 4}),
                         ("http", {"status": 500})]:
        folder = tmp_path / name
        folder.mkdir()
        other, _, _ = setup_transport(folder, monkeypatch, **kwargs)
        if name == "total":
            other(None, body)
        with pytest.raises(ResponsesAdapterError):
            other(None, body)
        assert (other.budget_termination is not None) == (name == "total")


def test_history_and_compaction_each_get_fresh_conditioning_segment(tmp_path, monkeypatch, body):
    transport, tokenizer, seen = setup_transport(tmp_path, monkeypatch)
    transport(None, body)
    second = deepcopy(body)
    second["messages"] += [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1",
        "type": "function", "function": {"name": "functions__read_file", "arguments": '{"paths":["a"]}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "host observation"}]}]
    transport(None, second)
    compacted = deepcopy(body)
    compacted["messages"] = [{"role": "user", "content": "actual compacted summary"}]
    transport(None, compacted)
    assert len(tokenizer.encoded) == 3
    assert [x["input_ids"] for x in seen] == [[11, 12, 13, 21], [11, 12, 13, 22], [11, 12, 13, 23]]
    assert all(len(s["prompt_token_ids"]) == 4 for s in transport.segments)
    history = tokenizer.rendered[1][0]
    assert history[-2]["tool_calls"][0]["function"]["arguments"] == {"paths": ["a"]}
    assert history[-1]["content"] == "host observation"
    assert isinstance(second["messages"][-2]["tool_calls"][0]["function"]["arguments"], str)


@pytest.mark.parametrize("change", [
    lambda o: o["meta_info"].pop("output_token_logprobs"),
    lambda o: o["meta_info"]["output_token_logprobs"][0].__setitem__(0, None),
    lambda o: o["meta_info"]["output_token_logprobs"][0].__setitem__(0, float("inf")),
    lambda o: o["meta_info"]["output_token_logprobs"][0].__setitem__(1, True),
    lambda o: o["meta_info"].__setitem__("prompt_tokens", 90),
    lambda o: o["meta_info"].__setitem__("completion_tokens", 3),
    lambda o: o.__setitem__("output_ids", [501, 503]),
    lambda o: o["meta_info"].__setitem__("finish_reason", {"type": "abort"}),
])
def test_invalid_sampling_evidence_fails_closed_without_retry(tmp_path, monkeypatch, body, change):
    transport, _, seen = setup_transport(tmp_path, monkeypatch, output_change=change)
    with pytest.raises(ResponsesAdapterError):
        transport(None, body)
    with pytest.raises(ResponsesAdapterError, match="already failed"):
        transport(None, body)
    assert len(seen) == 1 and transport.segments == []
    assert transport.safe_metadata["failed"]


def test_http_error_no_redirect_or_retry_and_no_error_body_leak(tmp_path, monkeypatch, body):
    transport, _, seen = setup_transport(tmp_path, monkeypatch, status=302)
    with pytest.raises(ResponsesAdapterError) as error:
        transport(None, body)
    assert "PRIVATE" not in str(error.value) and len(seen) == 1
    assert not any("PRIVATE" in p.read_text() for p in (tmp_path / "private").iterdir())


def test_total_budget_and_no_context_truncation(tmp_path, monkeypatch, body):
    transport, _, seen = setup_transport(tmp_path, monkeypatch, total_output_budget=4)
    transport(None, body)
    transport(None, body)
    with pytest.raises(ResponsesAdapterError):
        transport(None, body)
    assert [x["sampling_params"]["max_new_tokens"] for x in seen] == [4, 2]
    other, _, calls = setup_transport(tmp_path, monkeypatch, evidence_root=None, max_context_tokens=4)
    with pytest.raises(ResponsesAdapterError):
        other(None, body)
    assert not calls


def test_tool_free_compaction_counts_exact_prompt_ids_and_clamps_output(tmp_path, monkeypatch):
    from eva_agent.codex_providers import adapter

    # Exercise the actual tool-free history projection. The deterministic byte
    # tokenizer is a CPU fixture, not an estimate of a real Qwen token count.
    canonical_output = {"type": "function_call_output", "call_id": "earlier",
                        "output": '{"observed":"public fixture result"}'}
    inputs = [{"type": "message", "role": "user", "content": "Original task"},
              {"type": "function_call", "call_id": "earlier", "name": "read_file",
               "namespace": "mcp__fixture", "arguments": '{"path":"public.txt"}'},
              canonical_output,
              {"type": "message", "role": "user", "content": "Compact the actual preceding history."}]
    original = deepcopy(inputs)
    projection = adapter._tool_projection([], provider_family="qwen", preserve_qwen_tool_schemas=True)
    messages = adapter._messages(inputs, "Fixture instructions", projection, provider_family="qwen")
    assert all(message["role"] != "tool" and "tool_calls" not in message for message in messages)
    assert adapter._tool_history_transcript(canonical_output) in messages[-1]["content"]
    kwargs = {"tools": None, "tokenize": False, "add_generation_prompt": True, "enable_thinking": True}
    expected_prompt = json.dumps({"messages": messages, **kwargs})
    expected_ids = list(expected_prompt.encode("utf-8"))

    class ByteTokenizer(Tokenizer):
        def encode(self, prompt, *, add_special_tokens):
            assert add_special_tokens is False
            self.encoded.append(prompt)
            return list(prompt.encode("utf-8"))

    tokenizer = ByteTokenizer()
    transport, _, calls = setup_transport(tmp_path, monkeypatch, tokenizer=tokenizer,
        max_context_tokens=len(expected_ids) + 3, max_output_tokens=8192, total_output_budget=8192)
    outcome = transport(None, {"model": "Qwen/Qwen3.5-9B", "messages": messages, "max_tokens": 8192,
                              "chat_template_kwargs": {"enable_thinking": True}})
    assert outcome.status == 200 and len(calls) == 1
    assert calls[0]["input_ids"] == expected_ids
    assert calls[0]["sampling_params"]["max_new_tokens"] == 3  # not the requested 8192
    assert len(tokenizer.encoded) == 1 and inputs == original
    segment = transport.segments[0]
    assert segment["prompt_token_ids"] == expected_ids
    assert segment["output_token_ids"] == [501, 502]  # never re-encoded from sampled text
    assert segment["loss_mask"] == [1, 1] and segment["prompt_is_conditioning_only"]
    assert tokenizer.rendered[0][1]["tools"] is None


@pytest.mark.parametrize("control", [
    {"tool_choice": "required"}, {"tool_choice": {"type": "function", "function": {"name": "x"}}},
    {"temperature": 0.1}, {"stream": True}, {"n": 2},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "private"}]}]},
])
def test_unsupported_or_conflicting_controls_fail_before_inference(tmp_path, monkeypatch, body, control):
    transport, _, seen = setup_transport(tmp_path, monkeypatch)
    with pytest.raises(ResponsesAdapterError):
        transport(None, {**body, **control})
    assert seen == []


def test_concurrent_calls_are_serialized_and_request_budget_is_exact(tmp_path, monkeypatch, body):
    transport, _, seen = setup_transport(tmp_path, monkeypatch, max_requests=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(x.status == 200 for x in pool.map(lambda _: transport(None, body), range(2)))
    with pytest.raises(ResponsesAdapterError):
        transport(None, body)
    assert len(seen) == 2 and [s["segment_index"] for s in transport.segments] == [0, 1]


def test_installed_qwen_parser_keeps_flat_tool_names_and_hides_reasoning(body):
    pytest.importorskip("sglang.srt.function_call.function_call_parser")
    text = ('PRIVATE</think><tool_call>\n<function=functions__read_file>\n'
            '<parameter=paths>\n["public.txt"]\n</parameter>\n</function>\n</tool_call><|im_end|>')
    message = module._project_completion(text, body["tools"], True, None)
    assert message["reasoning_content"] == "PRIVATE"
    assert "PRIVATE" not in message["content"]
    call = message["tool_calls"][0]["function"]
    assert call["name"] == "functions__read_file"
    assert json.loads(call["arguments"]) == {"paths": ["public.txt"]}


def test_installed_qwen_parser_truncated_reasoning_is_not_visible():
    pytest.importorskip("sglang.srt.function_call.function_call_parser")
    message = module._project_completion("PRIVATE UNFINISHED THINKING", [], True, None)
    assert message["content"] == "" and message["reasoning_content"] == "PRIVATE UNFINISHED THINKING"
