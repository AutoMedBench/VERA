"""CPU-only fixtures; no native Codex process, model service or tool job.

The integration uses the real HTTP adapter and a mocked model, plus a strict
JSON parser producing the captured Codex parser-error *shape*. It proves
projection/continuation, not that a real model will choose to self-correct.
"""
from copy import deepcopy
import json
from pathlib import Path
import urllib.error

import pytest

from eva_agent.codex_providers import adapter as module
from eva_agent.codex_runtime import CodexToolOffer
from eva_agent.pipeline.digests import canonical_json_bytes
from eva_agent.training.codex_sglang_transport import _messages as training_messages
from test_codex_provider_routes import _routes, _post_json, _chat_response


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sigma"],
          "properties": {"sigma": {"type": "number", "minimum": 0}}}
TOOLS = [{"type": "namespace", "name": "mcp__evamed", "tools": [
    {"type": "function", "name": "fixture_tool", "description": "Exact test tool.", "parameters": SCHEMA}]}]
BAD = {"type": "function_call", "call_id": "call-invalid", "namespace": "mcp__evamed",
       "name": "fixture_tool", "arguments": '{"sigma":NaN}'}
ERROR = {"type": "function_call_output", "call_id": "call-invalid", "output": [
    {"type": "input_text", "text": "Wall time: 0.001 seconds\nOutput:"},
    {"type": "input_text", "text": "err: expected value at line 1 column 10"}]}
USER = {"type": "message", "role": "user", "content": "Do the fixture task."}


@pytest.fixture
def route(tmp_path):
    return _routes(tmp_path)["qwen_3_6_27b"]


def request(route, items=None):
    return {"model": route.model_id, "input": deepcopy(items if items is not None else [USER, BAD, ERROR]),
            "tools": deepcopy(TOOLS), "stream": False, "store": False, "parallel_tool_calls": True}


def project(route, value, enabled=True):
    binding = module.AdapterRouteBinding.from_route(route)
    return module.responses_to_chat(value, binding, preserve_qwen_tool_schemas=True,
        recover_completed_invalid_tool_history=enabled)


def transcripts(chat):
    prefix = module.TOOL_HISTORY_TRANSCRIPT_PREFIX
    result = []
    for message in chat["messages"]:
        content = message.get("content")
        if isinstance(content, str) and content.startswith(prefix):
            for part in content.split(prefix)[1:]:
                # Qwen coalesces a following user stage prompt with the exact
                # historical observation; decode its leading JSON, not prose.
                result.append(json.JSONDecoder().raw_decode(part.lstrip())[0])
    return result


def test_default_and_valid_histories_are_unchanged(route):
    source = request(route)
    before = canonical_json_bytes(source)
    old, _ = project(route, source, False)
    assert old["messages"][1]["tool_calls"][0]["function"]["arguments"] == BAD["arguments"]
    assert old["messages"][2]["role"] == "tool"
    good = {**BAD, "arguments": '{"sigma":0.1}'}
    valid = request(route, [USER, good, {**ERROR, "output": '{"ok":true}'}])
    assert project(route, valid, True) == project(route, valid, False)
    assert canonical_json_bytes(source) == before


def test_completed_invalid_pair_is_lossless_with_unchanged_active_tools(route):
    source = request(route)
    before = canonical_json_bytes(source)
    chat, projection = project(route, source)
    assert transcripts(chat) == [BAD, ERROR]
    assert projection.packed_namespaces == {}
    assert chat["tools"][0]["function"]["parameters"] == SCHEMA
    assert chat["tools"][0]["function"]["description"] == TOOLS[0]["tools"][0]["description"]
    assert all("tool_calls" not in row and row["role"] != "tool" for row in chat["messages"])
    assert training_messages(chat) == chat["messages"]
    assert canonical_json_bytes(source) == before


def test_mixed_parallel_group_preserves_causal_item_order_and_no_orphans(route):
    good = {**BAD, "call_id": "call-good", "arguments": '{"sigma":0.1}'}
    good_result = {"type": "function_call_output", "call_id": "call-good", "output": "actual valid result"}
    # Results can arrive in the opposite order; never reorder them to match calls.
    items = [USER, good, BAD, ERROR, good_result]
    chat, _ = project(route, request(route, items))
    assert transcripts(chat) == items[1:]
    assert all("tool_calls" not in row and row["role"] != "tool" for row in chat["messages"])


@pytest.mark.parametrize("change", ["pending", "duplicate_call", "duplicate_result", "orphan", "message",
                                    "success", "ambiguous_error", "pending_sibling"])
def test_invalid_history_without_closed_unambiguous_parser_error_is_rejected(route, change):
    items = deepcopy([USER, BAD, ERROR])
    if change == "pending": items.pop()
    if change == "duplicate_call": items.insert(2, deepcopy(BAD))
    if change == "duplicate_result": items.append(deepcopy(ERROR))
    if change == "orphan": items[-1]["call_id"] = "different"
    if change == "message": items.insert(2, deepcopy(USER))
    if change == "success": items[-1]["output"] = '{"success":true}'
    if change == "ambiguous_error": items[-1]["output"] = "tool failed, retry this"
    if change == "pending_sibling": items.insert(2, {**BAD, "call_id": "sibling", "arguments": '{"sigma":1}'})
    with pytest.raises(module.ResponsesAdapterError):
        project(route, request(route, items))


@pytest.mark.parametrize("stage", ["S1", "S2", "S3", "S4", "S5"])
def test_completed_old_tool_error_survives_stage_catalog_change(route, stage):
    source = request(route)
    current_tool = {"type": "function", "name": "current_" + stage,
        "description": "Exact current-stage tool.", "parameters": deepcopy(SCHEMA)}
    source["tools"] = [{"type": "namespace", "name": "mcp__current_stage", "tools": [current_tool]}]
    source["input"].append({"type": "message", "role": "user", "content": "Begin " + stage})
    original = canonical_json_bytes(source)
    chat, projection = project(route, source)
    assert transcripts(chat) == [BAD, ERROR]
    assert chat["messages"][-1]["content"].endswith("Begin " + stage)
    assert len(chat["tools"]) == 1
    assert chat["tools"][0]["function"]["parameters"] == current_tool["parameters"]
    assert chat["tools"][0]["function"]["description"] == current_tool["description"]
    assert set(projection.responses_to_chat) == {("mcp__current_stage", current_tool["name"])}
    assert training_messages(chat) == chat["messages"]
    assert canonical_json_bytes(source) == original
    # A pending old call or invented success is not made recoverable by retirement.
    for invalid_history in ([USER, BAD], [USER, BAD, {**ERROR, "output": "success"}]):
        with pytest.raises(module.ResponsesAdapterError):
            project(route, {**source, "input": invalid_history})


def test_retired_invalid_parallel_group_preserves_all_observations(route):
    good = {**BAD, "call_id": "old-good", "arguments": '{"sigma":0.1}'}
    result = {"type": "function_call_output", "call_id": "old-good", "output": "actual old result"}
    items = [USER, good, BAD, ERROR, result]
    source = request(route, items)
    source["tools"][0]["tools"][0]["name"] = "current_stage_tool"
    chat, projection = project(route, source)
    assert transcripts(chat) == items[1:]
    assert all("tool_calls" not in row and row["role"] != "tool" for row in chat["messages"])
    assert set(projection.responses_to_chat) == {("mcp__evamed", "current_stage_tool")}


def test_tool_free_compaction_keeps_existing_lossless_history_semantics(route):
    value = request(route)
    value["tools"] = []
    assert project(route, value, True) == project(route, value, False)


@pytest.mark.parametrize("arguments", ['{"sigma":Infinity}', '{"sigma":-Infinity}', '{"sigma":}'])
def test_nonfinite_or_malformed_completed_argument_bytes_are_not_repaired(route, arguments):
    bad = {**BAD, "arguments": arguments}
    chat, _ = project(route, request(route, [USER, bad, ERROR]))
    assert transcripts(chat)[0]["arguments"] == arguments


@pytest.mark.parametrize("priority", [False, True])
def test_gateway_version_is_explicit_and_signed_with_canonical_schemas(route, priority):
    gateway = module.ResponsesAdapterGateway({route.route_id: route}, preserve_qwen_tool_schemas=True,
        normalize_qwen_priority_messages=priority, recover_completed_invalid_tool_history=True)
    expected = (module.COMPLETED_INVALID_PRIORITY_QWEN_PROJECTION_VERSION if priority
        else module.COMPLETED_INVALID_QWEN_PROJECTION_VERSION)
    assert gateway.safe_metadata["projection_version"] == expected
    binding = gateway.bind_canonical_mcp_tools((CodexToolOffer("evamed/fixture_tool", "Exact test tool.", SCHEMA),))
    expected = (module.CANONICAL_COMPLETED_INVALID_PRIORITY_QWEN_PROJECTION_VERSION if priority
        else module.CANONICAL_COMPLETED_INVALID_QWEN_PROJECTION_VERSION)
    assert binding["projection_version"] == expected
    assert gateway.safe_metadata["completed_invalid_tool_history_projection"] == "completed-codex-parser-errors-lossless-v1"


def test_same_conversation_mock_invalid_error_then_valid_executes_only_valid_call(route):
    calls, executed = [], []
    def upstream(binding, body):
        calls.append(deepcopy(body))
        ordinal = len(calls)
        if ordinal == 2:
            assert transcripts(body)[-1]["output"] == "err: expected value at line 1 column 10"
            assert transcripts(body)[0]["arguments"] == BAD["arguments"]
        if ordinal < 3:
            arguments = BAD["arguments"] if ordinal == 1 else '{"sigma":0.1}'
            wire = {"id": "call-invalid" if ordinal == 1 else "call-corrected", "type": "function",
                "function": {"name": body["tools"][0]["function"]["name"], "arguments": arguments}}
            result = _chat_response(binding.model_id, content=None, tool_calls=[wire])
        else:
            result = _chat_response(binding.model_id, content="Fixture complete.")
        return module._UpstreamOutcome(200, json.dumps(result).encode(), 1)
    token = "cpu-fixture-local-token"
    with module.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=upstream,
        local_bearer_token=token, preserve_qwen_tool_schemas=True,
        recover_completed_invalid_tool_history=True) as gateway:
        endpoint = gateway.adapted_routes()[route.route_id].config.for_subprocess()[2][0]
        history = [deepcopy(USER)]
        for _ in range(3):
            status, payload, _ = _post_json(endpoint + "/responses", token, request(route, history))
            assert status == 200
            response = json.loads(payload)
            generated = [item for item in response["output"] if item["type"] == "function_call"]
            if not generated:
                break
            item = generated[0]
            history.append(item)
            def invalid_constant(value):
                # Captured native Codex shape; no actual tool execution occurs.
                raise ValueError("invalid JSON constant")
            try:
                arguments = json.loads(item["arguments"], parse_constant=invalid_constant)
            except ValueError:
                output = "err: expected value at line 1 column 10"
            else:
                executed.append((item["call_id"], arguments))
                output = '{"ok":true}'
            history.append({"type": "function_call_output", "call_id": item["call_id"], "output": output})
        assert executed == [("call-corrected", {"sigma": 0.1})] and len(calls) == 3
        assert history[1]["arguments"] == BAD["arguments"]
        for receipt in gateway.receipts:
            module.verify_adapter_receipt(receipt)
            assert receipt.payload["projection_version"] == module.COMPLETED_INVALID_QWEN_PROJECTION_VERSION
            assert receipt.payload["upstream_request_max_retries"] == 0
        with pytest.raises(urllib.error.HTTPError) as error:
            _post_json(endpoint + "/responses", token, request(route, [USER, BAD]))
        assert error.value.code == 400 and len(calls) == 3


def test_selected_local_qwen_template_renders_same_training_text_history(route):
    path = Path("/localhome/local-operator/operator_GB300-2/Qwen3.5-9B/chat_template.jinja")
    if not path.is_file():
        pytest.skip("actual local Qwen template is not installed on this test host")
    from jinja2 import Environment
    env = Environment()
    def reject(message):
        raise ValueError(message)
    env.globals["raise_exception"] = reject
    template = env.from_string(path.read_text())
    chat, _ = project(route, request(route))
    rendered = template.render(messages=chat["messages"], tools=chat["tools"],
        add_generation_prompt=True, enable_thinking=True)
    trained = template.render(messages=training_messages(chat), tools=chat["tools"],
        add_generation_prompt=True, enable_thinking=True)
    assert rendered == trained and module.TOOL_HISTORY_TRANSCRIPT_PREFIX in rendered
