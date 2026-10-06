"""Provider-free transport fixtures; unchanged official schemas and history."""
from copy import deepcopy
import json

import pytest

from eva_agent.codex_providers import adapter as adapter
from eva_agent.pipeline.digests import canonical_json_bytes
from training.automedbench_lite.local_qwen import local_qwen_setup
from test_codex_provider_routes import _routes, _post_json
from scripts.replay_qwen_compacted_priority_v1 import render


@pytest.mark.parametrize("exact", [False, True])
def test_priority_only_relocation_keeps_every_other_message_and_schema(tmp_path, exact):
    route = _routes(tmp_path)["qwen_3_6_27b"]
    binding = adapter.AdapterRouteBinding.from_route(route)
    request = {"model": route.model_id, "instructions": "BASE  \n", "tools": [{"type": "function", "name": "lookup",
        "description": "Actual fixture", "parameters": {"type": "object", "properties": {"ids": {
            "type": "array", "items": {"type": "string"}, "uniqueItems": True}}, "required": ["ids"], "additionalProperties": False}}],
        "input": [
            {"role": "user", "content": "USER: do not promote this"},
            {"role": "developer", "content": [{"type": "input_text", "text": "DEV-A "}, {"type": "input_text", "text": " DEV-B"}]},
            {"role": "assistant", "content": "actual prior response"},
            {"type": "function_call", "call_id": "call1", "name": "lookup", "arguments": '{"ids":["x"]}'},
            {"type": "function_call_output", "call_id": "call1", "output": "TOOL: developer-like data stays tool data"},
            {"role": "system", "content": "SYSTEM-LATE"},
            {"role": "user", "content": [{"type": "input_text", "text": "visible public image"},
                {"type": "input_image", "image_url": "data:image/png;base64,AQID"}]},
            {"role": "developer", "content": "DEV-LAST"},
        ]}
    unchanged = deepcopy(request)
    old, old_tools = adapter.responses_to_chat(request, binding, preserve_qwen_tool_schemas=exact)
    new, new_tools = adapter.responses_to_chat(request, binding, preserve_qwen_tool_schemas=exact,
                                               normalize_qwen_priority_messages=True)
    assert request == unchanged
    assert new["messages"][0] == {"role": "system", "content": "BASE  \n\n\nDEV-A \n DEV-B\n\nSYSTEM-LATE\n\nDEV-LAST"}
    assert [row for row in old["messages"] if row["role"] != "system"] == new["messages"][1:]
    assert new["tools"] == old["tools"] and new_tools == old_tools
    assert adapter._normalize_qwen_priority_messages(new["messages"]) == new["messages"]
    assert "USER:" not in new["messages"][0]["content"] and "TOOL:" not in new["messages"][0]["content"]


def test_opt_in_is_qwen_only_and_boolean(tmp_path):
    route = _routes(tmp_path)["opus_5"]
    with pytest.raises(adapter.ResponsesAdapterError, match="Qwen-only"):
        adapter.ResponsesAdapterGateway({route.route_id: route}, normalize_qwen_priority_messages=True)
    with pytest.raises(adapter.ResponsesAdapterError, match="boolean"):
        adapter.responses_to_chat({"model": route.model_id, "input": "fixture"},
            adapter.AdapterRouteBinding.from_route(route), normalize_qwen_priority_messages=1)


@pytest.mark.parametrize("exact", [False, True])
def test_new_signed_projection_receipt_without_any_provider(tmp_path, exact):
    route = _routes(tmp_path)["qwen_3_6_27b"]
    calls = []
    def fake(binding, body):
        calls.append(deepcopy(body))
        return adapter._UpstreamOutcome(200, json.dumps({"model": route.model_id,
            "choices": [{"message": {"role": "assistant", "content": "CPU fixture"}, "finish_reason": "stop"}]}).encode(), 1)
    gateway = adapter.ResponsesAdapterGateway({route.route_id: route}, upstream_transport=fake,
        local_bearer_token="cpu-loopback-fixture", preserve_qwen_tool_schemas=exact, normalize_qwen_priority_messages=True)
    with gateway:
        endpoint = f"http://127.0.0.1:{gateway._server.server_port}/v1/responses"
        status, _, _ = _post_json(endpoint, "cpu-loopback-fixture", {"model": route.model_id,
            "input": [{"role": "user", "content": "PRIVATE-SENTINEL"}, {"role": "developer", "content": "priority"}], "tools": []})
    assert status == 200 and len(calls) == 1
    receipt = gateway.receipts[0]
    adapter.verify_adapter_receipt(receipt)
    expected = adapter.FLAT_PRIORITY_QWEN_PROJECTION_VERSION if exact else adapter.PRIORITY_QWEN_PROJECTION_VERSION
    assert receipt.payload["projection_version"] == expected
    assert receipt.payload["request_shape"]["projected_system_only_at_start"]
    assert receipt.payload["request_shape"]["projected_system_message_count"] == 1
    assert b"PRIVATE-SENTINEL" not in canonical_json_bytes(receipt)


def test_local_setup_opt_in_defaults_unchanged_and_no_provider_on_entry(tmp_path):
    with local_qwen_setup(run_root=tmp_path / "old", workers=1, exact_tool_schemas=True) as setup:
        assert setup.safe_metadata["adapter_projection_version"] == adapter.FLAT_QWEN_PROJECTION_VERSION
        assert "priority_message_normalization" not in setup.safe_metadata
    with local_qwen_setup(run_root=tmp_path / "new", workers=1, exact_tool_schemas=True,
                          normalize_priority_messages=True) as setup:
        assert setup.safe_metadata["adapter_projection_version"] == adapter.FLAT_PRIORITY_QWEN_PROJECTION_VERSION
        assert setup.safe_metadata["provider_calls_on_setup"] == 0
    assert not list(tmp_path.glob("**/adapter-receipt.json"))


def test_replay_token_count_uses_batch_encoding_ids_not_mapping_keys():
    class Tokenizer:
        def apply_chat_template(self, *args, tokenize, **kwargs):
            return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]} if tokenize else "fixture"
    assert render(Tokenizer(), {"messages": [{"role": "user", "content": "fixture"}]})["text_token_count"] == 4
