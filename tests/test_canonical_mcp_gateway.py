"""Local HTTP-to-mock transport tests; no model endpoint or GPU is used."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from eva_agent.codex_runtime import CodexToolOffer
from eva_agent.codex_providers.adapter import (
    CANONICAL_QWEN_PROJECTION_VERSION, CANONICAL_PRIORITY_QWEN_PROJECTION_VERSION,
    ResponsesAdapterError, ResponsesAdapterGateway, verify_adapter_receipt,
)
from eva_agent.codex_providers.routes import CodexConfigOverrides, CodexProviderRoute, _PrivateText
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes


MODEL = "Qwen/Qwen3.5-9B-schema-fixture"
SCHEMA = {"type": "object", "properties": {
    "stage": {"enum": ["S3", "S4"]},
    "code": {"type": "string", "minLength": 1, "maxLength": 2000},
}, "required": ["stage", "code"], "additionalProperties": False}


def offers(server="medical"):
    return (CodexToolOffer(f"{server}/execute_code", "Execute the stage code.", deepcopy(SCHEMA)),)


def route():
    return CodexProviderRoute(route_id="qwen_schema", model_id=MODEL,
        model_env_name="MODEL_QWEN_3_5_9B", registry_role_id=None, provider_family="qwen",
        config=CodexConfigOverrides(provider_id="eva_schema", model_id=MODEL,
            credential_env_name="OPENAI_API_KEY", endpoint_env_name="OPENAI_BASE_URL",
            _credential=_PrivateText("synthetic-local"), _endpoint=_PrivateText("http://127.0.0.1:1/v1")))


def native_tools():
    schema = deepcopy(SCHEMA)
    schema["properties"]["stage"]["type"] = "string"
    del schema["properties"]["code"]["minLength"]
    del schema["properties"]["code"]["maxLength"]
    return [{"type": "namespace", "name": "mcp__medical",
        "description": "Tools in the mcp__medical namespace.", "tools": [
            {"type": "function", "name": "execute_code", "description": "Execute the stage code.",
             "parameters": schema, "strict": False}]}]


def make_gateway(captured, **kwargs):
    def transport(_binding, body):
        captured.append(deepcopy(body))
        return SimpleNamespace(status=200, latency_ms=0, body=json.dumps({
            "model": MODEL, "choices": [{"message": {"role": "assistant", "content": "OK"},
                                         "finish_reason": "stop"}]}).encode())
    return ResponsesAdapterGateway({"qwen_schema": route()}, upstream_transport=transport,
        preserve_qwen_tool_schemas=kwargs.pop("preserve_qwen_tool_schemas", True), **kwargs)


def post(gateway, tools="absent"):
    _, _, (endpoint, token) = gateway.adapted_routes()["qwen_schema"].config.for_subprocess()
    body = {"model": MODEL, "input": "Public schema fixture.", "stream": False}
    if tools != "absent":
        body["tools"] = tools
    req = urllib.request.Request(endpoint + "/responses", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.mark.parametrize("priority", [False, True])
def test_bound_gateway_restores_before_upstream_with_signed_audit(priority):
    captured = []
    original = native_tools()
    before = deepcopy(original)
    with make_gateway(captured, normalize_qwen_priority_messages=priority) as gateway:
        binding = gateway.bind_canonical_mcp_tools(offers())
        assert post(gateway, original)[0] == 200
        assert original == before
        tool = captured[0]["tools"][0]["function"]
        assert tool["name"] == "mcp__medical__execute_code"
        assert canonical_json_bytes(tool["parameters"]) == canonical_json_bytes(SCHEMA)
        assert tool["description"] == "Execute the stage code."
        signed = gateway.receipts[-1]
        verify_adapter_receipt(signed)
        expected = CANONICAL_PRIORITY_QWEN_PROJECTION_VERSION if priority else CANONICAL_QWEN_PROJECTION_VERSION
        assert signed.payload["projection_version"] == binding["projection_version"] == expected
        audit = signed.payload["request_shape"]["canonical_mcp_schema_projection"]
        assert audit["catalog_blake3"] == binding["canonical_catalog_blake3"]
        assert audit["native_tools_blake3"] == blake3_hex(original)
        assert audit["restored_count"] == 1
        assert signed.payload["raw_request_recorded"] is False
        assert signed.payload["upstream_request_max_retries"] == 0


@pytest.mark.parametrize("mutation", ["required", "description", "name"])
def test_mismatch_is_rejected_without_upstream_call(mutation):
    captured = []
    value = native_tools()
    tool = value[0]["tools"][0]
    if mutation == "required":
        tool["parameters"]["required"].remove("stage")
    else:
        tool[mutation] = "changed"
    with make_gateway(captured) as gateway:
        gateway.bind_canonical_mcp_tools(offers())
        status, body = post(gateway, value)
        assert status == 400 and body["error"]["type"] == "canonical_mcp_schema_mismatch"
        assert not captured
        verify_adapter_receipt(gateway.receipts[-1])
        assert gateway.receipts[-1].payload["upstream_http_status"] is None


@pytest.mark.parametrize("tools", ["absent", None, []])
def test_compaction_or_tool_free_request_never_injects_tools(tools):
    captured = []
    with make_gateway(captured) as gateway:
        gateway.bind_canonical_mcp_tools(offers())
        assert post(gateway, tools)[0] == 200
        assert not captured[0].get("tools")
        assert gateway.receipts[-1].payload["request_shape"]["canonical_mcp_schema_projection"]["restored_count"] == 0


def test_binding_is_immutable_idempotent_and_concurrent():
    captured = []
    with make_gateway(captured) as gateway:
        with ThreadPoolExecutor(max_workers=3) as pool:
            bindings = list(pool.map(lambda _: dict(gateway.bind_canonical_mcp_tools(offers())), range(3)))
        assert bindings[0] == bindings[1] == bindings[2]
        assert post(gateway, native_tools())[0] == 200
        assert dict(gateway.bind_canonical_mcp_tools(offers())) == bindings[0]
        with pytest.raises(ResponsesAdapterError, match="cannot change"):
            gateway.bind_canonical_mcp_tools(offers("another_server"))
        assert len(captured) == 1


def test_unbound_old_mode_unchanged_but_binding_after_request_fails():
    captured = []
    with make_gateway(captured) as gateway:
        assert post(gateway, native_tools())[0] == 200
        assert captured[0]["tools"][0]["function"]["parameters"] == native_tools()[0]["tools"][0]["parameters"]
        assert "canonical_mcp_schema_projection" not in gateway.receipts[-1].payload["request_shape"]
        with pytest.raises(ResponsesAdapterError, match="before the first request"):
            gateway.bind_canonical_mcp_tools(offers())


def test_binding_cannot_silently_enable_exact_projection():
    with make_gateway([], preserve_qwen_tool_schemas=False) as gateway:
        with pytest.raises(ResponsesAdapterError, match="requires exact"):
            gateway.bind_canonical_mcp_tools(offers())
