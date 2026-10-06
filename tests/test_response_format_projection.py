"""Provider-free structured-output projection and loopback gateway regressions."""
from copy import deepcopy
from http import HTTPStatus
import json
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from eva_agent.codex_pipeline.adapter import _judge_output_schema
from eva_agent.codex_providers.adapter import (
    AdapterRouteBinding, ResponsesAdapterError, ResponsesAdapterGateway,
    responses_to_chat, verify_adapter_receipt,
)
from eva_agent.codex_providers.routes import CodexConfigOverrides, CodexProviderRoute, _PrivateText
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes


MODEL = "aws/anthropic/bedrock-claude-opus-5"


def route():
    return CodexProviderRoute(
        route_id="opus_5", model_id=MODEL, model_env_name="MODEL_OPUS_5",
        registry_role_id="architect_opus_5", provider_family="anthropic",
        config=CodexConfigOverrides(provider_id="eva_opus_5", model_id=MODEL,
            credential_env_name="OPENAI_API_KEY", endpoint_env_name="OPENAI_BASE_URL",
            _credential=_PrivateText("synthetic-not-a-credential"),
            _endpoint=_PrivateText("https://provider.invalid/v1")),
    )


def request():
    return {"model": MODEL, "input": "Public transport fixture.", "tools": [], "stream": False}


def project(value):
    return responses_to_chat(value, AdapterRouteBinding.from_route(route()))[0]


def judge_schema():
    # Use the actual canonical Judge schema; no hand-written substitute.
    return json.loads(canonical_json_bytes(_judge_output_schema([{"item_id": "public-fixture"}])))


@pytest.mark.parametrize("strict", ["absent", None, False, True])
@pytest.mark.parametrize("description", [None, "Public structured verdict."])
def test_canonical_judge_schema_envelope_roundtrip(strict, description):
    value = {"type": "json_schema", "name": "codex_output_schema",
             "schema": judge_schema(), "description": description}
    if strict != "absent":
        value["strict"] = strict
    req = {**request(), "text": {"format": value, "verbosity": "low"}}
    before = deepcopy(req)
    result = project(req)["response_format"]
    restored = {"type": result["type"], **result["json_schema"]}
    assert restored == value
    assert canonical_json_bytes(restored["schema"]) == canonical_json_bytes(value["schema"])
    assert req == before
    # Upstream mutation cannot mutate the input's canonical schema object.
    result["json_schema"]["schema"]["required"].append("fixture-mutated-only")
    assert req == before


@pytest.mark.parametrize("text", [None, {}, {"format": None}, {"format": {"type": "text"}},
                                   {"verbosity": "low"}, {"format": {"type": "text"}, "verbosity": "high"}])
def test_unstructured_requests_keep_existing_projected_body(text):
    assert project({**request(), "text": text}) == project(request())


def test_json_object_maps_without_inventing_schema_or_strictness():
    projected = project({**request(), "text": {"format": {"type": "json_object"}}})
    assert projected["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("text", [
    [], "json_schema", {"format": []}, {"format": {}}, {"format": {"type": []}},
    {"format": {"type": "xml"}}, {"format": {"type": "text", "schema": {}}},
    {"format": {"type": "json_object", "strict": True}},
    {"format": {"type": "json_schema", "schema": {}}},
    {"format": {"type": "json_schema", "name": "bad name", "schema": {}}},
    {"format": {"type": "json_schema", "name": "x" * 65, "schema": {}}},
    {"format": {"type": "json_schema", "name": "x", "schema": []}},
    {"format": {"type": "json_schema", "name": "x", "schema": {}, "strict": 1}},
    {"format": {"type": "json_schema", "name": "x", "schema": {}, "strict": "true"}},
    {"format": {"type": "json_schema", "name": "x", "schema": {}, "description": []}},
    {"format": {"type": "json_schema", "name": "x", "schema": {}, "unknown": 1}},
])
def test_malformed_or_unsupported_format_is_not_silently_dropped(text):
    with pytest.raises(ResponsesAdapterError, match="format|text configuration"):
        project({**request(), "text": text})


def test_gateway_forwards_canonical_schema_and_retains_safe_format_commitment():
    captured = []
    verdict = {"item_scores": [], "hard_gates_passed": False, "summary": "Synthetic transport only."}

    def upstream(_binding, body):
        captured.append(deepcopy(body))
        return SimpleNamespace(status=HTTPStatus.OK, latency_ms=0, body=json.dumps({
            "model": MODEL, "choices": [{"message": {"role": "assistant", "content": json.dumps(verdict)},
                                         "finish_reason": "stop"}],
        }).encode())

    text_format = {"type": "json_schema", "name": "codex_output_schema", "strict": True,
                   "schema": judge_schema()}
    req = {**request(), "text": {"format": text_format}}
    with ResponsesAdapterGateway({"opus_5": route()}, upstream_transport=upstream,
                                 local_bearer_token="synthetic-local-token") as gateway:
        _, _, (endpoint, token) = gateway.adapted_routes()["opus_5"].config.for_subprocess()

        def post(body):
            query = urllib.request.Request(endpoint + "/responses", data=json.dumps(body).encode(), method="POST",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
            return urllib.request.urlopen(query, timeout=5)

        with post(req) as result:
            assert result.status == 200
        assert len(captured) == 1
        actual = captured[0]["response_format"]
        assert actual == {"type": "json_schema", "json_schema": {k: v for k, v in text_format.items() if k != "type"}}
        signed = gateway.receipts[-1]
        verify_adapter_receipt(signed)
        shape = signed.payload["request_shape"]
        assert shape["response_format_projection"] == "eva.responses-text-format-to-chat.v1"
        assert shape["response_format_type"] == "json_schema"
        assert shape["response_format_blake3"] == blake3_hex(actual)
        assert "schema" not in shape and "name" not in shape
        with pytest.raises(urllib.error.HTTPError) as error:
            post({**request(), "text": {"format": {"type": "xml"}}})
        assert error.value.code == 400 and len(captured) == 1
        assert gateway.receipts[-1].payload["upstream_http_status"] is None
