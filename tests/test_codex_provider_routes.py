from __future__ import annotations

import asyncio
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
from typing import Any
import urllib.error
import urllib.request

import pytest

import eva_agent.codex_providers.adapter as adapter_module
from eva_agent.codex_providers import (
    AdapterLimits,
    AdapterReceiptSigner,
    CANARY_PROMPT,
    CodexConfigOverrides,
    CodexExecLaunch,
    CodexProviderConfigurationError,
    ProcessOutcome,
    PROJECTION_VERSION,
    ResponsesAdapterError,
    ResponsesAdapterGateway,
    load_allowlisted_environment,
    load_codex_provider_routes,
    probe_route_once,
    probe_routes_concurrently,
    receipts_document,
    responses_to_chat,
    verify_adapter_receipt,
    verify_canary_receipt,
)
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes
from eva_agent.codex_providers.routes import _PrivateText


SECRET = "secret-canary-value-never-record"
ENDPOINT = "https://private-provider.invalid/v1"


def _registry(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "rlevo.med-research-model-registry.v1",
                "models": [
                    {
                        "role_id": "architect_opus_5",
                        "api_model_id": "aws/anthropic/bedrock-claude-opus-5",
                    },
                    {
                        "role_id": "critic_opus_4_8",
                        "api_model_id": "azure/anthropic/claude-opus-4-8",
                    },
                    {
                        "role_id": "cascade_qwen3_6_27b",
                        "api_model_id": "nvidia/qwen/qwen3.6-27b",
                    },
                    {
                        "role_id": "cascade_glm_5_2",
                        "api_model_id": "nvidia/zai-org/glm-5.2",
                    },
                    {
                        "role_id": "cascade_deepseek_v4_flash",
                        "api_model_id": "nvidia/deepseek-ai/deepseek-v4-flash",
                    },
                    {
                        "role_id": "cascade_gpt_5_6_sol",
                        "api_model_id": "registry/must-not-win",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _environment() -> dict[str, str]:
    return {
        "NVIDIA_INFERENCE_API_KEY": SECRET,
        "NVIDIA_INFERENCE_BASE_URL": ENDPOINT,
        "MODEL_GPT_5_6_SOL": "openai/openai/gpt-5.6-sol-exact",
        "MODEL_GEMINI_3_1_PRO": "gcp/google/gemini-3.1-pro-preview",
        "MODEL_GEMINI_3_5_FLASH": "gcp/google/gemini-3.5-flash",
        "MODEL_GEMINI_3_8_FLASH": "gcp/google/gemini-3.8-flash",
        "MODEL_GLM_5_1": "nvidia/zai-org/glm-5.1",
        "MODEL_GLM_5_3": "nvidia/zai-org/glm-5.3",
        "MODEL_GLM_5_3_FLASH": "nvidia/zai-org/glm-5.3-flash",
        "MODEL_QWEN_3_5_0_8B": "nvidia/qwen/qwen3.5-0.8b",
        "MODEL_QWEN_3_5_9B": "nvidia/qwen/qwen3.5-9b",
        "MODEL_QWEN_3_5_35B_A3B": "nvidia/qwen/qwen3.5-35b-a3b",
        "MODEL_QWEN_3_5_122B_A10B": "nvidia/qwen/qwen3.5-122b-a10b",
        "MODEL_QWEN_3_5_397B_A17B": "nvidia/qwen/qwen3-5-397b-a17b",
        "MODEL_DEEPSEEK_V4_PRO": "nvidia/deepseek-ai/deepseek-v4-pro",
        "UNRELATED_SECRET": "must-never-be-read",
    }


def _routes(tmp_path: Path):
    return load_codex_provider_routes(
        env_files=(),
        environment=_environment(),
        registry_path=_registry(tmp_path / "registry.json"),
    )


def _local_upstream_route(route, endpoint: str):
    return replace(
        route,
        config=CodexConfigOverrides(
            provider_id=route.config.provider_id,
            model_id=route.model_id,
            credential_env_name=route.config.credential_env_name,
            endpoint_env_name=route.config.endpoint_env_name,
            _credential=_PrivateText(SECRET),
            _endpoint=_PrivateText(endpoint),
        ),
    )


class _FakeChatUpstream:
    def __init__(self, responder):
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                assert self.path == "/v1/chat/completions"
                assert self.headers.get("Authorization") == f"Bearer {SECRET}"
                length = int(self.headers["Content-Length"])
                request = json.loads(self.rfile.read(length))
                owner.requests.append(request)
                status, response = owner.responder(request, len(owner.requests))
                payload = json.dumps(response, separators=(",", ":")).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _chat_response(model: str, *, content: str | None = "OK", tool_calls=None):
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls or [],
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _post_json(url: str, token: str, body: dict[str, Any]) -> tuple[int, bytes, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read(), response.headers["Content-Type"]


def _passed_jsonl(text: str = "OK") -> bytes:
    rows = (
        {"type": "thread.started", "thread_id": "not-recorded"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    )
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


def test_dotenv_parser_only_reads_allowlisted_names_and_never_executes(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "UNRELATED_SECRET=do-not-read\n"
        "NVIDIA_API_KEY_DAGUANG=chosen\n"
        "NVIDIA_INFERENCE_API_KEY=${NVIDIA_API_KEY_DAGUANG}\n"
        "MODEL_GPT_5_6_SOL='vendor/exact-model'\n",
        encoding="utf-8",
    )
    values = load_allowlisted_environment((dotenv,), environment={})
    assert "UNRELATED_SECRET" not in values.names
    assert values.get("NVIDIA_INFERENCE_API_KEY") == "chosen"
    assert values.get("MODEL_GPT_5_6_SOL") == "vendor/exact-model"
    assert "chosen" not in repr(values)
    with pytest.raises(CodexProviderConfigurationError, match="not allowlisted"):
        values.get("UNRELATED_SECRET")


def test_routes_preserve_exact_model_ids_and_cover_all_requested_families(tmp_path: Path) -> None:
    routes = _routes(tmp_path)
    assert routes["gpt_5_6_sol"].model_id == "openai/openai/gpt-5.6-sol-exact"
    assert routes["opus_5"].model_id == "aws/anthropic/bedrock-claude-opus-5"
    assert routes["opus_4_8"].model_id == "azure/anthropic/claude-opus-4-8"
    assert routes["gemini_3_1_pro"].model_id == "gcp/google/gemini-3.1-pro-preview"
    assert routes["gemini_3_5_flash"].model_id == "gcp/google/gemini-3.5-flash"
    assert routes["gemini_3_8_flash"].model_id == "gcp/google/gemini-3.8-flash"
    assert routes["glm_5_1"].model_id == "nvidia/zai-org/glm-5.1"
    assert routes["glm_5_2"].model_id == "nvidia/zai-org/glm-5.2"
    assert routes["glm_5_3"].model_id == "nvidia/zai-org/glm-5.3"
    assert routes["glm_5_3_flash"].model_id == "nvidia/zai-org/glm-5.3-flash"
    assert routes["qwen_3_5_0_8b"].model_id == "nvidia/qwen/qwen3.5-0.8b"
    assert routes["qwen_3_5_9b"].model_id == "nvidia/qwen/qwen3.5-9b"
    assert routes["qwen_3_5_35b_a3b"].model_id == "nvidia/qwen/qwen3.5-35b-a3b"
    assert routes["qwen_3_5_122b_a10b"].model_id == "nvidia/qwen/qwen3.5-122b-a10b"
    assert routes["qwen_3_5_397b_a17b"].model_id == "nvidia/qwen/qwen3-5-397b-a17b"
    assert routes["qwen_3_6_27b"].model_id == "nvidia/qwen/qwen3.6-27b"
    assert routes["deepseek_v4_flash"].model_id == "nvidia/deepseek-ai/deepseek-v4-flash"
    assert routes["deepseek_v4_pro"].model_id == "nvidia/deepseek-ai/deepseek-v4-pro"
    assert routes["opus_5"].provider_family == "anthropic"
    assert routes["qwen_3_6_27b"].provider_family == "qwen"
    assert routes["qwen_3_5_397b_a17b"].provider_family == "qwen"
    assert routes["glm_5_1"].provider_family == "glm"
    assert routes["glm_5_2"].provider_family == "glm"
    assert routes["glm_5_3"].provider_family == "glm"
    assert routes["glm_5_3_flash"].provider_family == "glm"
    serialized = json.dumps({key: dict(value.safe_metadata) for key, value in routes.items()})
    assert SECRET not in serialized
    assert ENDPOINT not in serialized
    assert SECRET not in repr(routes["gpt_5_6_sol"])
    assert ENDPOINT not in repr(routes["gpt_5_6_sol"])


def test_astra_teacher_route_requires_explicit_model_and_preserves_responses_transport() -> None:
    environment = _environment()
    assert not load_codex_provider_routes(
        env_files=(), environment=environment, route_ids=("gpt_6_astra",)
    )
    environment["MODEL_GPT_6_ASTRA"] = "gpt-6-astra"
    route = load_codex_provider_routes(
        env_files=(), environment=environment, route_ids=("gpt_6_astra",)
    )["gpt_6_astra"]
    assert route.model_id == "gpt-6-astra"
    assert route.provider_family == "openai"
    assert route.registry_role_id is None
    overrides, _, _ = route.config.for_subprocess()
    assert any('wire_api="responses"' in value for value in overrides)
    assert SECRET not in repr(route)


def test_hub_astra_routes_require_separate_explicit_ids_and_remain_distinct() -> None:
    from eva_agent.deployment.rollout_adapters import rollout_model_catalog_document
    from eva_agent.training.teacher_batch import TEACHER_ROUTES
    environment = _environment()
    ids = ("gpt_6_astra_openai", "gpt_6_astra_azure")
    assert not load_codex_provider_routes(env_files=(), environment=environment, route_ids=ids)
    environment.update({
        "MODEL_GPT_6_ASTRA_OPENAI": "openai/openai/gpt-6-astra",
        "MODEL_GPT_6_ASTRA_AZURE": "azure/openai/gpt-6-astra",
    })
    routes = load_codex_provider_routes(env_files=(), environment=environment, route_ids=ids)
    assert tuple(routes) == ids
    assert len({route.config.provider_id for route in routes.values()}) == 2
    for route in routes.values():
        assert route.route_id in TEACHER_ROUTES
        assert route.provider_family == "openai"
        assert route.registry_role_id is None
        overrides, _, _ = route.config.for_subprocess()
        assert any('wire_api="responses"' in value for value in overrides)
        assert any('request_max_retries=0' in value for value in overrides)
        assert SECRET not in repr(route) and ENDPOINT not in repr(route)
    catalog = rollout_model_catalog_document(routes, route_order=ids)
    assert {row["slug"] for row in catalog["models"]} == {route.model_id for route in routes.values()}
    assert all(row["default_reasoning_level"] == "low" and row["shell_type"] == "disabled" for row in catalog["models"])


def test_large_optional_teacher_route_is_absent_without_explicit_configuration(
    tmp_path: Path,
) -> None:
    environment = _environment()
    environment.pop("MODEL_QWEN_3_5_397B_A17B")
    routes = load_codex_provider_routes(
        env_files=(),
        environment=environment,
        registry_path=_registry(tmp_path / "registry.json"),
        route_ids=("qwen_3_5_397b_a17b", "glm_5_2", "deepseek_v4_pro"),
    )
    assert tuple(routes) == ("glm_5_2", "deepseek_v4_pro")


def test_adapter_loopback_credential_is_distinct_from_direct_provider_key(
    tmp_path: Path,
) -> None:
    route = _routes(tmp_path)["opus_5"]
    local_token = "local-loopback-only-never-serialize"
    gateway = ResponsesAdapterGateway(
        {"opus_5": route},
        limits=AdapterLimits(max_concurrency=4),
        local_bearer_token=local_token,
        local_credential_env_name="EVA_CODEX_OPUS5_ADAPTER_TOKEN",
    )
    planned = gateway.planned_adapted_routes()["opus_5"]
    with gateway:
        adapted = gateway.adapted_routes()["opus_5"]
        assert adapted.config.safe_blake3 == planned.config.safe_blake3
        _overrides, child_env, private = adapted.config.for_subprocess()
        assert adapted.config.credential_env_name == "EVA_CODEX_OPUS5_ADAPTER_TOKEN"
        assert child_env == {"EVA_CODEX_OPUS5_ADAPTER_TOKEN": local_token}
        assert route.config.credential_env_name == "NVIDIA_INFERENCE_API_KEY"
        assert private[1] == local_token
        public = json.dumps(dict(gateway.safe_metadata), sort_keys=True)
        assert local_token not in public
        assert SECRET not in public
        assert ENDPOINT not in public


def test_codex_launch_uses_responses_zero_retries_exact_prompt_and_minimal_env(tmp_path: Path) -> None:
    route = _routes(tmp_path)["gpt_5_6_sol"]
    launch = CodexExecLaunch(route, "codex", str(tmp_path), 60)
    argv, environment, _ = launch.for_subprocess()
    assert argv[-1] == CANARY_PROMPT
    assert "--ephemeral" in argv and "--ignore-user-config" in argv and "--json" in argv
    assert "--strict-config" in argv and "--ignore-rules" in argv
    overrides = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--config"]
    assert any(value.endswith('.wire_api="responses"') for value in overrides)
    assert any(value.endswith(".request_max_retries=0") for value in overrides)
    assert any(value.endswith(".stream_max_retries=0") for value in overrides)
    assert argv[argv.index("--model") + 1] == route.model_id
    assert environment["NVIDIA_INFERENCE_API_KEY"] == SECRET
    assert "UNRELATED_SECRET" not in environment
    assert SECRET not in repr(launch)
    assert ENDPOINT not in repr(launch)
    assert SECRET not in json.dumps(dict(route.config.safe_projection))
    assert ENDPOINT not in json.dumps(dict(route.config.safe_projection))


def test_one_attempt_pass_receipt_is_safe_and_blake3_bound(tmp_path: Path) -> None:
    route = _routes(tmp_path)["gemini_3_1_pro"]
    calls = 0

    async def fake(launch: CodexExecLaunch) -> ProcessOutcome:
        nonlocal calls
        calls += 1
        return ProcessOutcome(0, _passed_jsonl(), b"")

    receipt = asyncio.run(
        probe_route_once(
            route,
            cwd=tmp_path,
            runner=fake,
            now=lambda: "2026-09-07T00:00:00Z",
        )
    )
    assert calls == 1
    assert receipt.responses_accepted is True
    assert receipt.semantic_exact_ok is True
    assert receipt.route_status == "direct-pass"
    assert receipt.no_tool_calls is True
    assert receipt.request_attempt_count == 1
    verify_canary_receipt(receipt)
    serialized = canonical_json_bytes(receipt.to_dict())
    assert SECRET.encode() not in serialized
    assert ENDPOINT.encode() not in serialized
    with pytest.raises(CodexProviderConfigurationError, match="BLAKE3"):
        replace(receipt, latency_ms=receipt.latency_ms + 1)


def test_semantic_failure_is_not_retried(tmp_path: Path) -> None:
    route = _routes(tmp_path)["qwen_3_6_27b"]
    calls = 0

    async def fake(launch: CodexExecLaunch) -> ProcessOutcome:
        nonlocal calls
        calls += 1
        return ProcessOutcome(0, _passed_jsonl("NOT OK"), b"")

    receipt = asyncio.run(probe_route_once(route, cwd=tmp_path, runner=fake))
    assert calls == 1
    assert receipt.responses_accepted is True
    assert receipt.semantic_exact_ok is False
    assert receipt.status == "semantic_failure"
    assert receipt.route_status == "unavailable"
    assert receipt.semantic_retry_count == 0


def test_responses_protocol_failure_is_redacted_and_classified(tmp_path: Path) -> None:
    route = _routes(tmp_path)["gpt_5_6_sol"]

    async def fake(launch: CodexExecLaunch) -> ProcessOutcome:
        body = (
            '{"type":"error","message":"404 responses not found at '
            + ENDPOINT
            + " with "
            + SECRET
            + '"}\n{"type":"turn.failed"}\n'
        ).encode()
        return ProcessOutcome(1, body, (ENDPOINT + SECRET).encode())

    receipt = asyncio.run(probe_route_once(route, cwd=tmp_path, runner=fake))
    assert receipt.responses_accepted is False
    assert receipt.failure_class == "responses_protocol_incompatible"
    assert receipt.http_status == 404
    assert receipt.route_status == "needs-Responses-adapter"
    serialized = canonical_json_bytes(receipt.to_dict())
    assert SECRET.encode() not in serialized
    assert ENDPOINT.encode() not in serialized


def test_multiple_routes_execute_concurrently_without_outer_serializing(tmp_path: Path) -> None:
    routes = _routes(tmp_path)
    active = 0
    maximum = 0

    async def fake(launch: CodexExecLaunch) -> ProcessOutcome:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.02)
        active -= 1
        return ProcessOutcome(0, _passed_jsonl(), b"")

    receipts = asyncio.run(
        probe_routes_concurrently(
            routes,
            route_ids=("gpt_5_6_sol", "gemini_3_1_pro", "qwen_3_6_27b"),
            cwd=tmp_path,
            runner=fake,
        )
    )
    assert maximum == 3
    assert [receipt.route_id for receipt in receipts] == [
        "gpt_5_6_sol",
        "gemini_3_1_pro",
        "qwen_3_6_27b",
    ]
    document = receipts_document(receipts)
    assert document["receipt_count"] == 3
    assert SECRET not in json.dumps(document)
    assert ENDPOINT not in json.dumps(document)


def test_provider_adapter_does_not_mutate_canonical_evamed_values(tmp_path: Path) -> None:
    canonical_values = {
        "tool": {
            "name": "execute_code",
            "description": "Exact canonical description.",
            "input_schema": {
                "type": "object",
                "properties": {"stage": {"const": "S3"}},
                "required": ["stage"],
                "additionalProperties": False,
            },
        },
        "rubric": {"rubric_id": "exact", "items": [{"id": "r1", "weight": 1}]},
        "policy": {"allowed_tools": ["execute_code"], "stage": "S3"},
        "evidence": {"workspace_blake3": "a" * 64},
    }
    before = canonical_json_bytes(canonical_values)
    _routes(tmp_path)
    assert canonical_json_bytes(canonical_values) == before


def test_invalid_endpoint_fails_before_process_boundary(tmp_path: Path) -> None:
    environment = _environment()
    environment["NVIDIA_INFERENCE_BASE_URL"] = "https://service.example.invalid/redacted"
    with pytest.raises(CodexProviderConfigurationError, match="credential-free HTTPS"):
        load_codex_provider_routes(
            env_files=(),
            environment=environment,
            registry_path=_registry(tmp_path / "registry.json"),
        )


def test_adapter_projects_provider_options_without_mutating_tool_schema(tmp_path: Path) -> None:
    routes = _routes(tmp_path)
    tool = {
        "type": "function",
        "name": "batch_lookup",
        "description": "Look up independent records.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
            },
            "required": ["ids"],
            "additionalProperties": False,
        },
    }
    before = canonical_json_bytes(tool)
    qwen_binding = ResponsesAdapterGateway(
        {"qwen_3_6_27b": routes["qwen_3_6_27b"]}
    )._bindings[routes["qwen_3_6_27b"].model_id]
    qwen_chat, _ = responses_to_chat(
        {
            "model": qwen_binding.model_id,
            "input": "test",
            "tools": [tool],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "stream": False,
        },
        qwen_binding,
    )
    assert qwen_chat["parallel_tool_calls"] is True
    assert "uniqueItems" not in json.dumps(qwen_chat["tools"])
    assert canonical_json_bytes(tool) == before

    opus_binding = ResponsesAdapterGateway(
        {"opus_5": routes["opus_5"]}
    )._bindings[routes["opus_5"].model_id]
    opus_chat, _ = responses_to_chat(
        {
            "model": opus_binding.model_id,
            "input": "test",
            "tools": [tool],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "stream": False,
        },
        opus_binding,
    )
    assert "parallel_tool_calls" not in opus_chat
    assert "uniqueItems" in json.dumps(opus_chat["tools"])


def test_anthropic_projection_never_ends_with_assistant_prefill(tmp_path: Path) -> None:
    route = _routes(tmp_path)["opus_4_8"]
    binding = ResponsesAdapterGateway({route.route_id: route})._bindings[route.model_id]
    tool = {
        "type": "function",
        "name": "lookup",
        "description": "Lookup.",
        "parameters": {"type": "object", "properties": {}},
    }
    before = canonical_json_bytes(tool)
    chat, _ = responses_to_chat(
        {
            "model": route.model_id,
            "input": [
                {"type": "message", "role": "user", "content": "work"},
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call-1", "output": "{}"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "I will inspect another resource.",
                },
            ],
            "tools": [tool],
        },
        binding,
    )
    assert [message["role"] for message in chat["messages"]][-3:] == [
        "tool",
        "assistant",
        "user",
    ]
    assert chat["messages"][-1]["content"] == (
        "Continue the task from the completed tool result."
    )
    assert canonical_json_bytes(tool) == before


def test_anthropic_tool_free_compaction_transcribes_retained_tool_history(
    tmp_path: Path,
) -> None:
    """Mirror the signed v4 failure shape without crossing a provider boundary."""

    route = _routes(tmp_path)["opus_5"]
    binding = ResponsesAdapterGateway({route.route_id: route})._bindings[route.model_id]
    tool_items = []
    inputs = [{"type": "message", "role": "user", "content": "compact this history"}]
    for ordinal in range(14):
        call = {
            "type": "function_call",
            "call_id": f"call-{ordinal}",
            "namespace": "workspace",
            "name": "workspace_read",
            "arguments": json.dumps(
                {"path": f"evidence/{ordinal}.json", "offset": ordinal * 64},
                separators=(",", ":"),
            ),
        }
        output = {
            "type": "function_call_output",
            "call_id": f"call-{ordinal}",
            "output": json.dumps(
                {"bytes": f"verified-{ordinal}", "eof": True},
                separators=(",", ":"),
            ),
        }
        tool_items.extend((call, output))
        inputs.extend((call, output))
    inputs.extend(
        [
            {"type": "message", "role": "assistant", "content": "Summarizing."},
            {"type": "message", "role": "user", "content": "Continue compacting."},
            {"type": "message", "role": "assistant", "content": "Almost done."},
            {"type": "message", "role": "user", "content": "Return the summary."},
            {"type": "message", "role": "user", "content": "Keep exact evidence."},
        ]
    )
    assert len(inputs) == 34
    request = {
        "model": route.model_id,
        "input": inputs,
        "tools": [],
        "parallel_tool_calls": False,
        "stream": True,
    }
    before = canonical_json_bytes(request)

    chat, projection = responses_to_chat(request, binding)

    assert projection.chat_tools == ()
    assert "tools" not in chat
    assert "parallel_tool_calls" not in chat
    assert all(message["role"] != "tool" for message in chat["messages"])
    assert all("tool_calls" not in message for message in chat["messages"])
    transcript_rows = []
    for message in chat["messages"]:
        content = message.get("content")
        if isinstance(content, str) and content.startswith(
            adapter_module.TOOL_HISTORY_TRANSCRIPT_PREFIX
        ):
            transcript_rows.append(
                json.loads(
                    content[len(adapter_module.TOOL_HISTORY_TRANSCRIPT_PREFIX) :]
                )
            )
    assert transcript_rows == tool_items
    assert canonical_json_bytes(request) == before


def test_qwen_flat_opt_in_preserves_every_original_schema_and_inverse_name(tmp_path: Path) -> None:
    from training.automedbench_lite.public_tools import TOOLS
    route = _routes(tmp_path)["qwen_3_6_27b"]
    gateway = ResponsesAdapterGateway({route.route_id: route}, preserve_qwen_tool_schemas=True)
    binding = gateway._bindings[route.model_id]
    children = [{"type": "function", "name": row["name"], "description": row["description"],
                 "parameters": row["inputSchema"]} for row in TOOLS]
    # The exact mode must preserve even keywords removed by the historical route.
    children.append({"type": "function", "name": "fixture", "description": "Exact fixture",
        "parameters": {"type": "object", "additionalProperties": False,
            "properties": {"ids": {"type": "array", "uniqueItems": True}}, "required": ["ids"]}})
    request = {"model": route.model_id, "input": "Use the declared tools.",
        "tools": [{"type": "namespace", "name": "automed_eval", "tools": children}],
        "tool_choice": {"type": "function", "namespace": "automed_eval", "name": "fixture"}}
    original = canonical_json_bytes(request)
    chat, projection = responses_to_chat(request, binding, preserve_qwen_tool_schemas=True)
    assert len(chat["tools"]) == len(children)
    assert not projection.packed_namespaces
    for child, row in zip(children, chat["tools"]):
        function = row["function"]
        assert function["parameters"] == child["parameters"]
        assert function["description"] == child["description"]
        assert projection.decode_call(function["name"], '{"ids":[]}') == (
            "automed_eval", child["name"], '{"ids":[]}')
    assert chat["tool_choice"]["function"]["name"] == "automed_eval__fixture"
    assert canonical_json_bytes(request) == original
    assert gateway.safe_metadata["projection_version"] == adapter_module.FLAT_QWEN_PROJECTION_VERSION


def test_qwen_packed_namespace_v2_has_strict_schema_and_stable_digest(tmp_path: Path) -> None:
    route = _routes(tmp_path)["qwen_3_6_27b"]
    binding = ResponsesAdapterGateway({route.route_id: route})._bindings[route.model_id]
    tools = [
        {
            "type": "namespace",
            "name": "clinical",
            "description": "Clinical.",
            "tools": [
                {
                    "type": "function",
                    "name": "score",
                    "description": "Score patient.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ids": {"type": "array", "uniqueItems": True}
                        },
                        "required": ["ids"],
                    },
                },
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Lookup source.",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        }
    ]
    before = canonical_json_bytes(tools)
    chat, _ = responses_to_chat(
        {
            "model": route.model_id,
            "input": "x",
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        },
        binding,
    )
    assert len(chat["tools"]) == 1
    function = chat["tools"][0]["function"]
    assert function["name"] == "clinical__dispatch"
    assert function["parameters"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tool_name": {"type": "string", "enum": ["score", "lookup"]},
            "arguments": {"type": "object", "additionalProperties": True},
        },
        "required": ["tool_name", "arguments"],
    }
    assert blake3_hex(chat["tools"]) == (
        "a0149235339550e03f15c19bede79a25dde00161ad8fc35b772694332dce7608"
    )
    assert canonical_json_bytes(tools) == before

    forced, _ = responses_to_chat(
        {
            "model": route.model_id,
            "input": "x",
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "namespace": "clinical",
                "name": "score",
            },
        },
        binding,
    )
    assert forced["tools"][0]["function"]["parameters"]["properties"][
        "tool_name"
    ]["enum"] == ["score"]
    assert forced["tool_choice"] == {
        "type": "function",
        "function": {"name": "clinical__dispatch"},
    }


def test_qwen_coalesces_adjacent_text_roles_without_changing_tools(
    tmp_path: Path,
) -> None:
    routes = _routes(tmp_path)
    qwen = routes["qwen_3_5_397b_a17b"]
    qwen_binding = ResponsesAdapterGateway(
        {qwen.route_id: qwen}
    )._bindings[qwen.model_id]
    tool = {
        "type": "function",
        "name": "lookup",
        "description": "Lookup.",
        "parameters": {"type": "object", "properties": {}},
    }
    request = {
        "model": qwen.model_id,
        "instructions": "base system",
        "input": [
            {"type": "message", "role": "developer", "content": "developer"},
            {"type": "message", "role": "user", "content": "first user"},
            {"type": "message", "role": "user", "content": "second user"},
        ],
        "tools": [tool],
        "parallel_tool_calls": True,
    }
    before = canonical_json_bytes(tool)
    chat, _ = responses_to_chat(request, qwen_binding)
    assert chat["messages"] == [
        {"role": "system", "content": "base system\n\ndeveloper"},
        {"role": "user", "content": "first user\n\nsecond user"},
    ]
    assert chat["parallel_tool_calls"] is True
    assert chat["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert canonical_json_bytes(tool) == before

    google = routes["gemini_3_1_pro"]
    google_binding = ResponsesAdapterGateway(
        {google.route_id: google}
    )._bindings[google.model_id]
    google_request = {**request, "model": google.model_id}
    google_chat, _ = responses_to_chat(google_request, google_binding)
    assert [row["role"] for row in google_chat["messages"]] == [
        "system",
        "system",
        "user",
        "user",
    ]


def test_adapter_nonstream_parallel_tool_round_trip_and_signed_redacted_receipts(
    tmp_path: Path,
) -> None:
    source_route = _routes(tmp_path)["qwen_3_6_27b"]

    def respond(request: dict[str, Any], ordinal: int):
        if ordinal == 1:
            names = [tool["function"]["name"] for tool in request["tools"]]
            assert names == ["lookup", "clinical__dispatch"]
            return 200, _chat_response(
                source_route.model_id,
                content=None,
                tool_calls=[
                    {
                        "id": "call-a",
                        "type": "function",
                        "function": {"name": names[0], "arguments": '{"id":"a"}'},
                    },
                    {
                        "id": "call-b",
                        "type": "function",
                        "function": {
                            "name": names[1],
                            "arguments": '{"tool_name":"score","arguments":{"id":"b"}}',
                        },
                    },
                ],
            )
        assert ordinal == 2
        assistant = [message for message in request["messages"] if message["role"] == "assistant"][-1]
        tool_messages = [message for message in request["messages"] if message["role"] == "tool"]
        assert [call["id"] for call in assistant["tool_calls"]] == ["call-a", "call-b"]
        packed = json.loads(assistant["tool_calls"][1]["function"]["arguments"])
        assert packed == {"tool_name": "score", "arguments": {"id": "b"}}
        assert [message["tool_call_id"] for message in tool_messages] == ["call-a", "call-b"]
        assert [message["content"] for message in tool_messages] == ["A", "B"]
        return 200, _chat_response(source_route.model_id)

    tools = [
        {
            "type": "function",
            "name": "lookup",
            "description": "Lookup.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "namespace",
            "name": "clinical",
            "description": "Clinical tools.",
            "tools": [
                {
                    "type": "function",
                    "name": "score",
                    "description": "Score.",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
    ]
    token = "local-adapter-token"
    with _FakeChatUpstream(respond) as upstream:
        route = _local_upstream_route(source_route, upstream.endpoint)
        with ResponsesAdapterGateway(
            {route.route_id: route},
            limits=AdapterLimits(max_concurrency=4, upstream_timeout_seconds=10),
            signer=AdapterReceiptSigner.ephemeral(),
            local_bearer_token=token,
        ) as gateway:
            adapted = gateway.adapted_routes()[route.route_id]
            _, _, (endpoint, _) = adapted.config.for_subprocess()
            status, payload, content_type = _post_json(
                endpoint + "/responses",
                token,
                {
                    "model": route.model_id,
                    "input": [{"type": "message", "role": "user", "content": "run"}],
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "stream": False,
                    "store": False,
                },
            )
            assert status == 200 and content_type == "application/json"
            first = json.loads(payload)
            calls = [item for item in first["output"] if item["type"] == "function_call"]
            assert [item["call_id"] for item in calls] == ["call-a", "call-b"]
            assert calls[0]["name"] == "lookup" and "namespace" not in calls[0]
            assert calls[1]["name"] == "score" and calls[1]["namespace"] == "clinical"
            assert json.loads(calls[1]["arguments"]) == {"id": "b"}

            second_input = [
                {"type": "message", "role": "user", "content": "run"},
                *calls,
                {"type": "function_call_output", "call_id": "call-a", "output": "A"},
                {"type": "function_call_output", "call_id": "call-b", "output": "B"},
            ]
            status, payload, _ = _post_json(
                endpoint + "/responses",
                token,
                {
                    "model": route.model_id,
                    "input": second_input,
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "stream": False,
                    "store": False,
                },
            )
            assert status == 200
            final_output = json.loads(payload)["output"][0]
            assert final_output["content"][0]["text"] == "OK"
            assert final_output["phase"] == "final_answer"
            assert len(gateway.receipts) == 2
            for receipt in gateway.receipts:
                verify_adapter_receipt(receipt)
            document = gateway.receipts_document()
            assert gateway.safe_metadata["projection_version"] == PROJECTION_VERSION
            serialized = canonical_json_bytes(document)
            assert SECRET.encode() not in serialized
            assert upstream.endpoint.encode() not in serialized
            assert b'"run"' not in serialized and b'"OK"' not in serialized
            assert all(row["payload"]["upstream_request_max_retries"] == 0 for row in document["receipts"])


def test_adapter_streaming_events_are_responses_compatible_and_single_attempt(tmp_path: Path) -> None:
    source_route = _routes(tmp_path)["opus_5"]

    def respond(request: dict[str, Any], ordinal: int):
        assert ordinal == 1
        assert "parallel_tool_calls" not in request
        return 200, _chat_response(source_route.model_id, content="OK")

    token = "local-stream-token"
    with _FakeChatUpstream(respond) as upstream:
        route = _local_upstream_route(source_route, upstream.endpoint)
        with ResponsesAdapterGateway(
            {route.route_id: route}, local_bearer_token=token
        ) as gateway:
            adapted = gateway.adapted_routes()[route.route_id]
            _, _, (endpoint, _) = adapted.config.for_subprocess()
            status, payload, content_type = _post_json(
                endpoint + "/responses",
                token,
                {
                    "model": route.model_id,
                    "input": "Return OK",
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "stream": True,
                    "store": False,
                },
            )
            assert status == 200 and content_type == "text/event-stream"
            text = payload.decode()
            assert "event: response.created\n" in text
            assert "event: response.output_text.delta\n" in text
            assert "event: response.completed\n" in text
            assert text.endswith("data: [DONE]\n\n")
            assert len(upstream.requests) == 1
            assert len(gateway.receipts) == 1
            assert gateway.receipts[0].payload["request_max_retries"] == 0


def test_adapter_exact_upstream_output_budget_is_explicit_and_signed(tmp_path: Path) -> None:
    source_route = _routes(tmp_path)["opus_5"]

    def respond(request: dict[str, Any], ordinal: int):
        assert ordinal == 1
        assert request["max_tokens"] == 32_768
        return 200, _chat_response(source_route.model_id, content="OK")

    token = "local-budget-token"
    with _FakeChatUpstream(respond) as upstream:
        route = _local_upstream_route(source_route, upstream.endpoint)
        with ResponsesAdapterGateway(
            {route.route_id: route},
            local_bearer_token=token,
            upstream_max_tokens_override=32_768,
        ) as gateway:
            adapted = gateway.adapted_routes()[route.route_id]
            _, _, (endpoint, _) = adapted.config.for_subprocess()
            status, _payload, _content_type = _post_json(
                endpoint + "/responses",
                token,
                {
                    "model": route.model_id,
                    "input": "Return OK",
                    "max_output_tokens": 4_096,
                    "stream": False,
                    "store": False,
                },
            )
            assert status == 200
            assert gateway.safe_metadata["upstream_max_tokens_override"] == 32_768
            shape = gateway.receipts[0].payload["request_shape"]
            assert shape["requested_max_output_tokens"] == 4_096
            assert shape["effective_upstream_max_tokens"] == 32_768
            assert shape["upstream_max_tokens_policy"] == "exact_override"
            response = gateway.receipts[0].payload["response_shape"]
            assert response["upstream_finish_reason"] == "stop"
            assert response["upstream_output_tokens"] == 2
            verify_adapter_receipt(gateway.receipts[0])

    with pytest.raises(ResponsesAdapterError, match="output budget"):
        ResponsesAdapterGateway(
            {source_route.route_id: source_route},
            upstream_max_tokens_override=True,
        )


def test_adapter_receipt_redacts_unrecognized_request_item_type(tmp_path: Path) -> None:
    source_route = _routes(tmp_path)["opus_5"]
    private_item_type = "private-item-type-must-not-enter-receipt"

    def respond(_request: dict[str, Any], _ordinal: int):
        raise AssertionError("invalid request must not reach the upstream boundary")

    token = "local-invalid-shape-token"
    with _FakeChatUpstream(respond) as upstream:
        route = _local_upstream_route(source_route, upstream.endpoint)
        with ResponsesAdapterGateway(
            {route.route_id: route}, local_bearer_token=token
        ) as gateway:
            adapted = gateway.adapted_routes()[route.route_id]
            _, _, (endpoint, _) = adapted.config.for_subprocess()
            with pytest.raises(urllib.error.HTTPError) as error:
                _post_json(
                    endpoint + "/responses",
                    token,
                    {
                        "model": route.model_id,
                        "input": [{"type": private_item_type}],
                        "stream": False,
                        "store": False,
                    },
                )
            assert error.value.code == 400
            assert upstream.requests == []
            assert len(gateway.receipts) == 1
            receipt = gateway.receipts[0]
            assert receipt.payload["request_shape"]["input_item_types"] == ["other"]
            assert private_item_type.encode() not in canonical_json_bytes(
                gateway.receipts_document()
            )
            verify_adapter_receipt(receipt)


def test_adapter_receipt_buckets_unrecognized_upstream_finish_reason(tmp_path: Path) -> None:
    source_route = _routes(tmp_path)["opus_5"]
    private_finish_reason = "private-finish-reason-must-not-enter-receipt"

    def respond(_request: dict[str, Any], ordinal: int):
        assert ordinal == 1
        response = _chat_response(source_route.model_id, content="OK")
        response["choices"][0]["finish_reason"] = private_finish_reason
        return 200, response

    token = "local-private-finish-token"
    with _FakeChatUpstream(respond) as upstream:
        route = _local_upstream_route(source_route, upstream.endpoint)
        with ResponsesAdapterGateway(
            {route.route_id: route}, local_bearer_token=token
        ) as gateway:
            adapted = gateway.adapted_routes()[route.route_id]
            _, _, (endpoint, _) = adapted.config.for_subprocess()
            status, _payload, _content_type = _post_json(
                endpoint + "/responses",
                token,
                {
                    "model": route.model_id,
                    "input": "Return OK",
                    "stream": False,
                    "store": False,
                },
            )
            assert status == 200
            assert len(gateway.receipts) == 1
            receipt = gateway.receipts[0]
            assert receipt.payload["response_shape"]["upstream_finish_reason"] == "other"
            assert private_finish_reason.encode() not in canonical_json_bytes(
                gateway.receipts_document()
            )
            verify_adapter_receipt(receipt)


def test_adapter_receipt_tampering_and_model_identity_fail_closed(tmp_path: Path) -> None:
    route = _routes(tmp_path)["opus_5"]
    signer = AdapterReceiptSigner.ephemeral()
    receipt = signer.sign(
        {
            "schema": "eva.codex-responses-adapter-receipt-payload.v1",
            "request_id": "00000000-0000-4000-8000-000000000001",
            "created_at_utc": "2026-09-07T00:00:00Z",
            "route_id": route.route_id,
            "model_id": route.model_id,
            "provider_family": route.provider_family,
            "projection_version": PROJECTION_VERSION,
            "status": "adapter_error",
            "failure_class": "adapter_contract_error",
            "failure_message_blake3": blake3_hex("safe failure"),
            "adapter_http_status": 400,
            "upstream_http_status": None,
            "upstream_latency_ms": 0,
            "upstream_body_blake3": None,
            "request_shape": {},
            "response_shape": {},
            "request_max_retries": 0,
            "stream_max_retries": 0,
            "upstream_request_max_retries": 0,
            "credential_value_recorded": False,
            "endpoint_value_recorded": False,
            "raw_request_recorded": False,
            "raw_upstream_output_recorded": False,
            "raw_response_recorded": False,
        }
    )
    verify_adapter_receipt(receipt)
    tampered = replace(receipt, payload_blake3="0" * 64)
    with pytest.raises(ResponsesAdapterError, match="payload BLAKE3"):
        verify_adapter_receipt(tampered)

    def respond(request: dict[str, Any], ordinal: int):
        return 200, _chat_response("different/model")

    token = "local-identity-token"
    with _FakeChatUpstream(respond) as upstream:
        local_route = _local_upstream_route(route, upstream.endpoint)
        with ResponsesAdapterGateway(
            {local_route.route_id: local_route}, local_bearer_token=token
        ) as gateway:
            adapted = gateway.adapted_routes()[local_route.route_id]
            _, _, (endpoint, _) = adapted.config.for_subprocess()
            request = urllib.request.Request(
                endpoint + "/responses",
                data=json.dumps(
                    {
                        "model": route.model_id,
                        "input": "test",
                        "stream": False,
                        "store": False,
                    }
                ).encode(),
                method="POST",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=10)
            assert error.value.code == 400
            assert gateway.receipts[0].payload["status"] == "adapter_error"
            assert gateway.receipts[0].payload["failure_class"] == "adapter_contract_error"


@pytest.mark.skipif(shutil.which("codex") is None, reason="Codex CLI unavailable")
def test_real_codex_exec_accepts_local_adapter_sse(tmp_path: Path) -> None:
    source_route = _routes(tmp_path)["qwen_3_6_27b"]

    def respond(request: dict[str, Any], ordinal: int):
        assert ordinal == 1
        assert request["model"] == source_route.model_id
        assert isinstance(request["messages"], list)
        return 200, _chat_response(source_route.model_id, content="OK")

    with _FakeChatUpstream(respond) as upstream:
        route = _local_upstream_route(source_route, upstream.endpoint)
        with ResponsesAdapterGateway({route.route_id: route}) as gateway:
            adapted = gateway.adapted_routes()[route.route_id]
            receipt = asyncio.run(
                probe_route_once(
                    adapted,
                    cwd=tmp_path,
                    timeout_seconds=30,
                )
            )
            assert receipt.semantic_exact_ok is True
            assert receipt.request_attempt_count == 1
            assert len(upstream.requests) == 1
            assert len(gateway.receipts) == 1
            verify_adapter_receipt(gateway.receipts[0])
