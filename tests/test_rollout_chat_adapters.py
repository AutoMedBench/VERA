from __future__ import annotations

from http import HTTPStatus
import json
from pathlib import Path
import stat
from typing import Any
import urllib.request

import pytest

import eva_agent.codex_providers.adapter as adapter_module
from eva_agent.codex_providers import (
    AdapterLimits,
    AdapterReceiptSigner,
    ResponsesAdapterGateway,
    load_codex_provider_routes,
    verify_adapter_receipt,
)
from eva_agent.deployment import CampaignDeploymentError
from eva_agent.deployment import CODEX_FIRST_RELEASE_CONFIG_OVERRIDES
from eva_agent.deployment import (
    PreparedPersistentCodexRuntime,
    prepare_persistent_codex_runtime,
)
from eva_agent.deployment.codex_child_exec import build_sanitized_codex_exec_plan
from eva_agent.deployment import medresearch_v2
from eva_agent.deployment.rollout_adapters import (
    ROLLOUT_ADAPTED_ROUTE_IDS,
    ROLLOUT_ADAPTER_TOKEN_ENV,
    ROLLOUT_DIRECT_ROUTE_IDS,
    ROLLOUT_ROUTE_IDS,
    ShardedRolloutAdapterGateway,
    materialize_rollout_model_catalog,
    rollout_model_catalog_document,
)
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes, is_blake3


SECRET = "rollout-adapter-secret-never-record"
ENDPOINT = "https://provider.invalid/v1"


def _registry(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "rlevo.med-research-model-registry.v1",
                "models": [
                    {
                        "role_id": "architect_opus_5",
                        "api_model_id": "anthropic/opus-5",
                    },
                    {
                        "role_id": "critic_opus_4_8",
                        "api_model_id": "anthropic/opus-4.8",
                    },
                    {
                        "role_id": "cascade_deepseek_v4_flash",
                        "api_model_id": "deepseek/deepseek-v4-flash",
                    },
                    {
                        "role_id": "cascade_gpt_5_6_sol",
                        "api_model_id": "openai/gpt-5.6-sol",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _routes(tmp_path: Path):
    return load_codex_provider_routes(
        env_files=(),
        registry_path=_registry(tmp_path / "registry.json"),
        environment={
            "NVIDIA_INFERENCE_API_KEY": SECRET,
            "NVIDIA_INFERENCE_BASE_URL": ENDPOINT,
            "MODEL_GEMINI_3_1_PRO": "google/gemini-3.1-pro-preview",
        },
        route_ids=ROLLOUT_ROUTE_IDS,
    )


def _post_json(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == HTTPStatus.OK
        return json.loads(response.read())


def _chat_tool_response(model: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-search",
                                "type": "function",
                                "function": {
                                    "name": "medical__search",
                                    "arguments": '{"query":"trial"}',
                                },
                            },
                            {
                                "id": "call-read",
                                "type": "function",
                                "function": {
                                    "name": "medical__read",
                                    "arguments": '{"record_id":"r1"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        },
        separators=(",", ":"),
    ).encode()


def test_rollout_model_catalog_is_exact_private_and_disables_responses_lite(
    tmp_path: Path,
) -> None:
    routes = _routes(tmp_path)
    document = rollout_model_catalog_document(routes)
    assert [row["slug"] for row in document["models"]] == [
        routes[route_id].model_id for route_id in ROLLOUT_ROUTE_IDS
    ]
    assert all(row["use_responses_lite"] is False for row in document["models"])
    assert all(row["supports_parallel_tool_calls"] is True for row in document["models"])
    assert all(row["tool_mode"] == "direct" for row in document["models"])
    assert all(row["shell_type"] == "disabled" for row in document["models"])
    assert all(row["auto_compact_token_limit"] is None for row in document["models"])

    early_compaction = rollout_model_catalog_document(
        routes, auto_compact_token_limit=49_152
    )
    assert all(
        row["auto_compact_token_limit"] == 49_152
        for row in early_compaction["models"]
    )
    for invalid in (True, 0, -1, 131_072, "49152"):
        with pytest.raises(CampaignDeploymentError, match="auto-compact"):
            rollout_model_catalog_document(
                routes, auto_compact_token_limit=invalid  # type: ignore[arg-type]
            )

    catalog = materialize_rollout_model_catalog(
        routes,
        root=(tmp_path / "private-catalog").resolve(),
        auto_compact_token_limit=49_152,
    )
    assert json.loads(catalog.path.read_bytes())["models"][0][
        "auto_compact_token_limit"
    ] == 49_152
    catalog.verify()
    assert stat.S_IMODE(catalog.path.stat().st_mode) == 0o400
    assert stat.S_IMODE(catalog.path.parent.stat().st_mode) == 0o700
    assert catalog.safe_metadata["use_responses_lite"] is False
    assert catalog.safe_metadata["supports_parallel_tool_calls"] is True
    assert catalog.config_override.startswith("model_catalog_json=")
    assert materialize_rollout_model_catalog(
        routes,
        root=catalog.path.parent,
        auto_compact_token_limit=49_152,
    ).document_blake3 == catalog.document_blake3

    catalog.path.chmod(0o600)
    with pytest.raises(CampaignDeploymentError, match="bytes differ"):
        catalog.verify()


def test_rollout_model_catalog_is_accepted_by_pinned_codex_appserver(
    tmp_path: Path,
) -> None:
    from codex_cli_bin import bundled_codex_path
    from openai_codex import Codex, CodexConfig

    routes = _routes(tmp_path)
    catalog = materialize_rollout_model_catalog(
        routes,
        root=(tmp_path / "private-catalog").resolve(),
        auto_compact_token_limit=49_152,
    )
    assert json.loads(catalog.path.read_bytes())["models"][0][
        "auto_compact_token_limit"
    ] == 49_152
    isolation = tmp_path / "isolated"
    isolation.mkdir(mode=0o700)
    codex_args: list[str] = []
    for override in CODEX_FIRST_RELEASE_CONFIG_OVERRIDES:
        codex_args.extend(("--config", override))
    codex_args.extend(("--config", catalog.config_override))
    codex_args.extend(("app-server", "--strict-config", "--listen", "stdio://"))
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "eva_agent"
        / "deployment"
        / "codex_child_exec.py"
    )
    plan = build_sanitized_codex_exec_plan(
        python_bin=Path(__import__("sys").executable).resolve(),
        wrapper_script=wrapper.resolve(),
        codex_bin=bundled_codex_path().resolve(),
        isolation_root=isolation.resolve(),
        credential_env_names=(
            "NVIDIA_INFERENCE_API_KEY",
            ROLLOUT_ADAPTER_TOKEN_ENV,
        ),
        codex_args=tuple(codex_args),
    )
    client = Codex(
        config=CodexConfig(
            launch_args_override=plan.launch_args,
            cwd=str(tmp_path),
            env={
                "NVIDIA_INFERENCE_API_KEY": SECRET,
                ROLLOUT_ADAPTER_TOKEN_ENV: "local-only-token",
            },
            client_name="eva_rollout_catalog_test",
            client_title="EVA Rollout Catalog Test",
            client_version="0.1.0",
        )
    )
    try:
        client.__enter__()
        assert client.metadata is not None
        assert client.metadata.serverInfo.version
        catalog.verify()
    finally:
        client.__exit__(None, None, None)


def test_shard_launch_recipe_commits_exact_model_catalog_before_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_cli_bin import bundled_codex_path

    routes = _routes(tmp_path)
    catalog = materialize_rollout_model_catalog(
        routes, root=(tmp_path / "private-catalog").resolve()
    )
    initial_isolation = tmp_path / "initial-isolation"
    initial_isolation.mkdir(mode=0o700)
    base = prepare_persistent_codex_runtime(
        codex_bin=bundled_codex_path().resolve(),
        cwd=tmp_path.resolve(),
        child_env={
            "NVIDIA_INFERENCE_API_KEY": SECRET,
            ROLLOUT_ADAPTER_TOKEN_ENV: "local-only-token",
        },
        app_server_shards=2,
        worker_width=2,
        isolation_root=initial_isolation.resolve(),
    )
    monkeypatch.setattr(medresearch_v2, "PROJECT_ROOT", tmp_path.resolve())
    launches, metadata = medresearch_v2._shard_launches(
        base=base,
        codex_bin=bundled_codex_path().resolve(),
        child_environment={
            "NVIDIA_INFERENCE_API_KEY": SECRET,
            ROLLOUT_ADAPTER_TOKEN_ENV: "local-only-token",
        },
        shard_count=2,
        model_catalog=catalog,
    )
    assert len(launches) == 2
    assert all(
        launch.launch_args_override is not None
        and launch.launch_args_override.count(catalog.config_override) == 1
        for launch in launches
    )
    assert metadata["rollout_model_catalog"] == dict(catalog.safe_metadata)
    assert metadata["rollout_model_catalog_configured_at_app_server_start"] is True
    assert metadata["rollout_effective_config_overrides"][-1] == catalog.config_override
    assert is_blake3(metadata["launch_blake3"])
    PreparedPersistentCodexRuntime(
        runner=base.runner,
        launch_options=launches[0],
        startup_metadata=metadata,
    )


def test_sharded_rollout_gateway_is_route_affine_capacity_bound_and_redacted(
    tmp_path: Path,
) -> None:
    routes = _routes(tmp_path)
    signer = AdapterReceiptSigner.ephemeral()
    gateway = ShardedRolloutAdapterGateway(
        routes={route_id: routes[route_id] for route_id in ROLLOUT_ADAPTED_ROUTE_IDS},
        app_server_shards=2,
        required_capacity=7,
        signer=signer,
        canary_receipt_blake3s={
            route_id: blake3_hex({"canary": route_id})
            for route_id in ROLLOUT_ADAPTED_ROUTE_IDS
        },
    )
    assert gateway.shard_count == 2
    assert gateway.per_shard_capacity == 4
    assert gateway.aggregate_max_concurrency == 8
    assert set(gateway.planned_routes) == set(ROLLOUT_ADAPTED_ROUTE_IDS)
    assert all(gateway.handles_provider(route.config.provider_id) for route in gateway.planned_routes.values())
    assert not gateway.handles_provider(routes[ROLLOUT_DIRECT_ROUTE_IDS[0]].config.provider_id)
    assert SECRET not in repr(gateway)
    assert is_blake3(gateway.binding_blake3)

    observed_receipts = []
    gateway.install_receipt_sink(observed_receipts.append)
    gateway.start()
    assert tuple(gateway.child_environment) == (ROLLOUT_ADAPTER_TOKEN_ENV,)
    indices = []
    for route_id in ROLLOUT_ADAPTED_ROUTE_IDS:
        planned = gateway.planned_routes[route_id]
        with gateway.reserve_route(planned.config.provider_id) as (index, live):
            indices.append(index)
            assert live.route_id == route_id
            assert live.model_id == planned.model_id
            assert live.config.safe_blake3 == planned.config.safe_blake3
    assert indices == [0, 1, 0, 1]
    assert gateway.receipts_document()["receipt_count"] == 0
    gateway.close()
    assert dict(gateway.child_environment) == {}
    with pytest.raises(CampaignDeploymentError, match="not restartable"):
        gateway.start()


@pytest.mark.parametrize("route_id", ("deepseek_v4_flash", "gemini_3_1_pro"))
def test_chat_adapter_preserves_namespace_schema_and_parallel_calls(
    tmp_path: Path, route_id: str
) -> None:
    route = _routes(tmp_path)[route_id]
    captured: list[dict[str, Any]] = []

    def upstream(binding, request):
        assert binding.route_id == route_id
        captured.append(dict(request))
        return adapter_module._UpstreamOutcome(
            status=HTTPStatus.OK,
            body=_chat_tool_response(route.model_id),
            latency_ms=1,
        )

    tools = [
        {
            "type": "namespace",
            "name": "medical",
            "description": "Exact candidate-scoped medical tools.",
            "tools": [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search admitted evidence.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
                {
                    "type": "function",
                    "name": "read",
                    "description": "Read one admitted record.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"record_id": {"type": "string"}},
                        "required": ["record_id"],
                    },
                },
            ],
        }
    ]
    schema_bytes = canonical_json_bytes(tools)
    schema_blake3 = blake3_hex(tools)
    token = "local-rollout-token"
    signer = AdapterReceiptSigner.ephemeral()
    with ResponsesAdapterGateway(
        {route_id: route},
        limits=AdapterLimits(max_concurrency=4, upstream_timeout_seconds=5),
        signer=signer,
        upstream_transport=upstream,
        local_bearer_token=token,
        local_credential_env_name=ROLLOUT_ADAPTER_TOKEN_ENV,
    ) as gateway:
        adapted = gateway.adapted_routes()[route_id]
        _overrides, _environment, private = adapted.config.for_subprocess()
        response = _post_json(
            private[0] + "/responses",
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
        assert len(captured) == 1
        assert captured[0]["parallel_tool_calls"] is True
        assert [row["function"]["name"] for row in captured[0]["tools"]] == [
            "medical__search",
            "medical__read",
        ]
        calls = [row for row in response["output"] if row["type"] == "function_call"]
        assert [(row["namespace"], row["name"]) for row in calls] == [
            ("medical", "search"),
            ("medical", "read"),
        ]
        assert response["parallel_tool_calls"] is True
        assert canonical_json_bytes(response["tools"]) == schema_bytes
        assert blake3_hex(response["tools"]) == schema_blake3
        assert canonical_json_bytes(tools) == schema_bytes
        assert blake3_hex(tools) == schema_blake3
        assert len(gateway.receipts) == 1
        verify_adapter_receipt(gateway.receipts[0])
        assert gateway.receipts[0].payload["route_id"] == route_id
        assert gateway.receipts[0].payload["request_shape"][
            "parallel_tool_calls_requested"
        ] is True
