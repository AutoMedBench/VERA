"""Reusable existing Codex/Responses-gateway setup for explicit localhost Qwen.

Entering this context starts only the local adapter, not an inference server or
provider request. The caller owns thread creation, context/resume, and receipts.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from eva_agent.codex_providers import AdapterLimits, ResponsesAdapterGateway
from eva_agent.codex_providers.adapter import ChatCompletionsTransport
from eva_agent.codex_providers.routes import CodexConfigOverrides, CodexProviderRoute, _PrivateText
from eva_agent.codex_runtime import OpenAICodexBackend
from eva_agent.deployment.campaign import CODEX_FIRST_RELEASE_CONFIG_OVERRIDES
from eva_agent.deployment.rollout_adapters import rollout_model_catalog_document
from eva_agent.pipeline.digests import blake3_bytes, canonical_json_bytes
from eva_agent.training.persistent_teacher import _thread_config
from eva_agent.training.teacher_batch import persist_teacher_adapter_receipt
from eva_agent.training.teacher_launch import isolated_teacher_launch_options


@dataclass(frozen=True)
class LocalQwenSetup:
    launch: Any
    provider: str
    model: str
    thread_config: dict
    safe_metadata: dict

    def backend(self):
        """New app-server instance with the same durable isolated Codex home."""
        return OpenAICodexBackend(self.launch)


class ThinkingChatTransport:
    """Explicit Qwen thinking at the local provider boundary, without logging CoT.

    These diagnostic sidecars supplement, not replace, canonical adapter receipts.
    No prompts, credentials, model output, or private reasoning are persisted here.
    """

    def __init__(self, transport, audit_root: Path):
        self.transport = transport
        self.audit_root = audit_root

    def __call__(self, binding, body):
        controls = {"enable_thinking": True}
        forwarded = {**body, "chat_template_kwargs": {
            **body.get("chat_template_kwargs", {}), **controls}}
        attempt = self.audit_root / str(uuid4())
        attempt.mkdir(parents=True, mode=0o700)

        def record(name, document):
            with (attempt / name).open("x") as stream:
                json.dump(document, stream, sort_keys=True)

        record("request.json", {"schema": "eva.local-qwen-thinking-request.v1",
            "thinking_requested": True, "chat_template_kwargs": controls,
            "max_tokens": forwarded.get("max_tokens"), "automatic_retry": False,
            "raw_input_recorded": False})
        try:
            outcome = self.transport(binding, forwarded)
        except Exception as exc:
            record("outcome.json", {"status": "transport_error", "error_type": type(exc).__name__,
                                    "raw_output_recorded": False})
            raise
        observed = {"status": "returned", "http_status": outcome.status,
                    "reasoning_tokens": None, "reasoning_field_nonempty": False,
                    "raw_output_recorded": False, "private_reasoning_recorded": False}
        try:
            payload = json.loads(outcome.body)
            details = payload.get("usage", {}).get("completion_tokens_details") or {}
            count = details.get("reasoning_tokens")
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                observed["reasoning_tokens"] = count
            choices = payload.get("choices") or []
            observed["reasoning_field_nonempty"] = any(
                bool((choice.get("message") or {}).get("reasoning_content"))
                for choice in choices if isinstance(choice, dict))
        except (ValueError, TypeError, AttributeError):
            observed["response_metadata_readable"] = False
        record("outcome.json", observed)
        return outcome


@contextmanager
def local_qwen_setup(*, run_root: Path, model: str = "Qwen/Qwen3.5-9B",
                     endpoint: str = "http://127.0.0.1:30910/v1", context_length: int = 32768,
                     max_output_tokens: int = 4096, image_inputs: bool = False, workers: int = 3,
                     thinking: bool = False, exact_tool_schemas: bool = False,
                     normalize_priority_messages: bool = False,
                     codex_bin: str | Path | None = None,
                     auto_compact_token_limit: int | None = None,
                     upstream_transport=None, token_budget: bool = False,
                     upstream_timeout_seconds: float = 300.0,
                     capacity_wait_seconds: float = 0.0):
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port
            or parsed.path != "/v1" or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("local Qwen setup requires an explicit unauthenticated loopback /v1 route")
    if not 4096 <= context_length <= 131072 or not 1 <= max_output_tokens <= 32768 or not 1 <= workers <= 4:
        raise ValueError("local Qwen bounds differ")
    if not isinstance(thinking, bool):
        raise ValueError("thinking must be a boolean")
    if type(normalize_priority_messages) is not bool:
        raise ValueError("normalize_priority_messages must be a boolean")
    if not isinstance(token_budget, bool):
        raise ValueError("token_budget must be a boolean")
    if (type(upstream_timeout_seconds) not in (int, float)
            or not 1 <= upstream_timeout_seconds <= 900):
        raise ValueError("local Chat upstream timeout must be within 1..900 seconds")
    if token_budget and upstream_transport is not None:
        raise ValueError("Chat token budgeting cannot wrap a custom training transport")
    if auto_compact_token_limit is not None and (
            type(auto_compact_token_limit) is not int
            or not 1024 <= auto_compact_token_limit < context_length - max_output_tokens):
        raise ValueError("compaction threshold must leave space for a complete output")
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    source = CodexProviderRoute(route_id="qwen_3_5_9b", model_id=model,
        model_env_name="MODEL_QWEN_3_5_9B", registry_role_id=None, provider_family="qwen",
        config=CodexConfigOverrides(provider_id="eva_local_qwen", model_id=model,
            credential_env_name="OPENAI_API_KEY", endpoint_env_name="OPENAI_BASE_URL",
            _credential=_PrivateText("local-nonsecret"), _endpoint=_PrivateText(endpoint)))
    limits = AdapterLimits(max_concurrency=workers,
                           upstream_timeout_seconds=float(upstream_timeout_seconds),
                           capacity_wait_seconds=capacity_wait_seconds)
    capacity_root = root / "capacity-events"
    capacity_sink = None
    if limits.capacity_wait_seconds > 0:
        capacity_root.mkdir(mode=0o700)

        def persist_capacity_event(event):
            target = capacity_root / (event["event_id"] + ".json")
            with target.open("xb") as stream:
                stream.write(canonical_json_bytes(dict(event)))
            target.chmod(0o600)

        capacity_sink = persist_capacity_event
    transport = upstream_transport
    if token_budget:
        from .token_budget import TokenBudgetChatTransport

        transport = TokenBudgetChatTransport(ChatCompletionsTransport(limits),
            endpoint=endpoint, context_length=context_length, audit_root=root / "token-budget")
    if thinking:
        transport = ThinkingChatTransport(
            transport if transport is not None else ChatCompletionsTransport(limits),
            root / "thinking-controls")
    with ResponsesAdapterGateway({source.route_id: source}, limits=limits,
            upstream_transport=transport,
            preserve_qwen_tool_schemas=exact_tool_schemas,
            normalize_qwen_priority_messages=normalize_priority_messages,
            local_credential_env_name="OPENAI_API_KEY",
            upstream_max_tokens_override=max_output_tokens,
            receipt_sink=lambda receipt: persist_teacher_adapter_receipt(root, receipt),
            capacity_event_sink=capacity_sink) as gateway:
        live = gateway.adapted_routes()[source.route_id]
        catalog = rollout_model_catalog_document({live.route_id: live}, route_order=(live.route_id,))
        for entry in catalog["models"]:
            entry["context_window"] = entry["max_context_window"] = context_length
            entry["input_modalities"] = ["text", "image"] if image_inputs else ["text"]
        catalog_path = root / "local-qwen-model-catalog.json"
        payload = canonical_json_bytes(catalog)
        with catalog_path.open("xb") as stream:
            stream.write(payload)
        catalog_path.chmod(0o400)
        _, child_env, _ = live.config.for_subprocess()
        launch = isolated_teacher_launch_options(codex_bin=codex_bin or shutil.which("codex"), cwd=root,
            isolation_root=root / "codex-isolation", child_env=child_env,
            config_overrides=(*CODEX_FIRST_RELEASE_CONFIG_OVERRIDES,
                              f"model_catalog_json={json.dumps(str(catalog_path))}"))
        config = {**_thread_config(live), "model_context_window": context_length}
        if auto_compact_token_limit is not None:
            config["model_auto_compact_token_limit"] = auto_compact_token_limit
        yield LocalQwenSetup(launch, live.config.provider_id, model, config,
            {"provider_route": live.route_id, "model_alias": model, "endpoint": endpoint,
             "context_length": context_length, "max_output_tokens": max_output_tokens,
             "upstream_timeout_seconds": float(upstream_timeout_seconds) if upstream_transport is None else None,
             "upstream_timeout_scope": "local-chat-http-transport" if upstream_transport is None else "custom-transport-owned",
             "adapter_capacity_wait_seconds": float(limits.capacity_wait_seconds),
             "adapter_capacity_events_retained": limits.capacity_wait_seconds > 0,
             "adapter_capacity_event_directory": (
                 "capacity-events" if limits.capacity_wait_seconds > 0 else None),
             "adapter_capacity_event_upstream_claim": False,
             "auto_compact_token_limit": auto_compact_token_limit,
             "compaction_observed_by_this_setup": False,
             "thinking_explicitly_enabled": thinking,
             "text_token_budget_enabled": token_budget,
             "multimodal_token_count_verified_by_this_setup": False,
             "thinking_observed_by_this_setup": False,
             "exact_tool_schemas": exact_tool_schemas,
             **({"priority_message_normalization": "one-leading-block-preserve-priority-order-no-user-tool-promotion"}
                if normalize_priority_messages else {}),
             "adapter_projection_version": gateway.safe_metadata["projection_version"],
             "catalog_blake3": blake3_bytes(payload), "image_inputs_declared": image_inputs,
             "image_provider_canary_verified_by_this_setup": False,
             "checkpoint_identity_must_be_bound_by_server_receipt": True,
             "provider_calls_on_setup": 0, "inference_server_launched": False})
