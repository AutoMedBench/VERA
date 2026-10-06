"""Codex Responses route with a metered upstream and explicit Qwen controls."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from threading import Lock
import time
from uuid import uuid4


class ContextGuard:
    def __init__(self, transport, config, audit):
        self.transport, self.config, self.audit = transport, config, audit
        audit.mkdir(parents=True, mode=0o700)

    def __call__(self, binding, body):
        import httpx
        from training.automedbench_lite.adapter import write_once
        from training.automedbench_lite.token_budget import _RENDER_FIELDS, _text_only

        row = {"schema": "eva.benchmark-context-guard.v1", "context_length": self.config["context_length"],
               "max_output_tokens": body["max_tokens"], "input_truncated": False,
               "text_token_count_exact": False, "multimodal_expansion_counted": False}
        if _text_only(body.get("messages")):
            payload = {key: value for key, value in body.items() if key in _RENDER_FIELDS}
            with httpx.Client(timeout=30, trust_env=False, follow_redirects=False) as client:
                response = client.post(self.config["endpoint"] + "/tokenize", json=payload)
            if response.status_code != 200 or len(response.content) > 4 * 1024**2:
                raise RuntimeError("local_chat_tokenization_failed")
            value = response.json()
            count, tokens = value.get("count"), value.get("tokens")
            if type(count) is not int or not isinstance(tokens, list) or count != len(tokens):
                raise RuntimeError("local_chat_tokenization_count_invalid")
            row.update(prompt_tokens=count, text_token_count_exact=True)
            if count + body["max_tokens"] + 256 > self.config["context_length"]:
                write_once(self.audit / (str(uuid4()) + ".json"), {**row, "admitted": False})
                raise RuntimeError("full_output_reserve_does_not_fit_context")
        write_once(self.audit / (str(uuid4()) + ".json"), {**row, "admitted": True})
        return self.transport(binding, body)


class BudgetTransport:
    """Every actual model request, including compaction, consumes the same cap."""
    def __init__(self, transport, config: dict, audit: Path):
        self.transport, self.config, self.audit = transport, config, audit
        self.requests = 0
        self.started = time.monotonic()
        self.lock = Lock()
        self.exhausted = None
        self.rows = []
        audit.mkdir(parents=True, exist_ok=True, mode=0o700)

    def __call__(self, binding, body):
        from training.automedbench_lite.adapter import write_once

        with self.lock:
            expired = time.monotonic() >= self.config.get("track_deadline_monotonic", self.started + self.config["max_seconds"])
            if expired:
                self.exhausted = "wall_clock"
            elif self.requests >= self.config["max_turns"]:
                self.exhausted = "model_requests"
            if self.exhausted:
                path = self.audit / "budget-exhaustion.json"
                if not path.exists():
                    write_once(path, {"schema": "eva.codex-benchmark-provider-denial.v1",
                        "reason": self.exhausted, "actual_upstream_requests": self.requests,
                        "denied_request_number": self.requests + 1,
                        "max_model_requests": self.config["max_turns"],
                        "max_seconds": self.config["max_seconds"],
                        "observed_monotonic": time.monotonic(), "upstream_request_sent": False})
                raise RuntimeError("declared_" + ("model_request" if self.exhausted == "model_requests" else "wall_clock") + "_budget_exhausted")
            self.requests += 1
            number = self.requests
        forwarded = dict(body)
        thinking = self.config["mode"] == "think"
        forwarded["max_tokens"] = self.config["max_output_tokens"]
        forwarded["chat_template_kwargs"] = {**forwarded.get("chat_template_kwargs", {}),
                                               "enable_thinking": thinking}
        processor = self.config.get("thinking_logit_processor")
        if thinking and processor:
            forwarded["custom_logit_processor"] = processor
            forwarded["custom_params"] = {**forwarded.get("custom_params", {}),
                                            "thinking_budget": self.config["thinking_budget_tokens"] - 1}
        row = {"schema": "eva.codex-benchmark-provider-request.v1", "request_id": str(uuid4()),
               "request_number": number, "model": body.get("model"), "mode": self.config["mode"],
               "max_output_tokens": forwarded["max_tokens"], "thinking_enabled_requested": thinking,
               "manual_thinking_budget_requested": self.config["thinking_budget_tokens"] if thinking else 0,
               "manual_thinking_processor_supplied": bool(processor), "automatic_retry": False,
               "elapsed_before_request_seconds": time.monotonic() - self.started,
               "private_reasoning_recorded": False}
        try:
            outcome = self.transport(binding, forwarded)
            row["http_status"] = outcome.status
            try:
                data = json.loads(outcome.body)
                usage = data.get("usage") or {}
                row["usage"] = {key: value for key, value in usage.items()
                                if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                                and type(value) is int}
                details = usage.get("completion_tokens_details") or {}
                row["reasoning_tokens"] = details.get("reasoning_tokens")
                row["returned_model"] = data.get("model")
            except (ValueError, AttributeError, TypeError):
                row["usage_unavailable"] = True
            return outcome
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            row["elapsed_after_request_seconds"] = time.monotonic() - self.started
            write_once(self.audit / (row["request_id"] + ".json"), row)
            with self.lock:
                self.rows.append(row)


@contextmanager
def setup_provider(config: dict, run_root: Path, token: str, *, upstream_transport=None, upstream_wrapper=None):
    from eva_agent.codex_providers import AdapterLimits, ResponsesAdapterGateway
    from eva_agent.codex_providers.adapter import ChatCompletionsTransport
    from eva_agent.codex_providers.routes import CodexConfigOverrides, CodexProviderRoute, _PrivateText
    from eva_agent.codex_runtime import OpenAICodexBackend
    from eva_agent.codex_runtime.research_profile import profile_overrides
    from eva_agent.deployment.campaign import CODEX_FIRST_RELEASE_CONFIG_OVERRIDES
    from eva_agent.deployment.rollout_adapters import rollout_model_catalog_document
    from eva_agent.training.persistent_teacher import _thread_config
    from eva_agent.training.teacher_batch import persist_teacher_adapter_receipt
    from eva_agent.training.teacher_launch import isolated_teacher_launch_options
    from training.automedbench_lite.adapter import write_once

    run_root.mkdir(parents=True, mode=0o700)
    route = CodexProviderRoute(route_id="evamed_benchmark_qwen", model_id=config["model"],
        model_env_name="EVAMED_BENCHMARK_MODEL", registry_role_id=None, provider_family="qwen",
        config=CodexConfigOverrides(provider_id="evamed_benchmark_qwen", model_id=config["model"],
            credential_env_name="OPENAI_API_KEY", endpoint_env_name="OPENAI_BASE_URL",
            _credential=_PrivateText(token), _endpoint=_PrivateText(config["endpoint"])))
    limits = AdapterLimits(max_concurrency=1, max_request_bytes=32 * 1024**2,
                           upstream_timeout_seconds=min(600, config["max_seconds"]))
    upstream = upstream_transport if upstream_transport is not None else ChatCompletionsTransport(limits)
    if upstream_transport is None and config["route"] == "local":
        upstream = ContextGuard(upstream, config, run_root / "token-budget")
    if upstream_wrapper is not None:
        upstream = upstream_wrapper(upstream)
    meter = BudgetTransport(upstream, config, run_root / "requests")
    with ResponsesAdapterGateway({route.route_id: route}, limits=limits, upstream_transport=meter,
            preserve_qwen_tool_schemas=True, normalize_qwen_priority_messages=True,
            local_credential_env_name="OPENAI_API_KEY", upstream_max_tokens_override=config["max_output_tokens"],
            receipt_sink=lambda receipt: persist_teacher_adapter_receipt(run_root, receipt)) as gateway:
        live = gateway.adapted_routes()[route.route_id]
        catalog = rollout_model_catalog_document({live.route_id: live}, route_order=(live.route_id,))
        for entry in catalog["models"]:
            entry["context_window"] = entry["max_context_window"] = config["context_length"]
            entry["input_modalities"] = ["text", "image"]
        catalog_path = run_root / "model-catalog.json"
        catalog_path.write_text(json.dumps(catalog))
        catalog_path.chmod(0o400)
        _, environment, _ = live.config.for_subprocess()
        overrides = (*profile_overrides(), *CODEX_FIRST_RELEASE_CONFIG_OVERRIDES,
                     f"model_catalog_json={json.dumps(str(catalog_path))}")
        launch = isolated_teacher_launch_options(codex_bin=config["codex_bin"], cwd=run_root,
            isolation_root=run_root / "codex-isolation", child_env=environment,
            config_overrides=overrides)
        thread = {**_thread_config(live), "model_context_window": config["context_length"],
                  "model_auto_compact_token_limit": config["compact_at_tokens"],
                  "model_reasoning_effort": "xhigh" if config["mode"] == "think" else "none"}
        write_once(run_root / "binding.json", {"schema": "eva.codex-benchmark-provider-binding.v1",
            "model": config["model"], "route": config["route"], "context_tokens_requested": config["context_length"],
            "max_model_requests": config["max_turns"], "max_seconds": config["max_seconds"],
            "max_output_tokens": config["max_output_tokens"], "mode": config["mode"],
            "canonical_schemas_preserved": True, "raw_private_reasoning_retained": False})
        yield OpenAICodexBackend(launch), live.config.provider_id, thread, meter
