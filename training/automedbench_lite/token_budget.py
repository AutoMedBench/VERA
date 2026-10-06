"""Opt-in exact text-token headroom for the existing local Chat transport.

SGLang /v1/tokenize uses the same Chat template/message/tool processing as Chat
generation. It does NOT include multimodal processor placeholder expansion. Such
requests pass through unchanged with an explicit unverified-count receipt. This
wrapper neither truncates input nor retries generation, and retains only counts,
digests and fixed error categories. It does not prove a compaction was successful.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from eva_agent.codex_providers.adapter import ResponsesAdapterError
from eva_agent.pipeline.digests import blake3_hex


_RENDER_FIELDS = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls",
                  "reasoning_effort", "continue_final_message", "chat_template_kwargs", "response_format"}
_GENERATION_FIELDS = {"max_tokens", "max_completion_tokens", "temperature", "top_p",
                      "stream", "stream_options", "n", "logprobs", "top_logprobs"}
_SAFE_FAILURES = {"unsupported_rendering_controls", "explicit_output_budget_required",
                  "local_token_budget_messages_invalid", "local_token_budget_message_invalid",
                  "tokenize_response_too_large", "tokenize_count_invalid", "tokenize_response_invalid",
                  "text_prompt_exhausts_context"}


def http_error_category(status: int, body: bytes) -> str:
    """Classify fixed server validation phrases without retaining any error text."""
    if status != 400:
        return "http_rejection_other"
    try:
        value = json.loads(body)
        error = value.get("error", value) if isinstance(value, dict) else {}
        message = error.get("message", "") if isinstance(error, dict) else ""
        if not isinstance(message, str):
            return "unclassified_http_400"
    except (ValueError, UnicodeDecodeError):
        return "unclassified_http_400"
    lower = message.lower()
    patterns = (
        ("context_length_exceeded", ("requested token count exceeds the model's maximum context length",
                                      "is longer than the model's context length")),
        ("completion_budget_exceeds_context", ("max_completion_tokens is too large",)),
        ("missing_tools_for_choice", ("tools cannot be empty if tool choice",)),
        ("invalid_tool_schema", ("function has invalid 'parameters' schema", "schema_ is required for json_schema")),
        ("invalid_tool_history_arguments", ("assistant tool call function.arguments must be valid json",
                                            "assistant tool call function.arguments must be a json object")),
        ("chat_template_invalid", ("no user query found in messages", "system message must be at the beginning",
                                   "unexpected message role", "chat template tokenization requires a template manager")),
    )
    return next((name for name, phrases in patterns if any(p in lower for p in phrases)), "unclassified_http_400")


def _text_only(messages) -> bool:
    if not isinstance(messages, list) or not messages:
        raise ResponsesAdapterError("local_token_budget_messages_invalid")
    for message in messages:
        if not isinstance(message, dict):
            raise ResponsesAdapterError("local_token_budget_message_invalid")
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list) or any(
            not isinstance(part, dict) or part.get("type") != "text"
            or not isinstance(part.get("text"), str) for part in content
        ):
            return False
    return True


def _write(path: Path, document: dict):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(document, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


class TokenBudgetChatTransport:
    """Wrap INSIDE ThinkingChatTransport so tokenization sees final controls.

    ``endpoint`` must be the same explicit loopback /v1 endpoint bound to the
    underlying Chat transport. ``context_length`` comes from the independently
    verified serving configuration, not tokenizer.model_max_length. No credentials
    or proxy environment are used for tokenization. Constructor makes no request.
    """

    def __init__(self, transport, *, endpoint: str, context_length: int, audit_root: Path,
                 margin: int = 256, timeout_seconds: float = 30, tokenize_http_transport=None):
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port
                or parsed.path != "/v1" or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ResponsesAdapterError("token_budget_requires_explicit_loopback_v1")
        if (type(context_length) is not int or not 4096 <= context_length <= 131072
                or type(margin) is not int or not 0 <= margin < context_length
                or not 0 < timeout_seconds <= 120):
            raise ResponsesAdapterError("token_budget_bounds_invalid")
        self.transport, self.endpoint = transport, endpoint
        self.context_length, self.margin = context_length, margin
        self.timeout_seconds, self.http_transport = timeout_seconds, tokenize_http_transport
        self.audit_root = Path(audit_root)
        self.audit_root.mkdir(parents=True, mode=0o700, exist_ok=False)

    def __call__(self, binding, body):
        import httpx

        attempt = self.audit_root / str(uuid4())
        attempt.mkdir(mode=0o700)
        forwarded = deepcopy(body)
        record = {"schema": "eva.local-qwen-token-budget.v1", "context_length": self.context_length,
                  "margin_tokens": self.margin, "prompt_tokens": None, "token_count_exact": False,
                  "requested_max_tokens": body.get("max_tokens"), "effective_max_tokens": None,
                  "input_truncated": False, "generation_attempts": 0, "tokenization_attempts": 0,
                  "automatic_retry": False, "raw_input_recorded": False, "raw_output_recorded": False,
                  "token_ids_recorded": False, "request_blake3": blake3_hex(body)}
        phase = "input_validation"
        try:
            if set(body) - _RENDER_FIELDS - _GENERATION_FIELDS:
                raise ResponsesAdapterError("unsupported_rendering_controls")
            if any(value is not None and (type(value) is not int or value < 1)
                   for value in (body.get("max_tokens"), body.get("max_completion_tokens"))):
                raise ResponsesAdapterError("explicit_output_budget_required")
            requested = body.get("max_completion_tokens") or body.get("max_tokens")
            if type(requested) is not int or requested < 1:
                raise ResponsesAdapterError("explicit_output_budget_required")
            record["requested_max_tokens"] = requested
            text_only = _text_only(body.get("messages"))
            record["text_only"] = text_only
            if text_only:
                phase = "tokenization"
                payload = {key: deepcopy(value) for key, value in body.items() if key in _RENDER_FIELDS}
                record["tokenize_request_blake3"] = blake3_hex(payload)
                record["tokenization_attempts"] = 1
                with httpx.Client(timeout=self.timeout_seconds, trust_env=False, follow_redirects=False,
                                  transport=self.http_transport) as client:
                    with client.stream("POST", self.endpoint + "/tokenize", json=payload) as response:
                        chunks, size = [], 0
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > 2 * 1024 * 1024:
                                raise ResponsesAdapterError("tokenize_response_too_large")
                            chunks.append(chunk)
                        raw = b"".join(chunks)
                        record["tokenize_http_status"] = response.status_code
                if response.status_code != 200:
                    record["http_error_category"] = http_error_category(response.status_code, raw)
                    raise ResponsesAdapterError("tokenization_http_rejected")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ResponsesAdapterError("tokenize_response_invalid")
                count, tokens = value.get("count"), value.get("tokens")
                if (type(count) is not int or count < 1 or not isinstance(tokens, list)
                        or len(tokens) != count or any(type(t) is not int or t < 0 for t in tokens)):
                    raise ResponsesAdapterError("tokenize_count_invalid")
                record.update(prompt_tokens=count, token_count_exact=True,
                              count_source="same_endpoint_sglang_chat_template")
                remaining = self.context_length - count - self.margin
                record["remaining_output_slots"] = max(0, remaining)
                if remaining < 1:
                    record["failure_category"] = "text_prompt_exhausts_context"
                    raise ResponsesAdapterError("text_prompt_exhausts_context")
                effective = min(requested, remaining)
                field = "max_completion_tokens" if body.get("max_completion_tokens") is not None else "max_tokens"
                forwarded[field] = effective
                # If both aliases occur, bound both; SGLang prefers completion.
                if forwarded.get("max_tokens") is not None:
                    forwarded["max_tokens"] = min(forwarded["max_tokens"], effective)
                record.update(effective_max_tokens=effective, output_budget_clamped=effective < requested)
            else:
                record.update(effective_max_tokens=requested, output_budget_clamped=False,
                              count_source="multimodal_count_unverified", multimodal_forwarded_unchanged=True)
            phase = "generation"
            record["generation_attempts"] = 1
            outcome = self.transport(binding, forwarded)
            record["generation_http_status"] = outcome.status
            record["status"] = "returned" if outcome.status == 200 else "http_rejected"
            if outcome.status != 200:
                record["http_error_category"] = http_error_category(outcome.status, outcome.body)
            return outcome
        except Exception as exc:
            # Only our fixed codes or the allowlisted HTTP category may escape;
            # unknown transport/parser exceptions can contain private prompts.
            own_code = str(exc) if isinstance(exc, ResponsesAdapterError) else None
            category = (record.get("http_error_category") or record.get("failure_category")
                        or (own_code if own_code in _SAFE_FAILURES else phase + "_transport_error"))
            record.update(status="failed", failure_phase=phase, error_type=type(exc).__name__)
            record["failure_category"] = category
            raise ResponsesAdapterError("local_token_budget_failed: " + category) from None
        finally:
            _write(attempt / "outcome.json", record)
