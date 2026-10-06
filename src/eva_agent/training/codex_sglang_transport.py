"""Exact-token, text-only Qwen transport for native Codex on-policy rollouts.

One Chat request becomes one independent /generate request and one segment. The
caller constructs Slime Samples: prompt IDs are conditioning only; output IDs and
their returned logprobs are the only supervised suffix. Never join segments or
re-encode sampled text, even when Codex sends overlapping or compacted history.

Private segments may contain reasoning, medical observations, and token IDs. Do
not include them in public adapter receipts. The optional evidence directory is
exclusive and private (0700; files 0600). No credential from the adapter binding
is read. The echoed Chat model is a routing alias, NOT a returned model assertion.
SGLang/checkpoint identity must be independently bound by the integration layer.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

from eva_agent.codex_providers.adapter import ResponsesAdapterError, _UpstreamOutcome
from eva_agent.pipeline.digests import blake3_hex


_SAFE_FAILURE_CATEGORIES = frozenset({
    "completion_parser",
    "context_budget",
    "owned_output_budget",
    "owned_request_budget",
    "prompt_parser",
    "provider_http",
    "provider_transport",
    "request_control",
    "response_parser",
    "sampling_control",
    "token_alignment",
    "unexpected",
})


class _DiagnosedTransportError(ResponsesAdapterError):
    """Fixed-category failure carrying no raw request or response material."""

    def __init__(self, message: str, *, category: str, prompt_tokens: int | None = None):
        super().__init__(message)
        if category not in _SAFE_FAILURE_CATEGORIES:
            raise ValueError("unknown safe transport failure category")
        self.category = category
        self.prompt_tokens = prompt_tokens


def _diagnosed(message: str, *, category: str, prompt_tokens: int | None = None):
    return _DiagnosedTransportError(
        message,
        category=category,
        prompt_tokens=prompt_tokens,
    )


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ResponsesAdapterError(f"{label} must be a positive integer")
    return value


def _messages(body: Mapping[str, Any]) -> list[dict]:
    messages = deepcopy(body.get("messages"))
    if not isinstance(messages, list) or not messages:
        raise ResponsesAdapterError("Chat messages must be a nonempty array")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system", "user", "assistant", "tool"
        }:
            raise ResponsesAdapterError("unsupported text Chat message")
        content = message.get("content")
        if isinstance(content, list):
            if any(not isinstance(p, dict) or p.get("type") != "text"
                   or not isinstance(p.get("text"), str) for p in content):
                raise ResponsesAdapterError("multimodal content needs a separate exact-token transport")
            # Match SGLang's normalize_tool_content; Qwen renders other parts.
            if message["role"] == "tool":
                message["content"] = " ".join(p["text"] for p in content)
        elif content is None:
            message["content"] = ""
        elif not isinstance(content, str):
            raise ResponsesAdapterError("unsupported Chat content")
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                try:
                    function = call["function"]
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError
                    function["arguments"] = arguments
                except (KeyError, TypeError, ValueError) as exc:
                    raise ResponsesAdapterError("historical tool arguments must be a JSON object") from exc
    return messages


def _project_completion(text: str, tools: list[dict], thinking: bool, tokenizer: Any) -> dict:
    # Lazy imports: construction and mocked CPU tests need no SGLang/torch import.
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    reasoning, visible = ReasoningParser(model_type="qwen3", stream_reasoning=False,
        force_reasoning=thinking, tokenizer=tokenizer).parse_non_stream(text)
    for ending in ("<|im_end|>", "<|endoftext|>"):
        visible = visible.rstrip().removesuffix(ending)
    calls = []
    if tools:
        parser = FunctionCallParser(tools=[Tool.model_validate(t) for t in tools],
                                    tool_call_parser="qwen3_coder", tokenizer=tokenizer)
        # Match installed serving_chat: malformed model syntax remains text.
        try:
            visible, parsed = parser.parse_non_stream(visible)
        except (ValueError, TypeError, KeyError):
            parsed = []
        for call in parsed:
            calls.append({"id": "call_" + uuid4().hex, "type": "function",
                          "function": {"name": call.name, "arguments": call.parameters}})
    message = {"role": "assistant", "content": visible}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return message


class CodexSGLangTransport:
    """Synchronous, single-attempt ChatCompletionsTransport-compatible callable.

    ``segments`` returns defensive copies of PRIVATE token records. A trajectory
    must fail closed if any request fails; successful earlier segments are audit
    evidence, not permission to train an unjudged partial trajectory. Total output
    and request limits include Codex compaction/summarization provider requests.
    Forced tool choice and non-text inputs are explicitly unsupported, not silently
    weakened. The caller owns cancellation and a separate overall wall-clock limit.
    """

    def __init__(self, *, tokenizer: Any, generate_url: str, sampling_params: Mapping[str, Any],
                 max_context_tokens: int, max_output_tokens: int, total_output_budget: int,
                 max_requests: int, evidence_root: Path | None = None,
                 timeout_seconds: float = 300, http_transport: Any = None,
                 argument_projector: Any = None, custom_logit_processor: str | None = None,
                 custom_params: Mapping[str, Any] | None = None,
                 policy_loss_masker=None):
        url = urlsplit(generate_url)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username
                or url.password or url.query or url.fragment or url.path != "/generate"):
            raise ResponsesAdapterError("expected an unauthenticated explicit /generate URL")
        if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 900:
            raise ResponsesAdapterError("request timeout differs")
        self.tokenizer = tokenizer
        self.generate_url = generate_url
        self.sampling_params = deepcopy(dict(sampling_params))
        if self.sampling_params.get("n", 1) != 1:
            raise ResponsesAdapterError("exact-token transport requires one sampled choice")
        self.max_context_tokens = _positive_int(max_context_tokens, "context limit")
        self.max_output_tokens = _positive_int(max_output_tokens, "output limit")
        self.total_output_budget = _positive_int(total_output_budget, "total output limit")
        self.max_requests = _positive_int(max_requests, "request limit")
        self.timeout_seconds, self.http_transport = timeout_seconds, http_transport
        self._lock = threading.Lock()
        self._segments: list[dict] = []
        self._requests = self._generated = 0
        self._failed = False
        self._budget_termination = None
        self.argument_projector = argument_projector
        if custom_logit_processor is not None and (not isinstance(custom_logit_processor, str) or not custom_logit_processor):
            raise ResponsesAdapterError("invalid explicit custom logit processor")
        if custom_params and custom_logit_processor is None:
            raise ResponsesAdapterError("custom sampling parameters require an explicit processor")
        self.custom_logit_processor = custom_logit_processor
        self.custom_params = deepcopy(dict(custom_params or {}))
        if policy_loss_masker is not None and custom_logit_processor is None:
            raise ResponsesAdapterError("policy masking requires a bound custom processor")
        self.policy_loss_masker = policy_loss_masker
        self.evidence_root = Path(evidence_root) if evidence_root is not None else None
        if self.evidence_root is not None:
            self.evidence_root.mkdir(parents=True, mode=0o700, exist_ok=False)
            # Shared workspace parents can propagate setgid onto new folders.
            self.evidence_root.chmod(0o700)
        if self.argument_projector is not None:
            self._write("host-argument-type-binding.json", self.argument_projector.binding)
        if self.policy_loss_masker is not None:
            self._write("policy-loss-mask-binding.json", self.policy_loss_masker.binding)

    @property
    def budget_termination(self) -> dict | None:
        """Only an exhausted owned policy budget, never HTTP/parser/context errors."""
        with self._lock:
            return deepcopy(self._budget_termination)

    @property
    def segments(self) -> list[dict]:
        with self._lock:
            return deepcopy(self._segments)

    @property
    def safe_metadata(self) -> dict:
        with self._lock:
            return {"schema": "eva.codex-sglang-token-capture.v1", "requests": self._requests,
                    "segments": len(self._segments), "generated_tokens": self._generated,
                    "failed": self._failed, "returned_model": None,
                    "routing_alias_is_not_returned_model": True,
                    "sampled_completions_retokenized": False,
                    "segments_concatenated": False, "automatic_retry": False,
                    "private_token_evidence": self.evidence_root is not None}

    def _write(self, name: str, value: dict) -> None:
        if self.evidence_root is None:
            return
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        # Exclusive private evidence; a crash can leave a partial file, never a
        # forged complete receipt. Consumers must parse and verify the record.
        fd = os.open(self.evidence_root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def __call__(self, binding: Any, body: Mapping[str, Any]) -> _UpstreamOutcome:
        del binding  # Do not reveal upstream credentials or inherit proxy/auth env.
        with self._lock:
            if self._failed:
                raise ResponsesAdapterError("trajectory transport already failed; no retry")
            try:
                return self._generate(body)
            except Exception as exc:
                self._failed = True
                failure_id = str(uuid4())
                self._write(f"failure-{failure_id}.json", {
                    "schema": "eva.codex-sglang-capture-failure.v1",
                    "requests": self._requests, "segments": len(self._segments),
                    "error_type": (
                        "ResponsesAdapterError"
                        if isinstance(exc, _DiagnosedTransportError)
                        else type(exc).__name__
                    ),
                    "automatic_retry": False})
                category = (
                    exc.category
                    if isinstance(exc, _DiagnosedTransportError)
                    else "unexpected"
                )
                prompt_tokens = (
                    exc.prompt_tokens
                    if isinstance(exc, _DiagnosedTransportError)
                    else None
                )
                self._write(f"safe-failure-diagnostic-{failure_id}.json", {
                    "schema": "eva.codex-sglang-safe-failure-diagnostic.v1",
                    "category": category,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": self._generated,
                    "requests": self._requests,
                    "segments": len(self._segments),
                    "max_context_tokens": self.max_context_tokens,
                    "max_output_tokens": self.max_output_tokens,
                    "total_output_budget": self.total_output_budget,
                    "automatic_retry": False,
                    "raw_text_recorded": False,
                    "token_ids_recorded": False,
                    "request_body_recorded": False,
                })
                # Raw HTTP/parser exception text can contain private output.
                raise ResponsesAdapterError("exact-token SGLang request failed; private audit retained") from None

    def _generate(self, body: Mapping[str, Any]) -> _UpstreamOutcome:
        import httpx

        if self._requests >= self.max_requests:
            self._budget_termination = {"kind": "max_requests", "count": self._requests,
                                       "limit": self.max_requests, "generated_tokens": self._generated}
            raise _diagnosed(
                "Codex provider request budget exhausted",
                category="owned_request_budget",
            )
        allowed = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls",
                   "max_tokens", "temperature", "top_p", "chat_template_kwargs", "stream", "n",
                   "custom_logit_processor", "custom_params"}
        if set(body) - allowed or body.get("stream", False) is not False or body.get("n", 1) != 1:
            raise _diagnosed("unsupported Chat generation controls", category="request_control")
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise _diagnosed("requested model alias is missing", category="request_control")
        choice = body.get("tool_choice", "auto")
        if choice not in ("auto", "none", None):
            raise _diagnosed(
                "forced tool choice requires an explicit constraint implementation",
                category="request_control",
            )
        tools = deepcopy(body.get("tools") or [])
        if not isinstance(tools, list) or any(not isinstance(t, dict) or t.get("type") != "function"
                or not isinstance(t.get("function"), dict) for t in tools):
            raise _diagnosed("expected exact flat function tools", category="request_control")
        if choice == "none" and tools:
            raise _diagnosed(
                "tool suppression must be explicit upstream, not a schema mutation",
                category="request_control",
            )
        if self.argument_projector is not None:
            try:
                self.argument_projector.validate_tools(tools)
            except Exception:
                raise _diagnosed(
                    "tool argument projection differs",
                    category="request_control",
                ) from None
        kwargs = deepcopy(body.get("chat_template_kwargs") or {})
        if set(kwargs) - {"enable_thinking"} or type(kwargs.get("enable_thinking", True)) is not bool:
            raise _diagnosed("unsupported Qwen template controls", category="request_control")
        thinking = kwargs.get("enable_thinking", True)
        if (body.get("custom_logit_processor") != self.custom_logit_processor or
                dict(body.get("custom_params") or {}) != self.custom_params):
            raise _diagnosed("caller changed the bound thinking processor", category="sampling_control")
        if self.custom_logit_processor is not None and not thinking:
            raise _diagnosed("thinking processor enabled for an instant request", category="sampling_control")
        try:
            prompt = self.tokenizer.apply_chat_template(_messages(body), tools=tools or None,
                tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            prompt_ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        except Exception:
            raise _diagnosed("prompt projection failed", category="prompt_parser") from None
        if not prompt_ids or any(type(t) is not int or t < 0 for t in prompt_ids):
            raise _diagnosed("tokenizer returned invalid prompt IDs", category="prompt_parser")
        prompt_tokens = len(prompt_ids)
        try:
            requested = _positive_int(body.get("max_tokens", self.max_output_tokens), "Chat output limit")
        except ResponsesAdapterError:
            raise _diagnosed(
                "Chat output limit must be a positive integer",
                category="request_control",
                prompt_tokens=prompt_tokens,
            ) from None
        if len(prompt_ids) >= self.max_context_tokens:
            raise _diagnosed(
                "context budget exhausted; prompt never truncated",
                category="context_budget",
                prompt_tokens=prompt_tokens,
            )
        if self._generated >= self.total_output_budget:
            self._budget_termination = {"kind": "total_output_budget", "count": self._generated,
                                       "limit": self.total_output_budget, "requests": self._requests}
            raise _diagnosed(
                "Codex total output budget exhausted",
                category="owned_output_budget",
                prompt_tokens=prompt_tokens,
            )
        budget = min(requested, self.max_output_tokens, self.total_output_budget - self._generated,
                     self.max_context_tokens - len(prompt_ids),
                     self.sampling_params.get("max_new_tokens", self.max_output_tokens))
        if budget < 1:
            raise _diagnosed(
                "context or total output budget exhausted; prompt never truncated",
                category="context_budget",
                prompt_tokens=prompt_tokens,
            )
        params = deepcopy(self.sampling_params)
        # Slime owns sampling policy. A conflicting Codex control must not silently
        # alter the policy whose logprobs are later used for optimization.
        for name in ("temperature", "top_p"):
            if name in body and name in params and body[name] != params[name]:
                raise _diagnosed(
                    "Codex and rollout sampling controls disagree",
                    category="sampling_control",
                    prompt_tokens=prompt_tokens,
                )
            if name in body:
                params[name] = body[name]
        params.update(max_new_tokens=budget, skip_special_tokens=False)
        if self.custom_logit_processor is not None:
            # /generate consumes processor state from SamplingParams. The
            # Chat endpoint's top-level custom_params shape is not valid here.
            if 'custom_params' in params and params['custom_params'] != self.custom_params:
                raise _diagnosed('Slime and thinking processor parameters disagree', category='sampling_control')
            params['custom_params'] = deepcopy(self.custom_params)
        request_id = str(uuid4())
        payload = {"input_ids": prompt_ids, "sampling_params": params, "return_logprob": True,
                   "logprob_start_len": -1, "return_text_in_logprobs": False, "stream": False}
        if self.custom_logit_processor is not None:
            payload['custom_logit_processor'] = self.custom_logit_processor
        self._requests += 1
        self._write(f"request-{request_id}.json", {"schema": "eva.codex-sglang-token-request.v1",
                    "request_id": request_id, "chat_request_blake3": blake3_hex(body),
                    "tools_blake3": blake3_hex(tools), "requested_model": model, **payload})
        started = time.monotonic()
        try:
            with httpx.Client(timeout=self.timeout_seconds, trust_env=False,
                              follow_redirects=False, transport=self.http_transport) as client:
                response = client.post(self.generate_url, json=payload)
        except Exception:
            raise _diagnosed(
                "SGLang transport failed",
                category="provider_transport",
                prompt_tokens=prompt_tokens,
            ) from None
        if response.status_code != 200:
            self._write(f"http-failure-{request_id}.json", {"request_id": request_id,
                        "http_status": response.status_code, "response_body_retained": False})
            raise _diagnosed(
                "SGLang HTTP rejection",
                category="provider_http",
                prompt_tokens=prompt_tokens,
            )
        try:
            output = response.json()
        except Exception:
            raise _diagnosed(
                "SGLang response parsing failed",
                category="response_parser",
                prompt_tokens=prompt_tokens,
            ) from None
        if not isinstance(output, dict) or not isinstance(output.get("meta_info"), dict):
            raise _diagnosed(
                "SGLang returned no single-sample metadata",
                category="response_parser",
                prompt_tokens=prompt_tokens,
            )
        meta = output["meta_info"]
        rows = meta.get("output_token_logprobs")
        if (not isinstance(rows, list) or not rows or any(not isinstance(r, (list, tuple)) or len(r) < 2
                or type(r[1]) is not int or r[1] < 0 or type(r[0]) not in (int, float)
                or not math.isfinite(r[0]) or r[0] > 0 for r in rows)):
            raise _diagnosed(
                "missing or invalid actual sampled-token log probabilities",
                category="token_alignment",
                prompt_tokens=prompt_tokens,
            )
        output_ids, logprobs = [r[1] for r in rows], [float(r[0]) for r in rows]
        if len(output_ids) > budget or meta.get("prompt_tokens") != len(prompt_ids):
            raise _diagnosed(
                "SGLang token counts differ from request bounds",
                category="token_alignment",
                prompt_tokens=prompt_tokens,
            )
        if meta.get("completion_tokens") != len(output_ids):
            raise _diagnosed(
                "SGLang completion/logprob lengths differ",
                category="token_alignment",
                prompt_tokens=prompt_tokens,
            )
        if "output_ids" in output and output["output_ids"] != output_ids:
            raise _diagnosed(
                "SGLang output IDs differ from sampled-logprob IDs",
                category="token_alignment",
                prompt_tokens=prompt_tokens,
            )
        if ("output_token_logprobs_length" in meta
                and meta["output_token_logprobs_length"] != len(output_ids)):
            raise _diagnosed(
                "SGLang logprob count differs",
                category="token_alignment",
                prompt_tokens=prompt_tokens,
            )
        finish = meta.get("finish_reason")
        if not isinstance(finish, dict) or finish.get("type") not in {"stop", "length"}:
            raise _diagnosed(
                "SGLang generation did not finish successfully",
                category="response_parser",
                prompt_tokens=prompt_tokens,
            )
        text = output.get("text")
        if not isinstance(text, str):
            raise _diagnosed(
                "SGLang returned no decoded output",
                category="response_parser",
                prompt_tokens=prompt_tokens,
            )
        segment = {"schema": "eva.codex-sglang-token-segment.v1", "request_id": request_id,
                   "segment_index": len(self._segments), "requested_model": model,
                   "returned_model": output.get("model") if isinstance(output.get("model"), str) else None,
                   "chat_request_blake3": blake3_hex(body), "tools_blake3": blake3_hex(tools),
                   "prompt_token_ids": prompt_ids, "output_token_ids": output_ids,
                   "output_token_logprobs": logprobs, "output_text": text,
                   "finish_reason": deepcopy(finish), "meta_info": deepcopy(meta),
                   "sampling_params": params, "thinking_enabled": thinking,
                   "loss_mask": [1] * len(output_ids), "prompt_is_conditioning_only": True}
        if self.policy_loss_masker is not None:
            mask, provenance = self.policy_loss_masker(prompt_ids, output_ids)
            if len(mask) != len(output_ids) or any(type(bit) is not int or bit not in (0, 1) for bit in mask):
                raise _diagnosed("invalid bound policy loss mask", category="token_alignment")
            segment.update(loss_mask=mask, loss_mask_provenance=provenance)
        self._segments.append(segment)
        self._generated += len(output_ids)
        self._write(f"segment-{request_id}.json", segment)
        try:
            message = _project_completion(text, tools, thinking, self.tokenizer)
        except Exception:
            raise _diagnosed(
                "completion projection failed",
                category="completion_parser",
                prompt_tokens=prompt_tokens,
            ) from None
        if self.argument_projector is not None:
            try:
                message, projection = self.argument_projector.project_message(message, tools)
            except Exception:
                raise _diagnosed(
                    "completion argument projection failed",
                    category="completion_parser",
                    prompt_tokens=prompt_tokens,
                ) from None
            self._write(f"argument-projection-{request_id}.json", {**projection,
                "request_id": request_id, "raw_output_blake3": blake3_hex(text)})
        chat_finish = "length" if finish["type"] == "length" else (
            "tool_calls" if message.get("tool_calls") else "stop")
        chat = {"id": "chatcmpl_" + uuid4().hex, "object": "chat.completion",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": chat_finish}],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(output_ids),
                          "total_tokens": len(prompt_ids) + len(output_ids)}}
        return _UpstreamOutcome(200, json.dumps(chat, ensure_ascii=False, allow_nan=False).encode(),
                                max(0, round((time.monotonic() - started) * 1000)))
