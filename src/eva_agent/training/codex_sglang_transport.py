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
                 argument_projector: Any = None):
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
        self.evidence_root = Path(evidence_root) if evidence_root is not None else None
        if self.evidence_root is not None:
            self.evidence_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        if self.argument_projector is not None:
            self._write("host-argument-type-binding.json", self.argument_projector.binding)

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
                self._write(f"failure-{uuid4()}.json", {
                    "schema": "eva.codex-sglang-capture-failure.v1",
                    "requests": self._requests, "segments": len(self._segments),
                    "error_type": type(exc).__name__, "automatic_retry": False})
                # Raw HTTP/parser exception text can contain private output.
                raise ResponsesAdapterError("exact-token SGLang request failed; private audit retained") from None

    def _generate(self, body: Mapping[str, Any]) -> _UpstreamOutcome:
        import httpx

        if self._requests >= self.max_requests:
            self._budget_termination = {"kind": "max_requests", "count": self._requests,
                                       "limit": self.max_requests, "generated_tokens": self._generated}
            raise ResponsesAdapterError("Codex provider request budget exhausted")
        allowed = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls",
                   "max_tokens", "temperature", "top_p", "chat_template_kwargs", "stream", "n"}
        if set(body) - allowed or body.get("stream", False) is not False or body.get("n", 1) != 1:
            raise ResponsesAdapterError("unsupported Chat generation controls")
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise ResponsesAdapterError("requested model alias is missing")
        choice = body.get("tool_choice", "auto")
        if choice not in ("auto", "none", None):
            raise ResponsesAdapterError("forced tool choice requires an explicit constraint implementation")
        tools = deepcopy(body.get("tools") or [])
        if not isinstance(tools, list) or any(not isinstance(t, dict) or t.get("type") != "function"
                or not isinstance(t.get("function"), dict) for t in tools):
            raise ResponsesAdapterError("expected exact flat function tools")
        if choice == "none" and tools:
            raise ResponsesAdapterError("tool suppression must be explicit upstream, not a schema mutation")
        if self.argument_projector is not None:
            self.argument_projector.validate_tools(tools)
        kwargs = deepcopy(body.get("chat_template_kwargs") or {})
        if set(kwargs) - {"enable_thinking"} or type(kwargs.get("enable_thinking", True)) is not bool:
            raise ResponsesAdapterError("unsupported Qwen template controls")
        thinking = kwargs.get("enable_thinking", True)
        prompt = self.tokenizer.apply_chat_template(_messages(body), tools=tools or None,
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        prompt_ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        if not prompt_ids or any(type(t) is not int or t < 0 for t in prompt_ids):
            raise ResponsesAdapterError("tokenizer returned invalid prompt IDs")
        requested = _positive_int(body.get("max_tokens", self.max_output_tokens), "Chat output limit")
        if len(prompt_ids) >= self.max_context_tokens:
            raise ResponsesAdapterError("context budget exhausted; prompt never truncated")
        if self._generated >= self.total_output_budget:
            self._budget_termination = {"kind": "total_output_budget", "count": self._generated,
                                       "limit": self.total_output_budget, "requests": self._requests}
            raise ResponsesAdapterError("Codex total output budget exhausted")
        budget = min(requested, self.max_output_tokens, self.total_output_budget - self._generated,
                     self.max_context_tokens - len(prompt_ids),
                     self.sampling_params.get("max_new_tokens", self.max_output_tokens))
        if budget < 1:
            raise ResponsesAdapterError("context or total output budget exhausted; prompt never truncated")
        params = deepcopy(self.sampling_params)
        # Slime owns sampling policy. A conflicting Codex control must not silently
        # alter the policy whose logprobs are later used for optimization.
        for name in ("temperature", "top_p"):
            if name in body and name in params and body[name] != params[name]:
                raise ResponsesAdapterError("Codex and rollout sampling controls disagree")
            if name in body:
                params[name] = body[name]
        params.update(max_new_tokens=budget, skip_special_tokens=False)
        request_id = str(uuid4())
        payload = {"input_ids": prompt_ids, "sampling_params": params, "return_logprob": True,
                   "logprob_start_len": -1, "return_text_in_logprobs": False, "stream": False}
        self._requests += 1
        self._write(f"request-{request_id}.json", {"schema": "eva.codex-sglang-token-request.v1",
                    "request_id": request_id, "chat_request_blake3": blake3_hex(body),
                    "tools_blake3": blake3_hex(tools), "requested_model": model, **payload})
        started = time.monotonic()
        with httpx.Client(timeout=self.timeout_seconds, trust_env=False,
                          follow_redirects=False, transport=self.http_transport) as client:
            response = client.post(self.generate_url, json=payload)
            if response.status_code != 200:
                self._write(f"http-failure-{request_id}.json", {"request_id": request_id,
                            "http_status": response.status_code, "response_body_retained": False})
                raise ResponsesAdapterError("SGLang HTTP rejection")
            output = response.json()
        if not isinstance(output, dict) or not isinstance(output.get("meta_info"), dict):
            raise ResponsesAdapterError("SGLang returned no single-sample metadata")
        meta = output["meta_info"]
        rows = meta.get("output_token_logprobs")
        if (not isinstance(rows, list) or not rows or any(not isinstance(r, (list, tuple)) or len(r) < 2
                or type(r[1]) is not int or r[1] < 0 or type(r[0]) not in (int, float)
                or not math.isfinite(r[0]) or r[0] > 0 for r in rows)):
            raise ResponsesAdapterError("missing or invalid actual sampled-token log probabilities")
        output_ids, logprobs = [r[1] for r in rows], [float(r[0]) for r in rows]
        if len(output_ids) > budget or meta.get("prompt_tokens") != len(prompt_ids):
            raise ResponsesAdapterError("SGLang token counts differ from request bounds")
        if meta.get("completion_tokens") != len(output_ids):
            raise ResponsesAdapterError("SGLang completion/logprob lengths differ")
        if "output_ids" in output and output["output_ids"] != output_ids:
            raise ResponsesAdapterError("SGLang output IDs differ from sampled-logprob IDs")
        if ("output_token_logprobs_length" in meta
                and meta["output_token_logprobs_length"] != len(output_ids)):
            raise ResponsesAdapterError("SGLang logprob count differs")
        finish = meta.get("finish_reason")
        if not isinstance(finish, dict) or finish.get("type") not in {"stop", "length"}:
            raise ResponsesAdapterError("SGLang generation did not finish successfully")
        text = output.get("text")
        if not isinstance(text, str):
            raise ResponsesAdapterError("SGLang returned no decoded output")
        segment = {"schema": "eva.codex-sglang-token-segment.v1", "request_id": request_id,
                   "segment_index": len(self._segments), "requested_model": model,
                   "returned_model": output.get("model") if isinstance(output.get("model"), str) else None,
                   "chat_request_blake3": blake3_hex(body), "tools_blake3": blake3_hex(tools),
                   "prompt_token_ids": prompt_ids, "output_token_ids": output_ids,
                   "output_token_logprobs": logprobs, "output_text": text,
                   "finish_reason": deepcopy(finish), "meta_info": deepcopy(meta),
                   "sampling_params": params, "thinking_enabled": thinking,
                   "loss_mask": [1] * len(output_ids), "prompt_is_conditioning_only": True}
        self._segments.append(segment)
        self._generated += len(output_ids)
        self._write(f"segment-{request_id}.json", segment)
        message = _project_completion(text, tools, thinking, self.tokenizer)
        if self.argument_projector is not None:
            message, projection = self.argument_projector.project_message(message, tools)
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
