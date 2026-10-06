"""Loopback Responses-to-Chat adapter for exact Codex model routes.

Codex remains the sole agent runtime.  This module only adapts its OpenAI
Responses wire protocol to an upstream OpenAI-compatible Chat Completions
transport.  EvaMed tool, skill, rubric, policy, and evidence objects are never
rewritten: tools receive a temporary model-facing projection and the host still
validates every returned call against the original schema.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import secrets
import ssl
import threading
import time
from types import MappingProxyType
from typing import Any, Callable
import urllib.error
import urllib.request
from uuid import uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    is_blake3,
)

from .routes import (
    CREDENTIAL_ENV_NAMES,
    CodexConfigOverrides,
    CodexProviderConfigurationError,
    CodexProviderRoute,
    _PrivateText,
)


_RECEIPT_DOMAIN = b"eva.codex-responses-adapter-receipt.v1\x00"
PROJECTION_VERSION = (
    "eva.codex-responses-chat-projection.v5-tool-free-history-transcript"
)
FLAT_QWEN_PROJECTION_VERSION = "eva.codex-responses-chat-projection.v6-qwen-flat-exact-schemas"
PRIORITY_QWEN_PROJECTION_VERSION = "eva.codex-responses-chat-projection.v7-qwen-leading-priority-packed-namespaces"
FLAT_PRIORITY_QWEN_PROJECTION_VERSION = "eva.codex-responses-chat-projection.v7-qwen-leading-priority-flat-exact-schemas"
_VERIFIABLE_PROJECTION_VERSIONS = frozenset(
    {
        "eva.codex-responses-chat-projection.v2-packed-namespaces",
        "eva.codex-responses-chat-projection.v3-qwen-adjacent-role-coalescing",
        "eva.codex-responses-chat-projection.v4-chat-phase-qwen-role-coalescing",
        PROJECTION_VERSION,
        FLAT_QWEN_PROJECTION_VERSION,
        PRIORITY_QWEN_PROJECTION_VERSION,
        FLAT_PRIORITY_QWEN_PROJECTION_VERSION,
    }
)
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_FUNCTION_NAME = 64
_SUPPORTED_FAMILIES = frozenset({"anthropic", "deepseek", "google", "qwen"})
_KNOWN_REQUEST_KEYS = frozenset(
    {
        "background",
        "client_metadata",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "metadata",
        "model",
        "parallel_tool_calls",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "store",
        "stream",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_p",
        "truncation",
        "user",
    }
)


class ResponsesAdapterError(ValueError):
    """A request cannot be safely or exactly represented by the adapter."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResponsesAdapterError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ResponsesAdapterError(f"{label} contains a non-finite number")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResponsesAdapterError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ResponsesAdapterError(f"{label} is not a JSON object")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _int_or_zero(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


@dataclass(frozen=True, slots=True)
class AdapterLimits:
    """Resource bounds applied before private bytes cross either boundary."""

    max_request_bytes: int = 8 * 1024 * 1024
    max_upstream_response_bytes: int = 16 * 1024 * 1024
    max_concurrency: int = 32
    upstream_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not 1024 <= self.max_request_bytes <= 64 * 1024 * 1024:
            raise ResponsesAdapterError("adapter request byte limit is invalid")
        if not 1024 <= self.max_upstream_response_bytes <= 64 * 1024 * 1024:
            raise ResponsesAdapterError("adapter response byte limit is invalid")
        if not 1 <= self.max_concurrency <= 256:
            raise ResponsesAdapterError("adapter concurrency limit is invalid")
        if not 1 <= self.upstream_timeout_seconds <= 900:
            raise ResponsesAdapterError("adapter upstream timeout is invalid")


@dataclass(frozen=True, repr=False, slots=True)
class AdapterRouteBinding:
    route_id: str
    model_id: str
    provider_family: str
    credential_env_name: str
    endpoint_env_name: str
    _credential: _PrivateText
    _endpoint: _PrivateText

    def __post_init__(self) -> None:
        if self.provider_family not in _SUPPORTED_FAMILIES:
            raise ResponsesAdapterError("route does not require this adapter")

    @classmethod
    def from_route(cls, route: CodexProviderRoute) -> "AdapterRouteBinding":
        _, _, private = route.config.for_subprocess()
        endpoint, credential = private
        return cls(
            route_id=route.route_id,
            model_id=route.model_id,
            provider_family=route.provider_family,
            credential_env_name=route.config.credential_env_name,
            endpoint_env_name=route.config.endpoint_env_name,
            _credential=_PrivateText(credential),
            _endpoint=_PrivateText(endpoint),
        )

    def __repr__(self) -> str:
        return (
            "AdapterRouteBinding("
            f"route_id={self.route_id!r}, model_id={self.model_id!r}, "
            f"provider_family={self.provider_family!r}, "
            f"credential_env_name={self.credential_env_name!r}, "
            f"endpoint_env_name={self.endpoint_env_name!r}, "
            "credential=<redacted>, endpoint=<redacted>)"
        )

    def reveal_upstream_boundary(self) -> tuple[str, str]:
        """Reveal endpoint and credential only to the HTTP transport."""

        return (
            self._endpoint.reveal_at_process_boundary(),
            self._credential.reveal_at_process_boundary(),
        )

    @property
    def safe_metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "route_id": self.route_id,
                "model_id": self.model_id,
                "provider_family": self.provider_family,
                "credential_env_name": self.credential_env_name,
                "endpoint_env_name": self.endpoint_env_name,
                "credential_value_recorded": False,
                "endpoint_value_recorded": False,
            }
        )


@dataclass(frozen=True, slots=True)
class SignedAdapterReceipt:
    """Self-contained, redacted, Ed25519-signed adapter audit envelope."""

    payload: Mapping[str, Any]
    payload_blake3: str
    key_id: str
    public_key_base64: str
    public_key_blake3: str
    algorithm: str
    signature_base64: str
    envelope_blake3: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "eva.codex-responses-adapter-signed-receipt.v1",
            "signature_domain": _RECEIPT_DOMAIN[:-1].decode("ascii"),
            "payload": deepcopy(dict(self.payload)),
            "payload_blake3": self.payload_blake3,
            "key_id": self.key_id,
            "public_key_base64": self.public_key_base64,
            "public_key_blake3": self.public_key_blake3,
            "algorithm": self.algorithm,
            "signature_base64": self.signature_base64,
            "envelope_blake3": self.envelope_blake3,
        }


class AdapterReceiptSigner:
    """In-memory Ed25519 signer; private bytes have no serialization method."""

    __slots__ = ("_key", "key_id", "public_key_base64", "public_key_blake3")

    def __init__(self, key: Ed25519PrivateKey, *, key_id: str) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{2,127}", key_id):
            raise ResponsesAdapterError("adapter receipt signing key ID is invalid")
        public = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self._key = key
        self.key_id = key_id
        self.public_key_base64 = base64.b64encode(public).decode("ascii")
        self.public_key_blake3 = blake3_bytes(public)

    @classmethod
    def ephemeral(cls) -> "AdapterReceiptSigner":
        return cls(
            Ed25519PrivateKey.generate(),
            key_id="eva-codex-adapter-ephemeral-v1",
        )

    def __repr__(self) -> str:
        return (
            "AdapterReceiptSigner("
            f"key_id={self.key_id!r}, public_key_blake3={self.public_key_blake3!r}, "
            "private_key=<redacted>)"
        )

    def sign(self, payload: Mapping[str, Any]) -> SignedAdapterReceipt:
        clean_payload = deepcopy(dict(payload))
        payload_bytes = canonical_json_bytes(clean_payload)
        payload_blake3 = blake3_bytes(payload_bytes)
        signature = self._key.sign(_RECEIPT_DOMAIN + payload_bytes)
        envelope = {
            "schema": "eva.codex-responses-adapter-signed-receipt.v1",
            "signature_domain": _RECEIPT_DOMAIN[:-1].decode("ascii"),
            "payload": clean_payload,
            "payload_blake3": payload_blake3,
            "key_id": self.key_id,
            "public_key_base64": self.public_key_base64,
            "public_key_blake3": self.public_key_blake3,
            "algorithm": "Ed25519",
            "signature_base64": base64.b64encode(signature).decode("ascii"),
        }
        receipt = SignedAdapterReceipt(
            payload=MappingProxyType(clean_payload),
            payload_blake3=payload_blake3,
            key_id=self.key_id,
            public_key_base64=self.public_key_base64,
            public_key_blake3=self.public_key_blake3,
            algorithm="Ed25519",
            signature_base64=envelope["signature_base64"],
            envelope_blake3=blake3_hex(envelope),
        )
        verify_adapter_receipt(receipt)
        return receipt


def verify_adapter_receipt(receipt: SignedAdapterReceipt) -> None:
    document = receipt.to_dict()
    if document["schema"] != "eva.codex-responses-adapter-signed-receipt.v1":
        raise ResponsesAdapterError("adapter receipt schema differs")
    if document["signature_domain"] != _RECEIPT_DOMAIN[:-1].decode("ascii"):
        raise ResponsesAdapterError("adapter receipt signature domain differs")
    if receipt.algorithm != "Ed25519":
        raise ResponsesAdapterError("adapter receipt signature algorithm differs")
    expected_payload_keys = {
        "schema",
        "request_id",
        "created_at_utc",
        "route_id",
        "model_id",
        "provider_family",
        "projection_version",
        "status",
        "failure_class",
        "failure_message_blake3",
        "adapter_http_status",
        "upstream_http_status",
        "upstream_latency_ms",
        "upstream_body_blake3",
        "request_shape",
        "response_shape",
        "request_max_retries",
        "stream_max_retries",
        "upstream_request_max_retries",
        "credential_value_recorded",
        "endpoint_value_recorded",
        "raw_request_recorded",
        "raw_upstream_output_recorded",
        "raw_response_recorded",
    }
    payload = receipt.payload
    if set(payload) != expected_payload_keys:
        raise ResponsesAdapterError("adapter receipt payload fields differ")
    if payload.get("schema") != "eva.codex-responses-adapter-receipt-payload.v1":
        raise ResponsesAdapterError("adapter receipt payload schema differs")
    if payload.get("projection_version") not in _VERIFIABLE_PROJECTION_VERSIONS:
        raise ResponsesAdapterError("adapter receipt projection version differs")
    if any(
        payload.get(key) != 0
        for key in (
            "request_max_retries",
            "stream_max_retries",
            "upstream_request_max_retries",
        )
    ):
        raise ResponsesAdapterError("adapter receipt retry policy differs")
    if any(
        payload.get(key) is not False
        for key in (
            "credential_value_recorded",
            "endpoint_value_recorded",
            "raw_request_recorded",
            "raw_upstream_output_recorded",
            "raw_response_recorded",
        )
    ):
        raise ResponsesAdapterError("adapter receipt records private material")
    if not isinstance(payload.get("request_shape"), Mapping) or not isinstance(
        payload.get("response_shape"), Mapping
    ):
        raise ResponsesAdapterError("adapter receipt structural summary differs")
    for key in ("upstream_body_blake3", "failure_message_blake3"):
        value = payload.get(key)
        if value is not None and not is_blake3(value):
            raise ResponsesAdapterError(f"adapter receipt {key} differs")
    if payload.get("status") == "passed":
        if not (
            payload.get("failure_class") is None
            and payload.get("failure_message_blake3") is None
            and payload.get("adapter_http_status") == 200
            and payload.get("upstream_http_status") == 200
            and is_blake3(payload.get("upstream_body_blake3"))
        ):
            raise ResponsesAdapterError("adapter receipt success claim differs")
    elif payload.get("status") == "adapter_error":
        if not (
            isinstance(payload.get("failure_class"), str)
            and is_blake3(payload.get("failure_message_blake3"))
            and isinstance(payload.get("adapter_http_status"), int)
            and payload["adapter_http_status"] >= 400
        ):
            raise ResponsesAdapterError("adapter receipt failure claim differs")
    else:
        raise ResponsesAdapterError("adapter receipt status differs")
    payload_bytes = canonical_json_bytes(receipt.payload)
    if receipt.payload_blake3 != blake3_bytes(payload_bytes):
        raise ResponsesAdapterError("adapter receipt payload BLAKE3 differs")
    try:
        public = base64.b64decode(receipt.public_key_base64, validate=True)
        signature = base64.b64decode(receipt.signature_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ResponsesAdapterError("adapter receipt base64 differs") from exc
    if len(public) != 32 or receipt.public_key_blake3 != blake3_bytes(public):
        raise ResponsesAdapterError("adapter receipt public key differs")
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(
            signature, _RECEIPT_DOMAIN + payload_bytes
        )
    except (ValueError, InvalidSignature) as exc:
        raise ResponsesAdapterError("adapter receipt signature differs") from exc
    unsigned = {key: value for key, value in document.items() if key != "envelope_blake3"}
    if receipt.envelope_blake3 != blake3_hex(unsigned):
        raise ResponsesAdapterError("adapter receipt envelope BLAKE3 differs")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _UpstreamOutcome:
    status: int
    body: bytes
    latency_ms: int


class ChatCompletionsTransport:
    """Single-attempt HTTPS Chat Completions transport with no redirects."""

    __slots__ = ("_limits", "_opener")

    def __init__(self, limits: AdapterLimits) -> None:
        self._limits = limits
        context = ssl.create_default_context()
        self._opener = urllib.request.build_opener(
            _NoRedirect(), urllib.request.HTTPSHandler(context=context)
        )

    def __call__(self, binding: AdapterRouteBinding, body: Mapping[str, Any]) -> _UpstreamOutcome:
        endpoint, credential = binding.reveal_upstream_boundary()
        url = endpoint.rstrip("/") + "/chat/completions"
        encoded = _json_bytes(body)
        request = urllib.request.Request(
            url,
            data=encoded,
            method="POST",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {credential}",
                "Content-Type": "application/json",
                "User-Agent": "eva-agent-codex-responses-adapter/1",
            },
        )
        started = time.monotonic()
        try:
            with self._opener.open(
                request, timeout=self._limits.upstream_timeout_seconds
            ) as response:
                status = int(response.status)
                payload = response.read(self._limits.max_upstream_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            payload = exc.read(self._limits.max_upstream_response_bytes + 1)
        if len(payload) > self._limits.max_upstream_response_bytes:
            raise ResponsesAdapterError("upstream response exceeds the byte limit")
        return _UpstreamOutcome(
            status=status,
            body=payload,
            latency_ms=max(0, int(round((time.monotonic() - started) * 1000))),
        )


@dataclass(frozen=True, slots=True)
class _ToolProjection:
    chat_tools: tuple[Mapping[str, Any], ...]
    responses_to_chat: Mapping[tuple[str | None, str], str]
    chat_to_responses: Mapping[str, tuple[str | None, str]]
    packed_namespaces: Mapping[str, tuple[str, tuple[str, ...]]]

    def encode_call(self, namespace: str | None, name: str, arguments: str) -> tuple[str, str]:
        chat_name = self.responses_to_chat.get((namespace, name), name)
        packed = self.packed_namespaces.get(chat_name)
        if packed is None:
            return chat_name, arguments
        packed_namespace, child_names = packed
        if namespace != packed_namespace or name not in child_names:
            raise ResponsesAdapterError("packed namespace call does not match its tool projection")
        try:
            decoded = json.loads(arguments, object_pairs_hook=_pairs_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResponsesAdapterError("packed namespace arguments are not JSON") from exc
        if not isinstance(decoded, dict):
            raise ResponsesAdapterError("packed namespace arguments must be an object")
        return chat_name, _json_bytes(
            {"tool_name": name, "arguments": decoded}
        ).decode("utf-8")

    def decode_call(self, name: str, arguments: str) -> tuple[str | None, str, str]:
        packed = self.packed_namespaces.get(name)
        if packed is None:
            namespace, original = self.chat_to_responses.get(name, (None, name))
            return namespace, original, arguments
        namespace, child_names = packed
        try:
            value = json.loads(arguments, object_pairs_hook=_pairs_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResponsesAdapterError("packed namespace call is not JSON") from exc
        if not isinstance(value, dict) or set(value) != {"tool_name", "arguments"}:
            raise ResponsesAdapterError("packed namespace call envelope differs")
        child = value.get("tool_name")
        child_arguments = value.get("arguments")
        if not isinstance(child, str) or child not in child_names:
            raise ResponsesAdapterError("packed namespace child is unavailable")
        if not isinstance(child_arguments, dict):
            raise ResponsesAdapterError("packed namespace child arguments differ")
        return namespace, child, _json_bytes(child_arguments).decode("utf-8")


def _scrub_qwen_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrub_qwen_schema(child)
            for key, child in value.items()
            if key != "uniqueItems"
        }
    if isinstance(value, list):
        return [_scrub_qwen_schema(child) for child in value]
    return deepcopy(value)


def _project_function_name(namespace: str | None, name: str, used: set[str]) -> str:
    candidate = name if namespace is None else f"{namespace}__{name}"
    candidate = re.sub(r"[^A-Za-z0-9_-]", "_", candidate)
    if not candidate:
        candidate = "tool"
    if len(candidate) > _MAX_FUNCTION_NAME or candidate in used:
        digest = blake3_hex({"namespace": namespace, "name": name})[:12]
        candidate = candidate[: _MAX_FUNCTION_NAME - 13] + "_" + digest
    if candidate in used or _NAME.fullmatch(candidate) is None:
        raise ResponsesAdapterError("tool name projection is ambiguous")
    used.add(candidate)
    return candidate


def _tool_projection(
    tools: Any,
    *,
    provider_family: str,
    forced_tool: tuple[str | None, str] | None = None,
    preserve_qwen_tool_schemas: bool = False,
) -> _ToolProjection:
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        raise ResponsesAdapterError("Responses tools must be an array")
    projected: list[Mapping[str, Any]] = []
    forward: dict[tuple[str | None, str], str] = {}
    reverse: dict[str, tuple[str | None, str]] = {}
    packed_namespaces: dict[str, tuple[str, tuple[str, ...]]] = {}
    used: set[str] = set()

    def add(value: Mapping[str, Any], namespace: str | None) -> None:
        if value.get("type") != "function":
            raise ResponsesAdapterError("namespace contains a non-function tool")
        name = value.get("name")
        description = value.get("description", "")
        parameters = value.get("parameters", {"type": "object"})
        if not isinstance(name, str) or not name or not isinstance(description, str):
            raise ResponsesAdapterError("function tool identity differs")
        if not isinstance(parameters, dict):
            raise ResponsesAdapterError("function tool parameters differ")
        chat_name = _project_function_name(namespace, name, used)
        if provider_family == "qwen" and not preserve_qwen_tool_schemas:
            parameters = _scrub_qwen_schema(parameters)
        else:
            parameters = deepcopy(parameters)
        function = {
            "name": chat_name,
            "description": description,
            "parameters": parameters,
        }
        projected.append({"type": "function", "function": function})
        forward[(namespace, name)] = chat_name
        reverse[chat_name] = (namespace, name)

    for tool in tools:
        if not isinstance(tool, dict):
            raise ResponsesAdapterError("Responses tool entry differs")
        tool_type = tool.get("type")
        if tool_type == "function":
            add(tool, None)
        elif tool_type == "namespace":
            namespace = tool.get("name")
            children = tool.get("tools")
            if not isinstance(namespace, str) or not namespace or not isinstance(children, list):
                raise ResponsesAdapterError("namespace tool differs")
            if provider_family != "qwen" or preserve_qwen_tool_schemas:
                for child in children:
                    if not isinstance(child, dict):
                        raise ResponsesAdapterError("namespace child differs")
                    add(child, namespace)
                continue
            child_rows: list[tuple[str, str]] = []
            for child in children:
                if not isinstance(child, dict) or child.get("type") != "function":
                    raise ResponsesAdapterError("namespace child differs")
                name = child.get("name")
                description = child.get("description", "")
                parameters = child.get("parameters", {})
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(description, str)
                    or not isinstance(parameters, dict)
                ):
                    raise ResponsesAdapterError("namespace child identity differs")
                child_rows.append((name, description))
            if not child_rows:
                raise ResponsesAdapterError("namespace has no function tools")
            child_names = tuple(name for name, _ in child_rows)
            if len(set(child_names)) != len(child_names):
                raise ResponsesAdapterError("namespace child name is duplicated")
            visible_names = child_names
            if forced_tool is not None and forced_tool[0] == namespace:
                if forced_tool[1] not in child_names:
                    raise ResponsesAdapterError("forced namespace child is unavailable")
                visible_names = (forced_tool[1],)
            chat_name = _project_function_name(None, f"{namespace}__dispatch", used)
            summaries = []
            for name, description in child_rows:
                if name not in visible_names:
                    continue
                compact = " ".join(description.split())[:240]
                summaries.append(f"{name}: {compact}")
            packed_description = (
                f"Dispatch exactly one function from namespace {namespace}. "
                "Set tool_name to the child name and arguments to that child's JSON object. "
                "The Codex host validates the exact original child schema. Available: "
                + "; ".join(summaries)
            )[:12_000]
            projected.append(
                {
                    "type": "function",
                    "function": {
                        "name": chat_name,
                        "description": packed_description,
                        "parameters": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "tool_name": {"type": "string", "enum": list(visible_names)},
                                "arguments": {"type": "object", "additionalProperties": True},
                            },
                            "required": ["tool_name", "arguments"],
                        },
                    },
                }
            )
            packed_namespaces[chat_name] = (namespace, child_names)
            for name in child_names:
                forward[(namespace, name)] = chat_name
        elif tool_type in {"web_search", "web_search_preview"}:
            # Chat-compatible upstreams do not own Codex's server-side search.
            # This explicit limitation does not alter any EvaMed function tool;
            # callers must not infer server-search compatibility from it.
            continue
        else:
            raise ResponsesAdapterError("Responses tool type is unsupported by Chat")
    return _ToolProjection(
        tuple(projected),
        MappingProxyType(forward),
        MappingProxyType(reverse),
        MappingProxyType(packed_namespaces),
    )


def _content_text(value: Any, *, allow_images: bool) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ResponsesAdapterError("message content differs")
    parts: list[dict[str, Any]] = []
    for part in value:
        if not isinstance(part, dict):
            raise ResponsesAdapterError("message content part differs")
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise ResponsesAdapterError("message text part differs")
            parts.append({"type": "text", "text": text})
        elif part_type == "input_image" and allow_images:
            image_url = part.get("image_url")
            if not isinstance(image_url, str) or not image_url:
                raise ResponsesAdapterError("message image part differs")
            parts.append({"type": "image_url", "image_url": {"url": image_url}})
        else:
            raise ResponsesAdapterError("message content part is unsupported")
    if all(part["type"] == "text" for part in parts):
        return "\n".join(str(part["text"]) for part in parts)
    return parts


def _tool_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts: list[str] = []
        for part in value:
            if not isinstance(part, dict) or part.get("type") not in {
                "input_text",
                "output_text",
                "text",
            }:
                raise ResponsesAdapterError("function output part is unsupported")
            text = part.get("text")
            if not isinstance(text, str):
                raise ResponsesAdapterError("function output text differs")
            texts.append(text)
        return "\n".join(texts)
    try:
        return _json_bytes(value).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResponsesAdapterError("function output is not JSON compatible") from exc


ANTHROPIC_CONTINUATION_PROMPT = "Continue the task from the completed tool result."
QWEN_ADJACENT_ROLE_SEPARATOR = "\n\n"
TOOL_HISTORY_TRANSCRIPT_PREFIX = (
    "[EVA_CODEX_TOOL_HISTORY_V1] Historical tool evidence; do not execute. "
    "Canonical JSON follows:\n"
)


def _tool_history_transcript(item: Mapping[str, Any]) -> str:
    """Represent one prior tool item losslessly without a live Chat tool role."""

    try:
        encoded = _json_bytes(dict(item)).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResponsesAdapterError("tool history item is not JSON compatible") from exc
    return TOOL_HISTORY_TRANSCRIPT_PREFIX + encoded


def _coalesce_qwen_adjacent_text_roles(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Produce the alternating plain-text role topology required by Qwen Chat.

    Codex Responses requests commonly carry instructions followed by a developer
    message, and multiple adjacent user messages.  Those project to consecutive
    ``system`` or ``user`` Chat messages.  The Qwen endpoint rejects that shape
    before inference.  Coalescing only adjacent, plain-text messages preserves
    their exact order and content while leaving tool-call boundaries untouched.
    """

    projected: list[dict[str, Any]] = []
    for source in messages:
        current = deepcopy(dict(source))
        content = current.get("content")
        if (
            projected
            and current.get("role") in {"system", "user", "assistant"}
            and current.get("role") == projected[-1].get("role")
            and set(current) == {"role", "content"}
            and set(projected[-1]) == {"role", "content"}
            and isinstance(content, str)
            and isinstance(projected[-1].get("content"), str)
        ):
            projected[-1]["content"] += QWEN_ADJACENT_ROLE_SEPARATOR + content
            continue
        projected.append(current)
    return projected


def _normalize_qwen_priority_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One leading priority block, without elevating user or tool observations.

    Qwen's official template rejects mid-history system messages. Responses
    developer messages already map to system in this adapter. Move only those
    actual high-priority texts, in their original relative order. Do not merge
    or rearrange any nonpriority message, tool boundary, or image content.
    """
    priority, other = [], []
    for source in messages:
        current = deepcopy(dict(source))
        if current.get("role") == "system":
            if set(current) != {"role", "content"} or not isinstance(current.get("content"), str):
                raise ResponsesAdapterError("Qwen priority message is not plain instruction text")
            priority.append(current["content"])
        else:
            other.append(current)
    return ([{"role": "system", "content": QWEN_ADJACENT_ROLE_SEPARATOR.join(priority)}] if priority else []) + other


def _messages(
    input_value: Any,
    instructions: Any,
    tools: _ToolProjection,
    *,
    provider_family: str,
    normalize_qwen_priority_messages: bool = False,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    elif instructions is not None and instructions != "":
        raise ResponsesAdapterError("non-text Responses instructions are unsupported")
    if isinstance(input_value, str):
        input_value = [{"type": "message", "role": "user", "content": input_value}]
    if not isinstance(input_value, list):
        raise ResponsesAdapterError("Responses input must be text or an item array")
    pending_calls: list[dict[str, Any]] = []
    transcribe_tool_history = not tools.chat_tools

    def flush_calls() -> None:
        if pending_calls:
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": list(pending_calls)}
            )
            pending_calls.clear()

    for item in input_value:
        if not isinstance(item, dict):
            raise ResponsesAdapterError("Responses input item differs")
        item_type = item.get("type", "message")
        if item_type == "reasoning":
            continue
        if item_type == "message":
            flush_calls()
            role = item.get("role")
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant"}:
                raise ResponsesAdapterError("message role is unsupported")
            messages.append(
                {
                    "role": role,
                    "content": _content_text(item.get("content", ""), allow_images=role == "user"),
                }
            )
        elif item_type == "function_call":
            name = item.get("name")
            call_id = item.get("call_id")
            arguments = item.get("arguments")
            namespace = item.get("namespace")
            if namespace is not None and not isinstance(namespace, str):
                raise ResponsesAdapterError("function-call namespace differs")
            if not all(isinstance(value, str) and value for value in (name, call_id, arguments)):
                raise ResponsesAdapterError("function-call fields differ")
            if transcribe_tool_history:
                flush_calls()
                messages.append(
                    {
                        "role": "assistant",
                        "content": _tool_history_transcript(item),
                    }
                )
                continue
            chat_name, chat_arguments = tools.encode_call(namespace, name, arguments)
            pending_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": chat_name,
                        "arguments": chat_arguments,
                    },
                }
            )
        elif item_type == "function_call_output":
            flush_calls()
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ResponsesAdapterError("function-call output ID differs")
            output_text = _tool_output_text(item.get("output"))
            if transcribe_tool_history:
                messages.append(
                    {
                        "role": "user",
                        "content": _tool_history_transcript(item),
                    }
                )
                continue
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output_text,
                }
            )
        else:
            raise ResponsesAdapterError("Responses input item type is unsupported")
    flush_calls()
    if not messages:
        raise ResponsesAdapterError("Responses input contains no representable message")
    # Azure Anthropic rejects a final assistant message as an unsupported
    # prefill. Codex can produce that shape when commentary follows a built-in
    # MCP discovery result. Preserve the history and add one deterministic user
    # continuation; this is a request projection, not a retry or schema change.
    if provider_family == "anthropic" and messages[-1].get("role") == "assistant":
        messages.append({"role": "user", "content": ANTHROPIC_CONTINUATION_PROMPT})
    if provider_family == "qwen":
        messages = _coalesce_qwen_adjacent_text_roles(messages)
        if normalize_qwen_priority_messages:
            messages = _normalize_qwen_priority_messages(messages)
    return messages


def _tool_choice(value: Any, tools: _ToolProjection) -> Any:
    if value is None:
        return "auto"
    if isinstance(value, str) and value in {"none", "auto", "required"}:
        return value
    if isinstance(value, dict) and value.get("type") == "function":
        name = value.get("name")
        namespace = value.get("namespace")
        if not isinstance(name, str):
            raise ResponsesAdapterError("function tool choice differs")
        chat_name = tools.responses_to_chat.get((namespace, name), name)
        return {
            "type": "function",
            "function": {"name": chat_name},
        }
    raise ResponsesAdapterError("Responses tool choice is unsupported")


def responses_to_chat(
    request: Mapping[str, Any], binding: AdapterRouteBinding, *,
    preserve_qwen_tool_schemas: bool = False,
    normalize_qwen_priority_messages: bool = False,
) -> tuple[dict[str, Any], _ToolProjection]:
    """Project one stateless Responses turn into a Chat Completions turn."""

    if type(normalize_qwen_priority_messages) is not bool:
        raise ResponsesAdapterError("Qwen priority normalization selector must be boolean")
    if normalize_qwen_priority_messages and binding.provider_family != "qwen":
        raise ResponsesAdapterError("priority normalization is explicitly Qwen-only")

    unknown = set(request) - _KNOWN_REQUEST_KEYS
    if unknown:
        raise ResponsesAdapterError("Responses request has unsupported fields")
    if request.get("model") != binding.model_id:
        raise ResponsesAdapterError("request model is not bound to this route")
    if request.get("background") not in (None, False):
        raise ResponsesAdapterError("background Responses requests are unsupported")
    if request.get("store") not in (None, False):
        raise ResponsesAdapterError("stored Responses requests are unsupported")
    if request.get("stream") not in (None, False, True):
        raise ResponsesAdapterError("Responses stream flag differs")
    max_output_tokens = request.get("max_output_tokens")
    if max_output_tokens is not None and (
        type(max_output_tokens) is not int
        or not 1 <= max_output_tokens <= 131_072
    ):
        raise ResponsesAdapterError("Responses output budget differs")
    choice = request.get("tool_choice")
    forced_tool: tuple[str | None, str] | None = None
    if isinstance(choice, dict) and choice.get("type") == "function":
        choice_name = choice.get("name")
        choice_namespace = choice.get("namespace")
        if not isinstance(choice_name, str) or (
            choice_namespace is not None and not isinstance(choice_namespace, str)
        ):
            raise ResponsesAdapterError("function tool choice differs")
        forced_tool = (choice_namespace, choice_name)
    projection = _tool_projection(
        request.get("tools", []),
        provider_family=binding.provider_family,
        forced_tool=forced_tool,
        preserve_qwen_tool_schemas=preserve_qwen_tool_schemas,
    )
    chat: dict[str, Any] = {
        "model": binding.model_id,
        "messages": _messages(
            request.get("input"),
            request.get("instructions"),
            projection,
            provider_family=binding.provider_family,
            normalize_qwen_priority_messages=normalize_qwen_priority_messages,
        ),
    }
    if projection.chat_tools:
        chat["tools"] = [deepcopy(dict(value)) for value in projection.chat_tools]
        chat["tool_choice"] = _tool_choice(request.get("tool_choice"), projection)
    elif request.get("tool_choice") == "required":
        raise ResponsesAdapterError("required tool choice has no Chat-compatible tool")
    if binding.provider_family != "anthropic":
        chat["parallel_tool_calls"] = bool(request.get("parallel_tool_calls", True))
    for source, target in (
        ("max_output_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
    ):
        if request.get(source) is not None:
            chat[target] = request[source]
    return chat, projection


def _chat_text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        texts: list[str] = []
        for part in value:
            if not isinstance(part, dict):
                raise ResponsesAdapterError("upstream content part differs")
            text = part.get("text")
            if not isinstance(text, str):
                raise ResponsesAdapterError("upstream content text differs")
            texts.append(text)
        return "".join(texts)
    raise ResponsesAdapterError("upstream assistant content differs")


def _response_usage(value: Any) -> dict[str, Any]:
    usage = value if isinstance(value, dict) else {}
    input_tokens = _int_or_zero(usage.get("prompt_tokens"))
    output_tokens = _int_or_zero(usage.get("completion_tokens"))
    cached = 0
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached = _int_or_zero(prompt_details.get("cached_tokens"))
    reasoning = 0
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning = _int_or_zero(completion_details.get("reasoning_tokens"))
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": _int_or_zero(usage.get("total_tokens"))
        or input_tokens + output_tokens,
    }


def chat_to_response(
    upstream: Mapping[str, Any],
    request: Mapping[str, Any],
    binding: AdapterRouteBinding,
    tools: _ToolProjection,
) -> dict[str, Any]:
    """Validate one Chat completion and construct an exact Responses object."""

    if upstream.get("model") != binding.model_id:
        raise ResponsesAdapterError("upstream returned a different model identity")
    choices = upstream.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ResponsesAdapterError("upstream must return exactly one assistant choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ResponsesAdapterError("upstream assistant choice differs")
    content = _chat_text(message.get("content"))
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ResponsesAdapterError("upstream tool calls differ")
    output: list[dict[str, Any]] = []
    if content is not None and content != "":
        output.append(
            {
                "id": "msg_" + uuid4().hex,
                "type": "message",
                "role": "assistant",
                # Chat has no Codex channel field.  A message accompanying a
                # tool call is commentary; a tool-free assistant choice is the
                # terminal answer.  Preserve that distinction so Codex's
                # terminal-response provenance check remains authoritative.
                "phase": "commentary" if raw_calls else "final_answer",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict) or raw_call.get("type", "function") != "function":
            raise ResponsesAdapterError("upstream tool call type differs")
        function = raw_call.get("function")
        call_id = raw_call.get("id")
        if not isinstance(function, dict) or not isinstance(call_id, str) or not call_id:
            raise ResponsesAdapterError("upstream tool call envelope differs")
        chat_name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(chat_name, str) or not isinstance(arguments, str):
            raise ResponsesAdapterError("upstream tool call fields differ")
        namespace, name, arguments = tools.decode_call(chat_name, arguments)
        item: dict[str, Any] = {
            "id": "fc_" + uuid4().hex,
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        }
        if namespace is not None:
            item["namespace"] = namespace
        output.append(item)
    if not output:
        output.append(
            {
                "id": "msg_" + uuid4().hex,
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
                ],
            }
        )
    created = time.time()
    return {
        "id": "resp_" + uuid4().hex,
        "object": "response",
        "created_at": created,
        "completed_at": created,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": request.get("instructions"),
        "model": binding.model_id,
        "output": output,
        "parallel_tool_calls": bool(request.get("parallel_tool_calls", True)),
        "tool_choice": request.get("tool_choice", "auto"),
        "tools": deepcopy(request.get("tools", [])),
        "usage": _response_usage(upstream.get("usage")),
    }


def _stream_events(response: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    sequence = 0
    events: list[dict[str, Any]] = []

    def add(event_type: str, **values: Any) -> None:
        nonlocal sequence
        events.append({"type": event_type, **values, "sequence_number": sequence})
        sequence += 1

    in_progress = deepcopy(dict(response))
    in_progress["status"] = "in_progress"
    in_progress["completed_at"] = None
    in_progress["output"] = []
    add("response.created", response=deepcopy(in_progress))
    add("response.in_progress", response=deepcopy(in_progress))
    for output_index, final_item in enumerate(response["output"]):
        item = deepcopy(final_item)
        item["status"] = "in_progress"
        if item["type"] == "message":
            item["content"] = []
        elif item["type"] == "function_call":
            item["arguments"] = ""
        add("response.output_item.added", output_index=output_index, item=item)
        if final_item["type"] == "message":
            for content_index, part in enumerate(final_item["content"]):
                empty_part = deepcopy(part)
                empty_part["text"] = ""
                add(
                    "response.content_part.added",
                    item_id=final_item["id"],
                    output_index=output_index,
                    content_index=content_index,
                    part=empty_part,
                )
                add(
                    "response.output_text.delta",
                    item_id=final_item["id"],
                    output_index=output_index,
                    content_index=content_index,
                    delta=part["text"],
                    logprobs=[],
                )
                add(
                    "response.output_text.done",
                    item_id=final_item["id"],
                    output_index=output_index,
                    content_index=content_index,
                    text=part["text"],
                    logprobs=[],
                )
                add(
                    "response.content_part.done",
                    item_id=final_item["id"],
                    output_index=output_index,
                    content_index=content_index,
                    part=deepcopy(part),
                )
        else:
            add(
                "response.function_call_arguments.delta",
                item_id=final_item["id"],
                output_index=output_index,
                delta=final_item["arguments"],
            )
            add(
                "response.function_call_arguments.done",
                item_id=final_item["id"],
                output_index=output_index,
                name=final_item["name"],
                arguments=final_item["arguments"],
            )
        add(
            "response.output_item.done",
            output_index=output_index,
            item=deepcopy(final_item),
        )
    add("response.completed", response=deepcopy(dict(response)))
    return tuple(events)


def _sse_bytes(response: Mapping[str, Any]) -> bytes:
    chunks = []
    for event in _stream_events(response):
        chunks.append(f"event: {event['type']}\n".encode("ascii"))
        chunks.append(b"data: " + _json_bytes(event) + b"\n\n")
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)


def _request_shape(request: Mapping[str, Any]) -> dict[str, Any]:
    items = request.get("input")
    item_types: list[str] = []
    function_outputs = 0
    safe_item_types = {
        "message",
        "function_call",
        "function_call_output",
        "reasoning",
        "custom_tool_call",
        "custom_tool_call_output",
        "local_shell_call",
        "local_shell_call_output",
    }
    if isinstance(items, list):
        for item in items:
            item_type = item.get("type", "message") if isinstance(item, dict) else "invalid"
            item_types.append(
                item_type
                if isinstance(item_type, str) and item_type in safe_item_types
                else "other"
            )
            if item_type == "function_call_output":
                function_outputs += 1
    tools = request.get("tools")
    return {
        "stream": request.get("stream") is True,
        "input_item_count": len(items) if isinstance(items, list) else 1,
        "input_item_types": item_types,
        "function_call_output_count": function_outputs,
        "tool_count": len(tools) if isinstance(tools, list) else 0,
        "parallel_tool_calls_requested": request.get("parallel_tool_calls") is True,
    }


def _response_shape(response: Mapping[str, Any] | None) -> dict[str, Any]:
    output = response.get("output") if isinstance(response, Mapping) else None
    output_types = (
        [
            item.get("type")
            if isinstance(item.get("type"), str)
            and item.get("type") in {"message", "function_call"}
            else "other"
            for item in output
            if isinstance(item, dict)
        ]
        if isinstance(output, list)
        else []
    )
    return {
        "output_item_count": len(output) if isinstance(output, list) else 0,
        "output_item_types": output_types,
        "parallel_function_call_count": sum(value == "function_call" for value in output_types),
    }


class ResponsesAdapterGateway:
    """Authenticated loopback gateway serving one or more explicit routes."""

    def __init__(
        self,
        routes: Mapping[str, CodexProviderRoute],
        *,
        limits: AdapterLimits = AdapterLimits(),
        signer: AdapterReceiptSigner | None = None,
        upstream_transport: Callable[[AdapterRouteBinding, Mapping[str, Any]], _UpstreamOutcome]
        | None = None,
        local_bearer_token: str | None = None,
        local_credential_env_name: str | None = None,
        upstream_max_tokens_override: int | None = None,
        preserve_qwen_tool_schemas: bool = False,
        normalize_qwen_priority_messages: bool = False,
        receipt_sink: Callable[[SignedAdapterReceipt], None] | None = None,
    ) -> None:
        if not routes:
            raise ResponsesAdapterError("adapter requires at least one route")
        if any(key != route.route_id for key, route in routes.items()):
            raise ResponsesAdapterError("adapter route key differs from its route identity")
        bindings = [AdapterRouteBinding.from_route(route) for route in routes.values()]
        if len({binding.model_id for binding in bindings}) != len(bindings):
            raise ResponsesAdapterError("adapter model routes are ambiguous")
        self._bindings = {binding.model_id: binding for binding in bindings}
        self._source_routes = dict(routes)
        self._limits = limits
        self._signer = signer or AdapterReceiptSigner.ephemeral()
        self._upstream = upstream_transport or ChatCompletionsTransport(limits)
        self._local_token = local_bearer_token or secrets.token_urlsafe(32)
        if not self._local_token or any(character.isspace() for character in self._local_token):
            raise ResponsesAdapterError("adapter loopback token is invalid")
        self._local_credential_env_name = (
            local_credential_env_name or bindings[0].credential_env_name
        )
        if self._local_credential_env_name not in CREDENTIAL_ENV_NAMES:
            raise ResponsesAdapterError("adapter loopback credential env name is invalid")
        if upstream_max_tokens_override is not None and (
            type(upstream_max_tokens_override) is not int
            or not 1 <= upstream_max_tokens_override <= 131_072
        ):
            raise ResponsesAdapterError("adapter upstream output budget is invalid")
        if receipt_sink is not None and not callable(receipt_sink):
            raise ResponsesAdapterError("adapter receipt sink is invalid")
        self._upstream_max_tokens_override = upstream_max_tokens_override
        if not isinstance(preserve_qwen_tool_schemas, bool):
            raise ResponsesAdapterError("tool schema projection selector must be boolean")
        self._preserve_qwen_tool_schemas = preserve_qwen_tool_schemas
        if type(normalize_qwen_priority_messages) is not bool:
            raise ResponsesAdapterError("Qwen priority normalization selector must be boolean")
        if normalize_qwen_priority_messages and any(binding.provider_family != "qwen" for binding in bindings):
            raise ResponsesAdapterError("priority normalization requires Qwen-only routes")
        self._normalize_qwen_priority_messages = normalize_qwen_priority_messages
        self._projection_version = (FLAT_QWEN_PROJECTION_VERSION
                                    if preserve_qwen_tool_schemas else PROJECTION_VERSION)
        if normalize_qwen_priority_messages:
            self._projection_version = (FLAT_PRIORITY_QWEN_PROJECTION_VERSION
                if preserve_qwen_tool_schemas else PRIORITY_QWEN_PROJECTION_VERSION)
        self._receipt_sink = receipt_sink
        self._semaphore = threading.BoundedSemaphore(limits.max_concurrency)
        self._receipt_lock = threading.Lock()
        self._receipts: list[SignedAdapterReceipt] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __repr__(self) -> str:
        return (
            "ResponsesAdapterGateway("
            f"route_ids={tuple(sorted(route.route_id for route in self._bindings.values()))!r}, "
            f"max_concurrency={self._limits.max_concurrency!r}, "
            f"upstream_max_tokens_override={self._upstream_max_tokens_override!r}, "
            "loopback_token=<redacted>, upstream_endpoints=<redacted>, "
            "upstream_credentials=<redacted>)"
        )

    @property
    def receipts(self) -> tuple[SignedAdapterReceipt, ...]:
        with self._receipt_lock:
            return tuple(self._receipts)

    def install_receipt_sink(
        self, sink: Callable[[SignedAdapterReceipt], None]
    ) -> None:
        """Install one pre-start fail-closed durable receipt callback.

        The default remains entirely in-memory.  Premium construction uses
        this narrow hook to commit each provider-bound receipt before Codex
        can observe the corresponding response.
        """

        if not callable(sink):
            raise ResponsesAdapterError("adapter receipt sink is invalid")
        with self._receipt_lock:
            if (
                self._server is not None
                or self._thread is not None
                or self._receipts
                or self._receipt_sink is not None
            ):
                raise ResponsesAdapterError("adapter receipt sink installation is not fresh")
            self._receipt_sink = sink

    @property
    def safe_metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": "eva.codex-responses-adapter-config.v1",
                "projection_version": self._projection_version,
                **({"qwen_priority_message_normalization": "one-leading-block-preserve-priority-order-no-user-tool-promotion"}
                   if self._normalize_qwen_priority_messages else {}),
                "route_bindings": [
                    dict(binding.safe_metadata)
                    for binding in sorted(self._bindings.values(), key=lambda item: item.route_id)
                ],
                "bind_host": "127.0.0.1",
                "max_concurrency": self._limits.max_concurrency,
                "upstream_max_tokens_override": self._upstream_max_tokens_override,
                "request_max_retries": 0,
                "stream_max_retries": 0,
                "upstream_request_max_retries": 0,
                "loopback_token_recorded": False,
                "local_credential_env_name": self._local_credential_env_name,
                "raw_requests_recorded": False,
                "raw_upstream_outputs_recorded": False,
                "signing_key_id": self._signer.key_id,
                "signing_public_key_blake3": self._signer.public_key_blake3,
            }
        )

    def __enter__(self) -> "ResponsesAdapterGateway":
        if self._server is not None:
            raise ResponsesAdapterError("adapter gateway is already running")
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                gateway._serve(self)

            def do_GET(self) -> None:
                gateway._send_json(
                    self,
                    HTTPStatus.NOT_FOUND,
                    {"error": {"type": "not_found", "message": "resource unavailable"}},
                )

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = False

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="eva-codex-responses-adapter",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def _adapted_routes_for(
        self, *, endpoint: str, token: str
    ) -> Mapping[str, CodexProviderRoute]:
        result: dict[str, CodexProviderRoute] = {}
        for route_id, source in self._source_routes.items():
            config = CodexConfigOverrides(
                provider_id=f"eva_adapter_{route_id}",
                model_id=source.model_id,
                credential_env_name=self._local_credential_env_name,
                endpoint_env_name=source.config.endpoint_env_name,
                _credential=_PrivateText(token),
                _endpoint=_PrivateText(endpoint),
            )
            result[route_id] = CodexProviderRoute(
                route_id=source.route_id,
                model_id=source.model_id,
                model_env_name=source.model_env_name,
                registry_role_id=source.registry_role_id,
                provider_family=source.provider_family,
                config=config,
            )
        return MappingProxyType(result)

    def planned_adapted_routes(self) -> Mapping[str, CodexProviderRoute]:
        """Expose stable public route commitments without starting a server.

        Private placeholders never cross a process boundary.  The safe route
        projection deliberately excludes endpoint/token values, so its digest
        is identical to the live route returned by :meth:`adapted_routes`.
        """

        return self._adapted_routes_for(
            endpoint="http://127.0.0.1:1/v1",
            token="not-a-runtime-token",
        )

    def adapted_routes(self) -> Mapping[str, CodexProviderRoute]:
        if self._server is None:
            raise ResponsesAdapterError("adapter gateway is not running")
        endpoint = f"http://127.0.0.1:{self._server.server_port}/v1"
        return self._adapted_routes_for(endpoint=endpoint, token=self._local_token)

    def receipts_document(self) -> dict[str, Any]:
        payload = {
            "schema": "eva.codex-responses-adapter-receipts.v1",
            "adapter_config_blake3": blake3_hex(self.safe_metadata),
            "receipt_count": len(self.receipts),
            "receipts": [receipt.to_dict() for receipt in self.receipts],
            "raw_requests_recorded": False,
            "raw_upstream_outputs_recorded": False,
            "credential_values_recorded": False,
            "endpoint_values_recorded": False,
        }
        payload["document_blake3"] = blake3_hex(payload)
        return payload

    def _send_json(self, handler: BaseHTTPRequestHandler, status: int, value: Any) -> None:
        payload = _json_bytes(value)
        handler.send_response(int(status))
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(payload)

    def _serve(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path != "/v1/responses":
            self._send_json(
                handler,
                HTTPStatus.NOT_FOUND,
                {"error": {"type": "not_found", "message": "resource unavailable"}},
            )
            return
        authorization = handler.headers.get("Authorization", "")
        if not hmac.compare_digest(authorization, f"Bearer {self._local_token}"):
            self._send_json(
                handler,
                HTTPStatus.UNAUTHORIZED,
                {"error": {"type": "authentication_error", "message": "authentication failed"}},
            )
            return
        if not self._semaphore.acquire(blocking=False):
            self._send_json(
                handler,
                HTTPStatus.TOO_MANY_REQUESTS,
                {"error": {"type": "capacity_error", "message": "adapter capacity reached"}},
            )
            return
        try:
            self._serve_authorized(handler)
        finally:
            self._semaphore.release()

    def _serve_authorized(self, handler: BaseHTTPRequestHandler) -> None:
        request_id = str(uuid4())
        created_at = _utc_now()
        request_shape: dict[str, Any] = {}
        response_shape: dict[str, Any] = _response_shape(None)
        binding: AdapterRouteBinding | None = None
        upstream_status: int | None = None
        upstream_latency = 0
        upstream_body_blake3: str | None = None
        status = "adapter_error"
        failure_class: str | None = None
        failure_message_blake3: str | None = None
        http_status = HTTPStatus.BAD_REQUEST
        response: dict[str, Any] | None = None
        recorded = False

        def record() -> None:
            nonlocal recorded
            payload = {
                "schema": "eva.codex-responses-adapter-receipt-payload.v1",
                "request_id": request_id,
                "created_at_utc": created_at,
                "route_id": None if binding is None else binding.route_id,
                "model_id": None if binding is None else binding.model_id,
                "provider_family": None if binding is None else binding.provider_family,
                "projection_version": self._projection_version,
                "status": status,
                "failure_class": failure_class,
                "adapter_http_status": int(http_status),
                "upstream_http_status": upstream_status,
                "upstream_latency_ms": upstream_latency,
                "upstream_body_blake3": upstream_body_blake3,
                "failure_message_blake3": failure_message_blake3,
                "request_shape": request_shape,
                "response_shape": response_shape,
                "request_max_retries": 0,
                "stream_max_retries": 0,
                "upstream_request_max_retries": 0,
                "credential_value_recorded": False,
                "endpoint_value_recorded": False,
                "raw_request_recorded": False,
                "raw_upstream_output_recorded": False,
                "raw_response_recorded": False,
            }
            receipt = self._signer.sign(payload)
            # Mark this request before crossing the external durability hook.
            # A sink failure therefore closes the HTTP exchange without a
            # second signed attempt or a fail-open response to Codex.
            recorded = True
            if self._receipt_sink is not None:
                self._receipt_sink(receipt)
            with self._receipt_lock:
                self._receipts.append(receipt)

        try:
            content_type = handler.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                raise ResponsesAdapterError("request content type must be application/json")
            raw_length = handler.headers.get("Content-Length")
            if raw_length is None or not raw_length.isdigit():
                raise ResponsesAdapterError("request content length is required")
            length = int(raw_length)
            if length > self._limits.max_request_bytes:
                http_status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                raise ResponsesAdapterError("request exceeds the byte limit")
            request = _json_object(handler.rfile.read(length), label="Responses request")
            request_shape = _request_shape(request)
            model = request.get("model")
            binding = self._bindings.get(model) if isinstance(model, str) else None
            if binding is None:
                http_status = HTTPStatus.NOT_FOUND
                raise ResponsesAdapterError("request model has no adapter route")
            chat_body, tool_projection = responses_to_chat(request, binding,
                preserve_qwen_tool_schemas=self._preserve_qwen_tool_schemas,
                normalize_qwen_priority_messages=self._normalize_qwen_priority_messages)
            requested_max_tokens = request.get("max_output_tokens")
            if self._upstream_max_tokens_override is not None:
                chat_body["max_tokens"] = self._upstream_max_tokens_override
            request_shape = {
                **request_shape,
                "requested_max_output_tokens": requested_max_tokens,
                "effective_upstream_max_tokens": chat_body.get("max_tokens"),
                **({"qwen_priority_message_normalization": True,
                    "projected_system_message_count": sum(row["role"] == "system" for row in chat_body["messages"]),
                    "projected_system_only_at_start": all(index == 0 for index, row in enumerate(chat_body["messages"]) if row["role"] == "system")}
                   if self._normalize_qwen_priority_messages else {}),
                "upstream_max_tokens_policy": (
                    "exact_override"
                    if self._upstream_max_tokens_override is not None
                    else "passthrough"
                ),
            }
            try:
                upstream = self._upstream(binding, chat_body)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                http_status = HTTPStatus.BAD_GATEWAY
                failure_class = "upstream_transport_error"
                raise ResponsesAdapterError("upstream transport failed") from exc
            upstream_status = upstream.status
            upstream_latency = upstream.latency_ms
            upstream_body_blake3 = blake3_bytes(upstream.body)
            if upstream.status != HTTPStatus.OK:
                http_status = HTTPStatus.BAD_GATEWAY
                failure_class = "upstream_http_rejection"
                raise ResponsesAdapterError("upstream rejected the translated request")
            upstream_object = _json_object(upstream.body, label="upstream response")
            response = chat_to_response(upstream_object, request, binding, tool_projection)
            choices = upstream_object.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else None
            raw_finish_reason = (
                choice.get("finish_reason") if isinstance(choice, dict) else None
            )
            finish_reason = (
                raw_finish_reason
                if raw_finish_reason in {"stop", "length", "tool_calls", "content_filter"}
                else "other"
                if isinstance(raw_finish_reason, str)
                else None
            )
            usage = upstream_object.get("usage")
            completion_details = (
                usage.get("completion_tokens_details") if isinstance(usage, dict) else None
            )
            response_shape = {
                **_response_shape(response),
                "upstream_finish_reason": finish_reason,
                "upstream_output_tokens": (
                    _int_or_zero(usage.get("completion_tokens"))
                    if isinstance(usage, dict)
                    else 0
                ),
                "upstream_reasoning_tokens": (
                    _int_or_zero(completion_details.get("reasoning_tokens"))
                    if isinstance(completion_details, dict)
                    else 0
                ),
            }
            status = "passed"
            failure_class = None
            http_status = HTTPStatus.OK
            # Commit the signed audit record before exposing response bytes to
            # the client, avoiding an observation race and fail-open logging.
            record()
            if request.get("stream") is True:
                payload = _sse_bytes(response)
                handler.send_response(HTTPStatus.OK)
                handler.send_header("Content-Type", "text/event-stream")
                handler.send_header("Content-Length", str(len(payload)))
                handler.send_header("Cache-Control", "no-store")
                handler.send_header("Connection", "close")
                handler.end_headers()
                handler.wfile.write(payload)
            elif request.get("stream") in (False, None):
                self._send_json(handler, HTTPStatus.OK, response)
            else:
                raise ResponsesAdapterError("Responses stream flag differs")
        except ResponsesAdapterError as exc:
            if failure_class is None:
                failure_class = "adapter_contract_error"
            failure_message_blake3 = blake3_hex(str(exc))
            record()
            self._send_json(
                handler,
                http_status,
                {
                    "error": {
                        "type": failure_class,
                        "message": str(exc),
                        "request_id": request_id,
                    }
                },
            )
        finally:
            if not recorded:
                record()


def adapter_receipts_document(gateway: ResponsesAdapterGateway) -> dict[str, Any]:
    return gateway.receipts_document()


__all__ = [
    "AdapterLimits",
    "AdapterReceiptSigner",
    "AdapterRouteBinding",
    "ChatCompletionsTransport",
    "PROJECTION_VERSION",
    "ResponsesAdapterError",
    "ResponsesAdapterGateway",
    "SignedAdapterReceipt",
    "adapter_receipts_document",
    "chat_to_response",
    "responses_to_chat",
    "verify_adapter_receipt",
]
