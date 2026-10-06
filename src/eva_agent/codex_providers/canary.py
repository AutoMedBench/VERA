"""One-attempt, no-tool Codex Responses canaries with safe BLAKE3 receipts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex

from .routes import CodexProviderConfigurationError, CodexProviderRoute


CANARY_PROMPT = "Return exactly OK. Do not use tools."
_MAX_CAPTURE_BYTES = 2_000_000


@dataclass(frozen=True, repr=False)
class CodexExecLaunch:
    """Opaque launch material; only a process runner may reveal it."""

    route: CodexProviderRoute
    codex_bin: str
    cwd: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.codex_bin or not self.cwd:
            raise CodexProviderConfigurationError("Codex executable and cwd are required")
        if not 1 <= self.timeout_seconds <= 900:
            raise CodexProviderConfigurationError("canary timeout must be in [1,900]")

    def __repr__(self) -> str:
        return (
            "CodexExecLaunch("
            f"route_id={self.route.route_id!r}, model_id={self.route.model_id!r}, "
            f"codex_bin={self.codex_bin!r}, cwd={self.cwd!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, private_values=<redacted>)"
        )

    @property
    def safe_command_shape(self) -> tuple[str, ...]:
        return (
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "--json",
            "--color=never",
            "--sandbox=read-only",
            "--skip-git-repo-check",
            "--cd=<isolated-cwd>",
            f"--model={self.route.model_id}",
            "--config=model_provider=<provider-id>",
            "--config=model_providers.<id>.name=<safe-name>",
            "--config=model_providers.<id>.base_url=<redacted>",
            "--config=model_providers.<id>.env_key=<allowlisted-env-name>",
            "--config=model_providers.<id>.requires_openai_auth=false",
            "--config=model_providers.<id>.wire_api=responses",
            "--config=model_providers.<id>.request_max_retries=0",
            "--config=model_providers.<id>.stream_max_retries=0",
            f"prompt_blake3={blake3_hex(CANARY_PROMPT)}",
        )

    def for_subprocess(self) -> tuple[tuple[str, ...], Mapping[str, str], tuple[str, ...]]:
        overrides, secret_env, redactions = self.route.config.for_subprocess()
        arguments: list[str] = [
            self.codex_bin,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "--json",
            "--color",
            "never",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--cd",
            self.cwd,
            "--model",
            self.route.model_id,
        ]
        for override in overrides:
            arguments.extend(("--config", override))
        arguments.append(CANARY_PROMPT)
        child_env: dict[str, str] = {}
        # Never forward the ambient environment wholesale.
        for name in (
            "HOME",
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
        ):
            value = os.environ.get(name)
            if value:
                child_env[name] = value
        child_env.update(secret_env)
        return tuple(arguments), MappingProxyType(child_env), redactions


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False


ProcessRunner = Callable[[CodexExecLaunch], Awaitable[ProcessOutcome]]


async def run_subprocess_once(launch: CodexExecLaunch) -> ProcessOutcome:
    argv, environment, _ = launch.for_subprocess()
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=launch.cwd,
        env=dict(environment),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=launch.timeout_seconds
        )
    except asyncio.TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        return ProcessOutcome(None, stdout, stderr, timed_out=True)
    return ProcessOutcome(process.returncode, stdout, stderr)


def _redact(data: bytes, redactions: Sequence[str]) -> bytes:
    sanitized = data
    for private in redactions:
        if private:
            sanitized = sanitized.replace(private.encode("utf-8"), b"<redacted>")
    return sanitized


def _event_summary(stdout: bytes) -> tuple[int, int, str | None, bool, bool]:
    event_count = 0
    malformed_count = 0
    final_response: str | None = None
    tool_used = False
    turn_completed = False
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed_count += 1
            continue
        if not isinstance(event, dict):
            malformed_count += 1
            continue
        event_count += 1
        event_type = str(event.get("type", "")).lower().replace("/", "_").replace(".", "_")
        if event_type in {"turn_completed", "turn_complete"}:
            turn_completed = True
        item = event.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type", "")).lower().replace("/", "_").replace(".", "_")
            if item_type in {"agent_message", "agentmessage"}:
                text = item.get("text")
                if isinstance(text, str):
                    final_response = text
            if any(
                token in item_type
                for token in (
                    "tool_call",
                    "toolcall",
                    "command_execution",
                    "commandexecution",
                    "web_search",
                    "websearch",
                    "file_change",
                    "filechange",
                    "mcp_",
                )
            ):
                tool_used = True
        if any(token in event_type for token in ("tool_call", "command_execution", "web_search")):
            tool_used = True
    return event_count, malformed_count, final_response, tool_used, turn_completed


def _failure_class(outcome: ProcessOutcome, diagnostic: bytes) -> tuple[str | None, int | None]:
    if outcome.timed_out:
        return "timeout", None
    if outcome.returncode == 0:
        return None, None
    text = diagnostic.decode("utf-8", "replace").lower()
    statuses = [int(value) for value in re.findall(r"(?<!\d)([1-5]\d\d)(?!\d)", text)]
    http_status = next((value for value in statuses if 400 <= value <= 599), None)
    if "unknown configuration key" in text or "unrecognized" in text and "config" in text:
        return "codex_configuration_error", http_status
    if http_status in {404, 405, 415, 501} or (
        "responses" in text and any(word in text for word in ("unsupported", "not found", "unknown"))
    ):
        return "responses_protocol_incompatible", http_status
    if http_status in {401, 403} or any(word in text for word in ("unauthorized", "forbidden")):
        return "authentication_rejected", http_status
    if "model" in text and any(word in text for word in ("not found", "unknown", "unsupported", "invalid")):
        return "model_rejected", http_status
    return "provider_or_transport_error", http_status


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class CodexCanaryReceipt:
    schema: str
    created_at_utc: str
    route_id: str
    route_status: str
    model_id: str
    model_env_name: str
    registry_role_id: str | None
    provider_id: str
    wire_api: str
    request_max_retries: int
    stream_max_retries: int
    semantic_retry_count: int
    request_attempt_count: int
    credential_env_name: str
    endpoint_env_name: str
    credential_value_recorded: bool
    endpoint_value_recorded: bool
    raw_output_recorded: bool
    prompt_blake3: str
    safe_config_blake3: str
    safe_command_shape_blake3: str
    status: str
    responses_accepted: bool
    semantic_exact_ok: bool
    no_tool_calls: bool
    turn_completed: bool
    exit_code: int | None
    http_status: int | None
    failure_class: str | None
    latency_ms: int
    json_event_count: int
    malformed_jsonl_count: int
    final_response_blake3: str | None
    diagnostic_blake3: str
    receipt_blake3: str

    def __post_init__(self) -> None:
        if self.receipt_blake3 != blake3_hex(self.payload_without_digest()):
            raise CodexProviderConfigurationError("canary receipt BLAKE3 differs")

    def payload_without_digest(self) -> Mapping[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "receipt_blake3"}

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def create(cls, **values: Any) -> "CodexCanaryReceipt":
        payload = dict(values)
        payload["receipt_blake3"] = blake3_hex(payload)
        return cls(**payload)


def verify_canary_receipt(receipt: CodexCanaryReceipt) -> None:
    if receipt.schema != "eva.codex-responses-canary-receipt.v1":
        raise CodexProviderConfigurationError("canary receipt schema differs")
    if receipt.route_status not in {"direct-pass", "needs-Responses-adapter", "unavailable"}:
        raise CodexProviderConfigurationError("canary route status differs")
    if receipt.wire_api != "responses":
        raise CodexProviderConfigurationError("canary receipt wire API differs")
    if any(
        value != 0
        for value in (
            receipt.request_max_retries,
            receipt.stream_max_retries,
            receipt.semantic_retry_count,
        )
    ):
        raise CodexProviderConfigurationError("canary receipt retry policy differs")
    if receipt.request_attempt_count != 1:
        raise CodexProviderConfigurationError("canary receipt attempt count differs")
    if receipt.credential_value_recorded or receipt.endpoint_value_recorded or receipt.raw_output_recorded:
        raise CodexProviderConfigurationError("canary receipt records private material")
    expected_acceptance = (
        receipt.exit_code == 0
        and receipt.turn_completed
        and receipt.malformed_jsonl_count == 0
        and receipt.final_response_blake3 is not None
    )
    if receipt.responses_accepted != expected_acceptance:
        raise CodexProviderConfigurationError("canary acceptance claim differs")
    if receipt.semantic_exact_ok and not (receipt.responses_accepted and receipt.no_tool_calls):
        raise CodexProviderConfigurationError("canary semantic claim differs")
    if receipt.route_status == "direct-pass" and not receipt.semantic_exact_ok:
        raise CodexProviderConfigurationError("direct-pass claim differs")
    if receipt.route_status == "needs-Responses-adapter" and receipt.failure_class != "responses_protocol_incompatible":
        raise CodexProviderConfigurationError("Responses-adapter claim differs")
    if receipt.receipt_blake3 != blake3_hex(receipt.payload_without_digest()):
        raise CodexProviderConfigurationError("canary receipt BLAKE3 differs")


async def probe_route_once(
    route: CodexProviderRoute,
    *,
    cwd: str | Path,
    codex_bin: str = "codex",
    timeout_seconds: float = 120,
    runner: ProcessRunner = run_subprocess_once,
    now: Callable[[], str] = _utc_now,
    monotonic: Callable[[], float] = time.monotonic,
) -> CodexCanaryReceipt:
    """Run exactly one semantic attempt for one configured route."""

    launch = CodexExecLaunch(
        route=route,
        codex_bin=codex_bin,
        cwd=str(Path(cwd).resolve()),
        timeout_seconds=float(timeout_seconds),
    )
    started = monotonic()
    outcome = await runner(launch)
    latency_ms = max(0, int(round((monotonic() - started) * 1000)))
    _, _, redactions = launch.for_subprocess()
    sanitized_stdout = _redact(outcome.stdout, redactions)
    sanitized_stderr = _redact(outcome.stderr, redactions)
    oversized = len(sanitized_stdout) + len(sanitized_stderr) > _MAX_CAPTURE_BYTES
    if oversized:
        sanitized_stdout = sanitized_stdout[:_MAX_CAPTURE_BYTES]
        sanitized_stderr = sanitized_stderr[:_MAX_CAPTURE_BYTES]
    event_count, malformed, final_response, tool_used, turn_completed = _event_summary(sanitized_stdout)
    diagnostic = b"\n".join((sanitized_stdout, sanitized_stderr))
    failure_class, http_status = _failure_class(outcome, diagnostic)
    accepted = (
        not outcome.timed_out
        and not oversized
        and outcome.returncode == 0
        and turn_completed
        and malformed == 0
        and final_response is not None
    )
    semantic_ok = accepted and not tool_used and final_response == "OK"
    if semantic_ok:
        status, route_status = "passed", "direct-pass"
    elif failure_class == "responses_protocol_incompatible":
        status, route_status = "responses_rejected", "needs-Responses-adapter"
    elif outcome.timed_out:
        status, route_status = "timeout", "unavailable"
    elif oversized:
        status, route_status = "output_limit", "unavailable"
    elif malformed:
        status, route_status = "malformed_jsonl", "unavailable"
    elif outcome.returncode != 0:
        status, route_status = "responses_rejected", "unavailable"
    elif accepted:
        status, route_status = "semantic_failure", "unavailable"
    else:
        status, route_status = "incomplete", "unavailable"
    receipt = CodexCanaryReceipt.create(
        schema="eva.codex-responses-canary-receipt.v1",
        created_at_utc=now(),
        route_id=route.route_id,
        route_status=route_status,
        model_id=route.model_id,
        model_env_name=route.model_env_name,
        registry_role_id=route.registry_role_id,
        provider_id=route.config.provider_id,
        wire_api="responses",
        request_max_retries=0,
        stream_max_retries=0,
        semantic_retry_count=0,
        request_attempt_count=1,
        credential_env_name=route.config.credential_env_name,
        endpoint_env_name=route.config.endpoint_env_name,
        credential_value_recorded=False,
        endpoint_value_recorded=False,
        raw_output_recorded=False,
        prompt_blake3=blake3_hex(CANARY_PROMPT),
        safe_config_blake3=route.config.safe_blake3,
        safe_command_shape_blake3=blake3_hex(launch.safe_command_shape),
        status=status,
        responses_accepted=accepted,
        semantic_exact_ok=semantic_ok,
        no_tool_calls=not tool_used,
        turn_completed=turn_completed,
        exit_code=outcome.returncode,
        http_status=http_status,
        failure_class=failure_class,
        latency_ms=latency_ms,
        json_event_count=event_count,
        malformed_jsonl_count=malformed,
        final_response_blake3=(None if final_response is None else blake3_hex(final_response)),
        diagnostic_blake3=blake3_bytes(diagnostic),
    )
    verify_canary_receipt(receipt)
    return receipt


async def probe_routes_concurrently(
    routes: Mapping[str, CodexProviderRoute],
    *,
    route_ids: Sequence[str],
    cwd: str | Path,
    codex_bin: str = "codex",
    timeout_seconds: float = 120,
    runner: ProcessRunner = run_subprocess_once,
) -> tuple[CodexCanaryReceipt, ...]:
    """Probe configured routes concurrently, once each, preserving input order."""

    requested = tuple(route_ids)
    if len(set(requested)) != len(requested):
        raise CodexProviderConfigurationError("canary route request is duplicated")
    missing = [route_id for route_id in requested if route_id not in routes]
    if missing:
        raise CodexProviderConfigurationError(
            f"canary routes are not configured: {', '.join(missing)}"
        )
    receipts = await asyncio.gather(
        *(
            probe_route_once(
                routes[route_id],
                cwd=cwd,
                codex_bin=codex_bin,
                timeout_seconds=timeout_seconds,
                runner=runner,
            )
            for route_id in requested
        )
    )
    return tuple(receipts)


def receipts_document(receipts: Sequence[CodexCanaryReceipt]) -> dict[str, Any]:
    payload = {
        "schema": "eva.codex-responses-canary-receipts.v1",
        "prompt_blake3": blake3_hex(CANARY_PROMPT),
        "receipt_count": len(receipts),
        "receipts": [receipt.to_dict() for receipt in receipts],
        "raw_outputs_recorded": False,
        "credential_values_recorded": False,
        "endpoint_values_recorded": False,
    }
    payload["document_blake3"] = blake3_hex(payload)
    return payload


__all__ = [
    "CANARY_PROMPT",
    "CodexCanaryReceipt",
    "CodexExecLaunch",
    "ProcessOutcome",
    "ProcessRunner",
    "probe_route_once",
    "probe_routes_concurrently",
    "receipts_document",
    "run_subprocess_once",
    "verify_canary_receipt",
]
