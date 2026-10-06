"""Call-scoped, policy-visible execution diagnostics outside canonical results."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .contracts import freeze_json
from .digests import blake3_bytes, blake3_hex, canonical_json_bytes, canonical_value


DIAGNOSTIC_SCHEMA = "eva.execution-diagnostic.v1"
DIAGNOSTIC_PREFIX = "Host execution diagnostic (untrusted program output):\n"
MAXIMUM_STREAM_BYTES = 4096
MAXIMUM_CONTENT_BYTES = 2048


@dataclass
class DiagnosticScope:
    call_id: str
    tool: str
    content: tuple = ()


_SCOPE: ContextVar[DiagnosticScope | None] = ContextVar("eva_execution_diagnostic", default=None)


@contextmanager
def diagnostic_scope(call_id: str, tool: str):
    scope = DiagnosticScope(call_id, tool)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def diagnostics_requested() -> bool:
    scope = _SCOPE.get()
    return scope is not None and scope.tool == "execute_code"


def _safe_text(text: str) -> str:
    # Diagnostics originate in the bounded public sandbox, never host env dumps.
    # Still redact common secret spellings and host-only paths in error output.
    text = re.sub(r"-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*", "[redacted private key]", text)
    text = re.sub(r'''(?im)((?:["']?authorization["']?)\s*[:=]\s*)[^\r\n]+''', r"\1[redacted]", text)
    text = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[redacted]", text)
    text = re.sub(r'''(?im)((?:["']?(?:api[_-]?key|access[_-]?token|password|secret)["']?)\s*[:=]\s*)[^\r\n]+''', r"\1[redacted]", text)
    text = re.sub(r"\b(?:sk-|hf_)[A-Za-z0-9_-]{12,}", "[redacted token]", text)
    text = re.sub(r"(https?://)[^\s/@:]+:[^\s/@]+@", r"\1[redacted]@", text)
    text = re.sub(r"(?<![\w/])/(?:localhome|home|root|etc|run|proc|sys|var)(?:/[^\s\"'<>:;,)]*)?", "[host-path]", text)
    return text.encode("utf-8")[-MAXIMUM_STREAM_BYTES:].decode("utf-8", errors="ignore")


def _display_content(document: Mapping[str, Any]) -> tuple:
    """Prioritize stderr within the existing context budget, retaining exact text."""
    core = {key: value for key, value in document.items() if key != "diagnostic_blake3"}
    core["capture_blake3"] = document["diagnostic_blake3"]
    core["streams"] = {
        name: {**document["streams"][name], "display_truncated": False}
        for name in ("stdout", "stderr")
    }
    while True:
        payload = canonical_json_bytes({**core, "diagnostic_blake3": blake3_hex(core)})
        rendered = DIAGNOSTIC_PREFIX + payload.decode("utf-8")
        overflow = len(rendered.encode("utf-8")) - MAXIMUM_CONTENT_BYTES
        if overflow <= 0:
            return (freeze_json({"type": "text", "text": rendered}),)
        for name in ("stdout", "stderr"):
            row = core["streams"][name]
            raw = row["text"].encode("utf-8")
            if raw:
                row["text"] = raw[min(len(raw), overflow):].decode("utf-8", errors="ignore")
                row["display_truncated"] = True
                break
        else:
            raise ValueError("execution diagnostic metadata exceeds display budget")


def record_execution_diagnostics(capture: Mapping[str, Any], *, episode_dir: Path,
                                 stage: str, attempt: int) -> bool:
    """Persist only actual captured output; optional diagnostics never alter results."""
    scope = _SCOPE.get()
    if scope is None or scope.tool != "execute_code" or not capture:
        return False
    try:
        streams = {}
        for name in ("stdout", "stderr"):
            row = capture[name]
            if (set(row) != {"text", "byte_count", "truncated"}
                    or not isinstance(row["text"], str)
                    or type(row["byte_count"]) is not int or row["byte_count"] < 0
                    or type(row["truncated"]) is not bool):
                return False
            text = _safe_text(row["text"])
            streams[name] = {"text": text, "byte_count": row["byte_count"],
                             "truncated": row["truncated"] or len(row["text"].encode("utf-8")) > MAXIMUM_STREAM_BYTES}
        core = {"schema": DIAGNOSTIC_SCHEMA, "call_id": scope.call_id, "tool": scope.tool,
                "stage": stage, "attempt": attempt,
                "host_receipt_blake3": blake3_bytes((episode_dir / "host-receipt.json").read_bytes()),
                "streams": streams}
        document = {**core, "diagnostic_blake3": blake3_hex(core)}
        payload = canonical_json_bytes(document)
        with (episode_dir / "execution-diagnostics.json").open("xb") as stream:
            stream.write(payload)
        (episode_dir / "execution-diagnostics.json").chmod(0o400)
        scope.content = _display_content(document)
        return True
    except (OSError, KeyError, TypeError, ValueError):
        # A missing/invalid supplemental capture must not become a fabricated
        # tool result or replace an already completed signed execution outcome.
        return False


def diagnostic_content(native_output: Any, *, call_id: str) -> tuple:
    """Read the supplemental native content for policy/Judge projection."""
    output = canonical_value(native_output)
    result = output.get("result", {}) if isinstance(output, dict) else {}
    content = result.get("content", []) if isinstance(result, dict) else []
    selected = []
    for block in content:
        if not isinstance(block, dict) or not isinstance(block.get("text"), str):
            continue
        if not block["text"].startswith(DIAGNOSTIC_PREFIX):
            continue
        document = json.loads(block["text"][len(DIAGNOSTIC_PREFIX):])
        core = {k: v for k, v in document.items() if k != "diagnostic_blake3"}
        if (document.get("schema") != DIAGNOSTIC_SCHEMA or document.get("call_id") != call_id
                or document.get("tool") != "execute_code"
                or document.get("diagnostic_blake3") != blake3_hex(core)):
            raise ValueError("execution diagnostic call binding differs")
        selected.append(freeze_json(block))
    return tuple(selected)
