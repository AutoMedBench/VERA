"""Canonical BLAKE3 receipts for pipeline artifacts.

Application-level identities are UUIDs.  Content integrity is always BLAKE3;
there is deliberately no SHA fallback because silently changing digest domains
would make persisted evidence unverifiable.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import base64
import json
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from blake3 import blake3
except ImportError as exc:  # pragma: no cover - exercised by packaging, not tests
    raise RuntimeError("the EVA-Agent pipeline requires the 'blake3' package") from exc


class CanonicalValueError(ValueError):
    """A value cannot be represented in the pipeline's canonical JSON domain."""


def canonical_value(value: Any) -> Any:
    """Convert supported immutable values into deterministic JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalValueError("non-finite floats are not canonical")
        return value
    if isinstance(value, bytes):
        return {"$bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Enum):
        return canonical_value(value.value)
    if isinstance(value, Path):
        raise CanonicalValueError("filesystem paths must be projected explicitly")
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CanonicalValueError("canonical mappings require string keys")
        return {key: canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [canonical_value(item) for item in value]
    raise CanonicalValueError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes with a single trailing newline."""

    try:
        return (
            json.dumps(
                canonical_value(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, CanonicalValueError):
            raise
        raise CanonicalValueError("value is not canonical JSON") from exc


def blake3_hex(value: Any) -> str:
    """Digest a canonical value in the pipeline BLAKE3 content domain."""

    return blake3(canonical_json_bytes(value)).hexdigest()


def blake3_bytes(payload: bytes) -> str:
    """Digest literal file bytes without JSON projection."""

    return blake3(payload).hexdigest()


def is_blake3(value: Any) -> bool:
    """Recognize both EVA pipeline and rubric-registry BLAKE3 spellings.

    Pipeline receipts use the bare 64-character form.  The independently
    compiled rubric registry deliberately tags its digest as ``blake3:``.
    Bindings retain the compiler's exact bytes, so accepting the tag here is
    preferable to silently translating between digest domains.
    """

    if isinstance(value, str) and value.startswith("blake3:"):
        value = value[7:]
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "CanonicalValueError",
    "blake3_bytes",
    "blake3_hex",
    "canonical_json_bytes",
    "canonical_value",
    "is_blake3",
]
