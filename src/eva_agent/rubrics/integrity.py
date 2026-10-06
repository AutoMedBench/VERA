"""Canonical BLAKE3 integrity helpers for rubric artifacts.

Content commitments are BLAKE3 digests. Runtime identities are UUIDs and are
deliberately kept out of this module's digest namespace.
"""

from __future__ import annotations

import json
from typing import Any


class IntegrityDependencyError(RuntimeError):
    """A required content-integrity implementation is unavailable."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical JSON encoding used by rubric artifacts."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("rubric value is not canonical JSON") from exc
    return encoded.encode("utf-8")


def blake3_bytes(payload: bytes) -> str:
    """Return a tagged, lowercase BLAKE3-256 digest."""

    if not isinstance(payload, bytes):
        raise TypeError("BLAKE3 payload must be bytes")
    try:
        from blake3 import blake3
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise IntegrityDependencyError(
            "rubric integrity requires the 'blake3' package; no alternate digest is allowed"
        ) from exc
    return f"blake3:{blake3(payload).hexdigest()}"


def blake3_document(value: Any) -> str:
    """Commit a JSON-compatible value in the rubric canonical domain."""

    return blake3_bytes(canonical_json_bytes(value))


def is_blake3_digest(value: object) -> bool:
    """Return whether *value* is a tagged BLAKE3-256 digest."""

    return (
        isinstance(value, str)
        and value.startswith("blake3:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )
