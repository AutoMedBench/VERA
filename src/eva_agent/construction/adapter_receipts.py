"""Crash-durable premium-construction adapter receipt evidence.

The generic Responses adapter deliberately keeps receipts in memory.  Premium
construction installs this append-only sink so every provider-bound Opus 5
receipt is durably committed before the corresponding response can reach
Codex.  The signed receipt schema and signature domain remain unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID

from eva_agent.codex_providers import SignedAdapterReceipt, verify_adapter_receipt
from eva_agent.pipeline.digests import (
    blake3_hex,
    canonical_json_bytes,
    is_blake3,
)

from .legacy_exact import _ensure_directory, _open_directory_chain


_RECEIPT_DIRECTORY = "adapter-receipts-v1"
_SESSION_DIRECTORY = "session-catalogs"
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_SESSION_REFERENCE_BYTES = 64 * 1024 * 1024


class PremiumConstructionAdapterReceiptError(ValueError):
    """Durable premium-construction adapter evidence failed closed."""


def _signed_receipt(document: Any) -> SignedAdapterReceipt:
    expected = {
        "schema",
        "signature_domain",
        "payload",
        "payload_blake3",
        "key_id",
        "public_key_base64",
        "public_key_blake3",
        "algorithm",
        "signature_base64",
        "envelope_blake3",
    }
    if (
        not isinstance(document, Mapping)
        or set(document) != expected
        or not isinstance(document.get("payload"), Mapping)
    ):
        raise PremiumConstructionAdapterReceiptError(
            "durable adapter receipt envelope differs"
        )
    try:
        receipt = SignedAdapterReceipt(
            payload=MappingProxyType(dict(document["payload"])),
            payload_blake3=document["payload_blake3"],
            key_id=document["key_id"],
            public_key_base64=document["public_key_base64"],
            public_key_blake3=document["public_key_blake3"],
            algorithm=document["algorithm"],
            signature_base64=document["signature_base64"],
            envelope_blake3=document["envelope_blake3"],
        )
        verify_adapter_receipt(receipt)
    except Exception as exc:
        raise PremiumConstructionAdapterReceiptError(
            "durable adapter receipt signature differs"
        ) from exc
    return receipt


def _canonical_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PremiumConstructionAdapterReceiptError(
            f"{label} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise PremiumConstructionAdapterReceiptError(f"{label} is not canonical JSON")
    return MappingProxyType(value)


def _secure_directory(path: Path) -> Path:
    value = _ensure_directory(path, label="premium adapter receipt directory")
    descriptor = _open_directory_chain(value, create=False)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.getuid() or not stat.S_ISDIR(metadata.st_mode):
            raise PremiumConstructionAdapterReceiptError(
                "premium adapter receipt directory ownership differs"
            )
        os.fchmod(descriptor, 0o700)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = _open_directory_chain(value.parent, create=False)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return value


def _read_file(
    parent: Path, name: str, *, label: str, maximum_bytes: int
) -> bytes:
    directory = _open_directory_chain(parent, create=False)
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
        metadata = os.fstat(descriptor)
        if not (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o400
            and 0 < metadata.st_size <= maximum_bytes
        ):
            raise PremiumConstructionAdapterReceiptError(f"{label} topology differs")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size or len(payload) > maximum_bytes:
            raise PremiumConstructionAdapterReceiptError(f"{label} bounded read differs")
        return payload
    except OSError as exc:
        raise PremiumConstructionAdapterReceiptError(
            f"{label} cannot be reopened without following links"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory)


def _write_new(parent: Path, name: str, payload: bytes) -> None:
    directory = _open_directory_chain(parent, create=False)
    descriptor = -1
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(name, flags, 0o600, dir_fd=directory)
        metadata = os.fstat(descriptor)
        if not (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
        ):
            raise PremiumConstructionAdapterReceiptError(
                "durable adapter receipt target topology differs"
            )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise PremiumConstructionAdapterReceiptError(
                    "durable adapter receipt write stalled"
                )
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        # The file and its directory entry are both durable before returning
        # to the HTTP adapter and therefore before Codex sees a response.
        os.fsync(directory)
    except FileExistsError:
        raise PremiumConstructionAdapterReceiptError(
            "durable adapter receipt already exists"
        ) from None
    except OSError as exc:
        raise PremiumConstructionAdapterReceiptError(
            "durable adapter receipt commit failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory)


class PremiumConstructionAdapterReceiptStore:
    """One verified append-only catalog for an exact Opus adapter binding."""

    def __init__(
        self,
        state_root: str | Path,
        *,
        binding_blake3: str,
        route_id: str,
        model_id: str,
        provider_family: str,
        signing_key_id: str,
        signing_public_key_blake3: str,
        effective_upstream_max_tokens: int,
    ) -> None:
        if not (
            is_blake3(binding_blake3)
            and route_id == "opus_5"
            and isinstance(model_id, str)
            and model_id
            and provider_family == "anthropic"
            and isinstance(signing_key_id, str)
            and signing_key_id
            and is_blake3(signing_public_key_blake3)
            and effective_upstream_max_tokens == 32_768
        ):
            raise PremiumConstructionAdapterReceiptError(
                "premium adapter receipt policy differs"
            )
        state = Path(os.path.abspath(os.fspath(state_root)))
        if not state.is_absolute() or ".." in state.parts:
            raise PremiumConstructionAdapterReceiptError(
                "premium adapter receipt state root differs"
            )
        self.binding_blake3 = binding_blake3
        self.route_id = route_id
        self.model_id = model_id
        self.provider_family = provider_family
        self.signing_key_id = signing_key_id
        self.signing_public_key_blake3 = signing_public_key_blake3
        self.effective_upstream_max_tokens = effective_upstream_max_tokens
        self.root = _secure_directory(state / _RECEIPT_DIRECTORY)
        self.receipt_root = _secure_directory(self.root / binding_blake3)
        self.session_root = _secure_directory(self.root / _SESSION_DIRECTORY)
        policy_core = {
            "schema": "eva.premium-construction-adapter-receipt-policy.v1",
            "binding_blake3": binding_blake3,
            "route_id": route_id,
            "model_id": model_id,
            "provider_family": provider_family,
            "signing_key_id": signing_key_id,
            "signing_public_key_blake3": signing_public_key_blake3,
            "effective_upstream_max_tokens": effective_upstream_max_tokens,
            "upstream_max_tokens_policy": "exact_override",
            "receipt_schema": "eva.codex-responses-adapter-signed-receipt.v1",
            "path_layout": (
                f"{_RECEIPT_DIRECTORY}/<binding_blake3>/<envelope_blake3>.json"
            ),
            "append_only": True,
            "file_mode": "0400",
            "file_fsync_before_response": True,
            "parent_fsync_before_response": True,
            "raw_provider_material_recorded": False,
        }
        self.policy = MappingProxyType(
            {**policy_core, "policy_blake3": blake3_hex(policy_core)}
        )
        self._lock = RLock()
        self._digests: set[str] = set()
        with self._lock:
            self._digests = set(self._scan_receipts_locked())
            self._verify_session_references_locked()

    def _verify_expected(self, receipt: SignedAdapterReceipt) -> None:
        verify_adapter_receipt(receipt)
        payload = receipt.payload
        request_shape = payload.get("request_shape")
        if not (
            receipt.key_id == self.signing_key_id
            and receipt.public_key_blake3 == self.signing_public_key_blake3
            and payload.get("route_id") == self.route_id
            and payload.get("model_id") == self.model_id
            and payload.get("provider_family") == self.provider_family
            and isinstance(request_shape, Mapping)
            and request_shape.get("effective_upstream_max_tokens")
            == self.effective_upstream_max_tokens
            and request_shape.get("upstream_max_tokens_policy") == "exact_override"
        ):
            raise PremiumConstructionAdapterReceiptError(
                "durable adapter receipt route or output-budget policy differs"
            )

    @staticmethod
    def _provider_bound(receipt: SignedAdapterReceipt) -> bool:
        shape = receipt.payload.get("request_shape")
        return isinstance(shape, Mapping) and "upstream_max_tokens_policy" in shape

    def _scan_receipts_locked(self) -> tuple[str, ...]:
        directory = _open_directory_chain(self.receipt_root, create=False)
        try:
            names = sorted(os.listdir(directory))
        finally:
            os.close(directory)
        digests: list[str] = []
        for name in names:
            if len(name) != 69 or not name.endswith(".json") or not is_blake3(name[:-5]):
                raise PremiumConstructionAdapterReceiptError(
                    "durable adapter receipt filename differs"
                )
            payload = _read_file(
                self.receipt_root,
                name,
                label="durable adapter receipt",
                maximum_bytes=_MAX_RECEIPT_BYTES,
            )
            receipt = _signed_receipt(
                _canonical_object(payload, label="durable adapter receipt")
            )
            if receipt.envelope_blake3 != name[:-5]:
                raise PremiumConstructionAdapterReceiptError(
                    "durable adapter receipt filename BLAKE3 differs"
                )
            self._verify_expected(receipt)
            digests.append(receipt.envelope_blake3)
        if len(digests) != len(set(digests)):
            raise PremiumConstructionAdapterReceiptError(
                "durable adapter receipt inventory is duplicated"
            )
        return tuple(sorted(digests))

    def commit(self, receipt: SignedAdapterReceipt) -> None:
        """Commit one provider-bound signed receipt before HTTP response."""

        if not isinstance(receipt, SignedAdapterReceipt):
            raise PremiumConstructionAdapterReceiptError(
                "durable adapter receipt type differs"
            )
        # Parser/model lookup errors have not crossed the upstream boundary;
        # they remain in the generic in-memory audit but are outside this
        # provider-completion catalog.
        if not self._provider_bound(receipt):
            return
        self._verify_expected(receipt)
        document = receipt.to_dict()
        payload = canonical_json_bytes(document)
        with self._lock:
            _write_new(
                self.receipt_root,
                f"{receipt.envelope_blake3}.json",
                payload,
            )
            self._digests.add(receipt.envelope_blake3)

    def _catalog_for(self, digests: tuple[str, ...]) -> Mapping[str, Any]:
        core = {
            "schema": "eva.premium-construction-adapter-receipt-catalog.v1",
            "binding_blake3": self.binding_blake3,
            "policy_blake3": self.policy["policy_blake3"],
            "receipt_count": len(digests),
            "receipt_envelope_blake3s": digests,
            "relative_receipt_root": (
                f"{_RECEIPT_DIRECTORY}/{self.binding_blake3}"
            ),
            "append_only": True,
            "raw_provider_material_recorded": False,
        }
        return MappingProxyType(
            {**core, "catalog_root_blake3": blake3_hex(core)}
        )

    def catalog_document(self) -> Mapping[str, Any]:
        """Reopen all immutable files and return their content-addressed root."""

        with self._lock:
            observed = self._scan_receipts_locked()
            if set(observed) != self._digests:
                raise PremiumConstructionAdapterReceiptError(
                    "durable adapter receipt inventory changed unexpectedly"
                )
            return self._catalog_for(observed)

    def _verify_catalog(self, value: Any, *, available: set[str]) -> None:
        if not isinstance(value, Mapping):
            raise PremiumConstructionAdapterReceiptError(
                "session adapter catalog differs"
            )
        expected = {
            "schema",
            "binding_blake3",
            "policy_blake3",
            "receipt_count",
            "receipt_envelope_blake3s",
            "relative_receipt_root",
            "append_only",
            "raw_provider_material_recorded",
            "catalog_root_blake3",
        }
        if set(value) != expected:
            raise PremiumConstructionAdapterReceiptError(
                "session adapter catalog keys differ"
            )
        digests = tuple(value["receipt_envelope_blake3s"])
        expected_catalog = self._catalog_for(digests)
        if (
            canonical_json_bytes(value) != canonical_json_bytes(expected_catalog)
            or len(digests) != len(set(digests))
            or tuple(sorted(digests)) != digests
            or not set(digests).issubset(available)
        ):
            raise PremiumConstructionAdapterReceiptError(
                "session adapter catalog root differs"
            )

    def _verify_session_references_locked(self) -> None:
        directory = _open_directory_chain(self.session_root, create=False)
        try:
            names = sorted(os.listdir(directory))
        finally:
            os.close(directory)
        for name in names:
            if not name.endswith(".json"):
                raise PremiumConstructionAdapterReceiptError(
                    "adapter session reference filename differs"
                )
            try:
                session_id = str(UUID(name[:-5]))
            except (ValueError, AttributeError) as exc:
                raise PremiumConstructionAdapterReceiptError(
                    "adapter session reference identity differs"
                ) from exc
            if session_id != name[:-5]:
                raise PremiumConstructionAdapterReceiptError(
                    "adapter session reference filename differs"
                )
            payload = _read_file(
                self.session_root,
                name,
                label="adapter session reference",
                maximum_bytes=_MAX_SESSION_REFERENCE_BYTES,
            )
            document = _canonical_object(payload, label="adapter session reference")
            expected = {
                "schema",
                "session_id",
                "binding_blake3",
                "adapter_receipt_catalog",
                "adapter_receipt_catalog_root_blake3",
                "document_blake3",
            }
            core = {key: document[key] for key in expected if key != "document_blake3"}
            if not (
                set(document) == expected
                and document["schema"]
                == "eva.premium-construction-session-adapter-receipts.v1"
                and document["session_id"] == session_id
                and document["binding_blake3"] == self.binding_blake3
                and document["document_blake3"] == blake3_hex(core)
                and document["adapter_receipt_catalog_root_blake3"]
                == document["adapter_receipt_catalog"].get("catalog_root_blake3")
            ):
                raise PremiumConstructionAdapterReceiptError(
                    "adapter session reference commitment differs"
                )
            self._verify_catalog(
                document["adapter_receipt_catalog"], available=self._digests
            )

    def write_session_reference(self, session_id: str) -> Mapping[str, Any]:
        try:
            normalized = str(UUID(session_id))
        except (ValueError, AttributeError) as exc:
            raise PremiumConstructionAdapterReceiptError(
                "adapter session reference identity differs"
            ) from exc
        if normalized != session_id:
            raise PremiumConstructionAdapterReceiptError(
                "adapter session reference identity differs"
            )
        with self._lock:
            catalog = self.catalog_document()
            core = {
                "schema": "eva.premium-construction-session-adapter-receipts.v1",
                "session_id": session_id,
                "binding_blake3": self.binding_blake3,
                "adapter_receipt_catalog": dict(catalog),
                "adapter_receipt_catalog_root_blake3": catalog[
                    "catalog_root_blake3"
                ],
            }
            document = MappingProxyType(
                {**core, "document_blake3": blake3_hex(core)}
            )
            _write_new(
                self.session_root,
                f"{session_id}.json",
                canonical_json_bytes(document),
            )
            return document


__all__ = [
    "PremiumConstructionAdapterReceiptError",
    "PremiumConstructionAdapterReceiptStore",
]
