"""Read-only construction-readiness authority from signed EvaMed v24.

This adapter is deliberately narrower than the legacy supervisor.  It opens
the signed registry, status, checkpoint chain, and *construction* proofs only.
It never opens rollout, panel, judge, review, replay, or admission artifacts.
Rejected is used solely as a signed terminal exclusion bucket; rejection
evidence is not dereferenced and cannot influence ordering within that bucket.

SHA-256 names in this module are compatibility checks for the immutable legacy
authority.  Every new EVA receipt and identity uses BLAKE3 or UUID.
"""

from __future__ import annotations

import base64
import binascii
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex


REGISTRY_FILENAME = "candidate-registry.v24.json"
REGISTRY_ATTESTATION_FILENAME = "candidate-registry-host-attestation.json"
STATUS_FILENAME = "status.v24.json"
CHECKPOINT_DIRECTORY = "checkpoints"
REGISTRY_SCHEMA = "rlevo.med-research-campaign-supervisor-registry.v1"
STATUS_SCHEMA = "rlevo.med-research-campaign-supervisor-status.v1"
RECEIPT_SCHEMA = "rlevo.med-research-signed-host-receipt.v1"
PAYLOAD_SCHEMA = "rlevo.med-research-host-receipt-payload.v1"
CHECKPOINT_SCHEMA = "rlevo.med-research-campaign-supervisor-checkpoint.v1"
TRUST_STORE_SCHEMA = "rlevo.med-research-host-trust-store.v1"
CAMPAIGN_ID = "evamed-6000-campaign-supervisor-v1"
EXPECTED_REVISION = 24
EXPECTED_CANDIDATES = 9_000
STAGES = frozenset({"S1", "S2", "S3", "S4", "S5", "E2E"})
FAMILIES = frozenset(
    {"agentclinic", "automedbench", "healthbench-professional", "medxpertqa"}
)
EXPECTED_STATES = MappingProxyType(
    {
        "admitted": 0,
        "constructing": 0,
        "frozen": 7_645,
        "panelled": 0,
        "preregistered": 0,
        "promoted": 1_344,
        "rejected": 11,
        "replayed": 0,
        "reviewed": 0,
        "verified": 0,
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT = re.compile(
    r"^checkpoint-(?P<generation>[0-9]{8})-(?P<digest>[0-9a-f]{16})\.json$"
)
_MAX_JSON_BYTES = 40 * 1024 * 1024
_CONSTRUCTION_PROOFS = ("constructing", "verified", "promoted")


class SupervisorV24ReadinessError(ValueError):
    """The signed v24 construction-readiness authority failed closed."""


class ConstructionReadiness(str, Enum):
    PROMOTED = "promoted"
    FROZEN = "frozen"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class SignedReadinessRow:
    candidate_id: str
    source_family: str
    stage: str
    readiness: ConstructionReadiness
    proof_root_blake3: str

    def to_document(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "source_family": self.source_family,
            "stage": self.stage,
            "readiness": self.readiness.value,
            "proof_root_blake3": self.proof_root_blake3,
        }


@dataclass(frozen=True, slots=True)
class SignedSupervisorV24Readiness:
    """Compact, immutable projection of the verified signed authority."""

    campaign_id: str
    registry_revision: int
    candidate_set_upstream_sha256: str
    registry_logical_upstream_sha256: str
    registry_file_upstream_sha256: str
    registry_file_blake3: str
    registry_attestation_file_upstream_sha256: str
    registry_attestation_file_blake3: str
    status_file_upstream_sha256: str
    status_file_blake3: str
    checkpoint_head_file_upstream_sha256: str
    checkpoint_head_file_blake3: str
    checkpoint_chain_blake3: str
    trust_store_file_upstream_sha256: str
    trust_store_file_blake3: str
    proof_roots_blake3: str
    verified_construction_proofs_blake3: str
    rows: tuple[SignedReadinessRow, ...]
    authority_blake3: str

    @property
    def candidate_count(self) -> int:
        return len(self.rows)

    @property
    def readiness_by_candidate(self) -> Mapping[str, SignedReadinessRow]:
        return MappingProxyType({row.candidate_id: row for row in self.rows})

    def core_document(self) -> dict[str, Any]:
        return {
            "schema": "eva.signed-supervisor-v24-construction-readiness.v1",
            "campaign_id": self.campaign_id,
            "registry_revision": self.registry_revision,
            "candidate_count": len(self.rows),
            "candidate_set_upstream_sha256": self.candidate_set_upstream_sha256,
            "registry_logical_upstream_sha256": self.registry_logical_upstream_sha256,
            "registry_file_upstream_sha256": self.registry_file_upstream_sha256,
            "registry_file_blake3": self.registry_file_blake3,
            "registry_attestation_file_upstream_sha256": (
                self.registry_attestation_file_upstream_sha256
            ),
            "registry_attestation_file_blake3": self.registry_attestation_file_blake3,
            "status_file_upstream_sha256": self.status_file_upstream_sha256,
            "status_file_blake3": self.status_file_blake3,
            "checkpoint_head_file_upstream_sha256": (
                self.checkpoint_head_file_upstream_sha256
            ),
            "checkpoint_head_file_blake3": self.checkpoint_head_file_blake3,
            "checkpoint_chain_blake3": self.checkpoint_chain_blake3,
            "trust_store_file_upstream_sha256": self.trust_store_file_upstream_sha256,
            "trust_store_file_blake3": self.trust_store_file_blake3,
            "proof_roots_blake3": self.proof_roots_blake3,
            "verified_construction_proofs_blake3": (
                self.verified_construction_proofs_blake3
            ),
            "state_counts": dict(EXPECTED_STATES),
        }

    def to_document(self) -> dict[str, Any]:
        return {**self.core_document(), "authority_blake3": self.authority_blake3}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SupervisorV24ReadinessError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SupervisorV24ReadinessError(f"non-finite JSON constant is forbidden: {value}")


def _canonical_legacy_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise SupervisorV24ReadinessError("legacy value is not canonical JSON") from exc


def _legacy_sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _legacy_sha256_value(value: Any) -> str:
    return _legacy_sha256_bytes(_canonical_legacy_bytes(value))


def _exact_mapping(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise SupervisorV24ReadinessError(f"{label} keys differ")
    return value


def _upstream_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SupervisorV24ReadinessError(f"{label} is not a legacy SHA-256")
    return value


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SupervisorV24ReadinessError(f"{label} must be UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise SupervisorV24ReadinessError(f"{label} is invalid") from None
    return parsed


def _validate_root(path: str | Path, *, label: str) -> Path:
    target = Path(path)
    try:
        metadata = target.lstat()
    except OSError as exc:
        raise SupervisorV24ReadinessError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SupervisorV24ReadinessError(f"{label} must be a real directory")
    return target.resolve(strict=True)


def _safe_existing_file(path: Path, *, root: Path, label: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise SupervisorV24ReadinessError(f"{label} escapes the authority root") from None
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise SupervisorV24ReadinessError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SupervisorV24ReadinessError(f"{label} traverses a symlink")
    if not stat.S_ISREG(cursor.lstat().st_mode):
        raise SupervisorV24ReadinessError(f"{label} must be a regular file")
    return cursor


def _read_file(
    path: Path,
    *,
    root: Path,
    label: str,
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> bytes:
    target = _safe_existing_file(path, root=root, label=label)
    try:
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise SupervisorV24ReadinessError(f"{label} cannot be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum_bytes:
            raise SupervisorV24ReadinessError(f"{label} size differs")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise SupervisorV24ReadinessError(f"{label} exceeds the bounded size")
        return payload
    finally:
        os.close(descriptor)


def _decode_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except SupervisorV24ReadinessError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SupervisorV24ReadinessError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise SupervisorV24ReadinessError(f"{label} must be a JSON object")
    return value


def _read_json(
    path: Path, *, root: Path, label: str, maximum_bytes: int = _MAX_JSON_BYTES
) -> tuple[Mapping[str, Any], bytes]:
    payload = _read_file(path, root=root, label=label, maximum_bytes=maximum_bytes)
    return _decode_json(payload, label=label), payload


def _verify_receipt(
    receipt: Mapping[str, Any], *, trust_keys: Mapping[str, str], label: str
) -> Mapping[str, Any]:
    row = _exact_mapping(
        receipt,
        {"schema", "key_id", "payload", "payload_sha256", "signature_base64"},
        label=label,
    )
    if row["schema"] != RECEIPT_SCHEMA:
        raise SupervisorV24ReadinessError(f"{label} schema differs")
    key_id = row["key_id"]
    if not isinstance(key_id, str) or key_id not in trust_keys:
        raise SupervisorV24ReadinessError(f"{label} key is not trusted")
    payload = _exact_mapping(
        row["payload"],
        {"schema", "sandbox_id", "episode_id", "focus", "issued_at_utc", "observations", "events"},
        label=f"{label} payload",
    )
    if payload["schema"] != PAYLOAD_SCHEMA:
        raise SupervisorV24ReadinessError(f"{label} payload schema differs")
    _parse_utc(payload["issued_at_utc"], label=f"{label} issuance")
    payload_bytes = _canonical_legacy_bytes(payload)
    payload_sha256 = _legacy_sha256_bytes(payload_bytes)
    if row["payload_sha256"] != payload_sha256:
        raise SupervisorV24ReadinessError(f"{label} payload hash differs")
    try:
        public = base64.b64decode(trust_keys[key_id], validate=True)
        signature = base64.b64decode(row["signature_base64"], validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise SupervisorV24ReadinessError(f"{label} base64 differs") from exc
    if len(public) != 32 or len(signature) != 64:
        raise SupervisorV24ReadinessError(f"{label} signature size differs")
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(signature, payload_bytes)
    except (InvalidSignature, ValueError) as exc:
        raise SupervisorV24ReadinessError(f"{label} signature differs") from exc
    events = payload["events"]
    if not isinstance(events, list) or [event.get("ordinal") for event in events] != list(
        range(len(events))
    ):
        raise SupervisorV24ReadinessError(f"{label} event ordinals differ")
    return payload


def _safe_reference(
    value: Any, *, authority_root: Path, label: str
) -> tuple[Path, str, str]:
    ref = _exact_mapping(value, {"path", "sha256"}, label=label)
    relative = ref["path"]
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise SupervisorV24ReadinessError(f"{label} path differs")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or pure.as_posix() != relative or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise SupervisorV24ReadinessError(f"{label} path is unsafe")
    digest = _upstream_sha256(ref["sha256"], label=f"{label} hash")
    path = _safe_existing_file(authority_root / pure, root=authority_root, label=label)
    return path, relative, digest


def _resolve(value: Any, path: Sequence[str | int], *, label: str) -> Any:
    current = value
    for part in path:
        if isinstance(part, str) and isinstance(current, Mapping) and part in current:
            current = current[part]
        elif type(part) is int and isinstance(current, list) and part < len(current):
            current = current[part]
        else:
            raise SupervisorV24ReadinessError(f"{label} assertion path is unresolved")
    return current


def _assert_claims(value: Any, claims: Any, *, label: str) -> None:
    if not isinstance(claims, list) or len(claims) > 64:
        raise SupervisorV24ReadinessError(f"{label} claims differ")
    for ordinal, raw in enumerate(claims):
        claim = _exact_mapping(raw, {"path", "equals"}, label=f"{label}[{ordinal}]")
        path = claim["path"]
        if not isinstance(path, list) or not path or len(path) > 32:
            raise SupervisorV24ReadinessError(f"{label}[{ordinal}] path differs")
        actual = _resolve(value, path, label=f"{label}[{ordinal}]")
        if type(actual) is not type(claim["equals"]) or actual != claim["equals"]:
            raise SupervisorV24ReadinessError(f"{label}[{ordinal}] differs")


def _verify_construction_proof(
    *,
    candidate_id: str,
    stage: str,
    proof_name: str,
    proof: Any,
    authority_root: Path,
    trust_keys: Mapping[str, str],
) -> dict[str, str]:
    row = _exact_mapping(
        proof,
        {
            "attestation",
            "subject",
            "subject_binding",
            "attestation_kind",
            "event_kind",
            "expected_sandbox_id",
            "expected_focus",
            "required_observations",
            "required_subject",
        },
        label=f"{candidate_id} {proof_name} proof",
    )
    if row["expected_focus"] != stage:
        raise SupervisorV24ReadinessError(f"{candidate_id} {proof_name} focus differs")
    attestation_path, attestation_relative, expected_attestation_sha = _safe_reference(
        row["attestation"], authority_root=authority_root, label=f"{candidate_id} {proof_name} attestation"
    )
    attestation, attestation_bytes = _read_json(
        attestation_path,
        root=authority_root,
        label=f"{candidate_id} {proof_name} attestation",
    )
    if _legacy_sha256_bytes(attestation_bytes) != expected_attestation_sha:
        raise SupervisorV24ReadinessError(f"{candidate_id} {proof_name} attestation hash differs")
    payload = _verify_receipt(
        attestation,
        trust_keys=trust_keys,
        label=f"{candidate_id} {proof_name} attestation",
    )
    observations = payload["observations"]
    if not isinstance(observations, Mapping):
        raise SupervisorV24ReadinessError(f"{candidate_id} {proof_name} observations differ")
    event = {
        "ordinal": 0,
        "kind": row["event_kind"],
        "payload_sha256": _legacy_sha256_value(observations),
    }
    if not (
        payload["sandbox_id"] == row["expected_sandbox_id"]
        and payload["focus"] == stage
        and observations.get("attestation_kind") == row["attestation_kind"]
        and payload["events"] == [event]
    ):
        raise SupervisorV24ReadinessError(f"{candidate_id} {proof_name} identity differs")
    _assert_claims(
        observations,
        row["required_observations"],
        label=f"{candidate_id} {proof_name} observations",
    )
    if not (
        observations.get("candidate_id") == candidate_id
        and observations.get("proof_name") == proof_name
        and observations.get("derived_state") == proof_name
        and observations.get("campaign_id") == CAMPAIGN_ID
        and observations.get("passed") is True
    ):
        raise SupervisorV24ReadinessError(f"{candidate_id} {proof_name} state differs")
    subject_path, subject_relative, expected_subject_sha = _safe_reference(
        row["subject"], authority_root=authority_root, label=f"{candidate_id} {proof_name} subject"
    )
    subject, subject_bytes = _read_json(
        subject_path,
        root=authority_root,
        label=f"{candidate_id} {proof_name} subject",
    )
    if _legacy_sha256_bytes(subject_bytes) != expected_subject_sha:
        raise SupervisorV24ReadinessError(f"{candidate_id} {proof_name} subject hash differs")
    binding = _exact_mapping(
        row["subject_binding"],
        {"digest", "observation_path"},
        label=f"{candidate_id} {proof_name} subject binding",
    )
    expected_digest = (
        expected_subject_sha
        if binding["digest"] == "file_sha256"
        else _legacy_sha256_value(subject)
    )
    if binding["digest"] not in {"file_sha256", "canonical_json_sha256"}:
        raise SupervisorV24ReadinessError(f"{candidate_id} {proof_name} binding digest differs")
    if _resolve(observations, binding["observation_path"], label="subject binding") != expected_digest:
        raise SupervisorV24ReadinessError(f"{candidate_id} {proof_name} subject binding differs")
    _assert_claims(
        subject,
        row["required_subject"],
        label=f"{candidate_id} {proof_name} subject",
    )
    if not (
        subject.get("candidate_id") == candidate_id
        and subject.get("focus") == stage
        and subject.get("proof_name") == proof_name
        and subject.get("derived_state") == proof_name
        and subject.get("campaign_id") == CAMPAIGN_ID
    ):
        raise SupervisorV24ReadinessError(f"{candidate_id} {proof_name} subject state differs")
    return {
        "candidate_id": candidate_id,
        "proof_name": proof_name,
        "attestation_path": attestation_relative,
        "attestation_upstream_sha256": expected_attestation_sha,
        "attestation_blake3": blake3_bytes(attestation_bytes),
        "subject_path": subject_relative,
        "subject_upstream_sha256": expected_subject_sha,
        "subject_blake3": blake3_bytes(subject_bytes),
        "issued_at_utc": payload["issued_at_utc"],
    }


def _candidate_commitment(candidate: Mapping[str, Any]) -> str:
    return _legacy_sha256_value(
        {
            "candidate_id": candidate["candidate_id"],
            "family": candidate["family"],
            "focus": candidate["focus"],
            "source_shard": candidate["source_shard"],
            "harness": candidate["harness"],
            "frozen_artifact": candidate["frozen_artifact"],
            "frozen_assertions": candidate["frozen_assertions"],
        }
    )


def _candidate_set_sha256(candidates: Sequence[Mapping[str, Any]]) -> str:
    return _legacy_sha256_value(
        sorted(
            (
                {
                    "candidate_id": row["candidate_id"],
                    "candidate_commitment_sha256": _candidate_commitment(row),
                }
                for row in candidates
            ),
            key=lambda row: row["candidate_id"],
        )
    )


def _load_trust_store(
    path: Path, *, authority_root: Path
) -> tuple[Mapping[str, str], bytes]:
    value, payload = _read_json(path, root=authority_root, label="host trust store", maximum_bytes=1024 * 1024)
    row = _exact_mapping(
        value,
        {"schema", "created_at_utc", "algorithm", "status", "keys"},
        label="host trust store",
    )
    if not (
        row["schema"] == TRUST_STORE_SCHEMA
        and row["algorithm"] == "Ed25519"
        and row["status"] == "active"
        and isinstance(row["keys"], Mapping)
        and row["keys"]
    ):
        raise SupervisorV24ReadinessError("host trust store differs")
    _parse_utc(row["created_at_utc"], label="host trust store creation")
    for key_id, encoded in row["keys"].items():
        if not isinstance(key_id, str) or not isinstance(encoded, str):
            raise SupervisorV24ReadinessError("host trust key differs")
        try:
            if len(base64.b64decode(encoded, validate=True)) != 32:
                raise SupervisorV24ReadinessError("host trust key size differs")
        except (binascii.Error, ValueError):
            raise SupervisorV24ReadinessError("host trust key base64 differs") from None
    return MappingProxyType(dict(row["keys"])), payload


def _checkpoint_paths(supervisor_root: Path) -> tuple[Path, ...]:
    root = supervisor_root / CHECKPOINT_DIRECTORY
    if root.is_symlink() or not root.is_dir():
        raise SupervisorV24ReadinessError("checkpoint directory is unavailable or linked")
    rows: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = _CHECKPOINT.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise SupervisorV24ReadinessError("checkpoint directory has an unexpected entry")
        rows.append((int(match.group("generation")), path))
    rows.sort()
    if [generation for generation, _ in rows] != list(range(1, EXPECTED_REVISION + 1)):
        raise SupervisorV24ReadinessError("checkpoint chain generations differ")
    return tuple(path for _, path in rows)


def _verify_checkpoint_chain(
    *,
    supervisor_root: Path,
    authority_root: Path,
    trust_keys: Mapping[str, str],
    registry_file_sha256: str,
    candidate_set_sha256: str,
) -> tuple[Mapping[str, Any], tuple[dict[str, Any], ...], bytes]:
    previous_file_sha: str | None = None
    previous_payload_sha: str | None = None
    receipts: list[dict[str, Any]] = []
    head_status: Mapping[str, Any] | None = None
    head_bytes = b""
    for generation, path in enumerate(_checkpoint_paths(supervisor_root), start=1):
        receipt, raw = _read_json(path, root=authority_root, label=f"checkpoint {generation}")
        payload = _verify_receipt(receipt, trust_keys=trust_keys, label=f"checkpoint {generation}")
        observations = _exact_mapping(
            payload["observations"],
            {
                "attestation_kind",
                "schema",
                "campaign_id",
                "generation",
                "previous_checkpoint_file_sha256",
                "previous_checkpoint_payload_sha256",
                "registry_file_sha256",
                "candidate_set_sha256",
                "candidate_count",
                "terminal_count",
                "status_sha256",
                "provider_calls_performed",
                "exit_codes_considered",
                "status",
            },
            label=f"checkpoint {generation} observations",
        )
        file_sha = _legacy_sha256_bytes(raw)
        payload_sha = receipt["payload_sha256"]
        name = _CHECKPOINT.fullmatch(path.name)
        expected_event = {
            "ordinal": 0,
            "kind": "campaign-supervisor-checkpoint-attested",
            "payload_sha256": _legacy_sha256_value(observations),
        }
        expected_sandbox = "rlevo-medres-campaign-supervisor-" + _legacy_sha256_value(CAMPAIGN_ID)[:24]
        expected_episode = "campaign-supervisor-" + _legacy_sha256_value(CAMPAIGN_ID)[:16] + f"-g{generation:08d}"
        if not (
            observations["attestation_kind"] == "campaign_supervisor_checkpoint"
            and observations["schema"] == CHECKPOINT_SCHEMA
            and observations["campaign_id"] == CAMPAIGN_ID
            and observations["generation"] == generation
            and observations["previous_checkpoint_file_sha256"] == previous_file_sha
            and observations["previous_checkpoint_payload_sha256"] == previous_payload_sha
            and observations["candidate_set_sha256"] == candidate_set_sha256
            and observations["candidate_count"] == EXPECTED_CANDIDATES
            and observations["provider_calls_performed"] == 0
            and observations["exit_codes_considered"] is False
            and observations["status_sha256"] == _legacy_sha256_value(observations["status"])
            and name is not None
            and name.group("digest") == payload_sha[:16]
            and payload["sandbox_id"] == expected_sandbox
            and payload["episode_id"] == expected_episode
            and payload["focus"] == "E2E"
            and payload["events"] == [expected_event]
        ):
            raise SupervisorV24ReadinessError(f"checkpoint {generation} lineage differs")
        # Earlier revisions legitimately bind their then-current registries.
        if generation == EXPECTED_REVISION and observations["registry_file_sha256"] != registry_file_sha256:
            raise SupervisorV24ReadinessError("checkpoint head registry differs")
        receipts.append(
            {
                "generation": generation,
                "file_upstream_sha256": file_sha,
                "file_blake3": blake3_bytes(raw),
                "payload_upstream_sha256": payload_sha,
                "previous_file_upstream_sha256": previous_file_sha,
                "previous_payload_upstream_sha256": previous_payload_sha,
            }
        )
        previous_file_sha = file_sha
        previous_payload_sha = payload_sha
        head_status = observations["status"]
        head_bytes = raw
    if head_status is None:
        raise SupervisorV24ReadinessError("checkpoint head is absent")
    return head_status, tuple(receipts), head_bytes


def load_signed_supervisor_v24_readiness(
    supervisor_root: str | Path,
    *,
    authority_root: str | Path,
    trust_store_path: str | Path,
    worker_width: int = 64,
) -> SignedSupervisorV24Readiness:
    """Verify and project v24 without making any provider or outcome calls."""

    if type(worker_width) is not int or not 1 <= worker_width <= 512:
        raise SupervisorV24ReadinessError("worker_width must be in [1, 512]")
    authority = _validate_root(authority_root, label="legacy authority root")
    supervisor = _validate_root(supervisor_root, label="v24 supervisor root")
    try:
        supervisor.relative_to(authority)
    except ValueError:
        raise SupervisorV24ReadinessError("v24 supervisor root escapes authority") from None
    # ``abspath`` normalizes spelling without following links.  The component
    # walk below remains able to reject every symlink in the supplied path.
    trust_path = Path(os.path.abspath(trust_store_path))
    try:
        trust_path.relative_to(authority)
    except ValueError:
        raise SupervisorV24ReadinessError("trust store escapes authority") from None
    trust_keys, trust_bytes = _load_trust_store(trust_path, authority_root=authority)
    registry, registry_bytes = _read_json(
        supervisor / REGISTRY_FILENAME, root=authority, label="v24 registry"
    )
    registry_row = _exact_mapping(
        registry,
        {"schema", "campaign_id", "created_at_utc", "registry_revision", "slot_limits", "candidates", "registry_sha256"},
        label="v24 registry",
    )
    candidates = registry_row["candidates"]
    if not (
        registry_row["schema"] == REGISTRY_SCHEMA
        and registry_row["campaign_id"] == CAMPAIGN_ID
        and registry_row["registry_revision"] == EXPECTED_REVISION
        and isinstance(candidates, list)
        and len(candidates) == EXPECTED_CANDIDATES
    ):
        raise SupervisorV24ReadinessError("v24 registry identity differs")
    _parse_utc(registry_row["created_at_utc"], label="v24 registry creation")
    unsigned = dict(registry_row)
    claimed_registry_sha = unsigned.pop("registry_sha256")
    if claimed_registry_sha != _legacy_sha256_value(unsigned):
        raise SupervisorV24ReadinessError("v24 registry self-hash differs")
    candidate_ids = [row.get("candidate_id") for row in candidates if isinstance(row, Mapping)]
    if len(candidate_ids) != EXPECTED_CANDIDATES or candidate_ids != sorted(candidate_ids) or len(set(candidate_ids)) != EXPECTED_CANDIDATES:
        raise SupervisorV24ReadinessError("v24 candidate identities differ")
    for row in candidates:
        _exact_mapping(
            row,
            {"candidate_id", "family", "focus", "source_shard", "priority", "harness", "frozen_artifact", "frozen_assertions", "proofs"},
            label="v24 candidate",
        )
        if row["family"] not in FAMILIES or row["focus"] not in STAGES or not isinstance(row["proofs"], Mapping):
            raise SupervisorV24ReadinessError("v24 candidate family, stage, or proof root differs")
    registry_file_sha = _legacy_sha256_bytes(registry_bytes)
    registry_content_sha = _legacy_sha256_value(registry)
    candidate_set_sha = _candidate_set_sha256(candidates)
    registry_attestation, registry_attestation_bytes = _read_json(
        supervisor / REGISTRY_ATTESTATION_FILENAME,
        root=authority,
        label="v24 registry attestation",
        maximum_bytes=1024 * 1024,
    )
    attestation_payload = _verify_receipt(
        registry_attestation, trust_keys=trust_keys, label="v24 registry attestation"
    )
    expected_observations = {
        "attestation_kind": "campaign_supervisor_registry",
        "campaign_id": CAMPAIGN_ID,
        "registry_revision": EXPECTED_REVISION,
        "registry_file_sha256": registry_file_sha,
        "registry_content_sha256": registry_content_sha,
        "candidate_count": EXPECTED_CANDIDATES,
        "candidate_set_sha256": candidate_set_sha,
        "harness_owner": "rlevo-med-research",
        "required_stage_chain": ["S1", "S2", "S3", "S4", "S5"],
    }
    expected_registry_event = {
        "ordinal": 0,
        "kind": "campaign-supervisor-registry-attested",
        "payload_sha256": _legacy_sha256_value(expected_observations),
    }
    if not (
        attestation_payload["observations"] == expected_observations
        and attestation_payload["events"] == [expected_registry_event]
        and attestation_payload["focus"] == "E2E"
        and attestation_payload["sandbox_id"]
        == "rlevo-medres-campaign-supervisor-" + _legacy_sha256_value(CAMPAIGN_ID)[:24]
        and attestation_payload["episode_id"]
        == "campaign-supervisor-registry-" + str(claimed_registry_sha)[:24]
    ):
        raise SupervisorV24ReadinessError("v24 registry attestation differs")
    head_status, checkpoint_receipts, head_bytes = _verify_checkpoint_chain(
        supervisor_root=supervisor,
        authority_root=authority,
        trust_keys=trust_keys,
        registry_file_sha256=registry_file_sha,
        candidate_set_sha256=candidate_set_sha,
    )
    status, status_bytes = _read_json(
        supervisor / STATUS_FILENAME, root=authority, label="v24 status"
    )
    if status != head_status:
        raise SupervisorV24ReadinessError("v24 status differs from its signed checkpoint head")
    status = _exact_mapping(
        status,
        {
            "schema",
            "campaign_id",
            "observed_at_utc",
            "registry_revision",
            "registry_file_sha256",
            "registry_content_sha256",
            "registry_attestation_file_sha256",
            "trust_store_file_sha256",
            "candidate_set_sha256",
            "candidate_count",
            "terminal_count",
            "state_counts",
            "slot_allocation",
            "candidates",
            "next_legal_actions",
            "controls",
            "base_checkpoint",
        },
        label="v24 status",
    )
    status_rows = status["candidates"]
    if not (
        status.get("schema") == STATUS_SCHEMA
        and status.get("campaign_id") == CAMPAIGN_ID
        and status.get("registry_revision") == EXPECTED_REVISION
        and status.get("candidate_count") == EXPECTED_CANDIDATES
        and status.get("candidate_set_sha256") == candidate_set_sha
        and status.get("registry_file_sha256") == registry_file_sha
        and status.get("registry_content_sha256") == registry_content_sha
        and status.get("registry_attestation_file_sha256")
        == _legacy_sha256_bytes(registry_attestation_bytes)
        and status.get("trust_store_file_sha256") == _legacy_sha256_bytes(trust_bytes)
        and status.get("state_counts") == dict(EXPECTED_STATES)
        and isinstance(status_rows, list)
        and len(status_rows) == EXPECTED_CANDIDATES
    ):
        raise SupervisorV24ReadinessError("v24 signed status identity differs")
    registry_by_id = {row["candidate_id"]: row for row in candidates}
    status_by_id: dict[str, Mapping[str, Any]] = {}
    for status_row in status_rows:
        status_row = _exact_mapping(
            status_row,
            {
                "candidate_id",
                "family",
                "focus",
                "source_shard",
                "priority",
                "candidate_commitment_sha256",
                "frozen_artifact_file_sha256",
                "state",
                "state_rank",
                "active_operation",
                "proofs",
                "next_action",
            },
            label="v24 status candidate",
        )
        candidate_id = status_row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in status_by_id:
            raise SupervisorV24ReadinessError("v24 status candidate identity differs")
        status_by_id[candidate_id] = status_row
    if set(status_by_id) != set(registry_by_id):
        raise SupervisorV24ReadinessError("v24 registry and status candidate sets differ")

    # Verify construction proof files only for promoted rows.  No rejected,
    # panel, rollout, judge, review, replay, or admission evidence is opened.
    promoted_tasks: list[tuple[str, str, str, Any]] = []
    preliminary_rows: list[tuple[Mapping[str, Any], ConstructionReadiness, str]] = []
    for candidate_id in candidate_ids:
        candidate = registry_by_id[candidate_id]
        state_row = status_by_id[candidate_id]
        state = state_row.get("state")
        if state not in {item.value for item in ConstructionReadiness}:
            raise SupervisorV24ReadinessError(f"unsupported v24 readiness state: {state}")
        readiness = ConstructionReadiness(state)
        if not (
            state_row.get("family") == candidate["family"]
            and state_row.get("focus") == candidate["focus"]
            and state_row.get("candidate_commitment_sha256") == _candidate_commitment(candidate)
        ):
            raise SupervisorV24ReadinessError(f"{candidate_id} signed state binding differs")
        proof_root = blake3_hex(
            {
                "schema": "eva.legacy-supervisor-proof-root.v1",
                "candidate_id": candidate_id,
                "readiness": readiness.value,
                "legacy_registry_proof_metadata": candidate["proofs"],
                "signed_status_candidate_commitment_sha256": state_row.get(
                    "candidate_commitment_sha256"
                ),
            }
        )
        preliminary_rows.append((candidate, readiness, proof_root))
        if readiness is ConstructionReadiness.PROMOTED:
            if set(candidate["proofs"]) != set(_CONSTRUCTION_PROOFS):
                raise SupervisorV24ReadinessError(f"{candidate_id} construction proof chain differs")
            for proof_name in _CONSTRUCTION_PROOFS:
                promoted_tasks.append(
                    (candidate_id, candidate["focus"], proof_name, candidate["proofs"][proof_name])
                )
        elif readiness is ConstructionReadiness.FROZEN and candidate["proofs"]:
            raise SupervisorV24ReadinessError(f"{candidate_id} frozen proof root is not empty")

    def verify_task(task: tuple[str, str, str, Any]) -> dict[str, str]:
        candidate_id, stage, proof_name, proof = task
        return _verify_construction_proof(
            candidate_id=candidate_id,
            stage=stage,
            proof_name=proof_name,
            proof=proof,
            authority_root=authority,
            trust_keys=trust_keys,
        )

    with ThreadPoolExecutor(max_workers=worker_width, thread_name_prefix="eva-v24-proof") as pool:
        verified_proofs = tuple(pool.map(verify_task, promoted_tasks))
    proof_times: dict[str, list[datetime]] = {}
    for proof in verified_proofs:
        proof_times.setdefault(proof["candidate_id"], []).append(
            _parse_utc(proof["issued_at_utc"], label="construction proof chronology")
        )
    if any(times != sorted(times) for times in proof_times.values()):
        raise SupervisorV24ReadinessError("construction proof chronology regresses")
    rows = tuple(
        SignedReadinessRow(
            candidate_id=candidate["candidate_id"],
            source_family=candidate["family"],
            stage=candidate["focus"],
            readiness=readiness,
            proof_root_blake3=proof_root,
        )
        for candidate, readiness, proof_root in preliminary_rows
    )
    if Counter(row.readiness.value for row in rows) != Counter(
        {"frozen": 7_645, "promoted": 1_344, "rejected": 11}
    ):
        raise SupervisorV24ReadinessError("v24 readiness projection counts differ")
    proof_roots_blake3 = blake3_hex([row.to_document() for row in rows])
    verified_proofs_blake3 = blake3_hex(list(verified_proofs))
    provisional = SignedSupervisorV24Readiness(
        campaign_id=CAMPAIGN_ID,
        registry_revision=EXPECTED_REVISION,
        candidate_set_upstream_sha256=candidate_set_sha,
        registry_logical_upstream_sha256=str(claimed_registry_sha),
        registry_file_upstream_sha256=registry_file_sha,
        registry_file_blake3=blake3_bytes(registry_bytes),
        registry_attestation_file_upstream_sha256=_legacy_sha256_bytes(
            registry_attestation_bytes
        ),
        registry_attestation_file_blake3=blake3_bytes(registry_attestation_bytes),
        status_file_upstream_sha256=_legacy_sha256_bytes(status_bytes),
        status_file_blake3=blake3_bytes(status_bytes),
        checkpoint_head_file_upstream_sha256=_legacy_sha256_bytes(head_bytes),
        checkpoint_head_file_blake3=blake3_bytes(head_bytes),
        checkpoint_chain_blake3=blake3_hex(list(checkpoint_receipts)),
        trust_store_file_upstream_sha256=_legacy_sha256_bytes(trust_bytes),
        trust_store_file_blake3=blake3_bytes(trust_bytes),
        proof_roots_blake3=proof_roots_blake3,
        verified_construction_proofs_blake3=verified_proofs_blake3,
        rows=rows,
        authority_blake3="",
    )
    return replace(
        provisional,
        authority_blake3=blake3_hex(provisional.core_document()),
    )


__all__ = [
    "ConstructionReadiness",
    "SignedReadinessRow",
    "SignedSupervisorV24Readiness",
    "SupervisorV24ReadinessError",
    "load_signed_supervisor_v24_readiness",
]
