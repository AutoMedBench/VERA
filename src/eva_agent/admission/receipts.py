"""Domain-separated Ed25519 receipts for reviewed sandbox admissions.

The host private key is only used at the final supervisor boundary.  Runtime
identities remain UUIDs and all new content commitments remain BLAKE3.  A
legacy rlevo trust store can be consumed as a public-key source, but its
SHA-256 conventions never enter the new receipt domain.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from eva_agent.pipeline.contracts import AdmissionEvidence, freeze_json, uuid_text
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes, is_blake3


SIGNATURE_DOMAIN = "eva.ed25519-signed-envelope.v1"
ENVELOPE_SCHEMA = "eva.signed-envelope.v1"
RECEIPT_SCHEMA = "eva.admission-receipt-payload.v1"
RELAXED_RECEIPT_SCHEMA = "eva.admission-receipt-payload.v2"
TRANSITION_SCHEMA = "eva.supervisor-admission-transition-payload.v1"
TRUST_SCHEMAS = {
    "eva.ed25519-trust-store.v1",
    "rlevo.med-research-host-trust-store.v1",
}


class AdmissionReceiptError(ValueError):
    """An admission signature or exact claim failed closed."""


def _utc(value: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AdmissionReceiptError("issued_at_utc must be canonical UTC text")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise AdmissionReceiptError("issued_at_utc is invalid") from None
    if parsed.tzinfo != timezone.utc or parsed.isoformat().replace("+00:00", "Z") != value:
        raise AdmissionReceiptError("issued_at_utc must be canonical UTC text")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _key_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not 3 <= len(value) <= 128
        or not value[0].islower()
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in value)
    ):
        raise AdmissionReceiptError("signing key_id differs")
    return value


def _digest(value: str, label: str) -> str:
    if not is_blake3(value):
        raise AdmissionReceiptError(f"{label} must be BLAKE3")
    return value


@dataclass(frozen=True, slots=True)
class AdmissionClaims:
    candidate_id: str
    sandbox_id: str
    split: str
    pipeline_result_blake3: str
    strong_evaluation_blake3: str
    manifest_blake3: str
    rubric_blake3: str
    verification_report_blake3: str
    recommendation_blake3: str
    plan_blake3: str

    def __post_init__(self) -> None:
        uuid_text(self.candidate_id, label="candidate_id")
        uuid_text(self.sandbox_id, label="sandbox_id")
        if self.split not in {"train", "development", "sealed_evaluation"}:
            raise AdmissionReceiptError("admission split differs")
        for field_name in (
            "pipeline_result_blake3",
            "strong_evaluation_blake3",
            "manifest_blake3",
            "rubric_blake3",
            "verification_report_blake3",
            "recommendation_blake3",
            "plan_blake3",
        ):
            _digest(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class TrajectoryAdmissionClaims:
    """V2 admission claims for the best valid judged trajectory available."""

    candidate_id: str
    sandbox_id: str
    split: str
    pipeline_result_blake3: str
    selected_evaluation_blake3: str
    selected_evaluation_cohort: str
    admission_policy_blake3: str
    ability_separation_passed: bool
    manifest_blake3: str
    rubric_blake3: str
    verification_report_blake3: str
    recommendation_blake3: str
    plan_blake3: str

    def __post_init__(self) -> None:
        uuid_text(self.candidate_id, label="candidate_id")
        uuid_text(self.sandbox_id, label="sandbox_id")
        if self.split not in {"train", "development", "sealed_evaluation"}:
            raise AdmissionReceiptError("admission split differs")
        if self.selected_evaluation_cohort not in {"weak", "middle", "strong"}:
            raise AdmissionReceiptError("selected evaluation cohort differs")
        if type(self.ability_separation_passed) is not bool:
            raise AdmissionReceiptError("ability separation evidence differs")
        for field_name in (
            "pipeline_result_blake3", "selected_evaluation_blake3",
            "admission_policy_blake3", "manifest_blake3", "rubric_blake3",
            "verification_report_blake3", "recommendation_blake3", "plan_blake3",
        ):
            _digest(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class SignedEnvelope:
    payload: Mapping[str, Any]
    payload_blake3: str
    key_id: str
    algorithm: str
    signature_base64: str
    envelope_blake3: str

    def __post_init__(self) -> None:
        payload = freeze_json(self.payload)
        if not isinstance(payload, Mapping):
            raise AdmissionReceiptError("signed payload must be an object")
        object.__setattr__(self, "payload", payload)
        _digest(self.payload_blake3, "payload_blake3")
        _digest(self.envelope_blake3, "envelope_blake3")
        _key_id(self.key_id)
        if self.algorithm != "Ed25519":
            raise AdmissionReceiptError("signature algorithm differs")
        _signature_bytes(self.signature_base64)

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": ENVELOPE_SCHEMA,
            "signature_domain": SIGNATURE_DOMAIN,
            "payload": _thaw(self.payload),
            "payload_blake3": self.payload_blake3,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "signature_base64": self.signature_base64,
            "envelope_blake3": self.envelope_blake3,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "SignedEnvelope":
        expected = {
            "schema",
            "signature_domain",
            "payload",
            "payload_blake3",
            "key_id",
            "algorithm",
            "signature_base64",
            "envelope_blake3",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise AdmissionReceiptError("signed envelope shape differs")
        if value["schema"] != ENVELOPE_SCHEMA or value["signature_domain"] != SIGNATURE_DOMAIN:
            raise AdmissionReceiptError("signed envelope domain differs")
        return cls(
            payload=value["payload"],
            payload_blake3=value["payload_blake3"],
            key_id=value["key_id"],
            algorithm=value["algorithm"],
            signature_base64=value["signature_base64"],
            envelope_blake3=value["envelope_blake3"],
        )


@dataclass(frozen=True, slots=True)
class SignedAdmissionPair:
    admission: SignedEnvelope
    transition: SignedEnvelope


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _signature_bytes(value: str) -> bytes:
    if not isinstance(value, str):
        raise AdmissionReceiptError("signature must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise AdmissionReceiptError("signature base64 differs") from None
    if len(decoded) != 64 or base64.b64encode(decoded).decode("ascii") != value:
        raise AdmissionReceiptError("signature bytes differ")
    return decoded


def _signed_message(payload: Mapping[str, Any], payload_blake3: str) -> bytes:
    return canonical_json_bytes(
        {
            "signature_domain": SIGNATURE_DOMAIN,
            "payload_blake3": payload_blake3,
            "payload": payload,
        }
    )


def _envelope_core(
    *, payload: Mapping[str, Any], payload_blake3: str, key_id: str, signature_base64: str
) -> dict[str, Any]:
    return {
        "schema": ENVELOPE_SCHEMA,
        "signature_domain": SIGNATURE_DOMAIN,
        "payload": payload,
        "payload_blake3": payload_blake3,
        "key_id": key_id,
        "algorithm": "Ed25519",
        "signature_base64": signature_base64,
    }


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise AdmissionReceiptError("private key must be an absolute regular non-symlink file")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise AdmissionReceiptError("private key ownership or permissions are unsafe")
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (OSError, TypeError, ValueError):
        raise AdmissionReceiptError("private key could not be loaded") from None
    if not isinstance(key, Ed25519PrivateKey):
        raise AdmissionReceiptError("private key is not Ed25519")
    return key


def _sign(payload: Mapping[str, Any], *, key_id: str, key: Ed25519PrivateKey) -> SignedEnvelope:
    frozen = freeze_json(payload)
    if not isinstance(frozen, Mapping):
        raise AdmissionReceiptError("signed payload must be an object")
    payload_blake3 = blake3_hex(frozen)
    signature = key.sign(_signed_message(frozen, payload_blake3))
    encoded = base64.b64encode(signature).decode("ascii")
    core = _envelope_core(
        payload=frozen,
        payload_blake3=payload_blake3,
        key_id=_key_id(key_id),
        signature_base64=encoded,
    )
    envelope_fields = {
        field_name: core[field_name]
        for field_name in (
            "payload",
            "payload_blake3",
            "key_id",
            "algorithm",
            "signature_base64",
        )
    }
    return SignedEnvelope(**envelope_fields, envelope_blake3=blake3_hex(core))


def load_trust_store(path: Path) -> Mapping[str, Ed25519PublicKey]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise AdmissionReceiptError("trust store topology or size differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise AdmissionReceiptError("trust store could not be decoded") from None
    if (
        not isinstance(value, dict)
        or value.get("schema") not in TRUST_SCHEMAS
        or value.get("status") != "active"
        or value.get("algorithm") != "Ed25519"
        or not isinstance(value.get("keys"), dict)
        or not value["keys"]
    ):
        raise AdmissionReceiptError("trust store contract differs")
    result: dict[str, Ed25519PublicKey] = {}
    for key_id, encoded in value["keys"].items():
        _key_id(key_id)
        if not isinstance(encoded, str):
            raise AdmissionReceiptError("trust-store public key differs")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise AdmissionReceiptError("trust-store public key base64 differs") from None
        if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != encoded:
            raise AdmissionReceiptError("trust-store public key bytes differ")
        result[key_id] = Ed25519PublicKey.from_public_bytes(raw)
    return MappingProxyType(result)


def _verify_envelope(envelope: SignedEnvelope, keys: Mapping[str, Ed25519PublicKey]) -> None:
    if blake3_hex(envelope.payload) != envelope.payload_blake3:
        raise AdmissionReceiptError("signed payload BLAKE3 differs")
    core = _envelope_core(
        payload=envelope.payload,
        payload_blake3=envelope.payload_blake3,
        key_id=envelope.key_id,
        signature_base64=envelope.signature_base64,
    )
    if blake3_hex(core) != envelope.envelope_blake3:
        raise AdmissionReceiptError("signed envelope BLAKE3 differs")
    public_key = keys.get(envelope.key_id)
    if public_key is None:
        raise AdmissionReceiptError("signing key is not trusted")
    try:
        public_key.verify(
            _signature_bytes(envelope.signature_base64),
            _signed_message(envelope.payload, envelope.payload_blake3),
        )
    except InvalidSignature:
        raise AdmissionReceiptError("Ed25519 signature verification failed") from None


def issue_signed_envelope(
    payload: Mapping[str, Any],
    *,
    key_id: str,
    private_key_path: Path,
) -> SignedEnvelope:
    """Sign one already-domain-specific payload with the canonical EVA envelope.

    Admission and construction supervision have distinct payload schemas and
    state machines, but they intentionally share the same envelope encoding,
    key loading rules, and Ed25519 signature domain.  This narrow public
    wrapper prevents other supervisors from copying those primitives or
    pretending that their payload is an admission transition.
    """

    return _sign(
        payload,
        key_id=key_id,
        key=_load_private_key(Path(private_key_path)),
    )


def verify_signed_envelope(
    envelope: SignedEnvelope,
    *,
    trust_store_path: Path,
) -> None:
    """Verify one canonical EVA envelope without assigning payload semantics."""

    if not isinstance(envelope, SignedEnvelope):
        raise AdmissionReceiptError("signed envelope type differs")
    _verify_envelope(envelope, load_trust_store(Path(trust_store_path)))


def issue_admission_pair(
    claims: AdmissionClaims | TrajectoryAdmissionClaims,
    *,
    key_id: str,
    private_key_path: Path,
    issued_at_utc: str | None = None,
    receipt_id: str | None = None,
    transition_id: str | None = None,
) -> SignedAdmissionPair:
    """Sign one admission and one cross-linked supervisor transition."""

    issued = _utc(issued_at_utc or _now())
    receipt_uuid = uuid_text(receipt_id or str(uuid4()), label="receipt_id")
    transition_uuid = uuid_text(transition_id or str(uuid4()), label="transition_id")
    private_key = _load_private_key(private_key_path)
    relaxed = isinstance(claims, TrajectoryAdmissionClaims)
    receipt_payload = {
        "schema": RELAXED_RECEIPT_SCHEMA if relaxed else RECEIPT_SCHEMA,
        "receipt_id": receipt_uuid,
        "candidate_id": claims.candidate_id,
        "sandbox_id": claims.sandbox_id,
        "split": claims.split,
        "pipeline_result_blake3": claims.pipeline_result_blake3,
        **(
            {
                "selected_evaluation_blake3": claims.selected_evaluation_blake3,
                "selected_evaluation_cohort": claims.selected_evaluation_cohort,
                "admission_policy_blake3": claims.admission_policy_blake3,
                "ability_separation_required_for_admission": False,
            }
            if relaxed
            else {"strong_evaluation_blake3": claims.strong_evaluation_blake3}
        ),
        "manifest_blake3": claims.manifest_blake3,
        "rubric_blake3": claims.rubric_blake3,
        "verification_report_blake3": claims.verification_report_blake3,
        "recommendation_blake3": claims.recommendation_blake3,
        "pipeline_verification_passed": True,
        "ability_separation_passed": (
            claims.ability_separation_passed if relaxed else True
        ),
        "issued_at_utc": issued,
    }
    admission = _sign(receipt_payload, key_id=key_id, key=private_key)
    transition_payload = {
        "schema": TRANSITION_SCHEMA,
        "transition_id": transition_uuid,
        "candidate_id": claims.candidate_id,
        "sandbox_id": claims.sandbox_id,
        "from_state": "reviewed",
        "to_state": "admitted",
        "plan_blake3": claims.plan_blake3,
        "admission_envelope_blake3": admission.envelope_blake3,
        "issued_at_utc": issued,
    }
    transition = _sign(transition_payload, key_id=key_id, key=private_key)
    return SignedAdmissionPair(admission=admission, transition=transition)


def _expected_receipt(
    claims: AdmissionClaims | TrajectoryAdmissionClaims,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    relaxed = isinstance(claims, TrajectoryAdmissionClaims)
    return {
        "schema": RELAXED_RECEIPT_SCHEMA if relaxed else RECEIPT_SCHEMA,
        "receipt_id": payload.get("receipt_id"),
        "candidate_id": claims.candidate_id,
        "sandbox_id": claims.sandbox_id,
        "split": claims.split,
        "pipeline_result_blake3": claims.pipeline_result_blake3,
        **(
            {
                "selected_evaluation_blake3": claims.selected_evaluation_blake3,
                "selected_evaluation_cohort": claims.selected_evaluation_cohort,
                "admission_policy_blake3": claims.admission_policy_blake3,
                "ability_separation_required_for_admission": False,
            }
            if relaxed
            else {"strong_evaluation_blake3": claims.strong_evaluation_blake3}
        ),
        "manifest_blake3": claims.manifest_blake3,
        "rubric_blake3": claims.rubric_blake3,
        "verification_report_blake3": claims.verification_report_blake3,
        "recommendation_blake3": claims.recommendation_blake3,
        "pipeline_verification_passed": True,
        "ability_separation_passed": (
            claims.ability_separation_passed if relaxed else True
        ),
        "issued_at_utc": payload.get("issued_at_utc"),
    }


def verify_admission_pair(
    pair: SignedAdmissionPair,
    *,
    claims: AdmissionClaims | TrajectoryAdmissionClaims,
    trust_store_path: Path,
) -> AdmissionEvidence:
    """Verify signatures, exact claims, and receipt-to-transition linkage."""

    keys = load_trust_store(trust_store_path)
    _verify_envelope(pair.admission, keys)
    _verify_envelope(pair.transition, keys)
    receipt = pair.admission.payload
    transition = pair.transition.payload
    uuid_text(receipt.get("receipt_id"), label="receipt_id")
    uuid_text(transition.get("transition_id"), label="transition_id")
    _utc(receipt.get("issued_at_utc"))
    _utc(transition.get("issued_at_utc"))
    if _thaw(receipt) != _expected_receipt(claims, receipt):
        raise AdmissionReceiptError("admission receipt claims differ")
    expected_transition = {
        "schema": TRANSITION_SCHEMA,
        "transition_id": transition.get("transition_id"),
        "candidate_id": claims.candidate_id,
        "sandbox_id": claims.sandbox_id,
        "from_state": "reviewed",
        "to_state": "admitted",
        "plan_blake3": claims.plan_blake3,
        "admission_envelope_blake3": pair.admission.envelope_blake3,
        "issued_at_utc": receipt["issued_at_utc"],
    }
    if _thaw(transition) != expected_transition:
        raise AdmissionReceiptError("supervisor admission transition differs")
    return AdmissionEvidence(
        pipeline_result_blake3=claims.pipeline_result_blake3,
        evaluation_blake3=(
            claims.selected_evaluation_blake3
            if isinstance(claims, TrajectoryAdmissionClaims)
            else claims.strong_evaluation_blake3
        ),
        signed_admission_receipt_blake3=pair.admission.envelope_blake3,
        signed_supervisor_transition_blake3=pair.transition.envelope_blake3,
    )


__all__ = [
    "AdmissionClaims",
    "AdmissionReceiptError",
    "TrajectoryAdmissionClaims",
    "SignedAdmissionPair",
    "SignedEnvelope",
    "issue_signed_envelope",
    "issue_admission_pair",
    "load_trust_store",
    "verify_admission_pair",
    "verify_signed_envelope",
]
