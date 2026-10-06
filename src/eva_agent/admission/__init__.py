"""Cryptographically verified admission receipts for EVA-Agent releases."""

from .receipts import (
    AdmissionClaims,
    AdmissionReceiptError,
    SignedAdmissionPair,
    SignedEnvelope,
    TrajectoryAdmissionClaims,
    issue_admission_pair,
    issue_signed_envelope,
    load_trust_store,
    verify_admission_pair,
    verify_signed_envelope,
)
from .orchestration import (
    AdmissionContextResolver,
    Ed25519AdmissionSupervisor,
    Ed25519SupervisorSignatureVerifier,
    SupervisorAdmissionContext,
    SupervisorAdmissionError,
)

__all__ = [
    "AdmissionClaims",
    "AdmissionReceiptError",
    "SignedAdmissionPair",
    "SignedEnvelope",
    "TrajectoryAdmissionClaims",
    "issue_admission_pair",
    "issue_signed_envelope",
    "load_trust_store",
    "verify_admission_pair",
    "verify_signed_envelope",
    "AdmissionContextResolver",
    "Ed25519AdmissionSupervisor",
    "Ed25519SupervisorSignatureVerifier",
    "SupervisorAdmissionContext",
    "SupervisorAdmissionError",
]
