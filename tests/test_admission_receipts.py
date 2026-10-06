from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eva_agent.admission import (
    AdmissionClaims,
    AdmissionReceiptError,
    SignedAdmissionPair,
    SignedEnvelope,
    issue_admission_pair,
    load_trust_store,
    verify_admission_pair,
)
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.pipeline.ids import DeterministicUUIDFactory


def _key_material(tmp_path: Path, *, schema: str = "eva.ed25519-trust-store.v1"):
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "host-signing.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o600)
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust_path = tmp_path / "trust.json"
    trust_path.write_text(
        json.dumps(
            {
                "schema": schema,
                "status": "active",
                "algorithm": "Ed25519",
                "keys": {"eva-test-key": base64.b64encode(public).decode("ascii")},
            }
        ),
        encoding="utf-8",
    )
    return private_path, trust_path


def _claims() -> AdmissionClaims:
    ids = DeterministicUUIDFactory("admission-claims")
    return AdmissionClaims(
        candidate_id=ids.new("candidate"),
        sandbox_id=ids.new("sandbox"),
        split="train",
        pipeline_result_blake3=blake3_hex("pipeline-result"),
        strong_evaluation_blake3=blake3_hex("strong-evaluation"),
        manifest_blake3=blake3_hex("manifest"),
        rubric_blake3=blake3_hex("rubric"),
        verification_report_blake3=blake3_hex("verification-report"),
        recommendation_blake3=blake3_hex("recommendation"),
        plan_blake3=blake3_hex("campaign-plan"),
    )


@pytest.mark.parametrize(
    "schema",
    ["eva.ed25519-trust-store.v1", "rlevo.med-research-host-trust-store.v1"],
)
def test_signed_admission_pair_reopens_against_eva_and_legacy_trust_schemas(
    tmp_path: Path, schema: str
) -> None:
    private_path, trust_path = _key_material(tmp_path, schema=schema)
    claims = _claims()
    ids = DeterministicUUIDFactory(f"admission-receipt:{schema}")
    pair = issue_admission_pair(
        claims,
        key_id="eva-test-key",
        private_key_path=private_path,
        issued_at_utc="2026-09-07T12:00:00Z",
        receipt_id=ids.new("receipt"),
        transition_id=ids.new("transition"),
    )

    evidence = verify_admission_pair(pair, claims=claims, trust_store_path=trust_path)
    assert evidence.pipeline_result_blake3 == claims.pipeline_result_blake3
    assert evidence.evaluation_blake3 == claims.strong_evaluation_blake3
    assert evidence.signed_admission_receipt_blake3 == pair.admission.envelope_blake3
    assert evidence.signed_supervisor_transition_blake3 == pair.transition.envelope_blake3
    assert len(load_trust_store(trust_path)) == 1


def test_admission_payload_or_cross_link_tamper_fails_closed(tmp_path: Path) -> None:
    private_path, trust_path = _key_material(tmp_path)
    claims = _claims()
    ids = DeterministicUUIDFactory("admission-tamper")
    pair = issue_admission_pair(
        claims,
        key_id="eva-test-key",
        private_key_path=private_path,
        issued_at_utc="2026-09-07T12:00:00Z",
        receipt_id=ids.new("receipt"),
        transition_id=ids.new("transition"),
    )

    document = pair.admission.to_document()
    document["payload"]["split"] = "development"
    tampered_admission = SignedEnvelope.from_document(document)
    with pytest.raises(AdmissionReceiptError, match="payload BLAKE3"):
        verify_admission_pair(
            SignedAdmissionPair(tampered_admission, pair.transition),
            claims=claims,
            trust_store_path=trust_path,
        )

    transition_document = pair.transition.to_document()
    transition_document["payload"]["admission_envelope_blake3"] = blake3_hex("other")
    tampered_transition = SignedEnvelope.from_document(transition_document)
    with pytest.raises(AdmissionReceiptError, match="payload BLAKE3"):
        verify_admission_pair(
            SignedAdmissionPair(pair.admission, tampered_transition),
            claims=claims,
            trust_store_path=trust_path,
        )


def test_signing_key_permissions_and_malformed_trust_store_are_rejected(tmp_path: Path) -> None:
    private_path, trust_path = _key_material(tmp_path)
    private_path.chmod(0o644)
    with pytest.raises(AdmissionReceiptError, match="permissions"):
        issue_admission_pair(
            _claims(),
            key_id="eva-test-key",
            private_key_path=private_path,
            issued_at_utc="2026-09-07T12:00:00Z",
        )

    trust_path.write_bytes(b"{not-json")
    with pytest.raises(AdmissionReceiptError, match="could not be decoded"):
        load_trust_store(trust_path)
