from __future__ import annotations

from pathlib import Path
import sqlite3
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eva_agent.campaign.history import signed_historical_attempt_evidence
from eva_agent.orchestration.receipts import ReceiptJournal


def _key(path: Path) -> Path:
    private = Ed25519PrivateKey.generate()
    path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return path


def test_sealed_prior_claim_is_signed_and_globally_excluded(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    run_id = str(uuid4())
    candidate_id = str(uuid4())
    journal = ReceiptJournal(receipts, run_id)
    journal.start({"test": True})
    journal.append(
        event="candidate_execution_started",
        payload={"candidate_id": candidate_id},
        worker_id=str(uuid4()),
    )
    journal.finalize({"complete": True})

    evidence = signed_historical_attempt_evidence(
        receipt_roots=(receipts,), private_key_path=_key(tmp_path / "key.pem"), key_id="test"
    )
    assert evidence["attempted_candidate_ids"] == (candidate_id,)
    assert evidence["sealed_runs"][0]["run_id"] == run_id
    assert evidence["signature_base64"]
    assert len(evidence["evidence_blake3"]) == 64


def test_unsealed_run_is_not_authority_for_exclusion(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    candidate_id = str(uuid4())
    journal = ReceiptJournal(receipts, str(uuid4()))
    journal.start({"test": True})
    journal.append(
        event="candidate_execution_started",
        payload={"candidate_id": candidate_id},
        worker_id=str(uuid4()),
    )
    evidence = signed_historical_attempt_evidence(
        receipt_roots=(receipts,), private_key_path=_key(tmp_path / "key.pem"), key_id="test"
    )
    assert evidence["attempted_candidate_ids"] == ()
    assert evidence["sealed_runs"] == ()


def test_legacy_ledger_without_provider_boundary_is_reopened_conservatively(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "legacy.sqlite3"
    queued = str(uuid4())
    terminal = str(uuid4())
    attempted = str(uuid4())
    with sqlite3.connect(ledger) as connection:
        connection.executescript(
            """
            CREATE TABLE candidates(candidate_id TEXT PRIMARY KEY, status TEXT NOT NULL);
            CREATE TABLE attempts(candidate_id TEXT NOT NULL);
            """
        )
        connection.executemany(
            "INSERT INTO candidates(candidate_id,status) VALUES (?,?)",
            ((queued, "queued"), (terminal, "infrastructure_quarantine"), (attempted, "queued")),
        )
        connection.execute("INSERT INTO attempts(candidate_id) VALUES (?)", (attempted,))

    evidence = signed_historical_attempt_evidence(
        receipt_roots=(tmp_path / "missing-receipts",),
        ledger_paths=(ledger,),
        private_key_path=_key(tmp_path / "key.pem"),
        key_id="test",
    )

    assert evidence["attempted_candidate_ids"] == tuple(sorted((attempted, terminal)))
    assert evidence["ledgers"][0]["provider_boundary_state_available"] is False
    assert len(evidence["ledgers"][0]["candidate_schema_blake3"]) == 64
    assert len(evidence["ledgers"][0]["attempt_schema_blake3"]) == 64


def test_historical_ledger_missing_required_schema_fails_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "invalid.sqlite3"
    with sqlite3.connect(ledger) as connection:
        connection.execute("CREATE TABLE candidates(candidate_id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE attempts(candidate_id TEXT NOT NULL)")

    with pytest.raises(ValueError, match="schema differs"):
        signed_historical_attempt_evidence(
            receipt_roots=(tmp_path / "missing-receipts",),
            ledger_paths=(ledger,),
            private_key_path=_key(tmp_path / "key.pem"),
            key_id="test",
        )
