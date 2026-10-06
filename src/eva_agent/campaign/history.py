"""Signed, outcome-blind exclusion of candidates claimed by earlier runs."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import stat
import sqlite3
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eva_agent.orchestration.receipts import ReceiptJournal
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes
from eva_agent.orchestration.contracts import canonical_uuid


class HistoricalAttemptError(ValueError):
    """Historical attempt evidence could not be reopened safely."""


ATTEMPT_EVENTS = frozenset(
    {
        "candidate_execution_started",
        "candidate_provider_boundary_crossed",
        "candidate_reviewed",
        "candidate_rejected",
        "candidate_infrastructure_quarantined",
        "candidate_admitted",
    }
)


def signed_historical_attempt_evidence(
    *, receipt_roots: tuple[Path, ...], private_key_path: Path, key_id: str,
    ledger_paths: tuple[Path, ...] = (),
) -> Mapping[str, Any]:
    """Aggregate only completely sealed, self-verifying prior receipt runs.

    A start receipt is deliberately sufficient for exclusion: once a candidate
    has been claimed, replaying it in another ledger would bias scheduling even
    if the process died before the provider boundary.
    """

    if not receipt_roots or not key_id:
        raise HistoricalAttemptError("historical receipt roots and key ID are required")
    candidates: set[str] = set()
    runs: list[dict[str, Any]] = []
    for requested in receipt_roots:
        root = Path(requested)
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise HistoricalAttemptError("historical receipt root topology differs")
        if not root.exists():
            continue
        for run_root in sorted(path for path in root.iterdir() if path.is_dir()):
            run_id = run_root.name
            try:
                canonical_uuid(run_id, label="historical_run_id")
            except Exception:
                continue
            if not ReceiptJournal.verify_run(root, run_id):
                # An active or interrupted run is not immutable evidence yet.
                continue
            receipt_hashes: list[str] = []
            run_candidates: set[str] = set()
            for path in sorted(run_root.rglob("*.json")):
                document = json.loads(path.read_text(encoding="utf-8"))
                receipt_hashes.append(str(document["receipt_blake3"]))
                if document.get("event") not in ATTEMPT_EVENTS:
                    continue
                payload = document.get("payload")
                candidate_id = payload.get("candidate_id") if isinstance(payload, dict) else None
                if isinstance(candidate_id, str):
                    canonical_uuid(candidate_id, label="historical_candidate_id")
                    run_candidates.add(candidate_id)
            candidates.update(run_candidates)
            runs.append(
                {
                    "run_id": run_id,
                    "receipt_count": len(receipt_hashes),
                    "receipts_blake3": blake3_hex(tuple(sorted(receipt_hashes))),
                    "attempted_candidate_ids": tuple(sorted(run_candidates)),
                }
            )
    ledgers: list[dict[str, Any]] = []
    for requested in ledger_paths:
        path = Path(requested)
        if not path.exists():
            continue
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise HistoricalAttemptError("historical ledger topology differs")
        try:
            with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
                candidate_schema = tuple(
                    tuple(row) for row in connection.execute("PRAGMA table_info(candidates)")
                )
                attempt_schema = tuple(
                    tuple(row) for row in connection.execute("PRAGMA table_info(attempts)")
                )
                candidate_columns = {str(row[1]) for row in candidate_schema}
                attempt_columns = {str(row[1]) for row in attempt_schema}
                if not {"candidate_id", "status"}.issubset(candidate_columns) or not {
                    "candidate_id"
                }.issubset(attempt_columns):
                    raise HistoricalAttemptError(
                        "historical ledger schema differs"
                    )
                provider_boundary_available = (
                    "provider_boundary_state" in candidate_columns
                )
                crossed_clause = (
                    "candidate.provider_boundary_state='crossed' OR "
                    if provider_boundary_available
                    else ""
                )
                rows = connection.execute(
                    f"""
                    SELECT DISTINCT candidate.candidate_id
                    FROM candidates AS candidate
                    LEFT JOIN attempts ON attempts.candidate_id=candidate.candidate_id
                    WHERE {crossed_clause}
                       candidate.status IN ('reviewed','rejected','infrastructure_quarantine','admitted')
                       OR attempts.candidate_id IS NOT NULL
                    ORDER BY candidate.candidate_id
                    """
                ).fetchall()
        except HistoricalAttemptError:
            raise
        except sqlite3.Error as exc:
            raise HistoricalAttemptError("historical ledger cannot be reopened") from exc
        ledger_candidates = tuple(str(row[0]) for row in rows)
        for candidate_id in ledger_candidates:
            canonical_uuid(candidate_id, label="historical_candidate_id")
        candidates.update(ledger_candidates)
        ledgers.append(
            {
                "path": str(path.resolve()),
                "candidate_schema_blake3": blake3_hex(candidate_schema),
                "attempt_schema_blake3": blake3_hex(attempt_schema),
                "provider_boundary_state_available": provider_boundary_available,
                "attempted_candidate_ids": ledger_candidates,
                "logical_rows_blake3": blake3_hex(ledger_candidates),
            }
        )
    core = {
        "schema": "eva.historical-attempt-exclusion.v1",
        "key_id": key_id,
        "receipt_roots": tuple(str(Path(root).resolve()) for root in receipt_roots),
        "sealed_runs": tuple(runs),
        "ledgers": tuple(ledgers),
        "attempted_candidate_ids": tuple(sorted(candidates)),
    }
    core["evidence_blake3"] = blake3_hex(core)
    key_path = Path(private_key_path)
    info = key_path.lstat()
    if key_path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise HistoricalAttemptError("historical evidence signing key topology differs")
    private = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    if not isinstance(private, Ed25519PrivateKey):
        raise HistoricalAttemptError("historical evidence signing key differs")
    signed = canonical_json_bytes(core)
    return {
        **core,
        "signer_public_key_blake3": blake3_bytes(
            private.public_key().public_bytes_raw()
        ),
        "signature_base64": base64.b64encode(private.sign(signed)).decode("ascii"),
    }


__all__ = ["HistoricalAttemptError", "signed_historical_attempt_evidence"]
