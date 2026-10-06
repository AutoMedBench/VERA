from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest

from eva_agent.codex_providers import (
    AdapterReceiptSigner,
    PROJECTION_VERSION,
)
from eva_agent.construction.adapter_receipts import (
    PremiumConstructionAdapterReceiptError,
    PremiumConstructionAdapterReceiptStore,
)
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes


MODEL = "aws/anthropic/bedrock-claude-opus-5"


def _receipt(signer: AdapterReceiptSigner, *, request_id: str):
    return signer.sign(
        {
            "schema": "eva.codex-responses-adapter-receipt-payload.v1",
            "request_id": request_id,
            "created_at_utc": "2026-09-07T00:00:00Z",
            "route_id": "opus_5",
            "model_id": MODEL,
            "provider_family": "anthropic",
            "projection_version": PROJECTION_VERSION,
            "status": "passed",
            "failure_class": None,
            "failure_message_blake3": None,
            "adapter_http_status": 200,
            "upstream_http_status": 200,
            "upstream_latency_ms": 1,
            "upstream_body_blake3": blake3_hex("redacted upstream body"),
            "request_shape": {
                "requested_max_output_tokens": 4096,
                "effective_upstream_max_tokens": 32_768,
                "upstream_max_tokens_policy": "exact_override",
            },
            "response_shape": {
                "upstream_finish_reason": "stop",
                "upstream_output_tokens": 11,
                "upstream_reasoning_tokens": 7,
            },
            "request_max_retries": 0,
            "stream_max_retries": 0,
            "upstream_request_max_retries": 0,
            "credential_value_recorded": False,
            "endpoint_value_recorded": False,
            "raw_request_recorded": False,
            "raw_upstream_output_recorded": False,
            "raw_response_recorded": False,
        }
    )


def _store(tmp_path: Path, signer: AdapterReceiptSigner):
    return PremiumConstructionAdapterReceiptStore(
        tmp_path / "state",
        binding_blake3=blake3_hex("premium adapter binding"),
        route_id="opus_5",
        model_id=MODEL,
        provider_family="anthropic",
        signing_key_id=signer.key_id,
        signing_public_key_blake3=signer.public_key_blake3,
        effective_upstream_max_tokens=32_768,
    )


def test_receipt_is_oexcl_sealed_reopened_and_session_rooted(tmp_path: Path) -> None:
    signer = AdapterReceiptSigner.ephemeral()
    store = _store(tmp_path, signer)
    first = _receipt(signer, request_id="request-1")
    second = _receipt(signer, request_id="request-2")
    store.commit(first)
    store.commit(second)

    first_path = store.receipt_root / f"{first.envelope_blake3}.json"
    assert stat.S_IMODE(first_path.lstat().st_mode) == 0o400
    assert canonical_json_bytes(first.to_dict()) == first_path.read_bytes()
    catalog = store.catalog_document()
    assert catalog["receipt_count"] == 2
    assert catalog["receipt_envelope_blake3s"] == tuple(
        sorted((first.envelope_blake3, second.envelope_blake3))
    )
    assert catalog["catalog_root_blake3"] == blake3_hex(
        {key: value for key, value in catalog.items() if key != "catalog_root_blake3"}
    )

    session_id = "00000000-0000-4000-8000-000000000001"
    reference = store.write_session_reference(session_id)
    assert reference["adapter_receipt_catalog_root_blake3"] == catalog[
        "catalog_root_blake3"
    ]
    session_path = store.session_root / f"{session_id}.json"
    assert stat.S_IMODE(session_path.lstat().st_mode) == 0o400

    reopened = _store(tmp_path, signer)
    assert reopened.catalog_document() == catalog
    with pytest.raises(PremiumConstructionAdapterReceiptError, match="already exists"):
        reopened.commit(first)


def test_reopen_rejects_filename_signature_route_budget_and_mode_drift(
    tmp_path: Path,
) -> None:
    signer = AdapterReceiptSigner.ephemeral()
    store = _store(tmp_path, signer)
    receipt = _receipt(signer, request_id="request-1")
    store.commit(receipt)
    path = store.receipt_root / f"{receipt.envelope_blake3}.json"

    path.chmod(0o600)
    with pytest.raises(PremiumConstructionAdapterReceiptError, match="topology"):
        _store(tmp_path, signer)
    path.chmod(0o400)

    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["request_shape"]["effective_upstream_max_tokens"] = 4096
    path.chmod(0o600)
    path.write_bytes(canonical_json_bytes(document))
    path.chmod(0o400)
    with pytest.raises(PremiumConstructionAdapterReceiptError, match="signature"):
        _store(tmp_path, signer)


def test_non_provider_bound_adapter_error_is_not_persisted(tmp_path: Path) -> None:
    signer = AdapterReceiptSigner.ephemeral()
    store = _store(tmp_path, signer)
    receipt = signer.sign(
        {
            **dict(_receipt(signer, request_id="request-template").payload),
            "request_id": "request-without-upstream",
            "route_id": None,
            "model_id": None,
            "provider_family": None,
            "status": "adapter_error",
            "failure_class": "adapter_contract_error",
            "failure_message_blake3": blake3_hex("safe error"),
            "adapter_http_status": 400,
            "upstream_http_status": None,
            "upstream_body_blake3": None,
            "request_shape": {},
            "response_shape": {},
        }
    )
    store.commit(receipt)
    assert store.catalog_document()["receipt_count"] == 0


def test_wrong_key_and_non_exact_output_policy_fail_before_commit(tmp_path: Path) -> None:
    signer = AdapterReceiptSigner.ephemeral()
    store = _store(tmp_path, signer)
    wrong = AdapterReceiptSigner.ephemeral()
    with pytest.raises(PremiumConstructionAdapterReceiptError, match="policy"):
        store.commit(_receipt(wrong, request_id="wrong-key"))
    assert store.catalog_document()["receipt_count"] == 0

    payload = dict(_receipt(signer, request_id="wrong-budget").payload)
    payload["request_shape"] = {
        **payload["request_shape"],
        "effective_upstream_max_tokens": 4096,
    }
    with pytest.raises(PremiumConstructionAdapterReceiptError, match="policy"):
        store.commit(signer.sign(payload))
    assert store.catalog_document()["receipt_count"] == 0
