"""Opt-in proof of one exact local context-budget stop, never a clinical score.

The failed SDK receipt is not rewritten. A signed adapter failure joined by its
request UUID establishes the fixed local token-guard category. Separately stored
numeric token counts have no request-UUID join and are deliberately NOT used or
claimed here. This is not a timeout/429/provider/task failure fallback.
"""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from uuid import UUID

from eva_agent.codex_runtime import codex_turn_receipt_from_document
from eva_agent.codex_providers.adapter import SignedAdapterReceipt, verify_adapter_receipt
from eva_agent.pipeline.digests import blake3_hex, canonical_value
from .adapter import read_document, file_digest
from .policy_capture import require_joined_host_results
from .policy_terminal import require

FAILURE = "local_token_budget_failed: text_prompt_exhausts_context"


def _failure_id(error):
    require(isinstance(error, dict) and error.get("additionalDetails") is None,
            "context_terminal_error_details_differ")
    try:
        value = json.loads(error["message"])["error"]
        request_id = value["request_id"]
        UUID(request_id)
    except (KeyError, TypeError, ValueError):
        require(False, "context_terminal_not_exact_local_guard_error")
    require(set(value) == {"type", "message", "request_id"}
            and value["type"] == "adapter_contract_error" and value["message"] == FAILURE,
            "context_terminal_not_exact_local_guard_error")
    return request_id


def _source_binding(run, source_binding):
    """Bind the actual local wrapper source; do not claim an unretained guard hash."""
    if source_binding is None:
        launch = read_document(run / "baseline-launch.json")
        entries = [Path(value) for value in launch["command"]
                   if isinstance(value, str) and value.endswith("/training/automedbench_lite/track_entry.py")]
        require(len(entries) == 1, "context_terminal_actor_source_unbound")
        root = entries[0].parents[2]
        relative = "training/automedbench_lite/local_qwen.py"
        source = root / relative
        digest = launch["runtime_source_blake3"][relative]
    else:
        entries = [row for row in source_binding["actual_imported_sources"]
                   if row["path"].endswith("/training/automedbench_lite/local_qwen.py")]
        require(len(entries) == 1, "context_terminal_actor_source_unbound")
        source, digest = Path(entries[0]["path"]), entries[0]["blake3"]
    require(source.is_file() and not source.is_symlink() and file_digest(source) == digest,
            "context_terminal_local_wrapper_source_changed")
    return {"path": str(source), "blake3": digest}


def verify_context_terminal(turn_root, audit, raw_receipt, *, source_binding=None):
    """Read only; return proof for this attempted prefix, with reward/count NULL.

    Initially restricted to tracks with no submitted asynchronous model jobs.
    Those jobs require a separately proved drain, not inference from SDK status.
    """
    turn_root, audit = Path(turn_root), Path(audit)
    require(turn_root.parent == audit / "turns" and not turn_root.is_symlink(), "context_terminal_turn_scope")
    receipt = codex_turn_receipt_from_document(raw_receipt)
    require(receipt.status == "failed" and receipt.visibility == "actor-public"
            and receipt.events[-1].method == "turn/completed", "context_terminal_real_failed_terminal_required")
    terminal = canonical_value(receipt.events[-1].payload)
    require(terminal["threadId"] == receipt.thread_id and terminal["turn"]["id"] == receipt.turn_id
            and terminal["turn"]["status"] == "failed", "context_terminal_turn_binding_differs")
    request_id = _failure_id(terminal["turn"]["error"])
    errors = [canonical_value(event.payload) for event in receipt.events if event.method == "error"]
    require(len(errors) == 1 and errors[0].get("willRetry") is False
            and errors[0]["threadId"] == receipt.thread_id and errors[0]["turnId"] == receipt.turn_id
            and _failure_id(errors[0]["error"]) == request_id, "context_terminal_independent_error_or_retry")
    run = audit.parent.parent
    backend = read_document(audit.parent / "backend.json")
    require(backend.get("text_token_budget_enabled") is True
            and backend.get("model_alias") == receipt.model
            and type(backend.get("context_length")) is int and backend["context_length"] > 0,
            "context_terminal_local_guard_not_configured")
    source = _source_binding(run, source_binding)
    matches = []
    for path in (audit.parent / "local-runtime/adapter-receipts").glob("*/*.json"):
        raw = json.loads(path.read_bytes())
        if raw.get("payload", {}).get("request_id") == request_id:
            signed = SignedAdapterReceipt(**{field.name: raw[field.name] for field in fields(SignedAdapterReceipt)})
            verify_adapter_receipt(signed)
            require(canonical_value(signed.to_dict()) == raw, "context_terminal_signed_envelope_differs")
            matches.append((path, signed))
    require(len(matches) == 1, "context_terminal_signed_failure_missing_or_ambiguous")
    path, signed = matches[0]
    payload = signed.payload
    require(payload["model_id"] == receipt.model and payload["route_id"] == backend["provider_route"]
            and payload["projection_version"] == backend["adapter_projection_version"]
            and payload["status"] == "adapter_error" and payload["failure_class"] == "adapter_contract_error"
            and payload["adapter_http_status"] == 400 and payload["upstream_http_status"] is None
            and payload["upstream_body_blake3"] is None and payload["upstream_latency_ms"] == 0
            and payload["failure_message_blake3"] == blake3_hex(FAILURE),
            "context_terminal_not_signed_local_budget_failure")
    joined = require_joined_host_results(receipt, audit)
    require(not list((audit / "model-jobs").glob("*/submission.json")),
            "context_terminal_async_model_drain_not_proved")
    from .track_feedback import track_snapshot
    _, before = track_snapshot(turn_root / "before", "before")
    _, after = track_snapshot(turn_root / "after", "after")
    return {"schema": "eva.verified-local-context-budget-terminal.v1", "valid": True,
        "receipt_blake3": receipt.receipt_blake3, "actual_terminal_status": "failed",
        "adapter_request_id": request_id, "signed_adapter_envelope_blake3": signed.envelope_blake3,
        "signed_adapter_file_blake3": file_digest(path), "source_binding": source,
        "backend_document_blake3": backend["document_blake3"], "failure_category": "text_prompt_exhausts_context",
        "configured_context_length": backend["context_length"], "numeric_prompt_tokens": None,
        "numeric_guard_record_joined": False, "failed_request_generation_attempts": 0,
        "before_manifest_blake3": before["document_blake3"], "after_manifest_blake3": after["document_blake3"],
        "workspace_quiescence_verified": True, "quiescence_basis": "all_host_results_joined_no_async_model_jobs",
        **joined, "original_receipt_mutated": False, "provider_calls": 0, "reward": None,
        "later_stages_evaluated": False, "sft_eligible": False}
