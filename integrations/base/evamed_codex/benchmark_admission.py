"""Reopen actual terminal evidence before admitting partial policy outcomes."""
from __future__ import annotations

import json
from pathlib import Path

from training.automedbench_lite.adapter import read_document


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def provider_evidence(audit: Path) -> list[dict]:
    rows = [read_document(path) for path in sorted((audit / "provider/requests").glob("*.json"))
            if path.name != "budget-exhaustion.json"]
    rows.sort(key=lambda row: row["request_number"])
    _require([row["request_number"] for row in rows] == list(range(1, len(rows) + 1)),
             "provider_request_sequence_incomplete")
    _require(all(row.get("http_status") == 200 and not row.get("error_type") for row in rows),
             "provider_infrastructure_failure_observed")
    return [{"request_number": row["request_number"], "document_blake3": row["document_blake3"]} for row in rows]


def observed_infrastructure_failures(audit: Path) -> list[dict]:
    """A failed fixed analysis worker is infrastructure, even if Codex finished."""
    from blake3 import blake3
    failures = []
    for path in sorted((audit / 'code-executions').glob('*/cpu-publication-failure.json')):
        row = read_document(path)
        failures.append({'kind': 'cpu-publication-boundary-failure', 'execution_id': path.parent.name,
                         'document_blake3': row['document_blake3'], 'error_type': row['error_type']})
    for path in sorted(audit.glob('host-publication/deadline-publication-events.jsonl')):
        for row in (json.loads(line) for line in path.read_bytes().splitlines()):
            if row.get('event') == 'public_publication' and row.get('complete_interval_before_deadline') is not True:
                failures.append({'kind': 'host-publication-boundary-failure', 'event_id': row['event_id']})
    diagnostics = audit / "mcp-infrastructure-errors.jsonl"
    if diagnostics.exists():
        for line in diagnostics.read_bytes().splitlines():
            row = json.loads(line)
            failures.append({"kind": "mcp-host-failure", "error_id": row["error_id"],
                             "document_blake3": row["document_blake3"], "error_type": row["error_type"]})
    for path in sorted((audit / "model-jobs/authoritative").glob("*/process-exit.json")):
        ended = json.loads(path.read_bytes())
        worker_path = path.parent / "receipt.json"
        if ended.get("returncode") == 0 or not worker_path.exists():
            continue
        worker = json.loads(worker_path.read_bytes())
        # Policy termination of a still-running job is handled by the separate
        # budget proof. A worker's own explicit failed receipt is different.
        if worker.get("status") != "failed":
            continue
        _require(ended.get("worker_receipt_blake3") == blake3(worker_path.read_bytes()).hexdigest()
                 and worker.get("job_id") == ended.get("job_id") == path.parent.name
                 and ended.get("os_process_exit_observed") is True, "failed_worker_receipt_binding_changed")
        failures.append({"kind": "prescribed-analysis-worker-failure", "job_id": path.parent.name,
            "worker_receipt_blake3": ended["worker_receipt_blake3"],
            "process_exit_blake3": blake3(path.read_bytes()).hexdigest(), "returncode": ended["returncode"],
            "error_type": worker.get("error_type"), "policy_authored_worker": False})
    return failures


def require_scoring_admission(run: Path, track: str, actor: dict) -> str:
    """A budget outcome is admissible only with terminal, joined, quiescent proof.

    This does not turn an interrupted/failed policy into completed work. It only
    lets the native evaluator apply its own missing-output rules to frozen bytes.
    """
    _require(actor.get("score_admissible", True) is True and not actor.get("errors")
             and actor.get("purpose") != "infrastructure_smoke", "actor_infrastructure_or_diagnostic")
    _require(not observed_infrastructure_failures(run / "track-rollouts" / track),
             "actual_host_infrastructure_failure_not_score_admissible")
    if actor.get("completed_requested_turns") is True:
        return "completed"
    _require(actor.get("terminal_disposition") == "policy-budget-exhausted",
             "actor_incomplete_without_authenticated_policy_terminal")
    if actor.get('partial_policy_terminal_blake3'):
        from .benchmark_partial_admission import require_partial_scoring_admission
        return require_partial_scoring_admission(run, track, actor)
    from eva_agent.codex_runtime import codex_turn_receipt_from_document, verify_codex_turn_receipt
    from training.automedbench_lite.policy_capture import require_joined_host_results

    audit = run / "track-rollouts" / track
    proof = read_document(audit / "policy-budget-terminal.json")
    _require(proof["document_blake3"] == actor.get("policy_budget_terminal_blake3")
             and proof.get("schema") == "eva.evamed-policy-budget-terminal.v1"
             and proof.get("infrastructure_error") is None
             and proof.get("workspace_quiescence_verified") is True,
             "policy_terminal_proof_invalid")
    raw = json.loads((audit / "turns/01-e2e/receipt.json").read_bytes())
    receipt = codex_turn_receipt_from_document(raw)
    verify_codex_turn_receipt(receipt)
    _require(receipt.receipt_blake3 == proof["receipt_blake3"]
             and [receipt.receipt_blake3] == actor["turn_receipt_blake3s"]
             and receipt.status == proof["actual_terminal_status"]
             and receipt.status in {"completed", "interrupted", "failed"}, "actual_policy_terminal_changed")
    joins = require_joined_host_results(receipt, audit)
    hosts_path = audit / "mcp-events.jsonl"
    host_count = len(hosts_path.read_bytes().splitlines()) if hosts_path.exists() else 0
    _require(joins == proof["host_join"] and joins["joined_host_call_count"] == host_count,
             "unjoined_host_work_at_policy_terminal")
    budget = read_document(audit / "track-budget.json")
    _require(budget["document_blake3"] == proof["track_budget"]["budget_document_blake3"]
             and budget["timeout_seconds"] == 3600 and actor["max_model_requests"] == 100,
             "policy_budget_binding_changed")
    providers = provider_evidence(audit)
    _require(providers == proof["provider_requests"] and len(providers) == actor["actual_model_request_count"],
             "provider_request_evidence_changed")
    if proof["reason"] == "model_requests":
        denied = read_document(audit / "provider/requests/budget-exhaustion.json")
        _require(len(providers) == 100 and denied["reason"] == "model_requests"
                 and denied["actual_upstream_requests"] == 100 and denied["denied_request_number"] == 101
                 and denied["upstream_request_sent"] is False
                 and denied["document_blake3"] == proof["provider_denial_blake3"],
                 "request_limit_exhaustion_unproved")
    elif proof["reason"] == "wall_clock":
        outcome = proof["policy_budget"]
        _require(proof["track_budget"]["deadline_exhausted"] is True
                 and outcome.get("budget_exhausted") is True
                 and type(outcome.get("terminal_observed_ns")) is int
                 and outcome.get("infrastructure_error") is None,
                 "wall_clock_exhaustion_unproved")
    else:
        raise ValueError("policy_budget_reason_invalid")
    cleanup = read_document(audit / "evamed-job-cleanup.json")
    after = read_document(audit / "turns/01-e2e/after/manifest.json", maximum=16 * 1024**2)
    final = read_document(audit / "final/manifest.json", maximum=16 * 1024**2)
    _require(cleanup["document_blake3"] == actor["cleanup_document_blake3"] == proof["cleanup_document_blake3"]
             and cleanup["workspace_quiescent"] is True
             and after["document_blake3"] == proof["after_snapshot_blake3"]
             and final["document_blake3"] == actor["final_snapshot_blake3"] == proof["final_snapshot_blake3"],
             "policy_terminal_frozen_workspace_binding_changed")
    return "policy-budget-exhausted"
