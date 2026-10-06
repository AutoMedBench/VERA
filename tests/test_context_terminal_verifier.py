"""Real SDK fixture receipts and signatures, no provider or clinical scores."""
import json
from pathlib import Path
from uuid import uuid4

import pytest

from eva_agent.codex_providers.adapter import AdapterReceiptSigner, PROJECTION_VERSION
from eva_agent.pipeline.digests import blake3_hex
from training.automedbench_lite.adapter import write_once, file_digest
from training.automedbench_lite.context_terminal import verify_context_terminal, FAILURE
import test_automedbench_track_feedback as capture
from test_codex_runtime import _FakeBackend, _turn_event


@pytest.fixture
def context_stop(tmp_path, monkeypatch):
    request_id = str(uuid4())
    error = {"message": json.dumps({"error": {"type": "adapter_contract_error", "message": FAILURE,
                                             "request_id": request_id}}), "additionalDetails": None}
    def backend(script):
        def failed(thread_id, turn_id):
            prefix = script(thread_id, turn_id)[:3]
            return (*prefix,
                _turn_event("error", thread_id, turn_id, error=error, willRetry=False),
                _turn_event("turn/completed", thread_id, turn_id,
                    turn={"id": turn_id, "status": "failed", "items": [], "error": error}))
        return _FakeBackend(failed)
    monkeypatch.setattr(capture, "_FakeBackend", backend)
    value = capture.fixture(tmp_path, failed=True)
    run = value["run_root"];audit = run / "track-rollouts/classification";turn = audit / "turns/01-planning"
    raw = json.loads((turn / "receipt.json").read_bytes())
    source = tmp_path / "fixture-source/training/automedbench_lite/local_qwen.py"
    source.parent.mkdir(parents=True);source.write_text("# Declared synthetic local wrapper source; never executed.\n")
    write_once(run / "baseline-launch.json", {"command": [str(source.with_name("track_entry.py"))],
        "runtime_source_blake3": {"training/automedbench_lite/local_qwen.py": file_digest(source)}})
    write_once(audit.parent / "backend.json", {"text_token_budget_enabled": True,
        "model_alias": raw["model"], "context_length": 32768, "provider_route": "qwen_3_5_9b",
        "adapter_projection_version": PROJECTION_VERSION})
    signer = AdapterReceiptSigner.ephemeral()
    payload = {"schema": "eva.codex-responses-adapter-receipt-payload.v1", "request_id": request_id,
        "created_at_utc": "2026-09-10T00:00:00Z", "route_id": "qwen_3_5_9b", "model_id": raw["model"],
        "provider_family": "qwen", "projection_version": PROJECTION_VERSION, "status": "adapter_error",
        "failure_class": "adapter_contract_error", "failure_message_blake3": blake3_hex(FAILURE),
        "adapter_http_status": 400, "upstream_http_status": None, "upstream_latency_ms": 0,
        "upstream_body_blake3": None, "request_shape": {}, "response_shape": {},
        "request_max_retries": 0, "stream_max_retries": 0, "upstream_request_max_retries": 0,
        "credential_value_recorded": False, "endpoint_value_recorded": False, "raw_request_recorded": False,
        "raw_upstream_output_recorded": False, "raw_response_recorded": False}
    path = audit.parent / "local-runtime/adapter-receipts/qwen_3_5_9b/fixture.json"
    path.parent.mkdir(parents=True);path.write_text(json.dumps(signer.sign(payload).to_dict()))
    return turn, audit, raw, path, signer, payload, source


def test_fixed_signed_failure_is_budget_outcome_without_unjoined_numeric_claim(context_stop):
    turn, audit, raw, *_ = context_stop
    before = {str(p): p.read_bytes() for p in audit.parent.rglob("*") if p.is_file()}
    proof = verify_context_terminal(turn, audit, raw)
    assert proof["valid"] and proof["actual_terminal_status"] == "failed"
    assert proof["joined_host_call_count"] == 1 and proof["workspace_quiescence_verified"]
    assert proof["numeric_prompt_tokens"] is None and proof["numeric_guard_record_joined"] is False
    assert proof["provider_calls"] == 0 and proof["reward"] is None and proof["sft_eligible"] is False
    assert before == {str(p): p.read_bytes() for p in audit.parent.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", ["local429", "provider500", "wrong_failure", "wrong_request", "signature",
                                  "missing_host", "async_writer", "source_changed", "snapshot_changed"])
def test_unrelated_or_unproved_failures_stay_inadmissible(context_stop, change):
    turn, audit, raw, path, signer, payload, source = context_stop
    if change == "local429": payload["adapter_http_status"] = 429
    elif change == "provider500": payload["upstream_http_status"] = 500
    elif change == "wrong_failure": payload["failure_message_blake3"] = blake3_hex("upstream transport failed")
    elif change == "wrong_request": payload["request_id"] = str(uuid4())
    elif change == "missing_host": (audit / "mcp-events.jsonl").write_text("")
    elif change == "async_writer":
        job = audit / "model-jobs" / str(uuid4());job.mkdir(parents=True);(job / "submission.json").write_text("{}")
    elif change == "source_changed": source.write_text("# Changed source\n")
    elif change == "snapshot_changed":
        changed = turn / "after/files/task.json"
        changed.unlink();changed.write_text("{}")
    signed = signer.sign(payload).to_dict()
    if change == "signature": signed["payload_blake3"] = "0" * 64
    path.write_text(json.dumps(signed))
    with pytest.raises(ValueError): verify_context_terminal(turn, audit, raw)
