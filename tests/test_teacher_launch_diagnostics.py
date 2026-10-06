from __future__ import annotations

import json
from pathlib import Path

from eva_agent.codex_providers import AdapterReceiptSigner
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex
from eva_agent.training.teacher_batch import persist_teacher_adapter_receipt
from eva_agent.training.teacher_launch import isolated_teacher_launch_options


def _failed_adapter_receipt():
    return AdapterReceiptSigner.ephemeral().sign(
        {
            "schema": "eva.codex-responses-adapter-receipt-payload.v1",
            "request_id": "00000000-0000-4000-8000-000000000001",
            "created_at_utc": "2026-09-08T00:00:00Z",
            "route_id": "qwen_3_5_397b_a17b",
            "model_id": "nvidia/qwen/qwen3-5-397b-a17b",
            "provider_family": "qwen",
            "projection_version": "eva.codex-responses-chat-projection.v2-packed-namespaces",
            "status": "adapter_error",
            "failure_class": "upstream_http_rejection",
            "failure_message_blake3": blake3_hex("safe failure"),
            "adapter_http_status": 502,
            "upstream_http_status": 400,
            "upstream_latency_ms": 7,
            "upstream_body_blake3": blake3_bytes(b'{"error":"redacted"}'),
            "request_shape": {"tool_count": 7},
            "response_shape": {"output_item_count": 0},
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


def test_adapter_receipt_is_durable_and_self_contained(tmp_path: Path) -> None:
    receipt = _failed_adapter_receipt()
    first = persist_teacher_adapter_receipt(tmp_path, receipt)
    second = persist_teacher_adapter_receipt(tmp_path, receipt)
    assert first == second
    document = json.loads(first.read_text(encoding="utf-8"))
    assert document["envelope_blake3"] == receipt.envelope_blake3
    assert document["payload"]["upstream_http_status"] == 400
    assert document["payload"]["raw_request_recorded"] is False
    assert "redacted" not in first.read_text(encoding="utf-8")


def test_teacher_launch_uses_sanitized_isolated_codex_home(
    tmp_path: Path,
) -> None:
    isolation = (tmp_path / "private-runtime").resolve()
    launch = isolated_teacher_launch_options(
        codex_bin="/bin/true",
        cwd=tmp_path,
        isolation_root=isolation,
        config_overrides=(
            "project_doc_max_bytes=0",
            'model_catalog_json="/safe/models.json"',
        ),
        child_env={"NVIDIA_INFERENCE_API_KEY": "fixture-secret"},
    )
    argv = launch.launch_args_override
    assert argv is not None
    assert str(isolation) in argv
    assert argv[-4:] == ("app-server", "--strict-config", "--listen", "stdio://")
    assert "fixture-secret" not in "\x00".join(argv)
    assert "/localhome/local-operator/.codex" not in "\x00".join(argv)
    assert "openaiDeveloperDocs" not in "\x00".join(argv)
    assert "fixture-secret" not in repr(launch)
