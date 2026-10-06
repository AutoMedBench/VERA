from __future__ import annotations

import json
from pathlib import Path

import pytest

from eva_agent.codex_providers import AdapterReceiptSigner
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex
from eva_agent.training.agent_judge import (
    AgentJudgeSelectionError,
    AgentJudgeTask,
    _worker_environment,
    persist_agent_judge_adapter_receipt,
)


def _failed_adapter_receipt(*, route_id: str | None = "opus_5"):
    return AdapterReceiptSigner.ephemeral().sign(
        {
            "schema": "eva.codex-responses-adapter-receipt-payload.v1",
            "request_id": "00000000-0000-4000-8000-000000000001",
            "created_at_utc": "2026-09-08T00:00:00Z",
            "route_id": route_id,
            "model_id": "aws/anthropic/bedrock-claude-opus-5",
            "provider_family": "anthropic",
            "projection_version": "eva.codex-responses-chat-projection.v2-packed-namespaces",
            "status": "adapter_error",
            "failure_class": "upstream_http_rejection",
            "failure_message_blake3": blake3_hex("safe failure"),
            "adapter_http_status": 502,
            "upstream_http_status": 400,
            "upstream_latency_ms": 7,
            "upstream_body_blake3": blake3_bytes(b'{"error":"fixture-secret"}'),
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


def _task() -> AgentJudgeTask:
    return AgentJudgeTask(
        judge_task_id="00000000-0000-4000-8000-000000000002",
        source_task_id="source-task",
        sandbox_id="sandbox",
        candidate_id="candidate",
        actor_route_id="qwen_3_5_397b_a17b",
        stage="E2E",
        domain="medxpertqa",
        trajectory_path="/safe/result.json",
        judge_route_id="opus_5",
        judge_model_id="aws/anthropic/bedrock-claude-opus-5",
    )


def test_agent_judge_adapter_receipt_is_durable_safe_and_task_bound(
    tmp_path: Path,
) -> None:
    receipt = _failed_adapter_receipt()
    task = _task()
    first = persist_agent_judge_adapter_receipt(
        tmp_path,
        receipt,
        judge_task_id=task.judge_task_id,
        expected_route_id=task.judge_route_id,
    )
    second = persist_agent_judge_adapter_receipt(
        tmp_path,
        receipt,
        judge_task_id=task.judge_task_id,
        expected_route_id=task.judge_route_id,
    )

    assert first == second
    assert first.relative_to(tmp_path).parts[:3] == (
        "adapter-receipts",
        task.judge_task_id,
        task.judge_route_id,
    )
    document = json.loads(first.read_text(encoding="utf-8"))
    assert document["envelope_blake3"] == receipt.envelope_blake3
    assert document["payload"]["upstream_http_status"] == 400
    assert document["payload"]["raw_request_recorded"] is False
    assert "fixture-secret" not in first.read_text(encoding="utf-8")


def test_agent_judge_adapter_receipt_rejects_route_or_path_confusion(
    tmp_path: Path,
) -> None:
    receipt = _failed_adapter_receipt(route_id="qwen_3_5_397b_a17b")
    with pytest.raises(AgentJudgeSelectionError, match="route differs"):
        persist_agent_judge_adapter_receipt(
            tmp_path,
            receipt,
            judge_task_id=_task().judge_task_id,
            expected_route_id="opus_5",
        )
    with pytest.raises(AgentJudgeSelectionError, match="artifact component"):
        persist_agent_judge_adapter_receipt(
            tmp_path,
            _failed_adapter_receipt(),
            judge_task_id="../escape",
            expected_route_id="opus_5",
        )


def test_agent_judge_worker_environment_binds_durable_output_root(
    tmp_path: Path,
) -> None:
    task = _task()
    environment = _worker_environment(
        task,
        trajectory_blake3=blake3_hex("trajectory"),
        output_root=tmp_path / "selection",
    )
    assert environment["EVA_AGENT_JUDGE_OUTPUT_ROOT"] == str(
        (tmp_path / "selection").resolve()
    )
    assert environment["EVA_AGENT_JUDGE_TASK_ID"] == task.judge_task_id
