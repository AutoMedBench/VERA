from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.pipeline import DeterministicUUIDFactory
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.training import execution_verified_sft as candidate_sft


def _event(event_id: str, role: str, content, call_ids=()):
    core = {
        "event_id": event_id,
        "role": role,
        "content": content,
        "tool_call_ids": list(call_ids),
    }
    return {**core, "event_blake3": blake3_hex(core)}


def _task(tmp_path: Path) -> candidate_sft._SourceTask:
    source_root = tmp_path / "teacher"
    trajectory = source_root / "trajectories/sandbox/gpt_5_6_sol/result.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text("{}", encoding="utf-8")
    return candidate_sft._SourceTask(
        source_root=source_root,
        source_root_relative="runs/teacher",
        trajectory_path=trajectory,
        trajectory_relative="runs/teacher/trajectories/sandbox/gpt_5_6_sol/result.json",
        task_id="sandbox--gpt_5_6_sol",
        sandbox_id="sandbox",
        candidate_id="candidate",
        route_id="gpt_5_6_sol",
        stage="S1",
        domain="medical",
    )


def _inspection_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    messages = (
        _event("00000000-0000-4000-8000-000000000001", "system", "system"),
        _event("00000000-0000-4000-8000-000000000002", "user", {"task": "go"}),
        _event(
            "00000000-0000-4000-8000-000000000003",
            "assistant",
            {"codex_tool_calls": []},
            ("00000000-0000-4000-8000-000000000010",),
        ),
        _event(
            "00000000-0000-4000-8000-000000000004",
            "tool",
            {"status": "completed"},
            ("00000000-0000-4000-8000-000000000010",),
        ),
        _event(
            "00000000-0000-4000-8000-000000000005",
            "assistant",
            {"response": "Completed."},
        ),
    )
    document = {
        "schema": "eva.codex-teacher-full-trajectory.v1",
        "task_id": "sandbox--gpt_5_6_sol",
        "sandbox_id": "sandbox",
        "candidate_id": "candidate",
        "route_id": "gpt_5_6_sol",
        "selection_pending": True,
        "score_kind": "selection_pending",
        "semantic_retry_count": 0,
        "judge_calls": 0,
        "admission_writes": 0,
        "model_id": "openai/openai/gpt-5.6-sol",
        "provider": "eva_gpt_5_6_sol",
        "assistant_output": "Completed.",
        "messages": list(messages),
        "tool_trace": {
            "results": [
                {
                    "name": "materialize_plan",
                    "status": "completed",
                    "error_code": None,
                    "output": {
                        "gate_passed": True,
                        "stage": "S1",
                        "failed_check_ids": [],
                    },
                }
            ]
        },
        "workspace_before": {
            "tree_blake3": "e" * 64,
            "files": [],
        },
        "workspace_after": {
            "tree_blake3": "f" * 64,
            "files": [
                {"path": "work/stage-plan.json", "content_blake3": "a" * 64}
            ],
        },
        "provider_metadata": {"codex_turn_receipt": {"fixture": True}},
    }
    document["messages"][1]["content"] = {
        "task": "go",
        "execution_binding": {
            "execution_stages": {
                "S1": {"artifact_relative_path": "work/stage-plan.json"}
            }
        },
    }
    rubric = SimpleNamespace(
        rubric_id="rubric", digest="a" * 64, items=tuple(range(6))
    )
    trajectory = SimpleNamespace(
        document=document, source_blake3="b" * 64, rubric=rubric
    )
    result = SimpleNamespace(
        result_id="00000000-0000-4000-8000-000000000020",
        call_id="00000000-0000-4000-8000-000000000010",
        name="materialize_plan",
        status="completed",
        output={"gate_passed": True, "stage": "S1", "failed_check_ids": []},
        error_code=None,
        receipt_blake3="c" * 64,
    )
    evidence = SimpleNamespace(
        tool_trace=SimpleNamespace(results=(result,), trace_blake3="d" * 64),
        workspace_before=SimpleNamespace(tree_blake3="e" * 64, files=()),
        workspace_after=SimpleNamespace(
            tree_blake3="f" * 64,
            files=(
                SimpleNamespace(
                    path="work/stage-plan.json", content_blake3="a" * 64
                ),
            ),
        ),
    )
    receipt = SimpleNamespace(
        status="completed",
        role=candidate_sft.CodexRole.STRONG_ACTOR,
        thread_resumed=False,
        visibility="actor-public",
        model=document["model_id"],
        provider=document["provider"],
        final_response=document["assistant_output"],
        tool_calls=(SimpleNamespace(status="completed"),),
        receipt_blake3="1" * 64,
    )
    monkeypatch.setattr(candidate_sft, "validate_full_trajectory", lambda *_a, **_k: trajectory)
    monkeypatch.setattr(candidate_sft, "reconstruct_evidence", lambda *_a, **_k: evidence)
    monkeypatch.setattr(candidate_sft, "codex_turn_receipt_from_document", lambda _v: receipt)
    monkeypatch.setattr(candidate_sft, "verify_codex_turn_receipt", lambda _v: None)
    monkeypatch.setattr(candidate_sft, "_read_json", lambda _p: (document, b"{}"))
    return document, result, evidence, receipt


def test_inspection_accepts_only_exact_execution_complete_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document, result, evidence, receipt = _inspection_fixture(tmp_path, monkeypatch)
    task = _task(tmp_path)
    accepted, reason = candidate_sft._inspect_source(
        task, registry=SimpleNamespace(), allowed_routes=frozenset({task.route_id})
    )
    assert accepted is not None and reason is None

    document["assistant_output"] = "Blocked: no artifact submitted"
    receipt.final_response = document["assistant_output"]
    rejected, reason = candidate_sft._inspect_source(
        task, registry=SimpleNamespace(), allowed_routes=frozenset({task.route_id})
    )
    assert rejected is None and reason == "blocked_or_incomplete_terminal_response"
    document["assistant_output"] = "Completed."
    receipt.final_response = document["assistant_output"]

    result.status = "immutable_failure"
    rejected, reason = candidate_sft._inspect_source(
        task, registry=SimpleNamespace(), allowed_routes=frozenset({task.route_id})
    )
    assert rejected is None and reason == "immutable_or_failed_tool_result"
    result.status = "completed"

    result.output = {"gate_passed": False, "stage": "S1"}
    rejected, reason = candidate_sft._inspect_source(
        task, registry=SimpleNamespace(), allowed_routes=frozenset({task.route_id})
    )
    assert rejected is None and reason == "unmet_tool_gate"
    result.output = {"gate_passed": True, "stage": "S2"}
    rejected, reason = candidate_sft._inspect_source(
        task, registry=SimpleNamespace(), allowed_routes=frozenset({task.route_id})
    )
    assert rejected is None and reason == "task_stage_not_completed"

    result.output = {"gate_passed": True, "stage": "S1"}
    evidence.workspace_after.tree_blake3 = evidence.workspace_before.tree_blake3
    rejected, reason = candidate_sft._inspect_source(
        task, registry=SimpleNamespace(), allowed_routes=frozenset({task.route_id})
    )
    assert rejected is None and reason == "workspace_unchanged"


def test_candidate_slice_keeps_parallel_tool_group_atomic_and_pending(
    tmp_path: Path,
) -> None:
    calls = (
        "00000000-0000-4000-8000-000000000010",
        "00000000-0000-4000-8000-000000000011",
    )
    messages = (
        _event("00000000-0000-4000-8000-000000000001", "system", "system"),
        _event("00000000-0000-4000-8000-000000000002", "user", {"task": "go"}),
        _event(
            "00000000-0000-4000-8000-000000000003",
            "assistant",
            {"codex_tool_calls": []},
            calls,
        ),
        _event(
            "00000000-0000-4000-8000-000000000004",
            "tool",
            {"status": "completed", "ordinal": 2},
            (calls[1],),
        ),
        _event(
            "00000000-0000-4000-8000-000000000005",
            "tool",
            {"status": "completed", "ordinal": 1},
            (calls[0],),
        ),
    )
    source = candidate_sft._AcceptedSource(
        task=_task(tmp_path),
        document={"model_id": "openai/openai/gpt-5.6-sol"},
        source_blake3="a" * 64,
        rubric_id="rubric",
        rubric_digest="b" * 64,
        rubric_item_count=6,
        codex_turn_receipt_blake3="c" * 64,
        tool_trace_blake3="d" * 64,
        workspace_before_blake3="e" * 64,
        workspace_after_blake3="f" * 64,
        completion_result_id="00000000-0000-4000-8000-000000000020",
        completion_receipt_blake3="1" * 64,
        completion_artifact_path="work/stage-plan.json",
        messages=messages,
    )
    rows = candidate_sft._slice_source(
        source,
        dataset_id="00000000-0000-4000-8000-000000000099",
        id_factory=DeterministicUUIDFactory("candidate-sft"),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["atomic_multi_tool_target"] is True
    assert [item["tool_call_ids"][0] for item in row["target_tool_observations"]] == list(calls)
    assert row["strict_sft_eligible"] is False
    assert row["agent_judged"] is False
    assert row["selection_status"] == "pending_workspace_agent_judge"
    assert row["execution_verification"]["task_stage_completed"] is True
