"""CPU-only: real immutable workspace reads; no external Judge/model calls."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from eva_agent.codex_pipeline import CodexOpus5AgentJudge, CodexPipelineError
from eva_agent.codex_runtime import CodexRole, CodexRuntimeError, CodexSandbox, CodexThreadOptions
from eva_agent.pipeline import JudgeAssessment, JudgeRequest, RandomUUIDFactory, RubricItemScore, ToolCall
from eva_agent.pipeline.digests import blake3_hex, canonical_json_bytes, canonical_value
from eva_agent.pipeline.judge_material_view import POLICY_VISIBLE_VIEW, material_layout
from eva_agent.pipeline.judge_workspace import JudgeWorkspaceTools
from eva_agent.training.agent_judge import AgentJudgeSelectionError
from eva_agent.training.slime_agent_judge import (
    NativeAstraWorkspaceJudge, PolicyVisibleOpusWorkspaceJudge,
    grade_prepared_rollout, prepare_workspace_rollout,
)
from test_agent_judge_worker import _event, _fixture, _trace_event


def _prepared(tmp_path, message_bytes=629145, *, extra_after=None):
    registry, source, task = _fixture(tmp_path, extra_after=extra_after)
    rollout = json.loads(source.read_text())
    rollout["messages"][-1] = _event("assistant", "x" * message_bytes)
    rollout["assistant_output"] = ""
    source.write_bytes(canonical_json_bytes(rollout))
    prepared = prepare_workspace_rollout(
        rollout, rubric=registry.resolve("medxpertqa", "E2E"), source_path=source,
        judge_model_id=task.judge_model_id, material_view=POLICY_VISIBLE_VIEW,
    )
    return prepared


class _Capture:
    def run_once(self, options, turn_input):
        self.options, self.turn_input = options, turn_input
        raise CodexRuntimeError("provider-free capture")

    def run_once_with_timeout(self, options, turn_input, *, timeout_seconds):
        return self.run_once(options, turn_input)


def _judge(tmp_path, model, runner, cls=PolicyVisibleOpusWorkspaceJudge, fast=True):
    def options(request, offers):
        return CodexThreadOptions(
            role=CodexRole.JUDGE, model=request.judge_model_id, provider="provider-free",
            cwd=str(tmp_path), sandbox=CodexSandbox.READ_ONLY, offered_tools=offers,
        )

    return cls(
        options_factory=options, model_id=model, runner=runner,
        compact_evidence_index=True, fast_workspace_judge=fast,
        maximum_workspace_tool_calls=64, maximum_workspace_tool_frontiers=32,
        turn_timeout_seconds=600,
    )


def _capture(tmp_path, prepared, **kwargs):
    capture = _Capture()
    judge = _judge(tmp_path, prepared.task.judge_model_id, capture, **kwargs)
    with pytest.raises(CodexRuntimeError, match="provider-free capture"):
        grade_prepared_rollout(prepared, judge=judge, output_root=tmp_path)
    return capture


@pytest.mark.parametrize("message_bytes", [65536, 131071, 436589, 629145])
def test_native_turn_has_exact_private_chunk_plan_without_tail_gaps(tmp_path, message_bytes):
    prepared = _prepared(tmp_path, message_bytes)
    original = canonical_json_bytes(prepared.evidence)
    capture = _capture(tmp_path, prepared)
    private = canonical_value(capture.turn_input.judge_only_context)
    plan = private["judge_only_reference"]["required_actor_chunk_read_plan"]
    assert "required_actor_chunk_read_plan" not in capture.turn_input.public_context
    assert "required_actor_chunk_read_plan" not in capture.turn_input.public_text
    assert canonical_json_bytes(prepared.evidence) == original
    assert plan["maximum_workspace_calls"] == 64
    assert plan["maximum_execution_frontiers"] == 32
    assert plan["turn_timeout_seconds"] == 600
    assert plan["plan_is_not_evidence_of_reads"] is True
    assert "MULTIPLE" in plan["instruction"] and "no missing middle or tail" in plan["instruction"]
    assert plan["required_chunk_count"] == len(plan["chunks"]) < 62
    _, required, audit = material_layout(prepared.evidence)
    assert {row["path"] for row in plan["chunks"]} == set(required)
    assert not ({row["path"] for row in plan["chunks"]} & set(audit))
    files = {row.path: row for row in prepared.evidence.workspace_after.files}
    for path in required:
        chunks = [row for row in plan["chunks"] if row["path"] == path]
        total = files[path].byte_count
        assert len(chunks) == max(1, (total + 65535) // 65536)
        cursor = 0
        for chunk in chunks:
            assert chunk["snapshot"] == "after"
            assert chunk["offset"] == cursor
            assert chunk["total_bytes"] == total
            assert 1 <= chunk["max_bytes"] <= 65536
            assert chunk["max_bytes"] == min(65536, max(1, total - cursor))
            cursor += min(chunk["max_bytes"], total - cursor)
        assert cursor == total
    # Neither tool schemas nor exact compiled scoring objects gained plan fields.
    schemas = JudgeWorkspaceTools(prepared.evidence).response_api_schemas()
    assert [canonical_value(row.input_schema) for row in capture.options.offered_tools] == [
        canonical_value(row["parameters"]) for row in schemas
    ]
    assert private["compiled_rubric"]["digest"] == prepared.trajectory.rubric.digest
    verdict = private["judge_only_reference"]["exact_verdict_requirements"]
    output_schema = canonical_value(capture.turn_input.output_schema)
    row_schema = output_schema["properties"]["item_scores"]["items"]
    assert verdict["rubric_digest"] == prepared.trajectory.rubric.digest
    assert verdict["item_ids_in_order"] == [item["item_id"] for item in private["compiled_rubric"]["items"]]
    assert verdict["top_level_fields"] == output_schema["required"]
    assert verdict["item_score_fields"] == row_schema["required"]
    assert set(verdict["item_score_fields"]) == {"item_id", "score", "evidence_refs", "rationale"}
    assert verdict["additional_item_score_fields_allowed"] is row_schema["additionalProperties"] is False
    assert "score_note" in verdict["instruction"]
    assert "NOT proof that you read that file" in verdict["instruction"]


@pytest.mark.parametrize("cls,fast", [
    (CodexOpus5AgentJudge, True),
    (PolicyVisibleOpusWorkspaceJudge, False),
    (NativeAstraWorkspaceJudge, True),
])
def test_other_judge_profiles_do_not_receive_new_plan(tmp_path, cls, fast):
    prepared = _prepared(tmp_path)
    if cls is NativeAstraWorkspaceJudge:
        from dataclasses import replace
        prepared = replace(prepared, task=replace(
            prepared.task, judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra"))
    capture = _capture(tmp_path, prepared, cls=cls, fast=fast)
    assert "required_actor_chunk_read_plan" not in capture.turn_input.judge_only_context["judge_only_reference"]
    assert "exact_verdict_requirements" not in capture.turn_input.judge_only_context["judge_only_reference"]


def _planned_workspace_assessment(request, rubric, *, omit_message_tail=False):
    """Scripted model substitute issuing actual canonical workspace tools."""
    ids = RandomUUIDFactory()
    tools = JudgeWorkspaceTools(request.workspace_evidence, id_factory=ids)
    plan = request.judge_only_reference["required_actor_chunk_read_plan"]
    chunks = list(plan["chunks"])
    if omit_message_tail:
        tail = max(i for i, row in enumerate(chunks) if row["path"].endswith("/messages.json"))
        chunks.pop(tail)
    arguments = [{key: row[key] for key in ("snapshot", "path", "offset", "max_bytes")}
                 for row in chunks]
    arguments.extend({"snapshot": snapshot, "path": "input.txt", "offset": 0, "max_bytes": 65536}
                     for snapshot in ("before", "after"))
    calls = tuple(ToolCall(call_id=ids.new("read"), name="workspace_read", arguments=row)
                  for row in arguments)
    # The fixture really reads the original bytes. Merely attaching the plan
    # would not satisfy the unchanged read-byte coverage validator below.
    results = tools.execute_group(calls)
    assert all(result.status == "completed" for result in results)
    events = [
        _trace_event(ids, "system", "Read evidence."),
        _trace_event(ids, "user", {"judge_only_reference": request.judge_only_reference}),
        _trace_event(ids, "assistant", "Execute planned reads.", tuple(call.call_id for call in calls)),
        *(_trace_event(ids, "tool", {"receipt": row.receipt_blake3}, (row.call_id,)) for row in results),
        _trace_event(ids, "assistant", "Structured rubric assessment."),
    ]
    trace = tools.trace(policy_events=events, provider_turn_count=2)
    scores = tuple(RubricItemScore(
        item_id=item["item_id"], score=1.0,
        evidence_refs=("workspace:before:input.txt",), rationale="Fixture source bytes inspected.",
    ) for item in rubric.items)
    core = {
        "judgment_id": request.judgment_id, "judge_model_id": request.judge_model_id,
        "rubric_digest": rubric.digest, "agent_trace": trace, "item_scores": scores,
        "hard_gates_passed": True, "summary": "Scripted complete-read fixture.",
    }
    return JudgeAssessment(**core, assessment_blake3=blake3_hex(core))


@pytest.mark.parametrize("omit_tail", [False, True])
def test_planned_real_byte_reads_pass_existing_validator_but_missing_tail_does_not(
    tmp_path, monkeypatch, omit_tail,
):
    prepared = _prepared(tmp_path)
    # Mock only the model-facing judge method; the selected profile supplies
    # the real plan, and canonical tools/coverage/scoring remain actual code.
    monkeypatch.setattr(CodexOpus5AgentJudge, "judge", lambda self, request, rubric:
                        _planned_workspace_assessment(request, rubric, omit_message_tail=omit_tail))
    judge = _judge(tmp_path, prepared.task.judge_model_id, _Capture())
    if omit_tail:
        with pytest.raises(AgentJudgeSelectionError, match="did not completely read every actor evidence sidecar"):
            grade_prepared_rollout(prepared, judge=judge, output_root=tmp_path)
        assert not (tmp_path / "score.json").exists()
    else:
        grade = grade_prepared_rollout(prepared, judge=judge, output_root=tmp_path)
        assert grade["workspace_inspected"] is True
        assert grade["judge_tool_trace"]["workspace_read_count"] > 10
        assert grade["judge_tool_trace"]["retry_count"] == 0
        assert grade["rubric_digest"] == prepared.trajectory.rubric.digest


@pytest.mark.parametrize("fault", ["extra_score_note", "cited_but_unread_file", None])
def test_actual_shaped_report_and_vqa_verdict_errors_still_fail_native_validation(tmp_path, fault):
    """Exact observed failure shapes, through real CodexRuntime + scripted backend."""
    from test_codex_pipeline_adapter import _RuntimeFactory, _judge_options, _judge_script

    mentioned = "notes/options-file-observation-v2.md"
    prepared = _prepared(tmp_path, message_bytes=100, extra_after={mentioned: b"fixture note\n"})
    base = _judge_script(prepared.evidence, prepared.trajectory.rubric,
                         path="input.txt", cite="workspace:after:input.txt")

    def script(thread, turn, bridge=None):
        notifications = []
        for notification in base(thread, turn, bridge):
            item = notification.payload.get("item", {})
            if item.get("type") == "agentMessage":
                terminal = json.loads(item["text"])
                if fault == "extra_score_note":
                    # Report's IDs were correct; only this extra first-row key broke schema.
                    terminal["item_scores"][0]["score_note"] = "fixture extra field"
                elif fault == "cited_but_unread_file":
                    # VQA appended an existing but unread note to its sixth row.
                    terminal["item_scores"][-1]["evidence_refs"].append(f"workspace:after:{mentioned}")
                notification = replace(notification, payload={
                    **notification.payload, "item": {**item, "text": json.dumps(terminal)},
                })
            notifications.append(notification)
        return tuple(notifications)

    factory = _RuntimeFactory(script)
    judge = PolicyVisibleOpusWorkspaceJudge(
        factory, _judge_options, model_id=prepared.task.judge_model_id,
        compact_evidence_index=True, fast_workspace_judge=True,
        maximum_workspace_tool_calls=64, maximum_workspace_tool_frontiers=32,
        turn_timeout_seconds=600,
    )
    request = JudgeRequest(
        judgment_id=prepared.task.judge_task_id, judge_model_id=prepared.task.judge_model_id,
        policy_visible_context=prepared.evidence.policy_visible_context,
        workspace_evidence=prepared.evidence, judge_only_reference={"fixture": True},
    )
    if fault:
        message = "row identity differs" if fault == "extra_score_note" else "did not inspect"
        with pytest.raises(CodexPipelineError, match=message):
            judge.judge(request, prepared.trajectory.rubric)
        # A failed structured verdict is never rewritten or reissued by the guidance.
        with pytest.raises(CodexPipelineError, match="already consumed"):
            judge.judge(request, prepared.trajectory.rubric)
    else:
        assessment = judge.judge(request, prepared.trajectory.rubric)
        assert [row.item_id for row in assessment.item_scores] == [
            item["item_id"] for item in prepared.trajectory.rubric.items
        ]
        assert assessment.agent_trace.inspected_evidence_refs == ("workspace:after:input.txt",)
    assert sum(len(backend.turns) for backend in factory.backends) == 1
