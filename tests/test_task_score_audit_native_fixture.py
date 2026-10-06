"""Provider-free native wrapper integration; no benchmark or clinical score."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json

from eva_agent.codex_runtime import CodexRole, CodexSandbox, CodexThreadOptions
from eva_agent.pipeline import DeterministicUUIDFactory
from eva_agent.pipeline.digests import canonical_value
from eva_agent.training import slime_agent_judge as core
from test_automedbench_track_feedback import fixture
from test_codex_pipeline_adapter import _RuntimeFactory, _event
from test_eva_task_score_audit import source
from training.automedbench_lite import track_feedback
from training.benchmark_feedback.automed_codex import read_json, write_private
from training.eva_rsi import task_score_audit as audit
from training.eva_rsi.controller import write
from training.eva_rsi.evidence import commitment


def _zero_audit_script(evidence, rubric, summary):
    reads = [("after", row) for row in evidence.workspace_after.files]
    reads.append(("before", evidence.workspace_before.files[0]))
    scores, hard_passed = [], True
    for item in rubric.items:
        document = canonical_value(item)
        minimum = min(level["score_bps"] for level in document["partial_credit"]["levels"])
        gate = document.get("hard_gate")
        hard_passed = hard_passed and (not isinstance(gate, dict)
                                       or minimum >= gate["minimum_score_bps"])
        scores.append({"item_id": document["item_id"], "score": minimum / 10_000,
                       "evidence_refs": [summary["evidence_refs"][0]],
                       "rationale": "Inspected committed workspace content."})
    terminal = json.dumps({"item_scores": scores, "hard_gates_passed": hard_passed,
                           "summary": json.dumps(summary)})

    def script(thread, turn, bridge):
        observations = bridge.execute_group(tuple(("workspace_read", {
            "snapshot": snapshot, "path": selected.path, "offset": 0, "max_bytes": 65536,
        }) for snapshot, selected in reads))
        calls = [{"id": f"read-{index}", "type": "mcpToolCall",
                  "server": "evamed-judge", "tool": "workspace_read",
                  "arguments": canonical_value(observation.call.arguments), "status": "inProgress"}
                 for index, observation in enumerate(observations)]
        events = [_event("turn/started", thread, turn,
                         turn={"id": turn, "status": "inProgress"})]
        events.extend(_event("item/started", thread, turn, item=call) for call in calls)
        events.extend(_event("item/completed", thread, turn, item={**call, "status": "completed",
            "result": {"content": [{"type": "text", "text": "committed observation"}],
                       "structuredContent": canonical_value(observation.structured_content),
                       "isError": False}})
            for call, observation in zip(calls, observations, strict=True))
        events.extend((_event("item/completed", thread, turn,
            item={"id": "answer", "type": "agentMessage", "phase": "final_answer",
                  "text": terminal}),
            _event("turn/completed", thread, turn,
                   turn={"id": turn, "status": "completed"})))
        return tuple(events)

    return script


def test_native_task_score_decorator_and_startup_failure_boundary(tmp_path, monkeypatch):
    args = fixture(tmp_path / "actor")
    feedback_root = tmp_path / "feedback"
    _, rubric, _ = track_feedback.prepare_track_feedback(
        run_root=args["run_root"], checkpoint_identity=args["checkpoint_identity"],
        output_root=feedback_root, stage="S1", track="classification")
    preflight = read_json(feedback_root / "preflight.json")
    rollout = read_json(feedback_root / "rollout.json")
    prepared = core.prepare_workspace_rollout(
        rollout, rubric=rubric, source_path=feedback_root / "rollout.json",
        judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra")
    cite = next(ref for ref in prepared.source_workspace_refs
                if ref.startswith("workspace:after:"))
    summary = {"schema": audit.SUMMARY_SCHEMA, "native_result_consistent": True,
        "task_score_0_100": 0, "evidence_refs": [cite], "explanation": "Fixture checked."}
    annotation = {"schema": audit.SUMMARY_SCHEMA, "fixture": True}
    holder, observed = {}, []
    script = _zero_audit_script(prepared.evidence, rubric, summary)
    factory = _RuntimeFactory(lambda thread, turn, unused: script(thread, turn, holder["bridge"]))

    class BoundMCP:
        @contextmanager
        def open_judge(self, options, bridge):
            holder["bridge"] = bridge
            yield options

    class CapturingNative(core.NativeAstraWorkspaceJudge):
        def judge(self, request, selected_rubric):
            observed.append(request)
            return super().judge(request, selected_rubric)

    delegate = CapturingNative(factory, lambda request, offers: CodexThreadOptions(
        role=CodexRole.JUDGE, model=request.judge_model_id, provider="eva_native_astra",
        cwd=str(tmp_path.resolve()), sandbox=CodexSandbox.READ_ONLY, offered_tools=offers),
        id_factory=DeterministicUUIDFactory("task-score-native"),
        turn_mcp_factory=BoundMCP(), compact_evidence_index=True, fast_workspace_judge=True,
        maximum_workspace_tool_calls=64, maximum_workspace_tool_frontiers=32)
    inner = feedback_root / "judge" / preflight["sample_id"]
    inner.mkdir(parents=True)
    write_private(inner / "rollout.json", rollout)
    provenance = core._backend_provenance(
        prepared, "native_astra", native_turn_timeout_seconds=600,
        maximum_concurrent_judges=4)
    write_private(inner / "judge-backend.json", provenance)
    write_private(feedback_root / "judge-attempt.json", {
        "attempt_id": "provider-free", "backend": "native_astra", "semantic_attempt_count": 1,
        "retry_count": 0, "automatic_fallback": False, "native_turn_timeout_seconds": 600,
        "judge_implementation_blake3": commitment(core.__file__)["blake3"]})
    grade = core.grade_prepared_rollout(
        prepared, judge=audit.TaskScoreJudge(delegate, annotation), output_root=inner)
    receipt = delegate.receipt_for(prepared.task.judge_task_id)
    write_private(inner / "judge-codex-receipt.json", canonical_value(receipt))
    grade.update(judge_backend="native_astra", judge_provenance=provenance)
    write_private(inner / "grade.json", grade)

    verified = track_feedback.verify_feedback(feedback_root)
    assert verified["valid"] and verified["status"] == "scored"
    assert delegate._maximum_workspace_tool_frontiers == 32
    assert observed[0].judge_only_reference["task_score_audit"] == annotation

    row, _, _ = source(tmp_path / "native")
    result_path = tmp_path / "native-result.json"
    write(result_path, row)
    starts = 0

    @contextmanager
    def unavailable(*args, **kwargs):
        nonlocal starts
        starts += 1
        raise RuntimeError("provider-free startup failure")
        yield

    monkeypatch.setattr(audit, "bind_actor", lambda *args: None)
    monkeypatch.setattr(core, "_open_native_astra_judge", unavailable)
    failed = audit.run_task_audit(feedback_root, result_path, tmp_path / "unavailable")
    assert starts == 1 and failed["status"] == "unavailable"
    assert failed["task_score_0_100"] is None and failed["actor_rerolls"] == 0
    assert failed["additional_task_audit_attempts"] == 1
    failure = audit.read(failed["failure"]["path"])
    assert failure["automatic_retry"] is False and failure["failure_receipt_verified"] is None
