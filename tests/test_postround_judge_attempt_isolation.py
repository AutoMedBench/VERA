"""Provider-free contract tests for opt-in post-round Judge attempts."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json

import pytest

from eva_agent.codex_pipeline import CodexPipelineError
from eva_agent.codex_runtime import (
    CodexRole, CodexRuntime, CodexSandbox, CodexThreadOptions, CodexTurnInput,
)
from eva_agent.pipeline import DeterministicUUIDFactory
from eva_agent.pipeline.digests import canonical_value
from eva_agent.training import slime_agent_judge
from test_automedbench_track_feedback import fixture
from test_codex_pipeline_adapter import _RuntimeFactory, _event, _judge_options
from test_codex_runtime import _FakeBackend, _simple_script
from training.automedbench_lite import track_feedback
from training.automedbench_lite.judge_attempt_isolation import (
    BINDING_NAME, replacement_limit, run_stage_judge_attempts, verify_stage_judge_attempts,
)
from training.automedbench_lite.track_adapter import BY_TRACK
from training.eva_rsi.production_eval import evaluation_scope, judge_full_round


def _completed_judge_receipt(tmp_path):
    options = CodexThreadOptions(
        role=CodexRole.JUDGE, model="gpt-6-astra", provider="eva_native_astra",
        cwd=str(tmp_path.resolve()), sandbox=CodexSandbox.READ_ONLY,
        base_instructions="Provider-free fixture.")
    value = CodexTurnInput(public_text="Inspect the fixture.",
                           judge_only_context={"fixture": "private-reference"})

    async def generate():
        async with CodexRuntime(_FakeBackend(_simple_script)) as runtime:
            handle = await runtime.start_thread(options)
            return await runtime.run_turn(handle, value)

    return asyncio.run(generate())


def _complete_zero_judge_script(evidence, rubric, cite):
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
                       "evidence_refs": [cite], "rationale": "Inspected committed workspace content."})
    terminal = json.dumps({"item_scores": scores, "hard_gates_passed": hard_passed,
                           "summary": "verified"})

    def script(thread, turn, bridge=None):
        assert bridge is not None
        events = [_event("turn/started", thread, turn,
                         turn={"id": turn, "status": "inProgress"})]
        started, completed = [], []
        observations = bridge.execute_group(tuple(("workspace_read", {
            "snapshot": snapshot, "path": selected.path, "offset": 0, "max_bytes": 65536,
        }) for snapshot, selected in reads))
        for index, observation in enumerate(observations):
            call = {"id": f"judge-read-{index}", "type": "mcpToolCall",
                    "server": "evamed-judge", "tool": "workspace_read",
                    "arguments": canonical_value(observation.call.arguments),
                    "status": "inProgress"}
            started.append(_event("item/started", thread, turn, item=call))
            completed.append(_event("item/completed", thread, turn,
                item={**call, "status": "completed", "result": {
                    "content": [{"type": "text", "text": "committed observation"}],
                    "structuredContent": canonical_value(observation.structured_content),
                    "isError": False}}))
        events.extend((*started, *completed))
        events.extend((_event("item/completed", thread, turn,
            item={"id": "answer", "type": "agentMessage", "phase": "final_answer",
                  "text": terminal}),
            _event("turn/completed", thread, turn, turn={"id": turn, "status": "completed"})))
        return tuple(events)

    return script


def test_actual_core_invalid_verdict_then_first_valid_zero_is_accepted(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    stage_root = args["output_root"] / "S1"
    prepared_root = stage_root / "prepared"
    _, rubric, _ = track_feedback.prepare_track_feedback(
        run_root=args["run_root"], checkpoint_identity=args["checkpoint_identity"],
        output_root=prepared_root, stage="S1", track="classification")
    invalid_receipt = _completed_judge_receipt(tmp_path)
    opened = 0

    class InvalidJudge:
        transport_frontier_audit = None

        def judge(self, request, rubric):
            raise CodexPipelineError("judge terminal response is not JSON",
                                     receipt=invalid_receipt)

        def receipt_for(self, judgment_id):
            return invalid_receipt

    @contextmanager
    def native(prepared, output_root, **options):
        nonlocal opened
        opened += 1
        if opened == 1:
            yield InvalidJudge()
            return
        cite = next(ref for ref in prepared.source_workspace_refs
                    if ref.startswith("workspace:after:"))
        holder = {}
        script = _complete_zero_judge_script(prepared.evidence, rubric, cite)
        factory = _RuntimeFactory(lambda thread, turn, unused: script(
            thread, turn, holder["bridge"]))

        class BoundMCP:
            @contextmanager
            def open_judge(self, options, bridge):
                holder["bridge"] = bridge
                yield options

        yield slime_agent_judge.NativeAstraWorkspaceJudge(
            factory,
            lambda request, offers: replace(
                _judge_options(request, offers), provider="eva_native_astra"),
            id_factory=DeterministicUUIDFactory("postround-valid-judge"),
            turn_mcp_factory=BoundMCP(), compact_evidence_index=True,
            fast_workspace_judge=True, maximum_workspace_tool_frontiers=32)

    monkeypatch.setattr(slime_agent_judge, "_open_native_astra_judge", native)
    accepted, verification, binding = run_stage_judge_attempts(
        stage_root, prepared_root, replacements=1, judge=track_feedback.judge_track_once,
        verify=track_feedback.verify_feedback)

    assert opened == 2 and verification["valid"] and verification["status"] == "scored"
    assert verification["score"]["reward_bps"] == 0
    assert binding == accepted / BINDING_NAME
    reopened = verify_stage_judge_attempts(binding, verify=track_feedback.verify_feedback)
    assert reopened["valid"] and reopened["score"]["reward_bps"] == 0
    policy = json.loads((stage_root / "judge-attempt-isolation.json").read_text())
    assert policy["judge_attempts_made"] == 2
    assert policy["additional_judge_attempts_made"] == 1
    assert policy["first_valid_grade_accepted"] is True
    assert policy["actor_rerun_by_policy"] is False
    first = sorted((stage_root / "attempts").glob("*/result.json"),
                   key=lambda path: json.loads(path.read_text())["attempt_index"])[0]
    rejected = json.loads(first.read_text())
    assert rejected["replacement_eligible"] is True and rejected["score"] is None


def test_non_verdict_failure_is_not_retried_or_converted_to_zero(tmp_path):
    args = fixture(tmp_path)
    stage_root = args["output_root"] / "S1"
    prepared_root = stage_root / "prepared"
    track_feedback.prepare_track_feedback(
        run_root=args["run_root"], checkpoint_identity=args["checkpoint_identity"],
        output_root=prepared_root, stage="S1", track="classification")
    calls = 0

    def unavailable(root):
        nonlocal calls
        calls += 1
        raise TimeoutError("provider-free fixture")

    with pytest.raises(TimeoutError):
        run_stage_judge_attempts(stage_root, prepared_root, replacements=2,
                                 judge=unavailable, verify=lambda root: {})
    assert calls == 1
    result = json.loads(next((stage_root / "attempts").glob("*/result.json")).read_text())
    assert result["replacement_eligible"] is False and result["score"] is None
    policy = json.loads((stage_root / "judge-attempt-isolation.json").read_text())
    assert policy["status"] == "unavailable" and policy["replacement_limit_exhausted"] is False


def test_default_track_feedback_keeps_historical_stage_layout(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    monkeypatch.setattr(track_feedback, "judge_track_once",
                        lambda root: {"valid": True, "status": "scored"})
    monkeypatch.setattr(track_feedback, "verify_feedback", lambda root: {
        "valid": True, "status": "scored", "score": {"reward_bps": 0},
        "source_rollout_blake3": "f" * 64})
    roots = track_feedback.evaluate_track_feedback(
        args["run_root"], args["checkpoint_identity"], args["output_root"])
    assert roots == (args["output_root"] / "S1",)
    assert (roots[0] / "rollout.json").is_file()
    assert not (roots[0] / "prepared").exists()
    assert not (roots[0] / "attempts").exists()
    assert "postround_judge_verdict_replacements" not in json.loads(
        (args["output_root"] / "coverage.json").read_text())


def test_full_round_routes_bounded_additional_judge_attempt_setting(tmp_path, monkeypatch):
    scope = evaluation_scope({"evaluation_mode": "full_single_pass"})
    calls = []

    def boundary(run, identity, output, *, stages, track, registry_path,
                 postround_judge_verdict_replacements):
        calls.append((track, postround_judge_verdict_replacements))
        return tuple(output / stage for stage in stages)

    monkeypatch.setattr(track_feedback, "evaluate_track_feedback", boundary)
    roots = judge_full_round(tmp_path / "run", tmp_path / "identity",
        tmp_path / "feedback", stages=scope["stages"], registry_path=scope["registry"],
        postround_judge_verdict_replacements=2)
    assert len(roots) == 35
    assert set(calls) == {(track, 2) for track in BY_TRACK}
    coverage = json.loads((tmp_path / "feedback/coverage.json").read_text())
    assert coverage["postround_judge_verdict_replacements"] == 2
    assert coverage["first_valid_grade_including_zero_is_accepted"] is True
    assert coverage["additional_judge_attempts_are_not_actor_rerolls"] is True
    for value in (-1, 3, True, 1.5):
        with pytest.raises(ValueError):
            replacement_limit(value)
