from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from eva_agent.pipeline import (
    AdmissionEvidence,
    Cohort,
    DeterministicUUIDFactory,
    SFTSliceError,
    TrajectoryEvent,
    slice_admitted_trajectory,
)
from eva_agent.pipeline.digests import blake3_hex
from eva_agent.pipeline.runner import evidence_core, evaluation_core, result_core

from test_pipeline_e2e import _build_pipeline


class _AdmissionVerifier:
    def verify(self, *, result, evaluation, evidence):
        return (
            evidence.pipeline_result_blake3 == result.result_blake3
            and evidence.evaluation_blake3 == evaluation.evaluation_blake3
            and evidence.signed_admission_receipt_blake3 == blake3_hex("signed-admission")
            and evidence.signed_supervisor_transition_blake3 == blake3_hex("signed-transition")
        )


def test_sft_slice_keeps_multi_tool_decision_atomic_and_lineage(tmp_path: Path) -> None:
    pipeline, episode, rubric, targets, _artifacts, _clients, _judge = _build_pipeline(tmp_path)
    result = pipeline.run(episode=episode, rubric=rubric, targets=targets)
    strong = next(row for row in result.evaluations if row.cohort is Cohort.STRONG)
    admission = AdmissionEvidence(
        pipeline_result_blake3=result.result_blake3,
        evaluation_blake3=strong.evaluation_blake3,
        signed_admission_receipt_blake3=blake3_hex("signed-admission"),
        signed_supervisor_transition_blake3=blake3_hex("signed-transition"),
    )
    examples = slice_admitted_trajectory(
        result=result,
        evaluation=strong,
        admission=admission,
        verifier=_AdmissionVerifier(),
        id_factory=DeterministicUUIDFactory("sft-slice"),
    )

    assert len(examples) == 2
    tool_decision, final_decision = examples
    assert tool_decision.atomic_tool_call_group is True
    assert len(tool_decision.supervised_assistant_decision.tool_call_ids) == 2
    assert all(event.role != "tool" for event in tool_decision.prefix)
    assert [event.role for event in final_decision.prefix].count("tool") == 2
    assert all(example.private_reference_included is False for example in examples)
    assert all(example.hidden_reasoning_included is False for example in examples)
    assert all(example.source_pipeline_result_blake3 == result.result_blake3 for example in examples)
    assert all(example.signed_admission_receipt_blake3 == blake3_hex("signed-admission") for example in examples)
    assert all(len(example.lineage_blake3) == 64 for example in examples)

    weak = next(row for row in result.evaluations if row.cohort is Cohort.WEAK)
    with pytest.raises(SFTSliceError, match="strong-model"):
        slice_admitted_trajectory(
            result=result,
            evaluation=weak,
            admission=replace(admission, evaluation_blake3=weak.evaluation_blake3),
            verifier=_AdmissionVerifier(),
            id_factory=DeterministicUUIDFactory("bad-slice"),
        )


def test_sft_slice_preserves_native_action_decision_and_observation_order(
    tmp_path: Path,
) -> None:
    """A verified native sidecar stays sliceable without changing SFT schema."""

    pipeline, episode, rubric, targets, _artifacts, _clients, _judge = _build_pipeline(tmp_path)
    result = pipeline.run(episode=episode, rubric=rubric, targets=targets)
    strong = next(row for row in result.evaluations if row.cohort is Cohort.STRONG)
    ids = DeterministicUUIDFactory("sft-native-sidecar")
    call_id = ids.new("native-call")

    def event(role: str, content, tool_call_ids=()):
        core = {
            "event_id": ids.new("native-event"),
            "role": role,
            "content": content,
            "tool_call_ids": tuple(tool_call_ids),
        }
        return TrajectoryEvent(**core, event_blake3=blake3_hex(core))

    native_decision = event(
        "assistant",
        {
            "codex_tool_calls": [{
                "call_id": call_id,
                "upstream_item_id": "native-file-change",
                "tool_type": "fileChange",
                "name": "fileChange",
                "arguments": {
                    "changes": [{
                        "path": "result.md",
                        "kind": {"type": "add"},
                        "diff": "+verified\n",
                    }]
                },
                "receipt_blake3": blake3_hex("native-call-receipt"),
            }]
        },
        (call_id,),
    )
    native_observation = event(
        "tool",
        {
            "codex_native_action": {
                "tool_type": "fileChange",
                "name": "fileChange",
                "status": "completed",
                "output": {"status": "completed"},
                "receipt_blake3": blake3_hex("native-call-receipt"),
            }
        },
        (call_id,),
    )
    events = (
        *strong.evidence.policy_events[:-1],
        native_decision,
        native_observation,
        strong.evidence.policy_events[-1],
    )
    partial_evidence = replace(
        strong.evidence, policy_events=events, bundle_blake3=""
    )
    evidence = replace(
        partial_evidence,
        bundle_blake3=blake3_hex(evidence_core(partial_evidence)),
    )
    partial_evaluation = replace(
        strong, evidence=evidence, evaluation_blake3=""
    )
    evaluation = replace(
        partial_evaluation,
        evaluation_blake3=blake3_hex(evaluation_core(partial_evaluation)),
    )
    evaluations = tuple(
        evaluation if row.cohort is Cohort.STRONG else row
        for row in result.evaluations
    )
    partial_result = replace(result, evaluations=evaluations, result_blake3="")
    projected_result = replace(
        partial_result, result_blake3=blake3_hex(result_core(partial_result))
    )
    admission = AdmissionEvidence(
        pipeline_result_blake3=projected_result.result_blake3,
        evaluation_blake3=evaluation.evaluation_blake3,
        signed_admission_receipt_blake3=blake3_hex("signed-admission"),
        signed_supervisor_transition_blake3=blake3_hex("signed-transition"),
    )

    examples = slice_admitted_trajectory(
        result=projected_result,
        evaluation=evaluation,
        admission=admission,
        verifier=_AdmissionVerifier(),
        id_factory=ids,
    )

    native_example = next(
        row for row in examples if row.supervised_assistant_decision.event_id == native_decision.event_id
    )
    final_example = next(
        row
        for row in examples
        if row.supervised_assistant_decision.event_id
        == strong.evidence.policy_events[-1].event_id
    )
    assert native_example.atomic_tool_call_group is False
    assert native_example.supervised_assistant_decision.tool_call_ids == (call_id,)
    assert native_observation in final_example.prefix


@pytest.mark.parametrize("eligibility_metadata", [
    {"schema": "eva.codex-provider-rollout-projection.v3-controlled-budget"},
    {"schema": "eva.codex-provider-rollout-projection.v4-workspace-terminal"},
    {"controlled_budget_termination": {"sft_eligible": False}},
    {"sft_eligible": False},
])
def test_sft_rejects_budget_partial_and_host_marker_even_with_signed_admission(
    tmp_path: Path, eligibility_metadata,
) -> None:
    from eva_agent.codex_pipeline.budget import TERMINAL_MARKER

    pipeline, episode, rubric, targets, *_ = _build_pipeline(tmp_path)
    result = pipeline.run(episode=episode, rubric=rubric, targets=targets)
    strong = next(row for row in result.evaluations if row.cohort is Cohort.STRONG)
    terminal = strong.evidence.policy_events[-1]
    terminal = replace(terminal, content=TERMINAL_MARKER, event_blake3=blake3_hex({
        "event_id": terminal.event_id, "role": terminal.role,
        "content": TERMINAL_MARKER, "tool_call_ids": terminal.tool_call_ids,
    }))
    partial = replace(strong.evidence, safe_provider_metadata=eligibility_metadata,
                      assistant_output=TERMINAL_MARKER,
                      policy_events=(*strong.evidence.policy_events[:-1], terminal), bundle_blake3="")
    evidence = replace(partial, bundle_blake3=blake3_hex(evidence_core(partial)))
    partial_evaluation = replace(strong, evidence=evidence, evaluation_blake3="")
    evaluation = replace(partial_evaluation, evaluation_blake3=blake3_hex(evaluation_core(partial_evaluation)))
    partial_result = replace(result, evaluations=tuple(
        evaluation if row.cohort is Cohort.STRONG else row for row in result.evaluations), result_blake3="")
    result = replace(partial_result, result_blake3=blake3_hex(result_core(partial_result)))
    admission = AdmissionEvidence(
        pipeline_result_blake3=result.result_blake3,
        evaluation_blake3=evaluation.evaluation_blake3,
        signed_admission_receipt_blake3=blake3_hex("signed-admission"),
        signed_supervisor_transition_blake3=blake3_hex("signed-transition"),
    )
    assert _AdmissionVerifier().verify(result=result, evaluation=evaluation, evidence=admission)
    with pytest.raises(SFTSliceError, match="SFT-ineligible"):
        slice_admitted_trajectory(result=result, evaluation=evaluation, admission=admission,
                                 verifier=_AdmissionVerifier(), id_factory=DeterministicUUIDFactory("no-budget-sft"))
