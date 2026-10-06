"""Admission-gated slicing of policy-visible trajectories into SFT examples."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    AdmissionEvidence,
    AdmissionEvidenceVerifier,
    Cohort,
    ContractError,
    ModelEvaluation,
    PipelineResult,
    RuntimeIdFactory,
    SFTExample,
    TrajectoryEvent,
    uuid_text,
)
from .digests import blake3_hex


class SFTSliceError(ContractError):
    """A trajectory is not safely eligible for supervised slicing."""


def _forbidden_structure(value: Any) -> str | None:
    forbidden = {
        "answer_key",
        "chain_of_thought",
        "gold",
        "gold_answer",
        "hidden_reasoning",
        "judge_only_reference",
        "private_reference",
        "reference_answer",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if normalized in forbidden:
                return str(key)
            found = _forbidden_structure(child)
            if found is not None:
                return found
    elif isinstance(value, (tuple, list)):
        for child in value:
            found = _forbidden_structure(child)
            if found is not None:
                return found
    return None


def _validate_visible_timeline(events: tuple[TrajectoryEvent, ...]) -> None:
    declared: set[str] = set()
    observed: set[str] = set()
    for event in events:
        if event.role == "assistant":
            if set(event.tool_call_ids) & declared:
                raise SFTSliceError("trajectory reuses a tool-call identity")
            declared.update(event.tool_call_ids)
        elif event.role == "tool":
            if len(event.tool_call_ids) != 1:
                raise SFTSliceError("a tool observation must bind exactly one prior call")
            call_id = event.tool_call_ids[0]
            if call_id not in declared or call_id in observed:
                raise SFTSliceError("tool observation precedes, duplicates, or replaces its call")
            observed.add(call_id)
    if observed != declared:
        raise SFTSliceError("trajectory omits a policy-visible tool observation")


def _lineage_core(example: SFTExample) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "sandbox_id": example.sandbox_id,
        "source_rollout_id": example.source_rollout_id,
        "source_bundle_blake3": example.source_bundle_blake3,
        "source_evaluation_blake3": example.source_evaluation_blake3,
        "source_pipeline_result_blake3": example.source_pipeline_result_blake3,
        "rubric": example.rubric,
        "prefix_event_blake3": tuple(event.event_blake3 for event in example.prefix),
        "supervised_assistant_decision_blake3": (
            example.supervised_assistant_decision.event_blake3
        ),
        "atomic_tool_call_group": example.atomic_tool_call_group,
        "signed_admission_receipt_blake3": example.signed_admission_receipt_blake3,
        "signed_supervisor_transition_blake3": (
            example.signed_supervisor_transition_blake3
        ),
        "private_reference_included": example.private_reference_included,
        "hidden_reasoning_included": example.hidden_reasoning_included,
    }


def slice_admitted_trajectory(
    *,
    result: PipelineResult,
    evaluation: ModelEvaluation,
    admission: AdmissionEvidence,
    verifier: AdmissionEvidenceVerifier,
    id_factory: RuntimeIdFactory,
) -> tuple[SFTExample, ...]:
    """Create one example per visible assistant decision after signed admission.

    A same-turn group of independent tool calls remains one supervised
    assistant target.  Its tool observations are not targets and appear only
    in the prefixes of subsequent decisions.
    """

    from eva_agent.codex_pipeline.budget import PROJECTION as BUDGET_PROJECTION, WORKSPACE_PROJECTION

    provider_metadata = evaluation.evidence.safe_provider_metadata
    if (
        provider_metadata.get("schema") in {BUDGET_PROJECTION, WORKSPACE_PROJECTION}
        or "controlled_budget_termination" in provider_metadata
        or "workspace_terminal" in provider_metadata
        or provider_metadata.get("sft_eligible") is False
    ):
        # A real workspace Judge may score a budget-bounded RL trajectory, but
        # that does not turn its host terminal marker (or preceding partial
        # decisions) into an admitted SFT target. Signed admission cannot
        # override this explicit source eligibility boundary.
        raise SFTSliceError("controlled-budget or explicitly SFT-ineligible trajectory cannot be sliced")

    admission_policy_passed = (
        result.separation.policy_passed
        if result.separation.admission_policy_schema is None
        else result.separation.admission_policy_passed
    )
    if not admission_policy_passed:
        raise SFTSliceError("pipeline result is not eligible for signed admission")
    if (
        result.separation.admission_policy_schema is None
        and evaluation.cohort is not Cohort.STRONG
    ):
        raise SFTSliceError("only the admitted strong-model golden trajectory may be sliced")
    if (
        result.separation.admission_policy_schema is not None
        and evaluation.cohort.value != result.separation.selected_evaluation_cohort
    ):
        raise SFTSliceError(
            "only the selected strong-model or v2 admitted trajectory may be sliced"
        )
    if evaluation not in result.evaluations:
        raise SFTSliceError("evaluation is not bound to this pipeline result")
    if (
        admission.pipeline_result_blake3 != result.result_blake3
        or admission.evaluation_blake3 != evaluation.evaluation_blake3
    ):
        raise SFTSliceError(
            "only the selected strong-model or v2 admitted trajectory may be sliced; admission lineage differs"
        )
    if not verifier.verify(result=result, evaluation=evaluation, evidence=admission):
        raise SFTSliceError("signed admission evidence did not verify")
    events = evaluation.evidence.policy_events
    _validate_visible_timeline(events)
    forbidden = _forbidden_structure(
        {
            "policy_visible_context": evaluation.evidence.policy_visible_context,
            "policy_events": tuple(event.content for event in events),
        }
    )
    if forbidden is not None:
        raise SFTSliceError(f"private or hidden field entered visible trajectory: {forbidden!r}")

    examples: list[SFTExample] = []
    for index, decision in enumerate(events):
        if decision.role != "assistant":
            continue
        example_id = id_factory.new("sft-example")
        uuid_text(example_id, label="SFT example_id")
        provisional = SFTExample(
            example_id=example_id,
            sandbox_id=evaluation.evidence.sandbox_manifest.sandbox_id,
            source_rollout_id=evaluation.evidence.rollout_id,
            source_bundle_blake3=evaluation.evidence.bundle_blake3,
            source_evaluation_blake3=evaluation.evaluation_blake3,
            source_pipeline_result_blake3=result.result_blake3,
            rubric=evaluation.evidence.sandbox_manifest.rubric,
            prefix=tuple(events[:index]),
            supervised_assistant_decision=decision,
            atomic_tool_call_group=len(decision.tool_call_ids) > 1,
            signed_admission_receipt_blake3=admission.signed_admission_receipt_blake3,
            signed_supervisor_transition_blake3=(
                admission.signed_supervisor_transition_blake3
            ),
            private_reference_included=False,
            hidden_reasoning_included=False,
            lineage_blake3="",
        )
        examples.append(
            SFTExample(
                **{
                    **provisional.__dict__,
                    "lineage_blake3": blake3_hex(_lineage_core(provisional)),
                }
            )
        )
    if not examples:
        raise SFTSliceError("admitted trajectory contains no assistant decision")
    return tuple(examples)


__all__ = ["SFTSliceError", "slice_admitted_trajectory"]
