"""Prospective focus-only wrapper; canonical stage guidance remains unchanged."""
from __future__ import annotations

from eva_agent.pipeline import Stage
from .teacher_batch import TeacherBatchError
from .teacher_worker import TeacherCandidateContext, teacher_actor_developer_instructions


def focus_only_teacher_instructions(context: TeacherCandidateContext) -> str:
    guidance = context.stage_tool_guidance
    if guidance is None or guidance.focus not in {Stage.S1, Stage.S2, Stage.S3}:
        raise TeacherBatchError("focus-only teacher requires S1, S2, or S3 guidance")
    if guidance.focus is Stage.S3:
        original = teacher_actor_developer_instructions(context)
        return (
            f"{original}\n\n"
            "SUPPLEMENTAL S3-ONLY ROLLOUT CONTRACT (prospective runtime wrapper): "
            "The host has already verified the S1 and S2 prerequisites host-side. "
            "Those hydrated artifacts are not promised inside the separate execution container. "
            "Each execute_code call starts with a fresh empty writable /workspace; "
            "the actual public contract inputs are mounted read-only beneath /inputs. "
            "Read each declared input at /inputs/<contract-relative-path>, or legitimately "
            "list /inputs to resolve its public mounted paths. Do not read hydrated "
            "work/stage-plan.json or work/evidence-selection.json from the execution "
            "container. Create the parent directories of the contract's output artifact "
            "beneath /workspace before writing it. "
            "Leave exactly the output file or files declared by the bound contract, "
            "and no extra files in /workspace. Do not write diagnostic, trace, scratch, "
            "or provenance sidecar files; keep such diagnostics in memory or stdout, "
            "and include required evidence within the declared artifact's schema. "
            "Use the unchanged execute_code MCP tool with stage='S3' and authored Python "
            "code satisfying the supplied bound pilot-execution contract. Execution is "
            "performed by the existing bounded host tool, not by Codex shell or network "
            "commands. Stop tool execution immediately after execute_code for S3 reports "
            "gate_passed=true with no failed checks, then summarize only the observed "
            "pilot result and receipt/artifact references actually returned by the host. "
            "Do not recreate S1/S2, call execute_code with stage='S4', or call submit_results. "
            "Do not continue because a result advertises a later stage. If S3 fails, retain "
            "and report the actual failure without claiming a successful pilot. "
            "The skill registry remains bound to S3: every search_skills or load_skill "
            "call must pass stage='S3'. Load a relevant visible skill when useful, using "
            "only IDs actually advertised by the current catalog or successful discovery. "
            "Do not invent skill IDs, widen permissions, or force irrelevant skill calls."
        )
    stage = guidance.focus.value
    terminal = "materialize_plan" if guidance.focus is Stage.S1 else "materialize_evidence_selection"
    original = teacher_actor_developer_instructions(context)
    return (
        f"{original}\n\n"
        "SUPPLEMENTAL FOCUS-ONLY ROLLOUT CONTRACT (prospective runtime wrapper): "
        f"This task evaluates only {stage}. Complete its required prerequisites and "
        f"stop tool execution immediately after {terminal} reports gate_passed=true. "
        "Then give a brief factual summary naming the materialized artifact and "
        "the completed focus stage. Do not attempt any later-stage frontier, "
        "execute_code, or submit_results, even if host output advertises next_stage. "
        "This stopping instruction takes precedence over earlier instructions "
        "to complete all guidance frontiers. If the focus gate fails, report the "
        "observed failure honestly; do not claim successful completion. "
        f"The skill registry remains bound to {stage} throughout this task: "
        f"every search_skills or load_skill call must pass stage='{stage}', "
        "regardless of host next_stage. Use only exact skill IDs advertised by "
        "the current stage-visible catalog or returned by successful discovery. "
        "When a relevant visible skill can help the focus-stage task, load it "
        "before completing the gate; never invent a skill ID or widen permissions."
    )
