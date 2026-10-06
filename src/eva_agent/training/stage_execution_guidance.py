"""Shared, public execution advice; never a tool-schema or permission change."""
from __future__ import annotations

from eva_agent.pipeline.contracts import Stage


GUIDANCE_VERSION = "eva.stage-execution-guidance.v2"


def stage_execution_guidance(stage: Stage | str) -> str:
    """Return the same additive advice for training and evaluation consumers.

    S1/S2/E2E are unchanged. Consumers retain the returned text with their actual
    phase context; this helper does not execute, repair, or validate model code.
    """
    focus = Stage(stage)
    if focus not in {Stage.S3, Stage.S4, Stage.S5}:
        return ""
    execution = (
        f"When execute_code is offered and permitted for {focus.value}, emit its "
        f"required stage='{focus.value}' argument BEFORE the long code argument. "
        if focus is not Stage.S5 else
        "For S5, inspect actual prior results and use only the currently permitted "
        "review or terminal actions; do not introduce a new execution stage. "
        "If execute_code is explicitly permitted, put its required, permitted "
        "stage argument BEFORE the long code argument. "
    )
    return (
        f"PUBLIC BUDGET-AWARE EXECUTION GUIDANCE ({GUIDANCE_VERSION}; {focus.value}): "
        + execution
        + "This ordering advice applies only to execute_code: never add a stage "
        "argument to another tool whose schema does not declare it. "
        "Keep each execute_code call compact, syntactically complete, and fully closed. "
        "Independent permitted parallel-safe reads, searches, and load_skill calls may "
        "run concurrently. Keep dependent calls, writes, and execution steps ordered, "
        "and observe their results before the next dependent action. "
        "Keep code comfortably within the current response budget; avoid embedding "
        "long case prose, repeated data, or verbose diagnostics. Read the actual "
        "declared input files instead. Use bounded file inspection, small writes or "
        "edits, and execution steps only where the offered tools permit them. "
        "Observe each result before revising the code. Do not assume files persist "
        "between execution calls: a fresh-workspace contract requires a self-contained "
        "program producing all of that call's declared outputs. Write only allowed "
        "artifacts and inspect their real contents. Use supplied, bound digests or "
        "compute hashes from actual file bytes; never hash placeholder text or invent "
        "hashes, paths, execution results, or completion. If the budget or execution "
        "ends before completion, report the observed partial state. These instructions "
        "do not change tool schemas, stage permissions, output contracts, or scoring, "
        "and do not guarantee task success."
    )
