"""A separate workspace-agent audit of the native task score, not another A score.

Native benchmark computation remains the numerical authority. One fresh judge
checks its public aggregate and the same retained actor context/workspace. This
never rewrites the accepted process grades, reruns the actor, or exposes private
benchmark references. An unavailable audit is explicit and does not erase A/T.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from uuid import uuid4

from eva_agent.pipeline.digests import canonical_value
from eva_agent.training.slime_agent_judge import NativeAstraWorkspaceJudge
from .controller import write
from .evidence import commitment, read, require

SCHEMA = "eva.automedbench-task-score-audit.v1"
SUMMARY_SCHEMA = "eva.automedbench-task-score-review.v1"


def native_result(reference, *, track):
    require(commitment(reference["path"]) == reference, "task_audit_native_result_changed")
    row = read(reference["path"])
    require(row.get("track") == track and row.get("status") == "scored", "task_audit_native_unavailable")
    score = row.get("task_score_0_1")
    require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
            "task_audit_native_scale_invalid")
    source = row["native_score"]
    require(commitment(source["path"]) == source, "task_audit_native_score_changed")
    document = read(source["path"])
    require(document.get("schema") == "eva.automedbench-native-whole-track-score.v1"
            and document.get("track") == track and document.get("native_result") == row["native_result"]
            and row["native_result"].get("schema") == "eva.automedbench-native-track-result.v1"
            and row["native_result"].get("task_score_0_1") == score
            and row["native_result"].get("full_public_subset") is True
            and row["native_result"].get("private_values_exported") is False,
            "task_audit_native_source_invalid")
    return row, document


def bind_actor(row, document, rollout):
    reference = row["actor_rollout"]
    require(commitment(reference["path"]) == reference, "task_audit_actor_changed")
    actor = read(reference["path"])
    metadata = rollout["provider_metadata"]
    receipts = metadata["codex_turn_receipts"]
    require(actor.get("track") == row["track"] == metadata["track"]
        and actor.get("run_id") == document["run_id"]
        and actor.get("document_blake3") == document["actor_document_blake3"]
        and actor.get("completed_requested_turns") is True and not actor.get("errors")
        and metadata["evaluated_stage"] == "S5"
        and len(receipts) == 5
        and actor.get("turn_receipt_blake3s") == [r["receipt_blake3"] for r in receipts],
        "task_audit_must_match_complete_actor")
    files = {r["path"]: r for r in rollout["workspace_after"]["files"]}
    for submitted in document["submitted_files"]:
        retained = files.get("outputs/agents_outputs/" + submitted["path"], {})
        require(retained.get("content_blake3") == submitted["blake3"]
            and retained.get("byte_count") == submitted["bytes"], "task_audit_submission_not_in_snapshot")


def instruction(row, document):
    return {"schema": SUMMARY_SCHEMA, "native_task_result": row["native_result"],
        "submitted_files": document["submitted_files"],
        "instruction": (
            "This is a separate task-result audit, not a replacement process evaluation or a clinical claim. "
            "Inspect the actual actor context and before/after workspace with the offered read-only tools. "
            "Score the exact unchanged rubric as requested by the underlying Judge contract. "
            "hard_gates_passed means ONLY the conjunction of compiled non-null hard gates; with none it "
            "must be true even if every item score is zero. In the existing summary STRING return JSON "
            "with exactly schema, native_result_consistent, task_score_0_100, evidence_refs, explanation. "
            "schema is " + SUMMARY_SCHEMA + ". native_result_consistent is a boolean: judge whether the "
            "host-computed native result and retained submission inventory are consistent with the actual "
            "workspace you inspected. It is NOT whether performance is good. A legitimate zero may be "
            "consistent. task_score_0_100 is exactly native_task_result.task_score_0_1*100 when consistent, "
            "otherwise null; do not substitute a subjective score or recompute private reference metrics. "
            "evidence_refs is a nonempty list of original source workspace references actually read, "
            "including an after-workspace reference. If submissions exist, read and cite at least one "
            "actual submitted output or its source artifact. explanation is a brief observable conclusion, "
            "not private reasoning. The aggregate native score is judge-only and was not shown to the actor."
        )}


class TaskScoreJudge(NativeAstraWorkspaceJudge):
    """Keep native coverage limits while decorating only this fresh request."""
    def __init__(self, delegate, annotation):
        self.delegate = delegate
        self.annotation = annotation

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def judge(self, request, rubric):
        return self.delegate.judge(replace(request, judge_only_reference={
            **request.judge_only_reference, "task_score_audit": self.annotation}), rubric)


def validate_review(assessment, prepared, row, document):
    value = json.loads(assessment.summary)
    require(isinstance(value, dict) and set(value) == {
        "schema", "native_result_consistent", "task_score_0_100", "evidence_refs", "explanation"}
        and value["schema"] == SUMMARY_SCHEMA, "task_audit_summary_schema")
    accepted = value["native_result_consistent"]
    require(type(accepted) is bool, "task_audit_acceptance_type")
    expected = row["task_score_0_1"] * 100
    score = value["task_score_0_100"]
    require((accepted and type(score) in (int, float) and math.isfinite(score)
             and abs(score - expected) < 1e-8) or (not accepted and score is None),
            "task_audit_cannot_change_native_score")
    refs = value["evidence_refs"]
    actual_reads = {f'workspace:{result.arguments["snapshot"]}:{result.arguments["path"]}'
        for result in assessment.agent_trace.results
        if result.name == "workspace_read" and result.status == "completed"}
    require(isinstance(refs, list) and refs and all(isinstance(ref, str) for ref in refs)
        and set(refs) <= actual_reads & set(prepared.source_workspace_refs)
        and any(ref.startswith("workspace:after:") for ref in refs), "task_audit_workspace_not_read")
    readable = [item for item in document["submitted_files"] if Path(item["path"]).suffix in {".json", ".csv", ".txt"}]
    if accepted and readable:
        submitted = {"workspace:after:outputs/agents_outputs/" + item["path"] for item in readable}
        require(bool(set(refs) & submitted), "task_audit_submission_not_read")
    require(isinstance(value["explanation"], str) and 0 < len(value["explanation"]) <= 2000,
            "task_audit_explanation")
    return value


def verify_task_audit(root):
    """Reopen the real native grade/receipt/reads before accepting audit claims."""
    from training.benchmark_feedback.automed_codex import (
        _bound_feedback_rubric, read_feedback_rollout, verify_feedback)
    from eva_agent.pipeline.codec import _judgment
    from eva_agent.training.slime_agent_judge import prepare_workspace_rollout

    root = Path(root)
    source = read(root / "source.json")
    row, document = native_result(source["native_result"], track=source["track"])
    require(all(Path(source[name]["path"]).resolve().parent == Path(source["source_feedback_root"]).resolve()
        for name in ("rollout", "preflight")), "task_audit_source_root_differs")
    for name in ("rollout", "preflight"):
        ref = source[name]
        require(commitment(ref["path"]) == ref, "task_audit_actor_source_changed")
        require((root / (name + ".json")).read_bytes() == Path(ref["path"]).read_bytes(),
                "task_audit_actor_copy_changed")
    feedback = verify_feedback(root)
    preflight = read(root / "preflight.json")
    rollout = read_feedback_rollout(root / "rollout.json", preflight)
    bind_actor(row, document, rollout)
    require(feedback["case_id"] == source["track"] and source["actor_rerolls"] == 0,
            "task_audit_track_binding")
    rubric = _bound_feedback_rubric(preflight, rollout, None)
    prepared = prepare_workspace_rollout(rollout, rubric=rubric, source_path=root / "rollout.json",
        judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra",
        material_view=preflight.get("judge_material_view", "historical-v1"))
    assessment = _judgment(read(root / "judge" / preflight["sample_id"] / "assessment.json"))
    review = validate_review(assessment, prepared, row, document)
    require(read(root / "task-audit-instruction.json") == instruction(row, document),
            "task_audit_instruction_changed")
    return {"schema": SCHEMA, "track": source["track"], "status": "verified",
        "agent_audit_verified": True, "native_result_consistent": review["native_result_consistent"],
        "task_score_source": "native_evaluator", "task_score_0_100": review["task_score_0_100"],
        "native_result": source["native_result"], "source_feedback_root": source["source_feedback_root"],
        "evidence_refs": review["evidence_refs"], "explanation": review["explanation"],
        "judge_receipt_blake3": feedback["judge_codex_receipt_blake3"],
        "actor_rerolls": 0, "additional_task_audit_attempts": 1, "process_scores_replaced": False}


def run_task_audit(feedback_root, result_path, output_root, *, native_turn_timeout_seconds=600):
    from training.benchmark_feedback.automed_codex import (
        _bound_feedback_rubric, read_feedback_rollout, verify_feedback, write_private)
    from eva_agent.training import slime_agent_judge as core
    from eva_agent.codex_runtime import verify_codex_turn_receipt
    core.validate_native_judge_timeout(native_turn_timeout_seconds)
    root, feedback_root = Path(output_root), Path(feedback_root)
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    judge = prepared = None
    track = read(result_path)["track"]
    source = {"schema": SCHEMA, "track": track, "actor_rerolls": 0,
        "source_feedback_root": str(feedback_root.resolve()), "native_result": commitment(result_path),
        **{name: commitment(feedback_root / (name + ".json")) for name in ("rollout", "preflight")}}
    write(root / "source.json", source, exclusive=True)
    try:
        row, document = native_result(source["native_result"], track=track)
        original = verify_feedback(feedback_root)
        require(original["valid"] is True and original["case_id"] == track, "task_audit_source_feedback")
        for name in ("rollout", "preflight"):
            with os.fdopen(os.open(root / (name + ".json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
                stream.write(Path(source[name]["path"]).read_bytes())
        preflight = read(root / "preflight.json")
        rollout = read_feedback_rollout(root / "rollout.json", preflight)
        bind_actor(row, document, rollout)
        rubric = _bound_feedback_rubric(preflight, rollout, None)
        view = preflight.get("judge_material_view", "historical-v1")
        prepared = core.prepare_workspace_rollout(rollout, rubric=rubric, source_path=root / "rollout.json",
            judge_model_id="gpt-6-astra", judge_route_id="gpt_6_astra", material_view=view)
        write_private(root / "judge-attempt.json", {"attempt_id": str(uuid4()), "backend": "native_astra",
            "semantic_attempt_count": 1, "retry_count": 0, "automatic_fallback": False,
            "native_turn_timeout_seconds": native_turn_timeout_seconds,
            "judge_implementation_blake3": commitment(core.__file__)["blake3"],
            **({"material_view": view, "actual_imported_judge_source": str(Path(core.__file__).resolve())}
               if view != "historical-v1" else {})})
        inner = root / "judge" / preflight["sample_id"]
        inner.mkdir(parents=True, mode=0o700, exist_ok=False)
        write_private(inner / "rollout.json", rollout)
        annotation = instruction(row, document)
        write_private(root / "task-audit-instruction.json", annotation)
        provenance = core._backend_provenance(prepared, "native_astra",
            native_turn_timeout_seconds=native_turn_timeout_seconds, maximum_concurrent_judges=4)
        write_private(inner / "judge-backend.json", provenance)
        with core._open_native_astra_judge(prepared, inner,
                native_turn_timeout_seconds=native_turn_timeout_seconds) as judge:
            grade = core.grade_prepared_rollout(prepared, judge=TaskScoreJudge(judge, annotation), output_root=inner)
            receipt = judge.receipt_for(prepared.task.judge_task_id)
            write_private(inner / "judge-codex-receipt.json", canonical_value(receipt))
        grade.update(judge_backend="native_astra", judge_provenance=provenance)
        write_private(inner / "grade.json", grade)
        result = verify_task_audit(root)
    except Exception as exc:
        receipt = getattr(exc, "receipt", None)
        if receipt is None and judge is not None and prepared is not None:
            try:
                receipt = judge.receipt_for(prepared.task.judge_task_id)
            except Exception:
                pass
        if receipt is not None:
            write(root / "failure-receipt.json", canonical_value(receipt), exclusive=True)
            try:
                verify_codex_turn_receipt(receipt)
                receipt_verified = True
            except Exception:
                receipt_verified = False
        else:
            receipt_verified = None
        failure = {"error_type": type(exc).__name__, "automatic_retry": False,
            "actor_rerolls": 0, "task_score_0_100": None, "failure_receipt_verified": receipt_verified}
        write(root / "failure.json", failure, exclusive=True)
        result = {"schema": SCHEMA, "track": track, "status": "unavailable",
            "agent_audit_verified": False, "task_score_0_100": None,
            "task_score_source": "native_evaluator", "native_result": source["native_result"],
            "source_feedback_root": source["source_feedback_root"], "failure": commitment(root / "failure.json"),
            "actor_rerolls": 0, "additional_task_audit_attempts": 1, "process_scores_replaced": False}
    write(root / "result.json", result, exclusive=True)
    return result


def audit_native_scores(feedback_roots, native_summary_path, output_root, *, native_turn_timeout_seconds=600):
    """At most four independent task reviews; missing native results stay N/A."""
    summary = read(native_summary_path)
    choices = {}
    for path in feedback_roots:
        preflight = read(Path(path) / "preflight.json")
        key = preflight["case_id"]
        if key not in choices or preflight["stage"] > choices[key][0]:
            choices[key] = (preflight["stage"], Path(path))
    root = Path(output_root)
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    def one(row):
        track = row["track"]
        target = root / track / str(uuid4())
        path = Path(native_summary_path).parent / (track + "-result.json")
        require(read(path) == row, "task_audit_native_summary_row_differs")
        if row["status"] != "scored" or track not in choices:
            target.mkdir(parents=True, mode=0o700, exist_ok=False)
            result = {"schema": SCHEMA, "track": track, "status": "unavailable",
                "agent_audit_verified": False, "task_score_0_100": None,
                "task_score_source": "native_evaluator", "native_result": commitment(path),
                "reason": "native_score_or_process_evidence_unavailable", "actor_rerolls": 0,
                "additional_task_audit_attempts": 0, "process_scores_replaced": False}
            write(target / "result.json", result, exclusive=True)
        else:
            run_task_audit(choices[track][1], path, target, native_turn_timeout_seconds=native_turn_timeout_seconds)
        return {"track": track, "result": commitment(target / "result.json")}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, summary["tracks"]))
