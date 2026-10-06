"""Read-only, evidence-linked process/task report; never score, judge or launch.

Only metadata leaves this module. Policy/Judge narrative, arguments, tool outputs
and token text are reopened by existing verifiers but never copied to the report.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .adapter import read_document
from .track_adapter import BY_TRACK
from training.eva_rsi.evidence import commitment, require

STAGES = ("S1", "S2", "S3", "S4", "S5")
PHASES = ("01-planning", "02-setup", "03-smoke", "04-full-subset", "05-review")


def _read(path, maximum=64 * 1024**2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum,
            "report_metadata_file_invalid")
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), "report_document_invalid")
    return value


def _bound(reference):
    require(isinstance(reference, dict) and set(reference) == {"path", "blake3"}
            and commitment(reference["path"]) == reference, "report_reference_differs")
    return _read(reference["path"])


def _numeric(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _feedback(root, family):
    if family:
        from training.eva_rsi.judge_identity import reopen_family_feedback
        return reopen_family_feedback(root, family, include_skill_source_binding=True)
    from training.benchmark_feedback.automed_codex import verify_feedback
    return verify_feedback(Path(root), include_skill_source_binding=True)


def _receipt(path):
    from eva_agent.codex_runtime import codex_turn_receipt_from_document, verify_codex_turn_receipt
    raw = _read(path, 512 * 1024**2)
    verify_codex_turn_receipt(codex_turn_receipt_from_document(raw))
    return raw


def runtime_usage(records):
    """Aggregate verified receipt metadata, once per actual native thread/turn.

    SDK usage.total is cumulative within a thread. Deltas are attributable only
    with the thread's beginning and every intervening stage retained. `last` is
    not a complete per-turn request total and is deliberately never summed.
    """
    unique, tools, errors, threads = {}, {}, [], {}
    for row in records:
        raw = row["receipt"]
        key = (raw["thread_id"], raw["turn_id"])
        if key in unique:
            require(unique[key]["receipt"]["receipt_blake3"] == raw["receipt_blake3"],
                    "report_duplicate_turn_differs")
            continue
        unique[key] = row
        threads.setdefault(raw["thread_id"], []).append(row)
        for tool in raw["tool_calls"]:
            call = (raw["thread_id"], tool["tool_call_id"])
            require(call not in tools or tools[call] == tool["receipt_blake3"], "report_duplicate_call_differs")
            tools[call] = tool["receipt_blake3"]
    details = []
    for rows in threads.values():
        rows.sort(key=lambda r: STAGES.index(r["stage"]))
        previous, previous_stage, attributable = None, None, not rows[0]["receipt"]["thread_resumed"]
        for row in rows:
            raw, stage = row["receipt"], row["stage"]
            index = STAGES.index(stage)
            if previous_stage is not None and index != previous_stage + 1:
                attributable = False
            total = raw.get("usage", {}).get("total", {})
            counts = [total.get(key) for key in ("inputTokens", "outputTokens")]
            valid = all(type(value) is int and value >= 0 for value in counts)
            if previous is not None and valid and any(a < b for a, b in zip(counts, previous)):
                valid = False
                errors.append({"stage": stage, "reason": "sdk_cumulative_usage_decreased"})
            if not valid:
                attributable = False
            delta = ([a-b for a,b in zip(counts, previous or (0, 0))] if valid and attributable else [None, None])
            completed = [event["payload"].get("turn", {}) for event in raw["events"]
                if event["method"] == "turn/completed" and isinstance(event.get("payload"), dict)]
            timing = completed[-1] if completed else {}
            if timing.get("id") != raw["turn_id"]:
                timing = {}
            duration = timing.get("durationMs")
            started, ended = timing.get("startedAt"), timing.get("completedAt")
            times_valid = (_numeric(started, 0, 10**12) and _numeric(ended, started, 10**12))
            details.append({"stage": stage, "thread_id": raw["thread_id"], "turn_id": raw["turn_id"],
                "status": raw["status"], "receipt": row["source"], "receipt_blake3": raw["receipt_blake3"],
                "tool_calls": len(raw["tool_calls"]), "input_tokens": delta[0], "output_tokens": delta[1],
                "duration_seconds": duration / 1000 if _numeric(duration, 0, 10**12) else None,
                "started_at_unix_seconds": started if times_valid else None,
                "completed_at_unix_seconds": ended if times_valid else None,
                "sdk_version": raw["sdk_version"], "codex_server_version": raw["server_version"]})
            previous = counts if valid else None
            previous_stage = index
    def complete_sum(key):
        return sum(row[key] for row in details) if details and all(row[key] is not None for row in details) else None
    starts = [r["started_at_unix_seconds"] for r in details]
    ends = [r["completed_at_unix_seconds"] for r in details]
    return {"turns": len(details), "tool_calls": len(tools),
        "input_tokens": complete_sum("input_tokens"), "output_tokens": complete_sum("output_tokens"),
        "sum_turn_seconds": complete_sum("duration_seconds"),
        "wall_span_seconds": max(ends)-min(starts) if details and None not in starts + ends else None,
        "token_coverage": sum(r["input_tokens"] is not None for r in details),
        "timing_coverage": sum(r["duration_seconds"] is not None for r in details),
        "receipt_details": details, "errors": errors}


def _runtime(run, track):
    records, missing, invalid = [], [], []
    for stage, phase in zip(STAGES, PHASES):
        path = run / "track-rollouts" / track / "turns" / phase / "receipt.json"
        if not path.exists():
            missing.append(stage)
            continue
        try:
            records.append({"stage": stage, "receipt": _receipt(path), "source": commitment(path)})
        except Exception as error:
            invalid.append({"stage": stage, "error_type": type(error).__name__})
    result = runtime_usage(records)
    result.update(missing_receipt_stages=missing, invalid_receipts=invalid,
                  scope="actor_only; excludes Judge/scorer/serving startup")
    if invalid:
        # Valid partial metadata is still retained, never labelled a full total.
        result["input_tokens"] = result["output_tokens"] = result["sum_turn_seconds"] = None
    return result


def _track_elapsed(run, track, actor, runtime):
    root = run / "track-rollouts" / track
    outcome_path, budget_path = root / "track-budget-outcome.json", root / "track-budget.json"
    result = {"track_elapsed_seconds": None, "track_elapsed_status": "not_recorded",
        "reported_time_seconds": runtime["sum_turn_seconds"], "reported_time_source": "sdk_turn_duration_sum"}
    if not outcome_path.exists() and not (actor and actor.get("track_budget")):
        return result
    try:
        outcome, budget = read_document(outcome_path), read_document(budget_path)
        require(actor is not None and actor.get("track_budget") == outcome
            and outcome.get("schema") == "eva.automedbench-track-budget-outcome.v1"
            and budget.get("schema") == "eva.automedbench-track-budget.v1"
            and outcome.get("budget_document_blake3") == budget["document_blake3"]
            and outcome.get("actual_turn_count") == actor.get("actual_turn_count")
            and outcome.get("app_server_pids") == actor.get("app_server_pids"),
            "report_track_budget_source_differs")
        require(outcome.get("policy") == budget.get("policy") == "admitted-track-wallclock-3600-v1"
            and outcome.get("timeout_seconds") == budget.get("timeout_seconds") == 3600
            and budget.get("queue_before_admission_charged") is False
            and budget.get("job_waits_and_runtime_restart_charged") is True,
            "report_track_budget_policy_differs")
        elapsed = outcome.get("elapsed_seconds")
        require(_numeric(elapsed, 0, 10**12)
            and type(outcome.get("deadline_exhausted")) is bool
            and outcome["deadline_exhausted"] == (elapsed >= 3600)
            and math.isclose(outcome.get("remaining_seconds", -1), max(0, 3600-elapsed), abs_tol=1e-6)
            and math.isclose(outcome.get("cleanup_overrun_seconds", -1), max(0, elapsed-3600), abs_tol=1e-6),
            "report_track_budget_time_invalid")
        result.update(track_elapsed_seconds=elapsed, track_elapsed_status="verified",
            reported_time_seconds=elapsed, reported_time_source="admitted_track_elapsed",
            track_budget=commitment(budget_path), track_budget_outcome=commitment(outcome_path),
            timeout_seconds=3600, deadline_exhausted=outcome["deadline_exhausted"],
            cleanup_overrun_seconds=outcome["cleanup_overrun_seconds"],
            elapsed_scope="admission through turns/job waits/runtime restart/cleanup; excludes pre-admission queue")
    except Exception as error:
        result.update(track_elapsed_status="verification_failed", reported_time_seconds=None,
            reported_time_source="unavailable", track_elapsed_error_type=type(error).__name__)
    return result


def _native(index, run, checkpoint):
    result = {track: {"T": None, "status": "unavailable", "reason": "native_score_not_recorded",
        "task_score_source": "native_evaluator", "task_judge_audit_status": "not_performed",
        "agent_audit_verified": False} for track in BY_TRACK}
    if not index.get("native_task_scores"):
        return result
    summary = _bound(index["native_task_scores"])
    require(summary.get("schema") == "eva.rsi-native-task-score-summary.v1"
        and Path(summary["benchmark_run_root"]).resolve() == run
        and summary["checkpoint_identity"] == checkpoint, "report_native_summary_source_differs")
    seen = set()
    for row in summary["tracks"]:
        track = row["track"]
        require(track in BY_TRACK and track not in seen, "report_duplicate_or_unknown_native_track")
        seen.add(track)
        if row["status"] != "scored":
            result[track]["reason"] = "native_result_unavailable"
            result[track]["result_reference"] = index["native_task_scores"]
            continue
        try:
            _bound(row["native_score"])
            path = Path(row["native_score"]["path"])
            score, attempt = read_document(path), read_document(path.parent / "attempt.json")
            actor = read_document(run / "track-rollouts" / track / "rollout.json")
            native = score["native_result"]
            require(score.get("schema") == "eva.automedbench-native-whole-track-score.v1"
                and score["track"] == track and native.get("track") == track
                and score["actor_document_blake3"] == actor["document_blake3"]
                and attempt["actor_document_blake3"] == actor["document_blake3"]
                and Path(attempt["source_run_root"]).resolve() == run
                and actor["completed_requested_turns"] is True and not actor.get("errors")
                and row["native_result"] == native, "report_native_actor_binding_differs")
            value = native.get("task_score_0_1")
            require(native.get("schema") == "eva.automedbench-native-track-result.v1"
                and native.get("status") == "scored" and _numeric(value, 0, 1)
                and row["task_score_0_1"] == value, "report_native_scale_unknown_or_invalid")
            require(native.get("full_public_subset") is True
                and native.get("selected_case_count") == BY_TRACK[track].count,
                "report_native_full_subset_unproved")
            result[track].update(T=value*100, status="scored", reason=None,
                raw_value=value, raw_metric="task_score_0_1", raw_scale=[0, 1],
                benchmark_metric=BY_TRACK[track].metric, private_raw_metric_exported=False,
                result_reference=row["native_score"], native_facts=native.get("native_facts", {}))
        except Exception as error:
            result[track].update(reason="native_evidence_verification_failed", error_type=type(error).__name__)
    return result


def _versions(index, run):
    out = {"codex_version_requested": index.get("matched_codex_version_requested"), "bindings": []}
    if index.get("actor_runtime_binding"):
        value = _bound(index["actor_runtime_binding"])
        require(value.get("checkpoint_identity") == commitment(index["checkpoint_identity"]),
                "report_actor_runtime_checkpoint_differs")
        out.update({key: value.get(key) for key in ("codex_version", "codex_binary_blake3",
            "profile_requested", "selected_harness_root", "actual_imported_sources")})
        out["bindings"].append(index["actor_runtime_binding"])
    elif (run / "baseline-launch.json").is_file():
        path = run / "baseline-launch.json"
        value = _read(path)
        out.update({key: value.get(key) for key in ("codex_version", "codex_binary_blake3",
            "runtime_source_blake3", "selected_memory_sources")})
        out["bindings"].append(commitment(path))
    for filename, fields in (("backend.json", ("adapter_projection_version", "context_length",
            "max_output_tokens", "thinking_explicitly_enabled", "auto_compact_token_limit")),
            ("supra-profile.json", ("profile", "protocol", "mode", "context_tokens", "compact_at_tokens"))):
        path = run / "track-rollouts" / filename
        if path.is_file():
            value = read_document(path)
            out[filename] = {key: value.get(key) for key in fields}
            out["bindings"].append(commitment(path))
    return out


def _task_audit(root):
    # Additive producer is optional; no import (and no provider) without a ref.
    from training.eva_rsi.task_score_audit import verify_task_audit
    return verify_task_audit(root)


def _task_audits(index, tasks):
    seen = set()
    for row in index.get("task_agent_audits", []):
        track = row["track"]
        require(track in tasks and track not in seen, "report_duplicate_or_unknown_task_audit")
        seen.add(track)
        task = tasks[track]
        task["task_audit_reference"] = row["result"]
        try:
            value = _bound(row["result"])
            native_row = _bound(value["native_result"])
            require(value.get("schema") == "eva.automedbench-task-score-audit.v1"
                and value["track"] == track and native_row["track"] == track,
                "report_task_audit_source_differs")
            if task["T"] is not None:
                require(native_row.get("native_score") == task["result_reference"],
                        "report_task_audit_native_source_differs")
            task["task_judge_audit_status"] = "unavailable"
            if value["status"] == "unavailable":
                require(value["agent_audit_verified"] is False, "report_unavailable_audit_claims_verification")
                continue
            require(value["status"] == "verified"
                and _task_audit(Path(row["result"]["path"]).parent) == value
                and value["agent_audit_verified"] is True
                and task["T"] is not None and native_row["native_score"] == task["result_reference"]
                and value["task_score_source"] == "native_evaluator",
                "report_task_audit_verification_differs")
            consistent = value["native_result_consistent"]
            require(type(consistent) is bool and ((consistent and value["task_score_0_100"] == task["T"])
                or (not consistent and value["task_score_0_100"] is None)), "report_task_audit_cannot_change_score")
            task.update(task_judge_audit_status="verified_consistent" if consistent else "verified_inconsistent",
                agent_audit_verified=True, native_result_consistent=consistent,
                task_audit_evidence_refs=value["evidence_refs"],
                task_audit_judge_receipt_blake3=value["judge_receipt_blake3"])
        except Exception as error:
            task.update(task_judge_audit_status="verification_failed", agent_audit_verified=False,
                        task_audit_error_type=type(error).__name__)


def build_report(run_root, evaluation_index):
    run, index_path = Path(run_root).resolve(strict=True), Path(evaluation_index).resolve(strict=True)
    index = _read(index_path)
    require(index.get("schema") in {"eva.rsi-evaluation-index.v1", "eva.rsi-evaluation-index.v2"}
        and Path(index["benchmark_run_root"]).resolve() == run, "report_index_run_differs")
    checkpoint = commitment(index["checkpoint_identity"])
    attempt = read_document(run / "track-rollouts/attempt.json")
    family = None
    if index.get("judge_comparison_family"):
        from training.eva_rsi.judge_identity import verify_judge_comparison_family
        family = verify_judge_comparison_family(index["judge_comparison_family"])
    feedbacks, failures, conflicts = {}, [], set()
    for root in dict.fromkeys(str(Path(path).resolve()) for path in index["feedback_roots"]):
        try:
            feedback = _feedback(root, family)
            key = feedback["case_id"], feedback["stage"]
            require(key[0] in BY_TRACK and key[1] in STAGES and feedback["valid"] is True
                and feedback["status"] == "scored", "report_feedback_identity_invalid")
            require(feedback["checkpoint_identity_blake3"] == checkpoint["blake3"]
                and feedback["skill_source_binding"]["run_attempt_document_blake3"] == attempt["document_blake3"],
                "report_feedback_source_differs")
            value = feedback["score"]["reward_bps"]
            require(type(value) is int and 0 <= value <= 10000, "report_feedback_reward_invalid")
            if key in feedbacks:
                conflicts.add(key)
            feedbacks[key] = {"A": value/100, "raw_reward_bps": value, "status": "verified",
                "feedback_root": root, "rubric_digest": feedback["rubric_digest"],
                "judge_receipt_blake3": feedback["judge_codex_receipt_blake3"],
                "actual_workspace_reads": feedback["actual_workspace_reads"],
                "raw_round_identity": feedback["round_identity"]}
        except Exception as error:
            failures.append({"feedback_root": root, "status": "unavailable", "error_type": type(error).__name__})
    try:
        tasks = _native(index, run, checkpoint)
    except Exception as error:
        tasks = {track: {"T": None, "status": "unavailable", "reason": "native_summary_verification_failed",
            "error_type": type(error).__name__, "task_score_source": "native_evaluator",
            "task_judge_audit_status": "not_performed", "agent_audit_verified": False} for track in BY_TRACK}
    _task_audits(index, tasks)
    rows = []
    for track in BY_TRACK:
        stages = {stage: {**(feedbacks.get((track, stage), {}) if (track, stage) not in conflicts else {}),
            "T": None, "task_status": "stage_task_metric_not_recorded"} for stage in STAGES}
        for stage in STAGES:
            stages[stage].setdefault("A", None)
            stages[stage].setdefault("status", "conflicting_feedback" if (track, stage) in conflicts else "unavailable")
        values = [stage["A"] for stage in stages.values() if stage["A"] is not None]
        A, T = sum(values)/len(values) if values else None, tasks[track]["T"]
        available = A is not None and T is not None
        actor_path = run / "track-rollouts" / track / "rollout.json"
        actor_completed, actor = None, None
        if actor_path.is_file():
            actor = read_document(actor_path)
            actor_completed = actor.get("completed_requested_turns") is True and not actor.get("errors")
        runtime = _runtime(run, track)
        runtime.update(_track_elapsed(run, track, actor, runtime))
        rows.append({"track": track, "A": A, "A_stage_coverage": len(values), "A_stages_expected": 5,
            "A_provisional": len(values) < 5, "all_five_A_stages_verified": len(values) == 5,
            "actor_completed_requested_turns": actor_completed,
            "T": T, "A_plus_T": A+T if available else None,
            "mean_A_T": (A+T)/2 if available else None,
            "overall_provisional": len(values) < 5 if available else None,
            "stages": stages, "task": tasks[track], "runtime": runtime})
    return {"schema": "eva.automedbench-seven-track-report.v1", "benchmark_run_root": str(run),
        "evaluation_index": commitment(index_path), "checkpoint_identity": checkpoint,
        "harness": _versions(index, run), "tracks": rows,
        "verified_process_stage_count": sum(r["A_stage_coverage"] for r in rows),
        "unavailable_process_stage_count": 35-sum(r["A_stage_coverage"] for r in rows),
        "numeric_task_count": sum(r["T"] is not None for r in rows), "feedback_failures": failures,
        "definitions": {
            "A": "Mean of independently verified observed stage rewards / 100; 0..100; missing stages excluded, coverage explicit.",
            "T": "Native evaluator task_score_0_1 times 100; retains raw value/scale/metric. Not a Judge-issued score.",
            "combined": "A+T and (A+T)/2 require both numeric; partial A makes both provisional.",
            "tokens": "Actor-only SDK cumulative per-thread totals differenced once per unique turn; no last-request or prefix double counting; not training tokens.",
            "time": "Prefer source-bound admitted track elapsed including job waits/restart/cleanup, excluding pre-admission queue. Historical fallback is explicitly SDK durationMs sum, not wall time. SDK sum/span remain separate. Excludes Judge/scorer/serving.",
            "turns": "Unique actual thread_id/turn_id receipts, including failed/interrupted; tool_calls do not mean successful executions.",
            "task_audit": "Separate from process assessment; absent explicit verified task-audit evidence means not performed.",
        }, "new_rollouts": 0, "new_judgments": 0, "new_native_scores": 0,
        "missing_is_not_zero": True, "clinical_generalization_claimed": False}


def render_markdown(report):
    def fmt(value):
        return "N/A" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)
    lines = ["Seven-track report (A: process rubric; T: native task evaluator)", "",
        "| Track | A | Coverage | Workflow complete | T | A+T | Mean | Turns | Time seconds / source | SDK seconds | Input / output tokens |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|"]
    for row in report["tracks"]:
        runtime = row["runtime"]
        lines.append("| " + " | ".join((row["track"], fmt(row["A"]),
            f'{row["A_stage_coverage"]}/5' + (" provisional" if row["A_provisional"] else ""),
            {True: "yes", False: "no", None: "unknown"}[row["actor_completed_requested_turns"]],
            fmt(row["T"]), fmt(row["A_plus_T"]), fmt(row["mean_A_T"]), str(runtime["turns"]),
            fmt(runtime["reported_time_seconds"]) + " / " + runtime["reported_time_source"],
            fmt(runtime["sum_turn_seconds"]), f'{fmt(runtime["input_tokens"])} / {fmt(runtime["output_tokens"])}')) + " |")
    lines += ["", "Time prefers verified admitted-track elapsed (waits/restart/cleanup included); historical SDK fallback is explicitly labelled and is not wall time. Tokens are cumulative-SDK deltas; missing evidence is N/A.",
        "Partial A makes combined scores provisional. Native T is not a workspace-Judge verdict.", "",
        "| Track / stage A | S1 | S2 | S3 | S4 | S5 | Task audit |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in report["tracks"]:
        lines.append("| " + " | ".join([row["track"], *[fmt(row["stages"][s]["A"]) for s in STAGES],
            row["task"]["task_judge_audit_status"]]) + " |")
    lines += ["", "Stage T: not measured in the retained producer contract; full-track T is not copied into S1–S5.",
        f'Verified process stages: {report["verified_process_stage_count"]}/35; numeric native tasks: {report["numeric_task_count"]}/7.',
        "", "Codex requested: " + str(report["harness"].get("codex_version_requested") or "unknown") +
        "; observed build(s): " + ", ".join(sorted({d["codex_server_version"] for row in report["tracks"]
            for d in row["runtime"]["receipt_details"]})) + ".",
        "Actor profile: " + str(report["harness"].get("profile_requested")
            or report["harness"].get("supra-profile.json", {}).get("profile") or "unknown") +
        "; adapter projection: " + str(report["harness"].get("backend.json", {}).get("adapter_projection_version") or "unknown") + ".",
        "Harness/profile source bindings (exact hashes retained in JSON):"]
    lines += ["- " + row["path"] for row in report["harness"]["bindings"]]
    lines += ["", "Evidence index: " + report["evaluation_index"]["path"]]
    return "\n".join(lines) + "\n"
