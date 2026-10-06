"""Opt-in reporting over retained trajectories; never a controller/training input."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from eva_agent.codex_runtime import codex_turn_receipt_from_document
from training.automedbench_lite.adapter import read_document, write_once
from training.automedbench_lite.track_feedback import PHASES
from training.automedbench_lite.track_adapter import BY_TRACK
from .evidence import commitment, require


def _failure_category(error):
    # Only source-defined public codes, never arbitrary exception/provider text.
    known = {"document_topology_or_size_invalid", "document_digest_invalid", "track_checkpoint_binding",
        "track_no_verified_grade_missing_is_not_zero", "track_input_manifest_changed",
        "track_skill_catalog_changed", "actual_selected_sdk_catalog_differs", "actual_selected_sdk_body_differs",
        "extra_sdk_memory_binding_differs", "actor_skill_input_formatter_source_differs",
        "retained_report_phase_source_changed", "retained_report_single_stage_result_required",
        "retained_report_feedback_unverified"}
    return str(error) if str(error) in known else "retained_stage_validation_or_judge_unavailable"


def _read(path, maximum=16 * 1024**2):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= maximum,
            "retained_report_document_unavailable")
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), "retained_report_document_shape")
    return value


def _argument(argv, name):
    indices = [i for i, value in enumerate(argv) if value == name]
    require(len(indices) == 1 and indices[0] + 1 < len(argv), "retained_report_source_argument")
    return Path(argv[indices[0] + 1]).resolve(strict=True)


def bind_source(source_attempt):
    """Only the actual evaluation command's context, settings and actor binding."""
    source = Path(source_attempt).resolve(strict=True)
    context, request = _read(source / "context.json"), _read(source / "request.json")
    actor = _read(source / "evaluation-actor-binding.json")
    require(context["phase"] == request["phase"] == "evaluation"
            and Path(context["attempt_root"]).resolve() == source
            and request["attempt_id"] == source.name, "retained_report_source_attempt_differs")
    require(actor["schema"] == "eva.rsi-evaluation-actor-binding.v1"
            and actor["codex_version"] == "codex-cli 0.153.4"
            and actor["model_path"] == context["model_path"], "retained_report_actor_binding_differs")
    run = _argument(actor["equivalent_actor_cli"], "--run-root")
    require(run.is_relative_to(source / "benchmark"), "retained_report_run_outside_attempt")
    identity = actor["checkpoint_identity"]
    require(commitment(identity["path"]) == identity
            and _read(identity["path"])["exact_final_model_path"] == context["model_path"],
            "retained_report_checkpoint_differs")
    attempt = read_document(run / "track-rollouts/attempt.json")
    require(attempt["server_binding"]["identity_file_blake3"] == identity["blake3"]
            and set(attempt["tracks"]) == set(BY_TRACK) and attempt["one_attempt_per_track"] is True,
            "retained_report_track_attempt_differs")
    settings_path = _argument(request["argv"], "--settings")
    settings = _read(settings_path)
    require(settings["evaluation_mode"] == "full_single_pass"
            and settings["baseline_gate_mode"] == "terminal-attempts-v1"
            and settings.get("judge_material_view") == "policy-visible-audit-v2"
            and settings.get("native_judge_timeout_seconds") == 600,
            "retained_report_judge_profile_differs")
    from training.automedbench_lite.judge_attempt_isolation import replacement_limit
    replacements = replacement_limit(settings.get("postround_judge_verdict_replacements", 0))
    require(type(settings.get("allow_context_budget_terminal", False)) is bool,
            "retained_report_context_option_invalid")
    references = [commitment(source / name) for name in
                  ("context.json", "request.json", "evaluation-actor-binding.json", "worker.json")]
    references += [identity, commitment(settings_path), commitment(run / "track-rollouts/attempt.json"),
                   commitment(run / "track-run-manifest.json")]
    return {"source_attempt": str(source), "run_root": str(run), "context": context, "actor": actor,
            "checkpoint_identity": identity, "settings": settings, "references": references,
            "judge_replacements": replacements}


def source_exited(bound):
    source = Path(bound["source_attempt"])
    if not (source / "exit.json").is_file():
        return False
    value, owner = _read(source / "exit.json"), _read(source / "worker.json")
    require(value["attempt_id"] == source.name
            and (type(value.get("exit_code")) is int or isinstance(value.get("error_type"), str))
            and owner["request"] == commitment(source / "request.json")
            and type(owner["pid"]) is int and owner["pid"] > 0
            and isinstance(owner["start_identity"], str) and owner["start_identity"].isdigit(),
            "retained_report_exit_binding_differs")
    process = Path(f'/proc/{owner["pid"]}')
    if process.exists():
        # An unreadable existing process is not evidence of absence.
        try:
            birth = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            if process.exists():
                raise
            birth = None
        if birth == owner["start_identity"]:
            return False
    session = _read(source / "serving/session-exit.json")
    require(session["cleanup"].get("verified_members_remaining") == 0,
            "retained_report_serving_cleanup_unproved")
    return True


def _live_gate_blocked(bound):
    """A sealed zero-turn outcome prevents the source's own later Judge pool.

    Do not overlap an unmodified source that could independently grade the same
    stages. Its strict terminal-attempts gate is preserved, not bypassed here.
    """
    run = Path(bound["run_root"])
    for track in BY_TRACK:
        audit = run / "track-rollouts" / track
        path = audit / "track-budget-outcome.json"
        if not path.exists():
            continue
        try:
            row, budget = read_document(path), read_document(audit / "track-budget.json")
        except (OSError, ValueError):
            continue  # Publication may still be in progress.
        if (row.get("schema") == "eva.automedbench-track-budget-outcome.v1"
                and row.get("budget_document_blake3") == budget["document_blake3"]
                and type(row.get("actual_turn_count")) is int and row["actual_turn_count"] == 0
                and row.get("app_server_pids") == []
                and not any((audit / "turns" / phase / "receipt.json").exists() for phase in PHASES.values())):
            return commitment(path)
    return None


def stage_readiness(bound, track, stage, *, exited):
    """No provider call on exists-only or partly published JSON evidence."""
    audit = Path(bound["run_root"]) / "track-rollouts" / track
    prefix = list(PHASES)[:list(PHASES).index(stage) + 1]
    references = []
    try:
        outcome_path = audit / "track-budget-outcome.json"
        if outcome_path.exists():
            outcome = read_document(outcome_path)
            budget = read_document(audit / "track-budget.json")
            count = outcome.get("actual_turn_count")
            if (outcome.get("schema") == "eva.automedbench-track-budget-outcome.v1"
                    and outcome.get("budget_document_blake3") == budget["document_blake3"]
                    and type(count) is int and 0 <= count < len(prefix)
                    and not (audit / "turns" / PHASES[stage] / "receipt.json").exists()):
                return {"status": "unavailable", "reason": "source_track_terminal_without_requested_phase",
                        "actual_turn_count": count, "track_outcome": commitment(outcome_path), "score": None}
        for name in prefix:
            target = audit / "turns" / PHASES[name]
            for filename in ("request.json", "before/manifest.json", "after/manifest.json"):
                read_document(target / filename, maximum=16 * 1024**2)
                references.append(commitment(target / filename))
            receipt_path = target / "receipt.json"
            receipt = codex_turn_receipt_from_document(_read(receipt_path, 512 * 1024**2))
            references.append(commitment(receipt_path))
            if receipt.status == "interrupted":
                read_document(target / "policy-budget-terminal.json")
            if (target / "policy-budget-terminal.json").exists():
                read_document(target / "policy-budget-terminal.json")
                references.append(commitment(target / "policy-budget-terminal.json"))
        # The producer writes snapshots non-atomically. A later phase request
        # or terminal track outcome also establishes that capture_turn returned.
        position = list(PHASES).index(stage)
        marker = (audit / "turns" / list(PHASES.values())[position + 1] / "request.json"
                  if position < 4 else audit / "track-budget-outcome.json")
        terminal = audit / "track-budget-outcome.json"
        if not exited:
            read_document(terminal if terminal.exists() else marker, maximum=16 * 1024**2)
    except Exception as error:
        return {"status": "unavailable" if exited else "pending", "reason": "phase_prefix_not_sealed",
                "error_type": type(error).__name__, "score": None}
    return {"status": "ready", "source_references": references, "score": None}


def _stage_job(bound, track, stage, target, readiness, *, execute, registry):
    from training.automedbench_lite.track_feedback import evaluate_track_feedback, prepare_track_feedback
    from training.benchmark_feedback.automed_codex import verify_feedback
    options = dict(track=track, registry_path=registry, allow_policy_budget_terminal=True,
        material_view="policy-visible-audit-v2",
        allow_context_budget_terminal=bound["settings"].get("allow_context_budget_terminal", False),
        context_terminal_source_binding=bound["actor"])
    try:
        require(all(commitment(ref["path"]) == ref for ref in readiness["source_references"]),
                "retained_report_phase_source_changed")
        if execute:
            roots = evaluate_track_feedback(bound["run_root"], bound["checkpoint_identity"]["path"],
                target / "feedback", stages=(stage,), native_turn_timeout_seconds=600,
                postround_judge_verdict_replacements=bound["judge_replacements"], **options)
            require(len(roots) == 1, "retained_report_single_stage_result_required")
            require(Path(roots[0]).resolve().is_relative_to(target.resolve()),
                    "retained_report_feedback_outside_fresh_output")
            verified = verify_feedback(Path(roots[0]))
            require(verified["valid"] is True and verified["status"] == "scored"
                    and verified["case_id"] == track and verified["stage"] == stage,
                    "retained_report_feedback_unverified")
            result = {"status": "verified", "feedback_root": str(roots[0]),
                      "score": verified["score"]["reward_bps"],
                      "verification": commitment(Path(roots[0]) / "verification.json")}
        else:
            _, _, preflight = prepare_track_feedback(run_root=bound["run_root"],
                checkpoint_identity=bound["checkpoint_identity"]["path"], output_root=target / "prepared",
                stage=stage, **options)
            result = {"status": "prepared_eligible" if preflight["judge_eligible"] else "unavailable",
                      "preflight": commitment(target / "prepared/preflight.json"), "score": None}
        require(all(commitment(ref["path"]) == ref for ref in readiness["source_references"]),
                "retained_report_phase_source_changed")
    except Exception as error:
        result = {"status": "unavailable", "reason": "stage_preflight_or_judge_failed",
                  "error_type": type(error).__name__, "failure_category": _failure_category(error), "score": None}
    result.update(track=track, stage=stage, execution_requested=execute,
                  source_references=readiness["source_references"], observer_retry_count=0)
    write_once(target / "result.json", result)
    return result


def _progress(bound, output, completed, pending, *, observer_status, exited):
    active = set(pending.values())
    stages = []
    for track in BY_TRACK:
        for stage in PHASES:
            row = completed.get((track, stage), {})
            status = row.get("status", "running" if (track, stage) in active else "pending")
            if status in {"prepared_eligible", "ready"}:
                status = "pending"  # Provider-free preflight is never a numeric grade.
            stages.append({"track": track, "stage": stage, "status": status,
                "score_0_100": row["score"] / 100 if status == "verified" else None,
                "verification": row.get("verification") if status == "verified" else None})
    value = {"schema": "eva.retained-eval-report-progress.v1",
        "source_attempt_root": bound["source_attempt"], "benchmark_run_root": bound["run_root"],
        "actor_runtime_binding": commitment(Path(bound["source_attempt"]) / "evaluation-actor-binding.json"),
        "checkpoint_identity": bound["checkpoint_identity"], "stages": stages,
        "counts": {status: sum(row["status"] == status for row in stages)
                   for status in ("pending", "running", "verified", "unavailable")},
        "observer_status": observer_status, "source_exit_observed": exited,
        "diagnostic_only": True, "training_admission": False}
    temporary = output / ("progress." + str(uuid4()) + ".json")
    write_once(temporary, value)
    os.replace(temporary, output / "progress.json")


def _finish(bound, output, outcomes, *, execute, exited, stop_reason):
    from training.automedbench_lite.evaluation_report import build_report, render_markdown
    source, run = Path(bound["source_attempt"]), Path(bound["run_root"])
    roots = [row["feedback_root"] for row in outcomes if row["status"] == "verified"]
    index = {"schema": "eva.rsi-evaluation-index.v1", "evaluation_mode": "diagnostic_subset",
        "baseline_semantics": "existing diagnostic retained trajectories; report only, not training admission",
        "diagnostic_only": True, "full_seven_track_evaluation": False, "new_rollouts_per_track": 0,
        "benchmark_run_root": str(run), "checkpoint_identity": bound["checkpoint_identity"]["path"],
        "actor_runtime_binding": commitment(source / "evaluation-actor-binding.json"),
        "feedback_roots": roots, "tracks_requested": list(BY_TRACK), "stages_requested": list(PHASES),
        "matched_codex_version_requested": "0.153.4", "judge_material_view": "policy-visible-audit-v2",
        "source_references": bound["references"], "source_exit_observed": exited,
        "diagnostic_artifacts": [str(output / "coverage.json")], "new_actor_rollouts": 0}
    if exited:
        index["source_terminal_references"] = [commitment(source / name)
            for name in ("exit.json", "worker.json", "serving/session-exit.json")]
    if roots and bound["judge_replacements"]:
        from training.automedbench_lite.judge_attempt_isolation import BINDING_NAME
        index.update(postround_judge_verdict_replacements=bound["judge_replacements"],
            postround_judge_attempt_isolation=[commitment(Path(root) / BINDING_NAME) for root in roots])
    native = source / "native-task-scores/summary.json"
    native_failure = None
    try:
        _attach_task_evidence(bound, output, outcomes, index, native, execute=execute, exited=exited)
    except Exception as error:
        native_failure = {"reason": "native_score_or_task_audit_unavailable", "error_type": type(error).__name__}
    write_once(output / "coverage.json", {"schema": "eva.retained-eval-report-coverage.v1",
        "stages": outcomes, "stop_reason": stop_reason, "source_exit_observed": exited,
        "task_evidence_failure": native_failure,
        "diagnostic_only": True, "training_admission": False, "missing_is_not_zero": True})
    write_once(output / "evaluation-index.json", index)
    report = build_report(run, output / "evaluation-index.json")
    write_once(output / "seven-track-report.json", report)
    with (output / "seven-track-report.md").open("x") as stream:
        stream.write(render_markdown(report))
    return write_once(output / "result.json", {"schema": "eva.retained-eval-report-result.v1",
        "diagnostic_only": True, "training_admission": False, "execute": execute,
        "source_exit_observed": exited, "stop_reason": stop_reason,
        "logical_stage_dispatches": sum(row.get("execution_requested") is True for row in outcomes),
        "verified_stage_count": len(roots), "new_actor_rollouts": 0, "new_native_scores": 0,
        "task_audits": index.get("task_agent_audits", []), "task_evidence_failure": native_failure,
        "report": commitment(output / "seven-track-report.json"), "index": commitment(output / "evaluation-index.json")})


def _attach_task_evidence(bound, output, outcomes, index, native, *, execute, exited):
    source, run = Path(bound["source_attempt"]), Path(bound["run_root"])
    if exited and native.is_file():
        row = _read(native)
        require(row["schema"] == "eva.rsi-native-task-score-summary.v1"
                and Path(row["benchmark_run_root"]).resolve() == run
                and row["checkpoint_identity"] == bound["checkpoint_identity"],
                "retained_report_native_source_differs")
        index["native_task_scores"] = commitment(native)
        original_index = source / "evaluation-index.json"
        if original_index.is_file():
            previous = _read(original_index)
            require(Path(previous["benchmark_run_root"]).resolve() == run
                    and previous.get("native_task_scores") == index["native_task_scores"],
                    "retained_report_source_index_differs")
            if previous.get("task_agent_audits"):
                index["task_agent_audits"] = previous["task_agent_audits"]
        if execute and bound["settings"].get("task_agent_score_audit") is True and "task_agent_audits" not in index:
            from .task_score_audit import audit_native_scores
            s5 = [row["feedback_root"] for row in outcomes if row["status"] == "verified" and row["stage"] == "S5"]
            index["task_agent_audits"] = audit_native_scores(s5, native, output / "task-agent-audits",
                                                           native_turn_timeout_seconds=600)


def observe(source_attempt, output_root, *, execute=False, poll_seconds=10, max_wait_seconds=14400):
    """Default is one provider-free materialization pass; --execute watches once."""
    require(type(execute) is bool and type(poll_seconds) is int and 1 <= poll_seconds <= 60
            and type(max_wait_seconds) is int and 1 <= max_wait_seconds <= 21600,
            "retained_report_observer_limits")
    bound = bind_source(source_attempt)
    source, output = Path(bound["source_attempt"]), Path(output_root).resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output),
            "retained_report_output_must_be_separate")
    from eva_agent.training.slime_agent_judge import resolve_judge_concurrency
    require(Path(os.environ.get("EVA_HARNESS_ROOT", "")).resolve() == Path(bound["actor"]["selected_harness_root"]).resolve(),
            "retained_report_selected_harness_differs")
    concurrency = resolve_judge_concurrency()
    require(not execute or concurrency == 4, "retained_report_requires_four_judge_slots")
    registry = Path(__file__).resolve().parents[2] / "rubrics/source/domain-stage-tables.v2.json"
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    write_once(output / "observer.json", {"schema": "eva.retained-eval-report-observer.v1",
        "source_attempt": str(source), "source_references": bound["references"], "execute": execute,
        "judge_backend": "native_astra", "native_turn_timeout_seconds": 600,
        "judge_material_view": "policy-visible-audit-v2", "max_parallel_tasks": 4,
        "core_concurrency": concurrency, "registry": commitment(registry),
        "observer_source": commitment(Path(__file__)), "harness_root": os.environ["EVA_HARNESS_ROOT"],
        "postround_judge_verdict_replacements": bound["judge_replacements"],
        "poll_seconds": poll_seconds, "max_wait_seconds": max_wait_seconds,
        "source_mutations": False, "new_actor_rollouts": 0, "training_admission": False})
    started, claims, completed, pending = time.monotonic(), set(), {}, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            require(all(commitment(ref["path"]) == ref for ref in bound["references"]),
                    "retained_report_source_binding_changed")
            exited = source_exited(bound)
            expired = time.monotonic() - started >= max_wait_seconds
            gate_blocking = _live_gate_blocked(bound) if execute and not exited else None
            can_dispatch = not execute or exited or gate_blocking is not None
            for track in BY_TRACK:
                for stage in PHASES:
                    key = (track, stage)
                    if key in claims:
                        continue
                    readiness = stage_readiness(bound, track, stage, exited=exited)
                    if readiness["status"] != "ready":
                        if readiness["status"] == "unavailable":
                            claims.add(key)
                            completed[key] = {"track": track, "stage": stage, **readiness,
                                              "execution_requested": False}
                        continue
                    if not can_dispatch or expired:
                        continue
                    if execute and (source / "workspace-feedback" / track / stage).exists():
                        claims.add(key)
                        completed[key] = {"track": track, "stage": stage, "status": "unavailable", "score": None,
                            "reason": "source_stage_judge_already_present", "execution_requested": False}
                        continue
                    if len(pending) >= 4:
                        break
                    target = output / "stages" / track / stage
                    target.mkdir(parents=True, mode=0o700, exist_ok=False)
                    write_once(target / "claim.json", {"track": track, "stage": stage, **readiness,
                        "source_attempt": str(source), "execute": execute,
                        "source_exit_observed_at_claim": exited, "source_gate_blocking_outcome": gate_blocking})
                    claims.add(key)
                    pending[pool.submit(_stage_job, bound, track, stage, target, readiness,
                                        execute=execute, registry=registry)] = key
            for future in list(pending):
                if future.done():
                    key = pending.pop(future)
                    completed[key] = future.result()
                    print(json.dumps({"track": key[0], "stage": key[1], "status": completed[key]["status"],
                                      "score": completed[key]["score"]}), flush=True)
            _progress(bound, output, completed, pending, observer_status="running" if execute else "preflight",
                      exited=exited)
            # Preflight is a single source observation, but drain all ready jobs.
            if (not execute or exited or expired) and not pending:
                unclaimed_ready = any((t, s) not in claims and stage_readiness(bound, t, s, exited=exited)["status"] == "ready"
                                      for t in BY_TRACK for s in PHASES)
                if not unclaimed_ready or expired:
                    break
            time.sleep(min(poll_seconds, 1) if not execute else poll_seconds)
    outcomes = [completed.get((track, stage), {"track": track, "stage": stage,
        **stage_readiness(bound, track, stage, exited=exited), "execution_requested": False})
        for track in BY_TRACK for stage in PHASES]
    _progress(bound, output, {(row["track"], row["stage"]): row for row in outcomes}, {},
              observer_status="preflight_complete" if not execute else "source_exited" if exited else "watchdog",
              exited=exited)
    return _finish(bound, output, outcomes, execute=execute, exited=exited,
                   stop_reason="preflight_only" if not execute else "source_exited" if exited else "observer_watchdog")
