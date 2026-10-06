"""Explicit C/D supplementation, with fresh actors and immutable source evidence.

This produces a two-track component, never a synthetic seven-track run or a
controller admission. Import and default preflight do not launch serving,
actors, native scorers, or Judges. Execution is one-shot in a fresh operator
attempt, only after the original worker and owned serving session have exited.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from training.automedbench_lite.adapter import read_document
from training.automedbench_lite.track_adapter import BY_TRACK
from training.automedbench_lite.track_feedback import PHASES
from .controller import write
from .evidence import commitment, read, require
from . import production_eval as production
from .retained_eval_report import bind_source, source_exited, stage_readiness, _stage_job

TRACKS = ("classification", "detection")


def _zero_turn_sources(bound):
    references = []
    for track in TRACKS:
        audit = Path(bound["run_root"]) / "track-rollouts" / track
        outcome = read_document(audit / "track-budget-outcome.json")
        budget = read_document(audit / "track-budget.json")
        cleanup = read_document(audit / "track-deadline-cleanup.json")
        require(outcome.get("schema") == "eva.automedbench-track-budget-outcome.v1"
            and outcome.get("budget_document_blake3") == budget["document_blake3"]
            and type(outcome.get("actual_turn_count")) is int and outcome["actual_turn_count"] == 0
            and outcome.get("app_server_pids") == []
            and cleanup.get("workspace_quiescent") is True
            and cleanup.get("error_category") is None
            and not any((audit / "turns" / phase / "receipt.json").exists() for phase in PHASES.values()),
            "supplement_only_original_zero_turn_tracks")
        references += [commitment(audit / name) for name in
            ("track-budget.json", "track-budget-outcome.json", "track-deadline-cleanup.json")]
    return references


def _bind(context, settings):
    require(settings.get("allow_classification_detection_supplement") is True,
            "supplement_explicit_opt_in_required")
    bound = bind_source(settings["supplemental_source_attempt"])
    source, output = Path(bound["source_attempt"]), Path(context["attempt_root"]).resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output),
            "supplement_output_must_be_separate")
    require(context.get("phase") == "evaluation" and all(context.get(key) == bound["context"].get(key)
        for key in ("loop_id", "round", "model_path", "checkpoint_root", "architecture_model_path",
                    "skill_catalog_id", "skill_selection", "s_target")), "supplement_context_identity_differs")
    selected = Path(os.environ.get("EVA_HARNESS_ROOT", "")).resolve(strict=True)
    require(selected == Path(bound["actor"]["selected_harness_root"]).resolve(strict=True),
            "supplement_selected_actor_harness_differs")
    workers = settings.get("supplemental_workers", 2)
    require(type(workers) is int and 1 <= workers <= 2, "supplement_worker_limit")
    scope = {"mode": "full_single_pass", "tracks": list(TRACKS), "stages": tuple(PHASES),
        "registry": Path(__file__).resolve().parents[2] / "rubrics/source/domain-stage-tables.v2.json"}
    effective = {**settings, "evaluation_workers": workers}
    profile = production.actor_profile(effective, scope)
    require(profile["name"] == "supra-v1.3" and profile["args"].get("track_timeout") == 3600
        and profile["tool_output_token_limit"] == 2048, "supplement_requires_full_track_supra_profile")
    require(settings.get("judge_material_view") == "policy-visible-audit-v2"
        and settings.get("native_judge_timeout_seconds") == 600
        and settings.get("baseline_gate_mode") == "terminal-attempts-v1"
        and settings.get("skill_catalog_identity_mode", "verified-content-v1") == "verified-content-v1",
        "supplement_judge_or_skill_profile_differs")
    from training.automedbench_lite.judge_attempt_isolation import replacement_limit
    replacements = replacement_limit(settings.get("postround_judge_verdict_replacements", 0))
    require(replacements == bound["judge_replacements"]
        and settings.get("allow_context_budget_terminal", False)
            == bound["settings"].get("allow_context_budget_terminal", False),
        "supplement_terminal_or_judge_replacement_profile_differs")
    require(type(settings.get("task_agent_score_audit", False)) is bool,
            "supplement_task_audit_option_invalid")
    references = bound["references"] + _zero_turn_sources(bound)
    exited = source_exited(bound)
    if exited:
        references += [commitment(source / name) for name in ("exit.json", "serving/session-exit.json")]
    require(not any((output / name).exists() for name in
        ("supplemental-request.json", "benchmark", "serving", "workspace-feedback", "supplemental-index.json")),
        "supplement_output_already_used")
    return bound, output, scope, effective, profile, references, exited


def preflight_supplemental(context, settings):
    """Read-only plan. An active original run is reported blocked, never shared."""
    bound, output, scope, effective, profile, references, exited = _bind(context, settings)
    production.native_scoring_options(effective)
    sources = production.actor_runtime_sources(include_track_budget=True)
    binary = production.codex_binary_binding(effective["codex_bin"])
    return {"schema": "eva.rsi-supplemental-evaluation-preflight.v1", "ready": exited,
        "status": "READY" if exited else "WAITING_SOURCE",
        "reason": None if exited else "original_worker_or_serving_not_exited",
        "output_root": str(output), "original_source_attempt": bound["source_attempt"],
        "source_references": references, "tracks_requested": list(TRACKS),
        "prepared_public_input_tracks": list(BY_TRACK), "executed_tracks": list(TRACKS),
        "stages_requested": list(PHASES), "profile_requested": profile,
        "checkpoint_identity_original": bound["checkpoint_identity"],
        "actual_imported_sources": sources, **binary, "provider_calls": 0,
        "gpu_launches": 0, "full_seven_track_evaluation": False,
        "original_five_tracks_rerun": False, "controller_state_modified": False}


def _score_subset(run, identity, output, options):
    """Existing native scorer, only C/D; missing/incomplete/error remains N/A."""
    from training.automedbench_lite.track_scoring import score_track
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    rows = []
    for track in TRACKS:
        actor_path = run / "track-rollouts" / track / "rollout.json"
        row = {"track": track, "status": "unavailable", "task_score_0_1": None,
            "scorer_invoked": False, "reason": "actor_workflow_incomplete"}
        try:
            actor = read_document(actor_path)
            row["actor_rollout"] = commitment(actor_path)
            if actor.get("completed_requested_turns") is True and not actor.get("errors"):
                row["scorer_invoked"] = True
                score = score_track(run, track, options["evaluator_python"],
                    timeout=options["timeout"], score_output_root=output)
                path = output / track / "score.json"
                require(read_document(path) == score, "native_score_document_differs")
                row.update(status="scored", reason=None, native_result=score["native_result"],
                    task_score_0_1=score["native_result"]["task_score_0_1"], native_score=commitment(path))
        except Exception as error:
            row.update(status="unavailable", task_score_0_1=None,
                reason="native_scorer_or_evidence_unavailable", error_type=type(error).__name__)
        write(output / (track + "-result.json"), row, exclusive=True)
        rows.append(row)
    path = output / "summary.json"
    write(path, {"schema": "eva.rsi-native-task-score-summary.v1", "benchmark_run_root": str(run),
        "checkpoint_identity": commitment(identity), "tracks_requested": list(TRACKS), "tracks": rows,
        "scored_tracks": sum(row["status"] == "scored" for row in rows),
        "unavailable_tracks": sum(row["status"] != "scored" for row in rows),
        "evaluator_python": str(options["evaluator_python"]), "lpips_torch_home": None,
        "timeout_seconds_per_track": options["timeout"], "new_policy_rollouts": 0,
        "automatic_retry": False, "rubric_rewards_modified": False,
        "used_for_stage_target_selection": False, "missing_is_not_zero": True}, exclusive=True)
    return path


def _judge_subset(bound, output, registry):
    jobs = []
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for track in TRACKS:
            for stage in PHASES:
                ready = stage_readiness(bound, track, stage, exited=True)
                if ready["status"] != "ready":
                    results[track, stage] = {**ready, "track": track, "stage": stage}
                    continue
                target = output / track / stage
                target.mkdir(parents=True, mode=0o700, exist_ok=False)
                write(target / "claim.json", {"track": track, "stage": stage,
                    "source_references": ready["source_references"], "logical_dispatches": 1}, exclusive=True)
                jobs.append(((track, stage), pool.submit(_stage_job, bound, track, stage, target,
                    ready, execute=True, registry=registry)))
        for key, job in jobs:
            results[key] = job.result()
    rows = [results[track, stage] for track in TRACKS for stage in PHASES]
    path = output / "coverage.json"
    write(path, {"schema": "eva.rsi-supplemental-stage-coverage.v1", "tracks_requested": list(TRACKS),
        "stages_requested": list(PHASES), "stages": rows, "max_parallel_stages": 4,
        "missing_is_not_zero": True, "actor_rerolls": 0}, exclusive=True)
    return rows, path


def run_supplemental(context, settings, *, execute=False):
    plan = preflight_supplemental(context, settings)
    if not execute:
        return plan
    require(plan["ready"], "supplement_original_worker_or_serving_active")
    bound, output, scope, effective, profile, references, exited = _bind(context, settings)
    require(exited and all(commitment(ref["path"]) == ref for ref in plan["source_references"]),
            "supplement_original_source_changed")
    # Exclusive claim precedes any preparation or launch. No restart in this root.
    write(output / "supplemental-request.json", {**plan, "context": context,
        "settings": effective, "source_references": references, "execution_requested": True,
        "runner_source": commitment(__file__), "pid": os.getpid()}, exclusive=True)
    try:
        return _execute(context, effective, bound, output, scope, profile, plan)
    except BaseException as error:
        write(output / "supplemental-failure.json", {"schema": "eva.rsi-supplemental-failure.v1",
            "status": "unavailable", "error_type": type(error).__name__,
            "automatic_retry": False, "controller_state_modified": False}, exclusive=True)
        raise


def _execute(context, settings, original, output, scope, profile, plan):
    from training.automedbench_lite.track_adapter import TrackRelease, prepare_track_run
    from training.automedbench_lite.track_actor import run_tracks
    from .serving import serving_session
    cds, missing = TrackRelease(Path(settings["cds_receipt"])), TrackRelease(Path(settings["missing4_receipt"]))
    # Existing preparation is intentionally unchanged: all seven public input
    # workspaces are staged, but only C/D are selected by the actual actor.
    run = prepare_track_run({name: cds if name in {"classification", "detection", "segmentation"}
        else missing for name in BY_TRACK}, output / "benchmark")
    with serving_session(model_path=Path(context["model_path"]), checkpoint_root=Path(context["checkpoint_root"]),
        architecture_model_path=Path(context["architecture_model_path"]), output_root=output / "serving",
        public_image=Path(settings["public_image"]), port=settings.get("evaluation_port", 30911),
        python_executable=Path(settings["training_python"]), context_length=profile["args"]["context_length"]) as server:
        args, argv = production.actor_arguments(run, output, settings, scope, profile,
            identity_path=server.identity_path, canary_path=server.canary_path,
            selection=context.get("skill_selection"), catalog_id=context["skill_catalog_id"])
        actor = {"schema": "eva.rsi-evaluation-actor-binding.v1", "profile_requested": profile,
            **{key: plan[key] for key in ("codex_binary", "codex_version", "codex_binary_blake3")},
            "actual_actor_function": "training.automedbench_lite.track_actor.run_tracks",
            "equivalent_actor_cli": argv, "equivalent_cli_executed_as_subprocess": False,
            "selected_harness_root": os.environ["EVA_HARNESS_ROOT"],
            "actual_imported_sources": plan["actual_imported_sources"],
            "checkpoint_identity": commitment(server.identity_path), "model_path": context["model_path"],
            "memory_behavior_verified_by_configuration": False, "canonical_skill_ids_replaced": False,
            **({"skill_selection": context["skill_selection"]} if context.get("skill_selection") else {})}
        write(output / "evaluation-actor-binding.json", actor, exclusive=True)
        asyncio.run(run_tracks(args, memory_profile=profile["memory_profile"], supra_profile=profile["supra_profile"]))
        identity = server.identity_path
    require(read(output / "serving/session-exit.json")["cleanup"].get("verified_members_remaining") == 0,
            "supplement_serving_cleanup_unproved")
    actual = read_document(run / "track-rollouts/attempt.json")
    summary = read_document(run / "track-rollouts/summary.json")
    require(actual["tracks"] == list(TRACKS) and actual["planned_coding_rollouts"] == 2
        and actual["one_attempt_per_track"] is True and actual["full_seven_track_evaluation"] is False
        and {row["track"] for row in summary["tracks"]} == set(TRACKS), "supplement_actor_scope_differs")
    native = _score_subset(run, identity, output / "native-task-scores", production.native_scoring_options(settings))
    skill_index = production.evaluation_skill_index(run, output, context, settings, scope)
    judge_bound = {"run_root": str(run), "checkpoint_identity": commitment(identity), "actor": actor,
        "settings": settings, "judge_replacements": original["judge_replacements"]}
    rows, coverage = _judge_subset(judge_bound, output / "workspace-feedback", scope["registry"])
    roots = [row["feedback_root"] for row in rows if row["status"] == "verified"]
    extra = {}
    if roots and original["judge_replacements"]:
        from training.automedbench_lite.judge_attempt_isolation import BINDING_NAME
        extra.update(postround_judge_verdict_replacements=original["judge_replacements"],
            postround_judge_attempt_isolation=[commitment(Path(root) / BINDING_NAME) for root in roots])
    if settings.get("task_agent_score_audit", False):
        from .task_score_audit import audit_native_scores
        extra["task_agent_audits"] = audit_native_scores(
            [row["feedback_root"] for row in rows if row["status"] == "verified" and row["stage"] == "S5"],
            native, output / "task-agent-audits", native_turn_timeout_seconds=600)
    index = {**skill_index, "evaluation_mode": "diagnostic_subset", "full_seven_track_evaluation": False,
        "baseline_gate_mode": "terminal-attempts-v1", "tracks_requested": list(TRACKS),
        "stages_requested": list(PHASES), "new_rollouts_per_track": 1,
        "prepared_public_input_tracks": list(BY_TRACK), "executed_tracks": list(TRACKS),
        "matched_codex_version_requested": "0.153.4", "benchmark_run_root": str(run),
        "checkpoint_identity": str(identity), "actor_runtime_binding": commitment(output / "evaluation-actor-binding.json"),
        "feedback_roots": roots, "judge_material_view": "policy-visible-audit-v2",
        "allow_context_budget_terminal": settings.get("allow_context_budget_terminal", False),
        "native_task_scores": commitment(native), "original_source_attempt": original["source_attempt"],
        "supplemental_request": commitment(output / "supplemental-request.json"),
        "diagnostic_artifacts": [str(coverage), str(run / "track-rollouts/summary.json"),
            str(output / "serving/session-exit.json")], "stage_phase_names_are_success_claims": False,
        "source_indexes_mutated": False, "synthetic_seven_track_run_created": False,
        "missing_is_not_zero": True, **extra,
        **({"skill_selection": context["skill_selection"]} if context.get("skill_selection") else {})}
    path = output / "supplemental-index.json"
    write(path, index, exclusive=True)
    result = {"schema": "eva.rsi-supplemental-evaluation-result.v1", "status": "collected",
        "index": commitment(path), "verified_stage_count": len(roots),
        "unavailable_stage_count": len(rows) - len(roots), "controller_state_modified": False,
        "standalone_training_admission": False, "original_five_tracks_rerun": False}
    write(output / "supplemental-result.json", result, exclusive=True)
    return result
