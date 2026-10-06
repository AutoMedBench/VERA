"""One new checkpoint, explicit evaluation coverage, actual workspace Judge.

Historical settings retain their declared classification diagnostic. New full
round settings run all seven tracks and all five stages, without score cutoffs.
Neither mode infers memory/context success from a configuration flag.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from .controller import write
from .evidence import commitment, read, require, verify_evaluation


def actor_profile(settings, scope):
    """Explicit modern full-round defaults; historical diagnostics stay unchanged."""
    name = settings.get("evaluation_actor_profile", "supra-v1.3" if scope["mode"] == "full_single_pass" else "legacy")
    require(name in {"legacy", "memory-v1.2", "supra-v1.3"}, "unsupported_evaluation_actor_profile")
    modern = name != "legacy"
    values = {"workers": settings.get("evaluation_workers", 3 if modern else 4),
        "context_length": settings.get("evaluation_context_length", 32768),
        "auto_compact_token_limit": settings.get("evaluation_compact_tokens", 20480 if modern else 12288),
        "output_tokens": settings.get("evaluation_output_tokens", 4096),
        "adapter_capacity_wait_seconds": settings.get("evaluation_capacity_wait_seconds", 60 if modern else 0),
        "turn_timeout": settings.get("evaluation_turn_timeout_seconds", 900)}
    require(all(type(value) is int for value in values.values()), "evaluation_profile_requires_integer_limits")
    require(1 <= values["workers"] <= 4 and values["context_length"] == 32768
        and values["output_tokens"] == 4096 and values["turn_timeout"] == 900
        and values["adapter_capacity_wait_seconds"] in (0, 60), "unsupported_evaluation_profile_limits")
    require(values["auto_compact_token_limit"] == (20480 if modern else 12288),
            "evaluation_compaction_profile_differs")
    tool_limit = settings.get("evaluation_tool_output_token_limit")
    require(tool_limit is None or type(tool_limit) is int and tool_limit == 2048,
            "unsupported_evaluation_tool_output_limit")
    if tool_limit is not None:
        values["tool_output_token_limit"] = tool_limit
    track_timeout = settings.get("evaluation_track_timeout_seconds")
    if track_timeout is not None:
        require(type(track_timeout) is int and track_timeout == 3600 and modern,
                "unsupported_total_track_timeout")
        values["track_timeout"] = track_timeout
    return {"name": name, "memory_profile": name == "memory-v1.2", "supra_profile": name == "supra-v1.3",
        "thinking": True, "reserve_tokens": 2048, "tool_output_token_limit": tool_limit, "args": values}


def actor_arguments(run, attempt, settings, scope, profile, *, identity_path, canary_path,
                    selection=None, catalog_id=None):
    """The same arguments drive in-process execution and the retained equivalent CLI."""
    port = settings.get("evaluation_port", 30911)
    args = SimpleNamespace(run_root=run, runtime_manifest=Path(settings["public_model_runtime"]),
        public_python=Path(settings["training_python"]), image=settings["cpu_image"],
        server_canary=canary_path, server_identity=identity_path,
        codex_bin=Path(settings["codex_bin"]), tracks=scope["tracks"],
        endpoint=f"http://127.0.0.1:{port}/v1", **profile["args"])
    root = Path(__file__).resolve().parents[2]
    argv = [settings["training_python"], str(root / "training/automedbench_lite/track_entry.py"), "run"]
    for name in ("run_root", "runtime_manifest", "public_python", "image", "server_canary", "server_identity", "codex_bin", "endpoint"):
        argv.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    argv.extend(["--tracks", *scope["tracks"]])
    for name, value in profile["args"].items():
        argv.extend(["--" + name.replace("_", "-"), str(value)])
    if profile["memory_profile"]: argv.append("--memory-profile-v12")
    if profile["supra_profile"]: argv.append("--supra-profile")
    if selection is not None:
        from .skill_selection import read_selection
        read_selection(selection, catalog_id=catalog_id)
        args.skill_selection = Path(selection["path"])
        args.skill_selection_blake3 = selection["blake3"]
        args.skill_catalog_id = catalog_id
        argv.extend(["--skill-selection", selection["path"], "--skill-selection-blake3", selection["blake3"],
                     "--skill-catalog-id", catalog_id])
    return args, argv


def evaluation_skill_index(run, attempt, context, settings, scope):
    """Reuse the existing versioned content proof; never rewrite canonical IDs."""
    mode = settings.get("skill_catalog_identity_mode",
        "verified-content-v1" if scope["mode"] == "full_single_pass" else "mounted-v1")
    require(mode in {"mounted-v1", "verified-content-v1"}, "unsupported_evaluation_skill_identity_mode")
    if mode == "mounted-v1": return {"schema": "eva.rsi-evaluation-index.v1"}
    from .skill_identity import build_skill_content_binding
    proof = build_skill_content_binding(run)
    require(proof["content_identity_blake3"] == context["skill_catalog_id"], "evaluation_skill_content_differs")
    path = attempt / "skill-content-binding.json"
    write(path, proof, exclusive=True)
    return {"schema": "eva.rsi-evaluation-index.v2", "skill_catalog_identity_mode": mode,
        "skill_content_identity": commitment(path)}


def actor_runtime_sources(*, include_track_budget=False):
    """Resolve the actual imported profile before any serving/provider launch."""
    from eva_agent.codex_runtime import research_memory, supra
    from eva_agent.training import stage_execution_guidance
    from training.automedbench_lite import track_actor, track_memory_v12, local_qwen
    from . import skill_selection
    modules = (track_actor, track_memory_v12, local_qwen, research_memory, supra, skill_selection, stage_execution_guidance)
    if include_track_budget:
        from training.automedbench_lite import track_budget, policy_capture, policy_terminal, job_wait
        modules += (track_budget, policy_capture, policy_terminal, job_wait)
    selected = os.environ.get("EVA_HARNESS_ROOT")
    if selected:
        expected = Path(selected).resolve(strict=True) / "src"
        require(all(Path(module.__file__).resolve().is_relative_to(expected)
                    for module in (research_memory, supra)), "evaluation_harness_bootstrap_differs")
    return [commitment(Path(module.__file__)) for module in modules]


def codex_binary_binding(binary):
    """A requested path is not proof that the actual binary is the pinned build."""
    binary = Path(binary).resolve(strict=True)
    require(binary.is_file() and os.access(binary, os.X_OK), "evaluation_codex_binary_unavailable")
    before = binary.stat()
    version = subprocess.check_output([str(binary), "--version"], text=True, timeout=15).strip()
    require(version == "codex-cli 0.153.4", "evaluation_codex_version_differs")
    proof = commitment(binary)
    after = binary.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "evaluation_codex_binary_changed")
    return {"codex_binary": str(binary), "codex_version": version, "codex_binary_blake3": proof["blake3"]}


def evaluation_scope(settings):
    from training.automedbench_lite.track_adapter import BY_TRACK

    mode = settings.get("evaluation_mode", "diagnostic_subset")
    require(mode in {"diagnostic_subset", "full_single_pass"}, "unsupported_evaluation_scope")
    full = mode == "full_single_pass"
    root = Path(__file__).resolve().parents[2]
    return {"mode": mode, "tracks": list(BY_TRACK) if full else ["classification"],
        "stages": ("S1", "S2", "S3", "S4", "S5") if full else ("S1", "S2", "S3"),
        "registry": root / "rubrics/source" / ("domain-stage-tables.v2.json" if full else "domain-stage-tables.v1.json")}


def native_scoring_options(settings):
    """Validate explicit local runtime paths before serving; never download here."""
    evaluator = Path(settings.get("native_evaluator_python", settings["training_python"]))
    require(evaluator.is_file() and os.access(evaluator, os.X_OK), "native_evaluator_python_unavailable")
    cache = settings.get("native_lpips_torch_home")
    require(cache is None or isinstance(cache, str) and Path(cache).is_dir(),
            "native_lpips_cache_directory_unavailable")
    timeout = settings.get("native_score_timeout_seconds", 600)
    require(type(timeout) is int and 1 <= timeout <= 3600, "native_score_timeout_invalid")
    return {"evaluator_python": evaluator.absolute(), "lpips_torch_home": Path(cache).resolve() if cache else None,
        "timeout": timeout}


def score_native_round(run, identity_path, output_root, *, evaluator_python, lpips_torch_home=None, timeout=600):
    """Score retained complete workflows once; native metrics never become rubric rewards.

    Keep an independent result for every track, including unavailable results.
    A scorer/dependency/evidence failure does not erase another track's result
    or prevent the separate workspace Judge from evaluating actual stage work.
    """
    from training.automedbench_lite.adapter import read_document
    from training.automedbench_lite.track_adapter import BY_TRACK
    from training.automedbench_lite.track_scoring import score_track

    output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    rows = []
    for track in BY_TRACK:
        actor_path = run / "track-rollouts" / track / "rollout.json"
        row = {"track": track, "status": "unavailable", "task_score_0_1": None,
            "scorer_invoked": False, "reason": "actor_rollout_missing"}
        try:
            if actor_path.exists():
                actor = read_document(actor_path)
                row["actor_rollout"] = commitment(actor_path)
                row["reason"] = "actor_workflow_incomplete"
                if actor.get("completed_requested_turns") is True and not actor.get("errors"):
                    row["scorer_invoked"] = True
                    score = score_track(run, track, evaluator_python, timeout=timeout,
                        lpips_torch_home=lpips_torch_home if track == "enhancement" else None,
                        score_output_root=output_root)
                    score_path = output_root / track / "score.json"
                    require(read_document(score_path) == score, "native_score_document_differs")
                    row.update(status="scored", reason=None, task_score_0_1=score["native_result"]["task_score_0_1"],
                        native_result=score["native_result"], native_score=commitment(score_path))
        except Exception as error:
            # No raw exception text: it may include private paths or provider text.
            row.update(status="unavailable", task_score_0_1=None, reason="native_scorer_or_evidence_unavailable",
                error_type=type(error).__name__)
            failure_path = output_root / track / "failure.json"
            if failure_path.is_file():
                row["native_failure"] = commitment(failure_path)
        write(output_root / (track + "-result.json"), row, exclusive=True)
        rows.append(row)
    summary_path = output_root / "summary.json"
    write(summary_path, {"schema": "eva.rsi-native-task-score-summary.v1", "benchmark_run_root": str(run.resolve()),
        "checkpoint_identity": commitment(identity_path), "tracks_requested": list(BY_TRACK), "tracks": rows,
        "scored_tracks": sum(row["status"] == "scored" for row in rows),
        "unavailable_tracks": sum(row["status"] == "unavailable" for row in rows),
        "evaluator_python": str(evaluator_python), "lpips_torch_home": str(lpips_torch_home) if lpips_torch_home else None,
        "timeout_seconds_per_track": timeout, "new_policy_rollouts": 0, "automatic_retry": False,
        "rubric_rewards_modified": False, "used_for_stage_target_selection": False,
        "missing_is_not_zero": True}, exclusive=True)
    return summary_path


def judge_full_round(run, identity_path, output_root, *, stages, registry_path,
                    native_turn_timeout_seconds=240, attempted_stages=None, material_view="historical-v1",
                    allow_context_budget_terminal=False, context_terminal_source_binding=None,
                    policy_recovery_references=None, postround_judge_verdict_replacements=0):
    """Independent tracks may be judged in parallel; each stage runs once.

    Drain all already started track jobs and preserve errors, never turn a
    failed Judge into a zero score or an automatic retry.
    """
    from training.automedbench_lite.track_adapter import BY_TRACK
    from training.automedbench_lite.track_feedback import evaluate_track_feedback
    from training.automedbench_lite.judge_attempt_isolation import replacement_limit
    from eva_agent.training.slime_agent_judge import validate_native_judge_timeout

    validate_native_judge_timeout(native_turn_timeout_seconds)
    judge_replacements = replacement_limit(postround_judge_verdict_replacements)
    require(material_view in {"historical-v1", "policy-visible-audit-v2"}, "unsupported_judge_material_view")
    timeout_options = ({"native_turn_timeout_seconds": native_turn_timeout_seconds}
                       if native_turn_timeout_seconds != 240 else {})
    if material_view != "historical-v1":
        timeout_options["material_view"] = material_view
    require(type(allow_context_budget_terminal) is bool and
            (not allow_context_budget_terminal or attempted_stages is not None),
            "context_terminal_requires_explicit_attempt_gate")
    if allow_context_budget_terminal:
        timeout_options["allow_context_budget_terminal"] = True
        if context_terminal_source_binding is not None:
            timeout_options["context_terminal_source_binding"] = context_terminal_source_binding
    if attempted_stages is not None:
        require(set(attempted_stages) == set(BY_TRACK) and all(
            isinstance(value, list) and value and value == list(stages)[:len(value)]
            for value in attempted_stages.values()), "invalid_attempted_stage_prefixes")
        timeout_options["allow_policy_budget_terminal"] = True
    recoveries={} if policy_recovery_references is None else policy_recovery_references
    require(isinstance(recoveries,dict) and set(recoveries)<=set(BY_TRACK)
        and (not recoveries or attempted_stages is not None)
        and all(isinstance(refs,dict) and len(refs)==1 and set(refs)=={attempted_stages[track][-1]}
                for track,refs in recoveries.items()), "recovery_requires_explicit_terminal_stage_reference")
    output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    roots, failures = {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(evaluate_track_feedback, run, identity_path, output_root / track,
                           stages=stages, track=track, registry_path=registry_path,
                           **({"policy_recovery_references":recoveries[track]} if track in recoveries else {}),
                           **({"postround_judge_verdict_replacements": judge_replacements}
                              if judge_replacements else {}),
                           **timeout_options): track for track in BY_TRACK}
        for job in as_completed(jobs):
            track = jobs[job]
            try:
                roots[track] = job.result()
            except Exception as error:
                failures.append({"track": track, "error_type": type(error).__name__, "rubric_score": None})
    coverage = {"schema": "eva.rsi-seven-track-judge-coverage.v1", "tracks_requested": list(BY_TRACK),
        "stages_requested": list(stages), "max_parallel_tracks": 4, "automatic_retry": False,
        "native_turn_timeout_seconds": native_turn_timeout_seconds,
        "judge_material_view": material_view,
        "feedback_roots": [str(path) for track in BY_TRACK for path in roots.get(track, ())],
        "failures": sorted(failures, key=lambda row: row["track"]), "missing_is_not_zero": True}
    if judge_replacements:
        coverage.update(postround_judge_verdict_replacements=judge_replacements,
            automatic_additional_judge_attempts=True, same_judge_attempt_retried=False,
            additional_judge_attempts_are_not_actor_rerolls=True,
            first_valid_grade_including_zero_is_accepted=True)
    if recoveries:
        coverage["policy_recovery_references"]=recoveries
        coverage["recovered_workspace_is_not_original_terminal_snapshot"]=True
    if attempted_stages is not None:
        coverage.update(attempted_stages=attempted_stages,
            unreachable_stages={track:[stage for stage in stages if stage not in attempted_stages[track]] for track in BY_TRACK},
            unreachable_stage_scores=None,baseline_gate_mode="terminal-attempts-v1")
    write(output_root / "coverage.json", coverage, exclusive=True)
    require(not failures and all(len(roots.get(track, ())) ==
        (len(attempted_stages[track]) if attempted_stages is not None else len(stages)) for track in BY_TRACK),
            "full_round_workspace_judging_incomplete")
    return tuple(path for track in BY_TRACK for path in roots[track])


def evaluate_round(context, settings):
    # Import integration dependencies at execution, never on config/status calls.
    from training.automedbench_lite.track_adapter import TrackRelease, prepare_track_run, BY_TRACK
    from training.automedbench_lite.track_actor import run_tracks
    from training.automedbench_lite.track_feedback import evaluate_track_feedback
    from training.eva_rsi.serving import serving_session

    scope = evaluation_scope(settings)
    task_audit = settings.get("task_agent_score_audit", False)
    detailed_report = settings.get("seven_track_detailed_report", False)
    require(type(task_audit) is bool and type(detailed_report) is bool
        and (not (task_audit or detailed_report) or scope["mode"] == "full_single_pass"),
        "detailed_scores_require_full_evaluation")
    material_view = settings.get("judge_material_view",
        "policy-visible-audit-v2" if scope["mode"] == "full_single_pass" else "historical-v1")
    require(material_view in {"historical-v1", "policy-visible-audit-v2"}, "unsupported_judge_material_view")
    profile = actor_profile(settings, scope)
    gate_mode = settings.get("baseline_gate_mode", "complete-workflows-v1")
    require(gate_mode in {"complete-workflows-v1", "terminal-attempts-v1"}, "unsupported_baseline_gate_mode")
    require(gate_mode != "terminal-attempts-v1" or scope["mode"] == "full_single_pass",
            "terminal_attempt_gate_requires_explicit_full_scope")
    allow_context = settings.get("allow_context_budget_terminal", False)
    require(type(allow_context) is bool and (not allow_context or gate_mode == "terminal-attempts-v1"),
            "context_terminal_requires_explicit_attempt_gate")
    from training.automedbench_lite.judge_attempt_isolation import replacement_limit
    judge_replacements = replacement_limit(settings.get("postround_judge_verdict_replacements", 0))
    if profile["name"] != "legacy":
        parameters = inspect.signature(run_tracks).parameters
        require("memory_profile" in parameters and "supra_profile" in parameters,
                "current_evaluation_actor_profile_not_installed")
    native_options = native_scoring_options(settings) if scope["mode"] == "full_single_pass" else None
    imported_sources = (actor_runtime_sources(**({"include_track_budget": True}
        if "track_timeout" in profile["args"] else {})) if profile["name"] != "legacy" else [])
    binary_binding = codex_binary_binding(settings["codex_bin"]) if profile["name"] != "legacy" else {}
    from eva_agent.training.slime_agent_judge import validate_native_judge_timeout
    judge_timeout = settings.get("native_judge_timeout_seconds", 600 if scope["mode"] == "full_single_pass" else 240)
    validate_native_judge_timeout(judge_timeout)
    timeout_options = ({"native_turn_timeout_seconds": judge_timeout} if judge_timeout != 240 else {})
    if material_view != "historical-v1":
        timeout_options["material_view"] = material_view
    attempt = Path(context["attempt_root"])
    cds = TrackRelease(Path(settings["cds_receipt"]))
    missing = TrackRelease(Path(settings["missing4_receipt"]))
    run = prepare_track_run({name: cds if name in {"classification", "detection", "segmentation"}
                            else missing for name in BY_TRACK}, attempt / "benchmark")
    port = settings.get("evaluation_port", 30911)
    diagnostics = []
    runtime_index = {}
    with serving_session(model_path=Path(context["model_path"]),
            checkpoint_root=Path(context["checkpoint_root"]),
            architecture_model_path=Path(context["architecture_model_path"]),
            output_root=attempt / "serving", public_image=Path(settings["public_image"]), port=port,
            python_executable=Path(settings["training_python"]), context_length=profile["args"]["context_length"]) as server:
        args, argv = actor_arguments(run, attempt, settings, scope, profile,
            identity_path=server.identity_path, canary_path=server.canary_path,
            selection=context.get("skill_selection"), catalog_id=context["skill_catalog_id"])
        if profile["name"] != "legacy":
            plan_path = attempt / "evaluation-actor-binding.json"
            write(plan_path, {"schema": "eva.rsi-evaluation-actor-binding.v1", "profile_requested": profile,
                **binary_binding,
                "actual_actor_function": "training.automedbench_lite.track_actor.run_tracks",
                "equivalent_actor_cli": argv, "equivalent_cli_executed_as_subprocess": False,
                "selected_harness_root": os.environ.get("EVA_HARNESS_ROOT"),
                "actual_imported_sources": imported_sources,
                "checkpoint_identity": commitment(server.identity_path), "model_path": context["model_path"],
                "memory_behavior_verified_by_configuration": False,
                **({"skill_selection": context["skill_selection"]} if context.get("skill_selection") else {}),
                "canonical_skill_ids_replaced": False}, exclusive=True)
            diagnostics.append(str(plan_path))
            runtime_index = {"actor_runtime_binding": commitment(plan_path)}
            asyncio.run(run_tracks(args, memory_profile=profile["memory_profile"], supra_profile=profile["supra_profile"]))
        else:
            asyncio.run(run_tracks(args))
        identity_path = server.identity_path
    # The GPU serving context is released before independent remote Judging.
    native_index = {}
    if native_options is not None:
        native_summary = score_native_round(run, identity_path, attempt / "native-task-scores", **native_options)
        diagnostics.append(str(native_summary))
        native_index = {"native_task_scores": commitment(native_summary)}
    skill_index = evaluation_skill_index(run, attempt, context, settings, scope)
    if scope["mode"] == "full_single_pass":
        from .production import require_complete_baseline
        gate_document = {"evaluation_mode": scope["mode"], "benchmark_run_root": str(run),
            "checkpoint_identity": str(identity_path), "baseline_gate_mode": gate_mode,
            "allow_context_budget_terminal": allow_context, **runtime_index}
        gate_proof = require_complete_baseline(gate_document)
        judge_options = ({"attempted_stages": {track:row["attempted_stages"] for track,row in gate_proof["tracks"].items()}}
                         if gate_mode == "terminal-attempts-v1" else {})
        if allow_context:
            judge_options["allow_context_budget_terminal"] = True
            if "actor_runtime_binding" in runtime_index:
                judge_options["context_terminal_source_binding"] = read(runtime_index["actor_runtime_binding"]["path"])
        if judge_replacements:
            judge_options["postround_judge_verdict_replacements"] = judge_replacements
        feedback = judge_full_round(run, identity_path, attempt / "workspace-feedback",
                                    stages=scope["stages"], registry_path=scope["registry"], **timeout_options, **judge_options)
    else:
        replacement_options = ({"postround_judge_verdict_replacements": judge_replacements}
                               if judge_replacements else {})
        feedback = evaluate_track_feedback(run, identity_path, attempt / "workspace-feedback",
                                          stages=scope["stages"], registry_path=scope["registry"],
                                          **timeout_options, **replacement_options)
    judge_attempt_index = {}
    if judge_replacements:
        from training.automedbench_lite.judge_attempt_isolation import BINDING_NAME
        bindings = [commitment(Path(root) / BINDING_NAME) for root in feedback]
        judge_attempt_index = {"postround_judge_verdict_replacements": judge_replacements,
            "postround_judge_attempt_isolation": bindings}
    task_audit_index = {}
    if task_audit:
        from .task_score_audit import audit_native_scores
        task_audit_index["task_agent_audits"] = audit_native_scores(feedback, native_summary,
            attempt / "task-agent-audits", native_turn_timeout_seconds=judge_timeout)
    index_path = attempt / "evaluation-index.json"
    write(index_path, {**skill_index, "evaluation_mode": scope["mode"],
        "judge_material_view": material_view,
        "baseline_gate_mode": gate_mode, **runtime_index, **native_index, **task_audit_index,
        "allow_context_budget_terminal": allow_context,
        **judge_attempt_index,
        **({"skill_selection": context["skill_selection"]} if context.get("skill_selection") else {}),
        "checkpoint_identity": str(identity_path), "feedback_roots": [str(path) for path in feedback],
        "diagnostic_artifacts": [str(run / "track-rollouts/summary.json"),
            str(attempt / "workspace-feedback/coverage.json"), str(attempt / "serving/session-exit.json"), *diagnostics],
        "benchmark_run_root": str(run), "tracks_requested": scope["tracks"],
        "full_seven_track_evaluation": scope["mode"] == "full_single_pass", "matched_codex_version_requested": "0.153.4",
        "new_rollouts_per_track": 1, "stage_phase_names_are_success_claims": False}, exclusive=True)
    result = verify_evaluation(index_path, context)
    if gate_mode == "terminal-attempts-v1":
        from .terminal_attempts import require_judged_attempts
        require_judged_attempts(read(index_path), gate_proof)
    if detailed_report:
        from training.automedbench_lite.evaluation_report import build_report, render_markdown
        report = build_report(run, index_path)
        report_path = attempt / "seven-track-report.json"
        write(report_path, report, exclusive=True)
        with os.fdopen(os.open(attempt / "seven-track-report.md", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(render_markdown(report) + "\n")
        result["seven_track_report"] = commitment(report_path)
    return result
