"""Verify an explicit same-checkpoint composition of disjoint track sources.

The outer document is an adapter artifact.  It does not rewrite either source
evaluation index, manufacture a seven-track actor run, or normalize the raw
checkpoint identity retained by a Judge grade.  Consumers may normalize the
already-verified rows in memory for stage planning only.
"""
from __future__ import annotations

import math
from pathlib import Path

from eva_agent.pipeline.digests import blake3_hex, canonical_value
from training.automedbench_lite.adapter import read_document
from training.automedbench_lite.track_adapter import BY_TRACK
from training.automedbench_lite.track_feedback import PHASES
from training.benchmark_feedback.automed_codex import verify_feedback

from .evidence import commitment, read, require
from .skill_identity import verify_skill_content_binding
from .terminal_attempts import require_judged_attempts, verify_terminal_track_source


SCHEMA = "eva.rsi-composite-evaluation-index.v1"
PROOF_SCHEMA = "eva.verified-composite-evaluation.v1"
MODE = "same-checkpoint-zero-turn-track-supplement-v1"
SUPPLEMENTED = ("classification", "detection")
ORIGINAL = tuple(track for track in BY_TRACK if track not in SUPPLEMENTED)


def _bound(reference):
    require(isinstance(reference, dict) and set(reference) == {"path", "blake3"}
            and commitment(reference["path"]) == reference,
            "composite_source_commitment_changed")
    return read(reference["path"])


def _checkpoint(identity_path):
    path = Path(identity_path).resolve(strict=True)
    value = read(path)
    require(value.get("schema") == "eva.qwen-final-serving-checkpoint-identity.v1"
            and value.get("status") == "launch_prepared"
            and isinstance(value.get("hf_asset_commitments"), dict)
            and isinstance(value.get("checkpoint_preflight"), dict),
            "composite_checkpoint_identity_invalid")
    stable = {
        "architecture_model_path": str(Path(value["architecture_model_path"]).resolve(strict=True)),
        "checkpoint_root": str(Path(value["checkpoint_root"]).resolve(strict=True)),
        "exact_final_model_path": str(Path(value["exact_final_model_path"]).resolve(strict=True)),
        "final_checkpoint_iteration": value["final_checkpoint_iteration"],
        "checkpoint_preflight": value["checkpoint_preflight"],
        "hf_asset_commitments": value["hf_asset_commitments"],
    }
    require(type(stable["final_checkpoint_iteration"]) is int
            and stable["final_checkpoint_iteration"] >= 0,
            "composite_checkpoint_iteration_invalid")
    return {"identity": commitment(path), "stable": stable,
            "stable_blake3": blake3_hex(canonical_value(stable))}


def _zero_turn_eligibility(run, track, identity, summary_rows):
    """Prove an omitted original row had no actor/provider turn to retain."""
    audit = run / "track-rollouts" / track
    row = read_document(audit / "rollout.json")
    require(row == next(item for item in summary_rows if item.get("track") == track)
            and row.get("source_checkpoint") == identity["exact_final_model_path"]
            and row.get("actual_turn_count") == 0
            and row.get("turn_receipt_blake3s") == []
            and row.get("completed_requested_turns") is False
            and isinstance(row.get("errors"), list) and bool(row["errors"]),
            "composite_original_supplement_track_not_zero_turn")
    require(not any((audit / "turns" / phase / "receipt.json").exists() for phase in PHASES.values()),
            "composite_original_supplement_track_has_receipt")
    budget = read_document(audit / "track-budget.json")
    outcome = read_document(audit / "track-budget-outcome.json")
    cleanup = read_document(audit / "track-deadline-cleanup.json")
    elapsed = outcome.get("elapsed_seconds")
    require(budget.get("schema") == "eva.automedbench-track-budget.v1"
            and outcome.get("schema") == "eva.automedbench-track-budget-outcome.v1"
            and outcome.get("budget_document_blake3") == budget["document_blake3"]
            and outcome.get("policy") == budget.get("policy") == "admitted-track-wallclock-3600-v1"
            and outcome.get("timeout_seconds") == budget.get("timeout_seconds") == 3600
            and budget.get("queue_before_admission_charged") is False
            and budget.get("job_waits_and_runtime_restart_charged") is True
            and outcome.get("actual_turn_count") == 0 and outcome.get("app_server_pids") == []
            and type(outcome.get("deadline_exhausted")) is bool
            and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0
            and outcome["deadline_exhausted"] == (elapsed >= outcome.get("timeout_seconds", -1))
            and math.isclose(outcome.get("remaining_seconds", -1),
                             max(0, outcome["timeout_seconds"] - elapsed), abs_tol=1e-6)
            and math.isclose(outcome.get("cleanup_overrun_seconds", -1),
                             max(0, elapsed - outcome["timeout_seconds"]), abs_tol=1e-6),
            "composite_original_zero_turn_budget_invalid")
    require(cleanup.get("schema") == "eva.automedbench-track-deadline-cleanup.v1"
            and cleanup.get("workspace_quiescent") is True
            and cleanup.get("signals") == [] and cleanup.get("policy_time_extended") is False
            and cleanup.get("scope") == "exact_owned_model_supervisors_only",
            "composite_original_zero_turn_cleanup_invalid")
    manifest = read_document(run / "track-run-manifest.json")
    actor = next(item for item in manifest["tracks"] if item.get("track") == track)
    workspace = run / actor["workspace_relative"]
    require(all(not any((workspace / relative).rglob("*"))
                for relative in ("outputs/agents_outputs", "notes", "code"))
            and not any((audit / "model-jobs").glob("*/submission.json")),
            "composite_original_zero_turn_workspace_changed")
    return {"track": track, "actor_rollout": commitment(audit / "rollout.json"),
        "track_budget": commitment(audit / "track-budget.json"),
        "track_budget_outcome": commitment(audit / "track-budget-outcome.json"),
        "cleanup": commitment(audit / "track-deadline-cleanup.json"),
        "actual_turn_count": 0, "provider_turns": 0, "workspace_quiescent": True,
        "original_failure_retained": True, "eligible_for_supplement": True}


def _skill(reference, run):
    supplied = _bound(reference)
    result = verify_skill_content_binding(Path(reference["path"]), expected_run_root=run,
        expected_content_id=supplied["content_identity_blake3"])
    return result


def _judge_isolation(index):
    from training.automedbench_lite.judge_attempt_isolation import (
        BINDING_NAME, replacement_limit, verify_stage_judge_attempts,
    )
    if "approved_stage_rejudgments" in index:
        # An explicit user-approved implementation correction is not an
        # automatic failed-verdict replacement policy. Preserve every original
        # isolation proof; the two new one-shot grades carry separate records.
        from .report_rejudge import verify_rejudge_index_sidecar
        approved = verify_rejudge_index_sidecar(index)
        original = _bound(approved["original_index"])
        standalone = approved["standalone_roots"]
        require("approved_stage_rejudgments" not in original
                and isinstance(standalone, list) and len(standalone) == 2
                and all(isinstance(root, str) for root in standalone)
                and len(set(standalone)) == 2
                and not set(standalone) & set(original.get("feedback_roots", []))
                and index.get("feedback_roots") == original.get("feedback_roots", []) + standalone
                and all(index.get(key) == original.get(key) for key in (
                    "postround_judge_attempt_isolation", "postround_judge_verdict_replacements",
                    "benchmark_run_root", "checkpoint_identity", "actor_runtime_binding")),
                "composite_approved_rejudge_source_or_policy_differs")
        results = _judge_isolation(original)
        pairs = set()
        for root_value in standalone:
            root = Path(root_value).resolve(strict=True)
            require(str(root) == root_value and not (root / BINDING_NAME).exists(),
                    "composite_approved_rejudge_must_be_standalone")
            feedback = verify_feedback(root)
            pair = (feedback.get("case_id"), feedback.get("stage"))
            require(feedback.get("valid") is True and feedback.get("status") == "scored"
                    and pair in {("report", "S4"), ("report", "S5")} and pair not in pairs,
                    "composite_approved_rejudge_pair_or_verification_invalid")
            pairs.add(pair)
            results.append({"schema": "eva.verified-approved-report-stage-rejudgment.v1",
                "valid": True, "track": pair[0], "stage": pair[1],
                "accepted_attempt_root": root_value, "approval": index["approved_stage_rejudgments"],
                "judge_receipt_blake3": feedback["judge_codex_receipt_blake3"],
                "score": feedback["score"], "actor_rerolls": 0,
                "automatic_judge_replacements": False})
        return results
    references = index.get("postround_judge_attempt_isolation")
    roots = index.get("feedback_roots", [])
    if references is None:
        require("postround_judge_verdict_replacements" not in index
                and not any((Path(root) / BINDING_NAME).exists() for root in roots),
                "composite_unbound_judge_attempt_isolation")
        return []
    replacement_limit(index.get("postround_judge_verdict_replacements"))
    require(isinstance(references, list) and len(references) == len(roots)
            and all(commitment(ref["path"]) == ref for ref in references),
            "composite_judge_attempt_isolation_commitment")
    results = [verify_stage_judge_attempts(Path(ref["path"]), verify=verify_feedback)
               for ref in references]
    require([row["accepted_attempt_root"] for row in results]
            == [str(Path(root).resolve()) for root in roots],
            "composite_judge_attempt_isolation_roots_differ")
    return results


def _feedback_rows(index, source_name, selected_tracks):
    rows = []
    seen = set()
    family = None
    if "judge_comparison_family" in index:
        from .judge_identity import verify_judge_comparison_family
        require(index.get("judge_identity_mode") == "declared-annotation-family-v1",
                "composite_judge_family_opt_in_required")
        family = verify_judge_comparison_family(index["judge_comparison_family"])
    for root_value in index.get("feedback_roots", []):
        root = Path(root_value).resolve(strict=True)
        if family:
            from .judge_identity import reopen_family_feedback
            feedback = reopen_family_feedback(root, family, include_skill_source_binding=True)
        else:
            feedback = verify_feedback(root, include_skill_source_binding=True)
        key = (feedback["case_id"], feedback["stage"])
        require(feedback.get("valid") is True and feedback.get("status") == "scored"
                and key[0] in selected_tracks and key not in seen,
                "composite_feedback_scope_or_identity_invalid")
        seen.add(key)
        rows.append({"root": str(root), "source_name": source_name, "track": key[0], "stage": key[1],
            "judge_receipt_blake3": feedback["judge_codex_receipt_blake3"],
            "raw_round_identity": feedback["round_identity"],
            "skill_source_binding": feedback["skill_source_binding"],
            "score_document": feedback["score"], "rubric_digest": feedback["rubric_digest"]})
    return rows


def _source(index_reference, skill_reference, *, source_name, selected_tracks,
            expected_attempt_tracks, skill_selection):
    index = _bound(index_reference)
    require(index.get("schema") in {"eva.rsi-evaluation-index.v1", "eva.rsi-evaluation-index.v2"}
            and index.get("evaluation_mode") == "diagnostic_subset"
            and set(index.get("tracks_requested", [])) >= set(selected_tracks)
            and index.get("matched_codex_version_requested") == "0.153.4",
            "composite_source_index_scope_invalid")
    if source_name == "original":
        require(index.get("new_rollouts_per_track") == 0
                and index.get("full_seven_track_evaluation") is False
                and set(index.get("tracks_requested", [])) == set(BY_TRACK),
                "composite_original_index_semantics_invalid")
    else:
        require(index.get("new_rollouts_per_track") == 1
                and index.get("full_seven_track_evaluation") is False
                and index.get("tracks_requested") == list(SUPPLEMENTED)
                and index.get("stages_requested") == list(PHASES),
                "composite_supplement_index_semantics_invalid")
    run = Path(index["benchmark_run_root"]).resolve(strict=True)
    attempt = read_document(run / "track-rollouts/attempt.json")
    runtime_reference = index.get("actor_runtime_binding")
    runtime = _bound(runtime_reference)
    require(index.get("skill_selection") == skill_selection
            and attempt.get("skill_selection") == skill_selection
            and runtime.get("skill_selection") == skill_selection,
            "composite_source_skill_selection_differs")
    checkpoint = _checkpoint(index["checkpoint_identity"])
    require(runtime.get("checkpoint_identity") == checkpoint["identity"]
            and Path(runtime.get("model_path", "")).resolve()
                == Path(checkpoint["stable"]["exact_final_model_path"]),
            "composite_source_runtime_checkpoint_differs")
    proof = verify_terminal_track_source(index, selected_tracks=selected_tracks,
        expected_attempt_tracks=expected_attempt_tracks)
    require_judged_attempts(index, proof)
    skill = _skill(skill_reference, run)
    if "skill_content_identity" in index:
        require(index.get("skill_catalog_identity_mode") == "verified-content-v1"
                and index["skill_content_identity"] == skill_reference,
                "composite_source_skill_sidecar_differs")
    feedback = _feedback_rows(index, source_name, set(selected_tracks))
    isolation = _judge_isolation(index)
    require(all(row["raw_round_identity"].get("checkpoint_id") == checkpoint["identity"]["blake3"]
                and row["raw_round_identity"].get("skill_catalog_id") == skill["mounted_catalog_blake3"]
                for row in feedback),
            "composite_feedback_raw_identity_differs")
    expected = {(track, stage) for track, row in proof["tracks"].items()
                for stage in row["attempted_stages"]}
    require({(row["track"], row["stage"]) for row in feedback} == expected,
            "composite_feedback_coverage_differs")
    return {"index": index_reference, "benchmark_run_root": str(run),
        "actor_runtime_binding": runtime_reference,
        "checkpoint_identity": checkpoint["identity"], "checkpoint": checkpoint,
        "skill_content_binding": skill_reference,
        "skill_content_id": skill["content_identity_blake3"],
        "mounted_catalog_blake3": skill["mounted_catalog_blake3"],
        "skill_attempt_document_blake3": skill["attempt"]["document_blake3"],
        "attempt_document_blake3": proof["run_attempt_document_blake3"],
        "selected_tracks": list(selected_tracks), "terminal_proof": proof,
        "judge_attempt_isolation": isolation, "feedback": feedback}


def verify_composite_index(document_or_path):
    """Reopen the two immutable sources and return a planner-ready proof union."""
    if isinstance(document_or_path, (str, Path)):
        path = Path(document_or_path).resolve(strict=True)
        document = read(path)
        outer = commitment(path)
    else:
        require(isinstance(document_or_path, dict), "composite_index_document_invalid")
        document = document_or_path
        outer = {"path": None, "blake3": blake3_hex(canonical_value(document))}
    expected_map = {track: ("supplement" if track in SUPPLEMENTED else "original") for track in BY_TRACK}
    require(document.get("schema") == SCHEMA and document.get("composition_mode") == MODE
            and document.get("supplemented_tracks") == list(SUPPLEMENTED)
            and document.get("track_sources") == expected_map
            and document.get("source_indexes_mutated") is False
            and document.get("synthetic_seven_track_run_created") is False
            and document.get("canonical_source_schemas_unchanged") is True
            and document.get("missing_is_not_zero") is True,
            "composite_index_policy_invalid")
    require("skill_selection" in document
            and (document["skill_selection"] is None
                 or isinstance(document["skill_selection"], dict)),
            "composite_skill_selection_binding_missing")
    require(document["original_index"] != document["supplement_index"]
            and document["original_skill_content_binding"] != document["supplement_skill_content_binding"],
            "composite_sources_must_be_distinct")
    original = _source(document["original_index"], document["original_skill_content_binding"],
        source_name="original", selected_tracks=ORIGINAL, expected_attempt_tracks=tuple(BY_TRACK),
        skill_selection=document["skill_selection"])
    supplement = _source(document["supplement_index"], document["supplement_skill_content_binding"],
        source_name="supplement", selected_tracks=SUPPLEMENTED, expected_attempt_tracks=SUPPLEMENTED,
        skill_selection=document["skill_selection"])
    require(Path(original["benchmark_run_root"]) != Path(supplement["benchmark_run_root"]),
            "composite_actor_runs_must_be_distinct")
    require(original["checkpoint"]["stable"] == supplement["checkpoint"]["stable"],
            "composite_checkpoint_lineage_or_assets_differ")
    require(original["skill_content_id"] == supplement["skill_content_id"],
            "composite_skill_content_identity_differs")
    original_run = Path(original["benchmark_run_root"])
    original_summary = read_document(original_run / "track-rollouts/summary.json")
    original_identity = read(original["checkpoint_identity"]["path"])
    zero = {track: _zero_turn_eligibility(original_run, track, original_identity,
                                           original_summary["tracks"])
            for track in SUPPLEMENTED}
    feedback = original.pop("feedback") + supplement.pop("feedback")
    tracks = {track: {"source_name": expected_map[track],
        **(original if expected_map[track] == "original" else supplement)["terminal_proof"]["tracks"][track]}
        for track in BY_TRACK}
    return {"schema": PROOF_SCHEMA, "valid": True, "composite_index": outer,
        "sources": {"original": original, "supplement": supplement}, "tracks": tracks,
        "feedback": feedback, "original_zero_turn_supplement_eligibility": zero,
        "comparison_checkpoint_identity": original["checkpoint_identity"],
        "checkpoint_equivalence_blake3": original["checkpoint"]["stable_blake3"],
        "skill_content_id": original["skill_content_id"], "provider_calls": 0,
        "originals_mutated": False, "synthetic_seven_track_run_created": False,
        "unreachable_is_not_zero": True}
