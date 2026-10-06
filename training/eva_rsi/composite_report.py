"""Report-only fixed original-five plus classification/detection supplementation.

Each source is independently reopened by the existing report builder. Missing
grades remain unavailable; this module does not run the strict training gate,
rewrite either source, choose better scores, judge, score, or launch an actor.
"""
from copy import deepcopy
from pathlib import Path

from eva_agent.pipeline.digests import blake3_hex
from training.automedbench_lite.adapter import read_document
from training.automedbench_lite.evaluation_report import build_report
from training.automedbench_lite.track_adapter import BY_TRACK
from .composite_eval import _checkpoint
from .evidence import commitment, read, require

SCHEMA = "eva.automedbench-composite-seven-track-report.v1"
SUPPLEMENTED = ("classification", "detection")


def _source(index_path, name):
    path = Path(index_path).resolve(strict=True)
    index = read(path)
    run = Path(index["benchmark_run_root"]).resolve(strict=True)
    attempt = read_document(run / "track-rollouts/attempt.json")
    expected = set(BY_TRACK) if name == "original" else set(SUPPLEMENTED)
    require(set(attempt.get("tracks", [])) == expected, "composite_report_actor_scope_differs")
    checkpoint = _checkpoint(index["checkpoint_identity"])
    reference = commitment(path)
    report = build_report(run, path)
    require(report["evaluation_index"] == reference
            and report["checkpoint_identity"] == checkpoint["identity"]
            and commitment(path) == reference, "composite_report_source_changed")
    rows = {row["track"]: row for row in report["tracks"]}
    require(len(rows) == len(report["tracks"]) == len(BY_TRACK) and set(rows) == set(BY_TRACK),
            "composite_report_track_rows_invalid")
    binding = {"source_name": name, "evaluation_index": reference, "benchmark_run_root": str(run),
        "checkpoint_identity": checkpoint["identity"], "harness": report["harness"],
        "raw_mounted_catalog_blake3": attempt.get("verified_skill_catalog_blake3"),
        "judge_runtime": {"status": "not_aggregated", "inferred_from_actor_version": False,
            "evidence": "Exact native Judge receipts remain at each verified stage feedback_root; "
                        "source_report retains those roots and judge_receipt_blake3 values."},
        "source_report_blake3": blake3_hex(report)}
    return binding, checkpoint, report, rows


def build_composite_report(original_index, supplement_index):
    original = _source(original_index, "original")
    supplement = _source(supplement_index, "supplement")
    require(original[0]["evaluation_index"] != supplement[0]["evaluation_index"]
            and original[0]["benchmark_run_root"] != supplement[0]["benchmark_run_root"],
            "composite_report_sources_not_distinct")
    require(original[1]["stable"] == supplement[1]["stable"],
            "composite_report_checkpoint_lineage_or_assets_differ")
    failed = {}
    for track in SUPPLEMENTED:
        audit = Path(original[0]["benchmark_run_root"]) / "track-rollouts" / track
        path = audit / "rollout.json"
        actor = read_document(path)
        require(actor.get("actual_turn_count") == 0 and actor.get("turn_receipt_blake3s") == []
                and actor.get("completed_requested_turns") is False
                and isinstance(actor.get("errors"), list) and actor["errors"],
                "composite_report_original_track_not_failed_zero_turn")
        failed[track] = {"source_rollout": commitment(path), "actual_turn_count": 0,
            "error_count": len(actor["errors"]), "failure_preserved": True,
            "source_report_row": deepcopy(original[3][track]),
            "budget_metadata": {name: commitment(audit / name) for name in (
                "track-budget.json", "track-budget-outcome.json", "track-deadline-cleanup.json")
                if (audit / name).is_file()}}
    rows = []
    for track in BY_TRACK:
        source = supplement if track in SUPPLEMENTED else original
        rows.append({**deepcopy(source[3][track]), "source": deepcopy(source[0])})
    verified = sum(row["A_stage_coverage"] for row in rows)
    return {"schema": SCHEMA, "composition_mode": "original-five-plus-cd-only-report-v1",
        "track_sources": {row["track"]: row["source"]["source_name"] for row in rows},
        "supplemented_tracks": list(SUPPLEMENTED), "tracks": rows,
        "sources": {name: {**deepcopy(source[0]), "source_report": deepcopy(source[2])}
                    for name, source in (("original", original), ("supplement", supplement))},
        "original_failed_tracks": failed,
        "checkpoint_equivalence": {"stable_blake3": original[1]["stable_blake3"],
            "comparison": "same resolved checkpoint/HF lineage and retained asset/header commitments",
            "full_weight_payload_equality_verified": False,
            "raw_identity_documents_normalized": False},
        "verified_process_stage_count": verified, "unavailable_process_stage_count": 35-verified,
        "numeric_task_count": sum(row["T"] is not None for row in rows),
        "definitions": deepcopy(original[2]["definitions"]),
        "report_only": True, "training_admission_verified": False,
        "mixed_actor_source_versions": True, "synthetic_seven_track_run_created": False,
        "source_indexes_mutated": False, "score_based_source_selection": False,
        "new_rollouts": 0, "new_judgments": 0, "new_native_scores": 0,
        "missing_is_not_zero": True, "clinical_generalization_claimed": False}


def render_markdown(report):
    def fmt(value):
        return "N/A" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)
    lines = ["Mixed-source seven-track report — report only, not training admission", "",
        "Original five tracks retained; only classification/detection come from the supplement. "
        "This is not a single seven-track actor run. Original zero-turn failures are preserved.", "",
        "| Track | Source | A | Coverage | T | A+T | Mean | Turns | Time seconds / source | Input / output tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for row in report["tracks"]:
        runtime = row["runtime"]
        lines.append("| " + " | ".join((row["track"], row["source"]["source_name"], fmt(row["A"]),
            f'{row["A_stage_coverage"]}/5' + (" provisional" if row["A_provisional"] else ""),
            fmt(row["T"]), fmt(row["A_plus_T"]), fmt(row["mean_A_T"]), str(runtime["turns"]),
            fmt(runtime["reported_time_seconds"]) + " / " + runtime["reported_time_source"],
            fmt(runtime["input_tokens"]) + " / " + fmt(runtime["output_tokens"]))) + " |")
    lines += ["", "A is the mean of available verified process stages; T is the native full-task score. "
        "Missing stages/scores remain N/A. Combined values with partial A remain provisional.", "",
        "| Track / stage A | S1 | S2 | S3 | S4 | S5 | Task audit |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in report["tracks"]:
        lines.append("| " + " | ".join([row["track"], *[fmt(row["stages"][stage]["A"])
            for stage in ("S1", "S2", "S3", "S4", "S5")], row["task"]["task_judge_audit_status"]]) + " |")
    lines += ["", f'Verified process stages: {report["verified_process_stage_count"]}/35; '
        f'numeric native tasks: {report["numeric_task_count"]}/7.',
        "Stage-level T is not inferred from full-track T. Track times are not summed into parallel wall time.",
        "Judge runtime versions are not inferred from actor versions; exact Judge receipts remain at "
        "the verified stage feedback roots retained in the source reports.",
        "", "Source-specific checkpoint and harness bindings (full versions/hashes retained in JSON):"]
    for name, source in report["sources"].items():
        harness = source["harness"]
        lines += [f'- {name}: index {source["evaluation_index"]["path"]}; run {source["benchmark_run_root"]}',
            f'  checkpoint {source["checkpoint_identity"]["path"]}; '
            f'Actor Codex {harness.get("codex_version") or harness.get("codex_version_requested") or "unknown"}; '
            f'CORE {harness.get("selected_harness_root") or "unknown"}']
    return "\n".join(lines) + "\n"
