"""Report all seven predeclared tracks, keeping unknown results visible."""
from __future__ import annotations

import math
import json
import ast
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from training.automedbench_lite.adapter import file_digest, read_document, write_once
from training.automedbench_lite.track_adapter import BY_TRACK
from .benchmark_admission import observed_infrastructure_failures, require_scoring_admission


ROOT = Path(__file__).resolve().parents[3]


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _frozen_instruction_literals(record):
    relative = "evamed-codex/src/evamed_codex/benchmark.py"
    expected = next(row["blake3"] for row in record.harness["sources"] if row["path"] == relative)
    path = record.root / "frozen-sources" / relative
    _require(file_digest(path) == expected, "retry_frozen_instruction_source_changed")
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_track")
    result = {}
    for name in ("base", "developer"):
        values = [node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
        _require(len(values) == 1, "retry_instruction_literal_ambiguous")
        result[name] = ast.literal_eval(values[0])
    return result


def _actor_sources(record):
    prefix = "evamed-codex/src/evamed_codex/"
    return {row["path"]: row["blake3"] for row in record.harness["sources"]
            if not (row["path"].startswith(prefix) and (
                Path(row["path"]).name.startswith(("portable_", "slime_")) or
                Path(row["path"]).name in {"benchmark_report.py", "benchmark_judge.py"}))}


@dataclass
class ReportRun:
    root: Path
    manifest: dict
    harness: dict
    lineage: dict | None

    @classmethod
    def open(cls, root):
        root = root.resolve(strict=True)
        path = root / "infra-retry-lineage.json"
        return cls(root, read_document(root / "track-run-manifest.json"),
                   read_document(root / "harness-manifest.json", maximum=16 * 1024**2),
                   read_document(path, maximum=16 * 1024**2) if path.exists() else None)

    def task(self, track):
        matches = [row for row in self.manifest["tracks"] if row["track"] == track]
        _require(len(matches) == 1, "track_manifest_entry_not_unique")
        return matches[0]

    def actor_path(self, track):
        return self.root / "track-rollouts" / track / "rollout.json"


def verify_retry(previous: ReportRun, retry: ReportRun) -> dict[str, dict]:
    """Verify actual failed receipts and the original single-retry reservation."""
    from eva_agent.codex_runtime import codex_turn_receipt_from_document, verify_codex_turn_receipt
    from eva_agent.pipeline.digests import canonical_value

    value = retry.lineage
    _require(value is not None and value.get("schema") == "eva.benchmark-infra-retry-lineage.v1",
             "retry_lineage_missing")
    _require(value["primary_run_root"] == str(previous.root) and value["retry_run_root"] == str(retry.root)
             and value["primary_manifest_blake3"] == previous.harness["document_blake3"]
             and value["retry_manifest_blake3"] == retry.harness["document_blake3"], "retry_run_binding_invalid")
    _require(value.get("retry_number") == value.get("maximum_infrastructure_retries_per_track") == 1
             and value.get("original_attempts_preserved") is True
             and value.get("fresh_actor_workspaces") is True, "retry_count_or_preservation_invalid")
    _require(previous.harness["config"] == retry.harness["config"]
             and value.get("profile_configuration_equal") is True, "retry_configuration_changed")
    _require(all(previous.harness[key] == retry.harness[key] for key in (
        "tool_catalog", "tool_catalog_blake3", "canonical_skill_manifest_blake3", "codex_binary_blake3",
        "model_download_receipt_blake3")) and _frozen_instruction_literals(previous) == _frozen_instruction_literals(retry),
        "retry_instructions_tools_model_or_binary_changed")
    original_sources = {row["path"]: row["blake3"] for row in previous.harness["sources"]}
    changes = [{"path": row["path"], "previous": original_sources.get(row["path"]), "replacement": row["blake3"]}
               for row in retry.harness["sources"] if original_sources.get(row["path"]) != row["blake3"]]
    _require(changes == value["source_changes"], "retry_source_changes_not_fully_declared")
    result = {}
    for proof in value["tracks"]:
        track = proof["track"]
        _require(track not in result, "duplicate_retry_track")
        before, after = previous.task(track), retry.task(track)
        _require(all(before[key] == after[key] for key in (
            "task_file_blake3", "input_manifest_file_blake3", "case_count", "source_acquisition_document_blake3"))
            and proof["public_task_file_blake3"] == after["task_file_blake3"]
            and proof["input_manifest_file_blake3"] == after["input_manifest_file_blake3"], "retry_public_task_changed")
        actor = read_document(previous.actor_path(track))
        _require(not (previous.root / "native-scores" / track / "score.json").exists(),
                 "genuine_scored_attempt_cannot_be_replaced")
        _require(actor.get("terminal_disposition") == "infrastructure-unknown"
                 and actor.get("purpose") == "benchmark"
                 and actor["document_blake3"] == proof["previous_actor_document_blake3"]
                 and proof.get("native_or_judge_score_used_to_choose_retry") is False,
                 "retry_previous_attempt_not_proved_infrastructure")
        audit = previous.root / "track-rollouts" / track
        receipt = codex_turn_receipt_from_document(json.loads((audit / "turns/01-e2e/receipt.json").read_bytes()))
        verify_codex_turn_receipt(receipt)
        _require(receipt.receipt_blake3 == proof["previous_codex_receipt_blake3"]
                 and actor["turn_receipt_blake3s"] == [receipt.receipt_blake3], "retry_previous_codex_binding_invalid")
        observed = observed_infrastructure_failures(audit)
        for call in receipt.tool_calls:
            output = canonical_value(call.output)
            if not isinstance(output, dict):
                continue
            message = (output.get("error") or {}).get("message", "")
            if (call.status == "failed" and output.get("result") is None
                    and "timed out awaiting tools/call after 200s" in message):
                observed.append({"kind": "actual_mcp_response_timeout", "codex_tool_call_id": call.tool_call_id,
                    "codex_tool_receipt_blake3": call.receipt_blake3, "tool": call.mcp_tool,
                    "duration_ms": output.get("durationMs"), "host_result_available": False})
        _require(observed and observed == proof["observed_infrastructure_failures"],
                 "retry_actual_infrastructure_evidence_differs")
        reservation = read_document(ROOT / "evamed-codex/receipts/benchmark-retries" /
                                    (previous.root.name + "-" + track + ".json"))
        _require(reservation.get("retry_number") == 1 and reservation.get("track") == track
                 and reservation.get("primary_run_root") == str(previous.root)
                 and reservation.get("retry_run_root") == str(retry.root)
                 and reservation.get("lineage_document_blake3") == value["document_blake3"],
                 "single_retry_reservation_invalid")
        result[track] = {"run_root": str(previous.root), "run_id": previous.manifest["run_id"],
            "actor_document_blake3": actor["document_blake3"], "codex_receipt_blake3": receipt.receipt_blake3,
            "disposition": "excluded_proved_infrastructure", "infrastructure_evidence": observed,
            "model_requests": actor["actual_model_request_count"], "elapsed_seconds": actor["wall_seconds"],
            "lineage_document_blake3": value["document_blake3"], "retry_number": 1}
    _require(set(result) == {row["track"] for row in retry.manifest["tracks"]},
             "retry_manifest_contains_unapproved_tracks")
    return result


def verify_preparation_replacement(previous, replacement, track):
    value = read_document(replacement.root / "primary-equivalence.json")
    _require(value.get("schema") == "eva.benchmark-track-source-equivalence.v1"
             and value.get("track") == track and value.get("actual_actor_retry") is False
             and value.get("primary_run_root") == str(previous.root)
             and value.get("replacement_run_root") == str(replacement.root)
             and value.get("primary_manifest_blake3") == previous.harness["document_blake3"]
             and value.get("replacement_manifest_blake3") == replacement.harness["document_blake3"],
             "prepared_only_replacement_binding_invalid")
    before, after = previous.task(track), replacement.task(track)
    _require(not previous.actor_path(track).exists()
             and before["task_file_blake3"] == after["task_file_blake3"] == value["task_file_blake3"]
             and before["input_manifest_file_blake3"] == after["input_manifest_file_blake3"] == value["input_manifest_file_blake3"]
             and _actor_sources(previous) == _actor_sources(replacement),
             "prepared_only_replacement_actor_sources_or_inputs_changed")


def select_tracks(records: list[ReportRun]):
    by_root = {record.root: record for record in records}
    _require(len(by_root) == len(records), "duplicate_report_run")
    if records:
        _require(all(record.harness["config"] == records[0].harness["config"] for record in records),
                 "mixed_benchmark_configuration_in_repeat1_report")
    edges = {}
    for retry in records:
        if retry.lineage is None:
            continue
        previous = by_root.get(Path(retry.lineage["primary_run_root"]).resolve())
        _require(previous is not None, "retry_primary_run_must_be_reported")
        _require(previous.lineage is None, "second_infrastructure_retry_not_allowed")
        for track, evidence in verify_retry(previous, retry).items():
            _require(track not in edges, "multiple_infrastructure_retries_not_allowed")
            edges[track] = (previous, retry, evidence)
    selected = {}
    for track in BY_TRACK:
        prepared = [r for r in records if any(t["track"] == track for t in r.manifest["tracks"])]
        actual = [r for r in prepared if r.actor_path(track).exists()]
        if track in edges:
            previous, retry, evidence = edges[track]
            _require(all(r.root in {previous.root, retry.root} for r in actual), "unlinked_duplicate_actual_attempt")
            selected[track] = (retry, [evidence], [str(r.root) for r in prepared if r.root != retry.root
                                                and not r.actor_path(track).exists()])
        else:
            _require(len(actual) <= 1, "repeat1_duplicate_actual_track_attempt")
            if prepared:
                chosen = actual[0] if actual else prepared[0]
                if actual:
                    for previous in prepared:
                        if previous.root != chosen.root:
                            verify_preparation_replacement(previous, chosen, track)
                selected[track] = (chosen, [], [str(r.root) for r in prepared if r.root != chosen.root])
    return selected


def infrastructure_reasons(record: ReportRun, track: str, actor: dict) -> list[str]:
    if actor.get("terminal_disposition") != "infrastructure-unknown":
        return []
    reasons = []
    path = record.root / "track-rollouts" / track / "mcp-infrastructure-errors.jsonl"
    expected = {row.get("document_blake3") for row in actor.get("infrastructure_failures", [])}
    if path.exists():
        from training.automedbench_lite.adapter import blake3, canonical
        for line in path.read_bytes().splitlines():
            row = json.loads(line)
            core = {key: value for key, value in row.items() if key != "document_blake3"}
            _require(row.get("document_blake3") in expected and
                     row["document_blake3"] == blake3(canonical(core)).hexdigest(),
                     "reported_host_failure_evidence_changed")
            if row.get("error_code") == "track_file_changed_during_snapshot":
                reasons.append("Workspace snapshot encountered an in-progress artifact publication.")
    runtime = "EVA-Harness/src/eva_agent/codex_runtime/runtime.py"
    source_row = next((row for row in record.harness["sources"] if row["path"] == runtime), None)
    if source_row is not None:
        source = record.root / "frozen-sources" / runtime
        _require(file_digest(source) == source_row["blake3"], "reported_runtime_failure_source_changed")
        lines = source.read_text().splitlines()
        for error in actor.get("errors", []):
            for frame in error.get("frames", []):
                line = frame.get("line", 0)
                if (frame.get("file") == "runtime.py" and frame.get("function") == "run_turn"
                        and 1 <= line <= len(lines) and "completed with active tool calls" in lines[line - 1]):
                    reasons.append("At the deadline, the native terminal still had an active tool; the ordinary receipt guard rejected its incomplete lifecycle.")
    return list(dict.fromkeys(reasons)) or ["Infrastructure evidence is incomplete; no native task score is admitted."]


def supplementary_native_score(record, track, actor, overlay):
    """Reopen the reviewed same-actor metadata repair; never replace a valid score."""
    import importlib.util
    overlay = Path(overlay).resolve(strict=True)
    selection_path = overlay / (track + "-selected-score.json")
    if not selection_path.exists():
        return None
    original = record.root / "native-scores" / track
    _require(not (original / "score.json").exists(), "supplement_cannot_replace_valid_native_score")
    failure = read_document(original / "failure.json")
    _require(failure["error_code"] == "model_terminal_receipt_not_in_stage_snapshot",
             "supplement_failure_outside_reviewed_scope")
    selection = read_document(selection_path)
    lineage = read_document(overlay / (track + "-scorer-lineage.json"), maximum=16 * 1024**2)
    score_path = overlay / track / "score.json"
    score = read_document(score_path, maximum=64 * 1024**2)
    _require(selection["schema"] == "eva.same-actor-native-score-selection.v1" and
             lineage["schema"] == "eva.same-actor-native-scorer-compatibility-lineage.v1" and
             all(row["track"] == track and row["source_run_root"] == str(record.root) and
                 row["actor_document_blake3"] == actor["document_blake3"] and
                 row["new_actor_rollouts"] == 0 for row in (selection, lineage)) and
             selection["native_score_path"] == str(score_path) and
             selection["score_document_blake3"] == score["document_blake3"] and
             selection["scorer_lineage_blake3"] == lineage["document_blake3"] and
             lineage["harness_manifest_blake3"] == record.harness["document_blake3"] and
             lineage["original_scoring_failure"] == failure and
             lineage["native_metric_math_changed"] is False and lineage["actor_artifacts_changed"] is False,
             "supplement_actor_or_lineage_changed")
    review_path = ROOT / "evamed-codex/receipts/native-receipt-overlay-independent-review.json"
    _require(file_digest(review_path) == "f3cd1fd5c4c5d181186d8eaec47839fb496e78838074d7957218bfa621f4c993",
             "supplement_independent_review_changed")
    review = json.loads(review_path.read_bytes())
    source_rows = {row["path"]: row["blake3"] for row in lineage["overlay_sources"]}
    _require(len(source_rows) == len(lineage["overlay_sources"]) == 4, "supplement_source_inventory_changed")
    for relative, digest in review["sources"].items():
        _require(file_digest(ROOT / relative) == digest and
                 (relative.endswith("scripts/score-qualified-native.py") or source_rows.get(relative) == digest),
                 "supplement_reviewed_source_changed")
    frozen = {row["path"]: row["blake3"] for row in record.harness["sources"]}
    for name in ("track_native_worker.py", "scorer_runtime.py"):
        relative = "evamed-codex/evaluation-overlays/native-receipt-v2/training/automedbench_lite/" + name
        _require(source_rows.get(relative) == file_digest(ROOT / relative) ==
                 frozen["EVA-Agent/training/automedbench_lite/" + name], "supplement_metric_source_changed")
    loader_path = ROOT / "evamed-codex/scripts/score-qualified-native.py"
    spec = importlib.util.spec_from_file_location("supplement_scorer_loader", loader_path)
    loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    audit = record.root / "track-rollouts" / track
    inventory = read_document(audit / "turns/01-e2e/after/manifest.json", maximum=16 * 1024**2)
    task = record.task(track)
    jobs = loader.load_scorer().completed_model_evidence(audit=audit,
        workspace=record.root / task["workspace_relative"], track=track,
        input_binding=task["input_manifest_file_blake3"],
        hosts=[json.loads(line) for line in (audit / "mcp-events.jsonl").read_bytes().splitlines()],
        visible_files={row["path"]: row for row in inventory["files"]})
    extensions = [{"job_id": job["job_id"], **job["host_only_journal_extension"]}
                  for job in jobs if job["host_only_journal_extension"]]
    _require(extensions and extensions == lineage["verified_public_receipt_journal_extensions"],
             "supplement_actual_journal_evidence_changed")
    return score_path, {"selection_document_blake3": selection["document_blake3"],
        "lineage_document_blake3": lineage["document_blake3"], "source_run_root": str(record.root),
        "original_scoring_failure": failure, "new_actor_rollouts": 0,
        "native_metric_math_changed": False, "public_receipt_extensions_reverified": len(extensions)}


def build_report(runs: list[Path], output: Path, *, native_score_overlay: Path | None = None):
    rows = {track: {"track": track, "status": "not_run", "task_score_0_100": None,
                    "case_count": spec.count} for track, spec in BY_TRACK.items()}
    identities = set()
    for track, (record, excluded, prepared) in select_tracks([ReportRun.open(run) for run in runs]).items():
        run, manifest = record.root, record.manifest
        actor_path = record.actor_path(track)
        row = rows[track]
        row.update(run_root=str(run), run_id=manifest["run_id"], status="prepared",
                   excluded_infrastructure_attempts=excluded, preparation_only_runs=prepared,
                   selected_attempt_number=2 if excluded else 1,
                   infrastructure_retries_used=1 if excluded else 0)
        if not actor_path.exists():
            continue
        actor = read_document(actor_path)
        if actor.get("score_admissible", True) is not True or actor.get("purpose") == "infrastructure_smoke":
            raise ValueError("diagnostic_run_cannot_enter_benchmark_report")
        identities.add((actor["requested_model"], actor["harness"], actor["runner"], actor["route"]))
        row.update(status="actor_completed_score_pending" if actor["completed_requested_turns"] else "actor_incomplete",
            actor_document_blake3=actor["document_blake3"], model=actor["requested_model"], harness=actor["harness"],
            runner=actor["runner"], route=actor["route"], model_requests=actor["actual_model_request_count"],
            elapsed_seconds=actor["wall_seconds"], errors=actor["errors"], budget_exhausted=actor.get("budget_exhausted"),
            terminal_disposition=actor.get("terminal_disposition"),
            infrastructure_reasons=infrastructure_reasons(record, track, actor))
        path = run / "native-scores" / track / "score.json"
        if native_score_overlay is not None:
            supplement = supplementary_native_score(record, track, actor, native_score_overlay)
            if supplement is not None:
                path, row["same_actor_scorer_compatibility"] = supplement
        if not path.exists():
            failure = path.with_name("failure.json")
            if failure.exists():
                row.update(status="scoring_failed", scoring_failure=read_document(failure))
            continue
        score = read_document(path, maximum=64 * 1024**2)
        disposition = require_scoring_admission(run, track, actor)
        value = score["native_result"]["task_score_0_1"]
        if (score["actor_document_blake3"] != actor["document_blake3"] or
                score["native_result"]["selected_case_count"] != row["case_count"] or
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("native_score_binding_invalid")
        row.update(status="native_scored", task_score_0_100=100 * value,
                   terminal_disposition=disposition,
                   native_score_document_blake3=score["document_blake3"])
    if len(identities) > 1:
        raise ValueError("mixed_model_harness_or_route_in_repeat1_report")
    complete = all(row["status"] == "native_scored" for row in rows.values())
    valid = [row["task_score_0_100"] for row in rows.values() if row["status"] == "native_scored"]
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = write_once(output / "report.json", {"schema": "eva.automedbench-codex-repeat1-report.v1",
        "report_id": str(uuid4()), "repeats": 1, "declared_tracks": 7, "scored_tracks": len(valid),
        "complete_seven_track_report": complete, "tracks": list(rows.values()),
        "seven_track_macro_0_100": sum(valid) / 7 if complete else None,
        "available_track_mean_0_100": sum(valid) / len(valid) if valid else None,
        "unknown_results_counted_as_zero": False, "official_five_repeat_leaderboard": False,
        "score_scope": "official_native_task_metric_math; separate_from_process_agent_judgments"})
    lines = ["# AutoMedBench Lite — Codex repeat 1", "",
             f"Native scores available: {len(valid)}/7 tracks. Unknown outcomes remain visible.", "",
             "| Track | Cases | Status | Task score /100 | Model requests | Seconds |",
             "| --- | ---: | --- | ---: | ---: | ---: |"]
    for row in rows.values():
        value = row["task_score_0_100"]
        score = f"{value:.1f}" if value is not None else "—"
        lines.append(f"| {row['track']} | {row['case_count']} | {row['status']} | {score} | "
                     f"{row.get('model_requests', '—')} | {round(row['elapsed_seconds'], 1) if 'elapsed_seconds' in row else '—'} |")
    lines += ["", "Seven-track macro: " + (f"{report['seven_track_macro_0_100']:.1f}/100" if complete else "unavailable until all seven tracks are scored"),
              "", "These are native task metrics. Opus stage-rubric judgments and reward-verifier agreement are separate artifacts.", ""]
    for row in rows.values():
        if row.get("infrastructure_reasons"):
            lines.append(f"{row['track']}: infrastructure-invalid and unscored. " + " ".join(row["infrastructure_reasons"]))
        if row.get("excluded_infrastructure_attempts"):
            lines.append(f"{row['track']}: selected attempt 2, after the single allowed infrastructure retry. "
                         "The failed first attempt and its verified evidence remain in report.json.")
    if any(row.get("same_actor_scorer_compatibility") for row in rows.values()):
        lines += ["", "Supplementary same-actor scores use the independently reviewed receipt-journal metadata compatibility overlay. "
                  "Original scorer failures remain in each row's lineage in report.json; native metric math and actor artifacts are unchanged. "
                  "No actor was rerun for this correction."]
    (output / "report.md").write_text("\n".join(lines))
    return report
