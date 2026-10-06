"""Synthetic CPU-only report fixtures; none are benchmark or Judge results."""
import copy
import json

import pytest

from training.automedbench_lite import evaluation_report as report
from training.automedbench_lite.adapter import write_once
from training.eva_rsi.evidence import commitment, EvidenceError


def put(path, value, *, committed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if committed:
        return write_once(path, value)
    path.write_text(json.dumps(value))
    return value


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    run = tmp_path / "run"
    attempt = put(run / "track-rollouts/attempt.json", {"schema": "fixture"}, committed=True)
    identity = tmp_path / "identity.json"
    put(identity, {"exact_final_model_path": str(tmp_path / "model")})
    index = {"schema": "eva.rsi-evaluation-index.v1", "benchmark_run_root": str(run),
        "checkpoint_identity": str(identity), "feedback_roots": [], "matched_codex_version_requested": "0.153.4"}
    values = {}
    def stage(name, stage, bps, root=None):
        root = root or str(tmp_path / (name + stage))
        values[root] = {"case_id": name, "stage": stage, "valid": True, "status": "scored",
            "checkpoint_identity_blake3": commitment(identity)["blake3"],
            "skill_source_binding": {"run_attempt_document_blake3": attempt["document_blake3"]},
            "score": {"reward_bps": bps}, "rubric_digest": "fixture-rubric",
            "judge_codex_receipt_blake3": "fixture-receipt", "actual_workspace_reads": 2,
            "round_identity": {"judge_id": "fixture-judge"}}
        index["feedback_roots"].append(root)
        return values[root]
    monkeypatch.setattr(report, "_feedback", lambda root, family: values[root])
    def build():
        path = tmp_path / "index.json"
        put(path, index)
        return report.build_report(run, path)
    return run, index, stage, build


def track(result, name="classification"):
    return next(row for row in result["tracks"] if row["track"] == name)


def native(fixture, value=0, *, schema="eva.automedbench-native-track-result.v1"):
    run, index, _, _ = fixture
    actor = put(run / "track-rollouts/classification/rollout.json", {
        "completed_requested_turns": True, "errors": []}, committed=True)
    native_value = {"schema": schema, "status": "scored", "track": "classification",
        "task_score_0_1": value, "selected_case_count": 100, "full_public_subset": True,
        "native_facts": {"submission_format_valid": False}, "agent_judged": False}
    path = run.parent / "native/classification/score.json"
    put(path, {"schema": "eva.automedbench-native-whole-track-score.v1", "track": "classification",
        "native_result": native_value, "actor_document_blake3": actor["document_blake3"]}, committed=True)
    put(path.parent / "attempt.json", {"source_run_root": str(run),
        "actor_document_blake3": actor["document_blake3"]}, committed=True)
    summary = run.parent / "native/summary.json"
    put(summary, {"schema": "eva.rsi-native-task-score-summary.v1", "benchmark_run_root": str(run),
        "checkpoint_identity": commitment(index["checkpoint_identity"]), "tracks": [{
            "track": "classification", "status": "scored", "task_score_0_1": value,
            "native_result": native_value, "native_score": commitment(path)}]})
    index["native_task_scores"] = commitment(summary)
    return path


def test_partial_A_with_native_zero_is_numeric_and_provisional(fixture):
    _, _, stage, build = fixture
    stage("classification", "S1", 10000)
    stage("classification", "S3", 0)
    native(fixture, 0)
    result = build()
    row = track(result)
    assert (row["A"], row["T"], row["A_plus_T"], row["mean_A_T"]) == (50, 0, 50, 25)
    assert row["A_stage_coverage"] == 2 and row["A_provisional"] and row["overall_provisional"]
    assert not row["all_five_A_stages_verified"]
    assert row["task"]["task_judge_audit_status"] == "not_performed"
    assert row["task"]["raw_value"] == 0 and row["task"]["raw_scale"] == [0, 1]
    assert all(s["T"] is None for s in row["stages"].values())
    assert result["verified_process_stage_count"] == 2 and result["unavailable_process_stage_count"] == 33
    assert len(result["tracks"]) == 7 and track(result, "vqa")["mean_A_T"] is None
    text = report.render_markdown(result)
    assert "2/5 provisional" in text and "0.00" in text and "not a workspace-Judge verdict" in text


def test_complete_A_does_not_require_perfect_scores_or_fill_missing_T(fixture):
    _, _, stage, build = fixture
    for s in report.STAGES:
        stage("classification", s, 1667)
    row = track(build())
    assert row["A"] == pytest.approx(16.67)
    assert row["all_five_A_stages_verified"] and not row["A_provisional"]
    assert row["A_plus_T"] is None and row["mean_A_T"] is None


@pytest.mark.parametrize("value,schema", [(10, "eva.automedbench-native-track-result.v1"),
    (0.8, "unrecognized-normalization")])
def test_unknown_or_invalid_native_scale_is_unavailable_not_zero(fixture, value, schema):
    native(fixture, value, schema=schema)
    assert track(fixture[3]())["T"] is None


def test_native_and_process_source_mismatches_never_borrow_score(fixture):
    _, _, stage, build = fixture
    stage("classification", "S1", 10000)["skill_source_binding"]["run_attempt_document_blake3"] = "wrong-run"
    path = native(fixture, .9)
    path.write_text(path.read_text() + " ")
    result = build()
    assert track(result)["A"] is None and track(result)["T"] is None
    assert result["feedback_failures"][0]["status"] == "unavailable"


def test_duplicate_root_dedup_but_two_stage_roots_are_not_score_selected(fixture):
    _, index, stage, build = fixture
    stage("classification", "S1", 10000)
    index["feedback_roots"] *= 2
    assert track(build())["A_stage_coverage"] == 1
    stage("classification", "S1", 0, index["feedback_roots"][0] + "-other")
    row = track(build())
    assert row["A"] is None and row["stages"]["S1"]["status"] == "conflicting_feedback"


def receipt(stage, inp, out, *, thread="thread1", resumed=False, duration=1500):
    number = report.STAGES.index(stage)
    raw = {"thread_id": thread, "turn_id": str(number), "receipt_blake3": "digest"+str(number),
        "status": "completed", "thread_resumed": resumed, "sdk_version": "fixture-sdk",
        "server_version": "fixture-codex", "usage": {"total": {"inputTokens": inp, "outputTokens": out},
            "last": {"inputTokens": 999999, "outputTokens": 999999}},
        "tool_calls": [{"tool_call_id": str(number), "receipt_blake3": str(number)}],
        "events": [{"method": "turn/completed", "payload": {"turn": {"id": str(number),
            "startedAt": 100+number*2, "completedAt": 102+number*2, "durationMs": duration}}}]}
    return {"stage": stage, "receipt": raw, "source": {"path": "fixture", "blake3": raw["receipt_blake3"]}}


def test_cumulative_sdk_receipts_and_duplicate_prefixes_count_once():
    first, second = receipt("S1", 100, 10), receipt("S2", 250, 30, resumed=True)
    result = report.runtime_usage([first, second, copy.deepcopy(first)])
    assert (result["turns"], result["tool_calls"], result["input_tokens"], result["output_tokens"]) == (2, 2, 250, 30)
    assert result["receipt_details"][1]["input_tokens"] == 150
    assert result["sum_turn_seconds"] == 3 and result["wall_span_seconds"] == 4
    other = receipt("S1", 40, 4, thread="newthread")
    assert report.runtime_usage([first, second, other])["input_tokens"] == 290


@pytest.mark.parametrize("rows", [
    [receipt("S2", 100, 10, resumed=True)],
    [receipt("S1", 100, 10), receipt("S3", 500, 50, resumed=True)],
    [receipt("S1", 100, 10), receipt("S2", 50, 15, resumed=True)],
])
def test_missing_thread_origin_gap_or_decreasing_total_is_not_invented(rows):
    result = report.runtime_usage(rows)
    assert result["input_tokens"] is None and result["output_tokens"] is None


def test_missing_usage_time_and_conflicting_turns():
    value = receipt("S1", None, None, duration=None)
    value["receipt"]["events"] = []
    result = report.runtime_usage([value])
    assert result["input_tokens"] is None and result["sum_turn_seconds"] is None and result["wall_span_seconds"] is None
    other = copy.deepcopy(value)
    other["receipt"]["receipt_blake3"] = "conflict"
    with pytest.raises(EvidenceError, match="report_duplicate_turn_differs"):
        report.runtime_usage([value, other])


def test_report_uses_actor_receipts_not_repeated_judge_prefixes(fixture, monkeypatch):
    run, _, stage, build = fixture
    stage("classification", "S1", 10000)
    stage("classification", "S2", 0)
    originals = {}
    for name, value in [("01-planning", receipt("S1", 100, 10)), ("02-setup", receipt("S2", 220, 25, resumed=True))]:
        path = run / "track-rollouts/classification/turns" / name / "receipt.json"
        put(path, value["receipt"])
        originals[str(path)] = path.read_bytes()
    monkeypatch.setattr(report, "_receipt", lambda path: json.loads(path.read_text()))
    result = build()
    assert track(result)["runtime"]["input_tokens"] == 220
    assert track(result)["runtime"]["output_tokens"] == 25
    assert all(__import__('pathlib').Path(path).read_bytes() == body for path, body in originals.items())


def test_index_run_mismatch_rejected(fixture):
    _, index, _, build = fixture
    index["benchmark_run_root"] += "-different"
    with pytest.raises(EvidenceError, match="report_index_run_differs"):
        build()


@pytest.mark.parametrize("consistent", [True, False])
def test_task_audit_reopened_separately_never_changes_A_or_T(fixture, monkeypatch, consistent):
    run, index, stage, build = fixture
    stage("classification", "S1", 3333)
    native(fixture, .25)
    summary = json.loads(__import__('pathlib').Path(index["native_task_scores"]["path"]).read_text())
    native_path = run.parent / "native/classification-result.json"
    put(native_path, summary["tracks"][0])
    audit_path = run.parent / "audit/result.json"
    value = {"schema": "eva.automedbench-task-score-audit.v1", "track": "classification", "status": "verified",
        "agent_audit_verified": True, "task_score_source": "native_evaluator", "native_result_consistent": consistent,
        "task_score_0_100": 25 if consistent else None, "native_result": commitment(native_path),
        "evidence_refs": ["workspace:after:outputs/agents_outputs/predictions.csv"],
        "judge_receipt_blake3": "fixture-audit-receipt"}
    put(audit_path, value)
    index["task_agent_audits"] = [{"track": "classification", "result": commitment(audit_path)}]
    calls = []
    def verify(root):
        calls.append(root)
        return value
    monkeypatch.setattr(report, "_task_audit", verify)
    row = track(build())
    assert row["A"] == 33.33 and row["T"] == 25
    assert row["task"]["task_judge_audit_status"] == ("verified_consistent" if consistent else "verified_inconsistent")
    assert calls == [audit_path.parent]
    monkeypatch.setattr(report, "_task_audit", lambda root: {**value, "agent_audit_verified": False})
    row = track(build())
    assert row["task"]["task_judge_audit_status"] == "verification_failed" and row["T"] == 25


def test_invalid_native_summary_does_not_erase_verified_A(fixture):
    _, index, stage, build = fixture
    stage("classification", "S1", 10000)
    native(fixture, .5)
    index["native_task_scores"]["blake3"] = "invalid"
    row = track(build())
    assert row["A"] == 100 and row["T"] is None
    assert row["task"]["reason"] == "native_summary_verification_failed"


@pytest.mark.parametrize("elapsed", [150.0, 3605.0])
def test_track_elapsed_prefers_actual_bound_outcome_including_cleanup(fixture, elapsed):
    run, _, _, build = fixture
    root = run / "track-rollouts/classification"
    budget = put(root / "track-budget.json", {"schema": "eva.automedbench-track-budget.v1",
        "policy": "admitted-track-wallclock-3600-v1", "timeout_seconds": 3600,
        "queue_before_admission_charged": False, "job_waits_and_runtime_restart_charged": True}, committed=True)
    outcome = put(root / "track-budget-outcome.json", {"schema": "eva.automedbench-track-budget-outcome.v1",
        "policy": budget["policy"], "timeout_seconds": 3600, "budget_document_blake3": budget["document_blake3"],
        "elapsed_seconds": elapsed, "remaining_seconds": max(0, 3600-elapsed),
        "deadline_exhausted": elapsed >= 3600, "cleanup_overrun_seconds": max(0, elapsed-3600),
        "actual_turn_count": 0, "app_server_pids": [123]}, committed=True)
    put(root / "rollout.json", {"completed_requested_turns": False, "errors": [],
        "actual_turn_count": 0, "app_server_pids": [123], "track_budget": outcome}, committed=True)
    runtime = track(build())["runtime"]
    assert runtime["reported_time_seconds"] == elapsed and runtime["sum_turn_seconds"] is None
    assert runtime["reported_time_source"] == "admitted_track_elapsed"
    assert runtime["cleanup_overrun_seconds"] == max(0, elapsed-3600)
    # Existing outer document remains parseable; the outcome's exact bytes changed.
    (root / "track-budget-outcome.json").write_text('{"schema":"changed"}')
    runtime = track(build())["runtime"]
    assert runtime["track_elapsed_status"] == "verification_failed" and runtime["reported_time_seconds"] is None
