"""CPU-only observer fixtures: no actor, native Judge, scorer or provider calls."""
import json
import os
from pathlib import Path
import threading
import time

import pytest

from training.automedbench_lite.adapter import read_document, write_once
from training.eva_rsi.evidence import commitment
from training.eva_rsi import retained_eval_report as observer


def raw(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    run = root / "benchmark/run"
    run.mkdir(parents=True)
    harness = Path(os.environ["EVA_HARNESS_ROOT"])
    monkeypatch.setenv("EVA_SLIME_JUDGE_CONCURRENCY", "4")
    settings = tmp_path / "settings.json"
    raw(settings, {"evaluation_mode": "full_single_pass", "baseline_gate_mode": "terminal-attempts-v1",
        "native_judge_timeout_seconds": 600, "judge_material_view": "policy-visible-audit-v2",
        "postround_judge_verdict_replacements": 2, "allow_context_budget_terminal": True})
    identity = root / "serving/checkpoint-identity.json"
    raw(identity, {"exact_final_model_path": "/fixture/model"})
    raw(root / "context.json", {"phase": "evaluation", "attempt_root": str(root), "model_path": "/fixture/model"})
    raw(root / "request.json", {"attempt_id": root.name, "phase": "evaluation", "argv": ["--settings", str(settings)]})
    raw(root / "worker.json", {"pid": os.getpid(), "start_identity": "1", "request": commitment(root / "request.json")})
    raw(root / "evaluation-actor-binding.json", {"schema": "eva.rsi-evaluation-actor-binding.v1",
        "codex_version": "codex-cli 0.153.4", "model_path": "/fixture/model",
        "checkpoint_identity": commitment(identity), "equivalent_actor_cli": ["--run-root", str(run)],
        "selected_harness_root": str(harness)})
    (run / "track-rollouts").mkdir()
    write_once(run / "track-rollouts/attempt.json", {"server_binding": {
        "identity_file_blake3": commitment(identity)["blake3"]}, "tracks": list(observer.BY_TRACK),
        "one_attempt_per_track": True})
    write_once(run / "track-run-manifest.json", {"fixture_only": True})
    return root


def mock_report(monkeypatch):
    monkeypatch.setattr("training.automedbench_lite.evaluation_report.build_report",
        lambda run, index: {"fixture_only": True, "index": str(index)})
    monkeypatch.setattr("training.automedbench_lite.evaluation_report.render_markdown", lambda report: "fixture only\n")


def ready_prefixes(monkeypatch, count=6):
    selected = [(track, stage) for track in observer.BY_TRACK for stage in observer.PHASES][:count]
    monkeypatch.setattr(observer, "stage_readiness", lambda bound, track, stage, exited:
        {"status": "ready", "source_references": [], "score": None} if (track, stage) in selected
        else {"status": "unavailable" if exited else "pending", "score": None, "reason": "fixture_missing"})
    return selected


def test_source_binding_rejects_changed_checkpoint_and_output_inside_source(source, tmp_path):
    bound = observer.bind_source(source)
    assert bound["run_root"] == str(source / "benchmark/run")
    with pytest.raises(ValueError, match="output_must_be_separate"):
        observer.observe(source, source / "forbidden")
    raw(source / "serving/checkpoint-identity.json", {"exact_final_model_path": "/other"})
    with pytest.raises(ValueError, match="checkpoint_differs"):
        observer.bind_source(source)


def test_default_materializes_without_provider_or_source_writes(source, tmp_path, monkeypatch):
    selected = ready_prefixes(monkeypatch, 2)
    monkeypatch.setattr(observer, "source_exited", lambda bound: False)
    mock_report(monkeypatch)
    def prepare(**kwargs):
        target = kwargs["output_root"]
        target.mkdir(parents=True)
        write_once(target / "preflight.json", {"judge_eligible": True})
        assert kwargs["material_view"] == "policy-visible-audit-v2"
        return {}, {}, {"judge_eligible": True}
    monkeypatch.setattr("training.automedbench_lite.track_feedback.prepare_track_feedback", prepare)
    monkeypatch.setattr("training.automedbench_lite.track_feedback.evaluate_track_feedback",
        lambda *a, **k: pytest.fail("default cannot dispatch Judge"))
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "preflight"
    result = observer.observe(source, output, poll_seconds=1)
    assert result["logical_stage_dispatches"] == 0 and result["verified_stage_count"] == 0
    assert all(path.read_bytes() == value for path, value in before.items())
    assert len(list(output.glob("stages/*/*/claim.json"))) == len(selected)
    with pytest.raises(FileExistsError):
        observer.observe(source, output)
    index = observer._read(output / "evaluation-index.json")
    assert index["evaluation_mode"] == "diagnostic_subset" and index["new_rollouts_per_track"] == 0
    assert not index["full_seven_track_evaluation"]


def test_four_slots_once_only_and_failure_releases_slot(source, tmp_path, monkeypatch):
    selected = ready_prefixes(monkeypatch)
    monkeypatch.setattr(observer, "source_exited", lambda bound: True)
    raw(source / "exit.json", {"attempt_id": source.name, "exit_code": 1, "error_type": None})
    raw(source / "serving/session-exit.json", {"cleanup": {"verified_members_remaining": 0}})
    mock_report(monkeypatch)
    lock, calls, state = threading.Lock(), [], {"active": 0, "peak": 0}
    def evaluate(run, identity, output, **kwargs):
        key = (kwargs["track"], kwargs["stages"][0])
        assert kwargs["native_turn_timeout_seconds"] == 600
        assert kwargs["postround_judge_verdict_replacements"] == 2
        assert kwargs["context_terminal_source_binding"]["model_path"] == "/fixture/model"
        with lock:
            calls.append(key); state["active"] += 1; state["peak"] = max(state["peak"], state["active"])
        try:
            time.sleep(.05)
            if key == selected[1]:
                raise ValueError("synthetic invalid preflight; must not retry")
            root = output / kwargs["stages"][0]
            raw(root / "verification.json", {"valid": True, "status": "scored", "case_id": key[0],
                "stage": key[1], "score": {"reward_bps": 0}})
            raw(root / "postround-judge-attempt-binding.json", {"synthetic_fixture": True})
            return (root,)
        finally:
            with lock: state["active"] -= 1
    monkeypatch.setattr("training.automedbench_lite.track_feedback.evaluate_track_feedback", evaluate)
    monkeypatch.setattr("training.benchmark_feedback.automed_codex.verify_feedback",
        lambda root: observer._read(root / "verification.json"))
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "executed-fixture"
    result = observer.observe(source, output, execute=True, poll_seconds=1)
    assert sorted(calls) == sorted(selected) and state == {"active": 0, "peak": 4}
    assert result["logical_stage_dispatches"] == 6 and result["verified_stage_count"] == 5
    progress = read_document(output / "progress.json")
    assert progress["counts"] == {"pending": 0, "running": 0, "verified": 5, "unavailable": 30}
    for row in progress["stages"]:
        assert row["score_0_100"] == (0 if row["status"] == "verified" else None)
        assert (row["verification"] is not None) == (row["status"] == "verified")
    assert all(path.read_bytes() == value for path, value in before.items())


def test_partial_snapshot_json_pending_until_source_exit(source):
    bound = observer.bind_source(source)
    target = Path(bound["run_root"]) / "track-rollouts/classification/turns/01-planning"
    target.mkdir(parents=True)
    (target / "request.json").write_text('{"unfinished":')
    assert observer.stage_readiness(bound, "classification", "S1", exited=False)["status"] == "pending"
    result = observer.stage_readiness(bound, "classification", "S1", exited=True)
    assert result["status"] == "unavailable" and result["score"] is None


def test_live_dispatch_requires_sealed_zero_turn_gate_block(source):
    bound = observer.bind_source(source)
    assert observer._live_gate_blocked(bound) is None
    audit = Path(bound["run_root"]) / "track-rollouts/classification"
    audit.mkdir()
    budget = write_once(audit / "track-budget.json", {"fixture_only": True})
    write_once(audit / "track-budget-outcome.json", {"schema": "eva.automedbench-track-budget-outcome.v1",
        "budget_document_blake3": budget["document_blake3"], "actual_turn_count": 0, "app_server_pids": []})
    assert observer._live_gate_blocked(bound) == commitment(audit / "track-budget-outcome.json")
    unavailable = observer.stage_readiness(bound, "classification", "S1", exited=False)
    assert unavailable["status"] == "unavailable" and unavailable["actual_turn_count"] == 0
    assert unavailable["score"] is None
    raw(audit / "turns/01-planning/receipt.json", {})
    assert observer._live_gate_blocked(bound) is None


def test_task_audit_only_after_exit_only_s5_and_original_native_source(source, tmp_path, monkeypatch):
    bound = observer.bind_source(source)
    bound["settings"]["task_agent_score_audit"] = True
    native = source / "native-task-scores/summary.json"
    raw(native, {"schema": "eva.rsi-native-task-score-summary.v1", "benchmark_run_root": bound["run_root"],
                 "checkpoint_identity": bound["checkpoint_identity"]})
    calls = []
    def audit(roots, path, output, **kwargs):
        calls.append((roots, path)); return []
    monkeypatch.setattr("training.eva_rsi.task_score_audit.audit_native_scores", audit)
    outcomes = [{"status": "verified", "stage": stage, "feedback_root": stage} for stage in ("S1", "S5")]
    index = {}
    observer._attach_task_evidence(bound, tmp_path / "out", outcomes, index, native, execute=True, exited=False)
    assert calls == [] and index == {}
    observer._attach_task_evidence(bound, tmp_path / "out", outcomes, index, native, execute=True, exited=True)
    assert calls == [(["S5"], native)]
    assert index["native_task_scores"] == commitment(native)


def test_source_exit_requires_bound_os_wait_and_worker_absence(source):
    bound = observer.bind_source(source)
    assert observer.source_exited(bound) is False
    raw(source / "exit.json", {"attempt_id": source.name, "exit_code": 1, "error_type": None})
    birth = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
    owner = {"pid": os.getpid(), "start_identity": birth, "request": commitment(source / "request.json")}
    raw(source / "worker.json", owner)
    assert observer.source_exited(bound) is False
    raw(source / "worker.json", {**owner, "start_identity": "0"})  # Synthetic PID reuse, no process killed.
    raw(source / "serving/session-exit.json", {"cleanup": {"verified_members_remaining": 0}})
    assert observer.source_exited(bound) is True
    raw(source / "exit.json", {"attempt_id": "different", "exit_code": 1, "error_type": None})
    with pytest.raises(ValueError, match="exit_binding_differs"):
        observer.source_exited(bound)


def test_failure_categories_never_echo_private_exception_text():
    assert observer._failure_category(ValueError("document_topology_or_size_invalid")) == "document_topology_or_size_invalid"
    assert observer._failure_category(ValueError("private-fixture-not-to-copy")) == "retained_stage_validation_or_judge_unavailable"
