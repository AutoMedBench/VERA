"""Synthetic CPU commands test orchestration only, never real learning/medical quality."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from eva_agent.rubrics import compile_registry
from eva_agent.training.coevolution import EvaluationStatus, RoundIdentity, VerifiedStageEvaluation, plan_stage_target
from training.eva_rsi.controller import Controller, read, write
from training.eva_rsi.evidence import EvidenceError, optimizer_events, require
from test_rubric_compiler import rubric, uid

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/eva_rsi_fake_command.py"


@pytest.fixture
def table():
    return compile_registry({"schema": "eva.rubric-registry.v1", "registry_id": uid(9), "registry_version": 1,
                             "rubrics": [rubric(1, domain="medical-research", stage="S1")]}).resolve("medical-research", "S1")


def setup(tmp_path, table, mode="normal", sleep=0):
    config = {"schema": "eva.rsi-loop-config.v1", "rounds": 10, "updates_per_round": 50,
              "cwd": str(ROOT), "architecture_model_path": str(tmp_path / "original-model"),
              "initial_model_path": str(tmp_path / "trained-hf"), "initial_checkpoint_root": str(tmp_path / "initial-dcp"),
              "skill_catalog_id": "fixture-catalog", "commands": {}}
    for phase in ("baseline_eval", "train", "evaluation"):
        config["commands"][phase] = [sys.executable, str(FIXTURE), "{context}", "--mode", mode, "--sleep", str(sleep)]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    base = Controller.initialize(tmp_path / "loop", path)

    def evaluation(path, context):
        document = read(path)
        require(document.get("synthetic_test_fixture") and document["valid"], "fixture_feedback_invalid")
        require(document["model_path"] == context["model_path"], "fixture_checkpoint_mismatch")
        identity = RoundIdentity("fixture-model", context["model_path"], context["skill_catalog_id"], "fixture-Judge")
        score = table.score({item["item_id"]: 0 for item in table.items})  # A real zero is valid evidence, not infrastructure.
        row = VerifiedStageEvaluation("fixture-case", identity, table, EvaluationStatus.SCORED, str(path), score)
        return {"proposal": plan_stage_target([row]), "evaluation_mode": "diagnostic_subset",
                "synthetic_test_fixture": True, "memory_context_pass": None}

    def checkpoint(training, context):
        proof = read(Path(training["training_root"]) / "checkpoint-fixture.json")
        require(proof["synthetic_test_fixture"], "fixture_checkpoint_required")
        n = proof["durable_updates"]
        return {**proof, "durable_learning_updates": sum(row["gradient_norm"] > 0 and row["sampled_changed_values"] > 0
                                                        for row in training["events"][:n]),
                "exact_optimizer_resume": False}

    return Controller(base.root, evaluation_verifier=evaluation, checkpoint_verifier=checkpoint)


def run_round(controller, transitions=9):
    return controller.run(execute=True, max_transitions=transitions, poll_seconds=.01)


def test_full_ten_rounds_resume_never_repeats_completed_rounds(tmp_path, table):
    controller = setup(tmp_path, table)
    first = run_round(controller)
    assert first["completed_rounds"] == 1 and first["checkpoint_backed_updates"] == 50
    original_attempts = read(controller.root / "state.json")["attempts"]
    final = controller.run(execute=True, poll_seconds=.01)
    assert final["status"] == "complete" and final["completed_rounds"] == 10
    assert final["checkpoint_backed_updates"] == final["observed_optimizer_executions"] == 500
    assert final["retained_attempts"] == 21
    state = read(controller.root / "state.json")
    assert state["attempts"][:3] == original_attempts
    assert all(row["skill_decision"]["action"] == "retain_catalog" for row in state["rounds"])
    assert controller.run(execute=True) == final


def test_readonly_status_and_explicit_execution_gate(tmp_path, table):
    controller = setup(tmp_path, table)
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in controller.root.rglob('*') if p.is_file()}
    assert controller.status()["checkpoint_backed_updates"] == 0
    with pytest.raises(EvidenceError, match="explicit"):
        controller.run()
    after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in controller.root.rglob('*') if p.is_file()}
    assert before == after


def test_duplicate_lines_do_not_inflate_fifty(tmp_path, table):
    controller = setup(tmp_path, table, mode="duplicates")
    result = run_round(controller)
    assert result["completed_rounds"] == 1 and result["observed_optimizer_executions"] == 50
    attempt = next(row for row in read(controller.root / "state.json")["attempts"] if row["phase"] == "train")
    assert read(Path(attempt["root"]) / "training-verification.json")["duplicate_events"] == 20


def test_partial_checkpoint_only_credits_prefix_and_remainder_one(tmp_path, table):
    controller = setup(tmp_path, table, mode="partial_checkpoint")
    result = run_round(controller, transitions=12)
    assert result["completed_rounds"] == 1 and result["checkpoint_backed_updates"] == 50
    assert result["observed_optimizer_executions"] == 51 and result["retained_errors"] == 1
    state = read(controller.root / "state.json")
    trains = [row for row in state["attempts"] if row["phase"] == "train"]
    assert [read(Path(row["root"]) / "context.json")["remaining_updates"] for row in trains] == [50, 1]
    assert trains[0]["exit"]["exit_code"] == 7 and len(state["rounds"][0]["segments"]) == 2


def test_zero_learning_is_separate_not_auto_overshot(tmp_path, table):
    controller = setup(tmp_path, table, mode="zero_learning")
    result = run_round(controller)
    assert result["checkpoint_backed_updates"] == 50 and result["checkpoint_backed_learning_updates"] == 0
    assert result["retained_attempts"] == 3 and result["memory_context_pass"] is None
    assert "three_consecutive_no_learning_updates" in result["warnings"]
    assert read(controller.root / "state.json")["rounds"][0]["learning_outcome"] == "inconclusive_learning_signal"


@pytest.mark.parametrize("mode", ["bad_evaluation", "missing_checkpoint"])
def test_launch_exit_zero_is_not_round_completion_or_automatic_retry(tmp_path, table, mode):
    controller = setup(tmp_path, table, mode=mode)
    result = controller.run(execute=True, poll_seconds=.01)
    assert result["status"] == "blocked" and result["completed_rounds"] == 0
    attempts = result["retained_attempts"]
    assert controller.run(execute=True)["retained_attempts"] == attempts


def test_stop_and_new_supervisor_reuses_detached_worker(tmp_path, table):
    controller = setup(tmp_path, table, sleep=.15)
    result = run_round(controller, transitions=1)
    assert result["active_attempt"] is not None
    write(controller.root / "stop-request.json", {"fixture_stop": True})
    assert controller.run(execute=True)["retained_attempts"] == 1
    controller.resume()
    result = run_round(controller, transitions=8)
    assert result["completed_rounds"] == 1 and result["retained_attempts"] == 3
    assert len(list(controller.root.glob('retained-stop-*.json'))) == 1
    for path in controller.root.glob('attempts/*/exit.json'):
        assert path.stat().st_mode & 0o777 == 0o600


def test_conflicting_event_identity_rejected(tmp_path):
    a = {"event": "optimizer_step", "rollout_id": 0, "step_id": 0, "successful_update": True,
         "gradient_norm": 1.0, "sampled_changed_values": 1, "trainable_parameters": 8953803264}
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(a) + '\n' + json.dumps({**a, "sampled_changed_values": 2}) + '\n')
    with pytest.raises(EvidenceError, match="conflicting"):
        optimizer_events(path)


def test_cli_status_is_provider_gpu_free(tmp_path, table):
    controller = setup(tmp_path, table)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_eva_rsi_loop_v1.py'), 'status',
                             '--run-root', str(controller.root)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["observed_optimizer_executions"] == 0
