import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest

from training.eva_rsi.controller import Controller, read, write, start_identity
from training.eva_rsi.evidence import EvidenceError, commitment


def failed_training(tmp_path):
    config = {"schema": "eva.rsi-loop-config.v1", "rounds": 10, "updates_per_round": 50,
        "cwd": str(tmp_path), "architecture_model_path": "/architecture",
        "initial_model_path": "/initial/model", "initial_checkpoint_root": "/initial/checkpoint",
        "skill_catalog_id": "skills", "commands": {
            p: [sys.executable, "/runner.py", "{context}"] for p in ("baseline_eval", "train", "evaluation")}}
    write(tmp_path / "config.json", config)
    controller = Controller.initialize(tmp_path / "loop", tmp_path / "config.json")
    state, _ = controller.load()
    identity = str(uuid4())
    root = controller.root / "attempts" / identity
    state.update(phase="train", status="blocked", active_attempt=identity,
        model_path="/verified/prefix/hf", checkpoint_root="/verified/prefix/checkpoints",
        evaluation="/verified/baseline", errors=[{"code": "command_failed", "attempt_id": identity}],
        rounds=[{"round": 1, "s_target": "S3", "durable_updates": 1, "durable_learning_updates": 0,
                 "segments": ["/verified/prefix/checkpoint-verification.json"], "complete": False}])
    state["attempts"].append({"attempt_id": identity, "root": str(root), "phase": "train", "round": 1,
                               "status": "running", "exit": None})
    write(root / "context.json", controller.context(state, config, root))
    write(root / "request.json", {"attempt_id": identity, "phase": "train"})
    write(root / "worker.json", {"pid": os.getpid(), "start_identity": "0",
                                 "request": commitment(root / "request.json")})
    write(root / "exit.json", {"attempt_id": identity, "exit_code": 1, "error_type": None})
    write(root / "training/run-receipt.json", {"stage": "grpo", "status": "failed", "argv": [
        "train.py", "--load", state["checkpoint_root"], "--save", str(root / "training/checkpoints"),
        "--save-hf", str(root / "training/hf/iter_{rollout_id:07d}")]})
    write(root / "training/rollouts/retained-grade.json", {"retained": True, "reward": 0})
    controller.save(state)
    return controller, state, root


def test_failed_zero_update_retirement_preserves_prefix_and_all_attempt_bytes(tmp_path):
    c, before, root = failed_training(tmp_path)
    original = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    receipt = c.retire_failed_zero_update(reason="reviewed-code-boundary-failure", execute=True)
    after, config = c.load()
    assert after["active_attempt"] is None and after["status"] == "ready" and after["phase"] == "train"
    assert after["attempts"][-1]["status"] == "failed_zero_update"
    assert {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()} == original
    for key in ("round", "rounds", "model_path", "checkpoint_root", "evaluation", "errors", "pending_training"):
        assert after[key] == before[key]
    assert c.context(after, config, c.root / "attempts" / str(uuid4()))["remaining_updates"] == 49
    assert receipt["credited_updates"] == 0 and receipt["retry_launched"] is False
    assert c.status()["checkpoint_backed_updates"] == 1


@pytest.mark.parametrize("fault", ["step", "checkpoints", "hf", "checkpoint-proof", "malformed-event"])
def test_any_optimizer_step_or_checkpoint_blocks_zero_update_retirement(tmp_path, fault):
    c, before, root = failed_training(tmp_path)
    if fault in ("step", "malformed-event"):
        path = root / "training/optimizer-steps.jsonl"
        path.write_text(json.dumps({"event": "optimizer_step"}) if fault == "step" else "{partial")
    elif fault == "checkpoint-proof":
        write(root / "checkpoint-verification.json", {"pending": True})
    else:
        (root / "training" / fault).mkdir()
    with pytest.raises((EvidenceError, json.JSONDecodeError)):
        c.retire_failed_zero_update(reason="fixture", execute=True)
    assert read(c.root / "state.json") == before
    assert not (c.root / "recovery-records").exists()


@pytest.mark.parametrize("fault", ["live", "pending", "exit-zero", "receipt-running", "output-other", "context-other"])
def test_live_unbound_or_pending_training_cannot_be_retired(tmp_path, fault):
    c, before, root = failed_training(tmp_path)
    if fault == "live":
        doc = read(root / "worker.json"); doc["start_identity"] = start_identity(os.getpid()); write(root / "worker.json", doc)
    elif fault == "pending":
        before["pending_training"] = {"checkpoint": "pending"}; write(c.root / "state.json", before)
    elif fault == "exit-zero":
        doc = read(root / "exit.json"); doc["exit_code"] = 0; write(root / "exit.json", doc)
    elif fault in ("receipt-running", "output-other"):
        path = root / "training/run-receipt.json"; doc = read(path)
        if fault == "receipt-running": doc["status"] = "running"
        else: doc["argv"][doc["argv"].index("--save") + 1] = "/another/output"
        write(path, doc)
    else:
        doc = read(root / "context.json"); doc["model_path"] = "/different/model"; write(root / "context.json", doc)
    with pytest.raises(EvidenceError):
        c.retire_failed_zero_update(reason="fixture", execute=True)
    assert read(c.root / "state.json") == before
    assert not (c.root / "recovery-records").exists()


def test_cli_requires_explicit_execute_and_can_retire_only_the_temporary_fixture(tmp_path):
    c, before, _ = failed_training(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/run_eva_rsi_loop_v1.py"
    argv = [sys.executable, str(script), "retire-failed-zero-update", "--run-root", str(c.root), "--reason", "fixture"]
    result = subprocess.run(argv, text=True, capture_output=True)
    assert result.returncode != 0 and "execution_requires_explicit_flag" in result.stdout
    assert read(c.root / "state.json") == before
    result = subprocess.run([*argv, "--execute"], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["schema"] == "eva.rsi-failed-zero-update-retirement.v1"
    assert c.load()[0]["active_attempt"] is None
