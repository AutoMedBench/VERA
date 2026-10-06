from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest

from training.eva_rsi.controller import Controller, read, write, start_identity
from training.eva_rsi.evidence import EvidenceError, commitment


def setup(tmp_path):
    config = {"schema": "eva.rsi-loop-config.v1", "rounds": 10, "updates_per_round": 50,
        "cwd": str(tmp_path), "architecture_model_path": "/architecture",
        "initial_model_path": "/initial/model", "initial_checkpoint_root": "/initial/checkpoint",
        "skill_catalog_id": "original-skills", "commands": {
            p: [sys.executable, "/old/runner.py", "{context}"] for p in ("baseline_eval", "train", "evaluation")}}
    path = tmp_path / "original.json"
    write(path, config)
    return Controller.initialize(tmp_path / "loop", path), config


def ready_state(controller):
    state, _ = controller.load()
    state.update(phase="train", evaluation="/retained/baseline-verification.json", round=2,
        model_path="/updated/model", checkpoint_root="/updated/dcp",
        rounds=[{"round": 1, "durable_updates": 50, "durable_learning_updates": 48, "complete": True},
                {"round": 2, "durable_updates": 1, "durable_learning_updates": 1, "complete": False}],
        errors=[{"code": "retained_previous_failure"}])
    controller.save(state)
    return state


def test_reconfiguration_retains_history_counts_and_original_bytes(tmp_path):
    c, config = setup(tmp_path)
    before = ready_state(c)
    original = (c.root / "config.json").read_bytes()
    changed = deepcopy(config)
    changed["commands"]["train"][1] = "/new/runner.py"
    path = tmp_path / "new.json"
    write(path, changed)
    transition = c.reconfigure(path, reason="reviewed-stage45-runtime", execute=True)
    after, actual = c.load()
    assert actual == changed and (c.root / "config.json").read_bytes() == original
    assert after["config"]["path"].startswith(str(c.root / "config-revisions"))
    assert transition["previous_config"] == before["config"]
    assert {k:v for k,v in after.items() if k not in {"config", "config_transition"}} == {
        k:v for k,v in before.items() if k not in {"config", "config_transition"}}
    assert c.status()["checkpoint_backed_updates"] == 51


@pytest.mark.parametrize("change", ["noop", "identity", "running", "pending", "secret", "outside", "symlink"])
def test_unsafe_reconfigure_or_config_pointer_rejected(tmp_path, change):
    c, config = setup(tmp_path)
    state = ready_state(c)
    changed = deepcopy(config)
    if change != "noop":
        changed["commands"]["train"][1] = "/new/runner.py"
    if change == "identity": changed["skill_catalog_id"] = "different"
    if change == "running": state.update(active_attempt="active", status="running")
    if change == "pending": state["pending_training"] = {"proof": "pending"}
    if change == "secret": changed["commands"]["train"].append("sk-" + "x" * 30)
    path = tmp_path / "new.json"
    write(path, changed)
    if change == "outside": state["config"] = commitment(path)
    if change == "symlink":
        original = c.root / "config.json"
        original.unlink(); original.symlink_to(path)
    write(c.root / "state.json", state)
    with pytest.raises(EvidenceError):
        c.reconfigure(path, reason="fixture", execute=True)
    assert not (c.root / "config-revisions").exists()


def prelaunch(c):
    state = ready_state(c)
    identity = str(uuid4())
    root = c.root / "attempts" / identity
    write(root / "request.json", {"attempt_id": identity, "phase": "train"}, exclusive=True)
    write(root / "context.json", {"remaining_updates": 49, "s_target": "S3"}, exclusive=True)
    write(root / "exit.json", {"attempt_id": identity, "exit_code": 1, "error_type": None}, exclusive=True)
    write(root / "worker.json", {"pid": os.getpid(), "start_identity": "previous-process-birth",
        "request": commitment(root / "request.json")}, exclusive=True)
    state.update(active_attempt=identity, status="blocked")
    state["attempts"].append({"attempt_id": identity, "root": str(root), "phase": "train", "status": "running", "exit": None})
    write(c.root / "state.json", state)
    return state, root


def test_prelaunch_retirement_preserves_attempt_files_and_update_ledger(tmp_path):
    c, _ = setup(tmp_path)
    before, root = prelaunch(c)
    original = {p.name:p.read_bytes() for p in root.iterdir()}
    receipt = c.retire_prelaunch(reason="baseline-view-compatibility", execute=True)
    after, _ = c.load()
    assert after["active_attempt"] is None and after["status"] == "ready"
    assert after["attempts"][-1]["status"] == "failed_prelaunch"
    for key in ("round", "rounds", "evaluation", "model_path", "checkpoint_root", "skill_catalog_id", "errors"):
        assert after[key] == before[key]
    assert {p.name:p.read_bytes() for p in root.iterdir()} == original
    assert receipt["credited_updates"] == 0 and not receipt["retry_launched"]


@pytest.mark.parametrize("fault", ["training", "live", "zero", "wrong_id", "no_execute"])
def test_retirement_never_accepts_partial_training_or_live_attempt(tmp_path, fault):
    c, _ = setup(tmp_path)
    before, root = prelaunch(c)
    if fault == "training": (root / "training").mkdir()
    if fault == "live":
        owner = read(root / "worker.json"); owner["start_identity"] = start_identity(os.getpid()); write(root / "worker.json", owner)
    if fault in {"zero", "wrong_id"}:
        exit_value = read(root / "exit.json")
        exit_value["exit_code" if fault == "zero" else "attempt_id"] = 0 if fault == "zero" else str(uuid4())
        write(root / "exit.json", exit_value)
    with pytest.raises(EvidenceError):
        c.retire_prelaunch(reason="fixture", execute=fault != "no_execute")
    assert read(c.root / "state.json") == before


def test_cli_retirement_requires_explicit_execute_provider_free(tmp_path):
    c, _ = setup(tmp_path)
    before, _ = prelaunch(c)
    script = Path(__file__).resolve().parents[1] / "scripts/run_eva_rsi_loop_v1.py"
    result = subprocess.run([sys.executable, str(script), "retire-prelaunch", "--run-root", str(c.root),
        "--reason", "fixture"], text=True, capture_output=True)
    assert result.returncode != 0
    assert "execution_requires_explicit_flag" in result.stdout
    assert read(c.root / "state.json") == before
