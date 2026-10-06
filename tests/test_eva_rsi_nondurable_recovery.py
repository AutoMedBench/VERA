"""Explicit disposal of a failed unsaved prefix; no real model/GPU processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from training.eva_rsi.controller import read, write, start_identity
from training.eva_rsi.evidence import EvidenceError, commitment, optimizer_events
from test_eva_rsi_zero_update_recovery import failed_training


def nondurable(tmp_path):
    controller, state, root = failed_training(tmp_path)
    event = {"event": "optimizer_step", "rollout_id": 0, "step_id": 0,
             "trainable_parameters": 8_953_803_264, "successful_update": True,
             "gradient_norm": 1.8696848, "sampled_changed_values": 2045}
    source = root / "training/optimizer-steps.jsonl"
    source.write_text(json.dumps(event) + "\n")
    verified = {"run_receipt": commitment(root / "training/run-receipt.json"),
                "event_source": commitment(source), **optimizer_events(source), "launch_status": "failed",
                "checkpoint_root": str(root / "training/checkpoints"), "training_root": str(root / "training"),
                "exact_optimizer_resume": False}
    write(root / "training-verification.json", verified)
    state["attempts"][-1].update(status="verified_with_retained_command_failure", exit=read(root / "exit.json"),
                                  optimizer_executions=1, learning_updates_observed=1)
    state.update(active_attempt=None, phase="durable_checkpoint", pending_training={
        "attempt_root": str(root), "context": read(root / "context.json"), "training": verified})
    controller.save(state)
    return controller, state, root


def test_retirement_retains_unsaved_observation_and_previous_durable_prefix(tmp_path):
    c, before, root = nondurable(tmp_path)
    original = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    pending = c.status()["pending_training_observations_not_yet_checkpoint_credited"]
    assert pending["executions"] == pending["learning_updates"] == 1
    assert c.status()["active_training_observations_not_yet_checkpoint_credited"] is None
    receipt = c.retire_failed_nondurable(reason="unsaved-prefix-reviewed", execute=True)
    after, config = c.load()
    assert after["phase"] == "train" and after["status"] == "ready" and after["pending_training"] is None
    assert after["active_attempt"] is None and after["attempts"][-1]["status"] == "failed_nondurable"
    for key in ("round", "rounds", "model_path", "checkpoint_root", "evaluation", "errors", "skill_catalog_id"):
        assert after[key] == before[key]
    assert {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()} == original
    assert receipt["preserved_pending_training"] == before["pending_training"]
    assert receipt["observed_discarded_optimizer_executions"] == receipt["observed_discarded_nonzero_learning_updates"] == 1
    assert receipt["credited_updates"] == receipt["credited_learning_updates"] == 0
    assert receipt["retry_launched"] is False
    assert c.status()["checkpoint_backed_updates"] == 1 and c.status()["checkpoint_backed_learning_updates"] == 0
    assert c.context(after, config, root / "unused")["remaining_updates"] == 49


@pytest.mark.parametrize("artifact", ["checkpoints", "hf", "rollouts/unexpected.distcp",
    "rollouts/unexpected.safetensors", "rollouts/latest_checkpointed_iteration.txt", "../checkpoint-verification.json"])
def test_any_saved_output_requires_existing_checkpoint_verifier(tmp_path, artifact):
    c, before, root = nondurable(tmp_path)
    path = root / "training" / artifact
    if artifact in ("checkpoints", "hf"):
        path.mkdir()
    else:
        write(path, {"saved": True})
    with pytest.raises(EvidenceError, match="saved_checkpoint_requires_verifier"):
        c.retire_failed_nondurable(reason="fixture", execute=True)
    assert read(c.root / "state.json") == before and not (c.root / "recovery-records").exists()


@pytest.mark.parametrize("fault", ["live", "changed-event", "changed-receipt", "exit-zero",
                                   "inconsistent-counts", "changed-context", "active"])
def test_live_changed_nonfailed_or_inconsistent_pending_prefix_rejected(tmp_path, fault):
    c, before, root = nondurable(tmp_path)
    if fault == "live":
        doc = read(root / "worker.json"); doc["start_identity"] = start_identity(os.getpid()); write(root / "worker.json", doc)
    elif fault == "changed-event":
        with (root / "training/optimizer-steps.jsonl").open("a") as stream:
            stream.write("\n")
    elif fault == "changed-receipt":
        doc = read(root / "training/run-receipt.json"); doc["status"] = "running"; write(root / "training/run-receipt.json", doc)
    elif fault == "exit-zero":
        doc = read(root / "exit.json"); doc["exit_code"] = 0; write(root / "exit.json", doc)
    elif fault == "inconsistent-counts":
        before["pending_training"]["training"]["executions"] = 2
        write(root / "training-verification.json", before["pending_training"]["training"])
        c.save(before)
    elif fault == "changed-context":
        doc = read(root / "context.json"); doc["model_path"] = "/changed"; write(root / "context.json", doc)
    else:
        before["active_attempt"] = before["attempts"][-1]["attempt_id"]; c.save(before)
    with pytest.raises(EvidenceError):
        c.retire_failed_nondurable(reason="fixture", execute=True)
    assert read(c.root / "state.json") == before and not (c.root / "recovery-records").exists()


def test_cli_without_execute_is_readonly_and_fixture_execute_retires(tmp_path):
    c, before, _ = nondurable(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/run_eva_rsi_loop_v1.py"
    argv = [sys.executable, str(script), "retire-failed-nondurable", "--run-root", str(c.root), "--reason", "fixture"]
    rejected = subprocess.run(argv, capture_output=True, text=True)
    assert rejected.returncode != 0 and "execution_requires_explicit_flag" in rejected.stdout
    assert read(c.root / "state.json") == before
    accepted = subprocess.run([*argv, "--execute"], capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert json.loads(accepted.stdout)["schema"] == "eva.rsi-failed-nondurable-retirement.v1"
