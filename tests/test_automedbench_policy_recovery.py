import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.automedbench_lite import policy_recovery
from training.automedbench_lite.adapter import EvaluationError


def test_matching_process_birth_is_not_gone():
    fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    identity = {"pid": os.getpid(), "start_ticks": fields[19]}
    assert not policy_recovery._birth_is_gone(identity)
    assert policy_recovery._birth_is_gone({**identity, "start_ticks": "different-birth"})


def test_stable_snapshot_keeps_original_bytes_outside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    audit = tmp_path / "recovery"
    workspace.mkdir()
    (workspace / "inputs-manifest.json").write_text(json.dumps({"files": []}))
    # MutableInventory requires the normal committed input manifest.
    from training.automedbench_lite.adapter import write_once
    (workspace / "inputs-manifest.json").unlink()
    write_once(workspace / "inputs-manifest.json", {"files": []})
    (workspace / "task.json").write_text("fixture-task")
    monkeypatch.setattr(policy_recovery.time, "sleep", lambda _: None)
    observations, stable = policy_recovery._stable_snapshot(workspace, audit, 1.0)
    assert observations["first"]["file_count"] == 2
    assert stable["file_count"] == 2
    assert (audit / "after/files/task.json").read_text() == "fixture-task"
    assert not (workspace / "workspace-blobs").exists()


def test_stability_rejects_a_workspace_change(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    from training.automedbench_lite.adapter import write_once
    write_once(workspace / "inputs-manifest.json", {"files": []})
    target = workspace / "task.json"
    target.write_text("before")
    monkeypatch.setattr(policy_recovery.time, "sleep", lambda _: target.write_text("after"))
    with pytest.raises(EvaluationError, match="workspace_not_stable"):
        policy_recovery._stable_snapshot(workspace, tmp_path / "recovery", 1.0)


def test_output_inside_original_run_is_rejected(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(EvaluationError, match="outside_original"):
        policy_recovery._outside_original(run / "new-sidecar", run)


def test_builder_and_verifier_bind_current_workspace_without_provider(tmp_path, monkeypatch):
    from training.automedbench_lite.adapter import write_once
    run, workspace, audit, turn = (tmp_path / name for name in ("run", "workspace", "audit", "turn"))
    for path in (run, workspace, audit, turn):
        path.mkdir()
    write_once(workspace / "inputs-manifest.json", {"files": []})
    (workspace / "task.json").write_text("stable-public-task")
    paths = {"run": run, "workspace": workspace, "audit": audit, "turn": turn}
    owner_path = tmp_path / "owner.json"
    owner = write_once(owner_path, {"schema": policy_recovery.OWNERSHIP_SCHEMA})
    source = {"receipt": SimpleNamespace(tool_calls=(1, 2, 3)),
        "joined": {"joined_host_call_count": 2, "native_control_call_count": 1,
                   "native_control_operations": ["list_mcp_resources"]},
        "before": SimpleNamespace(tree_blake3="before-tree"),
        "before_manifest": {"immutable_inputs_manifest_blake3":
            policy_recovery.file_digest(workspace / "inputs-manifest.json")},
        "last_host_workspace": None,
        "archival_failure": {"failure_error": "AttributeError",
                             "terminal_infrastructure_error": "AttributeError"},
        "source_commitments": {"receipt_blake3": "receipt-binding"}}
    monkeypatch.setattr(policy_recovery, "_run_paths", lambda *_: paths)
    monkeypatch.setattr(policy_recovery, "_ownership", lambda *_: owner)
    monkeypatch.setattr(policy_recovery, "_source_evidence", lambda *_: source)
    with policy_recovery.tempfile.TemporaryDirectory() as temporary:
        source["last_host_workspace"] = policy_recovery._stable_core(
            policy_recovery.MutableInventory(workspace, Path(temporary)).capture())
    recovery = tmp_path / "recovery"
    result = policy_recovery.build_recovery(run_root=run, track="detection", phase="01-planning",
        ownership_path=owner_path, output=recovery, stability_seconds=0)
    assert (recovery / "ownership.json").read_bytes() == owner_path.read_bytes()
    assert result["provider_replayed"] is False and result["turn_retried"] is False
    policy_recovery.verify_recovery(run_root=run, track="detection", phase="01-planning",
        ownership_path=recovery / "ownership.json", recovery_root=recovery)
    (workspace / "task.json").write_text("changed-after-recovery")
    with pytest.raises(EvaluationError, match="stability_binding_invalid"):
        policy_recovery.verify_recovery(run_root=run, track="detection", phase="01-planning",
            ownership_path=recovery / "ownership.json", recovery_root=recovery)


def test_post_exit_observation_is_explicitly_not_historical_birth_proof(tmp_path, monkeypatch):
    from training.automedbench_lite.adapter import write_once
    run, workspace, audit, turn = (tmp_path / name for name in ("run", "workspace", "audit", "turn"))
    for path in (run, workspace, audit, turn):
        path.mkdir()
    write_once(workspace / "inputs-manifest.json", {"files": []})
    (workspace / "task.json").write_text("stable-public-task")
    baseline = write_once(run / "baseline-process.json", {"actor_pid": 1_000_000_001, "command": []})
    exit_receipt = write_once(run / "baseline-process-exit.json", {"actor_pid": 1_000_000_001,
        "returncode": 0, "os_process_exit_observed": True})
    thread = write_once(audit / "thread.json", {"app_server_pid": 1_000_000_002})
    paths = {"run": run, "workspace": workspace, "audit": audit, "turn": turn}
    with policy_recovery.tempfile.TemporaryDirectory() as temporary:
        current = policy_recovery._stable_core(
            policy_recovery.MutableInventory(workspace, Path(temporary)).capture())
    source = {"receipt": SimpleNamespace(receipt_blake3="receipt-binding"),
              "last_host_workspace": current}
    monkeypatch.setattr(policy_recovery, "_run_paths", lambda *_: paths)
    monkeypatch.setattr(policy_recovery, "_source_evidence", lambda *_: source)
    monkeypatch.setattr(policy_recovery, "_matching_track_mcp_pids", lambda *_: [])
    value = policy_recovery.observe_post_exit_ownership(run_root=run, track="vqa",
        phase="03-smoke", output=tmp_path / "post-exit.json", stability_seconds=0)
    assert value["baseline_process_document_blake3"] == baseline["document_blake3"]
    assert value["baseline_process_exit_document_blake3"] == exit_receipt["document_blake3"]
    assert value["thread_document_blake3"] == thread["document_blake3"]
    assert value["historical_pid_birth_capture_available"] is False
    assert value["historical_process_quiescence_claimed"] is False
    assert value["matching_track_mcp_process_count"] == 0
