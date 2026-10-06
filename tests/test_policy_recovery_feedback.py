"""Current-time recovery on public CPU fixtures, never an actual-run repair."""
import json
from pathlib import Path
import shutil

import pytest

from training.automedbench_lite import policy_recovery, track_feedback
from training.automedbench_lite.adapter import read_document, write_once
from test_automedbench_track_feedback import fixture, synthetic_budget_binding


def recovery_fixture(tmp_path, monkeypatch):
    args = fixture(tmp_path, interrupted=True)
    target = args["run_root"] / "track-rollouts/classification/turns/01-planning"
    terminal = synthetic_budget_binding(args)
    (target / "policy-budget-terminal.json").unlink()
    write_once(target / "policy-budget-terminal.json", {**{k:v for k,v in terminal.items() if k!="document_blake3"},
        "workspace_quiescence_verified": False, "infrastructure_error": "AttributeError"})
    write_once(target / "failure.json", {"phase": target.name, "error": "AttributeError", "reward": None})
    # Preserve the old fixture snapshot elsewhere; do not pretend it survived capture.
    shutil.move(target / "after", tmp_path / "unused-fixture-after")
    ownership = tmp_path / "ownership.json"
    owner = write_once(ownership, {"schema": policy_recovery.OWNERSHIP_SCHEMA, "fixture_only": True})
    monkeypatch.setattr(policy_recovery, "_ownership", lambda *_: owner)
    recovery = tmp_path / "current-recovery"
    policy_recovery.build_recovery(run_root=args["run_root"], track="classification", phase=target.name,
        ownership_path=ownership, output=recovery, stability_seconds=0)
    return args, target, {"ownership_path": str(ownership), "recovery_root": str(recovery)}


def test_recovery_uses_current_snapshot_and_retains_original_failure(tmp_path, monkeypatch):
    args, target, reference = recovery_fixture(tmp_path, monkeypatch)
    original = {name:(target/name).read_bytes() for name in ("receipt.json", "failure.json", "policy-budget-terminal.json")}
    with pytest.raises(ValueError, match="document_topology_or_size_invalid"):
        track_feedback.prepare_track_feedback(**args)
    rollout, _, report = track_feedback.prepare_track_feedback(**args, policy_recovery_reference=reference,
        allow_policy_budget_terminal=True, material_view="policy-visible-audit-v2")
    assert report["judge_eligible"] and report["actual_turn_statuses"] == ["interrupted"]
    assert report["stage_completion_claimed"] is False
    source = rollout["provider_metadata"]["source_turns"][0]
    assert source["after_manifest_blake3"] is None
    assert source["recovery_current_after_manifest_blake3"] == read_document(
        Path(reference["recovery_root"]) / "after/manifest.json")["document_blake3"]
    assert rollout["provider_metadata"]["verified_policy_recovery_proofs"][0]["original_archival_failure"]["error"] == "AttributeError"
    assert not (target/"after").exists()
    assert all((target/name).read_bytes()==raw for name,raw in original.items())
    assert not (args["output_root"] / "judge-attempt.json").exists()


def test_recovery_does_not_establish_suffix_stages(tmp_path, monkeypatch):
    args, _, reference = recovery_fixture(tmp_path, monkeypatch)
    calls=[]
    monkeypatch.setattr(track_feedback, "judge_track_once", lambda root, **kw: calls.append(root.name) or {"valid":True})
    monkeypatch.setattr(track_feedback, "verify_feedback", lambda root: {"valid":True,"score":{"fixture_only":True},"source_rollout_blake3":"x"})
    roots=track_feedback.evaluate_track_feedback(args["run_root"],args["checkpoint_identity"],args["output_root"],
        policy_recovery_references={"S1":reference})
    assert calls==["S1"] and len(roots)==1
    coverage=json.loads((args["output_root"]/"coverage.json").read_text())
    assert all(row["rubric_score"] is None and row["reason"]=="recovery_does_not_establish_later_phase" for row in coverage["stages"][1:])


def test_changed_current_workspace_still_rejected(tmp_path,monkeypatch):
    args,_,reference=recovery_fixture(tmp_path,monkeypatch)
    workspace=next((args["run_root"]/"actors").iterdir())
    (workspace/"notes/unaccounted.md").write_text("new write after recovery")
    with pytest.raises(ValueError,match="stability_binding_invalid"):
        track_feedback.prepare_track_feedback(**args,policy_recovery_reference=reference)
    assert not args["output_root"].exists()
