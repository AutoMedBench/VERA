"""Routing tests only; recovery evidence verifier has independent real-byte tests."""
import json
from pathlib import Path
import shutil

import pytest

from training.automedbench_lite.adapter import read_document, write_once
from training.eva_rsi import terminal_attempts
from test_terminal_attempt_gate import seven, replace_doc


def test_unregistered_receipt_adds_explicit_coverage_without_rewriting_zero_counter(seven, tmp_path, monkeypatch):
    document, _ = seven
    run = Path(document["benchmark_run_root"])
    summary = read_document(run / "track-rollouts/summary.json")
    row = summary["tracks"][0]
    track = row["track"]
    turn = run / "track-rollouts" / track / "turns/01-planning"
    raw = json.loads((turn / "receipt.json").read_text())
    failure = write_once(turn / "failure.json", {"phase":"01-planning", "error":"AttributeError", "reward":None})
    row.update(actual_turn_count=0, turn_receipt_blake3s=[], errors=[{k:v for k,v in failure.items() if k!="document_blake3"}])
    summary["tracks"][0] = replace_doc(run / "track-rollouts" / track / "rollout.json", row)
    replace_doc(run / "track-rollouts/summary.json", summary)
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()
    shutil.move(turn / "after", recovery_root / "after")
    recovery = {"valid":True, "reference":{"ownership_path":"fixture-owner", "recovery_root":str(recovery_root)},
        "proof":{"document_blake3":"r"*64, "async_model_jobs_present":False,
            "source_commitments":{"receipt_blake3":raw["receipt_blake3"],
                "before_manifest_document_blake3":read_document(turn/"before/manifest.json")["document_blake3"]}},
        "original_archival_failure":failure,
        "recovery_current_after_manifest_blake3":read_document(recovery_root/"after/manifest.json")["document_blake3"],
        "workspace_after_semantics":"verified_current_time_recovery_not_original_terminal_snapshot"}
    calls=[]
    monkeypatch.setattr(terminal_attempts,"verify_feedback_recovery_reference",
        lambda *args: calls.append(args) or recovery)
    original = (run / "track-rollouts/summary.json").read_bytes()
    with pytest.raises(ValueError): terminal_attempts.verify_terminal_attempts(document)
    assert not calls
    result=terminal_attempts.verify_terminal_attempts({**document,"policy_recovery_references":{track:{"S1":recovery["reference"]}}})
    observed=result["tracks"][track]
    assert observed["archival_summary_turn_count"]==0 and observed["verified_retained_turn_count"]==1
    assert observed["recovered_unregistered_turn_count"]==1 and observed["attempted_stages"]==["S1"]
    assert observed["sources"][0]["after_manifest_blake3"] is None
    assert observed["unreachable_scores"] is None and not (turn/"after").exists()
    assert (run/"track-rollouts/summary.json").read_bytes()==original
    with pytest.raises(ValueError,match="one_unregistered_final_turn"):
        terminal_attempts.verify_terminal_attempts({**document,"policy_recovery_references":{track:{"S2":recovery["reference"]}}})
