"""Strict mixed-policy consumption; no providers or actual grade generation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from training.automedbench_lite.judge_attempt_isolation import BINDING_NAME, run_stage_judge_attempts
from training.eva_rsi import composite_eval
from training.eva_rsi.evidence import commitment


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def retained(tmp_path, monkeypatch):
    # Actual existing isolation writer/verifier, with only the Judge boundary
    # represented by a small explicit fixture result (not a medical grade).
    stage = tmp_path / "old/classification/S1"
    prepared = stage / "prepared"
    write(prepared / "preflight.json", {"case_id": "classification", "stage": "S1"})
    write(prepared / "rollout.json", {"fixture_only": True})
    old_grade = {"valid": True, "status": "scored", "case_id": "classification", "stage": "S1",
        "score": {"reward_bps": 0}, "judge_codex_receipt_blake3": "a" * 64}
    old_root, _, binding = run_stage_judge_attempts(stage, prepared, replacements=2,
        judge=lambda *args, **kwargs: deepcopy(old_grade), verify=lambda root: deepcopy(old_grade))
    original = {"schema": "eva.rsi-evaluation-index.v1", "feedback_roots": [str(old_root)],
        "postround_judge_verdict_replacements": 2, "postround_judge_attempt_isolation": [commitment(binding)],
        "benchmark_run_root": str(tmp_path / "original-actor"), "checkpoint_identity": "/fixture/identity",
        "actor_runtime_binding": {"path": "/fixture/actor-binding", "blake3": "b" * 64}}
    original_path = tmp_path / "original-index.json"
    write(original_path, original)
    standalone = []
    for name in ("S4", "S5"):
        root = tmp_path / "rejudgment" / name
        write(root / "verification.json", {"valid": True, "status": "scored", "case_id": "report",
            "stage": name, "score": {"reward_bps": 0}, "judge_codex_receipt_blake3": name})
        standalone.append(str(root))
    sidecar = tmp_path / "approved.json"
    write(sidecar, {"fixture_only": True, "original_index": commitment(original_path),
        "standalone_roots": standalone})
    index = {**deepcopy(original), "feedback_roots": [str(old_root), *standalone],
             "approved_stage_rejudgments": commitment(sidecar)}
    proof = {"original_index": commitment(original_path), "standalone_roots": standalone}
    # C owns the independently tested sidecar/source verifier. This boundary
    # mock only lets these tests focus on the composite's additional policy join.
    producer = ModuleType("training.eva_rsi.report_rejudge")
    producer.verify_rejudge_index_sidecar = lambda index: deepcopy(proof)
    monkeypatch.setitem(sys.modules, "training.eva_rsi.report_rejudge", producer)
    calls = []
    def verify(root, **kwargs):
        calls.append(str(root))
        return json.loads((Path(root) / "verification.json").read_text())
    monkeypatch.setattr(composite_eval, "verify_feedback", verify)
    return index, original, original_path, old_root, standalone, producer, calls


def test_existing_policy_plus_two_explicit_standalone_grades_preserves_original(retained):
    index, original, original_path, old_root, standalone, _, calls = retained
    before = original_path.read_bytes()
    result = composite_eval._judge_isolation(index)
    assert [row["accepted_attempt_root"] for row in result] == [str(old_root), *standalone]
    assert result[0]["maximum_additional_judge_attempts"] == 2
    assert result[0]["additional_judge_attempts_made"] == 0
    assert all(row["automatic_judge_replacements"] is False for row in result[1:])
    assert all(row["score"]["reward_bps"] == 0 for row in result)
    assert calls == [str(old_root), *standalone]
    assert original_path.read_bytes() == before
    assert composite_eval._judge_isolation(original)[0] == result[0]


@pytest.mark.parametrize("tamper", ["remove_old", "drop_isolation", "extra_root", "checkpoint", "wrong_stage"])
def test_approval_does_not_allow_unrelated_roots_or_dropped_original_policy(retained, tamper):
    index, _, _, _, standalone, _, _ = retained
    if tamper == "remove_old": index["feedback_roots"].pop(0)
    elif tamper == "drop_isolation": index["postround_judge_attempt_isolation"] = []
    elif tamper == "extra_root": index["feedback_roots"].append("/unapproved-root")
    elif tamper == "checkpoint": index["checkpoint_identity"] = "/other-checkpoint"
    else:
        path = Path(standalone[0]) / "verification.json"
        row = json.loads(path.read_text()); row["stage"] = "S3"; write(path, row)
    with pytest.raises(ValueError, match="composite_approved_rejudge"):
        composite_eval._judge_isolation(index)


def test_missing_or_rejected_approval_never_silently_drops_isolation(retained):
    index, _, _, _, _, producer, _ = retained
    without = {key: value for key, value in index.items() if key != "approved_stage_rejudgments"}
    with pytest.raises(ValueError, match="isolation_commitment"):
        composite_eval._judge_isolation(without)
    def rejected(index):
        raise ValueError("approved source sidecar failed verification")
    producer.verify_rejudge_index_sidecar = rejected
    with pytest.raises(ValueError, match="sidecar failed verification"):
        composite_eval._judge_isolation(index)
