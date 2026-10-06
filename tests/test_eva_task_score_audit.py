"""CPU-only audit-boundary fixtures, never benchmark scores."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from training.eva_rsi import task_score_audit as audit
from training.eva_rsi.controller import write
from training.eva_rsi.evidence import commitment


def source(tmp_path):
    receipts = [{"receipt_blake3": str(i) * 64} for i in range(5)]
    actor_path = tmp_path / "actor.json"
    actor = {"track": "classification", "run_id": "fixture-run", "document_blake3": "a" * 64,
        "completed_requested_turns": True, "errors": [],
        "turn_receipt_blake3s": [r["receipt_blake3"] for r in receipts]}
    write(actor_path, actor)
    native = {"schema": "eva.automedbench-native-track-result.v1", "task_score_0_1": .25,
        "full_public_subset": True, "private_values_exported": False}
    document = {"schema": "eva.automedbench-native-whole-track-score.v1", "track": "classification",
        "run_id": actor["run_id"], "actor_document_blake3": actor["document_blake3"],
        "native_result": native, "submitted_files": [{"path": "predictions.csv", "bytes": 10, "blake3": "b" * 64}]}
    native_path = tmp_path / "native.json"
    write(native_path, document)
    row = {"track": "classification", "status": "scored", "task_score_0_1": .25,
        "actor_rollout": commitment(actor_path), "native_score": commitment(native_path), "native_result": native}
    rollout = {"provider_metadata": {"track": "classification", "evaluated_stage": "S5",
        "codex_turn_receipts": receipts}, "workspace_after": {"files": [{
            "path": "outputs/agents_outputs/predictions.csv", "byte_count": 10, "content_blake3": "b" * 64}]}}
    return row, document, rollout


def review(value=25, *, accepted=True, read_output=True):
    path = "outputs/agents_outputs/predictions.csv" if read_output else "task.json"
    ref = "workspace:after:" + path
    summary = {"schema": audit.SUMMARY_SCHEMA, "native_result_consistent": accepted,
        "task_score_0_100": value, "evidence_refs": [ref], "explanation": "Fixture output checked."}
    result = SimpleNamespace(name="workspace_read", status="completed", arguments={"snapshot": "after", "path": path})
    return SimpleNamespace(summary=json.dumps(summary), agent_trace=SimpleNamespace(results=[result])), SimpleNamespace(source_workspace_refs=[ref])


def test_native_and_five_turn_actor_submission_binding(tmp_path):
    row, document, rollout = source(tmp_path)
    path = tmp_path / "result.json"
    write(path, row)
    assert audit.native_result(commitment(path), track="classification") == (row, document)
    audit.bind_actor(row, document, rollout)
    changed = deepcopy(rollout)
    changed["provider_metadata"]["codex_turn_receipts"][4]["receipt_blake3"] = "wrong-run"
    with pytest.raises(ValueError, match="match_complete_actor"):
        audit.bind_actor(row, document, changed)
    changed = deepcopy(rollout)
    changed["workspace_after"]["files"][0]["content_blake3"] = "wrong-output"
    with pytest.raises(ValueError, match="submission_not_in_snapshot"):
        audit.bind_actor(row, document, changed)


def test_task_review_accepts_native_scale_and_disagreement(tmp_path):
    row, document, _ = source(tmp_path)
    assessment, prepared = review()
    assert audit.validate_review(assessment, prepared, row, document)["task_score_0_100"] == 25
    assessment, prepared = review(None, accepted=False, read_output=False)
    assert audit.validate_review(assessment, prepared, row, document)["native_result_consistent"] is False


@pytest.mark.parametrize("score", [0, 90, True, float("nan")])
def test_subjective_score_cannot_replace_native_t(tmp_path, score):
    row, document, _ = source(tmp_path)
    assessment, prepared = review(score)
    with pytest.raises(ValueError, match="cannot_change_native"):
        audit.validate_review(assessment, prepared, row, document)


def test_task_review_requires_actual_output_read_when_textual(tmp_path):
    row, document, _ = source(tmp_path)
    assessment, prepared = review(read_output=False)
    with pytest.raises(ValueError, match="submission_not_read"):
        audit.validate_review(assessment, prepared, row, document)


def test_native_zero_is_valid_performance_not_unavailable(tmp_path):
    row, document, _ = source(tmp_path)
    row["task_score_0_1"] = 0
    assessment, prepared = review(0)
    assert audit.validate_review(assessment, prepared, row, document)["task_score_0_100"] == 0


def test_missing_native_score_no_provider_and_no_zero(tmp_path, monkeypatch):
    row = {"track": "classification", "status": "unavailable", "task_score_0_1": None}
    write(tmp_path / "classification-result.json", row)
    write(tmp_path / "summary.json", {"tracks": [row]})
    monkeypatch.setattr(audit, "run_task_audit", lambda *a, **k: pytest.fail("must not call provider"))
    refs = audit.audit_native_scores([], tmp_path / "summary.json", tmp_path / "audits")
    result = audit.read(refs[0]["result"]["path"])
    assert result["status"] == "unavailable" and result["task_score_0_100"] is None
    assert result["additional_task_audit_attempts"] == 0
