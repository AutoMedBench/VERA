"""CPU-only proof annotation; original policy messages remain byte-identical."""
from copy import deepcopy

import pytest

from eva_agent.pipeline.digests import canonical_json_bytes
from eva_agent.training import slime_agent_judge as module
from test_slime_agent_judge import _generic_rollout


def recovered(tmp_path):
    source, rubric, path, model = _generic_rollout(tmp_path)
    proof = {"schema": "eva.automedbench-policy-recovery.v1", "document_blake3": "r" * 64,
        "actual_terminal_status": "interrupted", "recovery_snapshot_is_original_terminal_snapshot": False,
        "workspace_quiescence_at_original_terminal_claimed": False, "current_workspace_stability_observed": True,
        "original_after_snapshot_missing": True, "provider_replayed": False, "turn_retried": False,
        "later_stages_evaluated": False, "recovery_current_after_tree_blake3": source["workspace_after"]["tree_blake3"],
        "source_commitments": {"receipt_blake3": source["provider_receipt_blake3"],
            "failure_document_blake3": "f" * 64, "policy_budget_terminal_document_blake3": "t" * 64}}
    source["provider_metadata"].update(policy_recovery_feedback_policy="explicit_verified_current_workspace_recovery.v1",
        verified_policy_recovery_proofs=[{"valid": True, "proof": proof,
            "original_archival_failure": {"document_blake3": "f" * 64, "error": "AttributeError"},
            "original_policy_terminal": {"document_blake3": "t" * 64, "workspace_quiescence_verified": False}}],
        source_turns=[{"after_manifest_blake3": None, "recovery_document_blake3": "r" * 64}])
    return source, rubric, path, model


def test_recovery_is_judge_only_annotation_not_fake_policy_event(tmp_path):
    source, rubric, path, model = recovered(tmp_path)
    policy = canonical_json_bytes(source["messages"])
    prepared = module.prepare_workspace_rollout(source, rubric=rubric, source_path=path, judge_model_id=model)
    seen = []
    class CaptureRequest:
        def judge(self, request, rubric):
            seen.append(request)
            raise RuntimeError("CPU fixture stops before any assessment")
    with pytest.raises(RuntimeError, match="CPU fixture"):
        module.grade_prepared_rollout(prepared, judge=CaptureRequest(), output_root=tmp_path)
    annotation = seen[0].judge_only_reference["source_workspace_observation"]
    assert annotation["not_actor_observed_context"] is True
    assert annotation["verified_current_after_recovery"]["original_archival_failure"]["error"] == "AttributeError"
    assert canonical_json_bytes(prepared.trajectory.document["messages"]) == policy
    assert "source_workspace_observation" not in seen[0].policy_visible_context
    assert not (tmp_path / "assessment.json").exists()


@pytest.mark.parametrize("change", ["tree", "time", "missing_optin", "failure"])
def test_invalid_recovery_annotation_rejected_before_judge(tmp_path, change):
    source, rubric, path, model = recovered(tmp_path)
    metadata = source["provider_metadata"]
    proof = metadata["verified_policy_recovery_proofs"][0]["proof"]
    if change == "tree": proof["recovery_current_after_tree_blake3"] = "x" * 64
    elif change == "time": proof["recovery_snapshot_is_original_terminal_snapshot"] = True
    elif change == "failure": proof["source_commitments"]["failure_document_blake3"] = "x" * 64
    else: metadata.pop("policy_recovery_feedback_policy")
    with pytest.raises(ValueError, match="recovery"):
        module.prepare_workspace_rollout(source, rubric=rubric, source_path=path, judge_model_id=model)


def test_default_has_no_new_judge_annotation(tmp_path):
    source, rubric, path, model = _generic_rollout(tmp_path)
    original = deepcopy(source)
    prepared = module.prepare_workspace_rollout(source, rubric=rubric, source_path=path, judge_model_id=model)
    assert module._current_after_recovery_annotation(prepared) == {}
    assert source == original
