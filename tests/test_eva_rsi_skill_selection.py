"""Real consumer composition around synthetic attribution; no model calls."""
from types import SimpleNamespace
from pathlib import Path

import pytest

from training.eva_rsi.controller import read, write
from training.eva_rsi.evidence import EvidenceError, commitment
from training.eva_rsi.skill_selection import apply_verified_attribution, read_selection, render_selection
from test_eva_rsi_skill_attribution import controller_fixture


def verified(root, *rows, stage="S3"):
    write(root / "actual-attribution-fixture.json", {"fixture_only": True})
    return {"opus_attribution": "verified", "result": commitment(root / "actual-attribution-fixture.json"),
        "suggestions": [{"action": action, "skill_id": identifier} for action, identifier in rows],
        "verified_selection_source": {"stage": stage, "catalog_id": "catalog",
            "catalog": [{"skill_id": "existing/skill", "allowed_stages": ["S3"], "content_blake3": "a" * 64},
                        {"skill_id": "existing/other", "allowed_stages": ["S1"], "content_blake3": "b" * 64}]}}


def selected(root):
    decision = verified(root, ("add", "existing/skill"))
    return apply_verified_attribution(root / "selection", decision, previous=None, catalog_id="catalog")["selection"]


def test_existing_add_remove_changes_only_future_stage_preferences(tmp_path):
    first = selected(tmp_path)
    original = read(first["path"])
    assert original["stages"]["S3"] == {"preferred": ["existing/skill"], "deprioritized": []}
    decision = verified(tmp_path, ("remove", "existing/skill"))
    second = apply_verified_attribution(tmp_path / "selection", decision, previous=first, catalog_id="catalog")
    assert second["status"] == "applied" and second["canonical_catalog_changed"] is False
    assert read_selection(second["selection"])["stages"]["S3"] == {
        "preferred": [], "deprioritized": ["existing/skill"]}
    assert read(first["path"]) == original
    assert render_selection(second["selection"], stage="S1", visible_ids=["existing/other"]) == ""


def test_unsupported_or_conflicting_proposals_and_unavailable_opus_retain_without_block(tmp_path):
    first = selected(tmp_path)
    for decision in (verified(tmp_path, ("add", None), ("remove", "existing/other")),
                     verified(tmp_path, ("add", "existing/skill"), ("remove", "existing/skill")),
                     {"opus_attribution": "failed_attempt"}):
        outcome = apply_verified_attribution(tmp_path / "selection", decision, previous=first, catalog_id="catalog")
        assert outcome["status"] == "retained" and outcome["selection"] == first
        assert outcome["reason"] and not outcome["selection_guidance_changed"]


def test_selection_checks_current_verified_stage_permissions_and_exact_bytes(tmp_path):
    reference = selected(tmp_path)
    with pytest.raises(EvidenceError, match="not_stage_eligible"):
        render_selection(reference, stage="S3", visible_ids=["unrelated/skill"])
    with pytest.raises(EvidenceError, match="catalog_differs"):
        read_selection(reference, catalog_id="different-catalog")
    document = read(reference["path"]); document["stages"]["S3"]["preferred"] = []
    write(Path(reference["path"]), document)
    with pytest.raises(EvidenceError, match="commitment_differs"):
        read_selection(reference)


def test_controller_applies_verified_guidance_and_carries_it_to_next_attempt_without_pause(tmp_path):
    controller, state = controller_fixture(tmp_path)
    state["rounds"][0]["skill_decision"] = verified(tmp_path, ("add", "existing/skill"))
    controller.save(state)
    before = (controller.root / "config.json").read_bytes()
    controller.advance_internal(state)
    state, config = controller.load()
    outcome = state["rounds"][0]["selection_decision"]
    assert outcome["status"] == "applied" and state["status"] == "ready"
    assert state["round"] == 2 and state["phase"] == "target_selection"
    assert state["rounds"][0]["durable_updates"] == 50 and state["rounds"][0]["complete"]
    assert state["skill_catalog_id"] == "catalog" and state["evaluation"] == "evaluation.json"
    assert controller.context(state, config, tmp_path / "next-attempt")["skill_selection"] == outcome["selection"]
    assert (controller.root / "config.json").read_bytes() == before


def test_both_actual_prompt_consumers_use_identical_public_guidance_without_judge_rationale(tmp_path, monkeypatch):
    from training.automedbench_lite import track_actor
    from eva_agent.training.codex_slime_rollout import selection_instructions
    reference = selected(tmp_path)
    discovery = {"skills": [{"skill_id": "existing/skill"}]}
    monkeypatch.setattr(track_actor, "build_public_skill_discovery", lambda *args: discovery)
    monkeypatch.setattr(track_actor, "render_public_skill_discovery", lambda *args: "UNCHANGED_DISCOVERY")
    before = track_actor.phase_prompt(None, 2, "classification")
    after = track_actor.phase_prompt(None, 2, "classification", selection=reference, catalog_id="catalog")
    surface = SimpleNamespace(public_metadata=lambda stage: {"visible_skill_ids": ["existing/skill"]})
    native = selection_instructions(SimpleNamespace(skills_factory=surface), "S3", {
        "skill_selection": reference, "skill_catalog_id": "catalog"})
    assert after[2] == before[2] + native and after[3] is before[3]
    assert '"preferred":["existing/skill"]' in native and "remain available" in native
    assert "actual-attribution-fixture" not in native and "reward" not in native


def test_production_eval_command_and_runtime_arguments_bind_the_same_selection(tmp_path):
    from training.eva_rsi.production_eval import actor_arguments, actor_profile
    reference = selected(tmp_path)
    settings = {"public_model_runtime": "/models", "training_python": "/python", "cpu_image": "image", "codex_bin": "/codex"}
    scope = {"mode": "full_single_pass", "tracks": ["classification"]}
    args, argv = actor_arguments(tmp_path, tmp_path, settings, scope, actor_profile(settings, scope),
        identity_path=tmp_path / "identity", canary_path=tmp_path / "canary", selection=reference, catalog_id="catalog")
    assert str(args.skill_selection) == argv[argv.index("--skill-selection") + 1] == reference["path"]
    assert args.skill_selection_blake3 == argv[argv.index("--skill-selection-blake3") + 1] == reference["blake3"]
    assert "--supra-profile" in argv


def test_selection_mismatch_rejected_before_evaluation_evidence_reopening(tmp_path):
    from training.eva_rsi.evidence import verify_evaluation
    reference = selected(tmp_path)
    path = tmp_path / "index.json"; write(path, {"skill_selection": reference})
    with pytest.raises(EvidenceError, match="selection_context_differs"):
        verify_evaluation(path, {})
