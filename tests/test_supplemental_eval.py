"""Supplement orchestration fixtures only; no real actor/Judge/GPU operations."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.automedbench_lite.adapter import write_once, read_document
from training.eva_rsi import supplemental_eval as module
from training.eva_rsi.evidence import commitment
from test_retained_eval_report import source, raw


@pytest.fixture
def fixture(source, tmp_path, monkeypatch):
    context = module.read(source / "context.json")
    context.update(loop_id="fixture-loop", round=1, checkpoint_root="/fixture/checkpoints",
        architecture_model_path="/fixture/architecture", skill_catalog_id="a" * 64, s_target="S3")
    raw(source / "context.json", context)
    settings = module.read(tmp_path / "settings.json")
    settings.update(allow_classification_detection_supplement=True, supplemental_source_attempt=str(source),
        evaluation_track_timeout_seconds=3600, evaluation_tool_output_token_limit=2048,
        codex_bin="/fixture/codex", training_python="/fixture/python", cds_receipt="/fixture/cds",
        missing4_receipt="/fixture/missing", public_model_runtime="/fixture/runtime",
        cpu_image="fixture-image", public_image="/fixture/public.jpg")
    raw(tmp_path / "settings.json", settings)
    run = source / "benchmark/run"
    for track in module.TRACKS:
        audit = run / "track-rollouts" / track
        audit.mkdir()
        budget = write_once(audit / "track-budget.json", {"timeout_seconds": 3600})
        write_once(audit / "track-budget-outcome.json", {
            "schema": "eva.automedbench-track-budget-outcome.v1", "actual_turn_count": 0,
            "budget_document_blake3": budget["document_blake3"], "app_server_pids": []})
        write_once(audit / "track-deadline-cleanup.json", {"workspace_quiescent": True, "error_category": None})
    monkeypatch.setattr(module.production, "actor_runtime_sources", lambda **kwargs: [commitment(module.__file__)])
    monkeypatch.setattr(module.production, "codex_binary_binding", lambda path: {
        "codex_binary": path, "codex_version": "codex-cli 0.153.4", "codex_binary_blake3": "b" * 64})
    monkeypatch.setattr(module.production, "native_scoring_options", lambda settings: {
        "evaluator_python": Path(settings["training_python"]), "timeout": 600})
    output = tmp_path / "fresh-operator-attempt"
    return SimpleNamespace(source=source, source_run=run, output=output,
        context={**context, "attempt_root": str(output)}, settings=settings)


def terminal(value):
    raw(value.source / "exit.json", {"attempt_id": value.source.name, "exit_code": 1, "error_type": None})
    raw(value.source / "serving/session-exit.json", {"cleanup": {"verified_members_remaining": 0}})


def test_default_waiting_is_read_only_and_execution_is_blocked(fixture, monkeypatch):
    from training.eva_rsi import serving
    monkeypatch.setattr(serving, "serving_session", lambda **kwargs: pytest.fail("must not launch"))
    before = {path: path.read_bytes() for path in fixture.source.rglob("*") if path.is_file()}
    value = module.run_supplemental(fixture.context, fixture.settings)
    assert value["status"] == "WAITING_SOURCE" and value["provider_calls"] == value["gpu_launches"] == 0
    assert value["tracks_requested"] == ["classification", "detection"]
    assert value["stages_requested"] == list(module.PHASES)
    assert value["profile_requested"]["args"]["track_timeout"] == 3600
    assert value["profile_requested"]["args"]["workers"] == 2
    assert len(value["prepared_public_input_tracks"]) == 7 and len(value["executed_tracks"]) == 2
    assert not fixture.output.exists()
    with pytest.raises(ValueError, match="original_worker_or_serving_active"):
        module.run_supplemental(fixture.context, fixture.settings, execute=True)
    assert not fixture.output.exists()
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("change", ["optin", "checkpoint", "round", "same_root", "budget", "workers", "turn"])
def test_invalid_scope_or_identity_never_launches(fixture, change):
    terminal(fixture)
    if change == "optin": fixture.settings.pop("allow_classification_detection_supplement")
    if change == "checkpoint": fixture.context["model_path"] = "/different"
    if change == "round": fixture.context["round"] = 2
    if change == "same_root": fixture.context["attempt_root"] = str(fixture.source / "new")
    if change == "budget": fixture.settings["evaluation_track_timeout_seconds"] = 900
    if change == "workers": fixture.settings["supplemental_workers"] = 3
    if change == "turn":
        raw(fixture.source_run / "track-rollouts/classification/turns/01-planning/receipt.json", {})
    with pytest.raises(ValueError):
        module.run_supplemental(fixture.context, fixture.settings, execute=True)
    assert not fixture.output.exists()


def test_original_cleanup_is_required(fixture):
    terminal(fixture)
    raw(fixture.source / "serving/session-exit.json", {"cleanup": {"verified_members_remaining": 1}})
    with pytest.raises(ValueError, match="cleanup_unproved"):
        module.run_supplemental(fixture.context, fixture.settings, execute=True)
    assert not fixture.output.exists()


def actor_fixture(fixture, monkeypatch, *, fail=False):
    from training.automedbench_lite import track_adapter, track_actor, track_feedback
    from training.benchmark_feedback import automed_codex
    from training.eva_rsi import serving
    terminal(fixture)
    events, active, calls = [], [], []
    monkeypatch.setattr(track_adapter, "TrackRelease", lambda path: SimpleNamespace(receipt_path=path))
    def prepare(releases, output_root):
        assert set(releases) == set(track_adapter.BY_TRACK)
        run = output_root / "new-fixture-run"
        run.mkdir(parents=True)
        write_once(run / "track-run-manifest.json", {"fixture_only": True})
        events.append("prepare")
        return run
    monkeypatch.setattr(track_adapter, "prepare_track_run", prepare)
    @contextmanager
    def session(**kwargs):
        root = kwargs["output_root"]
        root.mkdir()
        identity, canary = root / "checkpoint-identity.json", root / "image-canary.json"
        raw(identity, {"exact_final_model_path": kwargs["model_path"].as_posix(), "fresh_identity": True})
        raw(canary, {"checkpoint_identity_blake3": commitment(identity)["blake3"]})
        assert kwargs["context_length"] == 32768
        events.append("serving_start"); active.append(True)
        try:
            yield serving.ServingSession(identity, canary, root, 123)
        finally:
            active.pop(); events.append("serving_stopped")
            raw(root / "session-exit.json", {"cleanup": {"verified_members_remaining": 0}})
    monkeypatch.setattr(serving, "serving_session", session)
    async def actor(args, *, memory_profile=False, supra_profile=False):
        assert active and not memory_profile and supra_profile
        assert args.tracks == ["classification", "detection"] and args.track_timeout == 3600
        assert args.workers == 2 and args.auto_compact_token_limit == 20480
        events.append("actor")
        if fail: raise RuntimeError("fixture actor failure")
        root = args.run_root / "track-rollouts"
        root.mkdir()
        write_once(root / "attempt.json", {"tracks": args.tracks, "planned_coding_rollouts": 2,
            "one_attempt_per_track": True, "full_seven_track_evaluation": False})
        rows = []
        for track in args.tracks:
            (root / track).mkdir()
            row = write_once(root / track / "rollout.json", {"track": track,
                "completed_requested_turns": False, "errors": []})
            rows.append(row)
        write_once(root / "summary.json", {"tracks": rows})
    monkeypatch.setattr(track_actor, "run_tracks", actor)
    monkeypatch.setattr("training.automedbench_lite.track_scoring.score_track",
        lambda *args, **kwargs: pytest.fail("incomplete workflow must not invoke scorer"))
    def skills(run, output, context, settings, scope):
        path = output / "skill-content-binding.json"
        raw(path, {"fixture_only": True})
        return {"schema": "eva.rsi-evaluation-index.v2", "skill_catalog_identity_mode": "verified-content-v1",
            "skill_content_identity": commitment(path)}
    monkeypatch.setattr(module.production, "evaluation_skill_index", skills)
    selected = {("classification", "S1"), ("classification", "S2"), ("detection", "S1")}
    monkeypatch.setattr(module, "stage_readiness", lambda bound, track, stage, exited:
        {"status": "ready", "source_references": []} if (track, stage) in selected
        else {"status": "unavailable", "score": None, "reason": "fixture_unattempted"})
    def evaluate(run, identity, output, **options):
        assert not active
        assert options["allow_policy_budget_terminal"] and options["allow_context_budget_terminal"]
        assert options["material_view"] == "policy-visible-audit-v2"
        assert options["native_turn_timeout_seconds"] == 600 and options["postround_judge_verdict_replacements"] == 2
        actor_binding = options["context_terminal_source_binding"]
        assert actor_binding["checkpoint_identity"] == commitment(identity)
        assert actor_binding["profile_requested"]["thinking"] is True
        key = (options["track"], options["stages"][0]); calls.append(key)
        if key == ("classification", "S2"):
            raise RuntimeError("fixture Judge unavailable, never retry")
        root = output / options["stages"][0]
        raw(root / "verification.json", {"valid": True, "status": "scored", "case_id": key[0],
            "stage": key[1], "score": {"reward_bps": 0}})
        raw(root / "postround-judge-attempt-binding.json", {"fixture_only": True})
        return (root,)
    monkeypatch.setattr(track_feedback, "evaluate_track_feedback", evaluate)
    monkeypatch.setattr(automed_codex, "verify_feedback", lambda root: module.read(root / "verification.json"))
    return events, calls


def test_fresh_two_track_component_and_stage_failure_not_zero(fixture, monkeypatch):
    events, calls = actor_fixture(fixture, monkeypatch)
    before = {path: path.read_bytes() for path in fixture.source.rglob("*") if path.is_file()}
    result = module.run_supplemental(fixture.context, fixture.settings, execute=True)
    assert events == ["prepare", "serving_start", "actor", "serving_stopped"]
    assert sorted(calls) == [("classification", "S1"), ("classification", "S2"), ("detection", "S1")]
    assert result["verified_stage_count"] == 2 and result["unavailable_stage_count"] == 8
    index = module.read(result["index"]["path"])
    assert index["schema"] == "eva.rsi-evaluation-index.v2" and not index["full_seven_track_evaluation"]
    assert index["evaluation_mode"] == "diagnostic_subset" and index["new_rollouts_per_track"] == 1
    assert index["tracks_requested"] == ["classification", "detection"]
    assert index["skill_catalog_identity_mode"] == "verified-content-v1"
    assert len(index["postround_judge_attempt_isolation"]) == len(index["feedback_roots"]) == 2
    assert index["actor_runtime_binding"]["path"].startswith(str(fixture.output))
    assert module.read(index["native_task_scores"]["path"])["tracks_requested"] == list(module.TRACKS)
    assert all(row["task_score_0_1"] is None for row in module.read(index["native_task_scores"]["path"])["tracks"])
    coverage = module.read(fixture.output / "workspace-feedback/coverage.json")
    assert [row["score"] for row in coverage["stages"] if row["status"] == "verified"] == [0, 0]
    assert all(row["score"] is None for row in coverage["stages"] if row["status"] == "unavailable")
    assert all(path.read_bytes() == data for path, data in before.items())
    with pytest.raises(ValueError, match="output_already_used"):
        module.run_supplemental(fixture.context, fixture.settings, execute=True)
    assert len(calls) == 3


def test_actor_failure_cleans_server_and_never_launches_judges(fixture, monkeypatch):
    events, calls = actor_fixture(fixture, monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="fixture actor"):
        module.run_supplemental(fixture.context, fixture.settings, execute=True)
    assert events[-1] == "serving_stopped" and calls == []
    failure = module.read(fixture.output / "supplemental-failure.json")
    assert failure["error_type"] == "RuntimeError" and not failure["automatic_retry"]
    assert not (fixture.output / "supplemental-index.json").exists()


def test_native_subset_only_scores_complete_track_and_preserves_real_zero(tmp_path, monkeypatch):
    run, output = tmp_path / "run", tmp_path / "native"
    identity = tmp_path / "identity.json"; raw(identity, {"fixture_only": True})
    for track in module.TRACKS:
        target = run / "track-rollouts" / track
        target.mkdir(parents=True)
        write_once(target / "rollout.json", {"completed_requested_turns": track == "classification", "errors": []})
    calls = []
    def score(run, track, python, **options):
        calls.append(track)
        target = options["score_output_root"] / track
        target.mkdir()
        return write_once(target / "score.json", {"native_result": {"task_score_0_1": 0}})
    monkeypatch.setattr("training.automedbench_lite.track_scoring.score_track", score)
    path = module._score_subset(run, identity, output, {"evaluator_python": Path("/fixture/python"), "timeout": 600})
    value = module.read(path)
    assert calls == ["classification"]
    assert value["tracks"][0]["status"] == "scored" and value["tracks"][0]["task_score_0_1"] == 0
    assert value["tracks"][1]["status"] == "unavailable" and value["tracks"][1]["task_score_0_1"] is None
