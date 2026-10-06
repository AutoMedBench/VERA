"""CPU-only boundaries; synthetic scorer values are not medical evaluations."""
from pathlib import Path
import sys

import pytest

from training.automedbench_lite import track_scoring
from training.automedbench_lite.adapter import EvaluationError, write_once
from training.eva_rsi import production_eval as module
from training.eva_rsi.evidence import commitment, read


def actor(run, track, *, complete=True, errors=()):
    path = run / "track-rollouts" / track / "rollout.json"
    path.parent.mkdir(parents=True)
    write_once(path, {"completed_requested_turns": complete, "errors": list(errors), "fixture_only": True})


def test_per_track_scores_failures_missing_and_incomplete_remain_distinct(tmp_path, monkeypatch):
    run, output = tmp_path / "actor", tmp_path / "native-task-scores"
    identity = tmp_path / "identity.json"
    identity.write_text('{"fixture_only":true}')
    for track in ("classification", "detection", "vqa", "enhancement"):
        actor(run, track)
    actor(run, "segmentation", complete=False)
    actor(run, "synthesis", errors=[{"fixture_only": "owned terminal stop"}])
    old = run / "native-scores" / "vqa" / "failure.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"fixture_only":"historical failure","reward":null}')
    old_bytes = old.read_bytes()
    calls = []
    cache = tmp_path / "pinned-cache"
    cache.mkdir()

    def score(source, track, evaluator, *, timeout, lpips_torch_home, score_output_root):
        calls.append(track)
        assert source == run and evaluator == Path(sys.executable) and timeout == 600
        assert score_output_root == output
        assert lpips_torch_home == (cache if track == "enhancement" else None)
        root = score_output_root / track
        root.mkdir()
        if track == "vqa":
            write_once(root / "failure.json", {"reward": None, "error_code": "fixture_dependency_missing"})
            raise EvaluationError("private exception detail must not enter summary")
        return write_once(root / "score.json", {"fixture_only": True,
            "native_result": {"task_score_0_1": 0.0 if track == "detection" else 0.25}})

    monkeypatch.setattr(track_scoring, "score_track", score)
    path = module.score_native_round(run, identity, output, evaluator_python=Path(sys.executable),
        lpips_torch_home=cache)
    summary = read(path)
    rows = {row["track"]: row for row in summary["tracks"]}
    assert calls == ["classification", "detection", "vqa", "enhancement"]
    assert (summary["scored_tracks"], summary["unavailable_tracks"]) == (3, 4)
    assert rows["detection"]["status"] == "scored" and rows["detection"]["task_score_0_1"] == 0.0
    for track in ("segmentation", "synthesis", "vqa", "report"):
        assert rows[track]["status"] == "unavailable" and rows[track]["task_score_0_1"] is None
    assert rows["report"]["reason"] == "actor_rollout_missing"
    assert rows["segmentation"]["reason"] == rows["synthesis"]["reason"] == "actor_workflow_incomplete"
    assert rows["vqa"]["native_failure"] == commitment(output / "vqa/failure.json")
    assert rows["classification"]["native_score"] == commitment(output / "classification/score.json")
    assert rows["classification"]["actor_rollout"] == commitment(run / "track-rollouts/classification/rollout.json")
    assert summary["checkpoint_identity"] == commitment(identity)
    assert summary["new_policy_rollouts"] == 0 and summary["automatic_retry"] is False
    assert summary["rubric_rewards_modified"] is summary["used_for_stage_target_selection"] is False
    assert "private exception" not in path.read_text()
    assert old.read_bytes() == old_bytes
    assert len(list(output.glob("*-result.json"))) == 7
    with pytest.raises(FileExistsError):
        module.score_native_round(run, identity, output, evaluator_python=Path(sys.executable))
    assert len(calls) == 4  # No retry or overwrite of any score attempt.


def test_invalid_actor_evidence_is_unavailable_without_scorer_dispatch(tmp_path, monkeypatch):
    run = tmp_path / "actor"
    actor(run, "classification")
    path = run / "track-rollouts/classification/rollout.json"
    path.write_text('{"completed_requested_turns":true,"document_blake3":"wrong"}')
    identity = tmp_path / "identity.json"
    identity.write_text('{}')
    monkeypatch.setattr(track_scoring, "score_track", lambda *a, **k: pytest.fail("must not dispatch invalid evidence"))
    summary = read(module.score_native_round(run, identity, tmp_path / "scores", evaluator_python=Path(sys.executable)))
    row = summary["tracks"][0]
    assert row["status"] == "unavailable" and row["task_score_0_1"] is None
    assert row["scorer_invoked"] is False and row["error_type"] == "EvaluationError"
    assert summary["unavailable_tracks"] == 7


def test_native_runtime_options_validate_offline_without_importing_torch(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    result = module.native_scoring_options({"training_python": sys.executable,
        "native_lpips_torch_home": str(cache), "native_score_timeout_seconds": 42})
    assert result == {"evaluator_python": Path(sys.executable).absolute(),
        "lpips_torch_home": cache, "timeout": 42}
    for settings, error in (
        ({"native_evaluator_python": str(tmp_path / "absent")}, "native_evaluator_python_unavailable"),
        ({"native_lpips_torch_home": str(tmp_path / "absent")}, "native_lpips_cache_directory_unavailable"),
        ({"native_score_timeout_seconds": True}, "native_score_timeout_invalid"),
    ):
        with pytest.raises(ValueError, match=error):
            module.native_scoring_options({"training_python": sys.executable, **settings})
