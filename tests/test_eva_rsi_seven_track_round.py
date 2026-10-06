"""CPU scheduling fixtures, not actual evaluation or medical scores."""
import json
from pathlib import Path

import pytest

from training.automedbench_lite import track_feedback
from training.automedbench_lite.track_adapter import BY_TRACK
from training.eva_rsi.production_eval import evaluation_scope, judge_full_round


def test_full_round_explicitly_selects_all_tracks_stages_and_new_exact_registry():
    scope = evaluation_scope({"evaluation_mode": "full_single_pass"})
    assert set(scope["tracks"]) == set(BY_TRACK) and len(scope["tracks"]) == 7
    assert scope["stages"] == ("S1", "S2", "S3", "S4", "S5")
    assert scope["registry"].name == "domain-stage-tables.v2.json"
    assert evaluation_scope({})["tracks"] == ["classification"]
    with pytest.raises(ValueError, match="unsupported_evaluation_scope"):
        evaluation_scope({"evaluation_mode": "invented"})


@pytest.mark.parametrize("timeout", [240, 600])
@pytest.mark.parametrize("failed_track,missing", [(None, False), ("report", False), (None, True)])
def test_parallel_judging_retains_each_track_once_and_failure_is_not_zero(tmp_path, monkeypatch, failed_track, missing, timeout):
    calls = []
    scope = evaluation_scope({"evaluation_mode": "full_single_pass"})
    def simulated_boundary(run, identity, output, *, stages, track, registry_path, native_turn_timeout_seconds=240):
        calls.append(track)
        assert native_turn_timeout_seconds == timeout
        assert run == Path("/fixture/run") and identity == Path("/fixture/identity.json")
        assert registry_path == scope["registry"] and stages == scope["stages"]
        if track == failed_track:
            raise RuntimeError("fixture remote Judge failure")
        selected = stages[:-1] if missing and track == "report" else stages
        return tuple(output / stage for stage in selected)
    monkeypatch.setattr(track_feedback, "evaluate_track_feedback", simulated_boundary)
    output = tmp_path / "feedback"
    def run():
        return judge_full_round(Path("/fixture/run"), Path("/fixture/identity.json"), output,
                               stages=scope["stages"], registry_path=scope["registry"],
                               native_turn_timeout_seconds=timeout)
    if failed_track or missing:
        with pytest.raises(ValueError, match="full_round_workspace_judging_incomplete"):
            run()
    else:
        roots = run()
        assert len(roots) == 35
        assert roots == tuple(output / track / stage for track in BY_TRACK for stage in scope["stages"])
    assert sorted(calls) == sorted(BY_TRACK)  # No automatic retry or dropped track.
    coverage = json.loads((output / "coverage.json").read_text())
    assert coverage["automatic_retry"] is False and coverage["max_parallel_tracks"] == 4
    assert coverage["native_turn_timeout_seconds"] == timeout
    assert coverage["missing_is_not_zero"] is True
    if failed_track:
        assert coverage["failures"] == [{"track": "report", "error_type": "RuntimeError", "rubric_score": None}]
        assert len(coverage["feedback_roots"]) == 30
    elif missing:
        assert len(coverage["feedback_roots"]) == 34


def test_explicit_attempt_prefix_judges_observed_stages_not_unreachable_suffix(tmp_path,monkeypatch):
    scope=evaluation_scope({'evaluation_mode':'full_single_pass'})
    attempted={track:['S1'] for track in BY_TRACK};calls=[]
    def boundary(run,identity,output,*,stages,track,registry_path,allow_policy_budget_terminal=False):
        assert allow_policy_budget_terminal is True and len(stages)==5
        calls.append(track)
        return (output/'S1',)
    monkeypatch.setattr(track_feedback,'evaluate_track_feedback',boundary)
    output=tmp_path/'feedback'
    roots=judge_full_round(tmp_path/'run',tmp_path/'identity',output,stages=scope['stages'],
        registry_path=scope['registry'],attempted_stages=attempted)
    assert len(roots)==7 and set(calls)==set(BY_TRACK)
    coverage=json.loads((output/'coverage.json').read_text())
    assert coverage['unreachable_stage_scores'] is None
    assert all(value==['S2','S3','S4','S5'] for value in coverage['unreachable_stages'].values())


def test_context_prefix_flag_is_explicit_and_requires_attempt_gate(tmp_path,monkeypatch):
    scope=evaluation_scope({'evaluation_mode':'full_single_pass'});calls=[]
    source_binding={'actual_imported_sources':[{'path':'/fixture/local_qwen.py','blake3':'a'*64}]}
    def boundary(run,identity,output,*,stages,track,registry_path,
                 allow_policy_budget_terminal=False,allow_context_budget_terminal=False,
                 context_terminal_source_binding=None):
        assert allow_policy_budget_terminal and allow_context_budget_terminal
        assert context_terminal_source_binding == source_binding
        calls.append(track);return (output/'S1',)
    monkeypatch.setattr(track_feedback,'evaluate_track_feedback',boundary)
    with pytest.raises(ValueError,match='explicit_attempt_gate'):
        judge_full_round(tmp_path,tmp_path/'identity',tmp_path/'blocked',stages=scope['stages'],
            registry_path=scope['registry'],allow_context_budget_terminal=True)
    assert calls==[]
    assert len(judge_full_round(tmp_path,tmp_path/'identity',tmp_path/'feedback',stages=scope['stages'],
        registry_path=scope['registry'],attempted_stages={track:['S1'] for track in BY_TRACK},
        allow_context_budget_terminal=True,context_terminal_source_binding=source_binding))==7


def test_explicit_material_view_reaches_every_track_judge(tmp_path, monkeypatch):
    scope = evaluation_scope({'evaluation_mode': 'full_single_pass'})
    calls = []
    def boundary(run, identity, output, *, stages, track, registry_path, material_view):
        assert material_view == 'policy-visible-audit-v2'
        calls.append(track)
        return tuple(output / stage for stage in stages)
    monkeypatch.setattr(track_feedback, 'evaluate_track_feedback', boundary)
    roots = judge_full_round(tmp_path / 'run', tmp_path / 'identity', tmp_path / 'feedback',
        stages=scope['stages'], registry_path=scope['registry'], material_view='policy-visible-audit-v2')
    assert len(roots) == 35 and set(calls) == set(BY_TRACK)
    coverage = json.loads((tmp_path / 'feedback/coverage.json').read_text())
    assert coverage['judge_material_view'] == 'policy-visible-audit-v2'
