"""Synthetic profile tests and source-diff guards; no model/score production."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex
from training.eva_rsi import judge_identity as module
from training.eva_rsi.evidence import commitment

OLD = '''def prepare_workspace_rollout():
    return PreparedAgentJudgeTask(task="fixture")
def grade_prepared_rollout():
    return {"semantic_attempt_count":1}
'''
NEW = '''def prepare_workspace_rollout():
    prepared = PreparedAgentJudgeTask(task="fixture")
    _current_after_recovery_annotation(prepared)
    return prepared
def _current_after_recovery_annotation(prepared):
    return {"fixture_annotation":True}
def grade_prepared_rollout():
    return {"semantic_attempt_count":1, **_current_after_recovery_annotation(prepared)}
'''


def family(tmp_path, monkeypatch, change=None):
    shared={"judge_backend":"native_astra","requested_model":"gpt-6-astra","provider":"eva_native_astra",
        "maximum_execution_frontiers":32,"maximum_workspace_calls":64,"maximum_transport_projection_turns":65,
        "turn_timeout_seconds":600,"material_view":"policy-visible-audit-v2",
        "material_view_implementation_blake3":"c"*64,"materializer_implementation_blake3":"d"*64}
    rows=[]
    for supports,letter in ((False,"a"),(True,"b")):
        profile={**shared,"implementation_blake3":letter*64}
        if supports and change=="timeout":profile["turn_timeout_seconds"]=900
        if supports and change=="unknown":profile["unknown_profile_field"]=True
        rows.append({"judge_id":blake3_hex(profile),"profile":profile,"core_root":"/fixture",
            "core_commit":letter*40,"annotation_support":supports})
    core={"schema":module.SCHEMA,"declared_difference":module.DIFFERENCE,"profiles":rows}
    value={**core,"document_blake3":blake3_hex(core)}
    path=tmp_path/"family.json";path.write_text(json.dumps(value))
    monkeypatch.setattr(module,"_source",lambda row: (NEW if row["annotation_support"] else OLD,"/fixture/slime_agent_judge.py"))
    return commitment(path),rows


def test_explicit_family_preserves_raw_id_and_normalizes_comparison_only(tmp_path,monkeypatch):
    ref,rows=family(tmp_path,monkeypatch)
    verified=module.verify_judge_comparison_family(ref)
    ids=[]
    for row in rows:
        feedback={"judge_profile":row["profile"],"round_identity":{"model_id":"fixture","checkpoint_id":"same",
            "skill_catalog_id":"same","judge_id":row["judge_id"]}}
        original=deepcopy(feedback)
        result=module.normalize_judge_identity(feedback,verified)
        assert result["raw_round_identity"]==original["round_identity"] and feedback==original
        ids.append(result["comparison_round_identity"]["judge_id"])
    assert ids[0]==ids[1] and ids[0] not in {row["judge_id"] for row in rows}
    assert verified["implementations_identical"] is False


@pytest.mark.parametrize("change",["timeout","unknown"])
def test_different_protocol_or_unknown_field_rejected(tmp_path,monkeypatch,change):
    ref,_=family(tmp_path,monkeypatch,change)
    with pytest.raises(ValueError,match="judge_family"):
        module.verify_judge_comparison_family(ref)


def test_source_diff_cannot_hide_changed_scoring_or_default():
    module._annotation_only(OLD,NEW)
    with pytest.raises(ValueError,match="undeclared_source_difference"):
        module._annotation_only(OLD,NEW.replace('"semantic_attempt_count":1','"semantic_attempt_count":2'))


def test_unknown_actual_grade_profile_rejected(tmp_path,monkeypatch):
    ref,rows=family(tmp_path,monkeypatch);verified=module.verify_judge_comparison_family(ref)
    with pytest.raises(ValueError,match="undeclared_profile"):
        module.normalize_judge_identity({"round_identity":{"judge_id":"unknown"},"judge_profile":rows[0]["profile"]},verified)


@pytest.mark.parametrize("changed", [None, "material_view_implementation_blake3", "materializer_implementation_blake3"])
def test_family_reopens_all_material_sources_at_declared_commit(tmp_path, monkeypatch, changed):
    source_bytes = {module.RELATIVE: b"# frozen Judge\n", **{
        relative: ("# frozen " + field + "\n").encode()
        for field, relative in module.MATERIAL_SOURCES.items()}}
    for relative, body in source_bytes.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    profile = {"implementation_blake3": blake3_bytes(source_bytes[module.RELATIVE]), **{
        field: blake3_bytes(source_bytes[relative]) for field, relative in module.MATERIAL_SOURCES.items()}}
    commit = "a" * 40
    calls = []

    def frozen_git(argv):
        calls.append(argv)
        assert argv[:4] == ["git", "-C", str(tmp_path), "show"]
        revision, relative = argv[4].split(":", 1)
        assert revision == commit
        return source_bytes[relative]

    monkeypatch.setattr(module.subprocess, "check_output", frozen_git)
    if changed:
        (tmp_path / module.MATERIAL_SOURCES[changed]).write_bytes(b"# unrelated new imported implementation\n")
    row = {"core_root": str(tmp_path), "core_commit": commit, "profile": profile}
    if changed:
        with pytest.raises(ValueError, match="material_source_version_differs"):
            module._source(row)
    else:
        assert module._source(row) == (source_bytes[module.RELATIVE], str(tmp_path / module.RELATIVE))
        assert len(calls) == 3


def test_reopen_passes_frozen_family_sources_not_current_runtime_sources(tmp_path, monkeypatch):
    from training.benchmark_feedback import automed_codex
    ref, rows = family(tmp_path, monkeypatch)
    verified = module.verify_judge_comparison_family(ref)
    row = rows[0]
    source = tmp_path / "grade"
    source.mkdir()
    (source / "verification.json").write_text(json.dumps({"round_identity": {"judge_id": row["judge_id"]}}))
    seen = {}

    def reopen(root, **kwargs):
        assert root == source
        seen.update(kwargs)
        return {"round_identity": {"judge_id": row["judge_id"]}, "judge_profile": row["profile"]}

    monkeypatch.setattr(automed_codex, "verify_feedback", reopen)
    module.reopen_family_feedback(source, verified, include_skill_source_binding=True)
    assert seen["judge_implementation_source"] == "/fixture/slime_agent_judge.py"
    assert seen["judge_material_sources"] == {
        field: str(Path("/fixture") / relative) for field, relative in module.MATERIAL_SOURCES.items()}
    assert seen["include_skill_source_binding"] is True
