import json
from pathlib import Path
from uuid import uuid4

import pytest

from training.automedbench_lite.adapter import blake3, canonical, write_once
from training.automedbench_lite.skill_surface import SKILL_TOOLS
from training.automedbench_lite.track_tools import (MutableInventory, TrackTools, TOOLS, TOOLS_V1,
    TOOLS_V2, EXTENDED_MODEL_TOOL, GENERATIVE_MODEL_TOOL)
from training.automedbench_lite.track_actor import PHASES, snapshot


def prepared(tmp_path):
    workspace = tmp_path / "actor"
    workspace.mkdir()
    for directory in ("inputs/CASE", "notes", "code", "outputs/agents_outputs", "public-guidance"):
        (workspace / directory).mkdir(parents=True, exist_ok=True)
    (workspace / "inputs/CASE/image.jpg").write_bytes(b"immutable image fixture")
    write_once(workspace / "task.json", {"track": "classification", "case_ids": ["CASE"]})
    write_once(workspace / "inputs-manifest.json", {"case_ids": ["CASE"], "files": [{"path": "inputs/CASE/image.jpg",
        "bytes": 23, "blake3": blake3(b"immutable image fixture").hexdigest()}]})
    return workspace


def test_mutable_snapshots_exclude_large_inputs_and_retain_history(tmp_path):
    workspace = prepared(tmp_path)
    inventory = MutableInventory(workspace, tmp_path / "audit")
    note = workspace / "notes/progress.md"
    note.write_text("first actual state")
    first = snapshot(inventory, tmp_path / "before")
    note.write_text("next actual state")
    second = snapshot(inventory, tmp_path / "after")
    assert all(not row["path"].startswith("inputs/") for row in first["files"])
    assert (tmp_path / "before/files/notes/progress.md").read_text() == "first actual state"
    assert (tmp_path / "after/files/notes/progress.md").read_text() == "next actual state"
    assert first["immutable_inputs_manifest_blake3"] == second["immutable_inputs_manifest_blake3"]


def test_canonical_skills_unchanged_and_five_real_stage_intents():
    assert tuple(TOOLS[-2:]) == tuple(SKILL_TOOLS)
    assert [stage for _, stage, _ in PHASES] == ["S1", "S2", "S3", "S4", "S5"]
    assert "actual app-server restart" not in PHASES[0][2]
    assert "read" in PHASES[3][2]
    assert tuple(row for row in TOOLS_V2 if row["name"] != EXTENDED_MODEL_TOOL["name"]) == TOOLS_V1
    assert tuple(row for row in TOOLS if row["name"] != GENERATIVE_MODEL_TOOL["name"]) == TOOLS_V2


@pytest.mark.parametrize("track,controls", [("vqa", {"max_new_tokens": 257, "multi_image_mode": "montage"}),
    ("vqa", {"max_new_tokens": 32}), ("report", {"max_new_tokens": 512}),
    ("report", {"max_new_tokens": 64, "prompt": "Generate <|control|> report"})])
def test_generative_job_rejects_missing_or_invalid_controls_before_launch(tmp_path, track, controls):
    workspace = prepared(tmp_path)
    tools = TrackTools(workspace=workspace, audit_root=tmp_path / "audit", image="unused", skills=None,
        model_manifest=tmp_path / "model.json", public_python=Path("/unused/python"))
    tools.contract["track"] = track
    with pytest.raises(Exception):
        tools._submit({"case_ids": ["CASE"], **controls}, generative=True)
    assert not list(tools.job_root.glob("*/submission.json"))


def test_enhancement_requires_explicit_controls_before_job_creation(tmp_path):
    workspace = prepared(tmp_path)
    tools = TrackTools(workspace=workspace, audit_root=tmp_path / "audit", image="unused", skills=None,
        model_manifest=tmp_path / "model.json", public_python=Path("/unused/python"))
    tools.contract["track"] = "enhancement"
    for arguments in ({"case_ids": ["CASE"]}, {"case_ids": ["CASE"], "sigma": .05, "hu_min": 10, "hu_max": 1}):
        with pytest.raises(Exception, match="explicit_enhancement"):
            tools._submit(arguments, extended=True)
    with pytest.raises(Exception, match="does_not_support"):
        tools._submit({"case_ids": ["CASE"]})
    assert not list(tools.job_root.glob("*/submission.json"))


def test_note_event_pairing_and_stage_loader(tmp_path):
    workspace = prepared(tmp_path)
    class Skills:
        def call(self, name, arguments, stage):
            assert stage == "S2"
            return {"actual_fixture_stage": stage}
    tools = TrackTools(workspace=workspace, audit_root=tmp_path / "audit", image="unused", skills=Skills(),
        model_manifest=tmp_path / "model.json", public_python=Path("/unused/python"))
    write_once(tools.audit_root / "phase-policy.json", {"skill_stage": "S2"})
    response = tools.call("automed_write_note", {"name": "progress.md", "content": "observed fixture"}, 7)
    row = json.loads((tools.audit_root / "mcp-events.jsonl").read_text())
    assert response["structuredContent"]["event_id"] == row["event_id"]
    assert row["request_id"] == 7 and not row["is_error"]
    assert not any(r["path"] == "notes/progress.md" for r in row["workspace_before"]["files"])
    assert any(r["path"] == "notes/progress.md" for r in row["workspace_after"]["files"])
    assert row["response_blake3"] == blake3(canonical(response)).hexdigest()


def test_model_status_bound_to_own_job_and_small_summary(tmp_path):
    workspace = prepared(tmp_path)
    tools = TrackTools(workspace=workspace, audit_root=tmp_path / "audit", image="unused", skills=None,
        model_manifest=tmp_path / "model.json", public_python=Path("/unused/python"))
    job_id = str(uuid4())
    (tools.job_root / job_id).mkdir()
    write_once(tools.job_root / job_id / "submission.json", {"track": "classification", "case_ids": ["CASE"]})
    audit = tools.job_root / "authoritative" / job_id
    audit.mkdir(parents=True)
    (audit / "receipt.json").write_text(json.dumps({"status": "running", "raw_output_sample": "must not enter result"}))
    summary = tools._status({"job_id": job_id})
    assert summary["status"] == "running" and "raw_output_sample" not in summary
    assert "must not enter result" not in json.dumps(summary)
    with pytest.raises(Exception): tools._status({"job_id": str(uuid4())})
