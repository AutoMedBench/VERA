import json
from pathlib import Path

import pytest

from training.automedbench_lite.adapter import EvaluationError, blake3, canonical, write_once
from training.automedbench_lite.docker_runtime import create_command
from training.automedbench_lite.public_tools import PublicTools
from training.automedbench_lite.actor import snapshot
from training.automedbench_lite.skill_surface import SKILL_TOOLS
from eva_agent.harness.skills import SEARCH_SKILLS_INPUT_SCHEMA, LOAD_SKILL_INPUT_SCHEMA


@pytest.fixture
def public_tools(tmp_path):
    workspace = tmp_path / "actor"
    workspace.mkdir()
    (workspace / "inputs").mkdir()
    (workspace / "outputs").mkdir()
    write_once(workspace / "task.json", {"input_path": "inputs/public.txt", "track": "classification"})
    (workspace / "inputs/public.txt").write_text("public fixture")
    return PublicTools(workspace=workspace, audit_root=tmp_path / "audit", image="sha256:" + "a" * 64)


def test_every_call_retains_exact_before_after_and_uuid(public_tools):
    response = public_tools.call("automed_write_note", {"name": "progress.md", "content": "first"}, 17)
    public_tools.call("automed_read_file", {"path": "notes/progress.md"}, 18)
    public_tools.call("automed_write_note", {"name": "progress.md", "content": "second"}, 19)
    rows = [json.loads(line) for line in (public_tools.audit_root / "mcp-events.jsonl").read_bytes().splitlines()]
    assert response["structuredContent"]["event_id"] == rows[0]["event_id"]
    assert rows[0]["request_id"] == 17
    assert len({row["event_id"] for row in rows}) == 3
    assert rows[1]["workspace_before"] == rows[1]["workspace_after"]
    assert rows[0]["workspace_after"] == rows[2]["workspace_before"]
    for row in rows:
        core = {key: value for key, value in row.items() if key != "event_blake3"}
        assert blake3(canonical(core)).hexdigest() == row["event_blake3"]
        for inventory in (row["workspace_before"], row["workspace_after"]):
            for item in inventory["files"]:
                blob = public_tools.audit_root / "workspace-blobs" / item["blake3"]
                assert blake3(blob.read_bytes()).hexdigest() == item["blake3"]


def test_public_paths_fail_closed_and_budget_survives_server_restart(public_tools):
    response = public_tools.call("automed_read_file", {"path": "../scorer/private.json"}, 1)
    assert response["isError"]
    restarted = PublicTools(workspace=public_tools.workspace, audit_root=public_tools.audit_root, image=public_tools.image)
    assert restarted.call_count == 1
    restarted.call_count = 64
    assert restarted.call("automed_read_file", {"path": "task.json"}, 2)["isError"]


def test_snapshot_retains_actual_bytes_mode_and_immutable_manifest(public_tools, tmp_path):
    document = snapshot(public_tools.workspace, tmp_path / "snapshot")
    for row in document["files"]:
        assert row["mode"] == (public_tools.workspace / row["path"]).stat().st_mode & 0o777
        assert (tmp_path / "snapshot/files" / row["path"]).read_bytes() == (public_tools.workspace / row["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        snapshot(public_tools.workspace, tmp_path / "snapshot")


def test_docker_command_is_cpu_public_mount_only(public_tools, tmp_path):
    command = create_command(image=public_tools.image, name="eva-test", workspace=public_tools.workspace,
                             submission=tmp_path / "solution.py")
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command and "--gpus" not in command
    assert command.count("--mount") == 4
    assert "docker.sock" not in " ".join(command) and "scorer" not in " ".join(command)
    with pytest.raises(EvaluationError):
        create_command(image="mutable:tag", name="test", workspace=public_tools.workspace, submission=tmp_path / "code")


def test_skill_schema_is_canonical_and_actual_result_is_unchanged(public_tools):
    assert SKILL_TOOLS[0]["inputSchema"] == SEARCH_SKILLS_INPUT_SCHEMA
    assert SKILL_TOOLS[1]["inputSchema"] == LOAD_SKILL_INPUT_SCHEMA
    observed = []
    original_result = {"skill_id": "fixture", "content": "Actual public skill bytes", "content_blake3": "a" * 64,
                       "delivery": "policy-visible-tool-observation"}
    class Catalog:
        def call(self, name, arguments, *, stage):
            observed.append((name, arguments, stage))
            return original_result
    public_tools.skills = Catalog()
    write_once(public_tools.audit_root / "phase-policy.json", {"skill_stage": "S3"})
    arguments = {"skill_id": "fixture", "stage": "S3"}
    response = public_tools.call("load_skill", arguments, 19)
    assert response["structuredContent"]["result"] == original_result
    assert observed == [("load_skill", arguments, "S3")]
    assert response["isError"] is False
